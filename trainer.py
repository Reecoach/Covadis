# -*- coding: utf-8 -*-
"""
Trainer for robust defender against worst case attacker
"""

from __future__ import annotations

import os
import json
import tensorflow as tf
from dataclasses import asdict
from fig_utils import *

from models import CovertEncoderMLP, CovertDefenderMLP
from data_utils import get_frequency, parse_feature, preprocess_feature_pool, postprocess_feature_indices
from info_utils import (
    kl_divergence,
    entropy_vec,
    row_entropy,
    col_entropy,
    compute_mutual_info,
    mutual_info_XY_under_PN,
    expected_relative_cost,
)

from config import TrainerConfig


class Trainer:
    def __init__(self, config: dict):
        if hasattr(config, "__dataclass_fields__"):
            self.cfg = asdict(config)   # dataclass → dict
        else:
            self.cfg = dict(config)
        self._best_def_ema = float("inf")
        self._no_improve = 0
        self._ema_loss_def = None

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _clip_grads(self, grads):
        clip_norm = self.cfg.get("clip_norm", None)
        if clip_norm is None:
            return grads
        clipped = []
        for g in grads:
            if g is None:
                clipped.append(None)
            else:
                clipped.append(tf.clip_by_norm(g, clip_norm))
        return clipped

    def _anneal_temps(self, attackers, defender):
        factor = float(self.cfg["temperature_anneal"])
        minT = float(self.cfg["min_temperature"])
        for a in attackers:
            a.temperature.assign(tf.maximum(a.temperature * factor, minT))
        defender.temperature.assign(tf.maximum(defender.temperature * factor, minT))

    # --------------------------------------------------
    # Domain resolution
    # --------------------------------------------------
    def _resolve_domains(self, benign_pool_int):
        values_support = sorted(set(benign_pool_int))

        vmin, vmax = min(values_support), max(values_support)
        values_cont = list(range(vmin, vmax + 1))

        def pick(domain):
            if domain == "support":
                return values_support
            elif domain == "continuous":
                return values_cont
            else:
                raise ValueError(f"Unknown domain: {domain}")

        values_X = pick(self.cfg["attacker_domain"])          # attacker + defender input
        values_Y = pick(self.cfg["defender_output_domain"])   # defender output

        if self.cfg["attacker_domain"] != self.cfg["defender_input_domain"]:
            raise ValueError(
                "attacker_domain must equal defender_input_domain in current model"
            )

        return values_support, values_cont, values_X, values_Y

    def _build_PN_for_domain(self, benign_pool_int, domain_values):
        """
        Build benign prior on a given discrete domain (X or Y).
        Zero probability states are allowed; will be handled by epsilon in KL.
        """
        freq = get_frequency(data=benign_pool_int, assume_contiguous=False)
        total = float(sum(freq.values()))
        if total <= 0:
            raise ValueError("Empty benign frequency when building P_N")

        probs = []
        for v in domain_values:
            probs.append(float(freq.get(v, 0.0)) / total)

        P = np.asarray(probs, dtype=np.float32)
        if P.sum() <= 0:
            # fallback to uniform if extremely degenerate
            P[:] = 1.0 / len(P)
        else:
            P /= P.sum()
        return P

    # --------------------------------------------------
    # Model builders
    # --------------------------------------------------
    def _build_attackers(self, K: int, Nx: int):
        M = int(self.cfg["num_attackers"]) if self.cfg.get("use_attacker_ensemble", False) else 1
        attackers = []
        for _ in range(M):
            attackers.append(
                CovertEncoderMLP(
                    K,
                    Nx,
                    emb_dim=self.cfg["enc_emb_dim"],
                    hidden=self.cfg["enc_hidden"],
                    temperature=self.cfg["enc_init_temp"],
                    fixed_pk=self.cfg["fixed_pk"]
                )
            )
        return attackers

    def _build_defender(self, values_X, values_Y):
        return CovertDefenderMLP(
            values_X,
            values_Y,
            hidden=self.cfg["def_hidden"],
            temperature=self.cfg["def_init_temp"],
            allow_shorten=self.cfg["allow_shorten"],
            band=self.cfg["band"],
            use_context=self.cfg["use_context"],
            enable_mask=self.cfg["enable_mask"],
        )

    def _build_optimizers(self, P_N_X, attackers, defender):
        att_opts = [tf.keras.optimizers.AdamW(self.cfg["lr_attacker"]) for _ in attackers]
        def_opt = tf.keras.optimizers.AdamW(self.cfg["lr_defender"])

        for att, opt in zip(attackers, att_opts):
            _ = att(None, training=False)
            opt.build(att.trainable_variables)

        pn_kw = P_N_X if self.cfg["use_context"] else None
        _ = defender(None, P_N=pn_kw, training=False)
        def_opt.build(defender.trainable_variables)

        return att_opts, def_opt

    # --------------------------------------------------
    # Worst attacker selection
    # --------------------------------------------------
    def _choose_worst_attacker(self, attackers, defender, P_N_X):
        I_vals = []
        triples = []

        pn_kw = P_N_X if self.cfg["use_context"] else None

        for att in attackers:
            P_K, Q = att(None, training=False)
            T = defender(None, P_N=pn_kw, training=False)
            I, P_X, P_Y = compute_mutual_info(P_K, Q, T)
            I_vals.append(I)
            triples.append((P_K, Q, P_X, P_Y, I))

        worst_idx = int(tf.argmax(tf.stack(I_vals)).numpy())
        return worst_idx, triples[worst_idx]

    # --------------------------------------------------
    # Logging and early stop
    # --------------------------------------------------
    def _log_metrics(self, step: int, app: str, logd: dict, filepath_save: str):
        if (step % int(self.cfg["log_every"])) != 0:
            return

        record = {
            "app": app,
            "step": int(step),
            "MI": float(logd["MI"]),
            "Cost": float(logd["Cost"]),
            "KLX": float(logd["KLX"]),
            "KLY": float(logd["KLY"]),
            "HY": float(logd["HY"]),
            "HYK": float(logd["HYK"]),
            "LossAtt": float(logd["LossAtt"]),
            "LossDef": float(logd["LossDef"]),
            "lambda": float(logd.get("lambda", 0.0)),
        }

        lam = record["lambda"]
        print(
            f"[{record['app']:<10} | step {record['step']:5d}]  "
            f"MI={record['MI']:.4f}  "
            f"Cost={record['Cost']:.3f}  "
            f"λ={lam:.3f}  "
            f"L_att={record['LossAtt']:.3f}  "
            f"L_def={record['LossDef']:.3f}  "
            f"KLX={record['KLX']:.3f}  "
            f"KLY={record['KLY']:.3f}"
        )

        with open(filepath_save, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _early_stop_update(self, step: int, loss_def_value: float):
        if not self.cfg.get("use_adaptive_steps", False):
            return False

        alpha = float(self.cfg.get("ema_alpha", 0.1))
        if self._ema_loss_def is None:
            self._ema_loss_def = loss_def_value
        else:
            self._ema_loss_def = alpha * loss_def_value + (1.0 - alpha) * self._ema_loss_def

        min_delta = float(self.cfg.get("earlystop_min_delta", 1e-4))
        min_rel = float(self.cfg.get("earlystop_min_rel", 0.0))

        abs_improve = self._best_def_ema - self._ema_loss_def
        rel_improve = abs_improve / max(abs(self._best_def_ema), 1e-8)
        improved = (abs_improve > min_delta) or (rel_improve > min_rel)

        if improved:
            self._best_def_ema = self._ema_loss_def
            self._no_improve = 0
        else:
            self._no_improve += 1

        min_steps = int(self.cfg.get("min_steps", 0))
        patience = int(self.cfg.get("earlystop_patience", 200))

        if step >= min_steps and self._no_improve >= patience:
            return True
        return False

    # --------------------------------------------------
    # Save models
    # --------------------------------------------------
    def _save_defender_bundle(
            self,
            app: str,
            feature_type: str,
            T: np.ndarray,
            values_X,
            values_Y,
            defender,
            save_dir: str,
    ):
        """
        Save a self-contained defender bundle.

        Semantics (IMPORTANT):
          - T[i][j] = P(Y_index = j | X_index = i)
          - T is row-stochastic
          - values_X / values_Y are discrete indices produced by preprocess_feature_pool
          - raw_values are representative values in original domain, obtained via postprocess
        """
        os.makedirs(save_dir, exist_ok=True)

        # --------------------------------------------------
        # 0. Defensive normalization (row-stochastic)
        #    This does NOT change semantics, only guarantees invariants.
        # --------------------------------------------------
        T = np.asarray(T, dtype=float)
        row_sum = T.sum(axis=1, keepdims=True)
        row_sum[row_sum <= 0] = 1.0
        T = T / row_sum

        # --------------------------------------------------
        # 1. Postprocess index -> raw representative values
        # --------------------------------------------------
        X_raw = postprocess_feature_indices(
            values_X,
            feature_type=feature_type,
            ipd_precision_ms=self.cfg.get("ipd_precision_ms", 1),
            log_ipd=self.cfg.get("log_ipd", False),
        )

        Y_raw = postprocess_feature_indices(
            values_Y,
            feature_type=feature_type,
            ipd_precision_ms=self.cfg.get("ipd_precision_ms", 1),
            log_ipd=self.cfg.get("log_ipd", False),
        )

        # --------------------------------------------------
        # 2. Assemble bundle (NO numeric logic beyond semantics)
        # --------------------------------------------------
        bundle = {
            # identity
            "app": app,
            "feature_type": feature_type,

            # core policy
            # T[i][j] = P(Y_index=j | X_index=i)
            "T": T.tolist(),

            # X / Y semantics
            # index_values: exactly what preprocess_feature_pool outputs
            # raw_values: instantiated original-domain representatives
            "X": {
                "index_values": list(values_X),
                "raw_values": list(X_raw),
            },
            "Y": {
                "index_values": list(values_Y),
                "raw_values": list(Y_raw),
            },

            # preprocess semantics (explicit, reproducible)
            "preprocess": {
                "feature_type": feature_type,
                "log_ipd": bool(self.cfg.get("log_ipd", False)),
                "ipd_precision_ms": int(self.cfg.get("ipd_precision_ms", 1)),
                "max_ipd_ms": self.cfg.get("max_ipd_ms", None),
                "max_size_bytes": self.cfg.get("max_size_bytes", None),
                "max_ipd_p": self.cfg.get("max_ipd_p", None),
                "max_size_p": self.cfg.get("max_size_p", None),
            },

            # defender meta (non-semantic, but useful)
            "defender_meta": {
                "hidden": list(self.cfg["def_hidden"]),
                "allow_shorten": bool(self.cfg["allow_shorten"]),
                "band": self.cfg["band"],
                "use_context": bool(self.cfg["use_context"]),
                "temperature": float(defender.temperature.numpy()),
            },

            # explicit kernel contract (for humans and future code)
            "kernel_semantics": {
                "type": "row_stochastic",
                "meaning": "T[i][j] = P(Y=j | X=i)",
                "shape": [int(len(values_X)), int(len(values_Y))],
            },
        }

        # --------------------------------------------------
        # 3. Save bundle JSON
        # --------------------------------------------------
        bundle_path = os.path.join(
            save_dir, f"{app}_{feature_type}_defender_bundle.json"
        )
        with open(bundle_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False, indent=2)

        # --------------------------------------------------
        # 4. Save defender weights separately
        # --------------------------------------------------
        weights_path = os.path.join(
            save_dir, f"{app}_{feature_type}_defender.weights.h5"
        )
        defender.save_weights(weights_path)

        print(f"[Saved] defender bundle to {bundle_path}")

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    def _training_loop(
            self,
            app,
            attackers,
            defender,
            att_opts,
            def_opt,
            values_X,
            values_Y,
            P_N_X,
            P_N_Y,
            log_path,
    ):
        # --------------------------------------------------
        # Dual variable
        # --------------------------------------------------
        if self.cfg.get("use_dual", False):
            lambda_dual = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        else:
            lambda_dual = tf.constant(0.0, dtype=tf.float32)

        P_N_X_tf = tf.constant(P_N_X, tf.float32)
        P_N_Y_tf = tf.constant(P_N_Y, tf.float32)
        pn_kw = P_N_X if self.cfg.get("use_context", False) else None

        attacker_steps = int(self.cfg.get("attacker_steps", 1))
        defender_steps = int(self.cfg.get("defender_steps", 1))

        # --------------------------------------------------
        # Training loop
        # --------------------------------------------------
        for step in range(int(self.cfg["max_steps"])):

            # ================================
            # 1. Attacker update
            # ================================
            for _ in range(attacker_steps):
                for att, opt in zip(attackers, att_opts):
                    with tf.GradientTape() as tape:
                        P_K, Q = att(None, training=True)
                        T = defender(None, P_N=pn_kw, training=False)

                        I, P_X, _ = compute_mutual_info(P_K, Q, T)

                        loss_att = (
                                -float(self.cfg["alpha"]) * I
                                + float(self.cfg["gamma"]) * kl_divergence(P_X, P_N_X_tf)
                        )

                        if float(self.cfg.get("lambda_ent_Q", 0.0)) > 0:
                            loss_att += float(self.cfg["lambda_ent_Q"]) * col_entropy(Q)
                        if float(self.cfg.get("lambda_row_ent_Q", 0.0)) > 0:
                            loss_att += float(self.cfg["lambda_row_ent_Q"]) * row_entropy(Q)

                    grads = tape.gradient(loss_att, att.trainable_variables)
                    grads = self._clip_grads(grads)
                    grads_vars = [
                        (g, v)
                        for g, v in zip(grads, att.trainable_variables)
                        if g is not None
                    ]
                    if grads_vars:
                        opt.apply_gradients(grads_vars)

            # ================================
            # 2. Choose worst attacker (ONCE)
            # ================================
            _, (P_Kw, Qw, P_Xw, P_Yw, Iw) = self._choose_worst_attacker(
                attackers, defender, P_N_X
            )

            # ================================
            # 3. Defender update (multiple steps)
            # ================================
            for _ in range(defender_steps):
                with tf.GradientTape() as tape:
                    T = defender(None, P_N=pn_kw, training=True)

                    I, P_X, P_Y = compute_mutual_info(P_Kw, Qw, T)
                    cost = expected_relative_cost(T, P_N_X, values_X, values_Y)

                    if self.cfg.get("use_dual", False):
                        loss_def = (
                                float(self.cfg["alpha"]) * I
                                + lambda_dual * (cost - float(self.cfg["cost_cap"]))
                        )
                    else:
                        loss_def = float(self.cfg["alpha"]) * I + float(self.cfg["beta"]) * cost

                    if float(self.cfg.get("lambda_py_kl", 0.0)) > 0:
                        loss_def += float(self.cfg["lambda_py_kl"]) * kl_divergence(
                            P_Y, P_N_Y_tf
                        )

                    if float(self.cfg.get("lambda_ent_T", 0.0)) > 0:
                        loss_def -= float(self.cfg["lambda_ent_T"]) * row_entropy(T)

                    if float(self.cfg.get("lambda_ixy_pn", 0.0)) > 0:
                        loss_def += float(self.cfg["lambda_ixy_pn"]) * mutual_info_XY_under_PN(
                            T, P_N_X
                        )

                grads = tape.gradient(loss_def, defender.trainable_variables)
                grads = self._clip_grads(grads)
                grads_vars = [
                    (g, v)
                    for g, v in zip(grads, defender.trainable_variables)
                    if g is not None
                ]
                if grads_vars:
                    def_opt.apply_gradients(grads_vars)

                # dual update (do every defender step is OK, or you can restrict to last)
                if self.cfg.get("use_dual", False):
                    dual_lr = float(self.cfg["dual_lr"])
                    lambda_dual.assign(
                        tf.maximum(
                            0.0,
                            lambda_dual + dual_lr * (cost - float(self.cfg["cost_cap"]))
                        )
                    )

            # ================================
            # 4. Logging
            # ================================
            if (step % int(self.cfg["log_every"])) == 0:
                logd = {
                    "MI": float(I.numpy()),
                    "Cost": float(cost.numpy()),
                    "KLX": float(kl_divergence(P_X, P_N_X_tf).numpy()),
                    "KLY": float(kl_divergence(P_Y, P_N_Y_tf).numpy()),
                    "HY": float(entropy_vec(P_Y).numpy()),
                    "HYK": float(entropy_vec(P_Y).numpy() - float(I.numpy())),
                    "LossAtt": float(
                        (
                                -float(self.cfg["alpha"]) * Iw
                                + float(self.cfg["gamma"]) * kl_divergence(P_Xw, P_N_X_tf)
                        ).numpy()
                    ),
                    "LossDef": float(loss_def.numpy()),
                    "lambda": float(lambda_dual.numpy())
                    if self.cfg.get("use_dual", False)
                    else 0.0,
                }
                self._log_metrics(step, app, logd, filepath_save=log_path)

            # ================================
            # 5. Early stopping
            # ================================
            if self._early_stop_update(step, float(loss_def.numpy())):
                print(
                    f"[{app}] Early stop at step={step} "
                    f"(EMA(best)={self._best_def_ema:.6f}, "
                    f"EMA(now)={self._ema_loss_def:.6f}, "
                    f"no_improve={self._no_improve})"
                )
                break

            # ================================
            # 6. Temperature annealing
            # ================================
            if self.cfg.get("use_temp_anneal", False):
                self._anneal_temps(attackers, defender)

        # --------------------------------------------------
        # After training
        # --------------------------------------------------
        attacker_states = []
        for att in attackers:
            P_K, Q = att(None, training=False)
            attacker_states.append(
                {
                    "P_K": P_K.numpy(),
                    "Q": Q.numpy(),
                }
            )

        worst_idx, (P_Kw, Qw, _, _, _) = self._choose_worst_attacker(
            attackers, defender, P_N_X
        )

        pn_kw = P_N_X if self.cfg.get("use_context", False) else None
        T = defender(None, P_N=pn_kw, training=False).numpy()
        print(f"T's shape: {T.shape}")

        extra = {
            "attackers": attacker_states,
            "worst_idx": worst_idx,
            "P_K_worst": P_Kw.numpy(),
            "Q_worst": Qw.numpy(),
            "T_defender": T,
        }

        return defender, extra

    # --------------------------------------------------
    # Public entry
    # --------------------------------------------------
    def train_one_app(self, app: str, benign_pool_int: list, save_dir: str, log_path: str = "logs.json"):
        if not benign_pool_int:
            raise ValueError(f"Empty benign_pool after preprocessing for app {app}")

        train_frac = float(self.cfg.get("train_frac", 1.0))
        train_len = int(len(benign_pool_int) * train_frac)
        benign_pool_int = benign_pool_int[:max(1, train_len)]

        values_support, values_cont, values_X, values_Y = self._resolve_domains(benign_pool_int)

        P_N_X = self._build_PN_for_domain(benign_pool_int, values_X)
        P_N_Y = self._build_PN_for_domain(benign_pool_int, values_Y)

        Nx = len(values_X)
        K = int(self.cfg["K"])

        attackers = self._build_attackers(K, Nx)
        defender = self._build_defender(values_X, values_Y)

        att_opts, def_opt = self._build_optimizers(P_N_X, attackers, defender)

        defender, extra = self._training_loop(
            app=app,
            attackers=attackers,
            defender=defender,
            att_opts=att_opts,
            def_opt=def_opt,
            values_X=values_X,
            values_Y=values_Y,
            P_N_X=P_N_X,
            P_N_Y=P_N_Y,
            log_path=log_path,
        )

        self._save_defender_bundle(
            app=app,
            feature_type=self.cfg["feature_type"],
            T=extra["T_defender"],
            values_X=values_X,
            values_Y=values_Y,
            defender=defender,
            save_dir=save_dir,
        )

        return defender


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    print(
        f"training at TF {tf.__version__}, GPU "
        f"{'available' if tf.config.list_physical_devices('GPU') else 'NOT available'}"
    )

    cfg = TrainerConfig()

    print("Config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    trainer = Trainer(cfg)

    for app in cfg.applications:
        benign_path = f"{cfg.traffic_dir}/{cfg.dataset_name}-{app}-parsed.json"
        print(f"=== Training defender for app: {app} ===")

        _, _, benign_pool_float = parse_feature(
            benign_path,
            feature_type=cfg.feature_type,
            direction_key=cfg.direction_key
        )

        benign_pool_int = preprocess_feature_pool(
            benign_pool_float,
            cfg.feature_type,
            ipd_precision_ms=cfg.ipd_precision_ms,
            max_ipd_ms=cfg.max_ipd_ms,
            max_size_bytes=cfg.max_size_bytes,
            max_ipd_p=float(cfg.max_ipd_p),
            max_size_p=float(cfg.max_size_p),
            log_ipd = cfg.log_ipd
        )
        # plot_histogram(benign_pool_int, cfg.feature_type)

        trainer.train_one_app(app, benign_pool_int, cfg.save_dir, log_path="logs.json")
        print(f"Saved defender for {app}.\n")
