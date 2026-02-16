# coding: utf8
"""
NCC robustness tester for a trained Covadis defender on a single 1D traffic feature.

What this script does
1 Loads benign traffic feature sequences from parsed json
2 Splits benign into train and test
3 Loads a trained defender model per application
4 Embeds NCC into benign train feature pool
5 Applies a disruptor to both NCC embedded features and benign test features
6 Measures
   Mutual information in bits per symbol between covert codes and observed feature
   Decoding accuracy after disruption
   Relative cost on NCC embedded stream and on benign test stream

Dependencies
numpy
tensorflow
scikit learn
Your local modules:
  ncc.py must provide Encoder Embedder Extractor
  models.py must provide CovertDefenderMLP
"""

from __future__ import annotations

import os
import json
import time
import bisect
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import traceback
import numpy as np
import tensorflow as tf
from sklearn.metrics.cluster import contingency_matrix

from collections import defaultdict
from config import TesterConfig
from nccs import *
from models import CovertDefenderMLP
from data_utils import (
    get_frequency,
    parse_feature,
    preprocess_feature_pool,
    check_accuracy,
)

# ============================================================
# IO and preprocessing utilities
# ============================================================

def build_PN_for_context(dataset_int: List[int], domain_values: List[int]) -> List[float]:
    """
    Build benign prior over a given discrete domain.
    """
    from collections import Counter
    c = Counter(int(x) for x in dataset_int)

    total = 0
    for v in domain_values:
        total += c.get(int(v), 0)

    if total <= 0:
        return [1.0 / len(domain_values)] * len(domain_values)

    return [c.get(int(v), 0) / total for v in domain_values]


def try_load_N_values(model_dir: str, app: str, feature_type: str) -> Optional[List[int]]:
    """
    Legacy helper (currently unused in main flow).
    Try multiple legacy file names.
    Returns None if not found.
    """
    candidates = [
        os.path.join(model_dir, f"{app}_{feature_type}s.json"),
        os.path.join(model_dir, f"{app}_sizes.json"),
        os.path.join(model_dir, f"{app}_N_values.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            nv = obj.get("N_values", None)
            if nv is None:
                continue
            return [int(x) for x in list(nv)]
    return None


def pretty_print_results(results, group_by="ncc"):
    """
    Pretty print results as aligned tables.
    """

    assert group_by in ("ncc", "disruptor")

    COL_W = {
        "app": 18,
        "row": 10,
        "dmi": 8,
        "dmi_pct": 7,
        "mi_after": 10,
        "acc": 7,
        "cost": 9,
    }

    def fmt(x, w):
        return f"{str(x):>{w}}"

    def delta_mi_ratio(mi_embed, mi_after):
        if mi_embed <= 0:
            return 0.0
        return 100.0 * (mi_embed - mi_after) / mi_embed

    grouped = defaultdict(list)

    if group_by == "ncc":
        key_name = "ncc_method"
        row_name = "disruptor"
    else:
        key_name = "disruptor"
        row_name = "ncc_method"

    for r in results:
        grouped[r[key_name]].append(r)

    for key, rows in grouped.items():
        print("\n" + "=" * 110)
        print(f"{key_name}: {key}")
        print("=" * 110)

        header = (
            f"{fmt('app', COL_W['app'])} | "
            f"{fmt(row_name, COL_W['row'])} | "
            f"{fmt('ΔMI', COL_W['dmi'])} | "
            f"{fmt('ΔMI%', COL_W['dmi_pct'])} | "
            f"{fmt('MI_after', COL_W['mi_after'])} | "
            f"{fmt('acc', COL_W['acc'])} | "
            f"{fmt('cost_avg', COL_W['cost'])}"
        )
        print(header)
        print("-" * len(header))

        rows = sorted(rows, key=lambda x: (x["application"], x[row_name]))

        for r in rows:
            mi_embed = r["MI_embed_bits_per_symbol"]
            mi_after = r["MI_perturb_bits_per_symbol"]

            dmi = mi_embed - mi_after
            dmi_pct = delta_mi_ratio(mi_embed, mi_after)

            s_dmi = f"{dmi:.4f}"
            s_dmi_pct = f"{dmi_pct:.1f}%"
            s_mi_after = f"{mi_after:.4f}"
            s_acc = f"{r['acc_after']:.4f}"
            s_cost = f"{r['cost_avg']:.2f}"

            print(
                f"{fmt(r['application'], COL_W['app'])} | "
                f"{fmt(r[row_name], COL_W['row'])} | "
                f"{fmt(s_dmi, COL_W['dmi'])} | "
                f"{fmt(s_dmi_pct, COL_W['dmi_pct'])} | "
                f"{fmt(s_mi_after, COL_W['mi_after'])} | "
                f"{fmt(s_acc, COL_W['acc'])} | "
                f"{fmt(s_cost, COL_W['cost'])}"
            )

# ============================================================
# Metrics
# ============================================================

def mi_bits_per_symbol(x: List[int], y: List[int]) -> float:
    """
    Mutual information I(X Y) in bits per symbol for two equal length discrete sequences.
    """
    if len(x) != len(y) or len(x) == 0:
        return 0.0

    C = contingency_matrix(x, y)
    n = float(C.sum())
    if n <= 0:
        return 0.0

    Pxy = C / n
    Px = C.sum(axis=1) / n
    Py = C.sum(axis=0) / n

    eps = 1e-12
    nz = Pxy > 0
    logPxy = np.zeros_like(Pxy, dtype=np.float64)
    logPxy[nz] = np.log2(Pxy[nz] + eps)

    logPx = np.log2(Px + eps)
    logPy = np.log2(Py + eps)
    logPxPy = logPx[:, None] + logPy[None, :]

    mi = np.sum(Pxy[nz] * (logPxy[nz] - logPxPy[nz]))
    return float(mi)


def relative_cost_mean(original: List[int], perturbed: List[int]) -> float:
    """
    Mean relative increase max((p - o) / o, 0) with safe handling of zero.
    """
    if len(original) == 0 or len(original) != len(perturbed):
        return 0.0
    acc = 0.0
    cnt = 0
    for o, p in zip(original, perturbed):
        oo = float(o)
        pp = float(p)
        if oo <= 0:
            continue
        d = (pp - oo) / oo
        if d < 0:
            d = 0.0
        acc += d
        cnt += 1
    if cnt == 0:
        return 0.0
    return float(acc / cnt)


# ============================================================
# Perturbation utilities
# ============================================================

def compute_ditto_pattern(benign_pool: List[int], pattern_length: int) -> List[int]:
    percentiles = [(i + 1) * 100.0 / float(pattern_length) for i in range(pattern_length)]
    arr = np.asarray(benign_pool, dtype=float)
    try:
        pattern = np.percentile(arr, percentiles, method="higher")
    except TypeError:
        pattern = np.percentile(arr, percentiles, interpolation="higher")
    return sorted(set(int(p) for p in pattern))


def normalize_row_stochastic_numpy(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    row_sum = T.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    return T / row_sum

def normalize_column_stochastic_numpy(T: np.ndarray) -> np.ndarray:
    """
    Normalize columns so that sum_j T[j, i] = 1 for each i.
    """
    T = np.asarray(T, dtype=float)
    col_sum = T.sum(axis=0, keepdims=True)
    col_sum[col_sum <= 0] = 1.0
    return T / col_sum


def normalize_upper_triangular_numpy(T: np.ndarray, ensure_upper: bool) -> np.ndarray:
    """
    Make row stochastic for square kernels.
    Optionally enforce upper triangle only.
    If a row becomes all zero, fall back to diagonal if allowed.
    """
    T = np.asarray(T, dtype=float)
    N = T.shape[0]

    if ensure_upper:
        allowed = np.triu(np.ones_like(T, dtype=bool), k=0)
    else:
        allowed = np.ones_like(T, dtype=bool)

    T = np.where(allowed, np.maximum(T, 0.0), 0.0)

    row_sum = T.sum(axis=1)
    T_norm = np.zeros_like(T)
    denom = row_sum[:, None]
    np.divide(T, denom, out=T_norm, where=denom > 0)

    zero_rows = np.where(row_sum <= 0)[0]
    if zero_rows.size > 0:
        diag_allowed = allowed[zero_rows, zero_rows]
        zr_diag = zero_rows[diag_allowed]
        if zr_diag.size > 0:
            T_norm[zr_diag, zr_diag] = 1.0

        zr_nondiag = zero_rows[~diag_allowed]
        for i in zr_nondiag:
            cols = np.where(allowed[i])[0]
            if cols.size > 0:
                T_norm[i, cols[0]] = 1.0

    return T_norm



def _map_value_to_N_index_ceiling(N_values: List[int], v: int) -> int:
    """
    Map a value v to an index j such that N_values[j] is the smallest value >= v.
    If all values are < v, return last index.
    """
    j = bisect.bisect_left(N_values, int(v))
    if j >= len(N_values):
        j = len(N_values) - 1
    if j < 0:
        j = 0
    return int(j)


def generate_perturbation_matrix(
    method: str,
    N_values: List[int],
    defender: Optional[CovertDefenderMLP] = None,
    P_N_ctx: Optional[List[float]] = None,
    fixed_value: Optional[int] = None,
    ditto_pattern: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Returns row stochastic matrix T over N_values in index space.

    For non model methods this is square [N, N].

    method supports
      model         (not used in this helper anymore)
      random_discrete
      fixed
      ditto
    """
    if method == "model":
        raise ValueError("model method is handled directly in ModelDisruptor.apply")

    N = len(N_values)
    T = np.zeros((N, N), dtype=np.float64)

    if method == "random_discrete":
        for i in range(N):
            valid = N - i
            if valid <= 0:
                continue
            T[i, i:] = 1.0 / float(valid)

    elif method == "fixed":
        if fixed_value is None:
            fixed_value = int(max(N_values))
        j = _map_value_to_N_index_ceiling(N_values, int(fixed_value))
        T[:, j] = 1.0

    elif method == "ditto":
        if ditto_pattern is None:
            raise ValueError("ditto_pattern is required for method ditto")
        maxN = int(max(N_values))
        for i, v in enumerate(N_values):
            targets = [p for p in ditto_pattern if int(p) >= int(v)]
            chosen_val = maxN if len(targets) == 0 else int(min(targets))
            j = _map_value_to_N_index_ceiling(N_values, chosen_val)
            if int(N_values[j]) < int(v):
                j = _map_value_to_N_index_ceiling(N_values, int(v))
            T[i, j] = 1.0

    else:
        raise ValueError(f"unknown perturbation method: {method}")

    return normalize_row_stochastic_numpy(T)


def apply_perturbation(
    embedded_feature: List[int],
    T: np.ndarray,
    N_values: List[int],
    value_to_index: Dict[int, int],
    rng: np.random.Generator,
    oov_scale_K: int = 1,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Legacy perturbation in a single domain N_values.

    In domain values use row distribution in T.
    Out of domain values are mapped conservatively so feature does not decrease.
    Returns
      embedded_indices
      perturbed_indices
      perturbed_feature
    """
    max_val = int(max(N_values))
    N = len(N_values)

    embedded_idx = []
    perturbed_idx = []
    perturbed_feature = []

    for val in embedded_feature:
        v = int(val)

        if v in value_to_index:
            i = value_to_index[v]
            j = int(rng.choice(T.shape[1], p=T[i]))
            new_val = int(N_values[j])

            embedded_idx.append(i)
            perturbed_idx.append(j)
            perturbed_feature.append(new_val)
            continue

        if v <= max_val:
            new_val = max_val
        else:
            r = int(rng.integers(1, max(int(oov_scale_K), 1) + 1))
            new_val = int(v * r)

        i0 = bisect.bisect_left(N_values, v)
        if i0 >= N:
            i0 = N - 1
        j0 = bisect.bisect_left(N_values, new_val)
        if j0 >= N:
            j0 = N - 1

        embedded_idx.append(int(i0))
        perturbed_idx.append(int(j0))
        perturbed_feature.append(int(new_val))

    return embedded_idx, perturbed_idx, perturbed_feature


def _apply_state_disruption(
    feature_seq: List[int],
    T: np.ndarray,
    N_values: List[int],
    value_to_index: Dict[int, int],
    rng: np.random.Generator,
    *,
    allow_shorten: bool,
) -> List[int]:
    """
    Legacy disruption in a single domain N_values.
    """
    ensure_upper = bool(not allow_shorten)
    T2 = normalize_upper_triangular_numpy(T, ensure_upper=ensure_upper)
    _, _, perturbed = apply_perturbation(feature_seq, T2, N_values, value_to_index, rng)
    return perturbed

def map_value_to_nearest_index(
    values: List[int],
    v: int,
) -> int:
    """
    Map v to nearest index in sorted values.
    Assumes values is non-empty and sorted.
    """
    if v <= values[0]:
        return 0
    if v >= values[-1]:
        return len(values) - 1

    j = bisect.bisect_left(values, v)
    if j == 0:
        return 0
    if j >= len(values):
        return len(values) - 1

    left = values[j - 1]
    right = values[j]
    if abs(v - left) <= abs(right - v):
        return j - 1
    else:
        return j


def _apply_state_disruption_model_bundle(
    feature_seq: List[int],
    T: np.ndarray,
    X_values: List[int],
    Y_values: List[int],
    x_value_to_index: Dict[int, int],
    rng: np.random.Generator,
) -> List[int]:
    """
    Apply defender bundle kernel.

    Semantics (CORRECT):
      T[i][j] = P(Y=j | X=i)
      For each input X index i, sample Y from row i.
    """
    T2 = normalize_row_stochastic_numpy(T)
    print(f"disruption matrix shape: {T2.shape}")

    Xv = list(X_values)
    Yv = list(Y_values)

    Nx = len(Xv)
    Ny = len(Yv)

    x_max = Xv[-1]

    out = []

    for v in feature_seq:
        v = int(v)

        # resolve X index
        if v in x_value_to_index:
            i = x_value_to_index[v]
        elif v <= x_max:
            i = map_value_to_nearest_index(Xv, v)
        else:
            i = Nx - 1  # true OOV

        # --- CORRECT: sample from ROW i ---
        p = T2[i, :]
        s = p.sum()
        if s <= 0:
            # safe fallback: identity / nearest
            p = np.zeros((Ny,), dtype=float)
            p[min(i, Ny - 1)] = 1.0
        else:
            p = p / s

        j = int(rng.choice(Ny, p=p))
        out.append(int(Yv[j]))

    return out


# ============================================================
# Disruptor refactor
# ============================================================

# @dataclass
# class DisruptContext:
#     cfg: TesterConfig
#     rng: np.random.Generator
#     defender: CovertDefenderMLP
#     benign_train: List[int]
#     X_values: List[int]
#     Y_values: List[int]
#     x_value_to_index: Dict[int, int]
#     y_value_to_index: Dict[int, int]

@dataclass
class DisruptContext:
    cfg: TesterConfig
    rng: np.random.Generator
    bundle: Dict[str, Any]
    X_values: List[int]
    Y_values: List[int]
    x_value_to_index: Dict[int, int]
    y_value_to_index: Dict[int, int]

    benign_train: List[int]



class DisruptorBase:
    name: str

    def apply(self, feature_seq: List[int], ctx: DisruptContext) -> List[int]:
        raise NotImplementedError

def apply_wang_random_perturbation(
    feature_seq: List[int],
    pct: int,
    rng: np.random.Generator,
) -> List[int]:
    """
    Wang style random perturbation:
    multiply by (1 + u), where u ~ Uniform[0, pct] percent.

    This operates directly in value domain and ignores defender kernel.
    """
    if pct <= 0:
        return list(feature_seq)

    arr = np.asarray(feature_seq, dtype=float)
    randk = rng.integers(0, int(pct) + 1, size=arr.shape)
    out = arr * (1.0 + randk.astype(np.float64) / 100.0)
    return np.rint(out).astype(int).tolist()


class WangRandomDisruptor(DisruptorBase):
    """
    Backward compatible with previous behavior:
    any disruptor string starting with "random" and carrying a numeric suffix
    uses Wang style multiplicative noise in value domain.
    """
    name = "wang_random"

    def __init__(self, pct: int):
        self.pct = int(pct)

    def apply(self, feature_seq: List[int], ctx: DisruptContext) -> List[int]:
        return apply_wang_random_perturbation(feature_seq, int(self.pct), ctx.rng)


class DiscreteRandomDisruptor(DisruptorBase):
    name = "random_discrete"

    def apply(self, feature_seq: List[int], ctx: DisruptContext) -> List[int]:
        T = generate_perturbation_matrix(
            method="random_discrete",
            N_values=ctx.Y_values,
        )
        return _apply_state_disruption(
            feature_seq,
            T,
            ctx.Y_values,
            ctx.y_value_to_index,
            ctx.rng,
            allow_shorten=bool(ctx.bundle.get("allow_shorten", True))
        )


class FixedDisruptor(DisruptorBase):
    name = "fixed"

    def __init__(self, fixed_value: Optional[int] = None):
        self.fixed_value = fixed_value

    def apply(self, feature_seq: List[int], ctx: DisruptContext) -> List[int]:
        fixed_value = int(max(ctx.Y_values)) if self.fixed_value is None else int(self.fixed_value)
        T = generate_perturbation_matrix(
            method="fixed",
            N_values=ctx.Y_values,
            fixed_value=fixed_value,
        )
        return _apply_state_disruption(
            feature_seq,
            T,
            ctx.Y_values,
            ctx.y_value_to_index,
            ctx.rng,
            allow_shorten= bool(ctx.bundle.get("allow_shorten", True))
        )


class DittoDisruptor(DisruptorBase):
    name = "ditto"

    def apply(self, feature_seq: List[int], ctx: DisruptContext) -> List[int]:
        pattern = compute_ditto_pattern(ctx.benign_train, int(ctx.cfg.ditto_pattern_length))
        T = generate_perturbation_matrix(
            method="ditto",
            N_values=ctx.Y_values,
            ditto_pattern=pattern,
        )
        return _apply_state_disruption(
            feature_seq,
            T,
            ctx.Y_values,
            ctx.y_value_to_index,
            ctx.rng,
            allow_shorten= bool(ctx.bundle.get("allow_shorten", True))
        )


class ModelDisruptor(DisruptorBase):
    name = "model"

    def apply(self, feature_seq: List[int], ctx: DisruptContext) -> List[int]:
        bundle = ctx.bundle

        T = bundle["T"]
        X_values = ctx.X_values
        Y_values = ctx.Y_values

        return _apply_state_disruption_model_bundle(
            feature_seq=feature_seq,
            T=T,
            X_values=X_values,
            Y_values=Y_values,
            x_value_to_index=ctx.x_value_to_index,
            rng=ctx.rng,
        )



def _parse_disruptor_spec(name: str) -> DisruptorBase:
    """
    Compatibility rules
    1 If name starts with "random" and includes an integer suffix, use WangRandomDisruptor(pct).
      Examples: random_20, random-20, random 20
    2 If name equals "random", keep old behavior and default pct = 20.
    3 If name equals "random_discrete" or "discrete_random", use DiscreteRandomDisruptor.
    4 model, fixed, ditto keep their meanings.
    """
    s = str(name).strip().lower()

    if s in ("random_discrete", "discrete_random"):
        return DiscreteRandomDisruptor()

    if s.startswith("random"):
        tail = s.replace("random_", "random ").replace("random-", "random ").split()
        if len(tail) >= 2:
            try:
                pct = int(tail[-1])
            except Exception:
                pct = 20
        else:
            pct = 20
        return WangRandomDisruptor(pct=pct)

    if s == "model":
        return ModelDisruptor()
    if s == "fixed":
        return FixedDisruptor()
    if s == "ditto":
        return DittoDisruptor()

    raise ValueError(f"unknown disruptor: {name}")


# ============================================================
# Tester
# ============================================================

class Tester:
    def __init__(self, cfg: TesterConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(int(cfg.seed))

        if cfg.P_K_values == (None,):
            pk = [1.0 / float(cfg.K)] * int(cfg.K)
            self.cfg.P_K_values = (pk,)

    def build_instructions(self) -> List[Dict[str, Any]]:
        out = []
        for app in self.cfg.applications:
            for ncc_method in self.cfg.ncc_methods:
                for disruptor in self.cfg.disruptor_methods:
                    for P_K in self.cfg.P_K_values:
                        out.append(
                            {
                                "application": app,
                                "ncc_method": ncc_method,
                                "disruptor": disruptor,
                                "P_K": list(P_K),
                            }
                        )
        return out

    def load_benign_feature_pool(self, app: str) -> List[int]:
        path = os.path.join(self.cfg.traffic_dir, f"{self.cfg.dataset_name}-{app}-parsed.json")
        _, _, pool_float = parse_feature(
            path,
            feature_type=self.cfg.feature_type,
            direction_key=self.cfg.direction_key,
        )

        if len(pool_float) == 0:
            raise ValueError(f"empty benign pool for app {app}")
        return pool_float

    def split_train_test(self, pool_int: List[int]) -> Tuple[List[int], List[int]]:
        n = len(pool_int)
        n_train = int(float(n) * float(self.cfg.train_frac))
        if n_train < 1:
            n_train = 1
        if n_train >= n:
            n_train = n - 1
        return pool_int[:n_train], pool_int[n_train:]

    def load_defender(
        self,
        app: str,
        feature_type: str,
        benign_train_int: List[int],
    ) -> Tuple[CovertDefenderMLP, List[int], List[int]]:
        cfg_path = os.path.join(self.cfg.model_dir, f"{app}_{feature_type}_defender.config.json")
        weights_path = os.path.join(self.cfg.model_dir, f"{app}_{feature_type}_defender.weights.h5")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(cfg_path)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(weights_path)

        with open(cfg_path, "r", encoding="utf-8") as f:
            dcfg = json.load(f)

        # Legacy only N_values
        if "X_values" in dcfg and "Y_values" in dcfg:
            X_values = [int(x) for x in dcfg["X_values"]]
            Y_values = [int(x) for x in dcfg["Y_values"]]
        else:
            N_values_saved = [int(x) for x in list(dcfg["N_values"])]
            X_values = N_values_saved
            Y_values = N_values_saved

        defender = CovertDefenderMLP(
            X_values,
            Y_values,
            hidden=tuple(dcfg["hidden"]),
            temperature=float(dcfg["temperature"]),
            allow_shorten=bool(dcfg["allow_shorten"]),
            band=dcfg["band"],
            use_context=bool(dcfg["use_context"]),
            enable_mask=bool(dcfg.get("enable_mask", True)),
        )

        if bool(dcfg["use_context"]):
            P_N_ctx = build_PN_for_context(benign_train_int, X_values)
            pn = tf.constant(P_N_ctx, tf.float32)
        else:
            pn = None

        _ = defender(None, P_N=pn, training=False)
        defender.load_weights(weights_path)

        return defender, X_values, Y_values

    def load_defender_bundle(self, app: str) -> Dict[str, Any]:
        """
        Load defender bundle as the single source of truth.
        """
        path = os.path.join(
            self.cfg.model_dir,
            f"{app}_{self.cfg.feature_type}_defender_bundle.json",
        )
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        with open(path, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        # basic sanity checks
        assert "T" in bundle
        assert "X" in bundle and "Y" in bundle

        bundle["T"] = np.asarray(bundle["T"], dtype=float)
        bundle["X"]["raw_values"] = [int(v) for v in bundle["X"]["raw_values"]]
        bundle["Y"]["raw_values"] = [int(v) for v in bundle["Y"]["raw_values"]]
        # print(f"X length = {len(bundle['X']['raw_values'])}")
        # print(f"Y length = {len(bundle['Y']['raw_values'])}")

        return bundle

    def run_one(
        self,
        app: str,
        ncc_method: str,
        disruptor: str,
        P_K: List[float],
    ) -> Dict[str, Any]:
        """
        NCC experiment runner based on the new nccs framework.
        """

        # 1 Load benign
        benign_pool_float = self.load_benign_feature_pool(app)
        benign_pool_int = preprocess_feature_pool(
            benign_pool_float,
            feature_type = self.cfg.feature_type,
            ipd_precision_ms = self.cfg.ipd_precision_ms,
            max_ipd_ms = None,
            max_size_bytes = None,
            max_ipd_p = None,
            max_size_p = None,
            log_ipd = False
        )
        benign_train, benign_test = self.split_train_test(benign_pool_int)

        # 2 Load defender with two domains
        bundle = self.load_defender_bundle(app)

        X_values = bundle["X"]["raw_values"]
        Y_values = bundle["Y"]["raw_values"]

        x_value_to_index = {v: i for i, v in enumerate(X_values)}
        y_value_to_index = {v: i for i, v in enumerate(Y_values)}
        Y_set = set(Y_values)


        # 3 Encoder
        encoder_cfg = EncoderConfig(
            mode="sampling",
            sampling=SamplingConfig(
                pk_mode="manual",
                K=int(self.cfg.K),
                L=int(self.cfg.L),
                P_K_manual=P_K,
            ),
            seed=int(self.cfg.seed),
        )
        encoder = Encoder(encoder_cfg)
        codes = encoder.encode(bits=None)

        # 4 Embedder and Extractor
        feat_quantizer = FeatureQuantizerConfig(
            enable=True,
            scale=1.0,
            method="round",
        )

        embedder_cfg = EmbedderConfig(
            method=ncc_method,
            K=int(self.cfg.K),
            feat_quantizer=feat_quantizer,
        )
        extractor_cfg = ExtractorConfig(
            method=ncc_method,
            K=int(self.cfg.K),
            feat_quantizer=feat_quantizer,
        )

        embedder = Embedder(embedder_cfg)
        extractor = Extractor(extractor_cfg)

        # 5 NCC embedding
        embedded_feature = embedder.embed(codes, benign_train)
        embedded_feature = [int(v) for v in embedded_feature]

        extracted_codes = extractor.extract(embedded_feature, benign_train)
        embed_acc = check_accuracy(codes, extracted_codes)

        oov_embed = sorted(set(embedded_feature) - Y_set)
        oov_test = sorted(set(benign_test) - Y_set)

        # 6 Disruptor
        ctx = DisruptContext(
            cfg=self.cfg,
            rng=self.rng,
            bundle=bundle,
            X_values=X_values,
            Y_values=Y_values,
            x_value_to_index=x_value_to_index,
            y_value_to_index=y_value_to_index,
            benign_train=benign_train,
        )

        disruptor_obj = _parse_disruptor_spec(disruptor)

        perturbed_feature = disruptor_obj.apply(embedded_feature, ctx)
        perturbed_test_feature = disruptor_obj.apply(benign_test, ctx)

        # 8 decode after perturbation
        extracted_codes_after = extractor.extract(perturbed_feature, benign_train)
        acc_after = check_accuracy(codes, extracted_codes_after)

        # 9 metrics
        mi_embed = mi_bits_per_symbol(
            [int(c) for c in codes],
            [int(v) for v in embedded_feature],
        )
        mi_perturb = mi_bits_per_symbol(
            [int(c) for c in codes],
            [int(v) for v in perturbed_feature],
        )

        cost_cc = relative_cost_mean(embedded_feature, perturbed_feature)
        cost_benign = relative_cost_mean(benign_test, perturbed_test_feature)

        return {
            "application": app,
            "ncc_method": ncc_method,
            "disruptor": disruptor,
            "K": int(self.cfg.K),
            "P_K": list(P_K),
            "L": int(self.cfg.L),

            "embed_accuracy": float(embed_acc),
            "acc_after": float(acc_after),

            "MI_embed_bits_per_symbol": float(mi_embed),
            "MI_perturb_bits_per_symbol": float(mi_perturb),

            "cost_cc": float(cost_cc),
            "cost_benign": float(cost_benign),
            "cost_avg": float(0.5 * (cost_cc + cost_benign)),

            "oov_embed": len(oov_embed),
            "oov_test": len(oov_test),
        }

    def run_all(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        instructions = self.build_instructions()

        print(f"Running {len(instructions)} test cases")

        for inst in instructions:
            print(
                f"[Test] app={inst['application']} "
                f"ncc={inst['ncc_method']} "
                f"disruptor={inst['disruptor']} "
                f"P_K={inst['P_K']}"
            )
            try:
                res = self.run_one(
                    app=inst["application"],
                    ncc_method=inst["ncc_method"],
                    disruptor=inst["disruptor"],
                    P_K=inst["P_K"],
                )
                results.append(res)
                print("  -> OK:", {
                    "MI_p": round(res["MI_perturb_bits_per_symbol"], 4),
                    "acc": round(res["acc_after"], 4),
                    "cost": round(res["cost_avg"], 4),
                })
            except Exception:
                traceback.print_exc()

        return results

    def save_results(self, results: List[Dict[str, Any]]):

        if self.cfg.results_dir is None:
            self.cfg.results_dir = "out"

        os.makedirs(self.cfg.results_dir, exist_ok=True)

        dataset = self.cfg.dataset_name
        feature = self.cfg.feature_type
        K = int(self.cfg.K)
        L = int(self.cfg.L)

        timestamp = time.strftime("%Y%m%d-%H%M%S")

        filename = (
            f"results_{dataset}_"
            f"{feature}_"
            f"K{K}_L{L}_"
            f"{timestamp}.json"
        )

        results_path = os.path.join(self.cfg.results_dir, filename)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {results_path}")

    def load_results(self, results_path: str) -> List[Dict[str, Any]]:
        """
        Load experiment results saved by save_results().
        """
        if not os.path.exists(results_path):
            raise FileNotFoundError(results_path)

        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        if not isinstance(results, list):
            raise ValueError("Loaded results is not a list")

        if results and not isinstance(results[0], dict):
            raise ValueError("Loaded results entries are not dicts")

        return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    cfg = TesterConfig()

    print("Tester config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    tester = Tester(cfg)
    results = tester.run_all()

    pretty_print_results(results, group_by="disruptor")
    tester.save_results(results)
