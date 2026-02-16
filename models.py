# -*- coding: utf-8 -*-
"""
Models for adversarial covert channel game

Provides
  CovertEncoderMLP
  CovertDefenderMLP
"""

from __future__ import annotations

import tensorflow as tf

keras = tf.keras
layers = tf.keras.layers


def build_mlp(units, activation="gelu", use_ln=True, name=None):
    seq = []
    for i, u in enumerate(units):
        seq.append(layers.Dense(u, name=None if name is None else f"{name}_dense{i}"))
        if use_ln:
            seq.append(layers.LayerNormalization(epsilon=1e-6))
        if i < len(units) - 1:
            if activation == "gelu":
                seq.append(layers.Activation(tf.nn.gelu))
            else:
                seq.append(layers.Activation("relu"))
    return keras.Sequential(seq, name=name)



class CovertEncoderMLP(keras.Model):
    """
    Attacker

    Outputs
      P_K: [K]
      Q:   [K, N] row wise softmax

    Options
      fixed_pk: if True, use uniform P_K and do not train it
    """

    def __init__(
        self,
        K,
        N,
        emb_dim=64,
        hidden=(128, 128),
        temperature=1.0,
        fixed_pk: bool = False,
        name="CovertEncoderMLP",
    ):
        super().__init__(name=name)
        self.K = int(K)
        self.N = int(N)
        self.fixed_pk = bool(fixed_pk)

        self.temperature = tf.Variable(
            float(temperature), trainable=False, dtype=tf.float32
        )

        # --------------------------------------------------
        # Message source prior P_K
        # --------------------------------------------------
        if self.fixed_pk:
            # non-trainable uniform prior
            self._P_K = tf.constant(
                tf.fill([self.K], 1.0 / self.K), dtype=tf.float32
            )
            self.PK_logits = None
        else:
            # learnable prior
            self.PK_logits = self.add_weight(
                "PK_logits",
                shape=(self.K,),
                initializer="zeros",
                trainable=True,
            )
            self._P_K = None

        # --------------------------------------------------
        # Encoder Q
        # --------------------------------------------------
        self.E = self.add_weight(
            "E",
            shape=(self.K, emb_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.mlp = build_mlp(list(hidden) + [self.N], name="enc_mlp")

    def call(self, inputs=None, training=False):
        # P_K
        if self.fixed_pk:
            P_K = self._P_K
        else:
            P_K = tf.nn.softmax(self.PK_logits, axis=0)

        # Q
        Q_logits = self.mlp(self.E, training=training)
        Q = tf.nn.softmax(
            Q_logits / tf.maximum(self.temperature, 1e-6),
            axis=1,
        )

        return P_K, Q


class CovertDefenderMLP(keras.Model):
    """
    Defender

    Output
      T: [Nx, Ny] row-wise stochastic kernel

    Options
      allow_shorten : forbid y < x if False
      band          : |y - x| <= band if not None
      use_context   : use benign prior P_N over X to gate shaping strength
      enable_mask   : whether to apply structural mask at all

    Contract
      If use_context True then P_N (over X domain) must be provided in every call
    """

    def __init__(
        self,
        values_X,
        values_Y=None,
        hidden=(128, 128),
        temperature=1.0,
        allow_shorten=True,
        band=None,
        use_context=False,
        enable_mask=True,
        ctx_dim=16,
        name="CovertDefenderMLP",
    ):
        super().__init__(name=name)

        # domains
        if values_Y is None:
            values_Y = values_X

        self.values_X = tf.constant(list(values_X), dtype=tf.float32)
        self.values_Y = tf.constant(list(values_Y), dtype=tf.float32)

        self.Nx = int(self.values_X.shape[0])
        self.Ny = int(self.values_Y.shape[0])

        self.temperature = tf.Variable(
            float(temperature), trainable=False, dtype=tf.float32
        )

        self.allow_shorten = bool(allow_shorten)
        self.band = band if band is None else int(band)
        self.use_context = bool(use_context)
        self.enable_mask = bool(enable_mask)

        self.ctx_dim = int(ctx_dim)

        # context gating
        if self.use_context:
            self.ctx_proj = layers.Dense(
                self.ctx_dim, activation=tf.nn.gelu, name="ctx_proj"
            )
            self.ctx_gate = layers.Dense(
                1, activation="sigmoid", name="ctx_gate"
            )

        # row wise MLP, operates on X domain
        self.mlp = build_mlp(list(hidden) + [self.Ny], name="def_mlp")

        # ordinal indices for X
        self.idx_X = tf.range(self.Nx, dtype=tf.float32)

    # --------------------------------------------------
    # Row feature construction
    # --------------------------------------------------
    def _features(self):
        # physical value feature (X domain)
        x = self.values_X
        x_norm = (x - tf.reduce_mean(x)) / (tf.math.reduce_std(x) + 1e-6)

        # ordinal index feature (X domain)
        i = self.idx_X
        i_norm = (i - (self.Nx - 1) / 2.0) / ((self.Nx - 1) / 2.0 + 1e-6)

        F = tf.stack([x_norm, i_norm], axis=1)
        return F

    # --------------------------------------------------
    # Structural mask (value based)
    # --------------------------------------------------
    def _build_mask(self):
        mask = tf.ones((self.Nx, self.Ny), dtype=tf.float32)

        x_vals = self.values_X[:, None]   # [Nx, 1]
        y_vals = self.values_Y[None, :]   # [1, Ny]

        if not self.allow_shorten:
            # forbid y < x in value space
            mask = mask * tf.cast(y_vals >= x_vals, tf.float32)

        if self.band is not None:
            band_mask = tf.cast(tf.abs(y_vals - x_vals) <= self.band, tf.float32)
            mask = mask * band_mask

        return mask

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def call(self, inputs=None, P_N=None, training=False):
        if self.use_context and P_N is None:
            raise ValueError("use_context=True requires P_N over X domain")

        # row wise features for X
        F = self._features()
        logits = self.mlp(F, training=training)

        # context gating using P_N over X
        if self.use_context:
            P_N = tf.convert_to_tensor(P_N, dtype=tf.float32)
            ctx = self.ctx_proj(P_N[None, :])   # [1, ctx_dim]
            gate = self.ctx_gate(ctx)           # [1, 1]
            logits = logits * gate              # broadcast to [Nx, Ny]

        # temperature
        logits = logits / tf.maximum(self.temperature, 1e-6)

        # structural mask
        if self.enable_mask:
            mask = self._build_mask()
            logits = logits + (1.0 - mask) * (-1e9)

        T = tf.nn.softmax(logits, axis=1)
        return T

