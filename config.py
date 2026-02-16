# -*- coding: utf-8 -*-
"""
Unified configuration for Covadis training and testing.

Design principles
- Single source of truth for feature and data semantics
- Trainer and Tester share the same base configuration
- Clear separation of concerns:
  Data / Feature / Model / Training / Testing / Evaluation
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


# ============================================================
# Base shared configuration
# ============================================================

@dataclass
class BaseConfig:
    """
    Configuration shared by both trainer and tester.
    Any change here affects BOTH training and evaluation semantics.
    """

    # --------------------------------------------------------
    # Dataset and IO
    # --------------------------------------------------------
    traffic_dir: str = "traffic"
    dataset_name: str = "ISCX"

    # --------------------------------------------------------
    # Feature definition (CRITICAL: must be consistent)
    # --------------------------------------------------------
    feature_type: str = "ipd"                 # "ipd" or "size"
    direction_key: str = "ip_direction"       # "pair_direction" | "ip_direction" | "direction"

    # Preprocessing
    ipd_precision_ms: int = 1                # bin size in ms
    max_ipd_ms: int = 5_000                   # hard clip for safety. Larger feature values are not disrupted
    max_ipd_p: float = 100.0
    max_size_bytes: Optional[int] = 2_000
    max_size_p: float = 100.0
    log_ipd: bool = True

    # Attacker output domain
    # "support": only values observed in benign data (S)
    # "continuous": contiguous filled numeric range (D)
    attacker_domain: str = "support"  # "support" | "continuous"

    # Defender input domain (what defender can observe)
    # Normally should be "support"
    defender_input_domain: str = "support"  # "support" | "continuous"

    # Defender output domain (what defender can shape to)
    # Usually "continuous"
    defender_output_domain: str = "continuous"  # "support" | "continuous"

    # --------------------------------------------------------
    # Dataset split
    # --------------------------------------------------------
    train_frac: float = 0.5                   # fraction of benign pool used for training

    # --------------------------------------------------------
    # Symbolic NCC setup
    # --------------------------------------------------------
    K: int = 2                                # number of covert symbols
    L: int = 10_000                           # covert sequence length
    seed: int = 42                            # global random seed

    # --------------------------------------------------------
    # Feature quantization during testing
    # --------------------------------------------------------
    precision: int = 1                        # rounding granularity (should usually be 1)

    # --------------------------------------------------------
    # Applications under test
    # --------------------------------------------------------
    applications: Tuple[str, ...] = (
        "messaging",
        # "audio_streaming",
        # "video_streaming",
        # "browsing",
    )


# ============================================================
# Trainer configuration
# ============================================================

@dataclass
class TrainerConfig(BaseConfig):
    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    save_dir: str = "Covadis_model"
    # Directory to save trained defender weights and metadata
    # One defender per application will be stored here.

    # --------------------------------------------------------
    # Game objective weights
    # --------------------------------------------------------
    alpha: float = 3.0
    # Weight of mutual information I(K;Y)
    # Appears in:
    #   - attacker loss:   maximize  I(K;Y)
    #   - defender loss:   minimize  I(K;Y)
    # Larger alpha => stronger emphasis on secrecy against NCC

    beta: float = 1.0
    # Weight of defender cost penalty (when NOT using dual)
    # Larger beta => more conservative traffic shaping

    gamma: float = 0.1
    # Weight of KL(P_X || P_N) in attacker loss
    # Penalizes attacker for deviating from benign marginal
    # Larger gamma => attacker forced to stay stealthier

    lambda_py_kl: float = 3.0
    # Weight of KL(P_Y || P_N) in defender loss
    # Encourages shaped output distribution to resemble benign traffic
    # Controls statistical stealth of the defender

    lambda_ixy_pn: float = 1.0
    # Weight of mutual information I(X;Y) under benign prior P_N
    # Penalizes defender mappings that introduce unnecessary structure
    # Helps avoid unnatural deterministic shaping

    # --------------------------------------------------------
    # Dual constraint (cost-constrained optimization)
    # --------------------------------------------------------
    use_dual: bool = True
    # If True, defender minimizes:
    #   alpha * I(K;Y) + lambda * (cost - cost_cap)
    # instead of using fixed beta * cost

    cost_cap: float = 0.60
    # Upper bound on expected relative cost of shaping
    # Interpreted as a soft deployment budget

    dual_lr: float = 0.01
    # Learning rate for updating the dual variable lambda
    # Larger values enforce the cost constraint more aggressively

    # --------------------------------------------------------
    # Attacker (Encoder) MLP
    # --------------------------------------------------------
    enc_emb_dim: int = 8
    # Embedding dimension for discrete symbols in attacker encoder

    enc_hidden: Tuple[int, ...] = (16, 16)
    # Hidden layer sizes of attacker MLP

    enc_init_temp: float = 1.0
    # Initial temperature for Gumbel-softmax sampling in attacker
    # Higher => smoother, more stochastic encoding

    fixed_pk: bool = False
    # If True, P_K is fixed at uniform distribution

    # --------------------------------------------------------
    # Defender MLP
    # --------------------------------------------------------
    def_hidden: Tuple[int, ...] = (64, 64, 32)
    # Hidden layer sizes of defender MLP

    def_init_temp: float = 1.0
    # Initial temperature for defender's soft routing matrix T

    allow_shorten: bool = False
    # If False, defender is forbidden to map x -> y where y < x
    # Enforces no packet shortening (only padding or delaying)

    band: Optional[int] = None
    # If not None, enforces |y - x| <= band
    # Limits how far defender can reshape each symbol

    use_context: bool = True
    # If True, defender conditions on benign prior P_N
    # Enables context-aware shaping rather than a fixed kernel

    enable_mask: bool = True
    # If True, all feature values can only be increased

    # --------------------------------------------------------
    # Regularization
    # --------------------------------------------------------
    lambda_ent_Q: float = 1e-4
    # Entropy regularization on columns of Q (attacker)
    # Prevents encoding collapse to a single symbol

    lambda_row_ent_Q: float = 0.0
    # Entropy regularization on rows of Q
    # Encourages each message symbol to spread over outputs

    lambda_ent_T: float = 1e-3
    # Entropy regularization on defender kernel T
    # Encourages stochastic rather than deterministic shaping

    # --------------------------------------------------------
    # Optimizers and schedules
    # --------------------------------------------------------
    lr_attacker: float = 1e-3
    # Learning rate for attacker encoder

    lr_defender: float = 1e-4
    # Learning rate for defender kernel
    # Typically smaller to ensure stable shaping policies

    use_temp_anneal: bool = True
    # Whether to anneal Gumbel-softmax temperatures

    temperature_anneal: float = 0.9995
    # Multiplicative decay factor per step

    min_temperature: float = 0.3
    # Lower bound for temperature to avoid premature hard collapse

    clip_norm: float = 1.0
    # Gradient norm clipping for both attacker and defender

    # --------------------------------------------------------
    # Training dynamics
    # --------------------------------------------------------
    use_adaptive_steps: bool = True
    # Enable EMA-based early stopping on defender loss

    min_steps: int = 1000
    # Minimum number of training steps before early stopping is allowed

    max_steps: int = 5000
    # Hard upper bound on training iterations

    earlystop_patience: int = 500
    # Number of steps without sufficient improvement before stopping

    earlystop_min_delta: float = 1e-4
    # Absolute improvement threshold for EMA loss

    earlystop_min_rel: float = 0.001
    # Relative improvement threshold for EMA loss

    ema_alpha: float = 0.1
    # Smoothing factor for EMA of defender loss

    # --------------------------------------------------------
    # Inner game
    # --------------------------------------------------------
    attacker_steps: int = 1
    # Number of attacker gradient steps per defender step

    defender_steps: int = 2  # e.g. 3, 5 for ipd

    use_attacker_ensemble: bool = True
    # Whether to maintain multiple attackers and select worst case

    num_attackers: int = 1
    # Number of attacker instances in the ensemble



    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    log_every: int = 10
    # Log metrics every N training steps


# ============================================================
# Tester configuration
# ============================================================

@dataclass
class TesterConfig(BaseConfig):
    """
    Configuration used ONLY during evaluation.
    """

    # --------------------------------------------------------
    # Model input
    # --------------------------------------------------------
    model_dir: str = "Covadis_model"

    # --------------------------------------------------------
    # Result output
    # --------------------------------------------------------
    results_dir: str = "out"

    # --------------------------------------------------------
    # NCC embedding methods
    # --------------------------------------------------------
    ncc_methods: Tuple[str, ...] = (
        "replay",
        "fixed",
        "modulo",
        "model",
    )

    # --------------------------------------------------------
    # Disruptor methods
    # --------------------------------------------------------
    disruptor_methods: Tuple[str, ...] = (
        "model",
        "random_20",
        "random_50",
        "random_100",
        "fixed",
        "ditto",
    )

    # --------------------------------------------------------
    # Source distributions
    # If None, uniform distribution is used
    # --------------------------------------------------------
    P_K_values: Tuple[Optional[List[float]], ...] = (None,)

    # --------------------------------------------------------
    # Ditto disruptor
    # --------------------------------------------------------
    ditto_pattern_length: int = 4
