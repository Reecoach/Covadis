# -*- coding: utf-8 -*-
"""
Data utilities

Provides
  parse_feature
  get_frequency_nopadding
  get_frequency_padding
"""

from __future__ import annotations

import json
import numpy as np

def sanitize_positive_floats(feature_pool_float):
    values = []
    for x in feature_pool_float:
        if x is None:
            continue
        try:
            v = float(x)
        except Exception:
            continue
        if v <= 0:
            continue
        values.append(v)
    return values


def compute_clip_value(
    values,
    *,
    feature_type: str,
    max_size_p=None,
    max_ipd_p=None,
    max_size_bytes=None,
    max_ipd_ms=None,
):
    clip_value = None

    if feature_type == "size" and max_size_p is not None:
        clip_value = float(np.percentile(values, max_size_p))

    if feature_type == "ipd" and max_ipd_p is not None:
        clip_value = float(np.percentile(values, max_ipd_p))

    if feature_type == "size" and max_size_bytes is not None:
        clip_value = (
            float(max_size_bytes)
            if clip_value is None
            else min(clip_value, float(max_size_bytes))
        )

    if feature_type == "ipd" and max_ipd_ms is not None:
        hard_cap = float(max_ipd_ms) / 1000.0
        clip_value = hard_cap if clip_value is None else min(clip_value, hard_cap)

    return clip_value


# ======================
# SIZE
# ======================

def discretize_size(values, clip_value=None):
    out = []
    for v in values:
        if clip_value is not None and v > clip_value:
            continue
        out.append(int(v))
    return out


def postprocess_size(index_values):
    return [int(v) for v in index_values]


# ======================
# IPD (millisecond domain)
# ======================

def discretize_ipd_ms(values, clip_value=None, ipd_precision_ms=1):
    """
    Discretize IPD in millisecond domain.
    Output: integer bins with clear physical meaning.
    """
    prec_ms = max(int(ipd_precision_ms), 1)
    out = []

    for v in values:
        if clip_value is not None and v > clip_value:
            continue
        ipd_ms = int(round(v * 1000.0))
        b = ipd_ms // prec_ms
        if b < 1:
            b = 1
        out.append(b)

    return out


def postprocess_ipd_ms(index_values, ipd_precision_ms=1):
    prec_ms = max(int(ipd_precision_ms), 1)
    return [int(b) * prec_ms for b in index_values]


# ======================
# IPD (log-compressed bins)
# ======================

def compress_ipd_log(index_values, base: int = 10.0):
    """
    Apply log compression on already discretized IPD bins.
    Still outputs integer tokens.
    """
    out = []
    for b in index_values:
        z = int(np.log1p(int(b)) * base)
        if z < 1:
            z = 1
        out.append(z)
    return out


def decompress_ipd_log(index_values, base: int = 10.0):
    """
    Inverse of compress_ipd_log (approximate).
    """
    out = []
    for z in index_values:
        b = int(np.expm1(float(z) / base))
        if b < 1:
            b = 1
        out.append(b)
    return out


# ======================
# Unified preprocess
# ======================

def preprocess_feature_pool(
    feature_pool_float,
    feature_type,
    *,
    ipd_precision_ms=1,
    max_size_bytes=None,
    max_ipd_ms=None,
    max_ipd_p=None,
    max_size_p=None,
    log_ipd=False,
):
    values = sanitize_positive_floats(feature_pool_float)
    if not values:
        return []

    clip_value = compute_clip_value(
        values,
        feature_type=feature_type,
        max_size_p=max_size_p,
        max_ipd_p=max_ipd_p,
        max_size_bytes=max_size_bytes,
        max_ipd_ms=max_ipd_ms,
    )

    if feature_type == "size":
        return discretize_size(values, clip_value)

    elif feature_type == "ipd":
        bins = discretize_ipd_ms(
            values,
            clip_value,
            ipd_precision_ms,
        )
        if log_ipd:
            bins = compress_ipd_log(bins)
        return bins

    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")


# ======================
# Unified postprocess
# ======================

def postprocess_feature_indices(
    index_values,
    feature_type,
    *,
    ipd_precision_ms=1,
    log_ipd=False,
):
    if feature_type == "size":
        return postprocess_size(index_values)

    elif feature_type == "ipd":
        bins = index_values
        if log_ipd:
            bins = decompress_ipd_log(bins)
        return postprocess_ipd_ms(bins, ipd_precision_ms)

    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")



def parse_feature(
    json_path: str,
    feature_type: str = "ipd",
    direction_key: str = "pair_direction",
):
    """
    Extract 1D traffic feature sequences from a traffic json dict

    feature_type
      ipd computes within each direction
      size reads ip_len

    direction_key
      pair_direction or ip_direction or direction

    Returns
      forward_feat, backward_feat, all_feat
    """
    if feature_type not in {"ipd", "size"}:
        raise ValueError("feature_type must be ipd or size")
    if direction_key not in {"pair_direction", "ip_direction", "direction"}:
        raise ValueError("direction_key must be pair_direction, ip_direction, or direction")

    with open(json_path, "r", encoding="utf-8") as f:
        traffic = json.load(f)

    forward_feat = []
    backward_feat = []

    for _, packets in traffic.items():
        if not packets:
            continue

        fwd_packets = []
        bwd_packets = []

        for pkt in packets:
            d = pkt.get(direction_key)
            if d == 1:
                fwd_packets.append(pkt)
            elif d == -1:
                bwd_packets.append(pkt)

        def extract_from_packets(pkts):
            feats = []
            if feature_type == "size":
                for p in pkts:
                    if "ip_len" in p and p["ip_len"] is not None:
                        feats.append(float(p["ip_len"]))
                return feats

            pkts_sorted = sorted(pkts, key=lambda x: x.get("timestamp", 0.0))
            for i in range(1, len(pkts_sorted)):
                t_cur = pkts_sorted[i].get("timestamp", None)
                t_prev = pkts_sorted[i - 1].get("timestamp", None)
                if t_cur is None or t_prev is None:
                    continue
                ipd = float(t_cur) - float(t_prev)
                if ipd > 0:
                    feats.append(ipd)
            return feats

        forward_feat.extend(extract_from_packets(fwd_packets))
        backward_feat.extend(extract_from_packets(bwd_packets))

    all_feat = forward_feat + backward_feat
    return forward_feat, backward_feat, all_feat


def check_accuracy(true_codes, pred_codes) -> float:
    """
    Symbol-wise accuracy between two 1D sequences.
    If lengths differ, compare only the overlapping prefix.

    Accuracy = (# equal symbols) / min(len(true), len(pred))

    Args
    ----
    true_codes : array-like
        Ground-truth symbol sequence
    pred_codes : array-like
        Predicted / extracted symbol sequence

    Returns
    -------
    acc : float in [0, 1]
    """
    if true_codes is None or pred_codes is None:
        return 0.0

    true_arr = np.asarray(true_codes).astype(int)
    pred_arr = np.asarray(pred_codes).astype(int)

    L = min(true_arr.size, pred_arr.size)
    if L == 0:
        return 0.0

    return float(np.mean(true_arr[:L] == pred_arr[:L]))


def get_frequency(
    data: list,
    *,
    assume_contiguous: bool = False,
) -> dict:
    """
    Compute empirical frequency distribution over discrete values.

    Parameters
    ----------
    data : list
        Discrete integer values
    assume_contiguous : bool
        If False (default):
            Only observed values are included.
        If True:
            Assume state space is a contiguous integer interval [min, max],
            and include zero-probability states.
            WARNING: only safe for small ranges.

    Returns
    -------
    freq : dict
        Mapping value -> probability
    """
    if not data:
        return {}

    total = len(data)

    if not assume_contiguous:
        freq = {}
        for v in data:
            freq[v] = freq.get(v, 0) + 1
        return {k: c / total for k, c in freq.items()}

    # contiguous padding mode
    mn = int(min(data))
    mx = int(max(data))

    # safety check (strongly recommended)
    if mx - mn > 10_000:
        raise ValueError(
            f"Contiguous frequency range too large: [{mn}, {mx}]"
        )

    freq = {i: 0 for i in range(mn, mx + 1)}
    for v in data:
        freq[int(v)] += 1

    return {k: c / total for k, c in freq.items()}
