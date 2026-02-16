import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from dataclasses import dataclass

@dataclass
class BaselineStats:
    bytes: float       # bytes
    duration: float    # seconds
    goodput: float     # B/s (optional; will be recomputed anyway)


import numpy as np

def simulate_setting(
    baseline,
    n_samples: int,
    bytes_increase_range=(0.4, 0.5),       # size -> bytes only
    duration_increase_range=(0.2, 0.4),    # ipd -> duration only

    bytes_noise_std=0.1,                  # 5% jitter on bytes (suggest 0.03-0.06)
    duration_noise_std=0.1,               # 5% jitter on duration (suggest 0.03-0.06)

    # make duration noise right-skewed like real networks
    duration_noise_model="lognormal",      # "normal" | "lognormal"

    # extra interaction noise for size+ipd to avoid too-regular patterns
    interaction_std=0.1,                  # 2% extra multiplicative effect

    mode="size_only",                      # "no" | "size_only" | "ipd_only" | "size_ipd"
    seed=None,
):
    rng = np.random.default_rng(seed)

    bytes_list, duration_list, goodput_list = [], [], []

    for _ in range(n_samples):
        # -------- bytes factor (size only affects bytes) --------
        bytes_factor = 1.0
        if mode in ("size_only", "size_ipd"):
            bytes_factor += rng.uniform(*bytes_increase_range)

        # multiplicative jitter
        bytes_factor *= max(1e-6, 1.0 + rng.normal(0.0, bytes_noise_std))

        # optional interaction (small) for size+ipd
        if mode == "size_ipd" and interaction_std > 0:
            bytes_factor *= max(1e-6, 1.0 + rng.normal(0.0, interaction_std))

        bytes_i = baseline.bytes * bytes_factor

        # -------- duration factor (ipd only affects duration) --------
        duration_factor = 1.0
        if mode in ("ipd_only", "size_ipd"):
            duration_factor += rng.uniform(*duration_increase_range)

        # right-skewed jitter
        if duration_noise_model == "lognormal":
            # lognormal with median ~ 1, controlled spread
            # approx: exp(N(0, sigma)) has median 1
            duration_factor *= rng.lognormal(mean=0.0, sigma=duration_noise_std)
        else:
            duration_factor *= max(1e-6, 1.0 + rng.normal(0.0, duration_noise_std))

        # optional interaction on duration too (small)
        if mode == "size_ipd" and interaction_std > 0:
            duration_factor *= rng.lognormal(mean=0.0, sigma=interaction_std)

        duration_i = baseline.duration * duration_factor

        # -------- goodput derived --------
        goodput_i = bytes_i / max(duration_i, 1e-9)

        bytes_list.append(bytes_i)
        duration_list.append(duration_i)
        goodput_list.append(goodput_i)

    return {
        "bytes": np.array(bytes_list),
        "duration": np.array(duration_list),
        "goodput": np.array(goodput_list),
    }



def summarize(samples):
    # sample std (ddof=1) as usual for "mean ± std"
    return {k: (v.mean(), v.std(ddof=1)) for k, v in samples.items()}





if __name__ == "__main__":
    baseline = BaselineStats(bytes=90769, duration=516.5, goodput=90769 / 516.5)

    n = 10

    size_only = summarize(simulate_setting(
        baseline, n,
        bytes_increase_range=(0.4, 0.5),
        duration_increase_range=(0.0, 0.0),  # size-only不动duration（也可以留给noise）
        mode="size_only",
        seed=1
    ))

    ipd_only = summarize(simulate_setting(
        baseline, n,
        bytes_increase_range=(0.0, 0.0),  # ipd-only不动bytes
        duration_increase_range=(0.2, 0.3),  # 你设定ipd让duration变长的幅度
        mode="ipd_only",
        seed=2
    ))

    size_ipd = summarize(simulate_setting(
        baseline, n,
        bytes_increase_range=(0.4, 0.5),
        duration_increase_range=(0.2, 0.3),
        mode="size_ipd",
        seed=3
    ))

    print(size_only)
    print(ipd_only)
    print(size_ipd)

