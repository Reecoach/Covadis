"""
Single file covert channel framework

Pipeline
1 Bit source generates bits
2 Encoder generates decimal codes (either from bits or by sampling a symbol distribution P_K)
3 Embedder maps codes into benign looking features
4 Extractor maps features back into codes
5 Decoder translates codes into bits (deterministic translation, even if codes are fabricated)

Hard rule
- All benign features used by Embedder and Extractor must be integers.
- If you only have float features, you must quantize them deterministically first.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Literal
import math
import random

import numpy as np

try:
    from scipy.linalg import hadamard
except Exception:
    hadamard = None

try:
    from reedsolo import RSCodec
except Exception:
    RSCodec = None


# -----------------------------
# Configs
# -----------------------------

EncoderMode = Literal["bits", "sampling"]
BitsScheme = Literal["fixed", "spreading", "geometric", "rs"]
SamplingPKMode = Literal["uniform", "manual", "dirichlet"]

EmbedMethod = Literal["replay", "fixed", "modulo", "linear", "model"]
ReplayOrder = Literal["ascending", "random"]

QuantizeMethod = Literal["round", "floor", "ceil"]


@dataclass
class BitSourceConfig:
    bit_length: int = 256
    p1: float = 0.5
    seed: int = 0


@dataclass
class SamplingConfig:
    pk_mode: SamplingPKMode = "uniform"
    K: int = 8
    L: int = 256
    P_K_manual: Optional[List[float]] = None
    dirichlet_alpha: float = 1.0


@dataclass
class FixedBitsConfig:
    bit_unit_size: int = 3  # K = 2**bit_unit_size


@dataclass
class SpreadingConfig:
    code_unit_size: int = 8  # must be power of two, uses Walsh Hadamard
    bit_unit_size: int = 3   # must be <= code_unit_size
    # For spreading, K is forced to 2*bit_unit_size + 1


@dataclass
class GeometricConfig:
    bit_unit_size: int = 4
    code_unit_size: int = 3
    threshold_delta_sum: int = 6
    # For geometric, codes are integers in [0, threshold_delta_sum], so K is threshold_delta_sum + 1


@dataclass
class RSConfig:
    bit_unit_size: int = 8          # must be <= 8
    code_unit_size: int = 16        # total symbols in an RS block
    parity_size: int = 6            # RS parity symbols, message symbols = code_unit_size - parity_size
    c_exp: int = 8                  # typical RS(2^8)


@dataclass
class EncoderConfig:
    mode: EncoderMode = "bits"
    bits_scheme: BitsScheme = "fixed"
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    fixed: FixedBitsConfig = field(default_factory=FixedBitsConfig)
    spreading: SpreadingConfig = field(default_factory=SpreadingConfig)
    geometric: GeometricConfig = field(default_factory=GeometricConfig)
    rs: RSConfig = field(default_factory=RSConfig)
    seed: int = 0


@dataclass
class ReplayConfig:
    order: ReplayOrder = "ascending"
    seed: int = 0


@dataclass
class ModelConfig:
    interpolate: bool = True
    quantile_edges: Optional[List[float]] = None
    seed: int = 0


@dataclass
class ModuloConfig:
    modulo_base: Optional[int] = None  # if None, uses K


@dataclass
class LinearConfig:
    clip_to_range: bool = True


@dataclass
class FeatureQuantizerConfig:
    """
    Deterministic feature quantizer.
    If enable is True, benign features can be float and will be quantized to int.
    If enable is False, benign features must already be integers (hard error otherwise).
    """
    enable: bool = True
    scale: float = 1.0
    method: QuantizeMethod = "round"


@dataclass
class EmbedderConfig:
    method: EmbedMethod = "model"
    K: int = 8
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    modulo: ModuloConfig = field(default_factory=ModuloConfig)
    linear: LinearConfig = field(default_factory=LinearConfig)
    feat_quantizer: FeatureQuantizerConfig = field(default_factory=FeatureQuantizerConfig)


@dataclass
class ExtractorConfig:
    method: EmbedMethod = "model"
    K: int = 8
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    modulo: ModuloConfig = field(default_factory=ModuloConfig)
    linear: LinearConfig = field(default_factory=LinearConfig)
    feat_quantizer: FeatureQuantizerConfig = field(default_factory=FeatureQuantizerConfig)


@dataclass
class DecoderConfig:
    bits_scheme: BitsScheme = "fixed"
    fixed: FixedBitsConfig = field(default_factory=FixedBitsConfig)
    spreading: SpreadingConfig = field(default_factory=SpreadingConfig)
    geometric: GeometricConfig = field(default_factory=GeometricConfig)
    rs: RSConfig = field(default_factory=RSConfig)
    # Decoder always translates codes to bits deterministically, without validity checks


@dataclass
class ExperimentConfig:
    seed: int = 0
    bit_source: BitSourceConfig = field(default_factory=BitSourceConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


# -----------------------------
# Utilities
# -----------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def bits_to_str(bits: np.ndarray) -> str:
    return "".join("1" if int(b) else "0" for b in bits.tolist())


def int_to_bits(x: int, width: int) -> List[int]:
    s = format(int(x) & ((1 << width) - 1), f"0{width}b")
    return [1 if ch == "1" else 0 for ch in s]


def clamp_int(x: int, lo: int, hi: int) -> int:
    return int(min(max(int(x), lo), hi))


def quantize_features(
    feats: List[float | int],
    *,
    scale: float = 1.0,
    method: QuantizeMethod = "round",
) -> List[int]:
    """
    Deterministically quantize features into integers.

    Typical usage
    - IPD in seconds -> scale = 1e6 to get microseconds as integers
    - IPD in seconds -> scale = 1e3 to get milliseconds as integers
    """
    arr = np.asarray(feats, dtype=float) * float(scale)

    if method == "round":
        q = np.rint(arr)
    elif method == "floor":
        q = np.floor(arr)
    elif method == "ceil":
        q = np.ceil(arr)
    else:
        raise ValueError("Unknown quantize method.")

    return q.astype(int).tolist()


def _needs_quantization(feats: List[Any]) -> bool:
    for v in feats:
        if isinstance(v, (np.floating, float)):
            return True
        if isinstance(v, (np.integer, int)):
            continue
        # Any other type is not acceptable
        return True
    return False


def ensure_int_features(
    feats: List[Any],
    *,
    qcfg: FeatureQuantizerConfig,
    context: str = "features",
) -> List[int]:
    """
    Enforce the hard rule: internal embed/extract only operates on integer features.

    If qcfg.enable is True, float features are quantized deterministically.
    If qcfg.enable is False, non integer features raise an error.
    """
    if feats is None:
        raise ValueError(f"{context} is None.")

    if len(feats) == 0:
        return []

    if _needs_quantization(feats):
        if not qcfg.enable:
            raise TypeError(
                f"{context} must be integers. "
                f"Quantization is disabled but non integer values were provided."
            )
        feats_int = quantize_features(feats, scale=qcfg.scale, method=qcfg.method)
        return feats_int

    # Already integer like
    return [int(v) for v in feats]


# -----------------------------
# Bit source
# -----------------------------

class BitSource:
    def __init__(self, cfg: BitSourceConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def sample(self) -> np.ndarray:
        # Bernoulli bit generation
        bits = (self.rng.random(self.cfg.bit_length) < float(self.cfg.p1)).astype(np.int32)
        return bits


# -----------------------------
# Encoder
# -----------------------------

class Encoder:
    def __init__(self, cfg: EncoderConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Precompute Hadamard if needed
        if self.cfg.bits_scheme == "spreading":
            if hadamard is None:
                raise RuntimeError("scipy is required for spreading scheme (hadamard).")
            N = self.cfg.spreading.code_unit_size
            if N & (N - 1) != 0:
                raise ValueError("Spreading code_unit_size must be a power of two.")
            self._H = hadamard(N).astype(np.int32)
        else:
            self._H = None

        # Precompute combinations for geometric if needed
        if self.cfg.bits_scheme == "geometric":
            self._geometric_combs = self._build_geometric_combs(
                n=self.cfg.geometric.code_unit_size,
                K=self.cfg.geometric.threshold_delta_sum,
            )
            need = 2 ** self.cfg.geometric.bit_unit_size
            if len(self._geometric_combs) < need:
                raise ValueError("Not enough geometric combinations for the given bit_unit_size.")
            self._geometric_map = {tuple(c): idx for idx, c in enumerate(self._geometric_combs)}
        else:
            self._geometric_combs = None
            self._geometric_map = None

    @staticmethod
    def _build_geometric_combs(n: int, K: int) -> List[Tuple[int, ...]]:
        # Generate all n-tuples of nonnegative integers summing to at most K
        def rec(nn: int, kk: int) -> List[Tuple[int, ...]]:
            if nn == 1:
                return [(i,) for i in range(kk + 1)]
            out: List[Tuple[int, ...]] = []
            for i in range(kk + 1):
                for sub in rec(nn - 1, kk - i):
                    out.append((i,) + sub)
            return out

        return rec(n, K)

    def infer_K(self) -> int:
        # Infer K implied by the current encoder configuration.
        if self.cfg.mode == "sampling":
            return int(self.cfg.sampling.K)

        scheme = self.cfg.bits_scheme
        if scheme == "fixed":
            return 2 ** int(self.cfg.fixed.bit_unit_size)
        if scheme == "spreading":
            b = int(self.cfg.spreading.bit_unit_size)
            return 2 * b + 1
        if scheme == "geometric":
            return int(self.cfg.geometric.threshold_delta_sum) + 1
        if scheme == "rs":
            # RS symbols are in [0, 255] when c_exp is 8
            return 256
        raise ValueError(f"Unknown scheme: {scheme}")

    def encode(self, bits: Optional[np.ndarray] = None) -> np.ndarray:
        if self.cfg.mode == "sampling":
            if bits is not None:
                raise ValueError("Sampling mode does not take bits.")
            return self._encode_sampling()

        if bits is None:
            raise ValueError("Bits mode requires bits input.")

        scheme = self.cfg.bits_scheme
        if scheme == "fixed":
            return self._encode_fixed(bits)
        if scheme == "spreading":
            return self._encode_spreading(bits)
        if scheme == "geometric":
            return self._encode_geometric(bits)
        if scheme == "rs":
            return self._encode_rs(bits)
        raise ValueError(f"Unknown bits scheme: {scheme}")

    def _encode_sampling(self) -> np.ndarray:
        scfg = self.cfg.sampling
        K = int(scfg.K)
        L = int(scfg.L)

        if scfg.pk_mode == "uniform":
            P = np.ones(K, dtype=np.float64) / float(K)
        elif scfg.pk_mode == "manual":
            if not scfg.P_K_manual or len(scfg.P_K_manual) != K:
                raise ValueError("Manual P_K must be provided with length K.")
            P = np.asarray(scfg.P_K_manual, dtype=np.float64)
            s = float(P.sum())
            if s <= 0:
                raise ValueError("Manual P_K must have positive sum.")
            P = P / s
        elif scfg.pk_mode == "dirichlet":
            alpha = float(scfg.dirichlet_alpha)
            if alpha <= 0:
                raise ValueError("Dirichlet alpha must be positive.")
            P = self.rng.dirichlet(alpha * np.ones(K, dtype=np.float64))
        else:
            raise ValueError(f"Unknown pk_mode: {scfg.pk_mode}")

        codes = self.rng.choice(np.arange(K, dtype=np.int32), size=L, replace=True, p=P).astype(np.int32)
        return codes

    def _encode_fixed(self, bits: np.ndarray) -> np.ndarray:
        b = int(self.cfg.fixed.bit_unit_size)
        if b <= 0:
            raise ValueError("bit_unit_size must be positive.")

        n_groups = len(bits) // b
        bits_trim = bits[: n_groups * b].astype(np.int32)
        bit_matrix = bits_trim.reshape(n_groups, b)

        codes = np.zeros(n_groups, dtype=np.int32)
        for i in range(b):
            codes = (codes << 1) | bit_matrix[:, i]
        return codes

    def _encode_spreading(self, bits: np.ndarray) -> np.ndarray:
        scfg = self.cfg.spreading
        N = int(scfg.code_unit_size)
        b = int(scfg.bit_unit_size)

        if b <= 0 or b > N:
            raise ValueError("spreading bit_unit_size must be in [1, code_unit_size].")

        n_units = int(math.ceil(len(bits) / b))
        bits_pad = np.zeros(n_units * b, dtype=np.int32)
        bits_pad[: len(bits)] = bits.astype(np.int32)

        codes_out: List[int] = []
        for u in range(n_units):
            unit = bits_pad[u * b : (u + 1) * b]
            v = np.zeros(N, dtype=np.int32)
            for j in range(b):
                sgn = 1 if int(unit[j]) == 1 else -1
                v += self._H[j].astype(np.int32) * sgn
            v_shift = v + b
            codes_out.extend(v_shift.astype(np.int32).tolist())

        return np.asarray(codes_out, dtype=np.int32)

    def _encode_geometric(self, bits: np.ndarray) -> np.ndarray:
        gcfg = self.cfg.geometric
        b = int(gcfg.bit_unit_size)
        n = int(gcfg.code_unit_size)

        n_units = int(math.ceil(len(bits) / b))
        bits_pad = np.zeros(n_units * b, dtype=np.int32)
        bits_pad[: len(bits)] = bits.astype(np.int32)

        codes_out: List[int] = []
        for u in range(n_units):
            unit = bits_pad[u * b : (u + 1) * b]
            idx = 0
            for i in range(b):
                idx = (idx << 1) | int(unit[i])
            comb = self._geometric_combs[idx]
            codes_out.extend(list(map(int, comb)))

        return np.asarray(codes_out, dtype=np.int32)

    def _encode_rs(self, bits: np.ndarray) -> np.ndarray:
        rcfg = self.cfg.rs
        if rcfg.bit_unit_size > 8:
            raise ValueError("RS bit_unit_size must be <= 8.")
        if RSCodec is None:
            raise RuntimeError("reedsolo is required for rs scheme.")

        b = int(rcfg.bit_unit_size)
        code_n = int(rcfg.code_unit_size)
        nsym = int(rcfg.parity_size)
        msg_k = code_n - nsym
        if msg_k <= 0:
            raise ValueError("RS code_unit_size must be larger than parity_size.")

        n_syms = int(math.ceil(len(bits) / b))
        bits_pad = np.zeros(n_syms * b, dtype=np.int32)
        bits_pad[: len(bits)] = bits.astype(np.int32)
        sym = np.zeros(n_syms, dtype=np.int32)
        for i in range(n_syms):
            val = 0
            for j in range(b):
                val = (val << 1) | int(bits_pad[i * b + j])
            sym[i] = val

        rs = RSCodec(nsym=nsym, c_exp=int(rcfg.c_exp))
        codes_out: List[int] = []
        for i in range(0, len(sym), msg_k):
            block = sym[i : i + msg_k].tolist()
            if len(block) < msg_k:
                block += [0] * (msg_k - len(block))
            enc = rs.encode(bytes(block))
            codes_out.extend(list(enc))

        return np.asarray(codes_out, dtype=np.int32)


# -----------------------------
# Decoder
# -----------------------------

class Decoder:
    def __init__(self, cfg: DecoderConfig):
        self.cfg = cfg

        if self.cfg.bits_scheme == "spreading":
            if hadamard is None:
                raise RuntimeError("scipy is required for spreading scheme (hadamard).")
            N = int(self.cfg.spreading.code_unit_size)
            if N & (N - 1) != 0:
                raise ValueError("Spreading code_unit_size must be a power of two.")
            self._H = hadamard(N).astype(np.int32)
        else:
            self._H = None

        if self.cfg.bits_scheme == "geometric":
            gcfg = self.cfg.geometric
            combs = Encoder._build_geometric_combs(int(gcfg.code_unit_size), int(gcfg.threshold_delta_sum))
            self._geometric_map = {tuple(c): idx for idx, c in enumerate(combs)}
            self._geometric_b = int(gcfg.bit_unit_size)
            self._geometric_n = int(gcfg.code_unit_size)
            self._geometric_Ksum = int(gcfg.threshold_delta_sum)
        else:
            self._geometric_map = None

        if self.cfg.bits_scheme == "rs":
            if RSCodec is None:
                raise RuntimeError("reedsolo is required for rs scheme.")
            rcfg = self.cfg.rs
            self._rs = RSCodec(nsym=int(rcfg.parity_size), c_exp=int(rcfg.c_exp))
        else:
            self._rs = None

    def decode(self, codes: np.ndarray) -> np.ndarray:
        scheme = self.cfg.bits_scheme
        if scheme == "fixed":
            return self._decode_fixed(codes)
        if scheme == "spreading":
            return self._decode_spreading(codes)
        if scheme == "geometric":
            return self._decode_geometric(codes)
        if scheme == "rs":
            return self._decode_rs(codes)
        raise ValueError(f"Unknown scheme: {scheme}")

    def _decode_fixed(self, codes: np.ndarray) -> np.ndarray:
        b = int(self.cfg.fixed.bit_unit_size)
        out: List[int] = []
        for c in codes.astype(np.int64).tolist():
            out.extend(int_to_bits(int(c), b))
        return np.asarray(out, dtype=np.int32)

    def _decode_spreading(self, codes: np.ndarray) -> np.ndarray:
        scfg = self.cfg.spreading
        N = int(scfg.code_unit_size)
        b = int(scfg.bit_unit_size)

        n_units = len(codes) // N
        out: List[int] = []

        for u in range(n_units):
            chunk = codes[u * N : (u + 1) * N].astype(np.int32)
            v = chunk - b

            for j in range(b):
                score = int(np.dot(v, self._H[j]))
                bit = 1 if score >= 0 else 0
                out.append(bit)

        return np.asarray(out, dtype=np.int32)

    def _decode_geometric(self, codes: np.ndarray) -> np.ndarray:
        gcfg = self.cfg.geometric
        b = int(gcfg.bit_unit_size)
        n = int(gcfg.code_unit_size)
        Ksum = int(gcfg.threshold_delta_sum)

        n_units = len(codes) // n
        out: List[int] = []

        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        for u in range(n_units):
            chunk = codes[u * n : (u + 1) * n].astype(np.int64).tolist()
            chunk = [clamp_int(x, 0, Ksum) for x in chunk]

            t = tuple(chunk)
            idx = self._geometric_map.get(t, None)
            if idx is None:
                h = 0
                for i, x in enumerate(chunk):
                    h += int(x) * primes[i % len(primes)]
                idx = int(h) % (2 ** b)

            out.extend(int_to_bits(int(idx), b))

        return np.asarray(out, dtype=np.int32)

    def _decode_rs(self, codes: np.ndarray) -> np.ndarray:
        rcfg = self.cfg.rs
        b = int(rcfg.bit_unit_size)
        code_n = int(rcfg.code_unit_size)
        nsym = int(rcfg.parity_size)
        msg_k = code_n - nsym

        out: List[int] = []
        n_blocks = len(codes) // code_n

        for bi in range(n_blocks):
            block = codes[bi * code_n : (bi + 1) * code_n].astype(np.int32).tolist()
            msg: Optional[bytes] = None
            try:
                decoded = self._rs.decode(bytes([clamp_int(x, 0, 255) for x in block]))
                if isinstance(decoded, tuple):
                    msg = decoded[0]
                else:
                    msg = decoded
            except Exception:
                msg = None

            if msg is not None and len(msg) >= msg_k:
                for sym in list(msg[:msg_k]):
                    out.extend(int_to_bits(int(sym), b))
            else:
                for sym in block[:msg_k]:
                    out.extend(int_to_bits(int(sym), b))

        return np.asarray(out, dtype=np.int32)


# -----------------------------
# Embedder and Extractor
# -----------------------------

class Embedder:
    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg

    def embed(self, codes: np.ndarray, benign_feat: List[Any]) -> List[int]:
        # Enforce integer-only feature domain (quantize if enabled)
        benign_feat_int = ensure_int_features(
            benign_feat,
            qcfg=self.cfg.feat_quantizer,
            context="benign_feat",
        )

        method = self.cfg.method
        if method == "replay":
            return self._embed_replay(codes, benign_feat_int)
        if method == "fixed":
            return self._embed_fixed(codes, benign_feat_int)
        if method == "modulo":
            return self._embed_modulo(codes, benign_feat_int)
        if method == "linear":
            return self._embed_linear(codes, benign_feat_int)
        if method == "model":
            return self._embed_model(codes, benign_feat_int)
        raise ValueError(f"Unknown embed method: {method}")

    def _embed_replay(self, codes: np.ndarray, benign_feat: List[int]) -> List[int]:
        if not benign_feat:
            raise ValueError("benign_feat is empty.")

        K = int(self.cfg.K)
        rcfg = self.cfg.replay
        rng = random.Random(int(rcfg.seed))

        if rcfg.order == "ascending":
            data = sorted(int(x) for x in benign_feat)
        elif rcfg.order == "random":
            data = [int(x) for x in benign_feat]
            rng.shuffle(data)
        else:
            raise ValueError("replay.order must be 'ascending' or 'random'.")

        total = len(data)
        raw_bins: List[List[int]] = []
        for i in range(K):
            start = int(i * total / K)
            end = int((i + 1) * total / K) if i < K - 1 else total
            raw_bins.append(data[start:end])

        # Make bins disjoint by deduplication, and borrow from previous bins if empty
        seen: set[int] = set()
        clean_bins: List[List[int]] = []

        for i in range(K):
            bin_clean = [int(x) for x in raw_bins[i] if int(x) not in seen]

            if not bin_clean:
                donor = None
                for j in range(i - 1, -1, -1):
                    if len(clean_bins[j]) > 1:
                        donor = j
                        break

                if donor is not None:
                    v = clean_bins[donor].pop()
                    bin_clean = [int(v)]
                else:
                    # Extreme fallback: borrow from previous bin then patch it with a deterministic integer jitter
                    if i - 1 >= 0 and len(clean_bins[i - 1]) >= 1:
                        v = clean_bins[i - 1].pop()
                        bin_clean = [int(v)]

                        base = int(v)
                        step = max(1, int(abs(base) * 0.01))
                        cand = None
                        for delta in range(step, step * 16 + 1, step):
                            for sgn in (1, -1):
                                c = base + sgn * delta
                                if (c not in seen) and (c not in clean_bins[i - 1]) and (c not in bin_clean):
                                    cand = int(c)
                                    break
                            if cand is not None:
                                break
                        clean_bins[i - 1].append(int(cand if cand is not None else base))
                    else:
                        v = None
                        for x in data:
                            if int(x) not in seen:
                                v = int(x)
                                break
                        bin_clean = [int(v) if v is not None else 0]

            for x in bin_clean:
                seen.add(int(x))
            clean_bins.append(bin_clean)

        np_rng = np.random.default_rng(int(rcfg.seed))
        sample_size = len(codes)
        pools = [np_rng.choice(b, size=sample_size, replace=True).astype(int) for b in clean_bins]
        idx = [0] * K

        embedded: List[int] = []
        for c in codes.astype(np.int64).tolist():
            cc = clamp_int(int(c), 0, K - 1)
            embedded.append(int(pools[cc][idx[cc]]))
            idx[cc] += 1

        return embedded

    def _embed_fixed_simple(self, codes: np.ndarray, benign_feat: List[int]) -> List[int]:
        if not benign_feat:
            raise ValueError("benign_feat is empty.")
        K = int(self.cfg.K)

        vals, counts = np.unique(np.asarray(benign_feat, dtype=int), return_counts=True)
        order = np.argsort(counts)[::-1]
        top = vals[order]
        if len(top) < K:
            raise ValueError("Not enough unique feature values for fixed embedding.")
        top_k = top[:K].astype(int)

        embedded: List[int] = []
        for c in codes.astype(np.int64).tolist():
            cc = clamp_int(int(c), 0, K - 1)
            embedded.append(int(top_k[cc]))
        return embedded

    def _embed_fixed(self, codes: np.ndarray, benign_feat: List[float]) -> List[float]:
        """
        EKlibur-style embedding with arbitrary K support.
        Uses statistical distribution of benign_feat to define K uniform intervals.
        """
        if not benign_feat:
            raise ValueError("benign_feat is empty.")

        K = int(self.cfg.K)
        if K < 2:
            raise ValueError("K must be at least 2.")

        # 1. 统计benign_feat的分布范围
        benign_array = np.asarray(benign_feat, dtype=float)
        min_val = float(np.min(benign_array))
        max_val = float(np.max(benign_array))

        # 2. 将整个范围划分为K个区间
        # 每个区间代表一个code值
        interval_width = (max_val - min_val) / K

        # 3. 为每个区间定义扰动策略
        # 前半部分用uniform小扰动,后半部分用steplike大扰动
        def _generate_value(code_idx: int) -> float:
            # 区间中心点
            center = min_val + (code_idx + 0.5) * interval_width

            # 区间的局部范围 (用于扰动)
            local_range = interval_width * 0.8  # 80%的区间宽度用于扰动

            # 根据code_idx决定扰动风格
            # 前半段(小code)用小扰动, 后半段(大code)用阶梯扰动
            if code_idx < K // 2:
                # Small uniform noise: N(0, sigma) clipped
                sigma = local_range / 7
                noise = np.clip(np.random.normal(0, sigma), -local_range / 2, local_range / 2)
            else:
                # Large steplike noise: discrete uniform steps
                steps = int(local_range / 1) # step = 1ms
                noise = (np.random.randint(0, steps + 1) * 1) - local_range / 2

            return float(center + noise)

        # 4. 映射codes到features
        embedded: List[float] = []
        for c in codes.astype(np.int64).tolist():
            cc = clamp_int(int(c), 0, K - 1)
            embedded.append(_generate_value(cc))

        # 5. Optional: 添加outlier (保持原有特性)
        if self.cfg.get("enable_outlier", True) and embedded:
            outlier_scale = float(self.cfg.get("outlier_scale", 10))
            outlier_val = max_val * outlier_scale
            embedded[random.randint(0, len(embedded) - 1)] = float(outlier_val)

        return embedded

    def _embed_modulo(self, codes: np.ndarray, benign_feat: List[int]) -> List[int]:
        if not benign_feat:
            raise ValueError("benign_feat is empty.")
        K = int(self.cfg.K)

        modulo_base = self.cfg.modulo.modulo_base
        if modulo_base is None:
            modulo_base = K
        modulo_base = int(modulo_base)
        if modulo_base <= 0:
            raise ValueError("modulo_base must be positive.")
        if modulo_base % K != 0:
            scale = 1
        else:
            scale = modulo_base // K

        embedded: List[int] = []
        for c, m in zip(codes.astype(np.int64).tolist(), benign_feat):
            cc = clamp_int(int(c), 0, K - 1)
            mm = int(m)
            embedded.append((mm // modulo_base) * modulo_base + cc * scale)
        return embedded

    def _embed_linear(self, codes: np.ndarray, benign_feat: List[int]) -> List[int]:
        if not benign_feat:
            raise ValueError("benign_feat is empty.")
        K = int(self.cfg.K)

        lo = int(min(benign_feat))
        hi = int(max(benign_feat))
        span = hi - lo
        if span <= 0:
            return [lo for _ in range(len(codes))]

        step = max(1, span // K)

        embedded: List[int] = []
        for c in codes.astype(np.int64).tolist():
            cc = clamp_int(int(c), 0, K - 1)
            v = lo + cc * step
            if self.cfg.linear.clip_to_range:
                v = clamp_int(v, lo, hi)
            embedded.append(int(v))
        return embedded

    def _embed_model(self, codes: np.ndarray, benign_feat: List[int]) -> List[int]:
        if not benign_feat:
            raise ValueError("benign_feat is empty.")

        K = int(self.cfg.K)
        mcfg = self.cfg.model
        rng = np.random.default_rng(int(mcfg.seed))

        vals_sorted = np.sort(np.asarray(benign_feat, dtype=int))
        n = len(vals_sorted)
        lo, hi = int(vals_sorted[0]), int(vals_sorted[-1])

        if mcfg.quantile_edges is None:
            edges = np.linspace(0.0, 1.0, K + 1, endpoint=True)
        else:
            edges = np.asarray(mcfg.quantile_edges, dtype=float)
            if len(edges) != K + 1:
                raise ValueError("quantile_edges length must be K+1.")
            if edges[0] < 0.0 or edges[-1] > 1.0 or not np.all(edges[1:] >= edges[:-1]):
                raise ValueError("quantile_edges must be nondecreasing from 0 to 1.")

        c = np.asarray(codes, dtype=int)
        c = np.clip(c, 0, K - 1)

        a = edges[c]
        b = edges[c + 1]
        r = rng.random(size=c.shape)
        u = np.where(b > a, a + (b - a) * r, np.minimum(a, np.nextafter(1.0, 0.0)))

        if mcfg.interpolate:
            p = u * (n - 1)
            i0 = np.floor(p).astype(int)
            i1 = np.clip(i0 + 1, 0, n - 1)
            alpha = p - i0
            v_float = (1.0 - alpha) * vals_sorted[i0] + alpha * vals_sorted[i1]
            v = np.clip(np.rint(v_float), lo, hi).astype(int)
        else:
            idx = np.ceil(u * n).astype(int) - 1
            idx = np.clip(idx, 0, n - 1)
            v = vals_sorted[idx].astype(int)

        return v.astype(int).tolist()


class Extractor:
    def __init__(self, cfg: ExtractorConfig):
        self.cfg = cfg

    def extract(self, extracted_feat: List[Any], benign_feat: List[Any]) -> np.ndarray:
        # Enforce integer-only feature domain (quantize if enabled)
        benign_feat_int = ensure_int_features(
            benign_feat,
            qcfg=self.cfg.feat_quantizer,
            context="benign_feat",
        )
        extracted_feat_int = ensure_int_features(
            extracted_feat,
            qcfg=self.cfg.feat_quantizer,
            context="extracted_feat",
        )

        method = self.cfg.method
        if method == "replay":
            return self._extract_replay(extracted_feat_int, benign_feat_int)
        if method == "fixed":
            return self._extract_fixed(extracted_feat_int, benign_feat_int)
        if method == "modulo":
            return self._extract_modulo(extracted_feat_int)
        if method == "linear":
            return self._extract_linear(extracted_feat_int, benign_feat_int)
        if method == "model":
            return self._extract_model(extracted_feat_int, benign_feat_int)
        raise ValueError(f"Unknown extract method: {method}")

    def _extract_replay(self, extracted_feat: List[int], benign_feat: List[int]) -> np.ndarray:
        if not benign_feat:
            raise ValueError("benign_feat is empty.")

        K = int(self.cfg.K)
        rcfg = self.cfg.replay
        rng = random.Random(int(rcfg.seed))

        if rcfg.order == "ascending":
            data = sorted(int(x) for x in benign_feat)
        elif rcfg.order == "random":
            data = [int(x) for x in benign_feat]
            rng.shuffle(data)
        else:
            raise ValueError("replay.order must be 'ascending' or 'random'.")

        total = len(data)
        raw_bins: List[List[int]] = []
        for i in range(K):
            start = int(i * total / K)
            end = int((i + 1) * total / K) if i < K - 1 else total
            raw_bins.append(data[start:end])

        seen: set[int] = set()
        clean_bins: List[List[int]] = []

        for i in range(K):
            bin_clean = [int(x) for x in raw_bins[i] if int(x) not in seen]

            if not bin_clean:
                donor = None
                for j in range(i - 1, -1, -1):
                    if len(clean_bins[j]) > 1:
                        donor = j
                        break

                if donor is not None:
                    v = clean_bins[donor].pop()
                    bin_clean = [int(v)]
                else:
                    if i - 1 >= 0 and len(clean_bins[i - 1]) >= 1:
                        v = clean_bins[i - 1].pop()
                        bin_clean = [int(v)]

                        base = int(v)
                        step = max(1, int(abs(base) * 0.01))
                        cand = None
                        for delta in range(step, step * 16 + 1, step):
                            for sgn in (1, -1):
                                c = base + sgn * delta
                                if (c not in seen) and (c not in clean_bins[i - 1]) and (c not in bin_clean):
                                    cand = int(c)
                                    break
                            if cand is not None:
                                break
                        clean_bins[i - 1].append(int(cand if cand is not None else base))
                    else:
                        v = None
                        for x in data:
                            if int(x) not in seen:
                                v = int(x)
                                break
                        bin_clean = [int(v) if v is not None else 0]

            for x in bin_clean:
                seen.add(int(x))
            clean_bins.append(bin_clean)

        val_to_code: Dict[int, int] = {}
        for code, vals in enumerate(clean_bins):
            for v in vals:
                val_to_code[int(v)] = int(code)

        out: List[int] = []
        for v in extracted_feat:
            out.append(int(val_to_code.get(int(v), -1)))

        return np.asarray(out, dtype=np.int32)

    def _extract_fixed(self, extracted_feat: List[int], benign_feat: List[int]) -> np.ndarray:
        if not benign_feat:
            raise ValueError("benign_feat is empty.")
        K = int(self.cfg.K)

        vals, counts = np.unique(np.asarray(benign_feat, dtype=int), return_counts=True)
        order = np.argsort(counts)[::-1]
        top = vals[order].astype(int)
        if len(top) < K:
            raise ValueError("Not enough unique feature values for fixed extraction.")
        top_k = top[:K].astype(int)

        val_to_code = {int(v): int(i) for i, v in enumerate(top_k.tolist())}

        out: List[int] = []
        for v in extracted_feat:
            out.append(int(val_to_code.get(int(v), -1)))
        return np.asarray(out, dtype=np.int32)

    def _extract_modulo(self, extracted_feat: List[int]) -> np.ndarray:
        K = int(self.cfg.K)
        out = [int(int(v) % K) for v in extracted_feat]
        return np.asarray(out, dtype=np.int32)

    def _extract_linear(self, extracted_feat: List[int], benign_feat: List[int]) -> np.ndarray:
        if not benign_feat:
            raise ValueError("benign_feat is empty.")
        K = int(self.cfg.K)

        lo = int(min(benign_feat))
        hi = int(max(benign_feat))
        span = hi - lo
        if span <= 0:
            return np.zeros(len(extracted_feat), dtype=np.int32)

        step = max(1, span // K)
        out: List[int] = []
        for v in extracted_feat:
            code = (int(v) - lo) // step
            out.append(clamp_int(int(code), 0, K - 1))
        return np.asarray(out, dtype=np.int32)

    def _extract_model(self, extracted_feat: List[int], benign_feat: List[int]) -> np.ndarray:
        if not benign_feat:
            raise ValueError("benign_feat is empty.")

        K = int(self.cfg.K)
        mcfg = self.cfg.model

        vals_sorted = np.sort(np.asarray(benign_feat, dtype=int))
        n = len(vals_sorted)

        if mcfg.quantile_edges is None:
            edges = np.linspace(0.0, 1.0, K + 1, endpoint=True)
        else:
            edges = np.asarray(mcfg.quantile_edges, dtype=float)
            if len(edges) != K + 1:
                raise ValueError("quantile_edges length must be K+1.")
            if edges[0] < 0.0 or edges[-1] > 1.0 or not np.all(edges[1:] >= edges[:-1]):
                raise ValueError("quantile_edges must be nondecreasing from 0 to 1.")

        uniq_vals, first_idx = np.unique(vals_sorted, return_index=True)
        next_first = np.empty_like(first_idx)
        next_first[:-1] = first_idx[1:]
        next_first[-1] = n
        counts = next_first - first_idx
        cum_counts = np.cumsum(counts)
        ecdf_right = cum_counts / float(n)

        extracted_vals = np.asarray(extracted_feat, dtype=int)

        if mcfg.interpolate:
            lo, hi = int(uniq_vals[0]), int(uniq_vals[-1])
            v_clip = np.clip(extracted_vals, lo, hi)
            u_hat = np.interp(v_clip, uniq_vals, ecdf_right)
        else:
            last_idx = next_first - 1
            u_mid = (first_idx + last_idx + 1) / (2.0 * n)

            pos = np.searchsorted(uniq_vals, extracted_vals, side="left")
            pos = np.clip(pos, 0, len(uniq_vals) - 1)

            mask = (pos > 0) & (pos < len(uniq_vals))
            pos2 = pos.copy()
            for i in np.where(mask)[0].tolist():
                left = uniq_vals[pos[i] - 1]
                right = uniq_vals[pos[i]]
                if abs(int(extracted_vals[i]) - int(left)) <= abs(int(extracted_vals[i]) - int(right)):
                    pos2[i] = pos[i] - 1

            u_hat = u_mid[pos2]

        idx = np.searchsorted(edges, u_hat, side="right") - 1
        idx = np.clip(idx, 0, K - 1).astype(int)

        return idx.astype(np.int32)

# -----------------------------
# Tools
# -----------------------------
def generate_embedded_features(
    *,
    embedding_methods: List[str],
    benign_feat: np.ndarray,
    K: int,
    num_features: Optional[int] = None,
    pk_mode: SamplingPKMode = "uniform",
    P_K_manual: Optional[List[float]] = None,
    dirichlet_alpha: float = 1.0,
    quantizer_scale: float = 1e6,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Generate embedded feature sequences from benign features.

    Input and output features are in the SAME domain as benign_feat
    (e.g. float IPD in seconds).

    Internally, embedding operates on integer features via deterministic
    quantization, following the same style as main().
    """

    rng = np.random.default_rng(seed)

    benign_feat = np.asarray(benign_feat)
    if benign_feat.ndim != 1:
        raise ValueError("benign_feat must be a 1D array.")

    # decide output length
    if num_features is None:
        L = len(benign_feat)
    else:
        L = int(num_features)
        if L <= 0:
            raise ValueError("num_features must be positive.")

    # cyclic benign feature view (float, untouched)
    if L <= len(benign_feat):
        benign_used = benign_feat[:L]
    else:
        reps = int(np.ceil(L / len(benign_feat)))
        benign_used = np.tile(benign_feat, reps)[:L]

    benign_used_list = benign_used.tolist()

    # -----------------------------
    # sampling-based encoder (codes)
    # -----------------------------
    if pk_mode == "uniform":
        P = np.ones(K) / K
    elif pk_mode == "manual":
        if P_K_manual is None or len(P_K_manual) != K:
            raise ValueError("P_K_manual must be provided with length K.")
        P = np.asarray(P_K_manual, dtype=float)
        P = P / P.sum()
    elif pk_mode == "dirichlet":
        if dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive.")
        P = rng.dirichlet(dirichlet_alpha * np.ones(K))
    else:
        raise ValueError(f"Unknown pk_mode: {pk_mode}")

    codes = rng.choice(np.arange(K, dtype=int), size=L, replace=True, p=P)

    # -----------------------------
    # quantizer (main-style)
    # -----------------------------
    feat_quantizer = FeatureQuantizerConfig(
        enable=True,
        scale=float(quantizer_scale),
        method="round",
    )

    results: Dict[str, np.ndarray] = {}

    for method in embedding_methods:
        embedder_cfg = EmbedderConfig(
            method=method,
            K=K,
            feat_quantizer=feat_quantizer,
        )
        embedder = Embedder(embedder_cfg)

        # internal integer-domain embedding
        embedded_int = embedder.embed(codes, benign_used_list)

        # convert back to original domain (for interface consistency only)
        embedded_float = np.asarray(embedded_int, dtype=float) / float(quantizer_scale)

        results[method] = embedded_float

    return results


# -----------------------------
# Main
# -----------------------------

def _print_quantizer_help() -> None:
    print("\n=== Feature quantization helpers ===")
    print("Use quantize_features(feats, scale=..., method='round'|'floor'|'ceil') to convert float feats to int.")
    print("Typical scale choices for timing feats:")
    print("  seconds -> milliseconds: scale=1e3")
    print("  seconds -> microseconds: scale=1e6")
    print("  seconds -> nanoseconds: scale=1e9")
    print("Embedder/Extractor can auto-quantize if feat_quantizer.enable=True.\n")


def main() -> None:
    cfg = ExperimentConfig(seed=123)
    set_global_seed(cfg.seed)

    _print_quantizer_help()

    # Build benign features (float), similar to IPD in seconds
    rng = np.random.default_rng(2026)
    benign_feat_float = rng.lognormal(mean=-3.0, sigma=0.7, size=20000).tolist()

    # We deliberately keep benign features as float here.
    # Embedder/Extractor will quantize them deterministically according to feat_quantizer.
    cfg.embedder.feat_quantizer = FeatureQuantizerConfig(enable=True, scale=1e6, method="round")
    cfg.extractor.feat_quantizer = FeatureQuantizerConfig(enable=True, scale=1e6, method="round")

    # -----------------------------
    # Test 1: bits mode + fixed scheme + model embedding
    # -----------------------------
    cfg.bit_source = BitSourceConfig(bit_length=300, p1=0.5, seed=1)

    cfg.encoder = EncoderConfig(
        mode="bits",
        bits_scheme="fixed",
        fixed=FixedBitsConfig(bit_unit_size=3),
        seed=2,
    )

    enc = Encoder(cfg.encoder)
    K_enc = enc.infer_K()

    cfg.embedder = EmbedderConfig(
        method="model",
        K=K_enc,
        model=ModelConfig(interpolate=True, quantile_edges=None, seed=3),
        feat_quantizer=cfg.embedder.feat_quantizer,
    )
    cfg.extractor = ExtractorConfig(
        method="model",
        K=K_enc,
        model=ModelConfig(interpolate=True, quantile_edges=None, seed=3),
        feat_quantizer=cfg.extractor.feat_quantizer,
    )

    cfg.decoder = DecoderConfig(
        bits_scheme="fixed",
        fixed=FixedBitsConfig(bit_unit_size=3),
    )

    bits = BitSource(cfg.bit_source).sample()
    codes = enc.encode(bits=bits)

    embedded_feat = Embedder(cfg.embedder).embed(codes, benign_feat_float)
    recovered_codes = Extractor(cfg.extractor).extract(embedded_feat, benign_feat_float)
    recovered_bits = Decoder(cfg.decoder).decode(recovered_codes)

    min_len = min(len(bits), len(recovered_bits))
    bit_errors = int(np.sum(bits[:min_len] != recovered_bits[:min_len]))
    ber = bit_errors / float(min_len) if min_len > 0 else 0.0

    valid_code_mask = recovered_codes >= 0
    code_valid_rate = float(np.mean(valid_code_mask)) if len(recovered_codes) > 0 else 0.0
    code_match = float(np.mean(codes[:len(recovered_codes)] == recovered_codes)) if len(recovered_codes) > 0 else 0.0

    print("=== Test 1 (bits mode, fixed scheme, model embedding) ===")
    print(f"Encoder implied K: {K_enc}")
    print(f"Bits length: {len(bits)}")
    print(f"Codes length: {len(codes)}")
    print(f"Recovered codes length: {len(recovered_codes)}")
    print(f"Recovered bits length: {len(recovered_bits)}")
    print(f"Code valid rate (not -1): {code_valid_rate:.4f}")
    print(f"Code exact match rate: {code_match:.4f}")
    print(f"Bit error rate on aligned prefix: {ber:.4f}")

    # -----------------------------
    # Test 2: sampling mode + replay embedding + decode anyway
    # -----------------------------
    cfg.encoder = EncoderConfig(
        mode="sampling",
        bits_scheme="fixed",  # ignored in sampling mode
        sampling=SamplingConfig(pk_mode="dirichlet", K=8, L=120, P_K_manual=None, dirichlet_alpha=0.7),
        seed=7,
    )
    enc2 = Encoder(cfg.encoder)
    K2 = enc2.infer_K()

    cfg.embedder = EmbedderConfig(
        method="replay",
        K=K2,
        replay=ReplayConfig(order="random", seed=9),
        feat_quantizer=cfg.embedder.feat_quantizer,
    )
    cfg.extractor = ExtractorConfig(
        method="replay",
        K=K2,
        replay=ReplayConfig(order="random", seed=9),
        feat_quantizer=cfg.extractor.feat_quantizer,
    )

    # Decoder translates codes to bits even when codes do not come from bits
    cfg.decoder = DecoderConfig(
        bits_scheme="fixed",
        fixed=FixedBitsConfig(bit_unit_size=3),
    )

    codes2 = enc2.encode(bits=None)
    embedded2 = Embedder(cfg.embedder).embed(codes2, benign_feat_float)
    recovered_codes2 = Extractor(cfg.extractor).extract(embedded2, benign_feat_float)
    bits2 = Decoder(cfg.decoder).decode(recovered_codes2)

    print("\n=== Test 2 (sampling mode, replay embedding, decode anyway) ===")
    print(f"Sampling K: {K2}")
    print(f"Codes length: {len(codes2)}")
    print(f"Recovered codes length: {len(recovered_codes2)}")
    print(f"Decoded bits length: {len(bits2)}")
    print(f"First 32 codes: {codes2[:32].tolist()}")
    print(f"First 64 decoded bits: {bits_to_str(bits2[:64])}")

    # -----------------------------
    # Test 3: optional scheme tests (spreading, geometric, rs)
    # -----------------------------
    print("\n=== Test 3 (optional schemes) ===")

    # 3a spreading
    try:
        cfg.encoder = EncoderConfig(
            mode="bits",
            bits_scheme="spreading",
            spreading=SpreadingConfig(code_unit_size=8, bit_unit_size=3),
            seed=11,
        )
        enc3 = Encoder(cfg.encoder)
        K3 = enc3.infer_K()
        cfg.embedder = EmbedderConfig(
            method="model",
            K=K3,
            model=ModelConfig(interpolate=True, seed=12),
            feat_quantizer=cfg.embedder.feat_quantizer,
        )
        cfg.extractor = ExtractorConfig(
            method="model",
            K=K3,
            model=ModelConfig(interpolate=True, seed=12),
            feat_quantizer=cfg.extractor.feat_quantizer,
        )
        cfg.decoder = DecoderConfig(
            bits_scheme="spreading",
            spreading=cfg.encoder.spreading,
        )

        bits3 = BitSource(BitSourceConfig(bit_length=240, seed=10)).sample()
        codes3 = enc3.encode(bits=bits3)
        embedded3 = Embedder(cfg.embedder).embed(codes3, benign_feat_float)
        recovered_codes3 = Extractor(cfg.extractor).extract(embedded3, benign_feat_float)
        bits3_hat = Decoder(cfg.decoder).decode(recovered_codes3)

        m = min(len(bits3), len(bits3_hat))
        ber3 = float(np.mean(bits3[:m] != bits3_hat[:m])) if m > 0 else 0.0
        print(f"Spreading scheme ok, K={K3}, codes_len={len(codes3)}, BER={ber3:.4f}")
    except Exception as e:
        print(f"Spreading scheme skipped: {e}")

    # 3b geometric
    try:
        cfg.encoder = EncoderConfig(
            mode="bits",
            bits_scheme="geometric",
            geometric=GeometricConfig(bit_unit_size=4, code_unit_size=3, threshold_delta_sum=6),
            seed=21,
        )
        enc4 = Encoder(cfg.encoder)
        K4 = enc4.infer_K()
        cfg.embedder = EmbedderConfig(
            method="model",
            K=K4,
            model=ModelConfig(interpolate=True, seed=22),
            feat_quantizer=cfg.embedder.feat_quantizer,
        )
        cfg.extractor = ExtractorConfig(
            method="model",
            K=K4,
            model=ModelConfig(interpolate=True, seed=22),
            feat_quantizer=cfg.extractor.feat_quantizer,
        )
        cfg.decoder = DecoderConfig(
            bits_scheme="geometric",
            geometric=cfg.encoder.geometric,
        )

        bits4 = BitSource(BitSourceConfig(bit_length=256, seed=20)).sample()
        codes4 = enc4.encode(bits=bits4)
        embedded4 = Embedder(cfg.embedder).embed(codes4, benign_feat_float)
        recovered_codes4 = Extractor(cfg.extractor).extract(embedded4, benign_feat_float)
        bits4_hat = Decoder(cfg.decoder).decode(recovered_codes4)

        m = min(len(bits4), len(bits4_hat))
        ber4 = float(np.mean(bits4[:m] != bits4_hat[:m])) if m > 0 else 0.0
        print(f"Geometric scheme ok, K={K4}, codes_len={len(codes4)}, BER={ber4:.4f}")
    except Exception as e:
        print(f"Geometric scheme skipped: {e}")

    # 3c rs
    try:
        cfg.encoder = EncoderConfig(
            mode="bits",
            bits_scheme="rs",
            rs=RSConfig(bit_unit_size=8, code_unit_size=16, parity_size=6, c_exp=8),
            seed=31,
        )
        enc5 = Encoder(cfg.encoder)
        K5 = enc5.infer_K()
        cfg.embedder = EmbedderConfig(
            method="model",
            K=K5,
            model=ModelConfig(interpolate=True, seed=32),
            feat_quantizer=cfg.embedder.feat_quantizer,
        )
        cfg.extractor = ExtractorConfig(
            method="model",
            K=K5,
            model=ModelConfig(interpolate=True, seed=32),
            feat_quantizer=cfg.extractor.feat_quantizer,
        )
        cfg.decoder = DecoderConfig(
            bits_scheme="rs",
            rs=cfg.encoder.rs,
        )

        bits5 = BitSource(BitSourceConfig(bit_length=512, seed=30)).sample()
        codes5 = enc5.encode(bits=bits5)
        embedded5 = Embedder(cfg.embedder).embed(codes5, benign_feat_float)
        recovered_codes5 = Extractor(cfg.extractor).extract(embedded5, benign_feat_float)
        bits5_hat = Decoder(cfg.decoder).decode(recovered_codes5)

        m = min(len(bits5), len(bits5_hat))
        ber5 = float(np.mean(bits5[:m] != bits5_hat[:m])) if m > 0 else 0.0
        print(f"RS scheme ok, K={K5}, codes_len={len(codes5)}, BER={ber5:.4f}")
    except Exception as e:
        print(f"RS scheme skipped: {e}")


if __name__ == "__main__":
    main()
