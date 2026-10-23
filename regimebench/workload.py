"""Workload descriptor and the regime grid.

A ``Workload`` is one forward step over a list of sequences. Sequence ``i`` has
``kv_len`` cached positions (already in the KV cache) and ``q_len`` new tokens
that this step processes (decode: q_len 1; prefill: kv_len 0, q_len T; chunked
prefill / spec verify: kv_len > 0, q_len > 1). New token ``j`` of sequence
``i`` sits at position ``kv_len + j`` and attends positions ``0 .. kv_len + j``
(causal). The step writes the new K/V into positions ``kv_len .. kv_len+q_len-1``
of that sequence's cache and returns logits for the LAST new token of each
sequence. Re-running a step rewrites the same slots, so the work is constant
and steps can be launched back to back.

Tokens are packed in sequence order: ``cu_q[i] .. cu_q[i+1]`` are sequence i's
new tokens (``cu_q`` has len n_seq + 1, ``cu_q[0] == 0``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .geometry import QWEN3_06B, ModelGeometry

# Points whose fixture + arm caches would exceed this many KV bytes are skipped.
KV_BYTES_CAP = 40 * 2**30


@dataclass(frozen=True)
class Seq:
    kv_len: int
    q_len: int

    def __post_init__(self) -> None:
        if self.kv_len < 0 or self.q_len < 1:
            raise ValueError(f"invalid Seq kv={self.kv_len} q={self.q_len}")


@dataclass(frozen=True)
class Workload:
    id: str                       # unique, e.g. "R1-b32-kv128"
    regime: str                   # "R0".."R8"
    seqs: tuple[Seq, ...]
    axis: dict = field(default_factory=dict, compare=False, hash=False)  # varied knobs, for plots
    check: bool = False           # run the accuracy gate at this point
    model: ModelGeometry = QWEN3_06B

    @property
    def n_seq(self) -> int:
        return len(self.seqs)

    @property
    def total_q(self) -> int:
        return sum(s.q_len for s in self.seqs)

    @property
    def total_kv(self) -> int:
        return sum(s.kv_len for s in self.seqs)

    @property
    def max_kv(self) -> int:
        return max(s.kv_len for s in self.seqs)

    @property
    def max_q(self) -> int:
        return max(s.q_len for s in self.seqs)

    @property
    def cu_q(self) -> list[int]:
        out = [0]
        for s in self.seqs:
            out.append(out[-1] + s.q_len)
        return out

    @property
    def is_pure_decode(self) -> bool:
        return all(s.q_len == 1 for s in self.seqs)

    @property
    def cache_tokens(self) -> int:
        """Cache slots the step needs: every sequence's kv_len + q_len."""
        return sum(s.kv_len + s.q_len for s in self.seqs)

    @property
    def kv_cache_bytes(self) -> int:
        return self.cache_tokens * self.model.kv_bytes_per_token

    def to_dict(self) -> dict:
        return {"id": self.id, "regime": self.regime, "axis": dict(self.axis),
                "n_seq": self.n_seq, "total_q": self.total_q,
                "total_kv": self.total_kv, "max_kv": self.max_kv,
                "check": self.check,
                "seqs": [[s.kv_len, s.q_len] for s in self.seqs]}

    @staticmethod
    def from_dict(d: dict) -> "Workload":
        return Workload(id=d["id"], regime=d["regime"],
                        seqs=tuple(Seq(kv, q) for kv, q in d["seqs"]),
                        axis=dict(d.get("axis", {})), check=bool(d.get("check", False)))


def _uniform(regime: str, wid: str, n: int, kv: int, q: int, axis: dict,
             check: bool = False) -> Workload:
    return Workload(wid, regime, tuple(Seq(kv, q) for _ in range(n)), axis, check)


K = 1024


def regime_grid() -> list[Workload]:
    """Every point of the study. Anchor R0 = decode, batch 1, kv 128."""
    w: list[Workload] = []
    w.append(_uniform("R0", "R0-b1-kv128", 1, 128, 1, {"B": 1, "kv": 128}, check=True))
    for b in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        w.append(_uniform("R1", f"R1-b{b}-kv128", b, 128, 1, {"B": b, "kv": 128},
                          check=b in (8, 64)))
    for kv in (128, 512, 1 * K, 2 * K, 4 * K, 8 * K, 16 * K, 32 * K):
        w.append(_uniform("R2", f"R2-b1-kv{kv}", 1, kv, 1, {"B": 1, "kv": kv},
                          check=kv in (1 * K, 8 * K)))
    for b in (1, 8, 32, 128):
        for kv in (1 * K, 4 * K, 16 * K):
            w.append(_uniform("R3", f"R3-b{b}-kv{kv}", b, kv, 1, {"B": b, "kv": kv},
                              check=(b, kv) == (8, 4 * K)))
    for t in (16, 64, 256, 1 * K, 4 * K, 8 * K):
        w.append(_uniform("R4", f"R4-b1-T{t}", 1, 0, t, {"B": 1, "T": t},
                          check=t in (16, 256)))
    for b in (4, 16):
        for t in (512, 2 * K):
            w.append(_uniform("R5", f"R5-b{b}-T{t}", b, 0, t, {"B": b, "T": t},
                              check=(b, t) == (4, 512)))
    for q in (2, 4, 8, 16):
        w.append(_uniform("R6", f"R6-b1-kv1024-q{q}", 1, 1 * K, q, {"q": q, "kv": 1 * K},
                          check=q in (4, 16)))
    for d in (8, 32, 128):
        seqs = (Seq(1 * K, 512),) + tuple(Seq(1 * K, 1) for _ in range(d))
        w.append(Workload(f"R7-p512-d{d}", "R7", seqs, {"D": d}, check=d == 8))
    rng = random.Random(20261022)
    seqs = tuple(Seq(rng.randint(128, 8 * K), 1) for _ in range(32))
    w.append(Workload("R8-b32-ragged", "R8", seqs, {"B": 32, "kv": "U[128,8K]"}, check=True))
    return [x for x in w if x.kv_cache_bytes <= KV_BYTES_CAP]


def skipped_by_cap() -> list[str]:
    """Grid points dropped by KV_BYTES_CAP (recorded in the report)."""
    kept = {x.id for x in regime_grid()}
    full: list[str] = []
    for b in (1, 8, 32, 128):
        for kv in (1 * K, 4 * K, 16 * K):
            full.append(f"R3-b{b}-kv{kv}")
    return [i for i in full if i not in kept]


# Decode points targeted by the bespoke megakernel `mk` (docs/briefs/): room (best baseline / SOL, 2026-10-22 runs)
# >= 2. R3-b8-kv16K, R3-b32-kv4K, R3-b128-kv1K and R8 are out of scope. `mk_kv`: the kv >= 1K points (attention focus).
_MK_POINTS = ("R0-b1-kv128", *(f"R1-b{b}-kv128" for b in (1, 2, 4, 8, 16, 32, 64, 128, 256)),
              *(f"R2-b1-kv{kv}" for kv in (128, 512, 1 * K, 2 * K, 4 * K, 8 * K, 16 * K, 32 * K)),
              "R3-b1-kv1024", "R3-b1-kv4096", "R3-b1-kv16384", "R3-b8-kv1024", "R3-b8-kv4096", "R3-b32-kv1024",
              *(f"R6-b1-kv1024-q{q}" for q in (2, 4, 8, 16)))
OWNER_POINTS: dict[str, tuple[str, ...]] = {
    "mk": _MK_POINTS,
    "mk_kv": tuple(p for p in _MK_POINTS if int(p.split("-kv")[1].split("-")[0]) >= 1 * K),
}


def select(ids_or_regimes: list[str] | None) -> list[Workload]:
    """Filter the grid by workload ids, regime names (``R1``) or presets (``mk``, ``mk_kv``); None = all."""
    grid = regime_grid()
    if not ids_or_regimes:
        return grid
    want = set(ids_or_regimes)
    for name in ids_or_regimes:
        want |= set(OWNER_POINTS.get(name, ()))
    return [x for x in grid if x.id in want or x.regime in want]
