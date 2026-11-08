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
    for d in (32, 128):   # MK_BIGT (2026-11-06): one 2K prefill chunk next to d decodes (2080 / 2176 tokens per step)
        seqs = (Seq(1 * K, 2 * K),) + tuple(Seq(1 * K, 1) for _ in range(d))
        w.append(Workload(f"R7-p2048-d{d}", "R7", seqs, {"D": d, "P": 2 * K}, check=d == 32))
    rng = random.Random(20261022)
    seqs = tuple(Seq(rng.randint(128, 8 * K), 1) for _ in range(32))
    w.append(Workload("R8-b32-ragged", "R8", seqs, {"B": 32, "kv": "U[128,8K]"}, check=True))
    for kv in (2 * K, 8 * K, 12 * K, 15 * K, 17 * K, 24 * K, 31 * K, 32 * K, 33 * K):
        w.append(_uniform("R9", f"R9-b8-kv{kv}", 8, kv, 1, {"B": 8, "kv": kv}))
    for b, kv in ((16, 512), (16, 1 * K), (16, 2 * K), (16, 4 * K), (16, 8 * K), (32, 4 * K), (32, 7 * K)):   # h64 threshold sweep
        w.append(_uniform("R9", f"R9-b{b}-kv{kv}", b, kv, 1, {"B": b, "kv": kv}))
    for b in (1, 2, 4, 8, 16):   # iso total KV: 240K cache tokens split over b sequences
        kv = 240 * K // b
        w.append(_uniform("R9", f"R9-b{b}-kv{kv}", b, kv, 1, {"B": b, "kv": kv}))
    w.extend(r10_points())
    return [x for x in w if x.kv_cache_bytes <= KV_BYTES_CAP]


# R10: decode-only steps with 33-256 requests at long KV (the Azure 2024 trace, conv L3 / L4: 34-49 % of stock GPU time).
# Census (probe/mk_core/r10/census_d33.py, stock step logs of the M2 grid): GPU-time-weighted n_req p10/p50/p90 37/54/84 (conv L3),
# 47/87/198 (conv L4); mean kv per step p10..p90 2.3K..3.3K, max kv 5.9K..8.1K, kv CV within a step ~0.6 (code 0.8).
# Memory cap, computed from job 11565 (R9-b1-kv245760: 6 KV copies of 26.25 GiB + 5.3 GiB other = 162.8 GiB in use, then OOM
# on the next 26.25 GiB; device 178.35 GiB): a timed point holds 4 KV copies at once (fixture, mk:auto packed clone,
# fi-graph graph-pdl-best and graph-pdl-cublas paged caches; run with --variants "mk=auto;fi-graph=graph-pdl-best,graph-pdl-cublas"),
# so 4 x 40 GiB + 5.3 = 165.3 GiB (13 GiB margin). An accuracy point also holds a second-seed fixture and one temporary arm
# copy (6 copies): <= 25 GiB. vLLM runs afterwards in its own process.
R10_KV_CAP = 40 * 2**30
R10_CHECK_CAP = 25 * 2**30
_R10_GRID = ((48, (1 * K, 2 * K, 4 * K)), (64, (1 * K, 2 * K, 4 * K)), (96, (1 * K, 2 * K, 3 * K)),
             (128, (1 * K, 2 * K, 2816)), (192, (1 * K, 1792)), (256, (1 * K, 1408)))
_R10_CHECK = ("R10-b64-kv2048", "R10-b192-kv1024", "R10-conv-L3-n54")


def r10_trace_steps() -> list[dict]:
    """Real R10 steps (per-request kv lists) from regimebench/r10_steps.json (source run + line in each entry)."""
    import json
    from pathlib import Path
    return json.loads((Path(__file__).with_name("r10_steps.json")).read_text())["steps"]


def r10_points() -> list[Workload]:
    w = [_uniform("R10", f"R10-b{b}-kv{kv}", b, kv, 1, {"B": b, "kv": kv}, check=f"R10-b{b}-kv{kv}" in _R10_CHECK)
         for b, kvs in _R10_GRID for kv in kvs]
    for st in r10_trace_steps():
        kv = st["kv"]
        w.append(Workload(st["id"], "R10", tuple(Seq(k, 1) for k in kv),
                          {"B": len(kv), "kv": round(sum(kv) / len(kv)), "trace": st["id"].split("-", 1)[1]},
                          check=st["id"] in _R10_CHECK))
    for x in w:
        cap = R10_CHECK_CAP if x.check else R10_KV_CAP
        if x.kv_cache_bytes > cap:
            raise ValueError(f"{x.id}: {x.kv_cache_bytes / 2**30:.1f} GiB KV > R10 cap {cap / 2**30:.0f} GiB")
    return w


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
    # Stage 1 KV sweep at fixed batch 8 (one mk build: NPAD 16, splitkv at every point)
    "mk_b8kv": ("R3-b8-kv1024", "R3-b8-kv4096", "R3-b8-kv16384", *(f"R9-b8-kv{kv}" for kv in (2 * K, 8 * K, 12 * K, 24 * K, 32 * K))),
    # power-of-two KV check: +-1K around 16K and 32K
    "mk_iso240": tuple(f"R9-b{b}-kv{240 * K // b}" for b in (1, 2, 4, 8, 16)),
    # R10 (2026-10-31): decode, 33-256 requests, long KV; uniform grid + 3 real trace steps (heterogeneous kv)
    "mk_r10": tuple(f"R10-b{b}-kv{kv}" for b, kvs in _R10_GRID for kv in kvs) + ("R10-conv-L3-n54", "R10-conv-L4-n87", "R10-code-L3-n40"),
    # MK_BIGT step 1 (branch mk-bigt): steps of 257..8192 tokens on the one-launch large-T build (placeholder attention); R5-b16-T2048 (32K tokens) is out of range
    "mk_bigt": ("R4-b1-T1024", "R4-b1-T4096", "R4-b1-T8192", "R5-b4-T512", "R5-b4-T2048", "R5-b16-T512",
                "R7-p512-d8", "R7-p512-d32", "R7-p512-d128", "R7-p2048-d32", "R7-p2048-d128"),
    "mk_b8kv_p2": ("R3-b8-kv16384", *(f"R9-b8-kv{kv}" for kv in (15 * K, 17 * K, 24 * K, 31 * K, 32 * K, 33 * K))),
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
