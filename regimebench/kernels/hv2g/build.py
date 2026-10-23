"""hv2g host side: source assembly, NVRTC builds, launch descriptors, RoPE table. Everything here but the kernel
loading runs without a GPU (torch CPU tensors / plain lists)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .coop import nvrtc_compile, parse_ptxas

HERE = Path(__file__).resolve().parent
SOURCE_FILES = ("hv2g_common.cuh", "hv2g_mt1.cu", "hv2g_mtn.cu")
MAXTOK = 256        # tokens per MTN launch
CTX = 128           # attention chunk (kernel constant); the cache is padded with CTX rows
H, NL, QH, KVH, HD, INTER, VOCAB = 1024, 28, 16, 8, 128, 3072, 151936
TBQ = 8
MT1_DYN_SMEM = 5632 + 86016 + 43008   # stage + sGU + sD (as hv2)
MTN_DYN_SMEM = TBQ * H * 4


def source() -> str:
    return "\n".join((HERE / f).read_text() for f in SOURCE_FILES)


def source_sha256() -> str:
    return hashlib.sha256(source().encode()).hexdigest()


def kernel_name(mt1: bool) -> str:
    return "qwen3_step_mt1" if mt1 else "qwen3_step_mtn"


def dyn_smem(mt1: bool) -> int:
    return MT1_DYN_SMEM if mt1 else MTN_DYN_SMEM


def scratch_floats(mt1: bool) -> int:
    return 8192 if mt1 else MAXTOK * (H + 4096 + INTER + 2048)


SYNC_INTS = 128


@lru_cache(maxsize=None)
def compile_build(mt1: bool, nb: int = 148, arch: str = "sm_100") -> tuple[bytes, dict, str]:
    """-> (cubin, ptxas info dict, ptxas log)."""
    cubin, log = nvrtc_compile(source(), [f"NB={nb}", f"HV2G_MT1={int(mt1)}", f"MAXTOK={MAXTOK}"], arch=arch,
                               ptxas_verbose=True, name="hv2g.cu")
    return cubin, parse_ptxas(log), log


def rope_table(max_pos: int):
    """fp32 CPU [max_pos, 128]: row p = [cos(p * inv_freq) x64, sin(p * inv_freq) x64], theta 1e6, fp64 angles."""
    import torch
    inv = 1e6 ** (-torch.arange(0, HD, 2, dtype=torch.float64) / HD)
    ang = torch.arange(max_pos, dtype=torch.float64)[:, None] * inv[None, :]
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=1).float().contiguous()


@dataclass
class Chunk:
    """One MTN launch: tokens [t0, t0 + ntok) of the step; lm lists are local to the chunk."""
    t0: int
    ntok: int
    lm_tok: list[int] = field(default_factory=list)   # local token index
    lm_seq: list[int] = field(default_factory=list)   # output slot (sequence index)
    lm_off: int = 0                                   # offset of this chunk's entries in the global lm arrays


@dataclass
class Descriptors:
    t_base: list[int]        # per token: first cache row of its sequence
    t_vis: list[int]         # per token: visible rows = position + 1 (own row = t_base + t_vis - 1)
    lm_tok: list[int]        # concatenated per-chunk lists (local indices)
    lm_seq: list[int]
    chunks: list[Chunk]
    cache_rows: int
    max_pos: int             # RoPE rows needed


def build_descriptors(seqs, maxtok: int = MAXTOK) -> Descriptors:
    """``seqs``: iterable of (kv_len, q_len). Token j of sequence i sits at position kv_len + j of cache region
    cu_cache[i] (fixture packing: kv_len + q_len rows per sequence); the last token of each sequence feeds the LM head."""
    seqs = [(int(a), int(b)) for a, b in seqs]
    t_base, t_vis, last_of = [], [], {}
    row = 0
    for i, (kv, q) in enumerate(seqs):
        for j in range(q):
            t_base.append(row)
            t_vis.append(kv + j + 1)
        last_of[len(t_base) - 1] = i
        row += kv + q
    total = len(t_base)
    chunks: list[Chunk] = []
    lm_tok: list[int] = []
    lm_seq: list[int] = []
    for t0 in range(0, total, maxtok):
        n = min(maxtok, total - t0)
        c = Chunk(t0=t0, ntok=n, lm_off=len(lm_tok))
        for t in range(t0, t0 + n):
            if t in last_of:
                c.lm_tok.append(t - t0)
                c.lm_seq.append(last_of[t])
        lm_tok += c.lm_tok
        lm_seq += c.lm_seq
        chunks.append(c)
    return Descriptors(t_base, t_vis, lm_tok, lm_seq, chunks, row, max(t_vis) if t_vis else 1)
