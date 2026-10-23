"""mk host side: source assembly, NVRTC builds per token bucket (NPAD), RoPE table, tensor-map specs. Compiling needs no GPU."""

from __future__ import annotations

import hashlib
import os
import re
import struct
from functools import lru_cache
from pathlib import Path

from ..hv2g.build import MAXTOK, build_descriptors, rope_table  # noqa: F401  (read-only reuse of hv2g descriptors / RoPE table)
from .coop import nvrtc_compile, parse_ptxas

# NVRTC / ptxas keep a compile cache that returns an empty log on a hit: the ptxas table needs the log
os.environ.setdefault("CUDA_CACHE_DISABLE", "1")
HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
BUCKETS = (16, 32, 64, 128, 256)
CTX = 128                     # attention padding rows after the last cache layer (placeholder chunk size)
H, NL, QH, KVH, HD, INTER, VOCAB = 1024, 28, 16, 8, 128, 3072, 151936
GRID = 148
MAXTOK_MK = 256
SYNC_INTS = 128
KERNEL = "mk_step"
ATTN_PLACEHOLDER = HERE / "attn_placeholder.cuh"
ATTN_SPLITKV = COMMON / "splitkv" / "splitkv.cuh"
SKV_BAR_OFF, SKV_BAR_BYTES = 147968, 64   # splitkv-v1 (128 x 2 TMA ring): skv::kBarOff, skv::kBarBytes; attend scratch = kBarOff + kBarBytes = 148,032 B (compile-checked by MK_EXPECT_DYN)
OWN_SOURCES = ("mk_common.cuh", "mk_step.cu")
DEFAULT_KNOBS = {"MK_KU": 128, "MK_RING": 2, "MK_XTMA": 0, "MK_STAMPS": 0, "MK_NULL": 0, "MK_STG": 4, "MK_EPI2": 0, "MK_DBG_EXIT": 0, "MK_PFA": 1, "MK_LMP": 1, "MK_LMR": 0, "MK_TS": 128, "MK_TSB": 1}
WEIGHT_MAPS = ("wq", "wk", "wv", "wo", "wg", "wu", "wd", "embed")   # tensor-map order (kernel: maps 0..7)
WORKSPACE_MAPS = ("xb",)   # kernel map 8 (MK_TS): bf16 [MAXTOK_MK][H] workspace of the tile-stream QKV (B tiles by TMA)
TSB_MAPS = ("xb2", "wgu", "act")   # kernel maps 9, 10, 11 (MK_TSB): xb2 bf16 [MAXTOK][H] (B of GU), wgu bf16 [NL * 6144][H] (A of GU), act bf16 [MAXTOK][INTER] (B of D)
WS_ORDER = WORKSPACE_MAPS + TSB_MAPS
TS_TT = 128   # TS token tile (mk_common.cuh TS_TT)
TS_EXCH = 12288   # MK_TSB: GU exchange region after the ring (mk_common.cuh TS_EXCH)
TK_SIZE = 64      # u32 tickets: O at [0, 16), D at [16, 32) (mk_step.cu TK_D)


def ts_tt(npad: int) -> dict:
    """Token tile TT per tile-stream phase (mk_common.cuh TS_TT, GU_TT, D_TT): the box rows of the B maps xb, xb2, act."""
    return {"qkv": TS_TT, "gu": 128 if npad == 256 else 64, "d": 128}


def ts_ks(npad: int) -> dict:
    """Split-K per tile-stream phase (mk_common.cuh TS_KS, GU_KS, D_KS)."""
    return {"qkv": 2 if npad == 256 else 4, "gu": 1, "d": 8 if npad == 256 else 12}


def tsb_on(npad: int, knobs: dict | None = None) -> bool:
    """MK_TSB active at this bucket (mk_common.cuh TSB = TS && MK_TSB)."""
    k = {**DEFAULT_KNOBS, **(knobs or {})}
    return bool(k["MK_TSB"]) and bool(k["MK_TS"]) and npad >= ts_min(k["MK_TS"]) and k["MK_KU"] == 128


def interleave_gu(wg, wu):
    """wgu [NL * 6144][H] from wg, wu [NL][3072][H] (torch tensors): per layer, for p in 0..47: rows [128 p, 128 p + 64) = wg rows [64 p, 64 p + 64),
    rows [128 p + 64, 128 p + 128) = wu rows [64 p, 64 p + 64)."""
    import torch
    nl, inter, h = wg.shape
    assert tuple(wu.shape) == (nl, inter, h) and inter % 64 == 0
    return torch.stack([wg.reshape(nl, inter // 64, 64, h), wu.reshape(nl, inter // 64, 64, h)], dim=2).reshape(nl * 2 * inter, h)


def bucket_for(ntok: int) -> int:
    """Smallest NPAD >= ntok."""
    for b in BUCKETS:
        if ntok <= b:
            return b
    raise ValueError(f"{ntok} tokens exceed the largest bucket {BUCKETS[-1]}")


def attn_header(attn: str = "placeholder", path: Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    if attn == "placeholder":
        return ATTN_PLACEHOLDER
    if attn == "splitkv":
        if not ATTN_SPLITKV.exists():
            raise FileNotFoundError(f"MK_ATTN=splitkv needs {ATTN_SPLITKV} (import a splitkv-vN tag first: "
                                    "git checkout splitkv-vN -- regimebench/kernels/common/splitkv)")
        return ATTN_SPLITKV
    raise ValueError(f"unknown MK_ATTN {attn!r}")


def source(attn: str = "placeholder", attn_path: Path | None = None) -> str:
    """Attention header text, then mk_common.cuh, then mk_step.cu."""
    return "\n".join([attn_header(attn, attn_path).read_text()] + [(HERE / f).read_text() for f in OWN_SOURCES])


def source_sha256(attn: str = "placeholder", attn_path: Path | None = None) -> str:
    return hashlib.sha256(source(attn, attn_path).encode()).hexdigest()


def strip_knobs(src: str) -> str:
    """Source without the knob code: `#if MK_STAMPS ... #endif` blocks removed, mk_null(p) calls replaced by false."""
    src = re.sub(r"^#if MK_STAMPS\n.*?^#endif[^\n]*\n", "", src, flags=re.S | re.M)
    return re.sub(r"mk_null\(\d+\)", "false", src)


def knob_defines(knobs: dict | None) -> list[str]:
    k = {**DEFAULT_KNOBS, **(knobs or {})}
    unknown = set(k) - set(DEFAULT_KNOBS)
    if unknown:
        raise KeyError(f"unknown knobs {sorted(unknown)}")
    return [f"{n}={int(v)}" for n, v in sorted(k.items())]


# placeholder SKV_SMEM_BYTES: SKV_CTX x 128 bf16 K/V chunk + 16 warps x 128 fp32 partial o + 2 x 16 fp32 (m, l)
ATTN_SMEM_PLACEHOLDER = CTX * 128 * 2 + 16 * 128 * 4 + 2 * 16 * 4


def lm_ring(npad: int, lmr: int) -> int:
    """MK_LMR as the kernel uses it (mk_common.cuh LMR): 0 = per bucket, 4 slots at NPAD 16, 3 at NPAD 32 (job 9586), 2 at NPAD 64 (smem)."""
    return lmr if lmr else {16: 4, 32: 3}.get(npad, 2)


def ts_min(ts: int) -> int:
    """Smallest NPAD on the tile stream for MK_TS (mk_common.cuh TS_MIN): 0 = off, 1 = 128, else the value."""
    return 128 if ts == 1 else ts


def ts_ring(ts: int = 1, tt: int = TS_TT, tsb: int = 0) -> int:
    """Tile-stream ring depth (mk_common.cuh ts_ringn; TS_R for the QKV phase): slots of 32 KB A + tt x 256 B B that fit the 227 KB budget
    (static 11280, align 1024, barriers 384, + the GU exchange region with tsb)."""
    if not ts:
        return 1
    return min(8, (227 * 1024 - 11280 - 1024 - 384 - (TS_EXCH if tsb else 0)) // (32768 + tt * 256))


def skv_off_bars(npad: int, ku: int = 128, ring: int = 2, attn: str = "placeholder", lmp: int = 1, lmr: int = 0, ts: int = 0, tsb: int = 0) -> int:
    """OFF_BARS of the kernel layout (mk_common.cuh): end of the overlay regions; under attn = splitkv at least kBarOff and = 512 (mod 1024)."""
    nbox = ku // 64
    ub, xs = nbox * 16384, nbox * npad * 128
    lay = ring * ub + 2 * xs
    if ts and npad >= ts_min(ts) and ku == 128:
        tt = ts_tt(npad)
        if tsb:
            lay = max(lay, max(ts_ring(1, tt[p], 1) * (32768 + tt[p] * 256) for p in ("qkv", "gu", "d")) + TS_EXCH)
        else:
            lay = max(lay, ts_ring(1) * (32768 + TS_TT * 256))
    elif lmp and npad <= 64 and ku == 128:
        lay = max(lay, 8 * xs + lm_ring(npad, lmr) * ub)
    if attn == "splitkv":
        lay = max(lay, SKV_BAR_OFF)
        return (lay - 512 + 1023) // 1024 * 1024 + 512
    return lay


def dyn_smem(npad: int, ku: int = 128, ring: int = 2, attn_bytes: int = ATTN_SMEM_PLACEHOLDER, lmp: int = 1, lmr: int = 0, ts: int = 0, tsb: int = 0, attn: str = "placeholder") -> int:
    """Dynamic smem of a bucket (mirrors mk_common.cuh DYN_SMEM; the kernel traps if the launch gives less).
    lmp: MK_LMP (the pipelined LM head only applies at npad <= 64: smem = [8 X chunks][lmr x 32 KB ring], 384 B of barriers).
    attn = "splitkv" (SPEC-P2): OFF_BARS = skv_off_bars(..) (>= skv::kBarOff, = 512 mod 1024), A1's 64 B mbarrier tail at [OFF_BARS, +64), mk's barrier block (384 / 256 B) behind it."""
    if attn == "splitkv":
        if ku != 128:
            raise ValueError("MK_ATTN=splitkv needs MK_KU 128")
        tail = 384 if ((lmp and npad <= 64) or (ts and npad >= ts_min(ts))) else 256
        return 1024 + skv_off_bars(npad, ku, ring, attn, lmp, lmr, ts, tsb) + SKV_BAR_BYTES + tail
    nbox = ku // 64
    ub, xs = nbox * 16384, nbox * npad * 128
    off_bars = ring * ub + max(2 * xs, attn_bytes)
    if ts and npad >= ts_min(ts) and ku == 128:   # MK_TS: tile-stream ring overlays the other regions; 384 B of barriers
        tt = ts_tt(npad)
        if tsb:   # MK_TSB: ring bytes = the largest over the phases (QKV / D at TT 128, GU at TT 128 or 64), then the GU exchange region
            ring_b = max(ts_ring(1, tt[p], 1) * (32768 + tt[p] * 256) for p in ("qkv", "gu", "d"))
            return 1024 + max(off_bars, ring_b + TS_EXCH) + 384
        return 1024 + max(off_bars, ts_ring(1) * (32768 + TS_TT * 256)) + 384
    if lmp and npad <= 64 and ku == 128:
        return 1024 + max(off_bars, 8 * xs + lm_ring(npad, lmr) * ub) + 384
    return 1024 + off_bars + 256


@lru_cache(maxsize=None)
def compile_build(npad: int, nb: int = GRID, knobs: tuple = (), arch: str = "sm_100a", attn: str = "placeholder",
                  attn_path: str | None = None, stripped: bool = False) -> tuple[bytes, dict, str]:
    """-> (cubin, ptxas info, log). ``knobs``: sorted tuple of (name, value)."""
    if npad not in BUCKETS:
        raise ValueError(f"NPAD {npad} not in {BUCKETS}")
    src = source(attn, Path(attn_path) if attn_path else None)
    if stripped:
        src = strip_knobs(src)
    k = {**DEFAULT_KNOBS, **dict(knobs)}
    exp = dyn_smem(npad, k["MK_KU"], k["MK_RING"], lmp=k["MK_LMP"], lmr=k["MK_LMR"], ts=k["MK_TS"], tsb=k["MK_TSB"], attn=attn) if attn_path is None and (attn == "placeholder" or k["MK_KU"] == 128) else 0
    defs = [f"NPAD={npad}", f"NB={nb}", f"MK_EXPECT_DYN={exp}", *knob_defines(dict(knobs))]
    defs += os.environ.get("MK_EXTRA_DEFINES", "").split()   # debug only (e.g. SKV_DEBUG_WAIT=1); not part of the knob set
    if attn == "splitkv":
        defs.append("MK_SKV=1")
    cubin, log = nvrtc_compile(src, defs, arch=arch, ptxas_verbose=True, name="mk.cu")
    return cubin, parse_ptxas(log), log


# ---- A1 (attn = splitkv) KV tensor maps: 2 * kNumMaps 3D bf16 maps (K then V), box rows 128, 64, .., 1, 128 B each ----
SKV_STAGE_ROWS = 128
SKV_NUM_MAPS = SKV_STAGE_ROWS.bit_length()   # log2(128) + 1 = 8 (skv::kNumMaps)


def skv_map_specs(rows: int, n_layers: int = NL) -> list[tuple[list[int], list[int], list[int]]]:
    """(dims, strides_bytes, box) of the 16 A1 maps in kernel order (K: box rows 128..1, then V): dims {128, 8, n_layers * rows}, strides {256, 2048} B,
    box {128, 1, r}, no swizzle (a1bench.py tensor_maps). ``rows`` = cache rows per layer (mk: the fixture's packed rows; the CTX padding rows lie behind the last layer)."""
    out = []
    for _kv in range(2):
        for rs in range(SKV_NUM_MAPS):
            out.append(([HD, KVH, n_layers * rows], [HD * 2, KVH * HD * 2], [HD, 1, SKV_STAGE_ROWS >> rs]))
    return out


def encode_skv_maps(kc_ptr: int, vc_ptr: int, rows: int, n_layers: int = NL) -> bytes:
    """16 x 128 B tensor maps over the K / V caches (device pointers); needs cuInit (a GPU host)."""
    from cuda.bindings import driver as d
    blob = b""
    for i, (dims, strides, box) in enumerate(skv_map_specs(rows, n_layers)):
        res = d.cuTensorMapEncodeTiled(
            d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, int(kc_ptr if i < SKV_NUM_MAPS else vc_ptr),
            [d.cuuint64_t(v) for v in dims], [d.cuuint64_t(v) for v in strides], [d.cuuint32_t(v) for v in box],
            [d.cuuint32_t(1)] * 3, d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
            d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE, d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
        if int(res[0]) != 0:
            raise RuntimeError(f"cuTensorMapEncodeTiled(skv map {i}): {res[0]}")
        blob += struct.pack("<16Q", *[int(v) for v in res[1].opaque])
    return blob


def auto_attn(seqs) -> str:
    """attn = "auto": splitkv when every sequence has q_len == 1 (A1 v1: pair_info is indexed by token), else the placeholder. ``seqs``: iterable of (kv_len, q_len)."""
    return "splitkv" if all(int(q) == 1 for _, q in seqs) else "placeholder"


# ---- tensor maps (host; 2D bf16 SW128, box {64 k, 128 rows}, one per weight tensor, 128 B each in global memory) ----
def map_specs(weights: dict) -> list[tuple[list[int], list[int], list[int]]]:
    """(dims, strides_bytes, box) per map in WEIGHT_MAPS order; dims = {K, layers * rows}. ``weights``: name -> tensor-like with .shape."""
    out = []
    for name in WEIGHT_MAPS:
        shp = tuple(weights[name].shape)
        if name == "embed":
            rows, k = shp
        else:
            rows, k = shp[0] * shp[1], shp[2]
        out.append(([k, rows], [k * 2], [64, 128]))
    return out


def workspace_map_specs(npad: int = 256, names: tuple = WORKSPACE_MAPS) -> list[tuple[list[int], list[int], list[int]]]:
    """(dims, strides_bytes, box) of the workspace maps ``names`` (default map 8 ``xb``: bf16 [MAXTOK_MK][H], dims {H, MAXTOK_MK}, box {64 k, TS_TT tokens}, SW128).
    MK_TSB: ``xb2`` {H, MAXTOK_MK} box {64, TT_GU}; ``wgu`` {H, NL * 6144} box {64, 128}; ``act`` {INTER, MAXTOK_MK} box {64, TT_D}. TT per bucket: ts_tt(npad)."""
    tt = ts_tt(npad)
    spec = {"xb": ([H, MAXTOK_MK], [H * 2], [64, tt["qkv"]]),
            "xb2": ([H, MAXTOK_MK], [H * 2], [64, tt["gu"]]),
            "wgu": ([H, NL * 2 * INTER], [H * 2], [64, 128]),
            "act": ([INTER, MAXTOK_MK], [INTER * 2], [64, tt["d"]])}
    return [spec[n] for n in names]


def encode_maps(weights: dict, ptrs: dict, workspace: dict | None = None, npad: int = 256) -> bytes:
    """128 B x 8 tensor maps with the real device addresses (+ 128 B per workspace map of ``workspace`` = {"xb": device ptr, and with MK_TSB "xb2", "wgu", "act"},
    appended in WS_ORDER = maps 8, 9, 10, 11). Needs cuInit (a GPU host)."""
    from cuda.bindings import driver as d
    blob = b""
    names = list(WEIGHT_MAPS)
    specs = list(map_specs(weights))
    if workspace is not None:
        wn = tuple(n for n in WS_ORDER if n in workspace)
        names += list(wn)
        specs += workspace_map_specs(npad, wn)
        ptrs = {**ptrs, **workspace}
    for name, (dims, strides, box) in zip(names, specs):
        res = d.cuTensorMapEncodeTiled(
            d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, len(dims), int(ptrs[name]),
            [d.cuuint64_t(v) for v in dims], [d.cuuint64_t(v) for v in strides], [d.cuuint32_t(v) for v in box],
            [d.cuuint32_t(1)] * len(dims), d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
        if int(res[0]) != 0:
            raise RuntimeError(f"cuTensorMapEncodeTiled({name}): {res[0]}")
        blob += struct.pack("<16Q", *[int(v) for v in res[1].opaque])
    return blob
