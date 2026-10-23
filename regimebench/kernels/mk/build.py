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
OWN_SOURCES = ("mk_common.cuh", "mk_step.cu")
DEFAULT_KNOBS = {"MK_KU": 128, "MK_RING": 2, "MK_XTMA": 0, "MK_STAMPS": 0, "MK_NULL": 0, "MK_STG": 4, "MK_EPI2": 0, "MK_DBG_EXIT": 0, "MK_PFA": 0}
WEIGHT_MAPS = ("wq", "wk", "wv", "wo", "wg", "wu", "wd", "embed")   # tensor-map order (kernel: maps 0..7)


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


def dyn_smem(npad: int, ku: int = 128, ring: int = 2, attn_bytes: int = ATTN_SMEM_PLACEHOLDER) -> int:
    """Dynamic smem of a bucket (mirrors mk_common.cuh DYN_SMEM; the kernel traps if the launch gives less)."""
    nbox = ku // 64
    ub, xs = nbox * 16384, nbox * npad * 128
    return 1024 + ring * ub + max(2 * xs, attn_bytes) + 256


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
    exp = dyn_smem(npad, k["MK_KU"], k["MK_RING"]) if attn == "placeholder" and attn_path is None else 0
    defs = [f"NPAD={npad}", f"NB={nb}", f"MK_EXPECT_DYN={exp}", *knob_defines(dict(knobs))]
    cubin, log = nvrtc_compile(src, defs, arch=arch, ptxas_verbose=True, name="mk.cu")
    return cubin, parse_ptxas(log), log


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


def encode_maps(weights: dict, ptrs: dict) -> bytes:
    """128 B x 8 tensor maps with the real device addresses. Needs cuInit (a GPU host)."""
    from cuda.bindings import driver as d
    blob = b""
    for name, (dims, strides, box) in zip(WEIGHT_MAPS, map_specs(weights)):
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
