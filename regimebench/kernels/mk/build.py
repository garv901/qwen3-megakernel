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
SKV_STAGE_BYTES = 128 * 512   # one 128-row K + V ring stage of the q_len 1 build
SKV_BAR_OFF, SKV_BAR_BYTES = 147968, 64   # splitkv-v1 (128 x 2 TMA ring): skv::kBarOff, skv::kBarBytes; attend scratch = kBarOff + kBarBytes = 148,032 B (compile-checked by MK_EXPECT_DYN)
OWN_SOURCES = ("mk_common.cuh", "mk_step.cu")
DEFAULT_KNOBS = {"MK_KU": 128, "MK_RING": 2, "MK_XTMA": 0, "MK_STAMPS": 0, "MK_NULL": 0, "MK_STG": 4, "MK_EPI2": 0, "MK_DBG_EXIT": 0, "MK_PFA": 1, "MK_LMP": 1, "MK_LMR": 0, "MK_TS": 128, "MK_TSB": 1, "MK_LMTS": 1, "MK_DBG_NOATT": 0, "MK_GU_TT": 0, "MK_FIN": -1, "MK_PFE": 0, "MK_QR": 0}
WEIGHT_MAPS = ("wq", "wk", "wv", "wo", "wg", "wu", "wd", "embed")   # tensor-map order (kernel: maps 0..7)
WORKSPACE_MAPS = ("xb",)   # kernel map 8 (MK_TS): bf16 [MAXTOK_MK][H] workspace of the tile-stream QKV (B tiles by TMA)
TSB_MAPS = ("xb2", "wgu", "act")   # kernel maps 9, 10, 11 (MK_TSB): xb2 bf16 [MAXTOK][H] (B of GU), wgu bf16 [NL * 6144][H] (A of GU), act bf16 [MAXTOK][INTER] (B of D)
WS_ORDER = WORKSPACE_MAPS + TSB_MAPS
TS_TT = 128   # TS token tile (mk_common.cuh TS_TT)
TS_EXCH = 12288   # MK_TSB: GU exchange region after the ring (mk_common.cuh TS_EXCH)
TK_SIZE = 192     # u32 tickets: O at [0, 16), D at [16, 32) (mk_step.cu TK_D); MK_FIN seen counters at [32, 64); MK_QR QKV tickets [64, 128), rope seen [128, 192)


def ts_tt(npad: int, gu_tt: int = 0) -> dict:
    """Token tile TT per tile-stream phase (mk_common.cuh TS_TT, GU_TT, D_TT): the box rows of the B maps xb, xb2, act. gu_tt: MK_GU_TT (0 = per-bucket default)."""
    return {"qkv": TS_TT, "gu": gu_tt or (96 if npad == 256 else 48), "d": 128}


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


# splitkv header knobs that mk passes through when set (else the header default applies): engine (MK-attn P8), item prefetch (P9), merge chunk (P4), KV evict_first + 1-row tail by LDG (v5),
# half-CTA consumers + stage geometry (splitkv-v8 P10)
SKV_PASS = ("SKV_FORCE_ENGINE", "SKV_ITEM_PREFETCH", "SKV_MERGE_CHUNK", "SKV_KV_EVICT_FIRST", "SKV_TAIL_LDG", "SKV_COMBINE_HALF", "SKV_STAGES", "SKV_HALVES", "SKV_STAGE_ROWS", "SKV_GROUPS")


def skv_stage_units(knobs: dict) -> int:
    """KV ring size of a q_len 1 build in 128-row stage units (skv_off_bars ``skv_stages``): SKV_STAGES x SKV_STAGE_ROWS / 128 (h64: 4 x 64 rows = 2)."""
    return int(knobs.get("SKV_STAGES", 2)) * int(knobs.get("SKV_STAGE_ROWS", 128)) // 128


def skv_bar_bytes(knobs: dict) -> int:
    """skv::kBarBytes of a build: max(64, 16 x SKV_STAGES) (splitkv-v9: 128 at the g4a 8-stage ring); the header defaults (2 / 4 stages) give 64."""
    return max(SKV_BAR_BYTES, 16 * int(knobs.get("SKV_STAGES", 2)))


def skv_knob_defines(knobs: dict) -> list[str]:
    """The SKV_* knobs of a build as NAME=value defines (skv_geometry / skv_map_specs ``extra``)."""
    return [f"{n}={int(v)}" for n, v in sorted(knobs.items()) if n.startswith("SKV_")]


def knob_defines(knobs: dict | None) -> list[str]:
    k = {**DEFAULT_KNOBS, **(knobs or {})}
    unknown = set(k) - set(DEFAULT_KNOBS) - set(SKV_PASS)
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


def skv_off_bars(npad: int, ku: int = 128, ring: int = 2, attn: str = "placeholder", lmp: int = 1, lmr: int = 0, ts: int = 0, tsb: int = 0, skv_stages: int = 2, gu_tt: int = 0) -> int:
    """OFF_BARS of the kernel layout (mk_common.cuh): end of the overlay regions; under attn = splitkv at least kBarOff and = 512 (mod 1024).
    ``skv_stages``: SKV_STAGES of a q_len 1 build (128-row stages, 64 KB each: kBarOff = SKV_BAR_OFF + (stages - 2) * 64 KB)."""
    nbox = ku // 64
    ub, xs = nbox * 16384, nbox * npad * 128
    lay = ring * ub + 2 * xs
    if ts and npad >= ts_min(ts) and ku == 128:
        tt = ts_tt(npad, gu_tt)
        if tsb:
            lay = max(lay, max(ts_ring(1, tt[p], 1) * (32768 + tt[p] * 256) for p in ("qkv", "gu", "d")) + TS_EXCH)
        else:
            lay = max(lay, ts_ring(1) * (32768 + TS_TT * 256))
    elif lmp and npad <= 64 and ku == 128:
        lay = max(lay, 8 * xs + lm_ring(npad, lmr) * ub)
    if attn == "splitkv":
        lay = max(lay, SKV_BAR_OFF + (skv_stages - 2) * SKV_STAGE_BYTES)
        return (lay - 512 + 1023) // 1024 * 1024 + 512
    return lay


def dyn_smem(npad: int, ku: int = 128, ring: int = 2, attn_bytes: int = ATTN_SMEM_PLACEHOLDER, lmp: int = 1, lmr: int = 0, ts: int = 0, tsb: int = 0, attn: str = "placeholder", skv_stages: int = 2, gu_tt: int = 0, skv_bar: int = SKV_BAR_BYTES) -> int:
    """Dynamic smem of a bucket (mirrors mk_common.cuh DYN_SMEM; the kernel traps if the launch gives less).
    lmp: MK_LMP (the pipelined LM head only applies at npad <= 64: smem = [8 X chunks][lmr x 32 KB ring], 384 B of barriers).
    attn = "splitkv" (SPEC-P2): OFF_BARS = skv_off_bars(..) (>= skv::kBarOff, = 512 mod 1024), A1's 64 B mbarrier tail at [OFF_BARS, +64), mk's barrier block (384 / 256 B) behind it."""
    if attn == "splitkv":
        if ku != 128:
            raise ValueError("MK_ATTN=splitkv needs MK_KU 128")
        tail = 384 if ((lmp and npad <= 64) or (ts and npad >= ts_min(ts))) else 256
        return 1024 + skv_off_bars(npad, ku, ring, attn, lmp, lmr, ts, tsb, skv_stages, gu_tt) + skv_bar + tail
    nbox = ku // 64
    ub, xs = nbox * 16384, nbox * npad * 128
    off_bars = ring * ub + max(2 * xs, attn_bytes)
    if ts and npad >= ts_min(ts) and ku == 128:   # MK_TS: tile-stream ring overlays the other regions; 384 B of barriers
        tt = ts_tt(npad, gu_tt)
        if tsb:   # MK_TSB: ring bytes = the largest over the phases (QKV / D at TT 128, GU at TT 128 or 64), then the GU exchange region
            ring_b = max(ts_ring(1, tt[p], 1) * (32768 + tt[p] * 256) for p in ("qkv", "gu", "d"))
            return 1024 + max(off_bars, ring_b + TS_EXCH) + 384
        return 1024 + max(off_bars, ts_ring(1) * (32768 + TS_TT * 256)) + 384
    if lmp and npad <= 64 and ku == 128:
        return 1024 + max(off_bars, 8 * xs + lm_ring(npad, lmr) * ub) + 384
    return 1024 + off_bars + 256


@lru_cache(maxsize=None)
def compile_build(npad: int, nb: int = GRID, knobs: tuple = (), arch: str = "sm_100a", attn: str = "placeholder",
                  attn_path: str | None = None, stripped: bool = False, skv_qmax: int = 1) -> tuple[bytes, dict, str]:
    """-> (cubin, ptxas info, log). ``knobs``: sorted tuple of (name, value). ``skv_qmax``: attn = splitkv only, 1 = default build, 2..16 = the R6 build (-DSKV_QMAX, SPEC-P2b)."""
    if npad not in BUCKETS:
        raise ValueError(f"NPAD {npad} not in {BUCKETS}")
    src = source(attn, Path(attn_path) if attn_path else None)
    if stripped:
        src = strip_knobs(src)
    k = {**DEFAULT_KNOBS, **dict(knobs)}
    exp = dyn_smem(npad, k["MK_KU"], k["MK_RING"], lmp=k["MK_LMP"], lmr=k["MK_LMR"], ts=k["MK_TS"], tsb=k["MK_TSB"], attn=attn, skv_stages=skv_stage_units(k), gu_tt=k["MK_GU_TT"], skv_bar=skv_bar_bytes(k)) if attn_path is None and (attn == "placeholder" or k["MK_KU"] == 128) else 0
    defs = [f"NPAD={npad}", f"NB={nb}", f"MK_EXPECT_DYN={exp}", *knob_defines(dict(knobs))]
    defs += os.environ.get("MK_EXTRA_DEFINES", "").split()   # debug only (e.g. SKV_DEBUG_WAIT=1); not part of the knob set
    if attn == "splitkv":
        defs.append("MK_SKV=1")
        defs += skv_defines(skv_qmax, npad)
    elif skv_qmax != 1:
        raise ValueError("skv_qmax needs attn = splitkv")
    cubin, log = nvrtc_compile(src, defs, arch=arch, ptxas_verbose=True, name="mk.cu")
    return cubin, parse_ptxas(log), log


# ---- A1 (attn = splitkv) KV tensor maps: 2 * kNumMaps 3D bf16 SW128 maps (K then V), box {64, 1, rows}, rows = stage rows, /2, .., 1, 128 B each ----
SKV_STAGE_ROWS = 128
SKV_NUM_MAPS = SKV_STAGE_ROWS.bit_length()   # log2(128) + 1 = 8 (skv::kNumMaps) of the default build
SKV_QMAX_R6 = 16                             # the R6 build (q_len 2..16): -DSKV_QMAX=16


def skv_defines(qmax: int = 1, npad: int = 16) -> list[str]:
    """Extra NVRTC defines of a splitkv build: -DSKV_QMAX=16 for the R6 build (q_len > 1). splitkv-v3 builds are TMA-only by default
    (SKV_FORCE_ENGINE 1 in the header, job 9708): the arm encodes the KV maps at every plan. The cache name / build key is (npad, knobs, attn, qmax)."""
    if qmax not in (1, SKV_QMAX_R6):
        raise ValueError(f"skv_qmax {qmax}: 1 (default build) or {SKV_QMAX_R6} (R6 build)")
    return [f"SKV_QMAX={qmax}"] if qmax > 1 else []


def skv_qmax_of(seqs) -> int:
    """Build of a splitkv workload: SKV_QMAX 16 iff some sequence has q_len > 1 (R6), else 1 (the default build: QMAX 16 is slower at q_len 1, b8 kv1K 18.6 vs 11.5 us). ``seqs``: (kv_len, q_len)."""
    return SKV_QMAX_R6 if any(int(q) > 1 for _, q in seqs) else 1


def skv_geometry(defines=()) -> tuple[int, int]:
    """(stage_rows, stages) of a splitkv build = the header defaults overridden by its defines (a1bench.py geometry): the first #define of SKV_STAGE_ROWS / SKV_STAGES
    in splitkv.cuh is the SKV_QMAX > 1 default, the last the q_len 1 default. kNumMaps = log2(stage_rows) + 1 maps per K / V."""
    d = {k: (v or "1") for k, _, v in (x.partition("=") for x in [*defines, *os.environ.get("MK_EXTRA_DEFINES", "").split()])}
    qmax = int(d.get("SKV_QMAX", 1))
    txt = ATTN_SPLITKV.read_text()

    def dflt(name):
        found = re.findall(rf"#define {name}\s+(\d+)", txt)
        return int(found[0] if qmax > 1 else found[-1])
    return int(d.get("SKV_STAGE_ROWS", dflt("SKV_STAGE_ROWS"))), int(d.get("SKV_STAGES", dflt("SKV_STAGES")))


def skv_num_maps(qmax: int = 1) -> int:
    """skv::kNumMaps of the build: 8 (default, 128 rows), 7 (R6, 64 rows)."""
    return skv_geometry(skv_defines(qmax))[0].bit_length()


def skv_map_specs(rows: int, n_layers: int = NL, qmax: int = 1, extra=()) -> list[tuple[list[int], list[int], list[int]]]:
    """(dims, strides_bytes, box) of the 2 * kNumMaps A1 maps in kernel order (K: box rows stage_rows .. 1, then V): dims {128, 8, n_layers * rows}, strides {256, 2048} B,
    box {64, 1, r}, CU_TENSOR_MAP_SWIZZLE_128B (SKV_TMA_MMA 1, the header default; a1bench.py tensor_maps with mma = True; the kernel issues each box at dim 0 and 64).
    ``rows`` = cache rows per layer (mk: the fixture's packed rows; the CTX padding rows lie behind the last layer). ``qmax``: the build (skv_defines); ``extra``: its SKV_* knob defines (skv_knob_defines; SKV_STAGE_ROWS)."""
    stage_rows = skv_geometry([*skv_defines(qmax), *extra])[0]
    out = []
    for _kv in range(2):
        for rs in range(stage_rows.bit_length()):
            out.append(([HD, KVH, n_layers * rows], [HD * 2, KVH * HD * 2], [64, 1, stage_rows >> rs]))
    return out


def encode_skv_maps(kc_ptr: int, vc_ptr: int, rows: int, n_layers: int = NL, qmax: int = 1, extra=()) -> bytes:
    """2 * kNumMaps x 128 B tensor maps over the K / V caches (device pointers), SWIZZLE_128B, L2 promotion none, as a1bench.py tensor_maps (mma); needs cuInit (a GPU host)."""
    from cuda.bindings import driver as d
    blob = b""
    specs = skv_map_specs(rows, n_layers, qmax, extra)
    for i, (dims, strides, box) in enumerate(specs):
        res = d.cuTensorMapEncodeTiled(
            d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, int(kc_ptr if i < len(specs) // 2 else vc_ptr),
            [d.cuuint64_t(v) for v in dims], [d.cuuint64_t(v) for v in strides], [d.cuuint32_t(v) for v in box],
            [d.cuuint32_t(1)] * 3, d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
            d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE, d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
        if int(res[0]) != 0:
            raise RuntimeError(f"cuTensorMapEncodeTiled(skv map {i}): {res[0]}")
        blob += struct.pack("<16Q", *[int(v) for v in res[1].opaque])
    return blob


def auto_attn(seqs) -> str:
    """attn = "auto": splitkv (A1 v2) when every sequence has q_len == 1 and (the longest kv_len >= SKV_AUTO_MIN_KV or n_seq >= SKV_AUTO_MIN_SEQ), else the placeholder.
    Job 9702 (splitkv-v2 vs placeholder, 28 points): kv >= 512 all win (R2-kv512 -67 .. kv32768 -8051, R3-b8-kv1024 -20); kv128: b16 -1.3, b32 -125, b64 -133, b128 -272,
    b256 -897, but b1 / b2 +23..+41 and b4 / b8 +81..+83 (the combine at S > 1). q_len 2..16 (R6, SKV_QMAX 16 build; job 9703, kv1024): q2 -172, q4 -161, q8 -142,
    q16 -405 -> splitkv whenever some q_len > 1 (max 16). ``seqs``: iterable of (kv_len, q_len)."""
    seqs = [(int(k), int(q)) for k, q in seqs]
    if not seqs or any(q > SKV_QMAX_R6 for _, q in seqs):
        return "placeholder"
    if any(q > 1 for _, q in seqs):
        return "splitkv"
    return "splitkv" if max(k for k, _ in seqs) >= SKV_AUTO_MIN_KV or len(seqs) >= SKV_AUTO_MIN_SEQ else "placeholder"


SKV_AUTO_MIN_KV = 512   # job 9702: kv512 -67 us (b1)
SKV_AUTO_MIN_SEQ = 16   # job 9702: kv128 b16 -1.3, b8 +81 us


# splitkv-v8 half-CTA consumers with 64-row stages (MK-attn P10). Job 9771 (vs the tune without it, 1 GPU x 7 rounds): b256 -417.6, b128 -179.8, b64 -75.6,
# R3-b32-kv1K -84.0, b16-kv128 -38.7. Not at 1-item-per-CTA plans (MK-attn job 9770: b1 kv32K +6.5 us/layer) -> NPAD 16 only with >= 16 sequences.
SKV_H64 = (("SKV_HALVES", 1), ("SKV_STAGES", 4), ("SKV_STAGE_ROWS", 64))
# splitkv-v9 quarter-CTA consumers, 32-row stages x 8 (MK-attn P11). Job 9775 (vs h64): b256 -49.0, b128 -91.4, b64 -34.7. It loses at <= 2 items per CTA
# (MK-attn job 9774: R3-b32-kv1K +10.9, b16-kv128 +1.6 us/layer) -> NPAD >= 64 only.
SKV_G4A = (("SKV_GROUPS", 4), ("SKV_HALVES", 1), ("SKV_STAGES", 8), ("SKV_STAGE_ROWS", 32))
# splitkv-v10 8 merge-free groups of 2 warps, 16-row stages x 16 (MK-attn P12). Job 9782 (vs g4a, MK_FIN 2): b256 -63.0, b128 +13.7 -> NPAD 256 only
SKV_G8 = (("SKV_GROUPS", 8), ("SKV_HALVES", 1), ("SKV_STAGES", 16), ("SKV_STAGE_ROWS", 16))


def skv_tune(npad: int, qmax: int, seqs) -> tuple:
    """Per-bucket splitkv header knobs of a workload (MkArm skvtune=1), as sorted (name, value) pairs; knobs set on the arm win.
    Job 9714 (v4, us/step vs the header default, 1 GPU x 7 rounds): at NPAD <= 32, q_len 1:
    - one sequence with kv <= SKV_LDG_MAX_KV: LDG-only (SKV_FORCE_ENGINE 0): b1 kv1K -21.8, 2K -30.0, 4K -9.1, 8K -4.5, 16K -16.6 (kv32K +4.5);
    - else SKV_MERGE_CHUNK 20 (one merge round at S = 19..20; 0 spill at NPAD 16 / 32 TMA-only only): b1 kv32K -1.9, R3-b8-kv4K -15.2, R3-b32-kv1K -6.7.
    Job 9725 (v5, vs the above): TMA builds add SKV_TAIL_LDG 1 (1-row item tail by LDG) + SKV_KV_EVICT_FIRST 1 (KV loads L2 evict_first: the GEMM
    phases after attend recover most of their run-time loss): b64 -61.7, b128 -216.3, b256 -424.0, R3-b32-kv1K -32.7 (on chunk 20); EF on the
    LDG-only b1 build: kv1K +1.7, kv16K +14.7 (not used there).
    Job 9748 (vs chunk 20 + TAIL + EF at NPAD 16): TAIL never helps at NPAD 16 (b16-kv128 +10.5, R3-b8-kv1K +20.9, b1 kv32K +11.2, R3-b8-kv4K +17.4);
    EF helps at long kv (b1 kv32K -51, R3-b8-kv4K -35) and hurts at short kv (b16-kv128 +4.2, R3-b8-kv1K +7.6); NPAD 32 keeps both (b32-kv128 +31.8 / +42.1
    without); R6 build: EF q4 +10.8, q16 +4.7 -> EF 0 (splitkv-v6: SKV_KV_EVICT_FIRST default 1 = TMA KV loads, so the off buckets set 0 explicitly).
    ``seqs``: iterable of (kv_len, q_len)."""
    seqs = [(int(k), int(q)) for k, q in seqs]
    if not seqs:
        return ()
    if qmax != 1:
        return (("SKV_KV_EVICT_FIRST", 0),)
    if npad > 32:
        return tuple(sorted((("SKV_KV_EVICT_FIRST", 1), *(SKV_G8 if npad == 256 else SKV_G4A), ("SKV_TAIL_LDG", 1))))
    if npad > 16:
        return tuple(sorted((("SKV_KV_EVICT_FIRST", 1), *SKV_H64, ("SKV_MERGE_CHUNK", 20), ("SKV_TAIL_LDG", 1))))
    kv = max(k for k, _ in seqs)
    if len(seqs) == 1 and kv <= SKV_LDG_MAX_KV:
        return (("SKV_FORCE_ENGINE", 0),)
    return tuple(sorted((("SKV_KV_EVICT_FIRST", int(kv >= SKV_EF_MIN_KV)), ("SKV_MERGE_CHUNK", 20), *(SKV_H64 if len(seqs) >= 16 else ()))))


SKV_EF_MIN_KV = 4096   # job 9748, NPAD 16 TMA builds: EF wins at kv 4K / 32K, loses at kv128 / 1K
SKV_LDG_MAX_KV = 16384   # job 9714: LDG-only wins at b1 kv1K..16K, loses at 32K


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


def workspace_map_specs(npad: int = 256, names: tuple = WORKSPACE_MAPS, gu_tt: int = 0) -> list[tuple[list[int], list[int], list[int]]]:
    """(dims, strides_bytes, box) of the workspace maps ``names`` (default map 8 ``xb``: bf16 [MAXTOK_MK][H], dims {H, MAXTOK_MK}, box {64 k, TS_TT tokens}, SW128).
    MK_TSB: ``xb2`` {H, MAXTOK_MK} box {64, TT_GU}; ``wgu`` {H, NL * 6144} box {64, 128}; ``act`` {INTER, MAXTOK_MK} box {64, TT_D}. TT per bucket: ts_tt(npad)."""
    tt = ts_tt(npad, gu_tt)
    spec = {"xb": ([H, MAXTOK_MK], [H * 2], [64, tt["qkv"]]),
            "xb2": ([H, MAXTOK_MK], [H * 2], [64, tt["gu"]]),
            "wgu": ([H, NL * 2 * INTER], [H * 2], [64, 128]),
            "act": ([INTER, MAXTOK_MK], [INTER * 2], [64, tt["d"]])}
    return [spec[n] for n in names]


def encode_maps(weights: dict, ptrs: dict, workspace: dict | None = None, npad: int = 256, gu_tt: int = 0) -> bytes:
    """128 B x 8 tensor maps with the real device addresses (+ 128 B per workspace map of ``workspace`` = {"xb": device ptr, and with MK_TSB "xb2", "wgu", "act"},
    appended in WS_ORDER = maps 8, 9, 10, 11). Needs cuInit (a GPU host)."""
    from cuda.bindings import driver as d
    blob = b""
    names = list(WEIGHT_MAPS)
    specs = list(map_specs(weights))
    if workspace is not None:
        wn = tuple(n for n in WS_ORDER if n in workspace)
        names += list(wn)
        specs += workspace_map_specs(npad, wn, gu_tt)
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
