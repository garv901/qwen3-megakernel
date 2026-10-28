"""P2 (MK-attn PASS-02): standalone bench + checker of A1 v1 (regimebench/kernels/common/splitkv/splitkv.cuh).

NVRTC-builds splitkv.cuh + a1bench.cu (sm_100a), cooperative launch 148 x 512, 28 layers, per layer: filler (cold pool) | barrier | attend | barrier | combine.
  --check     one launch (mode head|ao|dump), layers {0, last}: ao vs torch fp32 reference (rel-L2 <= 1e-2) + combine_head SW128 bytes vs pack_b_slot (bit-exact)
  --time      5 launches after 1 warmup, L2 flushed before each; A = block-0 stamp(after attend barrier) - stamp(after filler barrier) (layer 0 dropped),
              C = max over CTAs of the end of the combine phase - the A stamp. median, p10, p90 over launches x layers.
  --dry-run   print the plan per point (no GPU, no NVRTC);  --compile-only  NVRTC + ptxas report only (no GPU)
  --comb head|ao|none   combine phase of the timed kernel (head: O-proj emulation, 128 units over 148 CTAs; ao: combine_all_to_ao + barrier)
  --engine auto|ldg|tma|both   auto = planner (LDG iff max CTA share <= plan's ldg_max_bytes, default 160 KB)
  --build NAME=DEF1,DEF2,..   (repeatable) extra NVRTC build (header macros, e.g. v64=SKV_STAGE_ROWS=64,SKV_STAGES=3); every point / engine runs on each build.
              The default build (header defaults + -D) runs too unless --no-default. A build with SKV_PROF_NOCOMP=1 is timing-only (no --check).
              A build whose attend smem exceeds 160 KB is skipped. ptxas -v of every build is printed (regs / spills / stack must be 0 spills, 0 stack).
  --dyn-smem BYTES  dynamic smem of every launch (default 165248 = MK-core's NPAD 16/32 allocation; asserted >= attend_smem_bytes()), so the L1 carve-out matches mk
  --build mma=SKV_TMA_MMA=1  builds the mma.sync consumer: the host then encodes SWIZZLE_128B maps (box 64 x 1 x rows) for that build; the dynamic smem base is 1024 B
              aligned (a1bench.cu traps otherwise). The sizes are unchanged (attend_smem_bytes() 148,032 at 128 x 2).
  --iface 1|2       1 (default): bf16 post-RoPE q. 2: builds with -DSKV_IFACE=2; attend does the q/k RMS-norm, RoPE and the new-row KV write from fp32 qkv sums
              (random qkv * 2, rr1 ~ U[0.5, 1.5], qn / kn randn * 0.1 + 1, hv2g rope_table); the new cache rows start as NaN; --check compares k / v rows to a
              torch fp32 reference (<= 1 bf16 ulp, count of non-equal elements reported) and the attention output to the v1 reference on the reference rows.
  --plan-opt k=v    (repeatable) keyword passed to plan(), e.g. one_item=0, ldg_max_bytes=131072, no_split=1 (bool; "auto" is the default)
  --ticket          -DSKV_TICKET=1 for every build (also detected from --build ...,SKV_TICKET=1): attend merges S > 1 pairs itself (deferred last-arriver ticket), so there is no combine phase
              (mode forced to 0, C = 0). Tickets are allocated and zeroed once per point / plan and keep counting across timing launches; --check also verifies tickets[pair] == 28 * S.
  R6 (P5): points b1kv1024q2 / q4 / q8 / q16, b1kv4096q16, b8kv1024q4 (q_len > 1, causal) need a build with SKV_QMAX >= q_len, e.g. --build r6=SKV_QMAX=16 (implies 64 x 4 stages and the TMA
              engine; the plan's engine is forced to TMA for such builds). A point whose q_len exceeds a build's SKV_QMAX is skipped for that build; the torch reference is causal per token.
Output: one JSON line per (point, mode) to results/2026-10-23-mk-attn/p2_<job>.jsonl (rows carry build, defines, dyn_smem, plan_opts), and a table.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import numpy as np
import os
import statistics
import struct
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from regimebench.kernels.common.splitkv import header_source                      # noqa: E402
from regimebench.kernels.common.splitkv import plan as skv_plan                    # noqa: E402

NL = 28
FILL_MB = 31.5
POINTS = {  # name -> (batch, kv_len[, q_len]); q_len > 1 points (R6) need a build with SKV_QMAX >= q_len (e.g. --build r6=SKV_QMAX=16)
    "b1kv128": (1, 128), "b1kv512": (1, 512), "b1kv1024": (1, 1024), "b1kv2048": (1, 2048), "b1kv4096": (1, 4096), "b1kv8192": (1, 8192),
    "b1kv16384": (1, 16384), "b1kv32768": (1, 32768), "b8kv4096": (8, 4096), "b8kv1024": (8, 1024), "b32kv1024": (32, 1024),
    "b64kv128": (64, 128), "b256kv128": (256, 128), "b128kv128": (128, 128), "b4kv128": (4, 128), "b8kv128": (8, 128), "b16kv128": (16, 128),
    "b1kv1024q2": (1, 1024, 2), "b1kv1024q4": (1, 1024, 4), "b1kv1024q8": (1, 1024, 8), "b1kv1024q16": (1, 1024, 16), "b1kv4096q16": (1, 4096, 16),
    "b8kv1024q4": (8, 1024, 4)}
R6_BUILD = "r6=SKV_QMAX=16"   # the job's R6 build (SKV_QMAX > 1 implies 64 x 4 stages, TMA engine, SKV_TMA_MMA 1)
NPADS = (16, 32, 64, 128, 256)
DYN_SMEM = 165248      # default --dyn-smem: MK-core's NPAD 16/32 allocation
MAX_ATTEND_SMEM = int(os.environ.get("A1UB_MAX_SMEM", 160 * 1024))   # attend scratch budget in every mk bucket


def header_default(name: str, qmax: int = 1) -> int:
    """Default of a header macro. SKV_STAGE_ROWS / SKV_STAGES have two defaults: the first #define is the SKV_QMAX > 1 one, the second the q_len 1 one."""
    import re
    found = re.findall(rf"#define {name}\s+(-?\d+)", header_source())
    return int(found[0] if qmax > 1 else found[-1])


def parse_defines(defines) -> dict:
    out = {}
    for d in defines:
        k, _, v = d.partition("=")
        out[k] = v if v else "1"
    return out


def geometry(defines) -> tuple:
    """(stage_rows, stages) of a build (header defaults overridden by its defines)."""
    d = parse_defines(defines)
    qm = int(d.get("SKV_QMAX", 1))
    return int(d.get("SKV_STAGE_ROWS", header_default("SKV_STAGE_ROWS", qm))), int(d.get("SKV_STAGES", header_default("SKV_STAGES", qm)))


def attend_smem_bytes(defines) -> int:
    """Mirror of skv::attend_smem_bytes() (checked at NVRTC time by the static_assert in a1bench.cu)."""
    rows, stages = geometry(defines)
    d = parse_defines(defines)
    if d.get("SKV_TC", "0") != "0":        # SKV_TC: K ring + V ring + Q / P^T buffers + exchange + 256 B of mbarriers
        return (int(d.get("SKV_TC_KST", 2)) + int(d.get("SKV_TC_VST", 2))) * 32768 + 16384 + 512 + 256
    return stages * rows * 512 + 16 * 2 * 132 * 4 + max(64, 16 * stages)


def hnd_strides(cache_rows: int):
    """(head, row) byte strides of the 3D KV maps. Default = mk's layout [L, rows, 8 heads, 128] (256, 2048): an item reads 256 B per position at 2 KB stride.
    A1UB_HND=1 (a1ub, timing only: the data is a scrambled view of the same buffer): [8 heads, L * rows, 128] (L * rows * 256, 256), an item's positions
    are contiguous 256 B rows, as in an HND paged cache (fi: page 16 x 256 B = 4 KB contiguous per head)."""
    return (NL * cache_rows * 256, 256) if os.environ.get("A1UB_HND") == "1" else (256, 2048)


def fused_map_params(stage_rows: int, cache_rows: int):
    """P13 (SKV_TMA_FUSE) 4D bf16 SWIZZLE_128B map of K or V: (dims, strides of dims 1..3 in bytes, box, element strides).
    dims (64 dims in half, NL * cache_rows rows, 2 halves, 8 kv heads); box (64, stage_rows, 2, 1); coordinates {0, row, 0, kvh}."""
    return (64, NL * cache_rows, 2, 8), (2048, 128, 256), (64, stage_rows, 2, 1), (1, 1, 1, 1)


def npad_of(total_q: int) -> int:
    for n in NPADS:
        if total_q <= n:
            return n
    raise ValueError(f"total_q {total_q} > 256")


def source() -> str:
    return header_source() + "\n// ---- a1bench.cu ----\n" + (Path(__file__).with_name("a1ub.cu")).read_text()


def compile_cubin(defines=(), ldg_db: int | None = None):
    """NVRTC (not coop.nvrtc_compile: that helper has no hook for -lineinfo) -> (cubin, log)."""
    from cuda.bindings import nvrtc
    src = source()
    defs = list(defines) + ([f"SKV_LDG_DB={ldg_db}"] if ldg_db is not None else [])
    expect = f"A1_EXPECT_SMEM={attend_smem_bytes(defs)}"
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"a1bench.cu", 0, [], [])
    opts = [b"--gpu-architecture=sm_100a", b"-std=c++17", b"--fmad=true", b"-default-device", b"-lineinfo", b"--ptxas-options=-v"]
    opts += [f"-D{d}".encode() for d in defs + [expect]]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    _, n = nvrtc.nvrtcGetProgramLogSize(prog)
    log = b" " * n
    nvrtc.nvrtcGetProgramLog(prog, log)
    text = log.decode(errors="replace").strip("\x00 \n")
    if int(err) != 0:
        raise RuntimeError("NVRTC compile failed:\n" + text)
    _, n = nvrtc.nvrtcGetCUBINSize(prog)
    cubin = b" " * n
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    nvrtc.nvrtcDestroyProgram(prog)
    if "Compiling entry function" not in text and os.environ.get("CUDA_CACHE_DISABLE") != "1":
        # NVRTC used the driver compute cache (read at library load), so ptxas did not run and the -v report is empty: get the log from a
        # fresh process with the cache disabled
        import subprocess
        code = ("import sys; sys.path.insert(0, %r); import a1ub; sys.stdout.write(a1ub.compile_cubin(%r)[1])"
                % (str(Path(__file__).parent), defs))
        text = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                              env={**os.environ, "CUDA_CACHE_DISABLE": "1"}).stdout
    return cubin, text


def ptxas_report(log: str) -> dict:
    """{kernel: dict(regs, stack, spill_stores, spill_loads, smem)} from a --ptxas-options=-v log."""
    import re
    out, cur = {}, None
    for line in log.splitlines():
        m = re.search(r"Compiling entry function '(\w+)'", line)
        if m:
            cur = out.setdefault(m.group(1), {})
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m and cur is not None:
            cur.update(stack=int(m.group(1)), spill_stores=int(m.group(2)), spill_loads=int(m.group(3)))
        m = re.search(r"Used (\d+) registers", line)
        if m and cur is not None:
            cur["regs"] = int(m.group(1))
            m2 = re.search(r"(\d+) bytes smem", line)
            cur["smem"] = int(m2.group(1)) if m2 else 0
    return out


def seqs_of(point: str):
    b, kv, *q = POINTS[point]
    return [(kv, q[0] if q else 1)] * b


def qlen_of(point: str) -> int:
    return POINTS[point][2] if len(POINTS[point]) > 2 else 1


def parse_plan_opts(items) -> dict:
    import ast
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v                                                         # e.g. no_split=auto
        if k == "no_split" and out[k] in (0, 1):
            out[k] = bool(out[k])                                              # no_split=1 / 0 -> True / False
    return out


def make_plan(point: str, n_cta: int, engine: str, plan_opts=None, iface: int = 1):
    opts = dict(plan_opts or {})
    active = int(opts.pop("active", n_cta))                                # --plan-opt active=N: plan for N CTAs, CTAs N.. get no items (SM-count sweep)
    pl = skv_plan.plan(seqs_of(point), n_cta=min(active, n_cta), iface=iface, **opts)
    if pl.n_cta < n_cta:
        pad = n_cta - pl.n_cta
        pl.items = np.concatenate([pl.items, np.zeros((pad,) + pl.items.shape[1:], dtype=pl.items.dtype)])
        pl.item_off = np.concatenate([pl.item_off, np.full(pad, pl.item_off[-1], dtype=pl.item_off.dtype)])
        pl.cta_positions = list(pl.cta_positions) + [0] * pad
        pl.n_cta = n_cta
    if engine in ("ldg", "tma") and qlen_of(point) == 1:
        pl.engine = int(engine == "tma")
    return pl


def print_dry_run(n_cta: int, plan_opts=None):
    print(f"{'point':<11} {'engine':<6} {'S max':>5} {'items/CTA max':>13} {'part bytes':>11} {'CTA KB max':>10} {'CTA KB min':>10} {'items':>6} "
          f"{'one_item':>8} {'bal 1-item':>10} {'bal greedy':>10}")
    for pt in POINTS:
        s = make_plan(pt, n_cta, "auto", plan_opts).summary()
        b1 = s["balance_one_item"]
        print(f"{pt:<11} {s['engine']:<6} {s['s_max']:>5} {s['items_per_cta_max']:>13} {s['part_bytes']:>11} {s['cta_bytes_max'] / 1024:>10.1f} "
              f"{s['cta_bytes_min'] / 1024:>10.1f} {s['n_items']:>6} {str(s['one_item']):>8} {'-' if b1 is None else format(b1, '.3f'):>10} "
              f"{'-' if s['balance_greedy'] is None else format(s['balance_greedy'], '.3f'):>10}  no_split={s['no_split']}")


# ------------------------------------------------------------------------------------------------ GPU side
class Build:
    """One NVRTC build: defines, ptxas report, per-TPAD kernels."""

    def __init__(self, name: str, defines, ldg_db, dyn_smem: int):
        self.name = name
        self.defines = list(defines) + ([f"SKV_LDG_DB={ldg_db}"] if ldg_db is not None else [])
        self.stage_rows, self.stages = geometry(self.defines)
        self.smem = attend_smem_bytes(self.defines)
        self.timing_only = parse_defines(self.defines).get("SKV_PROF_NOCOMP", "0") != "0" or parse_defines(self.defines).get("SKV_TC_NOMATH", "0") != "0"
        self.qmax = int(parse_defines(self.defines).get("SKV_QMAX", 1))   # > 1: the R6 consumer (TMA engine only; runs q_len 1 points too)
        self.iface = int(parse_defines(self.defines).get("SKV_IFACE", header_default("SKV_IFACE")))
        self.ticket = int(parse_defines(self.defines).get("SKV_TICKET", header_default("SKV_TICKET"))) != 0   # attend leaves ao complete: no combine, tickets input
        self.mma = int(parse_defines(self.defines).get("SKV_TMA_MMA", header_default("SKV_TMA_MMA"))) != 0   # swizzled maps + mma consumer
        self.fuse = int(parse_defines(self.defines).get("SKV_TMA_FUSE", header_default("SKV_TMA_FUSE"))) != 0   # P13: 2 fused 4D maps appended
        self.skip = self.smem > MAX_ATTEND_SMEM
        assert dyn_smem >= self.smem or self.skip, f"--dyn-smem {dyn_smem} < attend_smem_bytes() {self.smem} of build {name}"
        self.dyn_smem = dyn_smem
        self.kernels = {}
        self.cubin, self.log, self.ptxas = None, "", {}
        if not self.skip:
            self.cubin, self.log = compile_cubin(self.defines)
            self.ptxas = ptxas_report(self.log)

    def bad_ptxas(self) -> list:
        return [k for k, v in self.ptxas.items() if v.get("spill_stores") or v.get("spill_loads") or v.get("stack") or v.get("regs", 0) > 128]


class Bench:
    def __init__(self, ldg_db: int | None, defines=(), builds=None, dyn_smem: int = DYN_SMEM, include_default: bool = True):
        """builds: list of (name, [defines]) extra builds; the default build ("default": header defaults + defines) is first unless include_default=False."""
        import torch
        from cuda.bindings import driver
        from regimebench.kernels.hv2g.coop import Args, CoopKernel, check, i32
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        self.torch, self.driver, self.Args, self.i32, self.check = torch, driver, Args, i32, check
        self.nb = torch.cuda.get_device_properties(0).multi_processor_count
        self.dyn_smem = dyn_smem
        self.builds = {}
        specs = ([("default", list(defines))] if include_default else []) + [(n, list(defines) + list(d)) for n, d in (builds or [])]
        for n, d in specs:
            self.builds[n] = Build(n, d, ldg_db, dyn_smem)
        first = next((b for b in self.builds.values() if not b.skip), None)
        self.cubin, self.log, self.ptxas = (first.cubin, first.log, first.ptxas) if first else (None, "", {})
        self.CoopKernel = CoopKernel
        self.pool = torch.ones(2 << 30, dtype=torch.uint8, device="cuda")
        self.flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
        self.cache = None  # (rows, k, v, {stage_rows: tmaps}) of the last point
        self.v2 = {}       # iface 2 inputs (qkv, rr1, qn, kn, rope table), generated once

    def kernel(self, build: Build, npad: int):
        if npad not in build.kernels:
            build.kernels[npad] = self.CoopKernel(build.cubin, f"a1_k{npad}", build.dyn_smem)
        return build.kernels[npad]

    def tensor_maps(self, kc, vc, cache_rows: int, stage_rows: int = 32, mma: bool = False, fuse: bool = False):
        """2 * kNumMaps CUtensorMaps (K then V, box rows stage_rows, /2, .., 1: log2(stage_rows) + 1 each), 128 B each, in a device buffer (64 B aligned).
        mma (SKV_TMA_MMA=1): box (64 dims, 1, rows) with CU_TENSOR_MAP_SWIZZLE_128B; the kernel issues each box twice (dim coordinate 0 and 64).
        fuse (SKV_TMA_FUSE=1, P13): 2 more maps (K then V) after the others, see fused_map_params."""
        d, torch = self.driver, self.torch
        nmaps = stage_rows.bit_length()      # log2(rows) + 1 for a power of two
        blob = b""
        for t in (kc, vc):
            for rs in range(nmaps):
                rows = stage_rows >> rs
                res = d.cuTensorMapEncodeTiled(
                    d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, int(t.data_ptr()),
                    [d.cuuint64_t(v) for v in (128, 8, NL * cache_rows)], [d.cuuint64_t(v) for v in hnd_strides(cache_rows)],
                    [d.cuuint32_t(v) for v in (64 if mma else 128, 1, rows)], [d.cuuint32_t(v) for v in (1, 1, 1)],
                    d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
                    d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B if mma else d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
                    d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE, d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
                if int(res[0]) != 0:
                    raise RuntimeError(f"cuTensorMapEncodeTiled: {res[0]}")
                blob += struct.pack("<16Q", *[int(v) for v in res[1].opaque])
        if fuse:
            dims, strides, box, estr = fused_map_params(stage_rows, cache_rows)
            for t in (kc, vc):
                res = d.cuTensorMapEncodeTiled(
                    d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, int(t.data_ptr()),
                    [d.cuuint64_t(v) for v in dims], [d.cuuint64_t(v) for v in strides],
                    [d.cuuint32_t(v) for v in box], [d.cuuint32_t(v) for v in estr],
                    d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
                    d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE, d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
                if int(res[0]) != 0:
                    raise RuntimeError(f"cuTensorMapEncodeTiled (fused): {res[0]}")
                blob += struct.pack("<16Q", *[int(v) for v in res[1].opaque])
        buf = torch.frombuffer(bytearray(blob), dtype=torch.uint8).cuda()
        assert buf.data_ptr() % 64 == 0
        return buf

    def data(self, point: str, pl, stage_rows: int = 32, mma: bool = False, fuse: bool = False):
        """KV [28, rows, 8, 128] bf16 (randn * 0.1, seeded), q [total_q, 16, 128]; cached while the row count is unchanged."""
        torch = self.torch
        rows = pl.cache_rows
        if self.cache is None or self.cache[0] != rows:
            self.cache = None
            torch.cuda.empty_cache()
            g = torch.Generator(device="cuda")
            g.manual_seed(1234)
            kc = torch.empty(NL, rows, 8, 128, dtype=torch.bfloat16, device="cuda")
            vc = torch.empty_like(kc)
            for l in range(NL):
                kc[l].normal_(0, 0.1, generator=g)
                vc[l].normal_(0, 0.1, generator=g)
            self.cache = (rows, kc, vc, {})
        g = torch.Generator(device="cuda")
        g.manual_seed(99)
        q = torch.empty(pl.total_q, 16, 128, dtype=torch.bfloat16, device="cuda")
        q.normal_(0, 16.0, generator=g)   # sharp-ish logits (sd ~1.6) so the online-softmax rescale is exercised
        key = (stage_rows, mma, fuse)
        if key not in self.cache[3]:
            self.cache[3][key] = self.tensor_maps(self.cache[1], self.cache[2], rows, stage_rows, mma, fuse)
        return q, self.cache[1], self.cache[2], self.cache[3][key]

    def v2_inputs(self, max_pos: int):
        """qkv fp32 [256, 4096] (randn * 2), rr1 fp32 [256] (U[0.5, 1.5]), qn / kn bf16 [28, 128] (randn * 0.1 + 1), rope fp32 [>= max_pos, 128] (hv2g rope_table)."""
        torch = self.torch
        if not self.v2 or self.v2["rope"].shape[0] < max_pos:
            from regimebench.kernels.hv2g.build import rope_table
            g = torch.Generator(device="cuda")
            g.manual_seed(4321)
            qkv = torch.randn(256, 4096, dtype=torch.float32, device="cuda", generator=g) * 2
            rr1 = torch.rand(256, dtype=torch.float32, device="cuda", generator=g) + 0.5
            qn = (torch.randn(28, 128, dtype=torch.float32, device="cuda", generator=g) * 0.1 + 1).to(torch.bfloat16)
            kn = (torch.randn(28, 128, dtype=torch.float32, device="cuda", generator=g) * 0.1 + 1).to(torch.bfloat16)
            self.v2 = {"qkv": qkv, "rr1": rr1, "qn": qn, "kn": kn, "rope": rope_table(max_pos).cuda()}
        return self.v2

    def run(self, point: str, engine: str, mode: int, check: bool, launches: int, build: str = "default", plan_opts=None):
        """mode bits: 1 head combine, 2 ao combine, 4 dump xs. -> dict of results."""
        torch = self.torch
        bld = self.builds[build]
        iface = bld.iface
        if bld.ticket:
            mode = 0                                                           # ao is complete after attend (+ the grid barrier): no combine_head / combine_all_to_ao
        pl = make_plan(point, self.nb, engine, plan_opts, iface)
        if bld.qmax > 1:
            pl.engine = 1                                                  # R6 builds contain only the TMA engine
        q, kc, vc, tmaps = self.data(point, pl, bld.stage_rows, bld.mma, bld.fuse)
        v2 = None
        if iface == 2:
            v2 = self.v2_inputs(max(k + 1 for k, _ in pl.seqs))
            new_rows = torch.tensor([pl.row_base[s] + kv for s, (kv, _) in enumerate(pl.seqs)], dtype=torch.long, device="cuda")
            kc.view(NL, pl.cache_rows, 1024)[:, new_rows] = float("nan")      # unwritten new rows must fail the check (timing: rewritten every layer anyway)
            vc.view(NL, pl.cache_rows, 1024)[:, new_rows] = float("nan")
            q = torch.zeros_like(q)                                            # q is not an input of iface 2
        v2p = [v2[k].data_ptr() for k in ("qkv", "rr1", "qn", "kn", "rope")] if v2 else [0] * 5
        dev = pl.to_torch()
        tq = pl.total_q
        npad = npad_of(tq)
        part = torch.zeros(max(pl.part_floats, 4), dtype=torch.float32, device="cuda")
        ao_stride = tq * 2048 if check else 0
        ao = torch.empty(NL * tq * 2048 if check else tq * 2048, dtype=torch.bfloat16, device="cuda")
        ao.view(torch.int16).fill_(0x7FC0)   # NaN: an unwritten output cannot pass
        dbg = torch.zeros(16 * 2 * npad * 128, dtype=torch.uint8, device="cuda")
        bar = torch.zeros(4, dtype=torch.int32, device="cuda")
        tickets = torch.zeros(max(pl.n_pairs, 1), dtype=torch.int32, device="cuda")   # SKV_TICKET: zeroed once per plan (here), never between launches
        stamps = torch.zeros(NL * 4, dtype=torch.int64, device="cuda")
        cend = torch.zeros(NL * self.nb, dtype=torch.int64, device="cuda")
        sink = torch.zeros(self.nb * 512, dtype=torch.int32, device="cuda")
        fill_vec = int(FILL_MB * 1e6) // self.nb // 16
        fill_adv = (fill_vec * self.nb * 16 + 4095) & ~4095
        pool_mod = self.pool.numel() - fill_adv - 4096
        k = self.kernel(bld, npad)
        Args, i32, C = self.Args, self.i32, ctypes

        def launch(it):
            args = Args([q.data_ptr(), kc.data_ptr(), vc.data_ptr(), i32(pl.cache_rows), i32(NL), dev["items"].data_ptr(), i32(pl.stride), dev["item_off"].data_ptr(),
                         dev["pair_info"].data_ptr(), part.data_ptr(), ao.data_ptr(), tmaps.data_ptr(), i32(pl.engine), i32(tq),
                         self.pool.data_ptr(), i32(fill_vec), C.c_size_t(fill_adv), C.c_size_t(pool_mod), C.c_size_t(it * NL * fill_adv),
                         bar.data_ptr(), stamps.data_ptr(), cend.data_ptr(), sink.data_ptr(), i32(mode), dbg.data_ptr(), C.c_size_t(ao_stride)] + v2p + [tickets.data_ptr()])
            self.flush.fill_(it & 255)
            bar.zero_()
            torch.cuda.synchronize()
            k.launch(self.nb, args, torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()

        res = {"point": point, "engine": pl.engine_name, "s_max": pl.s_max, "items_max": pl.max_items, "part_bytes": pl.part_bytes,
               "n_cta": self.nb, "mode": mode, "build": build, "defines": bld.defines, "dyn_smem": bld.dyn_smem,
               "attend_smem": bld.smem, "plan_opts": plan_opts or {}, "one_item": pl.one_item, "timing_only": bld.timing_only, "iface": iface, "mma": bld.mma,
               "ticket": bld.ticket, "no_split": pl.no_split}
        if check:
            launch(0)
            res.update(self.check_outputs(pl, q, kc, vc, ao, dbg, npad, mode, v2))
            if bld.ticket:                                                     # every layer adds exactly S per pair with S > 1
                exp = torch.zeros(max(pl.n_pairs, 1), dtype=torch.int32)
                for s_, (_, ql) in enumerate(pl.seqs):
                    for kvh in range(8):
                        S_ = int(pl.pair_info[pl.tok0s[s_] * 8 + kvh, 1])
                        exp[s_ * 8 + kvh] = NL * S_ if S_ > 1 else 0
                res["tickets_ok"] = bool((tickets.cpu() == exp).all())
                res["pass"] = bool(res["pass"] and res["tickets_ok"])
            return res
        A, Cc = [], []
        W, Wmax, Wmin, Imb, Lat, per_cta = [], [], [], [], [], []
        for it in range(launches + 1):
            launch(it)
            if it == 0:
                continue
            st = stamps.cpu().view(NL, 4)
            ce = cend.cpu().view(NL, self.nb)
            for l in range(1, NL):
                A.append(float(st[l, 1] - st[l, 0]) / 1e3)
                if mode & 8:   # a1ub split: ce = per-CTA attend end; st[l, 0] = block 0 release of the pre-attend barrier
                    w = (ce[l].double() - float(st[l, 0])) / 1e3
                    W.append(float(w.median())); Wmax.append(float(w.max())); Wmin.append(float(w.min()))
                    Imb.append(float(w.max() - w.median())); Lat.append(float(st[l, 1] - ce[l].max()) / 1e3)
                    per_cta.append(w.tolist())
                else:
                    Cc.append(float(ce[l].max() - st[l, 1]) / 1e3)
        if not Cc:
            Cc = [0.0]
        if mode & 8:
            import numpy as _np
            pc = _np.median(_np.array(per_cta), axis=0)                       # per-CTA work, median over layers x launches
            cb = _np.array(pl.cta_bytes, dtype=_np.float64)
            res.update(W_med=statistics.median(W), W_max=statistics.median(Wmax), W_min=statistics.median(Wmin), imb_med=statistics.median(Imb),
                       lat_med=statistics.median(Lat), cta_work=[round(float(x), 2) for x in pc], cta_bytes=[int(x) for x in cb],
                       cta_gbps=[round(float(b / w / 1e3), 2) if w > 0 else 0.0 for b, w in zip(cb, pc)])

        def pct(v, f):
            v = sorted(v)
            return v[min(len(v) - 1, int(f * (len(v) - 1) + 0.5))]

        res.update(A_med=statistics.median(A), A_p10=pct(A, .1), A_p90=pct(A, .9), C_med=statistics.median(Cc), C_p10=pct(Cc, .1), C_p90=pct(Cc, .9),
                   launches=launches, kv_bytes_layer=pl.cache_rows * 2048 * 2)
        return res

    def v2_reference(self, pl, v2, l):
        """torch fp32 v0 math (SPEC P3-iface2) for layer l -> q bf16 [tq, 16, 128], k_new / v_new bf16 [tq, 8, 128] (token t = sequence t, pos = kv_len)."""
        torch = self.torch
        tq = pl.total_q
        pos = torch.tensor([kv for kv, _ in pl.seqs], dtype=torch.long, device=v2["qkv"].device)
        rope = v2["rope"][pos]                                                 # [tq, 128]
        cos, sin = rope[:, :64].repeat(1, 2)[:, None, :], rope[:, 64:].repeat(1, 2)[:, None, :]
        rr = v2["rr1"][:tq, None, None]
        qkv = v2["qkv"][:tq]

        def nr(x, w):                                                          # x [tq, heads, 128] raw sums
            e = x * rr
            rn = torch.rsqrt((e * e).sum(-1, keepdim=True) / 128 + 1e-6)
            e = e * rn * w.float()
            rot = torch.cat([-e[..., 64:], e[..., :64]], dim=-1)
            return (e * cos + rot * sin).to(torch.bfloat16)

        qr = nr(qkv[:, :2048].reshape(tq, 16, 128), v2["qn"][l])
        kr = nr(qkv[:, 2048:3072].reshape(tq, 8, 128), v2["kn"][l])
        vr = (qkv[:, 3072:4096].reshape(tq, 8, 128) * rr).to(torch.bfloat16)
        return qr, kr, vr

    def ulp_diff(self, a, b):
        """|ordered-key difference| of two bf16 tensors (sign-magnitude -> monotonic int), in bf16 ulps; NaN patterns give large values."""
        torch = self.torch

        def key(x):
            u = x.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
            return torch.where(u >= 0x8000, -(u & 0x7FFF), u & 0x7FFF)
        return (key(a) - key(b)).abs()

    def check_outputs(self, pl, q, kc, vc, ao, dbg, npad, mode, v2=None):
        torch = self.torch
        tq = pl.total_q
        out = {"rel_l2": {}, "max_abs": {}, "row_rel_max": {}}
        ok = True
        aov = ao.view(NL, tq, 2048)
        if v2 is not None:
            out["kv_rows"] = {}
        for l in (0, NL - 1):
            if v2 is not None:
                q, kr, vr = self.v2_reference(pl, v2, l)
                ridx = torch.tensor([pl.row_base[s] + kv for s, (kv, _) in enumerate(pl.seqs)], dtype=torch.long, device="cuda")
                gk, gv = kc[l][ridx], vc[l][ridx]                              # what the kernel wrote
                dk, dv = self.ulp_diff(gk, kr), self.ulp_diff(gv, vr)
                neq = int((dk != 0).sum()) + int((dv != 0).sum())
                over = int((dk > 1).sum()) + int((dv > 1).sum())
                out["kv_rows"][str(l)] = {"elements": int(dk.numel() * 2), "non_equal": neq, "over_1ulp": over, "k_non_equal": int((dk != 0).sum()),
                                          "v_non_equal": int((dv != 0).sum())}
                ok &= over == 0
                kcl, vcl = kc[l:l + 1].clone(), vc[l:l + 1].clone()            # attention reference on the reference rows
                kcl[0][ridx], vcl[0][ridx] = kr, vr
                ref = skv_plan.torch_reference(q, kcl, vcl, 0, pl.seqs, pl.row_base)
            else:
                ref = skv_plan.torch_reference(q, kc, vc, l, pl.seqs, pl.row_base)
            got = aov[l].float()
            rel = float((got - ref).norm() / ref.norm())
            mx = float((got - ref).abs().max())
            out["rel_l2"][str(l)], out["max_abs"][str(l)] = rel, mx
            ok &= (rel <= 1e-2) and rel == rel
            rr = ((got - ref).reshape(-1, 128).norm(dim=1) / ref.reshape(-1, 128).norm(dim=1).clamp_min(1e-6)).max()   # worst (token, q-head) row: one wrong item must not hide in the global rel-L2
            out["row_rel_max"][str(l)] = float(rr)
            ok &= float(rr) <= 5e-2
        if mode & 1 and mode & 4:
            d = dbg.cpu().view(16, 2 * npad * 128).numpy()
            nbad = 0
            for h in range(16):
                exp = skv_plan.pack_b_slot(aov[NL - 1][:, h * 128:(h + 1) * 128].cpu(), npad)
                nbad += int((d[h] != exp).sum())
            out["sw128_bad_bytes"] = nbad
            ok &= nbad == 0
        out["pass"] = bool(ok)
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--comb", default="head", choices=["head", "ao", "none", "split"])
    ap.add_argument("--engine", default="auto", choices=["auto", "ldg", "tma", "both"])
    ap.add_argument("--points", default="all", help="comma list of point names (default all): " + ",".join(POINTS))
    ap.add_argument("--launches", type=int, default=5)
    ap.add_argument("--job", default=os.environ.get("SLURM_JOB_ID", "local"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--ldg-db", type=int, default=None, help="SKV_LDG_DB (0 single, 1 double, 2 half buffer; default = header default)")
    ap.add_argument("-D", action="append", default=[], help="extra NVRTC define for every build, e.g. -DSKV_KV_NC=1")
    ap.add_argument("--build", action="append", default=[], metavar="NAME=DEF1,DEF2", help="extra NVRTC build, e.g. v64=SKV_STAGE_ROWS=64,SKV_STAGES=3 (repeatable)")
    ap.add_argument("--no-default", action="store_true", help="run only the --build builds, not the default build")
    ap.add_argument("--dyn-smem", type=int, default=DYN_SMEM, help="dynamic smem of every launch (default 165248, MK-core NPAD 16/32)")
    ap.add_argument("--iface", type=int, default=1, choices=[1, 2], help="interface version: 2 builds -DSKV_IFACE=2 (qkv / rr1 / rope inputs, KV write inside attend)")
    ap.add_argument("--ticket", action="store_true", help="-DSKV_TICKET=1 for every build (deferred ticket merge inside attend; no combine phase)")
    ap.add_argument("--plan-opt", action="append", default=[], metavar="K=V", help="keyword for plan(), e.g. one_item=0 (repeatable)")
    a = ap.parse_args()
    points = list(POINTS) if a.points == "all" else a.points.split(",")
    for p in points:
        if p not in POINTS:   # any b<B>kv<K>[q<Q>] name is accepted (P9 sweeps)
            mt = __import__("re").fullmatch(r"b(\d+)kv(\d+)(?:q(\d+))?", p)
            if not mt:
                ap.error(f"unknown point {p}")
            POINTS[p] = (int(mt[1]), int(mt[2])) + ((int(mt[3]),) if mt[3] else ())
    builds = []
    for spec in a.build:
        name, _, rest = spec.partition("=")
        if not name or name == "default" or any(n == name for n, _ in builds):
            ap.error(f"bad or duplicate build name in --build {spec}")
        builds.append((name, [d for d in rest.split(",") if d]))
    plan_opts = parse_plan_opts(a.plan_opt)
    if a.iface == 2:
        a.D = list(a.D) + ["SKV_IFACE=2"]
    if a.ticket:
        a.D = list(a.D) + ["SKV_TICKET=1"]
    n_cta = 148
    if a.dry_run:
        print_dry_run(n_cta, plan_opts)
        return 0
    if a.compile_only:
        specs = ([("default", list(a.D))] if not a.no_default else []) + [(n, list(a.D) + d) for n, d in builds]
        for n, d in specs:
            bd = Build(n, d, a.ldg_db, a.dyn_smem)
            print(f"build {n}: defines {bd.defines} attend smem {bd.smem}" + (" SKIPPED (> 160 KB)" if bd.skip else ""))
            for k, v in bd.ptxas.items():
                print("  ", k, v)
            if not bd.skip:
                print("   cubin bytes", len(bd.cubin), "BAD ptxas" if bd.bad_ptxas() else "ptxas ok")
        return 0
    if not (a.check or a.time):
        ap.error("one of --check --time --dry-run --compile-only")
    b = Bench(a.ldg_db, a.D, builds, a.dyn_smem, include_default=not a.no_default)
    for n, bd in b.builds.items():
        if bd.skip:
            print(f"build {n}: SKIPPED, attend smem {bd.smem} B > {MAX_ATTEND_SMEM} B", flush=True)
            continue
        print(f"build {n}: defines {bd.defines} attend smem {bd.smem} dyn smem {bd.dyn_smem}" + ("  BAD PTXAS (spill / stack / regs)" if bd.bad_ptxas() else ""), flush=True)
        for k, v in sorted(bd.ptxas.items()):
            print("ptxas", n, k, v, flush=True)
    out = Path(a.out) if a.out else REPO / "results" / "2026-10-23-mk-attn" / f"p2_{a.job}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, bad = [], 0
    combmode = {"head": 1, "ao": 2, "none": 0, "split": 8}[a.comb]

    def emit(r):
        rows.append(r)
        print(json.dumps(r), flush=True)
        with out.open("a") as f:
            f.write(json.dumps(r) + "\n")

    live = [(bn, bd) for bn, bd in b.builds.items() if not bd.skip]
    for pt in points:                                   # points outermost: the KV cache of a point is generated once and shared by every build
        engines = ["ldg", "tma"] if a.engine == "both" else [a.engine]
        for eng in engines:
            for bn, bd in live:
                if qlen_of(pt) > bd.qmax or (bd.qmax > 1 and bd.iface != 1):
                    print(f"build {bn}: point {pt} skipped (needs SKV_QMAX >= {qlen_of(pt)}; SKV_QMAX > 1 is interface v1 only)", flush=True)
                    continue
                if a.check and bd.timing_only:
                    print(f"build {bn}: timing-only (SKV_PROF_NOCOMP), --check skipped", flush=True)
                elif a.check:
                    r = b.run(pt, eng, 1 | 2 | 4, True, 1, bn, plan_opts)
                    r["kind"] = "check"
                    bad += not r["pass"]
                    emit(r)
                if a.time:
                    r = b.run(pt, eng, combmode, False, a.launches, bn, plan_opts)
                    r.update(kind="time", comb=a.comb, job=a.job)
                    emit(r)
    print(f"\n{'build':<10} {'point':<11} {'kind':<5} {'engine':<6} {'S max':>5} {'A med (p10/p90) us':>22} {'C med us':>9} {'check rel-L2 (l0 / l27)':>24} result")
    for r in rows:
        if r["kind"] == "check":
            rl = r["rel_l2"]
            print(f"{r['build']:<10} {r['point']:<11} check {r['engine']:<6} {r['s_max']:>5} {'-':>22} {'-':>9} {rl['0']:>11.2e} /{rl[str(NL - 1)]:>10.2e} "
                  f"{'PASS' if r['pass'] else 'FAIL'} (sw128 bad bytes {r.get('sw128_bad_bytes')})")
        else:
            print(f"{r['build']:<10} {r['point']:<11} time  {r['engine']:<6} {r['s_max']:>5} {r['A_med']:>8.2f} ({r['A_p10']:.2f}/{r['A_p90']:.2f})".ljust(63)
                  + f" {r['C_med']:>9.2f} {'-':>24} {r['comb']}" + (" timing-only" if r["timing_only"] else ""))
    print("checks failed:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
