# SPEC mk v0: first e2e tcgen05 decode megakernel arm `mk` (Qwen3-0.6B, B200, 1-256 tokens/step)

## Why (measured, jobs)
- hv2g MTN (`regimebench/kernels/hv2g/hv2g_mtn.cu`, jobs 9464/9468/9471) runs every regime but its FFMA GEMV grows
  ~+150-180 us per token (b8 1482 us, b256 37404 us; fi-graph b8 973, b256 2242). [M]
- `probe/xover` (job 9521): a tcgen05 swap-AB GEMM phase on 148 CTAs is flat in T (GU 7.74 us at T=1, 14.1 at 256;
  D 5.10 / 9.09) and beats FFMA from T=2. User decision 2026-10-23: tcgen05 for every token count, no FFMA path. [M]
- v0 = hv2g MTN's skeleton and glue + xover's tcgen05 GEMM phase + hv2g's attention as a placeholder behind the A1
  interface. It is the first e2e build; the per-phase optimisation passes come after. Correctness first, no tuning.

## Read first
`docs/briefs/MK_CORE_BRIEF.md` (context only), `regimebench/kernels/common/README.md` (A1 interface, ownership),
`regimebench/arms/{base.py,hv2g.py}`, `regimebench/kernels/hv2g/{build.py,coop.py,hv2g_common.cuh,hv2g_mtn.cu,NOTES.md}`,
`probe/xover/{xover.cu,NOTES.md}` (tcgen05 helpers, verified), `regimebench/fixture.py`, `regimebench/workload.py`,
`/raid/garv901/vibe-megakernel/megabench/kernels/dense_tc/tc_common.py` (NVRTC sm_100a build, host tensor-map encode
via `cuda.bindings.driver.cuTensorMapEncodeTiled`, maps in global memory used through a generic pointer: copy that
pattern; `CUtensorMap` is not defined under NVRTC). Read-only: everything outside "Files you own".

## Design (implement exactly; where something does not work, stop, write down why, pick the nearest fix, report it)
### Launch
148 CTAs x 512 threads, 1 CTA/SM, cooperative launch (copy `coop.py` into `kernels/mk/`; add sm_100a), monotonic grid
barrier with arrive / wait split (hv2g `bar_arrive` / `bar_wait`, counter reset by the last CTA at the end), one launch
per step (every mk point has total_q <= 256), CUDA graph in the arm. Token bucket = compile-time `NPAD` in
{16, 32, 64, 128, 256} (one cubin per bucket, entry `mk_step`); runtime `ntok <= NPAD`. B-tile rows >= ntok are zero.
TMEM: 256 columns allocated at kernel start by warp 0, freed at the end.

### Buffers (arm-owned scratch; fp32 unless stated)
`x[256][1024]` residual; `qkv[256][4096]` (red.add target); `qb[256][16][128]` bf16 (q after q-norm + RoPE: the A1
input); `ao[256][2048]` bf16 (attention output: the A1 output, O-proj B operand); `gu[256][6144]` (red.add target,
rows 0..3071 gate, 3072..6143 up); `rr1[256]`, `rr2[256]`, `rrf[256]` (per-token RMSNorm reciprocal for ln1, ln2,
final norm). Zeroing (a buffer is zeroed by a phase in which nobody reads or writes it, behind a barrier from both its
last reader and its next writer): `gu` in phase 1, `qkv` in phase 4, both also in the prologue.

### Phases per layer (6 grid barriers, as hv2g MTN)
1. **QKV GEMM**: B tile = bf16(x[t][k] * ln1[k]) (deferred norm: the per-token scale is applied later). Out: red.add
   into qkv (unscaled). Also: CTA b computes rr1[t] = rsqrt(mean(x[t]^2) + eps) for t = b, b + 148, ... (x is not
   written in phases 1-3). Call site `skv_prefetch(l, blockIdx.x, gridDim.x)` after the CTA's weight TMAs are issued
   (no-op in the placeholder).
2. **q/k-norm + RoPE + KV write** (hv2g phase 2, one warp per (token, head) over 16 q-heads + 8 kv-heads): q-norm and
   k-norm are RMSNorms over the head, so the missing rr1[t] factor cancels (scale invariance; note the eps term changes
   by rr1^2, say in the report if the oracle band shows it); v is multiplied by rr1[t]. Writes qb (bf16), K/V cache rows.
3. **Attention** (placeholder, see below) -> ao (bf16).
4. **O GEMM** (K = 2048): B tile = ao (bf16 copy). Out: red.add into x (residual). Zero qkv.
5. **GU GEMM**: B tile = bf16(x * ln2). Out: red.add into gu (unscaled). CTA b computes rr2[t] for t = b, b+148, ...
   (x is not written in phase 5).
6. **D GEMM** (K = 3072): B tile = bf16(silu(rr2[t] * gate) * (rr2[t] * up)) from gu. Out: red.add into x.
Then **LM head**: rrf[t] per lm token (every CTA computes the ones it needs, x is final), B tile = bf16(x[lm_tok] *
fnorm); units are full row blocks (128 vocab rows x K 1024, accumulated in TMEM, no split-K), epilogue writes
logits[lm_seq][row] = rrf * acc (fp32) and keeps a per-CTA best per lm slot; last-CTA ticket reduces the argmax
(hv2g pattern, ties -> lowest index). 1187 row blocks over 148 CTAs.

### GEMM phase (one templated device function for all four shapes + LM)
xover ENG=1 code (`probe/xover/xover.cu`): weights W[N][K] bf16, swap-AB (A = 128-row x 128-k weight unit, K-major
SW128 by TMA 2D, 2 boxes of 64 k; B = staged token tile, N = NPAD), 8 x K16 `tcgen05.mma.cta_group::1.kind::f16` per
unit, TMEM accumulator per row block, `tcgen05.commit` -> mbarrier, TMEM ld by warps (quadrant = warp % 4), red.add of
t < ntok, split-K by unit ranges per CTA (u = rb * KC + kc, CTA b owns [b*U/148, (b+1)*U/148)). The staging functor
(per phase, above) replaces xover's plain X copy; staging reads fp32 / bf16 from global (`__ldcg`) and writes the SW128
K-major B layout exactly as xover's `stage()`. Tile knobs as macros with xover defaults so Pass 0b's winner drops in:
`MK_KU` (128), `MK_RING` (2), `MK_XTMA` (0; only the O phase may use 1 later). Weight maps: wq, wk, wv (QKV row blocks
0-15 / 16-23 / 24-31 map to the three tensors: 2048 and 1024 are multiples of 128), wo, wg / wu (gu rows), wd, embed;
each map 2D {K, L * rows}, box {64, 128}, SW128 (embed: {1024, 151936}).
**Barrier-window prefetch**: after a CTA finished reading the ring in a GEMM phase it issues the first `min(RING, n)`
units of its NEXT GEMM phase (QKV -> O skips phases 2 and 3: the ring is idle there, so O's units go out at the end of
phase 1), then arrives / waits at the barrier.

### Attention placeholder (`kernels/mk/attn_placeholder.cuh`)
The A1 interface (common/README.md): header with `SKV_`-prefixed macros only, no dependence on mk's macros, model
sizes as constexpr. Functions: `skv_prefetch(layer, cta, nct)` (no-op), `skv_attention(...)` = hv2g MTN phase 3
(`hv2g_mtn.cu:194-262`) with q read from `qb` (bf16, already normed + roped) instead of recomputed, writing `ao` bf16
[t][qh*128 + d]; `skv_combine(...)` no-op (the placeholder has no splits). Build flag `MK_ATTN` = `placeholder`
(default) | `splitkv` (prepends `regimebench/kernels/common/splitkv/splitkv.cuh` if present, else a clear error).
Keep hv2g's CTX padding rows and clamp rules (see hv2g NOTES "MTN design").

### Knobs (macros, default 0; with all 0 the kernel must be the same SASS as without the knob code: check by building
with and without the knob blocks compiled in and diffing `cuobjdump -sass`, write the result in NOTES.md)
- `MK_STAMPS=1`: per CTA %globaltimer at every bar_arrive and bar_wait exit (port of hv2 `R11_SKEW`,
  `kernels/hv2/hv2_submission.py:469-490`), into a `__device__` array the arm can read (`config` gets nothing; a
  separate script reads it).
- `MK_NULL=<bitmask>` bit p-1 = phase p (1..6), bit 6 = LM: the phase keeps its barrier(s) and its weight TMAs + ring
  waits but skips staging, MMA, TMEM read and output (wrong outputs by design; null-work oracle as in pi05).

### Arm `regimebench/arms/mk.py`, class `MkArm` (already registered in `arms/base.py`)
Mirror `arms/hv2g.py`: variants `("auto", "auto-direct")` (auto = smallest NPAD >= total_q, graph; -direct = no graph);
`supports()`: total_q in 1..256 and model qwen3-0.6b, else a reason. `prepare()`: packed cache with CTX zero rows
(hv2g), descriptors (reuse `kernels/hv2g/build.build_descriptors` by import: read-only reuse is fine), RoPE table,
tensor maps encoded once (pointers are fixed for the step), scratch, warm-up, graph capture. `config`: kernel, npad,
graph, grid/block/dyn_smem, regs, local_bytes, ptxas, attn ("placeholder"), knobs, `source_sha256` (all mk sources +
the attention header). `outputs()` as hv2g (gather k_new / v_new from the cache).

### Tests and checks
- `tests/test_mk.py` (CPU, pytest): NVRTC compiles every NPAD bucket (sm_100a), 0 spill, regs <= 128; bucket choice;
  `supports()` skip reasons; source hash changes when the attention header changes; knob build compiles.
- `scripts/gpu_check_mk.py` (copy the structure of `scripts/gpu_check_hv2g.py`): for each workload in a default list
  covering every bucket (R0-b1-kv128, R1-b2/b8/b16/b32/b64/b128/b256-kv128, R2-b1-kv8192, R6-b1-kv1024-q4, R3-b8-kv4096):
  accuracy vs the FP32 oracle band (`harness/accuracy.py`), next_token match, repeat-launch idempotence, one steady
  timing. Exit 1 on FAIL. Also `--null <mask>` and `--stamps` modes that print per-phase times (stamps: median over
  CTAs and layers 4..23 of wait-exit -> next arrive, per phase).

## Measurement (main agent runs these; you do not)
`sbatch` a 1-GPU main job for `gpu_check_mk.py` (all default points, ~20 min); then e2e
`slurm/regime.sbatch --arms mk,hv2g,fi-graph --workloads mk --tag mk-v0` on background.

## Acceptance (CPU, before any GPU job)
- `PYTHONPATH=$PWD /raid/garv901/vibe-megakernel/.venv/bin/python -m pytest -q tests` passes (all tests, not only yours).
- `cuobjdump -sass` of the NPAD=16 and NPAD=256 cubins: UTCHMMA present, UTCBAR, LDTM; no HMMA; FFMA only outside the
  GEMM phases (norms, attention, SiLU). Regs, spill (must be 0; if not, where and how many bytes), smem per bucket.
- Knob-off SASS identity check done (above).
- `regimebench/kernels/mk/NOTES.md`: files, design as built, every deviation from this SPEC and why, ptxas table,
  SASS counts. Do NOT submit Slurm jobs, do not run on a GPU, do not commit.

## Files you own
`regimebench/kernels/mk/` (new: `__init__.py`, `mk_common.cuh`, `mk_step.cu`, `attn_placeholder.cuh`, `build.py`,
`coop.py`, `NOTES.md`), `regimebench/arms/mk.py`, `tests/test_mk.py`, `scripts/gpu_check_mk.py`. Nothing else
(never `kernels/hv2*`, `kernels/common/`, `probe/`, shared harness files, `arms/base.py`). Another agent works in
`probe/mk_core/xover2/` at the same time: do not touch it.

## Report back
Files; ptxas table per bucket; SASS counts; test output (last lines); every deviation; open risks you see for the GPU
run (e.g. TMEM / mbarrier parity across phases, tensor-map alignment, red.add ordering), ranked.
