# SPEC: xover - FFMA vs tcgen05 GEMV/GEMM engine over tokens per step (T = 1..256), B200

## Why
The regime plan splits the bespoke Qwen3-0.6B decode megakernel into two arms: MK-small (1-8 tokens per step, FFMA
engine, like hv2) and MK-batch (9-256 tokens, tcgen05 engine). The user's hypothesis: one tcgen05 kernel covers all
token counts. Measured facts so far, all at T = 1 only:
- tcgen05.mma M128 N16..64 K16 costs ~58-59 cycles flat; N=128 64 c, N=256 128 c (dense_tc, hv2_tc mma_bench2 job 9425).
- Isolated hv2 phases on tcgen05 lose +0.5..+1.25 us/layer; TC_D in the kernel +49 us/step (hv2_tc/RESULT.md).
- Any tcgen05.mma lowers the SM clock 1964 -> ~1845 MHz for the whole kernel (tc_clk_bench.py, jobs 9420/9421).
Nobody measured T = 2..256. This microbenchmark answers: at which T does the tcgen05 engine become faster than an FFMA
engine for one GEMM phase, and how large is the FFMA advantage below that T? It also checks if a binary that CONTAINS
the tcgen05 path but runs the FFMA path keeps the full clock (the "one binary, runtime switch" option).

## What to build
One CUDA file `probe/xover/xover.cu` (nvcc, `-O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a`, driver API
for `cuTensorMapEncodeTiled` via `cudaGetDriverEntryPoint` as in the reference below), plus `probe/xover/run.sh`
(build + run all configs, prints JSON lines) and `probe/xover/xover.sbatch` (1 GPU, `main` partition, 20 min,
log to `probe/xover/logs/slurm-%j.log`, prints `nvidia-smi -L` first). No Python dependencies beyond the venv.

Known-good references (copy helpers, do not invent PTX from memory):
- `/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/ubench/gemm_tc.cu` (U10: 148-CTA tcgen05 GEMM task, TMA
  SW128 tiles, counter grid barrier, cold-weight pool walk, TMEM ld, correctness vs CUDA-core reference). Closest model.
- `/raid/garv901/vibe-megakernel/megabench/kernels/dense_tc/tc_common.py` HELPERS_SRC (`tc_idesc(M,N)`, `tc_a_desc`,
  `tmem_alloc`, `tc_mma`, commit, `tcgen05.ld` helpers; M=128 descriptors that are verified).
- `/raid/garv901/vibe-megakernel/megabench/kernels/hv2_tc/tc_clk_bench.py` (clock measurement: clock64 delta /
  globaltimer delta).

### Problem
Per phase: `Y[T][N] (+)= X[T][K] * W[N][K]^T`, W bf16 row-major (K contiguous), X bf16, Y fp32. Two shapes:
- `GU`: N = 6144 (gate+up of Qwen3-0.6B), K = 1024.
- `D` : N = 1024 (down), K = 3072.
T in {1, 2, 4, 8, 16, 32, 64, 128, 256}.

### Launch and timing (identical for every engine)
- 148 CTAs x 512 threads (16 warps), 1 CTA/SM, cooperative launch (`cudaLaunchCooperativeKernel`), dynamic smem
  sized for the largest config (same for all engines of one (shape, T), so occupancy is the same).
- One launch runs `nph` phases (default 112 = 4 x 28 layers). Between phases a counter grid barrier (copy U10's).
- Weights are cold: phase p reads a fresh slice of a >= 2 GB pool (wrap around), as U10 does.
- Work unit = 128 rows x 128 k of W (32 KB bf16). Units are numbered u = rb * (K/128) + kc (k-major inside a row
  block). CTA b owns units [b*U/148, (b+1)*U/148) (GU: U = 384, D: U = 192). Same assignment for all engines.
- Weight path (same for all engines): TMA 2D (SW128, box 64 k x 128 rows; 2 boxes per unit) into a smem ring of 2
  unit slots (64 KB), one mbarrier (expect_tx) per slot. The NEXT phase's first two units are issued after this
  phase's math has finished reading the ring and before the grid-barrier wait (barrier-window prefetch, as hv2). A CTA
  with a 3rd unit (GU) issues it into slot 0 as soon as unit 0's math has finished reading slot 0.
- X path (same for all engines): per unit, all threads stage the unit's X columns (128 k x T) from global (bf16
  [T][K], L2-hot) into one of 2 smem X slots in the SW128 K-major layout tcgen05 needs as the B operand (rows = tokens,
  zero rows up to Npad = max(16, roundup16(T)); 128 B rows, 2 k-atoms of 64 per unit; 64 KB per slot at T = 256).
  FFMA reads the same swizzled slot (XOR the 16 B chunk index with row % 8). Order per unit i: stage X(i) into slot
  i%2, `__syncthreads` + `fence.proxy.async`, math(i). ENG=1 may stage X(i+1) while MMA(i) runs; before it overwrites a
  slot it waits for the commit of the MMAs that read it. Smem total <= 64 + 128 KB + small.
- Output: Y is fp32 and updated with `red.global.add.f32` (t < T only). ENG=1 keeps a row block's sum in TMEM and
  adds it to Y when the CTA's unit range leaves the row block or ends. ENG=0 adds each unit's 128 x T sums straight from
  registers after its shuffle reduce (lane%4 == 0), so it does up to ~2x the red.adds of ENG=1 at D; write this down.
  A `__threadfence()` follows, then the barrier arrive.
- Timing: cudaEvent around the launch, us per phase = time / nph, median of 5 reps after 2 warm-up launches.
  Clock: thread 0 of every CTA records clock64 and globaltimer at kernel start and end; report the median MHz.

### Engines (`ENG` macro, one binary per engine; plus E2 below)
- `ENG=0` FFMA. Per unit: warp w owns rows 8w..8w+7; lane l owns row 8w + l/4 and k quarter (l%4)*32..+31 of the
  unit. The lane reads its 32 weights from smem once into fp32 registers, then loops over tokens in tiles of TB (macro,
  default 4): for each token, 8 x LDS.128 of X (bf16, convert by shift), 32 FFMA, keep TB fp32 sums; reduce over the 4
  lanes of a row with 2 shuffles per token; red.add as in Output. This is a reasonable, not a peak, FFMA engine: report its FFMA count.
- `ENG=1` tcgen05, swap-AB: A = weight unit (M = 128 rows, K-major SW128), B = X tile (N = Npad tokens, K-major
  SW128), D = fp32 in TMEM, N = Npad columns. One thread issues 8 x `tcgen05.mma.cta_group::1.kind::f16` (K = 16 each)
  per unit; units of one row block accumulate (enable-input-d = 1 after the first). `tcgen05.commit` -> mbarrier;
  then warps 0-3 (TMEM lane quadrants) `tcgen05.ld` the 128 x Npad accumulator and do the red.add (warps 4-15 may
  share columns: warp w reads lane quadrant w%4, column blocks w/4..). TMEM: allocate 256 columns once at kernel start
  (power of 2 >= Npad; at Npad = 256 allocate 256), dealloc at the end.
- `ENG=2` FFMA path of ENG=0 inside a binary built with BOTH paths (runtime flag `eng` selects; the tcgen05 code
  is compiled in), and the kernel runs `tcgen05.alloc` / `dealloc` as ENG=1 does, but never issues an MMA. Question:
  does the clock stay at the FFMA value? Also run the same binary with `eng=1` and report it as ENG=3 (must match ENG=1
  within noise).
- `ENG=-1` no-math oracle: same TMA, X staging, barrier, red.add of zeros. Gives the floor for this skeleton.

### Correctness
For each engine, shape and T in {1, 8, 64, 256}: one launch with nph = 1 and Y zeroed first; compare Y with a
CUDA-core reference kernel (fp32 accumulate of bf16 inputs; simple, one thread per (t, n)). Pass if
max |d| <= 2e-3 * max(1, |ref|). Use random W, X in [-1, 1]. Print PASS/FAIL per config. Timing configs that fail are
flagged in the JSON.

### SASS check (write into NOTES.md)
`cuobjdump -sass` of each binary: ENG=0 has FFMA and no UTCMMA/UTCHMMA (or any tcgen05 MMA mnemonic: find the exact
name in the dense_tc or gemm_tc binary first), ENG=1 has the MMA mnemonic; report regs and spill
(`-Xptxas -v`) per binary. 0 spill expected; if not, say so.

### Output
One JSON line per (eng, shape, T): `{"eng":..,"shape":..,"T":..,"us_phase":..,"us_reps":[..],"mhz":..,
"pass":..,"regs":..,"spill":..}`. Also print the two floors per (shape, T): weight stream time = bytes / 6.5 TB/s,
and FFMA math floor = T*N*K / (148 * 128 * 1.964e9) s.

## Acceptance (CPU side, before any GPU job)
- `bash probe/xover/run.sh --build-only` builds all binaries with nvcc on this machine (no GPU needed for nvcc).
- SASS check done and written to `probe/xover/NOTES.md`.
- Do NOT submit Slurm jobs and do not run on the GPU. The main agent reviews and submits.

## Files you own
`probe/xover/` in `/raid/garv901/regime-megakernel` only. Do not edit anything else. Do not commit.

## Report back
Files written, build result, regs/spill per binary, SASS mnemonics found, anything in this spec you changed and why
(for example a layout detail that did not work), and the exact command for the GPU run.
