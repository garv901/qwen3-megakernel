# SPEC-xover6: GEMM phases on a CTA subset next to a KV stream on the other SMs (standalone bench, PASS-19 gate)

Owner: mk-core-r10. Implementer: a Sonnet subagent. Pass: PASS-19 (read `probe/mk_core/PASS-19.md` first).
You implement, build, run, and report numbers. You do not decide keep or drop.

## Why
- [M] job 14897: at the R10 points (decode, 48-256 requests, kv 1K-4K) mk attention is 77-89 % of a layer and runs near
  the HBM rate; the GEMM phases (QKV / O / GU / D, 31 MB of weights per layer) take 27-43 us/layer and are latency bound
  (job 9765: GEMM sum per layer T=64 on 148 / 111 / 74 CTAs = 22.4 / 24.0 / 25.0 us).
- Idea: run the GEMM chain of one half of the batch on k_c CTAs while the other k_a = 148 - k_c SMs stream the KV of
  the other half. Two open hardware questions decide if this can pay:
  - Q1: what aggregate HBM rate does a TMA KV stream reach on k_a SMs (k_a < 148), as a function of bytes in flight per SM?
  - Q2: how long do the GEMM phases take on k_c CTAs WHILE that stream runs on the other SMs (HBM / L2 contention)?

## Files (new directory; do not edit xover4 or anything outside xover6)
- `probe/mk_core/xover6/xover6.cu`: copy of `probe/mk_core/xover4/xover4.cu` (CG = 1 only; you may delete the CG = 2 code
  paths or leave them compiled out), plus the stream kernel and the concurrent driver below.
- `probe/mk_core/xover6/run.sh` (build with `-Xptxas -v` into `logs/build.log`, `--build-only` flag),
  `probe/mk_core/xover6/xover6.sbatch` (the run plan), `probe/mk_core/xover6/NOTES.md` (design, deviations, regs /
  spills, SASS check), `probe/mk_core/xover6/sum.py` (tables from the JSON lines).
- Build: `/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v` (as xover4/run.sh).

## The bench
1. **GEMM role** = xover4 `k_ts` unchanged in its math (tile stream, TMA weights + X, tcgen05 MMA, TMEM epilogue,
   red.add, grid barrier `gbar` between phases). It must run on k_c CTAs chosen at RUN time (not only GRID_N at compile
   time): pass the CTA count as an argument and use it everywhere GRID is used for the tile split and the barrier
   (gbar already uses gridDim.x). The prior compile-time GRID_N path (job 9765) is the reference: at k_c = 148 / 111 / 74
   your run-time version must reproduce the GRID_N build within 3 % (run both once; report).
2. **Stream role** = a new kernel `k_kv` (512 threads, 1 CTA per SM by its dynamic smem): CTA b streams its
   contiguous share of `bytes_per_iter` bytes from a large KV pool (>= 8 GiB, bf16, > 60x L2, so the KV is never
   L2-resident; move the window forward every iteration so no iteration re-reads the previous one's bytes) with TMA:
   - 2D tensor map over the pool viewed as rows of 256 B x 8 heads (the mk KV row: 8 KV heads x 128 dims bf16 =
     2048 B per token row); 32 KB per stage = 128 token rows x 128 dims of one head (with SWIZZLE_128B the inner box
     dim is at most 128 B, so use the box geometry the splitkv header uses, e.g. 2 boxes of 64 dims), SWIZZLE_128B as mk's KV maps do
     (take the encode call and the L2 promotion / evict policy from the splitkv header's KV map:
     `regimebench/kernels/common/splitkv/splitkv.cuh`, search `cuTensorMapEncodeTiled` or the host-side map spec in
     `regimebench/kernels/mk/build.py`; KV loads use an L2 evict_first cache hint in mk (`SKV_KV_EVICT_FIRST`) - use the
     same `.L2::cache_hint` form, copied from the header).
   - ring of `depth` slots x 32 KB (depth in {2, 4, 6}); producer = one thread (warp 0 lane 0) issuing
     `cp.async.bulk.tensor.2d` with `mbarrier::complete_tx`; consumer = warps 4..15 wait full[s], read 16 B per thread of
     the slot (a sum into a register, written to a global sink once at the end so it is not dead code), arrive on
     empty[s]. Copy the TMA / mbarrier helper asm from xover4.cu (verified), not from memory.
   - no grid barrier; per-CTA globaltimer start / end per iteration into a global array.
3. **Driver modes** (one binary, args `mode k_a depth bytes_per_iter shape T TT KS iters`):
   - `kv`: only k_kv on k_a CTAs (other SMs idle). Report aggregate GB/s = bytes_per_iter / (max end - min start) per
     iteration, median over iterations, and per-CTA median rate.
   - `gemm`: only k_ts on k_c = 148 - k_a CTAs, one shape, `nph` phases (as xover4), us per phase median.
   - `both`: k_ts (k_c CTAs) and k_kv (k_a CTAs) at the same time on two streams. Launch k_ts first (cooperative is
     not required if all k_c CTAs are resident: check occupancy and that k_c + k_a <= 148 with 1 CTA / SM for both
     kernels; if the cooperative attribute blocks concurrency, launch k_ts without it and say so in NOTES). Size the
     work so both run >= 50 ms and overlap for >= 90 % of the shorter one's time (record both kernels' first start /
     last end globaltimer stamps and report the overlap fraction). Report the GEMM us/phase and the KV GB/s measured
     inside the overlap window only.
   - Correctness: keep xover4's PASS check for the GEMM role (it must PASS at k_c < 148 too); for k_kv, check that
     the byte count of every completed TMA equals the expect_tx (mbarrier completes) and the sink is non-zero.

## Run plan (xover6.sbatch; one job on `main`, 1 GPU, honest --time ~30 min; print `nvidia-smi -L`)
- `kv`: k_a in {148, 128, 116, 104, 92, 74} x depth {2, 4, 6}; bytes_per_iter 512 MiB; 20 iterations.
- `gemm`: k_c in {20, 32, 44, 56, 74} x shapes {QKV, O, GU, D} x T in {32, 64, 128}; for each (shape, T, k_c) sweep
  TT in {T, min(T, 64)} and KS in the xover4 list, report the best config and its time (the bench prints all).
- `both`: (k_a, k_c) in {(128, 20), (116, 32), (104, 44), (92, 56)} x depth {4, 6} x shapes {QKV, O, GU, D} at
  T in {32, 64, 128}, each with its best (TT, KS) from `gemm` at that k_c.
- Repeat the `kv` 148 / depth 4 and `gemm` k_c 74 T 64 GU configs at the start and end of the job (drift check).

## Acceptance (commands)
- `bash probe/mk_core/xover6/run.sh --build-only` succeeds; NOTES.md has regs and 0 spill / local memory for both
  kernels (`-Xptxas -v`), and the SASS check: `cuobjdump -sass bin/x6 | grep -c UTMALDG` > 0 in both kernels,
  `UTCHMMA` present in k_ts, no HMMA / FFMA loop in k_kv.
- `sbatch probe/mk_core/xover6/xover6.sbatch` from the worktree root `/raid/garv901/regime-mk-core` (check the
  `REPO=... branch sha` line), all GEMM PASS lines PASS.
- `python3 probe/mk_core/xover6/sum.py runs/slurm-<job>.log` prints: table A (kv: GB/s vs k_a x depth), table B
  (gemm: best us/phase and the 4-shape sum vs k_c x T), table C (both: GEMM sum and KV GB/s inside the overlap
  window, and their ratio to the alone values).

## Files you own
`probe/mk_core/xover6/` only. Read anything. Do not edit any other file, do not commit (the owner reviews and commits).
Use the run-gpu-job skill for the job (Slurm; `main`; at most 1 job at a time for this SPEC).

## Report back
Job id(s), GPU-h used, the three tables, regs / spills, SASS check, the k_c = 148 / 111 / 74 reproduction vs the
GRID_N build, the overlap fractions, anything that deviates from this SPEC and why.
