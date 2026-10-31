# xover6 notes (SPEC-xover6; jobs 14913 (bug: config filter) and 14919 (final, logs/sum-14919.txt, runs/slurm-14919.log))
Build: `bash run.sh --build-only` (bin/x6 from xover6.cu; bin/ref{148,111,74} = xover4.cu -DCG=1 -DGRID_N=n, the compile-time reference).
## Design
- k_ts = xover4 k_ts, CG = 1 only, tile split uses gridDim.x (run-time k_c); extra arg `stop` (CTA 0 stores 1 after the last grid barrier). Math unchanged.
- k_kv: 512 threads, dynamic smem always 227 KB (1 CTA / SM, never shares an SM with k_ts). Stage = 128 rows x 128 dims of one head = 2 SWIZZLE_128B boxes {64, 128}; map dims {1024, rows}, L2 promotion NONE, createpolicy evict_first +
  `.L2::cache_hint` (copied from splitkv.cuh). Global stage g -> rows (g / 8) * 128, head g % 8 (8 stages = 256 KB contiguous). CTA b owns stages [b*S/nb, (b+1)*S/nb); pool 8 GiB (0x3C fill), window (it % 16) per iteration.
  Producer = tid 0, consumers = warps 4..15 (16 B / thread / slot, integer sum -> sink), empty count 12 (one lane per warp). No grid barrier. Stamps per CTA and iteration.
- Producer end: a sentinel stage (plain arrive on full[s] + done[s] flag in smem). Stream length: `iters` in kv mode; in `both` mode the producer runs until k_ts sets the `stop` flag (cap 20000 iterations).
- `both`: k_ts launched first on stream A (NON-cooperative: all k_c CTAs are resident since k_c + k_a <= 148, 1 CTA / SM both; occupancy printed), k_kv on stream B. nph = max(112, 60 ms / alone us). GEMM us/phase = (ts last end - first start) / nph,
  window = [max starts, min ends] (overlap >= 0.9997 in every row, so ts time is inside the window); KV GB/s in the window prorates each CTA iteration by its overlap with the window. The best (TT, KS) is chosen by an in-process alone sweep at that k_c.
## Numbers reported per kv config
agg_gbs_total = all iterations / (last end - first start); agg_gbs_ss = skips the first 4 iterations; agg_gbs = median of per-iteration (max end - min start) as in the SPEC (no barrier, so CTA drift makes it much lower; not a rate).
## Deviations
- Per-iteration SPEC metric is shown but the total / steady-state rates are the meaningful ones (no grid barrier in k_kv).
- `both`: GEMM PASS is checked on the chosen config alone (1 phase, before the concurrent run); no result check under contention (red.add accumulates over thousands of phases).
- `both` uses a non-cooperative k_ts launch. `gemm`/`kv` modes: cooperative k_ts as xover4.
- Config filter: tiles >= min(k_c, 37) (xover4 rule NTILE >= 37; job 14913 used tiles >= k_c and missed configs like GU T128 KS 2; fixed in 14919).
- kv-alone value in table C = in-process 100-iteration steady-state run (not table A's 20 iterations).
- T = 32 gemm at k_c 111 / 148 not run (plan lists k_c 20..74 only).
## Build / SASS
k_ts 72 regs, k_kv 23 regs, both 0 stack / 0 spill stores / 0 loads (logs/build.log). SASS (logs/sass.txt): k_kv UTMALDG 2, UTCHMMA 0, HMMA 0, FFMA 0; k_ts UTMALDG 32, UTCHMMA 8, HMMA 0, FFMA 0.
