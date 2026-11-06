# B1 b1_stream: weight stream vs active SMs
Jobs: 14279 (sweep, 2 passes), 14281 (sweep, 2 passes, second job, same node `cayenne`), 14282 (large S, 2 passes). All priority, 1 GPU, B200 148 SM.
Raw: results/2026-10-29-occfuse/b1_stream/ (run{1,2}_JOB.jsonl, .st for S = 128/256/1024 KB; analysis_14279_14281.txt, fits_14279_14281.json, big{1,2}_14282.jsonl). Analysis: `analyze.py RESULTS 14279 14281`.

## Method
- Kernel `k_b1`: grid 148, 512 threads, cooperative launch, split barrier of ub.cuh (`bar_arrive` / `bar_wait`), 100 phases per launch, 2 warm-up + 7 timed launches. Stamps as B0.
- Active CTA: thread 0 = producer (waits `empty`, `expect_tx` 32 KB, 2 TMA boxes of 64 k x 128 rows SW128 per slot), thread 32 = consumer (waits `full`, arrives `empty`, no math). Ring state continues across phases. Inactive CTAs only do the barrier. No prefetch (`pf` = 0).
- A comes from the 2 GiB Pool (matrix [rows][1024] bf16, 8 k-steps per 128-row block); each (phase, CTA) reads its own consecutive window, so it is a new HBM window every phase (window wraps after about 2 GiB, > 2x L2). Activations: none.
- Placement: spread = tiles.py floor ranges (`((b+1)*N)/148 > (b*N)/148`), first = blockIdx < N.
- Time columns: `busy` = median over active CTAs of (arrive(p) - exit(p-1)), mean over phases 2..100 (resolves below the 0.256 us median grid); `last` = max over CTAs, same mean; `T` = Tmean (telescoping). Medians over the 7 launches. us_T/us_busy/us_last medians per launch are also in the JSONL.
- Sweep: N {8,16,32,48,64,96,128,148} x place {spread, first} x S {32..1024 KB (10 points)} x R {2,3,4,6,7} = 800 rows per pass.

## Deviations from METHODOLOGY.md
- R = 8 is not run: 8 x 32 KB + 3 KB fixed > 227 KB smem. R = 7 (232 448 B, exactly SMEM_MAX) replaces it.
- Correctness arm: none (no values are written; completion of every `full` wait implies all 32 KB of expect_tx landed). No FAIL rows possible.
- Stamps stored only for S in {128, 256, 1024} KB (1 launch per config) to limit size; metrics for all rows are computed host-side from the full stamps of every launch.
- Extra job 14282 with S up to 7424 KB (why streams 7.26 MB per SM) at N = 148 / 96 / 48, nph 50, spread only.
- Clock: not locked. mhz from clk_sample: median 1964 MHz, range 1962.4-1964.9 over all rows (this is a light-load run, no tcgen05; NVML 1965 MHz before and after, throttle 0x0).
- launch time / nph (events) vs Tmean: 7.67 vs 7.61 us (N=148, R=4, S=256), 12.92 vs 12.85 (S=512): +0.06-0.07 us (<1 %), within the 5 % limit.

## Results [M] (spread placement; busy work in us; jobs 14279 + 14281)
R = 4 (S in KB):

| N \ S | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|
| 8 | 0.82 | 1.19 | 2.22 | 3.48 | 6.00 | 11.02 |
| 16 | 0.88 | 1.62 | 2.27 | 3.52 | 6.03 | 11.05 |
| 32 | 1.29 | 1.65 | 2.30 | 3.57 | 6.06 | 11.10 |
| 48 | 1.31 | 1.66 | 2.33 | 3.61 | 6.13 | 11.18 |
| 64 | 1.35 | 1.70 | 2.40 | 3.76 | 6.40 | 11.56 |
| 96 | 1.42 | 1.88 | 2.74 | 4.53 | 7.94 | 14.77 |
| 128 | 1.52 | 2.11 | 3.24 | 5.58 | 10.02 | 18.95 |
| 148 | 1.59 | 2.26 | 3.56 | 6.31 | 11.43 | 21.78 |

R = 2 at S = 256 KB: 4.63 (N=8), 4.89 (48), 6.35 (128), 6.91 (148). R = 7 is within 0.15 us of R = 4 up to N = 64; at N = 148 it is slightly slower (S=1024: 21.57 vs 21.78 busy but see aggregate below). All tables (R 2/4/7, busy/last/T) are in analysis_14279_14281.txt.
- Spread vs first-N placement: the busy time differs by 0.026 us median, 0.255 us max over all 800 configs (first-N N=64 R=4 S=256: 3.90 vs 3.76 is the largest class). No placement dependence beyond this.
- Pass-to-pass (4 passes, 2 jobs): relative spread of busy 0.3 % median, 3.9 % max (small-S points); N=148 R=4 S=1024 busy 21.78 vs 21.79, last 22.83 vs 22.99.
- Unexplained [M]: at S = 32 and 64 KB busy steps by about 0.4 us between N = 16 and N = 32 (0.88 -> 1.29; 1.62 at 16 vs 1.19 at 8 already for S=64), independent of R. Not investigated.

## Fit t = t0 + S / min(r_sm, BW_agg / N) [M]
Spread placement, S >= 64 KB, all N, 288 rows (8 N x 9 S x 4 passes). Rates in GB/s (1e9 B/s). Weighted least squares (grid search, t0 solved).

busy work (median over active CTAs):

| R | t0 us | r_sm GB/s | BW_agg GB/s | rmse us |
|---|---|---|---|---|
| 2 | 0.735 | 63.1 | 6540 | 0.475 |
| 3 | 0.913 | 91.1 | 7165 | 0.284 |
| 4 | 0.985 | 102.3 | 7375 | 0.130 |
| 6 | 1.001 | 103.3 | 7410 | 0.109 |
| 7 | 0.999 | 103.3 | 7390 | 0.120 |

slowest CTA (last): R=2 t0 0.95 / 61.3 / 6125; R=3 1.12 / 89.2 / 6725; R=4 1.19 / 101.7 / 7005; R=6 1.19 / 103.0 / 6995; R=7 1.19 / 103.0 / 6975. First-N placement: within 0.05 us t0, 1 GB/s r_sm, 50 GB/s BW (fits_14279_14281.json).
R dependence: r_sm is ring-limited at R <= 3 (r_sm x R is not constant: 63, 91, 102 GB/s for R = 2, 3, 4; r_sm(2) x latency = 64 KB gives an implied TMA round trip of about 1.0 us at 63 GB/s) and saturates at about 103 GB/s from R = 4. BW_agg rises with R up to 4. The fit is a two-regime model; the real curves round the knee (rmse 0.11-0.13 us at R >= 4, up to 0.48 us at R = 2), see the 96/128 CTA rows.

## Gate [M]
Aggregate = N_active x S / t, GB/s (1e9 B/s), N = 148 spread.

| S KB | R | last (slowest CTA) | busy | T |
|---|---|---|---|---|
| 1024 | 4 | 6751-6804 | 7120-7126 | 6515-6538 |
| 7424 | 2 / 3 / 4 / 6 / 7 | 6745 / 7157 / 7141 / 6889 / 6830 | 6961 / 7360 / 7455 / 7657 / 7657 | 6708 / 7115 / 7102 / 6853 / 6795 |

(S = 7424 KB from job 14282, two passes agree to under 1 %.) Reference `why` (job 11678, aggregate = bytes / make-span, 7.26 MB per SM): static 5714 (static_contig 6399, queue 6323, static_il 6514, static_il_al 6727 best); per-SM-normalized x148: static 5.95, static_il 6.59 TB/s.
- Versus the best why arm (static_il_al 6727; il 6514-6727): b1 `last` 7141 at R = 4 is +6.2 % (+9.6 % vs static_il 6514). PASS within 10 %.
- Versus the plain why static arm (5714) and queue (6323): +25 % and +13 %. FAIL for those arms.
- At S = 1 MB (the largest S of the spec grid) b1 `last` = 6.75-6.80 TB/s, still rising with S (6.6 at 512 KB for R=7), the t0 term not yet amortized.
- Verdict: PASS against the why layouts that interleave channels (il / queue-class, 6.5-6.7 TB/s); b1 measures higher than the why static layout. Reason not tested; the b1 access pattern is 128 rows x 128 B at a 2 KB stride per box, 8 k-boxes adjacent (a different channel spread than why's contiguous 64 KB chunks) [H].
- N = 48 S = 256 KB busy work (spread, no prefetch) [H until B2]: R = 2 4.90, R = 3 3.90, R = 4 3.61, R = 6 3.60, R = 7 3.61 us; last 3.86 at R = 4; T 4.76. mk T_null GU busy = 2.62 us at b8 (prefetched). Difference at R = 4: 0.99 us (no prefetch here, so not comparable until B2).

## Caveats
- Both jobs ran on the same node (cayenne); no second node placement.
- No match between "busy" and "T" is attempted: T = busy + barrier part (about 1.15 us at R=4 N=148 S=256; includes the barrier latency and tails).
- Aggregate at N = 148 depends on R above 4 (R = 6/7 lower than R = 4 by 3-4 % for `last`); not explained.
- Fit is only for the plain ring (no compute, no prefetch, no B operand, L2 promotion 128 B, K = 1024 matrix layout). Timing of slots is by the consumer free at once; the producer waits for `empty` with try_wait.
