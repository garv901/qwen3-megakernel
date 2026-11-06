# B5 b5_epi: epilogue output modes
Jobs: 14293 (priority, cayenne; sweep pass 1 and pass 2 inside the job) and 14297 (priority, cayenne, same script; repeat). Same node in both jobs (no other placement). Run lines are under `timeout`; no hang, no failure. Analysis: `analyze.py RESULTS 14293 14297` -> `results/2026-10-29-occfuse/b5_epi/analysis.txt` (all tables) and `params.json`. Raw: `results/2026-10-29-occfuse/b5_epi/*_14293.jsonl|.st`, `*_14297.*`.
(The first session was interrupted while waiting for job 14293; the job completed normally. Job 14297 was submitted on resume. Two Slurm jobs used.)

## Method [M]
- Kernel `b5_epi.cu`: 148 CTAs, 512 threads, cooperative launch, mk split barrier (`bar_arrive`/`bar_wait`, B0 harness), 112 phases per launch, 2 warm-up + 7 timed launches (pass-to-pass spread is in analysis.txt).
- Per phase: thread 32 issues one K=64 fill (4 tcgen05 MMAs M128 x TT, smem operands prefilled in SW128 layout, values depend on the CTA) and commits to `accf`. Epilogue warps (4, 8 or 12; q = w & 3, grp = (w>>2)-1, G = nw/4) wait `accf`, then run the mode's epilogue. Chunk map as mk: non-pair modes: chunks grp, grp+G, two loads in flight, stride 2G (mk: G = 3); pair mode: one chunk per iteration, stride G. Chunks with column >= T are skipped, tokens >= T are not stored (warp-uniform, as mk).
- Modes: 1 plain fp32 store (token-major, row contiguous); 2 `red.add.f32` (CTAs b with the same b/KS add to the same tile address, all 148 CTAs run); 3 bf16 store; 4 GU pair (g warp q<2, u warp q>=2, smem exchange with 64-thread named barriers, silu(r g) * (r u), bf16 store); 5 argmax (mk lm epilogue: scale, fp32 logits store, `__reduce_max_sync` + ballot per token, per-quadrant smem best, per-CTA merge into pkey/pidx after the join).
- Event stamps (lane 0 of each epilogue warp, globaltimer, 32 ns steps; means over >= 148 x 111 phases per launch, then median over launches):
  - `loop` = max over warps of end of load/store loop - min over warps of start (start = just after the accfull wait). Includes `tc_fence_before` and `__syncwarp`.
  - `vis` = max over warps of end after `fence.acq_rel.gpu` (mode 4: plus `fence.proxy.async.global`) - same start. This is "written and visible" with every writer fencing.
  - `join` = end of the join `__syncthreads` (mode 5: after the merge and its sync).
  - `dT` = Tmean(config) - Tmean(orc 1, same TT): in-situ increment of the phase time from the barrier stamps. Null phase Tmean 1.93 (TT 16) to 2.01 (TT 128) us.
- Correctness [M]: `k_check` (CUDA-core reference, same bf16 operands, fp32 accumulate) on the last launch of every config; mode 2 expects nph x sum over the KS members. 690 configs, 650 M values checked, 0 failures (max error / tolerance 0.24). Oracle rows (orc 1 no epilogue, orc 2 no stores) are not checked and show pass = 0 by design; they are excluded from fits.
- Sweep: modes 1-5, TT 16/32/48/64/96/128, T in {1, 8, 16, TT} (<= TT), nw 4/8/12; mode 2 also KS 1/2/4/8/12/24.

## Deviations from METHODOLOGY
- No weight stream and no window prefetch (epilogue-only bench); accumulator filled by one K=64 MMA group. Rule 3 stamps are taken, but `T` is a barrier-to-barrier time that includes MMA fill + epilogue + join.
- KS with 148 CTAs: groups of KS consecutive CTAs; the last group is partial when 148 % KS != 0 (KS 8, 12, 24). Metrics use only full groups (144 CTAs for those KS, 148 for 1, 2, 4).
- Mode 5 excludes the cross-CTA ticket / final reduction (that is B6-like). Mode 4 per-token scale r comes from smem (mk loads it in a setup step while the MMAs run, outside the epilogue time). Mode 4/5 output leading dimension 1024.
- Clock: 1850 MHz (TT 16) to 1800 MHz (TT 128, clk_sample); no lock (B0 rule). Times are in us, not cycles.
- `.st` files only for T = TT, nw 12 configs (size).

## Results [M]
Full tables (all TT, T, nw): analysis.txt. Excerpt, us, KS = 1 (jobs 14293 + 14297 mean), nw 4 / 8 / 12:

| mode | TT/T | loop | vis | dT |
|---|---|---|---|---|
| 1 fp32 | 16/8 | .35/.36/.37 | .77/.77/.79 | .30/.29/.30 |
| 1 fp32 | 64/64 | .98/.77/.78 | 1.52/1.41/1.58 | 1.06/.95/1.13 |
| 1 fp32 | 128/128 | 1.93/1.49/1.50 | 2.48/2.31/2.12 | 2.02/1.91/1.73 |
| 3 bf16 | 16/8 | .25/.25/.25 | .66/.72/.73 | .19/.25/.25 |
| 3 bf16 | 128/128 | 1.71/1.16/1.10 | 2.19/1.66/1.59 | 1.73/1.20/1.17 |
| 4 pair | 16/8 | 1.27/1.28/1.30 | 1.69/1.70/1.71 | 1.25/1.26/1.28 |
| 4 pair | 16/16 | 1.93/1.94/1.96 | 2.35/2.36/2.38 | 1.91/1.92/1.94 |
| 4 pair | 64/64 | 7.46/4.14/4.19 | 7.87/4.58/4.61 | 7.39/4.20/4.21 |
| 4 pair | 128/128 | 14.6/8.05/6.35 | 15.0/8.48/6.83 | 14.5/8.11/6.43 |
| 5 argmax | 16/8 | 1.18/1.19/1.21 | 1.59/1.60/1.62 | 1.27/1.28/1.30 |
| 5 argmax | 128/128 | 12.9/6.83/5.46 | 13.4/7.33/5.87 | 13.0/6.96/5.52 |

Mode 2 (red.add, nw 12), vis / dT by KS (us):

| TT/T | KS 1 | 2 | 4 | 8 | 12 | 24 |
|---|---|---|---|---|---|---|
| 16/8 vis | .84 | .84 | .84 | .84 | .84 | .87 |
| 64/64 vis | 1.53 | 1.55 | 1.57 | 1.63 | 1.69 | 1.79 |
| 64/64 dT | 1.08 | 1.12 | 1.16 | 1.21 | 1.26 | 1.41 |
| 128/128 vis | 2.21 | 2.22 | 2.27 | 2.31 | 2.48 | 2.62 |
| 128/128 dT | 1.83 | 1.88 | 1.94 | 1.99 | 2.23 | 2.57 |

- [M] `loop` of mode 2 does not depend on KS (red.add is posted); contention shows only in `vis` (fence waits for the reds) and in `dT`. For T < 64 the KS effect is <= 0.06 us (KS 24); for T >= 64, +0.20 us (vis) / +0.30 us (dT) at KS 24.
- [M] At T <= 16 the loop is one chunk per group: nw 4, 8, 12 give the same time (+0.01 us per added group). Larger T: nw 4 -> 8 is the big step (iterations halve); nw 8 -> 12 gives -0 to -20 % at T = TT >= 96 and nothing at TT 64.
- [M] vis - loop = 0.32-0.45 us at small T (fence.acq_rel.gpu after the stores), 0.4-0.7 at T = TT >= 64.
- [M] Mode 4 and 5 cost about 0.07-0.08 us per valid token in the slowest group (silu with `__expf` and a divide; for mode 5 a warp reduce and ballot per token), plus 0.4-0.6 us per chunk iteration. Modes 1/3: 0.01 us per token and 0.26-0.32 us per iteration. Mode 4 with T = 1 costs 0.70 and T = 8 costs 1.27 (same chunk).
- [M] dT = -0.41 + 1.00 vis (rmse 0.06, modes 1, 3, 4, 5): the in-situ phase increment equals the vis-time minus 0.4 us. (The barrier's `red.release` replaces the per-warp fence.)
- [M] Oracle orc 2 (TMEM reads kept, one never-taken store): loop is 0.60 at TT 16 (0.44-0.46 at TT 64/128) for T <= 16, above the plain-store loop (0.36-0.42). The oracle changes the code (the dead store keeps v[0] live), so it is not a lower bound for the store cost. Not used in fits.

## Fit [M]
OLS over all (TT, T, nw) at KS = 1; n_iter = max over groups of loop iterations (mode 4: chunks, others: ceil(chunks/2) as the kernel loop), n_tok = max over groups of valid tokens, T = valid tokens. `t = a + b n_iter + c n_tok + d T` (us):

| term | a | b | c | d | rmse | max res |
|---|---|---|---|---|---|---|
| m1 loop | 0.011 | 0.321 | -0.0033 | 0.0081 | 0.030 | 0.107 |
| m1 vis | 0.656 | 0.047 | 0.0018 | 0.0107 | 0.063 | 0.140 |
| m2 loop (KS-free) | -0.092 | 0.408 | 0.0096 | 0.0036 | 0.048 | 0.095 |
| m2 vis (KS 1) | 0.490 | 0.178 | 0.0143 | 0.0053 | 0.075 | 0.161 |
| m3 loop | -0.046 | 0.260 | 0.0003 | 0.0051 | 0.030 | 0.081 |
| m3 vis | 0.513 | 0.105 | 0.0037 | 0.0057 | 0.051 | 0.161 |
| m4 loop | 0.161 | 0.489 | 0.0727 | 0.0100 | 0.027 | 0.086 |
| m4 vis | 0.598 | 0.399 | 0.0769 | 0.0110 | 0.037 | 0.157 |
| m5 loop | -0.067 | 0.607 | 0.0763 | 0.0061 | 0.091 | 0.208 |
| m5 vis | 0.392 | 0.522 | 0.0792 | 0.0061 | 0.080 | 0.166 |

KS term of mode 2: vis(KS) - vis(1) = 0.0008 (KS-1) + 0.00009 (KS-1) T (rmse 0.034); dT(KS) - dT(1) = 0.0005 (KS-1) + 0.00014 (KS-1) T (rmse 0.048). The linear model is a summary; the tables in analysis.txt are the data. Max residuals (0.1-0.2 us) are at T = TT with nw 4.

## Gate [H]: mode 4, TT 16, T 8 vs mk GU compute
- mk number, recomputed from st_14155 with `probe/mk_core/smallt/analyze.py` (b8, TS engine, GU busy work): full 5.95 - T_null 2.62 = 3.33 us (includes MMA, B, A stream wait beyond null, epilogue).
- This bench, mode 4 TT 16 T 8 (nw 4 / 8 / 12): loop 1.27 / 1.28 / 1.30, vis 1.69 / 1.70 / 1.71, join 1.78 / 1.79 / 1.81, dT 1.25 / 1.26 / 1.28 us, at 1850 MHz.
- So the epilogue alone is 0.38-0.51 of the 3.33 us (by loop / vis / dT: 1.28/3.33 = 0.38, 1.70/3.33 = 0.51, 1.26/3.33 = 0.38). [H] The remaining ~1.6-2.1 us (MMA drain after the last step, B wait, per-token setup and others) is not measured here.

## Caveats
- Both jobs ran on one node (cayenne). Pass-to-pass spread over configs: loop <= 0.04, vis <= 0.17, Tmean <= 0.29 us (worst case).
- The mean over epilogue events at 32 ns timer steps is unbiased but each event has +-32 ns jitter; differences below 0.05 us are noise.
- Epilogue starts at the accfull wait with a cold-start-free TMEM (the fill just completed); mk's epilogue of tile ti overlaps the MMAs of tile ti+1 (double accumulators); here no overlap.
- The CTA operands differ, so every CTA stores different values; red.add targets repeat across phases (L2-resident, as decode).
- No mk-with-exact-codegen check: lambdas / templates are rewritten, not copied; registers 104-106, no spills.
