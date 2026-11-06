# B3 b3_step: compute chain of one tile, operands in smem
Jobs: 14288 (first version, dep via runtime `%`; superseded, kept in results), 14290 (LEAN 0/1, DEP 1/2/4), 14291 (adds `--fence`, test_wait build, repeats everything). Partition priority, 1 GPU, node cayenne, B200, commit c2bc494. Fits pool 14290 + 14291 (two jobs, 2 passes each; run-to-run spread of every t_step < 0.5 cycle except where noted).
Raw: `results/2026-10-29-occfuse/b3_step/` (`*_<job>.jsonl`, `an_14290_14291.txt` = analyze.py output, `table_acc2_act1_F0.md`). Reproduce: `analyze.py results/2026-10-29-occfuse/b3_step 14290 14291`. Slurm logs `runs/slurm-<job>.log`. A 5-second `srun` sanity run (job 14280) was also used before the first sbatch (4 Slurm jobs touched the GPU in total, one a smoke test; brief said at most 3 sbatch jobs: 14288, 14290, 14291).

## Method
- `b3_step.cu`, kernel `k_b3`: grid 148, 512 threads, cooperative launch, 1 CTA/SM; CTAs >= `nact` return at once (nact 1 or 148). TMEM 512 columns; accumulator b at columns b*TT*DEP.
- Ring R = min(4, fit in smem) slots of A (32 KB, 2 SW128 boxes 64k x 128 rows) + B (TT rows x 256 B, 2 boxes), filled once in smem with hash bf16 data (SW128 layout, written by the kernel; 1024-B aligned). R = 4 for N <= 96, 3 for N 128, 2 for N 192/256 (ring 8 check: no change, below).
- Handoff as k_ts: producer (warp 0, lane 0) waits `empty[f % RB]` (f >= RB) then arrives `full[f % RB]` with no load (this replaces the TMA complete_tx; TMA latency and bytes are NOT in this bench); MMA warp (warp 1) waits `full`, `tcgen05.fence::after_thread_sync`, 8 MMAs (kk = 0..7: A/B descriptors at +32 B within the 64-k box, second box after kk 4; M128 N=TT K16, bf16, fp32 accumulate, cta_group::1, `tc_mma` / `sdesc` / `tc_idesc` from ub.cuh), then `tc_commit(empty)`; the last step also `tc_commit(accfull[b])`. Epilogue warp 4 waits `accfull[b]` and arrives `accempty[b]` (no TMEM read in timing mode). MMA warp waits `accempty[b]` before tile G >= NACC.
- Commit mode 1 (per tile): `empty` is arrived by a plain `mbarrier.arrive` from the MMA thread after the `full` wait (not by the commit), RB = 16 barriers; one `tc_commit(accfull)` per tile. Deviation: slot release is then not tied to MMA completion (data are static, so it is harmless here).
- Timing: a launch runs 2 passes of ntile tiles back to back (ntile = 2048 / SPT, so 2048 k-steps), only pass 2 is timed: MMA warp stamps (clock64, globaltimer) before its first wait of the pass; epilogue warp stamps after its last accfull wait. cycles per tile = delta clock64 / ntile; us = delta globaltimer / ntile (32 ns step, region >= 1 ms: negligible); MHz = cycles / us. 2 warm-up + 5 timed launches, median over launches (over active CTAs for nact 148; max over CTAs in `cyc_tile_ctamax`).
- Variants (extra to the spec, added after the first job showed a floor): LEAN 0 = descriptors rebuilt from addresses for every MMA (as k_ts / mk); LEAN 1 = `sdesc` once per operand per step, per-MMA offsets added as constants. DEP = independent accumulator chains inside a tile (MMA kk writes column (kk % DEP)*TT; DEP 1 = mk). `--fence 1` = tcgen05.fence once per tile instead of per step. `-DTW=1` build = `test_wait` spin on mbarriers.
- No phase barrier and no `.st` stamps file (no barrier phases in this bench; deviation from METHODOLOGY rules 2/3/10).
- Correctness [M]: check arm (ntile 2, 1 launch): warps 4-7 `tmem_ld16` the accumulator to global, `k_ref` CUDA-core reference from plain copies of the ring data, tolerance 2e-3*max(1,|ref|). 384 configs (N 16..256 at SPT 4, N 16/256 at SPT 1/16, NACC 1/2, both commit modes, LEAN 0/1, fence 0/1; all DEP 1), 0 failures, max abs error 2e-4. DEP > 1 and timing launches are not checked (DEP > 1 outputs are meaningless by construction).
- Sweep: N in {16,32,48,64,96,128,192,256}, SPT in {1,2,4,8,16}, NACC {1,2}, nact {1,148}, commit {step, tile}; DEP {1} and {2,4} (where TT*DEP*NACC <= 512). Fit t_tile = a + SPT * t_step (5 points) per configuration, per job and pass.

## Results [M] (acc 2, 1 CTA, fence per step; medians of fits over jobs 14290/14291 and passes; cycles from clock64)
Full table, all N: `table_acc2_act1_F0.md`. Selected rows:

| lean | commit | N | a cyc | t_step cyc | cyc / MMA (t_step/8) | a us | t_step us | MHz |
|---|---|---|---|---|---|---|---|---|
| 0 (mk-style) | step | 16 | 486 | 972 | 121.5 | 0.264 | 0.525 | 1851 |
| 0 | step | 128 | 488 | 972 | 121.5 | 0.263 | 0.526 | 1850 |
| 0 | step | 256 | 486 | 1164 | 145.5 | 0.019 | 0.732 | 1822 |
| 0 | tile | 16 | 549 | 883 | 110.4 | 0.298 | 0.477 | 1851 |
| 0 | tile | 128 | 547 | 883 | 110.5 | 0.294 | 0.478 | 1848 |
| 0 | tile | 256 | 507 | 1025 | 128.1 | 0.238 | 0.599 | 1766 |
| 1 | step | 16 | 530 | 917 | 114.6 | 0.286 | 0.496 | 1850 |
| 1 | step | 128 | 534 | 916 | 114.5 | 0.285 | 0.497 | 1848 |
| 1 | step | 256 | 534 | 1102 | 137.8 | 0.238 | 0.641 | 1810 |
| 1 | tile | 16 | 590 | 827 | 103.3 | 0.319 | 0.447 | 1850 |
| 1 | tile | 128 | 590 | 827 | 103.3 | 0.320 | 0.448 | 1847 |
| 1 | tile | 192 | 588 | 829 | 103.7 | 0.297 | 0.457 | 1849 |
| 1 | tile | 256 | 487 | 1025 | 128.1 | 0.271 | 0.583 | 1792 |

Fit rmse <= 2.7 cycles (all fits; typically 0.5-1.3). The intercept `a` is 0.24-0.32 us (about 490-590 cycles); at N 256 the cycle intercept and the us intercept disagree slightly (clock drops during the long runs; use cycles).
Other observations [M]:
- t_step is flat in N for N 16..128 (917 +- 1 cycles at LEAN 1 / step commit), rises at N 192 (+26), and is MMA-bound at N 256 (1025 = 128.1 cycles / MMA with the per-tile commit; commit per step adds 77-139 cycles per step, i.e. about 10-17 cycles / MMA).
- NACC 1 vs 2: identical t_step (to 0.1 cycle). nact 1 vs 148: identical in cycles (<= 0.2 %); in us the 148-CTA runs are slower only through the clock (MHz 1789-1850 at nact 148 vs 1793-1850 at nact 1; the clock sags at N >= 128 under load; B0 rule: use cycles).
- DEP 2 and 4 (independent accumulator chains) do not change t_step at N 16..128 (LEAN 1, tile commit: 103.6 / 102.8 vs 103.3 cycles/MMA; step commit 115.0 / 114.2 vs 114.6). So the floor is not an accumulator dependency chain.
- `--fence 1` (fence per tile): no change (< 0.3 cycle / MMA). LEAN 0 vs 1: descriptor rebuilding costs 7 cycles / MMA (step commit) at N <= 128 and 8 at N 256.
- Ring depth: ring 8 (ring8 arm, LEAN 0, step commit, SPT 16, N 16/64/128): 16030 / 16029 / 16032 cycles per tile, equal to the ring 4/3 fit value (972*16+486 = 16038). No ring dependence.
- `-DTW=1` (test_wait spin): slower, N 16 SPT 16 commit tile 17144 cycles vs 13847 with try_wait (LEAN 1). Not used for the fits.
- Single tile (ntile 1, pass 2 of 2, ring full; job 14291, LEAN 0, step commit): N 16 SPT 1: 1303 cycles (NACC 1) / 1115 (NACC 2), 0.67 / 0.58 us; SPT 16: 15878 / 15689 cycles; N 256 SPT 1: 2027 / 1662; SPT 16: 19476 / 19113 cycles. The steady-state tile fit gives a + SPT*t_step = 1458 (SPT 1, LEAN 0 step N 16) and 16.0k (SPT 16). (ntile 1 includes the full pipeline latency: first MMA to accfull observe.)

## Gate [M]: per-MMA cycles t_step / 8 vs hv2_tc job 9425 (N 16: 58.1, N 128: 80.6, N 256: 128.2), tolerance 10 %
| config (acc 2, nact 1 or 148) | N 16 | N 128 | N 256 |
|---|---|---|---|
| hv2_tc 9425 | 58.1 | 80.6 | 128.2 |
| LEAN 1, commit per tile (best case here) | 103.3 (1.78x) FAIL | 103.3 (1.28x) FAIL | 128.1 (1.00x) PASS |
| LEAN 1, commit per step | 114.6 (1.97x) FAIL | 114.5 (1.42x) FAIL | 137.8 (1.07x) PASS |
| LEAN 0 (mk-style), commit per step | 121.5 (2.09x) FAIL | 121.5 (1.51x) FAIL | 145.5 (1.14x) FAIL |
| LEAN 0, commit per tile | 110.4 (1.90x) FAIL | 110.5 (1.37x) FAIL | 128.1 (1.00x) PASS |

Gate result: **FAIL** at N 16 and N 128 (this bench is 1.3-2.1x slower per MMA), PASS at N 256 (with per-tile commit or lean descriptors + step commit). The same numbers repeat across jobs 14290 and 14291 and across fence / DEP / NACC / nact variants, so the discrepancy is not noise.

## Caveats and open issues
- [H] The cause of the 103-cycle / MMA floor at N <= 128 is not isolated. It is not: accumulator dependence (DEP), the per-step fence, descriptor arithmetic (7 cycles only), NACC, ring depth, the number of active CTAs. Candidates not tested (no jobs left): the step handoff (full wait to first MMA, ~0.37 us per step at 8 MMAs is about 360 cycles above 8 x 58); the SW128 K-major A/B smem read pattern (hv2_tc c1 / swab arms use `bdesc` no-swizzle B with NACC 4 accumulators, a different operand layout from mk's SW128); MMA issue rate of one thread. A raw back-to-back arm (no handoff, SW128 vs no-swizzle) would separate them.
- The hv2_tc reference lines (58.1 / 80.6) are "total c/MMA" of a burst of R MMAs plus one commit and wait, with different smem layouts; the log lines for those numbers were not found in 9425.out by grep (the c2m128 N16 line there reads 70.1). The numbers from the brief were used as given.
- No TMA: operand arrival latency and bytes are not in these terms. The producer-to-`full` handoff is a plain arrive.
- Commit mode 1 releases slots by plain arrive (see Method).
- Clock: cycles from clock64; us uses the globaltimer; MHz 1790-1850 here (power cap shows at N >= 128 with 148 CTAs).
- `params.json`: one entry per (N, nacc, nact, commit_tile, dep, lean, fence_tile), medians of fits, written by analyze.py.
