# B7 b7_splitk: price of splitting K for a whole phase

Tags: [M] measured (job id), [H] hypothesis. Measurements and fits only.
Job 14301 (priority, node cayenne, B200, 148 SMs, commit c2bc494). Results: `results/2026-10-29-occfuse/b7_splitk/` (`*_14301.jsonl`, QKV raw stamps `*.st`, `analysis_14301.txt` = full analyze output). Slurm log: `runs/slurm-14301.log`.
Earlier srun attempts 14298 (binary path typo, exit 127) and 14299 (cancelled before start) produced no data; they count against the 3-job budget. Total: 3 job ids, 1 with data.

## Method
- `b7_splitk.cu`: xover4 `k_ts` (CG 1: producer thread 0, MMA issuer thread 32, epilogue warps 4-15, ring of A+B slots, 2 TMEM accumulators, tile order (rb, tt, ks), CTA b owns tiles [b*NT/148, (b+1)*NT/148) as mk `ts_geom`) with the common/ harness: `bar_arrive` / `bar_wait` split barrier, per-CTA stamps, 112 phases per launch, 2 warm-up launches, 7 timed launches, A from the 2 GiB pool (new rows every phase), B by TMA from an L2-resident bf16 X, `clk_sample` MHz.
- Ring depth = as many slots as fit (cap 16): TT 16 -> 6, TT 32 -> 5, TT 128 -> 3 (mk's cap 8 is never reached for these TT).
- Window prefetch (pf 1): thread 0 issues A of the next phase's first min(ring, steps) slots between `bar_arrive` and `bar_wait`; B is issued after the barrier (after `fence.proxy.async.global`). Phase 0: A is issued before the start barrier. `--pf 0`: A and B both after the barrier. pf 0 was run at every point, not only the gate.
- Arm r (redadd): KS = 1 plain fp32 store; KS > 1 `red.add.f32` per element into Y [T][N] (Y zeroed before each launch).
- Arm w (ws): epilogue stores plain partials to WS [KS][T][N] fp32; barrier (stamp barrier 2p+1); then one reduce pass by all 148 CTAs x 512 threads (float4, `__ldcg`, sum over KS, plain store to Y); barrier 2p+2. Window prefetch is issued at the end barrier only (not in the middle barrier). KS = 1 not run for ws. WS for KS 24 x T 256 x N 6144 is 151 MB (> L2); the ws reduce reads come from L2 only when KS*T*N*4 is below about the L2 size.
- Shapes QKV (4096x1024), O (1024x2048), GU (6144x1024), D (1024x3072) are `--shapes` arguments; T in {8, 32, 128, 256}; TT = min(T rounded up to 16, 128) (T 256 = 2 token tiles of 128); KS in {1,2,3,4,6,8,12,24} where K/128 % KS == 0 (QKV, O, GU: 1,2,4,8 [O has 16 steps; 16 was not requested]; D: all 8). No act fusion.
- T = `us_Tmean` (telescoping mean of the median-over-CTA exit stamps over phases 2..112, median over 7 launches). Every value below is the median over the passes that measured the point (1 to 4 passes; `us_reps` and min/max are in the JSONL). Stamps are on the 32 ns timer, medians on the 0.256 us grid (B0), so `busy work`/`idle`/`lat` columns are quantised; `us_Tmean` is not.
- For ws rows `us_busy` = GEMM-stage busy work, `us_work2` = reduce-stage work (median over CTAs of arrive minus the previous exit).
- Occupancy quantities from `common/tiles.py ts()` (asserted equal to the kernel's own tile arithmetic in analyze.py): tiles, busy CTAs, tiles_max, steps per tile, max_steps = tiles_max x steps per tile.

## Correctness [M, job 14301]
Every config with orc 0 is checked against the CUDA-core `k_ref` (pool rows 0..N-1, phase 0, 1-phase launch from a zeroed / NaN-filled Y): 454 JSONL rows, 288 non-oracle sweep rows (+ repeats), 0 FAIL (tolerance 2e-3 max(1,|ref|)). All shapes, all T, both arms, both pf. The check covers every (shape, T, KS, arm, pf), more than the one T per shape the rule asks for. Timed launches run 112 phases (red.add values grow, as xover4; Y is re-zeroed per launch).

## Gate [M, job 14301]: QKV T128 TT128, KS4 vs KS8 (arm r, pf 1)
| pass | KS4 | KS8 | KS8 - KS4 |
|---|---|---|---|
| sanity (3 launches) | 6.197 | 7.412 | 1.215 |
| p1 sweep (7 launches) | 6.195 | 7.417 | 1.222 |
| gate2 (11 launches) | 6.199 | 7.424 | 1.225 |
| p2 sweep (7 launches) | 6.199 | 7.419 | 1.220 |

**PASS**: KS 4 is faster than KS 8 by 1.22 us (limit 0.8; job 9592: 6.07 vs 7.46 = 1.39). The four passes are inside 0.01 us of each other. pf 0: 6.395 vs 7.592 = 1.197, PASS. mhz 1818 (KS4) / 1831 (KS8). Repeat = second pass in the same job (same node and clock state), not a second job; the second-job placement check is not done.
Other xover3 9592 points (QKV TT128, `slurm-9592.log`): KS2 6.77 (here 8.20 pf 1 / 7.43 pf 0, +1.43 / +0.66), KS4 6.07 (here 6.20), KS8 7.46 (here 7.42). QKV T256 TT128 KS2: 7.11 there, 8.03 here. QKV T32 TT32 KS2/4/8: 6.25/5.09/5.34 there, 6.69/4.94/5.05 here. So the engine agrees at KS 4 and 8 (<= 0.13 us at T128) and is slower at KS 2 (0.4-1.4 us). [H] cause: xover3/xover4 `k_ts` issues the next phase's A from the producer warp right after its own last step (during the last MMAs and the epilogue), which this bench replaces by the mk barrier-window issue; with 64 busy CTAs (KS2) the window issue is not hidden (see pf 1 vs pf 0 below). Not tested (no job left; an arm `--pf 2` = producer issue after the last step would test it).

## Results [M]
T in us (arm r, pf 1 / pf 0), all points in `analysis_14301.txt`. Format: KS: T (busy CTAs, max steps).

| shape T | KS1 | KS2 | KS3 | KS4 | KS6 | KS8 | KS12 | KS24 |
|---|---|---|---|---|---|---|---|---|
| QKV 8 | 9.51 (32,8) | 6.55 (64,4) | | 4.83 (128,2) | | 5.06 (148,2) | | |
| QKV 32 | 9.49 | 6.69 | | 4.94 | | 5.05 | | |
| QKV 128 | 10.29 | 8.20 | | 6.20 | | 7.42 | | |
| QKV 256 | 10.33 (64,8) | 8.03 (128,4) | | 9.39 (148,4) | | 11.90 (148,4) | | |
| O 8 | 14.28 (8,16) | 9.74 (16,8) | | 6.53 (32,4) | | 4.70 (64,2) | | |
| O 128 | 14.96 | 10.57 | | 7.96 | | 5.82 | | |
| O 256 | 14.79 | 10.51 | | 7.96 | | 6.05 (128,2) | | |
| GU 8 | 9.56 (48,8) | 6.59 (96,4) | | 6.91 (148,4) | | 6.21 (148,3) | | |
| GU 128 | 10.39 | 8.13 | | 9.23 | | 10.24 | | |
| GU 256 | 10.42 (96,8) | 11.52 (148,8) | | 11.61 (148,6) | | 16.25 (148,6) | | |
| D 8 | 19.00 (8,24) | 12.15 (16,12) | 9.79 (24,8) | 8.41 (32,6) | 6.55 (48,4) | 5.82 (64,3) | 4.74 (96,2) | 4.86 (148,2) |
| D 32 | 18.99 | 12.09 | 9.71 | 8.30 | 6.70 | 5.97 | 4.87 | 5.00 |
| D 128 | 19.72 | 12.98 | 10.62 | 9.47 | 8.12 | 7.14 | 5.98 | 7.08 |
| D 256 | 19.54 (16,24) | 12.90 (32,12) | 10.65 (48,8) | 9.49 (64,6) | 7.79 (96,4) | 7.12 (128,3) | 9.15 (148,4) | 9.91 (148,3) |

(QKV/O/GU have K/128 = 8 / 16 / 8 steps, so KS 3, 6, 12, 24 exist only for D; O has KS 16 not run.) Full tables incl. pf 0, busy work, idle, last, lat and the ws arm are in the analysis file.

Observations [M]:
- Best KS per (shape, T) is the one that just fills the grid (busy 96-148, 2-3 steps per CTA) for T <= 128; at T 256 the best is the KS with <= 148 tiles and 1 tile per CTA (QKV KS2, 128 tiles; D KS8, 128; GU KS1, 96 tiles).
- Equal max steps, more pieces (paired differences, arm r, pf 1), e.g. QKV KS4 -> KS8 (2 steps, tiles_max 1 -> 2): T8 +0.23, T32 +0.11, T128 +1.22 (T256, KS2 -> KS4 at 4 steps: +1.36; KS2 -> KS8, tiles_max 1 -> 4: +3.87). D KS12 -> KS24 (2 steps): +0.12 (T8), +0.13 (T32), +1.10 (T128). GU KS2 -> KS4 (4 steps): +0.32, +0.24, +1.10 (T 8, 32, 128). So at T <= 32 an added piece costs 0.1-0.3 us once max steps are equal; at T >= 128 it costs about 1.1-1.4 us per extra 128 x 128 fp32 tile of red.add per CTA, and the cost keeps growing with tiles_max (GU T256 KS4 -> KS8: +4.64 at 3 -> 6 tiles).
- Oracle arms, QKV T128 (orc 2 = epilogue TMEM reads kept, stores off): KS4 6.20 -> 4.50 (-1.70); KS8 7.42 -> 4.69 (-2.73); KS2 8.20 -> 6.51 (-1.69). No-MMA (orc 1): -0.9 (KS4), -0.7 (KS8), -0.8 (KS2). No-B (orc 4): -0.5 / -0.6 / -1.2. No-A (orc 8): -0.7 / -0.7 / -1.7. So at TT 128 the red.add stores are 1.7 us (1 tile per CTA) and 2.7 us (2 tiles per CTA) of the phase; one tile costs about 1.0-1.7 us beyond the stores of the oracle-free floor (single-run oracle points, no repeats).
- pf 1 (window prefetch) minus pf 0 (arm r, median over points by busy CTAs): +0.3 us at busy 8-64 (range -0.03..+1.12), +0.17 at 96, -0.11 at 128, +0.22 at 148 (-0.32..+0.64). So in this bench the window prefetch does not help T at any occupancy and costs about 0.3 us at low occupancy (QKV KS2 T128: 8.20 vs 7.43). In the raw stamps (QKV T128 KS2, pf 1) some busy CTAs leave the barrier up to 1.5 us after the median, with pf 0 about 0.25 us. [H] the poll loads of thread 0 queue behind the incoming prefetch TMA data of that SM. Not tested further.
- ws arm vs r arm (same KS): ws is slower everywhere, +1.3 to +6.1 us; the reduce stage alone (`us_work2`) is 0.5 us at T 8 (QKV) and 1.5 / 2.3 / 3.3 us for QKV T128 KS 2 / 4 / 8 (read bytes 4.2 / 8.4 / 16.8 MB), plus a second barrier.
- launch/nph vs `us_Tmean` differ by median +1.5 % (max 3.3 %, 0 of 442 rows above 5 %). `stamps.metrics` on the raw QKV stamps (last timed launch) agrees with the JSONL (median of launches) within 0.025 us over 40 configs.
- clk_sample MHz 1803-1886 over the rows (r arm 1805-1850 at TT 128; ws rows higher, 1850-1870). NVML before the job: 825 MHz idle with throttle bit 0x1, after: 1965.

## Fit [M data, fit form is a description]
Form (arm r, pf 1, n = 80 points with all KS, shapes, T): occupancy part `a * max_steps` (tiles.py; busy CTAs enters through which KS gives which max_steps) + store part by bytes written per busiest CTA:

T = t0 + a * max_steps + c_red * red_tiles + c_pl * plain_tiles
red_tiles = tiles_max * min(T, TT)/128 for KS > 1 (one unit = a 128 x 128 fp32 tile = 64 KB red.add per CTA); plain_tiles = the same for KS = 1 (plain store).

| form | t0 | a (us/step) | c_red (us/unit) | c_pl (us/unit) | other | rmse | max res |
|---|---|---|---|---|---|---|---|
| steps only | 5.17 | 0.609 | | | | 1.398 | 7.42 |
| F1 | 3.81 | 0.646 | 1.38 | 0.96 | | 0.379 | 0.79 |
| F2 | 4.53 | 0.609 | 1.46 | 0.73 | -0.28 * log2 KS | 0.299 | 0.74 |
| F3 | 3.91 | 0.645 | 1.45 | 0.98 | -0.10 * tiles_max | 0.376 | 0.81 |
| F4 | 4.30 | 0.616 | 1.49 | 1.07 | -0.005 * busy CTAs | 0.337 | 0.87 |
| F1, pf 0 rows | 3.59 | 0.629 | 1.40 | 1.22 | | 0.319 | |

- Steps and store bytes together explain the table to 0.38 us rms; steps alone do not (1.40). t0 3.8 us is the empty-phase floor plus the first tile's fixed chain (barrier 1.15 + ring start + one epilogue; B0 and B3 give 1.16 and a ~0.26 us tile fixed cost).
- a = 0.61-0.65 us/step is above the B3 compute step (0.525 us) [M]; [H] the extra 0.1 comes from the A+B stream at ring 3-6 and the MMA/epilogue handoff (tiles with 8-24 steps: TT 128 ring 3 stream rate 91 GB/s in B1 gives 0.36 us/step, so it is not the stream alone).
- Per-TT fits (TT 16: c_red 0.74, TT 32: 0.19, TT 128: 1.38) are poorly identified because at T <= 32 tiles_max is 1-2 and the store term is small; the unit-based c_red is dominated by the T >= 128 points. At T <= 32 the paired same-steps differences (+0.1 to +0.3 us per piece) are the better number.
- Residuals: the largest (up to 0.8) are the KS with 148 busy CTAs at 2 steps (D KS24 T128 +0.81 over, i.e. model too slow; D KS3 T8 -0.74), no systematic term found among busy CTAs, log2 KS, tiles_max (F2-F4 reduce rmse by 0.04-0.08 only). [H] the 148-busy cases pay barrier arrival spread that F1 does not carry.
- Occupancy `f(max_steps, busy)` as a stream-saturation form (B1 rate min(r_sm, BW/busy), step = max(compute, bytes/rate)) was tried (nonlinear least squares by hand): rmse 0.45-0.95, worse than F1; no BW-saturation term is needed at these sizes (the `sat` regressor fitted ~0).

ws arm (n = 64, pf 1): T = 6.27 + 0.598 * max_steps + 1.525 * partial_tiles + 0.065 * read_MB, rmse 0.43 (max 1.45) (partial_tiles = tiles_max * min(T,TT)/128 for plain stores; read_MB = KS*T*N*4/1e6). Reduce-stage work alone = 0.62 + 0.134 us/MB (rmse 0.41). The ws t0 is 2.5 us above F1 (second barrier 1.15 + reduce pass fixed cost). params.json has all coefficients.

## Deviations / caveats
- Repeat of the gate was a second pass inside job 14301 (plus a third and fourth sample: sanity, gate2 with 11 launches), not a second job; both in one node/clock state. Slurm budget (3 jobs) was used up by two failed submissions (14298: path typo, 14299: cancelled) plus 14301.
- NSTMAX = 512 (default in common/ub.cuh); ws runs use 2 barriers per phase.
- ws arm has no window prefetch at the middle barrier (design choice; not measured).
- Phase 0 A prefetch is issued before the start barrier; B of phase 0 after it.
- Ring depth is the xover4 rule (cap 16), not mk's cap 8; equal for these TT.
- Oracle arms are single runs of QKV T128 KS 2/4/8 (7 launches), pf 1 only.
- `us_idle` is -1 when there are no idle CTAs (busy 148).
- Stamp medians are on the 0.256 us grid; differences < 0.25 us in busy/idle/last columns are not resolved; `us_Tmean` is.
- WS for large KS*T*N exceeds L2 (not tracked per point).
- KS 2 does not reproduce xover3 9592 (see Gate section).
