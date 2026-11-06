# B2 b2_prefetch: bytes hidden by the barrier-window prefetch
Jobs (priority, 1 GPU, B200, node cayenne, 1965 MHz idle clock; clk_sample 1962-1966 MHz in every row; nothing here runs tcgen05): 14287 (pilot: gate x2 + small sweep), 14289 (full sweep + gate + diagnostics), 14294 (repeat: gate, gate with R override, noB, small sweep).
Raw: results/2026-10-29-occfuse/b2_prefetch/ (`<tag>_<job>.jsonl`, `.st` for gate rows, analysis_14289_14294.txt, fit_hidden.json). Analysis: `analyze.py RESULTS gate3:14289 gate4:14294 gen_c:14289 gen_t:14289`.

## Method
- `b2_prefetch.cu`, built on b1_stream (copied). Grid 148, 512 threads, cooperative launch, ub.cuh split barrier, 100 phases/launch (2 warm-up + 5 timed launches). Stamps as B0.
- Phases alternate: odd = spin phase (all CTAs spin W; CTA 0 spins W + spread); even = stream phase (the measured one). In the barrier window of the spin phase (after `bar_arrive`, before `bar_wait`, as mk MK_PFA 1) thread 0 issues A of the first min(P, R, nsteps) slots of the next stream phase. P = 0 issues nothing.
- Stream phase = mk MK_NULL structure: thread 0 issues the not-yet-issued first R steps, then waits `full[j]` and refills step j+R at once; no MMA, no epilogue. Classic (mode 0): slot = A 32 KB (2 boxes 64 k x 128 rows), full count 1, ring position continues across phases, `__syncthreads` per step (as mk NUL gemm_phase). TS (mode 1): slot = A 32 KB + B (TT 16 rows x 128 k bf16, 2 boxes from an L2-resident 512 KB buffer), full count 2, mbarriers re-initialised in the window (mk `ts_first`), B boxes issued only after exit and `fence.proxy.async.global` (mk ts_phase).
- Per-CTA work: CTA c has floor((c+1)tot/148) - floor(c tot/148) tiles of `spt` steps (mk geom / ts_geom). Generic sweep: tot = N, spt = S/32 KB. Gate: classic QKV/GU/D tot 256/384/192, spt 1, R 2 (mk RING); TS QKV/GU/D tot 128/48/96, spt 2/8/2, R 5 (ts_ringn(16): (227 KB - 11280 - 2768 stamps - 1024 - 384 - 12288 exchange)/36864 = 5; computed from the source, mk was not run).
- Time: busy = mean over stream phases of the median over CTAs with tiles of arrive(p) - exit(p-1), exactly mk's definition (the window prefetch precedes exit(p-1), so it is not in it). win = same median of exit(p-1) - arrive(p-1) of the spin barrier = time from the issue (stamp is at arrive) to the exit. Floor = same config with 0 steps: busy 0.13-0.16 us (`exposed` = busy - floor).
- A from the 2 GiB Pool, a new window per (phase, CTA); the A of a phase is never in L2 (except the `--hot` diagnostic).
- No values are written; arrival of each `full` implies the bytes landed. No FAIL rows possible (same as B1). Every run under `timeout`; no hang occurred.
- Targets recomputed from st_14155 with the smallt/analyze.py definitions (C_null / T_null, b8, layers 4..23): classic QKV 0.69, GU 1.48, D 0.49; TS QKV 0.77, GU 2.64, D 0.83. They agree with the brief within 0.03. [M]
- mk windows (median over busy CTAs of exit - arrive of the preceding barrier, same runs): classic QKV 1.57, GU 1.14, D 1.80; TS QKV 1.60, GU 1.30, D 2.64. Preceding-phase T (null runs): classic 2.06 / 1.50 / 3.50, TS 2.19 / 1.37 / 4.21. [M]

## Deviations from METHODOLOGY.md
- **W does not set the window.** A spin of W on all CTAs only shifts every arrival equally: win stays 1.0-1.1 us (the barrier latency) and busy does not change (median |d busy| 0.006, max 0.078 us over 5200 pairs, W 0.5-6 vs 0). The window is set by the late CTA: win = 1.1 at spread 0 and 0.5, 1.67 at 1, 2.61 at 2, 4.58 at 4, 6.6 at 6 (the late CTA is active, so its own prefetch has a short window; the median ignores it). So "which W reproduces the target" is answered with the measured window `win`, and the spread values used to reach it. mk's preceding-phase T (2.06 ... 4.21) as an all-CTA W has no effect. [M]
- Spread swept {0, 0.5, 1, 2, 4, 6} (4 and 6 added to reach mk-like windows); W x spread crossed for {0, 0.5, 1, 2, 4, 6} x {0, 1, 2}. R 2/3/4 classic, 2/3/4/5 TS; N 48/96/128/148; S 32-1024 KB (10 points); P {0, 1, 2, R}. 21840 rows in job 14289. N = 148 uses the late CTA among the busy CTAs.
- Diagnostic arms added (not in the spec): `--noB` (TS without the B boxes, full count 1), `--hot` (A window repeats every phase: A hits L2), `--gR` (R override at the gate configs).
- Raw stamps stored for gate rows only (W in {0, 2}, spread in {0, 1, 2}).

## Results [M]
Busy work, us, P = R unless stated; classic = mode 0 gate configs, t* = TS (TT 16, R 5). Values are the mean over 4 passes (jobs 14287 x2, 14289, 14294; spread across passes <= 0.06 us) for A+B and 2 passes (14289, 14294) for noB.

| Config | target | P0 (no prefetch), win 1.0 | spread 0 (win) | spread 1 (win) | spread 2 (win) | mk window |
|---|---|---|---|---|---|---|
| cQKV | 0.69 | 2.20 | 1.08 (1.32) | 0.83 (1.90) | 0.82 (2.59) | 1.57 |
| cGU | 1.48 | 2.76 | 1.52 (1.50) | 1.38 (1.93) | 1.44 (2.65) | 1.14 |
| cD | 0.49 | 1.85 | 0.89 (1.14) | 0.70 (1.78) | 0.69 (2.62) | 1.80 |
| tQKV A+B | 0.77 | 2.35 | 1.18 (1.57) | 1.08 (1.93) | 1.08 (2.66) | 1.60 |
| tGU A+B | 2.64 | 4.44 | 3.33 (2.08) | 3.32 (1.98) | 3.33 (2.64) | 1.30 |
| tD A+B | 0.83 | 2.13 | 1.09 (1.45) | 1.08 (1.79) | 1.07 (2.65) | 2.64 |
| tQKV A only | 0.77 | 2.38 | 1.05 (1.57) | 0.88 (1.88) | 0.82 (2.63) | 1.60 |
| tGU A only | 2.64 | 3.72 | 2.62 (2.07) | 2.61 (1.98) | 2.63 (2.65) | 1.30 |
| tD A only | 0.83 | 2.12 | 0.93 (1.45) | 0.84 (1.78) | 0.82 (2.61) | 2.64 |

(The win column of a row at spread 0/1 differs between configs because the TMA issue and the barrier timing depend on the config; it is measured, not set.)

Gate (busy work, +-0.3 us, at the window nearest to the mk window; P = R):

| Target | mk | bench | verdict |
|---|---|---|---|
| classic QKV (148, 2 units, R 2) | 0.69 | 0.83 at win 1.9 (0.95 interpolated at mk's 1.57) | PASS (+0.14 to +0.26) |
| classic GU (148, 3 units, R 2) | 1.48 | 1.52 at win 1.5 (1.38 at 1.9) | PASS (+0.04) |
| classic D (148, 2 units, R 2) | 0.49 | 0.70 at win 1.8 | PASS (+0.21) |
| TS QKV (128, 2 steps, R 5, with B) | 0.77 | 1.08 at win 1.9-2.7 | FAIL (+0.31); A only 0.82-0.88 PASS |
| TS GU (48, 8 steps, R 5) | 2.64 | 3.33 at win 2.0-2.6 | FAIL (+0.69); A only 2.62 PASS (-0.02) |
| TS D (96, 2 steps, R 5) | 0.83 | 1.07 at win 2.6 | PASS (+0.24); A only 0.82 |

- Window dependence: busy falls as win goes from 1.1 to about 1.7-1.9 us, then is flat (cQKV P2: 1.10 / 0.91 / 0.86 for win 1.1 / 1.5 / 1.8+). P1 (one slot) gains less than P2 at every window.
- TS ring depth at the gate (jobs 14294 gateR, A+B): tGU R 2/3/4/5 P = R: 4.44 / 3.77 / 3.45 / 3.33 (spread 0); classic and tQKV/tD are R-independent for R >= 2 (all steps <= 3).
- Diagnostics (job 14289, 14294): `hot` (A in L2) leaves tGU at 3.28 and tQKV/tD at 1.07 (A+B); `hot` + noB: tGU 2.58, tQKV/tD 0.83. So in the bench the B boxes add 0.25 (QKV), 0.70 (GU), 0.25 (D) us to the TS busy, independent of A being in HBM or L2. The A-only TS numbers match the mk TS targets (0.82 / 2.62 / 0.82) within 0.05. [M]  Why mk's TS busy does not show the B cost is untested here [H: B boxes complete earlier in mk than in this bench, e.g. different xb placement or issue order].
- Non-monotonic window effect at N 148 (classic, R 4, S 256, P 4): busy 3.84 (win 1.0), 4.12 (2.6), 4.56 (4.6-6.6). Same at S 1024 (19.15 / 19.29 / 19.65). N 48: no effect (2.93 -> 2.97). [M] Cause not tested [H: with the window shorter than the prefetch transfer (19 MB at 7 TB/s = 2.7 us), the stream continues without a pipeline restart; a fully landed prefetch leaves a restart latency].
- Comparison with B1 (no prefetch, separate consumer thread): P0 busy at N 148 matches B1 t_stream (R 4: S 256 6.38 vs 6.25; S 1024 21.77 vs 22.03; within 2 %). At N 48 it is slower (S 1024: 12.99 vs 11.24; S 256: 3.78 vs 3.55) [M]. The B2 structure (one thread waits full and refills) differs from B1 (producer waits empty, consumer thread frees).

## Fit [M] (job 14289; 21840 rows)
hidden = busy(P=0) - busy(P), regressed on x = P * 32 KB / min(r_sm(R), BW_agg(R)/N) (B1 R-dependent parameters), per window class; all N, S, R, spreads and W:

| mode | window class | slope | intercept us | rmse us | slope through origin | n |
|---|---|---|---|---|---|---|
| classic | < 1.3 | 0.41 | 0.36 | 0.16 | 0.83 | 1197 |
| classic | 1.3-2.0 | 0.71 | 0.28 | 0.21 | 0.93 | 2575 |
| classic | >= 2.0 | 0.83 | 0.12 | 0.20 | 0.91 | 2948 |
| TS TT 16 | < 1.3 | 0.32 | 0.39 | 0.16 | 0.73 | 1517 |
| TS TT 16 | 1.3-2.0 | 0.51 | 0.34 | 0.21 | 0.78 | 3422 |
| TS TT 16 | >= 2.0 | 0.64 | 0.15 | 0.21 | 0.73 | 4301 |

So a slot prefetched in a window >= 2 us hides about 0.9 (classic) / 0.7 (TS) of its transfer time at the contended per-CTA rate (0.65 us per 32 KB at N 148, 0.32 us at N 48), with rmse 0.2 us; the TS value includes the B boxes issued after exit. The linear form does not capture the non-monotonic window effect above. exposed = busy - floor (0.15 us).

## Caveats
- One node (cayenne) for all jobs; "second job" repeats are on the same node.
- The spread is one late CTA (CTA 0, active when N = 148); mk's arrival spread is a distribution. The window is matched on the median over busy CTAs only.
- TS R = 5 and the mk windows come from reading the source / st_14155, not from running mk.
- The first-slot cost of the TS re-init (4 mbar_init + fence in the window) is in the window; not separated.
- Open: B-box cost vs mk (above); non-monotonic window effect at N 148; N 48 P0 gap vs B1.
