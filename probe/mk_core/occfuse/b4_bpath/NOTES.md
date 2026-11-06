# B4 b4_bpath: getting the token operand B into smem

Jobs: 14284 and 14286 (priority, 1 x B200, both: 2 passes each, all arms; 36 s and 40 s). Raw JSONL / stamps: `results/2026-10-29-occfuse/b4_bpath/`. Full analyzer output: `analysis_14284_14286.txt` there. Tags [M] measured, [H] hypothesis.

## Method
- Geometry/phases/barrier/stamps per METHODOLOGY (148 CTAs x 512 thr, cooperative, `bar_arrive`/`bar_wait`, nph 112, 2 warm-up + 7 timed launches per config). One bench phase = B delivery of T rows x kb into smem, then the split barrier (no window prefetch: nothing is prefetched across the barrier except the A stream, see Loaded).
- Arms: z = no B work (baseline); a1 = staged per k-step of 128 k (all 512 threads: `__ldcg` fp32 x as 2 x float4, `rsqrtf(ss)`, ln weight, bf16 pack, SW128 `st.shared`, `fence.proxy.async`, `__syncthreads`, per step); a2 = staged, one batch for the whole kb (STG = 4 chunks of loads in flight per thread, then 4 stores; one fence + sync at the end); b = TMA of a bf16 copy (all boxes of 64 k x TT rows issued at once, tensor map rows = T so rows T..TT are zero-filled by OOB, which gives the zeroed pad rows); c = TMA of the fp32 copy (the same x buffer viewed as bf16 pairs, boxes of 64 x TT, 2x the boxes of b, bytes only).
- Pad rows (T..TT) are stored as zero in a1/a2 (as mk).
- Metric: t_B = mean over CTAs and phases 2..112 of (arrive stamp - previous exit stamp) for the arm, minus the same for arm z at the same (T, kb, loaded) (z work = 0.17-0.18 us). Telescoping `Tmean` of the phase is also recorded (`us_Tmean`); median-per-phase `us_T`/`us_work` are on the 0.256 us grid and are not used for the tables.
- Loaded: thread 0 keeps a 2-slot ring (R = 2) of 32 KB A boxes (2 x 64 k x 128 rows from the 2 GiB Pool, new rows every issue, all 148 CTAs, so about 9.7 MB in flight GPU-wide) in smem slots separate from the B tile. Thread 0 never blocks on it: it re-arms a finished slot at `poll()` points: after each k-step (a1), after each load group (a2), in the TMA-wait spin (b, c). During the barrier wait nobody polls, so the ring idles there. A is never consumed (B1-like, no MMA).
- Smem: A ring 64 KB + B region 160 KB fixed in all arms. When the kb x TT tile does not fit (`nslots` < kb/64, JSONL field), sub-box j goes to slot j mod nslots (loads, stores and TMA bytes are as without wrapping; only the destination aliases). The wrapping points are (T, kb) = (128, 1024), (256, 512), (256, 1024) (a1/a2/b; c wraps more); correctness is checked only where nslots = nbox.
- x (fp32 [256][1024], 1 MB), ss, ln weight, the bf16 copy (0.5 MB) are written once and are L2-resident by being read by all CTAs every phase; they are not rewritten each phase.
- Correctness [M]: before timing, a 2-phase launch with a dump of the last CTA's B region is compared bit-exactly (all T x kb bf16, including zero pad rows and the SW128 swizzle position) to a CUDA-core reference (`k_ref`, which also produces the bf16 copy for b), for a1, a2, b whenever no wrapping. 600 of 600 checked rows PASS (300 per job), 0 FAIL. The reference was in turn checked against a host double computation (max rel err 3.9e-3, bf16 rounding). Arm c is not checked (bytes only).
- Per invocation one (loaded, mode) so a hang loses one arm (none occurred).

## Deviations from METHODOLOGY
- No prefetch-off/on arms (`--pf`): B4 has no per-phase weight prefetch; the A stream is the weight-side load.
- A stream polls only at B-path points (above), it does not saturate HBM continuously. [H] its effective rate was not measured here.
- c: fp32 TMA as bf16 pairs (2x boxes of 64 x TT) not an fp32 map.
- a2 groups loads by STG = 4 chunks per thread (register cap), not all loads at once; it has one fence + `__syncthreads`.
- stamps stored for 1 timed launch per config.
- Clock: fixed at 1965 MHz NVML before; clk_sample medians {a: 1958-1960, others 1964-1965} (no power cap here). No clock lock.

## Results: t_B in us (mean of 2 passes x 2 jobs; passes agree within 0.34 us, typically < 0.1)
```
==== loaded=0: t_B (us) = work(mode) - work(z); mean over passes/jobs; rows T, cols kb
-- a1 staged/k-step
T\kb      128      256      512     1024
1         0.60     1.24     2.46     4.91
8         0.66     1.33     2.61     5.15
16        0.79     1.53     2.92     5.70
32        1.09     2.12     4.05     7.78
64        1.76     3.43     6.53    12.40
128       2.90     5.45     9.91    18.49
256       4.68     8.81    16.51    31.68
-- a2 staged/batch
T\kb      128      256      512     1024
1         0.56     0.61     0.72     1.00
8         0.62     0.77     1.10     1.79
16        0.75     1.02     1.68     2.86
32        1.02     1.60     2.83     5.26
64        1.68     2.93     5.27     9.80
128       2.82     5.52    10.06    19.52
256       4.58     9.51    18.56    37.33
-- b TMA bf16
T\kb      128      256      512     1024
1         0.49     0.65     0.98     1.56
8         0.45     0.61     0.90     1.47
16        0.45     0.61     0.90     1.47
32        0.45     0.61     0.90     1.47
64        0.45     0.61     0.90     1.47
128       0.58     0.75     1.04     1.62
256       0.68     0.98     1.55     2.70
-- c TMA fp32
T\kb      128      256      512     1024
1         0.65     0.98     1.57     2.67
8         0.61     0.90     1.47     2.61
16        0.61     0.90     1.47     2.61
32        0.61     0.90     1.47     2.61
64        0.62     0.91     1.49     2.62
128       0.75     1.04     1.63     2.78
256       0.99     1.55     2.69     4.97
z baseline work (us): T1/kb128=0.18, T8/kb128=0.18, T16/kb128=0.18, T32/kb128=0.18, T64/kb128=0.18, T128/kb128=0.18, T256/kb128=0.18

==== loaded=1: t_B (us) = work(mode) - work(z); mean over passes/jobs; rows T, cols kb
-- a1 staged/k-step
T\kb      128      256      512     1024
1         0.99     2.90     5.31     9.90
8         1.03     2.93     5.35     9.96
16        1.16     2.95     5.30    10.02
32        1.45     3.16     5.59    10.49
64        2.08     4.53     8.91    20.18
128       3.21     6.49    12.19    29.94
256       5.00     9.77    18.37    50.22
-- a2 staged/batch
T\kb      128      256      512     1024
1         0.94     0.99     1.09     1.33
8         0.98     1.09     1.43     2.09
16        1.08     1.32     2.01     3.16
32        1.34     1.93     3.15     6.47
64        2.01     3.22     6.57    12.64
128       3.10     6.72    12.90    28.13
256       5.77    12.04    25.39    61.28
-- b TMA bf16
T\kb      128      256      512     1024
1         0.95     1.11     1.40     1.96
8         0.95     1.11     1.40     1.96
16        0.95     1.11     1.39     1.96
32        0.95     1.11     1.40     1.97
64        0.95     1.11     1.40     1.97
128       0.95     1.12     1.40     1.98
256       0.96     1.15     1.71     2.87
-- c TMA fp32
T\kb      128      256      512     1024
1         1.11     1.39     1.96     3.10
8         1.11     1.40     1.97     3.10
16        1.11     1.39     1.97     3.10
32        1.11     1.40     1.97     3.11
64        1.11     1.40     1.97     3.11
128       1.12     1.40     1.98     3.13
256       1.15     1.71     2.87     5.19
z baseline work (us): T1/kb128=0.18, T8/kb128=0.18, T16/kb128=0.18, T32/kb128=0.18, T64/kb128=0.18, T128/kb128=0.18, T256/kb128=0.18

```
Gate/gap tables:
```
==== T<=16 gap (staged - TMA bf16), us
loaded=0 T=1: kb128: a1-b=0.12 a2-b=0.07  kb256: a1-b=0.59 a2-b=-0.05  kb512: a1-b=1.48 a2-b=-0.27  kb1024: a1-b=3.35 a2-b=-0.56
loaded=0 T=8: kb128: a1-b=0.21 a2-b=0.18  kb256: a1-b=0.72 a2-b=0.16  kb512: a1-b=1.71 a2-b=0.20  kb1024: a1-b=3.68 a2-b=0.32
loaded=0 T=16: kb128: a1-b=0.34 a2-b=0.30  kb256: a1-b=0.91 a2-b=0.40  kb512: a1-b=2.03 a2-b=0.78  kb1024: a1-b=4.24 a2-b=1.40
loaded=1 T=1: kb128: a1-b=0.04 a2-b=-0.01  kb256: a1-b=1.79 a2-b=-0.12  kb512: a1-b=3.92 a2-b=-0.31  kb1024: a1-b=7.93 a2-b=-0.63
loaded=1 T=8: kb128: a1-b=0.08 a2-b=0.03  kb256: a1-b=1.82 a2-b=-0.03  kb512: a1-b=3.95 a2-b=0.03  kb1024: a1-b=8.00 a2-b=0.13
loaded=1 T=16: kb128: a1-b=0.21 a2-b=0.13  kb256: a1-b=1.84 a2-b=0.21  kb512: a1-b=3.91 a2-b=0.61  kb1024: a1-b=8.06 a2-b=1.20

```
Spread between the 4 repeats of every point: max 0.34 us (a2, T 256, kb 1024, idle), see `analysis_14284_14286.txt`.
==== clocks: mhz median per arm/loaded
z loaded=0: mhz med 1965 min 1964 max 1965
z loaded=1: mhz med 1964 min 1963 max 1965
a1 loaded=0: mhz med 1959 min 1948 max 1963
a1 loaded=1: mhz med 1960 min 1952 max 1964
a2 loaded=0: mhz med 1957 min 1946 max 1964
a2 loaded=1: mhz med 1959 min 1950 max 1964
b loaded=0: mhz med 1964 min 1950 max 1965
b loaded=1: mhz med 1963 min 1952 max 1964
c loaded=0: mhz med 1963 min 1951 max 1965
c loaded=1: mhz med 1963 min 1950 max 1964


`launch/phase vs T differ >5%: 92/1120; worst: [(0.112, 'a2', 1, 256, 0), (0.106, 'a2', 1, 256, 0), (0.102, 'a2', 1, 128, 0), (0.102, 'a2', 1, 128, 0), (0.093, 'a2', 1, 128, 0)]` (launch/phase vs `us_T`; flagged points are mostly a2/c at T >= 128, loaded or idle, the launch wall includes ~2 us launch overhead / nph).

## Fits (job 14284 + 14286, all T and kb, x-axis bytes: a: T*kb*4 fp32 bytes read; b: TT*kb*2 smem tile bytes; c: T*kb*4)
2-term t_B = lat + bytes / thr:

| mode | state | lat us | us per KB | rmse us |
|---|---|---|---|---|
| a1 | idle | 1.92 | 0.0300 | 1.29 |
| a1 | loaded | 2.91 | 0.0443 | 3.24 |
| a2 | idle | 0.55 | 0.0360 | 0.22 |
| a2 | loaded | -0.12 | 0.0564 | 1.29 |
| b | idle | 0.70 | 0.00382 | 0.30 |
| b | loaded | 1.18 | 0.00309 | 0.33 |
| c | idle | 1.12 | 0.00353 | 0.63 |
| c | loaded | 1.58 | 0.00321 | 0.66 |

The 2-term fit is poor for a1 and loaded a2 (the cost is not linear in bytes: loaded a1 jumps between T 32 and 64 at kb 1024, 10.4 to 20.3 us). 3-term t_B = lat + per_unit x (a1: k-steps; b, c: boxes; a2: boxes, coefficient ~0) + us_per_KB x KB (a1/a2/b: TT*kb*2 smem tile bytes incl. pad rows; c: T*kb*4):

| mode | state | lat us | per unit us | us per KB | rmse us |
|---|---|---|---|---|---|
| a1 | idle | 0.37 | 0.44 / k-step | 0.0560 | 0.49 |
| a1 | loaded | -0.69 | 1.09 / k-step | 0.0781 | 1.44 |
| a2 | idle | 0.42 | -0.03 | 0.0736 | 0.44 |
| a2 | loaded | -0.18 | -0.07 | 0.1160 | 1.20 |
| b | idle | 0.35 | 0.060 / box | 0.0024 | 0.094 |
| b | loaded | 0.79 | 0.067 / box | 0.0015 | 0.082 |
| c | idle | 0.37 | 0.062 / box | 0.0022 | 0.20 |
| c | loaded | 0.79 | 0.066 / box | 0.0018 | 0.19 |

(`params.json` holds both sets.) Observations [M]: the TMA modes are latency + per-box cost, nearly flat in T up to 64 (the t_B at T 8-64 is identical to 0.01 us, so rows do not cost anything until 128 / 256 where bytes reach 128 / 256 KB per kb 1024); T = 1 costs 0.03-0.2 us more than T = 8-16 for b/c at idle (not explained). Staged is linear in smem tile bytes (T rows incl. pad, i.e. TT) at roughly 0.03-0.07 us/KB idle: T 16 at kb 1024 is 2.9 (a2) / 5.7 (a1) us, T 256 at kb 1024 is 37 / 32 us. Staged a1 at T <= 32 costs about the same at every T (4.9-7.8 us at kb 1024, dominated by 8 x per-step latency 0.44 us + L2 load latency); a2 is cheaper than a1 at T <= 64 and slower at T >= 128 (STG 4 loads in flight per thread cap).

## Gate [M]
T = 256, idle, kb 1024: a1 31.7, a2 37.3, TMA bf16 2.70 us (ratio 11.7 / 13.8); loaded: a1 50.2, a2 61.3, b 2.87 (17.5 / 21.4). At every kb staged is 5-21 x slower than TMA. Direction of job 9595 (47.6 vs 27.9 us per phase sums, ratio 1.7) holds, the ratio here is far larger because this measures only B delivery. PASS (2 jobs agree within 0.34 us).
T <= 16 gap (staged minus TMA bf16, us), kb 128 (one mk k-step): idle T 1 / 8 / 16: a1 0.12 / 0.22 / 0.34, a2 0.08 / 0.18 / 0.30; loaded 0.04 / 0.08 / 0.21 (a1), -0.01 / 0.03 / 0.14 (a2). kb 1024: idle a1 3.3 / 3.7 / 4.2, a2 -0.56 / 0.32 / 1.39; loaded a1 7.9 / 7.9 / 8.0, a2 -0.63 / 0.13 / 1.20. No target.

## Caveats
- t_B is the in-phase time to the arrive stamp, not an overlap measurement: staged B occupies the same threads as the epilogue/MMA issue in mk (B8 / composition), TMA B does not.
- Loaded numbers depend on how often thread 0 re-arms the A ring; with a1 at T <= 32, kb >= 256 the loaded run costs 1.7-5 us more than idle at the same point although the B work is tiny (poll + A traffic competing for L2/SM ports) [H].
- fp32 copy arm c has 2x boxes of b, not a different TMA type.
- Power capping did not occur (1960-1965 MHz); busy-MMA runs (B0) saw ~1700. Staged arms had the lowest clk_sample (1949-1964).
- The idle pool of 148 CTAs reads the same x lines; per-SM-distinct x not tested.
