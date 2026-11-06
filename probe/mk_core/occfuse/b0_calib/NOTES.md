# B0 b0_calib: harness check
Jobs: 14277 and 14278 (priority, 1 GPU, both on node cayenne, different GPUs; commit c2bc494 worktree). Each job: two identical passes of arms gtstep + a + b (`run1`, `run2`), then clock arms `c_idle` / `c_busy` twice each (6 s, NVML sampled every 50 ms), then the `nvidia-smi -lgc` attempt.
Raw: `results/2026-10-29-occfuse/b0_calib/` (`run{1,2}_<job>.jsonl/.st`, `c_*`, `nvml_*`, `lgc_<job>.txt`, `an<job>.txt` = analyze.py output). Reproduce: `analyze.py results/2026-10-29-occfuse/b0_calib <job> [jobs...]`. No hang, all rows `pass 1` (no values are written; nothing to verify).

## Method
- `b0_calib.cu`: k_b0, grid 148, 512 threads, cooperative, 1 CTA/SM, ub.cuh `bar_arrive` / `bar_wait` with stamps, 112 phases/launch, 2 warm-up + 7 timed launches, first phase skipped (phases 2..112).
- Phase = (thread 0 of the active CTAs spins W) -> drain of previous window loads (thread 0, mbarrier) -> `bar_arrive` -> thread 0 issues `nld` x 32 KB TMA loads (2 boxes of 64k x 128 rows, SW128, from the 2 GiB Pool, a new row window per CTA and phase) -> `bar_wait`.
- Arm a: nld 0; N in {8,16,32,64,96,128,148}, W in {0,0.5,1,2,4} us; N = spinning CTAs, spread as tiles.py floor ranges ("spread"); extra `a_first` (first N blockIdx) at W = 1 us. Two spin implementations: `spin_ns` (globaltimer, ub.cuh) and a clock64 spin (cycles = W x last measured MHz).
- Arm b: nld 1..4, N in {8,48,148}, same W.
- Arm c: idle = every CTA spins 3 us per phase; busy = thread 32 of every CTA issues 48 tcgen05 MMAs (M128 N256 K16, SW128 smem operands filled with hash data, accumulate, one commit per phase), thread 0 waits on the commit. Batches of 32 back-to-back launches x 112 phases for 6 s; MHz per launch from `clk_sample` (clock64 / globaltimer, thread 0 of each CTA, median over CTAs); NVML via `nvidia-smi -lms 50`.
- Stamps stored for the first 2 timed launches per config (NSTMAX 232; `st_off`, `st_n` in the row).

## Deviations from METHODOLOGY.md
1. [M] **The globaltimer step is 32 ns here, not 0.256 us** (k_gtstep: min nonzero delta 32 ns on all 148 SMs, ~96k distinct values per 200k back-to-back reads; stamps mod 256 take 8 values 0, 32, .. 224, job 14277/14278). But medians of 148 exit stamps and their differences land on a **256 ns grid** (every per-phase T and lat median in the JSONL is a multiple of 0.256: 1.28, 0.768, 1.024 ...). So the 0.256 us quantization of the METHODOLOGY applies to the stamp medians, not to single stamps. mk's stamps behave the same (st_14155: the medians of lat are 1.024 / 0.768 only).
2. Because of (1), the tables below use **means over phases** computed from the raw stamps: T = (median exit(112) - median exit(1)) / 111 (telescoping; resolution 0.256/111 us), lat = mean over phases 2..112 of (median exit - max arrive). The JSONL `us_T` / `us_lat` are the METHODOLOGY medians (quantized); the means are in analyze.py output. Means come from the 2 stored launches per config (n = 2 x 111 phases per run).
3. Launch time / nph (events) is 0.07-0.08 us above T (launch overhead / 111): within 5 % for T >= 1.5 us, 6-7 % for the W = 0 configs (28 / 274 configs flagged > 5 %, all small T). [M]
4. spin W: the spin starts and ends on thread 0 only; it uses `spin_ns` as in ub.cuh (see fit slope below: no bias visible once the mean is used).

## Results [M] (job 14277; job 14278 agrees within 0.01 us everywhere, see an14278.txt)
Gate arm a, spread placement, globaltimer spin, mean over phases (T / lat, us), run 1:

| N \ W us | 0 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| 8 | 1.154 / 0.81 | 1.650 / 0.87 | 2.154 / 0.78 | 3.160 / 0.80 | 5.134 / 0.78 |
| 32 | 1.158 / 0.81 | 1.707 / 0.79 | 2.218 / 0.80 | 3.216 / 0.78 | 5.201 / 0.78 |
| 64 | 1.157 / 0.83 | 1.729 / 0.80 | 2.237 / 0.79 | 3.230 / 0.80 | 5.209 / 0.80 |
| 128 | 1.158 / 0.79 | 1.754 / 0.81 | 2.272 / 0.81 | 3.259 / 0.80 | 5.241 / 0.79 |
| 148 | 1.160 / 0.82 | 1.747 / 0.80 | 2.258 / 0.80 | 3.243 / 0.80 | 5.233 / 0.81 |

Fit T = t_bar(N) + s W (pooled 2 runs x 5 W; job 14277; job 14278 identical to +-0.003):

| N | t_bar us | s | rmse | max abs(T - (W + T(W=0))) |
|---|---|---|---|---|
| 8 | 1.154 | 0.997 | 0.006 | 0.022 |
| 16 | 1.178 | 1.005 | 0.012 | 0.040 |
| 32 | 1.193 | 1.005 | 0.021 | 0.059 |
| 64 | 1.203 | 1.006 | 0.028 | 0.081 |
| 96 | 1.214 | 1.008 | 0.033 | 0.098 |
| 128 | 1.221 | 1.011 | 0.039 | 0.122 |
| 148 | 1.213 | 1.010 | 0.033 | 0.101 |

T(W=0) is 1.155-1.16 at every N (empty phase, any N). The clock64 spin gives t_bar 1.14-1.21, same slopes. T - W grows with N at W > 0 (about +0.07 from N 8 to 148) [M]; hypothesis: the max over N spinners of the spin end (`spin_ns` overshoots by up to one 32 ns tick plus loop time) [H].
- Placement (spread vs first N), W = 1 us: identical T and lat within 0.01 us at every N [M].
- Empty-phase cost: **T(W=0) = 1.157 us** (both jobs, both runs; per-launch medians of 0.256 quantized T are 1.024-1.28). lat (barrier release latency) = **0.80 +- 0.03 us**, independent of N and W (0.78-0.84 over all arm-a cells).

Arm b, N = 148 (T / lat us, run 1, job 14277; W = 0 and 1 shown; full tables in an14277.txt):

| W us | nld 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0 | 1.160 / 0.82 | 1.762 / 0.91 | 2.168 / 0.97 | 2.610 / 1.05 | 3.132 / 1.20 |
| 1 | 2.258 / 0.80 | 2.559 / 0.89 | 2.913 / 0.95 | 3.343 / 1.07 | 3.931 / 1.49 |
| 4 | 5.233 / 0.81 | 5.573 / 0.92 | 5.935 / 1.00 | 6.316 / 1.10 | 6.830 / 1.44 |

- Fit at N 148, W 0: extra T vs empty phase = 0.12 + 0.455 us x nld (rmse 0.03) (T includes waiting for the loads to land before the next arrive; 148 x 32 KB x nld per phase). lat = 0.81 + 0.09 x nld (rmse 0.03), but lat is not linear in nld at W >= 0.5 us with nld 4 (1.44-1.49) [M]. At large W (2-4 us) and nld <= 2, lat is 0.90-1.0.
- N = 8 and 48: lat 0.91-1.0 at nld 1-2 (W <= 1), 1.04-1.25 at nld 3-4; nld 4 at N 8, W 1 us gives 1.82.
- Window loads cost T +0.30-0.34 us (nld 1) at W >= 1 us, i.e. the loads are not hidden when issued in the barrier window and awaited before the next arrive.

Clock (arm c) [M], jobs 14277 / 14278 (two 6 s runs each, means of the two):

| arm | clk_sample MHz median (min..max per launch) | NVML sm MHz (50 ms) | power W | NVML throttle reason | phase T us |
|---|---|---|---|---|---|
| idle (3 us spin) | 1964.5 (1963.5..1965) | 1965 | 210-420 | 0x0 | 4.35 |
| busy (48 MMAs) | 1693-1702 (1577..1821; first 10 % 1717-1736, last 10 % 1692-1703) | 1860-1867 | 965-1008 | 0x4 (SW power cap) in 128/173 samples | 5.38 |

- clk_sample and NVML agree at idle (0.03 %). Under MMA load clk_sample reads 9 % below the NVML sm clock (1700 vs 1860) [M]; cause unknown [H: NVML value is a smoothed / pre-cap figure; or clock64 vs globaltimer ratio changes with clock gating under tensor load].
- Busy-phase MMA cost implied: (5.38 - 1.16 us) / 48 MMAs = 88 ns = about 150 cycles at 1700 MHz per M128N256 MMA including commit and wait in a 48-MMA phase; not comparable to B3 (no steady state) and not a gate here.

## Clock rule
[M] `nvidia-smi -lgc 1965` as the job user: "The current user does not have permission to change clocks for GPU ..." (exit 4, both jobs, both GPUs); `sudo -n true`: "a password is required". **No lock possible without sudo, so none was applied.**
Recommendation: do not lock; **normalize by reporting clk_sample MHz with every row** (already in `mhz`, per-launch median) and compare benches by us plus MHz. Idle-ish phases (spin, barrier, TMA) run at 1964.5 MHz; a bench that keeps the tensor cores busy runs power-capped (clk_sample about 1700 MHz, NVML 1860) so its us is not comparable to a 1965 MHz run without the MHz column. Cycle-convert (us x MHz) only for pure-compute terms (B3); memory-bound terms (B1, B2) should be compared at equal MHz class and reported as measured.

## Gate
- t_bar / lat: mk st_14155 (computed here from the .npz, b8 classic, layers 4..23, median over layers x launches of median exit - max arrive): median 1.024 us for every phase (0.768 for TS O), means over the same samples 0.90-1.05 (QKV 1.02, RoPE 0.90, attn 0.93, O 0.99, GU 0.99, D 0.96).
  - Arm a (bare barrier): lat 0.80 +- 0.03 (all cells, both jobs), vs 1.02: delta -0.22 against the +-0.2 band (0.82-1.22): **FAIL by 0.02 us (marginal)** on the mean; against mk's mean lat (0.90-1.05) delta -0.10..-0.25.
  - Arm b with nld 1-2 (window loads, N 148, W <= 1 us): lat 0.89-0.98 mean, inside the band; nld 3-4: 1.05-1.5. So the barrier with in-flight window loads reproduces 1.02 and the bare barrier does not [M]; whether mk's lat comes from window loads, from heavier traffic or from the stamp point is not tested here [H].
- T = W + t_bar within 0.15: **PASS**. Using T(W=0) as t_bar: max deviation 0.12 (N 128), 0.10 (N 148), <= 0.08 for N <= 64, both runs and both jobs; using the fit (slope 1.00-1.01) rmse <= 0.04. Median-based per-launch T (quantized) shows the same within the 0.256 grid.
- Repeat: jobs 14277 and 14278 (different GPU, same node) plus runs 1 / 2 in each: every gate cell within 0.02 us.

## Caveats
- Placement: both jobs ran on cayenne; no second node.
- Means use 2 stored launches per config (222 phases); the JSONL medians over 7 launches are quantized to 0.256 us.
- Arm b exposes the load completion in the same phase (drain before arrive); B2 has to choose its own convention.
- MMA arm writes nothing and never reads TMEM; the data is hash bf16 (not zeros) so the power is representative of dense MMAs, but no correctness check applies.
- Pool: window rows wrap after about 108 phases (148 x 128 KB per phase vs 2 GiB), so a few phases may hit L2 (not checked).
