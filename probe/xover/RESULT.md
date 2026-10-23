# xover result (job 9521, 1x B200 GPU-c88e489a, 2026-10-23) - [M] unless tagged [H]

One GEMM phase, 148 CTAs x 512 thr, cold weights, same TMA/X-staging/barrier/red.add skeleton for all engines.
us per phase = median of 5 launches x 112 phases (rep spread <= 0.13 us). All engines PASS vs CUDA-core reference.

| T | GU oracle | GU FFMA | GU tcgen05 | D oracle | D FFMA | D tcgen05 |
|---|---|---|---|---|---|---|
| 1 | 4.96 | **6.91** | 7.74 | 3.53 | **5.01** | 5.10 |
| 2 | 5.03 | 7.88 | **7.76** | 3.58 | 5.73 | **5.12** |
| 4 | 5.11 | 9.80 | **7.79** | 3.64 | 6.94 | **5.14** |
| 8 | 5.30 | 13.94 | **7.88** | 3.79 | 9.69 | **5.18** |
| 16 | 5.76 | 22.11 | **8.03** | 4.26 | 15.19 | **5.25** |
| 64 | 8.84 | 71.55 | **8.56** | 6.50 | 48.51 | **5.75** |
| 256 | 22.25 | 269.62 | **14.10** | 14.63 | 181.19 | **9.09** |

- tcgen05 phase time is flat: +0.14 us GU / +0.08 us D from T=1 to T=8, +0.8 / +0.6 us to T=64.
- Clock: FFMA 1960 MHz, tcgen05 1834-1850 MHz (-6 %), as in jobs 9420/9421.
- Hybrid binary (both paths compiled, tcgen05.alloc executed, runtime switch): FFMA path keeps 1960 MHz
  (alloc alone does not drop the clock), but costs +0.5..+1.0 us/phase vs the pure FFMA binary, and the TC path
  +0.4..+0.6 us vs the pure TC binary. So "one source, two entry points" is better than one runtime-switched kernel.
- Caveat: this FFMA engine is weak (LDS weights, per-token shuffle + red.add): at T=8 GU it is ~16 % of the FFMA peak,
  and its T=1 math cost (+1.95 us over oracle) is ~3x hv2's G dot (584 ns). So T=2 is a LOWER bound on the crossover.
  [H] Upper bound: FFMA at 100 % peak costs 0.17*T us (GU) / 0.085*T (D) vs tcgen05's ~2.6 / ~1.5 us over the oracle
  -> crossover T ~ 15-17. A 50 %-of-peak FFMA -> T ~ 8. hv2g MTN runs at ~11 % of the ceiling.
- Oracle = floor for the FFMA output path only (its red.add pattern); tcgen05 beats it at T >= 64.
