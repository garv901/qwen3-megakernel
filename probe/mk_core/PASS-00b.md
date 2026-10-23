# PASS-00b: GEMM phase engine / tile choice (MK-core)

Job 9566 (main, GPU-c88e489a, mk-core c1fc660 sources, 2026-10-23). Log `probe/mk_core/xover2/logs/slurm-9566.log`.
Bench: `probe/mk_core/xover2/` (SPEC-xover2.md). us per phase = kernel time / 112 phases (LM: / 4); each phase includes
one grid barrier, the barrier-window weight prefetch, X load after the barrier, MMA, flush (red.add). 5 reps, median.
Correctness: 120 / 120 PASS vs the CUDA-core reference (T in {1, 8, 64, 256}, all compute arms; oracles skip by design).
One GPU, one placement: differences < ~0.3 us between arms are inside the noise rule (3 placements x 2 GPUs) and are [H].

Arms: a0 = xover ENG=1 (KU 128, LDG X staging, ring 2); a1 = KU 64, ring 4; a2 = X by TMA; a3 = a2 + ring max;
a4 = KU 64 + XTMA + ring max; a5 = a3 with cta_group::2 (cluster 2, M 256); o0 / o3 = oracles (no MMA) of a0 / a3.
SM clock: 1824-1846 MHz under tcgen05, 1960 in the oracles [M].

## us per phase [M] (fi = fi-graph kernel time per call, job 9564, upper bound)
| shape | arm | T=1 | 8 | 16 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|
| QKV | a0 / a2 / a3 / a5 | 5.98 / **5.24** / 5.51 / 5.53 | 6.04 / **5.29** / 5.51 / 5.51 | 6.22 / **5.44** / 5.67 / 5.69 | 6.90 / **6.03** / 6.21 / 6.46 | 8.29 / **7.17** / 7.39 / 7.42 | 11.98 / **10.86** / 10.92 / 11.02 |
| QKV | o3 / fi | 4.20 / 3.56 | 4.16 / - | 4.39 / 4.15 | 6.22 / 5.58 | 8.45 / - | 14.12 / ~6.1 |
| O | a0 / a2 / a3 / a5 | 4.45 / **4.15** / 4.21 / 4.28 | 4.54 / **4.19** / 4.31 / 4.37 | 4.60 / **4.23** / 4.39 / 4.43 | 4.64 / **4.23** / 4.40 / 4.48 | 5.55 / **4.90** / 4.96 / 4.98 | 7.75 / 7.09 / 7.07 / **6.99** |
| O | o3 / fi | 3.33 / 2.99 | 3.39 / - | 3.46 / 3.44 | 4.10 / 5.64 | 5.64 / - | 8.41 / ~6.0 |
| GU | a0 / a2 / a3 / a5 | 7.94 / 7.10 / **6.52** / 6.59 | 8.07 / 7.20 / **6.49** / 6.55 | 8.22 / 7.35 / **6.66** / 6.71 | 8.74 / 7.79 / **7.09** / 7.20 | 10.16 / 8.93 / **8.10** / 8.30 | 13.92 / 12.71 / 12.77 / **11.91** |
| GU | o3 / fi | 4.96 / 3.93 | 5.01 / - | 5.38 / 4.72 | 7.79 / 5.14 | 11.93 / - | 20.38 / ~6.1 |
| D | a0 / a2 / a3 / a5 | 5.27 / **4.55** / 4.83 / 4.85 | 5.36 / **4.61** / 4.88 / 4.92 | 5.43 / **4.68** / 4.96 / 4.99 | 5.89 / **5.12** / 5.20 / 5.40 | 6.87 / **5.56** / 5.96 / 6.05 | 9.26 / 8.03 / 8.32 / **7.92** |
| D | o3 / fi | 3.57 / 3.46 | 3.81 / - | 4.07 / 3.78 | 5.79 / 5.64 | 8.56 / - | 13.63 / ~6.0 |
| LM | a0 / a2 / a3 / a5 | 131.8 / 110.7 / **85.1** / 86.6 | 132.9 / 112.3 / **86.1** / 87.7 | 133.7 / 114.3 / **86.6** / 88.2 | 136.8 / 119.9 / **88.9** / 90.8 | 149.2 / 127.8 / **94.7** / 96.4 | 179.9 / 153.8 / 155.9 / **112.2** |
| LM | o3 / fi | 52.6 / 45.7 | 57.8 / - | 63.6 / 57.4 | 120.8 / 59.4 | 209.5 / - | 391.6 / 70.4 |
KU 64 (a1, a4) is slower than its KU 128 pair at every shape and T (+0.1..1.7 us, LM +6..48 us): dropped.
At T >= 64 the oracles are slower than the arms: their flush is a per-unit red.add of all T tokens (NOTES.md),
so o0 / o3 are a floor of load + barrier only at T <= 32.

## Reading [H unless tagged]
1. **X by TMA wins everywhere** (a2 vs a0: -0.3..-1.1 us per GEMM, LM -21..-26 us) [M, outside noise]. mk v0 stages
   X with LDG (it fuses the norm scale and SiLU*up into staging). Getting TMA X into mk needs a bf16 X in global that
   is already the B operand: fold ln1 / ln2 into the weight rows offline (exact up to bf16 rounding), and write
   bf16 x / bf16 act in the epilogue of the phase before (needs the row-owner flush, not split-K red.add, or a
   convert pass). This is a v1 pass of its own.
2. **Ring max**: helps GU (3 units / CTA: -0.6..-0.8 us) and LM (-25 us); at QKV / O / D it is equal or 0.1-0.3 us
   worse (inside noise). Use ring = min(fit, units per CTA) = a3 for every phase.
3. **cta_group::2** (a5): equal to a3 at T <= 128 (+0.0..0.2 us); at T = 256 it wins GU -0.86, D -0.40, O -0.08, and
   LM -43.7 us [M, LM outside noise]. Decision: **bucket NPAD 256 uses cta_group::2; buckets 16..128 stay ::1**
   (buckets are separate entry points, so no mixing inside a kernel). R1-b256's LM runs at T = 256.
4. **Gap to fi per GEMM is in the fixed per-phase cost, not the MMA**: at T <= 16 even the no-MMA oracle o3 is
   above fi's kernel (QKV 4.20 vs 3.56, GU 4.96 vs 3.93); the weight stream alone is 0.65-1.94 us. Per GEMM phase
   at T = 1 ~2.7-3.0 us is barrier + first-TMA latency + X load + flush. fi's 4 GEMMs at T = 1 sum to 13.9 us/layer;
   mk best (a2/a3) sums to 20.3 us/layer: **+6.4 us/layer = ~180 us/step to recover** (mk wins today only by the
   530-730 us/step of fused glue fi pays).
5. LM at T <= 128: a3 85-95 us vs o3 52.6 and fi 45.7-59.4. One TMEM accumulator: each CTA's 8 row blocks flush
   serially and stall the next MMA. Double-buffered accumulator (2 x 256 columns: fits 512) is the first LM lever
   (band: -25..-30 us/step [H]).

## Decisions for mk (recorded in LEDGER P0b)
- KU 128 everywhere; ring = min(fit in smem, units per CTA); cta_group::2 only in the NPAD 256 bucket.
- Optimisation order (to the pass plan): (a) per-phase fixed cost (barrier + first-TMA + X load), (b) X by TMA via
  folded norm weights + bf16 epilogue, (c) LM double-buffered accumulator, (d) NPAD 256 on cta_group::2.
GPU-h: 9566 used 0.01 (0.75 reserved).
