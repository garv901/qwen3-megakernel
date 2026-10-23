# PASS-04: xover3, tile-stream GEMM engine vs split-K (MK-core, P4 of PASS-PLAN.md)

## Before the build
- Question: at T = 32..256, does a tile-stream engine (tiles (rb, token tile TT, k split KS); A and B by TMA into a ring;
  producer / MMA / epilogue warps; 2 TMEM accumulators; KS = 1 plain store, else red.add) beat the split-K engine (xover2 a3)?
  Sets the S/L crossover and the P5 engine. Spec: specs/SPEC-xover3.md. Band [H]: T = 256 -6..-12 us/layer.

## Results [M] (job 9592, GPU-17dce9e8, one placement; us per phase incl. one grid barrier; 0 FAIL of 100 checked configs)
| shape | T | a3 | best ts | config TT / KS | d |
|---|---|---|---|---|---|
| QKV | 32 | 5.91 | 5.09 | 32 / 4 | -0.82 |
| QKV | 64 | 6.25 | 5.15 | 64 / 4 | -1.10 |
| QKV | 128 | 7.40 | 6.07 | 128 / 4 | -1.33 |
| QKV | 256 | 10.98 | 7.11 | 128 / 2 | -3.87 |
| O | 32 | 4.43 | 4.44 | 32 / 16 | +0.02 |
| O | 64 | 4.34 | 4.78 | 32 / 8 | +0.44 |
| O | 128 | 5.00 | 5.01 | 64 / 8 | +0.01 |
| O | 256 | 7.16 | 5.89 | 128 / 8 | -1.27 |
| GU | 32 | 6.81 | 6.09 | 32 / 8 | -0.71 |
| GU | 64 | 7.11 | 6.22 | 64 / 2 | -0.89 |
| GU | 128 | 8.12 | 6.98 | 128 / 2 | -1.14 |
| GU | 256 | 12.81 | 8.88 | 128 / 1 | -3.92 |
| D | 32 | 5.03 | 4.87 | 32 / 12 | -0.17 |
| D | 64 | 5.24 | 5.03 | 64 / 12 | -0.21 |
| D | 128 | 5.93 | 5.69 | 128 / 12 | -0.23 |
| D | 256 | 8.37 | 6.58 | 128 / 8 | -1.79 |

Sum of the 4 GEMM phases (us/layer), a3 -> best ts: T 32: 22.18 -> 20.49 (-1.68); T 64: 22.94 -> 21.18 (-1.76); T 128: 26.45 -> 23.76 (-2.69); T 256: 39.32 -> 28.48 (-10.84). fi ~24 at T = 256 (job 9564).

## Reading [H unless tagged]
- ts wins at every shape and T except O at T <= 128 (tie within 0.02; T = 64 +0.44 [M, one placement]).
- Every best config has tiles max 1 per CTA [M]: one tile per CTA with a deep ring (2-5 slots) beats several split-K units.
- KS = 1 (plain store, no atomics) wins only at GU T = 256 (TT 128: 96 tiles); elsewhere KS 2..12 is best, so atomics are not
  the main cost; the pipeline depth per CTA is.
- a3 already loads X by TMA, so ts vs a3 is the pipeline + tile shape, not the X path. mk stages X with LDG + fused norm (a0-like:
  +0.3..1.1 us per GEMM vs a2 in job 9566): a P5 port with staged B keeps that cost unless B comes from a bf16 copy.
- Gain at T = 256 is -300 us/step at most [H]; b256 gap to fi is ~3500 us. The L split GEMM vs attention vs glue is measured in job 9593.

## mk phase split at L (job 9593 stamps, v3; work median + arrival spread, us per layer) [M]
- R1-b256: QKV 12.4 + 2.8, RoPE 3.0 + 0.5, attn 108.3 + 3.1, O 6.0 + 0.8, GU 14.9 + 3.1, **D 16.7 + 14.0** (+ 6 x ~1.0 barrier
  latency) = 191 us/layer. GEMM phases 70.7 vs ts bench 28.5 (incl. barriers): **mk at L is far above both a3 and ts**.
- R1-b64: QKV 3.8 + 0.5, RoPE 1.1, attn 26.0 + 1.0, O 2.0, GU 5.1 + 1.0, D 4.2 + 2.6. GEMM ~20 us/layer, near the ts bench (21.2
  incl. barriers): small P5 gain at b64 (~-80 us/step [H]); attention dominates (MK-attn).
- (The spread script ran its default workloads; b128 / R3-b32 are not in 9593.)
- Cause at b256 [H]: fp32 LDG staging of B per unit (D: g and u fp32, 256 tokens), per-unit flush of 256 token columns, unit
  imbalance (D 192 units on 148 CTAs: 2 vs 1).

## Next: P4b = staged-B arm in xover3 (decides the B path of P5)
B = bf16(x_fp32 * r[t]) staged by the epilogue warps per k-step into the ring slot (generic stores + fence.proxy.async + arrive on
full[s]) vs B by TMA from bf16. If staged B costs < ~1 us/phase at T = 256, P5 keeps mk's fused staging; else P5 writes bf16
copies of x / act in the producing phase.
