# MK-core optimisation pass plan (after v0; for user approval)

Data: V0.md (jobs 9572, 9576), PASS-00 (fi timeline 9564, baseline 9562), PASS-00b (xover2 9566). Bands are [H].
Target = fi-graph at every point (fi is the best SOTA at all 28, job 9562).

## Two token-count regimes, two designs (each NPAD bucket is its own entry point)
- **S = NPAD 16 (b1..b16, R0, R2, R3-b1/b8, R6)**: latency-bound. 6 grid barriers ~ 13 of 27 us/layer at b1; GEMM work
  11.8 us/layer; LM 99 us. Keep split-K units (enough units for 148 CTAs at T <= 16); cut barriers and fixed costs.
- **L = NPAD 64..256 (b64..b256)**: GEMM-structure-bound. Split-K red.add traffic grows as N x T x K-chunks, and the fused
  fp32 LDG staging grows with T (D phase at b256: 33.6 us vs 9.3 in the bench). Switch to output-tile ownership (a tile =
  128 rows (::1) or 256 rows (::2, NPAD 256) x a token tile, full K, no split-K), as fi's nvjet / 2-CTA kernels do, with
  fused epilogues. NPAD 32 (b17..b32) goes to whichever wins in the bench (crossover measured, not assumed).

## Passes, in order
| pass | regime | change | band (us/step) | GPU-h |
|---|---|---|---|---|
| P1 | all | **Barrier cost**: microbench of the grid barrier alone (null-all build + stamps), then options: flag/epoch word instead of counter poll, arrive via `red.release` + `ld.acquire` poll spread over lanes, cluster-pair pre-arrive. | b1: -60..-100 (6 x 28 barriers x 0.4-0.6 us) | 0.3 |
| P2 | all | **Drop the phase-2 barrier**: q/k-norm + RoPE + KV write move into the attention phase (each attention CTA finalizes the q head / k,v row it reads from fp32 qkv; redundant k work is 128 elements). Needs an A1 interface change, agreed with MK-attn. | -50..-60 (1 barrier/layer) | 0.3 |
| P3 | all | **LM head**: TMA X + ring max (xover2 a3: 132 -> 85 at T = 1), double-buffered TMEM accumulator (2 x 256 cols), cta_group::2 at NPAD 256 (a5: 180 -> 112). | b1: -40..-50; b256: -150..-200 | 0.4 |
| P4 | L | **Bench xover3**: output-tile GEMM phase (full K, token tile 32/64/128, ::1 and ::2) vs split-K at T = 32..256, all 4 shapes, vs fi per call (5.1-6.7 us). Sets the S/L crossover. | - | 0.4 |
| P5 | L | **Output-tile engine in mk for NPAD >= crossover**, with fused epilogues: GU tile pairs g/u rows -> writes bf16 act = silu(g) u (D then stages bf16 by TMA); QKV tile = one head -> q/k-norm + RoPE + KV write in the epilogue; O / D: single writer per element (plain store, no atomics). | b256: -1000..-1300; b64: -250..-350 | 1.0 |
| P6 | S | **GEMM fixed cost**: ring = min(fit, units per CTA), whole CTA weight share issued before the barrier wait, X by TMA where the staging has no fused op. | b1: -40..-70 | 0.5 |
| P7 | all | e2e A/B of every kept pass on all 28 points (background), tag vN each. | - | background |

Projected [H, wide]: b1 872 -> ~620-700 (fi 787); b16 1030 -> ~800 + A1 attention gains (fi 1019);
b64 1700 -> ~1100-1250 with A1 at fi's attention (fi 1407); b256 5813 -> ~2000-2300 with A1 at fi's attention (fi 2235): b256 is
the hardest point and depends on both P5 and A1.

## Needs from MK-attn (attention is the largest term from b8 up: 3.9 -> 107 us/layer, V0.md)
A1 must reach fi's attention time: b16 ~5.5, b64 ~12.4, b256 ~33 us/layer (job 9564), R2-kv8192 not profiled yet.
P2 changes A1's input (fp32 qkv + rr1 + q/k-norm weights + RoPE instead of post-RoPE bf16 q).

## Budget
Used so far on main: ~0.1 GPU-h (jobs 9564-9576; all short). Plan above: ~3 GPU-h main + background e2e runs.
