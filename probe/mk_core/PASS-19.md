# PASS-19 (C2): hide the core chain under the KV stream (split-batch overlap) - silo gate first

Owner mk-core-r10. Status: **KILLED 2026-10-31** (silo jobs 14913 / 14919; result at the end). No mk code. Joint with mk-attn if the silo passes (MAILBOX 2026-10-31).

## Data that motivates it [M] (job 14897, BREAKDOWN.md)
- Attention is 77-89 % of the mk layer at R10, near the HBM rate at the best points (6.3-6.9 TB/s vs fi 6.5-7.1).
- mk core phases (QKV + RoPE + O + GU + D incl. barriers) are 27-43 us/layer: 27.3 (b48-kv4K), 29.3 (b64-kv2K),
  30.5 (b96-kv3K), 32.3 (b128-kv2816), 42.5 (b256-kv1K). fi non-attention: 34.6-49.6.
- The GEMM phases read 31 MB of weights per layer (~5 us at HBM rate): they are latency / sync bound, not bytes bound.
  So they leave HBM idle for ~27-43 us/layer that the KV stream could use.

## Hypothesis [H]
Run the batch as two halves. Per layer: slot 1 = attend(h0, l) on k_a CTAs || chain(h1) (O / GU / D of layer l-1,
QKV / RoPE of layer l) on the other k_c = 148 - k_a CTAs; slot 2 = the mirror. If the KV stream keeps >= 95 % of its
148-SM rate on k_a SMs and the chain on k_c CTAs at T = NPAD / 2 fits inside half the attention time, the layer costs
~ attention + 2 slot barriers + the doubled weight bytes (31 MB x 2 = ~9 us at HBM rate, overlapped with the KV).

## Expected band [H]
-15..-25 us/layer at b64-kv2K (114 -> ~90), -20..-35 at b128 / b256 (core 32-43). At b48-kv1K (attention ~35 us)
only part of the chain hides: ~-5..-10. 0.4-1.0 ms/step at the larger points.

## Gate (silo bench xover6, SPEC-xover6.md)
- Q1: KV TMA stream aggregate TB/s on k_a in {148, 128, 116, 104, 92} SMs x ring depth {2, 4, 6} x 32 KB.
- Q2: GEMM chain (O, GU, D, QKV at the mk shapes, tile stream, best KS) on k_c = 148 - k_a CTAs at T in {32, 64, 128}:
  time alone, and next to the Q1 stream (both roles in one cooperative kernel).
- Pass: some (k_a, depth) keeps >= 95 % of the 148-SM stream rate while the chain on k_c CTAs (with the stream
  running) is <= 45 us at T 32 / 64 (= half the attention of b64-kv2K / b48-kv4K at the measured rates).
- Kill: hidden time < 10 us/layer at b64-kv2K from Q1 x Q2, or the stream loses > 10 % next to the chain.

## Points (when it reaches mk)
mk_r10 uniform points first (b48-b256); trace steps after mk-attn's imbalance fix.

## Arms
mk (previous default) vs mk MK_SBP=1 vs fi-graph, same job, 7 rounds; gains under 5 us need 3 placements x 2 GPUs.

## Result [M] (xover6 jobs 14913 / 14919, mk-attn a1ub 14914 / 14917) -> killed
- Q1 stream alone: depth 4 / 6 keeps >= 99 % of the 148-SM rate (6866 GB/s) down to k_a 74. attend on 128 CTAs = on 148.
- Q2 GEMM 4-shape sum alone (us): k_c 20 48.6 / 50.2 / 56.6 (T 32 / 64 / 128), k_c 44 32.0 / 33.2 / 41.1, k_c 74 24.6 / 25.3 / 27.8.
- Both at once (depth 4): GEMM 1.28-1.65x slower, stream 0.87-0.92 of alone (depth 6: 1.8-2.6x, 0.74-0.84). The two roles
  interfere in HBM / L2: the chain is not free next to the stream, and the stream loses 8-13 % (kill rule: > 10 %).
- Net [H from M]: ~8 us/layer at b64-kv2K (hidden < 10: kill rule), ~13 at b128-kv2816, <= ~20 at b256, for a structural
  change (two plans, slot barriers, split chain barriers). Attention imbalance (R10-1) is the larger lever: done first.
