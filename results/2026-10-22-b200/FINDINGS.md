# Findings: hv2 megakernel design vs CUDA-graph + PDL baselines across serving regimes (B200, Qwen3-0.6B)

Data: `REPORT.md` / `summary.csv` in this folder, from runs `runs/*full-*` (slurm jobs 9471 R0+R1, 9472 R2+R8, 9473 R3,
9476 R6+R7, 9477 R4+R5). Same-point arms are interleaved in one job; regimes ran in different jobs.
Every hv2g and fi-graph result passed the FP32 oracle band (76/76 accuracy records, 3 seeds at check points).
Tags: **[M]** measured (numbers from REPORT.md), **[H]** hypothesis / model, not measured.

## Arms
- **hv2g auto**: hv2 generalized by the minimum needed to run every regime (runtime kv/position, token loop, causal mask,
  chunked prefill). MT1 build (1 token) is hv2's code path: 378 us vs hv2 375 us at R0 (+1.3 %, job 9468) [M]. MTN build
  (many tokens) is a first, untuned generalization (6 barriers/layer, FP32 FFMA, 80 B spills).
- **hv2-repeat**: unchanged hv2 binary launched N = tokens times (timing proxy, b1/kv128 work per launch).
- **fi-graph best**: FlashInfer + cuBLAS/cuDNN/cute-dsl GEMMs, attention backend autotuned per point (fa2, trtllm-gen,
  cute-dsl, cudnn), CUDA graph, PDL on/off; best of 4 variants.
- **vLLM 0.31**: engine step, wall and GPU span (`execute_model` + `sample_tokens` events; includes host-side gaps).

## Headline [M]
| regime | hv2g wins | geomean hv2g/best | crossover |
|---|---|---|---|
| R0 decode b1 kv128 | 1/1 | 0.45 | - |
| R1 batch (kv128) | 2/9 | 2.70 | B=4 |
| R2 long ctx (b1) | 4/8 | 1.29 | kv=4096 |
| R3 batch x KV | 1/9 | 2.25 | B=1: kv 4K; B>=8: kv 1K |
| R4 prefill b1 | 0/6 | 22.5 | T=16 |
| R5 prefill batch | 0/4 | 99.5 | all |
| R6 multi-token decode (kv 1K) | 1/4 | 1.34 | q=4 |
| R7 chunked prefill + decode | 0/3 | 24.6 | all |
| R8 ragged decode b32 | 0/1 | 3.89 | - |

The hv2 design wins only where it was tuned: one token, short context. There it is 2.2x faster than the best graph+PDL
baseline (378 vs 834 us) and at 40 % of HBM SOL, vs 18 % for the baseline.

## Per-regime reading
1. **Baselines at small work are launch/latency bound [M].** fi-graph runs ~312 kernels per step; at R0-R2 it sits at
   17-23 % SOL and grows slowly with work (B 1->64: 878 -> 1404 us; kv 128->32K: 884 -> 1612 us). PDL is worth ~10-15 %
   (graph-pdl-best vs graph-nopdl-best). At large KV it reaches 58-65 % SOL (R3 b8-16K, b32-4K, R8).
2. **Batch (R1, R3, R8) [M]:** hv2g grows ~linearly with tokens: +150-180 us per extra token (b2 783, b8 1482, b32 4894,
   b256 37404 us), i.e. ~11 % of the FP32-FFMA ceiling at every B >= 16. The baseline barely moves until B~64.
   [H] The FFMA-SOL column says an FFMA megakernel that reached its ceiling would stay below fi-graph up to B~64
   (FFMA-SOL 1050 vs fi 1404 us at B=64; 2100 vs 1760 at B=128). So for B 4-64 the gap is mostly implementation
   (register-blocked multi-token GEMV), and only for B >~ 100 does the design need tensor cores.
3. **Long context (R2) [M]:** hv2g adds ~1.35 us per 128-position chunk per layer (kv 1K 639, 4K 1431, 32K 8704 us); one
   q-head's attention is walked serially by its CTAs, and the 9 CTAs of a q-head repeat the same attention (hv2 layout).
   fi-graph at 32K: 1612 us (38 % SOL). Crossover kv 4K. Needed: split-KV / flash-decoding across all 148 CTAs with a
   combine step, and no redundant per-slice attention [H: expected to bring hv2g near its b1 advantage up to ~16K].
4. **Batch x KV (R3) [M]:** both batching and KV costs stack; hv2g at 9-31 % SOL vs fi 17-65 %. KV-bound points
   (R3 b8-16K, R8) need the KV stream spread over the whole grid (same fix as 3) plus the batch fix from 2.
5. **Prefill (R4, R5) and mixed (R7) [M]:** hv2g is 20-118x slower (0.4 % SOL); compute-bound beyond ~280-380 tokens
   per step (SOL ridge). Even the FFMA ceiling is ~30x above the tensor-core SOL (R4 T8K: FFMA-SOL 200 ms vs tensor SOL
   6.6 ms) [model]. No FP32-FFMA design can compete here: a megakernel must issue tensor-core GEMM tiles
   (tcgen05 / mma) for T >= tens of tokens. Earlier result (hv2_tc, hv2_mma): tensor-core GEMV loses at b1 (clock
   penalty, M128 waste), so this implies a dual-path kernel (FFMA GEMV for small token counts, tcgen05 tiles for large).
   R7 (512-token chunk + D decodes) is dominated by the chunk: same conclusion.
6. **Multi-token decode (R6) [M]:** q=2 ties (1033 vs 1075 us), q>=4 loses (q16: 2.3x). Spec-decode verification needs
   the multi-token GEMV fix from 2 first.
7. **vLLM [M]:** never the best baseline. Its GPU span is ~1.8 ms at R0 (vs fi 0.83 ms), because Python-side
   work inside `execute_model` leaves the GPU idle; it gets competitive only at large prefill (R5 b16-T2K: 43 ms vs fi 39 ms).

## What the megakernel needs, by regime [H, ranked by measured gap]
| regime | gap now | missing piece |
|---|---|---|
| prefill / mixed / B >~ 100 | 20-118x | tensor-core tile path (tcgen05) + tile scheduler; keep FFMA path for small token counts |
| batch 4-64, spec verify | 1.1-6.7x | multi-token GEMV with weights reused from registers across tokens (FFMA ceiling says this is enough up to B~64) |
| long context, KV-heavy | 1.4-5.4x | split-KV across all CTAs + combine; remove redundant attention per Wo slice |
| b1, short ctx | wins 2.2x | current design |

## Caveats
- hv2g MTN is a first generalization, not tuned; part of the batch gap is its 11 % FFMA efficiency, not the design limit.
- KV cap 40 GB dropped R3 b32-16K, b128-4K, b128-16K.
- hv2-repeat at long KV is not meaningful (it runs kv 128 work); it is a lower bound for "use the b1 kernel N times".
- Spec peaks (8 TB/s, 2.25 PFLOP/s) are datasheet-derived; measured probes: 6.55 TB/s copy, 1.66-1.70 PFLOP/s cuBLAS.
- R4/R5 used 5 rounds and one variant per megakernel arm; R6/R7 10 rounds; other regimes 20 rounds.
