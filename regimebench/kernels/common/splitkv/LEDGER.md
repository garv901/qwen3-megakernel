# splitkv LEDGER (MK-attn)

Budget: 20 GPU-h for the whole effort. Tags: [M] measured, [H] hypothesis.

| id | change | band | result (us per point) | job ids | GPU-h | kept/dropped | tag |
|---|---|---|---|---|---|---|---|
| P0b | fi-graph attention kernel timeline (graph-pdl-best) at 6 mk_kv points | - | attn kernel us/step (/layer), profiler durations incl. PDL overlap = upper bounds [M]: R2 kv1K trtllm 205.7 (7.3), kv4K 239.9 (8.6), kv32K 804.0 (28.7, 4.7 TB/s); R3-b8-kv4K trtllm 715.5 (25.6); R3-b32-kv1K cute-dsl 730.1 (26.1); R6-q16 fa2 201.5 + merge 87.2 = 288.7 (10.3). Steps 869/915/1535/1633/1694/1174 us, all PASS | 9563 | 0.5 (limit) | baseline | - |
| P0c | hv2g null-attention oracle (attention phase cost vs kv; `probe/mk_attn/p0_nullatt_ab.py`, 20 rounds interleaved) | - | hv2g attention us/step (/layer) = na0 - na1 [M]: kv128 9.5 (0.34), 512 141 (5.1), 1K 275 (9.8), 2K 541 (19.3), 4K 1069 (38.2), 8K 2114 (75.5), 16K 4195 (150), 32K 8351 (298); MT1 null step 364-366 us at every kv. MTN: R6-q16 646 (23.1; null 2021), R3-b8-kv4K 1434 (51.2; null 1388), R3-b32-kv1K 1719 (61.4; null 4500). Paired p10/p90 within 0.5-6 us. Sanity PASS all | 9565 | 0.67 (limit) | baseline | - |
| P1 | microbench: KV stream engines x item plan x bytes in flight, L2 prefetch, combine (`probe/mk_attn/ubench/kv_stream.cu`) | see PASS-01 | stream A us/layer incl. barrier (1.12): kv1K ldg 2.66, 4K ldg 5.25, 32K tma 26.75 (5.0 TB/s), b32-kv1K tma 26.75, b256-kv128 tma 26.30; L2 prefetch net <= 0 (P5 killed); combine S=18 T=1 +3.5..4.2 us (bench merge latency-bound) | 9567 | 0.33 (limit) | decided: ldg<=128KB/CTA else tma 4x16K, h1 | - |
| P2 | A1 v1 standalone (`splitkv.cuh` attend + combine_head, `plan.py`; bench `probe/mk_attn/a1bench.py`, 148 x 512, 28 layers, cold KV) | PASS-02: kv1K 2.8-3.4, 4K 5.5-6.3, 32K 26.5-28; C <= 1.0 | check PASS at all 13 points (rel-L2 1.6-1.7e-3 vs fp32 torch, SW128 bytes exact) [M]. A us/layer incl. barrier [M]: kv128 4.10, 512 6.43, 1K 7.01, 2K 7.78, 4K 9.34 (ldg); 8K 14.53 tma / 12.58 ldg; 16K 23.87, 32K 42.27, b8-kv4K 42.66, b8-kv1K 14.75, b32-kv1K 45.15, b64-kv128 20.80, b256-kv128 73.76 (tma). C (combine_head) 1.9-2.3 at b1, 3.7 b32, 5.7 b64, 12.5 b256; ao fallback C 3.9-4.1. SKV_KV_NC=0 (coherent) = nc within 0.05 us. Below band everywhere; kill criterion hit for the TMA path (kv32K / b32-kv1K slower than fi 28.7 / 26.1). Bug fixed: per-call mbarrier re-init hung the TMA ring at kv >= 16K (9581, 9598, 9599) | 9581, 9594, 9598, 9599, 9601 | 0.75 actual (1.95 limit) | kept as v1 baseline (correct); perf re-opened in P2b | [M] |
| P2b | perf variants: TMA stage geometry / slot release, compute-off profiling, item prefetch, one-item planner, engine crossover (`specs/P2b-perf.md`) | PASS-02 bands | all 91 checks PASS [M]. TMA kv32K: v1 geometry 38.50, 32x4 + empty-mbarrier release 42.56, 64x2 30.08, 64x4 30.66, 128x2 27.33; NOCOMP 32x4 27.26, 64x4 23.10 (ring alone beats P1's 26.75). b8-kv4K 128x2 27.14, b32-kv1K 29.47, b256-kv128 60.70. One-item planner b1: kv512 6.43 -> 3.90, 1K 6.69 -> 4.22, 4K 9.28 -> 6.94. Item prefetch: +0..+0.4 (no gain). LDG wins to b1 kv16K (16.00 vs tma 16.58-17.66); b8-kv1K tie | 9610 | 0.03 actual (0.33 limit) | kept: 128x2 R1, one-item, ldg <= 512 KB; prefetch off | [M] |
| P2c | P2b winners as defaults | - | check PASS 13/13 [M]. A us/layer: kv128 3.65, 512 3.94, 1K 4.19, 2K 4.83, 4K 6.75, 8K 10.18, 16K 16.00 (ldg), 32K 27.23, b8-kv4K 27.58, b8-kv1K 12.06, b32-kv1K 30.11 (tma), b64-kv128 19.01, b256-kv128 63.30. C 1.9-2.3 at b1. vs fi kernel (P0b): kv1K 4.19 vs 7.3, 4K 6.75 vs 8.6, 32K 27.23 vs 28.7 (win); b8-kv4K 27.58 vs 25.6, b32-kv1K 30.11 vs 26.1 (lose) | 9611 | 0.02 actual (0.17 limit) | kept (attend smem 148,032 B) | [M] |

## Requests (from MK-core)
(none open)

## Bugs found in MK-core code
(none)

## Notes
- Tests: `tests/test_hv2g.py::test_mtn_compiles` / `test_mt1_dyn_smem_fits_one_block_per_sm` fail when the driver compute cache
  already holds their PTX (ptxas does not run, the -v log is empty -> KeyError). Not an A1 file; reported to the user. Passes with
  `CUDA_CACHE_DISABLE=1`.
