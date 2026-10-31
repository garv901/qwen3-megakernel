# PLAN-R10 (mk-core-r10, 2026-10-31)

Regime R10: decode-only steps, 33-256 requests, long KV (Azure 2024 conv L3 / L4: 49 % / 34 % of stock GPU time).
Owner split: mk-core = everything in mk except `common/splitkv/` (mk-attn). State of the run: STATUS-core.md, LEDGER.md.

## 1. Facts [M] (census of the stock step logs, `probe/mk_core/r10/census_d33.py`)
- n_req, GPU-time weighted p10/p50/p90: conv L3 37/54/84, conv L4 47/87/198 (L4: 34 % of the R10 time has n >= 129).
- Mean kv per step p10..p90 2.3K..3.3K at every n; max kv 5.9K..8.1K; kv CV within a step ~0.6 (code 0.8).
- Total cache tokens per step p50 158K (L3) / 235K (L4), p90 256K / 605K. Stock forward p50 3.9 ms (L3), 4.7 ms (L4).
- Serving M1 (v17 mk): mk 1.23-1.43x stock on these steps, fi 0.84-0.92x. Serving builds one graph per NPAD bucket
  (`serve_tune(npad)`), so per-step kv-dependent header knobs do not exist there.

## 2. Points: preset `mk_r10` (commit 703eb18)
- Uniform q_len 1: b48 kv 1K/2K/4K; b64 1K/2K/4K; b96 1K/2K/3K; b128 1K/2K/2816; b192 1K/1792; b256 1K/1408.
- Real steps (`regimebench/r10_steps.json`): R10-conv-L3-n54, R10-conv-L4-n87, R10-code-L3-n40 (nearest to the p50).
- Memory cap (computed from the job 11565 OOM record): timed points <= 40 GiB fixture KV (4 copies: fixture, mk:auto,
  fi-graph pdl-best + pdl-cublas), check points <= 25 GiB (6 copies). Run with
  `--variants "mk=auto;fi-graph=graph-pdl-best,graph-pdl-cublas"`. Limitation: the n >= 129 trace steps
  (> 360K cache tokens) cannot be measured offline against fi-graph; the serving run (stop rule 2) covers them.
- Note: `tests/test_core_harness.py::test_smoke_selection` already failed at HEAD (R9 added in P18); R10 adds one more
  regime. Shared harness test, not changed; told the orchestrator.

## 3. Bytes model [H] (to be replaced by the baseline + stamps)
KV per step = cache tokens x 112 KB; per layer = tokens x 4 KB. b64-kv2K: 0.54 GB/layer -> ~80 us/layer at 6.6 TB/s;
b128-kv2816: 1.48 GB/layer -> ~220 us/layer; b256-kv1K: 1.07 GB/layer -> ~165 us/layer. mk GEMM + barrier phases at
NPAD 64-256 were 20-30 us/layer at kv128 (v17 stamps). So attention is [H] 75-90 % of the step; the core phases are
10-25 %. Weights per step 1.2 GB (6 us/layer at HBM rate): the GEMM phases are sync / latency bound, not bytes bound.

## 4. Steps
1. Done: tag mk-core-v18, workloads + tests, point list to mk-attn.
2. Baseline job 14896 (background): mk, fi-graph, vLLM at mk_r10, 20 rounds. Breakdown job 14897 (main): mk stamps
   (stamps_spread.py: work / spread / lat per phase) + fi-graph pdl-best timeline at 8 points. Fill BREAKDOWN.md
   (attention rows owner mk-attn).
3. First core pass, chosen by gap x share from the breakdown. Candidates, each to be tested against data:
   - **C1 GEMM phases next to a long KV stream.** Compare R10 GEMM stamps to the same NPAD at kv128 (v17: QKV / O / GU /
     D). If they are slower at R10, the cause is L2 (KV TMA evict_first on? the h64 / g4a / g8 consumers' traffic) or
     code-gen; probes: MK_NULL attend-off oracle at R10 shapes, EF A/B.
   - **C2 hide the core phases under the KV stream (nano-batch overlap).** If attention is HBM-bound and the GEMM
     phases are latency-bound, run the GEMM chain of one half of the batch while the other half streams KV
     (NanoFlow-style, inside one persistent kernel: CTA partition or co-resident warp roles). Upper bound = the GEMM +
     barrier time per layer (~20-30 us/layer = 0.6-0.8 ms/step). Structural and joint with mk-attn (smem, TMEM, barrier
     regions): propose in MAILBOX with numbers before any build; silo first (one layer, two half-batches, measured
     HBM-rate of the attention stream on fewer SMs vs the GEMM chain on the rest).
   - **C3 O-proj / combine boundary.** Plan S per sequence at R10 (no-split LPT at >= 148 pairs?): if s_max > 1 at some
     points, the combine sits on the O critical path; measure from the O stamp.
   - **C4 LM head at n_seq 33-256** (NPAD 64: LMP path; NPAD >= 128: tile stream). Small share [H]; check its stamp.
   - **C5 barrier count / finalizer placement** at NPAD 64 (MK_FIN off below 128, MK_WU only at 128): cheap knob A/Bs at
     R10 points before any new code.
   - **C6 per-bucket knobs that depend on kv** (skv_tune by max kv) do not exist in serving: list what the paged build
     loses and pick bucket defaults that hold at R10 (with mk-attn).
4. Silo -> port behind a knob -> same-job A/B vs previous mk and fi-graph -> ledger row -> tag. Re-run the breakdown
   after each kept pass.
5. Serving port (mine): after the offline stop rule or when the remaining gaps have no plan. Paged build at the R10
   buckets, dispatch rule for 33-256 requests, conv trace L3 / L4 for stock, fi, mk. Order with mk-attn in MAILBOX.

## 5. Budget (10 GPU-h)
Baseline + breakdown ~1.2; passes ~0.5-1 each (silo on main, A/B on background); final e2e + ablation ~1; serving
port + trace runs ~2-3 (the trace jobs are the largest item: plan them as few jobs).

## 6. Kill / stop
- A pass is dropped when its A/B is within noise (< 5 us needs 3 placements x 2 GPUs) or slower.
- C2 is killed if the silo shows the KV stream on (148 - k) SMs loses more than the GEMM time it hides.
- Effort stop: the brief's rule (offline: mk < best of fi-graph / vLLM at every mk_r10 point in one job; serving: mk
  per-step ratio < fi's on the R10 steps, conv L3 / L4 reported), or the ledgers show no plan left.
