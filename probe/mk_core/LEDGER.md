# MK-core ledger (moves to regimebench/kernels/mk/LEDGER.md once that directory is committed)

| id | phase | change | band [H] | result (us per point) | jobs | GPU-h | kept | tag |
|---|---|---|---|---|---|---|---|---|
| P0 | - | baseline hv2g, fi-graph, vllm on `--workloads mk` (20 rounds) | - | fi-graph best at all 28: 787 (R0-b1) .. 2235 (R1-b256); PASS-00.md | 9562 | 0.42 (background) | - | [M] |
| P0-fi | - | fi-graph graph-pdl-best timeline at R1-b1/b16/b64/b256 (per-GEMM targets) | - | fi GEMM 3.0-4.7 us/call T<=16, 5.1-6.7 T>=64 (all 2-CTA); glue 531-727 us/step; PASS-00.md | 9564 | 0.01 | - | [M] |
| P0b | GEMM | xover2: 5 shapes x T 1..256 x {KU, X path, ring depth, cta_group} (SPEC-xover2.md) | - | 120/120 PASS. XTMA -0.3..-1.1 us/GEMM; ring max GU -0.7, LM -25; KU64 worse; CG2 = CG1 at T<=128, wins T=256 (LM -44 us). Best mk sum +6.4 us/layer vs fi at T=1 (fixed per-phase cost). PASS-00b.md | 9566 | 0.01 | KU128, ring max, CG2 only NPAD 256 | [M] |
| v0 | all | first GPU data: hv2g MTN skeleton + xover tcgen05 phases + placeholder attention (SPEC-v0.md); dyn-smem fix 86a16ba | b1-b8 ~830, b64 ~900 | check: accuracy pass (ratio <= 0.54) at 11 points; 872 (b1) / 1030 (b16) / 1700 (b64) / 5813 (b256) vs fi 787 / 1019 / 1407 / 2235. Band missed at b64 (attention 25 us/layer). V0.md. e2e 9578 pending | 9568, 9569, 9572, 9576 | 0.06 | yes (baseline) | mk-core-v0 (9578: wins 6/28) |

| P1 | barrier | bench: v0 barrier is the cheapest of 7 (1.095 us); MK_PFA: prefetch after arrive | b1 -60..-100 | -66 (b1), -74 (b16), -26 (b64), -30 (b256), -48 (R2-kv8192); latency 1.02 us/barrier after | 9579, 9580, 9582, 9583 | 0.05 | yes (MK_PFA=1) | mk-core-v1 |
| v1 e2e | all | v1 = P1 (b83147c), all 28 points, mk vs fi-graph interleaved | - | mk 796 (b1) / 956 (b16) / 1671 (b64) / 2835 (b128) / 5735 (b256); -63..-86 vs v0 at 27/28 points; b128 +68 vs v0 [H: other GPU, 9578 GPU-623e, 9584 GPU-0192; recheck in the P3 A/B]. Wins vs P0 fi: 11/28 (R1-b1..b16, R2-kv128/512, R3-b8-kv1024, R6-q2/q4/q8); R0-b1 795.6 vs 787 still loses. Accuracy pass at all points. results/2026-10-23-mk/v1 | 9584 | 0 (background) | - | - |
| P3 | LM head | MK_LMP: X staged once, LM ring of LMR units, 2 TMEM accumulators, producer / MMA / epilogue warps (NPAD 16, 32) | b1 -35..-45 | -60 (b1), -68 (b16), -67 (b32), -51 (R2-kv8192), -60 (R6-q4); LM phase 99 -> 44.8 us at b1 (6.9 TB/s); check ALL PASS | 9585 (trap, arm smem), 9586, 9588 | 0.05 | yes (MK_LMP=1, LMR 4 / 3) | mk-core-v2 |
| v2 e2e | all | v2 (7f7aad1), 27 points (R1-b64 failed: source changed under the job) | - | see results/2026-10-23-mk/v2 | 9589 | 0 (background) | - | - |
| P3b | LM head | pipelined LM head at NPAD 64 (LMR 2) | -30..-40 | -36.8 (R1-b64) | 9590 | 0.03 | yes | mk-core-v3 |
| v3 e2e | all | v3 (b5cb2f2, snapshot worktree), all 28 points | - | mk 737 (b1) / 886 (b16) / 1134 (b32) / 1639 (b64) / 2837 (b128) / 5723 (b256); -46..-70 vs v1 at the NPAD <= 64 points, b128 +2.4 / b256 -11 (same LM path). **Wins vs P0 fi: 13/28** (R0-b1, R1-b1..b32, R2-kv128/512, R3-b8-kv1024, R6-q2/q4/q8). Losses: KV >= 1024 at b1 (attention placeholder: 1.03-5.9x), R3-b8-kv4096, R3-b32, R6-q16, R1-b64..b256 (GEMM at L + attention). Accuracy pass at all points. results/2026-10-23-mk/v3 | 9591 | 0 (background) | - | - |
| P5a | QKV (NPAD 256) | tile-stream engine, QKV with B = bf16 xb by TMA; xb by D ticket finalize + prologue; MK_TS = min NPAD (256) | -150..-250 (b256) | -210.8 (R1-b256); b128 +54.2 so off there (D finalize on the critical path, 9603); check 9609 ALL PASS 11/11 (b256 5529 us) | 9602 / 9603 / 9609 | 0.6 | yes (b256 only) | mk-core-v4 |
| P5b | GU / D (NPAD >= 128) | GU and D on the tile stream (act from the GU accumulators; xb2 / ss2 / xb by idle-CTA finalizers on tickets); MK_TS 128 + MK_TSB 1 default. Fixes on the way: closed-form idle rank (a 148-iteration 64-bit division loop cost ~20 us per O / D phase, job 9613), r2 once per token in smem (per-chunk ss2 loads ~8.6 us/layer, job 9618) | -500..-900 (b256) | b256 -957, b128 -167 (job 9622); check 9660 ALL PASS 11/11 (b128 2677, b256 4587 us) | 9612 / 9613 / 9615 / 9618 / 9622 / 9660 | 0.5 | yes | mk-core-v5 |
| P2 | attention | A1 v1 (splitkv-v1) behind attn=splitkv/auto (q_len 1 points); 128-reg cap: spills 48-264 B at NPAD >= 32. Hang: splitkv + MK_TS on + MK_TSB 0 at NPAD 256 (mk barrier, not A1; job 9662) -> static_assert, the default (TSB 1) passes. First points (job 9659): R2-kv8192 2954 -> 946, R3-b8-kv4096 1933 -> 1462 us | - | 9659 / 9662 | 0.3 | auto (kv >= 512) | mk-core-v6 |
| e2e v6 | all | 737b8ef: P5b default + attn auto (splitkv-v1 iff q_len 1 and max kv >= 512; A/B job 9674: kv128 loses to the O combine, R2-kv1024 -211, kv32768 -8059, R3-b1-kv16384 -4026, R3-b32-kv1024 -523). Report results/2026-10-23-mk/v6 (run 20261023T221359Z-mk-v6). vs best SOTA (p0 job 9562, cross-job): **23 / 28 points < 1**; losses R1-b64 1.17 (1643.9 vs 1407.1), b128 1.50 (2660.1 vs 1768.7), b256 2.05 (4572.7 vs 2235.3), R3-b32-kv1024 1.01 (1704.4 vs 1694.4), R6-q16 1.11 (1284.0 vs 1161.7). v3 -> v6: b256 5723 -> 4573, R2-kv32768 9512 -> 1437 | - | 9688 | 0.5 | - | mk-core-v6 |
| P3c | LM head (NPAD >= 128) | LM on the tile stream (ts_phase<PhLm, 128, 1>, A = embed, B = xb = bf16(x * fnorm) from the last layer's idle D finalizers + ssf); run-time fallback to v0 unless lm_tok is the identity; MK_LMTS 1 default | -80..-100 (b128), -150..-200 (b256) | b128 -96.4 [-98.0, -95.9], b256 -215.7 [-217.7, -212.8] (job 9701, 7 rounds); check 9701 ALL PASS (b64 / b128 / b256) | 9701 | 0.1 | yes | (v7) |

## Requests to MK-attn (2026-10-23, agreed by message)
- A1 v1 output: bf16 `[total_q, 2048]` row-major (O-proj tcgen05 B operand). Input q: bf16 `[total_q, 16, 128]`
  after q-norm + RoPE.
## Requests from MK-attn (2026-10-23, accepted)
- `skv_prefetch(layer, cta, nct)` call site in the QKV phase after the weight TMAs are issued (no-op in the placeholder).
  Smem for it: TBD after Pass 0b; L2 prefetch only until stated.
- `skv_combine` at the start of O-proj, barrier owned by MK-core (default; ticket-last-CTA option later, on data).
- Update 2026-10-23 (agreed by message): `skv_prefetch` stays a no-op (MK-attn job 9567: L2 KV prefetch net <= 0 [M]).
  O-proj needs one q-head per CTA (unit u = rb * 16 + kc, kc = head; 128 units, CTA b owns [b*128/148, (b+1)*128/148)).
  A1 v1 provides `skv_combine_head(layer, head, xs, ntok)`: all 512 threads, writes that head for tokens 0..NPAD-1 as bf16
  K-major SW128 into the B slot `xs` (layout of mk_step.cu:96-115 @0be7b19, rows >= ntok zero), 0 B scratch smem.
  Called after the O-proj barrier while the Wo TMA is in flight; MK-core fences proxy.async after it. Global ao
  [total_q, 2048] stays the fallback. Owed by MK-core: the unit rule of the NPAD 256 (cta_group::2) bucket.
- P2 interface v2 (2026-10-23, agreed in principle by message): A1 takes fp32 qkv split-K sums + rr1 + qn/kn + RoPE table +
  t_base/t_vis and does q/k-norm + RoPE + KV write. The last KV split of each (seq, kv-head) owns the new-token rows (only
  writer, uses them from smem). MK-attn estimate [H]: +0.1-0.3 us/layer at b1, +0.3-0.8 at b256. Lands behind a flag; v1
  (post-RoPE bf16 q) stays. MK-core sends the SPEC with exact layouts when P2 starts.
- Update 2026-10-23 (MK-attn correction, splitkv/CHANGELOG.md): attend() does NOT re-init its mbarriers per call (that hung the
  TMA ring at kv >= 16K, jobs 9581 / 9598 / 9599). attend_smem_bytes() = 82,496 B; bytes [0, skv::kBarOff = 82,432) are free between
  calls (GEMM / LM / TS rings, combine_head xs slot); the last 64 B (mbarriers) must stay unchanged for the whole kernel; ring_phase
  starts at 0 per CTA and the same variable is passed to every layer. Scratch base needs 128 B alignment (MK-attn, agreed). MK-core integration rule: place the attention scratch at
  OFF_BARS - 82,432 so its 64 B of mbarriers sit in a reserved hole of mk's barrier region (never overlaid); keep overlays below OFF_BARS.
- MK-attn bench barrier (= hv2g bar_arrive/bar_wait): 1.12 us (job 9567) vs 1.7-2.7 us "wait" in mk v0 stamps: v0 waits
  include imbalance / phase-edge work (P1 bench separates them).
