# MK-core ledger (moves to regimebench/kernels/mk/LEDGER.md once that directory is committed)

| id | phase | change | band [H] | result (us per point) | jobs | GPU-h | kept | tag |
|---|---|---|---|---|---|---|---|---|
| P0 | - | baseline hv2g, fi-graph, vllm on `--workloads mk` (20 rounds) | - | fi-graph best at all 28: 787 (R0-b1) .. 2235 (R1-b256); PASS-00.md | 9562 | 0.42 (background) | - | [M] |
| P0-fi | - | fi-graph graph-pdl-best timeline at R1-b1/b16/b64/b256 (per-GEMM targets) | - | fi GEMM 3.0-4.7 us/call T<=16, 5.1-6.7 T>=64 (all 2-CTA); glue 531-727 us/step; PASS-00.md | 9564 | 0.01 | - | [M] |
| P0b | GEMM | xover2: 5 shapes x T 1..256 x {KU, X path, ring depth, cta_group} (SPEC-xover2.md) | - | 120/120 PASS. XTMA -0.3..-1.1 us/GEMM; ring max GU -0.7, LM -25; KU64 worse; CG2 = CG1 at T<=128, wins T=256 (LM -44 us). Best mk sum +6.4 us/layer vs fi at T=1 (fixed per-phase cost). PASS-00b.md | 9566 | 0.01 | KU128, ring max, CG2 only NPAD 256 | [M] |
| v0 | all | first GPU data: hv2g MTN skeleton + xover tcgen05 phases + placeholder attention (SPEC-v0.md); dyn-smem fix 86a16ba | b1-b8 ~830, b64 ~900 | check: accuracy pass (ratio <= 0.54) at 11 points; 872 (b1) / 1030 (b16) / 1700 (b64) / 5813 (b256) vs fi 787 / 1019 / 1407 / 2235. Band missed at b64 (attention 25 us/layer). V0.md. e2e 9578 pending | 9568, 9569, 9572, 9576 | 0.06 | yes (baseline) | mk-core-v0 after 9578 |

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
