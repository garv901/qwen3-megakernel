# MK-core ledger (moves to regimebench/kernels/mk/LEDGER.md once that directory is committed)

| id | phase | change | band [H] | result (us per point) | jobs | GPU-h | kept | tag |
|---|---|---|---|---|---|---|---|---|
| P0 | - | baseline hv2g, fi-graph, vllm on `--workloads mk` (20 rounds) | - | fi-graph best at all 28: 787 (R0-b1) .. 2235 (R1-b256); PASS-00.md | 9562 | 0.42 (background) | - | [M] |
| P0-fi | - | fi-graph graph-pdl-best timeline at R1-b1/b16/b64/b256 (per-GEMM targets) | - | fi GEMM 3.0-4.7 us/call T<=16, 5.1-6.7 T>=64 (all 2-CTA); glue 531-727 us/step; PASS-00.md | 9564 | 0.01 | - | [M] |
| P0b | GEMM | xover2: 5 shapes x T 1..256 x {KU, X path, ring depth, cta_group} (SPEC-xover2.md) | - | 120/120 PASS. XTMA -0.3..-1.1 us/GEMM; ring max GU -0.7, LM -25; KU64 worse; CG2 = CG1 at T<=128, wins T=256 (LM -44 us). Best mk sum +6.4 us/layer vs fi at T=1 (fixed per-phase cost). PASS-00b.md | 9566 | 0.01 | KU128, ring max, CG2 only NPAD 256 | [M] |
| v0 | all | first e2e: hv2g MTN skeleton + xover tcgen05 phases + placeholder attention (SPEC-v0.md) | b1-b8 ~830, b64 ~900 | pending | - | - | - | mk-core-v0 |

## Requests to MK-attn (2026-10-23, agreed by message)
- A1 v1 output: bf16 `[total_q, 2048]` row-major (O-proj tcgen05 B operand). Input q: bf16 `[total_q, 16, 128]`
  after q-norm + RoPE.
## Requests from MK-attn (2026-10-23, accepted)
- `skv_prefetch(layer, cta, nct)` call site in the QKV phase after the weight TMAs are issued (no-op in the placeholder).
  Smem for it: TBD after Pass 0b; L2 prefetch only until stated.
- `skv_combine` at the start of O-proj, barrier owned by MK-core (default; ticket-last-CTA option later, on data).
