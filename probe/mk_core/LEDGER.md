# MK-core ledger (moves to regimebench/kernels/mk/LEDGER.md once that directory is committed)

| id | phase | change | band [H] | result (us per point) | jobs | GPU-h | kept | tag |
|---|---|---|---|---|---|---|---|---|
| P0 | - | baseline hv2g, fi-graph, vllm on `--workloads mk` (20 rounds) | - | pending | 9562 | <=1.25 | - | [M] when done |
| P0-fi | - | fi-graph graph-pdl-best timeline at R1-b1/b16/b64/b256 (per-GEMM targets) | - | pending | 9564 | <=0.5 | - | |
| P0b | GEMM | xover2: 5 shapes x T 1..256 x {KU, X path, ring depth, cta_group} (SPEC-xover2.md) | - | pending | - | - | - | |
| v0 | all | first e2e: hv2g MTN skeleton + xover tcgen05 phases + placeholder attention (SPEC-v0.md) | b1-b8 ~830, b64 ~900 | pending | - | - | - | mk-core-v0 |

## Requests to MK-attn (2026-10-23, agreed by message)
- A1 v1 output: bf16 `[total_q, 2048]` row-major (O-proj tcgen05 B operand). Input q: bf16 `[total_q, 16, 128]`
  after q-norm + RoPE.
## Requests from MK-attn (2026-10-23, accepted)
- `skv_prefetch(layer, cta, nct)` call site in the QKV phase after the weight TMAs are issued (no-op in the placeholder).
  Smem for it: TBD after Pass 0b; L2 prefetch only until stated.
- `skv_combine` at the start of O-proj, barrier owned by MK-core (default; ticket-last-CTA option later, on data).
