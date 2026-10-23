# kernels/common: code shared inside the bespoke megakernel `mk`

`mk` is ONE kernel (tcgen05 GEMM phases + split-KV decode attention, 1-256 tokens/step) built by two owner
sessions (`docs/briefs/`). Each directory has ONE owner; the other consumes it read-only at a tagged version.

| path | owner (branch) | consumer | tags |
|---|---|---|---|
| `regimebench/kernels/mk/`, `arms/mk.py` (skeleton, GEMM phases, LM head, barriers, the attention call sites) | MK-core (`mk-core`) | MK-attn | `mk-core-vN` |
| `regimebench/kernels/common/splitkv/` (A1: split-KV attention + combine, behind the interface below) | MK-attn (`mk-attn`) | MK-core | `splitkv-vN` |

## Hand-off rules
1. MK-core wires attention ONLY through the A1 interface below. Until `splitkv-v1` exists it puts a placeholder
   implementation at `kernels/mk/attn_placeholder.cuh` with the same interface (hv2g MTN attention, one item per
   (token, q-head), `kernels/hv2g/hv2g_mtn.cu:194-266`). Tag `mk-core-v0` once the arm runs e2e with the placeholder.
2. MK-attn develops A1 standalone first (its own bench vs the FlashInfer decode kernels), then tests it inside `mk`
   by `git merge mk-core-vN` into `mk-attn` and switching the build to `common/splitkv` (a build flag owned by
   MK-core, e.g. `MK_ATTN=splitkv`). MK-attn never edits `kernels/mk/`; MK-core never edits `common/splitkv/`.
3. MK-attn tags `splitkv-vN` when a version passes its gate: its GPU unit test vs a torch reference, the FP32 oracle
   band at every `mk` point it runs, and an e2e gain at `--workloads mk_kv`. The tag message lists job ids;
   `splitkv/CHANGELOG.md` records each version and any interface change.
4. MK-core imports a tag with `git checkout splitkv-vN -- regimebench/kernels/common/splitkv` and commits it.
   A bug found by one owner in the other's code goes in the finder's LEDGER.md with a repro; the owner fixes it.
5. An interface change needs both owners: the requester writes it in the other's ledger "Requests" section and waits
   for the owner's tag. The user/orchestrator settles disagreements.
6. Nothing in `kernels/hv2/` or `kernels/hv2g/` is edited (hv2 is frozen; hv2g is the study's reference arm). Copy.

## A1 interface contract (draft v0; MK-attn may change it before v1, with the change in CHANGELOG.md)
- **Packaging.** One header `splitkv/splitkv.cuh`. The mk build prepends the header text to its NVRTC source list (and
  hashes it into the arm's `source_sha256`). No global macro dependencies: model sizes come in as
  `constexpr`/template parameters, every macro it defines starts with `SKV_`, and it does not define or rely on
  `H`, `NL`, `QH`, `KVH`, `HD`, `CTX`, `NT`, `NW`, `PPW`, `NSL`, `INFINITY`.
- **KV layout** (the harness fixture, `regimebench/fixture.py:74-89`): contiguous bf16 `[L, cache_rows, KVH=8, D=128]`.
  A sequence owns rows `row_base .. row_base + kv_len + q_len`. The new tokens' K/V rows (after k-norm and RoPE) are
  written by the caller before attention.
- **Work item** = (sequence, kv-head, split). An item reads positions `[p0, p1)` of that sequence's cache once and
  serves both q-heads of the GQA pair and all `q_len` new tokens of the sequence. Causal visibility for new token t:
  positions `< kv_len + t + 1`. The item list (and the split count per sequence, chosen from kv length) is built by a
  host-side planner in splitkv/ that mk's arm calls in `prepare()` (no data-dependent work in `launch()`).
- **Inputs**: q after q-norm + RoPE, `[total_q, QH=16, 128]` (layout fixed in v1). **Partials**: per (token, q-head,
  split) `o[128]` fp32 unnormalised, `m`, `l` fp32, in caller-owned scratch of `skv_scratch_bytes(total_q, n_split_max)`.
- **Combine**: `skv_combine(...)` merges the splits of one (token, q-head) into the attention output that O-proj
  reads (layout fixed in v1: bf16 or fp32 `[total_q, QH*128]`, as the O-proj tcgen05 B/A operand wants it - MK-core
  states its need in MK-attn's Requests before v1). The caller decides where it runs (start of the next phase, or a
  ticket-last-CTA); the barrier or ticket is the caller's.
- **Tests**: `splitkv/test_splitkv_gpu.py` (torch reference at kv 128..32K, q_len 1..16, batch 1..32, GQA pairs)
  and a CPU NVRTC compile test.
