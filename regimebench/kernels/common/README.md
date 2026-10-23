# kernels/common: code shared by the bespoke megakernels

Owners: see `docs/briefs/`. Each subdirectory has ONE owner. Other arms consume it read-only, at a tagged version.

| module | owner | consumers | tag |
|---|---|---|---|
| `splitkv/` (A1, split-KV decode attention + combine) | MK-small (branch `mk-small`) | MK-batch (branch `mk-batch`) | `splitkv-v1`, `splitkv-v2`, ... |

## Hand-off rules
1. MK-small tags `splitkv-vN` on branch `mk-small` when the module passes its gate: the FP32 oracle band at every
   owned point it runs on, its own GPU unit test vs a torch reference, and an e2e win at R2 kv >= 4K.
   The tag message lists the job ids. `splitkv/CHANGELOG.md` records each version and any interface change.
2. MK-batch imports only that path: `git checkout splitkv-vN -- regimebench/kernels/common/splitkv` and commits it
   on `mk-batch` ("Import splitkv-vN"). MK-batch never edits these files. A bug found by MK-batch goes in its
   LEDGER.md with a repro; MK-small fixes it and tags the next version.
3. Until `splitkv-v1` exists, MK-batch uses hv2g MTN's attention (`kernels/hv2g/hv2g_mtn.cu:194-266`).
4. Nothing in `kernels/hv2/` or `kernels/hv2g/` is edited (hv2 is frozen; hv2g is the study's reference arm). Copy.

## A1 interface contract (draft v0; MK-small may change it before v1 and must write the change in CHANGELOG.md)
- **Packaging.** One header `splitkv/splitkv.cuh`. hv2g's build has no `#include`: an arm's build.py prepends the
  header text to its NVRTC source list (and hashes it into the arm's `source_sha256`). No global macro dependencies:
  model sizes come in as `constexpr`/template parameters, and every macro the header defines starts with `SKV_`.
  It must not define or rely on `H`, `NL`, `QH`, `KVH`, `HD`, `CTX`, `NT`, `NW`, `PPW`, `NSL`, `INFINITY`.
- **KV layout** (the harness fixture, `regimebench/fixture.py:74-89`): contiguous bf16
  `[L, cache_rows, KVH=8, D=128]`. A sequence owns rows `row_base .. row_base + kv_len + q_len`. The new tokens'
  K/V rows are written by the caller before attention.
- **Work item** = (sequence, kv-head, split). An item reads positions `[p0, p1)` of that sequence's cache once and
  serves both q-heads of the GQA pair and all `q_len` new tokens of the sequence. Causal visibility for new token t:
  positions `< kv_len + t + 1`.
- **Partials**: per (token, q-head, split): `o[128]` fp32 (unnormalised), `m` fp32, `l` fp32, in a caller-owned
  global scratch of size given by `skv_scratch_bytes(total_q, n_split_max)`. The split count is chosen by the caller
  at runtime from kv length.
- **Combine**: `skv_combine(...)` merges the splits of one (token, q-head) into the attention output that O-proj reads.
  The caller decides where it runs (the start of the next phase, or a ticket-last-CTA), and the barrier/ticket is the
  caller's.
- **Tests**: `splitkv/test_splitkv_gpu.py` (torch reference at kv 128..32K, q_len 1..16, GQA pairs) and a CPU NVRTC
  compile test.
