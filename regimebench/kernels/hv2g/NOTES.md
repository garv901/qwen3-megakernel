# hv2g: hv2 generalized (runtime context, many tokens)

Status (2026-10-23): GPU-checked. Jobs 9464/9468 pass every check point (FP32 oracle band); MT1 R0 379.7 us vs hv2 374.9 us (graph, +1.3 %).
Full-grid timings: results/2026-10-22-b200/. Original note follows: every correctness and
speed statement below is a design intent or hypothesis until `scripts/gpu_check_hv2g.py` has run.

## Files
- `hv2g_common.cuh`  constants, helpers copied unchanged from hv2 (default macro set resolved by hand), `attn_chunk`, `attn_merge`.
- `hv2g_mt1.cu`      MT1 kernel `qwen3_step_mt1` (ntok == 1). `hv2g_mtn.cu` MTN kernel `qwen3_step_mtn` (ntok >= 1).
- `build.py`         source assembly, NVRTC builds, descriptors, chunking, RoPE table. `coop.py` NVRTC / cooperative launch helpers.
- Arms: `regimebench/arms/hv2g.py`, `regimebench/arms/hv2_repeat.py` (frozen hv2 binary x N launches, proxy).

## MT1 (stage A): changes vs hv2
Same skeleton (148 x 512, 4 barriers/layer, register-resident weights, TMA GU/down, deferred norm, fused attention + split-K O-proj,
LM head + argmax). Only:
1. runtime args `kvlen`, `cache_rows`, `base_row`; cache is the fixture's packed `[L, cache_rows, KVH, D]`.
2. `rope` arg points at row `position` of the `[max_pos, 128]` table (host adds `position * 128 * 4` bytes). Position = kvlen.
3. new K/V are written into cache row `base_row + kvlen` (replaces the `k_write` / `v_write` outputs).
4. fused attention loops over 128-position chunks (16 warps x 8 positions). Chunk 0 is prefetched in the QKV -> fused window as in hv2;
   chunk c+1 is issued by each warp right after it consumed chunk c's V (V rows are warp / lane private: no extra `__syncthreads`).
   Per-warp online softmax (`attn_merge`), warp states merged as in hv2. New token folded into warp 0, chunk 0 (as hv2).
   Positions >= kvlen are masked (probability 0).
5. No clamp on KV addressing (a `min()` clamp cost 36 B of spills): the last chunk reads up to 127 rows past the last cached one. The host
   allocates every cache with CTX = 128 zero rows after the last layer so these reads are in bounds and finite; they are masked. They may
   see rows of the next layer; a stale L1 copy of such a row is harmless because the only row a later layer writes in that range is its
   own row `kvlen`, which is masked as a cache position (the new token comes from smem).
With kvlen == 128 the executed path is hv2's plus masks that are no-ops.

ptxas (NB=148, sm_100):
- MT1: `Used 128 registers, 46544 bytes smem`, 0 bytes stack, 0 spill stores, 0 spill loads (hv2 itself: 126 regs, 0 spills).
- MTN: `Used 128 registers, 80 bytes cumulative stack size, 38992 bytes smem`, 80 B spill stores / 84 B spill loads (not required to be 0).
Dynamic smem: MT1 134656 B (as hv2), MTN 32768 B.

## MTN (stage B): design
6 barriers per layer: QKV | K-norm + RoPE + cache write | attention | O-proj | GU | down.
- Every GEMV loads its weight tile into registers once per launch and applies it to all tokens, in token blocks whose activations are staged
  in dynamic smem: TB = 8 (QKV, GU, LM head), 4 (O-proj, K = 2048), 2 (down, K = 3072). Weights for a phase are loaded between the arrive
  and the wait of the barrier before it (plain `__ldg`, no TMA, no stage).
- O-proj is its own GEMV (Wo, K = 2048), rows split over CTAs, residual added by the row owner (no atomics, deterministic).
- Attention item = (token, q-head), one CTA per item (items strided over CTAs), 16 warps x 8 positions per 128-position chunk, same
  `attn_chunk` as MT1. q-norm + RoPE of q is recomputed by every warp. All new K/V of the launch are in the cache before the attention
  barrier, so there is no new-token special case; causality is the per-token visible count `t_vis` (= position + 1). Positions past `t_vis`
  are clamped to the last visible row (keeps the non-coherent V cp.async path from caching rows that a later layer of this launch writes) and masked.
- Descriptors (int32): `t_base[t]` first cache row of the token's sequence, `t_vis[t]` visible rows. Own K/V row = `t_base + t_vis - 1`, RoPE
  position `t_vis - 1`. `lm_tok` (local token index) / `lm_seq` (output slot) list the last token of every sequence.
- LM head + argmax only for those tokens, 8 at a time: each embed row is applied to all 8 staged tokens. Known limit: embed (311 MB) is re-streamed
  `ceil(n_seq / 8)` times, so a 256-sequence step streams it 32 times. Per-CTA best per slot goes to `pval/pidx [NB][MAXTOK]`; the last CTA
  (ticket) reduces per slot (ties -> lowest index) and resets ticket + barrier counter. Skipped when `n_lm == 0`, reset always runs.
- MAXTOK = 256 tokens per launch (scratch = 256 x 10240 fp32 = 10.5 MB). Larger steps run as consecutive cooperative launches inside ONE CUDA
  graph (pointer offsets per chunk, chunk-local lm lists); `config["launches_per_step"]` is the count. Weights are therefore re-streamed per chunk.
- Output: logits `[n_seq, V]` fp32 written directly by the kernel; `outputs()` gathers k_new / v_new from the arm's cache.

## Arm
`auto` = MT1 if total_q == 1 else MTN; `mtn` = always MTN; `<variant>-direct` skips the CUDA graph. `prepare` clones the fixture KV into a
packed cache (+ CTX zero rows), builds descriptors and the RoPE table (`max_pos = max visible count`), warms up, captures the launches.
The fixture is never written.

## Known limits (design, unmeasured)
- Long context: one item (or the one MT1 sequence) walks its chunks serially per layer: 32K context = 256 chunks x 28 layers on 16 CTAs in MT1.
  MT1 attention uses only 16 of 148 CTAs (9 slices share one q-head's attention redundantly, as in hv2).
- MTN has 6 barriers per layer (hv2: 4) and no TMA for weights; at ntok = 1 it is expected to be slower than MT1.
- GQA: the 2 q-heads of a kv-head read the same K/V separately (L2 hits at best).
- MTN attention balance: items = ntok x 16 over 148 CTAs; a prefill tail or a ragged mix leaves the CTAs with the long items as the critical path.
- MTN spills 80 B (not tuned).
- Garbage V rows are masked by probability 0; this assumes finite values in the cache (fixture: zeros / randn, so yes).
- Prefill-style steps of one sequence with q_len > 1 write all rows; a launch is idempotent (same rows, same values), so repeat launches are fine.
