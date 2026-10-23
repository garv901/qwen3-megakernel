# mk v0: first e2e tcgen05 decode megakernel arm (CPU-verified only: NVRTC compile, SASS, pytest; NOTHING has run on a GPU)

Status 2026-10-23: written against `probe/mk_core/specs/SPEC-v0.md`. Every correctness and speed statement below is a design intent
until `scripts/gpu_check_mk.py` has run.

## Files
- `mk_common.cuh`  constants, knobs, bf16 helpers, tcgen05 / TMA / mbarrier helpers copied from `probe/xover/xover.cu` (GPU-verified, job 9521),
  grid barrier from `hv2g_common.cuh`, stamps, and the engine wrappers (below).
- `mk_step.cu`     kernel `mk_step`: phase descriptors, `geom` (unit ownership), `gemm_phase` (one templated function for QKV, O, GU, D, LM), the layer loop,
  LM head and argmax ticket.
- `attn_placeholder.cuh`  A1 interface (`skv_prefetch`, `skv_attention`, `skv_combine`), hv2g MTN attention with q from bf16 `qb`, output bf16 `ao`.
- `build.py` source assembly (attention header, then `mk_common.cuh`, `mk_step.cu`), NVRTC per bucket (cached), tensor-map specs / encode, `strip_knobs`.
  `coop.py` copy of hv2g `coop.py` (default arch `sm_100a`). Arm: `regimebench/arms/mk.py`. Tests: `tests/test_mk.py`. GPU check: `scripts/gpu_check_mk.py`.

## Design as built
148 CTAs x 512 threads, cooperative launch, one launch per step, CUDA graph in the arm. `NPAD` in {16, 32, 64, 128, 256} compile-time (cubin per
bucket, entry `mk_step`), runtime `ntok <= NPAD`. TMEM 256 columns by warp 0 at start, freed at the end. 6 grid barriers per layer + 1 prologue
(hv2g `bar_arrive` / `bar_wait`, counter reset by the last argmax-ticket CTA).
- GEMM phase = xover ENG=1: weight unit (128 rows x `MK_KU` k, TMA 2D SW128 boxes of 64 k, ring of `MK_RING` slots, one mbarrier per slot) is A, the staged token
  tile (NPAD rows, K-major SW128, 2 slots, all NPAD rows written every unit, rows >= ntok zero) is B; `KU/16` MMAs per unit into TMEM column 0,
  commit -> mbarrier; at the end of a row-block segment all warps read TMEM (`tcgen05.ld 32x32b.x16`, quadrant = warp % 4) and `red.add` (t < ntok).
  Split-K by unit ranges: u = rb * KC + kc, CTA b owns [b*U/148, (b+1)*U/148); LM head: whole row blocks (1187 over 148 CTAs, K = 1024 in TMEM).
- Barrier-window prefetch: after the last epilogue of a GEMM phase tid 0 issues the first `min(RING, n)` units of the next GEMM phase
  (QKV -> O at the end of phase 1; O -> GU; GU -> D; D -> next layer QKV or LM; prologue -> layer 0 QKV).
- Staging functors per phase (all read global via `__ldcg`): QKV `bf16(x*ln1)`, O `ao`, GU `bf16(x*ln2)`, D `bf16(silu(rr2*gate) * (rr2*up))`,
  LM `bf16(x[lm_tok]*fnorm)`. Deferred norms: per-token `rr1`, `rr2` computed by warps 8..15 of CTA b for t = b + 148 i while unit 0's MMA runs
  (`extra()` hook; also zeroes `gu` in phase 1 and `qkv` in phase 4, both also in the prologue).
- Phase 2 (one warp per (token, head), 16 q + 8 kv heads): q/k norm + RoPE; q -> `qb` bf16, k, v -> cache rows (`v * rr1`).
- Phase 3: `skv_attention` (placeholder, smem overlaid on the B-tile slots, which are idle in phases 2 and 3). Phase 4 start: `skv_combine` (no-op).
- Buffers (arm scratch): `x`, `qkv`, `gu` fp32 [256][...], `qb`, `ao` bf16, `rr` = rr1 | rr2 (2 x 256 fp32), `sync`, `pkey`/`pidx` [148][256].
- Smem: dynamic `1024 + RING*UB + max(2*XS, 41088) + 256` (NPAD=256: 197888 B) + 11280 B static (LM tables).

## Engine wrappers (so that a later `MK_CG=2` build per NPAD bucket is a local change; NOT implemented)
All tcgen05 / TMA / TMEM calls of the phase code go through `mk_tmem_alloc`, `mk_tmem_dealloc`, `mk_mma_unit`, `mk_commit`, `mk_tmem_read16`, `mk_tma_unit`
in `mk_common.cuh`. Unit numbering and row-block ownership: `geom<PH>()` and `PH::src` in `mk_step.cu`. TMEM lane -> output row: `mk_row_of`.
A CG=2 build (cluster of 2, M = 256 swap-AB, as `probe/mk_core/xover2/xover2.cu` arm a5 / pi05 `h_bench.cu`) would have to change:
1. wrappers: `tcgen05.*.cta_group::2` forms, MMA issued by one CTA of the pair with M = 256 (A = 2 x 128 weight rows, one half loaded by each CTA), commit with
   `.multicast::cluster` mask, alloc/dealloc `cta_group::2`; idesc M field 256; `mk_row_of` / TMEM lane mapping per CTA half.
2. `geom` / `PH::src`: a unit covers 256 rows (row block index halves, U halves), split of ownership per cluster, `fill_unit` loads its half;
   B tile must be visible to both CTAs (each stages half the tokens' k? or both stage fully) and the commit/ring mbarriers need cluster-scope arrives.
3. launch: cluster dimension attribute (`cuLaunchKernelEx` with `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION` instead of `cuLaunchCooperativeKernel`; cooperative
   attribute too) in `coop.py` / `build.py` (add `-D MK_CG=2`, a cluster field in the cubin key); grid barrier needs all 148 CTAs (74 clusters) co-resident.
4. epilogue: two accumulator halves (CTA r reads its 128 rows), `red.add` index math; LM row-block partition in pairs (1187 is odd: one tail block).
5. `MK_CG` is today a `constexpr` in `mk_common.cuh` only (not a macro, not selectable).

## Deviations from SPEC-v0 (each with reason)
1. q/k norm are applied to `rr1 * qkv` (exact), not on the unscaled qkv with the cancellation trick: one extra FMUL per element, and the eps term no longer
   changes by rr1^2. The oracle band should not show it.
2. `rrf` is not a global buffer and not computed by a separate pass: the LM B-tile staging of each CTA's FIRST row block (it stages the full x rows of every
   lm token exactly once per row block) accumulates sums of squares in smem (shuffle reduce inside the row's 16 lanes, one smem atomicAdd per row chunk);
   `rrf = rsqrt(ss/1024 + eps)` is applied in the epilogue. Reason: avoids 1 MB L2 reads per CTA and a stall (the SPEC's "every CTA computes the ones it
   needs"). Summation order of the atomics is not fixed (fp32, ~1 ulp).
3. Tensor maps: encoded on the host with the real device addresses (pointers are fixed for a step), stored in one 1 KB device tensor (8 x 128 B), used through a
   generic global pointer; thread 0 executes `fence.proxy.tensormap::generic.acquire.gpu [map], 128` for each map once (form of dense_tc `map_patch`; the in-kernel
   `tensormap.replace` patching is not used). Unverified on a GPU that the acquire fence without an in-kernel modification is accepted / needed.
4. `MK_XTMA` exists as a macro but only 0 is implemented (static_assert).
5. Extra knobs (not in the SPEC): `MK_STG` (staging chunks in flight, default 4) and `MK_EPI2` (TMEM read of 2 chunks per warp iteration, default 0). With
   `MK_EPI2 = 1` ptxas spills (4..200 B depending on the bucket), with 0 there is no spill at all. Cost: the TMEM-read loop has less memory-level parallelism.
6. Argmax uses order-preserving u32 keys of the fp32 logits (`__reduce_max_sync`, `__ballot_sync`); `pkey` therefore holds keys, not floats.
7. `attn_placeholder.cuh` uses the hv2g `attn_chunk` with the NEWTOK branch removed (never used in MTN) and `sc - m` guards as hv2g; q is rounded to bf16 (A1 input).
8. NVRTC/ptxas keep a compile cache that returns an EMPTY log on a hit; `build.py` sets `CUDA_CACHE_DISABLE=1` (setdefault) so the ptxas table is always filled.
9. Stamps: one extra stamp at the end of the LM phase; array `mk_stamps[148][346]` (`MkStep.stamps()` reads it with `cuModuleGetGlobal`, unverified).

## ptxas (NVRTC 13.0, sm_100a, defaults: KU 128, RING 2, STG 4, EPI2 0; `-Xptxas -v`), all buckets: 0 spill, 0 stack, 1 barrier
| NPAD | regs | static smem B | dyn smem B | total B |
|---|---|---|---|---|
| 16 | 85 | 11280 | 79232 | 90512 |
| 32 | 85 | 11280 | 83200 | 94480 |
| 64 | 118 | 11280 | 99584 | 110864 |
| 128 | 120 | 11280 | 132352 | 143632 |
| 256 | 104 | 11280 | 197888 | 209168 (limit 232448) |
Knob builds (`MK_STAMPS=1`, `MK_NULL=0x7f`, `MK_KU=64`, `MK_RING=3|4`, NPAD 64) compile with 0 spill (tests).

## SASS (`cuobjdump -sass`), NPAD 16 | 256
UTCHMMA 40 | 40 (5 GEMM phases x 8 K16 steps), UTCBAR 5 | 5, LDTM 5 | 5, UTCATOMSWS 3 | 3, HMMA (non-UTC) 0 | 0, REDG.E.ADD.F32 64 | 64, UBLKCP/UTMALDG 22 | 22.
FFMA 468 | 1686, FMUL 305 | 1055, MUFU 98 | 322. FFMA split (builds with phases nulled): attention + phase 2 = 178 (both buckets: it does not depend on NPAD);
the rest (290 | 1508) sits inside the GEMM-phase functions but is staging / epilogue arithmetic that scales with the unrolled chunk count (SiLU `expf` + divide of the
D phase, norm products, LM rsqrt), not an FFMA dot product: the GEMM math is only UTCHMMA. Strictly "FFMA only outside the GEMM phases" is therefore NOT true
of the whole function bodies; it is true of the GEMM math.
Knob-off identity: `tests/test_mk.py::test_knob_off_sass_identical_to_stripped_source` compiles the source and the source with the knob code removed
(`build.strip_knobs`: `#if MK_STAMPS` blocks deleted, `mk_null(p)` -> `false`) for NPAD 32 and compares `cuobjdump -sass` with addresses removed: identical
(`MK_STAMPS=1` differs, as expected).

## Open risks for the GPU run (ranked)
1. TMEM / mbarrier parity across phases: ring slot f % RING and parity (f / RING) & 1 run over the whole launch; MMA mbarriers of units that are not at a segment
   end are never waited (as xover); the next phase's prefetch relies on the last commit implying all earlier MMAs are done. A hang here shows as a stuck launch.
2. Tensor maps in global memory (generic pointer, acquire fence form untested for host-written maps) and 128-byte alignment of the maps tensor (asserted).
3. red.add ordering: x, qkv, gu are accumulated in fp32 in an unfixed order (repeat launches differ at ~1e-6 relative; the check allows 1e-2 absolute on logits).
4. B-tile smem overlay: attention scratch lives on the B-tile slots; correct only because every staging call writes all NPAD rows (pad rows zeroed each unit).
5. LM-head cost: 9 row blocks per CTA, each restages x (8 units x NPAD x 128 k) and the epilogue does a warp-wide reduce per slot; expected slow at NPAD 256.
6. qkv/gu zeroing + red.add visibility relies on barrier release/acquire only (no extra fences), as hv2g scratch.
7. Cooperative launch co-residency: 209168 B smem at NPAD 256 plus 128 regs x 512 threads; one CTA per SM.
