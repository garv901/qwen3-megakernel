# SPEC-P2: A1 v1 split-KV attention in mk (MK_ATTN=splitkv)

## Why
- mk attention today is the placeholder: one (token, q-head) item per CTA iteration.
  - At R1-b256 it takes 108 us/layer (job 9593).
  - At b1 with KV >= 1K it is 1.03-5.9x fi (v3 e2e, job 9591).
- A1 v1, standalone, in us/layer including the barrier [M, MK-attn job 9611]:

| point | us/layer |
|---|---|
| kv128 | 3.65 |
| kv1K | 4.19 |
| kv4K | 6.75 |
| kv16K | 16.0 |
| kv32K | 27.2 |
| b8-kv4K | 27.6 |
| b8-kv1K | 12.1 |
| b32-kv1K | 30.1 |
| b64-kv128 | 19.0 |
| b256-kv128 | 63.3 |

  - combine_head takes 1.9-2.3 us at b1.

## Source
- Import the header with `git checkout splitkv-v1 -- regimebench/kernels/common/splitkv` (tag = 0f4ac0f, on the mk-attn branch; the tag is in the shared repo).
  - Never edit those files. Read `splitkv.cuh` (attend, combine_head, Params, the smem contract lines 470-490 and 585+), `plan.py` and `CHANGELOG.md` before you start.
  - Read `probe/mk_attn/a1bench.py` (in /raid/garv901/regime-mk-attn) for the host side: plan, to_torch and tensor_maps.
- `build.py` already has `attn="splitkv"` (`attn_header`). Keep the placeholder path working: it stays the default until the GPU A/B.

## Files
- `regimebench/kernels/mk/mk_step.cu`, `mk_common.cuh`, `build.py`, `NOTES.md`, `regimebench/arms/mk.py`, `tests/test_mk.py`.
- With attn=placeholder, the SASS must have a 0-line diff vs HEAD at all buckets, for MK_TS 0/256 x MK_TSB 0/1.

## Kernel (attn = splitkv; select it with `#ifdef` on a macro that build.py defines, e.g. `MK_SKV 1`)
- **smem**
  - ATTN_BYTES = skv::kSmemBytes (148,032). The scratch base is `sm + OFF_BARS - skv::kBarOff`.
  - That puts A1's 64 B mbarrier tail exactly at [OFF_BARS, OFF_BARS + 64).
  - So under splitkv, mk's own barrier block (tma / mma / LM / TS barriers, tmem slot) moves up by 64 B: it starts at OFF_BARS + 64. BARS_BYTES grows by 64.
  - The bytes [OFF_BARS, OFF_BARS + 64) are never written by mk.
- **OFF_BARS**: at least kBarOff, at least the current max over the layouts, and OFF_BARS ≡ 512 (mod 1024).
  - Then the scratch base is 1024 B aligned (MK-attn asked for that). The dyn smem base is 1024 aligned: check and say how.
  - Mirror this in build.py `dyn_smem(..., attn=)`.
  - Keep the static_assert dyn + static <= 227 KB at every bucket and knob set. Report any config that does not fit.
- All overlays (GEMM ring, B tiles, LM ring, TS ring, TS exchange) stay below OFF_BARS.
  - A1 allows anything in [0, kBarOff) between attend calls. The attention phase must not overlap any in-flight TMA or MMA into that region.
  - Argue this per phase boundary: RoPE → attention and attention → O. Note the prefetch in the barrier window before O: the Wo units land in the GEMM ring during the attention → O barrier, after attend returned. Check the prefetch position relative to the attend call.
- **Params**: build `skv::Params` in the kernel from new trailing kernel args (after `act`):
  - `const int4* items`, `int items_stride`, `const int* pair_info`, `float* part`, `const void* skv_maps`, `int engine`.
  - q = qb ([total_q, 16, 128] bf16 post-norm/RoPE, as the placeholder reads it), kc / vc = mk's caches, cache_rows = mk's rows per layer, n_layers = NLAY, ao = ao, total_q = ntok, item_off = nullptr.
- **ring_phase**: `unsigned ring_phase = 0;` per CTA, before the layer loop. Pass the same variable to every layer.
- **Phase 3**: `skv::attend<NT>(P, l, sm + OFF_BARS - skv::kBarOff, ring_phase)`. CTAs with no items return early without a __syncthreads: check that mk's next code needs no block barrier first, or add one.
- **skv_prefetch / skv_combine**: these are placeholder functions. Under splitkv, skv_prefetch is a no-op unless A1 has an equivalent (it has none).
- **Phase 4 (O)**: units u = rb * 16 + kc (KU 128, one unit per CTA at most: 128 units, 148 CTAs).
  - The B tile of the unit is head kc for tokens 0..NPAD-1. Under splitkv it is staged by `skv::combine_head<NT, NPAD>(P, l, kc, xs, ntok)` into the unit's B slot, instead of the `get` loop over ao.
  - Check the template args, the slot layout (combine_head writes the SW128 K-major B layout of mk's B slot) and the rows >= ntok zero rule against its header comment and `plan.pack_b_slot`.
  - Then `fence.proxy.async.shared::cta` and a __syncthreads before the MMA issue, as the staged path does.
  - Keep MK_KU 64 on the placeholder only (static_assert).
  - The Wo TMA prefetch in the window stays.
- **Pairs with S == 1**: A1 writes ao directly. combine_head handles both cases: read its code.
- **MK_NULL bit 2**: skip attend; then also skip combine_head (stage zeros or skip staging). The output is wrong by design; it must not hang.
- **MK_STAMPS**: unchanged positions.

## Host (arms/mk.py; build.py)
- **Option** `attn` in MkArm: "placeholder" | "splitkv" | "auto".
  - auto = splitkv when every seq has q_len == 1 (A1 v1 limit; pair_info is indexed by token), else the placeholder. That is a per-workload build choice; report which mk points get which path.
  - The default stays "placeholder" in this pass.
- **prepare()**: `pl = plan.plan([(s.kv_len, s.q_len) for s in w.seqs], n_cta=GRID)`, then `pl.to_torch()`.
  - Assert pl.row_base == mk's t_base layout (the fixture packs seq s at sum of the previous (kv_len + 1)) and pl.total_q == ntok.
  - Plan once per prepare: every replay attends over the same visible lengths (the cache does not grow between replays; MK-attn confirmed this condition).
  - Allocate `part` with torch.empty(pl.part_floats) fp32.
- **Tensor maps**: 2 * skv::kNumMaps (16) maps, K then V, box rows 128, 64, .., 1. dims {128, 8, NL * cache_rows}, strides {256, 2048} B, no swizzle; as in `a1bench.py tensor_maps`.
  - Note that the arm's cache has CTX extra zero rows at the end: dims use NL * rows (rows = mk's per-layer rows).
  - Put them in a separate 64 B-aligned device buffer `skv_maps`. Encode them only when pl.engine == 1, else pass 0.
- **Kernel cache key**: include attn.

## Checks (CPU only; never run on a GPU)
- **Build**: all 5 buckets x attn {placeholder, splitkv} x (MK_TS 256, MK_TSB 0 / 1) compile with 0 spill. Report regs, dyn smem, OFF_BARS and the scratch base % 1024.
- **0-diff**: placeholder vs HEAD at all buckets and knob sets above.
- **SASS counts** at NPAD 16 and 256 with splitkv: UTMALDG (2D + 3D), SYNCS, BAR.SYNC, no FFMA growth in the GEMM phases (report the total).
- **Tests**: compile tests; dyn_smem host == kernel (MK_EXPECT_DYN) for splitkv too; plan / row_base consistency on CPU for all `--workloads mk` points (q_len 1 only); the auto selection.
- **pytest**: exit code 0, not piped through tail. Do not commit.
- **Report**:
  - the files and a diff summary, the build table, the SASS counts and the 0-diff results,
  - the arguments (smem overlap per boundary, mbarrier tail, alignment, early-return CTAs, proxy fences, NULL),
  - which mk points use splitkv under auto, and every deviation.
