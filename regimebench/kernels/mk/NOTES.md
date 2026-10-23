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

## LM head, pipelined (MK_LMP, SPEC-lm; CPU-verified only: compile, SASS, pytest)
Knobs `MK_LMP` (default 1, job 9586) and `MK_LMR` (LM ring slots, default 0 = 4 at NPAD 16, 3 at 32, 2 at 64). `LMP = MK_LMP && NPAD <= 64` (NPAD 64 since P3b: the 12 epilogue warps loop over the 4 column chunks, `LM_EPI_WARPS` = accempty count); at NPAD >= 128 MK_LMP=1 keeps the v0 `gemm_phase<PhLm>` path
(`if constexpr`). MK_LMP=0: SASS identical to HEAD for all 5 buckets (checked by building `git show HEAD:` sources; also `test_lmp_off_sass_identical`).
Code: `lm_fill`, `lm_first`, `lm_phase` in `mk_step.cu`; constants `LM_RING_OFF`, `LMB_*`, `BARS_BYTES` in `mk_common.cuh`; `build.py:dyn_smem(..., lmp, lmr)`.
- Smem (from the 1024-aligned base): `[XL: 8 x XS][LM ring: LMR x 32 KB]`, overlaying every other region; `OFF_BARS = max(v0 layout, LM layout)`; 384 B of barriers
  (v0 bars at +0/+64, tmem slot +128, LM: full[LMR] +192, empty[LMR] +256, accfull[2] +320, accempty[2] +336). DYN_SMEM at NPAD 16 LMR 4: 165248; NPAD 32 LMR 4: 198016.
  Static smem 11280 B + dynamic must be <= 227 KB (static_assert): NPAD 16 allows LMR <= 5, NPAD 32 LMR <= 4 (SPEC asked to compile LMR 6 at NPAD 16: 230784 B, does not fit).
  MK_LMP needs MK_KU = 128.
- TMEM: 2 accumulators of NPAD columns (buffer b at column b * NPAD) inside the existing 256-column allocation.
- Roles: tid 0 producer (also issues the window prefetch), tid 32 MMA issuer, warps 4 .. 4 + 4 NPAD/16 - 1 epilogue (warp w: quadrant w & 3, column chunk (w >> 2) - 1;
  one warp per (quadrant, chunk) keeps the single-writer property of s_bk / s_bi; row blocks ascend, so strict > keeps the lowest index). Other warps idle until the final
  `__syncthreads`. accempty count = 4 * NPAD / 16 (the working epilogue warps only; idle warps must not arrive, an early arrival would pollute a later phase).
- X: all 512 threads stage all 8 k-chunks once (get math of v0, same SW128 layout as `stage_tile`), s_ss accumulated once per element, `fence.proxy.async`, `__syncthreads`.
- MK_NULL bit 6: tid 0 waits full[j % LMR] and refills (no staging / MMA / epilogue). Stamp: unchanged (after the join `__syncthreads`).
Arguments:
1. Parity. Every LM barrier is re-initialised in the prologue and used only in the LM phase. Convention as v0: use number u of a barrier is awaited with parity u & 1.
   full[s] use u = j / LMR (j = unit, s = j % LMR): MMA waits (j / LMR) & 1. empty[s]: the MMA of unit j - LMR commits it, so unit j >= LMR waits use (j / LMR - 1), parity
   ((j / LMR) - 1) & 1. accfull[b]: row block i (b = i & 1) is use i / 2, epilogue waits (i / 2) & 1. accempty[b]: the MMA waits at kc == 0 of row block i >= 2 for use i / 2 - 1.
2. First use. empty[]: no wait on first use (the producer starts at j = LMR; the first LMR units are filled by the window prefetch, whose slots have no previous LM use).
   accempty[b]: no wait for row blocks 0 and 1 (`i >= 2`). full[] / accfull[]: the wait is on the first phase (parity 0), nothing to skip. Counts: full 1 (arrive.expect_tx; TMA
   adds the tx bytes), empty 1 and accfull 1 (tcgen05.commit), accempty 4 NPAD / 16 (lane 0 of each epilogue warp after `tcgen05.wait::ld`, `fence::before_thread_sync`, `__syncwarp`).
3. Overlap with in-flight TMA. The window prefetch writes the LM ring, which overlays the GEMM ring, B tiles and attention scratch. Nothing of those can be in flight: every earlier
   phase's last unit ends with the MMA-commit wait (all earlier MMAs done; MMA completion is in order), every TMA of the D phase was waited by its MMA, the epilogue TMEM reads and
   the B-tile reads (generic and async) are finished by the last `__syncthreads` of `gemm_phase`, which both tid 0 (MK_PFA 0) and the `bar_arrive` `__syncthreads` (MK_PFA 1) follow.
   The v0 window prefetch is replaced, so no other TMA exists then. Generic-proxy writes into that smem (B tile, attention scratch) were fenced with `fence.proxy.async` and
   consumed before. XL staging (generic writes, XL = [0, 8 XS)) is disjoint from the ring ([8 XS, ..)) that the prefetch is filling; the staging's earlier readers/writers are covered by the
   same argument. The producer reuses a slot only after its empty barrier (MMA done with the slot). XL is never rewritten after the `fence.proxy.async` + `__syncthreads`.
4. TMEM lane quadrant. Epilogue warp w reads TMEM lanes 32 (w & 3) .. +31 only (`mk_tmem_read16(..., q = warp & 3, ...)`), as v0; every quadrant has 1 (NPAD 16) or 2 (NPAD 32) working warps.
   TMEM alloc / dealloc by warp 0 (all lanes) as before; the dealloc follows the join `__syncthreads`, after the last accfull wait (all MMAs and loads finished).
5. Buffer reuse: the MMA of row block i + 2 (same buffer) starts only after accempty[b] = all epilogue warps read buffer b of block i (first MMA overwrites: `kc == 0` -> accumulate flag off).
SASS (NPAD 16, MK_LMP 0 | 1; NPAD 32 identical to 16): UTCHMMA 40 | 40 (4 GEMM phases x 8 + LM 8 K16 steps), UTCBAR 5 | 6 (LM: empty + accfull commit), LDTM 5 | 5, UTMALDG 22 | 22,
UTCATOMSWS 3 | 3, SYNCS (mbarrier ops) 45 | 60 (ARRIVE 11 | 12, PHASECHK 30 | 32, EXCH (init) 4 | 16).
Build (regs / spill / dyn smem): see the report; all 5 buckets 0 spill for MK_LMP 0 and 1.

## Tile-stream QKV (MK_TS, SPEC-P5a; CPU-verified only: compile, SASS, pytest; NOTHING has run on a GPU)
Knob `MK_TS` (default 0). `TS = MK_TS && NPAD >= 128 && MK_KU == 128` (constexpr, `mk_common.cuh`). With TS the QKV GEMM runs on the tile-stream engine ported from
`probe/mk_core/xover3/xover3.cu k_ts` (SB = 0, GPU-verified in jobs 9592 / 9595) with B = bf16 `xb` loaded by TMA; O, GU, D and the LM head are unchanged (P5b).
MK_TS = 0: SASS identical to commit 6f1e1cc (the last one before this work) at all 5 buckets; MK_TS = 1 at NPAD <= 64 (and with MK_KU = 64): identical to MK_TS = 0
(`test_ts0_sass_identical_to_head`, `test_ts_off_sass_identical`). Keeping TS = 0 identical needed the TS code in `if constexpr` branches that DEFINE the hook lambda inside the branch
(a `fin` lambda defined outside the branch changed register allocation in the prologue).
- Data. `xb` bf16 [MAXTOK][HID] (arm: torch.zeros) = bf16(x * ln1[l]) (the value the v0 QKV `get` stages, before rr1); rows >= ntok keep their allocation zeros. `tk` u32 [8] tickets (zero at
  allocation; the last user of a row block resets its ticket). Both are the LAST two kernel arguments (after `cache_rows`), see Deviations. Tensor map 8 = xb: dims {HID, MAXTOK}, stride 2048 B, box {64, 128}, SW128
  (`build.py: workspace_map_specs`, `encode_maps(..., workspace={"xb": ptr})`; the arm appends the 128 B only for MK_TS builds; kernel `NMAP = TS ? 9 : 8`). QKV still red.adds UNSCALED sums into qkv; RoPE phase applies rr1.
- Writers of xb. (1) Prologue (TS only): each thread writes xb[i] = bf16(embed-x * ln1[0]) for its own i (same strided loop as x), `fence.proxy.async.global`, then the grid barrier arrive.
  (2) D phase of layer l (`gemm_phase<PhD>` gets a `fin(rb, nk)` hook after the epilogue of every row-block segment; nk = units of rb flushed by this CTA, tracked as `j - jseg + 1`): `__syncthreads`; tid 0
  `__threadfence`, `old = atomicAdd(&tk[rb], nk)`, last = (old + nk == 24); broadcast through the static `s_flag`; if last (and l + 1 < NLAY): `__threadfence`, all 512 threads `__ldcg` x[t][rb*128..+127] for t < ntok, write
  xb = pack_xw(x, ln1[l+1]) (16 B stores, 16 per row, 8 fp32 per item); `fence.proxy.async.global` by every writer thread; tid 0 `atomicExch(&tk[rb], 0)`. For l + 1 == NLAY the hook does nothing (tk stays 0).
- Geometry. Tile t = (rb * NTT + tt) * KS + ks, NRB = 32 (PhQkv), NTT = NPAD / 128, (TT, KS) = (128, 2) at NPAD 256 and (128, 4) at NPAD 128 (`TS_TT`, `TS_KS`), CTA b owns [b NTILE / 148, (b + 1) NTILE / 148)
  (NTILE = 128 in both buckets: 128 CTAs get one tile (4 or 2 k-steps), 20 get none). A = `PhQkv::src` rows, k = ks * 1024 / KS + j * 128; B = map 8 box (k, tt * TT), 2 boxes per step.
- Smem. Ring of `TS_R` slots [A 32 KB | B TT * 256 B = 32 KB] = 64 KB from the base, `TS_R = min(8, (227 KB - 11280 - 1024 - 384) / 64 KB)` = 3. `OFF_BARS = max(v0 layout, LM layout, TS ring)`; `BARS_BYTES = 384` for LMP or TS.
  DYN_SMEM with TS: 198016 at NPAD 128 (was 132352) and 256 (was 197888); `build.py: dyn_smem(..., ts)` / `ts_ring` mirror it (MK_EXPECT_DYN static_assert), plus static_asserts `DYN_SMEM + 11280 <= 227 KB` (TS), `2 TS_TT <= 256` (TMEM),
  `TS_R >= 2`. Barriers reuse the LM block (LMP is NPAD <= 64, TS is NPAD >= 128, never both): full[R] +192 (count 2), empty[R] +256 (1), accfull[2] +320 (1), accempty[2] +336 (12 = the epilogue warps).
  TMEM: the existing 256-column allocation = 2 accumulators x 128 columns (buffer b at column b * TT). Max column 255.
- Roles (`ts_phase`; after the grid barrier). tid 0 producer: `fence.proxy.async.global`, B boxes of the window-prefetched steps, then for j >= R: wait empty[j % R] (parity ((j / R) - 1) & 1), A + B of step j; finally a drain wait of
  empty for the last min(R, nst) steps. tid 32 MMA issuer (8 x K16 per step, N = 128, commit empty[s] per step and accfull[b] per tile). Warps 4..15 epilogue (quadrant w & 3, chunks (w >> 2) - 1 + 3m of 8; chunk
  skipped when its first column >= ntok; `epi(row, tt * 128 + 16 c, v)` = the v0 QKV red.add into qkv). Warps 2, 3 idle. Join: `__syncthreads`, `tc_fence_after`; then `extra()` (rr_of_x, zero gu) by all threads, as v0.
  MK_NULL bit 0 with TS: producer + full waits only (no MMA, no epilogue, `extra()` skipped as in the v0 NUL path).
- ts_first (replaces `prefetch_next<PhQkv>` in both MK_PF_ARRIVE uses): tid 0 re-inits the TS barriers, `fence.mbarrier_init`, issues A boxes of the first min(R, nst) steps (steps 0..R-1 in slots 0..R-1).
  The GEMM-ring barriers / `g.F` are not touched by the TS phase (no counter drift: QKV no longer advances `g.F`; fills and waits of the GEMM ring stay consistent with each other).
Arguments:
1. Parity. Every TS barrier is re-initialised in `ts_first` and used only in the following ts_phase, so counters are phase-local: step j (slot j % R, use j / R), tile ti (accumulator b = ti & 1, use ti >> 1).
   full[s]: the MMA waits (j / R) & 1 (count 2: A expect_tx arrival in the window, B expect_tx arrival after the grid barrier; tx bytes may go transiently positive, never negative, and a phase only completes with both arrivals + all bytes;
   an early A completion cannot flip the phase). empty[s]: the MMA commit of step j - R is use j / R - 1; the producer waits ((j / R) - 1) & 1 for j >= R (never for j < R: first use). accfull[b]: the epilogue waits (ti >> 1) & 1. accempty[b]:
   the MMA waits ((ti >> 1) - 1) & 1 for ti >= 2 (never fires today: nt <= 1; kept for generality). accempty count 12 = warps 4..15 (each lane 0 after `wait::ld` + `fence::before_thread_sync` + `__syncwarp`).
2. Re-init is legal. At `ts_first` (window after the D phase of layer l - 1, or the prologue) the previous TS phase finished a whole layer ago: it ended with the join `__syncthreads`, after (a) the epilogue warps' last accfull wait (all MMAs and
   TMEM loads done), (b) the producer's drain wait on the last use of every empty[] (the last commit arrivals have landed; the final accempty arrivals were executed by the epilogue warps before the join), (c) every TMA of the phase was consumed by an MMA
   that waited its full barrier, so no complete_tx is pending. Between, the GEMM phases wait only their own barriers (+0 / +64), never the TS block, and no thread waits on a TS barrier outside ts_phase. So nothing waits on, arrives at or completes
   bytes on a TS barrier when tid 0 re-inits it. The prologue init is the first init. `fence.mbarrier_init.release.cluster` precedes the first TMA expect_tx on the new barriers (same thread); the other threads first touch them after the grid barrier's `__syncthreads`.
3. Overlap with in-flight TMA (first fill of the ring, as "LM head, pipelined" argument 3). The window fill writes slots 0..R-1 = [0, 192 KB), which overlays the GEMM ring, B-tile slots and attention scratch of the previous phases. In the D-window nothing
   can be in flight there: every unit filled in D (and in its window prefetch) was consumed by an MMA that waited its full barrier; the last unit's segment-end waits the MMA commit (all earlier MMAs complete: in-order); epilogue TMEM reads and generic
   B-tile reads/writes finished before the last `__syncthreads` of `gemm_phase` (also before the `bar_arrive` `__syncthreads` of MK_PFA = 1; with MK_PFA = 0 the loop-end `__syncthreads`). The B tile's generic writes were released with `fence.proxy.async`
   in `stage_tile`. The D-phase finalize (`fin`) uses no dynamic smem (only static `s_flag`). In the prologue nothing exists yet. Inside the phase a slot is refilled only after empty[s] (all 8 MMAs of its previous step read it). After the phase,
   the O-phase window prefetch (`prefetch_next<PhO>`) comes after the join and the drain wait: no TS TMA is in flight. The barrier block (OFF_BARS..) is not overlaid.
4. TMEM lane quadrant. Epilogue warp w reads lanes 32 (w & 3) .. +31 via `mk_tmem_read16(v, tb + b * TT, q = warp & 3, c0)`; warps 4..15 cover each quadrant 3 times with disjoint chunk sets {g, g+3} + 6m, so each accumulator element has one reader
   (and one red.add issuer). MMA writes column range [b * 128, b * 128 + 128) of both accumulators; the first MMA of a tile has accumulate = 0. TMEM alloc / dealloc by warp 0 unchanged (256 columns).
5. Generic-to-async proxy for xb. Generic stores to xb (prologue per-thread, D-hook in the last CTA) are followed by `fence.proxy.async.global` in every writing thread, then the grid barrier arrive (`__syncthreads` + `red.release.gpu` by tid 0). The TMA
   reader (tid 0 of each CTA) runs `fence.proxy.async.global` after the barrier's acquire and before the first B box. Rows >= ntok are never written (allocation zeros). A later overwrite of xb (D hook of layer l + 1) cannot race with a TMA of layer l + 1's QKV:
   at least 4 grid barriers lie in between and each CTA's QKV loads are all complete before it arrives (the drain wait + the MMAs consumed every load).
6. Ticket memory ordering. Producer side (every CTA): red.adds of all 512 threads for row block rb, `__syncthreads` (CTA-scope barrier orders them before tid 0), tid 0 `__threadfence` (fence.sc.gpu: release, cumulative over what tid 0 has observed), then
   `atomicAdd(&tk[rb], nk)`. The ticket value reaches 24 only after all 24 units' atomics; an RMW chain, so each CTA's release fence synchronizes with the atomic of the last CTA (release sequence). Consumer side (last CTA): tid 0 `__threadfence` (acquire
   side) after the atomic, `__syncthreads` broadcasting `s_flag`, then the other threads' `ld.global.cg` of x see every other CTA's red.adds (red.add is performed at L2; `.cg` bypasses L1, so no stale line). This CTA's own red.adds are ordered by the same
   `__syncthreads` + fence. Then the xb stores + proxy fence + (grid barrier release). The ticket is reset by `atomicExch` after the last CTA's loads/stores; all 24 atomics of this layer are done by then, and the next use is in the next layer's D phase (grid barriers in between).
   The `fin` hook is block-uniform (the segment-end condition depends only on j / u), so its `__syncthreads` are legal.

### Build (ptxas, NVRTC 13.0 sm_100a, default knobs; MK_TS 0 | 1)
| NPAD | regs 0 | regs 1 | spill | dyn smem 0 | dyn smem 1 | total 1 (limit 232448) | ring R |
|---|---|---|---|---|---|---|---|
| 16 | 86 | 86 | 0 | 165248 | 165248 | 176528 | - |
| 32 | 80 | 80 | 0 | 165248 | 165248 | 176528 | - |
| 64 | 113 | 113 | 0 | 198016 | 198016 | 209296 | - |
| 128 | 115 | 122 | 0 | 132352 | 198016 | 209296 | 3 |
| 256 | 113 | 123 | 0 | 197888 | 198016 | 209296 | 3 |
MK_TS = 1 with MK_NULL 1 / 0x7f / MK_STAMPS 1 compile with 0 spill (tests).
SASS (NPAD 256, `cuobjdump -sass`; HEAD 6f1e1cc | MK_TS 1): UTCHMMA 40 | 40 (NPAD 128: 40 | 48: the k loop of SPT = 2 steps is unrolled), UTMALDG.2D 22 | 26, UTCBAR 5 | 6, LDTM 5 | 6, REDG.E.ADD.F32 64 | 80,
UTCATOMSWS 3 | 3, `SYNCS` (mbarrier ops) 45 | 72, `FENCE.VIEW.ASYNC.S` (fence.proxy.async.shared::cta) 11 | 11, `FENCE.VIEW.ASYNC.G` (fence.proxy.async.global) 0 | 3 (prologue, producer, D hook), MEMBAR.SC.GPU 2 | 4 (ticket
release + acquire), ATOMG.ADD 1 | 2.
Deviations from SPEC-P5a: (1) `xb` and `tk` are the LAST two kernel arguments (after `cache_rows`), not "after gu": inserting arguments in the middle moves the constant-bank offsets of every later argument and breaks the required
0-line SASS diff at MK_TS = 0 (and at NPAD <= 64 with MK_TS = 1). (2) `NMAP = TS ? 9 : 8` (a 9-map acquire loop would change MK_TS = 0 SASS). (3) `ts_phase` takes `ntok` and does the join + `tc_fence_after` itself; the QKV `extra()` is called by the
caller after it. (4) `fin` is defined inside the `if constexpr (TS)` branch and the TS = 0 branch calls `gemm_phase` exactly as before (register-allocation identity). (5) Epilogue chunks with first column >= ntok are skipped (no TMEM read, no red.add; they would have been masked).
(6) The producer drains the last min(R, nst) empty barriers before the join (needed for the re-init argument; k_ts never re-inits).

## GU and D on the tile stream, idle-CTA finalizers (MK_TSB, SPEC-P5b; CPU-verified only: compile, SASS, pytest; NOTHING has run on a GPU)
Knob `MK_TSB` (default 0; arm option `tsb`). `TSB = TS && MK_TSB` (so it needs MK_TS on at the bucket: NPAD >= TS_MIN, MK_KU 128). MK_TSB = 0: SASS identical to commit b7a78e2 at all 5 buckets x MK_TS 0 / 128 / 256 (checked, and
`test_tsb0_sass_identical_to_head`); MK_TSB = 1 where TS is off (NPAD <= 64, NPAD 128 with MK_TS 256) is identical to MK_TSB = 0.
- Phases with TSB: QKV (P5a) | attention | O = `gemm_phase<PhO>` (v0 engine) + ticket hook + idle-CTA finalizer (xb2, ss2) | GU = `ts_phase<PhGu, GU_TT, 1, map 9>` (A = map 10 wgu, epilogue writes act) | D = `ts_phase<PhD, 128, D_KS, map 11>` + tickets
  + idle-CTA finalizer (xb for the next QKV). Configs (TT / KS): NPAD 256: QKV 128 / 2, GU 128 / 1 (96 tiles), D 128 / 8 (128 tiles); NPAD 128: 128 / 4, 64 / 1 (96 tiles), 128 / 12 (96 tiles). `ts_phase`, `ts_first`, `ts_fill_a/b` take
  the B map index and use per-phase ring depth `TsC::R = ts_ringn(TT)` and slot `TsC::SLOT`.
- Data (arm, torch.zeros; trailing kernel args after `tk`): `xb2` bf16 [256][1024] = bf16(x * ln2[l]) (map 9, box {64, GU_TT}); `ss2` fp32 [256] = sum over k of x^2 after O (zeroed in the QKV `extra()` of the same layer);
  `act` bf16 [256][3072] = bf16(silu(r2 g) * (r2 u)), r2 = rsqrt(ss2 / 1024 + eps), the v0 D `get` math (map 11, box {64, 128}); `wgu` bf16 [28 * 6144][1024] (map 10, box {64, 128}), `build.interleave_gu(wg, wu)`: per layer, p in 0..47,
  rows [128 p, +64) = wg rows [64 p, +64), rows [128 p + 64, +64) = wu rows [64 p, +64) (tested on a small tensor against the formula); built once per weight set by the arm (352 MB). `tk` is u32 [64]: O tickets tk[rb * NTT + tt] (0..15), D tickets
  tk[16 + rb * NTT + tt]. Maps 8..11 are encoded per bucket (`build.workspace_map_specs(npad, names)`, box rows = TT of the phase). The `gu` buffer, its zeroing and `rr2` are not used with TSB.
- Smem. Ring bytes = max over the phases of R * (32 KB + TT * 256 B) = 196608 for every config (QKV / D: 3 x 64 KB; GU at TT 64: 4 x 48 KB; GU at TT 128: 3 x 64 KB), then the GU exchange region [196608, 208896) (12 KB, `TS_XOFF`), then OFF_BARS = 208896.
  DYN_SMEM = 210304 (+ 11280 static = 221584 <= 232448; static_asserts, MK_EXPECT_DYN mirrors `build.dyn_smem(..., tsb)`). The mbarrier block [OFF_BARS, OFF_BARS + 64) is untouched (the exchange region ends exactly at OFF_BARS). TMEM: 2 TT <= 256 columns per phase.
- GU epilogue (KS 1, act straight from TMEM). TMEM lanes 0..63 (quadrants 0, 1) = gate of channel 64 rb + 32 q + lane, lanes 64..127 (quadrants 2, 3) = up of channel 64 rb + 32 (q - 2) + lane (wgu row order). Epilogue warps 4..15: group = (warp >> 2) - 1 (0..2), quadrant q = warp & 3;
  warps w and w + 2 of a group (q and q + 2) run the same chunks (c = group, group + 3, ...; one 16-token chunk per iteration, no `two` pairing). The u warp stores its 16 values to the exchange slot [pid = 2 group + (q & 1)][16][32] fp32, then both warps `bar.sync 1 + pid, 64`;
  the g warp reads the 16 u values, `bar.sync` again (slot free), computes act for t < ntok and stores bf16 act[t][ch] (64 contiguous bytes per token and warp). Then every epilogue warp runs `fence.proxy.async.global` after the join (before the grid barrier arrive).
- D: `epi` = the v0 D red.add into x (columns < ntok). After the join, every CTA with nt > 0: tid 0 `__threadfence()`, then `atomicAdd(tk + 16 + t / D_KS, 1)` per tile (not at the last layer).
- Idle-CTA finalizers (`idle_finalize<IS_O, NTILE>`). Idle CTAs = those that own no unit/tile of the phase (O: 128 units, D: 128 / 96 tiles, grid 148); rank r among the idle CTAs by a scalar loop over b' < gridDim.x with the geom / ts_geom formula; jobs j = rb * NTT + tt
  (NTT = NPAD / 128, 16 or 8 jobs; <= 20 idle CTAs), idle CTA r takes j = r, r + nidle, ... tid 0 spins `ld.acquire.gpu` (`__nanosleep(64)`) until the ticket == target (O: 16, D: D_KS), resets it with `atomicExch`, `__syncthreads`; all 512 threads `__ldcg` x[t][rb 128 ..+127] for the
  job's tokens (t < ntok) and write xb2 = bf16(x * ln2[l]) plus `atomicAdd(ss2[t], sum of x^2 over the 128 columns)` (shuffle over the 16 lanes of a token first), or xb = bf16(x * ln1[l + 1]) (D, not at the last layer); `fence.proxy.async.global` by every thread; the
  grid barrier arrive follows (the MK_PF_ARRIVE window code runs after it for idle CTAs too).
- MK_NULL bits 4 / 5 with TSB: `ts_phase` NUL path (producer + full waits), no GU epilogue / fence, no D tickets, no finalizer. Bit 3: the O gemm_phase NUL path never runs the hook, so the O finalizer is skipped (otherwise it would hang). Stamps: no new stamps.
- Ring slot of GEMM units (g.F) now advances in O only (and LM at NPAD >= 128), prefetch_next<PhO> and the LM prefetch use g.F as before.
Arguments:
1. Parity / re-init. Same convention as P5a: every TS barrier block (full[R] count 2, empty[R], accfull[2], accempty[2] = 12 warps) is re-initialised by tid 0 in `ts_first` of its phase (windows: after O -> GU, after GU -> D, after D -> QKV, prologue -> QKV) and used only in
   the following `ts_phase`; counters are phase-local (step j: slot j % R, use j / R; tile ti: accumulator ti & 1, use ti >> 1). R differs per phase (3 / 4 / 3): the init loop of a phase covers only its R barriers, and only those are used. Re-init is legal because the previous TS phase (or
   none) ended with the join `__syncthreads` after the epilogue warps' last accfull wait (all MMAs and TMEM loads done), the producer's drain wait on the last use of every empty[] and every TMA consumed by an MMA that waited its full barrier (no pending complete_tx); between two TS phases
   the O gemm phase and attention only wait their own barriers (+0 / +64) never the TS block. The first fill of a window writes slots 0..R-1 of the new phase's layout only after the previous phase's drain, see 2. full[s] count 2 (A expect_tx in the window, B expect_tx after the grid barrier)
   as P5a. accempty count 12: every epilogue warp arrives once per tile outside the chunk loop, also warps whose chunks are all >= ntok, and the GU pair syncs happen inside the chunk loop with equal trip counts for both partners.
2. Overlap with in-flight TMA / smem. Window fills (ts_first) write ring slots 0..R-1 = [0, R * SLOT) <= [0, 196608). What may still touch that smem at a window: (a) after O: the O gemm ring / B tiles: every O unit was consumed by an MMA that waited its full barrier, the last segment waited the
   commit (in-order MMA completion), epilogue reads and B staging finished by the loop-end `__syncthreads` of `gemm_phase` (idle CTAs: nothing was filled); the finalizer uses no dynamic smem. (b) after GU / D: the previous TS ring: producer drained, MMAs done, join `__syncthreads`; the GU exchange
   region [196608, 208896) is not part of any ring. Inside a phase a slot is refilled only after empty[s]. The exchange region is also disjoint from the barrier block (ends at OFF_BARS) and, being below OFF_BARS, may overlay the attention scratch's free bytes ([OFF_BARS - 82432, OFF_BARS), free between calls per the A1
   rule); the hole [OFF_BARS, OFF_BARS + 64) is not used by TSB. The exchange region is generic-proxy only (no TMA writes it).
3. TMEM lane quadrant. Every epilogue warp reads only lanes 32 (warp & 3) .. +31 (`mk_tmem_read16` with q = warp & 3). QKV / D: each (quadrant, 16-column chunk) has exactly one reader as in P5a. GU: each (quadrant, chunk) is read by exactly one warp (group g and partner reads quadrants q and q + 2 of the same chunk set, the
   two quadrants are different warps), so the u warp reads lanes 64..127 and the g warp lanes 0..63 of the same columns; accumulators at columns b * TT (TT 64: 0, 64; TT 128: 0, 128), within the 256-column allocation.
4. Named barrier ids. The u / g pair of group G (0..2) and quadrant parity p (0 / 1) uses id 1 + 2 G + p (ids 1..6), count 64 = the two warps. Id 0 is `__syncthreads` (grid-barrier code, join, hooks). The attention placeholder (`attn_placeholder.cuh`) uses only `__syncthreads` (id 0); the A1 splitkv header
   is not in this repo (`kernels/common` has only a README without named-barrier use), so ids 1..6 must be kept free for it (to be written in the LEDGER). Both warps of a pair execute the same sequence u: {write, sync, sync}, g: {sync, read, sync} the same number of times (same chunk set, break condition `cb >= ntok` warp-uniform and equal),
   so generation k of the barrier completes only with both warps and no id is shared by two pairs. The second sync protects the slot: the u warp cannot write the next chunk before the g warp has read this one. The syncs happen only in the epilogue warps (4..15), which no other code waits on until the join `__syncthreads`.
5. Generic-to-async proxy fences. xb2, act and xb are written with generic stores and read by TMA (B boxes, map 9 / 11 / 8) in later phases. Writers: O finalizer (xb2), GU g warps (act), D finalizer (xb), prologue (xb): every writing thread runs `fence.proxy.async.global` after its stores and before the
   grid barrier arrive (`__syncthreads` + `red.release.gpu`); the consumers (producer thread of each CTA) run `fence.proxy.async.global` after the barrier's acquire and before the first B box of the phase (top of `ts_phase`, unchanged from P5a). Rows >= ntok of xb2 / act / xb are never written (allocation zeros). Reuse: xb2 is
   rewritten a layer later (O finalizer), after the GU phase's TMA loads of this layer completed (each CTA drained and arrived at >= 3 grid barriers in between); act likewise (GU of the next layer vs D of this layer); xb as P5a.
6. Ticket memory ordering. Worker side: every thread's red.adds of the tile / row block, then (O: inside gemm_phase's `fin` after `tc_fence_before`; D: after the `ts_phase` join) `__syncthreads`, tid 0 `__threadfence()` (fence.sc.gpu, cumulative over the CTA's red.adds it has observed through the barrier), then `atomicAdd`
   on the ticket (relaxed RMW). The RMW chain of one ticket forms release sequences headed by each worker's fence + atomic, so the value reaching the target (O: 16 unit adds, D: D_KS tile adds) is read by the finalizer's tid 0 `ld.acquire.gpu` as a synchronizes-with edge from every worker; `__syncthreads` extends it to the other 511 threads, whose `ld.global.cg` of x then
   see all the red.adds (performed at L2, `.cg` skips L1). The finalizer's x reads happen before its xb2 / xb stores; ss2 atomics are relaxed adds made visible to GU by the same fence/barrier release (the finalizer's `bar_arrive` is `__syncthreads` + `red.release.gpu`, GU readers pass `bar_wait` with `fence.acquire.gpu`).
   Reset: `atomicExch(ticket, 0)` by the finalizer's tid 0 right after it saw the target: all adds of that ticket for this layer happened (target reached) and none will follow until the next layer's phase, behind >= 3 grid barriers. O tickets are per (rb, tt): each worker flush adds nk to every token-tile ticket of its row block (two for NPAD 256), because a
   shared ticket could be reset by the first finalizer while the second waits. D at the last layer: no adds, no finalizer (tickets stay 0), so the launch ends with all tickets 0 (also for graph replays).
7. Finalizer deadlock freedom. Cooperative launch: all 148 CTAs are co-resident. A CTA waits on a ticket only if it is idle in that phase, i.e. owns no unit / tile, so it has no worker duty that others could wait for. Worker CTAs never wait for a finalizer within the phase: their phase code waits only on their own mbarriers (TMA of weights and of
   B written before the previous grid barrier, own MMAs) and on nothing global, then add to the tickets and arrive at the grid barrier. The ticket targets are reached by workers alone (all work of the phase is owned by non-idle CTAs and always completes, MMA tiles of tokens >= ntok included), so every finalizer's wait terminates, and the grid barrier completes only after all finalizers arrive.
   Jobs are mapped to idle CTAs by rank so each job has exactly one finalizer (nidle >= 16 >= jobs at NPAD 256; the loop handles nidle < jobs). The build has `static_assert(NB > 128)`: with a grid <= 128 CTAs there would be no idle CTA for O; the SPEC's fallback to the P5a last-arriver path is NOT built (deviation).

### Build (ptxas, NVRTC 13.0 sm_100a, default knobs; regs MK_TSB 0 | 1; 0 spill, 0 stack in all)
| NPAD | MK_TS | regs 0 | regs 1 | dyn smem 0 | dyn smem 1 | ring R qkv / gu / d |
|---|---|---|---|---|---|---|
| 16 / 32 / 64 | any | 86 / 80 / 113 | same | 165248 / 165248 / 198016 | same | - |
| 128 | 0 or 256 (TS off) | 115 | 115 | 132352 | 132352 | - |
| 128 | 128 | 122 | 122 | 198016 | 210304 | 3 / 4 / 3 |
| 256 | 0 | 113 | 113 | 197888 | 197888 | - |
| 256 | 128 or 256 | 123 | 128 | 198016 | 210304 | 3 / 3 / 3 |
SASS (NPAD 256, `cuobjdump -sass`; MK_TSB 0 | 1): UTCHMMA 40 | 56 (D: SPT 3 unrolled, GU SPT 8 loop), UTMALDG.2D 26 | 34, UTCBAR 6 | 10, LDTM 6 | 7, REDG.E.ADD.F32 80 | 80, ATOMG.E.ADD.STRONG.GPU 2 | 10 (+ 1 ATOMG.ADD.F32 = ss2), ATOMG.E.EXCH 3 | 4,
FENCE.VIEW.ASYNC.G 3 | 7, MEMBAR.SC.GPU 4 | 4, SYNCS.PHASECHK 34 | 46, SYNCS.EXCH.64 (mbarrier init) 24 | 44, NANOSLEEP 1 | 3, BAR.SYNC 33 | 36 (4 of them the named 64-thread barriers of the GU exchange).
Deviations from SPEC-P5b: (1) no last-arriver fallback when there are no idle CTAs: `static_assert(NB > 128)` instead (the arm compiles with NB = SM count; B200 148). (2) O tickets per (rb, tt) (the SPEC's one ticket per rb is reset by the first of the two token-tile finalizers at NPAD 256).
(3) The GU epilogue handles one chunk per iteration (no two-chunk TMEM read in flight) so one exchange slot per pair suffices; the slot is protected by a second `bar.sync` on the same id (the SPEC's "second named barrier"), not a different id; ids 1..6. (4) `fence.proxy.async.global` of the act writers runs after the `ts_phase` join (warps >= 4), not inside the chunk loop.
(5) `TS_R_FIT` removed: ring depth per phase `ts_ringn(TT)`; `TS_R` stays the QKV depth. (6) `wgu` is cached in the arm per (wg, wu) data_ptr pair (one entry); `xb2` / `ss2` / `act` / `wgu` and maps 9..11 exist only for TSB builds, the kernel arguments are 0 otherwise.

## A1 integration (attn = splitkv, MK_SKV; SPEC-P2)
- Header: `regimebench/kernels/common/splitkv` (tag splitkv-v1, imported unmodified). Default stays attn = placeholder (SASS 0-diff vs P5b at all buckets, MK_TS 0/256 x MK_TSB 0/1).
- Layout: scratch for `skv::attend` = `sm + OFF_BARS - kBarOff`; A1's 64 B mbarrier tail = [OFF_BARS, OFF_BARS+64), never written by mk. mk's barrier block starts at OFF_MB = OFF_BARS + 64.
  OFF_BARS = smallest value >= max(layout max, kBarOff) with OFF_BARS = 512 (mod 1024); `sm` is 1024 B aligned (kernel rounds the raw dyn base up; DYN_SMEM carries 1024 B slack), so the scratch base is 1024 B aligned. build.py `skv_off_bars` / `dyn_smem(attn=)` mirror it; MK_EXPECT_DYN checks it.
- Phase 3 = `skv::attend<NT>`; phase 4 stages the B slot of unit u = kc (one unit per CTA, NB >= 128) with `skv::combine_head<NPAD, NT>` before gemm_phase (template order is <TPAD, TNT> in the header), then gemm_phase's fence.proxy.async + __syncthreads precede the MMA.
- The Wo prefetch cannot sit in the phase 1->2 window (attend's TMA ring overlays the weight ring): under splitkv it is issued after attend, always after bar_arrive (MK_PFA ignored). `fence.proxy.async.global` after the K/V row write (phase 2) makes it visible to TMA loads in other CTAs. The 16 K/V tensor maps get a tensormap acquire in the prologue.
- NULL bit 2 skips attend and combine (no hang). KU 64 is placeholder-only.
- Open: with the 128-register cap the splitkv build spills at NPAD >= 32 (stack 48..248 B; NPAD 16: 0). The spill is in attend/combine/non-TS staging paths; see the P2 report for the numbers. Check the GPU A/B before enabling by default.

## splitkv-v2 integration (SPEC-P2b; CPU-verified only: compile, SASS, pytest; NOTHING has run on a GPU)
- Header: tag splitkv-v2 (1f027ac). K / V tensor maps are SWIZZLE_128B, box {64, 1, rows} (SKV_TMA_MMA 1, header default; encoding = a1bench.py `tensor_maps` with mma = True). Map count and rows follow the build:
  `build.skv_geometry(defines)` (header defaults + defines) -> 16 maps (128 rows) default, 14 maps (64 rows, 4 stages) for the R6 build. The kernel's map_acquire loop and the arm's blob follow `skv::kNumMaps` / `skv_num_maps(qmax)`.
  attend_smem_bytes / kBarOff = 148,032 / 147,968 in both builds (MK_EXPECT_DYN static_assert).
- O phase without combine: kernel argument `skv_comb` (last, after `skv_engine`) = `plan.s_max > 1`. 1: `combine_head` into the B slot as before. 0: every ao row was written by attend (S == 1 everywhere), and `stage_any(SkvB)` stages the B tile from `ao`
  inside gemm_phase with the placeholder O `get` (`__ldcg` uint4, zero rows >= ntok, then stage_tile + fence.proxy.async). Block-uniform run-time branch. Fences: attend writes `ao` with generic stores, the grid barrier (bar_arrive / bar_wait) orders them,
  and the staging reads with ld.global.cg (generic): no proxy fence is needed for this LDG staging (a TMA read of ao would need `fence.proxy.async.global` after attend's stores; not used).
- R6 build: `attn = splitkv` with any q_len > 1 compiles with `-DSKV_QMAX=16` (`compile_build(..., skv_qmax=16)`, in the lru key; the arm's `_kernel` key is (npad, device, attn, qmax)). q_len == 1 workloads keep the default build.
  Contract check vs CHANGELOG P5: hv2g packs kv_len + q_len rows per sequence (= plan.row_base, asserted in the arm), phase 2 writes the q_len new rows at t_base + (t_vis - 1) before attend, token t sees t_vis = kv_len + t + 1 positions, q is [total_q, 16, 128] (row of token t) in `qb`:
  all match. S may be > 1 in R6 (combine_head reads pair_info per token row). `auto_attn` is unchanged (q_len == 1 only).

## LM head on the tile stream (MK_LMTS, SPEC-P3c; CPU-verified only: compile, SASS, pytest; NOTHING has run on a GPU)
Knob `MK_LMTS` (default 0; arm option `lmts`). `LMTS = TSB && MK_LMTS` (NPAD >= 128 with the defaults); at run time also `lm_ident`. MK_LMTS 0: 0-line SASS diff vs commit 737b8ef at the 5 buckets (default knobs) and NPAD 256 with MK_TS 256 / MK_TSB 0
(`test_lmts_off_sass_identical_to_head`); MK_LMTS 1 at NPAD <= 64 = MK_LMTS 0 (TSB off).
- Design: the v0 LM (`gemm_phase<PhLm>`, fp32 x restaged per row block by LDG) is replaced by `ts_phase<PhLm, 128, 1, 8, mk_null(6)>` (A = embed row block rb by TMA map 7, B = xb by TMA map 8, TT 128, KS 1, SPT 8; NTILE = 1187 * NPAD / 128).
  xb at the LM head holds bf16(x * fnorm): the last layer's D phase now also runs the D tickets and `idle_finalize<false, NTILE, true>(x, fnorm, xb, ssf)` (SS = add sum x^2 per token into ssf = `ss2 + MAXTOK`).
  `ss2` is allocated `[2 * MAXTOK]` in the arm (no new kernel argument). ssf is zeroed in the QKV `extra()` of the last layer (`zero_f32(ss2 + MAXTOK, (ntok + 3) & ~3)`; MAXTOK % 4 == 0, so the rounded count stays in the buffer).
  After the last grid barrier all threads load `s_ss[t] = __ldcg(ssf + t)` (t < n_lm), `__syncthreads()`, then the phase runs with the EXISTING LM `epi` (it applies rsqrt(s_ss / HID + EPS) itself: logits store + per-(quadrant, slot) argmax). The `get` lambda is not used.
  The barrier window after the last D calls `ts_first<PhLm, 128, 1>(g, 0)` (A prefetch + TS mbarrier re-init) instead of `lm_first(g)`. After the phase everything is unchanged (dealloc, stamps, argmax reduce over s_bk / s_bi). No new smem (ring for TT 128 and TMEM 2 x 128 columns come from the TSB build).
- Run-time condition `lm_ident` = (n_lm == ntok) && (lm_tok[s] == s for every s < n_lm). Prologue: each thread ANDs its i's (`i >= n_lm || s_lmtok[i] == i`, MAXTOK 256 <= NT 512, so one i per thread), `__syncthreads_and` replaces the prologue `__syncthreads`
  (LMTS builds only). Same inputs in every CTA, so the value is block- and grid-uniform. True for every q_len 1 workload (all mk points that reach NPAD >= 128). lm_ident == 0: the v0 path exactly as before (`lm_first` in the window, `gemm_phase<PhLm>`; the last layer skips tickets / finalize).
- Argmax rule (strict `>` in epi stays valid): (a) a CTA's tiles t0 .. t0 + nt - 1 have t = rb * NTT + tt (KS 1), so rb ascends along the CTA's tile list; (b) in the non-PAIR epilogue branch warp group grp = (warp >> 2) - 1 runs chunks c = grp, grp + 6, ... and (two) c + 3, i.e. c in {grp, grp + 3, grp + 6, ..}:
  a slot s = tt * 128 + 16 c + i fixes tt and c, hence one warp group, hence one warp per quadrant: each (quadrant, slot) has exactly one writer warp, which sees rb ascend. Ties inside a row block: `ri = row - lane + (ffs(eq) - 1)` (lowest row among equal maxima), unchanged.
- ts_phase's non-PAIR branch stops at `cb >= ntok` and reads chunk c + 3 only if `cb + 48 < ntok`: `n_lm` is passed as ntok. B rows >= n_lm come from xb rows holding stale values of earlier steps or zeros; the epi ignores slots >= n_lm and MMA columns are independent, so they cannot affect slots < n_lm.
- Fences and ordering: (1) xb (fnorm rows) is written by the idle CTAs with generic stores + `fence.proxy.async.global` (end of idle_finalize), then the grid barrier; the ts producer does `fence.proxy.async.global` before its first B box (existing), same as the QKV path.
  (2) ssf is written with atomicAdd by the idle CTAs before the grid barrier and read with `__ldcg` after it. (3) The window `ts_first` re-initialises the TS mbarriers: the previous ts phase (last D) has drained (its join `__syncthreads` precedes the window), same as QKV after D.
  (4) D tickets: reset by the finalizer as today; their next use is the next step, behind grid barriers. (5) s_ss: nothing writes it between the GU re-zero and the LM load (prologue zero, GU setup writes r2 and re-zeroes it; the v0 LM `get` accumulation is not on this path).
- Known issue (not changed, MK_LMTS 0 must stay a 0-line SASS diff): `zero_f32(ss2, ntok)` skips a trailing partial float4, so for ntok % 4 != 0 the last ss2 entries are not zeroed (stale sums). This can matter for the GU r2 (rsqrt of a stale ss2) when ntok % 4 != 0; ssf avoids it by zeroing `(ntok + 3) & ~3`.
- Implementation note: the QKV `extra` lambda reads a `const bool zssf = LMTS && l == NLAY - 1` instead of `l` directly: a use of `l` in a discarded `if constexpr` branch of a non-template function still captures it and changed the NPAD <= 64 SASS (found by the 0-line diff test).
### Build (ptxas, NVRTC 13.0 sm_100a, default knobs; MK_LMTS 0 | 1; 0 spill, 0 stack in all)
| NPAD | regs 0 / 1 | local bytes 0 / 1 | UTCHMMA | UTMALDG.2D | UTCBAR | LDTM | ATOMG |
|---|---|---|---|---|---|---|---|
| 128 | 126 / 126 | 0 / 0 | 56 / 64 | 34 / 42 | 10 / 12 | 7 / 9 | 14 / 16 |
| 256 | 128 / 128 | 0 / 0 | 56 / 64 | 34 / 42 | 10 / 12 | 7 / 9 | 15 / 17 |
Deltas at NPAD 256 (MK_LMTS 1 - 0): UTCHMMA +8 (the LM phase's SPT 8 MMA loop), UTMALDG +8, LDTM +2 (one chunk + "two" read), UTCBAR +2, ATOMG +2 (the second idle_finalize instance, SS: ssf atomicAdd, tick reset).
