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
