# SPEC-lm: pipelined LM head (Pass P3, MK-core)

## Why
mk's LM phase (job 9583 stamps, R0-b1) takes ~99 us. Its byte floor is 311 MB / 6.5 TB/s = 48 us, and fi takes 46 us. It uses
the generic `gemm_phase` (mk_step.cu), which does per unit: wait TMA, MMA, wait the previous MMA, refill one slot, restage X,
`__syncthreads`, and stall on the single TMEM accumulator at every row-block flush. With ring 2 that is ~1-1.5 units in flight per SM.
X (the B operand = bf16(x[lm_tok] * fnorm)) is the same for every row block, so it does not need restaging.

## Scope
New code path for the LM phase only, behind knob `MK_LMP` (0 = v0 path; 1 = new path). Default 0 in this change; I flip it after the A/B.
Files: `regimebench/kernels/mk/mk_step.cu`, `mk_common.cuh`, `build.py` (DEFAULT_KNOBS + `dyn_smem`), `tests/test_mk.py`, `NOTES.md`.
No other files. `MK_LMP=0` must give a 0-line SASS diff vs HEAD (check it as described in the Checks section below).
The new path applies to NPAD 16 and 32 only (X of all 8 k-chunks fits in smem); at NPAD >= 64 `MK_LMP=1` keeps the v0 LM path
(static dispatch on NPAD). Write `constexpr bool LMP = MK_LMP && NPAD <= 32;`.

## Design (LMP)
Row blocks: unchanged (`geom<PhLm>()`: CTA owns row blocks [r0, r1), WHOLE units, unit u = rb * 8 + kc, kc = k-chunk of 128).
1. **Smem layout of the LM phase** (it may overlay everything: no other phase runs after it): from the 1024-aligned base:
   `[XL: 8 k-chunks x XS bytes][LM ring: LMR slots x UB bytes]`, XS = NBOX * NPAD * 128 (4 KB at NPAD 16, 8 KB at 32), UB = 32 KB.
   LMR = `MK_LMR` knob (default 4). Barriers: own block of LM mbarriers in the barrier region: full[LMR], empty[LMR],
   accfull[2], accempty[2]. Grow the barrier region (OFF_BARS area) as needed. Update DYN_SMEM in mk_common.cuh =
   max(v0 layout, LM layout) and `build.py:dyn_smem` identically. The compile-time check `MK_EXPECT_DYN == DYN_SMEM` must keep passing.
   Initialise the LM mbarriers in the kernel prologue together with the others (counts: full 1 (expect_tx), empty 1 (tcgen05.commit),
   accfull 1 (commit), accempty = number of epilogue warps).
2. **Prefetch before the LM barrier**: the D phase of the last layer currently calls `prefetch_next<PhLm>(g, 0)` in the barrier window
   (MK_PF_ARRIVE). With LMP it must instead issue the first min(LMR, n) LM units into the LM ring slots with the LM full barriers.
   Legal because every earlier MMA has completed (gemm_phase waits its last commit) - but the LM ring may overlap the GEMM ring /
   B-tile / attention regions, so check that no TMA into those regions can still be in flight at that point (there is none in v0: the
   window prefetch is the only one, and you replace it). State this argument in NOTES.md.
3. **After the barrier**: all 512 threads stage XL = all 8 k-chunks of B (same `get` math as v0: bf16(x[lm_tok[t]] * fnorm), zero for
   t >= n_lm, SW128 K-major layout of `stage_tile`) and accumulate the per-slot sum of squares s_ss exactly once per element (v0 does
   it in the first row block's staging). Then `fence.proxy.async`, `__syncthreads`.
4. **Warp roles** for the rest of the phase (n units = (r1 - r0) * 8):
   - **Producer** (warp 0, lane 0): for unit j = LMR0 .. n-1 (LMR0 = units already prefetched): slot s = j % LMR; wait empty[s]
     (parity of the previous use) then expect_tx + TMA (mk_tma_unit) into slot s with full[s].
   - **MMA issuer** (warp 1, lane 0): for unit j: rb = r0 + j / 8, kc = j % 8, buffer b = (j / 8) & 1; at kc == 0 wait
     accempty[b] (skip for the first use of each buffer); wait full[s]; tc_fence_after; mk_mma_unit into TMEM columns b * NPAD
     (A = slot s, B = XL chunk kc, first = kc == 0); commit to empty[s]; at kc == 7 also commit to accfull[b].
   - **Epilogue** (warps 4..15, 12 warps = 3 per TMEM lane quadrant, q = warp & 3): for each row block i: buffer b = i & 1; wait
     accfull[b] (parity (i / 2) & 1); tc_fence_after; read the NPAD / 16 column chunks of buffer b split over the warps of each
     quadrant; call the v0 LM `epi(row, c0, v, rb)` math (logits, per-quadrant best: keep the "only writer of (quadrant, slot)"
     property: one warp per (quadrant, chunk)); tc_fence_before; one lane per epilogue warp arrives on accempty[b].
   - Warps 2, 3: idle in the phase (fine).
   - End: `__syncthreads` (all roles done), then the existing TMEM dealloc / stamps / argmax code, unchanged.
5. **TMEM**: two accumulators of NPAD columns each: 2 * NPAD <= 256 columns at NPAD <= 128, so the existing 256-column allocation is enough.
6. **MK_NULL bit 6 (LM null)** with LMP: producer + full/empty waits only (no staging, MMA, epilogue), like the v0 NUL path.
   MK_STAMPS: the LM-end stamp stays where it is.
7. Parities: all LM barriers are used only in the LM phase, once per launch; compute parities from local counters
   (full[s]/empty[s]: use index j / LMR; acc: i / 2). Every mbarrier is re-initialised in every launch (prologue), so there is no
   cross-launch state. Watch the first-use waits on empty[] and accempty[] (no wait or the "previous phase" trick; say which in NOTES.md).

## Checks (CPU; do NOT run on a GPU)
- All 5 buckets compile for MK_LMP 0 and 1 (and MK_LMR 2, 4, 6 at NPAD 16), 0 spill; report regs and dyn smem per bucket.
- MK_LMP=0: SASS identical to HEAD (strip addresses; compare instruction lists, as `tests/test_mk.py` does for knobs).
- SASS of MK_LMP=1, NPAD 16: count UTCHMMA / UTCBAR / LDTM / UTMALDG / SYNCS (mbarrier) and list them in NOTES.md.
- Add tests in `tests/test_mk.py`: compile NPAD 16 and 32 with MK_LMP=1; the MK_LMP=0 0-diff check; dyn_smem matches
  (the static_assert covers it, so a compile is enough).
- Run the CPU suite with `CUDA_CACHE_DISABLE=1 PYTHONPATH=/raid/garv901/regime-mk-core /raid/garv901/vibe-megakernel/.venv/bin/python -m pytest -q tests`.
- Do not git commit. Report: the diff summary, the build table, the SASS counts, the parity / first-use / overlap arguments, and
  any deviation from this spec.
