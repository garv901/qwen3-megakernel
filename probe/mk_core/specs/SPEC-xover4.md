# SPEC-xover4: tile-stream engine with cta_group::2 pairs and ragged token tiles (standalone bench)

Owner: MK-core. Implementer: a subagent. Pass: P6 (PASS-06.md). CPU only: build + ptxas + SASS check. Do not run on the GPU.

## Why
- mk GU at b256 is 11.2 us/layer (stamps, job 9725). fi's 2-CTA nvjet GU is about 6.7 us (PASS-00).
- PTX rule: "All tcgen05 instructions within a kernel must specify the same value for the .cta_group qualifier".
  A cta_group::2 GU phase therefore forces every tcgen05 phase of that bucket build (QKV, O, GU, D, LM) into pair mode.
- So first we measure pair mode for all four GEMM shapes in one standalone bench.
- We also measure ragged token tiles (TT that does not divide T, e.g. 96). With these, 144 of 148 CTAs have work at T = 256. Today only 96 have work.

## Files (new directory, do not edit xover2 / xover3)
- `probe/mk_core/xover4/xover4.cu`: copy of `probe/mk_core/xover3/xover3.cu`, then the changes below.
- `probe/mk_core/xover4/run.sh`: builds two binaries, `bin/ts1` (-DCG=1) and `bin/ts2` (-DCG=2), with `-Xptxas -v` into `logs/build.log`. `--build-only` as in xover3. It also builds `bin/a3`, which is not needed: leave it out.
- `probe/mk_core/xover4/xover4.sbatch`: the run plan (below).
- `probe/mk_core/xover4/NOTES.md`: design notes, deviations, SASS check result, regs and spill per binary.

## Changes to the copy
1. **Remove the SB = 1 (staged-B) arm completely:** the stage_b lambda, the Xf / rs / Xs inputs, the k_mkx kernel and the SB argument. B always comes from the TMA X map.
   Keep the argument order of main except that SB is removed: `nph reps shape T TT KS`.
2. **Ragged token tiles (CG 1 and 2).**
   - NTT = ceil(T / TT).
   - The X map's outer dimension stays T, so the rows past T are TMA out-of-bounds (zero fill, OOB_FILL_NONE as in xover3). expect_tx stays the full box bytes. Write a NOTES line on whether this was verified (the PASS check will show it).
   - Epilogue: skip token columns >= T. Use a chunk guard `cb >= T -> break` as in mk ts_phase. Inside a chunk, guard per element only when T % 16 != 0. Here T is a multiple of 16, so a chunk guard is enough.
   - Y must still get every (t < T, n) element exactly once for KS = 1. The NaN-fill check catches misses.
   - Valid TT: multiples of 16 in [16, 256]. For CG = 2, also TT / 2 % 8 == 0 (a multiple of 16 is enough).
   - Sweep lists:
     - CG 1: TT in {48, 64, 96, 128, 256};
     - CG 2: TT in {48, 64, 96, 128, 192, 256};
     - skip TT > T, except that TT = T is always valid.
3. **CG = 2 (compile-time `-DCG=2`; CG = 1 must give the same kernel as the xover3 SB = 0 path, apart from the ragged-tile changes).**
   - Launch: cooperative + cluster dimension 2 through `cudaLaunchKernelEx`, as in `xover2.cu` main (lines ~340-352). Do the occupancy check with `cudaOccupancyMaxActiveClusters` (need >= 74), as xover2 does.
   - rank = `%cluster_ctarank` (read it with asm), pair p = blockIdx.x >> 1, and 74 pairs.
   - Pair tile: t = (rb * NTT + tt) * KS + ks, with rb over N / 256 row pairs. NTILE = (N / 256) * NTT * KS. Pair p owns [p * NTILE / 74, (p + 1) * NTILE / 74). Both ranks compute the same range, so both run the same loop trip counts.
   - Helpers: copy `tma2d2`, `mma_tc2`, `commit2`, `tmem_alloc_c2`, `tmem_free_c2` and `cluster_sync_all` verbatim from `xover2.cu` lines 105-116. Allocate TMEM_COLS = 512 with tmem_alloc_c2. Do not call relinquish for CG 2.
   - A (per ring slot, 32 KB at slot offset 0): rank r loads W rows `prow(p) + rb * 256 + 128 r`, 2 boxes of 64 k x 128 rows, with tma2d2.
   - B (slot offset 32768): rank r loads tokens `tt * TT + r * (TT / 2)`, TT / 2 rows, 2 boxes of 64 k x (TT / 2) rows at `+ a * (TT / 2) * 128`.
     - The X map box is {64, TT / 2} for CG 2.
     - The slot bytes are 32768 + (TT / 2) * 256 for CG 2. The ring depth follows from that, as in xover3: as many slots as fit, at most 16.
   - All TMAs of both ranks complete on **rank 0's** full[s]: tma2d2 clears the peer bit, as in xover2.
     - full[s] keeps count 2 on rank 0. Rank 0 alone does `expect_tx(full, 2 * A_BYTES)` (arrival 1) and `expect_tx(full, TT * 256)` (arrival 2: both halves of B).
     - Rank 1 issues the same TMAs but no expect_tx / arrive.
     - Rank 1's full barriers are initialised but never waited on.
   - MMA issuer: tid 32 of **rank 0 only**. It uses idesc = tc_idesc(256, TT) and mma_tc2. The A / B descriptors are the rank-local slot addresses; both ranks use the same smem offsets. The B k-atom stride is (TT / 2) * 128.
     - Commit with commit2 (multicast mask 3) to empty[s], and with a second commit2 to accf[b] at the last k-step of a tile.
     - Both CTAs' empty[s] and accf[b] then get the arrival at the same offset.
   - Producer: tid 0 on **both** ranks, waiting on its **own** empty[s] (multicast arrival). The loop structure is unchanged, including the next-phase A prefetch in the barrier window.
   - Epilogue (warps 4..15 on both ranks):
     - Each CTA reads its own TMEM. Lane 32 q + l = W row `rb * 256 + 128 rank + 32 q + l`. Column c = token `tt * TT + c` (the full TT columns, not a half). See the xover2 NOTES "D layout for M=256".
     - Write / red.add into Y as now.
     - acce[b] lives on rank 0 with count 2 * (4 * NEPI). Rank 0's warps arrive locally. Rank 1's warps arrive remotely: `mapa.shared::cluster.u32` of the acce address to rank 0, then `mbarrier.arrive.release.cluster.shared::cluster.b64 _, [addr];`.
     - Rank 0's MMA issuer waits acce[b] as now. That wait must use acquire at cluster scope: write an `mbar_wait_cluster` with `mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64`.
     - Before the arrive, every epilogue warp does `tcgen05.fence::before_thread_sync` (as now).
   - Start: `cluster_sync_all()` after mbarrier init + TMEM alloc and before any TMA (as xover2).
   - End: `__syncthreads(); cluster_sync_all();` before tmem_free_c2, so that no CTA frees TMEM while the peer's MMAs / TMAs are in flight.
   - Grid barrier `gbar` unchanged: all 148 CTAs.
   - LM shape: not in this bench (the shapes are QKV, O, GU, D). N % 256 == 0 for all four.
4. **JSON line:** add `"cg":CG` and `"ntile":NTILE`. Keep the other fields. Keep the PASS / FAIL stderr line per config, with cg.

## Correctness
- k_ref as in xover3 (phase 0 rows), for every config run. A config passes if bad == 0.
- Compile with `nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v`. 0 spill bytes is expected; report regs.

## SASS check (CPU, `cuobjdump -sass bin/ts2`)
- ts2 must contain UTCHMMA.2CTA, UTCBAR.2CTA.MULTICAST and UTMALDG.2D.2CTA.
- It must contain no non-2CTA UTCHMMA / UTCBAR / UTCATOM* (the one-cta_group rule).
- ts1 must contain no 2CTA forms.
- Write the mnemonic counts in NOTES.md.

## Run plan (xover4.sbatch, `--partition main --gres=gpu:1 --time 00:30:00`, output `probe/mk_core/xover4/logs/slurm-%j.log`)
- `nvidia-smi -L`; `bash run.sh --build-only`.
- For shape in QKV O GU D, T in 128 256: run `./bin/ts1 0 5 $s $t` and `./bin/ts2 0 5 $s $t` (all TT / KS).
- Do not submit it: MK-core submits.

## Done when
- Both binaries build with 0 spill bytes. The SASS check is as above.
- NOTES.md is written. No file outside `probe/mk_core/xover4/` was changed.
- Do not commit. MK-core reviews and commits.
