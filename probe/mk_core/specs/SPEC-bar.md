# SPEC-bar: grid-barrier microbench (Pass P1, MK-core)

## Why
mk v0 (job 9576, probe/mk_core/V0.md) spends 1.7-2.7 us per grid barrier, 6 barriers per layer: ~13 of 27 us/layer at b1.
Find the cheapest correct grid barrier for 148 CTAs x 512 threads on B200, then port it into mk as a knob.
Hypotheses [H]: (a) 148 same-address atomics serialize at the L2 slice; (b) polling CTAs hit the same line as the atomics;
(c) the separate fence.acquire after a relaxed poll costs extra.

## Files (new directory `probe/mk_core/bar/`; touch nothing else)
`bar.cu`, `run.sh` (build all arms + `-Xptxas -v` log in `logs/build.log`, then run every arm unless `--build-only`),
`bar.sbatch` (copy the pattern of `probe/mk_core/xover2/xover2.sbatch`, 20 min), `NOTES.md`, `.gitignore` (`bin/`).
nvcc: `/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v -DBAR=<n>`.

## Kernel
- Persistent, cooperative launch (`cudaLaunchCooperativeKernel`), grid = SM count (query it), 512 threads, dynamic smem
  100 KB (forces 1 CTA/SM like mk). Check occupancy >= 1 and abort otherwise.
- Loop `nit` barriers (default 20000). Between barriers optional work: `skew_ns` (default 0): CTA b spins
  `(b % 8) * skew_ns` ns on %globaltimer before arriving (tests imbalance tolerance). Run skew 0 and skew 100.
- Split arrive / wait like mk: `bar_arrive(); <gap>; bar_wait();` with gap = 0 (the mk prefetch window is cheap).
- Sync words in one cudaMalloc'd buffer, zeroed before each launch; separate 128-B lines for every counter / flag.
- Time: cudaEvent around the launch, us per barrier = total / nit, median of 5 launches. Also report SM MHz
  (clock64 vs globaltimer delta over the launch on CTA 0) as xover2 does.
- Validation launch (separate, nit = 2000): before arriving at barrier i, thread 0 of each CTA stores i to
  `slot[cta]` (st.relaxed.gpu); after the wait, thread 0 reads `slot[(cta + 37 * i) % G]` (ld.relaxed.gpu) and counts
  values < i as violations. Print `PASS`/`FAIL violations=N` per arm. Must be 0 for every arm.

## Arms (compile-time BAR)
- **0 = mk v0 exactly**: copy `bar_arrive` / `bar_wait` from `regimebench/kernels/mk/mk_common.cuh:154-178` (no stamps):
  `__syncthreads(); tid 0: red.release.gpu.global.add.u32 cnt, 1;` then `target += G; tid 0: poll ld.relaxed.gpu cnt
  until >= target; fence.acquire.gpu;` then `__syncthreads()`. Monotonic counter (no reset inside the launch).
- **1**: as 0, poll with `ld.acquire.gpu` and no separate fence.
- **2 = counter + flag**: tid 0 `atom.add.release.gpu` (returns old). If old == target - 1 (last arriver):
  `st.release.gpu flag, epoch`. Else poll `flag` with `ld.acquire.gpu` until == epoch. Counter and flag on separate lines.
- **3 = two-level**: CTAs in 8 contiguous groups (group = b * 8 / G). tid 0 atom.add.acq_rel on its group counter; the
  last of the group atom.add.release on the top counter; the last at the top st.release the flag; all poll the flag.
  Group counters on separate lines.
- **4 = per-CTA arrival flags, no atomics**: tid 0 `st.release.gpu arr[b] = epoch` (one 128-B line per CTA? NO: pack
  4-byte words contiguously, 148 x 4 B). CTA 0 warp 0 polls all G words (each lane loads words lane, lane+32, ...
  with ld.acquire.gpu, loop until all == epoch, `__all_sync`), then `st.release.gpu flag, epoch`; all CTAs poll flag.
- **5 = cooperative_groups `this_grid().sync()`** (reference; plain `grid.sync()` between iterations, no split).
- **6**: as 2, but the flag poll spreads: all 32 lanes of warp 0 poll the same flag (`__any_sync` exit) - tests whether
  one lane's poll loop latency matters. (Cheap to add; drop if it does not compile cleanly.)

## Output
One JSON line per (arm, skew): `{"bar":n,"skew_ns":s,"us_per_bar":x,"us_reps":[...],"mhz":m,"G":148,"regs":r}` plus the
PASS/FAIL validation line. NOTES.md: build table (regs, spill), SASS of the barrier loop per arm (cuobjdump -sass; list the
memory/atomic mnemonics: RED / ATOM / LDG.E.STRONG.GPU / MEMBAR / FENCE / CCTL / NANOSLEEP), and any deviation from this spec.

## Rules
CPU-only work: build all arms and do the SASS check; do NOT run on a GPU (no srun/sbatch). Do not git commit.
Do not touch any file outside `probe/mk_core/bar/`. Report: files, build table, SASS summary, deviations.
