# bar: grid-barrier microbench (SPEC-bar). CPU-built only; NOT yet run on a GPU.

Run: `bash run.sh [--build-only] [nit=20000] [reps=5] [skews=0,100]`, or `sbatch probe/mk_core/bar/bar.sbatch` from the repo root.

## Build (nvcc -O3 sm_100a, ptxas -v; 0 spill, 0 stack in every arm)
Regs of the timing kernel k_bar<false> (the validation kernel k_bar<true> is a separate instantiation).
| arm | what | regs (timing) | regs (validation) |
|---|---|---|---|
| 0 | mk v0: red.release + relaxed poll + fence.acquire | 16 | 24 |
| 1 | red.release + ld.acquire poll | 16 | 24 |
| 2 | atom.acq_rel counter + flag | 21 | 26 |
| 3 | two-level (8 groups) | 22 | 30 |
| 4 | per-CTA arrival words, CTA 0 gathers | 31 | 31 |
| 5 | cg grid.sync | 18 | 24 |
| 6 | as 2, warp-0 poll with __any_sync | 21 | 26 |

## SASS of the barrier loop (cuobjdump -sass, k_bar<false>)
- 0: BAR.SYNC; MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR then REDG.E.ADD.STRONG.GPU; poll loop = one LDG.E.STRONG.GPU + branch (no fence in the loop); after the loop one CCTL.IVALL (fence.acquire.gpu = CCTL.IVALL, no MEMBAR); BAR.SYNC.
- 1: same, but CCTL.IVALL sits INSIDE the poll loop (after every LDG.E.STRONG.GPU). No MEMBAR after the poll.
- 2: MEMBAR/ERRBAR before ATOMG.E.ADD.STRONG.GPU, CCTL.IVALL after it (acq_rel); last arriver: MEMBAR.ALL.GPU + STG.E.STRONG.GPU flag; others: LDG.E.STRONG.GPU + CCTL.IVALL per poll.
- 3: two ATOMG.E.ADD.STRONG.GPU (group, top) each followed by CCTL.IVALL, MEMBAR before each atom and before the STG.E.STRONG.GPU flag; poll as arm 2.
- 4: MEMBAR + STG.E.STRONG.GPU arrival word; CTA 0 warp 0 gather = LDG.E.STRONG.GPU (+CCTL.IVALL) loops, 5 loads/lane unrolled, MEMBAR + STG.E.STRONG.GPU flag; flag poll as arm 2.
- 5: grid.sync compiled to ATOM.E.ADD.STRONG.GPU + LD.E.STRONG.GPU (generic) with MEMBAR.ALL.GPU + CCTL.IVALL, BAR.SYNC x2.
- 6: as 2; the poll is LDG.E.STRONG.GPU on all lanes of warp 0 (+CCTL.IVALL), exit by __any_sync.
No NANOSLEEP in any arm.

## Deviations from the spec
1. Arms 2/3/6 use `atom.add.acq_rel` (spec: `.release`) for arm 2/6 and for the top counter of arm 3. A release-only RMW does not make the last arriver
   acquire the others' writes, so the barrier would not be a full barrier. Group counters in arm 3 are acq_rel as in the spec. Cost: one CCTL.IVALL after the atom.
2. Validation stores `i+1` (not `i`) and compares `< i+1`, so the zero-initialised slots cannot mask a miss at i = 0. Validation runs for skew 0 and 100.
3. Timing and validation are two instantiations of one templated kernel (`k_bar<VAL>`), so the timed loop carries no validation code.
4. Arm 5: arrive is empty, `grid.sync()` is in the wait (no split). Arm 6 lane 0 of warp 0 has `last` broadcast by `__shfl_sync`.
5. Skew: thread 0 spins before `bar_arrive`; other threads wait in its first `__syncthreads`.
6. mhz = CTA 0 clock64 delta / globaltimer delta over the whole launch, as xover2.
7. Extra CLI: args are `nit reps skews`; default skews "0,100". regs in the JSON come from cudaFuncGetAttributes of the timing kernel.
