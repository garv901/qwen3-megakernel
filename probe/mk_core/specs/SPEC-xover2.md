# SPEC xover2 (MK-core Pass 0b): tcgen05 GEMM phase per Qwen3-0.6B shape, T = 1..256, tile / ingress / cta_group arms

## Why (measured, jobs)
- `probe/xover` (job 9521, read `probe/xover/{SPEC,RESULT,NOTES}.md` and `xover.cu` first): one GEMM phase on 148 CTAs,
  tcgen05 swap-AB (weights on M = 128, tokens on N), is flat in T up to ~16 (GU 7.74 -> 8.03 us, D 5.10 -> 5.25) and
  rises at T = 256 (GU 14.10, D 9.09). Only GU (N 6144, K 1024) and D (N 1024, K 3072) were measured.
- The no-math oracle at T = 1 is GU 4.96 us vs a byte floor of 12.6 MB / 6.5 TB/s = 1.94 us: the skeleton (barrier,
  first-TMA latency, a 2-slot 64 KB weight ring, all-thread X staging) costs more than the bytes. [H]
- At T = 256 each 32 KB weight unit needs 64 KB of staged X (128 k x 256 tokens bf16): activation ingress is 2x the
  weight bytes. [H] cta_group::2 halves per-CTA X ingress (pi05 F design: -11 % at 50 tokens, `pi05_mk.cu` ~line 1504).
The mk megakernel needs, per phase shape and T, the best of: unit size, X path, ring depth, cta_group. This bench answers it.

## What to build
`probe/mk_core/xover2/xover2.cu` = a copy of `probe/xover/xover.cu` (keep its skeleton, timing, clock and correctness
code), extended as below; `run.sh` (build every binary with `-Xptxas -v` into `bin/`, log `logs/build.log`; unless
`--build-only`, run every binary over all shapes and T, JSON lines on stdout); `xover2.sbatch` (1 GPU, partition main,
`--time=00:45:00`, log `probe/mk_core/xover2/logs/slurm-%j.log`, prints `nvidia-smi -L` first, `cd` via
`$SLURM_SUBMIT_DIR`, NOT a hard-coded path to another worktree). Drop the FFMA engines (ENG 0 / 2): not needed.

### Shapes (N rows of W, K), `shape=` argument
| name | N | K | note |
|---|---|---|---|
| QKV | 4096 | 1024 | q 2048 + k 1024 + v 1024 rows |
| O | 1024 | 2048 | |
| GU | 6144 | 1024 | |
| D | 1024 | 3072 | |
| LM | 151936 | 1024 | 1187 row blocks of 128; nph = 4 for this shape (311 MB per phase), else 112 |
T in {1, 2, 4, 8, 16, 32, 64, 128, 256}. Pool >= 2 GiB as in xover (LM: phase p reads a fresh 311 MB slice, wrapped).

### Compile-time knobs (one binary per arm; macro defaults = xover ENG=1)
- `KU` = k per work unit: 128 (xover: 32 KB unit, 2 TMA boxes of 64 k) or 64 (16 KB unit, 1 box). Units numbered as in
  xover, u = rb * (K / KU) + kc; CTA b owns units [b*U/148, (b+1)*U/148).
- `XTMA` = 0: X staged by all threads with LDG + STS (xover). 1: X tile loaded by TMA from a bf16 [T][K] global tensor
  map (box {64 k, Npad rows}, SW128; rows >= T read as zero by the TMA OOB fill, so allocate X as [256][K] and set the
  map's outer dim to T), issued by tid 0 together with the weight unit, same mbarrier (expect_tx = both byte counts).
- `RING` = number of weight ring slots of one unit each. 2 = xover. `RING=0` means "max": the largest slot count such that
  RING * unit bytes + X slots + 2 KB fits in 227 KB for the largest Npad the binary runs; every unit the CTA owns up to
  that count is issued in the barrier window of the previous phase (before the grid-barrier wait), so a CTA whose whole
  share fits never waits for a TMA after the barrier. With XTMA the X slots follow the same ring (one X slot per weight
  slot). Write the slot count per (shape, Npad) into the JSON (`ring` field).
  (Context, not a task: in mk the X of QKV / GU / D is an fp32 split-K sum that still needs RMSNorm or SiLU*up, so
  XTMA there needs a bf16 producer; O-proj reads A1's bf16 output directly. The bench prices the path either way.)
- `CG` = 1 (cta_group::1, xover) or 2: cluster of 2 CTAs (cluster launch attribute, cooperative; check that 148 CTAs
  stay resident and print the occupancy), `tcgen05.mma.cta_group::2`, M = 256 per pair: rank r loads weight rows
  128 r .. 128 r + 127 of a 256-row unit pair and tokens Npad/2 * r .. of B (each CTA stages half of the X columns; N of
  the instruction = Npad, Npad >= 32 for CG=2, check the valid N range in the PTX ISA table for cta_group::2 M = 256).
  Units are 256 rows x KU; the pair owns the unit range of its rank-0 CTA index / 2. Commit multicast to both CTAs; each
  CTA reads its 128 TMEM lanes. Copy the helpers VERBATIM from
  `/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/ubench/h/h_bench.cu` (lines ~32-45: 2d/3d TMA cta_group::2,
  mma2, commit2, tmem_alloc/free c2) and the pair layout notes in `.../ubench/fg/fg_k1.cu` ~lines 509-590 (D layout:
  "CTA r holds its own 64 rows" for M = 128 there; for M = 256 confirm from the ISA / probe u13 notes which TMEM lanes
  hold which rows, and test it via the correctness check). Requires XTMA=1 (build error otherwise).
- `ORACLE` = 1: same loads, barrier, Y flush of zeros; no MMA, no TMEM read (as ENG=-1, but with this binary's load path).

### Binaries (build all; names in bin/)
| bin | KU | XTMA | RING | CG | ORACLE | question |
|---|---|---|---|---|---|---|
| a0 | 128 | 0 | 2 | 1 | 0 | = xover e1 (sanity: GU/D must match job 9521 within 0.2 us) |
| o0 | 128 | 0 | 2 | 1 | 1 | xover oracle |
| a1 | 64 | 0 | 2*2 (same bytes) | 1 | 0 | finer units / balance |
| a2 | 128 | 1 | 2 | 1 | 0 | TMA X path |
| a3 | 128 | 1 | 0 (max) | 1 | 0 | whole share prefetched |
| a4 | 64 | 1 | 0 (max) | 1 | 0 | a3 with finer units |
| a5 | 128 | 1 | 0 (max) | 2 | 0 | cta_group::2 |
| o3 | 128 | 1 | 0 (max) | 1 | 1 | floor of the a3 skeleton |

### Correctness
As xover: for every binary except oracles, every shape, T in {1, 8, 64, 256}: nph = 1, Y zeroed, compare with the
CUDA-core reference (LM: compare a 4096-row sample of N to keep it fast; say so). PASS iff max|d| <= 2e-3*max(1,|ref|).

### SASS / build check (write into `probe/mk_core/xover2/NOTES.md`)
Per binary: regs, spill (`-Xptxas -v`, must be 0; if not, say where), dynamic smem per Npad. `cuobjdump -sass`:
mnemonic counts for UTCHMMA (or the cta_group::2 form: find its exact name in the a5 SASS and in the pi05 h_bench binary
if one is built; do not guess), UTCBAR, LDTM, UTCATOMSWS, UTMALDG / UBLKCP (TMA), REDG; oracles have no UTCHMMA. No HMMA,
no FFMA in the math loop.

### Output (per JSON line)
`{"bin":..,"shape":..,"T":..,"us_phase":..,"us_reps":[..],"mhz":..,"pass":..,"regs":..,"spill":..,"ring":..,
"units_per_cta_max":..,"occupancy":..}` plus the floor line per (shape, T): weight bytes / 6.5 TB/s, and
X bytes (T*K*2 per phase, and the per-CTA staged X bytes summed over CTAs) / 6.5 TB/s.

## Acceptance (CPU only)
- `bash probe/mk_core/xover2/run.sh --build-only` builds all 8 binaries on this machine (nvcc, no GPU).
- `a0` source path equals xover ENG=1 when the knobs are at their defaults (diff the device code paths; say so).
- NOTES.md written with the table above. Do NOT submit Slurm jobs, do not run on a GPU, do not commit.

## Files you own
`probe/mk_core/xover2/` and this spec's report in `/raid/garv901/regime-mk-core` only. Read anything; edit nothing else
(never `probe/xover/`, never the vibe-megakernel tree).

## Report back
Files written; build result; regs / spill / smem per binary; SASS mnemonics found (exact names); every place where this
spec was wrong or unclear and what you did (e.g. valid N for cta_group::2, TMEM lane layout of M = 256, ring sizes that
fit); the exact `sbatch` command for the GPU run (from the worktree root).
