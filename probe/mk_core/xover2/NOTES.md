# xover2 notes (CPU-side checks only; nothing here has run on a GPU)

Build: `bash probe/mk_core/xover2/run.sh --build-only` (nvcc 12.x, sm_100a, -O3, -Xptxas -v; log in logs/build.log). All 8 binaries build.

## Build table (k_xover kernel; `spill` = ptxas spill stores/loads, all 0; stack 0)
| bin | KU | XTMA | RING | CG | ORACLE | regs | spill |
|---|---|---|---|---|---|---|---|
| a0 | 128 | 0 | 2 | 1 | 0 | 121 | 0 |
| o0 | 128 | 0 | 2 | 1 | 1 | 110 | 0 |
| a1 | 64 | 0 | 4 | 1 | 0 | 122 | 0 |
| a2 | 128 | 1 | 2 | 1 | 0 | 63 | 0 |
| a3 | 128 | 1 | max | 1 | 0 | 62 | 0 |
| a4 | 64 | 1 | max | 1 | 0 | 62 | 0 |
| a5 | 128 | 1 | max | 2 | 0 | 62 | 0 |
| o3 | 128 | 1 | max | 1 | 1 | 24 | 0 |
(512 thr x 1 CTA/SM caps regs at 128.)

## Dynamic smem (KiB, = 1 KiB align slack + 2 KiB barriers/tmem slot + weight ring + X slots), `ring` in brackets (computed by the host exactly as run.sh/main does)
Npad = max(16 (CG=2: 32), roundup16(T)). ring = RING, or for RING=0 min(slots that fit 227 KiB, max units per CTA (pair) of the shape).
- a0/o0/a2 (ring 2): T<=16: 75; T=64: 99; T=128: 131; T=256: 195 (all shapes).
- a1 (ring 4, 16 KB units, 2 X slots): T<=16: 71; 64: 83; 128: 99; 256: 131.
- a3/o3 (KU 128, XTMA, ring max), T=1..16 / 64 / 128 / 256: QKV [2] 75/99/131/195; O [1] 39/51/67/99; GU [3] 111/147/195 then [2] 195 at T=256;
  D [2] 75/99/131/195; LM [6] 219 (T<=16), [4] 195 (64), [3] 195 (128), [2] 195 (256).
- a4 (KU 64): QKV [4]; O [2]; GU [6] (T=256: [4]); D [3]; LM [12] 219 (T<=16), [9] (64), [7] 227 (128), [4] 195 (256). Max 227 KiB at LM T=128.
- a5 (CG=2, per-CTA X half): QKV [2] 75..131; O [1] 39..67; GU [3] 111..195; D [2] 75..131; LM [6] 219 (T<=16), [5] 203 (64), [4] 195 (128), [3] 195 (256).
Units per CTA (pair for CG=2) max: KU128 CG1: QKV 2, O 1 (128 units on 148 CTAs: 20 CTAs idle), GU 3, D 2, LM 65; KU64: 4, 2, 6, 3, 129; CG2: 2, 1, 3, 2, 65.

## SASS (cuobjdump -sass, function k_xover only; the 15 FFMA of the whole file are in k_init / k_ref)
| bin | UTCHMMA | UTCBAR | LDTM | UTCATOMSWS | UTMALDG (call sites) | REDG.E.ADD.F32 | other |
|---|---|---|---|---|---|---|---|
| a0 | 8 | 1 | 2 | 1 .AND + 2 .FIND_AND_SET.ALIGN | 6 (.2D) | 32 | |
| o0 | 0 | 0 | 0 | 0 | 6 | 7 | |
| a1 | 4 | 1 | 2 | 1 + 2 | 3 | 32 | |
| a2 | 8 | 1 | 2 | 1 + 2 | 12 | 32 | |
| a3 | 8 | 1 | 2 | 1 + 2 | 52 | 32 | |
| a4 | 4 | 1 | 2 | 1 + 2 | 26 | 32 | |
| a5 | 8 `UTCHMMA.2CTA` | 1 `UTCBAR.2CTA.MULTICAST` | 2 | 1 `.AND` + 2 `UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN` | 52 `UTMALDG.2D.2CTA` | 32 | 2 UCGABAR_ARV, 2 UCGABAR_WAIT (cluster syncs) |
| o3 | 0 | 0 | 0 | 0 | 52 | 7 | |
No HMMA, no FFMA, no UBLKCP in k_xover of any binary. The TMA is `UTMALDG.2D` (the UTMALDG count is the number of inlined fill() call sites x boxes).
The cta_group::2 forms found in the a5 SASS: `UTCHMMA.2CTA`, `UTCBAR.2CTA.MULTICAST`, `UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN`, `UTMALDG.2D.2CTA`. (The pi05 h_bench binary was not built.)
a0 vs xover e1 (built with -DENG=1): same tcgen05 mnemonics and counts (8 UTCHMMA, 1 UTCBAR, 2 LDTM, 32 REDG); k_xover has 1744 SASS instructions vs 1736;
the remaining differences are address arithmetic (ring / bar offsets) only. Source differences from xover ENG=1: barrier offsets (full at +0, mma at +256, tmem slot at +512 instead of +0/+16/+32),
generic KU/ring/KC expressions (ring is a compile-time constant for RING != 0), the row guard `nrow < N` in the flush, no `HAS_TC` runtime flag.

## Design notes / deviations from the spec text
- Oracles (o0, o3) = xover ENG=-1 literally: wait the unit's loads, per-unit red.add of zeros in the FFMA pattern (one lane of 4 per row, all T tokens), no MMA / TMEM.
  The tcgen05 arms do one flush per row-block segment, so at large T the oracle has more red.adds than a0/a3; it is the floor of the load path + barrier, not of the flush.
- cta_group::2: tma2d2 / mma_tc2 / commit2 copied verbatim from h_bench.cu; tmem alloc/free c2 take the column count as an operand (h_bench has a fixed 64; 256 needed here).
  No relinquish_alloc_permit for cta_group::2 (h_bench does none). Rank 0 thread 0 issues MMA and commit (multicast mask 3 arrives at both CTAs' mma barrier at the same offset);
  both CTAs' TMA complete_tx on rank 0's full barrier (rank 0 expects 2 x (weight + X bytes)); cluster sync after init/alloc and before dealloc.
- D layout for M=256: CTA r, TMEM lane 32q+l = its own row 128 r + 32 q + l, column j = token j (rank 0's B half = tokens 0..Npad/2-1, rank 1's = Npad/2..); from h_bench.cu comments
  ("RPC 128 (M = 256): lane = feature row 32 q + lane, columns 0..63 = tokens 0..63") and hv2_tc NOTES-micro. Tested only by the correctness check on GPU (not run).
- LM with CG=2: 1187 row blocks of 128 is odd, so the pair grid has 594 pair blocks; rank 1 of the last pair reads 128 rows past N (pool rows exist: rowsAvail = rows - roundup(N,256))
  and its flush skips rows >= N. Its weight bytes are ~0.08 % more than N*K*2.
- LM correctness compares a 4096-row sample of N (rows 0..2047 and N-2048..N-1), all T in {1,8,64,256}.
- O (N=1024, K=2048) at KU=128: only 128 units (CG=2: 64 pair units) for 148 CTAs (74 pairs): 20 CTAs (10 pairs) idle by the spec's numbering; KU=64 gives 256.
- TMA OOB: CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = zero fill (cuda.h doc of cuTensorMapEncodeTiled). The X map's outer dim is T; for CG=2 rank 1 may start its box entirely past T (T < Npad/2): unverified that the TMA still completes the full byte count (expected by PTX: OOB elements are filled and counted).
- XTMA arms: the per-slot X buffers follow the weight ring. Mid-phase refill of a slot waits for the commit of its last MMA (same as xover), only tid 0 waits.
- One TMEM accumulator (256 columns) as xover: a mid-phase flush stalls the next MMA (matters for LM, 8 row blocks per CTA; double-buffered accumulators not implemented).
- JSON: `occupancy` = resident CTAs per SM for CG=1 (need >= 1), max co-resident clusters of 2 for CG=2 (need 74); a config below the need is skipped with a stderr message.
- `nph` argument 0 (default) = 112 phases, LM 4. Correctness launches use nph = 1.
- X is loaded after the grid barrier in every arm. XTMA=0 stages X by all threads after the barrier. XTMA=1 has a separate mbarrier per slot (`xfull[s]`, at bars+512; tmem slot moved to bars+1024):
  the barrier-window prefetch issues weight TMAs only (`fill_w`); tid 0 issues the X TMAs (`fill_x`) at the top of the next phase, after `gbar`, for the prefetched units; in-phase refills issue both;
  the MMA issuer (and oracle o3) waits `full[s]` and `xfull[s]`. For CG=2 rank 0's `xfull` counts both CTAs' X bytes.
