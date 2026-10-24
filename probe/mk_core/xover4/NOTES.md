# xover4 notes (CPU-side checks only; nothing here has run on a GPU)

Build: `bash probe/mk_core/xover4/run.sh --build-only` (nvcc, sm_100a, -O3, -Xptxas -v; log in logs/build.log). Binaries: bin/ts1 (-DCG=1), bin/ts2 (-DCG=2).

## Design
- xover4.cu = xover3.cu minus the SB = 1 arm (stage_b, Xf / rs / Xs, k_mkx, SB argument) plus ragged token tiles and CG = 2. Args: `nph reps shape T TT KS`.
- Ragged: NTT = ceil(T / TT); X map outer dim T, rows >= T are TMA OOB (zero fill), expect_tx stays the full box bytes. Epilogue: chunk guard `tt*TT + 16c >= T -> break`
  (and `two` only if the second chunk starts below T). T is a multiple of 16, so no per-element guard.
  NOT verified: that a box (partly or entirely, e.g. T = 128, TT = 96, CG 2 rank 1 of tile 1) outside the tensor still completes its full byte count. The PASS check on GPU will show it (as xover2 note).
- CG = 2: cluster 2 + cooperative launch, rank from %cluster_ctarank, 74 pairs, tile range per pair, A 128 rows per rank, B TT/2 tokens per rank (box {64, TT/2}), slot = 32768 + (TT/2)*256 B,
  all TMAs on rank 0 full[s] (count 2: expect 2*A_BYTES, then TT*256), MMA by rank 0 tid 32 (M = 256, N = TT, B k-atom stride (TT/2)*128), commit2 to empty[s] and accf[b],
  producer tid 0 on both ranks waits its own empty[s], epilogue on both ranks reads its own TMEM (row = rb*256 + 128 rank + 32q + l, columns = all TT tokens),
  rank 1 epilogue warps arrive remotely (mapa + mbarrier.arrive.release.cluster) on rank 0 acce[b] (count 2*4*NEPI), the issuer waits it with mbar_wait_cluster (acquire.cluster).
  cluster_sync_all after init + alloc (before any TMA), and `__syncthreads(); cluster_sync_all();` before tmem_free_c2. TMEM_COLS 512, no relinquish for CG 2. gbar over all 148 CTAs.
- Valid TT: multiple of 16 in [16, 256], TT <= T (TT = T always valid). Lists: CG 1 {48,64,96,128,256}, CG 2 {48,64,96,128,192,256} (+ T).

## Build table (k_ts)
| bin | regs | spill (stack / stores / loads) |
|---|---|---|
| ts1 | 108 | 0 / 0 / 0 |
| ts2 | 92 | 0 / 0 / 0 |

## SASS (cuobjdump -sass, k_ts only)
| mnemonic | ts1 | ts2 |
|---|---|---|
| UTCHMMA | 8 | 8 `.2CTA` |
| UTCBAR | 2 | 2 `.2CTA.MULTICAST` |
| UTMALDG.2D | 32 | 32 `.2CTA` |
| UTCATOMSWS | 1 `.AND` + 2 `.FIND_AND_SET.ALIGN` | 1 `.AND` + 2 `.2CTA.FIND_AND_SET.ALIGN` |
| LDTM | 2 | 2 |
| REDG.E.ADD.F32 | 32 | 32 |
| UCGABAR_ARV / WAIT | 0 / 0 | 2 / 2 |
- ts1 has no 2CTA form. ts2 has UTCHMMA.2CTA, UTCBAR.2CTA.MULTICAST, UTMALDG.2D.2CTA, and no non-2CTA UTCHMMA / UTCBAR.
- ts2 keeps one non-2CTA `UTCATOMSWS.AND URZ, UR5` (single instruction in the TMEM allocation path). It is identical in the xover2 a5 SASS (CG = 2 build), so it is
  emitted by ptxas for the tcgen05.alloc / dealloc code, not by anything in the source with a cta_group::1 qualifier (the source has none in the CG 2 path). The strict "no UTCATOM*" check is not met by this one instruction; check on GPU.

## Deviations
- JSON `"bin"` stays "ts" (field `cg` tells the binary); `"sb"` removed; `nt` kept (= `ntile`), `tiles_mean` = NTILE / (148 / CG) (per CTA for CG 1, per pair for CG 2), `tiles_max` likewise.
- The skip rule `NTILE < 37` of xover3 is kept for both CG (no halving for 74 pairs).
- With a TT argument given, only TT values in the sweep lists (or T) are run; other values are ignored.
- Rank 0's own acce arrival uses the local `release.cta` arrive (spec: "arrive locally"); the cluster-scope acquire on the wait pairs with rank 1's cluster-scope release.
- k_ts epilogue branch condition is `warp >= 4 && (warp >> 2) - 1 < NCH` (SB part removed).
- Rank 1's tid 32 is idle (the issuer condition is `tid == 32 && rank == 0`).
