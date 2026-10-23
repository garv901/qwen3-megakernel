# SPEC-P5b: GU and D on the tile stream; bf16 B copies written off the critical path (NPAD >= TS_MIN)

## Why (PASS-04.md, PASS-05.md)
- At b256, GU is 15.1 + 3.1 us/layer and D is 16.7 + 14.3 in mk (job 9593). On the tile stream with B by TMA they are 8.65 / 6.48 (job 9595).
- P5a showed that a last-arriver finalize (bf16 conversion) on the critical path costs 3.7-5.0 us/layer (job 9603).
- P5b therefore does every conversion in **idle CTAs** (CTAs with no tile or unit in that phase), which wait on tickets.

## Scope
- Files: `regimebench/kernels/mk/mk_step.cu`, `mk_common.cuh`, `build.py`, `NOTES.md`, `regimebench/arms/mk.py`, `tests/test_mk.py`.
- All of it is under `TS` (MK_TS = minimum NPAD, default 256; buckets below TS_MIN unchanged).
- New knob `MK_TSB` (default 0): 1 enables P5b on top of P5a. Off = 0-line SASS diff vs HEAD at all buckets and MK_TS values.
- With MK_TSB=1, the P5a last-arriver D finalize is replaced by the idle-CTA finalize below.

## Data (new global buffers, all allocated with torch.zeros in the arm; pass them as trailing kernel args, as P5a did)
- `xb2`: bf16 [MAXTOK][HID]: bf16(x * ln2[l]) (B of GU). Map 9: box {64, TT_GU}.
- `ss2`: fp32 [MAXTOK]: sum over k of x[t][k]^2 after the O phase. Zero it in the QKV phase `extra()` of the same layer (as gu is zeroed).
- `act`: bf16 [MAXTOK][INTER]: silu(r2 * g) * (r2 * u), with r2 = rsqrt(ss2[t] / HID + EPS) (same math as the v0 D `get`, MK_SILU). Map 11: box {64, TT_D}.
- `wgu`: bf16 [NLAY * 2 * INTER][HID], built once in the arm from wg and wu.
  - Per layer: for p in 0..47, rows [128 p, 128 p + 64) = wg rows [64 p, 64 p + 64), and rows [128 p + 64, 128 p + 128) = wu rows [64 p, +64).
  - Map 10: dims {HID, NLAY * 6144}, box {64, 128} (same as the weight maps).
- Tickets: extend `tk` to 64 u32. Use separate ranges for O (8 row blocks) and D (8 x NTT tiles). Every finalizer resets its ticket.
- Box rows depend on the bucket's TT, so the arm encodes maps 8-11 per bucket.

## Phase configs (constexpr per bucket; from jobs 9592 / 9595)
| phase | NPAD 256: TT / KS | NPAD 128: TT / KS |
|---|---|---|
| QKV (P5a) | 128 / 2 | 128 / 4 |
| GU | 128 / 1 (96 tiles) | 64 / 1 (96 tiles) |
| D | 128 / 8 (128 tiles) | 128 / 12 (96 tiles) |

## Engine changes (`ts_phase`, `ts_first`, `ts_fill_a`, `ts_fill_b`)
- Make the B map index, TT and the A source per-phase template inputs.
  - A source: GU uses map 10 with `row = l * 6144 + 128 rb` (rb = 0..47). QKV and D keep `PH::src`.
  - B source: QKV map 8 (xb), GU map 9 (xb2), D map 11 (act).
- TMEM: 2 * TT <= 256 columns (true for all the configs above).
- Ring depth R per phase: as many slots of 32 KB + TT * 256 B as fit.
  - DYN_SMEM = max over all layouts, plus the GU exchange region below. Mirror it in build.py.
  - Keep the static_assert dyn + static <= 227 KB.
  - Reserve no part of [OFF_BARS, OFF_BARS + 64) for anything else: the A1 mbarrier hole, see LEDGER.
- Epilogue functor: give it (q, lane, warp group, rb, tt, column base cb, v[16]) per chunk. Keep the "one warp per (quadrant, chunk)" rule.

## GU epilogue (KS 1): act straight from TMEM
- TMEM lanes 0-63 (quadrants 0, 1) hold g of channel 64 rb + 32 q + lane. Lanes 64-127 (quadrants 2, 3) hold u of channel 64 rb + 32 (q - 2) + lane.
- Warps w and w + 2 of the same group (warp >> 2) handle the same column chunks.
  - The u warp writes its 16 values to an smem exchange slot [group][q & 1][16][32] fp32 (12 KB total).
  - It then syncs with its partner via a named barrier of 64 threads (ids 1-6; check that the attention placeholder and A1 headers do not use those ids, and say so).
  - The g warp reads the u values, computes act for the 16 tokens of the chunk (t < ntok only), and stores bf16 act[t][ch]. Each warp writes 64 contiguous bytes per token.
  - A second named barrier (or a double buffer) protects the slot reuse.
- The exchange region is dynamic smem placed after the ring, below OFF_BARS.
- Every act writer runs `fence.proxy.async.global` before the grid barrier arrive, because D reads act by TMA.
- The v0 GU `extra()` (rr_of_x for rr2) is not needed under TSB, because r2 comes from ss2. The `gu` fp32 buffer and its zeroing are not needed under TSB.

## D on the tile stream
- B = act by TMA (map 11). The epilogue red.adds into x, as v0 D does, for columns < ntok.
- After the roles join, every CTA with nt > 0 runs: for each of its tiles: tid 0 `__threadfence()`, then `atomicAdd(&tkD[rb * NTT + tt], 1)`.
- `ts_first<PhD>` replaces `prefetch_next<PhD>` in the GU barrier window.

## Idle-CTA finalizers (O and D phases)
- **Jobs:**
  - O: one per (rb, tt) with TT 128. NPAD 256: 16 jobs; 128: 8. Each needs ticket tkO[rb] == 16 (the O units are u = rb * 16 + kc; the O flush hook adds nk, as in P5a's D hook).
  - D: one per (rb, tt), needing tkD[rb * NTT + tt] == KS.
- **Who:** CTAs with no work in that phase. O has 128 units on 148 CTAs; D has 128 / 96 tiles.
  - Idle rank r = index of this CTA among the idle ones in blockIdx order. Compute it with a small loop over b' < blockIdx.x of the same unit/tile range formula.
  - Idle CTA r takes jobs r, r + nidle, ...
  - static_assert / device check: nidle >= 1. If a config has no idle CTA, fall back to the P5a last-arriver path and say so.
- **Wait:** tid 0 spins `ld.acquire.gpu` on the ticket, with `__nanosleep(64)` between polls, until it reaches the target. Then `__syncthreads`.
- **Convert:** all 512 threads `__ldcg` x[t][rb * 128 .. +127] for t in [tt * 128, min(ntok, tt * 128 + 128)).
  - O job: write xb2 = bf16(x * ln2[l]) and atomicAdd the per-token partial sum of squares into ss2[t] (warp-reduce over the 128 columns first).
  - D job: write xb = bf16(x * ln1[l + 1]) (skip for l = NLAY - 1).
- **Reset:** tid 0 resets the ticket to 0. Every writer runs `fence.proxy.async.global`, then the CTA arrives at the grid barrier.
- No deadlock: the workers never wait for finalizers within the phase (cooperative launch, all CTAs resident). Argue this in NOTES.md.
- The prologue still writes xb for layer 0 (P5a).

## MK_NULL and MK_STAMPS
- MK_NULL bits 4 (GU) and 5 (D) with TSB: producer + full waits only, no MMA, no epilogue, no tickets, no finalizers. The outputs are wrong by design; the run must not hang.
- MK_STAMPS: same stamp positions.

## Checks (CPU only; never run on a GPU)
- All 5 buckets with MK_TSB 0 / 1 x MK_TS 0 / 128 / 256 compile with 0 spill. Report regs, dyn smem and R per phase.
- 0-line SASS diff of MK_TSB=0 vs HEAD.
- SASS counts at NPAD 256 TSB=1.
- Tests in `tests/test_mk.py`, including a CPU test of the wgu interleave on a small random tensor vs the formula above.
- pytest exit code 0. Do not pipe it through tail.
- Do not commit.
- Report: the files, the build table, the SASS counts, the 0-diff, the pytest result, the arguments (parity / re-init, overlap, proxy fences for xb2 / act, ticket ordering, named barriers, finalizer deadlock freedom) and every deviation.
