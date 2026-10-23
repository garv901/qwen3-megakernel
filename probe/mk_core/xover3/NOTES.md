# xover3 notes (SPEC-xover3; CPU-side checks only, nothing here has run on a GPU)

Files: `xover3.cu` (engine `ts` = kernel `k_ts`, harness), `run.sh`, `xover3.sbatch`, `.gitignore`. Baseline a3 = `../xover2/xover2.cu` (unedited) built with `-DKU=128 -DXTMA=1 -DRING=0 -DCG=1 -DORACLE=0`.
Build: `bash probe/mk_core/xover3/run.sh --build-only` (log: `logs/build.log`).

## Build table
| bin | regs | spill (stores/loads) | stack | static smem | dyn smem |
|---|---|---|---|---|---|
| ts (k_ts) | 106 | 0 / 0 | 0 | 0 | by TT, see below |
| a3 (k_xover) | 62 | 0 / 0 | 0 | 0 | as xover2 NOTES |

Ring depth = min(16, (227 KiB - 3 KiB) / (32 KiB + TT * 256 B)), dyn smem = 3 KiB (1 KiB align slack + 2 KiB barriers / TMEM slot) + ring * slot:
TT 32: ring 5, 203 KiB (207872 B); TT 64: ring 4, 195 KiB (199680 B); TT 128: ring 3, 195 KiB; TT 256: ring 2, 195 KiB. The bench prints `ring` and `smem` per point.

## SASS (cuobjdump -sass, k_ts only, 5109 lines)
UTCHMMA 8 (the 8 K16 MMAs of one k-step, one code site, in the tile / step loop), UTCBAR 2 (empty[s] commit, accfull[b] commit), LDTM 2 (chunk c, chunk c + 3), UTMALDG 32
(call sites of fill_a / fill_b: prologue, B-prefetch, in-loop, A window; 2 boxes each), REDG 32 (KS > 1 path: 2 x 16), STG 36 (32 in the KS == 1 path + 4 timing stores), UTCATOMSWS 3
(alloc, relinquish, dealloc), SYNCS: ARRIVE.TRANS64 16 (expect_tx and arrive), EXCH.64 14 (mbarrier init: 2 loops + 2 acc), PHASECHK.TRANS64.TRYWAIT 10. FFMA 0, HMMA 0.
KS is a runtime parameter, so both the STG and the REDG epilogue exist in the one binary.

## Config count / runtime
Valid (T, TT, KS, shape) with TT <= T, KS | K/128 (QKV, GU: 1 2 4 8; O: 1 2 4 8 16; D: 1 2 3 4 6 8 12 24): 210. NT < 37 (reported as `{"skipped":"NT<37"}`): 48. Run: 162
(QKV 36, O 30, GU 40, D 56); 100 of them are checked (T = 64 and 256). Plus a3: 16 points (4 shapes x 4 T).
Per config: 1 check launch (T = 64, 256) + 2 warm-up + 5 timed launches of 112 phases (~8 ms of kernel at 10 us / phase, a few ms for k_ref) + tensor-map encode: about 0.1 - 0.2 s.
Per (shape, T) process start (CUDA context, 2 GiB pool fill) about 2 - 3 s, 32 processes. Estimated job: 4 - 6 minutes (xover2 job 9566: 5 min for 8 arms); `--time=00:30:00` leaves 5x margin.

## Design (as lm_phase, cta_group::1)
Roles: tid 0 producer, tid 32 MMA issuer, warps 4..15 epilogue (quadrant `warp & 3`, group `(warp >> 2) - 1` in 0..2 takes column chunks `g, g + 3, g + 6, ..` of TT/16), other threads idle until the
phase's `gbar` (`__syncthreads`). Barriers per CTA: full[s] (count 2), empty[s] (count 1, MMA commit), accfull[b] (count 1, commit), accempty[b] (count 4 * min(TT/16, 3)). TMEM 512 columns allocated,
accumulator b (b = global tile counter & 1) at columns b * TT. Step = 128 k = 8 MMAs (M 128, N TT, K 16). MMA descriptors: A atom a (k 64a..) at slot + 16384 a + 32 B (kk & 3); B at slot + 32768 + TT*128 a + same.
Global counters across phases: step f = p * nst + j (slot f % ring, use f / ring), tile G = p * nt + ti (buffer G & 1, use G >> 1). Every CTA has the same nst, nt each phase, so these are phase-continuous.

### Arguments
1. Parity. Use number u of a barrier is awaited with parity u & 1 (the mbarrier convention of mk). full[s]: step f is use f / ring; the MMA waits (f / ring) & 1. empty[s]: step f - ring commits the previous use
   (f / ring - 1); the producer (in-loop and window) waits ((f / ring) - 1) & 1. accfull[b]: tile G is use G >> 1, epilogue waits (G >> 1) & 1. accempty[b]: tile G >= 2 waits use (G >> 1) - 1.
   Counts: full 2 = one `arrive.expect_tx` for A (32768 B) + one for B (TT * 256 B); phase completes only when both arrived and all tx bytes landed (tx may go transiently positive, never negative: the
   expect precedes the bytes of its own boxes; the other arrival is still missing so an early A completion cannot flip the phase). empty / accfull 1 (one `tcgen05.commit`). accempty = working epilogue
   warps only (lane 0 after `wait::ld`, `fence::before_thread_sync`, `__syncwarp`); at TT = 32 the third warp group has no chunk and neither arrives nor counts (NEPI = min(TT/16, 3)).
   I also ran a CPU model of the 3 agents + async TMA / commit completion with random scheduling (ring 1..5, nt 1..5, SPT 1..8, 1..3 phases, grid barrier between phases): no deadlock, no parity aliasing
   (every wait succeeds exactly at phase u + 1), no slot overwrite before its empty phase, no MMA read before the full phase (`/raid/tmp/claude-1029/x3/sim.py`, scratch, not part of the bench).
2. First use. empty[] is never waited for f < ring (first use of the slot; the in-loop producer starts at j = ring, the prologue and window fills skip the wait when f < ring). accempty[b] is not waited for G < 2.
   full[] / accfull[] are first waited with parity 0 on a fresh barrier (blocks until the first completion). All barriers are initialised once per launch (launch = nph phases) and never reset; counters
   are continuous across the phases of a launch, and the grid barrier separates phases.
   Window prefetch (A of the next phase's first min(ring, nst) steps, before `gbar`): its slot (F1 + i) % ring was last used by step F1 + i - ring, which can still be in flight, so the window producer waits
   empty for it (xover2 does not need this because it waits all MMAs before the epilogue; here MMAs of the last ring steps may still run when the producer finishes). B of those steps is issued only after the
   grid barrier (B = X = previous phase's output in the megakernel), on the same full barrier (arrival 2).
3. Overlap with in-flight TMA. A slot is refilled only after empty[s] (all 8 MMAs of its previous step finished reading A and B; tcgen05.commit tracks completion of all prior MMAs). No generic-proxy
   writes to smem exist in this kernel (A and B come by TMA only), so no `fence.proxy.async` is needed. Epilogue TMEM reads of accumulator b finish before accempty[b], and the MMA overwrites b only after it
   (first MMA of a tile uses accumulate = 0). The last accfull wait of every epilogue warp precedes `gbar`, so all MMAs and TMEM loads finished before dealloc (after the final `__syncthreads`).
4. TMEM lane quadrant. Epilogue warp w reads lanes 32 (w & 3) .. +31 (address `tb + ((q*32) << 16) + b*TT + col`), warps 4..15 cover each quadrant 3 times, chunk sets disjoint, so every accumulator
   element is read and written by exactly one thread. D row m = TMEM lane (M = 128 per CTA) = W row 128 rb + 32 q + lane; the 32 lanes of a warp store 32 consecutive floats (128 B) of Y[t][..].
   TMEM alloc / dealloc by warp 0 (all lanes, `.sync.aligned`), 512 columns = 2 x 256 (TT max).
5. SW128 B layout. The B box is {64 k, TT tokens} from the X map (dims {K, T}, SW128, 128 B inner): TT rows of 128 B, 8-row groups of 1024 B swizzled by TMA. This is the xover2 X layout (Rows = Npad);
   descriptor `sdesc`: SBO 1024 B (64 << 32 in 16 B units), SWIZZLE_128B (2 << 61), K-major, +32 B per K16 inside the 64-k atom (as xover2). Second k-atom box at + TT * 128 B (multiple of 4 KiB). Slot
   stride 32768 + 256 TT is a multiple of 1024 for TT >= 32, and the base is 1024-aligned, so every atom (8-row group) is 1024-aligned as SW128 requires. T % TT == 0 (power of two sizes), no OOB rows.
6. Instruction descriptor: `tc_idesc(128, TT)`: kind f16, A and B bf16 (bits 7, 10), K-major both (bits 15 / 16 = 0), N >> 3 at bit 17 (6 bits: TT 256 -> 32), M >> 4 at bit 24 (8). M = 128 with N = TT in 32..256
   (multiple of 16) is legal for cta_group::1. D = fp32 (bit 4).
7. Y writes. KS == 1: each element written exactly once by `st.global` (the check pre-fills Y with 0xFF = NaN so a missed element fails). KS > 1: `red.global.add.f32`; the check pre-zeroes Y.

### Y zeroing in the timed region
As xover2: no per-phase zeroing at all. Timed phases for KS > 1 keep `red.add`-ing into the same Y (values grow by a few hundred at most, no overflow; the atomic traffic is identical to a zeroed run).
Y is memset only before the single checked phase (outside the timed region). Deviation from the spec text "zeroed before each timed phase": no zeroing in timed phases, justified as above.

## Results
(to be filled after the job: per shape a table rows TT x KS, columns T = 32 / 64 / 128 / 256 in us per phase; best (TT, KS) per (shape, T) vs a3; PASS/FAIL list.)

## P4b: staged-B arm "sb" (SPEC-xover3 addendum; CPU-side checks only, not run on a GPU)
Runtime flag `SB` (7th argument of `bin/ts`: `nph reps shape T TT KS SB`; `SB=1 bash run.sh` for the default sweep). SB = 0 is the job 9592 path. SB = 1: X is `Xf` fp32 [T][K] + `rs` fp32 [T] (bench data: Xf in [-1, 1),
r in [0.5, 1.5)); B chunk of a k-step = bf16(Xf[t][k] * r[t]) written by warps 4..15 (all 12, 384 threads) into the ring slot in the SW128 K-major layout of the TMA box: k-atom `ch >> 3` at `+ (ch >> 3) * TT * 128`, token row r at
`+ r * 128`, 16 B chunk `(ch & 7) ^ (r & 7)` (as `stage_tile` / `lm_phase` in mk_step.cu). Per warp and step: `SB_STG = 4` chunks in flight per thread (`__ldcg` float4 x 2 + `rs`), `F2FP` pack, `STS.128`, then `fence.proxy.async.shared::cta`,
`__syncwarp`, lane 0 `mbarrier.arrive` on full[s]. full[s] count = 13 (A expect_tx arrival by the producer + 12 warps) when SB = 1, 2 when SB = 0. The A boxes and their window prefetch are unchanged; the producer skips every `fill_b`
when SB = 1; B staging starts after the grid barrier (the epilogue warps leave `gbar` and go straight to step 0). Host: `k_mkx` builds the reference input Xs = bf16(Xf * r) with the same fp32 multiply + RN; `k_ref` uses Xs when SB = 1.
The epilogue is now a lambda `epi_tile(ti, G0)` called once from one loop (`if (SB) stage the tile's SPT steps; if (has_ch) epi_tile`), so the SB = 0 code is the old epilogue, unduplicated.
Warps whose chunk index `(warp >> 2) - 1 >= NCH` (TT = 32: the third warp group) stage but run no epilogue and never arrive on accempty (count `4 * NEPI` unchanged).

### Build table (ptxas, `logs/build.log`)
| bin | regs | spill (stores / loads) | stack | smem |
|---|---|---|---|---|
| ts (k_ts), SB runtime | 120 (was 106 at SB-less; limit 128 at 512 threads) | 0 / 0 | 0 | dyn as above (ring unchanged) |
| a3 (k_xover) | 62 | 0 / 0 | 0 | as before |

### SASS, k_ts only (`cuobjdump -sass bin/ts`, function split; the earlier numbers in this file are right, whole-binary greps include k_ref / k_init and give FFMA)
| opcode | job 9592 build | now | note |
|---|---|---|---|
| UTCHMMA / UTCBAR / LDTM.x16 | 8 / 2 / 2 | 8 / 2 / 2 | SB = 0 hot code identical |
| UTMALDG.2D | 32 | 32 | |
| REDG.ADD.F32 / STG.E | 32 / 32 (+4 STG.64) | 32 / 32 (+4 STG.64) | epilogue exists once |
| LDG.E.128 (staging) | 0 | 8 | 4 chunks x 2 float4 |
| FMUL / F2FP.BF16.PACK_AB | 0 / 0 | 32 / 16 | scale + pack, staging only |
| STS.128 | 0 | 4 | staged B stores |
| FENCE.VIEW.ASYNC.S (fence.proxy.async) | 3 | 4 | +1 after staging |
| SYNCS.ARRIVE.TRANS64 (+A1T0) | 16 (+1) | 16 (+2) | +1 plain arrive (staging) |
| PHASECHK.TRYWAIT | 10 | 12 | +2 (staging empty wait; accfull / acce unchanged) |
| FFMA / HMMA | 0 / 0 | 0 / 0 | |
| STL / LDL | 0 | 0 | |
SASS lines 5108 -> 6004. Qualitative SB = 0 comparison: MMA loop, commit, TMEM loads, STG / REDG, TMA call sites are the same instruction counts; additions are the staging block (F2FP, FMUL, LDG.128, STS.128, one proxy fence, integer
division for step -> (t, kc, tt) mapping, as `F2I` / `I2F` sequences) plus one `SB` branch per phase / tile. Register allocation rose 106 -> 120 for the whole kernel (the staging block); no spill.

### Config count / runtime
Run plan (`xover3.sbatch`): a3 16 points; ts SB = 0 and SB = 1 at T = 128 (49 configs) and 256 (70) = 119 each = 238 runs; SB = 1 at T = 32 / 64: top three of job 9592 per (shape, T), O T = 32 has two valid configs: 11 + 12 = 23 runs.
ts total 261 runs (job 9592: 162) + 16 a3 (same). Checks (T = 64, 256, SB = 1): 70 + 12 = 82; SB = 0 T = 256: 70. Processes: 16 + 16 + 23 = 55 (job 9592: 32, 2 - 3 s each).
Estimate: 9592 took about 5 min for 162 + 16 runs (~1.7 s per config, process starts included); 261 + 16 runs and 55 starts: about 8 - 10 min (SB = 1 phases may be a few us slower, 112 phases x 8 launches x +3 us = +3 ms per config, negligible). `--time=00:30:00` leaves 3x margin.

### Arguments
1. Parity (SB = 1). full[s] use u (step f, u = f / ring) completes when 13 arrivals landed: 1 (producer expect_tx A, 32768 B, with its tx bytes) + 12 (one per staging warp). The MMA waits parity u & 1 as before. Each warp
   arrives for use u only after it saw empty[s] use u - 1 (parity ((f / ring) - 1) & 1; none for f < ring: first use), and empty use u - 1 completes only when the MMA committed step f - ring, which waited for full use u - 1. So no arrival of use u
   can occur before full use u - 1 has completed, and full use u cannot complete early (needs all 12 warps + A): exactly 13 arrivals per use per phase, no aliasing. The empty barrier cannot run ahead of a lagging warp either: empty use u
   needs full use u, which needs this warp's arrival for use u. Counts of empty (1), accfull (1), accempty (4 * NEPI) are unchanged.
2. Full count. The SB = 1 count is 13 = 1 + 12 everywhere: init (`SB ? 13 : 2`, once per launch), producer arrival (A only), all 12 warps 4..15 arrive once per step, including warps without an epilogue chunk (TT = 32). The producer's window
   prefetch of A for the next phase is unchanged and does not arrive twice (no B arrival from the producer when SB = 1).
3. Deadlock freedom. MMA of tile i needs: full of tile i's steps (A by producer, B by the 12 warps), accempty of tile i - 2 (G >= 2). The producer's A for step f needs empty of step f - ring (MMA of an earlier step). A warp stages step f
   after empty of step f - ring (MMA of an earlier step, same or earlier tile); it stages in step order and finishes all steps of tile i before running the epilogue of tile i, which waits accfull of tile i (MMA of tile i only, which needs only
   tile i's staged steps + A, plus accempty of tile i - 2, whose epilogue ran before this warp staged tile i). A warp blocked in the epilogue of tile i has already staged all tile i steps; a warp blocked on empty for step f waits only on MMAs
   of steps < f, which need staging by all warps of steps < f; every warp stages in order, so the earliest unstaged step of the slowest warp has all its dependencies met. By induction on the global step number all waits are satisfied.
   Cost (not a deadlock risk): staging of tile i + 1 starts after the epilogue of tile i (no staging / epilogue overlap within a CTA; irrelevant at 1 tile per CTA, the best configs). Across phases, `gbar` is reached by every thread after the
   last accfull wait; the staging warps never wait on anything of the next phase before `gbar`.
   (Not model-checked beyond the xover3 CPU model; the SB change is the extra 12 arrivals and the staging wait order, argued above.)
4. Proxy fences. B bytes are written with generic `st.shared` and read by the MMA async proxy. Each writing thread executes `fence.proxy.async.shared::cta` after its stores; `__syncwarp` orders the lanes; lane 0's `mbarrier.arrive`
   (release.cta) then publishes them, and the MMA issuer's `mbarrier.try_wait` (acquire.cta) + `tcgen05.fence::after_thread_sync` precedes the MMAs (as in mk `stage_tile` / `lm_phase`, which fence in every thread before the barrier arrive).
   WAR: a slot is rewritten (generic stores) only after empty[s], which the commit completes after the MMAs of the previous use have read the slot (async proxy reads finished), and the TMA write of A into the same slot also waits for it.
   TMA A bytes and generic B bytes go to disjoint ranges of the slot (A 0..32767, B from 32768).
5. Layout. Same bytes as the TMA B box: row r (token) 128 B, 8-row groups 1024 B, 16 B chunk index XORed with `r & 7`, k-atom 1 at `+ TT * 128`; slot base is 1024-aligned (checked for TMA B before), descriptors are unchanged.
   Verified only by the check run (`k_ref` vs bf16(Xf * r)) which I cannot run here.
