# B6 b6_fin: serial materialization (finalizer) chain
Jobs: 14296 (pilot, pre-fix build; gate row valid, place-3 group hung on a bug, see below), 14300 (full sweep + gate repeats, 34 groups, no hang, 646 rows, 0 FAIL). Both on node cayenne, priority. Raw: `results/2026-10-29-occfuse/b6_fin/` (`all_14300.jsonl`, `analysis_14300.txt`, `.st`). Tables: `analyze.py`, fit + params.json: `fit.py`. Clock (clk_sample): median 1962 MHz, min 1942 (producers do no MMA, so no power cap). Slurm job 14292 was an earlier srun sanity run that failed at start (smem attribute bug); counts as a launch.

## Method [M]
- Kernel `b6_fin.cu`: 148 CTAs, 512 threads, coop launch, B0 split barrier + stamps, 112 phases, x double-buffered by phase parity (next buffer zeroed at phase start; it was last read a phase earlier).
- Producers: unit u = (rb = u / KS, kc = u % KS), NB = H / 128 row blocks, NTILE = NB*KS units, Pc = min(NTILE, 128) producer CTAs spread as mk (`lo = b*Pc/G`; units [lo*NTILE/Pc, (lo+1)*NTILE/Pc)). Each unit: red.add of 128 rows x T fp32 into x[T][H] (real red.add traffic), `__syncthreads`, tid 0 `__threadfence`, stamp, ticket `atomicAdd` per token tile (NTT = ceil(T/128), job = rb*NTT + tt, target KS), as mk `fin` lambda. A producer delay (spin, `pdelay`, default 1500 ns, about mk's O busy work 1.63 us) replaces the GEMM; also run with 0.
- Finalizer body = mk `idle_finalize` item loop (16 threads per token x 8 columns, `__ldcg` x2 float4, pack_xw with ln weight, uint4 store of bf16, shuffle-reduce sum of squares, `atomicAdd` ss), then `fence.proxy.async.global`.
- Placements (mk semantics as read in mk_step.cu):
  - a = MK_FIN 0: CTAs with no unit (rank = b - lo, nidle = G - Pc) take jobs r, r + nidle, ...; whole job (all tokens of a 128-token tile) per CTA; tid 0 polls `ld.acquire` + `__nanosleep(64)`, then resets the ticket.
  - b: the CTA whose ticket add returns `old + nk == KS` runs that job itself (no poll). Not an mk mode.
  - c = MK_FIN 2: all CTAs after their own tickets; w = blockIdx, job = w % NJOB, part = w / NJOB, nsub = max(1, G / NJOB), pr = ceil(128 / nsub) token rows per CTA, seen counter resets.
  - place 0 (base): producers only, no tickets: the "classic" arm for the gate.
- Arms (var): 0 full; 1 empty finalizer (poll/ticket + fence + arrive); 2 read + convert + ss, xb store suppressed (guarded by an impossible value).
- Stamps (global timer, 32 ns): tp[rb] = atomicMax of the time just before the ticket add (last producer arrival of the block); per finalizing CTA: observe time (after poll / after atomic return), end of the job loop (after `__syncthreads`, stores issued), end after fence. t_fin = max over CTAs of fence-end - max over rb of tp. Split from the CTA that ends last: obs = observe - tp of its block; rc = loop end - observe (read + convert + ss + store issue); wf = fence end - loop end. Means over phases 2..112, 7 launches (2 warm-up); quoted numbers are means (medians sit on the 32 ns grid, same numbers within 0.03).
- Correctness [M]: after each launch, `k_check` (CUDA cores) compares x with the sum of the KS partials in double (tol 2e-5), xb with bf16(x * ln w) bit-exact (var 0), ss with sum x^2 (rel 1e-3, var 0/2). 646/646 rows pass.
- Gate definition: mk "last" = max over CTAs of (arrive - previous exit) from the barrier stamps; placement a minus base (no tickets, no finalizer), same producers.

## Deviations
- Producers do a spin + red.add, not a GEMM; absolute "last" differs from mk (bench 4.61 with pdelay 1500 vs mk T-mode O 3.54); only the difference is compared.
- Pc capped at 128 for every placement (mk O: 128 units, 20 idle). For NTILE > 128 CTAs own several units (a still has 20 idle). For KS 4 at H 1024 and KS 12 (NTILE 32, 96) Pc = NTILE.
- Placement b is not an mk mode.
- Stamps of one launch kept for var 0 and base rows only.
- Run script runs one process per (placement, T) group under `timeout 100`.

## Results [M] (job 14300; pdelay 1500)
t_fin (full finalizer, us). H, KS rows; columns T = 1 8 16 32 64 128 256.
```
a idle   H1024 KS4   1.35 1.49 1.50 1.64 2.26 3.42 3.42   | H1024 KS16 1.57 1.61 1.68 1.84 2.40 3.48 3.55
         H2048 KS4   1.49 1.52 1.49 1.65 2.31 3.44 3.51   | H2048 KS16 1.63 1.66 1.74 1.85 2.40 3.56 6.49 (32 jobs on 20 idle CTAs: 2 sequential)
b last   H1024 KS16  1.76 1.84 1.88 2.02 2.61 3.75 7.25   (T 256: both token tiles finalized by the same CTA, sequential)
c all    H1024 KS16  1.65 1.67 1.73 1.77 1.86 1.91 1.95   | H2048 KS16 1.68 1.73 1.75 1.87 1.88 1.93 2.08
```
Full tables for all H, KS: `analysis_14300.txt`. KS dependence is small: a at H1024 T 8: KS 4/12/16/24 = 1.49/1.50/1.61/1.57; b: 1.84-1.86; c: 1.67-1.71.

Split (H1024 KS16, T 8; us): 
| placement | obs (ticket -> observed) | read+convert+ss (rc) | store extra (full - noStore) | fence | empty-arm t_fin | full t_fin |
|---|---|---|---|---|---|---|
| a idle | 0.84 | 0.75 (T 1: 0.72, T 64: 1.51, T 128: 2.64) | -0.01 (T 64: +0.09, T 128: +0.13) | 0.07 | 1.00 | 1.61 |
| b last producer | 1.08 | 0.74 | -0.02 | 0.09 | 1.24 | 1.84 |
| c all CTAs | 0.87 | 0.77 | -0.02 | 0.06 | 1.33 | 1.67 |
- read+convert (noStore - empty) at T 8: a 0.62, b 0.63, c 0.36; the xb write adds 0 to 0.13 us (stores are issued, not waited for; the fence costs 0.06-0.18).
- obs depends on the number of producer CTAs: a: 0.62-0.67 at 64-96 producers, 0.76-0.86 at 128 (KS 16, 24). 

Fit (mean over rows, var 0, T 1..256, H 1024/2048, KS 4..24; params.json; x = [1, n_it, (m-1)*n_it, (m-1)], n_it = ceil(min(T,128)/32) = 32-token iterations of one 512-thread job, m = jobs per finalizing CTA):
| placement | a (us) | per iteration (us) | extra job per iteration (us) | extra job fixed (us) | rmse (max res) |
|---|---|---|---|---|---|
| a idle (m = ceil(NJOB / nidle)) | 1.009 | 0.624 | 0.698 | 0.175 | 0.107 (0.284), n 58 |
| b last producer (m = ceil(T/128)) | 1.285 | 0.624 | 0.817 | 0.204 | 0.074 (0.155), n 56 |
| c all CTAs (n_it = 1 always; x = [1, min(T,128)]) | 1.705 | 0.002 per token | - | - | 0.064 (0.174), n 56 |
Empty-arm floor (T <= 128): a 0.96, b 1.24, c 1.32 us.
Pooled mean components: a obs 0.94 / rc 1.41 / fence 0.10; b 1.56 / 1.43 / 0.08; c 0.97 / 0.77 / 0.13 (these pool all T, including the 2-job cases).
pdelay 0 vs 1500: t_fin differs <= 0.02 (H1024 KS16, T 8/64/256, a/b/c).

## Gate [M]: PASS
O-like config (H 1024, 8 row blocks, KS 16 = 16 producers per block, 128 producer CTAs + 20 idle, T 8, placement a), last(a) - last(base):
- job 14300, pdelay 1500: 4.61 - 2.56 = +2.05 (means over phases 4.53 - 2.45 = +2.08); three passes, nl 7-9; identical each time.
- job 14300, pdelay 0: 3.07 - 1.02 = +2.05.
- job 14296 (second job): 4.35 - 2.56 = +1.79 (means 4.49 - 2.44 = +2.05).
Target +1.8 +/- 0.4 (1.4..2.2): inside, upper half. Last arriver is a finalizer CTA in 100 % of phases (mk: 100 %). t_fin there = 1.60; the rest of the difference is ticket cost on the producers. Per-phase values sit on a 0.256 grid (the median over CTAs/`last` stamps, 4.35 vs 4.61 is one grid step).

## Caveats / open
- [M] First build (job 14296): the seen counters sat past the end of the ticket allocation (1 KB), so placement c hung at its first config (timeout 124, recorded in `hangs_14296.txt`); fixed (4 KB, seen at tk + 512). Rows from 14296 pilot_a/pilot_gate are unaffected; the pilot_b rows of place 2 (kept in pilot_14296.jsonl) too.
- [M] a at H 2048 with KS >= 12 and T 256, and b at T 256: two jobs on one CTA run in sequence (obs of the last job 4.5 us for b). These points follow the m terms of the fit and are not a general finalizer cost.
- [H] The obs rise with 128 vs 96 producers may be red.add queueing in L2 ahead of the ticket; not isolated.
- Producer arrival spread is small (all start at the barrier exit with equal work); mk's producers are staggered by MMA/B timing, which can change obs.
- The xb "write" term is store issue + fence only; visibility to a TMA reader after the barrier is not measured here.
- The second pass for the gate was a second job (14296) plus 2 repeats inside 14300. Only one node.
