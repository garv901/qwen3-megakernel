# kvq_pf: record prefetch (KVQ_RPF) at the unit boundary, with and without R1 (KVQ_ACC)

Base: kvq_diag/kvq.cu (EOI 3, TL) + R1 `KVQ_ACC` 1 / 2 / 3 ported from kvq_alt (red.f32 / v2 / v4, normalize_head, accb double buffer, check scaling in kvq.py). ACC 4 (TMA reduce), R2 (tail arm) and R3 (merge queue) not ported.
Run: `MODE=sweep|tune|tl [VARS="a0r0 a0r3 .."] sbatch -p main probe/mk_core/kvq_pf/kvq_pf.sbatch` (calibration gate first). Variant names aXrY[dN][lN] = KVQ_ACC X, KVQ_RPF Y, KVQ_RD N, KVQ_RLD N. Tables: `analyze_pf.py [--md]` (results/2026-10-28-kvq-pf/pf_table.md), `analyze_tl.py` (tl_table.txt).
Jobs (all B200, `main`, 1 GPU): calibration + tuning 12260 12261 12270 12271 (results/.../v1_preguard); final sweep 12301 12302 12303 12304 (replicates 12301/12303 = a0r0 a0r3 a1r0, 12302/12304 = a1r3 a3r0 a3r3); timeline 12305. Calibration gate passed in every job.

## Mechanism
produce() at a unit boundary took the ticket (atomicAdd, prefetched one unit ahead by queue_pf) and then did two dependent global loads, units[uid] and uinfo[uid]: 0.93-1.0 us (interior stage 0.29). Everything else at the boundary (prep, meta smem write, TMA issue) is cheap.
KVQ_RPF 0: unchanged (SASS of the diag build + ACC).
KVQ_RPF 3 (the design that works): ticket as queue_pf (atomicAdd issued when unit k starts, claims unit k+1), the record of k+1 (units[], uinfo[]; 2 x LDG.128, registers) is issued KVQ_RD produce() calls later (default 1: the ticket value is read one stage after the atomic was issued), at the end of produce() after the stage's TMA ops. At the boundary the registers are read: nothing global on the path. If the loads were not issued yet (1-stage units, first unit of a layer) the boundary falls back to the old inline path. Static arms: record of the next unit of the static list issued right after the boundary (known address). Plain `queue` (pf 0) has no ticket ahead: unaffected (the prefetch step is skipped for it).
KVQ_RPF 1 (spec's "ticket already held": a second ticket claimed one unit ahead, record of k+1 loaded at unit k's start): built and checked, not better (see below), not used.
KVQ_RLD: 0 ld.global, 1 ld.global.nc, 2 cp.async 2 x 16 B into a smem slot (wait_all at the boundary). Consumers' view unchanged (per-slot smem meta written by prep()).
KVQ_RPF_EOI: not done (see below). KVQ_RPF 2 (two units ahead): not built: after RPF 3 the produce() boundary is within 0.1 us of interior and the rest of the bubble is the EOI step.

## Build
128 regs, 0 stack, 0 spills for ACC 0 (RPF 0 / 1 / 3, RLD 0 / 1 / 2, same as the diag build); 127 regs for ACC 1 / 3. 512 threads, dyn smem 165248 (as kvq).

## Timeline (KVQ_TL, job 12305, last layer of the last launch; us; medians over CTAs)
produce() at a unit-first stage (to the end of the TMA issue, `prodB`) against an interior stage (`prodI`), and the unit-boundary bubble (stage period across a unit boundary minus inside a unit):
| point / config | prodB RPF 0 | prodB RPF 3 | prodI RPF 0 / 3 | bubble RPF 0 | bubble RPF 3 | EOI RPF 0 / 3 |
|---|---|---|---|---|---|---|
| b8kv32768 queue_pf C512 cap32 | 0.93 | 0.40 | 0.28 / 0.30 | 0.78 | 0.43 | 1.27 / 1.03 |
| b8kv32768 queue_pf C256 cap32 | 0.93 | 0.40 | 0.29 / 0.30 | 0.77 | 0.46 | 1.28 / 1.02 |
| b8kv32768 queue_pf C256 none | 0.84 | 0.42 | 0.28 / 0.29 | 1.49 | 1.32 | 1.56 / 1.39 |
| ragged64 queue_pf C512 cap32 | 0.92 | 0.39 | 0.29 / 0.31 | 0.93 | 0.64 | 1.38 / 1.13 |
| ragged64 queue_pf C256 cap32 | 0.91 | 0.40 | 0.29 / 0.31 | 1.10 | 0.85 | 1.48 / 1.26 |
| ragged64 queue_pf C256 none | 0.94 | 0.43 | 0.29 / 0.30 | 1.60 | 1.42 | 1.67 / 1.45 |
| b8kv32768 static | 0.56 | 0.53 | | 0.46 | 0.46 | 0.90 / 0.90 |
prodB falls to 0.39-0.43 (interior + 0.1 us). Including the prefetch step the thread-0 time at a boundary (`pfullB`) is 0.52 us; interior calls 0.38 vs 0.31 (the single call that waits for the ticket). The bubble falls by 0.3-0.35 us per unit where units have >= 4 stages. What remains is the EOI step (1.0-1.4 us, 3 CTA barriers + smem merge + stores, all 16 warps): the EOI itself also got 0.25 us shorter (it used to wait for thread 0). C256 cap none (1-2 stage units, ~40-56 units per CTA): bubble still 1.3-1.4 us, the EOI dominates.
Static: boundary cost is small already (0.5-0.56, one record per CTA unit).

## Tuning (queue_pf A us, tune jobs 12260 / 12270 in v1_preguard; check PASS on all)
| variant | b8kv32768 C512 cap32 | b8kv32768 C256 none | ragged64 C512 cap32 | ragged64 C256 none | b1kv245760 C256 none |
|---|---|---|---|---|---|
| RPF 0 | 151.9 | 161.3 | 209.1 | 219.6 | 151.6 |
| RPF 1 RD 1 (extra ticket depth) | 152.5 | 160.5 | 209.7 | 215.6 | 151.2 |
| RPF 1 RD 1 + ld.nc | 152.7 | 160.9 | 209.9 | 215.7 | 151.4 |
| RPF 3 RD 0 | 151.6 | 148.3 | 206.5 | 201.1 | 138.6 |
| RPF 3 RD 1 | 151.5 | 148.1 | 206.3 | 201.0 | 138.7 |
| RPF 3 RD 1 + ld.nc | 151.5 | 148.1 | 206.4 | 201.0 | 138.6 |
| RPF 3 RD 1 + cp.async | 151.6 | 149.9 | 206.5 | 203.1 | 140.4 |
| RPF 3 RD 2 (job 12260) | 151.5 | 160.9 | 206.4 | 218.7 | 150.4 |
RD 2 loses the C256 gain (2-stage units: the loads would be issued after the boundary). RD 0 = 1 (the 1-call-old atomic has mostly returned). ld.nc = ld.global (units / uinfo are L2 hits and the cost is the dependent latency, which the prefetch hides). cp.async slightly worse (wait_all + lds on the critical path). RPF 1 (second ticket held) does not help: its boundary step still costs 0.82-0.88 us (thread 0 waits at the boundary; cause not found), and it claims 2 units ahead.

## Final sweep (jobs 12301-12304; 10 points x static / queue / queue_pf, C 512 1024 (+ 256 at b8kv32768, ragged64), cap 32 / none, pos, --check each config: 0 FAIL of 1176 configs (98 per variant per job); replicate jobs)
Noise: |job1 - job2| of A over all configs: median 0.10-0.16 us, max 0.8-1.1 us (static: median 0.11-0.23, max 0.4-0.7); C median 0.02-0.10.
Calibration (every job): static b8kv32768 A 154.75-154.93 us (gate 154.6 +-3%), C 2.02-2.18; b1kv245760 static C 2.34; a1ub --comb head C 1.73-1.76 / 1.98.
Tables: pf_table.md. dA (queue_pf, A, same config as the best RPF 0 config; RPF 3 - RPF 0, median of 2 jobs) at ACC 0: b8kv4096 -0.2, b8kv16384 -0.6, b8kv32768 -2.0, b8kv30720 -2.3, b1kv245760 -2.3, b1kv32768 -0.3, b64kv1024 -2.0, b128kv512 -1.6, ragged8 -2.2, ragged64 -3.8 (ACC 1 / 3 the same within 0.5). Best-vs-best (RPF 3 re-optimises C / cap; cap none and C256 gain the most): b8kv32768 -3.1, b8kv30720 -5.2, ragged64 -4.1. Static A: -0.8 .. +0.6 (ragged64 -0.8; the others within noise, a slight +0.2..0.6 at the b8 / b1 big-KV points with ACC 1 / 3).
Plain `queue` (no ticket ahead) gains nothing and was 0-3 us slower in the first (pre-guard) build because of the prefetch bookkeeping in produce(); with the guard (final) it is +0.1..+0.9 us vs RPF 0 (ACC 0, same configs; code-layout / register effect, near the 0.5 us noise).
Best A + C (us; % vs static of the ACC 0 RPF 0 build; queue_pf best C / cap per point):
see pf_table.md section 2 (reproduced in the final report).

## Deviations
1. RPF numbering: RPF 3 is the working design (ticket as queue_pf, lazy record load); RPF 1 = second ticket held (spec's literal RPF 1) was not better; RPF 2 not built.
2. KVQ_RPF_EOI not done: thread 0's boundary work is no longer the cause of the long EOI barrier wait (EOI is 0.25 us shorter), the residual EOI is the 16-warp merge + 3 barriers; moving it needs a redesign of the EOI, not a simple change.
3. ACC 4, R2, R3 not ported. The ACC check stores the scaled piece to the slot as in kvq_alt.
4. Tuning job runs before the final guard edit (`pf || sidx < send`) are kept in results/2026-10-28-kvq-pf/v1_preguard.
