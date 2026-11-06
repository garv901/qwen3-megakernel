# B8 b8_compose: whole GEMM phases of three designs vs mk stamps

Tags: [M] measured, [H] hypothesis. Measurements only; no conclusion about mk's design.
Jobs: 14303 mk stamps (b128/b256, sets C, T, C_null, T_null; `results/2026-10-29-occfuse/b8_compose/mk_stamps_14303`), 14304 pilot (failed at start: smem attribute, fixed), 14305 pilot (checks + smoke), 14306 final (all arms). 14302 was submitted to `priority` and cancelled before it ran (duplicate of 14303). 4 jobs ran; no job left.
Files: `b8_compose.cu`, `run.sbatch`, `mk_stamps.sbatch`, `analyze.py` (`analyze.py RESULTS report|terms`), `params.json` (compact numbers), results `report_14306.txt` (all tables), `main_14306_d{1,2,3}.{jsonl,st}`, `null_14306_*`, `pf0_14306_*`, `pf2_14306_*`, `fin0/fin2_14306*`.

## Method
- One kernel `k_b8`, 148 CTAs, 28 layers x (QKV, RoPE, attn, O, GU, D), mk's split barrier (`bar_arrive`, thread 0 window prefetch, `bar_wait`), per-CTA globaltimer stamps at arrive/exit of each barrier. RoPE and attn are spin fillers (no traffic). Metrics as mk (layers 4..23, median over layers and launches): T = median exit - previous median exit, work/busy/idle/last, Tmean = mean of per-phase T. [M]
- Phase engines are ports of mk's `gemm_phase` (classic: units 128x128k, contiguous ranges, ring 2 x 32 KB, B staged, red.add) and `ts_phase` (tile stream: producer/MMA/12 epilogue warps, B by TMA from bf16, ring R = min(204704/(32768+TT*256), 8), plain / red.add / GU-pair act epilogue), and `idle_finalize` (FIN 0 / FIN 2). Shapes QKV 4096x1024, O 1024x2048, GU 6144x1024 pair, D 1024x3072 (weights from a pool).
- Designs: 1 `classic` (all four phases classic, red.add, window prefetch). 2 `ts_ks1` (TS, KS 1 everywhere, GU pair epilogue). 3 `ts_split` (mk's TT/KS per NPAD bucket; QKV and D TS split-K, GU TS KS1 pair, O classic + finalizer, D finalizer; FIN 2 at NPAD >= 128 else 0).
- Prefetch pf 1 = window (mk). pf 0 = first A at phase start. pf 2 = producer-side issue of the next phase's A after the CTA's own last step (open item b).
- Correctness [M]: every design, phase and T in {8,96,192} checked against a CUDA-core reference (O_fin, D_fin outputs for design 3 too): 42 / 42 pass (job 14306). (Job 14305 had D fail at a tolerance that was too tight for K = 3072; tolerance is now relative to sum|bw|.)
- LM phase: not implemented (optional).

## Deviations / defects
1. [M] `--fill` parse bug: nested `strtok` applied only the first entry of the filler list. In job 14306 only T = 1 got its mk-derived filler (800 / 3360 ns); T >= 8 ran with the default 1100 ns RoPE and 3700 ns attn (bench RoPE + attn = 8.2 us at every T; mk: 7.7 us at b8, 18.9 at b32, 30.7 at b64, 56.3 at b128, 110-116 at b256). Fixed in the source after the job (`strtok_r`); not re-run (no job left). Effect: QKV and O see a shorter gap before them than in mk. GU and D follow GU and O directly in mk too, so their gap is unaffected. Layer times are not comparable with mk at T >= 32; sum of the four GEMM phases is used instead.
2. [M] pf 0 of design 1 (classic) hung (`timeout` 124, no row written) in job 14306. pf 2 of design 3 (ts_split) failed with `unspecified launch failure` at T 8, 64, 256 (watchdog build, no STUCK print). Both recorded as results; cause not found. pf 0 and pf 2 of the other design ran.
3. No `common/` change needed. `ub.cuh` `bar_wait`/`mbar_wait` spin without a bound; b8 has its own watchdog variants under `-DB8_DBG`.

## Bench vs mk, phase T (us), pf 1 [M] (job 14306 vs st_14155 / 14303; classic vs mk C, ts_split vs mk T)
| T | design | QKV b/mk | O b/mk | GU b/mk | D b/mk | sum4 b/mk |
|---|---|---|---|---|---|---|
| 1 | C | 6.14/4.10 | 3.84/2.82 | 8.45/5.89 | 6.91/4.61 | 25.3/17.4 |
| 1 | TS | 5.63/3.33 | 5.89/4.35 | 10.50/6.91 | 8.45/5.38 | 30.5/20.0 |
| 8 | C | 6.14/4.10 | 4.10/2.82 | 8.70/6.14 | 7.04/4.86 | 26.0/17.9 |
| 8 | TS | 5.89/3.58 | 5.89/4.61 | 10.75/7.17 | 8.45/5.38 | 31.0/20.7 |
| 32 | C | 6.66/4.35 | 4.35/3.07 | 9.22/6.40 | 7.68/5.38 | 27.9/19.2 |
| 32 | TS | 6.40/3.84 | 6.66/4.86 | 11.26/7.68 | 8.70/5.89 | 33.0/22.3 |
| 64 | C | 7.94/5.38 | 4.61/3.58 | 10.75/7.42 | 9.47/7.68 | 32.8/24.1 |
| 64 | TS | 6.91/4.10 | 7.68/6.40 | 13.31/9.73 | 9.98/6.66 | 37.9/26.9 |
| 128 | C | 11.26/8.45 | 5.63/4.61 | 14.34/10.75 | 14.34/12.29 | 45.6/36.1 |
| 128 | TS | 7.17/5.12 | 9.73/6.91 | 12.03/8.19 | 10.37/6.91 | 39.3/27.1 |
| 256 | C | 18.18/16.38 | 7.94/7.94 | 22.78/19.20 | 23.30/32.00 | 72.2/75.5 |
| 256 | TS | 8.19/6.40 | 11.78/9.47 | 15.10/10.50 | 11.01/7.42 | 46.1/33.8 |

Bench-only T (1 / 8 ... see `report_14306.txt`): T 16, 96, 192 and the `ts_ks1` design. `ts_ks1` is 2x slower than the others at every T (O 14.9-17.3, D 24.1-26.4 us; D busy 20.9 us for 24 steps in one tile = 0.87 us/step; B3 gives 0.525) [M].
Detail (work/busy/idle/last/lat/spread): `report_14306.txt`.

## Gates
- Small T ranking (classic beats TS), sum4 layer gap, bench vs mk: T1 5.1 vs 2.6; T8 5.0 vs 2.8; T32 5.1 vs 3.1; T64 5.1 vs 2.8 us. Ranking PASS; the magnitude gap is 1.7-2.0x mk's 2.5-3.0 (reported, not a gate). [M]
- Small T, each phase within 20 % of mk: FAIL. Bench is +23 to +69 % slower in 31 of 32 (design, phase, T <= 64) cells; one cell (TS O at T 64) is +20 %. [M]
- Large T (128, 256) TS beats classic: PASS in both (sum4 39.3 vs 45.6 at 128, 46.1 vs 72.2 at 256; mk 27.1 vs 36.1 and 33.8 vs 75.5; mk layer 84.0 vs 92.8 and 144.6 vs 191.9 us). [M]
- Large T each phase within 20 % of mk: FAIL. TS +24 to +50 % everywhere; classic within 20 % in 4 of 8 cells (QKV 256 +11, O 256 0, GU 256 +19, D 128 +17), D at 256 is -27 % (mk classic D 32.0 us, bench 23.3). [M]

## Null arms, open item (a) [M] (bench null vs mk T_null / C_null, T us / busy us)
Bench null floor is higher than mk's in every phase.
- Classic: T +0.5..+0.8 us (QKV 2.82 vs 2.05, O 2.30 vs 1.54, GU 4.10 vs 3.58, D 2.82 vs 2.05), busy +0.3..+0.5.
- TS (B2 B-box gap): QKV T 4.86 / busy 2.34 vs mk 2.30 / 0.80; GU 9.47 / 6.40 vs 4.10 / 2.62 at T 8 (9.73 / 7.36 vs 4.86 / 3.65 at T 256); D 5.12 / 2.34 vs 2.05 / 0.83; O (classic + finalizer) 2.30 / 0.99 vs 1.28 / 0.26. TS null T is flat in T (T 1 to 64) in both bench and mk.
- Constant offsets of this size (0.8 us classic, 2.1-3.7 us TS busy) match the size of the bench-minus-mk gap at small T in the table above (e.g. classic QKV +2.0 us, TS GU +3.6 us at T 8). [H] The bench TS phase carries a fixed per-phase cost that mk does not (init / alloc / null-path epilogue in the bench engine); not isolated.

## Prefetch arms, open item (b)
- pf 2 (producer-side issue after own last step), classic, layer us: T8 39.1 vs pf 1 34.2; T64 47.5 vs 40.9; T256 95.5 vs 80.6. Per phase at T 8: QKV 7.68 vs 6.14, O 5.12 vs 4.10, GU 10.24 vs 8.70, D 8.19 vs 7.04. pf 2 is slower than the window at every point measured. [M] pf 2 for ts_split failed to launch (defect 2). pf 0 classic hung (defect 2).
- Window-prefetch cost / gain, ts_split, pf 0 minus pf 1 layer us: T1 +2.3, T8 +2.4, T32 +2.2, T64 +2.5, T128 +0.9, T256 +1.4 (filler bug applies to both). [M]
- B7 KS 2 gap vs xover3: no separate KS 2 arm was run (job budget). The bench's own T 256 QKV (KS 2, TT 128): T 8.19, busy 6.50 us. Not compared with xover3. Open.
- Finalizer placement, ts_split (fin 0 / fin 2 layer us): T8 39.3 / 41.3, T64 46.1 / 46.5, T128 49.8 / 47.6, T256 55.3 / 54.1. [M]

## Term-sum check (params.json terms, no refit; `analyze.py terms`)
Pred = t_bar 1.2 + max(stream side, chain) + piece + finalizer. Stream side = B1 t0 + S/min(r_sm, BW/N) minus B2 hidden (slope 0.408 x min(R, steps) x 32 KB / rate) plus B4 B path (classic staged a1_idle per unit; TS TMA b_idle, 2 boxes per k-step). Chain = 0.525 us x max steps. Piece = B7 1.382 per red tile + 0.956 per plain tile. Finalizer = B6 (c_all at FIN 2, a_idle at FIN 0). Classic at T > 128 multiplies chain, B path, A stream and piece by ceil(T/128). GU-pair epilogue (B5) has no term in this sum.
29 of 108 (design, T, phase) cells are within 20 % (full table: `report_14306.txt`). Mean signed error (pred vs measured):
- classic: QKV -16 % (T <= 64), +8 % (T >= 96); O -12 / +28; GU -16 / +23; D -27 / -14. At T 192/256 the ceil(T/128) scaling over-predicts O (+38..+58 %) and GU (+30..+55 %).
- ts_split: QKV -39 / -29; O -15 / -1; GU -49 / -34; D -40 / -34.
- ts_ks1: -31 to -55 % in every phase.
Terms that are off (> 20 %) [M]:
1. A constant per-phase floor is missing from every term: bench null T is 2.3-4.1 us classic and 4.6-9.7 us TS against t_bar 1.2, so small-T cells (where no term is large) are under-predicted by 1.5-5 us.
2. B3 / B1 overlap for TS KS 1 tiles with many steps (ts_ks1 D 24 steps, O 16): measured 0.87-0.93 us per step vs chain 0.525 and stream 0.32 us; the max(stream, chain) is 40-45 % low.
3. B5: TS GU pair epilogue is absent from the sum; ts_split GU is 47-52 % low at T <= 64 (bench GU null busy alone is 6.4 us, mk 2.6).
4. ts_split O (classic units + B6 finalizer + B7 piece) is within -19 to +20 % at T >= 96 (-9 to -19 % at T <= 64), so B6 and B7 are not the largest misses there. ts_split D (TS split-K + finalizer) is -32 to -41 % at every T: with B6 included, B1/B3 plus the missing floor (item 1) carry the miss [M].
5. classic D at 256 measured 23.3 us (mk 32.0): measured is above the sum without the ceil(T/128) scaling and below it with it.
Predictor cells within 20 % mostly sit at classic T 32-128 where stream + B path dominate.

## Caveats
- Single job, single node; layers 4..23 over 5 launches. Median grid 0.256 us: the plain `analyze.py RESULTS` mode prints Tmean next to T for every arm (Tmean differs from T by < 0.15 us in the pilot rows).
- Filler defect (above) and the constant bench-minus-mk floor mean the 20 % phase gates fail for a reason that is partly harness [H]; not separated.
- mk b8/b32/b64 stamps come from st_14155 (MK_TS bucket knobs read from the npz); b128/b256 from job 14303.
