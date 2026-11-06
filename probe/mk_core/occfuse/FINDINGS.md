# occfuse findings

The status of each bench. The fitted terms are in `params.json`. The method is in `METHODOLOGY.md`.
Tags: [M] measured (job id given), [H] hypothesis.

| Bench | Term | Status | Gate | Jobs | Fitted terms (summary) |
|---|---|---|---|---|---|
| B0 calib | t_bar(N), clock rule | done [M] | PASS with window loads (0.89–0.98); bare 0.80 (−0.02 outside band) | 14277, 14278 | empty phase T 1.157 us at every N; T = t_bar(N) + 1.00·W, t_bar 1.15–1.22; window loads +0.09 lat each; clock: no lock, power cap ~1700 MHz under MMA |
| B1 stream | t_stream(S, N, R) | done [M] | PASS vs best `why` arm (7141 vs 6727 GB/s, +6 %); +25 % vs plain static | 14279, 14281, 14282 | t = t0 + S/min(r_sm, BW_agg/N); R≥4: t0 1.0 us, r_sm 103 GB/s, BW_agg 7.4 TB/s; R 2: r_sm 63, R 3: 91; no placement effect |
| B2 prefetch | exposed(S, P, W, R) | done [M], partial gate | classic QKV/GU/D PASS (0.83/1.52/0.70 vs 0.69/1.48/0.49); TS A-only PASS (0.82/2.62/0.82); TS with B boxes QKV/GU FAIL (+0.31/+0.69) | 14287, 14289, 14294 | a slot prefetched in the window hides ≈ 0.9 (classic) / 0.7 (TS) of its transfer; window = time to the last arriver, not the phase W |
| B3 step | t_tile = a + SPT · t_step(N) | done [M], gate FAIL at N ≤ 128 | per MMA 103–121 c vs burst 58 (N16) / 80.5 (N128); N256 128 PASS | 14290, 14291 | mk-style: a ≈ 490 c, t_step 972 c (0.525 us) flat for N 16–128, 1164 c at N 256; commit per step +10–17 c/MMA; descriptor rebuild +7 c/MMA |
| B4 bpath | t_B(T, k, mode) | done [M] | PASS: T 256 staged 4.7 vs TMA 0.68 us (kb 128) | 14284, 14286 | TMA bf16: 0.35 + 0.06/box, flat in T ≤ 64; staged a1: 0.37 + 0.44/k-step + 0.056 us/KB; T ≤ 16 gap staged−TMA 0.1–0.3 us per 128 k (idle) |
| B5 epi | t_epi(TT, T, mode, KS) | done [M]; gate [H] | GU pair TT16 T8: phase increment 1.26 us = 0.38 of mk GU compute (3.33) | 14293, 14297 | t = a + b·n_iter + c·n_tok_max + d·T per mode; GU pair / argmax ≈ 0.07–0.08 us per valid token in the slowest group; fence 0.3–0.45 us; red.add KS contention ≤ 0.06 us for T < 64, +0.3 at T 128 KS 24 |
| B6 fin | t_fin(T, placement) | done [M] | PASS: O-like tail +2.05 / +1.79 us vs mk +1.8 ± 0.4 | 14296, 14300 | idle-CTA finalizer (MK_FIN 0): 1.01 + 0.62·n_it (+0.70·n_it + 0.18 per extra job); ticket observe 0.84, read+convert 0.75, fence 0.07 us; all-CTA (MK_FIN 2) flat 1.7–2.1 us |
| B7 splitk | t_red(KS, T) | done [M] | PASS: QKV T128 KS4 6.20 vs KS8 7.42 (Δ 1.22; 9592: 1.39) | 14301 | T = 3.81 + 0.646·max_steps + 1.38·red_tiles + 0.96·plain_tiles (rmse 0.38); extra K piece +0.1–0.3 us at T ≤ 32, +1.1–1.4 at T ≥ 128; workspace reduce slower than red.add by 1.3–6.1 us |
| B8 compose | phase T per design | done [M]; signs PASS, 20 % per-phase gates FAIL | small T: classic wins by 5.0–5.1 us/layer (mk 2.6–3.1); large T: TS wins (T128 39.3 vs 45.6, T256 46.1 vs 72.2; mk 27.1 vs 36.1, 33.8 vs 75.5); 31/32 small-T cells +23–69 % over mk | 14303 (mk stamps b128/b256), 14305, 14306 | term-sum check: 29/108 cells within 20 % |

## Notes for each wave

### Wave 1 (B0)
- [M] The bare split barrier gives lat 0.80 us at every N (8–148) and W. With 1–2 window TMA loads in flight (the mk state: classic `prefetch_next` issues 2 slots), it gives 0.89–0.98, which matches mk st_14155 b8 (1.02). [H] mk's extra lat comes from the window loads; not tested in mk.
- [M] T = t_bar(N) + W, slope 1.00–1.01, max deviation 0.12 us. So late arrivers add their lateness one to one.
- [M] The globaltimer steps by 32 ns. Medians over CTAs fall on a 0.256 us grid, so `stamps.metrics` now also returns `Tmean` (a telescoping mean).
- [M] Clocks cannot be locked. Idle runs are at 1965 MHz. Steady tcgen05 load is power-capped (clk_sample ~1700, NVML ~1865, throttle 0x4). [H] The 9 % gap between clk_sample and NVML under load is unexplained.
- Open: both jobs ran on one node. No L2-hit check for pool wrap at nld 4.

### Wave 2 (B1)
- [M] The per-SM stream rate is limited by the ring at R ≤ 3 (R 2: 63 GB/s, R 3: 91) and saturates at about 103 GB/s from R = 4. BW_agg is 6.5–7.4 TB/s and is reached only when N·r_sm exceeds it, i.e. N ≳ 72. Below that, the phase time scales with bytes per SM, not total bytes. This is the occupancy term.
- [M] N 48 × 256 KB without prefetch: busy 3.6 us (R ≥ 4), 4.9 (R 2). mk TS GU null busy is 2.62 with prefetch → B2 must explain ≈ 1 us of hiding.
- Open: a +0.4 us step between N 16 and 32 at S ≤ 64 KB; R 6–7 3–4 % below R 4 at N 148. Same node for all jobs.

### Wave 2 (B4)
- [M] TMA of B costs 0.35 us + 0.06 us per box and does not change with T up to 64. Staging costs 0.44 us per k-step + 0.056 us/KB of the smem tile. At T ≤ 16 the gap is small (0.1–0.3 us per 128 k when idle). At T 256 staged is 7–14× slower.
- [M] Batched staging (a2) beats per-step staging (a1) at T ≤ 64 (kb 1024 idle: T 1 is 0.56 us faster than TMA; T 16 is 1.4 us slower than TMA vs 4.2 for a1). This is the "batched B staging in classic" idea, measured in isolation.
- Caveats: under a concurrent A stream, a1 at T ≤ 32 costs 1.7–5 us more than idle (unexplained; the A ring polling interacts with staging) [H]. Smem wraps for the largest (T, kb); correctness was checked only without wrap (600/600 bit-exact).

### Wave 3 (B2)
- [M] The prefetch window is set by the last-arriving CTA, not by how long all CTAs work: an all-CTA spin W does not change busy work (median Δ 0.006 us over 5200 pairs). mk's windows at b8 are 1.1–2.6 us.
- [M] A slot issued in the window hides about 0.9 of its transfer time (classic) and 0.7 (TS, where B boxes are issued after exit). With this, all classic MK_NULL floors and the TS A-only floors reproduce within 0.3 us.
- Open: the bench's TS B boxes add 0.25 (QKV/D) and 0.70 (GU) us, which mk's TS null busy does not show. mk's NUL path does issue `ts_fill_b` (mk_step.cu:435/439), so the difference is not a null-mode artifact. B8 must check this against mk directly.
- Open: at N 148, P 4 busy work grows with the window (3.84 → 4.56 us, window 1.0 → 4.6) [H: pipeline restart]. At N 48 with no prefetch, B2 is 15 % slower than B1 (different refill loop).

### Wave 3 (B3)
- [M] One mk-style k-step (8 MMAs M128×N×K16 that walk the SW128 tile, commit per step, mbarrier handoff) costs 972 cycles (0.525 us at 1850 MHz) for every N from 16 to 128, and 1164 cycles at N 256. Per tile there is a fixed a ≈ 490 cycles. 1 vs 2 accumulators, 1 vs 148 CTAs and independent accumulator chains make no difference. Correctness: 384/384.
- [M] The gate fails against the hv2_tc 9425 burst (the reference is correct: c1 58.0, swab128n128 80.5 c in logs/9425.out). The burst issues 1024 MMAs back to back on fixed descriptors; the mk-style step costs ≈ 103 c/MMA even with lean descriptors and one commit per tile. [H] The floor comes from walking the operand addresses or from the step handoff; a raw back-to-back arm with walking descriptors would separate them (not run).
- Correction to earlier explanations: a TS k-step at M128 N128 is ≈ 0.53 us, not 8 × 80.6 c ≈ 0.35 us.
- Deviation: B3 has no phase barrier or `.st` stamps (pure compute, cycles from clock64).

### Wave 3 (B5)
- [M] At T ≤ 16 the epilogue time is the same for 4, 8 and 12 warps. The phase increment of a GU pair epilogue at TT 16, T 8 is 1.26 us; plain stores cost about 0.3–0.4 us per loop iteration. The store fence is 0.3–0.45 us.
- [M] Split-K contention on red.add is small: ≤ 0.06 us at KS 24 for T < 64, +0.3 us phase time at T 128 (KS 1 → 24).
- [H] mk GU compute at b8 (TS, full − null busy) is 3.33 us. The epilogue explains 1.26; the remaining ≈ 2.1 us must be the MMA chain (8 steps × 0.525 = 4.2 us, partly overlapped with the stream) and B. B8 checks the sum.
- Caveats: one node; clock 1800–1850 MHz; no weight stream or MMA/epilogue overlap in the bench.

### Wave 3 (B6)
- [M] The finalizer chain at T 8, H 1024 costs 1.6 us after the last producer arrives: ticket observe 0.84 us, read + norm + convert 0.75, fence 0.07. The store itself is ≈ 0. With the ticket cost on the producers, the phase tail grows by 1.8–2.05 us, which matches mk's O tail at b8.
- [M] It grows with T in 32-token iterations (0.62 us each) and with jobs per finalizing CTA (+0.7 per iteration per extra job). The all-CTA mode is flat at 1.7–2.1 us for T ≤ 128.
- Open: ticket observe rises 0.64 → 0.82 us with 128 vs 64–96 producers [H: red.add queueing in L2]. In the bench all producers arrive together; mk's are staggered.

### Wave 4 (B7)
- [M] With occupancy held through max steps per CTA (0.646 us per step), each output tile a CTA stores costs 1.38 us with red.add (KS > 1) and 0.96 us with plain stores (KS 1), per 128×128 fp32 tile scaled by min(T, TT)/128. So an extra K piece is cheap at T ≤ 32 (+0.1–0.3 us) and expensive at T ≥ 128 (+1.1–1.4 us).
- [M] A workspace + reduce pass is always slower than red.add (+1.3 to +6.1 us).
- Open: QKV KS 2 is 1.1–1.4 us slower than xover3 9592 (KS 4/8 agree within 0.13) [H: xover3 issues the next phase's A from the producer after its last step, not in the barrier window]. The window prefetch costs +0.2–0.3 us at ≤ 64 or 148 busy CTAs here; some CTAs leave the barrier up to 1.5 us late with it on [H]. Both need a check in B8.

### Wave 4 (B8)
- [M] mk stamps at large T (job 14303, results/2026-10-29-occfuse/b8_compose/mk_stamps_14303/): layer 84.0 (TS) vs 92.8 us (classic) at b128, 144.6 vs 191.9 at b256. GEMM sum (QKV+O+GU+D): 27.1 vs 36.1 and 33.8 vs 75.5.
- [M] The composed bench gets the sign right at every T (classic wins at T ≤ 64, TS wins at T 128/256) but is 23–69 % slower than mk per phase. The bench's TS null floors are far above mk's (GU busy 6.4 vs 2.6 us, QKV/D 2.34 vs 0.80/0.83 at T 8), although B2 reproduced those floors in isolation (2.62, 0.82, 0.82). [H] So the excess is in the B8 engine (a per-phase fixed cost of the composed kernel), not a missing physical term.
- [M] Term-sum check misses: no constant per-phase floor term; single-tile many-step TS KS 1 phases run at 0.87–0.93 us/step (chain 0.525, stream 0.32: the overlap rule is wrong for them); no GU pair term in the sum; classic at T > 128 (column passes) not captured.
- Harness defects: in job 14306 the RoPE/attention filler applied only at T 1 (strtok bug, fixed in source, not re-run), so layer totals at T ≥ 8 are not comparable with mk; GEMM sums are. pf 0 classic hung; pf 2 TS split had a launch failure. B7's KS 2 vs xover3 arm not run.
- Status: B8 is not yet a valid ground truth for magnitudes. Next step (not started): isolate the B8 TS null floor against B2 (same config, same binary flags), re-run with the filler fix.
