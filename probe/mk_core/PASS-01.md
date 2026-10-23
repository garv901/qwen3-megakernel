# PASS-01: grid-barrier cost (MK-core, P1 of PASS-PLAN.md)

## Before the build
- Phase isolated: the grid barrier (6 per layer + prologue + LM). v0 stamps (job 9576): arrive -> exit 1.7-2.7 us for the
  phases with balanced work at b1 (QKV 1.70, O 2.08, GU 1.86, D 2.69; RoPE 2.05 after a 0.13 us phase).
  MK-attn's bench measures 1.12 us with the same hv2g barrier (job 9567) [M, other bench].
- Hypotheses [H]: (a) 148 same-address atomics serialize at one L2 slice; (b) polls hit the line that the atomics hit;
  (c) the spread of CTA arrival times (unit-count imbalance: QKV 256 units / 148 CTAs = 1 or 2, GU 384 = 2 or 3,
  D 192 = 1 or 2, O 128 = 0 or 1) is part of "wait".
- Bench: `probe/mk_core/bar/` (SPEC-bar.md, 7 arms: v0, acquire poll, counter + flag, two-level, per-CTA words, cg grid.sync,
  warp poll), skew 0 and 100 ns, validation of every arm. Job 9579.
- Band [H]: if the best arm is <= 0.6 x v0's barrier, -60..-100 us/step at b1 (6 x 28 barriers x 0.4-0.6 us).
- Then: port the best arm into mk as knob `MK_BAR` (default = v0 until the A/B), 0-line SASS diff when off;
  SASS check; e2e A/B (3 placements x 2 GPUs: the effect per step is < 5 us per barrier but ~100 us per step, so one
  placement per GPU is enough if the step delta is > 5x the CI); ledger row; tag mk-core-v1 if kept.
- Imbalance (c) is a separate lever: measured by the null-work stamps (MK_NULL 0x7f: barriers + TMA only) vs the bench.

## Results [M]
- Bench (job 9579, GPU-019267a2): us per barrier, skew 0 / 100 ns: v0 1.095 / 1.726; acquire poll 1.166 / 1.794; counter + flag
  1.742 / 2.427; two-level 2.463 / 3.166; per-CTA words 3.137 / 4.513; cg grid.sync 1.255 / 1.872; warp poll 1.668 / 2.358.
  All PASS validation. **v0's barrier is already the cheapest**: hypotheses (a), (b), (c-fence) are wrong.
- In mk (job 9580, stamps spread; globaltimer resolution ~0.256 us): latency (median exit - last arrive) 1.0-1.3 us for most
  barriers but 2.05 after O and 1.79 after D, also with all work nulled. Cause [H, confirmed by the A/B]: the next phase's
  weight prefetch was issued BEFORE the arrive, so the arrive's release fence (MEMBAR.ALL.GPU) waited behind 32-64 KB of TMA issue.
- Fix: knob `MK_PFA` (arrive, then prefetch). Off = 0-line SASS diff (NPAD 16, 256). A/B (job 9582, 9 interleaved rounds,
  one GPU): R0-b1 868.0 -> 801.8 (-66.1 [-67.3, -65.2]); R1-b16 -74.2; R1-b64 -25.9; R1-b256 -30.4; R2-b1-kv8192 -48.3.
  Accuracy unchanged (worst ratio 0.47-0.54). **Kept, default on.**
- After the fix (job 9583): latency 1.02 us at every barrier (= the bench floor). Remaining barrier cost = spread:
  attention 3.84 (b1, placeholder critical path), D 1.54 (192 units on 148 CTAs: max 2 = 1.5x the mean), RoPE 1.28, GU 0.77.
- Ring depth (job 9583, PFA on): ring 3 / 4 vs 2: R0-b1 -21.8 / -24.7, R1-b16 -30.7 / -27.2, R3-b8-kv4096 +0.7 / +16.0,
  R2-b1-kv8192 +29.5 / +66.7 [M]. The long-KV loss is the placeholder attention [H: smaller L1 carve-out]; ring stays 2 until A1
  lands, then re-measure.
- e2e: v0 (job 9578, tag mk-core-v0) beats fi at 6 / 28 (results/2026-10-23-mk/v0). v1 e2e job 9584.

## Next (not this pass)
- D unit imbalance (max 2 units vs mean 1.3): per-phase KU (D at KU 192: 128 units, max 1.5) -> P6.
- hv2g (FFMA) is 379 us at b1 vs mk 802: per-layer GEMM streams at 2.9-3.4 TB/s in mk (GU 12.6 MB in 3.92 us). P6
  (whole CTA share in flight before the barrier exit) targets this.
