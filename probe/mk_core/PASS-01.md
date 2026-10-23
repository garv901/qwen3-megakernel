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

## Results
(pending job 9579)
