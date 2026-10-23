# PASS-03: LM head pipeline (MK-core, P3 of PASS-PLAN.md)

Order note: P2 (drop the phase-2 barrier) waits on MK-attn's A1 v2 (SPEC-A1v2.md, accepted 2026-10-23, lands after their P2 job 9581).
P3 goes first; P2 follows when v2 is in.

## Before the build
- Phase isolated: LM head. Stamps (job 9583, PFA on): LM phase ~99 us at R0-b1, 112 at R1-b16 (V0.md, job 9576). Null-LM saves 62
  (b1) / 77 (b16) us (job 9576). Byte floor 48 us; fi 45.7 (b1) / 57.4 (b16) (job 9564). xover2 (job 9566): ring max + X by TMA ->
  85 us at T = 1; oracle (loads only) 52.6.
- Cause [H]: gemm_phase keeps ~1-1.5 units in flight (refill after the previous MMA, per-unit restage + __syncthreads) and the
  single TMEM accumulator stalls MMA at every row-block flush.
- Change (SPEC-lm.md, knob MK_LMP, NPAD 16 / 32): X staged once; producer / MMA / epilogue warp roles; LM ring of MK_LMR slots
  (default 4) over the whole smem; double-buffered TMEM accumulator.
- Band [H]: LM 99 -> 55-65 us at b1 and b16: -35..-45 us/step. Risk: the larger dyn smem (NPAD 16: ~108 -> ~165 KB) shrinks the L1
  carve-out, which costs the placeholder attention at long KV (ring 3: +30 us at R2-kv8192, job 9583). Measured in the A/B.
- Checks: 0-line SASS diff with MK_LMP=0; SASS counts; A/B with ab.py (R0-b1, R1-b16, R1-b32, R2-b1-kv8192, R6-q4) MK_LMP 0 / 1,
  MK_LMR 2 / 4 / 6; accuracy band at every point.

## Results
(pending)
