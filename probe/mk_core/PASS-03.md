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

## Results [M]
- Build (CPU): NPAD 16 86 regs / 0 spill, dyn 165248 B (LMR 4); NPAD 32 80 regs, 165248 B (LMR 3). Off = 0-line SASS diff at all 5
  buckets. LMR max: 5 at NPAD 16, 4 at NPAD 32 (227 KB).
- Job 9585: trap at the first launch: `arms/mk.py` sized the launch smem without the LM knobs (my SPEC missed that call site; fixed).
- A/B (job 9586, GPU-c098, 9 interleaved rounds), d vs MK_LMP=0 for LMR 2 / 3 / 4:
  R0-b1 -44.5 / -58.4 / -59.9; R1-b16 -62.9 / -65.3 / -67.8; R1-b32 -60.8 / **-66.9** / -53.1; R2-b1-kv8192 -29.3 / -51.6 / -50.8;
  R6-q4 -49.3 / -57.3 / -60.0. Accuracy unchanged. R1-b128 (same SASS in every arm) +18.8 / -2.5 / +6.2: **~20 us is the noise of
  the same binary at b128 in one job**, so the v1 vs v0 +68 at b128 (other GPU) is not a measured regression yet.
- **Kept, default on:** MK_LMP=1, MK_LMR=0 = 4 at NPAD 16, 3 at NPAD 32. R0-b1 734.7 us (fi 787, job 9562).
- Check (job 9588): ALL PASS at 11 points; LM phase (stamps) 44.8 (b1) / 48.9 (b16) / 54.0 (b32) us = 6.9 / 6.4 / 5.8 TB/s
  (was ~99 / 112). Still v0 LM path: 129 (b64) / 173 (b128) / 304 (b256) us.

- P3b (9a965f8): NPAD 64 on the pipelined path with LMR 2 (dyn 198016 B, 113 regs; the 12 epilogue warps loop over 4 column
  chunks). A/B (job 9590, 9 rounds): R1-b64 -36.8 [-37.6, -34.4]; R0-b1 -51.8, R1-b16 -68.5 (same as P3). **Kept.**
  MK_KU 64 keeps the v0 LM path (test fail at bc685f7, fixed 9a965f8).
- e2e v2 (job 9589, tag mk-core-v2): R1-b64 missing. The job read the kernel source after the P3b edit with the old build.py
  (static_assert dyn smem). The other 27 points are v2 SASS. From now on e2e jobs run from a detached snapshot worktree (REPO=...).

## Next
- P3c: the pipelined LM head at NPAD >= 128. NPAD 128 / 256: X does not fit
  (256 / 512 KB); options: X k-chunk ring (restage per k-chunk, 8 chunks, shared by all row blocks of a pass over a row group), or
  output tiles of N = 64 tokens with the weights re-read from L2 (2 / 4 passes). Epilogue cost at b256 (logits 256 x 151936 fp32
  = 156 MB of writes) is a floor of ~24 us by itself.

## P3c (2026-10-23, before the build): LM head on the tile stream at NPAD 128 / 256 (SPEC-P3c.md)
- Phase isolated: LM head at NPAD >= 128. Still the v0 path: 173 (b128) / 304 (b256) us per step (job 9588 stamps); fi 59-70 (job 9564).
- Change: knob MK_LMTS (TSB buckets): the last layer's idle D finalizers write xb = bf16(x * fnorm) + ssf; the LM runs as
  ts_phase<PhLm, TT 128, KS 1> with A = embed, B = xb by TMA (map 8), the existing logits + argmax epilogue. Run-time fallback to
  the v0 path unless lm_tok is the identity (all q_len 1 points).
- Band [H]: b256 -150..-200, b128 -80..-100 us/step (per-CTA ingest ~6.2 / 4.1 MB at the job 9595 per-SM rate).
- Checks: MK_LMTS 0 = 0-line SASS diff; A/B with ab.py at R1-b128 / R1-b256 (MK_LMTS 0 / 1), accuracy band at both; stamps.
- Result (job 9701, fd67607, GPU-17dce9e8): check ALL PASS at R1-b64 / b128 / b256 (0 local bytes); A/B MK_LMTS 0 -> 1, 7 rounds:
  b128 2670.0 -> 2573.7 (-96.4 [-98.0, -95.9]), b256 4588.2 -> 4373.3 (-215.7 [-217.7, -212.8]) [M]. **Kept, default on.**
  (Job 9690 / 9699: void, `--wrap` ran under sh; also ab.py takes raw knob names: {"MK_LMTS": 1}, not {"lmts": 1}.)
