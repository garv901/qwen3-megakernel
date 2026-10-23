# PASS-05: tile-stream engine for the L buckets (MK-core, P5 of PASS-PLAN.md)

Design from PASS-04 (jobs 9592, 9593, 9595). Buckets NPAD 128 / 256 only (knob MK_TS); S buckets unchanged (per-bucket designs).

## P5a (SPEC-P5a.md): engine + xb + QKV
- xb = bf16(x * ln1[l+1]) written by the last D split per row block (ticket) and by the prologue; QKV on ts with B by TMA from xb.
- Band [H]: QKV at b256 15.2 -> ~8 us/layer, D finalize +~1 -> -150..-200 us/step at b256; b128 -40..-80.
- Checks: 0-line SASS diff off; gpu_check_mk at b128 / b256 points; A/B R1-b128, R1-b256 (R1-b64 control: same SASS).

## P5b (next): xb2 = bf16(x * ln2) by O finalize; GU on ts (KS 1 where possible, g/u row pairs in one tile -> act bf16 in the
epilogue); D on ts with B = act by TMA; D finalize keeps writing xb. Band [H]: b256 -600..-900 us/step.
