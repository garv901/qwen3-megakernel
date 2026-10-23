# PASS-05: tile-stream engine for the L buckets (MK-core, P5 of PASS-PLAN.md)

Design from PASS-04 (jobs 9592, 9593, 9595). Buckets NPAD 128 / 256 only (knob MK_TS); S buckets unchanged (per-bucket designs).

## P5a (SPEC-P5a.md): engine + xb + QKV
- xb = bf16(x * ln1[l+1]) written by the last D split per row block (ticket) and by the prologue; QKV on ts with B by TMA from xb.
- Band [H]: QKV at b256 15.2 -> ~8 us/layer, D finalize +~1 -> -150..-200 us/step at b256; b128 -40..-80.
- Checks: 0-line SASS diff off; gpu_check_mk at b128 / b256 points; A/B R1-b128, R1-b256 (R1-b64 control: same SASS).

## P5b (next): xb2 = bf16(x * ln2) by O finalize; GU on ts (KS 1 where possible, g/u row pairs in one tile -> act bf16 in the
epilogue); D on ts with B = act by TMA; D finalize keeps writing xb. Band [H]: b256 -600..-900 us/step.

## P5a results [M]
- A/B (job 9602, 9 rounds), MK_TS off -> on at NPAD >= 128: R1-b256 **-210.8** [-213.4, -205.9]; R1-b128 **+54.2** [+52.4, +55.1];
  R1-b64 +4.5 (same SASS: noise of the placement). Accuracy unchanged (0.47-0.48).
- Stamps (job 9603), us/layer work + spread, off -> on: b128 QKV 6.18 + 1.28 -> 3.87 + 0.51; attn 50.3 + 1.5 -> 51.5 + 1.3;
  D 6.72 + 4.61 -> 7.94 + 7.04. b256 QKV 12.6 + 2.6 -> 7.1 + 0.3; D 16.7 + 14.3 -> 18.1 + 17.9.
  Reading: the QKV gain is as the bench; the **D finalize** (the last CTA of each of 8 row blocks converts 128 columns x ntok
  tokens after its own D units) adds 3.7 (b128) / 5.0 (b256) us/layer on the critical path; dyn smem 132 -> 198 KB at b128 costs
  attention ~+1 [H].
- **Kept at NPAD 256 only**: MK_TS = minimum NPAD (0 off, 1 = 128, default 256). b128 re-tested after P5b, where D on the tile
  stream spreads the finalize over 16 (rb, tt) tiles of 128 tokens.
