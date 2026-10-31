# PASS-20: combine off the O critical path (MK_CMB) - R10, plans with S > 1

Owner mk-core-r10. Status: **KEPT** (jobs 14986 / 14987 / 14988; LEDGER P20), default on at NPAD >= 64, tag mk-core-v20.

## Data [M]
- O-phase work (stamps, us/layer) at the same point, v18 (s_max 1) vs v19 (skvgrp: group plans, s_max 7-16):
  conv-L3 2.77 -> 8.48 (job 14897 -> 14975), conv-L4 5.44 -> 14.81. Same-NPAD no-split points in v19: b128-kv1K 6.35, b256 8.88
  vs split b96-kv2K 13.82, b192-kv1K 20.83. So the combine adds 4.3-12 us/layer (120-335 us/step); the v19 gaps to fi are
  55-603 us/step at these points.
- Cause (code): the O phase has 128 units (rb 0..7) x (head 0..15), one per CTA; each unit runs skv::combine_head of its head for
  all NPAD tokens before its MMA: every head is combined 8 times, serially inside one CTA (16 warps, NPAD / 16 rows each).

## Change
MK_CMB 1 (mk_step.cu, after attend's arrive, only when skv_comb): wait for attend, skv::combine_all_to_ao on every CTA (warp per
(token, q-head) row with S > 1, grid-strided over 148 x 16 warps), arrive; O stages B from ao (the S == 1 path). Cost: one grid
barrier (~2 us) + 1/1184 of the rows per warp. Uses the existing splitkv entry point (no interface change).

## Band [H]
-2..-10 us/layer at the split points (O back to its no-split cost + ~2-3 us). 0 at no-split points (skv_comb 0: no extra barrier).

## Gate
Same-job A/B, 7 rounds; keep if every split point gains and the no-split control is within noise. Then plan auto model (mk-attn):
the C cost it assumes falls, so more points may pick the group plan (tell mk-attn).
