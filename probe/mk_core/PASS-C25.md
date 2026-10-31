# PASS-C25 (core-P25, R10): last-arriver merge of split pairs (MK_CMB 3), no comb barrier -- KILLED (job 15055)

Gap: the P20 comb phase (MK_CMB 1: grid barrier, combine_all_to_ao on all CTAs, arrive) costs 3.3 us/layer (NPAD 64 SI
points) to 6.0 us/layer (b192) at split plans (BD v21, job 15005): work 0.2-2.5, spread 1.8-2.6, lat ~1.0. That is
92-168 us/step; the b192-kv1024 loser is mostly this phase plus the residual.

Change (mk only, header untouched; SKV_TICKET cannot be used: #error with SKV_HALVES):
- After attend, every CTA fences its partial stores, then thread 0 adds 1 per S > 1 item record to tk[192 + tok * 8 + kvh].
  The add that returns S - 1 completes the pair: that CTA resets the ticket (all adds of the layer have landed; next use
  is behind grid barriers) and merges the pair's 2 q-heads into ao (skv::merge_th, one warp per (tok, head)) before its
  attend arrive. The O phase reads ao after the existing attend barrier, so the comb barrier and phase are gone.
- TK_SIZE 192 -> 2240 (u32), zero once at prepare (as before); the winner's reset keeps the invariant.
- q_len 1 builds only (static_assert).

[H] Saving = comb barrier lat + spread (~3 us/layer at b192, ~2.5 at NPAD 64) minus the merge tail on the last CTAs
(a few pairs per CTA, ~0.1-0.3 us). Expected: b192 -60..-90 us/step, NPAD 64 split points -40..-70.
Risk: the merge lands on the last-finishing CTAs (critical path); a CTA that wins many pairs (g8 group plans put several
splits on one CTA) adds its merge serially.

Checks: default SASS identical (md5 of cuobjdump -sass at NPAD 64 kv4K / 64 / 128 / 256 = HEAD c2d3b39); MK_CMB 3
compiles at all four (static smem 12304, regs 128). A/B in one job vs v21 default (MK_CMB 1) at the split points, with
accuracy vs the FP32 band (ab.py) where the KV fits.
Kill: no gain > 5 us/step at b192, or any accuracy failure.

## Result (job 15055, 7 rounds, MK_CMB 3 - v21 default, us/step)
b48-kv4K +16.8, b64-kv4K +65.1, b96 kv1K / 2K / 3K +117.2 / +131.2 / +153.0, b192 kv1K / 1792 +214.8 / +211.3, conv-L3 +47.3,
code-L3 +46.9. Accuracy PASS where graded. Slower everywhere: the merge moves onto the last-finishing CTAs (serial ticket adds
by thread 0, then the winner's merges) and costs more than the barrier + all-CTA combine it replaces. Code removed;
MK_DBG_WAIT (bounded spins, debug only, default SASS identical) kept.
