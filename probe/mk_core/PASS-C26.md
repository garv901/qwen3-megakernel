# PASS-C26: early KV issue (lead (a); joint with mk-attn PASS-27)

## Hypothesis [H]
- Attend's first KV stages are issued only after the pre-attend grid barrier. Their first-byte latency sits on every layer's critical path.
- Past-row K/V does not depend on this layer's QKV or RoPE. A CTA's ring bytes and attention mbarriers are idle from the end of its own QKV tiles until attend.
- In that window lie the QKV barrier, the RoPE work and the pre-attend barrier: about 2.5-3.5 us/layer at NPAD 128/256 (15005).
- Issuing the first stages there hides the first-byte latency.
- mk-attn's silo ceiling is a lat_med of 0.93-1.09 us/layer, so the gain is at most ~28 us/step.

## Change (default off, SASS identical when off: verified with cc.py def at NPAD 64 kv4K / 64 / 128 / 256)
- mk side, knob MK_KVE (mk_step.cu):
  - After the QKV phase, before the QKV grid barrier, call `skv::attend_pre<NT>(SP, l, attn smem, ring_phase)`.
  - On the same thread, ring_phase stays intact from attend_pre to attend.
- splitkv side, knob SKV_EARLY (mk-attn, PASS-27):
  - Issues only stages whose rows all precede the new row.
  - Leaves the mbarrier init to attend: it is a no-op on layer 0.
  - Keeps the in-flight count in ring_phase bits 24..27.
- Phase 2 uses no shared memory (checked). The weight ring is drained at the end of the CTA's QKV tiles. O's prefetch is issued after attend, as today.

## Test
- Accuracy: the A/B grades every variant against the FP32 band.
- One job, A/B MK_KVE=1 + SKV_EARLY=1 vs default, at b96-kv1K, b128-kv2K, b128-kv2816, b192-kv1024, b192-kv1792, b256-kv1408 and b64-kv4K (SI).
- If the delta is under 5 us/step: 3 placements x 2 GPUs.
- Kill rule: median delta >= 0 at the split points, or any hang or accuracy failure.
