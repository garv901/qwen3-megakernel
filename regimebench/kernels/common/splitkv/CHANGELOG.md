# splitkv CHANGELOG (A1 interface and versions)

## Interface v1 (agreed with MK-core 2026-10-23, before splitkv-v1; changes vs the draft in `kernels/common/README.md`)
- q input: **bf16** `[total_q, QH=16, 128]`, after q-norm + RoPE (MK-core writes it).
- Combine output (O-proj input): **bf16** `[total_q, QH*128 = 2048]` row-major (K-major; mk stages it as the tcgen05
  B operand).
- `skv_combine` runs at the start of O-proj; the barrier before it is MK-core's (default). A ticket-last-CTA variant
  is an open option, to be priced on data.
- New hook `skv_prefetch(layer, cta, nct)`: called by MK-core inside the QKV phase after its weight issue (no-op in the
  placeholder). L2 prefetch only until MK-core states an smem budget.

- 2026-10-23 (MK-core answer): O-proj is split-K by q-head (unit = 128 rows x one head; CTA b owns unit
  floor(b*128/148) or none; 8 CTAs read each head). Combine = `skv::combine_head<NPAD,NT>(p, layer, head, xs, ntok)`:
  all 512 threads, one q-head, tokens 0..ntok-1, written as bf16 into MK-core's SW128 K-major B slot (element (t,k) ->
  `xs + (k/64)*NPAD*128 + t*128 + ((((k%64)/8) ^ (t&7)) << 4)`, k%8 inside the chunk; rows >= ntok zero), called after
  MK-core's O-proj barrier while its Wo TMA is in flight. Scratch smem: 0 B. Fallback kept: global `ao` bf16 [total_q, 2048].
- 2026-10-23: L2 prefetch of KV measured net <= 0 (P1, job 9567); `skv_prefetch` stays a no-op hook.
- 2026-10-23 (A1 v1 code, P2): `skv::attend<NT>(p, layer, smem, ring_phase)` uses `attend_smem_bytes()` (82,496 B) of
  dynamic smem. Between calls the caller may use bytes [0, kBarOff = 82,432) (TMA ring + merge buffer: GEMM stages, the
  combine_head `xs` slot), but must keep the last 64 B (the TMA mbarriers, initialised once, phase carried in `ring_phase`)
  unchanged. Start every CTA with `ring_phase = 0` and pass the same variable to every layer. (Re-initialising the mbarriers
  on every call hung the TMA ring at kv >= 16K: jobs 9581, 9598, 9599.)
  Partials / q / ao are read with ld.global.cg. The LDG engine's KV loads are coherent (`SKV_KV_NC` default 0; P2 job 9601
  measured nc = coherent within 0.05 us).
- 2026-10-23: interface v2 (MK-core SPEC-A1v2: A1 does q/k-norm + RoPE + the KV write from fp32 qkv sums; mk selects it with
  `-DSKV_IFACE=2`, the header defines `SKV_IFACE_MAX`) is accepted. It is not implemented yet.
- 2026-10-23 (P2b / P2c, jobs 9610 / 9611): `Params` gains `items_stride`; `items` is now `int32 [n_cta][stride][8]` 32 B records
  (`item_off` / `pair_info` are no longer read by attend; combine still uses `pair_info`). TMA ring geometry is a macro
  (`SKV_STAGE_ROWS` 32/64/128, `SKV_STAGES` 2..8); default 128 x 2 (128 KB ring): **attend_smem_bytes() = 148,032 B,
  kBarOff = 147,968 B** (v1 was 82,496 / 82,432). `kNumMaps = log2(rows) + 1` tensor maps per K / V (8 at 128 rows: the host
  encodes 16 maps). The mbarrier tail is `kBarBytes = max(64, 16 * stages)` bytes (full[s] at +8 s, empty[s] at +8 (stages + s));
  the caller keeps those bytes unchanged. Macros: `SKV_TMA_RELEASE` (default 1: per-slot empty mbarriers, no CTA barrier per
  stage), `SKV_ITEM_PREFETCH` (default 0, no gain), `SKV_PROF_NOCOMP` (profiling only). Planner: one item per CTA when
  n_pairs <= n_cta and balance <= 1.10 (b1); `ldg_max_bytes` default 512 KB (LDG up to b1 kv16K).

## Versions
(none yet)
