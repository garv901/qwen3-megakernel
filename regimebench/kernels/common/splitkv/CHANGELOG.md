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
  `-DSKV_IFACE=2`, the header defines `SKV_IFACE_MAX`) is accepted. Implemented in P3 (below).
- 2026-10-23 (P2b / P2c, jobs 9610 / 9611): `Params` gains `items_stride`; `items` is now `int32 [n_cta][stride][8]` 32 B records
  (`item_off` / `pair_info` are no longer read by attend; combine still uses `pair_info`). TMA ring geometry is a macro
  (`SKV_STAGE_ROWS` 32/64/128, `SKV_STAGES` 2..8); default 128 x 2 (128 KB ring): **attend_smem_bytes() = 148,032 B,
  kBarOff = 147,968 B** (v1 was 82,496 / 82,432). `kNumMaps = log2(rows) + 1` tensor maps per K / V (8 at 128 rows: the host
  encodes 16 maps). The mbarrier tail is `kBarBytes = max(64, 16 * stages)` bytes (full[s] at +8 s, empty[s] at +8 (stages + s));
  the caller keeps those bytes unchanged. Macros: `SKV_TMA_RELEASE` (default 1: per-slot empty mbarriers, no CTA barrier per
  stage), `SKV_ITEM_PREFETCH` (default 0, no gain), `SKV_PROF_NOCOMP` (profiling only). Planner: one item per CTA when
  n_pairs <= n_cta and balance <= 1.10 (b1); `ldg_max_bytes` default 512 KB (LDG up to b1 kv16K).
- 2026-10-23 (P3, CPU-verified only): interface v2 implemented behind `-DSKV_IFACE=2` (header: `SKV_IFACE_MAX 2`, `SKV_IFACE` default 1 = v1,
  unchanged). `Params` gains `qkv, rr1, qn, kn, rope` (v2 only). attend computes q (norm + RoPE, bf16-rounded, then kQScale) per item in every CTA; the
  item ending at `vis` (word 7) computes k_new / v_new, starts its row-slot-0 state from that single position and is the only writer of the cache rows;
  its stream stops at vis - 1 (LDG range and TMA producer). Records: word 5 = S | n_items << 8, word 7 = vis (`plan(..., iface=2)`).
  attend_smem_bytes() / kBarOff unchanged. Norm sum order: per lane fma chain over its 8 dims, then xor-butterfly 8, 4, 2, 1 over the half-warp.
- 2026-10-23 (P4, CPU-verified only, default unchanged): `-DSKV_TMA_MMA=1` adds an mma.sync (bf16 m16n8k16) consumer for the TMA engine (ldmatrix + movmatrix from the
  smem ring). **Host tensor-map format changes for such builds**: K and V maps use `CU_TENSOR_MAP_SWIZZLE_128B`, box `64 dims x 1 x rows` (rows = kStageRows >> rs as before),
  and each box is issued at dim coord 0 and 64 (4 TMA ops per box: K lo, K hi, V lo, V hi, each [rows][64] bf16). The attend smem base (TMA ring) must be 1024 B aligned.
  attend_smem_bytes() unchanged (148,032 at 128x2). `SKV_MERGE_CHUNK` (default 20) sets combine's partial float4 loads in flight; both-engine builds with SKV_TMA_MMA spill
  at 20 (ptxas heuristic, combine_head merge_th holds 80 registers), 18 gives 0 spills. TMA-only builds (`SKV_FORCE_ENGINE=1`): 122 regs, 0 spills. Both ifaces supported.
  Causal mask for q_len > 1 (R6) is a TODO stub in the mma path.
- 2026-10-23 (P5, CPU-verified only, default unchanged): `-DSKV_QMAX=2..16` (default 1) builds the R6 multi-token causal consumer: q_len 1..16 per sequence on a two-phase mma.sync consumer
  (TMA engine only, interface v1 only, `SKV_TMA_MMA` 1; defaults become `SKV_STAGE_ROWS` 64, `SKV_STAGES` 4: same ring bytes, attend_smem_bytes() / kBarOff unchanged at 148,032 / 147,968; all staging lives in the 16.9 KB merge buffer).
  `plan(seqs)` takes `(kv_len, q_len)`, q_len <= 16, sum q_len <= 256, seq < 4096: sequence s owns cache rows `row_base .. + kv_len + q_len - 1` (mk writes the q_len new rows BEFORE attention), token t sees
  positions p < kv_len + t + 1, items cut [0, kv_len + q_len), engine = TMA whenever some q_len > 1. Record `y` gains q_len - 1 in bits 24..27. Partials are token-major: slot(pair, split, t) = part_base + t * S + split;
  `pair_info` has one row per (token row, kvh) at (tok0 + t) * 8 + kvh = {part_base + t * S, S, tok0, vis_t}, so `combine_head` / `combine_all_to_ao` (which index pair_info by token row) are unchanged; `part_floats`
  grows with sum q_len. q is `[total_q, 16, 128]` bf16 as before (row of token tok0 + t). Output rows tok0 + t of `ao` (S == 1 pairs written by attend, S > 1 by combine). q_len == 1 sequences are valid inside an R6 build.

## Versions
(none yet)

## P6 (MK-attn): planner `no_split` + `SKV_TICKET` deferred merge (for MK-core: ao complete after attend)
- `plan(..., no_split="auto" | True | False)`: True = every (seq, kvh) pair is one item (S = 1, attend writes `ao` directly), pairs LPT-packed whole onto CTAs. "auto" (default) = True iff
  n_pairs >= n_cta and LPT max / mean CTA load <= 1.10 (b128-kv128, b256-kv128: S = 1 everywhere; b64-kv128, b32-kv1K, b1..b8: the previous cut, unchanged). `Plan.no_split`, `summary()["no_split"]`.
  Plans with S == 1 everywhere need no combine at all.
- `-DSKV_TICKET=1` (default 0; default builds are unchanged): attend merges S > 1 pairs itself, so `ao` is complete after attend + the caller's grid barrier; the caller does NOT call
  combine_head / merge_th / combine_all_to_ao (they are unchanged). All three consumers (LDG, mma TMA, R6 `SKV_QMAX > 1`), iface 1 and 2. attend_smem_bytes() / kBarOff unchanged.
- Ticket contract: `Params::tickets` = `unsigned*`, `tickets[pair]`, pair = seq * 8 + kvh, array size `Plan.n_pairs` (= 8 * n_seq <= 2048) u32, zeroed by the caller ONCE PER PLAN and never reset
  between layers or steps; zero it again whenever the plan (S of any pair) changes. No layer index. Every S > 1 item does `__threadfence(); red.release.gpu.add.u32 +1` after its partial store (no one
  waits), so every layer adds exactly S to its pair. After its last item every CTA re-reads its S > 1 records; one thread per item does `ld.acquire.gpu.u32`: the pair is complete iff
  `count % S == 0` (count > 0). Layers are separated by the caller's grid barrier, so the read at the end of layer l never sees adds of layer l + 1: it sees a count below the next multiple of S
  (incomplete) or exactly that multiple (complete). Every CTA that sees a pair complete merges it (256 threads per (pair, token row); two CTAs may merge the same pair and write identical
  bf16 values: benign). The issuer of the last add of a layer always sees all S (its read follows its own add; RMWs on one address are totally ordered), so every pair is merged by >= 1 CTA.
  The counter is u32: it wraps after 2^32 / S layer-steps; 2^32 % S != 0 unless S is a power of 2, so re-zero per prepare (fine in practice).
- Costs: one extra `__syncthreads` per attend call with SKV_TICKET (start of the deferred phase), one per S > 1 item in R6 (before the add); thread 0 stalls for the fence + red per S > 1 item.
- a1bench: `--ticket` (or `SKV_TICKET=1` in a build), `--plan-opt no_split=1`; ticket builds run mode 0 (no combine), tickets zeroed once per point, `--check` also verifies tickets[pair] == 28 * S.
- 2026-10-23 (P5 + P6, job 9692): R6 consumer (`-DSKV_QMAX=16`) verified on GPU: b1 kv1K q16 A 5.06 us/layer; use it only for
  R6 buckets (slower at q_len 1). Planner auto `no_split` limit LPT max/mean 1.10 -> 1.20 (`plan.LPT_MAX_IMB`): b32 kv1K and
  b64 / b128 / b256 kv128 now have S = 1 (ao written by attend, no combine). Callers must not call combine_head when
  `plan.s_max == 1`. `SKV_TICKET=1` is correct but slower than attend + combine (b1 kv1K 7.04 vs 6.88): do not use it.
- 2026-10-23 (P8, job 9708, tag splitkv-v3): `SKV_FORCE_ENGINE` default -1 -> 1 (TMA-only build; LDG compiled out). The TMA maps are
  needed at every plan (the planner's `engine` field is ignored by this build). `-DSKV_FORCE_ENGINE=0` (LDG-only) is faster at
  b1 kv <= 2K (0.1-0.2 us/layer) and b256 kv128 (1.4); `-1` restores the runtime choice.
