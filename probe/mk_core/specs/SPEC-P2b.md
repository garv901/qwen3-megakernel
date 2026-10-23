# SPEC-P2b: mk on splitkv-v2 (TMA-MMA maps, no_split plans without combine, R6 build)

## Why
- MK-attn tagged splitkv-v2 (1f027ac, jobs 9658 / 9661 / 9692); it is imported at f614b3a (`regimebench/kernels/common/splitkv/`,
  read CHANGELOG.md there: P4, P5, P6 sections). Do NOT edit anything under `kernels/common/splitkv/`.
- In mk today (job 9674 / stamps 9675) splitkv loses at kv128 because the O-phase combine costs 7.6 / 12.3 / 21.4 us/layer at
  b64 / b128 / b256. v2's planner gives S = 1 there (attend writes `ao`), so mk must skip combine and stage `ao` instead.
- R6 (q_len 2..16) can now run on splitkv with a separate build (`-DSKV_QMAX=16`): 4.6-5.1 us/layer standalone.

## Scope
Files: `regimebench/kernels/mk/mk_step.cu`, `mk_common.cuh`, `build.py`, `NOTES.md`, `regimebench/arms/mk.py`, `tests/test_mk.py`,
`scripts/gpu_check_mk.py` only if needed. Placeholder builds (`MK_ATTN` placeholder) must stay a 0-line SASS diff vs HEAD (f614b3a)
at all 5 buckets: test it.

## 1. K / V tensor maps (v2 default SKV_TMA_MMA=1)
- `build.py skv_map_specs` / `encode_skv_maps`: K and V maps now use `CU_TENSOR_MAP_SWIZZLE_128B` and box `{64, 1, rows}`
  (rows = SKV_STAGE_ROWS >> rs, as before; dims / strides unchanged). Copy the exact encoding from MK-attn's host code
  `/raid/garv901/regime-mk-attn/probe/mk_attn/a1bench.py` (read only; do not edit that tree) and cite it in a comment.
- The number of maps and the stage rows come from the header's macros: kNumMaps = log2(SKV_STAGE_ROWS) + 1 per K / V. The default
  build has STAGE_ROWS 128 (8 + 8 maps). The R6 build (SKV_QMAX 16) defaults to STAGE_ROWS 64 / STAGES 4 (7 + 7 maps). Make the
  Python side derive the count and rows per build (one function of the build's defines), and make the kernel's map region size
  follow skv's kNumMaps (check where mk sizes / copies the 16 maps: kernel args `skv_maps`, map_acquire loop).
- The attend scratch base must be 1024 B aligned: mk already places it at OFF_BARS - kBarOff with that alignment (static_assert
  exists); keep it, and check attend_smem_bytes / kBarOff are unchanged (148,032 / 147,968) for both builds.
- Do not set SKV_TMA_MMA or SKV_MERGE_CHUNK (header defaults).

## 2. O phase without combine (plan.s_max == 1)
- The arm plans with `plan(seqs)` (default no_split="auto"). Add a trailing kernel argument `int skv_comb` = 1 if `plan.s_max > 1`,
  else 0 (append after `skv_engine`; update MK_SKV_PARAMS / kernel signature and the arm's skv_args tuple).
- O phase (MK_SKV builds): if `skv_comb`, as today (combine_head into the B slot before gemm_phase). Else stage B from global `ao`
  bf16 [total_q, 2048] (attend wrote every pair's rows) with the same staging the placeholder build uses for its O phase (reuse that
  code: the placeholder O `get` reads ao). Block-uniform run-time branch. Fences: ao is written by attend with generic stores before
  the grid barrier and read with ld.global.cg / generic loads after it: state this in NOTES.md (no proxy fence needed for LDG staging;
  if the staging is by TMA, a fence.proxy.async.global after attend's ao stores is needed; say which applies).
- The arm must allocate `ao` for splitkv builds if it does not already (check) and pass it in Params as today.
- Report regs / local bytes of the splitkv builds at all 5 buckets before and after (spills must not grow by more than 64 B per
  bucket; report either way).

## 3. R6 build (q_len > 1 on splitkv)
- `attn="splitkv"` with any q_len > 1 (max q_len <= 16, sum q_len <= 256) builds with the extra define `-DSKV_QMAX=16` (part of the
  build key / cache name). q_len == 1 workloads keep the default build (never QMAX 16 for them: b8 kv1K is 18.6 vs 11.5 us).
- plan(seqs) takes (kv_len, q_len) pairs: pass the real q_len. mk already writes the q_len new K/V rows in phase 2, before attention,
  and q is [total_q, 16, 128] bf16: check both against CHANGELOG P5 (rows `row_base .. + kv_len + q_len - 1`, token t sees
  positions < kv_len + t + 1) and against mk's t_base / t_vis descriptors. Report anything that does not match instead of guessing.
- `supports()` / `resolve_attn`: allow attn="splitkv" for q_len <= 16. Leave `auto_attn` UNCHANGED (I set the auto rule after the A/B).
- With SKV_QMAX 16 the planner forces TMA and S may be > 1: combine stays (pair_info per token row, CHANGELOG P5).

## 4. Tests (CPU)
- Placeholder SASS identity vs f614b3a (all 5 buckets).
- splitkv builds compile at all 5 buckets with (ts, tsb) in ((0, 0), (128, 1)); R6 build (SKV_QMAX 16) compiles at NPAD 16 (and 32).
- `skv_map_specs` gives SW128 boxes {64, 1, rows}, 16 maps (default) / 14 maps (QMAX 16).
- skv_comb is 0 for the plan of R1-b256-kv128 and R1-b64-kv128, 1 for R0-b1-kv128 (CPU planner).
- `PYTHONPATH=$PWD /raid/garv901/vibe-megakernel/.venv/bin/python -m pytest -q tests` passes.

## Report
Diff summary, pytest tail, build table (regs / local per bucket, default and R6 builds), SASS counts that changed, deviations and why.
CPU only, no GPU, no Slurm, no commit.
