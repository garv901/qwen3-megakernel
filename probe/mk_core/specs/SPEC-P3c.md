# SPEC-P3c: LM head on the tile stream at NPAD >= 128 (TSB buckets)

## Why (PASS-03.md, job 9588 stamps; PASS-00 fi timeline job 9564)
- The LM head at NPAD 128 / 256 is still the v0 path (`gemm_phase<PhLm>`: fp32 x restaged per row block by LDG): 173 us (b128) and
  304 us (b256) per step. fi's LM head is 59-70 us.
- The tile stream (`ts_phase`) is already in the build at these buckets (TSB). With A = embed rows by TMA and B = bf16 rows by TMA
  (map 8, xb), the per-CTA ingest at b256 is ~2.1 MB of weights + 16 x 256 KB of B (L2 hits) ~ 6.2 MB, ~100-115 us at the
  per-SM rate of job 9595. At b128 it is ~4.1 MB, ~70 us. Band [H]: b256 -150..-200, b128 -80..-100 us/step.

## Scope
- Files: `regimebench/kernels/mk/mk_step.cu`, `mk_common.cuh`, `build.py` (knob default only), `NOTES.md`,
  `regimebench/arms/mk.py`, `tests/test_mk.py`.
- New knob `MK_LMTS` (default 0). Active only when `TSB` (so NPAD >= 128 with the defaults): `constexpr bool LMTS = TSB && MK_LMTS;`.
  Off = 0-line SASS diff vs HEAD at all 5 buckets (default knobs) and at NPAD 256 with MK_TS 256 / MK_TSB 0. Test it.
- Do not touch `kernels/common/splitkv/`. Do not change shared files (base.py, report.py, run.py, workload.py, harness/).

## Data
- `ssf`: fp32 sums of squares of x after the last layer, one per token. Do NOT add a kernel argument: allocate `ss2` in the arm as
  `[2 * MAXTOK]` when LMTS can be on (just always allocate 2 * MAXTOK) and use `ss2 + MAXTOK` as ssf.
- Zero ssf in the QKV `extra()` of the LAST layer only (`l == NLAY - 1`), next to the existing `zero_f32(ss2, ntok)`.
- `zero_f32(p, total)` skips a trailing partial float4. For ssf pass `(ntok + 3) & ~3` (MAXTOK-sized buffer, MAXTOK % 4 == 0).
  Leave the existing ss2 zeroing unchanged (MK_LMTS 0 must stay a 0-line SASS diff); note the ntok % 4 issue of ss2 in NOTES.md.

## Run-time condition
- The tile-stream LM reads B row s = token s from xb. That is valid only when `n_lm == ntok` and `lm_tok[s] == s` for every s < n_lm
  (true for every q_len 1 workload, i.e. all mk points that reach NPAD >= 128).
- In the prologue, after s_lmtok is filled: `lm_ident = __syncthreads_and(i >= n_lm || s_lmtok[i] == i)` over all i < MAXTOK
  (each thread checks its i's; MAXTOK <= NT? if not, loop and AND locally first), and `n_lm == ntok`. Keep it in one register / a
  __shared__ int. Block-uniform and grid-uniform (same inputs in all CTAs).
- `lm_ident == 0`: run the existing v0 LM path exactly as today (prefetch_next<PhLm> in the window, gemm_phase<PhLm>).

## Last layer D phase (l == NLAY - 1), LMTS and lm_ident only
- Today the last layer skips the D tickets and the idle finalize. With LMTS && lm_ident: do the D tickets as for other layers, and
  run `idle_finalize` with lnw = `fnorm`, out = `xb`, and the sums of squares into `ssf`.
- Generalise `idle_finalize<IS_O, NTILE>` with a third template bool `SS` (default `IS_O`): SS adds sum x^2 per token into the
  `ss` pointer. Call the last-layer finalize with `<false, NTILE, true>` and `ss = ss2 + MAXTOK`.
- Tickets are reset by the finalizer as today; their next use is in the next step, behind grid barriers. Same rule as now.

## LM phase (LMTS && lm_ident)
- Barrier window after the last D (`MK_PF_ARRIVE(... else lm_first(g))`): call `ts_first<PhLm, 128, 1>(g, 0)` instead of `lm_first(g)`.
- After the grid barrier:
  1. All threads: `s_ss[t] = __ldcg(ssf + t)` for t < n_lm (raw sums; the existing LM `epi` computes rsqrt(s_ss / HID + EPS) itself),
     then `__syncthreads()`. s_ss is zero for t >= n_lm already (prologue + GU re-zero). Check that nothing else writes s_ss in
     between (the GU setup writes r2 into s_ss and zeroes it after the GU phase: fine).
  2. `ts_phase<PhLm, 128, 1, 8, mk_null(6)>(g, 0, n_lm, epi)` with the EXISTING LM `epi` lambda (logits store + per-(quadrant, slot)
     argmax). Do not use the `get` lambda on this path.
- `PhLm::src` already gives map 7 (embed), row = rb * 128. `TsC<PhLm, 128, 1>`: NRB 1187, NTT = NPAD / 128, SPT 8, NTILE = 1187 * NTT.
  `ts_geom` uses 64-bit products: keep that (NTILE * 148 fits in int too, but keep the code shared).
- Argmax rule. The epi keeps the strict `>` update because row blocks ascend per writer. Verify and state in NOTES.md:
  (a) a CTA's tiles t0 .. t0 + nt - 1 visit rb in ascending order (t = rb * NTT + tt, KS 1);
  (b) a given slot s maps to one (tt, chunk c) and so to one warp group per quadrant in the non-PAIR epilogue branch
      (c in {g, g + 3, g + 6, ..}); so each (quadrant, slot) still has exactly one writer warp.
- ts_phase's non-PAIR branch reads chunk c and c + 3 ("two"), and stops at `cb >= ntok`: pass `n_lm` as ntok.
- The TS ring for TT 128 (ts_ringn(128)) and TMEM 2 x 128 columns are already sized by the build. No new smem.
- The xb map (map 8) has box {64, 128} at NPAD 128 / 256 (check `build.py ts_tt`). B rows >= n_lm come from xb rows that hold stale
  values of earlier steps or zeros: the epi ignores slots >= n_lm, and the MMA columns are independent, so that is fine. State it.
- After the phase: unchanged (dealloc, stamps, argmax reduce over s_bk / s_bi).

## Fences and ordering (state each in NOTES.md)
- xb (fnorm rows) is written by idle CTAs with generic stores + fence.proxy.async.global, then the grid barrier; the ts producer
  does fence.proxy.async.global before its first B box (existing). Same as the QKV path.
- ssf is written with atomicAdd by idle CTAs before the grid barrier and read with __ldcg after it.
- The window A prefetch (`ts_first`) reinitialises the TS mbarriers. The previous ts phase (last D) has drained (its join
  __syncthreads precedes the window). Same as QKV after D today.

## Build and checks (CPU only; I submit the GPU jobs)
- `PYTHONPATH=$PWD /raid/garv901/vibe-megakernel/.venv/bin/python -m pytest -q tests` passes.
- New tests in tests/test_mk.py: (1) MK_LMTS 0 = 0-line SASS diff vs HEAD (default knobs, all buckets; and NPAD 256 with MK_TS 256
  MK_TSB 0); (2) MK_LMTS 1 builds at NPAD 128 and 256 with default knobs; report regs and local bytes (spills must not grow vs
  MK_LMTS 0 at the same bucket; report the numbers either way); (3) MK_LMTS 1 at NPAD 16 / 32 / 64 = same SASS as MK_LMTS 0.
- SASS counts at NPAD 256, MK_LMTS 1 vs 0: UTCHMMA / UTMALDG / LDTM call-site deltas; report them.
- Arm: `MkArm(..., lmts=0)` option -> knob `MK_LMTS`, like `tsb`. `scripts/gpu_check_mk.py --knobs` and `probe/mk_core/debug/ab.py`
  already pass knobs. Leave the build.py default at 0.
- NOTES.md: section "LM head on the tile stream (P3c)" with the design, the arguments above and the build table.
- Do not commit. Report: diff summary, test output, build table, SASS counts, anything you deviated from and why.
