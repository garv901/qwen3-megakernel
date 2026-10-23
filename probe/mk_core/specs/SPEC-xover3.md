# SPEC-xover3: tile-stream GEMM engine vs split-K at T = 32..256 (Pass P4, MK-core)

## Why
At T = 256 the split-K engine (xover2 a3, job 9566) takes QKV 10.9 / O 7.1 / GU 12.8 / D 8.3 us per phase; fi ~6 each. In mk
(V0.md) the L-bucket GEMM phases are ~2x that. Suspects [H]: (1) split-K red.add traffic (every output element gets K/128 fp32
atomics: GU T = 256 = 8 x 6.3 MB = 50 MB per phase); (2) X restaged per unit. This bench measures the alternative engine and
gives, per (shape, T), the best tile config. It is also the prototype of the mk P5 engine, so build it like `lm_phase` in
`regimebench/kernels/mk/mk_step.cu` (read that first: producer / MMA / epilogue warps, LM ring, 2 TMEM accumulators).

## Bench
New dir `probe/mk_core/xover3/`: `xover3.cu`, `run.sh`, `xover3.sbatch`, `NOTES.md`, `.gitignore` (bin/, logs/ outputs).
Start from a copy of `probe/mk_core/xover2/xover2.cu` (reuse its harness: persistent 148 x 512 cooperative grid, grid barrier,
phase loop of 112 phases with timing, CUDA-core reference, tensor maps, SW128 helpers). Do not edit xover2.

Problem per phase: Y[T][N] (+)= X[T][K] * W[N][K]^T; shapes QKV N 4096 K 1024; O N 1024 K 2048; GU N 6144 K 1024; D N 1024 K 3072.
X is bf16 [T][K] in global (row-major, K-major operand), W bf16 [N][K]. Y fp32 [T][N].

### Engine "ts" (tile stream), cta_group::1, runtime parameters TT (token tile) and KS (k splits)
- Tile = (rb, tt, ks): rows 128 rb .. +127, tokens TT tt .. +TT-1, k range ks * K/KS .. +K/KS. NRB = N/128, NTT = T/TT.
  Tiles enumerated t = (rb * NTT + tt) * KS + ks; CTA b owns tiles [b*NT/148, (b+1)*NT/148), NT = NRB * NTT * KS.
- A tile is K/(KS*128) k-steps. Per k-step: one ring slot = A chunk (128 rows x 128 k bf16, 32 KB, 2 TMA boxes 64 x 128 SW128) +
  B chunk (TT tokens x 128 k bf16, TT*256 B, TMA boxes of 64 k x TT rows SW128), one full mbarrier with expect_tx of both.
- Ring depth = max that fits (227 KB minus barriers): print it.
- Warp roles (as lm_phase): producer tid 0 (empty / full barriers; the first min(ring, steps) loads issued in the barrier window
  before the grid barrier wait, as xover2 a3 does); MMA issuer tid 32 (N = TT, M = 128, K 16 per MMA, 8 MMAs per k-step;
  accumulate flag off at the first k-step of a tile; commit to empty[s]; at the last k-step of a tile commit to accfull[b]);
  2 TMEM accumulators of TT columns (b = tile index & 1; 2 TT <= 512, allocate 512 columns).
- Epilogue warps 4..15 (TMEM quadrant = warp & 3; split the TT/16 column chunks over the 3 warps of each quadrant): KS == 1:
  plain `st.global` fp32 (each element written once); KS > 1: `red.global.add.f32`. Arrive on accempty[b].
- Y zeroed before each timed phase only where KS > 1 needs it; zeroing is NOT in the timed region (use xover2's approach;
  say how in NOTES.md).
- Valid configs: TT in {32, 64, 128, 256}, TT <= T; KS in {1, 2, 3, 4, 6, 8, 12, 16, 24}, (K / 128) % KS == 0.
  Skip configs with NT < 37 (fewer than 1 tile per 4 CTAs) except report them as skipped.
- Baseline arm in the same binary or job: xover2 a3 (build and run its binary in the same job, same T values) so both are on one GPU.

### Measurements
- T in {32, 64, 128, 256}; all 4 shapes; all valid (TT, KS). Per point: us per phase (median of 5 reps, xover2 method), tiles per
  CTA max / mean, ring depth, SM MHz.
- Correctness: every config at T = 64 and T = 256 vs the CUDA-core reference (xover2 tolerance); print PASS/FAIL per config.
- Output: a table per shape (rows TT x KS, columns T) + the best config per (shape, T) vs a3. Write it to NOTES.md after the job
  (I run the job; you only build).

## Checks (CPU only; do NOT run on a GPU: no srun, sbatch, or CUDA calls)
- `bash probe/mk_core/xover3/run.sh --build-only` builds; ptxas: 0 spill, regs, smem. SASS: UTCHMMA, UTCBAR, LDTM, UTMALDG,
  REDG / STG counts in the ts kernel; no FFMA / HMMA in it.
- `xover3.sbatch`: main partition, 1 GPU, `--time=00:30:00`, runs the a3 baseline and all ts configs; logs to
  `probe/mk_core/xover3/logs/slurm-%j.log`. Estimate the runtime from the config count (xover2 job 9566 took about 5 min for 8 arms).
- Report: files, build table, SASS counts, config count, the parity / first-use / overlap arguments (as in mk NOTES "LM head,
  pipelined"), and every deviation. Do not git commit.
