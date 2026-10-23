# regimebench build spec (for implementation subagents)

## Goal
Measure the current best Qwen3-0.6B dense megakernel design (hv2, ~383 us at decode b1 / kv 128 on B200) across
serving regimes, against the best CUDA-graph + PDL multi-kernel baseline per regime and vLLM, with a speed-of-light
(SOL) number per point. This is a measurement study: do NOT tune any arm per regime beyond what is written here.
Every claim in docs/reports is either measured (job id + log path) or labelled hypothesis.

## Ground rules (all agents)
- Repo root: `/raid/garv901/regime-megakernel`. Python: `/raid/garv901/vibe-megakernel/.venv/bin/python`
  (torch 2.13 cu130, flashinfer 0.7.0.post1, vllm 0.31.0, cutlass-dsl 4.7.1, cudnn 9.20, matplotlib, pytest).
  Run tools from the repo root with `PYTHONPATH=/raid/garv901/regime-megakernel`.
- **CPU only.** Do not run anything on a GPU, do not call `sbatch`/`srun`, do not set `CUDA_VISIBLE_DEVICES` to a GPU.
  NVRTC / nvcc compilation is allowed (it needs no device). The main agent runs every GPU job and sends you results.
- **No git commands that change state** (no commit/branch/stash). The main agent commits.
- Edit only files you own (ownership table below). The frozen interfaces are
  `regimebench/{geometry.py, workload.py, fixture.py, arms/base.py}`: do not edit them; if one must change, say so in
  your final report with the exact change.
- `regimebench/_upstream/` holds read-only copies of the vibe-megakernel sources this repo is ported from (commit in
  `_upstream/UPSTREAM_COMMIT`). Read them; never import from `_upstream` or from `megabench`.
- Write CPU unit tests in `tests/` (pytest) for your pieces; they must pass with no GPU
  (`pytest -q tests/<yours>`; skip GPU-only paths with `pytest.mark.skipif(not torch.cuda.is_available())`).
- For GPU validation, write a small script `scripts/gpu_check_<area>.py` that the main agent can run; it must print a
  short PASS/FAIL table and exit non-zero on failure.
- Final report (your last message): files changed, what is done, what is untested, exact commands for the main agent
  to run on GPU, open questions. Keep it short and factual.

## Frozen interfaces (read these files first)
- `regimebench/geometry.py`: `QWEN3_06B` (H 1024, L 28, QH 16, KVH 8, D 128, I 3072, V 151936, rope_theta 1e6, eps 1e-6,
  tied embeddings).
- `regimebench/workload.py`: `Seq(kv_len, q_len)`, `Workload(id, regime, seqs, axis, check)`, `regime_grid()` (45 points,
  R0..R8), `select()`, `KV_BYTES_CAP`. Token j of sequence i is at position kv_len_i + j, attends 0..kv_len_i+j.
- `regimebench/fixture.py`: `make_inputs(w, seed, device)`: weights (MegaBench names, bf16), `tokens` [total_q],
  packed `kcache`/`vcache` [L, cache_rows, KVH, D] with per-sequence regions `cu_cache`, plus CPU `cu_q`, `kv_len`, `q_len`.
- `regimebench/arms/base.py`: `Arm` (`variants`, `supports`, `prepare`) and `Step` (`launch`, `outputs`, `close`,
  `config`); output contract `logits` fp32 [n_seq,V] (last new token per seq), `next_token` int64 [n_seq],
  `k_new`/`v_new` bf16 [total_q, L, KVH, D]. Timed `launch` writes K/V only into the arm's own cache; `outputs()` is
  untimed and may gather from the cache / cast logits.

## Model math (reference semantics; same as `_upstream` qwen3_torch / megabench tasks/common.py)
Per token: x = embed[tok]; per layer: xn = rms(x, ln1); q = xn Wq^T, k = xn Wk^T, v = xn Wv^T (per head D=128);
q = rms(q, qn), k = rms(k, kn) per head; RoPE (HF rotate-half, theta 1e6, position = kv_len + j);
write k, v to cache; o = softmax(q k^T / sqrt(D)) v over visible positions (GQA groups 2, causal);
x += o Wo^T; xn = rms(x, ln2); x += (silu(xn Wg^T) * (xn Wu^T)) Wd^T. Final: logits = rms(x, fnorm) embed^T (fp32 out).
BF16 activations, FP32 reductions for norm/softmax, BF16 scores/probs (see `ref/common.py`).

## Ownership and tasks

### S1 core (owns: `regimebench/{sol.py, probes.py, run.py, report.py, __main__.py}`, `regimebench/ref/*`,
`regimebench/harness/*`, `tests/test_core_*.py`, `tests/test_sol.py`, `tests/test_ref.py`, `slurm/*`, `pyproject.toml`,
`.gitignore`, `README.md`, `docs/METHODOLOGY.md`, `scripts/gpu_check_core.py`)
1. `ref/reference.py`: `forward(w: Workload, inputs, dtype) -> dict` with the four outputs, for ragged workloads
   (per-sequence loop is fine; vectorise within a sequence; causal mask for q_len > 1). `dtype=torch.bfloat16` =
   serving semantics above; `dtype=torch.float32` = FP32 oracle. Fix the vendored `ref/qwen3_torch.py` imports (or
   drop it if `reference.py` replaces it). Must handle R7 (one 512-token chunk + decode seqs) and R4 T=8K
   (chunk the attention so memory stays < 20 GB).
2. `harness/`: adapt vendored `accuracy.py` (outputs = `OUTPUT_NAMES`; oracle band: an arm passes when its rel-L2 to
   FP32 is <= 2x the BF16 reference's own rel-L2 to FP32, plus next_token agreement rate recorded), `records.py`
   (SCHEMA `regimebench/v1`, runs dir = `<repo>/runs`), keep `timing.py` (fix imports only). Report rewritten (below).
3. `sol.py`: per Workload:
   - bytes = weights once (all linear weights + embed/LM head 151936x1024, bf16; count embed once since tied, but
     state it) + KV read (2 * L * KVD * 2 B * total_kv) + KV write (same per total_q) + token ids + logits out fp32
     (n_seq * V * 4) + norms. R0 must reproduce the upstream megabench floor 1,207,502,352 B / 150.9 us at 8 TB/s
     (`_upstream/sol.py`; if R0 differs by a few KB because of the definition, match upstream exactly for R0 and
     document).
   - flops = 2 * (linear params per layer * L) * total_q + attention 4 * L * QH * D * sum over tokens of attended
     positions + LM head 2 * V * H * n_seq.
   - `T_bw = bytes / BW`, `T_fl = flops / PEAK`, `T_sol = max`, bound = "memory" | "compute"; also report the
     ridge (tokens/step where they cross) per point.
   - Spec constants: BW 8.0 TB/s, BF16 dense tensor-core PEAK 2.25e15 FLOP/s (B200; cite NVIDIA DGX B200 datasheet
     72 PFLOPS FP8 sparse / 8 GPUs -> BF16 dense 2.25 PF; leave a TODO note that the main agent verifies), FP32
     FFMA peak 148 SMs x 128 lanes x 2 x clock (record clock).
   - "measured" variant: same formula with BW from `probes.hbm_bw()` (wrap `harness.timing.hbm_copy_bw`) and PEAK
     from `probes.gemm_peak()` (cuBLAS bf16 8192x8192x8192 `torch.mm`, best of 20, TFLOP/s). Probes run once per job.
4. `run.py` + `__main__.py`: `python -m regimebench run --arms fi-graph,hv2g,hv2-repeat,vllm --workloads R0,R1,R2-b1-kv8192
   --tag NAME [--seeds 7301,8917,42811] [--rounds 20] [--smoke] [--dry-run] [--variants arm=v1,v2;...]`
   and `python -m regimebench report RUN_DIR... --out results/<date>-b200/`.
   Per workload: fixture (timing seed = first seed) -> probes (once) -> for each arm/variant: `supports()` (record a
   `skip` record with reason) -> `prepare` -> accuracy (when `w.check` or regime R0; all seeds; proxy arms skipped)
   -> warmup -> interleaved timing over all prepared steps with `harness.timing.interleaved` (randomised order each
   round). Two measures per round: `steady` = mean of n back-to-back launches between events, n chosen so one sample
   is ~20 ms of work (min 3, max 50, from a one-launch estimate); `isolated` = one launch between syncs (record p50).
   Free each arm's Step (`close()`, `torch.cuda.empty_cache()`) before the next workload; drop the fixture KV for
   points with `check=False` only after all arms prepared. Catch exceptions per arm/variant into an `error` record
   with traceback; never abort the whole run. vLLM: call `arms/vllm_worker.py` as a subprocess per workload (CLI
   contract in S3) and merge its JSONL. Record per timing: arm, variant, workload dict, config, steady/isolated stats
   (harness.timing.summarize), clock log, SOL dict (spec + measured), tokens/s = total_q / steady.
   `--smoke`: rounds 3, seeds first only, R0 + one point per regime.
5. `report.py`: from records -> `REPORT.md` + `summary.csv` + PNG plots per regime (x = varied axis, y = time per
   step log scale; lines: hv2g, hv2-repeat (dashed, "proxy"), best fi-graph variant, vllm, SOL spec and SOL measured).
   Main table per regime: point | SOL us (bound) | hv2g | best baseline (variant) | vllm | hv2g/best | %SOL hv2g |
   %SOL best. Plus: winner variant/backends per point, skips with reasons, errors, accuracy table.
6. `slurm/env.sh` (based on `_upstream/sota_env.sh`: same /raid/garv901/.cache caches so FlashInfer cubins and vLLM
   caches are reused; PYTHONPATH = this repo; PY = the venv python) and `slurm/regime.sbatch` (partition main, 1 GPU,
   28 CPUs, 2 h, log `runs/slurm-%j.log`, args forwarded to `python -m regimebench run`).
7. `pyproject.toml`, `.gitignore` (runs/, __pycache__, *.png in runs), `README.md`, `docs/METHODOLOGY.md` (regimes,
   SOL model, timing method, fairness rules: own KV per sequence, logits fp32 cast untimed, proxy arm).
8. `scripts/gpu_check_core.py`: reference BF16 vs FP32 on R0 and one point of each regime with check=True;
   SOL table print for the grid (CPU-only part also as a test).

### S2 fi-graph (owns: `regimebench/arms/fi_graph.py`, `tests/test_fi_graph.py`, `scripts/gpu_check_fi.py`)
Port `_upstream/sota_graph.py` (FlashInfer + cuBLAS / FlashInfer GEMM, attention autotune, PDL via FlashInfer
`enable_pdl=`/`pdl=` kwargs, graph capture) from `Shape(batch, context)` to `Workload`:
- Paged KV cache, page 16, HND (as upstream), each sequence has its own pages; copy the fixture prefix in `prepare`.
  Capacity = kv_len + q_len per sequence.
- Token-major activations [total_q, H]. RoPE with per-token positions (`flashinfer.apply_rope_pos_ids` or the
  cos/sin-cache variant with `positions`); KV append via `append_paged_kv_cache` with per-token batch indices and
  positions (`get_batch_indices_positions`).
- Attention selection per workload, autotuned once with PDL on, winner recorded in config:
  - pure decode: upstream candidates (`BatchDecodeWithPagedKVCacheWrapper` fa2 / trtllm-gen / auto / cute-dsl,
    tensor cores on/off; direct `flashinfer.decode.trtllm_batch_decode_with_kv_cache`).
  - any q_len > 1 (R4-R7): `BatchPrefillWithPagedKVCacheWrapper` causal with backends fa2 / fa3 (if it runs on
    sm100) / trtllm-gen (`flashinfer.prefill.trtllm_batch_context_with_kv_cache`) / cudnn (`flashinfer.cudnn` batch
    prefill, if available); R7 also a split plan: prefill wrapper for the q_len>1 sequence + decode wrapper for the
    rest. Candidates that raise or give wrong results (compare to a torch attention on the same inputs) are dropped
    and recorded.
- GEMMs with M = total_q (LM head M = n_seq after gathering last-token rows): per GEMM group autotune cuBLAS
  `torch.mm` vs `flashinfer.mm_bf16` backends (tgv, cudnn, tinygemm, cute-dsl), as upstream; fused QKV and fused
  gate/up weights are allowed (weight relayout in prepare is fine for baselines).
- Variants: `graph-pdl-best` (headline: graph + PDL + per-group best GEMM), `graph-pdl-cublas`, `graph-nopdl-best`,
  `graph-nopdl-cublas`; `eager-pdl-best` only for R0 (`supports` returns a reason otherwise). Record PDL coverage
  (`launches_with_pdl` / `launches_total`) as upstream.
- `launch()` = replay the captured graph (or the eager step). Logits written as the GEMM produces them; `outputs()`
  casts to fp32 and gathers k_new/v_new from the paged cache.
- Large prefill (R4 T 8K, R5 16x2K = 32K tokens): must fit in memory; workspace buffers sized from the workload.
- `scripts/gpu_check_fi.py`: for every check=True workload, all variants vs `regimebench.ref.reference.forward`
  (S1 is writing it in parallel; until it lands, compare to your own small torch reference) and print timings of
  one quick steady measurement per variant.

### S3 vllm (owns: `regimebench/arms/vllm_worker.py`, `tests/test_vllm_worker.py`, `scripts/gpu_check_vllm.py`)
Out-of-process vLLM 0.31 reference, ported from `_upstream/vllm_worker.py` (read `_upstream/ENV.md` §4 for PDL
notes and the cubin env var).
- CLI: `python -m regimebench.arms.vllm_worker --workload-json PATH --variant {pdl,nopdl} --out OUT.jsonl
  --run-id ID [--seed 7301] [--rounds N]`. Writes JSONL records in the S1 record shape (`record_type` "timing" with
  `arm="vllm"`, `variant`, `workload`, `config`, `steady` stats in ms per step; or "skip"/"error").
- Model: Qwen3-0.6B architecture from the local HF cache (`HF_HOME=/raid/hf`, offline; repo `Qwen/Qwen3-0.6B`) with
  `load_format="dummy"`, bf16, CUDA graphs on (default compilation config; record the cudagraph mode), prefix caching
  OFF, `max_num_batched_tokens >= total_q`, `max_num_seqs >= n_seq`, `max_model_len >= max kv + q + 64`,
  in-process engine core (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, as upstream variant mp0).
- Measure the time of the engine step that matches the workload, using step wall time from a direct
  `LLMEngine.step()` loop (record that it includes scheduler/host time):
  - pure decode (R0-R3, R8): add n_seq requests whose prompts have kv_len random tokens (`ignore_eos`, max_tokens
    large), run until all are decoding, then time K decode steps; kv grows by K during timing, so keep K <= 16 and
    record mean kv during timing.
  - prefill (R4, R5): time the single step that prefills all prompts (q_len tokens, kv 0); repeat with fresh
    requests per round (prefix caching off).
  - R6 (q tokens over kv 1K): vLLM has no plain multi-token decode without spec decoding; approximate as chunked
    prefill: prompt of kv_len + q tokens where the first kv_len are processed in an earlier step
    (`long_prefill_token_threshold` / `max_num_batched_tokens` so the last chunk is q tokens). If that cannot be
    made exact, emit a `skip` record with the reason.
  - R7: D decoding requests at kv ~1K, then add one 1536-token prompt processed so that the timed step has a
    512-token chunk at kv 1K (use the same chunking trick) plus the D decode tokens. If not exact, record the actual
    step composition (from scheduler output) in config.
  - Record the actual per-step scheduled token counts when available (scheduler output) to prove the step shape.
- Variants: `pdl` (default) and `nopdl` (monkeypatch as upstream).
- `scripts/gpu_check_vllm.py`: run R0, R1-b32, R2-kv8192, R4-T1024, R7-d8 once each and print step times.

### S4 megakernel (owns: `regimebench/kernels/hv2g/*`, `regimebench/arms/hv2g.py`, `regimebench/arms/hv2_repeat.py`,
`tests/test_hv2*.py`, `scripts/gpu_check_hv2g.py`)
Source: `regimebench/kernels/hv2/hv2_submission.py` (frozen hv2 = fx1 kernel + Host V2, sha256 4253a0db...; NVRTC
build, cooperative launch, 148 CTAs x 512 threads, macros baked in as #ifndef defaults). Never edit this file.
1. `arms/hv2_repeat.py` (do first): the unchanged hv2 binary as a timing proxy. `prepare` builds hv2 for the
   megabench case dict `{"params": {batch 1, context 128, layers 28, hidden 1024, q_heads 16, kv_heads 8,
   head_dim 128, intermediate 3072, vocab 151936}}`, feeds it the fixture weights plus its own kcache/vcache
   [28,128,8,128] (random) and token = tokens[0], preallocated outputs. One Step = N = total_q kernel launches (each
   one b1/kv128 decode). Variants: `graph` (N launches captured in one CUDA graph; see `_upstream/opus.py` for how a
   cooperative kernel is captured) and `direct`. `config["proxy"] = True`, `launches_per_step = N`; `outputs()`
   returns zeros of the right shapes (accuracy is skipped for proxy arms). `supports`: N <= 8192.
   Also `scripts/gpu_check_hv2g.py` must check that hv2-repeat with N=1 gives the same outputs as the hv2 `run()`.
2. `kernels/hv2g/`: a minimal generalization of hv2 that keeps its structure (phases, grid barriers, weight smem
   streaming, 148 x 512, fx1 macros). No regime-specific optimization: the point is to see how this design scales.
   - Stage A (runtime context and position): remove `CTX` as a compile-time size. Phase F loops over the
     sequence's positions in chunks of 128 (same 16 warps x 8 positions butterfly per chunk), online softmax
     (running max / sum / o) across chunks, V chunk staged in `sVc` (same 32 KB); the new token(s) handled as now.
     RoPE: cos/sin table [max_pos, 64] fp32 built in `prepare`, kernel reads row `position`. KV cache layout = the
     fixture's packed layout [L, cache_rows, KVH, D] with a per-sequence base row; the kernel writes the new K/V into
     the cache at the position (and still into its k/v outputs if convenient).
   - Stage B (many tokens / sequences): activations `[ntok][...]`; every GEMV keeps streaming each weight tile once
     per launch and applies it to all ntok tokens (inner token loop; stage x tiles in smem in token blocks as smem
     allows). Attention work items = (token, q-head) with per-token (seq base row, position) and causal visibility
     (positions <= position of that token; the earlier new tokens of the same sequence must already be in the cache
     -> write K/V for all tokens in the Q phase before the F barrier). LM head + argmax only for the last token of
     each sequence. Device descriptor arrays (int32): per token seq id / position / cache row; per sequence base row
     / last token index. Cap `MAXTOK` per launch (start 256, lower if scratch/smem demands); larger steps run as
     consecutive launches of <= MAXTOK tokens (prefill chunks in order: later chunks see earlier chunks' K/V in the
     cache), all inside one CUDA graph. Record `launches_per_step` in config.
   - Two builds: `MT1` (ntok == 1 compiled path, used whenever total_q == 1; must be as close as possible to hv2's
     code so R0 time stays within 3% of hv2) and `MTN` (general). Variant names: `auto` (headline: MT1 if
     total_q == 1 else MTN) and `mtn` (always MTN, to measure the general path's cost at R0).
   - Register/spill: print NVRTC ptxas info (`--ptxas-options=-v` / `-Xptxas -v`) for both builds and include it in
     your report. 0 spills required for MT1.
3. `arms/hv2g.py`: Arm wrapping the build: `prepare` copies the fixture KV into a cache of the packed layout (clone),
   builds the descriptors and RoPE table, captures launches in a CUDA graph (variant suffix `-direct` optional),
   `outputs()` gathers k_new/v_new from its cache, logits fp32 [n_seq, V].
4. `scripts/gpu_check_hv2g.py [--workloads ...]`: for hv2-repeat N=1 vs hv2 run() (bit-exact) and for hv2g on
   R0, R2-b1-kv1024, R2-b1-kv8192, R1-b8-kv128, R4-b1-T16, R4-b1-T256, R6-b1-kv1024-q4, R7-p512-d8, R8-b32-ragged:
   rel-L2 vs FP32 reference compared with the BF16 reference's own rel-L2 (pass = <= 2x), next_token match, plus a
   quick timing of hv2 vs hv2g MT1 at R0 (interleaved, 10 rounds x 20 launches).
   Use `regimebench.ref.reference.forward` (S1 writes it in parallel; agree on the signature
   `forward(w, inputs, dtype) -> {"logits","next_token","k_new","v_new"}`).
Work order: hv2_repeat -> stage A (compile-clean, MT1) -> stage B (MTN) -> arm -> check script. Report after each of
those if you finish early; the main agent will run GPU checks and send failures back to you.

## Record shape (S1 defines; others follow)
One JSON object per line: `schema, run_id, record_type ("timing"|"accuracy"|"skip"|"error"|"probe"|"env"),
timestamp_utc, slurm_job_id, arm, variant, workload (Workload.to_dict()), config, ...` plus type-specific fields:
timing: `steady` (summarize dict, ms per step), `isolated` (summarize dict), `n_launch_per_sample`, `sol`
(from sol.py; spec + measured), `tokens_per_s`; accuracy: `seed`, metrics per output, `verdict`; skip: `reason`;
error: `stage`, `traceback`.
