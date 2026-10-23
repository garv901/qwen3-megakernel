# SOTA decode comparison

Compares the Qwen3-0.6B CUDA decode megakernel (one cooperative launch per step)
against production decode stacks (FlashInfer + cuBLAS in a CUDA graph,
torch.compile max-autotune, vLLM end to end), plus a PDL / launch-mechanism
ablation. First deliverable: batch 1, KV prefix 128 (decode at position 128,
129 attended positions). Code is shape-generic (`shapes.Shape`).

Environment setup (venv, caches, FlashInfer/vLLM builds): see `ENV.md`. GPU jobs
source `slurm/env.sh` (all caches under `/raid/garv901/.cache`); request
`--cpus-per-task=28`, since `--gres=gpu:1` alone gets 1 CPU here.

## Quick start

    sbatch megabench/sota/slurm/b1_s128.sbatch --smoke --tag smoke   # ~10 min
    sbatch megabench/sota/slurm/b1_s128.sbatch --flush-l2 --tag b1s128   # ~35 min
    # or split to fit short Slurm windows (in-process arms must share a job):
    sbatch --partition=priority --time=00:30:00 megabench/sota/slurm/b1_s128.sbatch \
        --arms megakernel,sota-graph,torch-compile --flush-l2 --tag inproc
    sbatch --time=00:20:00 megabench/sota/slurm/b1_s128.sbatch --arms vllm --tag vllm
    python -m megabench.sota report <run_dir> [<run_dir> ...] --markdown out.md
    python -m megabench.sota.arms.megakernel --placement-probe   # scratch placement
    python -m megabench.sota.arms.<megakernel|sota_graph|torch_compile> --selftest
    python -m megabench.sota run --dry-run   # CPU: print the planned matrix
    python -m megabench.sota env             # environment JSON

Results land in `megabench/sota/runs/<UTC>-<tag>/records.jsonl` (exclusive-create,
one JSON record per line; `env`, `hbm_bw`, `shape`, `skip`, `error`, `correctness`,
`timing`, `profile`, plus merged vLLM worker records). Chrome traces go in `traces/`.

## Arms and variants

- `megakernel`: `direct` (headline; one cooperative launch on the stream), plus PDL
  and graph variants. The PDL variants use direct stream launches because PDL edges
  do not cross separate CUDA-graph replays.
- `sota-graph`: FlashInfer + cuBLAS. PDL ablation matrix: eager vs graph, PDL vs
  no-PDL, cuBLAS vs FlashInfer GEMM (`eager-nopdl` ... `graph-figemm-pdl`).
- `torch-compile`: `torch.compile(mode="max-autotune")` of `qwen3_torch.decode_step`.
- `vllm`: out of process (`vllm_worker.py`), engine ITL and per-step slope over
  generated lengths `--vllm-n`; variants `mp0`, `mp1`, `mp0-nopdl`.

## Timing methodology

- **steady (headline)**: chained launches (each step feeds its argmax token back),
  mean over `--steady-steps` per sample; this is the throughput-style decode step.
- **isolated**: one launch on an idle GPU between synchronizations (includes launch
  latency); **flushed** (`--flush-l2`) zeroes 2x L2 first, so weights come from HBM.
- Arms are **interleaved**: each round shuffles arm order (`seed + round`) and takes
  a block per arm, so clock/thermal drift cancels. Statistics: median, bootstrap CI
  (2000 resamples), ratio-of-medians CI. Timing uses the first seed only (kernels are
  data-independent). Each mode is warmed separately: switching chained/unchained
  re-records torch.compile's cudagraph trees (>1 s on the first call).
- **Clocks**: NVML snapshots per block (SM/mem clock, temp, power, throttle reasons);
  rounds with reasons beyond idle/app-clocks are flagged. Measured D2D copy
  bandwidth is recorded for the `% meas BW` column; SOL uses 8 TB/s.
- The profile record (`kernel_timeline`) gives gap/busy/overlap per step. The
  profiler inflates gaps for eager and ctypes launches. Under PDL a kernel's
  interval includes its dependency wait, so busy/gap don't decompose cleanly; the
  ablation's ground truth is the steady-time delta.
- **Megakernel scratch placement**: all megakernel variants of one arm share one
  scratch buffer (sync flags and partials), so the launch/PDL ablation compares them at
  identical placement. `--placement-probe` measures how the step time varies with
  where that buffer lands.

## Accuracy policy

Every arm is compared against two references: the **BF16 MegaBench reference**
(`tasks/dense.py`) and an **FP32 eager reference** (`qwen3_torch.decode_step` in
float32, the megakernel's own contract). Grading uses the FP32 reference
(`accuracy.grade_vs_oracle`):

- FP32 arms (the megakernel): logits rel_l2 <= 1e-3, argmax token equal, K/V
  rel_l2 <= 1e-2.
- BF16 arms: **oracle band**. The arm's rel_l2 against FP32 must stay within
  `NOISE_FACTOR` (2x) of the BF16 reference's own distance from FP32, for logits,
  K and V (pass); 4x is warn; beyond that is fail. On the b1-s128 fixture that band is
  ~4.6% for logits. With random synthetic weights, independent BF16 stacks differ
  from each other about as much as BF16 differs from FP32, so a fixed small
  tolerance against the BF16 reference is not meaningful. (On CPU,
  `qwen3_torch` in bf16 reproduces the BF16 reference bitwise; on GPU, Inductor's
  fusion removes intermediate roundings.)

MegaBench's `_compare` verdict is stored for information only. The FP32
megakernel fails its 0.002 logits tolerance against BF16 by design. Accuracy runs
on every seed, except for arms in `--single-seed-arms` (default `torch-compile`:
max-autotune recompiles in ~4 min per variant and seed). Inputs are checked for
mutation.

## Extending

- New megakernel: `--megakernel NAME=PATH[:CASE_ID]` (repeatable); records use arm
  `NAME`. The headline remains the first megakernel's `direct` variant.
- New shape: `--batch 1,8 --context 128,1024` (the sweep is the cross product). Batch
  > 1 needs `shapes.make_shape_inputs`/`reference_outputs` support and arm support.
- New arm: add a `module:Class` entry to `backend.ARMS`.

## Caveats

- vLLM uses the real checkpoint, not synthetic weights, and measures the engine
  (scheduler, sampling) rather than a bare step.
- GPU clocks cannot be locked here; the node may be shared. Check the clock and
  throttle summary in the report header.

## TP=2 Gemma 3 baselines (`sota/tp/`)

Strong non-megakernel baselines for one Gemma 3 4B decode step with TP=2 on two
GPUs (`distributed-step-gemma3-4b-tp2`, batch 1, context 128; the 27B case also
works). Run from inside a 2-GPU allocation; the runner spawns the two NCCL ranks
itself. One best configuration per baseline, no ablation matrix.

- `tp-graph` (`graph-pdl` headline, `graph-nopdl`): the whole step per rank in ONE
  CUDA graph, collectives included. FlashInfer attention (trtllm-gen HND, fa2 fallback),
  `gemma_rmsnorm`/`gemma_fused_add_rmsnorm`, `gelu_tanh_and_mul`, KV append, cuBLAS vs
  FlashInfer GEMM (autotuned once), the selected all-reduce, global argmax.
- `tp-compile` (`reduce-overhead`, `max-autotune`): `gemma3_tp.Gemma3TP` under
  `torch.compile(fullgraph=True, dynamic=False)` with cudagraph trees. Skipped, with the
  reason, if Inductor skipped cudagraphs or the collectives are not inside the replayed graph.
  PDL is off (Inductor never sets it).
- `tp-eager`: `tasks.distributed.reference`; the speedup baseline.
- `tp-submission`: any MegaBench distributed submission (`--submission PATH`).
- `tp-vllm`: out of process (`vllm_tp_worker.py`), vLLM TP=2, FlashInfer attention,
  `FULL_DECODE_ONLY` graphs for batch 1, default optimization level, async scheduling,
  dummy weights from a locally generated Gemma3 text config. Timing only (no accuracy).

All-reduce selection (`ar_select.py`) times a 2,560-element BF16 all-reduce in a CUDA
graph (100 per replay) for NCCL, symm-mem one-shot/multimem and FlashInfer one-shot and
fused AR+residual+RMSNorm, caches the winner as JSON under `$MEGABENCH_SOTA_CACHE`, and
every in-process arm uses it. Only the winner is reported.

Timing is chained steady steps plus isolated steps, lockstep across ranks (barrier before
each sample, shared shuffled order), reported as the max over ranks. Accuracy is checked per rank
against the BF16 reference and an FP32 oracle (`Gemma3TP` in float32, torch attention, NCCL)
with the oracle band; `next_token` must match.

    sbatch megabench/sota/slurm/tp_b1_s128.sbatch ar-select                # AR selection, ~5 min
    sbatch megabench/sota/slurm/tp_b1_s128.sbatch run --smoke \
        --arms tp-graph,tp-compile,tp-eager --tag tp-smoke                 # in-process arms
    sbatch megabench/sota/slurm/tp_b1_s128.sbatch run --smoke --arms tp-vllm --tag tp-vllm
    python -m megabench.sota.tp report <inproc_run_dir> <vllm_run_dir> --markdown out.md
    python -m megabench.sota.tp run --dry-run    # CPU: print the plan

`--smoke` runs one seed, 3 rounds, and only `tp-compile:reduce-overhead` (max-autotune
recompiles take minutes; request it with `--variants 'tp-compile:max-autotune'` and a longer
window). Full run: drop `--smoke`, use `--time=01:00:00`. In the full run, `tp-vllm` is
launched by the same command after the ranks exit; for short windows keep it a separate job.
