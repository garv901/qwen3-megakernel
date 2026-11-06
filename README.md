# regimebench: megakernels across serving regimes

This branch holds the code and the data of a measurement study on one NVIDIA B200 (148 SMs).
The study asks one question:

> How fast is one persistent megakernel for a full Qwen3-0.6B forward step in each serving regime?

We compare each megakernel with two baselines in each regime:

- **fi-graph**: the best multi-kernel step with CUDA graphs and PDL (FlashInfer and cuBLAS, tuned for each point).
- **vllm**: vLLM 0.31 with CUDA graphs, out of process.

Each point also has a speed-of-light (SOL) time from `regimebench/sol.py`.
The full rules are in [SPEC.md](SPEC.md) and [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## The regimes

One workload is one forward step over a set of sequences. Each sequence has a KV length and a query length.

| regime | step |
|---|---|
| R0 | decode, 1 sequence, KV 128 (the anchor point) |
| R1 | decode, batch 1 to 256, KV 128 |
| R2 | decode, batch 1, KV 128 to 32K |
| R3 | decode, batch x KV grid |
| R4 | prefill of 1 sequence, T = 16 to 8K |
| R5 | prefill of a batch |
| R6 | 2 to 16 new tokens on a KV of 1K |
| R7 | one 512-token prefill chunk and D decodes at KV 1K |
| R8 | ragged decode, batch 32, KV from 128 to 8K |
| R9 | decode, batch 8 or more, long KV (KV sweeps and equal total KV) |
| R10 | decode, 33 to 256 requests, KV 1K to 4K, and 3 real steps from the Azure 2024 trace |

`regimebench/workload.py` makes the grid. It also has the named point sets `mk`, `mk_kv`, `mk_b8kv`, `mk_iso240`
and `mk_r10`.

## The three megakernels

| arm | design | status |
|---|---|---|
| `hv2-repeat` | The frozen hv2 kernel (one b1 / KV 128 decode, FP32 FFMA). One step is N copies of this kernel. | Timing proxy only. The output is not correct, so we do not do an accuracy check. |
| `hv2g` | hv2 with a run-time KV length and many tokens in one launch. The structure of hv2 does not change. | Reference arm. It shows how the hv2 design scales. |
| `mk` | A new kernel: tcgen05 tensor-core GEMM phases, split-KV decode attention, 1 to 256 tokens in each step. | The main result of this branch. |

The sources are in `regimebench/kernels/hv2/` (frozen), `regimebench/kernels/hv2g/` and `regimebench/kernels/mk/`.

### How `mk` works

`mk` is one cooperative kernel: 148 CTAs x 512 threads, one launch for each step, inside a CUDA graph.
Each layer has GEMM phases (QKV, O, gate and up, down) and one attention phase. Grid barriers separate the phases.

The weights come into shared memory by TMA. The tcgen05 MMAs write their results into TMEM.
The attention comes from the split-KV header in `regimebench/kernels/common/splitkv/`.
A host planner in that directory makes the list of attention work items before the step starts.

### Different designs for different token counts

The arm compiles one cubin for each token bucket: NPAD = 16, 32, 64, 128 or 256.
For a step, it uses the smallest bucket that holds all the tokens of the step.
Each bucket has its own design. The default knobs are in `regimebench/arms/mk.py` and `regimebench/kernels/mk/build.py`.

| part | NPAD 16 and 32 | NPAD 64 | NPAD 128 and 256 |
|---|---|---|---|
| GEMM engine | split-K over weight units | split-K over weight units | tile stream (`MK_TS`, `MK_TSB`): QKV, gate and up, and down read bf16 token tiles by TMA |
| LM head | pipelined LM ring (`MK_LMP`) | pipelined LM ring (`MK_LMP`) | LM head on the tile stream (`MK_LMTS`) |
| gate and up token tile | - | - | `MK_GU_TT` = 48 at NPAD 128, 96 at NPAD 256 |
| attention (all q_len 1) | split-KV, engine set by `skv_tune` from the KV lengths. tcgen05 consumer (`SKV_TC`) at NPAD 16 when KV >= 4K | split-KV, TMA engine. Stage-interleaved consumer (`SKV_SI`) when KV >= 4K | split-KV, TMA engine. `SKV_SI` at NPAD 128 when KV >= 4K |
| split combine | at the start of the O phase | own phase after attention (`MK_CMB`) | own phase after attention (`MK_CMB`) |
| attention plan | balance for each consumer group (`skvgrp`) | balance for each consumer group | balance for each consumer group |

Steps with q_len > 1 (R6) use the split-KV build with `SKV_QMAX 16`.
`probe/mk_core/LEDGER.md` gives the job, the gain and the reason for each knob and for each design that we did not keep.
Examples of designs that we did not keep: the 2-CTA (`cta_group::2`) GEMM path, and the tile stream at NPAD 16 to 64.

## Results

All times are microseconds for each step. Each ratio compares arms in the same Slurm job.

### Decode points `mk` (R0, R1, R2, R3, R6): 28 points, job 9806

`mk` is faster than the best of fi-graph and vLLM at 28 of 28 points. The geometric mean of mk / best is 0.832.

| point | SOL | best baseline | vLLM | mk | mk / best |
|---|---|---|---|---|---|
| R0-b1-kv128 | 150.9 | 890.6 | 2369.6 | 737.3 | 0.83 |
| R1-b32-kv128 | 210.6 | 1180.5 | 2824.8 | 894.5 | 0.76 |
| R1-b128-kv128 | 395.5 | 1766.2 | 4344.2 | 1531.8 | 0.87 |
| R1-b256-kv128 | 641.9 | 2263.0 | 6227.5 | 2204.3 | 0.97 |
| R2-b1-kv32768 | 618.9 | 1614.6 | 3256.2 | 1410.7 | 0.87 |
| R3-b32-kv1024 | 621.7 | 1692.8 | 3440.2 | 1404.1 | 0.83 |

The full table is in [results/2026-10-23-mk/final/REPORT.md](results/2026-10-23-mk/final/REPORT.md).
The reports of the versions v0 to v17 are in the same directory.

### Large-batch decode `mk_r10` (R10): 19 points, job 15059

`mk` is faster than fi-graph at 7 of 19 points. The geometric mean of mk / best is 1.013. vLLM is slower than fi-graph at all points.

The wins are b48-kv1K, b64-kv1K, b64-kv2K, b128-kv1K, conv-L3, conv-L4 and code-L3.
The worst losses are at b192 (1.071 and 1.072).
At R10, attention is 77 % to 89 % of the mk layer time. The other mk phases are faster than the non-attention kernels of fi-graph.

### Other studies

| directory | subject |
|---|---|
| `results/2026-10-22-b200/` | first full grid: hv2g, hv2-repeat, fi-graph and vLLM |
| `results/2026-10-27-*`, `results/2026-10-28-*` | attention studies: tcgen05 consumer, KV queue, L2 effects |
| `results/2026-10-29-occfuse/`, `results/2026-10-29-smallt/` | GEMM phase studies at small token counts |
| `results/2026-10-29-dilution/` | why the forward gain of mk becomes smaller inside vLLM serving |
| `results/2026-10-29-plots/` | summary plots |

We did not push the large raw dumps of these studies (more than 1 MB for each file). The summaries and the small records are in this branch.

## Repository layout

| path | contents |
|---|---|
| `regimebench/geometry.py`, `workload.py`, `fixture.py` | model sizes, regime grid, input data (frozen interfaces) |
| `regimebench/arms/` | `fi_graph`, `hv2g`, `hv2_repeat`, `mk`, `vllm_worker` |
| `regimebench/kernels/` | `hv2` (frozen), `hv2g`, `mk`, `common/splitkv` |
| `regimebench/ref/` | BF16 serving reference and FP32 oracle |
| `regimebench/harness/` | timing, accuracy, JSONL records |
| `regimebench/sol.py`, `probes.py`, `run.py`, `report.py` | SOL model, GPU probes, run driver, report |
| `probe/mk_core/` | design notes (`PASS-*.md`, `specs/`), `LEDGER.md` and the standalone probes (xover, occfuse, kvq and others) |
| `probe/xover/` | FFMA and tcgen05 GEMM phase bench, T = 1 to 256 |
| `docs/briefs/` | the briefs of the two owners of `mk` (MK-core and MK-attn) |
| `scripts/gpu_check_*.py` | GPU checks for each arm |
| `slurm/` | Slurm job scripts |
| `tests/` | CPU tests (pytest) |

## How to run

The commands use the paths of our cluster. Change `PY` and `PYTHONPATH` for your system.
The software versions are: torch 2.13 (CUDA 13.0), FlashInfer 0.7.0.post1, vLLM 0.31.0, cutlass-dsl 4.7.1.

```
export PY=/raid/garv901/vibe-megakernel/.venv/bin/python PYTHONPATH=/raid/garv901/regime-megakernel
$PY -m regimebench run --arms mk,fi-graph,vllm --workloads mk --tag NAME \
    [--seeds 7301,8917,42811] [--rounds 20] [--smoke] [--dry-run] [--variants "arm=v1,v2;..."]
$PY -m regimebench report runs/<RUN_DIR>... --out results/<date>-b200/     # REPORT.md, summary.csv, R*.png
sbatch slurm/regime.sbatch --arms ... --workloads ... --tag NAME          # GPU job (1 GPU, 2 h)
$PY scripts/gpu_check_mk.py                                               # mk accuracy check on a GPU
$PY -m pytest -q tests/                                                   # CPU tests
```

The run driver writes records to `runs/<UTC stamp>-<tag>/records.jsonl` (schema `regimebench/v1`).
It does not write over a run. Git does not keep `runs/`.

## Rules of the study

- Each sequence has its own KV region. Sequences do not share pages.
- The timed launch writes K and V only into the cache of the arm. The FP32 logit cast is not in the timed part.
- An arm passes the accuracy check when its rel-L2 error to the FP32 oracle is at most 2x the error of the BF16 reference.
- Each claim in a report has a job ID and a log path, or a "hypothesis" label.
