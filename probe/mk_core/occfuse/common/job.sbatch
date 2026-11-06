#!/bin/bash
# occfuse job template. Copy to <bench>/run.sbatch and edit the RUN section. Submit from the repo root:
#   sbatch -p <priority|main> --time=<honest> probe/mk_core/occfuse/<bench>/run.sbatch
# (choose partition and time with slurm-windows first; see the run-gpu-job skill)
#SBATCH --job-name=occf-BENCH
#SBATCH --gres=gpu:1
#SBATCH --output=runs/slurm-%j.log
set -uo pipefail
BENCH=BENCH                                   # folder name, e.g. b1_stream
REPO=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
D=$REPO/probe/mk_core/occfuse/$BENCH; R=$REPO/results/2026-10-29-occfuse/$BENCH; J=${SLURM_JOB_ID:-local}
mkdir -p "$D/bin" "$R"
echo "REPO=$REPO $(git -C "$REPO" rev-parse --short HEAD) job $J $(hostname)"; nvidia-smi -L
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv | tee "$R/clocks_before_$J.csv"
NVCC=/usr/local/cuda/bin/nvcc; F="-O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v -I$REPO/probe/mk_core/occfuse/common"
$NVCC $F "$D/$BENCH.cu" -o "$D/bin/$BENCH" > "$R/build_$J.log" 2>&1 || { echo "build failed: $R/build_$J.log"; exit 1; }
grep -E "registers|spill" "$R/build_$J.log" | head -5
# ---- RUN (edit): every run line under a timeout, JSONL to $R/<name>_$J.jsonl, raw stamps to $R/<name>_$J.st ----
timeout 600 "$D/bin/$BENCH" --out "$R/run_$J.st" > "$R/run_$J.jsonl" || echo "!! run exit $?"
# ----
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv | tee "$R/clocks_after_$J.csv"
