#!/bin/bash
# B1 run. Submit from the repo root: sbatch -p <priority|main> --time=00:15:00 probe/mk_core/occfuse/b1_stream/run.sbatch
#SBATCH --job-name=occf-b1
#SBATCH --gres=gpu:1
#SBATCH --output=runs/slurm-%j.log
set -uo pipefail
BENCH=b1_stream
REPO=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
D=$REPO/probe/mk_core/occfuse/$BENCH; R=$REPO/results/2026-10-29-occfuse/$BENCH; J=${SLURM_JOB_ID:-local}
mkdir -p "$D/bin" "$R"
echo "REPO=$REPO $(git -C "$REPO" rev-parse --short HEAD) job $J $(hostname)"; nvidia-smi -L
Q=clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active
nvidia-smi --query-gpu=$Q --format=csv | tee "$R/clocks_before_$J.csv"
NVCC=/usr/local/cuda/bin/nvcc; F="-O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v -I$REPO/probe/mk_core/occfuse/common"
$NVCC $F "$D/$BENCH.cu" -o "$D/bin/$BENCH" > "$R/build_$J.log" 2>&1 || { echo "build failed: $R/build_$J.log"; exit 1; }
grep -E "registers|spill" "$R/build_$J.log" | head -5
B="$D/bin/$BENCH"
# ---- RUN: two identical full-sweep passes (gate points repeated); N x S x R x place ----
for r in 1 2; do
  timeout 600 $B --out "$R/run${r}_$J.st" --tag run$r > "$R/run${r}_$J.jsonl" 2> "$R/run${r}_$J.err" || echo "!! run$r exit $?"
  wc -l "$R/run${r}_$J.jsonl"
done
nvidia-smi --query-gpu=$Q --format=csv | tee "$R/clocks_after_$J.csv"
