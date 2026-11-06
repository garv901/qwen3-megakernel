#!/bin/bash
# B5 run. Submit from the repo root: sbatch -p priority --time=00:25:00 probe/mk_core/occfuse/b5_epi/run.sbatch
#SBATCH --job-name=occf-b5
#SBATCH --gres=gpu:1
#SBATCH --output=runs/slurm-%j.log
set -uo pipefail
BENCH=b5_epi
REPO=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
D=$REPO/probe/mk_core/occfuse/$BENCH; R=$REPO/results/2026-10-29-occfuse/$BENCH; J=${SLURM_JOB_ID:-local}
mkdir -p "$D/bin" "$R"
echo "REPO=$REPO $(git -C "$REPO" rev-parse --short HEAD) job $J $(hostname)"; nvidia-smi -L
Q=clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active
nvidia-smi --query-gpu=$Q --format=csv | tee "$R/clocks_before_$J.csv"
NVCC=/usr/local/cuda/bin/nvcc; F="-O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v -I$REPO/probe/mk_core/occfuse/common"
$NVCC $F "$D/$BENCH.cu" -o "$D/bin/$BENCH" > "$R/build_$J.log" 2>&1 || { echo "build failed: $R/build_$J.log"; exit 1; }
B="$D/bin/$BENCH"
# quick: all modes, correctness first
timeout 120 $B --modes 1,2,3,4,5 --TT 16,128 --T 8,0 --nw 12 --KS 1,4 --launches 3 --tag quick > "$R/quick_$J.jsonl" 2> "$R/quick_$J.err" || echo "!! quick exit $?"
grep -c '"pass":0' "$R/quick_$J.jsonl"
# passes: pass 1 and pass 2, each per mode (a hang only loses that mode); then oracle arms
for r in 1 2; do
  for m in 1 3 4 5 2; do
    timeout 300 $B --modes $m --launches 7 --out "$R/p${r}_m${m}_$J.st" --tag p$r > "$R/p${r}_m${m}_$J.jsonl" 2> "$R/p${r}_m${m}_$J.err" || echo "!! p$r m$m exit $?"
  done
  timeout 120 $B --modes 1 --orc 1 --T 0 --nw 4,8,12 --launches 7 --out "$R/p${r}_null_$J.st" --tag p$r > "$R/p${r}_null_$J.jsonl" 2> "$R/p${r}_null_$J.err" || echo "!! p$r null exit $?"
  timeout 120 $B --modes 1 --orc 2 --T 1,8,16,0 --nw 4,8,12 --launches 7 --tag p$r > "$R/p${r}_nost_$J.jsonl" 2> "$R/p${r}_nost_$J.err" || echo "!! p$r nost exit $?"
done
nvidia-smi --query-gpu=$Q --format=csv | tee "$R/clocks_after_$J.csv"
