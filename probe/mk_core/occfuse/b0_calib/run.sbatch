#!/bin/bash
# B0 run. Submit from the repo root: sbatch -p <priority|main> --time=00:15:00 probe/mk_core/occfuse/b0_calib/run.sbatch
#SBATCH --job-name=occf-b0
#SBATCH --gres=gpu:1
#SBATCH --output=runs/slurm-%j.log
set -uo pipefail
BENCH=b0_calib
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
# ---- RUN: two identical passes (gate points repeated) ----
for r in 1 2; do
  timeout 300 $B --arm all --out "$R/run${r}_$J.st" --tag run$r > "$R/run${r}_$J.jsonl" 2> "$R/run${r}_$J.err" || echo "!! run$r exit $?"
done
# ---- clock arms with an NVML sampler (50 ms) around each ----
clockarm() {  # name tag
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.gr,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -lms 50 > "$R/nvml_$1_$2_$J.csv" &
  local P=$!; sleep 0.5
  timeout 120 $B --arm $1 --dur 6 --tag $2 --out "$R/c_$1_$2_$J.st" > "$R/$1_$2_$J.jsonl" 2> "$R/$1_$2_$J.err" || echo "!! $1 exit $?"
  sleep 0.5; kill $P 2>/dev/null; wait $P 2>/dev/null
}
clockarm c_idle free; clockarm c_busy free; clockarm c_idle free2; clockarm c_busy free2
# ---- clock lock attempt without sudo (record the outcome) ----
{ echo "# nvidia-smi -lgc 1965 (no sudo)"; nvidia-smi -lgc 1965 2>&1; echo "exit=$?"; echo "# sudo -n true"; sudo -n true 2>&1; echo "exit=$?"; } | tee "$R/lgc_$J.txt"
if grep -q "exit=0" <(sed -n '2,3p' "$R/lgc_$J.txt"); then
  echo "lock worked: running locked clock arms"; clockarm c_idle locked; clockarm c_busy locked; nvidia-smi -rgc
fi
nvidia-smi --query-gpu=$Q --format=csv | tee "$R/clocks_after_$J.csv"
