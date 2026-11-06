#!/bin/bash
# B4 run. Submit from the repo root: sbatch -p <priority|main> --time=00:25:00 probe/mk_core/occfuse/b4_bpath/run.sbatch [PASSES=2] [gate-only: GATE=1]
#SBATCH --job-name=occf-b4
#SBATCH --gres=gpu:1
#SBATCH --output=runs/slurm-%j.log
set -uo pipefail
BENCH=b4_bpath
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
PASSES=${PASSES:-2}; GATE=${GATE:-0}
EXTRA=""; [ "$GATE" = 1 ] && EXTRA="--T 8,16,256 --kb 128,1024"
# one invocation per (loaded, mode): a hang loses only that arm
for r in $(seq 1 $PASSES); do
  for ld in 0 1; do for md in 3 1 2 4 0; do
    n=r${r}_l${ld}_m${md}
    timeout 400 $B --mode $md --loaded $ld $EXTRA --tag $n --out "$R/${n}_$J.st" > "$R/${n}_$J.jsonl" 2> "$R/${n}_$J.err" || echo "!! $n exit $?"
  done; done
done
nvidia-smi --query-gpu=$Q --format=csv | tee "$R/clocks_after_$J.csv"
