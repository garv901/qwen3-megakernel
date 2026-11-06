#!/bin/bash
# B3 run. Submit from the repo root: sbatch -p priority --time=00:15:00 probe/mk_core/occfuse/b3_step/run.sbatch
#SBATCH --job-name=occf-b3
#SBATCH --gres=gpu:1
#SBATCH --output=runs/slurm-%j.log
set -uo pipefail
BENCH=b3_step
REPO=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
D=$REPO/probe/mk_core/occfuse/$BENCH; R=$REPO/results/2026-10-29-occfuse/$BENCH; J=${SLURM_JOB_ID:-local}
mkdir -p "$D/bin" "$R"
echo "REPO=$REPO $(git -C "$REPO" rev-parse --short HEAD) job $J $(hostname)"; nvidia-smi -L
Q=clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active
nvidia-smi --query-gpu=$Q --format=csv | tee "$R/clocks_before_$J.csv"
NVCC=/usr/local/cuda/bin/nvcc; F="-O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v -I$REPO/probe/mk_core/occfuse/common"
$NVCC $F "$D/$BENCH.cu" -o "$D/bin/$BENCH" > "$R/build_$J.log" 2>&1 || { echo "build failed: $R/build_$J.log"; exit 1; }
grep -E "registers|spill" "$R/build_$J.log" | head -3
$NVCC $F -DTW=1 "$D/$BENCH.cu" -o "$D/bin/${BENCH}_tw" > "$R/build_tw_$J.log" 2>&1 || echo "tw build failed"
B="$D/bin/$BENCH"
run() { local name=$1; shift; timeout 400 $B "$@" --tag $name > "$R/${name}_$J.jsonl" 2> "$R/${name}_$J.err" || echo "!! $name exit $?"; }
# LEAN 0 = descriptors rebuilt per MMA (as k_ts / mk), LEAN 1 = hoisted per step. F (--fence) 0 = tcgen05.fence::after_thread_sync per k-step (k_ts), 1 = once per tile.
# DEP = independent accumulator chains inside a tile (1 = mk).
for L in 0 1; do for F in 0 1; do
  run check_L${L}_F$F --arm check --SPT 4 --lean $L --fence $F
  run check2_L${L}_F$F --arm check --N 16,256 --SPT 1,16 --lean $L --fence $F
done; done
for r in 1 2; do for L in 0 1; do for F in 0 1; do
  run sweep${r}_L${L}_F$F --arm sweep --dep 1 --lean $L --fence $F
  run dep${r}_L${L}_F$F --arm sweep --dep 2,4 --acc 1,2 --lean $L --fence $F --act 1
done; done; done
for L in 0 1; do for F in 0 1; do run single_L${L}_F$F --arm sweep --ntile 1 --acc 1,2 --dep 1 --act 1 --lean $L --fence $F; done; done
run ring8 --arm sweep --ring 8 --N 16,64,128 --acc 2 --act 1 --commit 0 --dep 1 --lean 0
# try_wait (default) vs test_wait spin on the mbarriers (-DTW=1 build)
for F in 0 1; do timeout 400 "$D/bin/${BENCH}_tw" --arm sweep --N 16,128,256 --acc 2 --act 1 --dep 1 --lean 1 --fence $F --tag tw_F$F > "$R/tw_F${F}_$J.jsonl" 2> "$R/tw_F${F}_$J.err" || echo "!! tw$F"; done
nvidia-smi --query-gpu=$Q --format=csv | tee "$R/clocks_after_$J.csv"
