#!/bin/bash
# B6 b6_fin. Submit from the repo root: sbatch -p priority --time=00:25:00 probe/mk_core/occfuse/b6_fin/run.sbatch [pilot|sweep|all|gate]
#SBATCH --job-name=occf-b6
#SBATCH --gres=gpu:1
#SBATCH --output=runs/slurm-%j.log
set -uo pipefail
MODE=${1:-sweep}
BENCH=b6_fin
REPO=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
D=$REPO/probe/mk_core/occfuse/$BENCH; R=$REPO/results/2026-10-29-occfuse/$BENCH; J=${SLURM_JOB_ID:-local}
mkdir -p "$D/bin" "$R"
echo "REPO=$REPO $(git -C "$REPO" rev-parse --short HEAD) job $J $(hostname) mode $MODE"; nvidia-smi -L
Q=clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active
nvidia-smi --query-gpu=$Q --format=csv | tee "$R/clocks_before_$J.csv"
NVCC=/usr/local/cuda/bin/nvcc; F="-O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v -DNSTMAX=232 -I$REPO/probe/mk_core/occfuse/common"
$NVCC $F "$D/$BENCH.cu" -o "$D/bin/$BENCH" > "$R/build_$J.log" 2>&1 || { echo "build failed"; exit 1; }
B=$D/bin/$BENCH
run() { # name args...
  n=$1; shift
  timeout 100 $B "$@" --out "$R/${n}_$J.st" >> "$R/${MODE}_$J.jsonl" 2> "$R/${n}_$J.err"; rc=$?
  echo "$n rc=$rc"; [ $rc -ne 0 ] && echo "!! $n exit $rc (124 = timeout/hang)" | tee -a "$R/hangs_$J.txt"
}
case $MODE in
pilot)  # gate point + a few, short
  run pilot_gate --T 8 --H 1024 --KS 16 --place 0,1 --var 0 --launches 3 --nph 60
  run pilot_b --T 8,256 --H 1024,2048 --KS 12 --place 2,3 --var 0,1,2 --launches 3 --nph 60
  run pilot_a256 --T 256 --H 1024 --KS 16 --place 1 --var 0 --launches 3 --nph 60 ;;
sweep)  # one process per (place, T) so that a hang loses one group only
  for pl in 0 1 2 3; do for T in 1 8 16 32 64 128 256; do
    run s_p${pl}_T$T --T $T --place $pl --var 0,1,2 --pdelay 1500
  done; done ;;
all)
  for pl in 0 1 2 3; do for T in 1 8 16 32 64 128 256; do
    run s_p${pl}_T$T --T $T --place $pl --var 0,1,2 --pdelay 1500
  done; done
  run g_pd1500 --T 8 --H 1024 --KS 16 --place 0,1 --var 0,1,2 --pdelay 1500 --rep 2 --launches 9
  run g_pd0 --T 8 --H 1024 --KS 16 --place 0,1 --var 0,1,2 --pdelay 0 --rep 2 --launches 9
  for pl in 0 1 2 3; do run pd0_p$pl --H 1024 --KS 16 --place $pl --var 0,1,2 --pdelay 0; done ;;
gate)   # gate points again (second pass) + pdelay 0
  run g_pd1500 --T 8 --H 1024 --KS 16 --place 0,1 --var 0,1,2 --pdelay 1500 --rep 2 --launches 9
  run g_pd0 --T 8 --H 1024 --KS 16 --place 0,1 --var 0,1,2 --pdelay 0 --rep 2 --launches 9 ;;
esac
nvidia-smi --query-gpu=$Q --format=csv | tee "$R/clocks_after_$J.csv"
