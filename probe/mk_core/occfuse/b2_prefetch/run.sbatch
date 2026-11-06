#!/bin/bash
# B2 run. Submit from the repo root: STAGE=<pilot|full|repeat> sbatch -p <priority|main> --time=00:25:00 --export=ALL,STAGE=full probe/mk_core/occfuse/b2_prefetch/run.sbatch
#SBATCH --job-name=occf-b2
#SBATCH --gres=gpu:1
#SBATCH --output=runs/slurm-%j.log
set -uo pipefail
BENCH=b2_prefetch; STAGE=${STAGE:-full}
REPO=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
D=$REPO/probe/mk_core/occfuse/$BENCH; R=$REPO/results/2026-10-29-occfuse/$BENCH; J=${SLURM_JOB_ID:-local}
mkdir -p "$D/bin" "$R"
echo "REPO=$REPO $(git -C "$REPO" rev-parse --short HEAD) job $J $(hostname) stage $STAGE"; nvidia-smi -L
Q=clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active
nvidia-smi --query-gpu=$Q --format=csv | tee "$R/clocks_before_$J.csv"
NVCC=/usr/local/cuda/bin/nvcc; F="-O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v -I$REPO/probe/mk_core/occfuse/common"
$NVCC $F "$D/$BENCH.cu" -o "$D/bin/$BENCH" > "$R/build_$J.log" 2>&1 || { echo "build failed: $R/build_$J.log"; exit 1; }
grep -E "registers|spill" "$R/build_$J.log" | head -3
B="$D/bin/$BENCH"
GWS="0:0,0.5:0,1:0,1.5:0,2:0,2.2:0,3.5:0,4:0,4.2:0,6:0,0:0.25,0:0.5,0:0.75,0:1,0:1.5,0:2,0:2.5,0:3"
run() { n=$1; shift; echo "== $n"; /usr/bin/time -f "%e s" timeout $TMO $B "$@" --tag $n --out "$R/${n}_$J.st" > "$R/${n}_$J.jsonl" 2> "$R/${n}_$J.err" || echo "!! $n exit $?"; wc -l "$R/${n}_$J.jsonl"; tail -2 "$R/${n}_$J.err"; }
case $STAGE in
  pilot)   # gate arm (2 passes) + a small stream sweep
    TMO=300; run gate1 --gate 1 --P 0,1,2,-1 --ws $GWS; run gate2 --gate 1 --P 0,1,2,-1 --ws $GWS
    TMO=300; run gen1 --N 48,148 --S 64,256,1024 --R 2,4 --P 0,1,2,-1 ;;
  full)    # stream sweep: spec grid (classic slot and TS slot at TT 16), gate arm, diagnostic gate arms (noB: A only; hot: A window repeats, L2 hits)
    SW="0:0,0.5:0,1:0,2:0,4:0,6:0,0:1,0.5:1,1:1,2:1,4:1,6:1,0:2,0.5:2,1:2,2:2,4:2,6:2,0:0.5,0:4,0:6"
    SS=32,64,96,128,192,256,384,512,768,1024
    TMO=900; run gen_c --mode 0 --N 48,96,128,148 --S $SS --R 2,3,4 --P 0,1,2,-1 --ws $SW
    TMO=900; run gen_t --mode 1 --N 48,96,128,148 --S $SS --R 2,3,4,5 --P 0,1,2,-1 --ws $SW
    TMO=300; run gate3 --gate 1 --P 0,1,2,-1 --ws $GWS
    TMO=300; run diagnoB --gate 1 --noB 1 --P 0,-1 --ws 0:0,0:1,0:2
    TMO=300; run diaghot --gate 1 --hot 1 --P 0,-1 --ws 0:0,0:1,0:2
    TMO=300; run diaghotnoB --gate 1 --hot 1 --noB 1 --P 0,-1 --ws 0:0,0:1,0:2 ;;
  repeat)  # second job: gate points again + gate with R overrides + small stream subset
    TMO=300; run gate4 --gate 1 --P 0,1,2,-1 --ws $GWS
    TMO=300; run gateR --gate 1 --P 0,-1 --gR 2,3,4 --ws 0:0,0:1,0:2
    TMO=300; run diagnoB4 --gate 1 --noB 1 --P 0,-1 --ws 0:0,0:1,0:2
    TMO=300; run gen4 --mode 0 --N 48,148 --S 64,256,1024 --R 2,4 --P 0,1,2,-1 ;;
esac
nvidia-smi --query-gpu=$Q --format=csv | tee "$R/clocks_after_$J.csv"
