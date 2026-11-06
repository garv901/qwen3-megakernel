#!/bin/bash
# B7 run. Submit from the repo root: STAGE=<full|repeat> sbatch -p <priority|main> --time=00:25:00 --export=ALL,STAGE=full probe/mk_core/occfuse/b7_splitk/run.sbatch
#SBATCH --job-name=occf-b7
#SBATCH --gres=gpu:1
#SBATCH --output=runs/slurm-%j.log
set -uo pipefail
BENCH=b7_splitk; STAGE=${STAGE:-full}
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
run() { n=$1; shift; echo "== $n"; /usr/bin/time -f "%e s" timeout $TMO $B "$@" --tag $n --out "$R/${n}_$J.st" > "$R/${n}_$J.jsonl" 2> "$R/${n}_$J.err" || echo "!! $n exit $?"; wc -l "$R/${n}_$J.jsonl"; tail -2 "$R/${n}_$J.err"; }
SH="QKV:4096:1024 O:1024:2048 GU:6144:1024 D:1024:3072"
case $STAGE in
  full)   # pass 1: sanity, per-(shape,T) sweep x KS x {redadd, ws} x pf {1,0}; oracle arms at the QKV T128 gate; pass 2: gate again (11 launches) + sweep again (pf 1)
    TMO=90; run sanity --shapes QKV:4096:1024 --T 128 --KS 4,8 --launches 3 --pf 1 --keepst QKV
    for s in $SH; do for T in 8 32 128 256; do TMO=240; run p1_${s%%:*}_T$T --shapes $s --T $T --pf 1,0 --keepst QKV; done; done
    TMO=120; run orc_nomma --orc 1 --shapes QKV:4096:1024 --T 128 --KS 2,4,8 --arm 0
    TMO=120; run orc_noepi --orc 2 --shapes QKV:4096:1024 --T 128 --KS 2,4,8 --arm 0
    TMO=120; run orc_noB --orc 4 --shapes QKV:4096:1024 --T 128 --KS 2,4,8 --arm 0
    TMO=120; run orc_noA --orc 8 --shapes QKV:4096:1024 --T 128 --KS 2,4,8 --arm 0
    TMO=180; run gate2 --pf 1,0 --shapes QKV:4096:1024 --T 128 --KS 2,4,8 --arm 0 --launches 11 --keepst QKV
    for s in $SH; do for T in 8 32 128 256; do TMO=240; run p2_${s%%:*}_T$T --shapes $s --T $T --pf 1; done; done ;;
esac
nvidia-smi --query-gpu=$Q --format=csv | tee "$R/clocks_after_$J.csv"
