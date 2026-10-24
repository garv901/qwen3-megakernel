#!/bin/bash
# xover4: build ts1 (-DCG=1) and ts2 (-DCG=2) with ptxas -v log; unless --build-only, run both over shapes x T (JSON lines on stdout, PASS/FAIL lines on stderr).
# Usage: bash run.sh [--build-only] [nph=0 (112)] [reps=5] [shape=all (QKV O GU D)] [T=0 (32 64 128 256)]
cd "$(dirname "$0")"; mkdir -p bin logs
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
BUILD_ONLY=0; [ "$1" = "--build-only" ] && { BUILD_ONLY=1; shift; }
: > logs/build.log
for cg in 1 2; do
  echo "== ts$cg (-DCG=$cg)" >> logs/build.log
  $NVCC -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v -DCG=$cg xover4.cu -o bin/ts$cg >> logs/build.log 2>&1 || { echo "build failed: ts$cg (see logs/build.log)"; exit 1; }
done
echo "built: $(ls bin | tr '\n' ' ') (ptxas log: logs/build.log)"
[ $BUILD_ONLY = 1 ] && exit 0
NPH=${1:-0}; REPS=${2:-5}; SH=${3:-all}; TS=${4:-0}
[ "$SH" = all ] && SH="QKV O GU D"; [ "$TS" = 0 ] && TS="32 64 128 256"
for s in $SH; do for t in $TS; do
  ./bin/ts1 $NPH $REPS $s $t
  ./bin/ts2 $NPH $REPS $s $t
done; done
