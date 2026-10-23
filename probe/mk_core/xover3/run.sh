#!/bin/bash
# xover3: build ts (tile-stream engine) and a3 (xover2 baseline: KU=128 XTMA=1 RING=0 CG=1, built from ../xover2/xover2.cu, not edited) with ptxas -v log;
# unless --build-only, run both over shapes x T (JSON lines on stdout, PASS/FAIL lines on stderr).
# Usage: bash run.sh [--build-only] [nph=0 (112)] [reps=5] [shape=all (QKV O GU D)] [T=0 (32 64 128 256)]
cd "$(dirname "$0")"; mkdir -p bin logs
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
BUILD_ONLY=0; [ "$1" = "--build-only" ] && { BUILD_ONLY=1; shift; }
: > logs/build.log
echo "== ts" >> logs/build.log
$NVCC -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v xover3.cu -o bin/ts >> logs/build.log 2>&1 || { echo "build failed: ts (see logs/build.log)"; exit 1; }
echo "== a3 (xover2 KU=128 XTMA=1 RING=0 CG=1 ORACLE=0)" >> logs/build.log
$NVCC -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v -DKU=128 -DXTMA=1 -DRING=0 -DCG=1 -DORACLE=0 -DBIN=\"a3\" \
  ../xover2/xover2.cu -o bin/a3 >> logs/build.log 2>&1 || { echo "build failed: a3 (see logs/build.log)"; exit 1; }
echo "built: $(ls bin | tr '\n' ' ') (ptxas log: logs/build.log)"
[ $BUILD_ONLY = 1 ] && exit 0
NPH=${1:-0}; REPS=${2:-5}; SH=${3:-all}; TS=${4:-0}
[ "$SH" = all ] && SH="QKV O GU D"; [ "$TS" = 0 ] && TS="32 64 128 256"
for s in $SH; do for t in $TS; do
  ./bin/a3 $NPH $REPS $s $t
  ./bin/ts $NPH $REPS $s $t
done; done
