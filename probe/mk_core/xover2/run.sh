#!/bin/bash
# xover2: build the 8 arms (+ ptxas -v log) and, unless --build-only, run every binary over all shapes and T (JSON lines on stdout,
# PASS/FAIL lines on stderr). Usage: bash run.sh [--build-only] [nph=0 (112, LM 4)] [reps=5] [shape=all] [T=0]
cd "$(dirname "$0")"; mkdir -p bin logs
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
BUILD_ONLY=0; [ "$1" = "--build-only" ] && { BUILD_ONLY=1; shift; }
: > logs/build.log
#      name KU XTMA RING CG ORACLE
ARMS=("a0 128 0 2 1 0" "o0 128 0 2 1 1" "a1 64 0 4 1 0" "a2 128 1 2 1 0" "a3 128 1 0 1 0" "a4 64 1 0 1 0" "a5 128 1 0 2 0" "o3 128 1 0 1 1")
for arm in "${ARMS[@]}"; do
  read -r n ku xtma ring cg oracle <<< "$arm"
  echo "== $n KU=$ku XTMA=$xtma RING=$ring CG=$cg ORACLE=$oracle" >> logs/build.log
  $NVCC -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v -DKU=$ku -DXTMA=$xtma -DRING=$ring -DCG=$cg -DORACLE=$oracle -DBIN=\"$n\" \
    xover2.cu -o bin/$n >> logs/build.log 2>&1 || { echo "build failed: $n (see logs/build.log)"; exit 1; }
done
echo "built: $(ls bin | tr '\n' ' ') (ptxas log: logs/build.log)"
[ $BUILD_ONLY = 1 ] && exit 0
for arm in "${ARMS[@]}"; do read -r n _ <<< "$arm"; ./bin/$n "$@"; done
