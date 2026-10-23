#!/bin/bash
# xover: build the 4 engine binaries (+ ptxas -v log) and, unless --build-only, run all configs (JSON lines on stdout,
# PASS/FAIL lines on stderr). Usage: bash run.sh [--build-only] [nph=112] [reps=5] [shape=all] [T=0]
cd "$(dirname "$0")"; mkdir -p bin logs
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
BUILD_ONLY=0; [ "$1" = "--build-only" ] && { BUILD_ONLY=1; shift; }
: > logs/build.log
for e in 0 1 2 -1; do
  n=$e; [ "$e" = "-1" ] && n=m1
  echo "== ENG=$e" >> logs/build.log
  $NVCC -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v -DENG=$e xover.cu -o bin/xover_e$n >> logs/build.log 2>&1 || { echo "build failed ENG=$e"; exit 1; }
done
echo "built: $(ls bin | tr '\n' ' ') (ptxas log: logs/build.log)"
[ $BUILD_ONLY = 1 ] && exit 0
for n in m1 0 1 2; do ./bin/xover_e$n "$@"; done
