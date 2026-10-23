#!/bin/bash
# bar: build the 7 arms (+ ptxas -v log) and, unless --build-only, run every binary (JSON lines on stdout, PASS/FAIL on stderr).
# Usage: bash run.sh [--build-only] [nit=20000] [reps=5] [skews=0,100]
cd "$(dirname "$0")"; mkdir -p bin logs
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
BUILD_ONLY=0; [ "$1" = "--build-only" ] && { BUILD_ONLY=1; shift; }
: > logs/build.log
for n in 0 1 2 3 4 5 6; do
  echo "== BAR=$n" >> logs/build.log
  $NVCC -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v -DBAR=$n bar.cu -o bin/bar$n >> logs/build.log 2>&1 || { echo "build failed: BAR=$n (see logs/build.log)"; exit 1; }
done
echo "built: $(ls bin | tr '\n' ' ') (ptxas log: logs/build.log)"
[ $BUILD_ONLY = 1 ] && exit 0
for n in 0 1 2 3 4 5 6; do ./bin/bar$n "$@"; done
