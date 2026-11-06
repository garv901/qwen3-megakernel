#!/bin/bash
# clc: build bin/clc (+ ptxas -v log) and, unless --build-only, run the sweep (JSON lines on stdout, PASS/FAIL on stderr).
# Usage: bash run.sh [--build-only] [clc.cu args: --validate --arms .. --chunks .. --reps N]
cd "$(dirname "$0")"; mkdir -p bin logs
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
BUILD_ONLY=0; [ "$1" = "--build-only" ] && { BUILD_ONLY=1; shift; }
$NVCC -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v ${NVFLAGS} clc.cu -o bin/clc > logs/build.log 2>&1 || { echo "build failed (see logs/build.log)"; cat logs/build.log; exit 1; }
echo "built: bin/clc (ptxas log: logs/build.log)"
[ $BUILD_ONLY = 1 ] && exit 0
./bin/clc "$@"
