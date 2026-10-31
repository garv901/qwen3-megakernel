#!/bin/bash
# xover6: build bin/x6 (xover6.cu, sm_100a, -Xptxas -v -> logs/build.log) and, as the GRID_N reference of job 9765, bin/ref{148,111,74} from ../xover4/xover4.cu (-DCG=1 -DGRID_N=n).
# Usage: bash run.sh [--build-only]   (without the flag: nothing else, the run plan is xover6.sbatch)
cd "$(dirname "$0")"; mkdir -p bin logs
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
: > logs/build.log
echo "== x6" >> logs/build.log
$NVCC -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v xover6.cu -o bin/x6 >> logs/build.log 2>&1 || { echo "build failed: x6 (see logs/build.log)"; exit 1; }
for g in 148 111 74; do
  echo "== ref$g (xover4.cu -DCG=1 -DGRID_N=$g)" >> logs/build.log
  $NVCC -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -DCG=1 -DGRID_N=$g ../xover4/xover4.cu -o bin/ref$g >> logs/build.log 2>&1 || { echo "build failed: ref$g"; exit 1; }
done
echo "built: $(ls bin | tr '\n' ' ') (ptxas log: logs/build.log)"
