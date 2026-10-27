#!/bin/bash
# Binaries of xover4_small.sbatch: a0 / a3 from ../xover2/xover2.cu (flags as xover2 / xover3 run.sh), ts3 from ../xover3/xover3.cu (SB arm), ts1 / ts1w from xover4.cu (WU 0 / 1).
cd "$(dirname "$0")"; mkdir -p bin logs
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
F="-O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v"
: > logs/build_small.log
b() { echo "== $1" >> logs/build_small.log; shift; $NVCC $F "$@" >> logs/build_small.log 2>&1 || { echo "build failed (see logs/build_small.log)"; exit 1; }; }
b a0 -DKU=128 -DXTMA=0 -DRING=2 -DCG=1 -DORACLE=0 -DBIN=\"a0\" ../xover2/xover2.cu -o bin/a0
b a3 -DKU=128 -DXTMA=1 -DRING=0 -DCG=1 -DORACLE=0 -DBIN=\"a3\" ../xover2/xover2.cu -o bin/a3
b ts3 ../xover3/xover3.cu -o bin/ts3
b ts1 -DCG=1 xover4.cu -o bin/ts1
b ts1w -DCG=1 -DWU=1 xover4.cu -o bin/ts1w
echo "built: a0 a3 ts3 ts1 ts1w (ptxas log: logs/build_small.log)"
