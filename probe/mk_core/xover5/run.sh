#!/bin/bash
# xover5: TMA vs cp.async ingress arms (SPEC: probe/mk_core/specs/SPEC-xover5.md). Builds into ./bin (ptxas log logs/build.log); unless
# --build-only, runs every arm over QKV / O / GU / D / LM x T 1..256 (JSON lines on stdout, PASS / FAIL lines to logs/check-<job>.log), ROUNDS rounds (default 2).
# Usage: bash run.sh [--build-only] [nph=0 (112, LM 4)] [reps=5] [shape=all] [T=0]
cd "$(dirname "$0")"; mkdir -p bin logs
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
BUILD_ONLY=0; [ "$1" = "--build-only" ] && { BUILD_ONLY=1; shift; }
: > logs/build.log
#      name XTMA RING WCPA     (KU 128, CG 1, no oracle)
ARMS=("a0 0 2 0" "c0 2 2 0" "x3 0 0 0" "c3 2 0 0" "a3 1 0 0" "w3 1 0 1" "wc3 2 0 1")
for arm in "${ARMS[@]}"; do
  read -r n xtma ring wcpa <<< "$arm"
  echo "== $n XTMA=$xtma RING=$ring WCPA=$wcpa" >> logs/build.log
  $NVCC -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v -DKU=128 -DXTMA=$xtma -DRING=$ring -DCG=1 -DORACLE=0 -DWCPA=$wcpa -DBIN=\"$n\" \
    xover5.cu -o bin/$n >> logs/build.log 2>&1 || { echo "build failed: $n (see logs/build.log)"; exit 1; }
done
echo "built: $(ls bin | tr '\n' ' ') (ptxas log: logs/build.log)"
[ $BUILD_ONLY = 1 ] && exit 0
for round in $(seq 1 ${ROUNDS:-2}); do
  echo "## round $round"
  for arm in "${ARMS[@]}"; do read -r n _ <<< "$arm"; ./bin/$n "$@" 2>> logs/check-${SLURM_JOB_ID:-local}.log | sed "s/^{/{\"round\":$round,/"; done
done
