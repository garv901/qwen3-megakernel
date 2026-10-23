#!/bin/bash
#SBATCH --job-name=mkc-spread
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=runs/slurm-%j.log
# MK-core P1: barrier latency vs arrival spread inside mk (stamps), full build and null-all build.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1 PYTHONPATH=$REPO
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi -L
W=R0-b1-kv128,R1-b16-kv128,R1-b64-kv128,R1-b256-kv128
"$PY" probe/mk_core/debug/stamps_spread.py $W 0
"$PY" probe/mk_core/debug/stamps_spread.py $W 0x7f
