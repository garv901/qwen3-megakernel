#!/bin/bash
#SBATCH --job-name=mkc-p1b
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=runs/slurm-%j.log
# MK-core P1: stamps spread with MK_PFA=1 (default), then ring-depth A/B on NPAD 16 points.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1 PYTHONPATH=$REPO
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi -L
"$PY" probe/mk_core/debug/stamps_spread.py R0-b1-kv128,R1-b16-kv128,R1-b64-kv128 0
"$PY" probe/mk_core/debug/stamps_spread.py R0-b1-kv128 0x7f
"$PY" probe/mk_core/debug/ab.py R0-b1-kv128,R1-b16-kv128,R2-b1-kv8192,R3-b8-kv4096 '[{"MK_RING":2},{"MK_RING":3},{"MK_RING":4}]' 9
