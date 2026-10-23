#!/bin/bash
#SBATCH --job-name=mkc-spk
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=runs/slurm-%j.log
# Stamps spread per phase for several knob sets: sbatch slurm/mk_spreadk.sbatch WORKLOADS 'JSON1' ['JSON2' ...]
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1 PYTHONPATH=$REPO
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi -L
W=$1; shift
for K in "$@"; do echo "== knobs $K"; "$PY" probe/mk_core/debug/stamps_spread.py "$W" 0 "$K"; done
