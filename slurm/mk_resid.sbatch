#!/bin/bash
#SBATCH --job-name=mkc-resid
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=runs/slurm-%j.log
# MK-core: per-step residual split (launch gap, prologue, layer 0, LM, argmax) from MK_STAMPS: sbatch slurm/mk_resid.sbatch WORKLOADS ['JSON knobs']
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1 PYTHONPATH=$REPO
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi -L
"$PY" probe/mk_core/debug/residual.py "$@"
