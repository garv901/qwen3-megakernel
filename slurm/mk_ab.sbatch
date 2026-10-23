#!/bin/bash
#SBATCH --job-name=mkc-ab
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=runs/slurm-%j.log
# MK-core knob A/B on one GPU: sbatch slurm/mk_ab.sbatch WORKLOADS 'JSON knob sets' [rounds]  (then optional stamps spread with $SPREAD_KNOBS)
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1 PYTHONPATH=$REPO
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi -L
"$PY" probe/mk_core/debug/ab.py "$@"
