#!/bin/bash
#SBATCH --job-name=mkc-each
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm-%j.log
# MK-core: gpu_check_mk.py one workload per process with a timeout (hang triage). sbatch slurm/mk_check_each.sbatch 'W1,W2,..' [TIMEOUT_S] [extra args]
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1 PYTHONPATH=$REPO
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi -L
WS=$1; T=${2:-120}; shift; shift || true
for w in ${WS//,/ }; do
  timeout -s KILL "$T" "$PY" scripts/gpu_check_mk.py --workloads "$w" "$@" 2>&1 | grep -vE "^=== " | tail -4
  echo "@@ $w rc ${PIPESTATUS[0]}"
done
