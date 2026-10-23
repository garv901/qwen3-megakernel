#!/bin/bash
#SBATCH --job-name=mkc-hang
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=runs/slurm-%j.log
# MK-core hang triage: gpu_check_mk.py per (workload, knob JSON) with a timeout; MK_EXTRA_DEFINES from the environment.
# sbatch slurm/mk_hang.sbatch TIMEOUT_S ATTN 'W:JSON' ['W:JSON' ...]
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1 PYTHONPATH=$REPO
echo "REPO=$REPO $(git rev-parse --short HEAD) MK_EXTRA_DEFINES=${MK_EXTRA_DEFINES:-}"
nvidia-smi -L
T=$1; A=$2; shift 2
for wk in "$@"; do
  w=${wk%%:*}; k=${wk#*:}
  echo "@@ $w $k"
  timeout -s KILL "$T" "$PY" scripts/gpu_check_mk.py --workloads "$w" --attn "$A" --knobs "$k" 2>&1 | grep -vE "^=== " | head -30
  echo "@@ rc ${PIPESTATUS[0]}"
done
