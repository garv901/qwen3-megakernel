#!/bin/bash
#SBATCH --job-name=mk-bigt-watch
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=runs/slurm-%j.log
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1 MK_EXTRA_DEFINES="${MK_EXTRA_DEFINES:-}"
for w in "$@"; do timeout 200 "$PY" scripts/mk_bigt_watch.py "$w" 2>&1 | grep -v Warning; echo "exit $?"; done
