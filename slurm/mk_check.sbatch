#!/bin/bash
#SBATCH --job-name=mkc-check
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm-%j.log
# MK-core: mk accuracy + idempotence + timing (all buckets), then per-phase stamps. Submit from the worktree root.
# Extra args go to both gpu_check_mk.py calls (e.g. --workloads ids).
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi -L
OUT=results/2026-10-23-mk
"$PY" scripts/gpu_check_mk.py --json "$OUT/mk_check_$SLURM_JOB_ID.json" "$@"
rc=$?
echo "check exit $rc"
[ $rc -eq 0 ] && "$PY" scripts/gpu_check_mk.py --stamps "$@"
echo "stamps exit $?"
