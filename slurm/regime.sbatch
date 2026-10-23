#!/bin/bash
#SBATCH --job-name=regime
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm-%j.log
# Usage: sbatch slurm/regime.sbatch --arms fi-graph,hv2g,hv2-repeat,vllm --workloads R0 --tag NAME [...]
# All arguments are forwarded to `python -m regimebench run`. Submit from the checkout root (main or a
# worktree): the job runs that checkout's code and writes its runs/ there.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi -L
exec "$PY" -m regimebench run "$@"
