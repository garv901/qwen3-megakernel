#!/bin/bash
#SBATCH --job-name=regime
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=02:00:00
#SBATCH --output=/raid/garv901/regime-megakernel/runs/slurm-%j.log
# Usage: sbatch slurm/regime.sbatch --arms fi-graph,hv2g,hv2-repeat,vllm --workloads R0 --tag NAME [...]
# All arguments are forwarded to `python -m regimebench run`.
set -euo pipefail
mkdir -p /raid/garv901/regime-megakernel/runs
source /raid/garv901/regime-megakernel/slurm/env.sh
cd /raid/garv901/regime-megakernel
nvidia-smi -L
exec "$PY" -m regimebench run "$@"
