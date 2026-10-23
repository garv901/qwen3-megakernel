#!/bin/bash
#SBATCH --job-name=mkc-p0fi
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm-%j.log
# MK-core Pass 0: fi-graph headline variant per-kernel timeline at the R1 points (GEMM kernels = per-phase targets).
# Submit from the worktree root.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi -L
OUT=results/2026-10-23-mk
"$PY" scripts/gpu_check_fi.py --variants graph-pdl-best --timeline --json "$OUT/p0_fi_timeline_$SLURM_JOB_ID.json" \
  --workloads R1-b1-kv128,R1-b16-kv128,R1-b64-kv128,R1-b256-kv128
echo "exit $?"
