#!/bin/bash
#SBATCH --job-name=mk-bigt-attn
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=runs/slurm-%j.log
# MK_BIGT step 3: unit check of the prefill attention (attn_prefill.cuh) vs an FP32 torch reference + per-layer timing. Args: case list (default all). Submit from the worktree root.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1
nvidia-smi -L
OUT=results/2026-11-08-mk-bigt-attn
mkdir -p "$OUT"
timeout 600 "$PY" scripts/mk_prefill_attn_check.py --cases "${1:-A,B,C,D,E,F}" --json "$OUT/attn_$SLURM_JOB_ID.json" 2>&1 | grep -v Warning
echo "exit ${PIPESTATUS[0]}"
