#!/bin/bash
#SBATCH --job-name=mk-bigt-check
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --output=runs/slurm-%j.log
# MK_BIGT (steps 1 + 3): accuracy (FP32 oracle) + idempotence + timing of the large-T build, then the BIGT build forced onto small steps,
# then the default bucket points (regression). Arg 1 / 2: only the preset / only forced + buckets (default: all). Each pass under a timeout (a grid-barrier hang must not eat the job). Submit from the worktree root.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi -L
OUT=results/2026-11-06-mk-bigt
mkdir -p "$OUT"
[ "${1:-all}" != 2 ] && timeout 1200 "$PY" scripts/gpu_check_mk.py --workloads mk_bigt --json "$OUT/bigt_$SLURM_JOB_ID.json"
echo "bigt exit $?"
[ "${1:-all}" != 1 ] && timeout 600 "$PY" scripts/gpu_check_mk.py --workloads R1-b8-kv128,R1-b128-kv128,R1-b256-kv128,R6-b1-kv1024-q4,R2-b1-kv8192 --knobs '{"bigt": 1}' --json "$OUT/bigt_forced_$SLURM_JOB_ID.json"
echo "forced exit $?"
[ "${1:-all}" != 1 ] && timeout 600 "$PY" scripts/gpu_check_mk.py --json "$OUT/buckets_$SLURM_JOB_ID.json"
echo "buckets exit $?"
