#!/bin/bash
#SBATCH --job-name=mkc-tc-a2
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=runs/slurm-%j.log
# MK-core A2: tcgen05 attention consumer (SKV_TC) in mk. Interleaved A/B (probe/mk_core/debug/ab.py, 7 rounds, accuracy graded per variant):
#   variant 0 = today's skv_tune, variant 1 = skv_tune(tc=1) (NPAD 16 / 32, q_len 1, kv >= SKV_TC_MIN_KV; b1 kv <= 16K stays LDG-only).
# Points: b8 kv sweep, iso total KV 240K (b1..b16), b16 / b32 kv 4K..8K, b1 kv32K, R3-b32-kv4K. Submit from the worktree root.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1 PYTHONPATH=$REPO
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD) (+ uncommitted SKV_TC wiring)"
nvidia-smi -L
W=${POINTS:-mk_b8kv,mk_iso240,R9-b16-kv4096,R9-b16-kv8192,R9-b32-kv4096,R9-b32-kv7168,R2-b1-kv32768,R3-b32-kv4096}
timeout 2400 "$PY" probe/mk_core/debug/ab.py "$W" '[{"skvtc": 0}, {"skvtc": 1}]' 7; echo "exit $?"
