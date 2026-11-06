#!/bin/bash
#SBATCH --job-name=kvq-cal
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=probe/mk_core/kvq/logs/slurm-%j.log
# calibration diagnosis: a1ub `ld` build with L2 evict_first on / off, and kvq static with the same two settings (same GPU, same job)
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1 PYTHONPATH=$REPO
nvidia-smi -L
J=${SLURM_JOB_ID:-local}; D=probe/mk_core/kvq; O=results/2026-10-28-kvq; mkdir -p $O
for X in "--swiz 0" "--swiz 0 --define KVQ_EF=0" "--swiz 1"; do
  echo "== kvq $X"
  timeout 120 "$PY" $D/kvq.py --points b8kv32768 --arms static --launches 20 --check $X --job $J --out $O/kvq_${J}_calib.jsonl 2>&1 | grep "^b8\|CALIB\|CHECK"
done
