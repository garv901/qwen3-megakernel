#!/bin/bash
#SBATCH --job-name=mkc-dbg
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=runs/slurm-%j.log
# MK-core v0 crash bisection: early-exit stages (MK_DBG_EXIT) and memcheck of the all-null and full builds.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1 PYTHONPATH=$REPO
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi -L
W=${W:-R0-b1-kv128}
for e in 1 2; do
  echo "--- dbg_exit $e (null 0x7f)"; timeout 300 "$PY" probe/mk_core/debug/v0_one.py "$W" 0x7f $e 2>&1 | grep -E '^OK|Error' | head -2
done
for m in 0x7f 0x0; do
  echo "--- memcheck null $m"
  MK_LINEINFO=1 timeout 600 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --report-api-errors no --show-backtrace device \
    --print-limit 30 "$PY" probe/mk_core/debug/v0_one.py "$W" $m 2>&1 | grep -v '^=========\s*$' | head -80
done
