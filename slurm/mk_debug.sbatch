#!/bin/bash
#SBATCH --job-name=mkc-dbg
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=runs/slurm-%j.log
# MK-core v0 crash bisection: null-mask launches (one process each), then compute-sanitizer memcheck of the full build.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1 PYTHONPATH=$REPO
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi -L
W=${W:-R0-b1-kv128}
for m in 0x7f 0x40 0x3f 0x7e 0x7d 0x7b 0x77 0x6f 0x5f 0x0; do   # all null; all but LM; only LM; then one real phase each (1..6)
  echo "--- null $m"; timeout 300 "$PY" probe/mk_core/debug/v0_one.py "$W" $m 2>&1 | grep -E '^OK|Error|error' | head -3
done
echo "--- memcheck full build"
MK_LINEINFO=1 timeout 600 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --show-backtrace device --print-limit 20 \
  "$PY" probe/mk_core/debug/v0_one.py "$W" 0 2>&1 | head -120
