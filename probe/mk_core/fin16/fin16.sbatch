#!/bin/bash
#SBATCH --job-name=mkc-fin16
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:35:00
#SBATCH --output=runs/slurm-%j.log
# Gap check of the P15 causes at NPAD 16 (tile stream MK_TS 16):
#  1. finalizer mode: MK_FIN 0 (P15 default below NPAD 128) vs 1 vs 2, stamps + e2e A/B (7 rounds) at R1-b1 / b8-kv128 (placeholder attention)
#  2. clock: mk SM clock (NVML) and the bench a0 / ts1w clock (clock64 / globaltimer) in the same job, T 8 / 16, 2 rounds
# Each stamps run has a timeout: a hang in a new MK_FIN mode drops that set from the A/B instead of blocking the job.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1 PYTHONPATH=$REPO
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi -L

SETS=('{"MK_TS":128}' '{"MK_TS":16}' '{"MK_TS":16,"MK_FIN":1}' '{"MK_TS":16,"MK_FIN":2}')
OK=()
echo "### stamps R1-b8-kv128"
for K in "${SETS[@]}"; do
  echo "== knobs $K"
  if timeout 300 "$PY" probe/mk_core/debug/stamps_spread.py R1-b8-kv128 0 "$K"; then OK+=("$K"); else echo "!! knobs $K failed or hung (exit $?)"; fi
done
LIST=$(IFS=,; echo "[${OK[*]}]")
echo "### A/B $LIST"
timeout 1200 "$PY" probe/mk_core/debug/ab.py R1-b1-kv128,R1-b8-kv128 "$LIST" 7 || echo "!! A/B exit $?"

echo "### mk clocks"
timeout 300 "$PY" probe/mk_core/debug/clocks.py R1-b8-kv128 '[{"MK_TS":128},{"MK_TS":16}]' || echo "!! clocks exit $?"

echo "### bench (xover4_small arms a0 / ts1w)"
cd "$REPO/probe/mk_core/xover4"
bash build_small.sh || exit 1
for round in 1 2; do
  echo "## round $round"
  for s in QKV O GU D; do for t in 8 16; do
    echo "## $s $t"
    ./bin/a0 0 5 $s $t
    ./bin/ts1w 0 5 $s $t
  done; done
done
