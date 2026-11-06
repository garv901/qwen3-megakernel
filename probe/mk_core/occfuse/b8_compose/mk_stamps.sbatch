#!/bin/bash
#SBATCH --job-name=occf-b8mk
#SBATCH --gres=gpu:1
#SBATCH --output=runs/slurm-%j.log
# B8: mk per-CTA stamps at T 128 / 256: C = classic (MK_TS 0), T = tile stream (MK_TS 128, default) and their MK_NULL 0x39 versions. No e2e A/B.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1 PYTHONPATH=$REPO
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi -L
OUT=results/2026-10-29-occfuse/b8_compose/mk_stamps_$SLURM_JOB_ID
WL=R1-b128-kv128,R1-b256-kv128
for S in '{"name":"T","MK_TS":128}' '{"name":"C","MK_TS":0}' '{"name":"T_null","MK_TS":128,"null":57}' '{"name":"C_null","MK_TS":0,"null":57}'; do
  echo "== $S"
  timeout 420 "$PY" probe/mk_core/smallt/dump.py "$OUT" "$WL" "[$S]" 5 || echo "!! $S failed or hung (exit $?)"
done
"$PY" probe/mk_core/smallt/analyze.py "$OUT" > "$OUT/analyze.txt" 2>&1; cat "$OUT/analyze.txt"
