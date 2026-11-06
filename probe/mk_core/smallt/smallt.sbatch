#!/bin/bash
#SBATCH --job-name=mkc-smallt
#SBATCH --gres=gpu:1
#SBATCH --output=runs/slurm-%j.log
# Small-T study (NPAD 16 / 32 / 64): classic (MK_TS 128) vs tile stream (MK_TS 16) vs QKV-only tile stream (MK_TS 16, MK_TSB 0).
# 1. per-CTA stamps (full and MK_NULL 0x39 = QKV / O / GU / D null: weight TMAs + ring waits + barriers only), placeholder attention
# 2. e2e A/B, 7 rounds, placeholder attention
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1 PYTHONPATH=$REPO
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi -L
OUT=results/2026-10-29-smallt/st_$SLURM_JOB_ID
WL=R1-b1-kv128,R1-b8-kv128,R1-b32-kv128,R1-b64-kv128
for S in '{"name":"C","MK_TS":128}' '{"name":"C_null","MK_TS":128,"null":57}' \
         '{"name":"T","MK_TS":16}' '{"name":"T_null","MK_TS":16,"null":57}' \
         '{"name":"Tq","MK_TS":16,"MK_TSB":0}'; do
  echo "== $S"
  timeout 420 "$PY" probe/mk_core/smallt/dump.py "$OUT" "$WL" "[$S]" 5 || echo "!! $S failed or hung (exit $?)"
done
"$PY" probe/mk_core/smallt/analyze.py "$OUT"
echo "### A/B"
timeout 900 "$PY" probe/mk_core/debug/ab.py "$WL" '[{"MK_TS":128,"attn":"placeholder"},{"MK_TS":16,"attn":"placeholder"},{"MK_TS":16,"MK_TSB":0,"attn":"placeholder"}]' 7 || echo "!! A/B exit $?"
