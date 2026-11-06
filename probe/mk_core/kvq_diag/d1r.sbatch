#!/bin/bash
#SBATCH --job-name=kvqd-d1r
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=probe/mk_core/kvq_diag/logs/slurm-%j.log
# D1 replicate at the decomposition configs only: all variants, static + queue / queue_pf, C 512 + 1024 cap 32 (pos), 3 points, 40 launches x 3 layers. Env: VARS (default all).
source "${SLURM_SUBMIT_DIR:-$PWD}/probe/mk_core/kvq_diag/common.sh"
P=b8kv32768,b1kv245760,ragged64
for v in ${VARS:-e0 e1 e2 e3 e4_544 e4_576 e4_1024 e5 e5_576}; do
  echo "== variant $v: ${VDEF[$v]}"
  timeout 200 "$PY" $D/kvq.py --points $P --arms static,queue,queue_pf --chunks 512,1024 --orders pos --caps 32 --launches 40 $(defs $v) --job $J --out $O/d1r_${J}_$v.jsonl 2>&1 | grep "^b\|^rag\|done"
done
