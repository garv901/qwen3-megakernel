#!/bin/bash
#SBATCH --job-name=kvqd-d1
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=probe/mk_core/kvq_diag/logs/slurm-%j.log
# D1: A per end-of-item variant. Env: VARS="e0 e1 .." (see common.sh), GATE_JOB. Per variant: static + queue / queue_pf, C 256 512 1024, caps 32 / none, pos, points b8kv32768 b1kv245760 ragged64.
# Check (coverage, XOR, and for EOI >= 3 partials + merge) on the C512 cap32 subset first, then the timed sweep (15 launches x 3 layers).
source "${SLURM_SUBMIT_DIR:-$PWD}/probe/mk_core/kvq_diag/common.sh"
P=b8kv32768,b1kv245760,ragged64
for v in ${VARS:-e0 e1 e2 e3}; do
  echo "== variant $v: ${VDEF[$v]}"
  timeout 150 "$PY" $D/kvq.py --points $P --arms static,queue,queue_pf --chunks 1024 --orders pos --caps 32 --check-only $(defs $v) --job $J 2>&1 | grep "^CHECK\|^done\|^ptxas:" | cut -c1-200
  timeout 420 "$PY" $D/kvq.py --points $P --arms static,queue,queue_pf --chunks 256,512,1024 --orders pos --caps 32,0 --launches 15 $(defs $v) --job $J --out $O/d1_${J}_$v.jsonl 2>&1 | grep "^b\|^rag\|done"
done
