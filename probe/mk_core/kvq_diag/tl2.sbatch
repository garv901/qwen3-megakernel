#!/bin/bash
#SBATCH --job-name=kvqd-tl2
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=probe/mk_core/kvq_diag/logs/slurm-%j.log
# D1 timeline at cap none (units of 2 / 4 / 8 stages): queue_pf at b8kv32768 and ragged64, C 256 512 1024, EOI 0 / 1 / 3 (KVQ_TL=1).
source "${SLURM_SUBMIT_DIR:-$PWD}/probe/mk_core/kvq_diag/common.sh"
for v in e3 e0 e1; do
  echo "== variant $v"
  timeout 300 "$PY" $D/kvq.py --points b8kv32768,ragged64 --arms queue_pf --chunks 256,512,1024 --orders pos --caps 0 --launches 5 $(defs $v) --define KVQ_TL=1 --tl-out $O/tl2_${J}_$v --job $J --out $O/tl2_${J}_$v.jsonl 2>&1 | grep "^b\|^rag\|done\|rror"
done
