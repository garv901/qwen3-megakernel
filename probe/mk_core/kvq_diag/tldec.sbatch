#!/bin/bash
#SBATCH --job-name=kvqd-tldec
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=probe/mk_core/kvq_diag/logs/slurm-%j.log
# D1 timeline: KVQ_TL=1 builds (clock64 per warp, last layer, last launch), EOI 3 / 0 / 1 / 5, static + queue + queue_pf at b8kv32768 and b1kv245760 (cap 32). Also A with TL on, to see its perturbation.
source "${SLURM_SUBMIT_DIR:-$PWD}/probe/mk_core/kvq_diag/common.sh"
for v in e3 e0 e1 e5; do
  echo "== variant $v"
  timeout 240 "$PY" $D/kvq.py --points b8kv32768,b1kv245760 --arms static,queue,queue_pf --chunks 256,512 --orders pos --caps 32 --launches 5 $(defs $v) --define KVQ_TL=1 --tl-out $O/tl_${J}_$v --job $J --out $O/tl_${J}_$v.jsonl 2>&1 | grep "^b\|done\|rror"
done
# Decomposition inputs: head vs wide (k 1 2 4 8) combine on the real static / queue / queue_pf configs of the three points (EOI 3, PS 132 = mk today), same job so same GPU.
P=b8kv32768,b1kv245760,ragged64
echo "== decomp: head vs wide on real configs"
timeout 500 "$PY" $D/kvq.py --points $P --arms static,queue,queue_pf --chunks 512,1024 --orders pos --caps 32 --launches 15 --comb-mode head,wide --wide-k 1,2,4,8 --job $J --out $O/decomp_${J}.jsonl 2>&1 | grep "^b\|^rag\|done"
