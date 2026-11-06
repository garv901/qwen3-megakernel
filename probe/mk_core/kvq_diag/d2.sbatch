#!/bin/bash
#SBATCH --job-name=kvqd-d2
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=probe/mk_core/kvq_diag/logs/slurm-%j.log
# D2: C(S) with forced S (arm fs: S equal pieces per pair, static round-robin), combine_head vs combine_wide (k slices per row), kMergeChunk 20 / 32, partial fp32 / bf16.
source "${SLURM_SUBMIT_DIR:-$PWD}/probe/mk_core/kvq_diag/common.sh"
P=b8kv32768,b1kv245760,ragged64,b8kv4096
S=1,2,4,8,16,32,64
echo "== check head / wide, PBF"
timeout 200 "$PY" $D/kvq.py --points b8kv32768,ragged64 --arms fs --chunks 3,32,64 --check-only --comb-mode head,wide --wide-k 1,2,4 --job $J 2>&1 | grep "^CHECK\|done" | cut -c1-200
timeout 100 "$PY" $D/kvq.py --points b8kv32768 --arms fs --chunks 3,32,64 --check-only --define KVQ_PBF=1 --comb-mode head,wide --wide-k 1,2 --job $J 2>&1 | grep "^CHECK\|done" | cut -c1-200
echo "== C(S): head + wide k 1 2 4 8 (fp32 partials, MK 20)"
timeout 480 "$PY" $D/kvq.py --points $P --arms fs --chunks $S --launches 15 --comb-mode head,wide --wide-k 1,2,4,8 --job $J --out $O/d2_${J}_fs.jsonl 2>&1 | grep "^b\|^rag\|done"
echo "== MK 32"
timeout 240 "$PY" $D/kvq.py --points $P --arms fs --chunks $S --launches 15 --comb-mode head,wide --wide-k 1,2 --define KVQ_MK=32 --job $J --out $O/d2_${J}_mk32.jsonl 2>&1 | grep "^b\|^rag\|done\|ptxas:"
echo "== bf16 partials"
timeout 240 "$PY" $D/kvq.py --points $P --arms fs --chunks $S --launches 15 --comb-mode head,wide --wide-k 1,2 --define KVQ_PBF=1 --job $J --out $O/d2_${J}_pbf.jsonl 2>&1 | grep "^b\|^rag\|done"
