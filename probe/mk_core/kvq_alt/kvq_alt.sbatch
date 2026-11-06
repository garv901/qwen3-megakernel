#!/bin/bash
#SBATCH --job-name=kvq-alt
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=probe/mk_core/kvq_alt/logs/slurm-%j.log
# kvq_alt sweeps. Submit from the repo root: MODE=<m> sbatch probe/mk_core/kvq_alt/kvq_alt.sbatch   (MODE: acc1 | acc2 | acc3 | acc4 | base | tail | mq | mqw | tail2)
#  1. calibration gate (same as kvq_comb.sbatch: a1ub --comb head vs kvq static C, kvq static A at b8kv32768 vs 154.3 us +- 3%). Gate fails -> no sweep.
#  2. acc<N>: R1 (KVQ_ACC=N) static / queue_pf / queue (C 256,512,1024 x caps 32,64,none) + hybrid, pos, --check;  base: the same with KVQ_ACC=0 (same-job control)
#     tail: R2 arm tail, f 5/10/20 % x k 2/4/8 (slots + combine);  mq: R3 tick 2 (queue_pf / queue, cap 32, pos and pair; static)
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1 PYTHONPATH=$REPO
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD) MODE=${MODE:-?}"
nvidia-smi -L
J=${SLURM_JOB_ID:-local}
D=probe/mk_core/kvq_alt
O=results/2026-10-28-kvq-alt; mkdir -p $O
echo "== 1. calibration"
timeout 300 "$PY" probe/mk_core/attn_ub/a1ub.py --time --comb head --points b8kv32768,b1kv245760 --engine tma --build mk=SKV_KV_EVICT_FIRST=1,SKV_MERGE_CHUNK=20,SKV_TC=1 --no-default --launches 10 --job $J --out $O/kvq_${J}_a1ub.jsonl 2>&1 | grep -v "^ptxas\|warning\|^$"; echo "exit $?"
timeout 240 "$PY" $D/kvq.py --points b8kv32768,b1kv245760 --arms static --check --launches 20 --job $J --out $O/kvq_${J}_calib.jsonl 2>&1 | grep "^CHECK\|^b\|CALIB\|^ptxas:"
GATE=$("$PY" - "$O/kvq_${J}_a1ub.jsonl" "$O/kvq_${J}_calib.jsonl" <<'PY'
import json, sys
a1 = {json.loads(l)["point"]: json.loads(l) for l in open(sys.argv[1]) if '"kind": "time"' in l}
kv = {json.loads(l)["point"]: json.loads(l) for l in open(sys.argv[2]) if '"A_med"' in l}
ok = True
for p in ("b8kv32768", "b1kv245760"):
    c1, c2 = a1[p]["C_med"], kv[p]["C_med"]
    g = abs(c2 - c1) <= max(0.1 * c1, 0.5)
    print(f"# {p}: a1ub head C {c1:.2f}  kvq static C {c2:.2f}  {'ok' if g else 'MISMATCH'}", file=sys.stderr)
    ok &= g and kv[p].get("check_status") == "PASS"
ok &= abs(kv["b8kv32768"]["A_med"] / 154.3 - 1) <= 0.03
print("1" if ok else "0")
PY
)
echo "gate: '$GATE'"
if [ "$GATE" != 1 ] || [ "${STAGE:-sweep}" = calib ]; then echo "no sweep"; exit 0; fi
P=${POINTS:-b8kv4096,b8kv16384,b8kv32768,b8kv30720,b1kv245760,b1kv32768,b64kv1024,b128kv512,ragged8,ragged64}
echo "== 2. sweep $MODE"
F="--points $P --orders pos --check --launches 10 --job $J"
case "$MODE" in
  acc[1-4]|base)
    N=${MODE#acc}; [ "$MODE" = base ] && N=0
    OUT=$O/kvq_${J}_$MODE.jsonl
    timeout 540 "$PY" $D/kvq.py $F --define KVQ_ACC=$N --arms static,queue_pf,queue --chunks 256,512,1024 --caps 32,64,0 --out $OUT 2>&1 | grep "^CHECK.*FAIL\|^b\|^rag\|done"
    timeout 330 "$PY" $D/kvq.py $F --define KVQ_ACC=$N --arms hybrid --chunks 256,512,1024,2048 --caps 32,64,0 --out $OUT 2>&1 | grep "^CHECK.*FAIL\|^b\|^rag\|done" ;;
  tail)
    timeout 700 "$PY" $D/kvq.py $F --arms tail --tail-f 0.05,0.1,0.2 --tail-k 2,4,8 --out $O/kvq_${J}_tail.jsonl 2>&1 | grep "^CHECK.*FAIL\|^b\|^rag\|done" ;;
  mqw)   # R3 waiting stats (--units-rec adds globaltimer reads: A is not comparable with mq)
    timeout 400 "$PY" $D/kvq.py --points b8kv16384,b8kv32768,b1kv245760,ragged8,ragged64 --orders pos,pair --check --launches 10 --job $J --units-rec --arms queue_pf --chunks 256,1024 --caps 32 --tick 2 --out $O/kvq_${J}_mqw.jsonl 2>&1 | grep "^CHECK.*FAIL\|^b\|^rag\|MQ\|done" ;;
  tail2)  # R2 with more queue chunks per pair (S <= S_static + 16 / 32: outside the plan's S <= 8)
    timeout 400 "$PY" $D/kvq.py --points b8kv16384,b8kv32768,b1kv245760,b1kv32768,ragged8,ragged64 --orders pos --check --launches 10 --job $J --arms tail --tail-f 0.05,0.1,0.2 --tail-k 16,32 --out $O/kvq_${J}_tail2.jsonl 2>&1 | grep "^CHECK.*FAIL\|^b\|^rag\|done" ;;
  mq)
    timeout 600 "$PY" $D/kvq.py $F --arms static,queue_pf,queue --chunks 256,512,1024 --orders pos,pair --caps 32 --tick 2 --out $O/kvq_${J}_mq.jsonl 2>&1 | grep "^CHECK.*FAIL\|^b\|^rag\|done" ;;
esac
