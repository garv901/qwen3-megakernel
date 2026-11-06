#!/bin/bash
#SBATCH --job-name=kvq-comb
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=probe/mk_core/kvq/logs/slurm-%j.log
# kvq + mk's combine (partials in mk's layout, combine_head after the attend barrier, SKV_TICKET variant). Submit from the repo root: sbatch probe/mk_core/kvq/kvq_comb.sbatch
#  1. calibration (same GPU, same job): a1ub --comb head (mk build knobs) vs kvq static C at b8kv32768 / b1kv245760; A at b8kv32768 vs 154.3 us. Gate fails -> no sweep.
#  2. sweep: 10 points x static / queue_pf / queue (C 256,512,1024) / hybrid (C 256..2048) x cap {32, 64, none}, pos order, --check on every config
#  3. pair-order control at b8kv32768, 4. ticket variant (queue_pf, cap 32; static)
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1 PYTHONPATH=$REPO
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi -L
J=${SLURM_JOB_ID:-local}
D=probe/mk_core/kvq
O=results/2026-10-28-kvq; mkdir -p $O
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
echo "== 2. sweep"
timeout 420 "$PY" $D/kvq.py --points $P --arms static,queue_pf,queue --chunks 256,512,1024 --orders pos --caps 32,64,0 --check --launches 10 --job $J --out $O/kvq_${J}.jsonl 2>&1 | grep "^CHECK.*FAIL\|^b\|^rag\|done"
timeout 240 "$PY" $D/kvq.py --points $P --arms hybrid --chunks 256,512,1024,2048 --orders pos --caps 32,64,0 --check --launches 10 --job $J --out $O/kvq_${J}.jsonl 2>&1 | grep "^CHECK.*FAIL\|^b\|^rag\|done"
echo "== 3. pair-order control"
timeout 60 "$PY" $D/kvq.py --points b8kv32768 --arms queue_pf --chunks 256 --orders pair --caps 32,64,0 --check --launches 10 --job $J --out $O/kvq_${J}.jsonl 2>&1 | grep "^CHECK.*FAIL\|^b\|done"
echo "== 4. ticket variant"
timeout 200 "$PY" $D/kvq.py --points $P --arms static,queue_pf --chunks 256,512,1024 --orders pos --caps 32 --tick 1 --check --launches 10 --job $J --out $O/kvq_${J}_ticket.jsonl 2>&1 | grep "^CHECK.*FAIL\|^b\|^rag\|done"
