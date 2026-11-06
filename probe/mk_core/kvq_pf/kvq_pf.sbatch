#!/bin/bash
#SBATCH --job-name=kvq-pf
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=probe/mk_core/kvq_pf/logs/slurm-%j.log
# kvq_pf: record prefetch (KVQ_RPF) x R1 accumulation (KVQ_ACC). Submit from the repo root:  MODE=<m> [VARS="a0r0 a0r1 .."] [STAGE=calib] sbatch probe/mk_core/kvq_pf/kvq_pf.sbatch
#  1. calibration gate (a1ub --comb head vs kvq static C, kvq static A at b8kv32768 vs 154.6 us +-3%, check PASS). Gate fails -> no sweep.
#  2. MODE sweep: per variant v in VARS (a<ACC>r<RPF>[d<RD>][l<RLD>], e.g. a3r1 = ACC 3 RPF 1): 10 points x static / queue / queue_pf, C 512 1024, caps 32 / none, pos, --check, plus C 256 at b8kv32768 / ragged64
#     MODE tune: RPF / RD / RLD variants on b8kv32768, ragged64, b1kv245760 (queue_pf, static), ACC 0
#     MODE tl: KVQ_TL=1 builds (clock64 timeline), static + queue_pf at b8kv32768 / ragged64, C 256 512 1024, cap 32 / none, per variant
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1 PYTHONPATH=$REPO
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD) MODE=${MODE:-?} VARS=${VARS:-}"
nvidia-smi -L
J=${SLURM_JOB_ID:-local}
D=probe/mk_core/kvq_pf
O=results/2026-10-28-kvq-pf; mkdir -p $O
defs() {   # a3r1d2l1 -> --define KVQ_ACC=3 --define KVQ_RPF=1 --define KVQ_RD=2 --define KVQ_RLD=1
  local v=$1 o=""
  [[ $v =~ a([0-9]) ]] && o="$o --define KVQ_ACC=${BASH_REMATCH[1]}"
  [[ $v =~ r([0-9]) ]] && o="$o --define KVQ_RPF=${BASH_REMATCH[1]}"
  [[ $v =~ d([0-9]) ]] && o="$o --define KVQ_RD=${BASH_REMATCH[1]}"
  [[ $v =~ l([0-9]) ]] && o="$o --define KVQ_RLD=${BASH_REMATCH[1]}"
  echo $o
}
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
ok &= abs(kv["b8kv32768"]["A_med"] / 154.6 - 1) <= 0.03
print("1" if ok else "0")
PY
)
echo "gate: '$GATE'"
if [ "$GATE" != 1 ] || [ "${STAGE:-sweep}" = calib ]; then echo "no sweep"; exit 0; fi
P=${POINTS:-b8kv4096,b8kv16384,b8kv32768,b8kv30720,b1kv245760,b1kv32768,b64kv1024,b128kv512,ragged8,ragged64}
case "${MODE:-sweep}" in
sweep)
  for v in ${VARS:-a0r0}; do
    echo "== sweep variant $v: $(defs $v)"
    OUT=$O/sw_${J}_$v.jsonl
    timeout 380 "$PY" $D/kvq.py --points $P --arms static,queue,queue_pf --chunks 512,1024 --caps 32,0 --orders pos --check --launches 10 $(defs $v) --job $J --out $OUT 2>&1 | grep "^CHECK.*FAIL\|^b\|^rag\|done\|rror"
    timeout 150 "$PY" $D/kvq.py --points b8kv32768,ragged64 --arms queue,queue_pf --chunks 256 --caps 32,0 --orders pos --check --launches 10 $(defs $v) --job $J --out $OUT 2>&1 | grep "^CHECK.*FAIL\|^b\|^rag\|done\|rror"
  done ;;
tune)
  for v in ${VARS:-a0r0 a0r1 a0r3}; do
    echo "== tune variant $v: $(defs $v)"
    timeout 200 "$PY" $D/kvq.py --points b8kv32768,ragged64,b1kv245760 --arms static,queue_pf --chunks 256,512,1024 --caps 32,0 --orders pos --check --launches 10 $(defs $v) --job $J --out $O/tune_${J}_$v.jsonl 2>&1 | grep "^CHECK.*FAIL\|^b\|^rag\|done\|rror"
  done ;;
tl)
  for v in ${VARS:-a0r0 a0r1 a0r3}; do
    echo "== timeline variant $v: $(defs $v)"
    timeout 240 "$PY" $D/kvq.py --points b8kv32768,ragged64 --arms static,queue_pf --chunks 256,512,1024 --caps 32,0 --orders pos --launches 5 $(defs $v) --define KVQ_TL=1 --tl-out $O/tl_${J}_$v --job $J --out $O/tl_${J}_$v.jsonl 2>&1 | grep "^b\|^rag\|done\|rror"
  done ;;
esac
