#!/bin/bash
#SBATCH --job-name=kvqd-gate
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=probe/mk_core/kvq_diag/logs/slurm-%j.log
# Calibration gate of the kvq_diag build (same as job 11854's gate): a1ub --comb head vs kvq static C at b8kv32768 / b1kv245760, static A at b8kv32768 vs 154.3 us (+-3%). Submit from the repo root.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1 PYTHONPATH=$REPO
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi -L
J=${SLURM_JOB_ID:-local}
D=probe/mk_core/kvq
O=results/2026-10-28-kvq-diag; mkdir -p $O
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
