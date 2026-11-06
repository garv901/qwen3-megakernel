#!/bin/bash
#SBATCH --job-name=kvq
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=probe/mk_core/kvq/logs/slurm-%j.log
# kvq: work-assignment arms on mk's real KV access pattern. Submit from the repo root: sbatch probe/mk_core/kvq/kvq.sbatch
#  1. calibration gate: static @ b8kv32768 (--check first), swiz 0 (a1ub `ld` build) and swiz 1: A within 3% of 156.864 us, imb within 0.8 us of 4.352 (3 timer steps; one step is 6% of imb) (globaltimer step 0.256 us). Fails -> no sweep.
#  2. sweep (swiz 0 ring): --check then timing of every point x arm x C x order
#  3. --units-rec runs (per-unit gaps) at b8kv32768 / b64kv1024
#  4. swizzled-ring subset (swiz 1)
# env: STAGE=calib stops after 1; POINTS, ARMS, CHUNKS override the sweep set.
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
for S in 0 1; do
  timeout 240 "$PY" $D/kvq.py --points b8kv32768 --arms static --swiz $S --check --launches 20 --job $J --out $O/kvq_${J}_calib.jsonl; echo "exit $?"
done
GATE=$("$PY" - "$O/kvq_${J}_calib.jsonl" <<'PY'
import json, sys
ok = {}
for l in open(sys.argv[1]):
    r = json.loads(l)
    if "A_med" not in r: continue
    ok[r["swiz"]] = r.get("check_status") == "PASS" and abs(r["A_med"] / 156.864 - 1) <= 0.03 and abs(r["imb_med"] - 4.352) <= 0.8
print("".join(str(s) for s in sorted(ok) if ok[s]))
PY
)
echo "gate pass for swiz builds: '$GATE'"
if [ "${STAGE:-sweep}" = calib ] || [[ "$GATE" != *0* ]]; then echo "no sweep (STAGE=calib or swiz-0 calibration gate failed)"; exit 0; fi
P=${POINTS:-b8kv4096,b8kv16384,b8kv32768,b8kv30720,b1kv245760,b1kv32768,b64kv1024,b128kv512,ragged8,ragged64}
A=${ARMS:-static,static_il,queue,queue_pf,hybrid}
CH=${CHUNKS:-128,256,512,1024,2048}
echo "== 2. sweep swiz=0 (the calibrated a1ub ld ring)"
timeout 650 "$PY" $D/kvq.py --points $P --arms $A --chunks $CH --swiz 0 --check --launches 10 --job $J --out $O/kvq_${J}.jsonl; echo "exit $?"
echo "== 3. unit overhead (--units-rec)"
timeout 80 "$PY" $D/kvq.py --points b8kv32768,b64kv1024 --arms static,queue,queue_pf --chunks 256,1024 --orders pos --swiz 0 --units-rec --launches 6 --job $J --out $O/kvq_${J}_rec.jsonl; echo "exit $?"
echo "== 4. swizzled ring (4 TMA ops per stage, a1ub mk access shape) on a subset"
timeout 200 "$PY" $D/kvq.py --points b8kv32768,b1kv245760,b64kv1024,ragged64 --arms $A --chunks 256,1024 --orders pos --swiz 1 --check --launches 8 --job $J --out $O/kvq_${J}_swiz1.jsonl; echo "exit $?"
