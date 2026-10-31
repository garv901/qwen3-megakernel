#!/bin/bash
#SBATCH --job-name=mkr10-bd
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=runs/slurm-%j.log
# R10 per-phase breakdown: mk phase stamps (MK_STAMPS, stamps_spread.py: work / spread / lat per phase, layers 4..23)
# and the fi-graph graph-pdl-best kernel timeline (gpu_check_fi.py --timeline) at representative mk_r10 points.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export PYTHONPATH=$REPO
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi -L
PTS=${PTS:-R10-b48-kv4096,R10-b64-kv2048,R10-b96-kv3072,R10-b128-kv2816,R10-b256-kv1024,R10-b256-kv1408,R10-conv-L3-n54,R10-conv-L4-n87}
echo "### mk stamps"
for p in ${PTS//,/ }; do timeout 600 "$PY" probe/mk_core/debug/stamps_spread.py "$p" 0 || echo "!! stamps $p exit $?"; done
echo "### fi timeline"
timeout 1800 "$PY" scripts/gpu_check_fi.py --workloads "$PTS" --variants graph-pdl-best --timeline || echo "!! fi timeline exit $?"
