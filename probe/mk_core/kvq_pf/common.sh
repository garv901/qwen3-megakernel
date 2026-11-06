# sourced by the kvq_diag sbatch scripts (cwd = repo root). Needs GATE_JOB (the gate job whose log must say gate: '1').
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1 PYTHONPATH=$REPO
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi -L
J=${SLURM_JOB_ID:-local}; D=probe/mk_core/kvq_diag; O=results/2026-10-28-kvq-diag; mkdir -p $O
if [ -n "${GATE_JOB:-}" ] && ! grep -q "^gate: '1'" $D/logs/slurm-${GATE_JOB}.log; then echo "calibration gate of job $GATE_JOB did not pass: stop"; exit 0; fi
declare -A VDEF=([e0]="KVQ_EOI=0" [e1]="KVQ_EOI=1" [e2]="KVQ_EOI=2" [e3]="KVQ_EOI=3" [e4_544]="KVQ_EOI=3 KVQ_PS=136" [e4_576]="KVQ_EOI=3 KVQ_PS=144" [e4_1024]="KVQ_EOI=3 KVQ_PS=256" [e5]="KVQ_EOI=5" [e5_576]="KVQ_EOI=5 KVQ_PS=144")
defs() { local o=""; for x in ${VDEF[$1]}; do o="$o --define $x"; done; echo $o; }
