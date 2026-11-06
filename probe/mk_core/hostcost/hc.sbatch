#!/bin/bash
#SBATCH --job-name=mk-hostcost
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/raid/garv901/regime-mk-core/runs/slurm-%j.log
# Where mk's host time goes in vLLM serving, and what skipping vLLM's unused attention-metadata build buys.
# Harness: worktree regime-mk-serve-host (branch mk-serve-host = mk-serve fe5725b + per-part host timers in the step log:
# pre_us = execute_model entry -> forward, meta_us = vLLM attention-metadata build, decide_us = step description +
# dispatch, parts = mk run() split {wait, plan, copy, replay}). Same mk build as the M2 grid.
# Per stage: arms stock,mk (metadata built, as in M2), then mk with --skip-attn-meta (tag suffix _skip).
#   T_L1 / T_L2: M2 Azure conv median-window trace at 10 / 40 req/s.  C_c1 / C_c8: random 128 in / 256 out, closed loop.
# Submit: sbatch --export=ALL,STAGES="T_L1 T_L2" probe/mk_core/hostcost/hc.sbatch
set -uo pipefail
export REPO=/raid/garv901/regime-mk-serve-host
source "$REPO/slurm/env.sh"
cd "$REPO"
OUT=/raid/garv901/regime-mk-core/runs/hostcost
R=/raid/garv901/regime-mk-serve/runs/e2e/replay/m2
mkdir -p "$OUT"
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD) dirty=$(git status --short | wc -l) OUT=$OUT"
nvidia-smi -L
sb() {  # tag arms [args...]
  local tag=$1 arms=$2; shift 2
  echo "=== $tag (arms=$arms) $(date -u +%T)"
  "$PY" scripts/serve_bench.py --arms "$arms" --warmup-trace "$R/conv_warmup.jsonl" --out-root "$OUT" --tag "$tag" "$@"
  echo "exit $? $tag"
}
for s in ${STAGES:-T_L1 T_L2 C_c1 C_c8}; do
  case $s in
    T_L*) l=${s#T_}
      a=(--trace "$R/conv_median_$l.jsonl" --meta trace=conv --meta window=median --meta load=$l)
      sb "$s" stock,mk "${a[@]}"
      sb "${s}_skip" mk "${a[@]}" --skip-attn-meta --meta skip_meta=1 ;;
    C_c*) c=${s#C_c}; n=$([ "$c" = 1 ] && echo 96 || echo 256)
      a=(--random --input-len 128 --output-len 256 --num-prompts "$n" --request-rate inf
         --bench-arg=--max-concurrency --bench-arg="$c" --meta conc="$c")
      sb "$s" stock,mk "${a[@]}"
      sb "${s}_skip" mk "${a[@]}" --skip-attn-meta --meta skip_meta=1 ;;
    *) echo "unknown stage $s" ;;
  esac
done
echo "done $(date -u +%T)"
