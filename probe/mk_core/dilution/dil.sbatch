#!/bin/bash
#SBATCH --job-name=mk-dilution
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/raid/garv901/regime-mk-core/runs/slurm-%j.log
# Why the mk gain dilutes in vLLM serving. Controlled closed-loop decode (random prompts, input 128, output 256,
# ignore-eos) through the mk-serve harness (regime-mk-serve: vLLM 0.31 + mkserve plugin, per-step log with
# fwd_us = model forward on the GPU, gap_us = GPU time between forwards, host_us = host time in the hooked call).
#   A: arms stock, fi, mk at concurrency 1 / 8 / 32 / 64 (mk covers <= 32 requests; 64 falls back to stock)
#   B: stock, mk at concurrency 1 with --no-async-scheduling (how much host time async scheduling hides)
# Every arm replays the M2 conv warmup trace (unmeasured) first. Analysis: probe/mk_core/dilution/analyze.py.
# Submit from anywhere: sbatch probe/mk_core/dilution/dil.sbatch
set -uo pipefail
export REPO=/raid/garv901/regime-mk-serve
source "$REPO/slurm/env.sh"
cd "$REPO"
OUT=/raid/garv901/regime-mk-core/runs/dilution
mkdir -p "$OUT"
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD) OUT=$OUT"
nvidia-smi -L
WARM=runs/e2e/replay/m2/conv_warmup.jsonl
run() {  # tag concurrency prompts arms [extra args...]
  local tag=$1 c=$2 n=$3 arms=$4; shift 4
  echo "=== $tag (C=$c, prompts=$n, arms=$arms) $(date -u +%T)"
  "$PY" scripts/serve_bench.py --arms "$arms" --random --input-len 128 --output-len 256 --num-prompts "$n" \
    --request-rate inf --bench-arg=--max-concurrency --bench-arg="$c" --warmup-trace "$WARM" \
    --out-root "$OUT" --tag "$tag" --meta stage="${tag%%_*}" --meta conc="$c" "$@"
  echo "exit $? $tag"
}
# Each arm costs ~9 min (server start + warmup replay ~7.5 min, bench ~1 min): split stages over jobs with
# STAGES, e.g. sbatch --export=ALL,STAGES="A_c64 B_c1sync" --time=01:05:00 probe/mk_core/dilution/dil.sbatch
for s in ${STAGES:-A_c1 A_c8 A_c32 A_c64 B_c1sync}; do
  case $s in
    A_c1) run A_c1 1 96 stock,fi,mk ;;
    A_c8) run A_c8 8 256 stock,fi,mk ;;
    A_c32) run A_c32 32 512 stock,fi,mk ;;
    A_c64) run A_c64 64 512 stock,fi,mk ;;
    B_c1sync) run B_c1sync 1 96 stock,mk --server-arg=--no-async-scheduling --meta async=0 ;;
    *) echo "unknown stage $s" ;;
  esac
done
echo "done $(date -u +%T)"
