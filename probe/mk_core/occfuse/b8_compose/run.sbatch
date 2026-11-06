#!/bin/bash
#SBATCH --job-name=occf-b8
#SBATCH --gres=gpu:1
#SBATCH --output=runs/slurm-%j.log
# B8 b8_compose. MODE=pilot (checks + a few timing points) or MODE=full (all designs x T, pf 0/1/2, null arms). Submit from the repo root:
#   sbatch -p main --time=00:20:00 --export=ALL,MODE=pilot probe/mk_core/occfuse/b8_compose/run.sbatch
set -uo pipefail
BENCH=b8_compose; MODE=${MODE:-pilot}
REPO=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
D=$REPO/probe/mk_core/occfuse/$BENCH; R=$REPO/results/2026-10-29-occfuse/$BENCH; J=${SLURM_JOB_ID:-local}
mkdir -p "$D/bin" "$R"
echo "REPO=$REPO $(git -C "$REPO" rev-parse --short HEAD) job $J $(hostname) MODE=$MODE"; nvidia-smi -L
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv | tee "$R/clocks_before_$J.csv"
NVCC=/usr/local/cuda/bin/nvcc; F="-O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v -I$REPO/probe/mk_core/occfuse/common"
$NVCC $F "$D/$BENCH.cu" -o "$D/bin/$BENCH" > "$R/build_$J.log" 2>&1 || { echo "build failed: $R/build_$J.log"; exit 1; }
grep -A3 "k_b8" "$R/build_$J.log" | grep -E "registers|spill" | head -3
B="$D/bin/$BENCH"
FILL=${FILL:-}
if [ "$MODE" = pilot ]; then
  timeout 300 "$B" --check 1 --checkT 8,96,192 --T 8,256 --design 1,2,3 --pf 1 --out "$R/pilot_$J.st" ${FILL:+--fill "$FILL"} > "$R/pilot_$J.jsonl" || echo "!! pilot main exit $?"
  # path smoke tests (each one process, own timeout): pf 0 / 2, null, fin placement 2 at small T, KS overrides
  timeout 120 "$B" --check 0 --T 8 --design 3 --pf 0,2 --launches 3 >> "$R/pilot_$J.jsonl" || echo "!! pf smoke exit $?"
  timeout 120 "$B" --check 0 --T 8,256 --design 1,3 --null 1 --launches 3 >> "$R/pilot_$J.jsonl" || echo "!! null smoke exit $?"
  timeout 120 "$B" --check 0 --T 8 --design 3 --fin 2 --launches 3 >> "$R/pilot_$J.jsonl" || echo "!! fin2 smoke exit $?"
else
  # full: every arm in its own process under timeout (a hang is a result). pf2 uses the watchdog build.
  $NVCC $F -DB8_DBG "$D/$BENCH.cu" -o "$D/bin/${BENCH}_dbg" > "$R/build_dbg_$J.log" 2>&1 || echo "!! dbg build failed"
  for d in 1 2 3; do
    timeout 300 "$B" --check 1 --checkT 8,96,192 --design $d --T 1,8,16,32,64,96,128,192,256 --pf 1 --out "$R/main_${J}_d$d.st" ${FILL:+--fill "$FILL"} > "$R/main_${J}_d$d.jsonl" || echo "!! main d$d exit $?"
  done
  for d in 1 3; do
    timeout 200 "$B" --check 0 --design $d --T 1,8,32,64,128,256 --pf 1 --null 1 --out "$R/null_${J}_d$d.st" ${FILL:+--fill "$FILL"} > "$R/null_${J}_d$d.jsonl" || echo "!! null d$d exit $?"
    timeout 200 "$B" --check 0 --design $d --T 1,8,32,64,128,256 --pf 0 --out "$R/pf0_${J}_d$d.st" ${FILL:+--fill "$FILL"} > "$R/pf0_${J}_d$d.jsonl" || echo "!! pf0 d$d exit $?"
    for t in 8 64 256; do
      timeout 60 "${B}_dbg" --check 0 --design $d --T $t --pf 2 --launches 5 --out "$R/pf2_${J}_d${d}_T$t.st" ${FILL:+--fill "$FILL"} > "$R/pf2_${J}_d${d}_T$t.jsonl" 2> "$R/pf2_${J}_d${d}_T$t.err" || echo "!! pf2 d$d T$t exit $?"
    done
  done
  timeout 200 "$B" --check 0 --design 3 --T 8,64,128,256 --pf 1 --fin 0 --out "$R/fin0_$J.st" ${FILL:+--fill "$FILL"} > "$R/fin0_$J.jsonl" || echo "!! fin0 exit $?"
  timeout 200 "$B" --check 0 --design 3 --T 8,64,128,256 --pf 1 --fin 2 --out "$R/fin2_$J.st" ${FILL:+--fill "$FILL"} > "$R/fin2_$J.jsonl" || echo "!! fin2 exit $?"
fi
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv | tee "$R/clocks_after_$J.csv"
