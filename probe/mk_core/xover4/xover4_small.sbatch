#!/bin/bash
#SBATCH --job-name=xover4s
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:25:00
#SBATCH --output=probe/mk_core/xover4/logs/slurm-%j.log
# Small-T question (NPAD <= 64 on the tile stream?): split-K vs tile stream at T = 32 / 64 (T = 128 control = the current crossover), 4 shapes.
# a0 = mk's classic engine (xover2: KU 128, X by LDG+STS, ring 2); a3 = split-K with X by TMA, ring max (xover3 baseline);
# ts1 / ts1w = tile stream, B by TMA from bf16 (WU 0 / 1, every TT incl. 16 / 32 and KS); ts3sb = xover3 tile stream with B staged from fp32 (SB 1), T = 32 / 64.
# Two rounds (round 2 without the T = 128 control) to see placement noise. JSON lines on stdout, PASS/FAIL on stderr.
nvidia-smi -L
cd "$SLURM_SUBMIT_DIR/probe/mk_core/xover4"
bash build_small.sh || exit 1
for round in 1 2; do
  echo "## round $round"
  TS="32 64"; [ $round = 1 ] && TS="32 64 128"
  for s in QKV O GU D; do for t in $TS; do
    echo "## $s $t"
    ./bin/a0 0 5 $s $t
    ./bin/a3 0 5 $s $t
    ./bin/ts1 0 5 $s $t
    ./bin/ts1w 0 5 $s $t
    [ $t -le 64 ] && ./bin/ts3 0 5 $s $t 0 0 1
  done; done
done
