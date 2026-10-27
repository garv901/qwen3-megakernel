#!/bin/bash
#SBATCH --job-name=xover4t
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --output=probe/mk_core/xover4/logs/slurm-%j.log
# Low-T question (NPAD 16 on the tile stream?): split-K (a0 = mk classic, a3) vs tile stream (ts1 / ts1w, TT 16 for T < 16) at T = 1 / 4 / 8 / 16,
# T = 32 as the link to job 10214; ts3 (B staged from fp32, SB 1) at T = 16 / 32. Two rounds. Same arms as xover4_small.sbatch.
nvidia-smi -L
cd "$SLURM_SUBMIT_DIR/probe/mk_core/xover4"
bash build_small.sh || exit 1
for round in 1 2; do
  echo "## round $round"
  for s in QKV O GU D; do for t in 1 4 8 16 32; do
    echo "## $s $t"
    ./bin/a0 0 5 $s $t
    ./bin/a3 0 5 $s $t
    ./bin/ts1 0 5 $s $t
    ./bin/ts1w 0 5 $s $t
    [ $t -ge 16 ] && ./bin/ts3 0 5 $s $t 0 0 1
  done; done
done
