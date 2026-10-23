#!/bin/bash
#SBATCH --job-name=xover3
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=probe/mk_core/xover3/logs/slurm-%j.log
# P4 + P4b run plan (SPEC-xover3). ts args: nph reps shape T TT KS SB (0 = all).
nvidia-smi -L
cd "$SLURM_SUBMIT_DIR/probe/mk_core/xover3"
bash run.sh --build-only || exit 1
SHAPES="QKV O GU D"
# a3 baseline: 4 shapes x T 32..256 (16 points)
for s in $SHAPES; do for t in 32 64 128 256; do ./bin/a3 0 5 $s $t; done; done
# SB = 0 (TMA B, job 9592 behaviour) and SB = 1 (staged B): every valid config at T = 128, 256
for s in $SHAPES; do for t in 128 256; do
  ./bin/ts 0 5 $s $t 0 0 0
  ./bin/ts 0 5 $s $t 0 0 1
done; done
# SB = 1 at T = 32, 64: the best three SB = 0 configs per (shape, T) of job 9592 (TT KS; O T = 32 has only two valid configs)
while read s t cfgs; do
  set -- $cfgs
  while [ $# -ge 2 ]; do ./bin/ts 0 5 $s $t $1 $2 1; shift 2; done
done <<'EOF'
QKV 32 32 4 32 8 32 2
QKV 64 64 4 64 8 64 2
O 32 32 16 32 8
O 64 32 8 64 8 64 16
GU 32 32 8 32 4 32 2
GU 64 64 2 64 4 64 8
D 32 32 12 32 24 32 8
D 64 64 12 64 24 32 8
EOF
