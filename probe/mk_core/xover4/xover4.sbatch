#!/bin/bash
#SBATCH --job-name=xover4
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=probe/mk_core/xover4/logs/slurm-%j.log
# P6 run plan (SPEC-xover4). ts args: nph reps shape T TT KS (0 = all). Every valid TT / KS config per shape and T.
nvidia-smi -L
cd "$SLURM_SUBMIT_DIR/probe/mk_core/xover4"
bash run.sh --build-only || exit 1
for s in QKV O GU D; do for t in 128 256; do
  ./bin/ts1 0 5 $s $t
  ./bin/ts2 0 5 $s $t
done; done
