#!/bin/bash
#SBATCH --job-name=bar
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=probe/mk_core/bar/logs/slurm-%j.log
nvidia-smi -L
cd "$SLURM_SUBMIT_DIR/probe/mk_core/bar"
bash run.sh
