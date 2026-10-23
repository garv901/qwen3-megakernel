#!/bin/bash
#SBATCH --job-name=xover3
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=probe/mk_core/xover3/logs/slurm-%j.log
nvidia-smi -L
cd "$SLURM_SUBMIT_DIR/probe/mk_core/xover3"
bash run.sh
