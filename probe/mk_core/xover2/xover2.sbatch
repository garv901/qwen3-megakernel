#!/bin/bash
#SBATCH --job-name=xover2
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:45:00
#SBATCH --output=probe/mk_core/xover2/logs/slurm-%j.log
nvidia-smi -L
cd "$SLURM_SUBMIT_DIR/probe/mk_core/xover2"
bash run.sh
