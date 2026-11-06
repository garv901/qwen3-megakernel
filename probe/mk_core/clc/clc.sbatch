#!/bin/bash
#SBATCH --job-name=clc
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=probe/mk_core/clc/logs/slurm-%j.log
nvidia-smi -L
cd "$SLURM_SUBMIT_DIR/probe/mk_core/clc"
bash run.sh --build-only || exit 1
./bin/clc > "$SLURM_SUBMIT_DIR/results/2026-10-27-clc/clc_${SLURM_JOB_ID}.jsonl"
