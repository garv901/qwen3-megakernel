#!/bin/bash
#SBATCH --job-name=xover5
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:40:00
#SBATCH --output=probe/mk_core/xover5/logs/slurm-%j.log
# TMA vs cp.async ingress (SPEC-xover5.md, PASS-cpa.md). Binaries live in probe/mk_core/xover5/bin only (no shared bin with xover2 / xover4).
nvidia-smi -L
cd "$SLURM_SUBMIT_DIR/probe/mk_core/xover5"
bash run.sh
