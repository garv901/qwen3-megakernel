#!/bin/bash
#SBATCH --job-name=xover
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --output=/raid/garv901/regime-megakernel/probe/xover/logs/slurm-%j.log
nvidia-smi -L
cd /raid/garv901/regime-megakernel/probe/xover
bash run.sh
