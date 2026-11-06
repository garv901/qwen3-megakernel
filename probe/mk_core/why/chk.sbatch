#!/bin/bash
#SBATCH --job-name=why_chk
#SBATCH --gres=gpu:1
#SBATCH --output=probe/mk_core/why/logs/slurm-%j.log
cd "$SLURM_SUBMIT_DIR/probe/mk_core/why"
./bin_why --arms queue,queue_al,static_il,queue_perm_al,static_il_al --reps 10 > "$SLURM_SUBMIT_DIR/results/2026-10-28-why/chk_$SLURM_JOB_ID.jsonl"
