#!/bin/bash
#SBATCH --job-name=mref_real
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/raid/garv901/regime-mk-core/probe/mk_core/mref_real/slurm-%j.out
set -euo pipefail
cd /raid/garv901/regime-mk-core
source slurm/env.sh
OUT=results/2026-10-28-mref-real
$PY probe/mk_core/mref_real/mref_real.py run --out $OUT
$PY probe/mk_core/mref_real/mref_real.py analyze --out $OUT
