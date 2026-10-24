#!/bin/bash
#SBATCH --job-name=xover4g
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=probe/mk_core/xover4/logs/slurm-%j.log
# Nano-batch question (MK-attn): tile-stream GEMM phases on 74 / 111 CTAs (one per SM) vs 148, CG 1, T = 64 / 128 / 256, every TT / KS config.
nvidia-smi -L
cd "$SLURM_SUBMIT_DIR/probe/mk_core/xover4"
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
for g in 74 111; do $NVCC -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -DCG=1 -DGRID_N=$g xover4.cu -o bin/ts1_g$g || exit 1; done
$NVCC -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -DCG=1 xover4.cu -o bin/ts1 || exit 1
for s in QKV O GU D; do for t in 64 128 256; do
  for b in ts1 ts1_g111 ts1_g74; do echo "## $b $s $t"; ./bin/$b 0 5 $s $t; done
done; done
