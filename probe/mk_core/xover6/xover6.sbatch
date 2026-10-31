#!/bin/bash
#SBATCH --job-name=xover6
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:45:00
#SBATCH --output=runs/slurm-%j.log
# SPEC-xover6 run plan. Submit from the worktree root: sbatch probe/mk_core/xover6/xover6.sbatch
# x6 args: mode k_a depth bytes_per_iter shape T TT KS iters   (TT / KS 0 = sweep {T, min(T,64)} x KS list; `both` picks the best (TT, KS) of its own gemm sweep at that k_c)
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
echo "REPO=$REPO branch $(git -C "$REPO" branch --show-current) sha $(git -C "$REPO" rev-parse --short HEAD)"
nvidia-smi -L
cd "$SLURM_SUBMIT_DIR/probe/mk_core/xover6"
bash run.sh --build-only || exit 1
B=536870912
x() { echo "## $*"; ./bin/x6 "$@"; }
# drift check, start
x kv 148 4 $B GU 64 0 0 20
x gemm 74 4 0 GU 64 0 0 0
# Q1: kv alone
for ka in 148 128 116 104 92 74; do for dp in 2 4 6; do x kv $ka $dp $B GU 64 0 0 20; done; done
# Q2 alone: gemm on k_c CTAs, k_c = 20 32 44 56 74 (k_a = 148 - k_c)
for ka in 128 116 104 92 74; do for s in QKV O GU D; do for t in 32 64 128; do x gemm $ka 4 0 $s $t 0 0 0; done; done; done
# reproduction: run-time k_c = 148 / 111 / 74 (k_a = 0 / 37 / 74) vs the compile-time GRID_N builds (xover4.cu, same config sweep: TT in {T, min(T,64)} is a subset of its list)
for ka in 0 37 74; do for s in QKV O GU D; do for t in 64 128; do x gemm $ka 4 0 $s $t 0 0 0; done; done; done
for g in 148 111 74; do for s in QKV O GU D; do for t in 64 128; do echo "## ref$g $s $t"; ./bin/ref$g 0 5 $s $t; done; done; done
# both: (k_a, k_c) = (128, 20) (116, 32) (104, 44) (92, 56) x depth 4 6 x shapes x T
for ka in 128 116 104 92; do for dp in 4 6; do for s in QKV O GU D; do for t in 32 64 128; do x both $ka $dp $B $s $t 0 0 0; done; done; done; done
# drift check, end
x gemm 74 4 0 GU 64 0 0 0
x kv 148 4 $B GU 64 0 0 20
echo done
