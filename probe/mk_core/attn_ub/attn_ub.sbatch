#!/bin/bash
#SBATCH --job-name=attn-ub
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=runs/slurm-%j.log
# attn_ub: where does mk's attention lose to fi at large KV? (b8 kv4K / 16K / 32K, iso-KV 240K, b1 kv32K)
#  1. a1ub (standalone splitkv attend, mk NPAD 16 tune: EF 1, MERGE_CHUNK 20), --comb split: per-CTA attend end -> work median / imbalance / barrier latency
#     builds: mk (128 x 2 mma), ff (FFMA consumer), ld (FFMA + NOCOMP = KV ring only, no math), ld64 (load only, 64 x 4)
#  2. 3-stage ring (213.5 KB, over mk's 160 KB budget: study only): mk3, ld3 + mk control at the same dyn smem
#  2b. same, KV maps re-encoded as an HND layout (A1UB_HND=1): is mk's 2 KB-strided 256 B-per-head access costing HBM efficiency?
#  3. fi trtllm-gen decode kernel alone (CUDA events) + one ncu capture at b8 kv32K
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
cd "$REPO"
export CUDA_CACHE_DISABLE=1 PYTHONPATH=$REPO
echo "REPO=$REPO $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
nvidia-smi -L
J=${SLURM_JOB_ID:-local}
D=probe/mk_core/attn_ub
O=results/2026-10-27-attn-ub; mkdir -p $O
P=${POINTS:-b8kv4096,b8kv16384,b8kv32768,b8kv30720,b1kv245760,b1kv32768}
K=SKV_KV_EVICT_FIRST=1,SKV_MERGE_CHUNK=20
LD=$K,SKV_TMA_MMA=0,SKV_PROF_NOCOMP=1
echo "== 1. a1ub 2-stage builds"
timeout 900 "$PY" $D/a1ub.py --time --comb split --points $P --no-default --launches 5 --job $J --out $O/ub_${J}_s2.jsonl \
  --build mk=$K --build ff=$K,SKV_TMA_MMA=0 --build ld=$LD --build ld64=$LD,SKV_STAGE_ROWS=64,SKV_STAGES=4; echo "exit $?"
echo "== 2. a1ub 3-stage builds (dyn smem 214016)"
A1UB_MAX_SMEM=232448 timeout 600 "$PY" $D/a1ub.py --time --comb split --points $P --no-default --launches 5 --dyn-smem 214016 --job $J --out $O/ub_${J}_s3.jsonl \
  --build mk=$K --build mk3=$K,SKV_STAGES=3 --build ld3=$LD,SKV_STAGES=3; echo "exit $?"
echo "== 2b. HND KV layout (A1UB_HND=1: per-head contiguous rows; timing only)"
A1UB_HND=1 timeout 600 "$PY" $D/a1ub.py --time --comb split --points $P --no-default --launches 5 --job $J --out $O/ub_${J}_hnd.jsonl \
  --build mkH=$K --build ldH=$LD; echo "exit $?"
echo "== 3. fi kernel"
timeout 300 "$PY" $D/fi_ub.py ${P//,/ }; echo "exit $?"
timeout 300 ncu --kernel-name regex:fmha --launch-skip 3 --launch-count 1 --section LaunchStats --section SpeedOfLight --section Occupancy \
  --metrics dram__bytes_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,launch__cluster_dim_x,launch__cluster_dim_y,launch__cluster_dim_z \
  "$PY" $D/fi_ub.py b8kv32768 --reps 1 --layers 1; echo "exit $?"
