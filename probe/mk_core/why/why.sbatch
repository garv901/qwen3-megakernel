#!/bin/bash
#SBATCH --job-name=why
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=probe/mk_core/why/logs/slurm-%j.log
# timed arms + per-chunk dump + E2 latency probe + ncu (E4). Usage: sbatch -p priority --time=00:20:00 probe/mk_core/why/why.sbatch
nvidia-smi -L
cd "$SLURM_SUBMIT_DIR/probe/mk_core/why"
R="$SLURM_SUBMIT_DIR/results/2026-10-28-why"; J=$SLURM_JOB_ID
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v why.cu -o bin_why > logs/build.log 2>&1 || exit 1
ARMS=static,static_rot37,static_rot74,static_rot111,static_il,il_shift3,static_perm,queue,queue_perm,static_lpt,static_contig,static_contig_rj,static_contig_ro
ARMS=$ARMS,static_al,static_rot74_al,static_il_al,il_shift3_al,static_perm_al,queue_al,queue_perm_al,static_lpt_al,static_contig_al,static_contig_rj_al,static_contig_ro_al
ARMS=$ARMS,static_sync1,static_sync2,static_sync4,static_sync16,static_sync64,static_al_sync1,static_al_sync4,static_al_sync16,static_sh1,static_sh8,static_sh8_sync4
ARMS=$ARMS,static_wrot1,static_wrot3,static_al_wrot1,static_sh1_wrot1,static_sh8_wrot1,static_wrot1_sync64
ARMS=$ARMS,queue_sh1,queue_sh2,queue_sh8,static_il_sh1,static_il_sh2,static_il_sh8
./bin_why --arms $ARMS --dump $R/dump_$J.jsonl > $R/why_$J.jsonl || exit 1
./bin_why --lat $R/lat_$J.u16 || exit 1
./bin_why --lat1 $R/lat1_$J.u16 || exit 1
M=gpu__time_duration.sum,dram__bytes_read.sum,dram__sectors_read.sum,dram__cycles_active_read.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors_lookup_miss.sum,lts__t_sectors_lookup_hit.sum,lts__t_sectors_srcunit_ltcfabric.sum,lts__t_requests_srcunit_ltcfabric.sum,lts__ltcfabric2lts_cycles_active.sum,lts__ltcfabric2lts_cycles_stalled.sum,lts__t_sectors_srcnode_gpc.sum,lts__t_requests_srcunit_l1.sum,lts__t_sectors_srcunit_l1.sum
for a in static static_il queue static_perm static_al static_il_al queue_al static_perm_al queue_sh1 static_sync4 static_contig static_wrot1; do
  /usr/local/cuda/bin/ncu -k regex:k_stream --launch-count 1 --metrics $M --print-metric-instances values --csv ./bin_why --ncu $a > $R/ncu_${J}_$a.csv 2> $R/ncu_${J}_$a.err
  echo "ncu $a rc=$?"
done
