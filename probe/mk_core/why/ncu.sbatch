#!/bin/bash
#SBATCH --job-name=why_ncu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=probe/mk_core/why/logs/slurm-%j.log
# E4: one k_stream launch per arm under ncu (the L2 flush kernel is excluded by -k).
cd "$SLURM_SUBMIT_DIR/probe/mk_core/why"
R="$SLURM_SUBMIT_DIR/results/2026-10-28-why"
M=gpu__time_duration.sum,dram__bytes_read.sum,dram__sectors_read.sum,dram__cycles_active_read.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors_lookup_miss.sum,lts__t_sectors_lookup_hit.sum,lts__t_sectors_srcunit_ltcfabric.sum,lts__t_requests_srcunit_ltcfabric.sum,lts__ltcfabric2lts_cycles_active.sum,lts__ltcfabric2lts_cycles_stalled.sum,lts__t_sectors_srcnode_gpc.sum
for a in static static_il queue static_perm static_rot74 queue_perm; do
  /usr/local/cuda/bin/ncu -k regex:k_stream --launch-count 1 --metrics $M --print-metric-instances values --csv ./bin_why --ncu $a > $R/ncu_${SLURM_JOB_ID}_$a.csv 2> $R/ncu_${SLURM_JOB_ID}_$a.err
  echo "$a rc=$?"; tail -2 $R/ncu_${SLURM_JOB_ID}_$a.err
done
