# clc: CLC vs atomic queue vs static split (finish-time spread of a 1 GiB streaming read)

Run: `bash run.sh [--build-only] [--validate] [--arms a,b] [--chunks bytes,..] [--reps 10]`, or `sbatch -p priority probe/mk_core/clc/clc.sbatch` from the repo root.
Table: `$PY analyze.py results/2026-10-27-clc/clc_<job>.jsonl`.

STATUS: job 10542 ran (2026-10-28, B200 GPU 0): every arm x C validated PASS, then 10 timed reps each. Results below.

## Build (nvcc -O3 sm_100a, ptxas -v; 0 spill, 0 stack in every arm)
| arm | regs |
|---|---|
| static, hw | 80 |
| queue, queue_pf (= guided kernel), clc, clc_np | 48 |
Static smem 64-88 B + 160 KB dynamic; occupancy asserted = 1 CTA/SM for every arm.

## CLC SASS
`UGETNEXTWORKID.SELFCAST` (the CLC try_cancel instruction) is emitted in the clc / clc_np kernels, with `SYNCS.ARRIVE.TRANS64`, `SYNCS.PHASECHK.TRANS64.TRYWAIT`, `FENCE.VIEW.ASYNC.S`, `ELECT` around it. Grepping `CLC|UCLC|try` finds nothing (the mnemonic is UGETNEXTWORKID).

## Validate (idle GPU 0, C = 1 MB): ALL PASS
Every chunk visited exactly once and XOR checksum = host checksum for static, queue, queue_pf, guided, hw, clc, clc_np.

## Deviations
1. CLC arm `clc` issues try_cancel for the next block before streaming the current one (the for_each_canceled_block sequence), i.e. it is already the prefetch variant. `clc_np` (extra arm) issues it after streaming and waits immediately.
2. `guided` reuses the queue_pf kernel with a host list: first 85% of bytes in N/148/4 chunks (rounded to 64 KB, 1.8 MB), to 97% in 256 KB, rest in 64 KB (1506 chunks).
3. Raw PTX for CLC (no libcu++ wrapper).
4. Per-chunk overhead = median gap between consecutive chunks on the same SM (includes CTA launch for hw).

## Results (job 10542, medians over 10 reps; make = max end - min start, tail = make - median SM finish, ovh is globaltimer-quantized)
```
arm             C  make us  tail us  sprd us    GB/s  ev GB/s   B/SM min-max (MB)  ovh ns   n
static      6.92M    181.0     7.16    36.30    5932     5761     7.26-7.26          -1  10
queue         64K    217.4     0.82     1.86    4938     4766     6.75-7.83         512  10
queue        256K    157.1     2.03     4.40    6835     6548     7.08-7.60         512  10
queue       1.00M    164.0     2.45    20.83    6548     6279     6.29-7.34         512  10
queue       4.00M    165.8     5.13    70.37    6476     6241     4.19-8.39         320  10
queue_pf      64K    189.7     0.78     1.92    5661     5462     7.01-7.54         256  10
queue_pf     256K    157.1     2.01     4.66    6833     6554     7.08-7.60         256  10
queue_pf    1.00M    163.4     2.93    22.37    6573     6317     6.29-7.34         256  10
queue_pf    4.00M    167.8     6.61    75.76    6397     6168     4.19-8.39         256  10
guided       list    163.9     1.94     2.93    6551     6309     6.75-7.67         256  10
hw            64K    221.4     0.92     2.03    4850     4724     6.88-7.67         512  10
hw           256K    160.3     1.93     4.62    6697     6473     6.82-7.60         512  10
hw          1.00M    162.6     2.55    17.73    6605     6386     6.29-7.34         512  10
hw          4.00M    165.6     4.19    68.96    6486     6242     4.19-8.39         512  10
clc           64K    197.7     1.13     1.97    5430     5270     7.08-7.47         256  10
clc          256K    157.9     2.33     4.66    6799     6554     7.08-7.47         256  10
clc         1.00M    163.9     2.94    22.08    6550     6316     6.29-7.34         256  10
clc         4.00M    167.8     6.69    75.89    6400     6168     4.19-8.39         256  10
clc_np        64K    216.8     0.88     1.84    4953     4810     6.88-7.67         256  10
clc_np       256K    156.9     2.03     4.45    6844     6553     7.08-7.60         256  10
clc_np      1.00M    164.2     2.20    20.77    6539     6316     6.29-7.34         256  10
clc_np      4.00M    165.8     4.39    70.61    6478     6218     4.19-8.39         256  10
```

## Conclusions
1. Static split (mk's planner shape): tail 7.2 us, spread 36 us, makespan 181 us. Its median SM finish (~174 us) is already slower than every dynamic arm's makespan at 256 KB (~157 us), so the static loss (~24 us, 13%) is mostly NOT the tail: 148 far-apart 7 MB streams run slower than many CTAs on nearby 256 KB chunks (likely DRAM locality; not isolated here).
2. Dynamic queue: 256 KB is the best C for every arm (157 us, tail ~2 us). 64 KB loses (one UNROLL batch per chunk drains the pipeline each chunk; prefetch recovers part: queue 217 -> queue_pf 190). 1-4 MB raise tail and spread. Guided (1506 chunks) has the smallest spread (2.9 us) but makespan 164 us, worse than uniform 256 KB: its 1.8 MB head chunks pay the large-C penalty.
3. CLC: at 256 KB, clc 157.9 / clc_np 156.9 / queue_pf 157.1 / queue 157.1 / hw 160.3 us: within noise of the atomic queue. CLC buys nothing measurable over the software queue.

## Rerun with static_il (job 11593, 2026-10-28; 27/27 PASS)
static_il: CTA b streams chunks b, b + 148, ...; nchunks rounded to a multiple of 148 (equal bytes per CTA), no atomics. Other arms reproduce job 10542 within ~0.5 us.
```
arm             C  make us  tail us  sprd us    GB/s  ev GB/s   B/SM min-max (MB)  ovh ns   n
static      6.92M    180.7     7.18    36.50    5943     5761     7.26-7.26          -1  10
static_il     64K    193.9     1.45     2.45    5537     5379     7.26-7.26           0  10
static_il    256K    165.4     1.68     4.27    6493     6236     7.25-7.26           0  10
static_il   1.00M    167.5     3.98     8.45    6410     6198     7.26-7.26           0  10
static_il   4.00M    176.8     5.08    12.11    6075     5874     7.26-7.26           0  10
queue         64K    217.6     0.96     2.03    4934     4777     6.75-7.86         512  10
queue        256K    157.5     1.95     4.48    6819     6519     7.08-7.60         512  10
queue       1.00M    164.0     2.30    20.69    6549     6311     6.29-7.34         512  10
queue       4.00M    166.3     5.46    70.80    6457     6203     4.19-8.39         256  10
queue_pf      64K    190.0     0.98     1.94    5651     5461     7.01-7.50         256  10
queue_pf     256K    157.4     2.17     4.74    6824     6554     7.08-7.60         256  10
queue_pf    1.00M    163.5     2.97    22.16    6566     6317     6.29-7.34         256  10
queue_pf    4.00M    168.1     7.34    74.99    6388     6168     4.19-8.39         256  10
guided       list    164.1     1.94     2.96    6544     6311     6.75-7.67         256  10
hw            64K    221.5     0.90     1.97    4848     4724     6.88-7.67         512  10
hw           256K    159.8     1.82     4.38    6721     6472     6.82-7.60         512  10
hw          1.00M    162.9     2.42    18.11    6590     6393     6.29-7.34         512  10
hw          4.00M    164.4     3.56    67.58    6530     6317     4.19-8.39         512  10
clc           64K    197.5     1.06     2.02    5437     5297     7.08-7.41         256  10
clc          256K    158.0     2.50     4.70    6796     6554     7.08-7.47         256  10
clc         1.00M    164.1     2.88    22.35    6544     6317     6.29-7.34         256  10
clc         4.00M    166.9     4.38    75.71    6432     6205     4.19-8.39         256  10
clc_np        64K    216.2     0.78     1.89    4966     4810     6.88-7.60         256  10
clc_np       256K    156.8     2.06     4.38    6847     6552     7.08-7.60         256  10
clc_np      1.00M    163.9     2.22    20.48    6553     6316     6.29-7.34         256  10
clc_np      4.00M    166.3     5.26    70.59    6458     6224     4.19-8.39         256  10
```
Split of the static -> queue gap at 256 KB (180.7 -> 157.5 us, 23.2 us):
- static -> static_il: 15.3 us. Same bytes per CTA, only the address order changes: locality of the concurrent reads.
- static_il -> queue: 7.9 us. static_il has a small tail (1.7 us) and spread (4.3 us), but every SM must read 7.26 MB; the queue gives slow SMs less (7.08-7.60 MB). Dynamic balance.
- static_il also beats queue at 64 KB (193.9 vs 217.6 us: no atomic between chunks) but not at 256 KB.
