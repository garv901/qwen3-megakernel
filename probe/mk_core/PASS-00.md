# PASS-00: baseline and per-phase targets (MK-core)

## P0-fi: fi-graph graph-pdl-best kernel timeline, R1 (job 9564, GPU-c09878e8, mk-core bad611f, 2026-10-23) [M]
Log `runs/slurm-9564.log`, JSON `results/2026-10-23-mk/p0_fi_timeline_9564.json`. All 4 points PASS (ratio 0.80-0.82).
Torch-profiler kernel times under PDL overlap (sum of kernel time > step time), so each kernel time is an upper bound
on its own cost. Kernel names are truncated: the GEMM -> phase mapping below is by weight bytes, marked [H].

| point | step us | GEMM kernels (us per call = us/step / calls) | LM head | norms + act + rope + KV append | attention |
|---|---|---|---|---|---|
| R1-b1 | 782 | cute-dsl split-K: 3.93 / 3.56 / 3.46 / 2.99 (GU / QKV / D / O [H]) | 45.7 | 175 + 134 + 127 + 46 + 49 = 531 | 123 |
| R1-b16 | 963 | cute-dsl split-K: 4.72 / 4.15 / 3.78 / 3.44 | 57.4 | 185 + 145 + 137 + 53 + 52 = 572 | 154 |
| R1-b64 | 1360 | tgv 2 x 5.64 (O, D) / nvjet 48x64 5.58 (QKV) / nvjet 2cta 64x32 5.14 (GU) | 59.4 | 186 + 168 + 136 + 57 + 57 = 604 | 346 (trtllm fmha) |
| R1-b256 | 2279 | nvjet 2cta 64x32 2 x 6.10 / 96x128 2cta 6.68 / 112x64 2cta 5.96 | 70.4 | 215 + 188 + 180 + 75 + 69 = 727 | 917 (fa2) |

## Reading [H]
- fi's GEMMs are 3.0-4.7 us per call at T <= 16 and 5.1-6.7 us at T = 64-256. xover's tcgen05 phase (job 9521) is GU
  7.74-8.03 / D 5.10-5.25 at T <= 16 and GU 14.1 / D 9.1 at T = 256. **Per GEMM, the xover phase is 1.3-2.3x slower
  than fi's kernel.** mk wins today only by fusion: ~530-730 us/step of fi's norm / act / RoPE / append kernels go away.
- So the GEMM phase skeleton is the first target (Pass 0b decides how): per-phase targets = fi's per-call times above
  (GU <= 3.9 us at T = 1, <= 6 us at T = 256, including mk's barrier).
- At T >= 64 every fi GEMM is a 2-CTA (cta_group::2) kernel: weight for the cta_group::2 arm (a5) in Pass 0b.
- b256 attention (917 us, fa2) is 40 % of fi's step: MK-attn's target, not this ledger's.

## P0: e2e baseline (job 9562, background, GPU-17dce9e8, mk-core bad611f, 2026-10-23) [M]
`slurm/regime.sbatch --arms hv2g,fi-graph,vllm --workloads mk --tag mk-p0`, 20 rounds, run dir `runs/20261023T175548Z-mk-p0`,
report `results/2026-10-23-mk/p0/` (REPORT.md, summary.csv). Steady us/step, best variant per arm:

| workload | hv2g | fi-graph (best variant) | vllm (best variant) | target = best SOTA |
|---|---|---|---|---|
| R0-b1-kv128 | 379.9 | 786.9 (graph-pdl-best) | 2443.2 (nopdl) | 786.9 |
| R1-b1-kv128 | 371.0 | 891.9 (graph-pdl-best) | 2404.9 (pdl) | 891.9 |
| R1-b2-kv128 | 780.0 | 897.6 (graph-pdl-best) | 2410.8 (nopdl) | 897.6 |
| R1-b4-kv128 | 998.8 | 926.8 (graph-pdl-best) | 2451.4 (nopdl) | 926.8 |
| R1-b8-kv128 | 1481.5 | 972.1 (graph-pdl-best) | 2584.5 (pdl) | 972.1 |
| R1-b16-kv128 | 2619.8 | 1018.7 (graph-pdl-best) | 2734.3 (nopdl) | 1018.7 |
| R1-b32-kv128 | 4881.4 | 1175.9 (graph-pdl-best) | 2917.9 (pdl) | 1175.9 |
| R1-b64-kv128 | 9310.8 | 1407.1 (graph-pdl-best) | 3343.1 (pdl) | 1407.1 |
| R1-b128-kv128 | 18354.0 | 1768.7 (graph-pdl-cublas) | 4259.8 (nopdl) | 1768.7 |
| R1-b256-kv128 | 37268.5 | 2235.3 (graph-pdl-best) | 6263.2 (pdl) | 2235.3 |
| R2-b1-kv128 | 371.3 | 889.7 (graph-pdl-best) | 2331.0 (pdl) | 889.7 |
| R2-b1-kv512 | 504.0 | 974.3 (graph-pdl-best) | 2502.7 (nopdl) | 974.3 |
| R2-b1-kv1024 | 636.7 | 964.5 (graph-pdl-best) | 2495.1 (nopdl) | 964.5 |
| R2-b1-kv2048 | 899.9 | 971.1 (graph-pdl-best) | 2525.7 (nopdl) | 971.1 |
| R2-b1-kv4096 | 1431.8 | 1003.5 (graph-pdl-best) | 2498.7 (pdl) | 1003.5 |
| R2-b1-kv8192 | 2471.9 | 1156.6 (graph-pdl-best) | 2706.8 (nopdl) | 1156.6 |
| R2-b1-kv16384 | 4563.6 | 1308.2 (graph-pdl-best) | 2916.5 (nopdl) | 1308.2 |
| R2-b1-kv32768 | 8706.4 | 1601.8 (graph-pdl-best) | 3313.3 (nopdl) | 1601.8 |
| R3-b1-kv1024 | 637.0 | 963.4 (graph-pdl-best) | 2516.6 (nopdl) | 963.4 |
| R3-b1-kv4096 | 1432.0 | 1001.6 (graph-pdl-best) | 2474.0 (nopdl) | 1001.6 |
| R3-b1-kv16384 | 4564.1 | 1305.8 (graph-pdl-best) | 2944.5 (nopdl) | 1305.8 |
| R3-b8-kv1024 | 1783.0 | 1154.8 (graph-pdl-best) | 2759.4 (pdl) | 1154.8 |
| R3-b8-kv4096 | 2809.1 | 1597.5 (graph-pdl-best) | 3184.7 (nopdl) | 1597.5 |
| R3-b32-kv1024 | 6186.5 | 1694.4 (graph-pdl-best) | 3481.8 (pdl) | 1694.4 |
| R6-b1-kv1024-q2 | 1035.2 | 1078.5 (graph-pdl-best) | 3687.5 (nopdl) | 1078.5 |
| R6-b1-kv1024-q4 | 1186.5 | 1103.3 (graph-pdl-best) | 6312.0 (pdl) | 1103.3 |
| R6-b1-kv1024-q8 | 1559.4 | 1142.1 (graph-pdl-best) | 6333.1 (nopdl) | 1142.1 |
| R6-b1-kv1024-q16 | 2674.6 | 1161.7 (graph-pdl-best) | 6472.6 (nopdl) | 1161.7 |

Reading: fi-graph is the best SOTA arm at all 28 points (vLLM is 1.8-5.6x slower), so the target at every point is fi-graph.
hv2g beats fi only at b1 short KV (R0/R1/R2-b1 up to kv2048, R3-b1-kv1024, R6-q2): its FFMA GEMMs lose from b4 up.
