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

## P0: e2e baseline (job 9562, background) - pending
