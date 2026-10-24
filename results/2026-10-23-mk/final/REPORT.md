# regimebench report

- Runs: 20261024T041006Z-mk-final; slurm jobs: 9806 (logs under runs/<run>/ and runs/slurm-<job>.log)
- GPU: NVIDIA B200 (148 SMs), driver 580.178.04, host cayenne
- Versions: torch 2.13.0, flashinfer-python 0.7.0.post1, vllm 0.31.0, triton 3.7.1, nvidia-ml-py 13.615.71; git 872b6d6b2c
- SOL spec: 8.0 TB/s, BF16 dense 2.25 PFLOP/s (datasheet-derived, unverified); measured: copy BW 6561 GB/s median, cuBLAS bf16 1648 TFLOP/s, SM clock at probe 1965 MHz
- All times are microseconds per step (steady = back-to-back launches). hv2-repeat is a timing proxy (N repeated b1 decodes), not a correct forward; shown only in R0/R1, where its b1/kv128 work matches the step (elsewhere it skips KV work and can fall below SOL).
- FFMA-SOL = max(T_bw, flops / (148 SMs x 128 lanes x 2 x probed SM clock)); a ceiling for an FP32-FFMA design like hv2g, not for a tensor-core one (hypothesis-grade model, not a measurement).
- Run 20261024T041006Z-mk-final (slurm job 9806, SM clock at probe 1965 MHz): regimes R0, R1, R2, R3, R6
- Cross-regime comparisons are across jobs (and possibly GPUs); ratios within a point are interleaved in one job and are the reliable comparison.

## Executive summary

| regime | points | hv2g wins | geomean hv2g/best | crossover (best baseline first beats hv2g) | median %SOL hv2g | median %SOL best |
|---|---|---|---|---|---|---|
| R0 | 1 | 0/0 | - | none | - | 16.9 |
| R1 | 9 | 0/0 | - | none | - | 17.5 |
| R2 | 8 | 0/0 | - | none | - | 19.5 |
| R3 | 6 | 0/0 | - | B=1: none; B=8: none; B=32: none | - | 26.3 |
| R6 | 4 | 0/0 | - | none | - | 14.5 |

## Megakernel arms vs best baseline

| point | SOL us | best baseline us | vllm us | mk us | mk/best |
|---|---|---|---|---|---|
| R0-b1-kv128 | 150.9 | 890.6 | 2369.6 | 737.3 | 0.83 |
| R1-b1-kv128 | 150.9 | 888.2 | 2476.2 | 746.2 | 0.84 |
| R1-b2-kv128 | 152.9 | 900.3 | 2496.7 | 758.9 | 0.84 |
| R1-b4-kv128 | 156.7 | 938.5 | 2408.8 | 761.1 | 0.81 |
| R1-b8-kv128 | 164.4 | 986.7 | 2526.4 | 777.6 | 0.79 |
| R1-b16-kv128 | 179.8 | 1026.6 | 2729.2 | 856.8 | 0.83 |
| R1-b32-kv128 | 210.6 | 1180.5 | 2824.8 | 894.5 | 0.76 |
| R1-b64-kv128 | 272.2 | 1402.1 | 3415.3 | 1243.1 | 0.89 |
| R1-b128-kv128 | 395.5 | 1766.2 | 4344.2 | 1531.8 | 0.87 |
| R1-b256-kv128 | 641.9 | 2263.0 | 6227.5 | 2204.3 | 0.97 |
| R2-b1-kv128 | 150.9 | 885.1 | 2364.9 | 750.7 | 0.85 |
| R2-b1-kv512 | 156.4 | 968.4 | 2422.8 | 776.8 | 0.80 |
| R2-b1-kv1024 | 163.8 | 962.6 | 2469.5 | 789.6 | 0.82 |
| R2-b1-kv2048 | 178.5 | 976.2 | 2480.6 | 813.1 | 0.83 |
| R2-b1-kv4096 | 207.8 | 1002.6 | 2494.4 | 864.2 | 0.86 |
| R2-b1-kv8192 | 266.5 | 1151.8 | 2651.3 | 962.3 | 0.84 |
| R2-b1-kv16384 | 384.0 | 1301.7 | 2935.5 | 1145.8 | 0.88 |
| R2-b1-kv32768 | 618.9 | 1614.6 | 3256.2 | 1410.7 | 0.87 |
| R3-b1-kv1024 | 163.8 | 965.6 | 2472.8 | 790.2 | 0.82 |
| R3-b1-kv4096 | 207.8 | 1004.3 | 2578.6 | 863.4 | 0.86 |
| R3-b1-kv16384 | 384.0 | 1300.6 | 2910.8 | 1146.2 | 0.88 |
| R3-b8-kv1024 | 267.2 | 1162.0 | 2688.5 | 1026.8 | 0.88 |
| R3-b8-kv4096 | 619.5 | 1588.4 | 3167.0 | 1420.5 | 0.89 |
| R3-b32-kv1024 | 621.7 | 1692.8 | 3440.2 | 1404.1 | 0.83 |
| R6-b1-kv1024-q2 | 163.8 | 1074.4 | 3666.5 | 825.3 | 0.77 |
| R6-b1-kv1024-q4 | 163.8 | 1111.5 | 6548.0 | 832.2 | 0.75 |
| R6-b1-kv1024-q8 | 163.9 | 1150.0 | 6520.1 | 851.5 | 0.74 |
| R6-b1-kv1024-q16 | 164.0 | 1167.0 | 6455.7 | 883.9 | 0.76 |

## R0

| point | SOL us (bound) | FFMA-SOL us | hv2g us | hv2g mtn us | hv2-repeat us (proxy) | best baseline us (variant) | vllm wall / gpu us | hv2g/best | %SOL hv2g | %FFMA-SOL hv2g | %SOL best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R0-b1-kv128 | 150.9 (memory) | 150.9 | - | - | - | 890.6 (graph-pdl-best) | 2369.6 / 1841.1 (nopdl) | - | - | - | 16.9 |

![R0](R0.png)

## R1

| point | SOL us (bound) | FFMA-SOL us | hv2g us | hv2g mtn us | hv2-repeat us (proxy) | best baseline us (variant) | vllm wall / gpu us | hv2g/best | %SOL hv2g | %FFMA-SOL hv2g | %SOL best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1-b1-kv128 | 150.9 (memory) | 150.9 | - | - | - | 888.2 (graph-pdl-best) | 2476.2 / 1837.0 (nopdl) | - | - | - | 17.0 |
| R1-b2-kv128 | 152.9 (memory) | 152.9 | - | - | - | 900.3 (graph-pdl-best) | 2496.7 / 1846.4 (pdl) | - | - | - | 17.0 |
| R1-b4-kv128 | 156.7 (memory) | 156.7 | - | - | - | 938.5 (graph-pdl-best) | 2408.8 / 1848.8 (nopdl) | - | - | - | 16.7 |
| R1-b8-kv128 | 164.4 (memory) | 164.4 | - | - | - | 986.7 (graph-pdl-best) | 2526.4 / 1864.8 (pdl) | - | - | - | 16.7 |
| R1-b16-kv128 | 179.8 (memory) | 262.5 | - | - | - | 1026.6 (graph-pdl-best) | 2729.2 / 1920.6 (pdl) | - | - | - | 17.5 |
| R1-b32-kv128 | 210.6 (memory) | 525.0 | - | - | - | 1180.5 (graph-pdl-best) | 2824.8 / 1966.3 (nopdl) | - | - | - | 17.8 |
| R1-b64-kv128 | 272.2 (memory) | 1050.1 | - | - | - | 1402.1 (graph-pdl-best) | 3415.3 / 2100.5 (nopdl) | - | - | - | 19.4 |
| R1-b128-kv128 | 395.5 (memory) | 2100.2 | - | - | - | 1766.2 (graph-pdl-best) | 4344.2 / 2400.7 (nopdl) | - | - | - | 22.4 |
| R1-b256-kv128 | 641.9 (memory) | 4200.4 | - | - | - | 2263.0 (graph-pdl-cublas) | 6227.5 / 3030.2 (pdl) | - | - | - | 28.4 |

![R1](R1.png)

## R2

| point | SOL us (bound) | FFMA-SOL us | hv2g us | hv2g mtn us | hv2-repeat us (proxy) | best baseline us (variant) | vllm wall / gpu us | hv2g/best | %SOL hv2g | %FFMA-SOL hv2g | %SOL best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R2-b1-kv128 | 150.9 (memory) | 150.9 | - | - | n/a | 885.1 (graph-pdl-best) | 2364.9 / 1838.8 (nopdl) | - | - | - | 17.1 |
| R2-b1-kv512 | 156.4 (memory) | 156.4 | - | - | n/a | 968.4 (graph-pdl-best) | 2422.8 / 1894.9 (nopdl) | - | - | - | 16.2 |
| R2-b1-kv1024 | 163.8 (memory) | 163.8 | - | - | n/a | 962.6 (graph-pdl-best) | 2469.5 / 1888.3 (nopdl) | - | - | - | 17.0 |
| R2-b1-kv2048 | 178.5 (memory) | 178.5 | - | - | n/a | 976.2 (graph-pdl-best) | 2480.6 / 1899.9 (pdl) | - | - | - | 18.3 |
| R2-b1-kv4096 | 207.8 (memory) | 207.8 | - | - | n/a | 1002.6 (graph-pdl-best) | 2494.4 / 1924.2 (nopdl) | - | - | - | 20.7 |
| R2-b1-kv8192 | 266.5 (memory) | 266.5 | - | - | n/a | 1151.8 (graph-pdl-best) | 2651.3 / 2084.2 (nopdl) | - | - | - | 23.1 |
| R2-b1-kv16384 | 384.0 (memory) | 384.0 | - | - | n/a | 1301.7 (graph-pdl-best) | 2935.5 / 2301.0 (nopdl) | - | - | - | 29.5 |
| R2-b1-kv32768 | 618.9 (memory) | 618.9 | - | - | n/a | 1614.6 (graph-pdl-best) | 3256.2 / 2676.4 (nopdl) | - | - | - | 38.3 |

![R2](R2.png)

## R3

| point | SOL us (bound) | FFMA-SOL us | hv2g us | hv2g mtn us | hv2-repeat us (proxy) | best baseline us (variant) | vllm wall / gpu us | hv2g/best | %SOL hv2g | %FFMA-SOL hv2g | %SOL best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R3-b1-kv1024 | 163.8 (memory) | 163.8 | - | - | n/a | 965.6 (graph-pdl-best) | 2472.8 / 1874.9 (pdl) | - | - | - | 17.0 |
| R3-b1-kv4096 | 207.8 (memory) | 207.8 | - | - | n/a | 1004.3 (graph-pdl-best) | 2578.6 / 1934.8 (pdl) | - | - | - | 20.7 |
| R3-b1-kv16384 | 384.0 (memory) | 384.0 | - | - | n/a | 1300.6 (graph-pdl-best) | 2910.8 / 2313.2 (nopdl) | - | - | - | 29.5 |
| R3-b8-kv1024 | 267.2 (memory) | 267.2 | - | - | n/a | 1162.0 (graph-pdl-best) | 2688.5 / 2023.1 (nopdl) | - | - | - | 23.0 |
| R3-b8-kv4096 | 619.5 (memory) | 619.5 | - | - | n/a | 1588.4 (graph-pdl-best) | 3167.0 / 2441.9 (pdl) | - | - | - | 39.0 |
| R3-b32-kv1024 | 621.7 (memory) | 621.7 | - | - | n/a | 1692.8 (graph-pdl-best) | 3440.2 / 2505.3 (pdl) | - | - | - | 36.7 |

![R3](R3.png)

## R6

| point | SOL us (bound) | FFMA-SOL us | hv2g us | hv2g mtn us | hv2-repeat us (proxy) | best baseline us (variant) | vllm wall / gpu us | hv2g/best | %SOL hv2g | %FFMA-SOL hv2g | %SOL best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R6-b1-kv1024-q2 | 163.8 (memory) | 163.8 | - | - | n/a | 1074.4 (graph-pdl-best) | 3666.5 / 3447.9 (nopdl) | - | - | - | 15.2 |
| R6-b1-kv1024-q4 | 163.8 (memory) | 163.8 | - | - | n/a | 1111.5 (graph-pdl-best) | 6548.0 / 6326.2 (pdl) | - | - | - | 14.7 |
| R6-b1-kv1024-q8 | 163.9 (memory) | 163.9 | - | - | n/a | 1150.0 (graph-pdl-best) | 6520.1 / 6285.0 (pdl) | - | - | - | 14.3 |
| R6-b1-kv1024-q16 | 164.0 (memory) | 244.4 | - | - | n/a | 1167.0 (graph-pdl-best) | 6455.7 / 6232.5 (nopdl) | - | - | - | 14.1 |

![R6](R6.png)

## Winner variants and backends

| point | best fi-graph variant | attn backend | gemm | PDL cover | best vllm variant |
|---|---|---|---|---|---|
| R0-b1-kv128 | graph-pdl-best | decode/cute-dsl/tc=True | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cute-dsl', 'down': 'cute-dsl', 'lm_head': 'cute-dsl'} | 254/312 | nopdl |
| R1-b1-kv128 | graph-pdl-best | decode/cute-dsl/tc=True | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cute-dsl', 'down': 'cute-dsl', 'lm_head': 'cute-dsl'} | 254/312 | nopdl |
| R1-b2-kv128 | graph-pdl-best | decode/cute-dsl/tc=True | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cute-dsl', 'down': 'cute-dsl', 'lm_head': 'cute-dsl'} | 254/312 | pdl |
| R1-b4-kv128 | graph-pdl-best | decode/cute-dsl/tc=True | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cute-dsl', 'down': 'cute-dsl', 'lm_head': 'cute-dsl'} | 254/312 | nopdl |
| R1-b8-kv128 | graph-pdl-best | decode/cute-dsl/tc=True | {'qkv': 'cute-dsl', 'o': 'tinygemm', 'gu': 'cute-dsl', 'down': 'tinygemm', 'lm_head': 'cute-dsl'} | 254/312 | pdl |
| R1-b16-kv128 | graph-pdl-best | decode/cute-dsl/tc=True | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cute-dsl', 'down': 'cute-dsl', 'lm_head': 'cute-dsl'} | 254/312 | pdl |
| R1-b32-kv128 | graph-pdl-best | decode/trtllm-gen/tc=True | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cublas', 'down': 'cute-dsl', 'lm_head': 'cudnn'} | 225/312 | nopdl |
| R1-b64-kv128 | graph-pdl-best | decode/trtllm-direct | {'qkv': 'cudnn', 'o': 'tgv', 'gu': 'cublas', 'down': 'tgv', 'lm_head': 'cublas'} | 197/312 | nopdl |
| R1-b128-kv128 | graph-pdl-best | decode/trtllm-direct | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cublas', 'down': 'cublas', 'lm_head': 'cute-dsl'} | 141/312 | nopdl |
| R1-b256-kv128 | graph-pdl-cublas | decode/auto/tc=False | {'qkv': 'cublas', 'o': 'cublas', 'gu': 'cublas', 'down': 'cublas', 'lm_head': 'cublas'} | 141/312 | pdl |
| R2-b1-kv128 | graph-pdl-best | decode/cute-dsl/tc=True | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cute-dsl', 'down': 'cute-dsl', 'lm_head': 'cute-dsl'} | 254/312 | nopdl |
| R2-b1-kv512 | graph-pdl-best | decode/trtllm-gen/tc=True | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cute-dsl', 'down': 'cute-dsl', 'lm_head': 'cute-dsl'} | 254/312 | nopdl |
| R2-b1-kv1024 | graph-pdl-best | decode/trtllm-direct | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cute-dsl', 'down': 'cute-dsl', 'lm_head': 'cute-dsl'} | 254/312 | nopdl |
| R2-b1-kv2048 | graph-pdl-best | decode/trtllm-direct | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cute-dsl', 'down': 'cute-dsl', 'lm_head': 'cute-dsl'} | 254/312 | pdl |
| R2-b1-kv4096 | graph-pdl-best | decode/trtllm-direct | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cute-dsl', 'down': 'cute-dsl', 'lm_head': 'cute-dsl'} | 254/312 | nopdl |
| R2-b1-kv8192 | graph-pdl-best | decode/trtllm-direct | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cute-dsl', 'down': 'cute-dsl', 'lm_head': 'cute-dsl'} | 254/312 | nopdl |
| R2-b1-kv16384 | graph-pdl-best | decode/trtllm-direct | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cute-dsl', 'down': 'cute-dsl', 'lm_head': 'cute-dsl'} | 254/312 | nopdl |
| R2-b1-kv32768 | graph-pdl-best | decode/cute-dsl/tc=True | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cute-dsl', 'down': 'cute-dsl', 'lm_head': 'cute-dsl'} | 254/312 | nopdl |
| R3-b1-kv1024 | graph-pdl-best | decode/trtllm-gen/tc=True | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cute-dsl', 'down': 'cute-dsl', 'lm_head': 'cute-dsl'} | 254/312 | pdl |
| R3-b1-kv4096 | graph-pdl-best | decode/trtllm-direct | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cute-dsl', 'down': 'cute-dsl', 'lm_head': 'cute-dsl'} | 254/312 | pdl |
| R3-b1-kv16384 | graph-pdl-best | decode/trtllm-gen/tc=True | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cute-dsl', 'down': 'cute-dsl', 'lm_head': 'cute-dsl'} | 254/312 | nopdl |
| R3-b8-kv1024 | graph-pdl-best | decode/cute-dsl/tc=True | {'qkv': 'cute-dsl', 'o': 'tinygemm', 'gu': 'cute-dsl', 'down': 'tinygemm', 'lm_head': 'cute-dsl'} | 254/312 | nopdl |
| R3-b8-kv4096 | graph-pdl-best | decode/cute-dsl/tc=True | {'qkv': 'cute-dsl', 'o': 'tinygemm', 'gu': 'cute-dsl', 'down': 'tinygemm', 'lm_head': 'cute-dsl'} | 254/312 | pdl |
| R3-b32-kv1024 | graph-pdl-best | decode/cute-dsl/tc=True | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cublas', 'down': 'cute-dsl', 'lm_head': 'cudnn'} | 225/312 | pdl |
| R6-b1-kv1024-q2 | graph-pdl-best | prefill/fa2 | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cute-dsl', 'down': 'cute-dsl', 'lm_head': 'cute-dsl'} | 254/313 | nopdl |
| R6-b1-kv1024-q4 | graph-pdl-best | prefill/fa2 | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cute-dsl', 'down': 'cute-dsl', 'lm_head': 'cute-dsl'} | 254/313 | pdl |
| R6-b1-kv1024-q8 | graph-pdl-best | prefill/fa2 | {'qkv': 'cute-dsl', 'o': 'tinygemm', 'gu': 'cute-dsl', 'down': 'tinygemm', 'lm_head': 'cute-dsl'} | 254/313 | pdl |
| R6-b1-kv1024-q16 | graph-pdl-best | prefill/fa2 | {'qkv': 'cute-dsl', 'o': 'cute-dsl', 'gu': 'cute-dsl', 'down': 'cute-dsl', 'lm_head': 'cute-dsl'} | 254/313 | nopdl |

## Accuracy (FP32 oracle band, pass = rel-L2 <= 2x BF16 reference's own)

| point | arm:variant | seeds | worst x band (logits/k/v) | next_token match | verdict |
|---|---|---|---|---|---|
| R0-b1-kv128 | fi-graph:eager-pdl-best | 3 | 0.96/0.85/0.85 | 1.00 | pass |
| R0-b1-kv128 | fi-graph:graph-nopdl-best | 3 | 0.96/0.85/0.85 | 1.00 | pass |
| R0-b1-kv128 | fi-graph:graph-nopdl-cublas | 3 | 0.90/0.82/0.82 | 1.00 | pass |
| R0-b1-kv128 | fi-graph:graph-pdl-best | 3 | 0.96/0.85/0.85 | 1.00 | pass |
| R0-b1-kv128 | fi-graph:graph-pdl-cublas | 3 | 0.90/0.82/0.82 | 1.00 | pass |
| R0-b1-kv128 | mk:auto | 3 | 0.60/0.57/0.57 | 1.00 | pass |
| R0-b1-kv128 | mk:auto-direct | 3 | 0.59/0.56/0.56 | 1.00 | pass |
| R1-b64-kv128 | fi-graph:graph-nopdl-best | 3 | 0.80/0.78/0.78 | 0.89 | pass |
| R1-b64-kv128 | fi-graph:graph-nopdl-cublas | 3 | 0.80/0.78/0.78 | 0.89 | pass |
| R1-b64-kv128 | fi-graph:graph-pdl-best | 3 | 0.80/0.78/0.78 | 0.89 | pass |
| R1-b64-kv128 | fi-graph:graph-pdl-cublas | 3 | 0.80/0.78/0.78 | 0.89 | pass |
| R1-b64-kv128 | mk:auto | 3 | 0.49/0.50/0.50 | 0.92 | pass |
| R1-b64-kv128 | mk:auto-direct | 3 | 0.49/0.50/0.50 | 0.97 | pass |
| R1-b8-kv128 | fi-graph:graph-nopdl-best | 3 | 0.82/0.81/0.81 | 1.00 | pass |
| R1-b8-kv128 | fi-graph:graph-nopdl-cublas | 3 | 0.80/0.80/0.80 | 1.00 | pass |
| R1-b8-kv128 | fi-graph:graph-pdl-best | 3 | 0.82/0.81/0.81 | 1.00 | pass |
| R1-b8-kv128 | fi-graph:graph-pdl-cublas | 3 | 0.80/0.80/0.80 | 1.00 | pass |
| R1-b8-kv128 | mk:auto | 3 | 0.50/0.52/0.52 | 1.00 | pass |
| R1-b8-kv128 | mk:auto-direct | 3 | 0.50/0.52/0.52 | 0.88 | pass |
| R2-b1-kv1024 | fi-graph:graph-nopdl-best | 3 | 0.78/0.77/0.77 | 1.00 | pass |
| R2-b1-kv1024 | fi-graph:graph-nopdl-cublas | 3 | 0.80/0.80/0.79 | 1.00 | pass |
| R2-b1-kv1024 | fi-graph:graph-pdl-best | 3 | 0.78/0.77/0.77 | 1.00 | pass |
| R2-b1-kv1024 | fi-graph:graph-pdl-cublas | 3 | 0.80/0.80/0.79 | 1.00 | pass |
| R2-b1-kv1024 | mk:auto | 3 | 0.42/0.44/0.44 | 1.00 | pass |
| R2-b1-kv1024 | mk:auto-direct | 3 | 0.40/0.43/0.42 | 1.00 | pass |
| R2-b1-kv8192 | fi-graph:graph-nopdl-best | 3 | 0.88/0.86/0.87 | 1.00 | pass |
| R2-b1-kv8192 | fi-graph:graph-nopdl-cublas | 3 | 0.87/0.84/0.85 | 1.00 | pass |
| R2-b1-kv8192 | fi-graph:graph-pdl-best | 3 | 0.88/0.86/0.87 | 1.00 | pass |
| R2-b1-kv8192 | fi-graph:graph-pdl-cublas | 3 | 0.87/0.84/0.85 | 1.00 | pass |
| R2-b1-kv8192 | mk:auto | 3 | 0.49/0.48/0.49 | 1.00 | pass |
| R2-b1-kv8192 | mk:auto-direct | 3 | 0.49/0.48/0.49 | 1.00 | pass |
| R3-b8-kv4096 | fi-graph:graph-nopdl-best | 3 | 0.86/0.85/0.86 | 0.88 | pass |
| R3-b8-kv4096 | fi-graph:graph-nopdl-cublas | 3 | 0.83/0.83/0.83 | 0.88 | pass |
| R3-b8-kv4096 | fi-graph:graph-pdl-best | 3 | 0.86/0.85/0.86 | 0.88 | pass |
| R3-b8-kv4096 | fi-graph:graph-pdl-cublas | 3 | 0.83/0.83/0.83 | 0.88 | pass |
| R3-b8-kv4096 | mk:auto | 3 | 0.46/0.47/0.47 | 0.88 | pass |
| R3-b8-kv4096 | mk:auto-direct | 3 | 0.46/0.47/0.47 | 0.88 | pass |
| R6-b1-kv1024-q16 | fi-graph:graph-nopdl-best | 3 | 1.01/0.80/0.80 | 1.00 | pass |
| R6-b1-kv1024-q16 | fi-graph:graph-nopdl-cublas | 3 | 1.02/0.79/0.79 | 0.00 | pass |
| R6-b1-kv1024-q16 | fi-graph:graph-pdl-best | 3 | 1.01/0.80/0.80 | 1.00 | pass |
| R6-b1-kv1024-q16 | fi-graph:graph-pdl-cublas | 3 | 1.02/0.79/0.79 | 0.00 | pass |
| R6-b1-kv1024-q16 | mk:auto | 3 | 0.53/0.45/0.45 | 1.00 | pass |
| R6-b1-kv1024-q16 | mk:auto-direct | 3 | 0.51/0.46/0.47 | 1.00 | pass |
| R6-b1-kv1024-q4 | fi-graph:graph-nopdl-best | 3 | 0.93/0.83/0.83 | 1.00 | pass |
| R6-b1-kv1024-q4 | fi-graph:graph-nopdl-cublas | 3 | 0.89/0.81/0.81 | 1.00 | pass |
| R6-b1-kv1024-q4 | fi-graph:graph-pdl-best | 3 | 0.93/0.83/0.83 | 1.00 | pass |
| R6-b1-kv1024-q4 | fi-graph:graph-pdl-cublas | 3 | 0.89/0.81/0.81 | 1.00 | pass |
| R6-b1-kv1024-q4 | mk:auto | 3 | 0.47/0.44/0.44 | 1.00 | pass |
| R6-b1-kv1024-q4 | mk:auto-direct | 3 | 0.47/0.43/0.43 | 1.00 | pass |

## Skips and errors

- SKIP fi-graph:eager-pdl-best @ R1-b1-kv128: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R1-b2-kv128: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R1-b4-kv128: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R1-b8-kv128: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R1-b16-kv128: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R1-b32-kv128: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R1-b64-kv128: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R1-b128-kv128: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R1-b256-kv128: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R2-b1-kv128: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R2-b1-kv512: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R2-b1-kv1024: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R2-b1-kv2048: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R2-b1-kv4096: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R2-b1-kv8192: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R2-b1-kv16384: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R2-b1-kv32768: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R3-b1-kv1024: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R3-b1-kv4096: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R3-b1-kv16384: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R3-b8-kv1024: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R3-b8-kv4096: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R3-b32-kv1024: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R6-b1-kv1024-q2: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R6-b1-kv1024-q4: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R6-b1-kv1024-q8: eager variant is only measured at R0
- SKIP fi-graph:eager-pdl-best @ R6-b1-kv1024-q16: eager variant is only measured at R0
