| point | queue arm | u/CTA q / static | total A+C excess (decomp job) | A(e3) excess | 1a balance gain (A at e0 minus ring) | 1b ring / unit fetch (e0 slope x du) | 2 EOI barriers | 3 EOI smem merge | 4 store (best aligned) | 5 store alignment (528 B vs best) | 6 combine parallelism (head - wide best) | 7 combine S-growth (wide best) | residual | A noise between 2 replicate jobs (max |diff| of any X_v) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b8kv32768 | queue C1024 cap32 | 12.5 / 1.4 | -0.93 | -2.91 (replicates -3.36) | -5.41 | +1.14 | +0.08 | +0.14 | +0.81 | -0.12 (e3/1024B) | -0.10 (k1/k1) | +2.08 | +0.45 | 0.45 |
| b8kv32768 | queue_pf C1024 cap32 | 12.5 / 1.4 | +0.03 | -1.92 (replicates -2.34) | -3.81 | +0.63 | -0.01 | +0.24 | +0.72 | -0.10 (e3/576B) | -0.16 (k1/k1) | +2.11 | +0.42 | 0.49 |
| b1kv245760 | queue C1024 cap32 | 1.7 / 1.0 | -1.15 | -2.43 (replicates -2.55) | -3.60 | +0.04 | +0.23 | -0.17 | +0.48 | +0.47 (e3/544B) | -0.86 (k4/k1) | +2.14 | +0.12 | 0.64 |
| b1kv245760 | queue_pf C1024 cap32 | 1.7 / 1.0 | +1.54 | +0.26 (replicates -0.49) | -1.17 | -0.01 | +0.16 | -0.10 | +0.18 | +0.46 (e3/544B) | -0.93 (k4/k1) | +2.21 | +0.75 | 1.50 |
| ragged64 | queue C1024 cap32 | 20.5 / 4.5 | -15.62 | -19.78 (replicates -19.70) | -21.48 | +2.40 | -0.19 | -0.37 | -0.25 | +0.18 (e3/1024B) | +1.79 (k1/k1) | +2.37 | -0.07 | 0.53 |
| ragged64 | queue_pf C1024 cap32 | 20.5 / 4.5 | -12.86 | -17.02 (replicates -16.93) | -17.53 | +1.08 | -0.13 | -0.34 | -0.29 | +0.29 (e3/1024B) | +1.82 (k1/k1) | +2.34 | -0.10 | 0.40 |
