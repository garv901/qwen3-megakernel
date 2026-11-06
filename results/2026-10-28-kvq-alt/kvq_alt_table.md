| point | design | config | A | C | A + C | vs static A + C | S_max |
|---|---|---|---|---|---|---|---|
| b8kv4096 | static (11854) | static | 23.4 | 2.05 | 25.4 | +0.0% | 4 |
| b8kv4096 | best queue 11854 | queue C1024 cap64 | 23.9 | 2.18 | 26.0 | +2.4% | 4 |
| b8kv4096 | 11854 ticket (queue_pf cap32) | queue_pf C1024 cap32 | 28.0 | 0.19 | 28.2 | +11.0% | 4 |
| b8kv4096 | control: queue same job (slots) | queue C1024 cap32 | 23.9 | 2.05 | 26.0 | +2.3% | 4 |
| b8kv4096 | static + R1 red.f32 | static | 23.5 | 1.36 | 24.8 | -2.2% | 4 |
| b8kv4096 | R1 red.f32 | queue C1024 cap32 | 23.9 | 1.38 | 25.2 | -0.6% | 4 |
| b8kv4096 | R1 red.v4 | queue C1024 capnone | 24.1 | 1.36 | 25.5 | +0.3% | 4 |
| b8kv4096 | static + R1 TMA-reduce | static | 23.5 | 1.38 | 24.8 | -2.1% | 4 |
| b8kv4096 | R1 TMA-reduce | queue C1024 cap64 | 24.1 | 1.38 | 25.5 | +0.3% | 4 |
| b8kv4096 | R2 tail split | tail f20% k4 | 23.8 | 2.13 | 25.9 | +2.1% | 7 |
| b8kv4096 | R3 merge queue pos | queue C1024 cap32 | 27.5 | 0.19 | 27.6 | +8.9% | 4 |
| b8kv4096 | R3 merge queue pair | queue C1024 cap32/pair | 27.6 | 0.22 | 27.8 | +9.5% | 4 |
| b8kv4096 | static + R3 | static | 26.3 | 0.18 | 26.5 | +4.3% | 4 |
| b8kv16384 | static (11854) | static | 82.6 | 2.02 | 84.6 | +0.0% | 4 |
| b8kv16384 | best queue 11854 | queue C512 cap32 | 78.1 | 3.87 | 82.0 | -3.1% | 26 |
| b8kv16384 | 11854 ticket (queue_pf cap32) | queue_pf C1024 cap32 | 90.7 | 0.18 | 90.9 | +7.4% | 16 |
| b8kv16384 | control: queue same job (slots) | queue C512 cap32 | 78.3 | 3.79 | 82.1 | -3.0% | 26 |
| b8kv16384 | static + R1 red.f32 | static | 82.4 | 1.42 | 83.9 | -0.8% | 4 |
| b8kv16384 | R1 red.f32 | queue C256 cap32 | 78.1 | 1.57 | 79.7 | -5.8% | 26 |
| b8kv16384 | R1 red.v4 | queue C512 cap32 | 78.5 | 1.55 | 79.9 | -5.6% | 26 |
| b8kv16384 | static + R1 TMA-reduce | static | 82.5 | 1.44 | 84.1 | -0.7% | 4 |
| b8kv16384 | R1 TMA-reduce | queue C512 cap32 | 78.6 | 1.47 | 80.0 | -5.5% | 26 |
| b8kv16384 | R2 tail split | tail f5% k8 | 81.3 | 2.27 | 83.5 | -1.3% | 10 |
| b8kv16384 | R3 merge queue pos | queue C1024 cap32 | 86.0 | 0.22 | 86.3 | +1.9% | 16 |
| b8kv16384 | R3 merge queue pair | queue_pf C1024 cap32/pair | 87.0 | 0.19 | 87.3 | +3.2% | 16 |
| b8kv16384 | static + R3 | static | 85.2 | 0.13 | 85.3 | +0.8% | 4 |
| b8kv32768 | static (11854) | static | 154.5 | 2.05 | 156.5 | +0.0% | 4 |
| b8kv32768 | best queue 11854 | queue C512 cap32 | 151.4 | 3.97 | 155.3 | -0.8% | 29 |
| b8kv32768 | 11854 ticket (queue_pf cap32) | queue_pf C1024 cap32 | 174.2 | 0.13 | 174.3 | +11.4% | 29 |
| b8kv32768 | control: queue same job (slots) | queue C1024 capnone | 151.1 | 4.26 | 155.3 | -0.8% | 32 |
| b8kv32768 | static + R1 red.f32 | static | 154.7 | 1.47 | 156.1 | -0.2% | 4 |
| b8kv32768 | R1 red.f32 | queue C256 cap64 | 149.5 | 1.54 | 151.1 | -3.5% | 52 |
| b8kv32768 | R1 red.v4 | queue C512 cap64 | 149.9 | 1.52 | 151.3 | -3.3% | 52 |
| b8kv32768 | static + R1 TMA-reduce | static | 155.0 | 1.41 | 156.4 | -0.1% | 4 |
| b8kv32768 | R1 TMA-reduce | queue C512 cap64 | 149.9 | 1.58 | 151.4 | -3.3% | 52 |
| b8kv32768 | R2 tail split | tail f10% k8 | 154.8 | 2.40 | 157.2 | +0.4% | 12 |
| b8kv32768 | R3 merge queue pos | queue C1024 cap32 | 163.0 | 0.10 | 163.1 | +4.2% | 29 |
| b8kv32768 | R3 merge queue pair | queue C256 cap32/pair | 168.7 | 0.10 | 168.8 | +7.8% | 29 |
| b8kv32768 | static + R3 | static | 157.9 | 0.13 | 158.0 | +0.9% | 4 |
| b8kv30720 | static (11854) | static | 146.2 | 1.98 | 148.2 | +0.0% | 4 |
| b8kv30720 | best queue 11854 | queue C1024 cap32 | 143.7 | 4.02 | 147.7 | -0.3% | 30 |
| b8kv30720 | 11854 ticket (queue_pf cap32) | queue_pf C1024 cap32 | 166.1 | 0.13 | 166.3 | +12.2% | 30 |
| b8kv30720 | control: queue same job (slots) | queue C1024 cap32 | 143.9 | 4.03 | 147.9 | -0.2% | 30 |
| b8kv30720 | static + R1 red.f32 | static | 145.9 | 1.44 | 147.4 | -0.6% | 4 |
| b8kv30720 | R1 red.f32 | queue C1024 cap64 | 143.3 | 1.54 | 144.8 | -2.3% | 30 |
| b8kv30720 | R1 red.v4 | queue C1024 capnone | 143.4 | 1.54 | 144.9 | -2.3% | 30 |
| b8kv30720 | static + R1 TMA-reduce | static | 146.3 | 1.47 | 147.6 | -0.4% | 4 |
| b8kv30720 | R1 TMA-reduce | queue C1024 cap32 | 143.6 | 1.50 | 145.2 | -2.1% | 30 |
| b8kv30720 | R2 tail split | tail f5% k8 | 146.2 | 2.40 | 148.6 | +0.2% | 12 |
| b8kv30720 | R3 merge queue pos | queue C1024 cap32 | 155.3 | 0.13 | 155.4 | +4.8% | 30 |
| b8kv30720 | R3 merge queue pair | queue_pf C256 cap32/pair | 159.6 | 0.10 | 159.7 | +7.7% | 30 |
| b8kv30720 | static + R3 | static | 149.2 | 0.10 | 149.3 | +0.7% | 4 |
| b1kv245760 | static (11854) | static | 147.5 | 2.27 | 149.8 | +0.0% | 19 |
| b1kv245760 | best queue 11854 | queue C1024 cap32 | 144.6 | 3.52 | 148.1 | -1.1% | 32 |
| b1kv245760 | 11854 ticket (queue_pf cap32) | queue_pf C512 cap32 | 151.5 | 0.13 | 151.6 | +1.2% | 32 |
| b1kv245760 | control: queue same job (slots) | queue C512 cap32 | 144.4 | 3.54 | 148.0 | -1.2% | 32 |
| b1kv245760 | static + R1 red.f32 | static | 147.2 | 1.23 | 148.4 | -1.0% | 19 |
| b1kv245760 | R1 red.f32 | queue C1024 capnone | 143.1 | 1.33 | 144.4 | -3.6% | 240 |
| b1kv245760 | R1 red.v4 | queue_pf C1024 capnone | 142.2 | 1.28 | 143.5 | -4.2% | 240 |
| b1kv245760 | static + R1 TMA-reduce | static | 148.0 | 1.25 | 149.3 | -0.4% | 19 |
| b1kv245760 | R1 TMA-reduce | queue_pf C1024 capnone | 142.2 | 1.34 | 143.6 | -4.2% | 240 |
| b1kv245760 | R2 tail split | tail f5% k8 | 154.4 | 3.33 | 157.8 | +5.3% | 27 |
| b1kv245760 | R3 merge queue pos | queue C256 cap32 | 149.6 | 0.13 | 149.8 | -0.0% | 32 |
| b1kv245760 | R3 merge queue pair | queue C256 cap32/pair | 152.0 | 0.10 | 152.1 | +1.5% | 32 |
| b1kv245760 | static + R3 | static | 150.4 | 0.10 | 150.5 | +0.5% | 19 |
| b1kv32768 | static (11854) | static | 23.7 | 2.38 | 26.1 | +0.0% | 19 |
| b1kv32768 | best queue 11854 | queue_pf C1024 cap64 | 24.1 | 3.68 | 27.5 | +5.2% | 32 |
| b1kv32768 | 11854 ticket (queue_pf cap32) | queue_pf C256 cap32 | 29.2 | 0.16 | 29.3 | +12.1% | 29 |
| b1kv32768 | control: queue same job (slots) | queue C1024 capnone | 23.9 | 3.60 | 27.5 | +5.1% | 32 |
| b1kv32768 | static + R1 red.f32 | static | 23.5 | 1.12 | 24.8 | -5.1% | 19 |
| b1kv32768 | R1 red.f32 | queue C512 cap64 | 22.6 | 1.12 | 23.8 | -9.1% | 52 |
| b1kv32768 | R1 red.v4 | queue C512 cap64 | 22.8 | 1.12 | 23.9 | -8.5% | 52 |
| b1kv32768 | static + R1 TMA-reduce | static | 23.7 | 1.12 | 24.9 | -4.6% | 19 |
| b1kv32768 | R1 TMA-reduce | queue C256 cap64 | 22.8 | 1.14 | 23.9 | -8.4% | 52 |
| b1kv32768 | R2 tail split | tail f5% k8 | 24.9 | 3.36 | 28.2 | +7.8% | 27 |
| b1kv32768 | R3 merge queue pos | queue C256 cap32 | 29.5 | 0.16 | 29.6 | +13.3% | 29 |
| b1kv32768 | R3 merge queue pair | queue C512 cap32/pair | 29.3 | 0.22 | 29.6 | +13.3% | 29 |
| b1kv32768 | static + R3 | static | 26.7 | 0.13 | 26.9 | +3.0% | 19 |
| b64kv1024 | static (11854) | static | 47.1 | 0.19 | 47.3 | +0.0% | 1 |
| b64kv1024 | best queue 11854 | queue_pf C1024 capnone | 46.8 | 0.22 | 47.0 | -0.5% | 1 |
| b64kv1024 | 11854 ticket (queue_pf cap32) | queue_pf C1024 cap32 | 46.9 | 0.19 | 47.1 | -0.3% | 1 |
| b64kv1024 | control: queue same job (slots) | queue C1024 cap32 | 46.8 | 0.19 | 47.1 | -0.5% | 1 |
| b64kv1024 | static + R1 red.f32 | static | 47.2 | 0.13 | 47.3 | +0.1% | 1 |
| b64kv1024 | R1 red.f32 | hybrid C2048 capnone | 43.7 | 3.10 | 46.8 | -1.0% | 4 |
| b64kv1024 | R1 red.v4 | hybrid C2048 cap64 | 43.9 | 3.17 | 46.9 | -0.8% | 4 |
| b64kv1024 | static + R1 TMA-reduce | static | 47.6 | 0.13 | 47.7 | +0.9% | 1 |
| b64kv1024 | R1 TMA-reduce | queue C1024 cap64 | 47.1 | 0.10 | 47.2 | -0.2% | 1 |
| b64kv1024 | R2 tail split | tail f20% k4 | 44.5 | 6.66 | 51.1 | +8.1% | 2 |
| b64kv1024 | R3 merge queue pos | queue C1024 cap32 | 47.4 | 0.16 | 47.5 | +0.5% | 1 |
| b64kv1024 | R3 merge queue pair | queue_pf C1024 cap32/pair | 47.4 | 0.16 | 47.6 | +0.5% | 1 |
| b64kv1024 | static + R3 | static | 47.4 | 0.22 | 47.6 | +0.6% | 1 |
| b128kv512 | static (11854) | static | 46.5 | 0.19 | 46.7 | +0.0% | 1 |
| b128kv512 | best queue 11854 | queue_pf C1024 cap64 | 48.1 | 0.16 | 48.3 | +3.4% | 1 |
| b128kv512 | 11854 ticket (queue_pf cap32) | queue_pf C512 cap32 | 48.2 | 0.18 | 48.4 | +3.7% | 1 |
| b128kv512 | control: queue same job (slots) | queue_pf C512 capnone | 48.2 | 0.16 | 48.4 | +3.6% | 1 |
| b128kv512 | static + R1 red.f32 | static | 46.9 | 0.13 | 47.0 | +0.8% | 1 |
| b128kv512 | R1 red.f32 | queue_pf C512 cap64 | 48.3 | 0.13 | 48.4 | +3.7% | 1 |
| b128kv512 | R1 red.v4 | queue_pf C1024 cap32 | 48.3 | 0.10 | 48.4 | +3.7% | 1 |
| b128kv512 | static + R1 TMA-reduce | static | 46.8 | 0.13 | 46.9 | +0.6% | 1 |
| b128kv512 | R1 TMA-reduce | queue_pf C1024 cap64 | 48.4 | 0.13 | 48.5 | +3.9% | 1 |
| b128kv512 | R2 tail split | tail f20% k8 | 50.5 | 12.50 | 63.0 | +34.9% | 2 |
| b128kv512 | R3 merge queue pos | queue_pf C1024 cap32 | 48.4 | 0.16 | 48.6 | +4.1% | 1 |
| b128kv512 | R3 merge queue pair | queue_pf C1024 cap32/pair | 48.4 | 0.19 | 48.6 | +4.1% | 1 |
| b128kv512 | static + R3 | static | 47.1 | 0.16 | 47.2 | +1.2% | 1 |
| ragged8 | static (11854) | static | 100.4 | 2.11 | 102.5 | +0.0% | 9 |
| ragged8 | best queue 11854 | queue C1024 cap64 | 91.2 | 7.04 | 98.2 | -4.2% | 57 |
| ragged8 | 11854 ticket (queue_pf cap32) | queue_pf C1024 cap32 | 112.9 | 0.13 | 113.1 | +10.3% | 31 |
| ragged8 | control: queue same job (slots) | queue C1024 cap64 | 91.3 | 5.98 | 97.3 | -5.0% | 57 |
| ragged8 | static + R1 red.f32 | static | 100.4 | 1.46 | 101.8 | -0.7% | 9 |
| ragged8 | R1 red.f32 | queue C1024 cap64 | 91.2 | 1.39 | 92.6 | -9.7% | 57 |
| ragged8 | R1 red.v4 | queue C1024 cap64 | 91.2 | 1.34 | 92.5 | -9.7% | 57 |
| ragged8 | static + R1 TMA-reduce | static | 101.1 | 1.47 | 102.4 | -0.0% | 9 |
| ragged8 | R1 TMA-reduce | queue C1024 cap64 | 91.4 | 1.41 | 92.8 | -9.4% | 57 |
| ragged8 | R2 tail split | tail f5% k8 | 96.2 | 2.37 | 98.5 | -3.9% | 17 |
| ragged8 | R3 merge queue pos | queue C1024 cap32 | 103.5 | 0.10 | 103.6 | +1.0% | 31 |
| ragged8 | R3 merge queue pair | queue C512 cap32/pair | 105.2 | 0.16 | 105.4 | +2.9% | 31 |
| ragged8 | static + R3 | static | 102.1 | 0.10 | 102.2 | -0.3% | 9 |
| ragged64 | static (11854) | static | 223.4 | 5.18 | 228.7 | +0.0% | 3 |
| ragged64 | best queue 11854 | queue C1024 cap32 | 203.9 | 9.38 | 213.3 | -6.8% | 30 |
| ragged64 | 11854 ticket (queue_pf cap32) | queue_pf C1024 cap32 | 231.5 | 0.13 | 231.7 | +1.3% | 30 |
| ragged64 | control: queue same job (slots) | queue C1024 cap64 | 203.8 | 9.34 | 213.2 | -6.8% | 30 |
| ragged64 | static + R1 red.f32 | static | 223.8 | 2.78 | 226.6 | -0.9% | 3 |
| ragged64 | R1 red.f32 | queue C1024 capnone | 204.0 | 3.20 | 207.2 | -9.4% | 30 |
| ragged64 | R1 red.v4 | queue C1024 capnone | 204.0 | 3.20 | 207.3 | -9.4% | 30 |
| ragged64 | static + R1 TMA-reduce | static | 224.3 | 2.78 | 227.1 | -0.7% | 3 |
| ragged64 | R1 TMA-reduce | queue C1024 cap64 | 204.2 | 3.17 | 207.4 | -9.3% | 30 |
| ragged64 | R2 tail split | tail f5% k8 | 214.1 | 6.78 | 220.9 | -3.4% | 11 |
| ragged64 | R3 merge queue pos | queue C1024 cap32 | 214.9 | 0.10 | 214.9 | -6.0% | 30 |
| ragged64 | R3 merge queue pair | queue C1024 cap32/pair | 220.6 | 0.10 | 220.6 | -3.5% | 30 |
| ragged64 | static + R3 | static | 226.4 | 0.13 | 226.6 | -0.9% | 3 |
