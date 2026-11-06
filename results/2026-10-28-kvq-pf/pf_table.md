# jobs ['12301', '12302', '12303', '12304']; configs 1176; check non-PASS: 0

## 1. dA from RPF (queue_pf, A us, RPF 3 - RPF 0; median over replicate jobs; per point the config with the lowest A in the RPF 0 build, and the best RPF 3 config)
| point | ACC | arm | cfg (rpf0) | A rpf0 | A rpf3 (same cfg) | dA same cfg | best rpf3 cfg | A best rpf3 | dA best-best |
|---|---|---|---|---|---|---|---|---|---|
| b8kv4096 | 0 | queue_pf | C1024 capnone | 24.2 | 24.0 | -0.2 | C1024 cap32 | 24.0 | -0.2 |
| b8kv4096 | 0 | queue | C1024 cap32 | 23.9 | 24.1 | +0.2 | C1024 cap32 | 24.1 | +0.2 |
| b8kv4096 | 0 | static | - | 23.4 | 23.4 | -0.1 | - | 23.4 | -0.1 |
| b8kv16384 | 0 | queue_pf | C512 cap32 | 79.2 | 78.6 | -0.6 | C512 capnone | 77.2 | -2.0 |
| b8kv16384 | 0 | queue | C512 cap32 | 78.4 | 78.8 | +0.4 | C512 cap32 | 78.8 | +0.4 |
| b8kv16384 | 0 | static | - | 82.5 | 82.7 | +0.2 | - | 82.7 | +0.2 |
| b8kv32768 | 0 | queue_pf | C1024 capnone | 151.9 | 149.9 | -2.0 | C256 capnone | 148.8 | -3.1 |
| b8kv32768 | 0 | queue | C1024 capnone | 151.4 | 151.7 | +0.3 | C1024 capnone | 151.7 | +0.3 |
| b8kv32768 | 0 | static | - | 154.9 | 155.1 | +0.2 | - | 155.1 | +0.2 |
| b8kv30720 | 0 | queue_pf | C1024 cap32 | 144.7 | 142.5 | -2.3 | C512 capnone | 139.5 | -5.2 |
| b8kv30720 | 0 | queue | C1024 cap32 | 143.9 | 144.3 | +0.4 | C1024 capnone | 144.2 | +0.3 |
| b8kv30720 | 0 | static | - | 146.3 | 146.5 | +0.2 | - | 146.5 | +0.2 |
| b1kv245760 | 0 | queue_pf | C1024 capnone | 143.0 | 140.7 | -2.3 | C512 capnone | 140.4 | -2.6 |
| b1kv245760 | 0 | queue | C1024 capnone | 143.1 | 143.2 | +0.1 | C1024 capnone | 143.2 | +0.1 |
| b1kv245760 | 0 | static | - | 148.1 | 148.3 | +0.2 | - | 148.3 | +0.2 |
| b1kv32768 | 0 | queue_pf | C1024 capnone | 24.1 | 23.9 | -0.3 | C1024 capnone | 23.9 | -0.3 |
| b1kv32768 | 0 | queue | C1024 capnone | 24.0 | 24.1 | +0.1 | C1024 capnone | 24.1 | +0.1 |
| b1kv32768 | 0 | static | - | 23.6 | 23.6 | +0.0 | - | 23.6 | +0.0 |
| b64kv1024 | 0 | queue_pf | C512 capnone | 45.4 | 43.4 | -2.0 | C512 cap32 | 43.4 | -2.0 |
| b64kv1024 | 0 | queue | C512 capnone | 45.3 | 45.9 | +0.6 | C512 capnone | 45.9 | +0.6 |
| b64kv1024 | 0 | static | - | 47.2 | 47.2 | +0.0 | - | 47.2 | +0.0 |
| b128kv512 | 0 | queue_pf | C512 capnone | 48.2 | 46.7 | -1.6 | C512 capnone | 46.7 | -1.6 |
| b128kv512 | 0 | queue | C1024 cap32 | 48.4 | 49.3 | +0.9 | C512 cap32 | 49.3 | +0.9 |
| b128kv512 | 0 | static | - | 46.6 | 46.8 | +0.1 | - | 46.8 | +0.1 |
| ragged8 | 0 | queue_pf | C512 capnone | 93.4 | 91.3 | -2.2 | C512 capnone | 91.3 | -2.2 |
| ragged8 | 0 | queue | C512 capnone | 92.9 | 93.5 | +0.6 | C512 capnone | 93.5 | +0.6 |
| ragged8 | 0 | static | - | 100.5 | 100.3 | -0.2 | - | 100.3 | -0.2 |
| ragged64 | 0 | queue_pf | C512 capnone | 205.8 | 202.0 | -3.8 | C256 capnone | 201.7 | -4.1 |
| ragged64 | 0 | queue | C1024 capnone | 204.0 | 204.8 | +0.8 | C1024 cap32 | 204.5 | +0.5 |
| ragged64 | 0 | static | - | 223.7 | 223.0 | -0.8 | - | 223.0 | -0.8 |
| b8kv4096 | 1 | queue_pf | C1024 cap32 | 24.2 | 24.0 | -0.2 | C1024 cap32 | 24.0 | -0.2 |
| b8kv4096 | 1 | queue | C1024 capnone | 23.9 | 24.2 | +0.3 | C1024 cap32 | 24.2 | +0.3 |
| b8kv4096 | 1 | static | - | 23.3 | 23.4 | +0.1 | - | 23.4 | +0.1 |
| b8kv16384 | 1 | queue_pf | C512 cap32 | 79.0 | 78.8 | -0.1 | C512 capnone | 77.4 | -1.5 |
| b8kv16384 | 1 | queue | C512 cap32 | 78.1 | 79.0 | +0.9 | C512 cap32 | 79.0 | +0.9 |
| b8kv16384 | 1 | static | - | 82.2 | 82.7 | +0.6 | - | 82.7 | +0.6 |
| b8kv32768 | 1 | queue_pf | C1024 capnone | 151.6 | 150.1 | -1.5 | C256 capnone | 148.7 | -2.9 |
| b8kv32768 | 1 | queue | C512 cap32 | 151.0 | 152.3 | +1.3 | C1024 capnone | 151.8 | +0.8 |
| b8kv32768 | 1 | static | - | 154.5 | 155.0 | +0.6 | - | 155.0 | +0.6 |
| b8kv30720 | 1 | queue_pf | C1024 capnone | 144.2 | 142.6 | -1.6 | C512 capnone | 140.0 | -4.2 |
| b8kv30720 | 1 | queue | C512 capnone | 143.2 | 145.4 | +2.2 | C1024 capnone | 144.4 | +1.2 |
| b8kv30720 | 1 | static | - | 146.1 | 146.4 | +0.4 | - | 146.4 | +0.4 |
| b1kv245760 | 1 | queue_pf | C1024 capnone | 142.9 | 140.9 | -2.0 | C512 capnone | 140.4 | -2.5 |
| b1kv245760 | 1 | queue | C1024 capnone | 142.9 | 143.2 | +0.3 | C1024 capnone | 143.2 | +0.3 |
| b1kv245760 | 1 | static | - | 147.4 | 147.8 | +0.4 | - | 147.8 | +0.4 |
| b1kv32768 | 1 | queue_pf | C1024 capnone | 23.9 | 24.1 | +0.2 | C1024 capnone | 24.1 | +0.2 |
| b1kv32768 | 1 | queue | C1024 capnone | 23.9 | 24.2 | +0.2 | C1024 capnone | 24.2 | +0.2 |
| b1kv32768 | 1 | static | - | 23.6 | 23.8 | +0.2 | - | 23.8 | +0.2 |
| b64kv1024 | 1 | queue_pf | C512 cap32 | 45.1 | 43.4 | -1.7 | C512 cap32 | 43.4 | -1.7 |
| b64kv1024 | 1 | queue | C512 cap32 | 45.1 | 45.9 | +0.8 | C512 cap32 | 45.9 | +0.8 |
| b64kv1024 | 1 | static | - | 47.1 | 47.4 | +0.3 | - | 47.4 | +0.3 |
| b128kv512 | 1 | queue_pf | C512 capnone | 48.2 | 46.6 | -1.5 | C512 capnone | 46.6 | -1.5 |
| b128kv512 | 1 | queue | C1024 capnone | 48.4 | 49.3 | +0.9 | C1024 capnone | 49.3 | +0.9 |
| b128kv512 | 1 | static | - | 46.7 | 46.7 | +0.0 | - | 46.7 | +0.0 |
| ragged8 | 1 | queue_pf | C512 capnone | 93.0 | 91.4 | -1.6 | C512 capnone | 91.4 | -1.6 |
| ragged8 | 1 | queue | C512 capnone | 92.5 | 93.5 | +1.0 | C512 capnone | 93.5 | +1.0 |
| ragged8 | 1 | static | - | 100.1 | 100.1 | +0.0 | - | 100.1 | +0.0 |
| ragged64 | 1 | queue_pf | C512 capnone | 205.4 | 202.8 | -2.6 | C256 capnone | 202.6 | -2.9 |
| ragged64 | 1 | queue | C1024 capnone | 204.0 | 205.2 | +1.2 | C1024 capnone | 205.2 | +1.2 |
| ragged64 | 1 | static | - | 223.9 | 223.3 | -0.6 | - | 223.3 | -0.6 |
| b8kv4096 | 3 | queue_pf | C1024 cap32 | 24.3 | 24.1 | -0.1 | C1024 cap32 | 24.1 | -0.1 |
| b8kv4096 | 3 | queue | C1024 cap32 | 24.1 | 24.4 | +0.3 | C1024 capnone | 24.4 | +0.3 |
| b8kv4096 | 3 | static | - | 23.5 | 23.5 | -0.0 | - | 23.5 | -0.0 |
| b8kv16384 | 3 | queue_pf | C512 cap32 | 79.7 | 79.0 | -0.7 | C512 capnone | 77.7 | -1.9 |
| b8kv16384 | 3 | queue | C512 cap32 | 78.9 | 79.3 | +0.4 | C512 cap32 | 79.3 | +0.4 |
| b8kv16384 | 3 | static | - | 82.4 | 82.5 | +0.1 | - | 82.5 | +0.1 |
| b8kv32768 | 3 | queue_pf | C1024 capnone | 152.0 | 150.3 | -1.7 | C512 capnone | 149.6 | -2.4 |
| b8kv32768 | 3 | queue | C1024 capnone | 151.3 | 152.0 | +0.6 | C1024 capnone | 152.0 | +0.6 |
| b8kv32768 | 3 | static | - | 154.8 | 155.2 | +0.4 | - | 155.2 | +0.4 |
| b8kv30720 | 3 | queue_pf | C1024 cap32 | 144.9 | 142.7 | -2.2 | C512 capnone | 140.3 | -4.6 |
| b8kv30720 | 3 | queue | C1024 capnone | 143.8 | 144.5 | +0.6 | C1024 cap32 | 144.5 | +0.6 |
| b8kv30720 | 3 | static | - | 146.4 | 146.7 | +0.3 | - | 146.7 | +0.3 |
| b1kv245760 | 3 | queue_pf | C1024 capnone | 142.6 | 141.1 | -1.5 | C512 capnone | 140.7 | -1.9 |
| b1kv245760 | 3 | queue | C1024 capnone | 142.7 | 142.8 | +0.2 | C1024 capnone | 142.8 | +0.2 |
| b1kv245760 | 3 | static | - | 147.5 | 147.7 | +0.2 | - | 147.7 | +0.2 |
| b1kv32768 | 3 | queue_pf | C1024 capnone | 24.3 | 24.3 | -0.0 | C1024 capnone | 24.3 | -0.0 |
| b1kv32768 | 3 | queue | C1024 capnone | 24.1 | 24.4 | +0.3 | C1024 capnone | 24.4 | +0.3 |
| b1kv32768 | 3 | static | - | 23.8 | 23.8 | +0.0 | - | 23.8 | +0.0 |
| b64kv1024 | 3 | queue_pf | C512 cap32 | 45.4 | 43.6 | -1.8 | C512 capnone | 43.5 | -1.9 |
| b64kv1024 | 3 | queue | C512 capnone | 45.4 | 46.2 | +0.8 | C512 cap32 | 46.1 | +0.7 |
| b64kv1024 | 3 | static | - | 47.2 | 47.3 | +0.1 | - | 47.3 | +0.1 |
| b128kv512 | 3 | queue_pf | C1024 cap32 | 48.3 | 46.7 | -1.6 | C1024 cap32 | 46.7 | -1.6 |
| b128kv512 | 3 | queue | C1024 capnone | 48.5 | 49.4 | +0.9 | C512 cap32 | 49.3 | +0.9 |
| b128kv512 | 3 | static | - | 46.8 | 46.8 | -0.0 | - | 46.8 | -0.0 |
| ragged8 | 3 | queue_pf | C512 capnone | 93.8 | 91.5 | -2.3 | C512 capnone | 91.5 | -2.3 |
| ragged8 | 3 | queue | C512 capnone | 93.0 | 93.9 | +0.8 | C512 capnone | 93.9 | +0.8 |
| ragged8 | 3 | static | - | 100.8 | 100.3 | -0.4 | - | 100.3 | -0.4 |
| ragged64 | 3 | queue_pf | C512 capnone | 206.4 | 202.9 | -3.5 | C512 capnone | 202.9 | -3.5 |
| ragged64 | 3 | queue | C1024 cap32 | 204.4 | 205.1 | +0.7 | C1024 cap32 | 205.1 | +0.7 |
| ragged64 | 3 | static | - | 223.7 | 222.7 | -1.0 | - | 222.7 | -1.0 |

## 2. best A + C (us) per point; static = ACC 0 RPF 0 (mk today), median of replicate jobs; vs static in %
| point | static A + C | queue_pf RPF3 mk combine (ACC0) | vs | queue_pf RPF3 + R1 red.f32 | vs | queue_pf RPF3 + R1 red.v4 | vs | queue_pf RPF0 + R1 red.v4 |
|---|---|---|---|---|---|---|---|---|
| b8kv4096 | 25.5 | 26.2 (C1024 cap32; A 24.0 C 2.22) | +2.5% | 25.4 (C1024 cap32; A 24.0 C 1.40) | -0.6% | 25.6 (C1024 cap32; A 24.1 C 1.40) | +0.1% | 25.7 |
| b8kv16384 | 84.6 | 81.2 (C1024 capnone; A 78.5 C 2.78) | -4.0% | 78.8 (C512 capnone; A 77.4 C 1.40) | -6.8% | 79.1 (C512 capnone; A 77.7 C 1.38) | -6.5% | 81.1 |
| b8kv32768 | 157.0 | 154.1 (C1024 capnone; A 149.9 C 4.24) | -1.8% | 150.1 (C256 capnone; A 148.7 C 1.43) | -4.4% | 151.0 (C512 capnone; A 149.6 C 1.46) | -3.8% | 153.4 |
| b8kv30720 | 148.4 | 146.6 (C1024 capnone; A 142.3 C 4.18) | -1.2% | 141.4 (C512 capnone; A 140.0 C 1.43) | -4.7% | 141.7 (C512 capnone; A 140.3 C 1.49) | -4.5% | 146.3 |
| b1kv245760 | 150.6 | 151.5 (C1024 cap32; A 147.8 C 3.74) | +0.6% | 141.6 (C512 capnone; A 140.4 C 1.24) | -6.0% | 142.0 (C512 capnone; A 140.7 C 1.24) | -5.7% | 144.0 |
| b1kv32768 | 26.0 | 27.5 (C1024 capnone; A 23.9 C 3.67) | +6.1% | 25.4 (C1024 capnone; A 24.1 C 1.25) | -2.2% | 25.5 (C1024 capnone; A 24.3 C 1.19) | -1.6% | 25.5 |
| b64kv1024 | 47.4 | 47.0 (C1024 cap32; A 46.8 C 0.20) | -0.9% | 46.5 (C512 cap32; A 43.4 C 3.08) | -1.9% | 46.7 (C512 cap32; A 43.6 C 3.10) | -1.6% | 47.1 |
| b128kv512 | 46.9 | 46.9 (C1024 cap32; A 46.7 C 0.22) | +0.0% | 46.9 (C512 capnone; A 46.6 C 0.19) | -0.1% | 46.9 (C1024 capnone; A 46.7 C 0.22) | +0.0% | 48.6 |
| ragged8 | 102.8 | 103.5 (C1024 capnone; A 95.3 C 8.20) | +0.8% | 92.9 (C512 capnone; A 91.4 C 1.49) | -9.6% | 93.0 (C512 capnone; A 91.5 C 1.45) | -9.6% | 95.3 |
| ragged64 | 228.9 | 214.9 (C1024 capnone; A 205.4 C 9.50) | -6.1% | 205.8 (C256 capnone; A 202.6 C 3.14) | -10.1% | 206.1 (C512 capnone; A 202.9 C 3.22) | -9.9% | 209.6 |

## 2b. same, A only and static + R1 (static row of each ACC build)
| point | static A | static C acc0 | static C acc1 | static C acc3 | static A+C acc3 | queue_pf RPF3 best A (acc3 build) | vs static A |
|---|---|---|---|---|---|---|---|
| b8kv4096 | 23.4 | 2.05 | 1.38 | 1.30 | 24.9 | 24.1 (C1024 cap32) | +3.1% |
| b8kv16384 | 82.5 | 2.12 | 1.38 | 1.37 | 83.8 | 77.7 (C512 capnone) | -5.7% |
| b8kv32768 | 154.9 | 2.13 | 1.42 | 1.34 | 156.2 | 149.6 (C512 capnone) | -3.4% |
| b8kv30720 | 146.3 | 2.13 | 1.39 | 1.33 | 147.7 | 140.3 (C512 capnone) | -4.1% |
| b1kv245760 | 148.1 | 2.41 | 1.26 | 1.28 | 148.7 | 140.7 (C512 capnone) | -5.0% |
| b1kv32768 | 23.6 | 2.38 | 1.22 | 1.21 | 25.1 | 24.3 (C1024 capnone) | +3.0% |
| b64kv1024 | 47.2 | 0.23 | 0.23 | 0.22 | 47.4 | 43.5 (C512 capnone) | -7.8% |
| b128kv512 | 46.6 | 0.23 | 0.22 | 0.22 | 47.1 | 46.7 (C1024 cap32) | +0.1% |
| ragged8 | 100.5 | 2.19 | 1.42 | 1.34 | 102.1 | 91.5 (C512 capnone) | -9.0% |
| ragged64 | 223.7 | 5.19 | 2.79 | 2.82 | 226.6 | 202.9 (C512 capnone) | -9.3% |

## 3. replicate noise: per variant, |job1 - job2| of A for every config (median / max, us); static rows separately
a0r0: jobs ['12301', '12303']  A: median 0.13 max 1.12; static A: median 0.23 max 0.64; C: median 0.08 max 0.30
a0r3: jobs ['12301', '12303']  A: median 0.11 max 0.80; static A: median 0.11 max 0.40; C: median 0.10 max 0.26
a1r0: jobs ['12301', '12303']  A: median 0.13 max 0.99; static A: median 0.21 max 0.56; C: median 0.08 max 0.26
a1r3: jobs ['12302', '12304']  A: median 0.10 max 0.85; static A: median 0.21 max 0.62; C: median 0.07 max 0.24
a3r0: jobs ['12302', '12304']  A: median 0.16 max 1.01; static A: median 0.12 max 0.70; C: median 0.08 max 0.26
a3r3: jobs ['12302', '12304']  A: median 0.13 max 0.83; static A: median 0.13 max 0.53; C: median 0.08 max 0.32
