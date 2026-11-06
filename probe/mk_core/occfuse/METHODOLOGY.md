# occfuse: microbenchmarks for the occupancy-vs-fusion cost model

## Purpose
These benchmarks measure the terms of a cost model for one GEMM phase of a decode megakernel. The model will decide, before any kernel is built, when to trade occupancy for fusion:
- **Occupancy:** split K so that more SMs stream weights.
- **Fusion:** a fused epilogue with KS = 1, or a bf16 copy ready for TMA, which removes a separate pass or a finalizer.

Each bench measures one term, as a function of the axes that matter. This folder does not build the model. The fitted terms go to `params.json`, and the bench-by-bench status goes to `FINDINGS.md`.

The model terms, for one phase on one CTA c:
```
t_phase ≈ t_bar + max_c [ exposed_stream(S_c, N_active, R, window) ⊕ t_B(T, k, mode) ⊕ t_chain(SPT_c, TT) + t_epi(TT, T, mode, KS) ] + t_fin(T, placement)
```
`⊕` marks terms that can overlap; how much they overlap is itself measured (B8). The inputs come from `common/tiles.py`: S_c (weight bytes per CTA), SPT_c (k-steps), N_active (busy CTAs), and idle CTAs.

Background: `probe/mk_core/smallt/NOTES.md` (job 14155) and LEDGER rows P15 and P15b. Bench and mk disagreed before, because the bench had no barrier-window prefetch, ran at a different clock, and defined time differently. The rules below fix all three.

## Rules for every bench
1. **Geometry as mk:** grid = SM count (148), 1 CTA/SM, 512 threads, cooperative launch (`launch_coop`). Weight boxes of 128 rows × 64 k bf16 with SW128 (`tmap_bf16(..., 64, 128)`). One k-step = 128 k = 32 KB of A per 128-row block. tcgen05 cta_group::1, M = 128, weights on M, N = TT (multiple of 16, 16..256).
2. **Phases:** one launch runs `nph` phases (default 112), separated by mk's split barrier: `bar_arrive` → (thread 0: prefetch of the next phase) → `bar_wait`. Use `common/ub.cuh`; do not write a new barrier.
3. **Time definitions:** record per-CTA stamps (`st`, NSTMAX per CTA) on every launch and compute with `common/stamps.py`:
   - **T** = median exit − previous median exit.
   - **work, busy / idle, last, spread, lat** as defined there.
   - Skip the first phase of each launch.
   - Also report launch time / nph (CUDA events) as a check. When the two disagree by more than 5 %, say so.
4. **Data placement:**
   - A (weights) comes from the `Pool` (2 GiB, more than 2× L2), and each phase reads a new row window, so A always comes from HBM.
   - Activations (x, xb, outputs) stay L2-resident, as in decode: they are small and rewritten every phase.
   - If a bench needs cold activations, use `k_l2flush` and say so.
5. **Prefetch:** on by default. Thread 0 issues the next phase's first `min(R, steps)` A loads between `bar_arrive` and `bar_wait` (mk `prefetch_next` / `ts_first`). Every bench also has a `--pf 0` arm.
6. **Clock:**
   - Record `clk_sample` at the start and end of the kernel → MHz (`mhz_of`), plus the NVML clocks before and after (`job.sbatch`).
   - mk ran at about 1965 MHz NVML, and xover benches at about 1850 (P15b).
   - **Rule (B0):** clocks cannot be locked (`nvidia-smi -lgc` has no permission). Do not lock. Report `mhz` from `clk_sample` on every row. Under steady tcgen05 load the GPU is power-capped (clk_sample about 1700 MHz, NVML about 1865, throttle 0x4). Idle runs are at 1965. Convert to cycles only for pure-compute terms (B3); compare everything else in us, with MHz next to it.
7. **Statistics:**
   - 2 warm-up launches, ≥ 5 timed launches.
   - Report median plus min and max over launches, of the per-launch medians over phases.
   - The globaltimer steps by 32 ns, but the median over CTAs of the stamps falls on a 0.256 us grid (B0, job 14277; mk stamps do the same). Never quote a single phase. For differences below 0.25 us, use `Tmean` from `stamps.metrics` (a telescoping mean over ≥ 50 phases).
   - Repeat the gate points in a second job (another node placement, if the queue gives one).
8. **Correctness:**
   - Any arm that writes values checks a sample against a CUDA-core reference (pattern: `k_ref` in `probe/mk_core/xover4/xover4.cu`). FAIL rows are excluded from fits.
   - Oracle arms (MMA off, epilogue off, B off, A off, TMEM reads off) are named in JSON with `"orc"` bits, as in xover4.
9. **Generality:**
   - Shapes N, K, T, TT, KS, H, the ring depth R and the active CTA set are command-line arguments.
   - The Qwen3-0.6B shapes are one config set, passed by the run script: QKV 4096×1024, O 1024×2048, GU 6144×1024, D 1024×3072, LM 151936×1024, H 1024.
   - Do not hardcode model numbers in the kernels.
10. **Output:**
    - JSONL to stdout, one row per (arm, config), with: `bench`, `arm`, every sweep axis, `us_T`, `us_work`, `us_busy`, `us_idle`, `us_last`, `us_launch_per_phase`, `us_reps` (per-launch T), `mhz`, `pass`, `regs`, `spill`, `smem`, `job`.
    - Raw stamps go to a `.st` file (`stamps_append`).
    - Each bench writes `params.json` with each fitted term (`formula`, `coeffs`, `units`, `fit_rmse`, `range`, `jobs`).
11. **Tags:** in NOTES, [M] is measured (job id given) and [H] is a hypothesis.

## Python
The system python has no numpy. Use `/raid/garv901/vibe-megakernel/.venv/bin/python` for analyze.py and common/*.py.

## Folder per bench
`<bnn_name>/`:
- `<bnn_name>.cu`: includes `ub.cuh`, then `host.h`.
- `run.sbatch`: from `common/job.sbatch`.
- `analyze.py`: JSONL + `.st` → tables + fit, using `common/stamps.py` and `common/tiles.py`.
- `NOTES.md`: method, deviations from this file, results tables with job ids, the fit, the gate result, caveats.
- `params.json`.

Raw outputs go to `results/2026-10-29-occfuse/<bnn_name>/`. Slurm logs go to `runs/slurm-<job>.log`.

## Bench specs

### B0 `b0_calib`: harness check (do first)
- **Question:** what the empty phase costs, and whether the harness timing matches mk's.
- **Arms:**
  - (a) Phases with only `bar_arrive` / `bar_wait`; the number of active CTAs N ∈ {8, 16, 32, 64, 96, 128, 148}. All CTAs take part in the barrier, and N is the count that runs a spin of W ∈ {0, 0.5, 1, 2, 4} us.
  - (b) Like (a), with thread 0 issuing 1–4 TMA loads (32 KB each) between arrive and wait (the window prefetch), so its effect on barrier latency is visible.
  - (c) Clock: an idle phase sequence vs. a phase sequence with all CTAs busy (tensor MMAs on smem operands, as in B3). Record MHz from `clk_sample` and NVML. Try `nvidia-smi -lgc 1965` only if it is permitted without sudo. Otherwise do not lock, and record why.
- **Outputs:** t_bar (lat) vs N and W, the timer step, MHz under idle and busy load.
- **Result (jobs 14277/14278):** bare barrier lat 0.80; with 1–2 window loads 0.89–0.98 (mk state); T = W + t_bar within 0.12. Passed under the prefetch-on default.
- **Gate:** lat ≈ mk stamps "lat" (1.02 us at b8, st_14155) ± 0.2 us. Also, a spin of W gives T ≈ W + t_bar within 0.15.

### B1 `b1_stream`: weight stream vs active SMs
- **Question:** the time for N active CTAs to each stream S bytes of A from HBM into an R-slot ring (TMA, consumer = a single thread that waits on `full` and frees the slot at once, with no math).
- **Sweep:**
  - N ∈ {8, 16, 32, 48, 64, 96, 128, 148}, with two placements: spread (as `tiles.py` floor ranges place tiles) and the first N blockIdx.
  - S ∈ {32, 64, 96, 128, 192, 256, 384, 512, 768, 1024} KB.
  - R ∈ {2, 3, 4, 6, 8}.
  - Prefetch off here (B2 covers it).
- **Outputs:** busy-CTA work and phase T. Fit `t = t0 + S / rate(N)` with `rate(N) = min(r_sm, BW_agg / N)`. Report r_sm, BW_agg, t0, and their dependence on R.
- **Gate:** at N = 148 and large S, the aggregate rate agrees with the `why` probe (results/2026-10-28-why, analysis_11678) within 10 %. Also check the 48 × 256 KB point against mk T_null GU busy (2.62 us at b8) after B2 adds the prefetch [H until B2].

### B2 `b2_prefetch`: bytes hidden by the barrier-window prefetch
- **Question:** how much of S is hidden when thread 0 issues the first P slots in the barrier window, given the time W from that issue to the barrier exit.
- **Method:** the previous phase is a spin of W (all CTAs). Optionally, one CTA spins W + spread to make a late arriver. Thread 0 then issues `min(P, R)` loads between `bar_arrive` and `bar_wait`, and the phase streams the rest of S.
- **Sweep:** S as B1; P ∈ {0, 1, 2, R} slots; W ∈ {0, 0.5, 1, 2, 4, 6} us; spread ∈ {0, 1, 2} us; R; N ∈ {48, 96, 128, 148}.
- **Outputs:** exposed stream time = busy work − (stream-free floor). Fit `exposed = max(0, t_stream(S) − hidden(P, W))`.
- **Gate (mk ground truth, st_14155 b8, `MK_NULL` runs; ± 0.3 us on busy work):**

  | Floor (busy work) | Configuration | Target |
  |---|---|---|
  | classic QKV | 148 CTAs, 2 units, R 2 | 0.67 |
  | classic GU | 148 CTAs, 3 units, R 2 | 1.46–1.50 |
  | classic D | 148 CTAs, 2 units, R 2 | 0.48 |
  | TS QKV | 128 CTAs, 2 steps, R as mk at TT 16 | 0.80 (includes B) |
  | TS GU | 48 CTAs, 8 steps | 2.62 |
  | TS D | 96 CTAs, 2 steps | 0.83 |

  W comes from the previous phase in mk: use mk's preceding phase T from st_14155 as W. Report which W reproduces each target.

### B3 `b3_step`: compute chain of one tile, operands in smem
- **Question:** the time of SPT k-steps (8 MMAs M128×N×K16 each), with the mbarrier handoff, when the data is already in smem.
- **Sweep:** N(TT) ∈ {16, 32, 48, 64, 96, 128, 192, 256}; SPT ∈ {1, 2, 4, 8, 16}; accumulators 1 vs 2 (tiles back to back); 1 CTA active vs all 148; commit per k-step vs per tile.
- **Outputs:** fit `t_tile = a + SPT · t_step(N)`. Report cycles and us.
- **Gate:** the per-MMA cost implied by t_step agrees with hv2_tc job 9425 (M128N16 58.1, M128N128 80.6, M128N256 128.2 cycles; `vibe-megakernel/megabench/kernels/hv2_tc/logs/9425.out`) within 10 %.

### B4 `b4_bpath`: getting the token operand B into smem
- **Question:** the latency and throughput of B for k steps.
- **Arms:**
  - (a) Staged, as mk `stage_tile`: all 512 threads `__ldcg` fp32 x (L2-resident), multiply by rsqrt(ss) and the ln weight, convert to bf16, store to SW128 smem, `fence.proxy.async`, `__syncthreads`. Two sub-arms: once per k-step (mk classic) and all steps in one batch (all loads issued, then all stores).
  - (b) TMA from a bf16 copy (box 64 k × TT rows).
  - (c) TMA of an fp32 copy (no MMA use, bytes only): sets the bytes-vs-latency scale.
- **Sweep:** T rows ∈ {1, 8, 16, 32, 64, 128, 256}; k per batch ∈ {128, 256, 512, 1024}; idle vs. a concurrent A stream (B1 at N = 148) on the same SM.
- **Outputs:** fit t_B(T, k, mode) = latency + bytes / throughput, for each mode.
- **Gate:** at T = 256, staged is clearly slower than TMA (the direction of job 9595: phase sums 47.6 vs 27.9 us). At T ≤ 16, report the gap; no target.

### B5 `b5_epi`: epilogue output modes
- **Question:** the time from a full TMEM accumulator (M128 × TT) to the output written, for each mode.
- **Modes:**
  - (1) plain fp32 store;
  - (2) `red.add` fp32, with KS CTAs adding to the same tile addresses (contention);
  - (3) bf16 store;
  - (4) GU pair act: g in lanes 0–63, u in lanes 64–127, smem exchange with named barriers, silu, bf16 store (pattern: mk `ts_phase` PAIR epilogue, `regimebench/kernels/mk/mk_step.cu` lines ~976–1006);
  - (5) argmax over rows (LM).
- **Sweep:** TT ∈ {16, 32, 48, 64, 96, 128}; valid tokens T ≤ TT; epilogue warps 4, 8, 12 (chunk map as mk: q = w & 3, chunks grp + 6m, + 3); KS ∈ {1, 2, 4, 8, 12, 24}.
- **Method:** fill TMEM with one MMA from smem, then time the epilogue only (event stamps per warp group, as xover4 EV).
- **Outputs:** fit t_epi(TT, T, mode, KS).
- **Gate:** report mode (4) at TT 16, T 8 against mk GU compute (full − null busy, st_14155 b8: about 3.3 us, which includes MMA and B). No pass/fail; mark [H].

### B6 `b6_fin`: serial materialization (finalizer) chain
- **Question:** after a split-K reduction into fp32 x [T × H], the time from the last producer's arrival to the bf16 copy being complete.
- **Method:**
  - Producers red.add their partial [rows × T] pieces (KS pieces per 128-row block), fence, and add a ticket.
  - A finalizer for each row block polls the ticket (`ld.acquire` + `__nanosleep`, as mk `idle_finalize`), reads x, computes ss / norm, writes bf16, and arrives.
  - Placements: (a) idle CTAs, one job each (MK_FIN 0); (b) the last producer of each block; (c) all CTAs after their own tickets (MK_FIN 2).
- **Sweep:** T ∈ {1, 8, 16, 32, 64, 128, 256}; H ∈ {1024, 2048}; row blocks 8–16; KS ∈ {4, 12, 16, 24}.
- **Outputs:** t_fin = last arrival − last producer arrival (from stamps). Split it into: ticket observe latency (empty finalizer), L2 read + convert, write + fence.
- **Gate:** the O-like configuration (H 1024, 8 row blocks, 16 producers per block, T 8, placement (a)) reproduces the mk O tail (st_14155 b8: T-mode O last 3.54 vs classic 1.73 → +1.8 ± 0.4 us).

### B7 `b7_splitk`: price of splitting K for a whole phase
- **Question:** for a fixed GEMM, how the phase time depends on KS once occupancy is equal.
- **Arms:** the TS engine (xover4 `k_ts`, copied, with the B0 barrier and stamps) at KS ∈ {1, 2, 3, 4, 6, 8, 12, 24}, red.add output; plus a workspace arm (partials to [KS][T][N] fp32, then one reduce pass by all CTAs after a barrier).
- **Sweep:** shapes QKV / O / GU / D (no act fusion here: plain output), T ∈ {8, 32, 128, 256}, TT = min(T rounded to 16, 128).
- **Outputs:** T per KS; for equal max steps per CTA (`tiles.py`), the extra cost per added piece (red.add bytes, number of epilogues).
- **Gate:** QKV T 128 TT 128: KS 4 faster than KS 8 by ≥ 0.8 us (job 9592: 6.07 vs 7.46).

### B8 `b8_compose`: whole phases (ground truth for the model)
- **Question:** the phase time of complete designs, to check the model built from B0–B7.
- **Designs:**
  - (1) classic split-K: units over all CTAs, B staged each unit, red.add, prefetch R 2 (port of xover2 a0 plus window prefetch);
  - (2) TS fused KS 1 (GU-like act epilogue or plain);
  - (3) TS split-K + finalizer (B6 chain) producing the next phase's bf16 copy.
- **Sweep:** shapes QKV, O, GU, D, LM × T ∈ {1, 8, 16, 32, 64, 96, 128, 192, 256}. Run them as phase sequences in mk order, so that each phase's window prefetch sees the real previous phase.
- **Outputs:** per (shape, T, design): T, busy, idle, last.
- **Gate:**
  - The design ranking matches st_14155 at T 1, 8, 32, 64 (the classic sequence beats TS by 2.5–3.0 us/layer).
  - At T 128 and 256 the signs match PASS-04 (TS wins).
  - Each phase within 20 % of mk stamps for the same configuration.

## Agent brief template (one sonnet agent per bench)
```
You are implementing and running microbenchmark <Bn> of probe/mk_core/occfuse (repo /raid/garv901/regime-mk-core, branch mk-core).
Read first: probe/mk_core/occfuse/METHODOLOGY.md (rules + your bench spec), common/ub.cuh, common/host.h, common/stamps.py, common/tiles.py,
and the reference files named in your spec. Goal: measure the term in your spec, fit it, and check the gate.
Rules:
- Write only in probe/mk_core/occfuse/<bnn_name>/ and results/2026-10-29-occfuse/<bnn_name>/. Do not edit common/, the mk kernel
  (regimebench/kernels/mk/), LEDGER.md, or other benches. If common/ needs a change, stop and report it.
- Do not commit. Do not run e2e model benchmarks or any LLM / agent runs.
- GPU work only through Slurm, per the run-gpu-job skill: run slurm-windows first, 1 GPU, partition priority (<= 30 min) or main,
  an honest --time, at most 4 jobs. Build on the login node first (nvcc, sm_100a) to catch compile errors before submitting.
- Every run line under `timeout`. A hang is a result: record it and continue.
- NOTES.md: method, deviations from METHODOLOGY.md, tables with job ids, fit, gate result, caveats. Tag [M] / [H].
  Report measurements and fits only; no conclusions about mk's design.
Return (final message): job ids, gate PASS / FAIL with numbers, the fitted terms table, open issues.
```
