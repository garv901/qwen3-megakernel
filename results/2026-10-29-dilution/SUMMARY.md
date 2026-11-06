# Why the mk gain dilutes inside vLLM serving

All comparisons are mk vs stock vLLM in the same vLLM 0.31 engine and the same job; only the forward differs.
Qwen3-0.6B, B200, mk build of the M2 grid (mk-serve fe5725b). Figures: `results/2026-10-29-plots/06_dilution.png`
(M2 grid, coverage view) and `07_step_dilution.png` (this study, per-step view).

## Answer

A forward gain reaches the step only while the forward is longer than vLLM's CPU loop for the next step.
vLLM prepares step N+1 on the CPU while the GPU runs step N (async scheduling). Stock's forward (1.1-2 ms) hides
that CPU work; mk's shorter forward (0.8-1.1 ms at small batch) does not, so the GPU waits and the gap between
forwards grows. There are four dilution sources, in order of size:

1. **CPU floor** (largest at small batch). Step time cannot drop below the CPU's time per step. With vLLM's
   default path that floor is ~1.37 ms on the Azure trace; mk forwards below ~1.05 ms gain nothing more.
2. **Unused attention-metadata build** (raises the floor). On every mk step vLLM still builds FlashInfer attention
   metadata (~100-110 us CPU) that mk never reads. Skipping it (`--skip-attn-meta`) lowers the floor to ~1.22 ms.
3. **Fixed GPU work between forwards** (~300 us per step: sampler, input copies). Not touched by mk; it dilutes the
   forward gain by plain arithmetic (forward / (forward + 300 us)), about 4-5 points at small batch.
4. **Coverage**. Steps mk does not run (prefill, mixed prefill + decode, batches > 32 requests) run stock and
   count fully in TPOT. Costs ~4 points of TPOT at 10 req/s; dominates at 80-100 req/s (mk runs ~2-8% of GPU time).

Open-loop load adds a feedback effect: faster steps drain the queue, so batches get smaller (40 req/s: mean decode
batch 12.0 -> 9.6) and mk runs ~27% more steps for the same tokens. Each step pays the fixed costs (3, and the
CPU loop) again; latency per token still improves because each token waits behind fewer others.

## Measured, same job per stage (jobs 14160 / 14161)

Gains vs stock. Forward and step gains are matched per step shape (request bucket x KV bucket, weighted by stock's
time); TPOT is the client mean time per output token.

| stage | forward | step, metadata built | step, metadata skipped | TPOT, built | TPOT, skipped |
|---|---|---|---|---|---|
| Azure conv trace, 10 req/s | 25.9% | 16.1% | 24.9% | 10.8% | **21.0%** |
| Azure conv trace, 40 req/s | 17.8% | 16.1% | 18.0% | 15.6% | **18.5%** |
| closed loop, 1 request | 24.6% | 4.3% | 13.0% | 3.0% | **11.3%** |
| closed loop, 8 requests | 21.7% | -10.5% | 7.1% | -9.9% | **4.7%** |

- At realistic low load (trace, 10 and 40 req/s) the skip removes most of the dilution: the step keeps almost all
  of the forward gain, and TPOT keeps 18-21%.
- Closed loop is the worst case: every step is a short decode with no idle time, so the CPU floor binds on every
  step. Even with the skip, mk's gap stays 100-160 us above stock's and about half the forward gain is lost.
- Without the skip, closed loop at 8 requests is slower than stock: the gap grows from 310 to 709 us.

## Where the CPU time goes (median per decode step, Azure trace 10 req/s)

| | stock | mk | mk, skip |
|---|---|---|---|
| vLLM step start -> forward (`pre_us`: schedule, input prep, metadata) | 719 | 720 | 630 |
| of which attention-metadata build | 100 | 109 | skipped |
| step description + dispatch | 29 | 43 | 42 |
| forward call | 72 (graph replay) | 160 | 108 |
| mk `run()` parts: ring wait / plan / H2D copy / graph replay | - | 7 / 30 / 30 / 10 | 6 / 28 / 29 / 10 |

- The mk planner (numba `fill_step`) is 7-17 us; in serving the plan part is ~30 us including the Python list ->
  numpy conversion and header writes. Moving it to another thread would not help: it holds the GIL.
- The single H2D copy costs ~30 us of CPU (the cuda-python driver call plus an event record), not data volume.
- vLLM's own loop (~620 us before the forward, plus sampler and output handling) is the bulk of the floor.

## Async scheduling, measured (job 14159, closed loop, 1 request)

| arm | forward | gap | step | TPOT |
|---|---|---|---|---|
| stock, async | 1247 | 282 | 1529 | 1.551 ms |
| mk, async | 828 | 528 | 1341 | 1.421 ms |
| fi, async | 970 | 387 | 1348 | 1.392 ms |
| stock, async off | 1108 | 1027 | 2133 | 2.170 ms |
| mk, async off | 820 | 997 | 1822 | 1.840 ms |

With async off nothing is hidden: the gap is ~1 ms for both arms, i.e. vLLM's CPU work per step is ~1 ms and async
scheduling hides ~0.7 ms of it behind a stock forward. With async on, mk and fi-graph end at the same ~1.34 ms step
although mk's forward is 140 us shorter: both are on the CPU floor (metadata built).

## Next levers (not done)

1. Make `--skip-attn-meta` the default for custom-forward arms (measured above; no correctness change expected,
   mk never reads the metadata; keep the shadow / correctness check on when switching).
2. Trim mk's own ~90 us: issue the H2D copy through a cheaper path (e.g. a torch pinned-tensor copy or putting the
   staging buffer in mapped host memory read by the kernel), avoid the list -> numpy conversion, drop the dispatcher's
   per-step description work for steps that cannot run mk.
3. Coverage: decode batches > 32 requests and mixed steps.
4. vLLM's own loop (~1 ms of CPU per step) bounds every forward below ~1.1 ms; only a leaner engine loop removes it.

## Data

- `runs/hostcost/` (jobs 14160, 14161; `probe/mk_core/hostcost/hc.sbatch`; harness worktree
  `/raid/garv901/regime-mk-serve-host`, branch `mk-serve-host`: step log adds `pre_us`, `meta_us`, `decide_us`, `parts`).
- `runs/dilution/` (jobs 14157-14159; `probe/mk_core/dilution/dil.sbatch`), tables in `analysis.md`
  (`probe/mk_core/dilution/analyze.py`).
- Between-job noise is ~10% on absolute stock timings (stock forward at 1 request: 1247 us in 14157, 1102 us in
  14161); compare arms within a job only.
