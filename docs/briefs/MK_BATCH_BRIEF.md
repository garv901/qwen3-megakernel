# Brief: MK-batch owner (bespoke Qwen3-0.6B decode megakernel, 9-256 tokens per step, B200)

You are the MK-batch owner: an Opus session that plans, reasons about the hardware, writes SPECs, reviews every
diff, and decides keep or revert. Sonnet subagents implement. Start in plan mode: read the files below, run Pass 0
and Pass 0b, then present your pass plan to the user before the first optimisation pass.

Working copy: `/raid/garv901/regime-mk-batch` (git worktree, branch `mk-batch`). Commit only there.
The sibling owner MK-small works in `/raid/garv901/regime-mk-small` (branch `mk-small`) at the same time and owns
the shared split-KV attention module (A1).

## Goal
One megakernel arm `mk_batch` that beats the best baseline (FlashInfer + CUDA graph + PDL, "fi-graph", and vLLM)
at every owned point, or a ledger that shows why the remaining gaps have no plan left.

Owned points (`--workloads mk_batch` expands to these 7; numbers [M], jobs 9471-9477):
| point | tokens/step | best baseline us | SOL us | hv2g us / best |
|---|---|---|---|---|
| R1-b16-kv128 | 16 | 1019 | 180 | 2.59 |
| R1-b32-kv128 | 32 | 1179 | 211 | 4.15 |
| R1-b64-kv128 | 64 | 1404 | 272 | 6.62 |
| R1-b128-kv128 | 128 | 1759 | 395 | 10.41 |
| R1-b256-kv128 | 256 | 2242 | 642 | 16.68 |
| R3-b32-kv1024 | 32 | 1680 | 622 | 3.68 |
| R6-b1-kv1024-q16 | 16 (causal among new tokens) | 1164 | 164 | 2.31 |
The baseline is launch/latency bound here (14-38 % SOL, ~312 kernels per step), so there is 2.7-7x of room.

## Read first (in this order)
1. Dashboard: https://claude.ai/artifact/7GnFfGth9jEXPu5LqA7Kmz (Artifact tool, action "read"; its numbers are in
   `results/2026-10-22-b200/summary.csv` here).
2. `results/2026-10-22-b200/FINDINGS.md`, `REPORT.md`; `README.md`, `SPEC.md`, `docs/METHODOLOGY.md`.
3. `regimebench/kernels/hv2g/{hv2g_common.cuh, hv2g_mtn.cu, hv2g_mt1.cu, build.py, coop.py, NOTES.md}` (skeleton,
   token loop, MTN attention), `regimebench/kernels/hv2/hv2_submission.py` (`R11_SKEW` stamps at lines 469-490),
   `regimebench/arms/{base.py, hv2g.py}`. `regimebench/kernels/common/README.md` (A1 contract and hand-off).
4. Tensor-core engine: `/raid/garv901/vibe-megakernel/megabench/kernels/dense_tc/{SPEC.md, README.md, tc_common.py,
   kernel_main.cu, mma_bench.py}` and its results `/raid/garv901/vibe-megakernel/megabench/mkprobe/runs/20261018T1900Z-TC-engine/RESULT.md`.
   `.../megabench/kernels/hv2_tc/{PLAN.md, RESULT.md, SPEC-micro.md}`, `.../kernels/hv2_mma/RESULT.md`.
5. `/raid/garv901/vibe-megakernel/megabench/mkprobe/handoff-step3/BRIEF.md`, `LEARNINGS.md` (F39-F64 closed ideas),
   `/raid/garv901/vibe-megakernel/megabench/mkprobe/markers/LIMITATIONS.md` (tracer limits).
6. pi05 megakernel (method, and the swap-AB design): `/raid/garv901/vibe-megakernel/megabench/sota/pi05/mk/pi05_mk.cu`
   (comments near "swap-AB", ~lines 268 and 1500), `mk/SPEC-load-hoist.md` (SPEC template), `cute/NOTES.md`,
   `probe/ubench/{gemm_tc.cu, gemm_tc2.cu, gemm_split.cu, sync_cost.cu, gbar.cu, chain_pdl.cu, hbm_ramp.cu}`,
   `probe/mk_stamps_ab.py`, `probe/mk_knob_ab.py`, `probe/mk_nullfg_ab.py`, `probe/mk_fgumma_ab.py`.

## Facts to build on (do not re-derive) - [M] measured, [H] hypothesis
- [M] hv2g MTN: 6 grid barriers/layer, FP32 FFMA token loop (TBQ=8, TBO=4, TBD=2), 80 B spills; +150-180 us per
  extra token (b16 2635, b32 4894, b256 37404 us); ~11 % of the FFMA ceiling. LM head does 8 sequences at a time,
  so the embed (~311 MB bf16) is re-streamed ceil(n_seq/8) times.
- [M] [H for >1 token] tcgen05.mma costs ~59 cycles per instruction flat for M 64/128 and N 8..64, 64 c at N=128,
  128 c at N=256, the same at 1 or 148 CTAs; commit+wait 283 c even when empty; the issue loop ran ~90 c/MMA.
  [H] So the MMA count, not N, sets the time: phase time may stay flat from 1 to ~128 tokens. Never measured.
- [M] dense_tc/tck2b at b1: whole step 656-668 us; its no-math skeleton is 518 us vs hv2 386 us, so **the dense_tc
  skeleton itself costs ~130 us** (dependency chain, per-CTA hoarding, TMA ring setup). Exposed matvec: tck2b 147 us.
- [M] Tensor cores drop the SM clock 1962 -> ~1845 MHz (-6 %) for the whole kernel. At b1 that lost; at 16+ tokens
  the baseline also uses tensor cores, so it is a cost to count, not a veto.
- [M] At b1, cta_group::2 (M256) and swap-AB gave no gain (hv2_tc PLAN.md:50-62). In pi05 at 50 tokens, swap-AB with
  cta_group::2 (weights on M=128, tokens on N=64, full K per CTA pair, no split-K) gave -11 % on F+G, because
  per-SM activation ingress was the limit, not math.
- [M] Bytes cost full bandwidth (~17.6 ns/KB/CTA/layer, F47/F56). Barrier and poll micro-tuning is CLOSED (user).
  A dynamic work queue does not pay (hv2_wq +9.6 us). Clusters of 8/16 drop occupancy below 128 CTAs; size 2 keeps 148 (F45).
- [M] Measurement: buffer placement moves one binary by up to 9 us. A/Bs under 5 us need 3 placements x 2 different
  physical GPUs (check `nvidia-smi -L` in each log). Probe builds perturb timing; IKET +1.5-2.4 %. Isolated
  micro-prices often have the wrong sign in the kernel (F46). Decide on the plain binary only.

## The loop (one sub-problem at a time)
0. **Pass 0, baseline** (once): `sbatch -p background slurm/regime.sbatch --arms hv2g,fi-graph,vllm --workloads
   mk_batch --tag mkb-p0` (20 rounds). `scripts/gpu_check_fi.py --timeline` on R1-b16, R1-b64, R1-b256 (kv128):
   the GEMM and attention kernels fi-graph uses and their us per step are your per-phase targets.
   **Pass 0b, the gate for the whole effort:** an isolated GEMM phase at 148 CTAs - weights on MMA M (128;
   cta_group::1 and ::2), tokens on N = 16..256, TMA weight ring (from `dense_tc/tc_common.py`), TMEM accumulators -
   for the QKV, O, gate/up and down shapes. Compare with the weight-stream time (bytes / 6.5 TB/s) and with the
   fi-graph/cuBLAS kernel at the same shape. [H] phase time stays near stream time up to N ~128.
   Kill: at N=64 the phase is > 1.5x stream time AND slower than the baseline kernel. Then stop and report to the
   user (the MK-small/MK-batch boundary moves) before more work.
1. Pick ONE losing phase. Write `probe/mk_batch/PASS-NN.md` before any build: hypothesis, expected gain band, kill
   criterion, arms, placements, points.
2. Isolate the phase as a standalone kernel at the regime shapes. Optimise it in silo until it beats both the current
   megakernel phase and the matching SOTA kernel(s) from the timeline.
3. SASS of every candidate (`cuobjdump -sass`, `nvdisasm -c`, `-Xptxas -v`): intended primitives present (tcgen05 MMA,
   TMEM ld/st, TMA copies; no stray HMMA/FFMA in the math loop), regs, 0 spill / local memory. Take the exact SASS
   mnemonics from a known-good microbenchmark first (`sota/pi05/probe/ubench/gemm_tc2.cu`, dense_tc `mma_bench.py`);
   do not trust mnemonic names from memory (hv2's TMA bulk copy is `UBLKCP`).
4. Ask per phase: FFMA vs HMMA vs UMMA at this token count; TMA vs cp.async vs LDG; TMEM accumulators; swap-AB (large
   dim on MMA M); cta_group::2; warp specialisation (TMA producer / MMA issuer / epilogue warps) and a regime-specific
   ping-pong schedule; split-K and split-KV; fewer grid barriers via dependency counters. Write a microbenchmark when
   a hardware question is open (as pi05 did to find swap-AB), and record the answer.
5. If the isolated phase wins, port it behind a knob and run e2e on all owned points, interleaved with fi-graph and the
   previous mk_batch version (paired A/B rules above). Keep only if e2e improves beyond noise; else revert and log why.
6. Ledger row in `kernels/mk_batch/LEDGER.md`: id, phase, change, band, result (us per point), job ids, GPU-h used,
   kept/dropped, [M]/[H]. Then go to 1.
Stop when every owned point beats the best baseline, or when the remaining gaps have no plan left.

## Seeded plan [H] (change it after Pass 0/0b if the data says so)
1. **Skeleton choice, on data:** default = hv2g's skeleton (cooperative grid barriers, weight prefetch between barrier
   arrive and wait) + the tcgen05 engine from dense_tc. Take dense_tc's skeleton only if Pass 0b shows its TMA ring
   is needed for ingress - and then explain how you avoid its ~130 us skeleton cost.
2. GEMM phases on tcgen05 (QKV, O, gate/up, down): tile shape and swap-AB / cta_group::2 per Pass 0b.
3. Fewer barriers per layer (MTN has 6): fuse k-norm/RoPE/cache write and O-proj where the data allows.
4. LM head for n_seq up to 256 as a GEMM tile (one embed stream, not one per 8 sequences).
5. Attention: hv2g MTN attention first (cheap at kv128: 6 of 7 points). Import A1 at `splitkv-v1` when MK-small tags
   it (`git checkout splitkv-v1 -- regimebench/kernels/common/splitkv`), mainly for R3-b32-kv1024 and R6-q16.
Bands [H]: b64 400-700 us (fi 1404), b256 900-1500 us (fi 2242).

## Files you own (everything else is read-only)
`regimebench/kernels/mk_batch/`, `regimebench/arms/mk_batch.py` (class `MkBatchArm`, already registered in
`arms/base.py`), `tests/test_mk_batch*.py`, `scripts/gpu_check_mk_batch.py`, `slurm/mk_batch_*.sbatch`,
`probe/mk_batch/`, `results/*-mk-batch/`. NOT `kernels/common/` (MK-small owns it; you import tagged versions and
report bugs in your ledger). Copy from `kernels/hv2g/` and dense_tc - never edit hv2, hv2g or the vibe-megakernel
tree. A change to a shared file (base.py, report.py, run.py, workload.py, harness/) goes to the user/orchestrator as
a request; it lands on `main`, then you `git merge main`.

## Roles and agents
- Sonnet subagents (`model: "sonnet"`, general-purpose): one self-contained SPEC each in `probe/mk_batch/specs/`
  (template: `sota/pi05/mk/SPEC-load-hoist.md` - Why (measured, jobs) / The change behind a knob, default OFF /
  Measurement / Acceptance with commands / Files you own / Report back). Two in parallel need disjoint ownership blocks.
- Subagents implement, build, run microbenchmarks and e2e jobs with the run-gpu-job skill, and report numbers. They
  never decide keep/revert. You review every diff before a job that counts.
- A spawned subagent cannot spawn subagents: you are the only fan-out point.

## GPU budget and rules
- Budget: **20 GPU-hours** for the whole effort (estimate 10-15). Log GPU-h per pass in the ledger. Ask the user
  before going over it, and before any use of the `exceptions` partition.
- The node is shared (8 GPUs; other users queue many jobs). At most 2 of your jobs at once. Microbenchmarks and
  smoke tests: `main` partition. E2E sweeps and ablations: `background`. Jobs cap at 2 h.
- Submit from the worktree root: `slurm/regime.sbatch` runs the code of the checkout you submit from (it prints
  `REPO=... branch sha` in the log - check it).
- `pytest` is not installed in the venv (`/raid/garv901/vibe-megakernel/.venv`); ask the user before installing anything.

## Deliverable at the end
1. Final e2e ablation on all 7 points: mk_batch final vs best fi-graph vs vLLM vs SOL, plus the cumulative ablation
   of each kept optimisation (one at a time, in the order kept), with job ids. Report via
   `python -m regimebench report RUN_DIR... --out results/<date>-mk-batch/` (the "Megakernel arms" table).
2. One Artifact page. All text in Simplified Technical English (load the `simplified-technical-english` skill).
   Diagrams with `/diagram-design:diagram-design` in standard mode. It contains: a short diagram of the final kernel
   (phases, warp roles, data movement, primitives); each pass briefly (what changed, why, gain or loss, kept or
   dropped); the e2e ablation table for every owned point; [M]/[H] tags on every claim.
3. Add a link to your page in the dashboard's next-steps section: ArtifactData `get` `files/index.html`, then
   `str_replace` on field `text` with `if_version` (MK-small may write the same document; on a version conflict read
   again and redo). Change nothing else on the dashboard.

## Rules
- Truth over breadth: small validated sets; every number has a job id; hypotheses are marked [H].
- Commit locally on `mk-batch` with `Garv Ghai <garv901@gmail.com>`; no AI attribution; never push.
