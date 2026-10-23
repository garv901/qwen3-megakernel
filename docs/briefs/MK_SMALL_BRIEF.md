# Brief: MK-small owner (bespoke Qwen3-0.6B decode megakernel, 1-8 tokens per step, B200)

You are the MK-small owner: an Opus session that plans, reasons about the hardware, writes SPECs, reviews every
diff, and decides keep or revert. Sonnet subagents implement. Start in plan mode: read the files below, run Pass 0,
then present your pass plan to the user before the first optimisation pass.

Working copy: `/raid/garv901/regime-mk-small` (git worktree, branch `mk-small`). Commit only there.
The sibling owner MK-batch works in `/raid/garv901/regime-mk-batch` (branch `mk-batch`) at the same time.

## Goal
One megakernel arm `mk_small` that beats the best baseline (FlashInfer + CUDA graph + PDL, "fi-graph", and vLLM)
at every owned point, or a ledger that shows why the remaining gaps have no plan left.

Owned points (`--workloads mk_small` expands to these 21; numbers [M], jobs 9471-9477):
| point | best baseline us | SOL us | hv2g us / best |
|---|---|---|---|
| R0-b1-kv128, R1-b1-kv128 | 834, 878 | 151 | 0.45, 0.43 |
| R1-b2 / b4 / b8 (kv128) | 889 / 922 / 973 | 153-164 | 0.88 / 1.08 / 1.52 |
| R2-b1 kv128 / 512 / 1K / 2K | 884 / 973 / 975 / 982 | 151-178 | 0.43-0.92 |
| R2-b1 kv4K / 8K / 16K / 32K | 1014 / 1161 / 1308 / 1612 | 208-619 | 1.41 / 2.14 / 3.49 / 5.40 |
| R3-b1 kv1K / 4K / 16K | 973 / 1012 / 1305 | 164-384 | 0.66 / 1.42 / 3.50 |
| R3-b8 kv1K / 4K | 1154 / 1584 | 267 / 619 | 1.55 / 1.78 |
| R6-b1-kv1024 q2 / q4 / q8 | 1075 / 1107 / 1162 | 164 | 0.96 / 1.08 / 1.34 |
You also OWN the shared split-KV attention module (A1) in `regimebench/kernels/common/splitkv/` and publish it to
MK-batch by git tag. Read `regimebench/kernels/common/README.md` (the contract and the hand-off rules) first.

## Read first (in this order)
1. Dashboard: https://claude.ai/artifact/7GnFfGth9jEXPu5LqA7Kmz (Artifact tool, action "read"; its numbers are in
   `results/2026-10-22-b200/summary.csv` here).
2. `results/2026-10-22-b200/FINDINGS.md`, `REPORT.md`; `README.md`, `SPEC.md`, `docs/METHODOLOGY.md`.
3. `regimebench/kernels/hv2/hv2_submission.py` (frozen b1 winner; `R11_SKEW` stamps at lines 469-490, `R9_PROBE`).
   `regimebench/kernels/hv2g/{hv2g_common.cuh, hv2g_mt1.cu, hv2g_mtn.cu, build.py, coop.py, NOTES.md}`,
   `regimebench/arms/{base.py, hv2g.py}`.
4. `/raid/garv901/vibe-megakernel/megabench/mkprobe/handoff-step3/BRIEF.md` and `LEARNINGS.md` (F39-F64 closed ideas).
   `/raid/garv901/vibe-megakernel/megabench/mkprobe/markers/LIMITATIONS.md` (tracer limits).
5. Tensor-core results (why MK-small stays FFMA): `/raid/garv901/vibe-megakernel/megabench/kernels/hv2_tc/{PLAN.md,RESULT.md}`,
   `.../kernels/hv2_mma/RESULT.md`.
6. Method to copy (not code): `/raid/garv901/vibe-megakernel/megabench/sota/pi05/` - `mk/SPEC-load-hoist.md` (SPEC
   template), `cute/NOTES.md` (log format), `probe/mk_stamps_ab.py`, `probe/mk_knob_ab.py`, `probe/mk_nullfg_ab.py`
   (per-stage A/B and null-work oracles), `probe/ubench/*.cu` (sync_cost, gbar, hbm_ramp, chain_pdl).

## Facts to build on (do not re-derive) - [M] measured, [H] hypothesis
- [M] hv2 = 148 CTAs x 512 threads, cooperative launch, 4 grid barriers per layer (Q = QKV; F = qk-norm, RoPE,
  attention, O-proj red.add; G = gate/up + SiLU; D = down), 113 in total. Hop ~1.05 us (~120 us/step). F->G release
  fence 0.64 us/layer, self-caused (F63/F64). 126 regs, 0 spill, 196 KB smem bin (bins 0/8/16/32/64/100/132/164/196/228).
  ~375 us steady, 40 % of HBM SOL. hv2g MT1 = hv2's path, 378 us at R0.
- [M] Attention layout: 144 CTAs = 16 q-heads x 9 Wo slices; all 9 CTAs of a head repeat the full attention; KV chunks
  of 128 run serially (`hv2g_mt1.cu:297-461`; `attn_chunk` `hv2g_common.cuh:196-258`, PPW==8 hard-wired). Cost:
  +1.35 us per 128 positions per layer (kv1K 639, 4K 1431, 32K 8704 us). No real split-KV was ever built.
- [M] At s128 attention is off the critical path (F58-F62): a perfect cluster-2 split saves <= 2.6 us; most of the
  attention time is fixed cost (max, exp, butterfly, syncs). So keep the hv2 F path for short kv; measure the switch.
- [M] hv2g MTN (6 barriers/layer, 80 B spills): +150-180 us per extra token, ~11 % of the FFMA ceiling.
  [H] An FFMA design at its ceiling stays weight-bound to ~8 tokens (math ~95 us/step at 8 tokens).
- [M] Any HMMA or tcgen05 drops the SM clock 1962 -> ~1845 MHz (-6 %, ~+22 us/step); every tensor-core GEMV lost
  at b1 (hv2_tc TC_D +49 us/step; hv2_mma +4..+20 us). Do not reopen this at 1-8 tokens without a new reason.
- [M] Bytes cost full bandwidth (~17.6 ns/KB/CTA/layer, F47/F56). Barrier and poll micro-tuning is CLOSED (user).
  A dynamic work queue does not pay (hv2_wq +9.6 us). Clusters of 8/16 drop occupancy below 128 CTAs (F45).
- [M] Measurement: buffer placement moves one binary by up to 9 us. A/Bs under 5 us need 3 placements x 2 different
  physical GPUs (check `nvidia-smi -L` in each log). Probe builds perturb timing (+49..+114 us); IKET +1.5-2.4 %.
  Isolated micro-prices often have the wrong sign in the kernel (F46). Decide on the plain binary only.

## The loop (one sub-problem at a time)
0. **Pass 0, baseline** (once): `sbatch -p background slurm/regime.sbatch --arms hv2g,fi-graph,vllm --workloads
   mk_small --tag mks-p0` (20 rounds). `scripts/gpu_check_fi.py --timeline` on R2-b1-kv4096, R1-b8-kv128,
   R6-b1-kv1024-q8. A per-phase breakdown of the starting kernel: port hv2's `R11_SKEW` stamps behind a knob (default
   OFF, 0-line SASS diff when off) plus null-work oracle builds (keep barriers, skip one phase's work, pi05 method).
   Output in `kernels/mk_small/NOTES.md`: phase -> us/layer -> matching baseline kernel(s) -> gap, per point class.
1. Pick ONE losing phase. Write `probe/mk_small/PASS-NN.md` before any build: hypothesis, expected gain band, kill
   criterion, arms, placements, points.
2. Isolate the phase as a standalone kernel at the regime shapes. Optimise it in silo until it beats both the current
   megakernel phase and the matching SOTA kernel(s) from the timeline.
3. SASS of every candidate (`cuobjdump -sass`, `nvdisasm -c`, `-Xptxas -v`): intended primitives present (FFMA vs
   HMMA vs UMMA, TMA bulk copies `UBLKCP`, cp.async `LDGSTS`), regs, 0 spill / local memory. Check the mnemonics first
   on a known-good microbenchmark (`sota/pi05/probe/ubench/`).
4. Ask per phase: FFMA register blocking across tokens; TMA vs cp.async vs LDG; split-K and split-KV; fewer grid
   barriers via dependency counters; warp specialisation and a ping-pong schedule for this regime (F50 killed
   ping-pong at b1 - say what is different). Write a microbenchmark when a hardware question is open, and record the answer.
5. If the isolated phase wins, port it behind a knob and run e2e on all owned points, interleaved with fi-graph and the
   previous mk_small version (paired A/B rules above). Keep only if e2e improves beyond noise; else revert and log why.
6. Ledger row in `kernels/mk_small/LEDGER.md`: id, phase, change, band, result (us per point), job ids, GPU-h used,
   kept/dropped, [M]/[H]. Then go to 1.
Stop when every owned point beats the best baseline, or when the remaining gaps have no plan left.

## Seeded pass order [H] (change it after Pass 0 if the table says so)
1. **A1 split-KV decode attention** (R2, R3 long kv). Items = (kv-head, kv-split) over all 148 CTAs; both q-heads of
   a GQA pair share one K/V read; attention runs once, O-proj reads the combined output (no 9x repeat). Split count
   at runtime from kv length; hv2 F path kept for short kv. Band: KV-read floor at 32K = 3.76 GB, ~470-570 us extra;
   target kv32K 800-1200 us (fi 1612). Kill: no point at kv >= 4K beats fi, or kv128 regresses > 3 us.
   On pass: tag `splitkv-v1` (see `kernels/common/README.md`) and tell the user.
2. **Multi-token FFMA GEMV** (R1 b2-b8, R6, R3-b8): weights in registers reused over T <= 8 tokens; one build per T
   (the arm picks it, as hv2g does). Band b8 450-650 us (fi 973). Kill: b4 not below fi.
3. **Multi-token attention**: A1 with q-tokens per item (batch: one sequence per item; R6: causal among new tokens).
4. **LM head / argmax for n_seq <= 8**, then the remaining gaps from the Pass 0 table.

## Files you own (everything else is read-only)
`regimebench/kernels/mk_small/`, `regimebench/arms/mk_small.py` (class `MkSmallArm`, already registered in
`arms/base.py`), `regimebench/kernels/common/splitkv/`, `tests/test_mk_small*.py`, `scripts/gpu_check_mk_small.py`,
`slurm/mk_small_*.sbatch`, `probe/mk_small/`, `results/*-mk-small/`. Copy from `kernels/hv2g/` - never edit hv2 or
hv2g. A change to a shared file (base.py, report.py, run.py, workload.py, harness/) goes to the user/orchestrator as a
request; it lands on `main`, then you `git merge main`.

## Roles and agents
- Sonnet subagents (`model: "sonnet"`, general-purpose): one self-contained SPEC each in `probe/mk_small/specs/`
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
1. Final e2e ablation on all 21 points: mk_small final vs best fi-graph vs vLLM vs SOL, plus the cumulative ablation
   of each kept optimisation (one at a time, in the order kept), with job ids. Report via
   `python -m regimebench report RUN_DIR... --out results/<date>-mk-small/` (the "Megakernel arms" table).
2. One Artifact page. All text in Simplified Technical English (load the `simplified-technical-english` skill).
   Diagrams with `/diagram-design:diagram-design` in standard mode. It contains: a short diagram of the final kernel
   (phases, warp roles, data movement, primitives); each pass briefly (what changed, why, gain or loss, kept or
   dropped); the e2e ablation table for every owned point; [M]/[H] tags on every claim.
3. Add a link to your page in the dashboard's next-steps section: ArtifactData `get` `files/index.html`, then
   `str_replace` on field `text` with `if_version` (MK-batch may write the same document; on a version conflict read
   again and redo). Change nothing else on the dashboard.

## Rules
- Truth over breadth: small validated sets; every number has a job id; hypotheses are marked [H].
- Commit locally on `mk-small` with `Garv Ghai <garv901@gmail.com>`; no AI attribution; never push.
