# Brief: MK-core owner (bespoke Qwen3-0.6B decode megakernel `mk`, tcgen05, 1-256 tokens/step, B200)

You are the MK-core owner: an Opus session that plans, reasons about the hardware, writes SPECs, reviews every
diff, and decides keep or revert. Sonnet subagents implement. Start in plan mode: read the files below, run Pass 0
and Pass 0b, then present your pass plan to the user before the first optimisation pass.

Working copy: `/raid/garv901/regime-mk-core` (git worktree, branch `mk-core`). Commit only there.
The sibling owner MK-attn (`/raid/garv901/regime-mk-attn`, branch `mk-attn`) builds the split-KV attention module
for the SAME kernel at the same time. Read `regimebench/kernels/common/README.md` (ownership, hand-off by git tag,
the A1 interface) first.

## Goal
ONE kernel, arm `mk`, for every decode point from 1 to 256 tokens per step: tcgen05 GEMM phases for all token counts
(no FFMA GEMV path - user decision 2026-10-23) and the split-KV attention (A1) from MK-attn. It must beat the best
baseline (FlashInfer + CUDA graph + PDL "fi-graph", and vLLM) at every point, or the ledger shows why the remaining
gaps have no plan left. hv2 (378 us at b1, FFMA) stays the b1 reference; beating it is not required.

You own: the kernel skeleton (launch, barriers, phase order, weight streaming), the tcgen05 GEMM phases (QKV, O,
gate/up, down), q/k-norm + RoPE + KV-cache write, the LM head + argmax, the arm, and the attention call sites. You
call attention only through the A1 interface; until `splitkv-v1` exists you use a placeholder with that interface.

Points (`--workloads mk` = 28; numbers [M], jobs 9471-9477; your main focus is R1, MK-attn's is `--workloads mk_kv`):
| point | tokens/step | best baseline us | SOL us | hv2g us / best |
|---|---|---|---|---|
| R0-b1-kv128 | 1 | 834 | 151 | 0.45 |
| R1-b1 / b2 / b4 / b8 (kv128) | 1-8 | 878 / 889 / 922 / 973 | 151-164 | 0.43 / 0.88 / 1.08 / 1.52 |
| R1-b16 / b32 / b64 / b128 / b256 | 16-256 | 1019 / 1179 / 1404 / 1759 / 2242 | 180-642 | 2.6-16.7 |
| R2-b1 kv128 .. 32K (8 points) | 1 | 884 .. 1612 | 151-619 | 0.43 .. 5.40 |
| R3-b1 kv1K/4K/16K, b8 kv1K/4K, b32 kv1K | 1-32 | 973-1680 | 164-622 | 0.66-3.68 |
| R6-b1-kv1024 q2/q4/q8/q16 | 2-16 | 1075 / 1107 / 1162 / 1164 | 164 | 0.96-2.31 |
The baseline runs ~312 kernels/step and sits at 14-38 % of SOL here, so a megakernel has 2.6-7x of room.

## Read first (in this order)
1. Dashboard: https://claude.ai/artifact/7GnFfGth9jEXPu5LqA7Kmz (Artifact tool, action "read"; numbers are in
   `results/2026-10-22-b200/summary.csv` here). `results/2026-10-22-b200/FINDINGS.md`, `REPORT.md`.
2. `README.md`, `SPEC.md`, `docs/METHODOLOGY.md` (harness, arms, SOL). `regimebench/arms/{base.py, hv2g.py}`.
3. **The crossover bench that set this design:** `probe/xover/{SPEC.md, RESULT.md, NOTES.md,
   xover.cu, run.sh}` (job 9521; `run.sh` rebuilds `bin/`). One GEMM phase, 148 CTAs: tcgen05 is flat
   in T (GU 7.74 -> 7.88 us, D 5.10 -> 5.18 from T=1 to 8; GU 14.1 / D 9.1 at T=256) and wins from T=2.
4. Skeleton and token loop to copy: `regimebench/kernels/hv2g/{hv2g_common.cuh, hv2g_mt1.cu, hv2g_mtn.cu, build.py,
   coop.py, NOTES.md}`; `regimebench/kernels/hv2/hv2_submission.py` (`R11_SKEW` stamps, lines 469-490).
5. tcgen05 engine: `/raid/garv901/vibe-megakernel/megabench/kernels/dense_tc/{SPEC.md, README.md, tc_common.py,
   kernel_main.cu, mma_bench.py}`, results `/raid/garv901/vibe-megakernel/megabench/mkprobe/runs/20261018T1900Z-TC-engine/RESULT.md`;
   `.../megabench/kernels/hv2_tc/{PLAN.md, RESULT.md, SPEC-micro.md}`.
6. `/raid/garv901/vibe-megakernel/megabench/mkprobe/handoff-step3/BRIEF.md`, `LEARNINGS.md` (F39-F64 closed ideas),
   `/raid/garv901/vibe-megakernel/megabench/mkprobe/markers/LIMITATIONS.md` (tracer limits).
7. pi05 megakernel (method, and the swap-AB design): `/raid/garv901/vibe-megakernel/megabench/sota/pi05/mk/pi05_mk.cu`
   (comments near "swap-AB", ~lines 268 and 1500), `mk/SPEC-load-hoist.md` (SPEC template), `cute/NOTES.md`,
   `probe/ubench/{gemm_tc.cu, gemm_tc2.cu, gemm_split.cu, sync_cost.cu, gbar.cu, chain_pdl.cu, hbm_ramp.cu}`,
   `probe/mk_stamps_ab.py`, `probe/mk_knob_ab.py`, `probe/mk_nullfg_ab.py`, `probe/mk_fgumma_ab.py`.

## Facts to build on (do not re-derive) - [M] measured, [H] hypothesis
- [M] tcgen05.mma ~59 cycles/instruction flat for M 64/128 and N 8..64, 64 c at N=128, 128 c at N=256, same at 1 or
  148 CTAs; commit+wait 283 c even when empty; the issue loop ran ~90 c/MMA. So the MMA count sets the time, and
  N (tokens) is almost free up to 64-128 (xover confirms in situ, job 9521).
- [M] tcgen05 (and HMMA) drop the SM clock 1962 -> ~1845 MHz (-6 %) for the whole kernel; `tcgen05.alloc` alone does
  not. A runtime-switched hybrid costs +0.4..1.0 us/phase on both paths (xover).
- [M] dense_tc/tck2b at b1: whole step 656-668 us; its no-math skeleton is 518 us vs hv2's ~386 us, so **the dense_tc
  skeleton itself costs ~130 us** (dependency chain, per-CTA hoarding). Exposed matvec: tck2b 147 us. More than
  1 token was never run e2e. [H] Per xover, tcgen05 GU+D cost ~12.8 us/layer at T=1-8 vs ~8.5 for a no-math oracle.
- [M] hv2 skeleton: 148 CTAs x 512 threads, cooperative launch, 4 grid barriers/layer, hop ~1.05 us (~120 us/step),
  weights loaded between barrier arrive and wait; F->G release fence 0.64 us/layer (F63/F64).
- [M] At b1, cta_group::2 (M256) and swap-AB gave no gain (hv2_tc PLAN.md:50-62). In pi05 at 50 tokens swap-AB with
  cta_group::2 (weights on M=128, tokens on N=64, full K per CTA pair, no split-K) gave -11 % on F+G, because per-SM
  activation ingress was the limit, not math.
- [M] hv2g MTN: 6 barriers/layer, LM head 8 sequences at a time (the ~311 MB bf16 embed is re-streamed ceil(n_seq/8) times).
- [M] Bytes cost full bandwidth (~17.6 ns/KB/CTA/layer, F47/F56). Barrier and poll micro-tuning is CLOSED (user).
  A dynamic work queue does not pay (hv2_wq +9.6 us). Clusters of 8/16 drop occupancy below 128 CTAs; size 2 keeps 148.
- [M] Measurement: buffer placement moves one binary by up to 9 us. A/Bs under 5 us need 3 placements x 2 different
  physical GPUs (check `nvidia-smi -L` in each log). Probe builds perturb timing; IKET +1.5-2.4 %. Isolated
  micro-prices often have the wrong sign in the kernel (F46). Decide on the plain binary only.

## The loop (one sub-problem at a time)
0. **Pass 0, baseline** (once): `sbatch -p background --time=<honest> slurm/regime.sbatch --arms hv2g,fi-graph,vllm
   --workloads mk --tag mk-p0` (20 rounds; coordinate with MK-attn - one of you runs it, both use it).
   `scripts/gpu_check_fi.py --timeline` on R1-b1, R1-b16, R1-b64, R1-b256: the GEMM kernels fi-graph uses and their
   us/step are your per-phase targets.
   **Pass 0b, skeleton + engine choice:** extend xover to all four GEMM shapes (QKV, O, gate/up, down) and T = 1..256,
   cta_group::1 vs ::2, tokens on N (weights on M); and compare with the fi-graph kernel at each shape. Then the
   first e2e build: `mk` v0 = hv2g-style skeleton + tcgen05 phases + the attention placeholder, all 28 points.
   Tag `mk-core-v0`. Kill/stop rule: if v0 does not beat fi at R1-b16..b256 and there is no per-phase plan to close
   it, stop and report to the user.
1. Pick ONE losing phase from the per-phase breakdown (port `R11_SKEW` stamps behind a knob, default OFF, 0-line SASS
   diff when off; plus null-work oracle builds that keep barriers and skip one phase's work - the pi05 method).
   Write `probe/mk_core/PASS-NN.md` before any build: hypothesis, expected gain band, kill criterion, arms,
   placements, points.
2. Isolate the phase as a standalone kernel at the regime shapes. Optimise it in silo until it beats both the current
   mk phase and the matching SOTA kernel(s) from the timeline.
3. SASS of every candidate (`cuobjdump -sass`, `nvdisasm -c`, `-Xptxas -v`): intended primitives present (tcgen05 MMA,
   TMEM ld/st, TMA copies; no stray HMMA/FFMA in the math loop), regs, 0 spill / local memory. Take the exact SASS
   mnemonics from a known-good microbenchmark first (`probe/xover`, `sota/pi05/probe/ubench/gemm_tc2.cu`); do not
   trust mnemonic names from memory.
4. Ask per phase: TMA vs cp.async vs LDG for weights and activations; TMEM accumulators and the epilogue (fuse
   SiLU, residual, norms); swap-AB; cta_group::2; warp specialisation (TMA producer / MMA issuer / epilogue warps)
   and a ping-pong schedule for this regime; split-K; fewer grid barriers via dependency counters. Write a
   microbenchmark when a hardware question is open (as pi05 did to find swap-AB), and record the answer.
5. If the isolated phase wins, port it behind a knob and run e2e on all 28 points, interleaved with fi-graph and the
   previous mk version (paired A/B rules above). Keep only if e2e improves beyond noise; else revert and log why.
6. Ledger row in `kernels/mk/LEDGER.md`: id, phase, change, band, result (us per point), job ids, GPU-h used,
   kept/dropped, [M]/[H]. Tag `mk-core-vN` after each kept pass so MK-attn can test on it. Then go to 1.
Stop when every point beats the best baseline, or when the remaining gaps have no plan left.

## Seeded plan [H] (change it after Pass 0/0b if the data says so)
1. Skeleton on data: default = hv2g's skeleton (cooperative grid barriers, weight prefetch between arrive and wait)
   + dense_tc's tcgen05 engine. dense_tc's skeleton only if Pass 0b shows its TMA ring is needed for ingress - and
   then say how you avoid its ~130 us cost.
2. GEMM phases on tcgen05, tile shape and swap-AB / cta_group::2 per Pass 0b; token-count builds as separate entry
   points (not a runtime switch, per xover), picked by the arm.
3. Fewer barriers per layer: fuse norms/RoPE/cache write into GEMM epilogues; O-proj after the A1 combine.
4. LM head for n_seq up to 256 as one GEMM over the embed (one stream, not one per 8 sequences).
5. Import each `splitkv-vN` from MK-attn and re-run e2e.
Bands [H]: b1-b8 500-650 us (fi 878-973); b64 550-800 (fi 1404); b256 900-1400 (fi 2242).

## Files you own (everything else is read-only)
`regimebench/kernels/mk/` (incl. `attn_placeholder.cuh`), `regimebench/arms/mk.py` (class `MkArm`, already registered
in `arms/base.py`), `tests/test_mk*.py`, `scripts/gpu_check_mk.py`, `slurm/mk_*.sbatch`, `probe/mk_core/`,
`results/*-mk/`. NOT `kernels/common/splitkv/` (MK-attn's; import tagged versions). Copy from hv2g, dense_tc and
xover - never edit hv2, hv2g or the vibe-megakernel tree. A change to a shared file (base.py, report.py, run.py,
workload.py, harness/) goes to the user/orchestrator as a request; it lands on `main`, then you `git merge main`.

## Roles and agents
- Sonnet subagents (`model: "sonnet"`, general-purpose): one self-contained SPEC each in `probe/mk_core/specs/`
  (template: `sota/pi05/mk/SPEC-load-hoist.md` - Why (measured, jobs) / The change behind a knob, default OFF /
  Measurement / Acceptance with commands / Files you own / Report back). Two in parallel need disjoint ownership blocks.
- Subagents implement, build, run microbenchmarks and e2e jobs with the run-gpu-job skill, and report numbers. They
  never decide keep/revert. You review every diff before a job that counts.
- A spawned subagent cannot spawn subagents: you are the only fan-out point.

## GPU budget and rules
- Budget: **20 GPU-hours** for the whole effort (estimate 10-15). Log GPU-h per pass in the ledger. Ask the user
  before going over it, and before any use of the `exceptions` partition.
- The node is shared (8 GPUs; other users queue many jobs). At most 2 of your jobs at once. `main` + `priority` share
  ONE per-user budget of 20 GPU-h (GPUs x --time, queued + running) with MK-attn and the user: keep `main` for
  microbenchmarks and smoke tests you wait on; e2e sweeps and ablations go to `background`. Run `slurm-windows`
  and pass an honest `--time` (the run-gpu-job skill). Jobs cap at 2 h.
- Submit from the worktree root: `slurm/regime.sbatch` runs the code of the checkout you submit from (it prints
  `REPO=... branch sha` in the log - check it).
- `pytest` is not installed in the venv (`/raid/garv901/vibe-megakernel/.venv`); ask the user before installing anything.

## Deliverable at the end
1. Final e2e ablation on all 28 points: mk final vs best fi-graph vs vLLM vs SOL, plus the cumulative ablation of each
   kept optimisation (one at a time, in the order kept, incl. MK-attn's A1 versions), with job ids. Report via
   `python -m regimebench report RUN_DIR... --out results/<date>-mk/` (the "Megakernel arms" table).
2. One Artifact page for the whole kernel, written together with MK-attn (you own the page; MK-attn sends its
   section - attention diagram, its passes, its ablation rows - as a Markdown file in `probe/mk_attn/PAGE_SECTION.md`
   on its branch). All text in Simplified Technical English (load the `simplified-technical-english` skill).
   Diagrams with `/diagram-design:diagram-design` in standard mode. It contains: a short diagram of the final kernel
   (phases, warp roles, data movement, primitives); each pass briefly (what changed, why, gain or loss, kept or
   dropped); the e2e ablation table for every point; [M]/[H] tags on every claim.
3. Add a link to the page in the dashboard's next-steps section: ArtifactData `get` `files/index.html`, then
   `str_replace` on field `text` with `if_version` (on a version conflict read again and redo). Change nothing else.

## Rules
- Truth over breadth: small validated sets; every number has a job id; hypotheses are marked [H].
- Commit locally on `mk-core` with `Garv Ghai <garv901@gmail.com>`; no AI attribution; never push.
