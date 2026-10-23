# Brief: MK-attn owner (split-KV decode attention A1 for the bespoke megakernel `mk`, B200)

You are the MK-attn owner: an Opus session that plans, reasons about the hardware, writes SPECs, reviews every
diff, and decides keep or revert. Sonnet subagents implement. Start in plan mode: read the files below, run Pass 0,
then present your pass plan to the user before the first optimisation pass.

Working copy: `/raid/garv901/regime-mk-attn` (git worktree, branch `mk-attn`). Commit only there.
The sibling owner MK-core (`/raid/garv901/regime-mk-core`, branch `mk-core`) builds the rest of the SAME kernel
(skeleton, tcgen05 GEMM phases, LM head, the arm) at the same time. Read `regimebench/kernels/common/README.md`
(ownership, hand-off by git tag, the A1 interface) first.

## Goal
The decode attention of `mk`: one split-KV module (A1) for every point, 1-256 tokens/step, kv 128-32K, batch 1-256,
multi-token causal (R6). Items = (sequence, kv-head, kv-split) spread over all 148 CTAs, one K/V read per GQA pair,
partials (m, l, o) + a combine that feeds O-proj. It must make `mk` beat the best baseline at the KV points
(`--workloads mk_kv`, 16 points) and must not cost anything measurable at kv128 (the other 12 points).

Points where attention decides the result (numbers [M], jobs 9471-9477):
| point | best baseline us | SOL us | hv2g us / best |
|---|---|---|---|
| R2-b1 kv1K / 2K / 4K / 8K / 16K / 32K | 975 / 982 / 1014 / 1161 / 1308 / 1612 | 164-619 | 0.66 / 0.92 / 1.41 / 2.14 / 3.49 / 5.40 |
| R3-b1 kv1K / 4K / 16K | 973 / 1012 / 1305 | 164-384 | 0.66 / 1.42 / 3.50 |
| R3-b8 kv1K / 4K; R3-b32 kv1K | 1154 / 1584; 1680 | 267 / 619; 622 | 1.55 / 1.78; 3.68 |
| R6-b1-kv1024 q2 / q4 / q8 / q16 | 1075 / 1107 / 1162 / 1164 | 164 | 0.96-2.31 |
KV bytes = 114,688 B per cached token across all 28 layers (bf16, 8 kv-heads x 128 x K+V): kv32K reads 3.76 GB,
~470 us at 8 TB/s, ~575 us at the measured 6.55 TB/s.

## Read first (in this order)
1. Dashboard: https://claude.ai/artifact/7GnFfGth9jEXPu5LqA7Kmz (Artifact tool, action "read"; numbers are in
   `results/2026-10-22-b200/summary.csv`). `results/2026-10-22-b200/FINDINGS.md`, `REPORT.md` (incl. the winner
   attention backend per point: fa2 / trtllm-gen / cute-dsl / cudnn).
2. `README.md`, `SPEC.md`, `docs/METHODOLOGY.md`; `regimebench/fixture.py` (KV layout), `regimebench/arms/fi_graph.py`
   (how the baseline calls its attention), `regimebench/kernels/common/README.md`.
3. Current attention: `regimebench/kernels/hv2g/hv2g_mt1.cu:297-461` (fused attention + O-proj, 16 q-heads x 9 Wo
   slices) and `hv2g_common.cuh:196-258` (`attn_chunk`, `attn_merge`), `hv2g_mtn.cu:194-266` (multi-token, one item
   per (token, q-head)), `kernels/hv2g/NOTES.md`.
4. `/raid/garv901/vibe-megakernel/megabench/mkprobe/handoff-step3/LEARNINGS.md` F43, F58-F62 (attention cost at s128
   is mostly fixed overhead), `.../mkprobe/markers/LIMITATIONS.md`.
5. The design decision context: `probe/xover/RESULT.md` (job 9521):
   the whole kernel is tcgen05, so it already runs at ~1845 MHz.
6. Method: `/raid/garv901/vibe-megakernel/megabench/sota/pi05/` - `mk/SPEC-load-hoist.md` (SPEC template),
   `cute/NOTES.md`, `probe/mk_nullatt_ab.py` (null-attention oracle), `probe/mk_attk2_ab.py`, `probe/ubench/{hbm_ramp.cu,
   attn_o_fused.cu, sync_cost.cu}`, `probe/specs/U7_attn_o_fused.md`.

## Facts to build on (do not re-derive) - [M] measured, [H] hypothesis
- [M] hv2g attention: every one of the 9 CTAs of a q-head repeats the full attention; KV chunks of 128 positions run
  serially; +1.35 us per 128 positions per layer (kv1K 639, 4K 1431, 32K 8704 us). GQA pairs read the same K/V twice.
  No real split-KV was ever built (only an oracle, F58: <= 2.6 us at s128 for a free 2-way exchange).
- [M] At s128 most of the attention time is fixed cost (max, exp, new-token fold, butterfly, syncs), not QK/PV dots
  (F62); `__expf` kept (-2.3 us, F43); a 512-thread combine cost +7..+8 us (F43). So the short-kv path must stay lean.
- [M] The clock penalty does not apply inside `mk`: the GEMM phases already use tcgen05 (~1845 MHz). So QK/PV may use
  HMMA or tcgen05 where it pays (multi-token: R6 q16 = 32 q-rows per GQA item; batch). Decide on data.
- [M] HBM streaming (pi05 hbm_ramp): small streams never reach peak (5 MB ~2.0 TB/s, 16.8 MB ~3.5, 34.6 MB ~4.1-4.4
  TB/s with ldg/cp.async/TMA). Per-layer KV at 32K is ~134 MB, so the stream should get near peak only if all 148 SMs
  keep enough bytes in flight - measure.
- [M] Bytes cost full bandwidth (~17.6 ns/KB/CTA/layer). Barrier/poll micro-tuning is CLOSED (user). A dynamic work
  queue does not pay (hv2_wq +9.6 us) - use a static item plan built in `prepare()`.
- [M] Measurement: buffer placement moves one binary by up to 9 us. A/Bs under 5 us need 3 placements x 2 different
  physical GPUs (check `nvidia-smi -L`). Probe builds perturb timing; IKET +1.5-2.4 %. Isolated micro-prices often
  have the wrong sign in the kernel (F46). Decide on the plain binary only.

## The loop (one sub-problem at a time)
0. **Pass 0, baseline** (once): use MK-core's `mk-p0` run of `--arms hv2g,fi-graph,vllm --workloads mk` if it exists
   (coordinate; else run `--workloads mk_kv`). `scripts/gpu_check_fi.py --timeline` on R2-b1-kv4096, R2-b1-kv32768,
   R3-b32-kv1024, R6-b1-kv1024-q16: the attention kernel fi-graph runs and its us/step (incl. its split-KV merge
   kernel) is your target per point. Also time hv2g's attention phase alone (null-attention oracle build of a hv2g
   copy in `probe/mk_attn/`) to get its per-layer cost vs kv.
1. Pick ONE sub-problem. Write `probe/mk_attn/PASS-NN.md` before any build: hypothesis, expected gain band, kill
   criterion, arms, placements, points.
2. Build it standalone first: `probe/mk_attn/` bench kernel at 148 CTAs x 512 threads with the real KV layout,
   per layer, cold KV, for kv 128..32K, batch 1..32, q_len 1..16. Optimise in silo until it beats both hv2g's
   attention phase and the fi-graph attention kernel (+ merge) at the same shapes.
3. SASS of every candidate (`cuobjdump -sass`, `nvdisasm -c`, `-Xptxas -v`): intended primitives (TMA / cp.async /
   LDG.128 for K/V, FFMA2 or HMMA/tcgen05 for the dots as chosen), regs, 0 spill / local memory. Take the exact SASS
   mnemonics from a known-good microbenchmark first; do not trust mnemonic names from memory.
4. Ask per pass: split count vs kv and batch (fill 148 SMs, enough bytes in flight per SM); TMA vs cp.async vs LDG
   for K/V; FFMA vs HMMA vs tcgen05 for QK/PV at this q-row count; warp specialisation (loader / math warps) and a
   ping-pong over KV chunks; where the combine runs (ticket-last-CTA vs start of O-proj) and its cost at kv128;
   overlap of the KV stream with the QKV GEMM phase (KV does not depend on this step's q). Write a microbenchmark
   when a hardware question is open, and record the answer.
5. If the standalone version wins, test it in `mk`: `git merge mk-core-vN` (latest tag) into `mk-attn`, build with
   MK-core's attention switch set to `common/splitkv`, and run e2e on all 28 points interleaved with fi-graph and
   `mk-core-vN` with the placeholder (paired A/B rules above). Keep only if `mk_kv` improves beyond noise and kv128
   points do not regress > 3 us. Then tag `splitkv-vN` (see the README rules). Else revert and log why.
6. Ledger row in `regimebench/kernels/common/splitkv/LEDGER.md`: id, change, band, result (us per point), job ids,
   GPU-h used, kept/dropped, [M]/[H]. Then go to 1.
Stop when `mk` beats the best baseline at every `mk_kv` point, or when the remaining gaps have no plan left.

## Seeded plan [H] (change it after Pass 0 if the data says so)
1. **A1 v1, single token, b1** (R2, R3-b1): items = (kv-head, split), 8 kv-heads x S splits ~ 148; both q-heads of the
   pair per item; combine per (q-head) at the start of O-proj. Band: kv32K attention ~500-650 us/step (floor ~470-575);
   `mk` at kv32K 1000-1250 us (fi 1612). Kill: at kv4K the standalone phase is not faster than hv2g's attention.
2. **Batch** (R3-b8, R3-b32): items = (sequence, kv-head, split), split count shrinks as batch grows.
3. **Multi-token causal** (R6 q2-q16): q_len rows per item, causal mask over the new tokens; tensor-core QK/PV if the
   row count pays.
4. **Short kv** (kv128-512, all R1 points): no split, cheapest path; the combine must be ~free here.

## Files you own (everything else is read-only)
`regimebench/kernels/common/splitkv/` (header, host planner, tests, CHANGELOG.md, LEDGER.md), `probe/mk_attn/`,
`tests/test_splitkv*.py`, `slurm/mk_attn_*.sbatch`, `results/*-mk-attn/`. NOT `kernels/mk/` or `arms/mk.py`
(MK-core's): ask for call-site or interface changes in MK-core's `kernels/mk/LEDGER.md` "Requests" section. Never
edit hv2, hv2g or the vibe-megakernel tree. A change to a shared file (base.py, report.py, run.py, workload.py,
harness/) goes to the user/orchestrator as a request; it lands on `main`, then you `git merge main`.

## Roles and agents
- Sonnet subagents (`model: "sonnet"`, general-purpose): one self-contained SPEC each in `probe/mk_attn/specs/`
  (template: `sota/pi05/mk/SPEC-load-hoist.md` - Why (measured, jobs) / The change behind a knob, default OFF /
  Measurement / Acceptance with commands / Files you own / Report back). Two in parallel need disjoint ownership blocks.
- Subagents implement, build, run microbenchmarks and e2e jobs with the run-gpu-job skill, and report numbers. They
  never decide keep/revert. You review every diff before a job that counts.
- A spawned subagent cannot spawn subagents: you are the only fan-out point.

## GPU budget and rules
- Budget: **20 GPU-hours** for the whole effort (estimate 8-12). Log GPU-h per pass in the ledger. Ask the user
  before going over it, and before any use of the `exceptions` partition.
- The node is shared (8 GPUs; other users queue many jobs). At most 2 of your jobs at once. `main` + `priority` share
  ONE per-user budget of 20 GPU-h (GPUs x --time, queued + running) with MK-core and the user: keep `main` for
  microbenchmarks and smoke tests you wait on; e2e sweeps go to `background`. Run `slurm-windows` and pass an honest
  `--time` (the run-gpu-job skill). Jobs cap at 2 h.
- Submit from the worktree root: `slurm/regime.sbatch` runs the code of the checkout you submit from (it prints
  `REPO=... branch sha` in the log - check it).
- CPU tests: `PYTHONPATH=$PWD $PY -m pytest -q tests` (venv `/raid/garv901/vibe-megakernel/.venv`, shared with other
  projects: ask the user before installing anything into it).

## Deliverable at the end
MK-core owns the single Artifact page for `mk`. You write your section as `probe/mk_attn/PAGE_SECTION.md` on your
branch, in Simplified Technical English (load the `simplified-technical-english` skill), with diagram source made
with `/diagram-design:diagram-design` in standard mode: a short diagram of A1 (items, splits, data movement, combine,
primitives); each pass briefly (what changed, why, gain or loss, kept or dropped); the standalone and e2e results
for every `mk_kv` point with job ids; [M]/[H] tags on every claim. Tell the user when it is ready.

## Rules
- Truth over breadth: small validated sets; every number has a job id; hypotheses are marked [H].
- Commit locally on `mk-attn` with `Garv Ghai <garv901@gmail.com>`; no AI attribution; never push.
