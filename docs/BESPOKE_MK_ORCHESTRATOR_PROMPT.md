# Refined prompt: parallel bespoke decode megakernels (Opus parents + Sonnet implementers)

## Context
The regime study (done, repo `/raid/garv901/regime-megakernel`, dashboard
https://claude.ai/artifact/7GnFfGth9jEXPu5LqA7Kmz) showed hv2 wins only at b1 / short context and planned
two bespoke decode megakernels (MK-small, MK-batch) for the high-headroom decode points; prefill and KV-heavy
points are out of scope. The user wants a prompt for a clean chat that plans (in plan mode) a large workflow:
one Opus parent per bespoke megakernel, each with Sonnet implementer subagents, each ending with an STE +
diagram-design artifact of its optimizations and e2e ablation results.

Deliverable of this turn: the refined prompt below (paste into a clean chat as-is). Nothing is executed now.

Design decisions baked into the prompt (my choices, flagged so the user can change them):
- **Two Opus parents** = the two planned kernels (MK-small, MK-batch), matching the dashboard plan.
- **Each Opus parent runs as its own top-level Claude Code session** (own git worktree + branch), started by
  the user with a brief the clean chat writes. Reason: a spawned subagent cannot itself spawn subagents, so an
  Opus parent that must fan out Sonnet implementers has to be a top-level session.
- **Shared split-KV attention (A1) is owned by MK-small**; MK-batch uses hv2g's attention until A1 lands, then
  imports it. Avoids two agents building the same engine.
- **Sonnet implementers may submit GPU jobs** (the user's request overrides the old "Sonnet = CPU only" rule),
  through the run-gpu-job skill; big sweeps go to the `background` partition.

---

## The prompt (copy everything inside the fence)

```text
You are the orchestrator for building bespoke Qwen3-0.6B decode megakernels on B200, one per serving
regime, that beat the best CUDA-graph + PDL baseline in that regime. Work in plan mode first. Your output is
a high-level workflow plus one self-contained brief per regime owner. Do not write kernel code yourself.

## Background (read these first, in this order)
1. Dashboard with all measured results, glossary and the next-steps plan:
   https://claude.ai/artifact/7GnFfGth9jEXPu5LqA7Kmz  (read it with the Artifact tool, action "read")
2. /raid/garv901/regime-megakernel/results/2026-10-22-b200/FINDINGS.md and REPORT.md (per-point tables).
3. /raid/garv901/regime-megakernel/README.md, SPEC.md, docs/METHODOLOGY.md (harness, arms, SOL model).
4. Current megakernel and its generalisation:
   - hv2 (frozen b1 winner, ~378 us/step, 40 % SOL): regimebench/kernels/hv2/hv2_submission.py
   - hv2g (runtime kv, token loop; MT1 = hv2 path, MTN = untuned many-token build):
     regimebench/kernels/hv2g/{hv2g_common.cuh, hv2g_mt1.cu, hv2g_mtn.cu, NOTES.md}
5. History of what was tried on hv2 (read before proposing anything):
   /raid/garv901/vibe-megakernel/megabench/mkprobe/handoff-step3/BRIEF.md, LEARNINGS.md (F39-F64 closed ideas)
6. Tensor-core experiments on this model:
   /raid/garv901/vibe-megakernel/megabench/kernels/dense_tc/{SPEC.md, README.md} (tcgen05 engine, tck2b)
   /raid/garv901/vibe-megakernel/megabench/kernels/hv2_tc/{PLAN.md, RESULT.md, SPEC-micro.md}
   /raid/garv901/vibe-megakernel/megabench/kernels/hv2_mma/RESULT.md
7. pi05 megakernel optimisation passes (method to copy, not code to copy):
   /raid/garv901/vibe-megakernel/megabench/sota/pi05/ - mk/pi05_mk.cu (swap-AB cta_group::2 design, see the
   comments near "swap-AB" around lines 268 and 1500), cute/{PLAN.md, NOTES.md, specs/},
   probe/ubench/ (microbenchmarks: gemm_tc, gemm_tc2, ab, gemm_split, sync_cost, gbar, chain_pdl, hbm_ramp),
   probe/mk_stamps_ab.py and probe/mk_*_ab.py (per-stage A/B method).

## Measured facts to build on (from the study; tagged)
- [M] Baseline fi-graph runs ~312 kernels/step; at small decode it sits at 14-23 % of SOL, so a megakernel
  has 4-7x of room there. KV-heavy points (baseline at 58-65 % SOL) and prefill are OUT OF SCOPE.
- [M] hv2g loses from 4 tokens/step (batch or spec-verify) and from ~4K context at b1.
- [M] Long context: hv2g adds ~1.35 us per 128-position chunk per layer; one q-head's KV is walked serially
  and 9 CTAs per head repeat the same attention.
- [M] At b1, every tensor-core GEMV lost: any HMMA or tcgen05 drops the SM clock 1962 -> ~1845 MHz (-6 %);
  tcgen05.mma costs ~59 cycles per instruction flat for N <= 64; tck2b whole step 664 us vs hv2 386 us.
- [M] Isolated micro-prices often have the wrong in-kernel sign; A/Bs under 5 us need 3 buffer placements x
  2 GPUs; probe/trace builds perturb timing - judge by the plain binary.
- [H] tcgen05 step time may stay flat from 1 to ~128 tokens (MMA-count bound), which would beat the baseline
  across batched decode. Unmeasured: this is MK-batch's first experiment.

## Regime owners (one Opus parent each)
- MK-small: 1-8 tokens per step, any context length. Points: R0; R1 b1-b8; R2 kv128-32K; R3 b1 and b8 up
  to kv 4K; R6 q2-q8. Engines: FFMA GEMV (hv2 lineage, register-blocked across tokens) + split-KV decode
  attention over all 148 CTAs with a combine step. MK-small OWNS the split-KV attention engine and publishes
  it as a reusable module.
- MK-batch: 9-256 tokens per step. Points: R1 b16-b256; R3 b32-kv1K; R6 q16. Engines: tcgen05 GEMM tiles
  (start from dense_tc / tck2b) + the split-KV attention from MK-small (use hv2g's attention until it lands).
The exact point list per owner is on the dashboard ("Decode points and which kernel targets each").

## What I want you to plan
A workflow where each regime owner is an Opus session that runs this loop, one sub-problem at a time:
1. Baseline: run its megakernel arm and fi-graph on all its points with the regime harness
   (python -m regimebench run, see README); get the fi-graph kernel timeline
   (scripts/gpu_check_fi.py --timeline) and a per-phase breakdown of the megakernel (IKET or stamp markers,
   see the tracer limits in megabench/mkprobe/markers/LIMITATIONS.md) to find which phase loses, by how much,
   against which baseline kernel.
2. Pick ONE losing phase. Write a pre-registered plan: expected gain band, and the kill criterion.
3. Isolate that phase as a standalone kernel; optimise it in silo until it beats both the current
   megakernel phase and the matching SOTA kernel(s) at the regime's shapes.
4. Inspect SASS of every candidate (cuobjdump -sass / nvdisasm, ptxas -v): confirm the intended primitives
   are present (FFMA vs HMMA vs tcgen05 UMMA, TMA bulk copies, TMEM loads/stores), no unexpected spills or
   local memory, register count. Verify the mnemonics against a known-good microbenchmark's SASS first.
5. Ask explicitly per phase: FFMA vs HMMA vs UMMA at this token count; TMA vs cp.async vs LDG; TMEM
   accumulators; swap-AB (put the large dimension on MMA M, as pi05 did); cta_group::2; warp
   specialisation (producer / MMA / epilogue warps) and a regime-specific ping-pong schedule; split-K and
   split-KV; fewer grid barriers via dependency counters. Devise microbenchmarks when a hardware question
   is open (as pi05 did to find swap-AB), and record the answer.
6. If the isolated phase wins, port it into the megakernel and run the e2e regime harness on all owned
   points, interleaved with fi-graph and the previous megakernel version. Keep it only if e2e improves
   beyond noise (paired A/B rules above); otherwise revert and log why.
7. Log every pass (kept or dropped) in a ledger with job ids, numbers and [M]/[H] tags, then go to step 2.
Stop when every owned point beats the best baseline, or when the remaining gaps have no plan left.

## Roles and parallelism
- The Opus parent plans, reasons about the hardware, reviews every diff and decides keep/revert. It
  delegates implementation to Sonnet subagents (model "sonnet") with one self-contained SPEC file each,
  with an explicit file-ownership block when two subagents work in parallel.
- Sonnet subagents implement, build, run microbenchmarks and e2e benchmarks, and report numbers. They may
  submit Slurm jobs via the run-gpu-job skill; sweeps and anything over ~1 GPU-hour go to the background
  partition. They never decide keep/revert.
- A spawned subagent cannot spawn its own subagents, so each Opus parent must be a separate top-level Claude
  Code session. Your job (orchestrator) is to write each parent's brief as a file the user starts a session
  with, plus the shared setup: git worktree and branch per owner in /raid/garv901/regime-megakernel, the
  directory each owner owns (e.g. regimebench/kernels/mk_small/, regimebench/arms/mk_small.py), how the
  shared split-KV module is handed from MK-small to MK-batch, and a GPU budget split.
- Keep the cost visible: estimate agents, jobs and GPU-hours per owner before starting.

## Deliverable at the end of each owner's optimisation passes
Each owner publishes one Artifact page. Write all text in Simplified Technical English (load the
simplified-technical-english skill) and draw the diagrams with /diagram-design:diagram-design in standard
mode. The page must contain:
- a short diagram of the final kernel (phases, warp roles, data movement, primitives used);
- each optimisation pass briefly: what changed, why, measured gain or loss, kept or dropped;
- the e2e ablation results for every owned point: megakernel vs best fi-graph vs vLLM vs SOL, plus the
  ablation of each kept optimisation (cumulative, one at a time), with job ids;
- [M]/[H] tags on every claim.
Then add a link to each owner's page in the dashboard's next-steps section.

## Rules
- Truth over breadth: small validated sets; every number has a job id; hypotheses are marked.
- Do not push to GitHub. Commit locally with the repo's identity (Garv Ghai <garv901@gmail.com>); no AI
  attribution in commits.
- Ask me before using the exceptions partition or anything that would exceed the GPU budget.

Start in plan mode: explore the files above (you may fan out Explore agents), then present the workflow,
the two owner briefs, the shared setup and the cost estimate for my approval.
```

---

## Notes for the user (not part of the prompt)
- Context pack the clean chat needs is all referenced by path or link inside the prompt; nothing else to attach.
- If you prefer the clean chat to run the owners itself instead of you starting two sessions, the alternative
  is the Workflow tool, but its agents also cannot spawn Sonnet subagents, so the Opus/Sonnet split would be
  lost. Separate sessions keep the requested structure.
- If you want more parallelism, MK-small can be split into two owners (split-KV long context; multi-token
  FFMA GEMV), at the cost of one more shared-code boundary.

## Verification
- Paste the fenced prompt into a clean chat; it should be able to open every path and the artifact link
  (all paths checked to exist in this session: regimebench/kernels/hv2g, megabench/kernels/{dense_tc,hv2_tc,
  hv2_mma}, megabench/sota/pi05/{mk,cute,probe/ubench}).
