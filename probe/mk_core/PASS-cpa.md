# PASS-cpa: TMA vs cp.async (LDGSTS) ingress for the GEMM phases (MK-core)

Bench: `probe/mk_core/xover5/` = fork of xover2 (xover2 untouched: `xover4/build_small.sh`, the concurrent NPAD < 128 tile-stream
study, builds its a0 / a3 from `xover2/xover2.cu`). Own `bin/`, `logs/`; no shared files with xover2 / xover3 / xover4.
Question (brief step 4, never run): per phase, TMA vs cp.async vs LDG for weights and activations. Prompted by SemiAnalysis'
Blackwell tensor-core microbenchmarks (TMA has a higher fixed issue / completion latency than cp.async on small transfers).

## Arms (KU 128, CG 1; one binary each; `run.sh`)
| bin | X path | weight path | ring | compares with |
|---|---|---|---|---|
| a0 | LDG + STS (all threads) | TMA | 2 | = xover2 a0 (opcode sequence identical) = mk at NPAD <= 64 |
| c0 | cp.async.cg 16 B (all threads), wait_all + fence.proxy.async before publish | TMA | 2 | a0: X path only |
| x3 | LDG + STS | TMA | max | a3 / c3 |
| c3 | cp.async | TMA | max | x3: X path; a3: cp.async vs TMA for X |
| a3 | TMA | TMA | max | = xover2 a3 (opcode sequence identical) |
| w3 | TMA | cp.async.cg 16 B (all threads, SW128 box layout), cp.async.mbarrier.arrive.noinc on full[s] (count 512) | max | a3: weight path only |
| wc3 | cp.async | cp.async | max | c3: weight path; a3: all-cp.async vs all-TMA |
Barrier-window weight prefetch is kept in every arm (cp.async weights are issued by all threads before the grid barrier).
X is issued only after the grid barrier in every arm (it is the previous phase's output in mk).

## Hypothesis [H] and band
- X path (c0 vs a0, c3 vs x3): cp.async removes the register round trip and lets the next unit's X load overlap the TMEM flush;
  band -0.0..-0.5 us per phase at T <= 64, ~0 at T >= 128. Versus TMA X (c3 vs a3): TMA stays ahead by 0..0.5 us (P0b: a2 - a0 = -0.3..-1.1).
- Weights (w3 vs a3): a bandwidth stream; cp.async costs 4 LDGSTS per thread per unit and no better latency once prefetched in the
  barrier window: band 0..+1 us per phase (worse); LM (311 MB / phase) is the sharpest probe of steady-state bandwidth.
## Kill / keep
A cp.async arm is worth a mk port only if it beats its TMA / LDG pair by > 0.3 us per phase at the same T in both rounds of both
placements (2 jobs), with PASS. Else record "TMA stays" in the ledger.
## Points
Shapes QKV / O / GU / D / LM, T 1..256 (correctness at T 1 / 8 / 64 / 256), nph 112 (LM 4), 5 reps, 2 rounds per job, 2 jobs.

## Results [M] (jobs 10272 / 10273, 2 B200s on cayenne, 2 rounds each = 4 paired samples; 560 PASS / 0 FAIL)
Full tables: `xover5/logs/summary-10272-10273.md` (`python3 xover5/sum.py xover5/logs/slurm-10272.log xover5/logs/slurm-10273.log`).
Sanity: a0 = job 9566 a0 within 0.2 us (QKV T=1 6.00 vs 5.98, GU 7.92 vs 7.94, LM 132.0 vs 131.8). Every delta below has min..max of
the 4 samples on the same side of 0 (spread <= 0.05 us for the 4 small GEMMs).
Delta us per phase (negative = first arm faster), T = 1 / 16 / 64 / 128 / 256:
| pair | QKV | O | GU | D | LM |
|---|---|---|---|---|---|
| c0 - a0 (X cp.async vs LDG, ring 2) | -.37/-.38/-.42/-.66/-.98 | -.20/-.23/-.24/-.47/-.52 | -.33/-.35/-.38/-.68/-1.17 | -.37/-.37/-.34/-.59/-.83 | -4.7/-4.5/-4.2/-9.2/-19.5 |
| c3 - a3 (X cp.async vs TMA) | +.35/+.41/+.46/+.48/+.21 | +.14/+.12/+.20/+.25/+.25 | +.68/+.70/+.73/+.80/-.39* | +.32/+.35/+.47/+.61/+.31 | +20.4/+20.5/+21.1/+21.1/-24.5* |
| w3 - a3 (W cp.async vs TMA) | +.46/+.55/+.37/+.22/+.24 | -.06/-.09/+.20/+.18/+.28 | +.74/+.84/+.82/+.81/+.09 | +.50/+.51/+.45/+.36/+.25 | +20.0/+20.0/+20.0/+19.5/+2.4 |
| wc3 - a3 (all cp.async vs all TMA) | +.86/+.94/+.86/+.78/+.50 | +.07/+.05/+.42/+.43/+.50 | +1.48/+1.59/+1.64/+1.70/+.51 | +.89/+.89/+.91/+.88/+.57 | +54/+54/+55/+56/+15 |
\* ring confound: X-by-TMA takes one X slot per ring slot, so at T = 256 a3 runs GU ring 2 vs c3 3, LM 2 vs 3 (LM also T 64 / 128:
4 / 3 vs 6 / 5). Rings are equal everywhere at T <= 32 and for QKV / O / D at every T.

## Reading
1. **Weights: TMA stays** [M]. cp.async weights lose +0.2..+0.9 us per phase on QKV / GU / D and +20 us on LM (bandwidth stream:
   4 LDGSTS per thread per 32 KB unit + 512 arrivals per fill). Only O at T <= 32 (1 unit per CTA) is ahead, by 0.06..0.09 us,
   below the 0.3 us kill bar. Closed.
2. **X: TMA > cp.async > LDG + STS** [M] at equal ring. cp.async over LDG + STS is -0.2..-0.4 us per phase at T <= 64
   (~-1.3 us/layer summed over the 4 GEMMs at ring 2, ~-36 us/step over 28 layers [H]), but TMA X beats cp.async X by a further
   0.1..0.7 us per phase. Both need a pre-normalised bf16 X in global (mk's NPAD <= 64 path fuses the norm scale / SiLU*up into
   LDG staging; cp.async cannot transform). So cp.async X is dominated wherever that bf16 copy exists, and that is the same
   prerequisite as the NPAD < 128 tile-stream study (xover4_small), which already uses B by TMA from bf16.
3. The one place cp.async X helps over TMA X is the smem budget: 2 X slots instead of one per ring slot lets the weight ring go
   deeper at large T (GU / LM at T = 256: -0.39 / -24.5 us). In mk, NPAD 256 already runs the tile stream (one tile per CTA,
   B by TMA), so this is not a live lever there [H].
Decision: no mk port. TMA for weights and activations; record in the ledger.
