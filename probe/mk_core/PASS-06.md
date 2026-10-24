# PASS-06: cta_group::2 tile stream (2-CTA GEMM engine) for NPAD 128 / 256

User-approved (2026-10-23): "go ahead with the 2-CTA gate/up kernel".

## Where we are [M]
- e2e v10 (job 9753): 26 / 28 points are below the best SOTA. The losses are:
  - R1-b128: 1960.5 vs 1768.7 us (1.108);
  - R1-b256: 2906.1 vs 2235.3 us (1.300).
- Stamps, us/layer of work:

  | phase | b256 (job 9725, EF) | b128 (job 9754) | fi b256 (PASS-00, PDL-overlapped upper bounds) |
  |---|---|---|---|
  | QKV | 5.4 | 4.5 | ~6 |
  | O | 6.8 | 4.7 | ~6 |
  | GU | 11.2 | 8.7 | ~6.7 |
  | D | 5.1 | 4.5 | ~6 |
  | attn | 46-58 | 26-30 | ~33 (917 us / 28) |

- GU is the GEMM that is far from fi's number. At b256 the attention gap (MK-attn's) is larger than all GEMM gaps together.

## Constraint [M, PTX ISA 9.7.16]
- All tcgen05 instructions in one kernel must use the same `.cta_group`. splitkv does not use tcgen05 (grep).
- So a 2-CTA GU means a CG = 2 build of the NPAD 128 / 256 buckets, where every tcgen05 phase is in pair mode: QKV / GU / D tile stream, O gemm_phase and the LM tile stream.
- The NPAD <= 64 buckets stay CG = 1. A per-bucket split is allowed by the user.

## Hypothesis [H]
- Per-CTA bytes set the TS phase time; ring depth matters second (TT256 / KS2 at 384 KB was slower than TT128 / KS1 at 512 KB, job 9592).
- GU T = 256, per-CTA bytes (A + B):

  | engine | tile | CTAs with work | per-CTA bytes |
  |---|---|---|---|
  | today | TT 128 / KS 1 | 96 | 512 KB |
  | CG 1, ragged | TT 96 | 144 | 448 KB |
  | CG 2 | M256 x N96 | 144 | 352 KB, ring 4 x 44 KB |
  | CG 2 | M256 x N128 | 96 | 384 KB |

- Expected: GU 11.2 -> ~8 us/layer at b256, about -90 us e2e. b128 is smaller (TT 48 pairs: 304 KB vs 384 KB today).

## Steps
1. xover4 bench (SPEC-xover4): CG 1 / 2 x ragged TT x KS, shapes QKV / O / GU / D, T = 128 / 256. One main job, 30 min.
2. Decide per shape whether a pair beats CG 1 by more than the barrier noise (0.3 us). Go only if the sum over the four phases at b256 is >= 2 us/layer (>= 56 us e2e). Otherwise take only the CG 1 ragged-TT wins and close the 2-CTA path in the LEDGER with the data.
3. If go: SPEC-P6, an MK_CG = 2 bucket build (engine wrappers + geometry + cluster launch in coop.py / build.py). Then A/B, e2e, LEDGER, tag.

## Budget
- Step 1: 0.5 GPU-h. Steps 3+: ~1.5 GPU-h.

## Result, step 1 (xover4, job 9756) [M]
- All 175 configs PASS vs the CUDA-core reference. This includes ragged tiles with partly OOB TMA boxes (expect_tx = full box bytes works).
- Best us/phase per engine (CG 1 / CG 2, TT / KS):

  | shape, T | CG 1 | CG 2 | CG 2 - CG 1 |
  |---|---|---|---|
  | QKV 128 | 5.64 (128/4) | 6.06 (128/4) | +0.42 |
  | QKV 256 | 6.88 (128/2) | 7.34 (128/2) | +0.46 |
  | O 128 | 4.81 (64/8) | 5.14 (128/16) | +0.33 |
  | O 256 | 5.60 (128/8) | 5.80 (128/8) | +0.20 |
  | GU 128 | 6.95 (128/2) | 7.41 (128/2) | +0.46 |
  | GU 256 | 8.45 (96/1) | 9.04 (256/2) | +0.59 |
  | D 128 | 5.51 (128/12) | 5.94 (128/12) | +0.43 |
  | D 256 | 6.37 (128/8) | 6.73 (128/8) | +0.36 |

- GU KS 1 only (the in-epilogue SiLU needs KS 1):
  - T = 256: CG 1 TT 96 8.45, TT 128 8.85; CG 2 TT 96 9.13, TT 128 9.20.
  - T = 128: CG 1 TT 64 8.13 (today's config) < TT 96 8.30 < TT 48 8.54; CG 2 TT 48 8.84.
- **The hypothesis is falsified:** per-CTA bytes do not set the phase time. CG 2 TT 96 (352 KB per CTA, 144 CTAs) is slower than CG 1 TT 128 (512 KB, 96 CTAs).
- **Decision:** cta_group::2 loses at every shape and T by 0.2-0.6 us/phase. Under the step-2 rule the 2-CTA path is closed: no MK_CG = 2 build.
- The only gain is CG 1 GU TT 96 at NPAD 256 (-0.40 us/phase standalone). It is plumbed as knob MK_GU_TT (2f55c92; default SASS identical) and tested in e2e A/B job 9757.
