# PASS-07: O / D finalizer tail at NPAD 128 / 256 (MK_FIN) + splitkv-v8 half-CTA attention in mk

## Where we are [M]
- e2e v11 (job 9760): 26 / 28. Losses: R1-b128 1.088, R1-b256 1.281 (vs best SOTA, p0 job 9562).
- fi b128 timeline (job 9761): step 1688 us. Attention 20.4 us/layer, GEMMs 23.5 us/layer.
- mk b128 (stamps): attention 25.3, GEMMs 21.3 us/layer.
- O / D stamps at b256 (job 9757): last arrive - median arrive = 4.10 (O) and 3.58 (D) us/layer. At b128 (job 9754): 4.86 / 3.84.
  The last arriver is an idle finalizer CTA (a CTA with no tiles):
  - It waits for the ticket of one (row block, 128-token tile) job.
  - Then it converts 128 tokens x 128 columns alone (up to 4 rounds of 512 items).

## Hypothesis [H]
- Splitting each finalizer job by token rows over nidle / njobs idle CTAs cuts the post-ticket work per CTA by that factor
  (b128: 8 jobs on ~20-60 idle CTAs).
- The reset of the ticket moves to the last of the sub CTAs (a seen counter in tk[32, 64)).

## Build
- MK_FIN 1 (d416c72). SASS is identical at MK_FIN 0.
- The variant that issued all loads before the stores (MK_FIN 1/2 of c220c68) deadlocked at b128 (job 9766, srun bisect).
  The deadlock came from the batched item loop, not from the split. It was not root-caused and was dropped.
- First data (srun, cross-run, 1 round): b128 1862.1 us vs 1946.0 (job 9766 baseline).

## splitkv-v8 (MK-attn P10, half-CTA consumers)
- Imported (f5f98d9). SASS is identical at default knobs (NPAD 128 / 256).
- mk passes SKV_HALVES / SKV_STAGE_ROWS through. The ring size is counted in 128-row units; the KV maps use the build's stage rows.
- h64 = SKV_HALVES 1, SKV_STAGE_ROWS 64, SKV_STAGES 4, on top of skv_tune.
- MK-attn standalone (job 9770), us/layer: b256 43.12 -> 34.23, b128 22.74 -> 20.00, b64 13.86 -> 11.67.

## Run
- Job 9771 (1 GPU x 7 rounds, in-job A/B):
  - {}, MK_FIN 1, h64, h64 + MK_FIN 1 at b128 / b256;
  - {}, h64, MK_FIN 1 at b64;
  - {} vs h64 at R3-b32-kv1024 and b16-kv128;
  - stamps at b128 / b256 with both.

## Results [M] (job 9771, f5f98d9, 1 GPU x 7 rounds, in-job delta vs today's defaults, us/step)

| point | {} | MK_FIN 1 | h64 | h64 + MK_FIN 1 |
|---|---|---|---|---|
| R1-b128-kv128 | 1946.9 | 1852.4 (-94.3) | 1767.1 (-179.8) | **1701.7 (-245.4)** |
| R1-b256-kv128 | 2844.7 | 2806.2 (-38.4) | **2426.5 (-417.6)** | 2475.7 (-369.0) |
| R1-b64-kv128 | 1376.3 | 1382.0 (+5.7) | **1300.7 (-75.6)** | - |
| R3-b32-kv1024 | 1507.6 | - | **1423.9 (-84.0)** | - |
| R1-b16-kv128 | 891.5 | - | **852.9 (-38.7)** | - |

- Stamps (h64 + MK_FIN 1), attention work in us/layer: b128 20.09 (fi 20.4, job 9761) and b256 34.13 (fi ~32.7).
- O spread at b128 is 4.86 -> 3.07 and D spread 3.84 -> 2.05. At b256 the O / D spreads did not change (4.10 / 3.58).
  At NPAD 256, NJOB = 16, so a job gets fewer idle CTAs. [H] the seen-counter cost is not paid back there.

## Decision
- h64 in skv_tune at NPAD >= 32 (q_len 1), and at NPAD 16 with >= 16 sequences (no 1-item-per-CTA plans).
- MK_FIN default -1 = per bucket: 1 at NPAD 128, else 0. Commit 5d436ab; e2e v12 = job 9772.
