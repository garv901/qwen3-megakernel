# Methodology

## Regimes
One `Workload` = one forward step over sequences `(kv_len, q_len)`; new token j of a sequence sits at position
`kv_len + j` and attends `0..kv_len+j` (causal). Logits are returned for the last new token of each sequence.
R0 decode b1 kv128 (anchor); R1 batch sweep (kv 128); R2 context sweep (b1); R3 batch x context; R4 single-sequence
prefill T=16..8K; R5 batched prefill; R6 multi-token over kv 1K (q 2..16); R7 one 512-token chunk + D decodes at kv 1K;
R8 ragged decode b32. Points whose KV exceeds `KV_BYTES_CAP` (40 GiB) are dropped and listed.

## SOL model (`sol.py`)
Optimistic useful-work floor. Bytes: all linear weights once, norm weights, tied embedding / LM head counted ONCE
(151936x1024 bf16 = 311,164,928 B; same as upstream megabench `sol.py`), KV prefix read, new K/V write, token ids,
fp32 logits out. R0 = 1,207,502,352 B = 150.94 us at 8 TB/s (identical to upstream). FLOPs: 2 x linear params x L x
total_q + 4 L QH D x (attended positions summed over new tokens) + 2 V H n_seq. `T_sol = max(bytes/BW, flops/PEAK)`;
`ridge_tokens` is the total_q where the two cross with bytes and non-linear FLOPs held fixed.
Spec constants: BW 8.0 TB/s; BF16 dense 2.25 PFLOP/s (DGX B200 datasheet 72 PFLOPS FP8 sparse / 8 GPUs, /4 for
dense BF16; hypothesis until verified). FP32 FFMA peak = 148 x 128 x 2 x clock (clock recorded). A "measured" SOL
uses the job's HBM copy bandwidth and the cuBLAS 8192^3 bf16 `torch.mm` peak (best of 20), probed once per job.

## Timing
Per workload all prepared steps are warmed, then timed with randomised arm order every round (`harness.timing.interleaved`).
`steady` = mean of n back-to-back launches per sample, n from a one-launch estimate so a sample is ~20 ms (3..50);
`isolated` = one launch between syncs (5 per round). Stats: median, bootstrap 95% CI, p05/p95. Clocks are logged each
block (NVML). tokens/s = total_q / steady median. vLLM runs out of process and reports engine step wall time
(includes scheduler/host time; documented in the record config).

## Accuracy
Reference (`ref/reference.py`): BF16 serving semantics (BF16 activations, FP32 norm/softmax reductions, BF16
scores/probs) and an FP32 oracle. An arm passes when, for logits / k_new / v_new, its rel-L2 to the FP32 oracle is
<= 2x the BF16 reference's own rel-L2 to the oracle; next_token agreement rate is recorded, not gated. Checked at
`check=True` points and R0, for every seed (7301, 8917, 42811).

## Fairness rules
- Every sequence has its own KV region/pages (no shared pages, no cross-sequence L2 reuse).
- Timed `launch` writes K/V only into the arm's own cache; fp32 logit casts and k_new/v_new gathers are untimed (`outputs()`).
- hv2-repeat is a proxy: N = total_q repeated b1/kv128 decodes of the unchanged hv2 binary; it computes the wrong
  thing on purpose, is never accuracy-graded and is plotted dashed.
- No arm is tuned per regime beyond what is written in SPEC.md. Baselines autotune once per workload (recorded).
- Every claim in reports is measured (run id, slurm job, log path) or labelled hypothesis.
