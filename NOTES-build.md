# Build notes

This file records how the repository was made, every check that ran on the CPU, and what is still open.
The CPU work came first. The GPU section at the end gives the results of the GPU runs.

## Source of the kernel

- Original: `megabench/mkprobe/runs/20261012T232636Z-HV2-host/src/hv2_submission.py` (`KERNEL_SRC`, 2850 lines, 437 conditional directives).
- Public kernel: `csrc/qwen3_step.cu` (818 lines, no conditional directive).
- The script `tools/prune.py` makes the public kernel. The call is `python tools/prune.py ORIGINAL.cu csrc/qwen3_step.cu --public`.
  The original text came from `KERNEL_SRC` with no change. A CPU test (`Prune.test_prune_reproduces_the_shipped_kernel`) runs the script again and compares the result with the shipped file.

### What the script does

1. It reads the source line by line and tracks every object-like `#define`.
2. It evaluates `#if`, `#ifdef`, `#ifndef`, `#elif`, `#else`, `#endif` with `defined()`, integer compare, `&&`, `||`, `!`, `?:` and arithmetic.
   It keeps only the live branch. No conditional was undecidable. (`NB` comes from the command line. No conditional uses it.)
3. It expands the macros that a switch branch defined (function-like macros, empty macros, `true`/`false`, `R10_NIMAX`-style expressions).
   Probe hooks (`P2_*`, `P3_*`, `R10_SK`) were empty at the defaults. Their call lines are gone.
4. It removes the `#define`s that nothing uses any more (89 removed). It keeps tuning constants that the code still uses.
5. It drops the comments that belong to removed switches: the trailing comment of a switch default, comment lines that follow such a define, and comment runs next to a removed switch directive (69 comment lines).
6. With `--public` it applies `HAND_FIXES`, renames four tuning constants and adds the header.

### Hand-fixes (the places the script cannot decide)

These are in the `HAND_FIXES` list of `tools/prune.py`. The script stops if one of them does not match the expected number of times.

1. `rmsd_to_smem<!R10_QKE >(...)` becomes `rmsd_to_smem<true>(...)`. The comment about the removed switch is replaced.
2. `{ rmsd_to_smem<false >(...); do {} while (0); }` becomes `rmsd_to_smem<false>(...);`.
3. The empty statement `if (l + 1 < NL) ;` is deleted.
4. `template <bool WAIT = false >` loses a stray space.
5. `(size_t)(l) *` becomes `(size_t)l *` (5 sites). `(size_t)(ll) *` becomes `(size_t)ll *` (3 sites). These came from the expanded `R10_WL(l)`.
6. The SiLU expression loses the parentheses of the expanded `R11_SILU(g)`.
7. `gemv_load_lf`: the unused template parameter `G` (a switch value) is removed. The call `gemv_load_lf<R10_LFQ>` becomes `gemv_load_lf`.
8. `if (R10_GUE_AT == 0 || !fused) issue_gue(l);` becomes `if (!fused) issue_gue(l);` (the default `R10_GUE_AT` is 3).
9. `R10_PFS` (default 0) is folded into the warp tests: `(unsigned)(warp - R10_PFS) < 2u` becomes `warp < 2` (2 sites), `warp == R10_PFS ?` becomes `warp == 0 ?` (3 sites), `warp == R10_PFS + 1` becomes `warp == 1`, `warp == R10_PFS + 2` becomes `warp == 2`.
10. Comments edited: "as in V1q" (2 sites), the `prep roles` comment, the `_dyn_smem` references in two `static_assert` messages.
11. Renames (regex, no semantic change): `R8_SPIN_*` to `SPIN_*`, `R8_STAGE_BYTES` to `STAGE_BYTES`, `r8_stage` to `stage`, `R10_LMH` to `LM_ROWS`.
12. Two comment lines are added above `SPIN_CYC` and `LM_ROWS`. The header comment of the file is added.

Left as is on purpose: unused helper functions that the original also carried (`row_dot`, `gemv_dot`, `load_act`, `rms_to_smem` and others).
Remaining tuning constants that were switch defaults: `SPIN_*` (barrier spin cycles), `LM_ROWS`.

## Gate: same binary

Compile options for both sources (hv2 `_compile`, sm_100): `--gpu-architecture=sm_100 -std=c++17 --fmad=true -default-device -DNB=148`.
`kernel.py` uses the same list. The normalized SASS is `cuobjdump -sass` without address and encoding comments (the `normalize_sass` of `opus_adapter`), hashed with SHA-256.

| Source | normalized SASS sha256 | regs | spill bytes | static smem |
|---|---|---|---|---|
| original hv2 `KERNEL_SRC` | `a4454be9373e5d21521a36079d3b91069e62261b1db300ca6b95798376811c07` | 126 | 0 | 47568 |
| `csrc/qwen3_step.cu` | `a4454be9373e5d21521a36079d3b91069e62261b1db300ca6b95798376811c07` | 126 | 0 | 47568 |

The two shas are equal. The sha starts with `a4454be9`, the fx1 sha recorded with the base options of megabench. The option sets happen to give the same SASS.
Every intermediate stage of the pruning (before and after macro expansion, with and without the hand-fixes) also gave this sha.

Dynamic shared memory: 5632 + 86016 + 43008 = 134656 bytes. The test compares this with `hv2._dyn_smem()`.
Scratch floats: 8192. Sync ints: 128. The tensor-map path (`R11_SKD`) is off at the defaults, so the host does not write a tensor map.

## CPU checks and their output

All commands ran with `/raid/garv901/vibe-megakernel/.venv/bin/python`, `HF_HOME=/raid/hf`, `HF_HUB_OFFLINE=1`.

- `python -m unittest tests.test_cpu -v`: 13 tests, OK. They cover: NVRTC compile for sm_100, 0 spill, regs 126 (<= 128), static smem 47568, SASS sha gate, no switch left in the source, `DYN_SMEM_BYTES` equal to `hv2._dyn_smem()`, scratch and sync sizes, input order, original-vs-public SASS equality, the prune script reproducing the shipped file, the macro evaluator.
- `python -m py_compile` on every `.py` file: OK.
- Reference port, on CPU in fp32 with synthetic inputs (seed 1), against `megabench.sota.qwen3_torch.decode_step(..., torch.float32)`:
  logits bit-equal (rel-L2 0.0), `next_token` equal, `k_write` and `v_write` bit-equal.
- Synthetic inputs: names, shapes and dtypes equal those of `megabench.tasks.dense.make_inputs` (16 tensors). The values differ (other generator calls).
- `weights.load_hf` on CPU from `/raid/hf` (Qwen/Qwen3-0.6B): all 13 weight tensors have the shape, dtype and contiguity of `make_inputs`. `kcache`, `vcache` and `token` come from `prefill.py`.
  Mapped keys: see the table in the docstring of `qwen3mk/weights.py` (`model.embed_tokens.weight`, `model.layers.{i}.input_layernorm.weight`, `self_attn.{q,k,v,o}_proj.weight`, `self_attn.{q,k}_norm.weight`, `post_attention_layernorm.weight`, `mlp.{gate,up,down}_proj.weight`, `model.norm.weight`). The config check asserts hidden 1024, 28 layers, 16/8 heads, head dim 128, intermediate 3072, vocab 151936, rope theta 1e6.
- `prefill` on CPU: a 173-token prompt gives a 128-token prefix (cut 44) and input id = token 129 of the prompt. Shapes `kcache`/`vcache` [28, 128, 8, 128] bf16. The reference step on it ran. A 19-token prompt is padded on the left with 109 pad tokens (they are attended, because the kernel has no mask) and the input id is the greedy token.

## GPU runs (B200, driver 580.178.04)

Commands, from the repository root:

```
python -m unittest tests.test_gpu -v
python run.py check --weights synthetic --seed 0     # also seeds 1 and 2
python run.py check --weights hf
python run.py bench --weights synthetic --iters 200
python run.py bench --weights hf
python run.py demo
python run.py demo --prompt "too short"             # must exit 2
```

Results:

- `tests.test_gpu`: OK. `reference.decode_step` agrees with the megabench reference.
- `check`: PASS for synthetic seeds 0, 1, 2 and for the HF weights.
  HF weights: `next_token` equal, logits rel-L2 4.8e-7, k/v rel-L2 1.6e-3 to 1.7e-3, k/v within 1 bf16 ulp.
- `bench --weights hf`: isolated 378.6 us (CUDA events), 391.7 us (wall clock). Steady 373.6 us for each step.
- `demo`: the default prompt (173 tokens) gives the same token as the reference. A prompt of 2 tokens gives an error and exit 2.

Two changes came from the first GPU run:

1. The k/v gate was an absolute max-abs of 0.02. With the HF weights, k has values above 64, where 1 bf16 ulp is 0.5.
   The gate is now in bf16 ulps of max(|ref|, rms(ref)), with a limit of 2, plus a rel-L2 limit of 1e-2.
2. `build_prefix` padded a short prompt on the left. The kernel has no attention mask, so the pad tokens took part in attention.
   A prompt with fewer than 128 tokens is now an error.

## Open issues

- The 200-launch sync check uses a loose gate (logits rel-L2 <= 1e-5, k/v within 2 bf16 ulps) because the O-projection uses float `red.add`. Repeated launches can differ by rounding.
- The chained path alternates two token buffers by rewriting two argument slots on the host before each launch. The host cost of this is not measured.
- `isolated` does not flush the L2 cache unless `flush_l2=True` is passed to `bench.isolated`. The CLI has no flag for it.
- The kernel only supports position 128.
