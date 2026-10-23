# MegaBench SOTA environment (B200 / sm_100)

Venv: `/raid/garv901/vibe-megakernel/.venv` (Python 3.12.3). Marker: `/raid/garv901/.cache/VENV_READY`.
Full freeze: `megabench/sota/requirements.lock`.

## 1. Versions and install

| package | version |
|---|---|
| torch | 2.13.0+cu130 (`torch.version.cuda == 13.0`) |
| vllm | 0.31.0 |
| flashinfer-python | 0.7.0.post1 |
| flashinfer-jit-cache (+ -sm80/89/90a/100a/103a/120f) | 0.7.0.post1+cu130 |
| triton | 3.7.1 |
| nvidia-cutlass-dsl | 4.7.1 |
| transformers | 5.17.0 |
| numpy / matplotlib / ninja / nvidia-ml-py | 2.3.5 / 3.11.2 / 1.13.2 / 13.615.71 |

Why cu130: vllm 0.31.0 from PyPI pins `torch==2.13.0`, and the default PyPI torch 2.13.0 wheel is already the cu130 build
(it pulls nvidia-*-13.0 libs). That matches the megakernel's validated env (torch 2.13.0+cu130, nvcc 13.1, driver 580).
No extra index was needed for torch.

```bash
export UV_CACHE_DIR=/raid/garv901/.cache/uv
UV=/raid/keisuke/.local/bin/uv
$UV venv --python 3.12 /raid/garv901/vibe-megakernel/.venv
PY=/raid/garv901/vibe-megakernel/.venv/bin/python
$UV pip install --python $PY vllm==0.31.0 matplotlib ninja nvidia-ml-py numpy
# FlashInfer prebuilt JIT kernels (all arches; the sm100a package is the one used on B200)
$UV pip install --python $PY flashinfer-jit-cache==0.7.0.post1 --extra-index-url https://flashinfer.ai/whl/cu130
# cubins: the PyPI package flashinfer-cubin==0.7.0.post1 does NOT exist (max 0.6.13). Download cubins instead (login node has network):
FLASHINFER_WORKSPACE_BASE=/raid/garv901/.cache/flashinfer \
FLASHINFER_CUBIN_DIR=/raid/garv901/.cache/flashinfer/cubins \
  $PY -m flashinfer download-cubin     # ~42.7k files, takes ~25+ min; log: /raid/garv901/.cache/cubin_dl.log
```
Never run `python` with cwd inside `site-packages/vllm` (its `tokenizers/` dir shadows the HF package and breaks `import vllm`).

## 2. Env vars for the Slurm script

```bash
C=/raid/garv901/.cache
export FLASHINFER_WORKSPACE_BASE=$C/flashinfer        # JIT cache lands in $BASE/.cache/flashinfer (env.py: BASE/.cache/flashinfer)
export FLASHINFER_CUBIN_DIR=$C/flashinfer/cubins      # cubin dir (overrides flashinfer-cubin pkg; not installed here)
export VLLM_HAS_FLASHINFER_CUBIN=1                    # tells vLLM cubins are local; skips the 5s artifactory probe (has_nvidia_artifactory),
                                                      # which otherwise disables TRTLLM-gen decode on a node without network. Set ONLY once the download finished.
export TORCHINDUCTOR_CACHE_DIR=$C/torchinductor
export TRITON_CACHE_DIR=$C/triton
export VLLM_CACHE_ROOT=$C/vllm                        # torch.compile cache, etc.
export VLLM_CONFIG_ROOT=$C/vllm_config
export XDG_CACHE_HOME=$C/xdg
export TORCH_EXTENSIONS_DIR=$C/torch_extensions
export CUDA_CACHE_PATH=$C/nv_compute_cache
export HF_HOME=/raid/hf
export HF_HUB_OFFLINE=1
```
Optional switches: `FLASHINFER_USE_CUDA_NORM=1` (see 3), `VLLM_ENABLE_V1_MULTIPROCESSING=0` (default `1`; set 0 to run the
engine core in-process so monkeypatches apply and profilers see the worker), `VLLM_USE_FLASHINFER_SAMPLER` (default True),
`VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE` (default 394 MiB), `VLLM_LORA_DISABLE_PDL` (LoRA only).

## 3. FlashInfer 0.7.0.post1 API (inspected)

`enable_pdl` / `pdl` default `None` means "auto": `device_support_pdl(device)` which is `compute_capability major >= 9`, so PDL is ON by default on sm_100.
`device_support_pdl(device: torch.device) -> bool`: False for non-cuda; True if major >= 9.

- `norm.rmsnorm(input, weight, eps=1e-06, out=None, enable_pdl=None) -> Tensor`. 2D `[T,H]` input. (3D input goes to the qk-rmsnorm path.)
- `norm.fused_add_rmsnorm(input, residual, weight, eps=1e-06, enable_pdl=None) -> None`. In place: `residual += input; input = rmsnorm(residual)`.
- Norm gotcha: code is `if enable_pdl is None or enable_pdl: enable_pdl = device_support_pdl(...)`, so passing `False` really disables PDL; `True` is clamped to device support.
- Norm backend: default is CuTe DSL kernels (`nvidia-cutlass-dsl`, first call JIT-compiles with cute). `FLASHINFER_USE_CUDA_NORM=1` forces the CUDA-JIT C++ kernels (`jit/norm.py`, from jit-cache). Also auto-falls back if DSL cannot target the arch. Both honor `enable_pdl`.
- `rope.apply_rope_with_cos_sin_cache_inplace(positions, query, key, head_size, cos_sin_cache, is_neox=True) -> None`. No PDL arg. query/key are `[nnz, heads*head_dim]` or `[nnz, heads, dim]` flattened per vLLM convention; cache is `[max_pos, rot_dim]` with cos|sin halves, fp32 typical.
- `rope.apply_rope_pos_ids_inplace(q, k, pos_ids, rotary_dim=None, interleave=False, rope_scale=1, rope_theta=10000.0) -> None`. No PDL arg. `[nnz, heads, dim]`; `interleave=False` is NeoX (half-split) style. Computes cos/sin on the fly from theta.
- `page.append_paged_kv_cache(append_key, append_value, batch_indices, positions, paged_kv_cache, kv_indices, kv_indptr, kv_last_page_len, kv_layout='NHD') -> None`. No PDL arg. `paged_kv_cache` is a (k_cache, v_cache) tuple or stacked tensor. batch_indices/positions come from `flashinfer.get_batch_indices_positions(append_indptr, seq_lens, nnz)`.
- `activation.silu_and_mul(input, out=None, enable_pdl=None) -> Tensor`. input `[..., 2*d]` (gate first, up second), out `[..., d]`. None -> auto PDL on.
- `BatchDecodeWithPagedKVCacheWrapper.__init__(float_workspace_buffer, kv_layout='NHD', use_cuda_graph=False, use_tensor_cores=False, paged_kv_indptr_buffer=None, paged_kv_indices_buffer=None, paged_kv_last_page_len_buffer=None, backend='auto', jit_args=None)`.
  Valid `backend`: `auto`, `fa2`, `fa3`, `trtllm-gen`, `cute-dsl`, `prims-ts` (docstring); the code also references `cutile`. With `use_cuda_graph=True` the three `*_buffer` args must be provided (fixed-address GPU int32 buffers). `use_tensor_cores=True` is the right setting for GQA (Qwen3-0.6B has 16 q / 8 kv heads, group 2). `prims-ts` requires HND and does not support `use_cuda_graph=True`. `trtllm-gen` requires HND layout (code raises for NHD).
- `.plan(indptr, indices, last_page_len, num_qo_heads, num_kv_heads, head_dim, page_size, *, pos_encoding_mode='NONE', window_left=-1, logits_soft_cap=None, q_data_type=..., kv_data_type=..., ...)` (kwargs). Has no PDL arg. indptr/indices/last_page_len int32. plan runs host-side work (not graph-capturable); call it before capture, and re-plan outside the graph if lengths change.
- `.run(q, paged_kv_cache, *args, q_scale=None, k_scale=None, v_scale=None, out=None, lse=None, return_lse=False, enable_pdl=None, window_left=None, sinks=None, q_len_per_req=None, skip_softmax_threshold_scale_factor=None, kv_cache_sf=None)`. `enable_pdl=None` -> `device_support_pdl(q.device)` (ON). Pass `out=` for graph-safe, allocation-free runs.
- `decode.trtllm_batch_decode_with_kv_cache(query, kv_cache, workspace_buffer, block_tables, seq_lens, max_seq_len, bmm1_scale=1.0, bmm2_scale=1.0, window_left=-1, out=None, out_dtype=None, ..., kv_layout='HND', enable_pdl=None, backend='auto', q_len_per_req=1, ...)`. `enable_pdl=None` -> device support (ON). `backend`: `auto`/`trtllm-gen`/`xqa` (xqa for sm90/sm12x). KV layout default HND. Needs the cubin download (trtllm-gen FMHA cubins) on first use; cubin download needs network, so pre-download on the login node.
- `mm_bf16(a, b, bias=None, pdl=False, out=None, out_dtype=torch.bfloat16, backend='cudnn')`. Note: `pdl` defaults to False (explicit opt-in). `a`: `[m,k]` bf16 row-major. `b`: `[k,n]` bf16 COLUMN-major, i.e. pass `weight[n,k].T` (a view of the usual nn.Linear weight, no copy). Backends: `cudnn`, `cutlass`, `tgv`, `cublaslt`, `tinygemm`, `cutile`, `cute-dsl`, `auto`. Supports `pdl`: `tgv`, `tinygemm`, `cute-dsl` (the cute-dsl M>32 fallback ignores it). `cutile` ignores `pdl` (must pass False). `bias`: tgv, tinygemm, cute-dsl, cublaslt. `tinygemm`/`cute-dsl` require bf16 out. `cute-dsl` is low-M (<=32) Blackwell kernels, never auto-selected, needs CuTe DSL >= 4.7 (4.7.1 installed). Also `flashinfer.gemm.tinygemm_bf16`, `tgv_gemm_sm100`, `bmm_bf16`.
- Sampling helpers (`flashinfer.sampling`): `sampling_from_logits`, `sampling_from_probs`, `top_k_sampling_from_probs`, `top_p_sampling_from_probs`, `top_k_top_p_sampling_from_logits`, `min_p_sampling_from_probs`, `softmax`, `top_k_mask_logits`. There is NO dedicated argmax; use `torch.argmax` (or flashinfer `top_k_top_p...` with k=1). Sampling from logits with temperature 0 is not the same as greedy argmax; use torch.argmax for greedy.

## 4. vLLM 0.31.0

`LLM(model, *, runner, convert, tokenizer, tokenizer_mode, skip_tokenizer_init, trust_remote_code, tensor_parallel_size, dtype='auto', quantization, revision, seed=0, gpu_memory_utilization=0.92, enforce_eager=False, attention_config=None, kv_cache_memory_bytes=None, compilation_config=None, ..., **kwargs)`.
Extra `**kwargs` are EngineArgs fields: `block_size`, `enable_prefix_caching`, `max_num_seqs`, `max_num_batched_tokens`, `max_model_len`, `async_scheduling`, `disable_log_stats` (default False), `load_format` (e.g. `dummy`/`auto`), `enable_chunked_prefill`, `linear_backend` (new: `--linear-backend`, selects e.g. flashinfer mm_bf16 backends for unquantized linears; default `auto` = torch/cuBLAS).

- Attention backend: `attention_config={"backend": "FLASHINFER"}` (AttentionBackendEnum; fields: `backend`, `use_trtllm_attention` (None/True/False), `disable_flashinfer_q_quantization`, ...). Not an env var any more.
- Compilation: `compilation_config={"mode": ..., "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1], "max_cudagraph_capture_size": ...}`. `CUDAGraphMode` values: `NONE`, `PIECEWISE`, `FULL`, `FULL_DECODE_ONLY`, `FULL_AND_PIECEWISE` (default for most configs). Also `cudagraph_num_of_warmups`, `cudagraph_copy_inputs`.
- Prompt of raw ids: `llm.generate([{"prompt_token_ids": [...]}], sp)` or `from vllm.inputs import TokensPrompt; TokensPrompt(prompt_token_ids=[...])`.
- `SamplingParams(temperature=0.0, max_tokens=N, ignore_eos=True, detokenize=False, min_tokens=N, ...)` (msgspec struct; fields include `n, temperature, top_p, top_k, seed, stop, ignore_eos, max_tokens, min_tokens, detokenize, output_kind, stream_interval`).
- Per-token timing: `RequestOutput.metrics` is `RequestStateStats` with `num_generation_tokens, num_preemptions, arrival_time (wall), queued_ts, scheduled_ts, first_token_ts, last_token_ts (engine-core monotonic), first_token_latency, is_corrupted`. It is populated only when stats are on: `disable_log_stats=False` (the default); with `True` `metrics` is None. Only first/last token timestamps exist (no per-token list): mean ITL = (last_token_ts - first_token_ts)/(num_generation_tokens-1). For per-token latency use streaming (AsyncLLM/`output_kind=DELTA`) with host timestamps, or `trace_decode_token_ids`-style tracing, which is not timing.
- `VLLM_ENABLE_V1_MULTIPROCESSING` default `1` (engine core in separate process).
- Blackwell cc10.0 for dense Qwen3: backend priority is `FLASHINFER`, `FLASH_ATTN`, `TRITON_ATTN`, ... so FLASHINFER is the default on sm100 anyway. Decode uses `trtllm_batch_decode_with_kv_cache` (trtllm-gen, backend value from `attn_metadata.decode.kernel`) when `can_use_trtllm_attention` (needs cubins reachable: local cubins via `VLLM_HAS_FLASHINFER_CUBIN=1`, else artifactory probe) and `num_tokens <= 256` with `kv_cache_dtype == auto`; otherwise native `BatchDecodeWithPagedKVCacheWrapper`. KV write is vLLM's `torch.ops._C_cache_ops.reshape_and_cache_flash`; RoPE is vLLM's own rotary op (FlashInfer pos encoding disabled, `use_flashinfer=False`); linears are torch/cuBLAS (`default_unquantized_gemm`) unless `--linear-backend` is set.
- RMSNorm / SiLU: RMSNorm goes through `vllm.ir.ops.rms_norm` / `fused_add_rms_norm`. On CUDA the default IR priority is `["native"]` if Inductor compile is active (backend inductor, mode != NONE, the default), so RMSNorm and SiLU-and-mul are then Inductor-generated Triton kernels fused with neighbors, NOT vLLM CUDA ops and NOT FlashInfer. With `enforce_eager` / no Inductor, priority is `["vllm_c","native"]` = vLLM csrc `_C.rms_norm` / `_C.fused_add_rms_norm` and `_C.silu_and_mul`. FlashInfer norm/activation is never used by vLLM for Qwen3 here. Override with `kernel_config`/`ir_op_priority` (config/kernel.py `IrOpPriorityConfig`: `rms_norm`, `fused_add_rms_norm`). `VLLM_USE_OINK_OPS` adds oink for rms_norm.

### PDL in vLLM 0.31 (for the "PDL off" ablation)
- vLLM never passes `enable_pdl` to FlashInfer attention (grep of `v1/attention/backends/flashinfer.py`: no `enable_pdl`). Both `wrapper.run(...)` and `trtllm_batch_decode_with_kv_cache(...)` get the default `None`, so FlashInfer decides: `device_support_pdl(q.device)` -> True on cc10. So PDL is ON for FlashInfer attention, via FlashInfer's own default. FlashInfer norm/activation are not called by vLLM for this model.
- There is no global vLLM env var or config for PDL (only `VLLM_LORA_DISABLE_PDL`, LoRA only).
- Explicit `enable_pdl`/`pdl` call sites in vllm: `utils/flashinfer.py:52,68` `flashinfer_bf16_mm(..., pdl)` and `model_executor/layers/utils.py:633` pass `pdl=current_platform.is_arch_support_pdl()` (only when `--linear-backend` selects a FlashInfer mm_bf16 backend); `_custom_ops.py:3286` `dsv3_fused_a_gemm(enable_pdl=False)` (DeepSeek only); `enable_pdl=False` hard-coded in `v1/attention/backends/mla/{tokenspeed_mla,prefill/*}.py` (MLA only); `enable_pdl=True` in `fused_moe/experts/trtllm_mxfp4_moe.py:403`; `launch_pdl=current_platform.is_arch_support_pdl()` Triton launches in mamba ops, fla, `models/common/ops/fused_qk_rmsnorm.py` (Qwen3 q/k-norm fusion only if `enable_qk_norm_rope_fusion`/`fuse_rope_kvcache`), DeepSeek/Kimi/qwen4_exp ops; `model_executor/kernels/linear/cute_dsl/*` `_use_pdl()`. None of these are on the default dense Qwen3 BF16 decode path except FlashInfer attention.
- `current_platform.is_arch_support_pdl()` (platforms/cuda.py:755) = cc major >= 9.
- Ablation recipe: yes, monkeypatching works if done in the process that issues the launches, before CUDA graph capture. FlashInfer modules do `from .utils import device_support_pdl`, so patch EVERY module-level binding, not only `flashinfer.utils`:
  ```python
  import sys
  for m in list(sys.modules.values()):
      if getattr(m, "__name__", "").startswith("flashinfer") and hasattr(m, "device_support_pdl"):
          m.device_support_pdl = lambda device: False
  from vllm.platforms.cuda import CudaPlatformBase
  CudaPlatformBase.is_arch_support_pdl = classmethod(lambda cls: False)
  ```
  (verify the class name in `platforms/cuda.py`; `current_platform` is an instance of it.) Import `flashinfer.decode`, `flashinfer.norm`, `flashinfer.activation` first so they are in `sys.modules`. Run with `VLLM_ENABLE_V1_MULTIPROCESSING=0` so the engine core/worker is in-process (otherwise patch via `sitecustomize.py` on `PYTHONPATH` or `collective_rpc`). `trtllm_batch_decode_with_kv_cache` is decorated/cached per call but reads `device_support_pdl` at call time, so the patch takes effect; graphs captured after the patch bake in non-PDL launches. Uncertain: whether the trtllm-gen cubin kernels themselves are launched with the PDL attribute only when `enable_pdl` is true (expected yes, not verified on GPU). Alternative that does not rely on patching: pass `attention_config={"use_trtllm_attention": False}` to force the native FlashInfer wrapper (which also takes `enable_pdl=None`), still needs the patch to turn PDL off. vLLM csrc norm/silu kernels (`_C.*`) are closed binaries here; PDL usage there is unknown, though in the default Inductor mode they are not used at all.

## 5. PyTorch 2.13 Inductor
- No PDL option: `[k for k in dir(torch._inductor.config) if 'pdl' in k]` and same for `config.triton` are empty. Inductor never sets `launch_pdl` for its generated Triton kernels (only a passthrough in `runtime/triton_heuristics.py` for `compile_meta.get("launch_pdl", False)`), so torch.compile baselines have PDL off.
- Relevant defaults: `max_autotune=False`, `max_autotune_gemm=False`, `max_autotune_gemm_backends="ATEN,TRITON,CPP"`, `max_autotune_pointwise=False`, `coordinate_descent_tuning=False`, `coordinate_descent_check_all_directions=False`, `epilogue_fusion=True`, `aggressive_fusion=False`, `freezing=False`, `fx_graph_cache=True`, `benchmark_kernel=False`, `force_disable_caches=False`, `triton.cudagraphs=False`, `triton.cudagraph_trees=True`, `triton.cudagraph_skip_dynamic_graphs=False`, `triton.unique_kernel_names=True`, `triton.persistent_reductions=True`, `triton.multi_kernel=0`, `triton.autotune_at_compile_time=None`, `triton.enable_persistent_tma_matmul=False`.
- `torch.compile(mode="max-autotune")` turns on max_autotune + `triton.cudagraphs`; `"max-autotune-no-cudagraphs"` leaves cudagraphs off (use it if you capture your own graph or want to report eager-launch numbers); `"reduce-overhead"` = cudagraphs only. Set `coordinate_descent_tuning=True` via `torch._inductor.config` for extra tuning (slow compile). Use `fullgraph=True, dynamic=False`.
- Cudagraph skip check after warmup/run: `from torch._dynamo.utils import counters; counters["inductor"]["cudagraph_skips"]` (a `Counter` key, 0 if absent; incremented in `output_code.py`/`cudagraph_utils.py` whenever a graph is not cudagraphed, e.g. mutated inputs, CPU ops, dynamic shapes). Assert it is 0 (or absent) for a valid "with CUDA graphs" baseline; also `torch._dynamo.utils.counters["inductor"]` for other skip reasons; log with `TORCH_LOGS=cudagraphs`.
- Triton cache is separate from the Inductor cache (`TRITON_CACHE_DIR` vs `TORCHINDUCTOR_CACHE_DIR`); both must be set under /raid/garv901/.cache.

## 6. Status notes
- CPU tests: `python -m unittest discover -s megabench/tests -t .` -> 74 tests, OK (new venv).
- Cubin download is long; check `/raid/garv901/.cache/cubin_dl.log` and `du -sh /raid/garv901/.cache/flashinfer/cubins` (full set is several GB).
