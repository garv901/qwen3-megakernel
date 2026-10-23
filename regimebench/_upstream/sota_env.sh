# Environment for megabench.sota GPU jobs; sourced by the sbatch scripts.
# Every cache lives under /raid/garv901/.cache (see ../ENV.md).
C=/raid/garv901/.cache
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH=/raid/garv901/vibe-megakernel
export HF_HOME=/raid/hf HF_HUB_OFFLINE=1
export FLASHINFER_WORKSPACE_BASE=$C/flashinfer FLASHINFER_CUBIN_DIR=$C/flashinfer/cubins
# Local trtllm-gen cubins; skips vLLM's artifactory probe. Only valid after
# `python -m flashinfer download-cubin` has finished (marker written by hand).
if [ -f "$C/flashinfer/CUBINS_COMPLETE" ]; then export VLLM_HAS_FLASHINFER_CUBIN=1; fi
export TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton
export VLLM_CACHE_ROOT=$C/vllm VLLM_CONFIG_ROOT=$C/vllm_config
export XDG_CACHE_HOME=$C/xdg TORCH_EXTENSIONS_DIR=$C/torch_extensions CUDA_CACHE_PATH=$C/nv_compute_cache
export MEGABENCH_SOTA_CACHE=$C/megabench_sota
mkdir -p /raid/garv901/vibe-megakernel/megabench/sota/runs "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$VLLM_CACHE_ROOT" "$MEGABENCH_SOTA_CACHE" "$FLASHINFER_WORKSPACE_BASE"
