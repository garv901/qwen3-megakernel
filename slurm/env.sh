# Environment for regimebench GPU jobs; sourced by slurm/regime.sbatch.
# Same caches as vibe-megakernel (/raid/garv901/.cache) so FlashInfer cubins and vLLM caches are reused.
C=/raid/garv901/.cache
# Repo = the checkout the job was submitted from (main or a worktree such as /raid/garv901/regime-mk-small).
REPO=${REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo /raid/garv901/regime-megakernel)}
export REPO
export PY=/raid/garv901/vibe-megakernel/.venv/bin/python
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH=$REPO
export HF_HOME=/raid/hf HF_HUB_OFFLINE=1
export FLASHINFER_WORKSPACE_BASE=$C/flashinfer FLASHINFER_CUBIN_DIR=$C/flashinfer/cubins
# Local trtllm-gen cubins; skips vLLM's artifactory probe (only valid once the download finished).
if [ -f "$C/flashinfer/CUBINS_COMPLETE" ]; then export VLLM_HAS_FLASHINFER_CUBIN=1; fi
export TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton
export VLLM_CACHE_ROOT=$C/vllm VLLM_CONFIG_ROOT=$C/vllm_config
export XDG_CACHE_HOME=$C/xdg TORCH_EXTENSIONS_DIR=$C/torch_extensions CUDA_CACHE_PATH=$C/nv_compute_cache
mkdir -p "$REPO/runs" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE"
