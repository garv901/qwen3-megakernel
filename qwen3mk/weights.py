"""Kernel inputs: real Qwen3-0.6B weights from Hugging Face, or synthetic ones.

Both functions return a dict that uses the kernel input names (see `INPUT_ORDER`).
`load_hf` returns the weights only. The KV cache and the token come from
`qwen3mk.prefill`. `synthetic` returns every input.

Hugging Face tensor names that `load_hf` maps (layer index i, stacked over 28 layers):

    model.embed_tokens.weight                      -> embed   (also the LM head: tied embeddings)
    model.layers.{i}.input_layernorm.weight        -> ln1
    model.layers.{i}.self_attn.q_proj.weight       -> wq      [2048, 1024]
    model.layers.{i}.self_attn.k_proj.weight       -> wk      [1024, 1024]
    model.layers.{i}.self_attn.v_proj.weight       -> wv      [1024, 1024]
    model.layers.{i}.self_attn.q_norm.weight       -> qn      [128]
    model.layers.{i}.self_attn.k_norm.weight       -> kn      [128]
    model.layers.{i}.self_attn.o_proj.weight       -> wo      [1024, 2048]
    model.layers.{i}.post_attention_layernorm.weight -> ln2
    model.layers.{i}.mlp.gate_proj.weight          -> wg      [3072, 1024]
    model.layers.{i}.mlp.up_proj.weight            -> wu      [3072, 1024]
    model.layers.{i}.mlp.down_proj.weight          -> wd      [1024, 3072]
    model.norm.weight                              -> fnorm
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from .kernel import INPUT_SPECS

MODEL_ID = "Qwen/Qwen3-0.6B"
WEIGHT_NAMES = ("ln1", "wq", "wk", "wv", "qn", "kn", "wo", "ln2", "wg", "wu", "wd", "fnorm", "embed")
STATE_NAMES = ("token", "kcache", "vcache")

_LAYER_KEYS = {
    "ln1": "input_layernorm.weight",
    "wq": "self_attn.q_proj.weight",
    "wk": "self_attn.k_proj.weight",
    "wv": "self_attn.v_proj.weight",
    "qn": "self_attn.q_norm.weight",
    "kn": "self_attn.k_norm.weight",
    "wo": "self_attn.o_proj.weight",
    "ln2": "post_attention_layernorm.weight",
    "wg": "mlp.gate_proj.weight",
    "wu": "mlp.up_proj.weight",
    "wd": "mlp.down_proj.weight",
}
_CONFIG = {"hidden_size": 1024, "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
           "head_dim": 128, "intermediate_size": 3072, "vocab_size": 151936, "rope_theta": 1e6}


def resolve_model_dir(model_id_or_path: str = MODEL_ID) -> Path:
    """A local directory, or the Hugging Face cache snapshot of a model id.

    The cache location follows `HF_HOME`. Set `HF_HUB_OFFLINE=1` to forbid downloads."""
    if os.path.isdir(model_id_or_path):
        return Path(model_id_or_path)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id_or_path, allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.json"]))


def load_hf(model_id_or_path: str = MODEL_ID, device="cpu") -> dict:
    """Load the 13 weight tensors (bf16, HF [out, in] layout, stacked over layers) onto `device`."""
    from safetensors import safe_open

    root = resolve_model_dir(model_id_or_path)
    cfg = json.loads((root / "config.json").read_text())
    for k, v in _CONFIG.items():
        if cfg.get(k) != v:
            raise ValueError(f"config.json {k}={cfg.get(k)!r}, the kernel needs {v!r}")
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors file in {root}")
    handles = {}
    for f in files:
        h = safe_open(str(f), framework="pt", device="cpu")
        for key in h.keys():
            handles[key] = h

    def get(key: str) -> torch.Tensor:
        return handles[key].get_tensor(key)

    out = {}
    for name, suffix in _LAYER_KEYS.items():
        out[name] = torch.stack([get(f"model.layers.{i}.{suffix}") for i in range(_CONFIG["num_hidden_layers"])])
    out["fnorm"] = get("model.norm.weight")
    out["embed"] = get("model.embed_tokens.weight")  # tied: the LM head is the same matrix
    for name in WEIGHT_NAMES:
        shape, dt = INPUT_SPECS[name]
        t = out[name]
        if tuple(t.shape) != shape:
            raise ValueError(f"{name}: shape {tuple(t.shape)}, expected {shape}")
        out[name] = t.to(device=device, dtype=dt).contiguous()
    return {name: out[name] for name in WEIGHT_NAMES}


def synthetic(seed: int = 0, device="cuda") -> dict:
    """Random inputs with the shapes and dtypes of the kernel. Weights scale as fan_in ** -0.5,
    norm weights are 1 + 0.1 * N(0, 1), the KV cache is 0.1 * N(0, 1), the token is random."""
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(shape, generator=gen, device=device)

    def weight(name, fan_in):
        return (randn(*INPUT_SPECS[name][0]) * fan_in ** -0.5).to(torch.bfloat16)

    def norm(name):
        return (1 + 0.1 * randn(*INPUT_SPECS[name][0])).to(torch.bfloat16)

    return {
        "token": torch.randint(151936, (), generator=gen, device=device, dtype=torch.int64),
        "ln1": norm("ln1"), "wq": weight("wq", 1024), "wk": weight("wk", 1024), "wv": weight("wv", 1024),
        "qn": norm("qn"), "kn": norm("kn"), "wo": weight("wo", 2048), "ln2": norm("ln2"),
        "wg": weight("wg", 1024), "wu": weight("wu", 1024), "wd": weight("wd", 3072),
        "fnorm": norm("fnorm"),
        "embed": (randn(151936, 1024) * 0.02).to(torch.bfloat16),
        "kcache": (randn(28, 128, 8, 128) * 0.1).to(torch.bfloat16),
        "vcache": (randn(28, 128, 8, 128) * 0.1).to(torch.bfloat16),
    }
