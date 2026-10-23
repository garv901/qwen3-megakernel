"""Qwen3-0.6B dense decode fixture and PyTorch reference."""

from __future__ import annotations

import torch

from .references.qwen3 import Config as Qwen3Config
from .references.qwen3 import RefDecoder

from ..cases import Case


QWEN3_WEIGHT_NAMES = (
    "ln1", "wq", "wk", "wv", "qn", "kn", "wo", "ln2", "wg", "wu",
    "wd", "fnorm", "embed",
)


def _qwen3_config(case: Case) -> Qwen3Config:
    p = case.params
    return Qwen3Config(
        hidden=p["hidden"], layers=p["layers"], q_heads=p["q_heads"],
        kv_heads=p["kv_heads"], head_dim=p["head_dim"],
        inter=p["intermediate"], vocab=p["vocab"],
        max_seq=p["context"] + 1,
    )


def make_inputs(case: Case, seed: int, device: str) -> dict[str, torch.Tensor]:
    p = case.params
    if p["batch"] != 1:
        raise NotImplementedError("Qwen3 oracle currently handles batch one")
    cfg = _qwen3_config(case)
    gen = torch.Generator(device=device).manual_seed(seed)
    h, layers, inter, d = cfg.hidden, cfg.layers, cfg.inter, cfg.head_dim
    qdim, kdim = cfg.q_heads * d, cfg.kv_heads * d

    def weight(*shape: int, fan_in: int) -> torch.Tensor:
        return (torch.randn(shape, generator=gen, device=device) *
                fan_in ** -0.5).to(torch.bfloat16)

    def norm(*shape: int) -> torch.Tensor:
        return (1 + 0.1 * torch.randn(shape, generator=gen,
                                      device=device)).to(torch.bfloat16)

    inputs = {
        "token": torch.randint(cfg.vocab, (), generator=gen,
                               device=device, dtype=torch.int64),
        "ln1": norm(layers, h),
        "wq": weight(layers, qdim, h, fan_in=h),
        "wk": weight(layers, kdim, h, fan_in=h),
        "wv": weight(layers, kdim, h, fan_in=h),
        "qn": norm(layers, d),
        "kn": norm(layers, d),
        "wo": weight(layers, h, qdim, fan_in=qdim),
        "ln2": norm(layers, h),
        "wg": weight(layers, inter, h, fan_in=h),
        "wu": weight(layers, inter, h, fan_in=h),
        "wd": weight(layers, h, inter, fan_in=inter),
        "fnorm": norm(h),
        "embed": (torch.randn((cfg.vocab, h), generator=gen,
                               device=device) * 0.02).to(torch.bfloat16),
        "kcache": (torch.randn((layers, p["context"], cfg.kv_heads, d),
                               generator=gen, device=device) * 0.1).to(torch.bfloat16),
        "vcache": (torch.randn((layers, p["context"], cfg.kv_heads, d),
                               generator=gen, device=device) * 0.1).to(torch.bfloat16),
    }
    return inputs


def reference(case: Case, t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cfg = _qwen3_config(case)
    weights = {name: t[name] for name in QWEN3_WEIGHT_NAMES}
    ref = RefDecoder(cfg, weights, device=str(t["token"].device))
    context = case.params["context"]
    ref.k_cache[:, :context] = t["kcache"]
    ref.v_cache[:, :context] = t["vcache"]
    ref.pos = context
    logits = ref.step(int(t["token"].item()))
    return {
        "logits": logits.float(),
        "next_token": logits.argmax().to(torch.int64),
        "k_write": ref.k_cache[:, context].contiguous(),
        "v_write": ref.v_cache[:, context].contiguous(),
    }
