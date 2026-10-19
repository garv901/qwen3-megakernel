"""FP32 reference for one Qwen3-0.6B decode step (batch 1, position 128).

This is a port of `decode_step` of the MegaBench torch reference with `dtype=float32`.
All activations are fp32. The new token's K and V enter attention unrounded.
Attention runs over the whole static cache under a `position <= pos` mask.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

LAYERS, QH, KVH, D = 28, 16, 8, 128
EPS = 1e-6
ROPE_THETA = 1e6
CONTEXT = 128
WEIGHT_NAMES = ("ln1", "wq", "wk", "wv", "qn", "kn", "wo", "ln2", "wg", "wu", "wd", "fnorm", "embed")


def rms(x: torch.Tensor, weight: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """RMSNorm. The reduction runs in fp32, then the result is cast to the dtype of x and weighted."""
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return normalized.to(x.dtype) * weight.to(x.dtype)


def rope_tables(s_max: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin [s_max, D], fp32, Hugging Face layout (the half-dim table twice)."""
    inv = ROPE_THETA ** (-torch.arange(0, D, 2, dtype=torch.float64) / D)
    freqs = torch.outer(torch.arange(s_max, dtype=torch.float64), inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().float().to(device), emb.sin().float().to(device)


def rot_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def to_fp32(inputs: dict) -> dict:
    """The weights of a kernel input dict, cast to fp32."""
    return {n: inputs[n].float() for n in WEIGHT_NAMES}


def decode_step(w: dict, token: torch.Tensor, kcache: torch.Tensor, vcache: torch.Tensor, pos: int = CONTEXT) -> dict:
    """One decode step in fp32.

    w: fp32 weights (see `to_fp32`). token: int64 scalar. kcache, vcache: [28, pos, 8, 128] (any float dtype).
    Returns `logits` [V] fp32, `next_token` int64 scalar, `k_write` and `v_write` [28, 8, 128] fp32.
    """
    dev = token.device
    s_max = pos + 1
    kc = torch.zeros(LAYERS, 1, s_max, KVH, D, dtype=torch.float32, device=dev)
    vc = torch.zeros_like(kc)
    kc[:, :, :pos] = kcache.to(device=dev, dtype=torch.float32).unsqueeze(1)
    vc[:, :, :pos] = vcache.to(device=dev, dtype=torch.float32).unsqueeze(1)
    cos_t, sin_t = rope_tables(s_max, dev)
    pos_t = torch.full((1,), pos, dtype=torch.int64, device=dev)
    bidx = torch.arange(1, device=dev)
    visible = torch.arange(s_max, device=dev)[None, :] <= pos_t[:, None]
    cos = cos_t[pos_t].unsqueeze(1)
    sin = sin_t[pos_t].unsqueeze(1)

    x = w["embed"][token.reshape(1)]                                # [1, H]
    ks, vs = [], []
    for l in range(LAYERS):
        xn = rms(x, w["ln1"][l])
        q = (xn @ w["wq"][l].T).view(1, QH, D)
        k = (xn @ w["wk"][l].T).view(1, KVH, D)
        v = (xn @ w["wv"][l].T).view(1, KVH, D)
        q = rms(q, w["qn"][l])
        k = rms(k, w["kn"][l])
        q = q * cos + rot_half(q) * sin
        k = k * cos + rot_half(k) * sin
        kc[l][bidx, pos_t] = k
        vc[l][bidx, pos_t] = v
        ks.append(k)
        vs.append(v)
        o = _attention(q, kc[l], vc[l], visible, QH // KVH)
        x = x + o.reshape(1, QH * D) @ w["wo"][l].T
        xn = rms(x, w["ln2"][l])
        gate = xn @ w["wg"][l].T
        up = xn @ w["wu"][l].T
        x = x + (F.silu(gate) * up) @ w["wd"][l].T

    logits = (rms(x, w["fnorm"]) @ w["embed"].T).float()
    return {"logits": logits[0], "next_token": logits.argmax(dim=-1)[0],
            "k_write": torch.stack(ks, dim=1)[0], "v_write": torch.stack(vs, dim=1)[0]}


def _attention(q, k_all, v_all, visible, groups: int):
    """q [1, QH, D]. k_all and v_all [1, S, KVH, D]. visible bool [1, S]."""
    k = k_all.repeat_interleave(groups, dim=2).transpose(1, 2)    # [1, QH, S, D]
    v = v_all.repeat_interleave(groups, dim=2).transpose(1, 2)
    scores = (q.unsqueeze(2) @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    scores = scores.masked_fill(~visible[:, None, None, :], -torch.inf)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return (probs @ v).squeeze(2)                                 # [1, QH, D]
