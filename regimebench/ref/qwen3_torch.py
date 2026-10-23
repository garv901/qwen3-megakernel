"""Pure-torch, graph-safe Qwen3 dense decode step (torch.compile arm and FP32 reference).

    decode_step(w, state, dtype, chain=False) -> dict[str, Tensor]

``w`` holds MegaBench weight names (``ln1 wq wk wv qn kn wo ln2 wg wu wd fnorm
embed``) already cast to ``dtype`` (see ``weights_for``). ``state`` is a
``DecodeState`` with static device tensors: ``token`` int64 [B],
``kcache``/``vcache`` [L, B, S_max, KVH, D] in ``dtype``, ``pos`` int64 [B]
(the decode position = cached prefix length), and fp32 RoPE ``cos``/``sin``
tables [S_max, D] in the HF duplicated layout. The step writes the new K/V into
slot ``pos[b]`` and returns ``logits`` fp32 [B, V], ``next_token`` int64 [B],
``k_write``/``v_write`` [B, L, KVH, D]. ``chain=True`` also copies
``next_token`` into ``state.token`` in place.

``dtype=torch.bfloat16`` reproduces ``megabench/tasks/references/qwen3.py``
(BF16 activations, FP32 norm/softmax reductions, BF16 scores and probabilities
as in ``tasks/common.py``). ``dtype=torch.float32`` is the FP32 reference used
to grade the FP32 megakernel: all activations FP32, the new token's K/V enter
attention unrounded. Attention runs over the whole static cache under a
``position <= pos[b]`` mask. No ``.item()``, no host tables, no data-dependent
shapes or control flow, so the function can be captured or compiled.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..tasks.common import rms
from ..tasks.dense import QWEN3_WEIGHT_NAMES
from .shapes import ModelGeometry, Shape


@dataclass
class DecodeState:
    token: torch.Tensor      # int64 [B]
    pos: torch.Tensor        # int64 [B]
    kcache: torch.Tensor     # [L, B, S_max, KVH, D]
    vcache: torch.Tensor     # [L, B, S_max, KVH, D]
    cos: torch.Tensor        # fp32 [S_max, D]
    sin: torch.Tensor        # fp32 [S_max, D]
    geometry: ModelGeometry


def rope_tables(geometry: ModelGeometry, s_max: int,
                device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [s_max, D] fp32, HF layout (matches ``references/qwen3.py``)."""
    d = geometry.head_dim
    inv = geometry.rope_theta ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    freqs = torch.outer(torch.arange(s_max, dtype=torch.float64), inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().float().to(device), emb.sin().float().to(device)


def weights_for(inputs: dict, dtype) -> dict:
    return {n: inputs[n].to(dtype) for n in QWEN3_WEIGHT_NAMES}


def build_state(shape: Shape, inputs: dict, dtype, device,
                s_max: int | None = None) -> DecodeState:
    """Static cache holding the fixture prefix (broadcast over the batch), pos = context."""
    g, b, ctx = shape.model, shape.batch, shape.context
    s_max = ctx + 1 if s_max is None else s_max
    if s_max <= ctx:
        raise ValueError("s_max must exceed the context")
    caches = []
    for name in ("kcache", "vcache"):
        c = torch.zeros(g.layers, b, s_max, g.kv_heads, g.head_dim,
                        dtype=dtype, device=device)
        c[:, :, :ctx] = inputs[name].to(device=device, dtype=dtype).unsqueeze(1)
        caches.append(c)
    cos, sin = rope_tables(g, s_max, device)
    return DecodeState(
        token=inputs["token"].to(device).reshape(1).expand(b).clone(),
        pos=torch.full((b,), ctx, dtype=torch.int64, device=device),
        kcache=caches[0], vcache=caches[1], cos=cos, sin=sin, geometry=g)


def _rot_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _attention(q, k_all, v_all, visible, groups: int):
    """q [B,QH,D]; k_all/v_all [B,S,KVH,D]; visible bool [B,S]; BF16-style scores/probs."""
    k = k_all.repeat_interleave(groups, dim=2).transpose(1, 2)   # [B,QH,S,D]
    v = v_all.repeat_interleave(groups, dim=2).transpose(1, 2)
    scores = (q.unsqueeze(2) @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    scores = scores.masked_fill(~visible[:, None, None, :], -torch.inf)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return (probs @ v).squeeze(2)                                # [B,QH,D]


def decode_step(w: dict, state: DecodeState, dtype, chain: bool = False) -> dict:
    g = state.geometry
    qh, kvh, d = g.q_heads, g.kv_heads, g.head_dim
    pos = state.pos
    bsz = pos.shape[0]
    bidx = torch.arange(bsz, device=pos.device)
    visible = torch.arange(state.kcache.shape[2], device=pos.device)[None, :] <= pos[:, None]
    cos = state.cos[pos].unsqueeze(1).to(dtype)                  # [B,1,D]
    sin = state.sin[pos].unsqueeze(1).to(dtype)

    x = w["embed"][state.token]                                  # [B,H]
    ks, vs = [], []
    for l in range(g.layers):
        xn = rms(x, w["ln1"][l], g.eps)
        q = (xn @ w["wq"][l].T).view(bsz, qh, d)
        k = (xn @ w["wk"][l].T).view(bsz, kvh, d)
        v = (xn @ w["wv"][l].T).view(bsz, kvh, d)
        q = rms(q, w["qn"][l], g.eps)
        k = rms(k, w["kn"][l], g.eps)
        q = q * cos + _rot_half(q) * sin
        k = k * cos + _rot_half(k) * sin
        state.kcache[l][bidx, pos] = k
        state.vcache[l][bidx, pos] = v
        ks.append(k)
        vs.append(v)
        o = _attention(q, state.kcache[l], state.vcache[l], visible, qh // kvh)
        x = x + o.reshape(bsz, qh * d) @ w["wo"][l].T
        xn = rms(x, w["ln2"][l], g.eps)
        gate = xn @ w["wg"][l].T
        up = xn @ w["wu"][l].T
        x = x + (F.silu(gate) * up) @ w["wd"][l].T

    logits = (rms(x, w["fnorm"], g.eps) @ w["embed"].T).float()
    next_token = logits.argmax(dim=-1)
    if chain:
        state.token.copy_(next_token)
    return {"logits": logits, "next_token": next_token,
            "k_write": torch.stack(ks, dim=1), "v_write": torch.stack(vs, dim=1)}
