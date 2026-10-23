"""Reference forward for a ragged Workload (BF16 serving semantics or FP32 oracle).

    forward(w, inputs, dtype) -> {"logits", "next_token", "k_new", "v_new"}

``inputs`` is the fixture (``regimebench/fixture.py``). ``dtype=torch.bfloat16``
is the serving-precision reference (BF16 activations, FP32 norm/softmax
reductions, BF16 scores/probs, see ``ref/common.py``); ``dtype=torch.float32``
is the FP32 oracle (everything FP32, fixture weights and cache upcast).

Outputs follow ``arms.base.expected_outputs``: logits fp32 [n_seq, V] (last new
token of every sequence), next_token int64 [n_seq], k_new / v_new
[total_q, L, KVH, D] in ``dtype`` (post-norm, post-RoPE K; raw V of every new
token, packed in cu_q order). The fixture is never mutated. Linear layers run
over all tokens at once; attention runs per sequence, query-chunked so the
score tensor stays small (R4 T=8K, kv 32K decode).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..workload import Workload
from .common import attention, rms, rope

# Max score-tensor elements per attention call: QH * q_chunk * S.
_SCORE_ELEMS = 1 << 28


def _attend_sequence(q, k_all, v_all, kv_len: int, g) -> torch.Tensor:
    """q [n,QH,D] for new tokens at positions kv_len..kv_len+n-1; k_all/v_all [kv_len+n,KVH,D]."""
    n, s = q.shape[0], k_all.shape[0]
    chunk = max(1, min(n, _SCORE_ELEMS // (g.q_heads * s)))
    outs = []
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        qc = q[a:b]
        if b == n and b - a == 1:
            allowed = None  # last new token sees every position
            kk, vv = k_all, v_all
        else:
            hi = kv_len + b  # keys beyond the chunk's last query are never visible
            kk, vv = k_all[:hi], v_all[:hi]
            pos_q = torch.arange(kv_len + a, kv_len + b, device=q.device)[:, None]
            pos_k = torch.arange(hi, device=q.device)[None, :]
            allowed = pos_k <= pos_q
        outs.append(attention(qc, kk, vv, allowed=allowed))
    return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]


@torch.no_grad()
def forward(w: Workload, inputs: dict, dtype=torch.bfloat16) -> dict:
    g = w.model
    qh, kvh, d, layers = g.q_heads, g.kv_heads, g.head_dim, g.layers
    device = inputs["tokens"].device
    total_q = w.total_q
    cu_q, cu_cache = w.cu_q, inputs["cu_cache"].tolist()

    embed = inputs["embed"].to(dtype)
    x = embed[inputs["tokens"]]                                   # [T, H]
    positions = torch.cat([torch.arange(s.kv_len, s.kv_len + s.q_len, device=device)
                           for s in w.seqs])
    k_new = torch.empty((total_q, layers, kvh, d), dtype=dtype, device=device)
    v_new = torch.empty_like(k_new)

    def lw(name: str, l: int) -> torch.Tensor:
        return inputs[name][l].to(dtype)

    for l in range(layers):
        xn = rms(x, lw("ln1", l), g.eps)
        q = (xn @ lw("wq", l).T).view(total_q, qh, d)
        k = (xn @ lw("wk", l).T).view(total_q, kvh, d)
        v = (xn @ lw("wv", l).T).view(total_q, kvh, d)
        q = rope(rms(q, lw("qn", l), g.eps), positions, g.rope_theta)
        k = rope(rms(k, lw("kn", l), g.eps), positions, g.rope_theta)
        k_new[:, l] = k
        v_new[:, l] = v
        o = torch.empty_like(q)
        kc, vc = inputs["kcache"][l], inputs["vcache"][l]
        for i, s in enumerate(w.seqs):
            a, b, base = cu_q[i], cu_q[i + 1], cu_cache[i]
            if s.kv_len:
                k_all = torch.cat([kc[base:base + s.kv_len].to(dtype), k[a:b]])
                v_all = torch.cat([vc[base:base + s.kv_len].to(dtype), v[a:b]])
            else:
                k_all, v_all = k[a:b], v[a:b]
            o[a:b] = _attend_sequence(q[a:b], k_all, v_all, s.kv_len, g)
        x = x + o.reshape(total_q, qh * d) @ lw("wo", l).T
        xn = rms(x, lw("ln2", l), g.eps)
        gate = xn @ lw("wg", l).T
        up = xn @ lw("wu", l).T
        x = x + (F.silu(gate) * up) @ lw("wd", l).T

    last = torch.tensor([c - 1 for c in cu_q[1:]], device=device)
    logits = (rms(x[last], inputs["fnorm"].to(dtype), g.eps) @ embed.T).float()
    return {"logits": logits, "next_token": logits.argmax(dim=-1),
            "k_new": k_new, "v_new": v_new}
