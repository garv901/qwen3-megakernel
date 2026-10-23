"""Synthetic inputs for one Workload (weights follow megabench tasks/dense.make_inputs).

``make_inputs(w, seed, device)`` returns a dict:

  weights   MegaBench names, bf16 (same recipe and generator order as
            megabench/tasks/dense.py, so a given seed gives the same weights):
            ln1 [L,H]  wq [L,QD,H]  wk [L,KD,H]  wv [L,KD,H]  qn [L,D]  kn [L,D]
            wo [L,H,QD]  ln2 [L,H]  wg [L,I,H]  wu [L,I,H]  wd [L,H,I]
            fnorm [H]  embed [V,H]
  tokens    int64 [total_q]              new tokens, packed in cu_q order
  kcache    bf16 [L, cache_tokens, KVH, D]  packed per-sequence regions
  vcache    bf16 [L, cache_tokens, KVH, D]
  cu_cache  int64 [n_seq+1] (CPU)        sequence i owns rows cu_cache[i] .. cu_cache[i+1]
                                         (= kv_len_i + q_len_i rows); the first kv_len_i rows
                                         hold the random prefix (randn * 0.1), the rest are zero
  cu_q      int64 [n_seq+1] (CPU)
  kv_len    int64 [n_seq]   (CPU)
  q_len     int64 [n_seq]   (CPU)

Every sequence has its own prefix values (no shared pages), so arms cannot get
L2 reuse across sequences that a real server would not get.
"""

from __future__ import annotations

import torch

from .workload import Workload

WEIGHT_NAMES = ("ln1", "wq", "wk", "wv", "qn", "kn", "wo", "ln2", "wg", "wu",
                "wd", "fnorm", "embed")

_CHUNK_ROWS = 1 << 15


def make_weights(w: Workload, seed: int, device: str) -> tuple[dict, torch.Generator]:
    g = w.model
    gen = torch.Generator(device=device).manual_seed(seed)
    h, layers, inter, d = g.hidden, g.layers, g.intermediate, g.head_dim
    qdim, kdim = g.q_dim, g.kv_dim

    def weight(*shape: int, fan_in: int) -> torch.Tensor:
        return (torch.randn(shape, generator=gen, device=device) *
                fan_in ** -0.5).to(torch.bfloat16)

    def norm(*shape: int) -> torch.Tensor:
        return (1 + 0.1 * torch.randn(shape, generator=gen, device=device)).to(torch.bfloat16)

    # Same draw order as megabench tasks/dense.make_inputs (token first).
    _ = torch.randint(g.vocab, (), generator=gen, device=device, dtype=torch.int64)
    out = {
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
        "embed": (torch.randn((g.vocab, h), generator=gen, device=device) * 0.02).to(torch.bfloat16),
    }
    return out, gen


def make_inputs(w: Workload, seed: int, device: str) -> dict:
    g = w.model
    inputs, gen = make_weights(w, seed, device)
    inputs["tokens"] = torch.randint(g.vocab, (w.total_q,), generator=gen,
                                     device=device, dtype=torch.int64)
    cu_cache = [0]
    for s in w.seqs:
        cu_cache.append(cu_cache[-1] + s.kv_len + s.q_len)
    rows = cu_cache[-1]
    for name in ("kcache", "vcache"):
        c = torch.zeros((g.layers, rows, g.kv_heads, g.head_dim),
                        dtype=torch.bfloat16, device=device)
        for i, s in enumerate(w.seqs):
            start = cu_cache[i]
            for off in range(0, s.kv_len, _CHUNK_ROWS):
                n = min(_CHUNK_ROWS, s.kv_len - off)
                c[:, start + off:start + off + n] = (torch.randn(
                    (g.layers, n, g.kv_heads, g.head_dim), generator=gen,
                    device=device) * 0.1).to(torch.bfloat16)
        inputs[name] = c
    inputs["cu_cache"] = torch.tensor(cu_cache, dtype=torch.int64)
    inputs["cu_q"] = torch.tensor(w.cu_q, dtype=torch.int64)
    inputs["kv_len"] = torch.tensor([s.kv_len for s in w.seqs], dtype=torch.int64)
    inputs["q_len"] = torch.tensor([s.q_len for s in w.seqs], dtype=torch.int64)
    return inputs
