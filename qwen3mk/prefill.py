"""A real-prompt KV cache for the megakernel.

The kernel decodes at position 128 with a KV prefix of exactly 128 tokens.
`build_prefix` tokenizes a prompt and cuts it to 128 tokens. `prefill`
runs the FP32 torch reference over those 128 positions and returns the KV cache
(bf16, [28, 128, 8, 128]) and the logits of the last position.

Rules for the 128-token prefix:
  - A prompt with more than 128 tokens is cut: the first 128 tokens are the prefix.
  - A shorter prompt is an error. The kernel has no attention mask, so it cannot
    skip pad tokens.
  - The input id of position 128 (the 129th token) is token 129 of the prompt when
    the prompt has one. Otherwise it is the greedy token after the prefix.
"""

from __future__ import annotations

import torch

from . import reference as ref

PREFIX = ref.CONTEXT


def load_tokenizer(model_id_or_path: str = "Qwen/Qwen3-0.6B"):
    from transformers import AutoTokenizer

    from .weights import resolve_model_dir

    return AutoTokenizer.from_pretrained(str(resolve_model_dir(model_id_or_path)))


def build_prefix(tokenizer, prompt: str) -> tuple[list[int], int | None, dict]:
    """(128 prefix ids, input id of position 128 or None, info). None means: take the greedy token."""
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    n = len(ids)
    info = {"prompt_tokens": n, "cut": max(0, n - PREFIX - 1)}
    if n >= PREFIX + 1:
        return ids[:PREFIX], ids[PREFIX], info
    if n == PREFIX:
        return ids, None, info
    raise ValueError(f"the prompt has {n} tokens. The kernel decodes position {PREFIX} and has no attention mask, "
                     f"so the prompt must have at least {PREFIX} tokens (padding would be attended)")


@torch.no_grad()
def prefill(w: dict, ids: list[int], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP32 forward over 128 positions with causal attention.

    w: fp32 weights (`reference.to_fp32`). Returns kcache and vcache (bf16, [28, 128, 8, 128])
    and the fp32 logits [V] of the last position. The cache holds K after the head norm and RoPE."""
    assert len(ids) == PREFIX
    dev = torch.device(device)
    t = torch.tensor(ids, dtype=torch.int64, device=dev)
    cos, sin = ref.rope_tables(PREFIX, dev)                            # [T, D]
    cos, sin = cos[:, None, :], sin[:, None, :]                        # [T, 1, D]
    causal = torch.ones(PREFIX, PREFIX, dtype=torch.bool, device=dev).tril()
    groups = ref.QH // ref.KVH
    x = w["embed"][t]                                                  # [T, H]
    kcache, vcache = [], []
    for l in range(ref.LAYERS):
        xn = ref.rms(x, w["ln1"][l])
        q = (xn @ w["wq"][l].T).view(PREFIX, ref.QH, ref.D)
        k = (xn @ w["wk"][l].T).view(PREFIX, ref.KVH, ref.D)
        v = (xn @ w["wv"][l].T).view(PREFIX, ref.KVH, ref.D)
        q = ref.rms(q, w["qn"][l])
        k = ref.rms(k, w["kn"][l])
        q = q * cos + ref.rot_half(q) * sin
        k = k * cos + ref.rot_half(k) * sin
        kcache.append(k)
        vcache.append(v)
        kk = k.repeat_interleave(groups, dim=1).transpose(0, 1)       # [QH, T, D]
        vv = v.repeat_interleave(groups, dim=1).transpose(0, 1)
        scores = (q.transpose(0, 1) @ kk.transpose(-1, -2)) * ref.D ** -0.5
        scores = scores.masked_fill(~causal[None], -torch.inf)
        o = torch.softmax(scores, dim=-1) @ vv                         # [QH, T, D]
        x = x + o.transpose(0, 1).reshape(PREFIX, ref.QH * ref.D) @ w["wo"][l].T
        xn = ref.rms(x, w["ln2"][l])
        x = x + (torch.nn.functional.silu(xn @ w["wg"][l].T) * (xn @ w["wu"][l].T)) @ w["wd"][l].T
    logits = (ref.rms(x[-1:], w["fnorm"]) @ w["embed"].T).float()[0]
    return (torch.stack(kcache).to(torch.bfloat16).contiguous(), torch.stack(vcache).to(torch.bfloat16).contiguous(), logits)


def prefix_state(w_bf16: dict, tokenizer, prompt: str, device="cuda") -> tuple[dict, dict]:
    """Kernel inputs (weights plus token, kcache, vcache) for a prompt, and an info dict.

    `w_bf16` holds the weights on `device` (`weights.load_hf`)."""
    ids, next_id, info = build_prefix(tokenizer, prompt)
    w32 = ref.to_fp32(w_bf16)
    kc, vc, last_logits = prefill(w32, ids, device)
    if next_id is None:
        next_id = int(last_logits.argmax())
        info["input_id_source"] = "greedy token after the prefix"
    else:
        info["input_id_source"] = "token 129 of the prompt"
    info["input_id"] = next_id
    inputs = dict(w_bf16)
    inputs.update(token=torch.tensor(next_id, dtype=torch.int64, device=device), kcache=kc.to(device), vcache=vc.to(device))
    return inputs, info
