"""Arm and Step interfaces shared by every in-process arm (adapted from megabench/sota/backend.py).

An ``Arm`` builds a ``Step`` for one ``Workload`` and variant. ``prepare`` may
do anything (copy the fixture KV into its own cache layout, autotune, compile,
capture CUDA graphs); it is not timed. ``Step.launch`` is the timed unit: it
enqueues exactly one full forward step for the workload on
``torch.cuda.current_stream()`` with no host sync, no allocation, and no
data-dependent Python control flow, so the harness can call it back to back.

Output contract (``expected_outputs``), produced by the most recent launch:
  logits     fp32  [n_seq, vocab]           last new token of each sequence
  next_token int64 [n_seq]                  argmax of logits
  k_new      bf16  [total_q, layers, kv_heads, head_dim]  post-norm/RoPE K of every new token (packed, cu_q order)
  v_new      bf16  [total_q, layers, kv_heads, head_dim]
The timed ``launch`` only has to write K/V into the arm's own cache (as a real
server does). ``outputs()`` is NOT timed and may gather k_new/v_new from that
cache (and copy logits) to build the four outputs above; an arm must not add
extra copies to ``launch`` just to produce them.

``precision`` is "bf16" for every arm in this study (graded with the FP32
oracle band, see harness/accuracy.py). A proxy arm that computes the wrong
thing on purpose (hv2-repeat) sets ``config["proxy"] = True`` and the harness
skips its accuracy gate.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable

from ..workload import Workload

OUTPUT_NAMES = ("logits", "next_token", "k_new", "v_new")


def expected_outputs(w: Workload) -> dict[str, tuple[tuple[int, ...], str]]:
    m = w.model
    return {
        "logits": ((w.n_seq, m.vocab), "float32"),
        "next_token": ((w.n_seq,), "int64"),
        "k_new": ((w.total_q, m.layers, m.kv_heads, m.head_dim), "bfloat16"),
        "v_new": ((w.total_q, m.layers, m.kv_heads, m.head_dim), "bfloat16"),
    }


@runtime_checkable
class Step(Protocol):
    arm: str
    variant: str
    precision: str
    # JSON-serializable facts recorded with every result, e.g.
    # {"graph": True, "pdl": {"launches_with_pdl": 140, "launches_total": 365},
    #  "attn_backend": "trtllm-gen", "gemm": {"qkv": "cublas", ...},
    #  "proxy": False, "launches_per_step": 1, "source_sha256": "..."}
    config: dict[str, Any]

    def launch(self) -> None: ...

    def outputs(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class Arm(Protocol):
    name: str

    def variants(self) -> tuple[str, ...]:
        """All variants, headline first."""
        ...

    def supports(self, w: Workload, variant: str) -> str | None:
        """None when runnable, otherwise a human-readable skip reason (recorded)."""
        ...

    def prepare(self, w: Workload, inputs: dict, variant: str, device: str) -> Step:
        """Build a ready Step. ``inputs`` is the fixture (regimebench/fixture.py); never mutate it."""
        ...


# In-process arms, imported lazily. vLLM runs out of process (arms/vllm_worker.py).
ARMS: dict[str, str] = {
    "fi-graph": "regimebench.arms.fi_graph:FiGraphArm",
    "hv2-repeat": "regimebench.arms.hv2_repeat:Hv2RepeatArm",
    "hv2g": "regimebench.arms.hv2g:Hv2gArm",
}
OUT_OF_PROCESS_ARMS = ("vllm",)


def load_arm(name: str, **options: Any) -> Arm:
    if name not in ARMS:
        raise KeyError(f"unknown in-process arm {name!r}; known: {sorted(ARMS)}")
    module_name, _, class_name = ARMS[name].partition(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**options)
