"""Arm and Step interfaces shared by every in-process benchmark arm.

An ``Arm`` builds a ``Step`` for one shape and variant. ``prepare`` may do
anything (relayout weights, autotune, compile, capture CUDA graphs); it is not
timed. ``Step.launch`` is the timed unit: it enqueues exactly one decode step
on ``torch.cuda.current_stream()`` with no host sync, no allocation, and no
data-dependent Python control flow, so the harness can call it back to back,
capture it, or profile it.

Output contract (see ``shapes.expected_outputs``): ``logits`` fp32,
``next_token`` int64, ``k_write``/``v_write`` bf16 = the new token's
post-norm/RoPE K and V for every layer. ``Step.outputs`` returns the static
buffers the most recent ``launch`` wrote.

Chained mode (``set_chained(True)``) feeds each step's ``next_token`` back as
the next step's input token on the device, ideally without an extra kernel
(the megakernel aliases its token/next pointers; graph arms write argmax into
the token buffer). Every step still decodes position ``shape.context`` over the
same cached prefix, so the work per step is constant. Accuracy checks use
unchained mode with the fixture token.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable

from .shapes import Shape

OUTPUT_NAMES = ("logits", "next_token", "k_write", "v_write")


@runtime_checkable
class Step(Protocol):
    arm: str
    variant: str
    precision: str  # "fp32" or "bf16": which reference the accuracy gate uses
    # JSON-serializable facts recorded with every result, for example
    # {"graph": True, "attn_backend": "trtllm-gen", "gemm": "cublas",
    #  "pdl": {"launches_with_pdl": 140, "launches_total": 365},
    #  "source_sha256": "...", "nvcc_flags": [...]}.
    config: dict[str, Any]

    def launch(self) -> None:
        ...

    def outputs(self) -> dict[str, Any]:
        ...

    def set_chained(self, on: bool) -> None:
        ...

    def close(self) -> None:
        ...


class Arm(Protocol):
    name: str

    def variants(self) -> tuple[str, ...]:
        """All variants, headline first."""
        ...

    def supports(self, shape: Shape, variant: str) -> str | None:
        """None when runnable, otherwise a human-readable skip reason."""
        ...

    def prepare(self, shape: Shape, inputs: dict, variant: str,
                device: str) -> Step:
        """Build a ready Step. ``inputs`` uses MegaBench names and must not be mutated."""
        ...


# In-process arms, imported lazily so CPU tests and the vLLM subprocess never
# pull in FlashInfer or nvcc builds. vLLM runs out of process (vllm_worker.py).
ARMS: dict[str, str] = {
    "megakernel": "megabench.sota.arms.megakernel:MegakernelArm",
    "opus": "megabench.sota.arms.opus:OpusArm",
    "sota-graph": "megabench.sota.arms.sota_graph:SotaGraphArm",
    "torch-compile": "megabench.sota.arms.torch_compile:TorchCompileArm",
}
OUT_OF_PROCESS_ARMS = ("vllm",)


def load_arm(name: str, **options: Any) -> Arm:
    if name not in ARMS:
        raise KeyError(f"unknown in-process arm {name!r}; known: {sorted(ARMS)}")
    module_name, _, class_name = ARMS[name].partition(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**options)
