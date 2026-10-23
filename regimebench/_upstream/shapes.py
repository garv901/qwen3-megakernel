"""Decode shapes, model geometry, fixtures, and byte floors for the SOTA harness.

A ``Shape`` is one decode step: ``batch`` sequences, each with ``context``
cached tokens, decoding the token at position ``context`` (so every sequence
attends ``context + 1`` positions). Only batch one has a MegaBench fixture
today; larger batches raise ``NotImplementedError`` so a later sweep has one
obvious place to add batched inputs and references.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ..cases import Case, select_cases
from ..sol import DEFAULT_B200_BANDWIDTH_TB_S, estimate_case

BASE_CASE_ID = "dense-step-qwen3-06b-b1-s128"


@dataclass(frozen=True)
class ModelGeometry:
    name: str
    hidden: int
    layers: int
    q_heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    rope_theta: float = 1e6
    eps: float = 1e-6
    tied_embeddings: bool = True

    @property
    def q_dim(self) -> int:
        return self.q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim


QWEN3_06B = ModelGeometry("qwen3-0.6b", hidden=1024, layers=28, q_heads=16,
                          kv_heads=8, head_dim=128, intermediate=3072,
                          vocab=151936)


@dataclass(frozen=True)
class Shape:
    batch: int
    context: int
    model: ModelGeometry = QWEN3_06B

    def __post_init__(self) -> None:
        if self.batch < 1 or self.context < 1:
            raise ValueError(f"invalid shape b{self.batch}-s{self.context}")

    @property
    def id(self) -> str:
        return f"b{self.batch}-s{self.context}"

    @property
    def attended(self) -> int:
        return self.context + 1

    @property
    def case_id(self) -> str:
        return f"dense-step-qwen3-06b-b{self.batch}-s{self.context}"

    def to_dict(self) -> dict:
        return {"id": self.id, "batch": self.batch, "context": self.context,
                "model": self.model.name}


def parse_shapes(batches: list[int], contexts: list[int]) -> list[Shape]:
    return [Shape(b, c) for b in batches for c in contexts]


def megabench_case(shape: Shape) -> Case:
    """The MegaBench case for ``shape``; non-catalog contexts reuse the dense recipe."""
    base = select_cases("all", [BASE_CASE_ID])[0]
    if shape.case_id == base.id:
        return base
    m = shape.model
    params = dict(base.params, batch=shape.batch, context=shape.context,
                  layers=m.layers, hidden=m.hidden, q_heads=m.q_heads,
                  kv_heads=m.kv_heads, head_dim=m.head_dim,
                  intermediate=m.intermediate, vocab=m.vocab)
    return dataclasses.replace(base, id=shape.case_id, params=params)


def make_shape_inputs(shape: Shape, seed: int, device: str) -> dict:
    """MegaBench synthetic inputs (names and layouts of ``tasks/dense.py``)."""
    if shape.batch != 1:
        raise NotImplementedError("batched fixtures are a sweep TODO")
    from ..tasks.dense import make_inputs
    return make_inputs(megabench_case(shape), seed, device)


def reference_outputs(shape: Shape, inputs: dict) -> dict:
    """BF16 serving-precision MegaBench reference outputs."""
    if shape.batch != 1:
        raise NotImplementedError("batched references are a sweep TODO")
    from ..tasks.dense import reference
    return reference(megabench_case(shape), inputs)


def expected_outputs(shape: Shape) -> dict[str, tuple[tuple[int, ...], str]]:
    """Output names, shapes, and dtypes every in-process Step must produce.

    Batch one matches the MegaBench reference exactly; batch B adds a leading
    batch dimension to every output.
    """
    m = shape.model
    lead = () if shape.batch == 1 else (shape.batch,)
    return {
        "logits": (lead + (m.vocab,), "float32"),
        "next_token": (lead, "int64"),
        "k_write": (lead + (m.layers, m.kv_heads, m.head_dim), "bfloat16"),
        "v_write": (lead + (m.layers, m.kv_heads, m.head_dim), "bfloat16"),
    }


def accessed_bytes(shape: Shape, width: int = 2) -> int:
    """Forge Eq. 2 (reproductions/ForgeMegakernel/config.py): weights once, KV per sequence."""
    m = shape.model
    weights = m.layers * (m.hidden * m.q_dim + 2 * m.hidden * m.kv_dim
                          + m.q_dim * m.hidden + 3 * m.hidden * m.intermediate)
    weights += m.vocab * m.hidden
    kv = 2 * m.layers * m.kv_dim * shape.context * shape.batch
    return width * (weights + kv + shape.batch * m.hidden)


def bytes_floor(shape: Shape,
                bandwidth_tb_s: float = DEFAULT_B200_BANDWIDTH_TB_S) -> dict:
    """HBM speed-of-light floor; MegaBench's ``sol`` model for batch one."""
    if shape.batch == 1:
        est = estimate_case(megabench_case(shape), bandwidth_tb_s)
        return {"bytes": est["minimum_bytes"], "floor_ms": est["hbm_floor_ms"],
                "bandwidth_tb_s": bandwidth_tb_s, "model": "megabench.sol"}
    total = accessed_bytes(shape)
    return {"bytes": total, "floor_ms": total / (bandwidth_tb_s * 1e12) * 1e3,
            "bandwidth_tb_s": bandwidth_tb_s, "model": "forge_eq2"}
