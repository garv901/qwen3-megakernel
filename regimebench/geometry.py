"""Qwen3-0.6B geometry (same numbers as vibe-megakernel megabench/sota/shapes.py)."""

from __future__ import annotations

from dataclasses import dataclass


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

    @property
    def kv_bytes_per_token(self) -> int:
        """BF16 K+V bytes for one cached token over all layers (114,688 for Qwen3-0.6B)."""
        return 2 * self.layers * self.kv_dim * 2


QWEN3_06B = ModelGeometry("qwen3-0.6b", hidden=1024, layers=28, q_heads=16,
                          kv_heads=8, head_dim=128, intermediate=3072,
                          vocab=151936)
