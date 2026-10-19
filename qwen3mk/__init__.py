"""Qwen3-0.6B batch-1 decode step as one persistent CUDA megakernel."""

from .kernel import INPUT_ORDER, Megakernel

__all__ = ["INPUT_ORDER", "Megakernel"]
