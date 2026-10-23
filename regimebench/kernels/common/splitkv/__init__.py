"""A1: split-KV decode attention for the megakernel (splitkv.cuh = device module, plan.py = host planner)."""
from pathlib import Path

HEADER_PATH = Path(__file__).with_name("splitkv.cuh")


def header_source() -> str:
    return HEADER_PATH.read_text()
