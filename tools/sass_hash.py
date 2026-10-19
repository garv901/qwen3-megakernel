#!/usr/bin/env python3
"""Compile a kernel source for sm_100 on the CPU and print its resources and a SASS hash.

NVRTC compiles the source to a cubin. `cuobjdump` prints the SASS and the
resource usage. The hash covers the SASS text without addresses and encodings.
Two sources with the same hash give the same machine code.

Usage: sass_hash.py [SOURCE.cu]   (default: csrc/qwen3_step.cu)
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CUOBJDUMP = os.environ.get("CUOBJDUMP", "/usr/local/cuda/bin/cuobjdump")

sys.path.insert(0, str(ROOT))
from qwen3mk.kernel import compile_cubin  # noqa: E402  (the same compile path as the runtime)

ARCH = "sm_100"
N_BLOCKS = 148


def _cuobjdump(flag: str, cubin: bytes) -> str:
    with tempfile.TemporaryDirectory(prefix="qmk_") as d:
        p = Path(d) / "k.cubin"
        p.write_bytes(cubin)
        return subprocess.run([CUOBJDUMP, flag, str(p)], capture_output=True, text=True, check=True).stdout


def normalize_sass(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = re.sub(r"/\*[^*]*\*/", "", line).strip()
        if line and not line.startswith(("code for", ".target", ".headerflags")):
            lines.append(re.sub(r"\s+", " ", line))
    return "\n".join(lines)


def sha(text: str | bytes) -> str:
    b = text.encode() if isinstance(text, str) else text
    return hashlib.sha256(b).hexdigest()


def res_usage(cubin: bytes) -> dict:
    txt = _cuobjdump("-res-usage", cubin)
    m = re.search(r"Function \w+:\s*\n\s*(.*)", txt)
    out = {}
    if m:
        for k, v in re.findall(r"(REG|STACK|SHARED|LOCAL):(\d+)", m.group(1)):
            out[k.lower()] = int(v)
    return out


def analyze(src: str, arch: str = ARCH, n_blocks: int = N_BLOCKS) -> dict:
    """Compile `src` with the kernel options and return regs, spill bytes, static smem and the normalized SASS sha."""
    cubin = compile_cubin(src, arch, n_blocks)
    usage = res_usage(cubin)
    return {"regs": usage.get("reg"), "spill": max(usage.get("stack", 0), usage.get("local", 0)),
            "static_smem": usage.get("shared"), "sass_sha": sha(normalize_sass(_cuobjdump("-sass", cubin)))}


def main(argv: list[str]) -> None:
    path = Path(argv[1]) if len(argv) > 1 else ROOT / "csrc" / "qwen3_step.cu"
    r = analyze(path.read_text())
    print(f"source      {path}")
    print(f"regs        {r['regs']}")
    print(f"spill bytes {r['spill']}")
    print(f"static smem {r['static_smem']}")
    print(f"sass sha    {r['sass_sha']}")


if __name__ == "__main__":
    main(sys.argv)
