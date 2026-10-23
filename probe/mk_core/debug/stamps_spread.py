"""Barrier latency vs arrival spread inside mk, from MK_STAMPS. Usage: stamps_spread.py WORKLOADS [NULL_MASK [JSON knobs]].
Per phase (layers 4..23, median over layers): spread = last arrive - median arrive, lat = median exit - last arrive,
and whether the last arriver is a CTA with the max unit count of that phase (unit imbalance)."""
import json
import sys

import numpy as np
import torch

from regimebench.arms.base import load_arm
from regimebench.fixture import make_inputs
from regimebench.workload import select

PH = ("QKV", "RoPE", "attn", "O", "GU", "D")
UNITS = {0: 256, 3: 128, 4: 384, 5: 192}   # KU 128 split-K units per GEMM phase
mask = int(sys.argv[2], 0) if len(sys.argv) > 2 else 0
arm = load_arm("mk", stamps=1, null=mask)
if len(sys.argv) > 3:
    arm.knobs.update({k: int(v) for k, v in json.loads(sys.argv[3]).items()})
for w in select(sys.argv[1].split(",")):
    inputs = make_inputs(w, 7401, "cuda")
    step = arm.prepare(w, inputs, "auto-direct", "cuda")
    for _ in range(3):
        step.launch()
    torch.cuda.synchronize()
    st = step.stamps().astype(np.int64)
    G = st.shape[0]
    nbar = 1 + 6 * 28
    arr, wex = st[:, 0:2 * nbar:2], st[:, 1:2 * nbar:2]
    print(f"{w.id} null=0x{mask:x} (G {G}): phase | work med | spread (last-med arrive) | lat (med exit - last arrive) | last arriver has max units")
    for p in range(6):
        sp, lat, wk, lastmax = [], [], [], []
        for layer in range(4, 24):
            b = 1 + 6 * layer + p
            a = arr[:, b]
            sp.append(a.max() - np.median(a)); lat.append(np.median(wex[:, b]) - a.max())
            wk.append(np.median(a - wex[:, b - 1]))
            if p in UNITS:
                U = UNITS[p]
                n = np.array([(c + 1) * U // G - c * U // G for c in range(G)])
                lastmax.append(n[int(a.argmax())] == n.max())
        lm = f"{np.mean(lastmax):.2f}" if lastmax else "-"
        print(f"  {PH[p]:5s} {np.median(wk) / 1e3:6.2f} | {np.median(sp) / 1e3:6.2f} | {np.median(lat) / 1e3:6.2f} | {lm}")
    step.close(); del step, inputs
    torch.cuda.empty_cache()
