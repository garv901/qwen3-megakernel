"""Interleaved A/B of mk knob sets on one GPU. Usage: ab.py WORKLOADS 'JSON list of knob dicts' [rounds=7].
Per workload: every variant graded vs the FP32 band (worst ratio), then `rounds` rounds of (each variant: 20 graph launches),
order rotated each round; prints median us per variant and the delta to variant 0 with the min..max of the per-round deltas."""
import json
import statistics
import sys

import torch

from regimebench.arms.base import load_arm
from regimebench.fixture import make_inputs
from regimebench.harness import accuracy
from regimebench.ref.reference import forward
from regimebench.workload import select

sets = json.loads(sys.argv[2])
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 7
arms = []
for ks in sets:
    ks = dict(ks)
    a = load_arm("mk", attn=ks.pop("attn", "auto"), skvtc=ks.pop("skvtc", 1))   # keys "attn", "skvtc": MkArm options, the rest are knobs
    a.knobs.update({k: int(v) for k, v in ks.items()})
    arms.append(a)


def batch(step, n=20):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        step.launch()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / n * 1e3


for w in select(sys.argv[1].split(",")):
    inputs = make_inputs(w, 7401, "cuda")
    ref16, ref32 = forward(w, inputs, torch.bfloat16), forward(w, inputs, torch.float32)
    band = accuracy.oracle_band(ref16, ref32)
    steps = [a.prepare(w, inputs, "auto", "cuda") for a in arms]
    grades = []
    for s in steps:
        s.launch(); torch.cuda.synchronize()
        grades.append(accuracy.grade_vs_oracle(accuracy.compare(s.outputs(), ref32), band)["worst_ratio"])
        for _ in range(3):
            s.launch()
    torch.cuda.synchronize()
    t = [[] for _ in steps]
    for r in range(rounds):
        order = list(range(len(steps)))
        order = order[r % len(order):] + order[:r % len(order)]
        for i in order:
            t[i].append(batch(steps[i]))
    med = [statistics.median(x) for x in t]
    line = [f"{w.id}:"]
    for i, ks in enumerate(sets):
        d = [t[i][r] - t[0][r] for r in range(rounds)]
        line.append(f"[{i} {ks}] {med[i]:.1f} us (ratio {grades[i]:.2f})" + ("" if i == 0 else f" d {statistics.median(d):+.1f} [{min(d):+.1f},{max(d):+.1f}]"))
    print(" | ".join(line), flush=True)
    for s in steps:
        s.close()
    del steps, inputs
    torch.cuda.empty_cache()
