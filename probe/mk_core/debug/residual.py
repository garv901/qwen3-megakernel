"""Where the per-step residual (step - 28 x median layer) goes, from MK_STAMPS (entry / exit stamps at MK_NSTAMP - 2 / - 1).
Usage: residual.py WORKLOADS [JSON knobs]. Graph variant; step = 20 graph launches / 20 (event time); stamps of the last launch.
Prints (us): step, span (last CTA exit - first CTA entry), launch gap (step - span), entry spread, prologue (entry -> barrier 0
arrive, median / max), barrier 0 exit (from first entry), layer 0 vs median layer (4..23), LM (last layer barrier exit -> LM end,
median / max), argmax tail (last exit - last LM end), and residual = step - 28 x median layer."""
import json
import sys

import numpy as np
import torch

from regimebench.arms.base import load_arm
from regimebench.fixture import make_inputs
from regimebench.workload import select

arm = load_arm("mk", stamps=1)
if len(sys.argv) > 2:
    kj = json.loads(sys.argv[2])
    arm.attn = kj.pop("attn", arm.attn)
    arm.knobs.update({k: int(v) for k, v in kj.items()})


def batch(step, n=20):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        step.launch()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / n * 1e3


print("point | NPAD | step | span | launch gap | entry spread | prologue med/max | bar0 exit | layer0 / med layer | LM med/max | argmax tail | residual")
for w in select(sys.argv[1].split(",")):
    inputs = make_inputs(w, 7401, "cuda")
    step = arm.prepare(w, inputs, "auto", "cuda")
    for _ in range(3):
        step.launch()
    torch.cuda.synchronize()
    ts = sorted(batch(step) for _ in range(5))
    t_step = ts[2]
    st = step.stamps().astype(np.int64)
    G, NS = st.shape
    cmb = int(step.kern.knobs.get("MK_CMB", -1))
    cmb = (cmb if cmb >= 0 else int(step.npad >= 64)) and getattr(step, "plan", None) is not None and step.plan.s_max > 1
    NP = 7 if cmb else 6
    nbar = 1 + NP * 28
    arr, wex = st[:, 0:2 * nbar:2], st[:, 1:2 * nbar:2]
    lm_end = st[:, 2 * nbar]
    ent, ext = st[:, NS - 2], st[:, NS - 1]
    t0 = ent.min()
    us = lambda x: x / 1e3
    mex = np.median(wex, axis=0)   # median exit per barrier
    lay = [mex[NP * (l + 1)] - mex[NP * l] for l in range(28)]
    med = float(np.median(lay[4:24]))
    span = ext.max() - t0
    pro = arr[:, 0] - ent
    lm = lm_end - wex[:, nbar - 1]
    print(f"{w.id} | {step.npad} | {t_step:.1f} | {us(span):.1f} | {t_step - us(span):.1f} | {us(ent.max() - t0):.1f} | "
          f"{us(np.median(pro)):.1f} / {us(pro.max()):.1f} | {us(wex[:, 0].max() - t0):.1f} | {us(lay[0]):.1f} / {us(med):.1f} | "
          f"{us(np.median(lm)):.1f} / {us(lm.max()):.1f} | {us(ext.max() - lm_end.max()):.1f} | {t_step - 28 * us(med):.1f}", flush=True)
    step.close(); del step, inputs
    torch.cuda.empty_cache()
