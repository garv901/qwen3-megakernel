"""Per-CTA mk stamps for the small-T study. Usage: dump.py OUTDIR WORKLOADS 'JSON list of {"name", "null", knobs...}' [launches=5].
Saves OUTDIR/<workload>__<name>.npz: st [launches][G][NSTAMP] (globaltimer ns), knobs. Attention is the placeholder for every set
(MK_TS 16 / 32 + splitkv hangs: static_assert in mk_step.cu), so classic and tile stream see the same attention phase."""
import json
import os
import sys

import numpy as np
import torch

from regimebench.arms.base import load_arm
from regimebench.fixture import make_inputs
from regimebench.workload import select

out, wl, sets = sys.argv[1], sys.argv[2].split(","), json.loads(sys.argv[3])
nl = int(sys.argv[4]) if len(sys.argv) > 4 else 5
os.makedirs(out, exist_ok=True)
for s in sets:
    s = dict(s)
    name, mask = s.pop("name"), int(s.pop("null", 0))
    arm = load_arm("mk", stamps=1, null=mask, attn=s.pop("attn", "placeholder"))
    arm.knobs.update({k: int(v) for k, v in s.items()})
    for w in select(wl):
        inputs = make_inputs(w, 7401, "cuda")
        step = arm.prepare(w, inputs, "auto-direct", "cuda")
        for _ in range(3):
            step.launch()
        sts = []
        for _ in range(nl):
            step.launch(); torch.cuda.synchronize()
            sts.append(step.stamps().astype(np.int64))
        np.savez(f"{out}/{w.id}__{name}.npz", st=np.stack(sts), knobs=json.dumps(dict(arm.knobs)), null=mask)
        print(f"saved {w.id} {name} null=0x{mask:x} shape {sts[0].shape}", flush=True)
        step.close(); del step, inputs
        torch.cuda.empty_cache()
