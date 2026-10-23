"""One direct mk launch of one workload; exit 0 if it completes. Usage: v0_one.py WORKLOAD [NULL_MASK [DBG_EXIT]]."""
import sys

import torch

from regimebench.arms.base import load_arm
from regimebench.fixture import make_inputs
from regimebench.workload import select

w = select([sys.argv[1]])[0]
mask = int(sys.argv[2], 0) if len(sys.argv) > 2 else 0
inputs = make_inputs(w, 7401, "cuda")
dbg = int(sys.argv[3]) if len(sys.argv) > 3 else 0
arm = load_arm("mk", null=mask)
arm.knobs["MK_DBG_EXIT"] = dbg
step = arm.prepare(w, inputs, "auto-direct", "cuda")
step.launch()
torch.cuda.synchronize()
print(f"OK {w.id} null=0x{mask:x} dbg_exit={dbg} sync={step.sync[:8].tolist()}", flush=True)
