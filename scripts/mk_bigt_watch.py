"""MK_BIGT hang triage without touching the kernel: launch on its own stream, then read the grid-barrier counter (sync[0]) and the
finalizer tickets (tk) through a second stream while the kernel spins (the copy engine still runs). PYTHONPATH=. python scripts/mk_bigt_watch.py WID"""
import json, os, sys, time
import torch
from regimebench.arms.base import load_arm
from regimebench.fixture import make_inputs
from regimebench.workload import select

arm = load_arm("mk", bigt=1)
arm.knobs.update(json.loads(os.environ.get("MK_KNOBS", "{}")))
w = select([sys.argv[1]])[0]
inputs = make_inputs(w, 7401, "cuda")
step = arm.prepare(w, inputs, "auto-direct", "cuda")
torch.cuda.synchronize()
nb, ntok = step.nb, w.total_q
ntt = (ntok + 127) // 128
print(f"=== {w.id} ntok {ntok} NTT {ntt} grid {nb}; barriers: 1 prologue + 6 per layer x 28", flush=True)
s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
hs = torch.empty(8, dtype=torch.int32).pin_memory()
ht = torch.empty(step.tk.numel(), dtype=torch.int32).pin_memory()
done = torch.cuda.Event()
with torch.cuda.stream(s1):
    step.kern.kernel.launch(step.nb, step.args, s1.cuda_stream)
    done.record(s1)
for it in range(2):
    time.sleep(3)
    if done.query():
        print(f"t={3 * (it + 1)}s: kernel finished (no hang this time)", flush=True)
        break
    with torch.cuda.stream(s2):
        hs.copy_(step.sync[:8], non_blocking=True)
        ht.copy_(step.tk, non_blocking=True)
    s2.synchronize()
    c = int(hs[0])
    b, part = divmod(c, nb)
    lay, ph = divmod(b - 1, 6) if b >= 1 else (-1, -1)
    nz = {i: int(v) for i, v in enumerate(ht.tolist()) if v}
    print(f"t={3 * (it + 1)}s: bar_cnt {c} = {b} full barriers + {part} arrivals (layer {lay}, barrier {ph} of the layer) ticket {int(hs[2])}; "
          f"nonzero tk ({len(nz)}): {dict(list(nz.items())[:40])}", flush=True)
    print(f"   tk regions: O [0,{8*ntt}) D [{8*ntt},{16*ntt}) seenO [{16*ntt},{24*ntt}) seenD [{24*ntt},{32*ntt})", flush=True)
sys.stdout.flush()
os._exit(0)   # a hung cooperative kernel cannot be torn down cleanly
