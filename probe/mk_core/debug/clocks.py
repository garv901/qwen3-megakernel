"""SM clock / power / throttle reasons while mk replays a workload (NVML sampled from a thread). Usage: clocks.py WORKLOADS 'JSON list of knob dicts' [seconds=6]."""
import json
import statistics
import sys
import threading
import time

import pynvml
import torch

from regimebench.arms.base import load_arm
from regimebench.fixture import make_inputs
from regimebench.workload import select

sets = json.loads(sys.argv[2])
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
pynvml.nvmlInit()
dev = torch.cuda.current_device()
h = pynvml.nvmlDeviceGetHandleByUUID("GPU-" + str(torch.cuda.get_device_properties(dev).uuid))   # the job's GPU (CUDA_VISIBLE_DEVICES remaps indices)


def sample(stop, out):
    while not stop.is_set():
        out.append((pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM), pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
                    pynvml.nvmlDeviceGetCurrentClocksEventReasons(h), pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)))
        time.sleep(0.02)


for w in select(sys.argv[1].split(",")):
    inputs = make_inputs(w, 7401, "cuda")
    for ks in sets:
        ks = dict(ks)
        a = load_arm("mk", attn=ks.pop("attn", "auto"))
        a.knobs.update({k: int(v) for k, v in ks.items()})
        step = a.prepare(w, inputs, "auto-direct", "cuda")
        for _ in range(20):
            step.launch()
        torch.cuda.synchronize()
        out, stop = [], threading.Event()
        t = threading.Thread(target=sample, args=(stop, out)); t.start()
        n, t0 = 0, time.time()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        while time.time() - t0 < secs:
            for _ in range(50):
                step.launch()
            n += 50
            torch.cuda.synchronize()
        e1.record(); torch.cuda.synchronize()
        stop.set(); t.join()
        us = e0.elapsed_time(e1) * 1000 / n
        out = out[len(out) // 4:]   # drop the ramp
        reasons = {}
        for o in out:
            reasons[o[2]] = reasons.get(o[2], 0) + 1
        print(f"{w.id} {json.dumps(ks)} attn={step.attn if hasattr(step, 'attn') else '?'}: {us:.1f} us/step, SM MHz med {statistics.median(o[0] for o in out)} "
              f"min {min(o[0] for o in out)}, power med {statistics.median(o[1] for o in out):.0f} W, temp {statistics.median(o[3] for o in out)}, "
              f"reasons {{{', '.join(f'0x{k:x}: {v}' for k, v in sorted(reasons.items()))}}}", flush=True)
        step.close()
        del step
