"""Timing of the megakernel: isolated launches and a steady chain.

isolated: one launch on an idle GPU, between two synchronizations. Two clocks run:
  CUDA events around the launch, and the host wall clock around launch plus synchronize.
steady: back-to-back chained launches. The next_token output of one launch is the
  token input of the next. There is no host sync inside a chain. The value is the CUDA-event
  time of the whole chain divided by its length.

Every number is a median over repeated measurements.
"""

from __future__ import annotations

import statistics
import time

import torch

from .kernel import Megakernel


def isolated(mk: Megakernel, inputs: dict, iters: int = 200, warmup: int = 20, flush_l2: bool = False) -> dict:
    """Median microseconds of one isolated launch (event clock and wall clock)."""
    mk.bind(inputs)
    dev = mk.device
    flush = None
    if flush_l2:
        flush = torch.empty(2 * torch.cuda.get_device_properties(dev).L2_cache_size, dtype=torch.uint8, device=dev)
    for _ in range(warmup):
        mk.launch()
    torch.cuda.synchronize(dev)
    ev, wall = [], []
    for _ in range(iters):
        if flush is not None:
            flush.zero_()
        torch.cuda.synchronize(dev)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        mk.launch()
        end.record()
        torch.cuda.synchronize(dev)
        wall.append((time.perf_counter() - t0) * 1e6)
        ev.append(start.elapsed_time(end) * 1e3)
    return {"event_us_median": statistics.median(ev), "wall_us_median": statistics.median(wall),
            "event_us_min": min(ev), "event_us_max": max(ev), "iters": iters}


def steady(mk: Megakernel, inputs: dict, steps: int = 50, reps: int = 20, warmup: int = 200) -> dict:
    """Median microseconds per step over `reps` chains of `steps` chained launches."""
    mk.bind(inputs, chained=True)
    dev = mk.device
    for _ in range(warmup):
        mk.launch()
    torch.cuda.synchronize(dev)
    per_step = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(steps):
            mk.launch()
        end.record()
        torch.cuda.synchronize(dev)
        per_step.append(start.elapsed_time(end) * 1e3 / steps)
    return {"us_per_step_median": statistics.median(per_step), "us_per_step_min": min(per_step),
            "us_per_step_max": max(per_step), "steps": steps, "reps": reps}


def gpu_info(device) -> dict:
    """GPU name and the current SM clock in MHz (None when NVML is missing)."""
    dev = torch.device(device)
    info = {"name": torch.cuda.get_device_name(dev), "sm_count": torch.cuda.get_device_properties(dev).multi_processor_count,
            "sm_clock_mhz": None}
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(dev.index or 0)
        info["sm_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
    except Exception:
        try:
            import subprocess

            out = subprocess.run(["nvidia-smi", f"--id={dev.index or 0}", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, check=True).stdout.strip()
            info["sm_clock_mhz"] = int(out.splitlines()[0])
        except Exception:
            pass
    return info
