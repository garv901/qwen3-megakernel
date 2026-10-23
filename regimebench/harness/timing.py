"""GPU timing, clock logging, HBM bandwidth probe, and kernel timelines.

Pure statistics (``summarize``, ``ratio_ci``, ``gap_stats``) need no GPU; torch
is imported inside the functions that touch CUDA.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import tempfile
import time
from pathlib import Path
from typing import Callable, NamedTuple

# --------------------------------------------------------------------------
# Statistics


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 100]) of an ascending list."""
    if not sorted_values:
        return float("nan")
    at = (len(sorted_values) - 1) * q / 100
    lo = int(at)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (at - lo)


def _boot_median(x: list[float], rng: random.Random) -> float:
    n = len(x)
    return statistics.median(x[rng.randrange(n)] for _ in range(n))


def summarize(samples: list[float], seed: int = 0) -> dict:
    """Location/spread statistics plus a 2000-resample bootstrap CI of the median."""
    x = sorted(float(v) for v in samples)
    if not x:
        return {"n": 0}
    med = statistics.median(x)
    rng = random.Random(seed)
    boots = sorted(_boot_median(x, rng) for _ in range(2000))
    return {"n": len(x), "median": med, "mean": statistics.fmean(x),
            "std": statistics.stdev(x) if len(x) > 1 else 0.0,
            "p05": percentile(x, 5), "p95": percentile(x, 95),
            "min": x[0], "max": x[-1],
            "mad": statistics.median(abs(v - med) for v in x),
            "ci95_median": [percentile(boots, 2.5), percentile(boots, 97.5)]}


def ratio_ci(a: list[float], b: list[float], seed: int = 0) -> dict:
    """median(a) / median(b) with a bootstrap 95% CI (independent resampling)."""
    if not a or not b:
        return {"ratio": None, "ci95": None}
    rng = random.Random(seed)
    boots = sorted(_boot_median(list(a), rng) / _boot_median(list(b), rng)
                   for _ in range(2000))
    return {"ratio": statistics.median(a) / statistics.median(b),
            "ci95": [percentile(boots, 2.5), percentile(boots, 97.5)]}


def diff_ci(a: list[float], b: list[float], seed: int = 0) -> dict:
    """median(a) - median(b) with a bootstrap 95% CI (independent resampling)."""
    if not a or not b:
        return {"diff": None, "ci95": None}
    rng = random.Random(seed)
    boots = sorted(_boot_median(list(a), rng) - _boot_median(list(b), rng)
                   for _ in range(2000))
    return {"diff": statistics.median(a) - statistics.median(b),
            "ci95": [percentile(boots, 2.5), percentile(boots, 97.5)]}


# Two-sided 95% Student-t quantiles by degrees of freedom (1.96 beyond the table).
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}


def t_interval(values: list[float]) -> dict:
    """Mean with a Student-t 95% CI, for a few independent replicates (e.g. rounds).

    Use it when samples come in correlated groups: one estimate per group, then
    this interval over the groups. A bootstrap over the pooled samples ignores the
    group effect and is too narrow.
    """
    x = [float(v) for v in values]
    if len(x) < 2:
        return {"mean": x[0] if x else None, "ci95": None, "n": len(x)}
    m, se = statistics.fmean(x), statistics.stdev(x) / math.sqrt(len(x))
    df = len(x) - 1
    t = _T975[max(k for k in _T975 if k <= df)] if df <= 30 else 1.96
    return {"mean": m, "ci95": [m - t * se, m + t * se], "n": len(x)}


def linear_fit(xs: list[float], ys: list[float]) -> dict:
    """Ordinary least squares y = intercept + slope * x, with residuals."""
    n = len(xs)
    if n < 2 or len(set(xs)) < 2:
        return {"slope": None, "intercept": None, "residuals": []}
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    return {"slope": slope, "intercept": intercept,
            "residuals": [y - (intercept + slope * x) for x, y in zip(xs, ys)]}


def slope_ci(groups: dict[float, list[float]], seed: int = 0) -> dict:
    """Least-squares slope of the per-x medians with a bootstrap 95% CI.

    ``groups`` maps x (for example max_seq_length) to its samples; each
    bootstrap draw resamples every group independently and refits.
    """
    xs = sorted(groups)
    if len(xs) < 2:
        return {"slope": None, "intercept": None, "ci95": None, "residuals": {}}
    fit = linear_fit(xs, [statistics.median(groups[x]) for x in xs])
    rng = random.Random(seed)
    boots = sorted(linear_fit(xs, [_boot_median(list(groups[x]), rng) for x in xs])["slope"]
                   for _ in range(2000))
    return {"slope": fit["slope"], "intercept": fit["intercept"],
            "ci95": [percentile(boots, 2.5), percentile(boots, 97.5)],
            "residuals": dict(zip(xs, fit["residuals"]))}


# --------------------------------------------------------------------------
# Single measurements (step.launch() enqueues one decode step, no host sync)


def steady(step, n_steps: int) -> float:
    """Mean ms per step over ``n_steps`` back-to-back launches (caller sets chained)."""
    import torch
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_steps):
        step.launch()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_steps


def isolated(step, flush=None) -> float:
    """ms for one launch on an idle GPU, optionally after an L2 flush."""
    import torch
    torch.cuda.synchronize()
    if flush is not None:
        flush.zero_()
        torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    step.launch()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def l2_flush_buffer(device):
    import torch
    size = torch.cuda.get_device_properties(device).L2_cache_size
    return torch.empty(2 * size, dtype=torch.uint8, device=device)


def warmup(steps: list, seconds: float = 2.0) -> None:
    import torch
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for step in steps:
            for _ in range(10):
                step.launch()
        torch.cuda.synchronize()
    torch.cuda.synchronize()


# --------------------------------------------------------------------------
# Clock / thermal logging

THROTTLE_BITS = {0x1: "gpu_idle", 0x2: "applications_clocks_setting",
                 0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x10: "sync_boost",
                 0x20: "sw_thermal_slowdown", 0x40: "hw_thermal_slowdown",
                 0x80: "hw_power_brake_slowdown", 0x100: "display_clock_setting"}
BENIGN_THROTTLE = 0x1 | 0x2


def decode_throttle(mask: int) -> list[str]:
    return [name for bit, name in THROTTLE_BITS.items() if mask & bit]


class ClockLogger:
    """NVML snapshots of clocks, temperature, power and throttle reasons.

    ``snapshot()`` returns ``{}`` when pynvml is missing or anything fails.
    """

    def __init__(self, device: int | str = 0):
        self.handle = None
        try:
            import pynvml
            import torch
            pynvml.nvmlInit()
            self.nvml = pynvml
            idx = torch.device(device).index or 0
            props = torch.cuda.get_device_properties(idx)
            self.handle = self._handle(pynvml, props, idx)
        except Exception:
            self.handle = None

    @staticmethod
    def _handle(nvml, props, idx):
        uuid = str(getattr(props, "uuid", ""))
        for cand in (uuid, uuid if uuid.startswith("GPU-") else "GPU-" + uuid):
            try:
                return nvml.nvmlDeviceGetHandleByUUID(cand)
            except Exception:
                pass
        try:
            bus = f"{props.pci_domain_id:08X}:{props.pci_bus_id:02X}:{props.pci_device_id:02X}.0"
            return nvml.nvmlDeviceGetHandleByPciBusId(bus)
        except Exception:
            return None

    def snapshot(self) -> dict:
        if self.handle is None:
            return {}
        n, h = self.nvml, self.handle
        try:
            get = getattr(n, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
                n.nvmlDeviceGetCurrentClocksThrottleReasons
            mask = int(get(h))
            return {"sm_mhz": n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM),
                    "mem_mhz": n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM),
                    "temp_c": n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU),
                    "power_w": n.nvmlDeviceGetPowerUsage(h) / 1000,
                    "throttle_mask": mask, "throttle": decode_throttle(mask),
                    "throttled": bool(mask & ~BENIGN_THROTTLE)}
        except Exception:
            return {}

    @staticmethod
    def summarize(entries: list[dict]) -> dict:
        """Min/median/max of clocks and the sorted list of throttled (arm, round) blocks."""
        snaps = [e["clock"] for e in entries if e.get("clock")]
        if not snaps:
            return {}
        out: dict = {"n": len(snaps)}
        for key in ("sm_mhz", "mem_mhz", "temp_c", "power_w"):
            v = [s[key] for s in snaps if key in s]
            if v:
                out[key] = {"min": min(v), "median": statistics.median(v), "max": max(v)}
        out["throttled_rounds"] = sorted({e["round"] for e in entries
                                          if e.get("clock", {}).get("throttled")})
        out["throttle_reasons"] = sorted({r for s in snaps for r in s.get("throttle", [])})
        return out


# --------------------------------------------------------------------------
# Interleaved measurement


class InterleavedResult(NamedTuple):
    samples: dict[str, list[float]]
    round_idx: dict[str, list[int]]       # round of each sample
    clocks: list[dict]                    # {"arm", "round", "clock"} per block


def interleaved(steps: dict, measure: Callable[[object], float], rounds: int,
                per_block: int, seed: int, clock: ClockLogger | None = None
                ) -> InterleavedResult:
    """Round-robin the arms in a fresh random order each round, ``per_block`` samples
    per arm per round, so clock and thermal drift hits every arm alike."""
    samples = {k: [] for k in steps}
    round_idx = {k: [] for k in steps}
    clocks: list[dict] = []
    for r in range(rounds):
        order = list(steps)
        random.Random(seed + r).shuffle(order)
        for name in order:
            for _ in range(per_block):
                samples[name].append(measure(steps[name]))
                round_idx[name].append(r)
            if clock is not None:
                clocks.append({"arm": name, "round": r, "clock": clock.snapshot()})
    return InterleavedResult(samples, round_idx, clocks)


# --------------------------------------------------------------------------
# HBM bandwidth


def hbm_copy_bw(nbytes: int = 2 << 30, reps: int = 20) -> dict:
    """Device-to-device copy bandwidth in GB/s (read + write = 2 * nbytes per copy)."""
    import torch
    src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    src.fill_(1)
    dst.copy_(src)
    torch.cuda.synchronize()
    gbps = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        dst.copy_(src)
        b.record()
        torch.cuda.synchronize()
        gbps.append(2 * nbytes / (a.elapsed_time(b) * 1e-3) / 1e9)
    del src, dst
    return {"median_gbps": statistics.median(gbps), "max_gbps": max(gbps),
            "nbytes": nbytes, "reps": reps}


# --------------------------------------------------------------------------
# Kernel timeline


def gap_stats(intervals: list[tuple[float, float]], n_steps: int,
              names: list[str] | None = None) -> dict:
    """Launch-gap statistics of kernel ``(start_us, end_us)`` intervals over ``n_steps`` steps.

    ``window`` = first start to last end; ``busy`` = union of intervals;
    ``gap = window - busy``. Inter-kernel gaps are ``start_i - max(end so far)`` in start
    order; negative values are overlaps (concurrent kernels, e.g. PDL) and are counted
    separately. ``names`` (parallel to ``intervals``) enables the top-10 kernel table.
    """
    if not intervals:
        return {"kernels_per_step": 0}
    order = sorted(range(len(intervals)), key=lambda i: intervals[i][0])
    iv = [intervals[i] for i in order]
    window = max(e for _, e in iv) - iv[0][0]
    busy, cur_s, cur_e = 0.0, iv[0][0], iv[0][1]
    gaps: list[float] = []
    max_end = iv[0][1]
    for s, e in iv[1:]:
        gaps.append(s - max_end)
        if s > cur_e:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
        max_end = max(max_end, e)
    busy += cur_e - cur_s
    pos = sorted(g for g in gaps if g >= 0)
    over = [-g for g in gaps if g < 0]
    out = {"kernels_per_step": len(iv) / n_steps,
           "sum_kernel_us_per_step": sum(e - s for s, e in iv) / n_steps,
           "window_us_per_step": window / n_steps,
           "busy_us_per_step": busy / n_steps,
           "gap_us_per_step": (window - busy) / n_steps,
           "gap_mean_us": statistics.fmean(pos) if pos else 0.0,
           "gap_p50_us": percentile(pos, 50) if pos else 0.0,
           "gap_p95_us": percentile(pos, 95) if pos else 0.0,
           "overlap_count": len(over),
           "overlap_mean_us": statistics.fmean(over) if over else 0.0}
    if names is not None:
        agg: dict[str, list[float]] = {}
        for (s, e), name in zip(intervals, names):
            c = agg.setdefault(name, [0, 0.0])
            c[0] += 1
            c[1] += e - s
        top = sorted(agg.items(), key=lambda kv: -kv[1][1])[:10]
        out["top_kernels"] = [{"name": k, "count": c, "total_us": t} for k, (c, t) in top]
    return out


def kernel_timeline(step, n_steps: int = 10, trace_dir: Path | None = None) -> dict:
    """Profile ``n_steps`` chained launches and summarize the kernel timeline.

    Caveat: the profiler (CUPTI plus CPU-side hooks) inflates host launch cost, so gaps
    for eager launches are an upper bound; CUDA-graph replays and the megakernel are
    much less affected.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile
    step.set_chained(True)
    step.launch()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n_steps):
            step.launch()
        torch.cuda.synchronize()
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / f"{step.arm}-{step.variant}.json"
    else:
        path = Path(tempfile.mkdtemp(prefix="sota-trace-")) / "trace.json"
    prof.export_chrome_trace(str(path))
    with open(path) as f:
        events = json.load(f).get("traceEvents", [])
    kernels = [e for e in events if e.get("cat") == "kernel" and "dur" in e]
    stats = gap_stats([(e["ts"], e["ts"] + e["dur"]) for e in kernels], n_steps,
                      [e.get("name", "?") for e in kernels])
    runtime = [e.get("name", "") for e in events if e.get("cat") == "cuda_runtime"]
    stats["graph_launches"] = sum(n.startswith("cudaGraphLaunch") for n in runtime)
    stats["kernel_launch_calls"] = sum(n.startswith(("cudaLaunchKernel",
                                                     "cudaLaunchCooperativeKernel"))
                                       for n in runtime)
    stats["n_steps"] = n_steps
    if trace_dir is None:
        path.unlink(missing_ok=True)
    return stats


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
