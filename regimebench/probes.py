"""Once-per-job hardware probes feeding the "measured" SOL variant (GPU required)."""

from __future__ import annotations

from .harness import timing


def hbm_bw(nbytes: int = 2 << 30, reps: int = 20) -> dict:
    """Device copy bandwidth; ``bw_bytes_s`` = median of read+write traffic (B/s)."""
    r = timing.hbm_copy_bw(nbytes, reps)
    return dict(r, bw_bytes_s=r["median_gbps"] * 1e9)


def gemm_peak(n: int = 8192, reps: int = 20) -> dict:
    """cuBLAS bf16 n^3 ``torch.mm``, best of ``reps``; ``peak_flops`` in FLOP/s."""
    import torch
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        torch.mm(a, b)
    torch.cuda.synchronize()
    ms = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        torch.mm(a, b)
        e.record()
        torch.cuda.synchronize()
        ms.append(s.elapsed_time(e))
    del a, b
    best = min(ms)
    return {"n": n, "reps": reps, "best_ms": best, "median_ms": sorted(ms)[len(ms) // 2],
            "peak_flops": 2 * n ** 3 / (best * 1e-3), "peak_tflops": 2 * n ** 3 / (best * 1e-3) / 1e12}


def run_probes() -> dict:
    """All probes once: hbm, gemm, SM clock snapshot (NVML, may be empty)."""
    clock = timing.ClockLogger(0).snapshot()
    return {"hbm": hbm_bw(), "gemm": gemm_peak(), "clock": clock}
