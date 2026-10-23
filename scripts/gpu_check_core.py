"""GPU check for the core pieces. Prints a PASS/FAIL table, exits non-zero on failure.

  1. SOL table for the whole grid (CPU-only part) and the R0 floor identity.
  2. Reference BF16 vs FP32 on R0 and the first check=True point of every regime: finite, shapes,
     rel-L2 of BF16 to FP32 (the oracle band) below 0.25, next_token agreement recorded.
  3. Probes: HBM copy BW and cuBLAS bf16 peak are positive and plausible (>2 TB/s, >0.5 PF).

Run: PYTHONPATH=/raid/garv901/regime-megakernel $PY scripts/gpu_check_core.py
"""

from __future__ import annotations

import sys
import time

import torch

from regimebench import probes
from regimebench.arms.base import expected_outputs
from regimebench.fixture import make_inputs
from regimebench.harness import accuracy as acc
from regimebench.ref.reference import forward
from regimebench.sol import sol
from regimebench.workload import regime_grid

BAND_MAX = 0.25


def main() -> int:
    rows, ok = [], True
    grid = regime_grid()
    print(f"{'point':22s} {'bytes':>14s} {'T_bw us':>9s} {'T_fl us':>9s} {'SOL us':>9s} bound   ridge")
    for w in grid:
        s = sol(w)
        print(f"{w.id:22s} {s['bytes']:14,d} {s['t_bw_us']:9.1f} {s['t_fl_us']:9.1f} "
              f"{s['t_sol_us']:9.1f} {s['bound']:7s} {s['ridge_tokens']:.0f}")
    r0 = sol(grid[0])
    rows.append(("R0 SOL bytes == 1207502352", r0["bytes"] == 1_207_502_352))

    pts, seen = [], set()
    for w in grid:
        if (w.regime == "R0" or w.check) and w.regime not in seen:
            seen.add(w.regime)
            pts.append(w)
    for w in pts:
        t0 = time.time()
        inp = make_inputs(w, 7301, "cuda")
        o16, o32 = forward(w, inp, torch.bfloat16), forward(w, inp, torch.float32)
        torch.cuda.synchronize()
        exp = expected_outputs(w)
        shapes = all(tuple(o16[k].shape) == exp[k][0] for k in exp)
        finite = all(torch.isfinite(o16[k].float()).all().item() for k in ("logits", "k_new", "v_new"))
        band = acc.oracle_band(o16, o32)
        m = acc.compare(o16, o32)
        good = shapes and finite and max(band.values()) < BAND_MAX
        rows.append((f"ref {w.id}: band logits {band['logits']:.3e} k {band['k_new']:.3e} "
                     f"v {band['v_new']:.3e}, next_token match {m['next_token']['match_rate']:.2f}, "
                     f"{time.time() - t0:.1f}s", good))
        del inp, o16, o32
        torch.cuda.empty_cache()

    bw, pk = probes.hbm_bw(), probes.gemm_peak()
    rows.append((f"probe HBM copy {bw['median_gbps']:.0f} GB/s (max {bw['max_gbps']:.0f})",
                 bw["median_gbps"] > 2000))
    rows.append((f"probe cuBLAS bf16 {pk['peak_tflops']:.0f} TFLOP/s", pk["peak_tflops"] > 500))

    print()
    for name, good in rows:
        print(f"{'PASS' if good else 'FAIL'}  {name}")
        ok &= good
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
