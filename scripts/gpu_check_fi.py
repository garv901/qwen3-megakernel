"""GPU check for the fi-graph arm: every check=True workload x variant vs the reference.

    PYTHONPATH=. python scripts/gpu_check_fi.py [--workloads R0,R7-p512-d8,...] [--variants v1,v2]
        [--seed 7301] [--json OUT.json]

Prints a PASS/FAIL table (accuracy grade vs the FP32 oracle band of ``harness/accuracy.py``,
next_token match rate, one quick steady timing per variant, chosen attention / GEMM backends,
PDL coverage). Exits non-zero on any FAIL or ERROR.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback

import torch

from regimebench.arms.base import expected_outputs, load_arm
from regimebench.fixture import make_inputs
from regimebench.harness import accuracy
from regimebench.ref.reference import forward
from regimebench.workload import select


def _steady_us(step, target_ms: float = 20.0) -> float:
    def run(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(n):
            step.launch()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / n
    run(3)
    est = run(3)
    n = int(min(max(target_ms / max(est, 1e-3), 3), 50))
    vals = sorted(run(n) for _ in range(10))
    return vals[len(vals) // 2] * 1e3


def _timeline(step, n: int = 5, top: int = 15) -> None:
    """Top kernels by total GPU time per step, kernel count, kernel-busy vs step time."""
    from torch.profiler import ProfilerActivity, profile
    for _ in range(3):
        step.launch()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        step.launch()
    b.record()
    torch.cuda.synchronize()
    step_us = a.elapsed_time(b) / n * 1e3
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(n):
            step.launch()
        torch.cuda.synchronize()
    agg, spans = {}, []
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA:
            continue
        dur = getattr(e, "device_time", None)
        dur = e.cuda_time if dur is None else dur  # us
        c = agg.setdefault(e.name, [0, 0.0])
        c[0] += 1
        c[1] += dur
        spans.append((e.time_range.start, e.time_range.end))
    spans.sort()
    busy, cur_s, cur_e = 0.0, None, None
    for s0, e0 in spans:
        if cur_e is None or s0 > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s0, e0
        else:
            cur_e = max(cur_e, e0)
    if cur_e is not None:
        busy += cur_e - cur_s
    total = sum(v[1] for v in agg.values()) / n
    print(f"  timeline: step {step_us:.1f} us (graph replay, mean of {n}); "
          f"{len(spans) / n:.0f} device ops/step; sum kernel time {total:.1f} us; "
          f"kernel-busy (union) {busy / n:.1f} us = {100 * busy / n / step_us:.0f}% of step")
    print(f"  {'us/step':>9} {'%step':>6} {'calls':>6}  kernel")
    for name, (cnt, t) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:top]:
        print(f"  {t / n:9.1f} {100 * t / n / step_us:6.1f} {cnt / n:6.1f}  {name[:110]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", default=None, help="ids/regimes; default = all check=True")
    ap.add_argument("--variants", default=None)
    ap.add_argument("--seed", type=int, default=7301)
    ap.add_argument("--json", default=None)
    ap.add_argument("--timeline", action="store_true",
                    help="for the first (headline) variant print top-15 kernels per step")
    args = ap.parse_args()
    dev = "cuda"
    arm = load_arm("fi-graph")
    variants = args.variants.split(",") if args.variants else list(arm.variants())
    wls = select(args.workloads.split(",") if args.workloads else None)
    if not args.workloads:
        wls = [w for w in wls if w.check]
    rows, failures = [], 0
    for w in wls:
        print(f"=== {w.id} (n_seq {w.n_seq}, total_q {w.total_q}, max_kv {w.max_kv})", flush=True)
        inputs = make_inputs(w, args.seed, dev)
        try:
            ref16 = forward(w, inputs, torch.bfloat16)
            ref32 = forward(w, inputs, torch.float32)
            band = accuracy.oracle_band(ref16, ref32)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            failures += 1
            rows.append((w.id, "-", "ERROR(reference)", "", "", "", "", ""))
            del inputs
            gc.collect()
            torch.cuda.empty_cache()
            continue
        for v in variants:
            reason = arm.supports(w, v)
            if reason:
                rows.append((w.id, v, "SKIP", reason, "", "", "", ""))
                continue
            step = None
            try:
                step = arm.prepare(w, inputs, v, dev)
                step.launch()
                torch.cuda.synchronize()
                out = step.outputs()
                for k, (shape, dt) in expected_outputs(w).items():
                    assert tuple(out[k].shape) == shape, f"{k} shape {tuple(out[k].shape)} != {shape}"
                    assert str(out[k].dtype).endswith(dt), f"{k} dtype {out[k].dtype} != {dt}"
                m32 = accuracy.compare(out, ref32)
                g = accuracy.grade_vs_oracle(m32, band)
                # idempotence: a second launch must reproduce the same logits
                step.launch()
                torch.cuda.synchronize()
                again = step.outputs()["logits"]
                idem = float((again - out["logits"]).abs().max())
                us = _steady_us(step)
                if args.timeline and v == variants[0]:
                    print(f"  [{w.id} {v}] attn={step.config['attn_backend']} gemm={step.config['gemm']}")
                    _timeline(step)
                c = step.config
                p = c["pdl"]
                ok = g["verdict"] == "pass" and idem <= 0.02 * float(out["logits"].abs().max())
                failures += not ok
                rows.append((w.id, v, "PASS" if ok else "FAIL",
                             f"ratio {g['worst_ratio']:.2f} (<=2) repeat-launch max_abs {idem:.1e}",
                             f"{m32['next_token']['match_rate']:.2f}", f"{us:.1f}",
                             f"{c['attn_backend']}",
                             f"gemm={c['gemm']} pdl={p['launches_with_pdl']}/{p['launches_total']}"))
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                failures += 1
                rows.append((w.id, v, "ERROR", f"{type(e).__name__}: {str(e)[:100]}", "", "", "", ""))
            finally:
                if step is not None:
                    step.close()
                del step
                gc.collect()
                torch.cuda.empty_cache()
        del inputs, ref16, ref32
        gc.collect()
        torch.cuda.empty_cache()
    hdr = ("workload", "variant", "verdict", "accuracy", "tok_match", "steady_us", "attn", "gemm/pdl")
    print("\n" + " | ".join(hdr))
    for r in rows:
        print(" | ".join(str(x) for x in r))
    print(f"\n{'ALL PASS' if not failures else f'{failures} FAILURE(S)'}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump([dict(zip(hdr, r)) for r in rows], f, indent=1)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
