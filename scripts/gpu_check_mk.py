"""GPU check for the mk arm (v0). Not run yet: written on a CPU-only host.

    PYTHONPATH=. python scripts/gpu_check_mk.py [--workloads ids] [--seed 7401] [--json OUT.json]      # accuracy + idempotence + timing
    PYTHONPATH=. python scripts/gpu_check_mk.py --null MASK [--workloads ids]                          # null-work build step times (outputs wrong by design)
    PYTHONPATH=. python scripts/gpu_check_mk.py --stamps [--workloads ids]                              # per-phase times from %globaltimer stamps

Default check, per workload: rel-L2 of logits / k_new / v_new to the FP32 reference against the BF16 reference's own rel-L2 (pass = worst ratio
<= 2, harness/accuracy.py), next_token match rate, repeat-launch idempotence (same logits to a small tolerance: red.add order is not fixed),
one steady timing (graph replay, median of 5 x 20 launches). Repeat check: second launch passes the band too and
the launch-to-launch rel-L2 is < 0.5 x the BF16 band (job 9576: 0.08-0.24 from red.add order); next_token must equal the argmax of mk's own logits. Exit code 1 on any FAIL / ERROR.
--null MASK: bit p-1 = phase p (1..6), bit 6 = LM head; prints the median step time next to the full build's.
--stamps: median over CTAs and layers 4..23 of (wait exit of the previous barrier -> arrive) per phase, and of (arrive -> wait exit) per barrier.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import traceback

import torch

from regimebench.arms.base import expected_outputs, load_arm
from regimebench.fixture import make_inputs
from regimebench.harness import accuracy
from regimebench.ref.reference import forward
from regimebench.workload import select

DEFAULT = ["R0-b1-kv128", "R1-b2-kv128", "R1-b8-kv128", "R1-b16-kv128", "R1-b32-kv128", "R1-b64-kv128", "R1-b128-kv128",
           "R1-b256-kv128", "R2-b1-kv8192", "R6-b1-kv1024-q4", "R3-b8-kv4096"]
PHASES = ("QKV", "qk-norm+RoPE+KV", "attention", "O", "GU", "D")


def timed(step, n: int = 20, rounds: int = 5) -> float:
    for _ in range(3):
        step.launch()
    torch.cuda.synchronize()
    out = []
    for _ in range(rounds):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(n):
            step.launch()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) / n * 1e3)
    return statistics.median(out)


def check(args, dev: str) -> int:
    arm = load_arm("mk", attn=args.attn)
    rows, failures = [], 0
    for w in select(args.workloads.split(",") if args.workloads else DEFAULT):
        print(f"=== {w.id} (n_seq {w.n_seq}, total_q {w.total_q}, max_kv {w.max_kv})", flush=True)
        inputs = make_inputs(w, args.seed, dev)
        step = None
        try:
            ref16 = forward(w, inputs, torch.bfloat16)
            ref32 = forward(w, inputs, torch.float32)
            band = accuracy.oracle_band(ref16, ref32)
            step = arm.prepare(w, inputs, "auto", dev)
            step.launch(); torch.cuda.synchronize()
            out = step.outputs()
            for k, (shape, dt) in expected_outputs(w).items():
                assert tuple(out[k].shape) == shape and str(out[k].dtype).endswith(dt), f"{k}: {out[k].shape} {out[k].dtype}"
            gr = accuracy.grade_vs_oracle(accuracy.compare(out, ref32), band)
            m32 = accuracy.compare(out, ref32)
            self_arg = float((out["next_token"].long() == out["logits"].argmax(-1)).float().mean())   # argmax of mk's own logits
            step.launch(); torch.cuda.synchronize()
            out2 = step.outputs()
            gr2 = accuracy.grade_vs_oracle(accuracy.compare(out2, ref32), band)
            idem = float((out2["logits"] - out["logits"]).abs().max())
            # red.add order is not fixed: repeat launches must each pass the band and differ by << the BF16 band
            rep = max(float((out2[k].float() - out[k].float()).norm() / out[k].float().norm().clamp_min(1e-30)) / max(band[k], accuracy.BAND_FLOOR)
                      for k in ("logits", "k_new", "v_new"))
            us = timed(step)
            ok = gr["verdict"] == "pass" and gr2["verdict"] == "pass" and rep < 0.5 and self_arg == 1.0
            failures += not ok
            c = step.config
            rows.append((w.id, "PASS" if ok else "FAIL",
                         f"ratio {gr['worst_ratio']:.2f} (logits {gr['ratios']['logits']:.2f} k {gr['ratios']['k_new']:.2f} v {gr['ratios']['v_new']:.2f}) "
                         f"tok_match {m32['next_token']['match_rate']:.2f} (ref argmax {m32['logits']['argmax_match_rate']:.2f}, self {self_arg:.2f}) "
                         f"repeat {rep:.3f} x band, max_abs {idem:.1e} | npad {c['npad']} regs {c['regs']} local {c['local_bytes']} | {us:.1f} us"))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc(); failures += 1
            rows.append((w.id, "ERROR", f"{type(e).__name__}: {str(e)[:120]}"))
        finally:
            if step is not None:
                step.close()
            del step, inputs
            gc.collect(); torch.cuda.empty_cache()
    print("\nworkload | verdict | detail")
    for r in rows:
        print(" | ".join(r))
    print(f"\n{'ALL PASS' if not failures else f'{failures} FAILURE(S)'}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump([dict(zip(("workload", "verdict", "detail"), r)) for r in rows], f, indent=1)
    return 1 if failures else 0


def null_mode(args, dev: str) -> int:
    full, nul = load_arm("mk", attn=args.attn), load_arm("mk", null=args.null, attn=args.attn)
    for w in select(args.workloads.split(",") if args.workloads else DEFAULT):
        inputs = make_inputs(w, args.seed, dev)
        a = full.prepare(w, inputs, "auto", dev)
        b = nul.prepare(w, inputs, "auto", dev)
        ta, tb = timed(a), timed(b)
        print(f"{w.id}: full {ta:.1f} us | null mask {args.null:#x} {tb:.1f} us | work in the masked phases {ta - tb:.1f} us", flush=True)
        a.close(); b.close(); del a, b, inputs
        gc.collect(); torch.cuda.empty_cache()
    return 0


def stamps_mode(args, dev: str) -> int:
    arm = load_arm("mk", stamps=1, attn=args.attn)
    nbar = 1 + 6 * 28
    for w in select(args.workloads.split(",") if args.workloads else DEFAULT):
        inputs = make_inputs(w, args.seed, dev)
        step = arm.prepare(w, inputs, "auto-direct", dev)
        for _ in range(3):
            step.launch()
        torch.cuda.synchronize()
        st = step.stamps().astype("int64")
        arr, wex = st[:, 0:2 * nbar:2], st[:, 1:2 * nbar:2]   # [cta, barrier]
        import numpy as np
        phase = {p: [] for p in range(6)}
        bwait = {p: [] for p in range(6)}
        for layer in range(4, 24):
            for p in range(6):
                b = 1 + 6 * layer + p
                phase[p].append(arr[:, b] - wex[:, b - 1])
                bwait[p].append(wex[:, b] - arr[:, b])
        lm = (st[:, 2 * nbar] - wex[:, nbar - 1]) / 1e3
        print(f"{w.id} (npad {step.npad}): per-phase work (us, median over CTAs and layers 4..23) | barrier wait")
        tot = 0.0
        for p in range(6):
            a, bw = float(np.median(np.concatenate(phase[p]))) / 1e3, float(np.median(np.concatenate(bwait[p]))) / 1e3
            tot += a + bw
            print(f"  {p + 1} {PHASES[p]:18s} {a:8.2f} | {bw:6.2f}")
        print(f"  layer sum {tot:.2f} us; LM phase {float(np.median(lm)):.2f} us; prologue barrier arrive->exit "
              f"{float(np.median(wex[:, 0] - arr[:, 0])) / 1e3:.2f} us", flush=True)
        step.close(); del step, inputs
        gc.collect(); torch.cuda.empty_cache()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", default=None)
    ap.add_argument("--seed", type=int, default=7401)
    ap.add_argument("--json", default=None)
    ap.add_argument("--null", type=lambda s: int(s, 0), default=None)
    ap.add_argument("--stamps", action="store_true")
    ap.add_argument("--attn", default="placeholder", help="MkArm attn option: placeholder | splitkv | auto")
    args = ap.parse_args()
    dev = "cuda"
    if args.null is not None:
        return null_mode(args, dev)
    if args.stamps:
        return stamps_mode(args, dev)
    return check(args, dev)


if __name__ == "__main__":
    sys.exit(main())
