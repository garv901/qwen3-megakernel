"""GPU check for the hv2 arms (hv2-repeat and hv2g).

    PYTHONPATH=. python scripts/gpu_check_hv2g.py [--workloads R0-b1-kv128,R2-b1-kv1024,...] [--seed 7401] [--json OUT.json]

1. hv2-repeat N = 1 against the frozen hv2 ``run()`` on the same inputs: logits / next_token / k_write / v_write must be bit-exact.
2. hv2g (variant ``auto``; plus ``mtn`` on the first workload and on R0) on each workload: rel-L2 of logits / k_new / v_new to the FP32
   reference against the BF16 reference's own rel-L2 (pass = worst ratio <= 2), next_token match, repeat-launch idempotence.
3. Interleaved timing of hv2 (frozen run()) vs hv2g MT1 at R0, 10 rounds x 20 launches (medians; informational, the 3% target is
   reported as OK/OVER but does not fail the check).
Prints a PASS/FAIL table; exit code 1 on any FAIL / ERROR in 1 or 2.
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

DEFAULT = ["R0-b1-kv128", "R2-b1-kv1024", "R2-b1-kv8192", "R1-b8-kv128", "R4-b1-T16", "R4-b1-T256",
           "R6-b1-kv1024-q4", "R7-p512-d8", "R8-b32-ragged"]


def check_repeat(seed: int, dev: str) -> tuple[bool, str]:
    from regimebench.arms import hv2_repeat as HR
    w = select(["R0-b1-kv128"])[0]
    inputs = make_inputs(w, seed, dev)
    arm = load_arm("hv2-repeat")
    step = arm.prepare(w, inputs, "direct", dev)
    step.launch()
    torch.cuda.synchronize()
    mine = {k: v.clone() for k, v in step.raw_outputs().items()}
    mod = HR.load_hv2()
    run = mod.build(HR.HV2_CASE)
    t = step._t
    ref = run({"token": t["token"], **{k: t[k] for k in HR.WEIGHT_KEYS}, "kcache": t["kcache"], "vcache": t["vcache"]})
    torch.cuda.synchronize()
    bad = [k for k in ("logits", "next_token", "k_write", "v_write")
           if not torch.equal(mine[k].reshape(-1), ref[k].reshape(-1))]
    step.close()
    return (not bad), ("bit-exact" if not bad else "MISMATCH in " + ",".join(bad))


def timing_r0(seed: int, dev: str) -> str:
    from regimebench.arms import hv2_repeat as HR
    w = select(["R0-b1-kv128"])[0]
    inputs = make_inputs(w, seed, dev)
    rep = load_arm("hv2-repeat").prepare(w, inputs, "direct", dev)
    t = rep._t
    run = HR.load_hv2().build(HR.HV2_CASE)
    hin = {"token": t["token"], **{k: t[k] for k in HR.WEIGHT_KEYS}, "kcache": t["kcache"], "vcache": t["vcache"]}
    g = load_arm("hv2g").prepare(w, inputs, "auto", dev)
    assert g.mt1, "R0 must use the MT1 build"

    def hv2_once():
        run(hin)

    def timed(fn, n=20):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(n):
            fn()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / n * 1e3
    for _ in range(3):
        hv2_once(); g.launch()
    torch.cuda.synchronize()
    a, b = [], []
    for _ in range(10):
        a.append(timed(hv2_once)); b.append(timed(g.launch))
    a.sort(); b.sort()
    ma, mb = a[len(a) // 2], b[len(b) // 2]
    rep.close(); g.close()
    return (f"hv2 run() median {ma:.1f} us | hv2g MT1 (graph) median {mb:.1f} us | ratio {mb / ma:.3f} "
            f"({'OK' if mb <= 1.03 * ma else 'OVER 3%'}; note hv2 run() pays Python launch + per-call allocs, graph does not)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", default=None, help="ids; default = the SPEC list")
    ap.add_argument("--seed", type=int, default=7401)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    dev = "cuda"
    ids = args.workloads.split(",") if args.workloads else DEFAULT
    rows, failures = [], 0
    try:
        ok, msg = check_repeat(args.seed, dev)
        failures += not ok
        rows.append(("R0-b1-kv128", "hv2-repeat N=1 vs hv2 run()", "PASS" if ok else "FAIL", msg, "", ""))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc(); failures += 1
        rows.append(("R0-b1-kv128", "hv2-repeat N=1 vs hv2 run()", "ERROR", f"{type(e).__name__}: {str(e)[:100]}", "", ""))
    gc.collect(); torch.cuda.empty_cache()
    arm = load_arm("hv2g")
    for wi, w in enumerate(select(ids)):
        print(f"=== {w.id} (n_seq {w.n_seq}, total_q {w.total_q}, max_kv {w.max_kv})", flush=True)
        inputs = make_inputs(w, args.seed, dev)
        try:
            ref16 = forward(w, inputs, torch.bfloat16)
            ref32 = forward(w, inputs, torch.float32)
            band = accuracy.oracle_band(ref16, ref32)
        except Exception:  # noqa: BLE001
            traceback.print_exc(); failures += 1
            rows.append((w.id, "-", "ERROR", "reference", "", ""))
            continue
        variants = ["auto"] + (["mtn"] if (wi == 0 or w.total_q == 1) else [])
        for v in variants:
            step = None
            try:
                step = arm.prepare(w, inputs, v, dev)
                step.launch(); torch.cuda.synchronize()
                out = step.outputs()
                for k, (shape, dt) in expected_outputs(w).items():
                    assert tuple(out[k].shape) == shape and str(out[k].dtype).endswith(dt), f"{k}: {out[k].shape} {out[k].dtype}"
                m32 = accuracy.compare(out, ref32)
                gr = accuracy.grade_vs_oracle(m32, band)
                step.launch(); torch.cuda.synchronize()
                idem = float((step.outputs()["logits"] - out["logits"]).abs().max())
                ok = gr["verdict"] == "pass"
                failures += not ok
                c = step.config
                detail = (f"ratio {gr['worst_ratio']:.2f} (<=2; logits {gr['ratios']['logits']:.2f} k {gr['ratios']['k_new']:.2f} "
                          f"v {gr['ratios']['v_new']:.2f}) repeat max_abs {idem:.1e} | {c['kernel']} launches {c['launches_per_step']} "
                          f"regs {c['regs']} local {c['local_bytes']}")
                rows.append((w.id, v, "PASS" if ok else "FAIL", detail, f"{m32['next_token']['match_rate']:.2f}", ""))
            except Exception as e:  # noqa: BLE001
                traceback.print_exc(); failures += 1
                rows.append((w.id, v, "ERROR", f"{type(e).__name__}: {str(e)[:100]}", "", ""))
            finally:
                if step is not None:
                    step.close()
                del step
                gc.collect(); torch.cuda.empty_cache()
        del inputs
        gc.collect(); torch.cuda.empty_cache()
    try:
        tmsg = timing_r0(args.seed, dev)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        tmsg = f"ERROR {type(e).__name__}: {str(e)[:100]}"
    hdr = ("workload", "check", "verdict", "detail", "tok_match", "")
    print("\n" + " | ".join(hdr[:5]))
    for r in rows:
        print(" | ".join(str(x) for x in r[:5]))
    print(f"\ntiming R0: {tmsg}")
    print(f"\n{'ALL PASS' if not failures else f'{failures} FAILURE(S)'}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"rows": [dict(zip(hdr, r)) for r in rows], "timing": tmsg}, f, indent=1)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
