"""GPU check for the vLLM worker: R0, R1-b32, R2-kv8192, R4-T1024, R7-d8 once each.

    PYTHONPATH=/raid/garv901/regime-megakernel python scripts/gpu_check_vllm.py [--variants pdl,nopdl] [--rounds 5]

Prints one row per (workload, variant): status, median step time, step shape, mean kv, cudagraph
mode. Exit code 1 if any row is not a timing record with an exact step shape.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from regimebench.workload import select

IDS = ["R0-b1-kv128", "R1-b32-kv128", "R2-b1-kv8192", "R4-b1-T1024", "R7-p512-d8"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="pdl")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--workloads", default=",".join(IDS))
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="vllm_check_"))
    rows, ok_all = [], True
    for wid in a.workloads.split(","):
        w = select([wid])[0]
        wj = tmp / f"{wid}.json"
        wj.write_text(json.dumps(w.to_dict()))
        for var in a.variants.split(","):
            out = tmp / f"{wid}-{var}.jsonl"
            t0 = time.time()
            p = subprocess.run([sys.executable, "-m", "regimebench.arms.vllm_worker",
                                "--workload-json", str(wj), "--variant", var, "--out", str(out),
                                "--run-id", "check", "--rounds", str(a.rounds)],
                               capture_output=True, text=True)
            recs = [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
            tim = [r for r in recs if r["record_type"] == "timing"]
            if tim:
                r = tim[0]
                c = r["config"]
                exact = c["step_shape_exact"]
                ok = exact and p.returncode == 0
                rows.append((wid, var, "PASS" if ok else "FAIL(shape)",
                             f"wall {r['steady']['median'] * 1e3:.0f} us / gpu "
                             + (f"{r['gpu']['median'] * 1e3:.0f} us" if r.get("gpu") else "n/a"),
                             json.dumps(c["step_shapes"][0]["hist"]),
                             f"kv {c['mean_kv_cached_before_step']:.0f}",
                             c["engine_config"]["cudagraph_mode"].split(".")[-1]
                             + (" graph" if c["cudagraph_expected_by_size"] else " nograph"),
                             f"{time.time() - t0:.0f}s"))
            else:
                ok = False
                why = next((r.get("reason") or r.get("traceback", "")[-300:] for r in recs
                            if r["record_type"] in ("skip", "error")), p.stderr[-300:])
                rows.append((wid, var, "FAIL", "-", why.replace("\n", " ")[:120], "", "", ""))
            ok_all &= ok
    for r in rows:
        print(" | ".join(r))
    print("ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
