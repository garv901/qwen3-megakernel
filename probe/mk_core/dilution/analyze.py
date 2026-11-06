"""Decompose why the mk gain dilutes in vLLM serving.

Inputs
  * runs/dilution/<UTC>-{A_c1,A_c8,A_c32,A_c64,B_c1sync}/<arm>/{steps.jsonl,bench.json}  (job dil.sbatch)
  * regime-mk-serve runs/e2e/<UTC>-m2_conv_<window>_L<n>/<arm>/{steps.jsonl,bench.json}   (M2 trace replays)
Step log fields (mkserve/steplog.py): fwd_us = model forward on the GPU (CUDA events around the model call);
gap_us = GPU time from the previous step's end event to this step's start event (sampler, input copies and
GPU idle while the host prepares the step); host_us = host time of the hooked forward call.

Output: results/2026-10-29-dilution/analysis.md and analysis.json.
Run: python3 probe/mk_core/dilution/analyze.py
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DIL = os.path.join(ROOT, "runs", "dilution")
M2 = "/raid/garv901/regime-mk-serve/runs/e2e"
OUT = os.path.join(ROOT, "results", "2026-10-29-dilution")
IDLE_US = 5000.0  # a gap longer than this is the engine waiting for work, not per-step overhead

# Offline regimebench, R1-b1-kv128 (us/step): mk job 11565, fi-graph best job 11565, vLLM job 9806 (nopdl:
# wall = LLMEngine.step, span = CUDA-event span of execute_model + sample_tokens).
OFFLINE = {"mk": 742.4, "fi": 891.4, "vllm_wall": 2476.2, "vllm_span": 1837.0}


def load_steps(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("fwd_us") is not None:
                out.append(r)
    return out


def is_decode(r: dict) -> bool:
    return r["label"].startswith(("D1", "DB"))


def med(xs):
    return st.median(xs) if xs else float("nan")


def latest(pattern: str) -> str | None:
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def step_stats(steps: list[dict]) -> dict:
    dec = [r for r in steps if is_decode(r)]
    gaps = [r["gap_us"] for r in dec if r.get("gap_us") is not None and r["gap_us"] < IDLE_US]
    return {
        "n_steps": len(steps), "n_decode": len(dec),
        "decode_covered": sum(r["fwd"] != "stock" for r in dec) / max(len(dec), 1),
        "fwd_med": med([r["fwd_us"] for r in dec]), "gap_med": med(gaps),
        "period_med": med([r["fwd_us"] + r["gap_us"] for r in dec
                           if r.get("gap_us") is not None and r["gap_us"] < IDLE_US]),
        "host_med": med([r["host_us"] for r in dec]),
        "batch_med": med([len(r["q"]) for r in dec]),
        "kv_med": med([st.mean(r["kv"]) for r in dec]),
    }


def bench(path: str) -> dict:
    b = json.load(open(path))
    keys = ("mean_tpot_ms", "median_tpot_ms", "median_itl_ms", "mean_ttft_ms", "output_throughput", "duration",
            "completed", "max_concurrent_requests")
    return {k: b.get(k) for k in keys}


def controlled() -> dict:
    res = {}
    for tag in ("A_c1", "A_c8", "A_c32", "A_c64", "B_c1sync"):
        d = latest(os.path.join(DIL, f"*-{tag}"))
        if not d:
            continue
        res[tag] = {}
        for arm in ("stock", "fi", "mk"):
            sp, bp = os.path.join(d, arm, "steps.jsonl"), os.path.join(d, arm, "bench.json")
            if os.path.exists(sp) and os.path.exists(bp):
                res[tag][arm] = {**step_stats(load_steps(sp)), **bench(bp), "dir": os.path.join(d, arm)}
    return res


def timeline(steps: list[dict]) -> dict:
    """GPU-timeline accounting (us): forward on custom path / on stock path, per-step gap, idle (gap >= IDLE_US)."""
    acc = defaultdict(float)
    for r in steps:
        acc["fwd_custom" if r["fwd"] != "stock" else ("fwd_stock_decode" if is_decode(r) else "fwd_stock_other")] \
            += r["fwd_us"]
        g = r.get("gap_us")
        if g is not None:
            acc["idle" if g >= IDLE_US else "gap"] += g
    acc["n"] = len(steps)
    acc["n_gap"] = sum(1 for r in steps if r.get("gap_us") is not None and r["gap_us"] < IDLE_US)
    dec = [r for r in steps if is_decode(r)]
    acc["n_decode"] = len(dec)
    acc["decode_fwd_mean"] = st.mean(r["fwd_us"] for r in dec) if dec else float("nan")
    acc["decode_batch_mean"] = st.mean(len(r["q"]) for r in dec) if dec else float("nan")
    return dict(acc)


def m2(window: str = "median") -> dict:
    res = {}
    for load in (1, 2, 3, 4):
        d = latest(os.path.join(M2, f"*-m2_conv_{window}_L{load}"))
        if not d:
            continue
        res[f"L{load}"] = {}
        for arm in ("stock", "mk"):
            sp, bp = os.path.join(d, arm, "steps.jsonl"), os.path.join(d, arm, "bench.json")
            if os.path.exists(sp) and os.path.exists(bp):
                steps = load_steps(sp)
                res[f"L{load}"][arm] = {**timeline(steps), **bench(bp), **step_stats(steps), "dir": os.path.join(d, arm)}
    return res


def pct(a, b):
    return (1 - a / b) * 100 if b else float("nan")


def write(ctrl: dict, real: dict) -> None:
    os.makedirs(OUT, exist_ok=True)
    json.dump({"controlled": ctrl, "m2_conv_median": real, "offline": OFFLINE}, open(os.path.join(OUT, "analysis.json"), "w"),
              indent=1)
    L = ["# Dilution analysis (generated by probe/mk_core/dilution/analyze.py)", ""]
    L += ["## 1. Controlled closed-loop decode (input 128, output 256, ignore-eos)", "",
          "Medians over decode steps; period = fwd + gap (gap < 5 ms). Gains are mk vs stock in the same job.", "",
          "| run | arm | batch | covered | fwd us | gap us | period us | host us | mean TPOT ms | median ITL ms |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, arms in ctrl.items():
        for arm, s in arms.items():
            L.append(f"| {tag} | {arm} | {s['batch_med']:.0f} | {s['decode_covered']:.2f} | {s['fwd_med']:.0f} | "
                     f"{s['gap_med']:.0f} | {s['period_med']:.0f} | {s['host_med']:.0f} | {s['mean_tpot_ms']:.3f} | "
                     f"{s['median_itl_ms']:.3f} |")
    L += ["", "| run | fwd gain % | period gain % | TPOT gain % | gap mk - stock us | fi: fwd / TPOT gain % |",
          "|---|---|---|---|---|---|"]
    for tag, a in ctrl.items():
        if "stock" in a and "mk" in a:
            s, m = a["stock"], a["mk"]
            fi = a.get("fi")
            fis = f"{pct(fi['fwd_med'], s['fwd_med']):.1f} / {pct(fi['mean_tpot_ms'], s['mean_tpot_ms']):.1f}" if fi else "-"
            L.append(f"| {tag} | {pct(m['fwd_med'], s['fwd_med']):.1f} | {pct(m['period_med'], s['period_med']):.1f} | "
                     f"{pct(m['mean_tpot_ms'], s['mean_tpot_ms']):.1f} | {m['gap_med'] - s['gap_med']:+.0f} | {fis} |")
    if "A_c1" in ctrl:
        a = ctrl["A_c1"]
        L += ["", "## 2. Offline benchmark vs the same forward inside serving (b1, short KV)", "",
              "| quantity | offline us | serving us (A_c1 median) | serving / offline |", "|---|---|---|---|"]
        for arm, key in (("mk", "mk"), ("fi", "fi")):
            if arm in a:
                L.append(f"| {arm} forward | {OFFLINE[key]:.0f} | {a[arm]['fwd_med']:.0f} | "
                         f"{a[arm]['fwd_med'] / OFFLINE[key]:.2f} |")
        if "stock" in a:
            s = a["stock"]
            L.append(f"| vLLM forward (serving: model forward only) | {OFFLINE['vllm_span']:.0f} (execute_model + "
                     f"sample span) | {s['fwd_med']:.0f} | {s['fwd_med'] / OFFLINE['vllm_span']:.2f} |")
            L.append(f"| vLLM step (offline wall; serving fwd + gap) | {OFFLINE['vllm_wall']:.0f} | {s['period_med']:.0f} | "
                     f"{s['period_med'] / OFFLINE['vllm_wall']:.2f} |")
    L += ["", "## 3. Real trace (Azure conv, median window, M2): GPU-timeline accounting, 180 s slice", "",
          "Seconds of GPU timeline per category; idle = gaps >= 5 ms (no runnable work).", "",
          "| load | arm | fwd mk path s | fwd stock decode s | fwd stock other s | gap s | idle s | mean TPOT ms |",
          "|---|---|---|---|---|---|---|---|"]
    for load, arms in real.items():
        for arm, t in arms.items():
            L.append(f"| {load} | {arm} | {t.get('fwd_custom', 0) / 1e6:.1f} | {t.get('fwd_stock_decode', 0) / 1e6:.1f} | "
                     f"{t.get('fwd_stock_other', 0) / 1e6:.1f} | {t.get('gap', 0) / 1e6:.1f} | {t.get('idle', 0) / 1e6:.1f} | "
                     f"{t['mean_tpot_ms']:.3f} |")
    L += ["", "| load | arm | steps | decode steps | mean decode batch | mean decode fwd us | mean gap us/step |",
          "|---|---|---|---|---|---|---|"]
    for load, arms in real.items():
        for arm, t in arms.items():
            L.append(f"| {load} | {arm} | {t['n']} | {t['n_decode']} | {t['decode_batch_mean']:.2f} | "
                     f"{t['decode_fwd_mean']:.0f} | {t.get('gap', 0) / max(t['n_gap'], 1):.0f} |")
    L += ["", "| load | busy s stock -> mk | of which decode fwd saved s | gap change s | other fwd change s | "
          "TPOT gain % |", "|---|---|---|---|---|---|"]
    for load, a in real.items():
        if "stock" in a and "mk" in a:
            s, m = a["stock"], a["mk"]
            busy = lambda t: sum(t.get(k, 0) for k in ("fwd_custom", "fwd_stock_decode", "fwd_stock_other", "gap"))
            dec = lambda t: t.get("fwd_custom", 0) + t.get("fwd_stock_decode", 0)
            L.append(f"| {load} | {busy(s) / 1e6:.1f} -> {busy(m) / 1e6:.1f} | {(dec(s) - dec(m)) / 1e6:.1f} | "
                     f"{(m.get('gap', 0) - s.get('gap', 0)) / 1e6:+.1f} | "
                     f"{(m.get('fwd_stock_other', 0) - s.get('fwd_stock_other', 0)) / 1e6:+.1f} | "
                     f"{pct(m['mean_tpot_ms'], s['mean_tpot_ms']):.1f} |")
    open(os.path.join(OUT, "analysis.md"), "w").write("\n".join(L) + "\n")
    print(os.path.join(OUT, "analysis.md"))


if __name__ == "__main__":
    write(controlled(), m2())
