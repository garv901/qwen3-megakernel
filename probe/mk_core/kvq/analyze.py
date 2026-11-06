"""kvq analysis: $PY analyze.py results/2026-10-28-kvq/kvq_<job>.jsonl [--best] [--md]. Table: point | arm | C | order | A | imb | tail | spread | B/CTA min-max | S_max | units."""
import json, sys
from collections import defaultdict

args = [a for a in sys.argv[1:] if not a.startswith("--")]
rows = [json.loads(l) for f in args for l in open(f)]
rows = [r for r in rows if "A_med" in r]
md = "--md" in sys.argv
hdr = ["point", "arm", "C", "order", "A us", "imb", "tail", "spread", "B/CTA MB min-max", "S_max", "units", "u/CTA"]


def line(r):
    return [r["point"], r["arm"], str(r["C"]) if r["arm"] != "static" else "-", r["order"], f"{r['A_med']:.1f}", f"{r['imb_med']:.2f}", f"{r['tail_med']:.2f}",
            f"{r['spread_med']:.2f}", f"{r['cta_bytes_min']/1e6:.2f}-{r['cta_bytes_max']/1e6:.2f}", str(r["s_max"]), str(r["n_units"]), f"{r['cta_units_min']}-{r['cta_units_max']}"]


def show(rs):
    if md:
        print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
        for r in rs: print("| " + " | ".join(line(r)) + " |")
    else:
        print("  ".join(f"{h:>10s}" for h in hdr))
        for r in rs: print("  ".join(f"{x:>10s}" for x in line(r)))


order = {"static": 0, "static_il": 1, "queue": 2, "queue_pf": 3, "hybrid": 4}
pts = list(dict.fromkeys(r["point"] for r in rows))
if "--best" in sys.argv:
    best = []
    for p in pts:
        st = [r for r in rows if r["point"] == p and r["arm"] == "static"]
        for arm in order:
            c = [r for r in rows if r["point"] == p and r["arm"] == arm]
            if c: best.append(min(c, key=lambda r: r["A_med"]))
    show(best)
    print()
    print("A vs static (best C / order per arm):")
    for p in pts:
        st = [r for r in rows if r["point"] == p and r["arm"] == "static"]
        if not st: continue
        s = st[0]["A_med"]
        print(f"{p:12s} static {s:8.1f} | " + " | ".join(f"{a} {min(r['A_med'] for r in rows if r['point']==p and r['arm']==a):.1f}" for a in order if a != "static" and any(r['point']==p and r['arm']==a for r in rows)))
else:
    show(sorted(rows, key=lambda r: (pts.index(r["point"]), order[r["arm"]], r["C"], r["order"])))
