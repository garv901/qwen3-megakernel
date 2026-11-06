"""kvq + combine analysis: $PY analyze_comb.py results/2026-10-28-kvq/kvq_<job>.jsonl [..._ticket.jsonl] [--best] [--md]. Table: point | arm | C | cap | A | C_comb | A + C | S_max | vs static (A + C)."""
import json, sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
rows = [json.loads(l) for f in args for l in open(f)]
rows = [r for r in rows if "A_med" in r]
md = "--md" in sys.argv
hdr = ["point", "arm", "C", "cap", "A us", "C_comb us", "A + C us", "S_max", "vs static (A + C)", "vs static (A)"]
order = {"static": 0, "queue_pf": 1, "queue": 2, "hybrid": 3}
name = lambda r: r["arm"] + ("+ticket" if r.get("ticket") else "")
pts = list(dict.fromkeys(r["point"] for r in rows))
st = {r["point"]: r for r in rows if r["arm"] == "static" and not r.get("ticket")}


def line(r):
    s = st.get(r["point"])
    vs = f"{100 * (r['AC_med'] / s['AC_med'] - 1):+.1f}%" if s else "-"
    va = f"{100 * (r['A_med'] / s['A_med'] - 1):+.1f}%" if s else "-"
    return [r["point"], name(r) + ("" if r["order"] in ("pos", "-") else "/" + r["order"]), str(r["C"]) if r["arm"] != "static" else "-", str(r.get("cap") or "none") if r["arm"] != "static" else "-",
            f"{r['A_med']:.1f}", f"{r['C_med']:.2f}", f"{r['AC_med']:.1f}", str(r["s_max_dev"]), vs, va]


def show(rs):
    out = [hdr] + [line(r) for r in rs]
    for l in out:
        print(("| " + " | ".join(l) + " |") if md else "  ".join(f"{x:>14s}" for x in l))
        if md and l is hdr:
            print("|" + "---|" * len(hdr))


if "--best" in sys.argv:
    best = []
    for p in pts:
        for key in sorted({(r["arm"], r.get("ticket", 0)) for r in rows if r["point"] == p}, key=lambda k: (order[k[0]], k[1])):
            c = [r for r in rows if r["point"] == p and (r["arm"], r.get("ticket", 0)) == key and r["order"] in ("pos", "-")]
            if c: best.append(min(c, key=lambda r: r["AC_med"]))
    show(best)
else:
    show(sorted(rows, key=lambda r: (pts.index(r["point"]), order[r["arm"]], r.get("ticket", 0), r["order"], r.get("cap") or 10**6, r["C"])))
