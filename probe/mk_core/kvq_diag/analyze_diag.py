"""kvq_diag tables. $PY analyze_diag.py d1|d2|decomp <jsonl ...> [--md]
d1:     A per EOI variant (rows) for static / queue / queue_pf configs at 3 points; per-unit slope of A over units per CTA (cap none, C 256/512/1024) per variant and the slope minus the e0 slope.
d2:     C(S) of the forced-S study (arm fs): rows S, columns head / wide k.
decomp: A + C excess of the queue over static, split into rows (see below), from d1 files (--d1 ...) and the decomp file.
Variant name: e<EOI>[/<partial stride B>]."""
import json, sys, statistics
from collections import defaultdict

args = [a for a in sys.argv[2:] if not a.startswith("--")]
mode = sys.argv[1]


def load(files):
    return [json.loads(l) for f in files for l in open(f) if '"A_med"' in l]


def vname(r):
    return f"e{r['eoi']}" + (f"/{r['ps'] * 4}B" if r["ps"] != 132 and not r["pbf"] else "") + ("/bf16" if r["pbf"] else "")


def fit(xs, ys):
    n = len(xs); mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")


VORD = ["e0", "e1", "e2", "e3", "e3/544B", "e3/576B", "e3/1024B", "e5", "e5/576B"]
if mode == "d1":
    rows = load(args)
    pts = list(dict.fromkeys(r["point"] for r in rows))
    print("### A (us) per end-of-item variant; configs: static, queue/queue_pf C1024 cap 32 (pieces >= len/32), C256 cap none")
    cfgs = [("static", 0, 0), ("queue", 1024, 32), ("queue_pf", 1024, 32), ("queue", 256, 0), ("queue_pf", 256, 0)]
    for p in pts:
        print(f"\n{p}\n| variant | " + " | ".join(f"{a} C{c} cap{k or 'none'} (u/CTA {next((r['n_units'] / 148 for r in rows if r['point'] == p and r['arm'] == a and r['C'] == c and r['cap'] == k), 0):.1f})" for a, c, k in cfgs) + " |\n|---|" + "---|" * len(cfgs))
        for v in VORD:
            line = []
            for a, c, k in cfgs:
                m = [r for r in rows if r["point"] == p and vname(r) == v and r["arm"] == a and r["C"] == c and r["cap"] == k]
                line.append(f"{m[0]['A_med']:.1f}" if m else "-")
            if any(x != "-" for x in line):
                print(f"| {v} | " + " | ".join(line) + " |")
    print("\n### per-unit slope: A vs units per CTA, queue + queue_pf pooled, cap none, C 256 / 512 / 1024 (us per unit); minus the e0 slope = per-unit end-of-item cost")
    print("| point | variant | slope us/unit | slope - e0 | A(e) - A(e0) at C256 per unit | (queue, C256, cap none) |\n|---|---|---|---|---|---|")
    for p in pts:
        s0 = None
        for v in VORD:
            for arm in ("queue", "queue_pf"):
                m = [r for r in rows if r["point"] == p and vname(r) == v and r["arm"] == arm and r["cap"] == 0 and r["C"] in (256, 512, 1024)]
                if len(m) < 3:
                    continue
                sl = fit([r["n_units"] / 148 for r in m], [r["A_med"] for r in m])
                if v == "e0":
                    s0 = (s0 or {}); s0[arm] = sl
                z = [r for r in rows if r["point"] == p and vname(r) == "e0" and r["arm"] == arm and r["cap"] == 0 and r["C"] == 256]
                mm = [r for r in m if r["C"] == 256][0]
                per_unit = (mm["A_med"] - z[0]["A_med"]) / (mm["n_units"] / 148) if z else float("nan")
                print(f"| {p} | {v} | {arm}: {sl:.3f} | {sl - (s0 or {}).get(arm, float('nan')):.3f} | {per_unit:.3f} | |")
elif mode == "d2":
    rows = load(args)
    pts = list(dict.fromkeys(r["point"] for r in rows))
    tags = list(dict.fromkeys((f"mk{r['mk']}" + ("/bf16" if r["pbf"] else ""), r["cmode"] + (f"-k{r['ck']}" if r["cmode"] == "wide" else "")) for r in rows))
    variants = list(dict.fromkeys(t[0] for t in tags))
    for var in variants:
        for p in pts:
            cols = [t[1] for t in tags if t[0] == var]
            print(f"\n{p}, {var}: C (us) vs forced S\n| S | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
            for S in sorted({r["C"] for r in rows if r["point"] == p}):
                line = []
                for c in cols:
                    m = [r for r in rows if r["point"] == p and r["C"] == S and f"mk{r['mk']}" + ("/bf16" if r["pbf"] else "") == var and r["cmode"] + (f"-k{r['ck']}" if r["cmode"] == "wide" else "") == c]
                    line.append(f"{m[0]['C_med']:.2f}" if m else "-")
                print(f"| {S} | " + " | ".join(line) + " |")
elif mode == "decomp":
    # d1r:<files> = replicate runs of the decomposition configs (A per variant); d1:<files> = the C sweep (slope, for the ring row); plain args = decomp job (head / wide combine on the same configs)
    r1f = [a[4:] for a in sys.argv[2:] if a.startswith("d1r:")]
    d1f = [a[3:] for a in sys.argv[2:] if a.startswith("d1:")]
    dec = [a for a in args if not a.startswith("d1")]
    RR, R1, R2 = load(r1f), load(d1f), load(dec)
    pts = list(dict.fromkeys(r["point"] for r in R2))
    jobs = sorted({r["job"] for r in RR})

    def X(p, arm, v, job):
        """queue minus static A in one job, mean over the C values (512, 1024) whose piece lists are identical (else C 1024 only)."""
        q = [r for r in RR if r["point"] == p and vname(r) == v and r["arm"] == arm and r["cap"] == 32 and r["job"] == job]
        s = [r for r in RR if r["point"] == p and vname(r) == v and r["arm"] == "static" and r["job"] == job]
        if not q or not s:
            return None
        if len({r["n_units"] for r in q}) > 1:
            q = [r for r in q if r["C"] == 1024]
        return statistics.mean(r["A_med"] for r in q) - s[0]["A_med"]

    def slope(p, arm, v):
        m = [r for r in R1 if r["point"] == p and vname(r) == v and r["arm"] == arm and r["cap"] == 0 and r["C"] in (256, 512, 1024)]
        return fit([r["n_units"] / 148 for r in m], [r["A_med"] for r in m])
    hdr = ["point", "queue arm", "u/CTA q / static", "total A+C excess (decomp job)", "A(e3) excess", "1a balance gain (A at e0 minus ring)", "1b ring / unit fetch (e0 slope x du)", "2 EOI barriers", "3 EOI smem merge", "4 store (best aligned)", "5 store alignment (528 B vs best)", "6 combine parallelism (head - wide best)", "7 combine S-growth (wide best)", "residual", "A noise between 2 replicate jobs (max |diff| of any X_v)"]
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for p in pts:
        for arm in ("queue", "queue_pf"):
            cq = [r for r in R2 if r["point"] == p and r["arm"] == arm and r["cap"] == 32 and r["C"] == 1024]
            cs = [r for r in R2 if r["point"] == p and r["arm"] == "static"]
            if not cq or not cs:
                continue
            Cq = {(r["cmode"], r["ck"]): r["C_med"] for r in cq}; Cs = {(r["cmode"], r["ck"]): r["C_med"] for r in cs}
            wq = min((k for k in Cq if k[0] == "wide"), key=Cq.get); ws = min((k for k in Cs if k[0] == "wide"), key=Cs.get)
            hq = [r for r in cq if r["cmode"] == "head"][0]; hs = [r for r in cs if r["cmode"] == "head"][0]
            tot = (hq["A_med"] + hq["C_med"]) - (hs["A_med"] + hs["C_med"])
            du = hq["n_units"] / 148 - hs["n_units"] / 148
            Xm = {v: [X(p, arm, v, j) for j in jobs] for v in VORD}
            Xm = {v: x for v, x in Xm.items() if all(t is not None for t in x)}
            M = {v: statistics.mean(x) for v, x in Xm.items()}
            noise = max(abs(x[0] - x[-1]) for x in Xm.values())
            al = min((v for v in M if v.startswith("e3/")), key=lambda v: M[v])
            ring = slope(p, arm, "e0") * du
            rows_ = [M["e0"] - ring, ring, M["e1"] - M["e0"], M["e2"] - M["e1"], M[al] - M["e2"], M["e3"] - M[al]]
            par = (Cq[("head", 1)] - Cq[wq]) - (Cs[("head", 1)] - Cs[ws]); sg = Cq[wq] - Cs[ws]
            res = tot - sum(rows_) - par - sg
            print(f"| {p} | {arm} C1024 cap32 | {hq['n_units'] / 148:.1f} / {hs['n_units'] / 148:.1f} | {tot:+.2f} | {hq['A_med'] - hs['A_med']:+.2f} (replicates {M['e3']:+.2f}) | " + " | ".join(f"{x:+.2f}" for x in rows_) +
                  f" ({al}) | {par:+.2f} (k{wq[1]}/k{ws[1]}) | {sg:+.2f} | {res:+.2f} | {noise:.2f} |")
