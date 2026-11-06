"""kvq_pf tables: $PY analyze_pf.py [--md]. Reads results/2026-10-28-kvq-pf/sw_<job>_<variant>.jsonl (variant aXrY: KVQ_ACC X, KVQ_RPF Y); replicate jobs are listed per variant (job ids by name).
 1. dA: queue_pf A of RPF 3 vs RPF 0 (best-A config of each, same C / cap, same job pairs), per point, ACC 0 / 1 / 3, per replicate job
 2. best A + C per point for static, queue_pf + RPF (mk combine, ACC 0), queue_pf + RPF + R1 (ACC 1 / 3), with % vs static (static of the same job, ACC 0 build). Medians over replicate jobs.
 3. noise: replicate spread of the winning configs."""
import glob, json, re, statistics, sys
R = "results/2026-10-28-kvq-pf/"
md = "--md" in sys.argv
data = {}   # (job, variant) -> rows
for f in glob.glob(R + "sw_*_a*r*.jsonl"):
    m = re.search(r"sw_(\d+)_(a\d+r\d+)\.jsonl", f)
    data[(m.group(1), m.group(2))] = [json.loads(l) for l in open(f) if '"A_med"' in l]
jobs = sorted({j for j, _ in data})
# replicate groups: jobs that ran the same variant set
reps = {}
for (j, v) in data:
    reps.setdefault(v, []).append(j)
pts = list(dict.fromkeys(r["point"] for rows in data.values() for r in rows))


def key(r):
    return (r["point"], r["arm"], r["C"], r.get("cap") or 0, r["order"])


def table(rows_):
    d = {}
    for rows in rows_:
        for r in rows:
            d.setdefault(key(r), []).append(r)
    return d


def med(v, f):
    return statistics.median(f(r) for r in v)


def agg(v):
    d = table([data[(j, v)] for j in reps.get(v, [])])
    return d


def fmt(rows):
    out = []
    for l in rows:
        out.append(("| " + " | ".join(l) + " |") if md else "  ".join(f"{x:>16s}" for x in l))
        if md and l is rows[0]:
            out.append("|" + "---|" * len(l))
    print("\n".join(out))


A = {v: agg(v) for v in reps}
fails = sum(1 for rows in data.values() for r in rows if r.get("check_status") != "PASS")
print(f"# jobs {jobs}; configs {sum(len(r) for r in data.values())}; check non-PASS: {fails}\n")
print("## 1. dA from RPF (queue_pf, A us, RPF 3 - RPF 0; median over replicate jobs; per point the config with the lowest A in the RPF 0 build, and the best RPF 3 config)")
hdr = ["point", "ACC", "arm", "cfg (rpf0)", "A rpf0", "A rpf3 (same cfg)", "dA same cfg", "best rpf3 cfg", "A best rpf3", "dA best-best"]
rows = [hdr]
for acc in (0, 1, 3):
    a0, a3 = A.get(f"a{acc}r0", {}), A.get(f"a{acc}r3", {})
    for p in pts:
        for arm in ("queue_pf", "queue", "static"):
            c0 = {k: v for k, v in a0.items() if k[0] == p and k[1] == arm}
            c3 = {k: v for k, v in a3.items() if k[0] == p and k[1] == arm}
            if not c0:
                continue
            k0 = min(c0, key=lambda k: med(c0[k], lambda r: r["A_med"]))
            kb = min(c3, key=lambda k: med(c3[k], lambda r: r["A_med"]))
            x0 = med(c0[k0], lambda r: r["A_med"]); x3 = med(c3[k0], lambda r: r["A_med"]); xb = med(c3[kb], lambda r: r["A_med"])
            xb0 = x0
            cs = lambda k: "-" if k[1] == "static" else f"C{k[2]} cap{k[3] or 'none'}"
            rows.append([p, str(acc), arm, cs(k0), f"{x0:.1f}", f"{x3:.1f}", f"{x3 - x0:+.1f}", cs(kb), f"{xb:.1f}", f"{xb - xb0:+.1f}"])
fmt(rows)
print("\n## 2. best A + C (us) per point; static = ACC 0 RPF 0 (mk today), median of replicate jobs; vs static in %")
hdr = ["point", "static A + C", "queue_pf RPF3 mk combine (ACC0)", "vs", "queue_pf RPF3 + R1 red.f32", "vs", "queue_pf RPF3 + R1 red.v4", "vs", "queue_pf RPF0 + R1 red.v4"]
rows = [hdr]
S = A["a0r0"]
for p in pts:
    st = med(S[(p, "static", 0, 0, "-")], lambda r: r["AC_med"])
    row = [p, f"{st:.1f}"]
    for v in ("a0r3", "a1r3", "a3r3"):
        c = {k: x for k, x in A[v].items() if k[0] == p and k[1] == "queue_pf"}
        k = min(c, key=lambda k: med(c[k], lambda r: r["AC_med"]))
        m = med(c[k], lambda r: r["AC_med"])
        row += [f"{m:.1f} (C{k[2]} cap{k[3] or 'none'}; A {med(c[k], lambda r: r['A_med']):.1f} C {med(c[k], lambda r: r['C_med']):.2f})", f"{100 * (m / st - 1):+.1f}%"] if v != "a0r3" else [f"{m:.1f} (C{k[2]} cap{k[3] or 'none'}; A {med(c[k], lambda r: r['A_med']):.1f} C {med(c[k], lambda r: r['C_med']):.2f})", f"{100 * (m / st - 1):+.1f}%"]
    c = {k: x for k, x in A["a3r0"].items() if k[0] == p and k[1] == "queue_pf"}
    k = min(c, key=lambda k: med(c[k], lambda r: r["AC_med"]))
    row.append(f"{med(c[k], lambda r: r['AC_med']):.1f}")
    rows.append(row)
fmt(rows)
print("\n## 2b. same, A only and static + R1 (static row of each ACC build)")
hdr = ["point", "static A", "static C acc0", "static C acc1", "static C acc3", "static A+C acc3", "queue_pf RPF3 best A (acc3 build)", "vs static A"]
rows = [hdr]
for p in pts:
    sA = med(S[(p, "static", 0, 0, "-")], lambda r: r["A_med"])
    sc = [med(A[f"a{a}r0"][(p, "static", 0, 0, "-")], lambda r: r["C_med"]) for a in (0, 1, 3)]
    sac3 = med(A["a3r0"][(p, "static", 0, 0, "-")], lambda r: r["AC_med"])
    c = {k: x for k, x in A["a3r3"].items() if k[0] == p and k[1] == "queue_pf"}
    k = min(c, key=lambda k: med(c[k], lambda r: r["A_med"]))
    m = med(c[k], lambda r: r["A_med"])
    rows.append([p, f"{sA:.1f}", f"{sc[0]:.2f}", f"{sc[1]:.2f}", f"{sc[2]:.2f}", f"{sac3:.1f}", f"{m:.1f} (C{k[2]} cap{k[3] or 'none'})", f"{100 * (m / sA - 1):+.1f}%"])
fmt(rows)
print("\n## 3. replicate noise: per variant, |job1 - job2| of A for every config (median / max, us); static rows separately")
for v in sorted(reps):
    js = sorted(reps[v])
    if len(js) < 2:
        continue
    d1, d2 = table([data[(js[0], v)]]), table([data[(js[1], v)]])
    dd = [abs(d1[k][0]["A_med"] - d2[k][0]["A_med"]) for k in d1 if k in d2]
    ds = [abs(d1[k][0]["A_med"] - d2[k][0]["A_med"]) for k in d1 if k in d2 and k[1] == "static"]
    dc = [abs(d1[k][0]["C_med"] - d2[k][0]["C_med"]) for k in d1 if k in d2]
    print(f"{v}: jobs {js}  A: median {statistics.median(dd):.2f} max {max(dd):.2f}; static A: median {statistics.median(ds):.2f} max {max(ds):.2f}; C: median {statistics.median(dc):.2f} max {max(dc):.2f}")
