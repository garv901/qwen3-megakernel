"""kvq_alt tables: $PY analyze_alt.py [--md] [--all] ; reads results/2026-10-28-kvq/kvq_11854.jsonl (baseline) and results/2026-10-28-kvq-alt/kvq_<job>_<mode>.jsonl (newest job per mode).
Per point and design: best config by A + C over the design's arms (queue_pf / queue / hybrid, pos order; R3 pos and pair separately; R2 = arm tail), plus the static row of each build. C = the combine (slots) or normalize (R1) phase or the barrier (R3)."""
import glob, json, re, sys
R = "results/"
md = "--md" in sys.argv


def load(p):
    return [json.loads(l) for l in open(p) if '"A_med"' in l]


def newest(mode):
    fs = sorted(glob.glob(f"{R}2026-10-28-kvq-alt/kvq_*_{mode}.jsonl"), key=lambda f: int(re.search(r"kvq_(\d+)_", f).group(1)))
    return load(fs[-1]) if fs else []


base = [r for r in load(R + "2026-10-28-kvq/kvq_11854.jsonl") if not r.get("ticket")]
tick11854 = [r for r in load(R + "2026-10-28-kvq/kvq_11854_ticket.jsonl")]
sets = {"base": newest("base"), "acc1": newest("acc1"), "acc2": newest("acc2"), "acc3": newest("acc3"), "acc4": newest("acc4"), "tail": newest("tail"), "mq": newest("mq")}
pts = list(dict.fromkeys(r["point"] for r in base))


def best(rows, pt, arms=("queue_pf", "queue", "hybrid"), order=("pos", "-")):
    c = [r for r in rows if r["point"] == pt and r["arm"] in arms and r["order"] in order]
    return min(c, key=lambda r: r["AC_med"]) if c else None


def cfg(r):
    if r is None:
        return "-"
    if r["arm"] == "static":
        return "static"
    if r["arm"] == "tail":
        return f"tail f{int(round(100 * r.get('tail_f', 0)))}% k{r['C']}"
    return f"{r['arm']} C{r['C']} cap{r.get('cap') or 'none'}" + ("/pair" if r["order"] == "pair" else "")


designs = [("static (11854)", lambda p: best(base, p, ("static",), ("-",))),
           ("best queue 11854", lambda p: best(base, p)),
           ("11854 ticket (queue_pf cap32)", lambda p: best(tick11854, p, ("queue_pf", "queue"))),
           ("control: queue same job (slots)", lambda p: best(sets["base"], p)),
           ("static + R1 red.f32", lambda p: best(sets["acc1"], p, ("static",), ("-",))),
           ("R1 red.f32", lambda p: best(sets["acc1"], p)),
           ("R1 red.v4", lambda p: best(sets["acc3"], p)),
           ("static + R1 TMA-reduce", lambda p: best(sets["acc4"], p, ("static",), ("-",))),
           ("R1 TMA-reduce", lambda p: best(sets["acc4"], p)),
           ("R2 tail split", lambda p: best(sets["tail"], p, ("tail",))),
           ("R3 merge queue pos", lambda p: best(sets["mq"], p, ("queue_pf", "queue"), ("pos",))),
           ("R3 merge queue pair", lambda p: best(sets["mq"], p, ("queue_pf", "queue"), ("pair",))),
           ("static + R3", lambda p: best(sets["mq"], p, ("static",), ("-",)))]
hdr = ["point", "design", "config", "A", "C", "A + C", "vs static A + C", "S_max"]
out = [hdr]
for p in pts:
    st = designs[0][1](p)
    for name, f in designs:
        r = f(p)
        if r is None:
            continue
        out.append([p, name, cfg(r), f"{r['A_med']:.1f}", f"{r['C_med']:.2f}", f"{r['AC_med']:.1f}", f"{100 * (r['AC_med'] / st['AC_med'] - 1):+.1f}%", str(r["s_max_dev"])])
for l in out:
    print(("| " + " | ".join(l) + " |") if md else "  ".join(f"{x:>26s}" for x in l))
    if md and l is hdr:
        print("|" + "---|" * len(hdr))
if "--all" in sys.argv:
    for mode in ("acc1", "acc3", "acc4", "base", "tail", "mq"):
        print(f"\n## all configs {mode}")
        for r in sets[mode]:
            print(f"{r['point']:12s} {cfg(r):28s} {r['order']:4s} A {r['A_med']:7.1f} C {r['C_med']:5.2f} A+C {r['AC_med']:7.1f} S {r['s_max_dev']}")
