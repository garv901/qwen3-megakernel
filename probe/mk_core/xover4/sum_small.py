#!/usr/bin/env python3
# Summary of xover4_small.sbatch: per (shape, T) median over rounds of a0 / a3 and the best config of ts1 / ts1w / ts3 (SB 1), us per phase.
# ts1 and ts1w both print "bin":"ts" (cg 1, no "sb"): within a "## shape T" block the first line of a (TT, KS) config is ts1, the second ts1w.
import json, re, sys, statistics as st
from collections import defaultdict

runs = defaultdict(list)   # (arm, shape, T, cfg) -> [us]
fails = []
blk, seen = None, set()
for line in open(sys.argv[1]):
    m = re.match(r"## (\w+) (\d+)$", line.strip())
    if m: blk, seen = m.groups(), set(); continue
    if line.startswith(("PASS", "FAIL")) or " FAIL " in line:
        if line.startswith("FAIL"): fails.append(line.strip())
        continue
    if not line.startswith('{"bin"'): continue
    r = json.loads(line)
    if r.get("orc"): continue
    if r["bin"] in ("a0", "a3"): arm, cfg = r["bin"], ""
    elif r.get("sb") == 1: arm, cfg = "ts3sb", f'{r["TT"]}/{r["KS"]}'
    else:
        cfg = f'{r["TT"]}/{r["KS"]}'
        arm = "ts1w" if cfg in seen else "ts1"; seen.add(cfg)
    runs[(arm, r["shape"], r["T"], cfg)].append(r["us_phase"])

arms = ["a0", "a3", "ts1", "ts1w", "ts3sb"]
best = {}
for (arm, s, T, cfg), v in runs.items():
    u = st.median(v)
    k = (arm, s, T)
    if k not in best or u < best[k][0]: best[k] = (u, cfg, v)
print("| shape | T | " + " | ".join(arms) + " | best ts - a0 |")
print("|---|---|" + "---|" * (len(arms) + 1))
tot = defaultdict(lambda: defaultdict(float))
for T in sorted({k[2] for k in best}):
    for s in ("QKV", "O", "GU", "D"):
        row = []
        for a in arms:
            b = best.get((a, s, T))
            row.append(f"{b[0]:.2f}" + (f" ({b[1]})" if b and b[1] else "") if b else "-")
            if b: tot[T][a] += b[0]
        ts = min((best[(a, s, T)][0] for a in ("ts1", "ts1w") if (a, s, T) in best), default=None)
        d = f"{ts - best[('a0', s, T)][0]:+.2f}" if ts is not None and ("a0", s, T) in best else "-"
        print(f"| {s} | {T} | " + " | ".join(row) + f" | {d} |")
print()
for T, d in tot.items():
    print(f"T {T} sum of 4 phases: " + ", ".join(f"{a} {d[a]:.2f}" for a in arms if a in d))
print("\nround spread (max - min over rounds, best configs):",
      max((max(b[2]) - min(b[2]) for b in best.values() if len(b[2]) > 1), default=0))
print("FAIL lines:", len(fails)); [print(" ", f) for f in fails]
