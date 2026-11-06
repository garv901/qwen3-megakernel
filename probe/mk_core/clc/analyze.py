#!/usr/bin/env python3
"""clc: summarize a clc JSONL (medians over reps). Usage: $PY analyze.py results/2026-10-27-clc/clc_<job>.jsonl"""
import json, sys, statistics as st
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
order = ["static", "static_il", "queue", "queue_pf", "guided", "hw", "clc", "clc_np"]
g = {}
for r in rows: g.setdefault((r["arm"], r["C"]), []).append(r)
def m(rs, k): return st.median(r[k] for r in rs)
def cs(c): return "guided" if c == 0 and False else (f"{c/1048576:.2f}M" if c >= 1048576 else f"{c//1024}K")
print(f"{'arm':9} {'C':>7} {'make us':>8} {'tail us':>8} {'sprd us':>8} {'GB/s':>7} {'ev GB/s':>8} {'B/SM min-max (MB)':>19} {'ovh ns':>7} {'n':>3}")
for (arm, c), rs in sorted(g.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1])):
    lab = "list" if arm == "guided" else cs(c)
    print(f"{arm:9} {lab:>7} {m(rs,'makespan_ns')/1e3:8.1f} {m(rs,'tail_ns')/1e3:8.2f} {m(rs,'spread_ns')/1e3:8.2f} {m(rs,'bw_gbs'):7.0f} {m(rs,'ev_gbs'):8.0f} "
          f"{m(rs,'bytes_min')/1e6:8.2f}-{m(rs,'bytes_max')/1e6:<8.2f} {m(rs,'ovh_ns'):7.0f} {len(rs):3d}")
