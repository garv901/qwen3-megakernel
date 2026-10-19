"""Write the trace charts of the README from docs/data/iket_trace.json (light and dark variants). Usage: gen_trace_charts.py OUT_DIR

layer-timeline: one layer, one row for each CTA. A bar is the work of the CTA in a phase (barrier exit to the end of the arrive sync).
layer-breakdown: for each phase, the median work, the arrive skew and the barrier hop.
"""
import json
import sys
from pathlib import Path

from gen_schematics import THEMES, Svg, legend

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
D = json.loads((Path(__file__).parent / "data" / "iket_trace.json").read_text())
PH = "QFGD"
SKEW = {"light": "#1baf7a", "dark": "#199e70"}  # categorical slot 3 (aqua)


def timeline(t, n):
    ctas = D["ctas"]
    rh, top, left = 2.4, 92, 64
    t_end = max(c["next_q_bwx"] for c in ctas)
    t_beg = min(min(c["bwx"]) for c in ctas)
    w_plot = 860
    sx = lambda v: left + w_plot * (v - t_beg) / (t_end - t_beg)
    h = top + rh * len(ctas) + 70
    s = Svg(left + w_plot + 40, h, t)
    s.text(24, 32, f"One layer (layer {D['layer']}) in the trace: 148 CTAs", 16, weight="bold")
    s.text(24, 54, "Each row is one CTA. A blue bar is phase work: barrier exit to the end of the arrive sync. "
                   "An orange line is the last arrive of the phase.", 12, t["fg2"])
    for us in range(0, int((t_end - t_beg) / 1000) + 1, 2):
        x = sx(t_beg + us * 1000)
        s.line(x, top - 6, x, top + rh * len(ctas), t["grid"], sw=1)
        s.text(x, top + rh * len(ctas) + 18, f"{us}", 11, t["fg2"], anchor="middle")
    s.text(left + w_plot, top + rh * len(ctas) + 34, "us from the first exit into phase Q", 11, t["fg2"], anchor="end")
    for i, c in enumerate(ctas):
        y = top + i * rh
        for p in range(4):
            a, b = sx(c["bwx"][p]), sx(c["sal"][p])
            s.rect(round(a, 1), round(y, 1), round(max(b - a, 0.8), 1), rh - 0.6, t["work"], rx=0)
    s.text(left - 8, top + 8, "CTA 0", 10.5, t["fg2"], anchor="end")
    s.text(left - 8, top + rh * len(ctas), "CTA 147", 10.5, t["fg2"], anchor="end")
    for p in range(4):
        last = max(c["sal"][p] for c in ctas)
        x = sx(last)
        s.line(x, top - 14, x, top + rh * len(ctas), t["sync"], sw=1.6)
        x0 = sx(sorted(c["bwx"][p] for c in ctas)[len(ctas) // 2])
        s.text((x0 + x) / 2, top - 18, PH[p], 13, anchor="middle", weight="bold")
    s.done(OUT / f"layer-timeline-{n}.svg", "Per-CTA trace of one layer")


def breakdown(t, n):
    b = D["breakdown"]["phases"]
    left, top, bh, gap, w_plot = 60, 92, 26, 16, 760
    tot = max(v["work"] + v["skew"] + v["hop"] for v in b.values())
    sx = lambda v: w_plot * v / (tot * 1.08)
    h = top + 4 * (bh + gap) + 44
    s = Svg(left + w_plot + 120, h, t)
    s.text(24, 32, "Where the time of a layer goes, for each phase", 16, weight="bold")
    s.text(24, 54, "Median over 148 CTAs and 28 layers, ns. Work = exit to arrive end. Skew = last arrive minus median arrive. "
                   "Hop = last arrive to the next exit.", 12, t["fg2"])
    cols = [("work", t["work"]), ("skew", SKEW[n]), ("hop", t["sync"])]
    for i, ph in enumerate(PH):
        y = top + i * (bh + gap)
        s.text(24, y + 17, ph, 13, weight="bold")
        x = left
        for k, (key, col) in enumerate(cols):
            wv = sx(b[ph][key])
            s.rect(round(x, 1), y, round(max(wv - 2, 1), 1), bh, col, rx=3)
            if key != "skew":
                s.text(x + wv / 2, y + 17, f"{b[ph][key]:.0f}", 11.5, t["on"], anchor="middle")
            x += wv
        s.text(x + 6, y + 17, f"{b[ph]['work'] + b[ph]['skew'] + b[ph]['hop']:.0f}", 12, t["fg"])
    legend(s, 24, h - 14, [("work", t["work"]), ("skew", SKEW[n]), ("hop", t["sync"])])
    s.done(OUT / f"layer-breakdown-{n}.svg", "Work, skew and hop for each phase")


for n, t in THEMES.items():
    timeline(t, n)
    breakdown(t, n)
print("wrote", sorted(p.name for p in OUT.glob("layer-*.svg")))
