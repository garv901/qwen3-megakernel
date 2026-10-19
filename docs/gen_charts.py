"""Write the data charts of the README (light and dark variants). Usage: gen_charts.py OUT_DIR

The values are the measured medians that the README tables give. Edit both places together.
"""
import sys
from pathlib import Path

from gen_schematics import FONT, THEMES, Svg

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

HBM_FLOOR_US = 151

# (label, change, isolated latency in us: mean of the two runs of the A/B test)
PATH = [
    ("Start", "fused F + split barriers", 484.0),
    ("1", "relaxed poll, load-free warp 0, spin, argmax", 455.0),
    ("2", "activations in registers", 439.5),
    ("3", "constants in barrier windows", 419.5),
    ("4", "one-sync RMSNorm, gate/up by TMA", 401.0),
    ("5", "earlier TMA issue", 393.1),
    ("6", "__expf", 393.0),
    ("7", "host: outputs after launch", 383.05),
]

# (system, setup, steady time per step in us); vLLM uses a different method (see the README)
RESULTS = [
    ("This kernel", "one persistent launch", 373.6),
    ("FlashInfer", "CUDA graph + PDL", 877.0),
    ("torch.compile", "max-autotune, CUDA graph", 1111.3),
    ("cuBLAS", "CUDA graph + PDL", 1118.0),
    ("vLLM 0.31", "decode slope, other method", 1161.6),
]


def hbars(t, name, title, sub, rows, xmax, ticks, floor=None, highlight=None, sub_x=74, lw=300):
    """Horizontal bars: rows = [(label, sublabel, value)]. One series, so no legend."""
    rw, top, bh, gap = 560, 84, 22, 12
    w = lw + rw + 90
    h = top + len(rows) * (bh + gap) + 40
    s = Svg(w, h, t)
    s.text(24, 32, title, 16, weight="bold")
    s.text(24, 54, sub, 12, t["fg2"])
    x0 = lw
    sx = lambda v: x0 + rw * v / xmax
    for v in ticks:
        s.line(sx(v), top - 8, sx(v), h - 34, t["grid"], sw=1)
        s.text(sx(v), h - 18, f"{v:g}", 11, t["fg2"], anchor="middle")
    s.text(x0 + rw, h - 4, "us", 11, t["fg2"], anchor="end")
    for i, (lab, sub_, v) in enumerate(rows):
        y = top + i * (bh + gap)
        s.text(24, y + 15, lab, 12.5, weight="bold" if lab == highlight else "normal")
        if sub_:
            s.text(sub_x, y + 15, sub_, 11.5, t["fg2"])
        color = t["work"] if lab == highlight or highlight is None else t["fg2"]
        bw = sx(v) - x0
        s.o.append(f"<path d='M{x0},{y} h{bw - 4} a4,4 0 0 1 4,4 v{bh - 8} a4,4 0 0 1 -4,4 h{-(bw - 4)} z' fill='{color}'/>")
        s.text(sx(v) + 6, y + 15, f"{v:.0f}", 12, t["fg"])
    if floor:
        s.line(sx(floor), top - 14, sx(floor), h - 34, t["sync"], sw=2, dash="5 4")
        s.text(sx(floor) + 6, top - 16, f"HBM floor {floor} us", 11.5, t["fg2"])
    s.done(OUT / f"{name}-{t['_n']}.svg", title)


for n, t in THEMES.items():
    t = dict(t, _n=n)
    hbars(t, "path", "Isolated latency of each kernel version",
          "Median of the A/B test. Lower is better.",
          [(a, b, v) for a, b, v in PATH], 500, [0, 100, 200, 300, 400, 500], floor=HBM_FLOOR_US, highlight="7")
    if RESULTS:
        hbars(t, "results", "Decode step time of Qwen3-0.6B, batch 1, KV cache 128, B200",
              "Median steady time for each step. Lower is better.",
              RESULTS, 1300, [0, 250, 500, 750, 1000, 1250], floor=HBM_FLOOR_US, highlight=RESULTS[0][0], sub_x=130, lw=330)

print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
