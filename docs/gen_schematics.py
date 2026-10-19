"""Write the schematic SVG diagrams of the README (light and dark variants). Usage: gen_schematics.py OUT_DIR"""
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if __name__ == "__main__" else None

THEMES = {
    "light": dict(bg="#fcfcfb", fg="#0b0b0b", fg2="#52514e", grid="#e4e3df", card="#f3f2ef",
                  work="#2a78d6", sync="#eb6834", load="#1baf7a", tma="#6250d6", red="#e34948", on="#ffffff"),
    "dark": dict(bg="#1a1a19", fg="#f0efec", fg2="#c3c2b7", grid="#3a3936", card="#242422",
                 work="#3987e5", sync="#d95926", load="#199e70", tma="#9085e9", red="#e66767", on="#ffffff"),
}
FONT = "font-family='-apple-system,Segoe UI,Helvetica,Arial,sans-serif'"
MONO = "font-family='ui-monospace,SFMono-Regular,Menlo,Consolas,monospace'"


class Svg:
    def __init__(self, w, h, t):
        self.w, self.h, self.t, self.o = w, h, t, []

    def rect(self, x, y, w, h, fill, rx=4, stroke=None, sw=1.5, dash=None):
        s = f" stroke='{stroke}' stroke-width='{sw}'" if stroke else ""
        d = f" stroke-dasharray='{dash}'" if dash else ""
        self.o.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{rx}' fill='{fill}'{s}{d}/>")

    def text(self, x, y, s, size=13, fill=None, anchor="start", weight="normal", mono=False):
        f = MONO if mono else FONT
        self.o.append(f"<text x='{x}' y='{y}' font-size='{size}' fill='{fill or self.t['fg']}' text-anchor='{anchor}' "
                      f"font-weight='{weight}' {f}>{s}</text>")

    def line(self, x1, y1, x2, y2, stroke, sw=1.5, dash=None, arrow=False):
        d = f" stroke-dasharray='{dash}'" if dash else ""
        a = " marker-end='url(#ah)'" if arrow else ""
        self.o.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{stroke}' stroke-width='{sw}'{d}{a}/>")

    def done(self, path, title):
        t = self.t
        head = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {self.w} {self.h}' width='{self.w}' height='{self.h}' role='img'>"
                f"<title>{title}</title><defs><marker id='ah' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' "
                f"orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='{t['fg2']}'/></marker></defs>"
                f"<rect width='{self.w}' height='{self.h}' fill='{t['bg']}' rx='8'/>")
        path.write_text(head + "".join(self.o) + "</svg>\n")


def legend(s, x, y, items):
    for label, color in items:
        s.rect(x, y - 10, 12, 12, color, rx=3)
        s.text(x + 18, y, label, 12, s.t["fg2"])
        x += 26 + 7.2 * len(label)


def overview(t, name):
    s = Svg(980, 322, t)
    s.text(24, 34, "One decode step = one cooperative kernel launch", 17, weight="bold")
    s.text(24, 56, "148 CTAs x 512 threads, one CTA on each SM of the B200. Static work schedule: each CTA owns fixed rows in each phase.", 12.5, t["fg2"])
    y, h = 104, 66
    s.rect(24, y, 92, h, t["card"], stroke=t["grid"])
    s.text(70, y + 30, "prologue", 12.5, anchor="middle")
    s.text(70, y + 47, "embed row", 11, t["fg2"], anchor="middle")
    # layer block
    s.rect(132, y - 16, 620, h + 32, "none", rx=8, stroke=t["fg2"], dash="5 4", sw=1.2)
    s.text(742, y - 22, "x 28 layers", 12.5, t["fg2"], anchor="end", weight="bold")
    phases = [("Q", "RMSNorm + QKV", "4096 x 1024"), ("F", "attention + O-proj", "Wo 1024 x 2048"),
              ("G", "gate/up + SiLU", "6144 x 1024"), ("D", "down", "1024 x 3072")]
    x = 148
    for i, (p, a, b) in enumerate(phases):
        s.rect(x, y, 126, h, t["work"], rx=5)
        s.text(x + 63, y + 20, p, 14, t["on"], anchor="middle", weight="bold")
        s.text(x + 63, y + 38, a, 11, t["on"], anchor="middle")
        s.text(x + 63, y + 55, b, 10.5, t["on"], anchor="middle")
        x += 126
        s.rect(x + 4, y - 6, 6, h + 12, t["sync"], rx=2)
        x += 14
    s.rect(24 + 92 + 2, y - 6, 6, h + 12, t["sync"], rx=2)
    s.rect(770, y, 186, h, t["load"], rx=5)
    s.text(863, y + 28, "LM head + argmax", 12.5, t["on"], anchor="middle", weight="bold")
    s.text(863, y + 46, "151936 x 1024 (311 MB)", 11, t["on"], anchor="middle")
    s.line(752, y + h / 2, 768, y + h / 2, t["fg2"], arrow=True)
    yy = 222
    s.text(24, yy, "Each orange bar is a grid barrier: 1 + 4 x 28 = 113 per step.", 12.5)
    s.text(24, yy + 20, "A barrier is needed where a phase reads data that all 148 CTAs wrote (RMSNorm needs the full x, each down row needs all of hid).", 12.5, t["fg2"])
    s.text(24, yy + 40, "Activations stay in fp32. Weights are bf16. The weight bytes per step are 1.21 GB, so the HBM floor at 8 TB/s is about 151 us.", 12.5, t["fg2"])
    legend(s, 24, 302, [("phase work", t["work"]), ("grid barrier", t["sync"]), ("LM head (bandwidth bound)", t["load"])])
    s.done(OUT / f"{name}-{t_name}.svg", "Decode step overview")


def fused(t, name):
    s = Svg(1070, 430, t)
    s.text(24, 34, "Attention and O-proj in one phase", 17, weight="bold")
    s.text(24, 56, "144 of the 148 CTAs do fused work: 16 heads x 9 row slices of Wo. Each CTA computes the attention of its head again.", 12.5, t["fg2"])
    # left: conventional
    x0, y0 = 24, 86
    s.text(x0, y0, "Two phases (not used)", 13.5, weight="bold")
    s.rect(x0, y0 + 14, 280, 40, t["work"], rx=5)
    s.text(x0 + 140, y0 + 39, "attention: 16 heads -> o (2048)", 12, t["on"], anchor="middle")
    s.rect(x0, y0 + 60, 280, 8, t["sync"], rx=2)
    s.text(x0 + 286, y0 + 68, "grid barrier", 11, t["fg2"])
    s.rect(x0, y0 + 74, 280, 40, t["work"], rx=5)
    s.text(x0 + 140, y0 + 99, "O-proj: x += Wo . o", 12, t["on"], anchor="middle")
    s.rect(x0, y0 + 120, 280, 8, t["sync"], rx=2)
    s.text(x0, y0 + 152, "5 grid barriers per layer.", 12, t["fg2"])
    s.text(x0, y0 + 170, "o makes a round trip through L2.", 12, t["fg2"])
    # right: fused grid
    gx, gy, cw, ch = 430, 104, 24, 22
    s.text(gx, y0, "One fused phase (this kernel)", 13.5, weight="bold")
    s.text(gx + 16 * cw / 2, gy - 4, "head h = 0 .. 15  (128 Wo columns each)", 11, t["fg2"], anchor="middle")
    for si in range(9):
        for h in range(16):
            hl = (si == 3 and h == 5)
            s.rect(gx + h * cw + 1, gy + 4 + si * ch + 1, cw - 2, ch - 2, t["red"] if hl else t["work"], rx=3)
    s.text(gx - 8, gy + 4 + 4.5 * ch + 4, "row slice s", 11, t["fg2"], anchor="end")
    s.text(gx - 8, gy + 4 + 4.5 * ch + 18, "0 .. 8", 11, t["fg2"], anchor="end")
    s.text(gx + 16 * cw / 2, gy + 4 + 9 * ch + 18, "Wo (1024 x 2048) cut into 144 tiles = 144 CTAs", 11.5, t["fg2"], anchor="middle")
    # callout for the highlighted CTA
    cx = gx + 16 * cw + 20
    s.rect(cx, gy, 1070 - 20 - cx, 214, t["card"], rx=6, stroke=t["red"], sw=1.5)
    s.text(cx + 10, gy + 20, "CTA (h=5, s=3)", 12.5, weight="bold")
    steps = ["1. read q of head 5 and new k, v", "   of kv head 2 (written in Q)", "2. attention over 129 positions", "   (also done by the 8 other CTAs", "   of head 5)",
             "3. partial = Wo[slice s, head h] . o", "   (112 or 116 rows x 128 columns)", "4. red.add partial into x", "   (fp32, in L2)"]
    for i, l in enumerate(steps):
        s.text(cx + 10, gy + 42 + i * 18, l, 11, t["fg"] if not l.startswith("   ") else t["fg2"])
    s.line(gx + 6 * cw - 2, gy + 4 + 3.5 * ch, cx - 2, gy + 60, t["red"], 1.2, dash="3 3")
    yy = 342
    s.text(24, yy, "Result: 4 grid barriers per layer, not 5 (28 fewer per step). The O-proj split-K adds no weight bytes.", 12.5, weight="bold")
    s.text(24, yy + 20, "The cost: 9 copies of the attention per head. The F-phase oracles show that this compute is mostly off the critical path:", 12.5, t["fg2"])
    s.text(24, yy + 40, "with all attention compute removed, the step is only 3.6 us faster.", 12.5, t["fg2"])
    legend(s, 24, 414, [("CTA tile", t["work"]), ("example CTA", t["red"]), ("grid barrier", t["sync"])])
    s.done(OUT / f"{name}-{t_name}.svg", "Fused attention and O-proj phase")


def barrier(t, name):
    s = Svg(980, 470, t)
    s.text(24, 34, "Split-phase grid barrier: arrive, overlap, wait", 17, weight="bold")
    s.text(24, 56, "One CTA, one hop from phase N to phase N+1. Not to scale.", 12.5, t["fg2"])
    lx, x0, x1 = 24, 190, 956
    lanes = [("warp 0 (thread 0)", 96), ("warps 1-15", 156), ("TMA engine", 216)]
    for lab, y in lanes:
        s.text(lx, y + 18, lab, 12.5, weight="bold")
        s.line(x0, y + 34, x1, y + 34, t["grid"], 1)
    # phase N work
    for _, y in lanes[:2]:
        s.rect(x0, y, 120, 28, t["work"], rx=4)
        s.text(x0 + 60, y + 19, "phase N work", 11.5, t["on"], anchor="middle")
    # arrive sync
    ax = x0 + 126
    s.rect(ax, 90, 7, 100, t["sync"], rx=2)
    s.text(ax + 3, 82, "__syncthreads", 10.5, t["fg2"], anchor="middle")
    # thread 0: fence + red
    y = 96
    s.rect(ax + 14, y, 92, 28, t["sync"], rx=4)
    s.text(ax + 60, y + 18, "fence + red", 11.5, t["on"], anchor="middle")
    s.rect(ax + 112, y, 62, 28, t["card"], rx=4, stroke=t["sync"], sw=1.2)
    s.text(ax + 143, y + 18, "spin", 11.5, anchor="middle")
    s.rect(ax + 180, y, 270, 28, t["card"], rx=4, stroke=t["sync"], sw=1.2, dash="4 3")
    s.text(ax + 315, y + 18, "relaxed poll: ld.relaxed.gpu counter (L2)", 11.5, anchor="middle")
    s.rect(ax + 456, y, 74, 28, t["sync"], rx=4)
    s.text(ax + 493, y + 18, "1 acquire", 11.5, t["on"], anchor="middle")
    # warps 1-15: window loads
    y = 156
    s.rect(ax + 14, y, 360, 28, t["load"], rx=4)
    s.text(ax + 194, y + 18, "window loads: phase N+1 weights (and warp 0's share)", 11.5, t["on"], anchor="middle")
    s.rect(ax + 380, y, 150, 28, t["card"], rx=4, stroke=t["grid"])
    s.text(ax + 455, y + 18, "wait", 11.5, t["fg2"], anchor="middle")
    # TMA lane
    y = 216
    s.rect(x0 + 40, y, 380, 28, t["tma"], rx=4)
    s.text(x0 + 230, y + 18, "TMA bulk copy into smem (issued in phase N, after its critical read)", 11.5, t["on"], anchor="middle")
    # wait sync + phase N+1
    wx = ax + 536
    s.rect(wx, 90, 7, 160, t["sync"], rx=2)
    s.text(wx + 3, 82, "__syncthreads", 10.5, t["fg2"], anchor="middle")
    for _, y in lanes[:2]:
        s.rect(wx + 14, y, x1 - wx - 14, 28, t["work"], rx=4)
        s.text(wx + 14 + (x1 - wx - 14) / 2, y + 19, "phase N+1", 11.5, t["on"], anchor="middle")
    s.text(ax + 60, 274, "arrive", 12, t["fg2"], anchor="middle", weight="bold")
    s.text(ax + 290, 274, "overlap window: bytes move while thread 0 waits", 12, t["fg2"], anchor="middle", weight="bold")
    s.text(wx + 3, 274, "wait done", 12, t["fg2"], anchor="middle", weight="bold")
    # SASS inset
    y = 300
    s.text(24, y, "Poll loop in SASS (sm_100)", 13, weight="bold")
    s.rect(24, y + 10, 450, 118, t["card"], rx=6)
    s.text(36, y + 32, "before: acquire load in the loop", 12, t["fg2"])
    for i, l in enumerate(["L: LDG.E.STRONG.GPU R4, [counter]", "   CCTL.IVALL        // every iteration", "   ISETP / @!P0 BRA L"]):
        s.text(36, y + 54 + i * 20, l, 12, mono=True)
    s.rect(500, y + 10, 456, 118, t["card"], rx=6)
    s.text(512, y + 32, "hv2: relaxed loop, one acquire at the end", 12, t["fg2"])
    for i, l in enumerate(["L: LDG.E.STRONG.GPU R4, [counter]", "   ISETP / @!P0 BRA L", "   CCTL.IVALL        // once"]):
        s.text(512, y + 54 + i * 20, l, 12, mono=True)
    legend(s, 24, 456, [("work", t["work"]), ("barrier protocol", t["sync"]), ("weight loads", t["load"]), ("TMA copy", t["tma"])])
    s.done(OUT / f"{name}-{t_name}.svg", "Split-phase grid barrier")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for t_name, t in THEMES.items():
        overview(t, "overview")
        fused(t, "fused-attn-oproj")
        barrier(t, "split-barrier")
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
