"""Per-step dilution: why a shorter mk forward gives a smaller step / TPOT gain inside vLLM, and what skipping vLLM's
unused attention-metadata build recovers.

Source: runs/hostcost (probe/mk_core/hostcost/hc.sbatch, jobs 14160 / 14161; harness regime-mk-serve-host = mk-serve
fe5725b + host timers, same mk build as the M2 grid). Per stage, one job runs stock, mk (metadata built, as in M2) and
mk with --skip-attn-meta (tag suffix _skip). T_L1 / T_L2: Azure 2024 conv median-window trace at 10 / 40 req/s;
C_c1 / C_c8: random 128 in / 256 out, closed loop at concurrency 1 / 8.
Decode steps only (labels D1-*, DB-*); mk arms keep only steps that ran on mk. For step N with successor N+1:
forward = fwd_us(N), step time = fwd_us(N) + gap_us(N+1) (gap = GPU time between the two forwards: sampler, input
copies, and GPU idle while the host finishes step N+1). Gaps >= 5 ms (no work) are dropped.
Gains are matched per step label (request bucket x KV bucket), median mk / stock, weighted by stock's time on the label.
Run: <venv>/bin/python results/2026-10-29-plots/make_step_dilution.py
"""
import glob
import json
import os
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "runs", "hostcost")
IDLE_US = 5000.0
INK, INK2, MUTED, GRID, SURF = "#1f1f1e", "#52514c", "#8a897f", "#e6e5e0", "#fcfcfb"
C_VL, C_MK, C_SK = "#1baf7a", "#6da7ec", "#104281"
STAGES = [("T_L1", "Azure trace\n10 req/s"), ("T_L2", "Azure trace\n40 req/s"), ("C_c1", "closed loop\n1 request"),
          ("C_c8", "closed loop\n8 requests")]


def run_dir(tag):
    return sorted(glob.glob(os.path.join(HC, f"*-{tag}")))[-1]


def steps(path, mk_only):
    """-> list of (label, fwd_us, step_us)."""
    rs = sorted((json.loads(l) for l in open(path)), key=lambda r: r["step"])
    rs = [r for r in rs if r.get("fwd_us") is not None]
    out = []
    for a, b in zip(rs, rs[1:]):
        g = b.get("gap_us")
        if (b["step"] != a["step"] + 1 or g is None or g >= IDLE_US or not a["label"].startswith(("D1", "DB"))
                or (mk_only and a["fwd"] == "stock")):
            continue
        out.append((a["label"], a["fwd_us"], a["fwd_us"] + g))
    return out


def arms(stage):
    d, ds = run_dir(stage), run_dir(stage + "_skip")
    return {"stock": (os.path.join(d, "stock"), False), "mk": (os.path.join(d, "mk"), True),
            "skip": (os.path.join(ds, "mk"), True)}


def matched(stock, mk):
    """Shape-matched (forward gain %, step gain %)."""
    S, M = defaultdict(lambda: ([], [])), defaultdict(lambda: ([], []))
    for src, dst in ((stock, S), (mk, M)):
        for lab, f, p in src:
            dst[lab][0].append(f)
            dst[lab][1].append(p)
    nf = df = ns = ds = 0.0
    for lab, (sf, sp) in S.items():
        if len(M[lab][0]) < 50:
            continue
        nf += sum(sf) * st.median(M[lab][0]) / st.median(sf)
        df += sum(sf)
        ns += sum(sp) * st.median(M[lab][1]) / st.median(sp)
        ds += sum(sp)
    return (1 - nf / df) * 100, (1 - ns / ds) * 100


def tpot(path):
    return json.load(open(os.path.join(path, "bench.json")))["mean_tpot_ms"]


data, ladder = defaultdict(list), []
for stage, _ in STAGES:
    a = arms(stage)
    s = {k: steps(os.path.join(p, "steps.jsonl"), m) for k, (p, m) in a.items()}
    if stage.startswith("T_"):
        for k in s:
            data[k] += s[k]
    fwd, step_mk = matched(s["stock"], s["mk"])
    _, step_sk = matched(s["stock"], s["skip"])
    t0 = tpot(a["stock"][0])
    row = (fwd, step_mk, step_sk, (1 - tpot(a["skip"][0]) / t0) * 100, (1 - tpot(a["mk"][0]) / t0) * 100)
    ladder.append(row)
    print(stage, "fwd %.1f  step(meta built) %.1f  step(skip) %.1f  TPOT(skip) %.1f  TPOT(meta built) %.1f" % row)

g0 = st.median(p - f for _, f, p in data["stock"])
floor = {k: st.median(p for _, f, p in data[k] if f < 950) for k in ("mk", "skip")}
print(f"stock gap {g0:.0f} us; step-time floor at forward < 950 us: mk {floor['mk']:.0f}, mk+skip {floor['skip']:.0f}")

BIN = 50.0


def binned(rows, min_n=300):
    b = defaultdict(list)
    for _, f, p in rows:
        b[int(f // BIN)].append(p)
    xs, med, lo, hi = [], [], [], []
    for k in sorted(b):
        v = sorted(b[k])
        if len(v) >= min_n:
            xs.append((k + 0.5) * BIN)
            med.append(v[len(v) // 2])
            lo.append(v[len(v) // 4])
            hi.append(v[3 * len(v) // 4])
    return xs, med, lo, hi


plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 6.4), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.2})

# ---- A
lo_x, hi_x = 750, 2300
for k, color, name in (("stock", C_VL, "stock vLLM"), ("mk", C_MK, "mk, vLLM still builds attention metadata"),
                       ("skip", C_SK, "mk, metadata build skipped")):
    xs, med, lo, hi = binned(data[k])
    ax.fill_between(xs, lo, hi, color=color, alpha=0.13, lw=0)
    ax.plot(xs, med, "-o", color=color, lw=2, ms=4.5, mec=SURF, mew=1, label=name)
x = [lo_x, hi_x]
ax.plot(x, [v + g0 for v in x], color=INK2, lw=1, ls=(0, (4, 3)))
ax.annotate(f"GPU-bound: step = forward + {g0:.0f} us\n(sampler, input copies)", (1750, 1750 + g0), xytext=(0, -75),
            textcoords="offset points", ha="left", fontsize=8.5, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
for k, dy, txt in (("mk", 8, "CPU floor ~{:.0f} us"), ("skip", -16, "CPU floor ~{:.0f} us without the metadata build")):
    ax.axhline(floor[k], xmax=0.3, color=C_MK if k == "mk" else C_SK, lw=1, ls=(0, (2, 2)))
    ax.annotate(txt.format(floor[k]), (lo_x + 10, floor[k]), xytext=(0, dy), textcoords="offset points",
                fontsize=8.5, color=INK2)
ax.set_xlim(lo_x, hi_x)
ax.set_ylim(900, 2700)
ax.set_xlabel("GPU forward time of step N (us)")
ax.set_ylabel("step time = forward + gap to step N+1 (us)")
ax.set_title("A. Step time vs forward time, every decode step (Azure trace)", loc="left", fontsize=11, color=INK,
             pad=10)
ax.grid(True, color=GRID, lw=0.8)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.text(0.98, 0.03, "Below the floor the GPU waits for the CPU:\na shorter forward no longer shortens the step.",
        transform=ax.transAxes, ha="right", fontsize=8.5, color=INK2)

# ---- B
cols = ["#104281", "#b7d3f6", "#2a78d6", "#6da7ec"]
names = ["forward (GPU)", "step, vLLM still builds metadata", "step, metadata build skipped",
         "TPOT, metadata build skipped"]
W = 0.2
for i, row in enumerate(ladder):
    for k, v in enumerate(row[:4]):
        xx = i + (k - 1.5) * (W + 0.01)
        bx.bar(xx, v, W, color=cols[k], edgecolor=SURF, lw=1.5, label=names[k] if i == 0 else None)
        bx.annotate(f"{v:.0f}%", (xx, v), xytext=(0, 3 if v >= 0 else -12), textcoords="offset points", ha="center",
                    fontsize=8, color=INK, fontweight="bold" if k in (0, 3) else "normal")
bx.axhline(0, color=INK2, lw=1)
bx.set_xticks(range(len(STAGES)))
bx.set_xticklabels([s[1] for s in STAGES], fontsize=9)
bx.tick_params(axis="x", length=0, pad=8)
bx.set_ylim(-14, 42)
bx.set_ylabel("time saved by mk vs stock vLLM (%)")
bx.set_title("B. Gain at each level, same job per stage", loc="left", fontsize=11, color=INK, pad=10)
bx.grid(True, axis="y", color=GRID, lw=0.8)
bx.set_axisbelow(True)
bx.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=1)

fig.suptitle("Why mk's forward gain shrinks once the whole step is counted", x=0.055, ha="left", fontsize=14,
             color=INK, fontweight="bold", y=0.985)
fig.text(0.055, 0.905, "vLLM prepares step N+1 on the CPU while the GPU runs step N (async scheduling). A stock "
         "forward is long enough to hide that work; a shorter mk forward\nis not, and the GPU waits. On mk steps vLLM "
         "also builds FlashInfer attention metadata (~100 us CPU) that mk never reads; skipping it lowers the floor.",
         fontsize=9.5, color=INK2)
fig.text(0.055, 0.015, "Data: runs/hostcost, jobs 14160 / 14161 (vLLM 0.31, Qwen3-0.6B, B200, mk build of the M2 grid). "
         "Panel A pools the 10 and 40 req/s trace runs; band = 25th-75th percentile per 50 us forward bin.\nPanel B: "
         "forward and step gains matched per step shape (request bucket x KV bucket), weighted by stock's time; TPOT = "
         "client mean time per output token.", fontsize=7.5, color=MUTED)
fig.subplots_adjust(left=0.055, right=0.985, top=0.81, bottom=0.15)
out = os.path.join(HERE, "07_step_dilution.png")
fig.savefig(out, dpi=150)
print(out)
