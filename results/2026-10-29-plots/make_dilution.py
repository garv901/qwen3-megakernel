"""Fair dilution ladder: mk vs stock vLLM inside the same vLLM engine, same Azure conv requests, same load.

Source: regime-mk-serve M2 replays (runs/e2e/<UTC>-m2_conv_<window>_L<n>/{stock,mk}), 3 windows pooled.
Step shapes are matched by the step label (request bucket + KV bucket, e.g. DB-9-32-k64); each shape is weighted by
the time stock spent on it, so batching differences between the two runs do not enter levels 1-3.
  1. forward, mk-eligible steps : forward-time gain over the shapes mk runs
  2. forward, all steps         : same, over every shape (shapes mk does not run count with their measured ratio)
  3. GPU timeline, all steps    : forward + gap to the next step (gap < 5 ms: sampler, input copies, host wait)
  4. TPOT                       : client mean time per output token (bench.json), mean of the 3 window ratios
Run: python3 results/2026-10-29-plots/make_dilution.py   (needs matplotlib: use the vibe-megakernel venv)
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
M2 = "/raid/garv901/regime-mk-serve/runs/e2e"
IDLE_US = 5000.0
INK, INK2, MUTED, GRID, SURF = "#1f1f1e", "#52514c", "#8a897f", "#e6e5e0", "#fcfcfb"
LEVELS = ["Forward pass,\nsteps mk runs", "Forward pass,\nall steps", "GPU timeline,\n+ gaps between steps",
          "Time per output token\n(user-visible)"]
COLORS = ["#104281", "#2a78d6", "#6da7ec", "#b7d3f6"]  # one hue, dark -> light as the measurement widens
LOADS = {1: "10 req/s", 2: "40 req/s", 3: "80 req/s", 4: "100 req/s"}


def shapes(path):
    """label -> (fwd list, period list, fwd path counter)."""
    fwd, per, path_n = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(int))
    for line in open(path):
        r = json.loads(line)
        if r.get("fwd_us") is None:
            continue
        lab = r["label"]
        fwd[lab].append(r["fwd_us"])
        g = r.get("gap_us")
        if g is not None and g < IDLE_US:
            per[lab].append(r["fwd_us"] + g)
        path_n[lab][r["fwd"]] += 1
    return fwd, per, path_n


def load_point(load):
    runs = sorted(glob.glob(os.path.join(M2, f"*-m2_conv_*_L{load}")))
    runs = [d for d in runs if any(w in d for w in ("_median_", "_peak_", "_trough_"))]
    acc = {a: (defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(int))) for a in ("stock", "mk")}
    tpot = []
    for d in runs:
        for a in ("stock", "mk"):
            f, p, n = shapes(os.path.join(d, a, "steps.jsonl"))
            for lab in f:
                acc[a][0][lab] += f[lab]
                acc[a][1][lab] += p[lab]
                for k, v in n[lab].items():
                    acc[a][2][lab][k] += v
        b = {a: json.load(open(os.path.join(d, a, "bench.json")))["mean_tpot_ms"] for a in ("stock", "mk")}
        tpot.append(b["mk"] / b["stock"])
    (sf, sp, _), (mf, mp, mn) = acc["stock"], acc["mk"]
    num = defaultdict(float)
    den = defaultdict(float)
    cov_t = tot_t = 0.0
    for lab in sf:
        w_f, w_p = sum(sf[lab]), sum(sp[lab])
        on_mk = mn[lab].get("mk", 0) > sum(mn[lab].values()) / 2
        rf = st.median(mf[lab]) / st.median(sf[lab]) if mf.get(lab) else 1.0
        rp = st.median(mp[lab]) / st.median(sp[lab]) if mp.get(lab) and sp.get(lab) else 1.0
        tot_t += w_f
        if on_mk:
            cov_t += w_f
            num[1] += w_f * rf
            den[1] += w_f
        num[2] += w_f * rf
        den[2] += w_f
        num[3] += w_p * rp
        den[3] += w_p
    gains = [(1 - num[k] / den[k]) * 100 if den[k] else float("nan") for k in (1, 2, 3)]
    gains.append((1 - st.mean(tpot)) * 100)
    return gains, cov_t / tot_t * 100, len(runs)


data = {l: load_point(l) for l in LOADS}
for l, (g, cov, n) in data.items():
    print(f"L{l}: windows={n} coverage={cov:.1f}% gains=" + ", ".join(f"{x:.1f}" for x in g))

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})
fig, ax = plt.subplots(figsize=(11, 6.4))
W = 0.19
for i, l in enumerate(LOADS):
    g, cov, _ = data[l]
    for k, v in enumerate(g):
        x = i + (k - 1.5) * (W + 0.015)
        ax.bar(x, v, W, color=COLORS[k], edgecolor=SURF, linewidth=1.5, label=LEVELS[k] if i == 0 else None)
        ax.annotate(f"{v:.0f}%", (x, max(v, 0)), xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8.5, color=INK, fontweight="bold" if k in (0, 3) else "normal")
ax.axhline(0, color=INK2, lw=1)
ax.set_xticks(range(len(LOADS)))
ax.set_xticklabels([f"L{l}  {LOADS[l]}\nmk runs {data[l][1]:.0f}% of forward time" for l in LOADS], fontsize=10)
ax.tick_params(axis="x", length=0, pad=8)
ax.annotate("TPOT gain beats the GPU gain here:\nfaster steps drain the queue, so\nbatches are smaller and tokens wait less",
            (1 + 1.5 * (W + 0.015), data[2][0][3]), xytext=(18, 26), textcoords="offset points", fontsize=8.5,
            color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_ylim(-4, 31)
ax.set_ylabel("time saved by mk vs stock vLLM (%)")
ax.grid(True, axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.01), handlelength=1.4)
fig.suptitle("How the mk gain dilutes inside vLLM serving", x=0.07, ha="left", fontsize=14, color=INK,
             fontweight="bold", y=0.975)
fig.text(0.07, 0.905, "Same vLLM engine, same Azure 2024 conversation requests, same load; only the forward differs. "
         "Each bar widens\nthe measurement by one layer, left to right: what is left of the kernel gain at that layer.",
         fontsize=9.5, color=INK2)
fig.text(0.07, 0.02, "Levels 1-3: per step shape (request bucket x KV bucket), median mk / stock, weighted by stock's time on "
         "that shape. Level 4: client mean TPOT.\nAll levels measured. Qwen3-0.6B, B200, vLLM 0.31. Data: regime-mk-serve "
         "runs/e2e m2_conv (3 windows x 180 s per load), mk build of Oct 24.", fontsize=7.5, color=MUTED)
fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.17)
out = os.path.join(HERE, "06_dilution.png")
fig.savefig(out, dpi=150)
print(out)
