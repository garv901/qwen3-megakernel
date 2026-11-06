"""Core trend plots: mk vs fi-graph vs vLLM, all from one job (14156, run 20261029T175019Z-mk-tc-vllm, mk c2bc494).

Run from the checkout root: <venv>/bin/python results/2026-10-29-plots/make_plots.py
"""
import collections
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = "runs/20261029T175019Z-mk-tc-vllm/records.jsonl"

INK, INK2, MUTED, GRID, SURF = "#1f1f1e", "#52514c", "#8a897f", "#e6e5e0", "#fcfcfb"
C_MK, C_FI, C_VL = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})


def load():
    t = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(RUN):
        r = json.loads(line)
        if r["record_type"] == "timing":
            t[r["workload"]["id"]][(r["arm"], r["variant"])].append(r["steady"]["median"] * 1000)
    mk, fi, vl = {}, {}, {}
    for w, g in t.items():
        g = {k: statistics.median(v) for k, v in g.items()}
        mk[w] = g[("mk", "auto")]
        fi[w] = min(v for (a, var), v in g.items() if a == "fi-graph" and not var.startswith("eager"))
        v = [x for (a, _), x in g.items() if a == "vllm"]  # LLMEngine.step wall, best of pdl / nopdl
        if v:
            vl[w] = min(v)
    return mk, fi, vl


def us_fmt(v, _):
    return f"{v / 1000:g} ms" if v >= 1000 else f"{v:g}"


def series(ids, xs, src):
    pts = [(x, src[w]) for x, w in zip(xs, ids) if w in src]
    return [p[0] for p in pts], [p[1] for p in pts]


def end_label(ax, xs, ys, text, color, dy=0):
    ax.annotate(text, (xs[-1], ys[-1]), xytext=(6, dy), textcoords="offset points",
                va="center", fontsize=9, color=INK, fontweight="bold")
    ax.plot([], [], color=color)


def plot(name, title, subtitle, xlabel, lines, xlog=True, xticks=None, xticklabels=None, note=None, ylim=None):
    """lines: list of (label, ids, xs, linestyle); each draws mk / fi / vLLM for its ids."""
    mk, fi, vl = DATA
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08})
    handles, all_r = [], []
    for li, (lab, ids, xs, ls) in enumerate(lines):
        suffix = f" ({lab})" if lab else ""
        for src, color, sys_name, mk_ in ((vl, C_VL, "vLLM", "s"), (fi, C_FI, "fi-graph", "^"), (mk, C_MK, "mk", "o")):
            x, y = series(ids, xs, src)
            if not x:
                continue
            h, = ax.plot(x, y, ls, color=color, lw=2, marker=mk_, ms=7, mec=SURF, mew=1.5,
                         label=sys_name + suffix)
            handles.append(h)
        x, r = zip(*[(xx, mk[w] / fi[w]) for xx, w in zip(xs, ids)])
        all_r += r
        axr.plot(x, r, ls, color=C_MK, lw=2, marker="o", ms=7, mec=SURF, mew=1.5,
                 label=f"mk / fi-graph{suffix}")
        idx = {0, len(r) - 1} if len(lines) > 1 else {0, len(r) - 1, max(range(len(r)), key=r.__getitem__)}
        for i in sorted(idx):
            axr.annotate(f"{r[i]:.3f}", (x[i], r[i]), xytext=(0, 9 if li == len(lines) - 1 else -15),
                         textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(us_fmt))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("step time (log)")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left", ncol=1 if len(lines) == 1 else 2, handlelength=3.5)
    axr.axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)))
    axr.annotate("parity", (1.0, 1.0), xycoords=("axes fraction", "data"), xytext=(-2, -4),
                 textcoords="offset points", ha="right", va="top", fontsize=8, color=MUTED)
    axr.set_ylabel("mk / fi-graph")
    axr.set_ylim(min(0.7, min(all_r) - 0.03), 1.06)
    axr.grid(True, color=GRID, lw=0.8)
    if len(lines) > 1:
        axr.legend(frameon=False, fontsize=8.5, loc="lower right", handlelength=3.5)
    if xlog:
        axr.set_xscale("log", base=2)
    if xticks:
        axr.set_xticks(xticks)
        axr.set_xticklabels(xticklabels or [str(v) for v in xticks])
        axr.xaxis.set_minor_formatter(NullFormatter())
    axr.set_xlabel(xlabel, labelpad=6)
    fig.suptitle(title, x=0.08, ha="left", fontsize=13, color=INK, fontweight="bold", y=0.975)
    fig.text(0.08, 0.925, subtitle, fontsize=9.5, color=INK2)
    foot = ("Same job 14156 (mk c2bc494), steady median per step. fi-graph = fastest graph variant per point.\n"
            "vLLM = offline LLMEngine.step wall time (includes host scheduling), best of pdl / nopdl.")
    if note:
        foot += "\n" + note
    fig.text(0.08, 0.015, foot, fontsize=7.5, color=MUTED)
    fig.subplots_adjust(top=0.9, bottom=0.165 if note else 0.145, left=0.11, right=0.95)
    out = os.path.join(HERE, name)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


K = 1024
DATA = load()

b = [1, 2, 4, 8, 16, 32, 64, 128, 256]
plot("01_fixed_kv128_vs_batch.png",
     "Fixed KV (128 / seq), growing batch",
     "R1: decode, q_len 1, every sequence has a 128-token KV cache. GEMMs grow with B; attention stays tiny.",
     "batch size B (sequences)",
     [("", [f"R1-b{x}-kv128" for x in b], b, "-")], xticks=b)

kv = [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]
kvl = ["128", "512", "1K", "2K", "4K", "8K", "16K", "32K"]
plot("02_fixed_b1_vs_kv.png",
     "Fixed batch (B = 1), growing KV",
     "R2: one sequence, decode q_len 1. Weight streaming dominates; KV adds a slowly growing attention cost.",
     "KV length (tokens)",
     [("", [f"R2-b1-kv{x}" for x in kv], kv, "-")], xticks=kv, xticklabels=kvl)

b8 = [("R1", 128), ("R3", 1024), ("R9", 2048), ("R3", 4096), ("R9", 8192), ("R9", 12288),
      ("R3", 16384), ("R9", 24576), ("R9", 30720), ("R9", 32768)]
b16 = [("R1", 128), ("R9", 4096), ("R9", 8192), ("R9", 15360)]
xt = [128, 1024, 4096, 16384, 32768]
plot("03_fixed_b8_b16_vs_kv.png",
     "Fixed batch (B = 8, B = 16), growing KV",
     "Decode q_len 1. Attention takes over as KV grows; mk's lead narrows toward parity past ~16K.",
     "KV length per sequence (tokens)",
     [("B=8", [f"{r}-b8-kv{x}" for r, x in b8], [x for _, x in b8], "-"),
      ("B=16", [f"{r}-b16-kv{x}" for r, x in b16], [x for _, x in b16], "--")],
     xticks=xt, xticklabels=["128", "1K", "4K", "16K", "32K"],
     note="Points joined across R1 / R3 / R9 workloads (same job).")

iso = [(1, 245760), (2, 122880), (4, 61440), (8, 30720), (16, 15360)]
plot("04_iso_kv240k_batch_vs_length.png",
     "Constant total KV (~240K tokens), batch vs length",
     "R9 iso-240K: B x kv held fixed (B=1 kv 240K ... B=16 kv 15K). Attention-dominated; near parity.",
     "batch size B (KV per sequence = 240K / B)",
     [("", [f"R9-b{x}-kv{k}" for x, k in iso], [x for x, _ in iso], "-")],
     xticks=[1, 2, 4, 8, 16], xticklabels=["1\n240K", "2\n120K", "4\n60K", "8\n30K", "16\n15K"], ylim=(2000, 10000),
     note="vLLM has no point at kv > 40K (B = 1, 2, 4): max_model_len exceeds Qwen3-0.6B's 40,960 positions.")

q = [1, 2, 4, 8, 16]
plot("05_b1_kv1k_vs_qlen.png",
     "Tokens per step per sequence (B = 1, KV 1K)",
     "R6: q new tokens verified in one pass (speculative-decode verify / MTP), causal over the new tokens.",
     "query tokens per sequence per step (q_len)",
     [("", ["R2-b1-kv1024"] + [f"R6-b1-kv1024-q{x}" for x in q[1:]], q, "-")], xticks=q,
     note="q_len 1 point is R2-b1-kv1024 (plain decode).")
