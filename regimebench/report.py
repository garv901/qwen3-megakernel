"""Records -> ``REPORT.md`` + ``summary.csv`` + per-regime PNG plots.

``python -m regimebench report RUN_DIR... --out results/<date>-b200/``

Every number comes from a ``timing`` / ``accuracy`` record (cited by run id and
slurm job id in the header); nothing is interpolated. "best baseline" is the
fastest ``fi-graph`` variant at that point (steady median).
"""

from __future__ import annotations

import csv
from pathlib import Path

from .harness.records import read_records
from .sol import fp32_ffma_peak
from .workload import Workload, regime_grid
import math
import statistics

DASH = "-"
FALLBACK_CLOCK_MHZ = 1965
HEADLINE = "hv2g"
# Megakernel arms compared against the best baseline in their own table (bespoke kernels: docs/briefs/).
MK_ARMS = ("hv2g", "mk_small", "mk_batch")
PROXY = "hv2-repeat"
# hv2-repeat runs b1/kv128 work per launch; it matches the real step only where every sequence is kv 128, q 1.
PROXY_REGIMES = {"R0", "R1"}
BASELINE = "fi-graph"
VLLM = "vllm"
REGIME_ORDER = [f"R{i}" for i in range(9)]
# regime -> (x axis key, group key or None, log x)
REGIME_AXES = {"R0": ("B", None), "R1": ("B", None), "R2": ("kv", None), "R3": ("kv", "B"),
               "R4": ("T", None), "R5": ("T", "B"), "R6": ("q", None), "R7": ("D", None),
               "R8": ("B", None)}


def _us(ms) -> float | None:
    return ms * 1e3 if isinstance(ms, (int, float)) else None


def _f(x, fmt="{:.1f}") -> str:
    return fmt.format(x) if isinstance(x, (int, float)) else DASH


class Data:
    """Indexed records."""

    def __init__(self, records: list[dict]):
        self.records = records
        self.timing: dict[str, dict[tuple[str, str], dict]] = {}   # wid -> (arm, variant) -> rec
        self.skips: list[dict] = []
        self.errors: list[dict] = []
        self.accuracy: list[dict] = []
        self.workloads: dict[str, dict] = {}
        self.env: dict = {}
        self.probe: dict = {}
        self.probes: dict[str, dict] = {}   # run_id -> probe record
        self.run_jobs: dict[str, str] = {}  # run_id -> slurm job id
        self.wid_run: dict[str, str] = {}   # workload id -> run_id of its timing records
        for r in records:
            t = r.get("record_type")
            wd = r.get("workload")
            if wd:
                self.workloads.setdefault(wd["id"], wd)
            if r.get("run_id") and r.get("slurm_job_id"):
                self.run_jobs.setdefault(r["run_id"], str(r["slurm_job_id"]))
            if t == "probe":
                self.probes.setdefault(r.get("run_id"), r)
            if t == "timing" and wd:
                self.wid_run.setdefault(wd["id"], r.get("run_id"))
                self.timing.setdefault(wd["id"], {})[(r["arm"], r["variant"])] = r
            elif t == "skip":
                self.skips.append(r)
            elif t == "error":
                self.errors.append(r)
            elif t == "accuracy":
                self.accuracy.append(r)
            elif t == "env" and not self.env:
                self.env = r.get("env", {})
            elif t == "probe" and not self.probe:
                self.probe = r

    def steady_us(self, wid: str, arm: str, variant: str) -> float | None:
        r = self.timing.get(wid, {}).get((arm, variant))
        return _us((r or {}).get("steady", {}).get("median"))

    def best(self, wid: str, arm: str, variants: tuple[str, ...] | None = None):
        """(variant, steady_us) of the fastest variant of ``arm`` at ``wid``, or None."""
        cands = [(v, self.steady_us(wid, a, v)) for (a, v) in self.timing.get(wid, {})
                 if a == arm and (variants is None or v in variants)]
        cands = [c for c in cands if c[1] is not None]
        return min(cands, key=lambda c: c[1]) if cands else None

    def clock_mhz(self, wid: str) -> float:
        c = (self.probes.get(self.wid_run.get(wid), {}).get("clock") or {}).get("sm_mhz")
        return c or FALLBACK_CLOCK_MHZ

    def ffma_sol_us(self, wid: str) -> float | None:
        """max(T_bw, flops / FP32 FFMA peak at the run's probed SM clock), in us."""
        s = self.sol(wid)
        if not s:
            return None
        t_fl = s["flops"] / fp32_ffma_peak(self.clock_mhz(wid) * 1e6) * 1e6
        return max(s["t_bw_us"], t_fl)

    def steady_gpu_us(self, wid: str, arm: str, variant: str) -> float | None:
        r = self.timing.get(wid, {}).get((arm, variant)) or {}
        return _us((r.get("steady_gpu") or {}).get("median"))

    def sol(self, wid: str, kind: str = "spec") -> dict | None:
        for r in self.timing.get(wid, {}).values():
            s = (r.get("sol") or {}).get(kind)
            if s:
                return s
        return None


def ordered_workloads(d: Data) -> list[dict]:
    grid = {w.id: i for i, w in enumerate(regime_grid())}
    return sorted(d.workloads.values(), key=lambda w: (grid.get(w["id"], 10 ** 6), w["id"]))


def point_row(d: Data, w: dict) -> dict:
    wid = w["id"]
    sol = d.sol(wid)
    sol_us = sol["t_sol_us"] if sol else None
    h = d.best(wid, HEADLINE, ("auto",)) or d.best(wid, HEADLINE)
    p = d.best(wid, PROXY)
    b = d.best(wid, BASELINE)
    v = d.best(wid, VLLM)
    pct = lambda x: 100 * sol_us / x if sol_us and x else None  # noqa: E731
    ffma = d.ffma_sol_us(wid)
    mtn = d.best(wid, HEADLINE, ("mtn",))
    vgpu = d.steady_gpu_us(wid, VLLM, v[0]) if v else None
    mk = {a: d.best(wid, a, ("auto",)) or d.best(wid, a) for a in MK_ARMS}
    return {"id": wid, "ffma_us": ffma, "mtn": mtn, "vllm_gpu": vgpu, "mk": mk,
            "pct_ffma_hv2g": 100 * ffma / h[1] if ffma and h else None,
            "sol_us": sol_us, "bound": sol["bound"] if sol else None,
            "hv2g": h, "proxy": p, "best": b, "vllm": v,
            "ratio": (h[1] / b[1]) if h and b else None,
            "pct_hv2g": pct(h[1] if h else None), "pct_best": pct(b[1] if b else None)}


def _cell(x) -> str:
    return _f(x[1]) if x else DASH


def regime_table(d: Data, regime: str) -> str:
    rows = [point_row(d, w) for w in ordered_workloads(d) if w["regime"] == regime]
    head = ["point", "SOL us (bound)", "FFMA-SOL us", "hv2g us", "hv2g mtn us", "hv2-repeat us (proxy)",
            "best baseline us (variant)", "vllm wall / gpu us", "hv2g/best", "%SOL hv2g",
            "%FFMA-SOL hv2g", "%SOL best"]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        best = f"{_f(r['best'][1])} ({r['best'][0]})" if r["best"] else DASH
        vl = (f"{_f(r['vllm'][1])} / {_f(r['vllm_gpu'])} ({r['vllm'][0]})") if r["vllm"] else DASH
        out.append("| " + " | ".join([
            r["id"], f"{_f(r['sol_us'])} ({r['bound'] or DASH})", _f(r["ffma_us"]), _cell(r["hv2g"]),
            _cell(r["mtn"]), _cell(r["proxy"]) if regime in PROXY_REGIMES else "n/a", best, vl,
            _f(r["ratio"], "{:.2f}"), _f(r["pct_hv2g"]), _f(r["pct_ffma_hv2g"]), _f(r["pct_best"])]) + " |")
    return "\n".join(out)


def mk_table(d: Data) -> str:
    """Every megakernel arm present vs best fi-graph, vLLM and SOL, one row per point."""
    arms = [a for a in MK_ARMS if any(a == arm for t in d.timing.values() for (arm, _) in t)]
    head = ["point", "SOL us", "best baseline us", "vllm us"] + [f"{a} us" for a in arms] + \
        [f"{a}/best" for a in arms]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for w in ordered_workloads(d):
        r = point_row(d, w)
        b = r["best"][1] if r["best"] else None
        vals = [r["mk"][a][1] if r["mk"][a] else None for a in arms]
        out.append("| " + " | ".join([r["id"], _f(r["sol_us"]), _cell(r["best"]), _cell(r["vllm"])]
                                      + [_f(v) for v in vals]
                                      + [_f(v / b, "{:.2f}") if v and b else DASH for v in vals]) + " |")
    return "\n".join(out)


def _crossover(rows: list[dict], ws: list[dict], xkey: str):
    """Axis value of the first point (ascending x) where best baseline beats hv2g (ratio > 1)."""
    pts = [(w["axis"].get(xkey), r) for w, r in zip(ws, rows) if r["ratio"] is not None]
    pts = [(x, r) for x, r in pts if isinstance(x, (int, float))]
    for x, r in sorted(pts, key=lambda t: t[0]):
        if r["ratio"] > 1:
            return x
    return None


def summary_rows(d: Data) -> list[dict]:
    out = []
    for reg in REGIME_ORDER:
        ws = [w for w in ordered_workloads(d) if w["regime"] == reg]
        if not ws:
            continue
        rows = [point_row(d, w) for w in ws]
        both = [r for r in rows if r["ratio"] is not None]
        ratios = [r["ratio"] for r in both]
        xkey, gkey = REGIME_AXES.get(reg, ("B", None))
        cross = None
        if gkey:  # one crossover per group
            groups: dict = {}
            for w, r in zip(ws, rows):
                groups.setdefault(w["axis"].get(gkey), ([], []))
                groups[w["axis"].get(gkey)][0].append(w)
                groups[w["axis"].get(gkey)][1].append(r)
            cross = "; ".join(f"{gkey}={g}: {_crossover(rr, wws, xkey) or 'none'}"
                              for g, (wws, rr) in groups.items())
        else:
            c = _crossover(rows, ws, xkey)
            cross = "none" if c is None else f"{xkey}={c}"
        med = lambda k: statistics.median([r[k] for r in rows if r[k] is not None]) \
            if any(r[k] is not None for r in rows) else None  # noqa: E731
        out.append({"regime": reg, "points": len(ws), "compared": len(both),
                    "wins": sum(1 for r in both if r["ratio"] < 1),
                    "gmean": math.exp(sum(map(math.log, ratios)) / len(ratios)) if ratios else None,
                    "crossover": cross, "pct_hv2g": med("pct_hv2g"), "pct_best": med("pct_best")})
    return out


def summary_table(d: Data) -> str:
    head = ["regime", "points", "hv2g wins", "geomean hv2g/best", "crossover (best baseline first beats hv2g)",
            "median %SOL hv2g", "median %SOL best"]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for r in summary_rows(d):
        out.append("| " + " | ".join([r["regime"], str(r["points"]), f"{r['wins']}/{r['compared']}",
                                      _f(r["gmean"], "{:.2f}"), str(r["crossover"]),
                                      _f(r["pct_hv2g"]), _f(r["pct_best"])]) + " |")
    return "\n".join(out)


def plot_overview(d: Data, out_dir: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = [(w, point_row(d, w)) for w in ordered_workloads(d)]
    pts = [(w, r) for w, r in pts if r["ratio"] is not None]
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(max(6, 0.28 * len(pts)), 4))
    for i, (w, r) in enumerate(pts):
        ax.scatter(i, r["ratio"], color=f"C{int(w['regime'][1:]) % 10}", s=22)
    for reg in REGIME_ORDER:
        idx = [i for i, (w, _) in enumerate(pts) if w["regime"] == reg]
        if idx:
            ax.axvspan(idx[0] - 0.5, idx[-1] + 0.5, alpha=0.06 * (1 + int(reg[1:]) % 2), color="gray")
            ax.text((idx[0] + idx[-1]) / 2, 1.02, reg, ha="center", va="bottom", fontsize=8,
                    transform=ax.get_xaxis_transform())
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(pts)))
    ax.set_xticklabels([w["id"] for w, _ in pts], rotation=90, fontsize=5)
    ax.set_ylabel("hv2g / best fi-graph (<1: hv2g faster)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "overview.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def winners_table(d: Data) -> str:
    head = ["point", "best fi-graph variant", "attn backend", "gemm", "PDL cover", "best vllm variant"]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for w in ordered_workloads(d):
        b = d.best(w["id"], BASELINE)
        v = d.best(w["id"], VLLM)
        cfg = d.timing[w["id"]][(BASELINE, b[0])]["config"] if b else {}
        pdl = cfg.get("pdl") if isinstance(cfg.get("pdl"), dict) else {}
        cover = (f"{pdl.get('launches_with_pdl', 0)}/{pdl['launches_total']}"
                 if "launches_total" in pdl else DASH)
        out.append("| " + " | ".join([w["id"], b[0] if b else DASH, str(cfg.get("attn_backend", DASH)),
                                      str(cfg.get("gemm", DASH)), cover, v[0] if v else DASH]) + " |")
    return "\n".join(out)


def accuracy_table(d: Data) -> str:
    groups: dict[tuple, list[dict]] = {}
    for r in d.accuracy:
        if r.get("arm") == "reference":
            continue
        groups.setdefault((r["workload"]["id"], r["arm"], r["variant"]), []).append(r)
    if not groups:
        return "_no accuracy records_"
    head = ["point", "arm:variant", "seeds", "worst x band (logits/k/v)", "next_token match", "verdict"]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for (wid, arm, v), rs in sorted(groups.items()):
        worst = max(rs, key=lambda r: r["worst_ratio"])
        ratios = "/".join(_f(worst["ratios"][k], "{:.2f}") for k in ("logits", "k_new", "v_new"))
        nt = min(r.get("next_token_match_rate", 0) for r in rs)
        verdict = "fail" if any(r["verdict"] == "fail" for r in rs) else "pass"
        out.append(f"| {wid} | {arm}:{v} | {len(rs)} | {ratios} | {_f(nt, '{:.2f}')} | {verdict} |")
    return "\n".join(out)


def issues(d: Data) -> str:
    lines = []
    for r in d.skips:
        wid = (r.get("workload") or {}).get("id", DASH)
        lines.append(f"- SKIP {r.get('arm')}:{r.get('variant')} @ {wid}: {r.get('reason')}")
    for r in d.errors:
        wid = (r.get("workload") or {}).get("id", DASH)
        tail = (r.get("traceback") or "").strip().splitlines()
        lines.append(f"- ERROR {r.get('arm')}:{r.get('variant')} @ {wid} [{r.get('stage')}]: "
                     f"{r.get('error') or ''} {('| ' + tail[-1]) if tail else ''}")
    return "\n".join(lines) or "_none_"


def header(d: Data, records: list[dict]) -> str:
    run_ids = sorted({r.get("run_id") for r in records if r.get("run_id")})
    jobs = sorted({str(r.get("slurm_job_id")) for r in records if r.get("slurm_job_id")})
    gpu = d.env.get("gpu", {})
    v = d.env.get("versions", {})
    hbm = d.probe.get("hbm", {})
    gemm = d.probe.get("gemm", {})
    clk = d.probe.get("clock", {})
    lines = ["# regimebench report", "",
             f"- Runs: {', '.join(run_ids) or DASH}; slurm jobs: {', '.join(jobs) or DASH} "
             f"(logs under runs/<run>/ and runs/slurm-<job>.log)",
             f"- GPU: {gpu.get('name', DASH)} ({gpu.get('sm_count', DASH)} SMs), driver "
             f"{d.env.get('driver') or DASH}, host {d.env.get('host', DASH)}",
             "- Versions: " + ", ".join(f"{k} {x or DASH}" for k, x in v.items())
             + f"; git {((d.env.get('git') or {}).get('sha') or DASH)[:10]}"
             + (" (dirty)" if (d.env.get("git") or {}).get("dirty") else ""),
             f"- SOL spec: 8.0 TB/s, BF16 dense 2.25 PFLOP/s (datasheet-derived, unverified); "
             f"measured: copy BW {_f(hbm.get('median_gbps'), '{:.0f}')} GB/s median, "
             f"cuBLAS bf16 {_f(gemm.get('peak_tflops'), '{:.0f}')} TFLOP/s, "
             f"SM clock at probe {clk.get('sm_mhz', DASH)} MHz",
             "- All times are microseconds per step (steady = back-to-back launches). "
             "hv2-repeat is a timing proxy (N repeated b1 decodes), not a correct forward; shown only in R0/R1, "
             "where its b1/kv128 work matches the step (elsewhere it skips KV work and can fall below SOL)."]
    lines.append("- FFMA-SOL = max(T_bw, flops / (148 SMs x 128 lanes x 2 x probed SM clock)); a ceiling for an "
                 "FP32-FFMA design like hv2g, not for a tensor-core one (hypothesis-grade model, not a measurement).")
    for rid in sorted({r.get("run_id") for r in records if r.get("run_id")}):
        regs = sorted({d.workloads[w]["regime"] for w, run in d.wid_run.items() if run == rid},
                      key=REGIME_ORDER.index)
        clk_r = (d.probes.get(rid, {}).get("clock") or {}).get("sm_mhz")
        lines.append(f"- Run {rid} (slurm job {d.run_jobs.get(rid, DASH)}, SM clock at probe {clk_r or DASH} MHz): "
                     f"regimes {', '.join(regs) or DASH}")
    lines.append("- Cross-regime comparisons are across jobs (and possibly GPUs); ratios within a point are "
                 "interleaved in one job and are the reliable comparison.")
    return "\n".join(lines)


# -- plots ---------------------------------------------------------------


def _axis_vals(w: dict, key: str):
    return w["axis"].get(key)


def plot_regime(d: Data, regime: str, out_dir: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ws = [w for w in ordered_workloads(d) if w["regime"] == regime]
    if not ws:
        return None
    xkey, gkey = REGIME_AXES.get(regime, ("B", None))
    groups: dict = {}
    for w in ws:
        groups.setdefault(_axis_vals(w, gkey) if gkey else None, []).append(w)
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 4), squeeze=False)
    for ax, (g, pts) in zip(axes[0], groups.items()):
        pts = sorted(pts, key=lambda w: (_axis_vals(w, xkey) if isinstance(_axis_vals(w, xkey), (int, float)) else 0))
        xs = [(_axis_vals(w, xkey) if isinstance(_axis_vals(w, xkey), (int, float)) else i)
              for i, w in enumerate(pts)]
        rows = [point_row(d, w) for w in pts]

        def series(vals, label, **kw):
            pairs = [(x, v) for x, v in zip(xs, vals) if v]
            if pairs:
                ax.plot(*zip(*pairs), label=label, **{"marker": "o", **kw})
        series([r["hv2g"][1] if r["hv2g"] else None for r in rows], "hv2g", color="C3")
        series([r["mtn"][1] if r["mtn"] else None for r in rows], "hv2g mtn", color="C1")
        for a, c in (("mk_small", "C4"), ("mk_batch", "C5")):
            series([r["mk"][a][1] if r["mk"][a] else None for r in rows], a, color=c)
        if regime in PROXY_REGIMES:
            series([r["proxy"][1] if r["proxy"] else None for r in rows], "hv2-repeat (proxy)",
                   color="C3", linestyle="--")
        series([r["best"][1] if r["best"] else None for r in rows], "best fi-graph", color="C0")
        series([r["vllm"][1] if r["vllm"] else None for r in rows], "vllm", color="C2")
        series([r["sol_us"] for r in rows], "SOL spec", color="k", linestyle=":", marker="")
        series([r["ffma_us"] for r in rows], "FFMA-SOL", color="k", linestyle=(0, (1, 3)), marker="")
        series([(d.sol(w["id"], "measured") or {}).get("t_sol_us") for w in pts],
               "SOL measured", color="gray", linestyle="-.", marker="")
        ax.set_yscale("log")
        if len(xs) > 2 and all(x > 0 for x in xs):
            ax.set_xscale("log", base=2)
        ax.set_xlabel(xkey)
        ax.set_ylabel("us per step")
        ax.set_title(f"{regime}" + (f" ({gkey}={g})" if gkey else ""))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / f"{regime}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# -- csv -----------------------------------------------------------------

CSV_FIELDS = ["workload", "regime", "arm", "variant", "steady_us", "steady_ci_lo_us",
              "steady_ci_hi_us", "isolated_us", "tokens_per_s", "sol_spec_us", "sol_measured_us", "ffma_sol_us",
              "bound", "pct_sol_spec", "n_launch_per_sample"]


def write_csv(d: Data, path: Path) -> None:
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, CSV_FIELDS)
        wr.writeheader()
        for w in ordered_workloads(d):
            for (arm, v), r in sorted(d.timing.get(w["id"], {}).items()):
                st = r.get("steady", {})
                sp = (r.get("sol") or {}).get("spec") or {}
                me = (r.get("sol") or {}).get("measured") or {}
                ci = st.get("ci95_median") or [None, None]
                med = _us(st.get("median"))
                wr.writerow({
                    "workload": w["id"], "regime": w["regime"], "arm": arm, "variant": v,
                    "steady_us": med, "steady_ci_lo_us": _us(ci[0]), "steady_ci_hi_us": _us(ci[1]),
                    "isolated_us": _us((r.get("isolated") or {}).get("median")),
                    "tokens_per_s": r.get("tokens_per_s"), "sol_spec_us": sp.get("t_sol_us"),
                    "sol_measured_us": me.get("t_sol_us"), "ffma_sol_us": d.ffma_sol_us(w["id"]), "bound": sp.get("bound"),
                    "pct_sol_spec": 100 * sp["t_sol_us"] / med if sp and med else None,
                    "n_launch_per_sample": r.get("n_launch_per_sample")})


def render(d: Data, records: list[dict], plot_files: dict[str, str]) -> str:
    parts = [header(d, records), "", "## Executive summary", "", summary_table(d), ""]
    if any(arm in ("mk_small", "mk_batch") for t in d.timing.values() for (arm, _) in t):
        parts += ["## Megakernel arms vs best baseline", "", mk_table(d), ""]
    if "overview" in plot_files:
        parts += ["![overview](overview.png)", ""]
    regimes = [r for r in REGIME_ORDER if any(w["regime"] == r for w in d.workloads.values())]
    for reg in regimes:
        parts += [f"## {reg}", "", regime_table(d, reg), ""]
        if reg in plot_files and reg != "overview":
            parts += [f"![{reg}]({plot_files[reg]})", ""]
    parts += ["## Winner variants and backends", "", winners_table(d), "",
              "## Accuracy (FP32 oracle band, pass = rel-L2 <= 2x BF16 reference's own)", "",
              accuracy_table(d), "", "## Skips and errors", "", issues(d), ""]
    return "\n".join(parts)


def main_report(runs: list[str], out: str) -> int:
    records: list[dict] = []
    for p in runs:
        path = Path(p)
        records += read_records(path / "records.jsonl" if path.is_dir() else path)
    d = Data(records)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots: dict[str, str] = {}
    for reg in REGIME_ORDER:
        try:
            pth = plot_regime(d, reg, out_dir)
        except Exception as exc:  # plotting must not block the tables
            print(f"plot {reg} failed: {exc!r}")
            pth = None
        if pth:
            plots[reg] = pth.name
    try:
        if plot_overview(d, out_dir):
            plots["overview"] = "overview.png"
    except Exception as exc:
        print(f"overview plot failed: {exc!r}")
    write_csv(d, out_dir / "summary.csv")
    (out_dir / "REPORT.md").write_text(render(d, records, plots))
    print(f"wrote {out_dir}/REPORT.md, summary.csv, {len(plots)} plots")
    return 0
