"""Markdown report from ``records.jsonl``: ``python -m megabench.sota report RUN [--markdown OUT]``."""

from __future__ import annotations

import argparse
from pathlib import Path

DASH = "—"
SOTA_ABLATION = ("eager-nopdl", "eager-pdl", "graph-nopdl", "graph-pdl",
                 "graph-figemm-nopdl", "graph-figemm-pdl")
MEGA_ABLATION = ("direct", "direct-pdl", "direct-pdl-early", "graph", "submission")
VLLM_ABLATION = ("mp0", "mp0-nopdl")
SOL_TB_S = 8.0


def _num(x, fmt="{:.4f}") -> str:
    return fmt.format(x) if isinstance(x, (int, float)) else DASH


def _med_ci(stats: dict | None, digits: int = 4) -> str:
    if not stats or "median" not in stats:
        return DASH
    lo, hi = stats["ci95_median"]
    return f"{stats['median']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _g(rec: dict, *names):
    for n in names:
        if rec.get(n) is not None:
            return rec[n]
    return None


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _pdl(config: dict | None) -> str:
    p = (config or {}).get("pdl")
    if isinstance(p, dict) and "launches_total" in p:
        return f"{p.get('launches_with_pdl', 0)}/{p['launches_total']}"
    return DASH


def _short_config(config: dict | None) -> str:
    c = config or {}
    parts = [str(c[k]) for k in ("attn_backend", "gemm") if c.get(k)]
    parts.append("pdl " + _pdl(c))
    return ", ".join(parts)


class Index:
    def __init__(self, records: list[dict], shape_id: str | None):
        keep = lambda r: shape_id is None or r.get("shape") in (None, shape_id)  # noqa: E731
        self.recs = [r for r in records if keep(r)]
        self.order: list[tuple[str, str]] = []
        self.timing: dict[tuple, dict] = {}
        self.profile: dict[tuple, dict] = {}
        self.correct: dict[tuple, list[dict]] = {}
        self.skips: dict[tuple, str] = {}
        self.errors: dict[tuple, list[dict]] = {}
        self.vllm: dict[str, dict] = {}
        for r in self.recs:
            t = r.get("record_type")
            key = (r.get("arm"), r.get("variant"))
            if r.get("arm") == "vllm" and t not in ("error", "skip"):
                # The slope record carries slope/ITL/context; env and engine-ITL
                # records are not table rows.
                if t == "timing" and r.get("mode") == "vllm_slope":
                    self.vllm.setdefault(r.get("variant"), {}).update(
                        {k: v for k, v in r.items() if v is not None})
                    if key not in self.order:
                        self.order.append(key)
                continue
            if t in ("timing", "profile", "correctness", "skip", "error") and key[0] and key not in self.order:
                self.order.append(key)
            if t == "timing":
                self.timing[key + (r["mode"], r["l2"])] = r
            elif t == "profile":
                self.profile[key] = r
            elif t == "correctness":
                self.correct.setdefault(key, []).append(r)
            elif t == "skip":
                self.skips[key] = r.get("reason", "")
            elif t == "error":
                self.errors.setdefault(key, []).append(r)

    def steady(self, key):
        return self.timing.get(key + ("steady", "warm"))

    def status(self, key) -> str:
        if self.steady(key) or (key[0] == "vllm" and key[1] in self.vllm):
            return "ok"
        if key in self.skips:
            return "skip"
        if key in self.errors:
            return "error"
        return "ok" if key in self.correct else DASH

    def accuracy(self, key) -> str:
        cs = self.correct.get(key)
        if not cs:
            return DASH
        order = {"pass": 0, "warn": 1, "fail": 2, "unknown": 3}
        grade = max((c["grade"] for c in cs), key=lambda g: order.get(g, 3))
        own = "vs_fp32" if cs[0].get("own_ref") == "fp32" else "vs_bf16"

        def worst(field):
            v = [c[field]["logits"]["rel_l2"] for c in cs if c.get(field)]
            return f"{max(v):.1e}" if v else DASH
        ref = "vs_fp32" if any(c.get("vs_fp32") for c in cs) else "vs_bf16"
        match = sum(1 for c in cs if (c.get(ref) or {}).get("logits", {}).get("argmax_match"))
        return (f"{grade}; fp32 {worst('vs_fp32')}; bf16 {worst('vs_bf16')}; "
                f"argmax={ref[3:]} {match}/{len(cs)}")


def _header(records: list[dict]) -> str:
    env = next((r for r in records if r.get("record_type") == "env"), {}).get("env", {})
    bw = next((r for r in records if r.get("record_type") == "hbm_bw"), {})
    gpu = env.get("gpu", {})
    v = env.get("versions", {})
    lines = ["# SOTA decode benchmark", ""]
    lines.append(f"- GPU: {gpu.get('name', DASH)} ({gpu.get('sm_count', DASH)} SMs, "
                 f"L2 {_num((gpu.get('l2_bytes') or 0) / 2**20, '{:.0f}')} MiB), driver "
                 f"{env.get('driver') or DASH}, host {env.get('host', DASH)}")
    lines.append("- Versions: " + ", ".join(f"{k} {x or DASH}" for k, x in v.items())
                 + f"; nvcc {env.get('nvcc') or DASH}; git {(env.get('git') or {}).get('sha', DASH)[:10] if (env.get('git') or {}).get('sha') else DASH}"
                 + (" (dirty)" if (env.get("git") or {}).get("dirty") else ""))
    sm = [r["clock"]["sm_mhz"] for r in records if r.get("record_type") == "timing"
          and r.get("clock", {}).get("sm_mhz")]
    thr = sorted({x for r in records if r.get("record_type") == "timing"
                  for x in r.get("throttled_rounds", [])})
    reasons = sorted({x for r in records if r.get("record_type") == "timing"
                      for x in r.get("clock", {}).get("throttle_reasons", [])})
    if sm:
        lines.append(f"- SM clock MHz: min {min(s['min'] for s in sm)}, max "
                     f"{max(s['max'] for s in sm)}; throttled rounds: {thr or 'none'}; "
                     f"reasons seen: {reasons or 'none'}")
    else:
        lines.append(f"- Clocks: {DASH} (NVML unavailable)")
    lines.append(f"- Measured HBM copy BW: {_num(bw.get('median_gbps'), '{:.0f}')} GB/s median, "
                 f"{_num(bw.get('max_gbps'), '{:.0f}')} max")
    return "\n".join(lines)


def _main_table(ix: Index, shape: dict | None) -> str:
    header = ["arm:variant", "status", "steady ms [CI]", "isolated ms", "flushed ms",
              "x vs mk direct [CI]", "GB/s", "%SOL@8TB/s", "% meas BW", "kernels/step",
              "accuracy", "config"]
    rows = []
    for key in ix.order:
        st = ix.steady(key)
        iso = ix.timing.get(key + ("isolated", "warm"))
        fl = ix.timing.get(key + ("isolated", "flushed"))
        prof = (ix.profile.get(key) or {}).get("stats", {})
        ratio = (st or {}).get("ratio_vs_headline") or {}
        rtxt = DASH
        if ratio.get("ratio") is not None:
            lo, hi = ratio["ci95"]
            rtxt = f"{ratio['ratio']:.2f} [{lo:.2f}, {hi:.2f}]"
        sol = (st or {}).get("sol", {})
        tbs = sol.get("gbps")
        v = ix.vllm.get(key[1], {}) if key[0] == "vllm" else {}
        steady_txt = _med_ci(st["stats"] if st else None)
        if key[0] == "vllm":
            steady_txt = _slope_txt(v)
        rows.append([f"{key[0]}:{key[1]}", ix.status(key), steady_txt,
                     _med_ci(iso["stats"], 3).split(" ")[0] if iso else DASH,
                     _med_ci(fl["stats"], 3).split(" ")[0] if fl else DASH, rtxt,
                     _num(tbs, "{:.0f}"),
                     _num(100 * tbs / (SOL_TB_S * 1e3), "{:.1f}") if tbs else DASH,
                     _num((st or {}).get("pct_measured_bw"), "{:.1f}"),
                     _num(prof.get("kernels_per_step"), "{:.1f}"), ix.accuracy(key),
                     _short_config((st or {}).get("config") or
                                   (ix.correct.get(key) or [{}])[0].get("config"))])
    return _table(header, rows)


def _slope(v: dict) -> float | None:
    """Median of per-trial slopes (consistent with its CI); else the all-points fit."""
    med = (v.get("stats") or {}).get("median")
    return med if med is not None else _g(v, "slope_ms", "per_step_ms", "slope")


def _slope_txt(v: dict) -> str:
    s = _slope(v)
    ci = (v.get("stats") or {}).get("ci95_median") or _g(v, "slope_ms_ci95", "slope_ci95", "slope_ci")
    if s is None:
        return DASH
    return f"{s:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else f"{s:.4f}"


def _ablation_table(ix: Index) -> str:
    # "overhead" = unprofiled steady step - profiled GPU-busy time: launch gaps
    # plus anything not inside a kernel. Robust to profiler-inflated host launches;
    # under PDL, busy includes dependency waits, so overhead can go slightly negative.
    header = ["arm:variant", "steady ms", "d vs baseline us", "d %", "busy us/step",
              "overhead us/step", "prof gap us/step", "mean gap us", "overlaps", "PDL"]
    groups = [("sota-graph", SOTA_ABLATION, "graph-nopdl"),
              ("megakernel", MEGA_ABLATION, "direct")]
    rows = []

    def steady_ms(key):
        if key[0] == "vllm":
            return _slope(ix.vllm.get(key[1], {}))
        st = ix.steady(key)
        return st["stats"]["median"] if st else None

    def add(key, base):
        ms = steady_ms(key)
        if ms is None and key not in ix.profile:
            return
        b = steady_ms((key[0], base))
        prof = (ix.profile.get(key) or {}).get("stats", {})
        cfg = (ix.steady(key) or {}).get("config")
        rows.append([f"{key[0]}:{key[1]}", _num(ms),
                     _num((ms - b) * 1e3, "{:+.1f}") if ms is not None and b else DASH,
                     _num((ms / b - 1) * 100, "{:+.1f}") if ms is not None and b else DASH,
                     _num(prof.get("busy_us_per_step"), "{:.1f}"),
                     _num(ms * 1e3 - prof["busy_us_per_step"], "{:+.1f}")
                     if ms is not None and prof.get("busy_us_per_step") is not None else DASH,
                     _num(prof.get("gap_us_per_step"), "{:.1f}"),
                     _num(prof.get("gap_mean_us"), "{:.2f}"),
                     _num(prof.get("overlap_count"), "{:.0f}"), _pdl(cfg)])
    for arm, variants, base in groups:
        names = [k[1] for k in ix.order if k[0] == arm]
        extra = [n for n in names if n not in variants and arm == "megakernel"]
        for v in list(variants) + extra:
            if (arm, v) in ix.order:
                add((arm, v), base)
    for v in VLLM_ABLATION:
        if ("vllm", v) in ix.order:
            add(("vllm", v), "mp0")
    return _table(header, rows) if rows else "_no data_"


def _vllm_table(ix: Index) -> str:
    if not ix.vllm:
        return "_no vLLM data_"
    rows = []
    for v, r in ix.vllm.items():
        rows.append([v, _slope_txt(r), _num(_g(r, "r2", "r_squared"), "{:.4f}"),
                     _num(_g(r, "itl_ms", "engine_itl_ms", "engine_itl_ms_mean"), "{:.4f}"),
                     _num(_g(r, "mean_context", "context_mean"), "{:.1f}")])
    return _table(["variant", "per-step slope ms [CI]", "r2", "engine ITL ms", "mean context"], rows)


def _issues(ix: Index) -> str:
    lines = [f"- SKIP {k[0]}:{k[1]}: {r}" for k, r in ix.skips.items()]
    for k, errs in ix.errors.items():
        for e in errs[:2]:
            tail = (e.get("error") or "").strip().splitlines()
            lines.append(f"- ERROR {k[0]}:{k[1]} [{e.get('stage')}]: {tail[-1] if tail else ''}")
    return "\n".join(lines) or "_none_"


def render(records: list[dict]) -> str:
    # Out-of-process workers (vLLM) record the shape as Shape.to_dict(); key by id.
    records = [dict(r, shape=r["shape"]["id"])
               if r.get("record_type") != "shape" and isinstance(r.get("shape"), dict) else r
               for r in records]
    shapes = [r["shape"] for r in records if r.get("record_type") == "shape"]
    ids = list(dict.fromkeys(s["id"] for s in shapes)) or \
        sorted({r["shape"] for r in records if r.get("shape")}) or [None]
    parts = [_header(records)]
    for sid in ids:
        sh = next((s for s in shapes if s["id"] == sid), None)
        floor = next((r["sol"] for r in records if r.get("record_type") == "shape"
                      and r["shape"]["id"] == sid), None)
        ix = Index(records, sid)
        title = f"## Shape {sid}" if sid else "## Results"
        if floor:
            title += (f" (SOL bytes {floor['bytes']:,}, floor {floor['floor_ms']:.4f} ms "
                      f"@ {floor['bandwidth_tb_s']} TB/s)")
        parts += [title, "", "### Main table", "", _main_table(ix, sh), "",
                  "### PDL / launch ablation", "", _ablation_table(ix), "",
                  "### vLLM", "", _vllm_table(ix), "", "### Skipped and errors", "", _issues(ix), ""]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="megabench.sota report")
    p.add_argument("paths", type=Path, nargs="+",
                   help="run dirs or records.jsonl files; several are merged "
                        "(e.g. in-process and vLLM jobs)")
    p.add_argument("--markdown", type=Path, default=None)
    a = p.parse_args(argv)
    from .records import read_records
    records: list[dict] = []
    for path in a.paths:
        records += read_records(path / "records.jsonl" if path.is_dir() else path)
    text = render(records)
    print(text)
    if a.markdown:
        with open(a.markdown, "x") as f:
            f.write(text + "\n")
    return 0
