"""B0 analysis. Usage: analyze.py RESULTS_DIR JOB   -> tables, fits, gate, params.json (written next to this file).
Reads run{1,2}_JOB.jsonl, c_idle/c_busy jsonl, nvml csv; uses ../common/stamps.py for the raw-stamp cross-check."""
import glob, json, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
import stamps

R, job = sys.argv[1], sys.argv[2]
here = os.path.dirname(os.path.abspath(__file__))
def rows(f):
    return [json.loads(l) for l in open(f) if l.startswith("{")] if os.path.exists(f) else []
runs = {r: rows(f"{R}/run{r}_{job}.jsonl") for r in (1, 2)}
def sel(rs, **kw): return [x for x in rs if all(x.get(k) == v for k, v in kw.items())]
# Raw-stamp means: per-phase medians are quantized (T, lat in multiples of 0.256 us: medians of exit stamps land on a 256 ns grid even though stamps step by 32 ns).
# So us_T / us_lat below are means over phases from the stored raw stamps (T = (med exit(last) - med exit(1)) / (nph - 1), telescoping); *_med keep the JSONL medians.
for r, rs in runs.items():
    f = f"{R}/run{r}_{job}.st"
    for x in rs:
        x["us_T_med"], x["us_lat_med"] = x.get("us_T"), x.get("us_lat")
        if x.get("st_n", 0) and os.path.exists(f):
            st = stamps.load(f, 148, x["nstmax"]); Ts, Ls = [], []
            for L in st[x["st_off"]:x["st_off"] + x["st_n"]]:
                arr, ex = L[:, 0::2], L[:, 1::2]; n = x["nph"]
                Ts.append((np.median(ex[:, n]) - np.median(ex[:, 1])) / 1e3 / (n - 1))
                Ls.append(np.mean([(np.median(ex[:, p]) - arr[:, p].max()) / 1e3 for p in range(2, n + 1)]))
            x["us_T"], x["us_lat"] = float(np.mean(Ts)), float(np.mean(Ls))
Ns = sorted({x["N"] for x in runs[1] if x["arm"] == "a"})
Ws = sorted({x["W_ns"] for x in runs[1] if x["arm"] == "a"})

print("== gtstep"); [print({k: x[k] for k in x if k.startswith(("step", "distinct"))}) for x in sel(runs[1], arm="gtstep")]
for sp, name in ((0, "globaltimer spin"), (1, "clock64 spin")):
    for r in (1, 2):
        print(f"\n== arm a, {name}, run {r}: us_T / us_lat per N (rows) x W (cols)")
        print("N\\W   " + " ".join(f"{w/1e3:>12.1f}" for w in Ws))
        for n in Ns:
            c = []
            for w in Ws:
                x = sel(runs[r], arm="a", N=n, W_ns=w, spin=sp)
                c.append(f"{x[0]['us_T']:6.3f}/{x[0]['us_lat']:5.3f}" if x else "     -")
            print(f"{n:<5} " + " ".join(f"{s:>12}" for s in c))

# fit T = t_bar(N) + a*W per N and spin mode (all W>=0 pooled over both runs); residual vs W + T(W=0)
fits = {}
print("\n== fit T = t_bar + s*W (pooled runs), residual of T - (W + T(0)) max over W")
for sp in (0, 1):
    for n in Ns:
        W, T, L = [], [], []
        for r in (1, 2):
            for x in sel(runs[r], arm="a", N=n, spin=sp): W.append(x["W_ns"] / 1e3); T.append(x["us_T"]); L.append(x["us_lat"])
        if len(W) < 3: continue
        A = np.vstack([np.ones(len(W)), W]).T; co, *_ = np.linalg.lstsq(A, T, rcond=None); rm = float(np.sqrt(np.mean((A @ co - T) ** 2)))
        t0 = np.median([t for w, t in zip(W, T) if w == 0]); res = max(abs(t - (w + t0)) for w, t in zip(W, T))
        fits[(sp, n)] = (co[0], co[1], rm, res, float(np.median(L)))
        print(f"spin={sp} N={n:<4} t_bar={co[0]:.3f} slope={co[1]:.3f} rmse={rm:.3f}  max|T-(W+T0)|={res:.3f}  lat_med={np.median(L):.3f}")

print("\n== lat at W=0 / W=1 us per run, N=148, spin gt")
for r in (1, 2):
    for w in (0, 1000):
        x = sel(runs[r], arm="a", N=148, W_ns=w, spin=0)
        if x: print(f"run{r} W={w} lat={x[0]['us_lat']:.3f} T={x[0]['us_T']:.3f} reps={x[0]['us_reps']}")

print("\n== arm b (TMA window loads): us_T / us_lat; delta of lat vs nld=0 same (N,W)")
for r in (1, 2):
    print(f"run {r}")
    for n in sorted({x["N"] for x in runs[r] if x["arm"] == "b"}):
        for w in Ws:
            base = sel(runs[r], arm="a", N=n, W_ns=w, spin=0)
            c = []
            for nld in (1, 2, 3, 4):
                x = sel(runs[r], arm="b", N=n, W_ns=w, nld=nld)
                if x: c.append(f"{x[0]['us_T']:6.3f}/{x[0]['us_lat']:5.3f}")
            print(f"N={n:<4} W={w/1e3:3.1f}  nld0 " + (f"{base[0]['us_T']:6.3f}/{base[0]['us_lat']:5.3f}" if base else "-") + "  nld1-4 " + "  ".join(c))

print("\n== placement first vs spread at W=1 us (T/lat)")
for n in Ns:
    f = sel(runs[1], arm="a_first", N=n); s = sel(runs[1], arm="a", N=n, W_ns=1000, spin=0)
    if f and s: print(f"N={n:<4} first {f[0]['us_T']:.3f}/{f[0]['us_lat']:.3f}  spread {s[0]['us_T']:.3f}/{s[0]['us_lat']:.3f}")

print("\n== launch/phase vs T check (>5% flagged)")
bad = 0; tot = 0
for r in (1, 2):
    for x in runs[r]:
        if x.get("arm") in ("a", "b", "a_first"):
            tot += 1
            if abs(x["us_launch_per_phase"] / x["us_T"] - 1) > 0.05: bad += 1
print(f"{bad}/{tot} configs differ >5%")
for x in sel(runs[1], arm="a", N=148, spin=0): print(f"W={x['W_ns']/1e3} T={x['us_T']:.3f} launch/phase={x['us_launch_per_phase']:.3f}")

print("\n== raw-stamp cross-check (stamps.py), run1 first stored launch")
x = sel(runs[1], arm="a", N=148, W_ns=0, spin=0)
if x and os.path.exists(f"{R}/run1_{job}.st"):
    st = stamps.load(f"{R}/run1_{job}.st", 148, x[0]["nstmax"]); L = x[0]["st_off"]
    print(stamps.metrics(st[L:L + 1], x[0]["nph"]))
    ex = st[L][:, 1::2].ravel(); d = np.diff(np.sort(np.unique(ex))); print("min nonzero stamp step ns:", int(d[d > 0].min()), " distinct exit stamps:", len(np.unique(ex)))

print("\n== clock arms")
for f in sorted(glob.glob(f"{R}/c_*_{job}.jsonl")) + sorted(glob.glob(f"{R}/c_idle_*_{job}.jsonl")) + sorted(glob.glob(f"{R}/c_busy_*_{job}.jsonl")):
    pass
seen = set()
for f in sorted(glob.glob(f"{R}/c_idle_*_{job}.jsonl") + glob.glob(f"{R}/c_busy_*_{job}.jsonl")):
    for x in rows(f):
        print({k: x[k] for k in ("arm", "tag", "N", "W_ns", "reps", "launches", "us_T", "mhz", "mhz_min", "mhz_max", "mhz_first10pct", "mhz_last10pct") if k in x})
for f in sorted(glob.glob(f"{R}/nvml_*_{job}.csv")):
    v = [int(l.split(",")[1].split()[0]) for l in open(f) if len(l.split(",")) > 3 and l.split(",")[1].strip().split()[0].isdigit()]
    p = [float(l.split(",")[3].split()[0]) for l in open(f) if len(l.split(",")) > 3 and l.split(",")[3].strip().split()[0].replace(".", "").isdigit()]
    if v: print(os.path.basename(f), "n=", len(v), "sm MHz med/min/max", int(np.median(v)), min(v), max(v), " power W med/max", np.median(p) if p else None, max(p) if p else None)

# params.json
t = {n: fits[(0, n)] for n in Ns if (0, n) in fits}
out = {"t_bar": {"formula": "T(N,W) = t_bar(N) + s*W (globaltimer spin, arm a); t_bar_us = coeffs.t_bar_us[N]", "coeffs": {"N": list(t), "t_bar_us": [round(v[0], 4) for v in t.values()], "slope": [round(v[1], 4) for v in t.values()]},
       "units": "us", "fit_rmse": max(v[2] for v in t.values()) if t else None, "range": {"N": [min(Ns), max(Ns)], "W_us": [0, 4]}, "jobs": [int(j) for j in sys.argv[2:]]}}
json.dump(out, open(os.path.join(here, "params.json"), "w"), indent=1)

# window-load terms (arm b, N=148, W=0, both runs): extra T and lat vs nld
pts = [(x["nld"], x["us_T"] - np.mean([y["us_T"] for y in sel(runs[r], arm="a", N=148, W_ns=0, spin=0)]), x["us_lat"]) for r in (1, 2) for x in sel(runs[r], arm="b", N=148, W_ns=0)]
if pts:
    n = np.array([p[0] for p in pts], float); dT = np.array([p[1] for p in pts]); lt = np.array([p[2] for p in pts]); A = np.vstack([np.ones(len(n)), n]).T
    cT, *_ = np.linalg.lstsq(A, dT, rcond=None); cL, *_ = np.linalg.lstsq(A, lt, rcond=None)
    print("\n== window loads, N=148 W=0: dT = a + b*nld:", cT, "rmse", np.sqrt(np.mean((A @ cT - dT) ** 2)), " lat = a + b*nld:", cL, "rmse", np.sqrt(np.mean((A @ cL - lt) ** 2)))
    out["t_window_exposed"] = {"formula": "extra T vs empty phase = a + b*nld (nld x 32 KB loads issued in the barrier window, N=148, W=0, drained before the next arrive)", "coeffs": {"a_us": round(float(cT[0]), 4), "b_us_per_load": round(float(cT[1]), 4)}, "units": "us", "fit_rmse": float(np.sqrt(np.mean((A @ cT - dT) ** 2))), "range": {"nld": [1, 4]}, "jobs": [int(j) for j in sys.argv[2:]]}
    out["lat_bar"] = {"formula": "lat (median exit - max arrive) = a + b*nld", "coeffs": {"a_us": round(float(cL[0]), 4), "b_us_per_load": round(float(cL[1]), 4), "lat_nld0_us": round(float(np.mean([x["us_lat"] for r in (1, 2) for x in sel(runs[r], arm="a", N=148, W_ns=0, spin=0)])), 4)}, "units": "us", "fit_rmse": float(np.sqrt(np.mean((A @ cL - lt) ** 2))), "range": {"nld": [0, 4]}, "jobs": [int(j) for j in sys.argv[2:]]}
cl = {}
for f in glob.glob(f"{R}/c_*_{job}.jsonl"):
    for x in rows(f): cl.setdefault(x["arm"], []).append(x["mhz"])
if cl: out["clock_mhz"] = {"formula": "clk_sample MHz (median over launches)", "coeffs": {k: round(float(np.mean(v)), 1) for k, v in cl.items()}, "units": "MHz", "fit_rmse": None, "range": {"W": "idle spin 3us / 48 tcgen05 MMAs per phase"}, "jobs": [int(j) for j in sys.argv[2:]]}
out["gtstep"] = {"formula": "globaltimer step", "coeffs": {"step_ns": 32, "median_quantization_ns": 256}, "units": "ns", "fit_rmse": None, "range": {}, "jobs": [int(j) for j in sys.argv[2:]]}
json.dump(out, open(os.path.join(here, "params.json"), "w"), indent=1)
