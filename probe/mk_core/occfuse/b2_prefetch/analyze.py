"""B2 analysis. Usage: analyze.py RESULTS_DIR TAG:JOB [TAG:JOB ...]   (e.g. gate1:14287 gen_c:14290)
Rows: RESULTS_DIR/<tag>_<job>.jsonl. Busy work = us_busymean (mean over stream phases of the median over busy CTAs of arrive - previous exit, as mk).
exposed = busy - floor (floor = same config, S 0, same W/spread). win = us_winmean = median over busy CTAs of exit - arrive of the spin barrier = time from issue to exit."""
import json, os, sys
import numpy as np
R = sys.argv[1]; rows = []
for a in sys.argv[2:]:
    t, j = a.split(":"); f = f"{R}/{t}_{j}.jsonl"
    for l in open(f):
        if l.startswith("{"): x = json.loads(l); x["src"] = a; rows.append(x)
print("rows", len(rows))
def key(x): return (x["name"], x["mode"], x["tot"], x["spt"], x["R"], x["W_us"], x["spread_us"])
floor = {}
for x in rows:
    if x["arm"] == "floor": floor.setdefault((x["name"], x["mode"], x["tot"], x["R"], x["W_us"], x["spread_us"]), []).append(x["us_busymean"])
def fl(x): v = floor.get((x["name"], x["mode"], x["tot"], x["R"], x["W_us"], x["spread_us"])); return float(np.mean(v)) if v else float("nan")
TARGET = {"cQKV": 0.69, "cGU": 1.48, "cD": 0.49, "tQKV": 0.77, "tGU": 2.64, "tD": 0.83}   # recomputed from st_14155 C_null / T_null (smallt/analyze.py definitions)
MKWIN = {"cQKV": 1.57, "cGU": 1.14, "cD": 1.80, "tQKV": 1.60, "tGU": 1.30, "tD": 2.64}      # median exit-arrive of the preceding barrier, busy CTAs
MKPREVT = {"cQKV": 2.06, "cGU": 1.50, "cD": 3.50, "tQKV": 2.19, "tGU": 1.37, "tD": 4.21}    # preceding phase T (null runs)
print("\n== gate (mean over passes). P0 = no prefetch, PR = P = R. busy in us, win = measured window")
for name in TARGET:
    g = [x for x in rows if x["arm"] == "gate" and x["name"] == name]
    if not g: continue
    print(f"\n{name}: target {TARGET[name]}  (mk window {MKWIN[name]}, preceding T {MKPREVT[name]}; N={g[0]['N']} nt_max={g[0]['nt_max']} spt={g[0]['spt']} R={g[0]['R']})")
    print("  W   spread |  win  |  P0 busy | P1 | P2 | PR busy | floor")
    ks = sorted({(x["W_us"], x["spread_us"]) for x in g})
    for W, s in ks:
        sel = [x for x in g if x["W_us"] == W and x["spread_us"] == s and x["R"] == g[0]["R"]]
        d = {}
        for x in sel: d.setdefault(x["P"], []).append(x["us_busymean"])
        win = np.mean([x["us_winmean"] for x in sel]); Rn = g[0]["R"]
        f = lambda p: f"{np.mean(d[p]):6.2f}" if p in d else "   -  "
        print(f"  {W:4.1f} {s:5.2f} | {win:5.2f} | {f(0)}   | {f(1)} | {f(2)} | {f(Rn)}  | {np.nanmean([fl(x) for x in sel]):.2f}")

# ---- stream sweep: hidden(P, window) and fit ----
B1 = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../b1_stream/params.json")))["t_stream"]["coeffs"]
def b1par(Rr):
    i = B1["R"].index(Rr) if Rr in B1["R"] else None
    if i is None: i = B1["R"].index(min(B1["R"], key=lambda q: abs(q - Rr)))
    return B1["t0_us"][i], B1["r_sm_GBps"][i], B1["BW_agg_GBps"][i]
def tB1(S, N, Rr): t0, r, bw = b1par(Rr); return t0 + S * 1024 / (min(r, bw / N) * 1e3)
st = [x for x in rows if x["arm"] == "stream"]
print("\n== stream rows", len(st))
idx = {}
for x in st: idx[(x["mode"], x["N"], x["S_kb"], x["R"], x["P"], x["W_us"], x["spread_us"])] = x
# W (all-CTA spin) has no effect: max |busy(W) - busy(W=0)| at spread 0
dW = [abs(idx[k]["us_busymean"] - idx[k[:5] + (0.0, 0.0)]["us_busymean"]) for k in idx if k[6] == 0.0 and k[5] > 0 and k[:5] + (0.0, 0.0) in idx]
print("all-CTA spin W vs W=0 (spread 0): median |d busy| %.3f, max %.3f us (n=%d)" % (np.median(dW), np.max(dW), len(dW)))
print("window (us_winmean) vs spread at W=0: ", {s: round(float(np.median([x["us_winmean"] for x in st if x["spread_us"] == s and x["W_us"] == 0])), 2) for s in sorted({x["spread_us"] for x in st})})
SLOT = lambda m: 32768  # A bytes per slot
fitout = {}
for mode in (0, 1):
    for cls, (lo, hi) in {"win<1.3": (0, 1.3), "1.3-2.0": (1.3, 2.0), ">=2.0": (2.0, 99)}.items():
        X, Y = [], []
        for k, x in idx.items():
            m_, N, S, Rr, P, W, sp = k
            if m_ != mode or P == 0 or not lo <= x["us_winmean"] < hi: continue
            b0 = idx.get((m_, N, S, Rr, 0, W, sp))
            if not b0: continue
            t0, r, bw = b1par(Rr)
            X.append(P * 32768 / (min(r, bw / N) * 1e3)); Y.append(b0["us_busymean"] - x["us_busymean"])
        if len(X) < 5: continue
        X, Y = np.array(X), np.array(Y); a, b = np.polyfit(X, Y, 1)
        rm = float(np.sqrt(np.mean((a * X + b - Y) ** 2)))
        # also hidden = h * X through the origin
        h = float(np.sum(X * Y) / np.sum(X * X)); rm0 = float(np.sqrt(np.mean((h * X - Y) ** 2)))
        print(f"mode {mode} {cls:8}: hidden = {a:.3f} * P*slot/rate(N,R) + {b:.3f}  rmse {rm:.3f} | through origin slope {h:.3f} rmse {rm0:.3f}  (n={len(X)})")
        fitout[f"mode{mode}|{cls}"] = dict(slope=float(a), intercept_us=float(b), rmse_us=rm, slope_origin=h, rmse_origin_us=rm0, n=len(X))
json.dump(fitout, open(f"{R}/fit_hidden.json", "w"), indent=1)
print("\n== exposed vs B1 t_stream: classic, spread 0 (win ~1.0) and spread 2 (win ~2.6), R=4: N x S, busy(P) / B1 t(S)")
for N in (48, 148):
    for S in (64, 128, 256, 512, 1024):
        row = []
        for sp in (0.0, 2.0):
            row.append("sp%g: " % sp + " ".join("%.2f" % idx[(0, N, S, 4, P, 0.0, sp)]["us_busymean"] for P in (0, 1, 2, 4) if (0, N, S, 4, P, 0.0, sp) in idx))
        print(f"N={N:3} S={S:4}  B1 {tB1(S, N, 4):6.2f} | " + " | ".join(row))
