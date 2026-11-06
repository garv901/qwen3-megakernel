"""B5 analysis. Usage: analyze.py RESULTS_DIR JOB [JOB2 ...] -> tables, fits, gate, params.json (next to this file).
Metrics (per epilogue phase, mean over eligible CTAs and phases 2..nph; us):
  loop = max over epilogue warps of end of the load/store loop - min over warps of start (start = after accfull wait)
  vis  = max end after fence.acq_rel.gpu (+ fence.proxy.async.global in mode 4) - same start  (output written and visible, writers fence)
  join = end of the join __syncthreads (mode 5: + per-CTA argmax merge) - same start
  dT   = Tmean(config) - Tmean(null epilogue, same TT, orc 1): in-situ increment of the phase time (telescoping mean of barrier exits)."""
import glob, json, os, sys
import numpy as np
R = sys.argv[1]; jobs = sys.argv[2:]
here = os.path.dirname(os.path.abspath(__file__))
rows = []
for j in jobs:
    for f in sorted(glob.glob(f"{R}/*_{j}.jsonl")):
        if os.path.basename(f).startswith("quick"): continue
        pid = os.path.basename(f).split("_")[0]
        for l in open(f):
            if l.startswith("{"):
                x = json.loads(l); x["jobid"] = j; x["pid"] = pid; rows.append(x)
key = lambda x: (x["mode"], x["TT"], x["T"], x["nw"], x["KS"], x["orc"])
def ncfg(TT, T, nw, pair):
    G = nw // 4; NCH = TT // 16; it = 0; ch = 0; tk = 0
    for g in range(G):
        cs = [c for c in range(g, NCH, G) if c * 16 < T]
        ch = max(ch, len(cs)); tk = max(tk, sum(min(16, T - c * 16) for c in cs))
        it = max(it, len(cs) if pair else (len(cs) + 1) // 2 if True else 0)
    return it, ch, tk
# non-pair: two chunks per iteration only if the second chunk is fully/partly valid; the count above matches the kernel's loop (two = c+G<NCH and cb+16G<T)
def n_iter_nonpair(TT, T, nw):
    G = nw // 4; NCH = TT // 16; best = 0
    for g in range(G):
        c = g; n = 0
        while c < NCH and c * 16 < T:
            n += 1; c += 2 * G
        best = max(best, n)
    return best
grp = {}
for x in rows: grp.setdefault(key(x), []).append(x)
good = {k: v for k, v in grp.items() if k[5] == 0 and all(r["pass"] for r in v)}
bad = [k for k, v in grp.items() if k[5] == 0 and not all(r["pass"] for r in v)]
def avg(k, f): return float(np.mean([r[f] for r in grp[k]]))
def spread(k, f): v = [r[f] for r in grp[k]]; return max(v) - min(v)
print("FAIL configs (orc 0):", bad, " configs:", len(good), " rows:", len(rows), " jobs:", jobs)
print("max maxerr_over_tol (orc 0):", max(r["maxerr_over_tol"] for k, v in good.items() for r in v), "  checked values total:", sum(r["nchk"] for k, v in good.items() for r in v))
print("max pass-to-pass spread of us_loop / us_vis / us_Tmean over configs:", max(spread(k, "us_loop") for k in good), max(spread(k, "us_vis") for k in good), max(spread(k, "us_Tmean") for k in good))
null = {k[1]: avg(k, "us_Tmean") for k in grp if k[5] == 1 and k[3] == 12}
print("null-epilogue Tmean by TT (us):", {t: round(v, 3) for t, v in sorted(null.items())})
MN = {1: "fp32 store", 2: "red.add fp32", 3: "bf16 store", 4: "GU pair act", 5: "argmax (+logits store)"}
Tl = lambda TT: sorted({k[2] for k in good if k[1] == TT})
for m in (1, 3, 4, 5):
    for f in ("us_loop", "us_vis"):
        print(f"\n==== mode {m} {MN[m]}: {f} (us), KS=1; rows TT/T, cols nw 4 / 8 / 12")
        for TT in (16, 32, 48, 64, 96, 128):
            for T in Tl(TT):
                k = [(m, TT, T, nw, 1, 0) for nw in (4, 8, 12)]
                print(f"TT{TT:<4}T{T:<4}" + " ".join(f"{avg(x, f):8.3f}" for x in k))
    print(f"-- mode {m}: dT vs null (us), cols nw 4/8/12, selected: " + "; ".join(f"TT{TT}/T{T}: " + "/".join(f"{avg((m,TT,T,nw,1,0),'us_Tmean')-null[TT]:.2f}" for nw in (4, 8, 12)) for TT, T in ((16, 8), (16, 16), (64, 64), (128, 128))))
print("\n==== mode 2 red.add, nw=12: vis (us) / dT (us) by KS; rows TT/T")
KSl = (1, 2, 4, 8, 12, 24)
print("            loop(KS=1)   " + " ".join(f"vis KS{k:<3}" for k in KSl) + "   | " + " ".join(f"dT KS{k:<3}" for k in KSl))
for TT in (16, 32, 48, 64, 96, 128):
    for T in Tl(TT):
        print(f"TT{TT:<4}T{T:<4} {avg((2,TT,T,12,1,0),'us_loop'):8.3f}    " + " ".join(f"{avg((2,TT,T,12,k,0),'us_vis'):8.3f}" for k in KSl) + "   | " + " ".join(f"{avg((2,TT,T,12,k,0),'us_Tmean')-null[TT]:8.3f}" for k in KSl))
print("\n==== mode 2: loop by nw at KS 1 and 24, TT64/T64 and TT128/T128: loop, vis, dT")
for TT in (64, 128):
    for nw in (4, 8, 12):
        print(f"TT{TT} nw{nw}: " + "  ".join(f"KS{k}: {avg((2,TT,TT,nw,k,0),'us_loop'):.2f}/{avg((2,TT,TT,nw,k,0),'us_vis'):.2f}/{avg((2,TT,TT,nw,k,0),'us_Tmean')-null[TT]:.2f}" for k in (1, 24)))
print("\n==== oracle: mode 1 no-store (orc 2) loop vs plain store loop, nw 12: TT/T  store, nostore")
for TT in (16, 64, 128):
    for T in Tl(TT):
        k2 = (1, TT, T, 12, 1, 2)
        if k2 in grp: print(f"TT{TT} T{T}: {avg((1,TT,T,12,1,0),'us_loop'):.3f} {avg(k2,'us_loop'):.3f}")
# ---------------- fits ----------------
def fit(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    c, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ c
    return c, float(np.sqrt(np.mean(r ** 2))), float(np.max(np.abs(r)))
terms = {}
print("\n==== fits (OLS over all TT, T, nw configs; KS=1 rows)")
for m in (1, 3, 4, 5, 2):
    for f in ("us_loop", "us_vis"):
        ks = [k for k in good if k[0] == m and k[4] == 1 and k[5] == 0]
        X, y = [], []
        for k in ks:
            _, TT, T, nw, _, _ = k
            if m == 4:
                it, ch, tk = ncfg(TT, T, nw, True); n_it = ch
            else:
                n_it = n_iter_nonpair(TT, T, nw); _, ch, tk = ncfg(TT, T, nw, False)
            X.append([1, n_it, tk, T]); y.append(avg(k, f))
        c, rm, mx = fit(X, y)
        print(f"mode {m} {f}: t = {c[0]:.3f} + {c[1]:.3f} * n_iter + {c[2]:.4f} * n_tok_max + {c[3]:.4f} * T   rmse {rm:.3f} max|res| {mx:.3f} (n={len(y)})")
        terms[f"m{m}_{f[3:]}"] = {"formula": "t = a + b*n_iter + c*n_tok_max + d*T (KS=1); n_iter = max over epilogue groups of loop iterations (mode 4: chunks; others: ceil(chunks/2)), n_tok_max = max over groups of valid tokens they process", "coeffs": {"a_us": c[0], "b_us_per_iter": c[1], "c_us_per_token_max_group": c[2], "d_us_per_token_total": c[3]},
                                    "units": "us", "fit_rmse": rm, "fit_max_abs_res": mx, "range": {"TT": [16, 128], "T": [1, 128], "nw": [4, 8, 12], "KS": [1]}, "jobs": jobs, "n_points": len(y)}
# KS dependence of the red.add visible time: vis(KS) - vis(KS=1) per config
print("\n==== mode 2: vis(KS) - vis(1) and dT(KS) - dT(1) vs KS (mean over configs, by n_tok_max bucket)")
d1 = {}; X, y, X2, y2 = [], [], [], []
for k in good:
    m, TT, T, nw, KS, o = k
    if m != 2 or KS == 1: continue
    k1 = (2, TT, T, nw, 1, 0)
    _, ch, tk = ncfg(TT, T, nw, False)
    dv = avg(k, "us_vis") - avg(k1, "us_vis"); dt = avg(k, "us_Tmean") - avg(k1, "us_Tmean")
    X.append([KS - 1, (KS - 1) * T]); y.append(dv); X2.append([KS - 1, (KS - 1) * T]); y2.append(dt)
    d1.setdefault((KS, "big" if T >= 64 else "small"), []).append((dv, dt))
for (KS, b), v in sorted(d1.items()): print(f"KS {KS:<3} T{'>=64' if b == 'big' else '<64 '}: dvis {np.mean([a for a, _ in v]):.3f}  dT {np.mean([a for _, a in v]):.3f}  (n={len(v)})")
c, rm, mx = fit(X, y); print(f"dvis = {c[0]:.4f}*(KS-1) + {c[1]:.5f}*(KS-1)*T  rmse {rm:.3f}")
terms["m2_dvis_KS"] = {"formula": "vis(KS) - vis(1) = a*(KS-1) + b*(KS-1)*T (nw 4/8/12, all TT)", "coeffs": {"a_us": c[0], "b_us_per_token": c[1]}, "units": "us", "fit_rmse": rm, "range": {"KS": [1, 24], "T": [1, 128]}, "jobs": jobs}
c, rm, mx = fit(X2, y2); print(f"dT   = {c[0]:.4f}*(KS-1) + {c[1]:.5f}*(KS-1)*T  rmse {rm:.3f}")
terms["m2_dT_KS"] = {"formula": "Tmean(KS) - Tmean(1) = a*(KS-1) + b*(KS-1)*T", "coeffs": {"a_us": c[0], "b_us_per_token": c[1]}, "units": "us", "fit_rmse": rm, "range": {"KS": [1, 24], "T": [1, 128]}, "jobs": jobs}
# in-situ dT fit: dT = a + b*vis
X, y = [], []
for k in good:
    if k[0] in (1, 3, 4, 5) and k[4] == 1: X.append([1, avg(k, "us_vis")]); y.append(avg(k, "us_Tmean") - null[k[1]])
c, rm, mx = fit(X, y); print(f"\ndT(in-situ phase increment) = {c[0]:.3f} + {c[1]:.3f} * vis   rmse {rm:.3f}  (modes 1,3,4,5)")
terms["dT_from_vis"] = {"formula": "dT = a + b*us_vis (modes 1,3,4,5, KS 1)", "coeffs": {"a_us": c[0], "b": c[1]}, "units": "us", "fit_rmse": rm, "range": {"us_vis": [0.6, 15]}, "jobs": jobs}
# gate
print("\n==== gate: mode 4, TT 16, T 8")
for nw in (4, 8, 12):
    k = (4, 16, 8, nw, 1, 0); print(f"nw {nw}: loop {avg(k,'us_loop'):.3f} vis {avg(k,'us_vis'):.3f} join {avg(k,'us_join'):.3f}  dT {avg(k,'us_Tmean')-null[16]:.3f}  mhz {avg(k,'mhz'):.0f}")
print("mk GU compute (st_14155 b8, TS busy full 5.95 - T_null busy 2.62) = 3.33 us (computed by probe/mk_core/smallt/analyze.py)")
print("clocks mhz by mode (median over configs):", {m: round(float(np.median([avg(k, 'mhz') for k in good if k[0] == m])), 0) for m in (1, 2, 3, 4, 5)})
json.dump({"b5_epi": terms}, open(os.path.join(here, "params.json"), "w"), indent=1)
