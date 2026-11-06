"""Fit t_fin (mean over phases, var 0, pdelay 1500) per placement; write params.json. Usage: fit.py JSONL..."""
import json, sys, math, collections
import numpy as np
rows = []
for f in sys.argv[1:]:
    for l in open(f):
        try: rows.append(json.loads(l))
        except Exception: pass
jobs = sorted({int(r["job"]) for r in rows})
sel = [r for r in rows if r["var"] == 0 and r["pdelay_ns"] == 1500 and r["place"] in (1, 2, 3) and r["pass"]]
def feats(r):
    T, nidle = r["T"], 148 - r["Pc"]; NTT = (T + 127) // 128; NJOB = r["NB"] * NTT
    nit = math.ceil(min(T, 128) / 32)
    if r["place"] == 1: m = math.ceil(NJOB / nidle)
    elif r["place"] == 2: m = NTT
    else: m = 1; nit = 1
    return nit, m
out = {}
for pl, nm in ((1, "a_idle"), (2, "b_lastprod"), (3, "c_all")):
    R = [r for r in sel if r["place"] == pl]
    y = np.array([r["t_fin_mean"] for r in R])
    if pl < 3:
        X = np.array([[1, f[0], (f[1] - 1) * f[0], (f[1] - 1)] for f in map(feats, R)], float); names = ["a_us", "b_us_per_32tok_iter", "extra_job_us_per_iter", "extra_job_us"]
    else:
        X = np.array([[1, min(r["T"], 128)] for r in R], float); names = ["a_us", "b_us_per_token(T<=128)"]
    c, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ c
    # components (mean over rows): obs, rc, wf
    comp = {k: float(np.mean([r[k] for r in R])) for k in ("t_obs_mean", "t_rc_mean", "t_wf_mean")}
    out[nm] = dict(formula="t_fin = " + " + ".join(f"{n}*x{i}" for i, n in enumerate(names)) + "; a/b: x = [1, n_it, (m-1)*n_it, (m-1)], n_it = ceil(min(T,128)/32), m = jobs per finalizing CTA (a: ceil(NJOB/nidle), b: ceil(T/128)); c: x = [1, min(T,128)]; NJOB = (H/128)*ceil(T/128)",
                   coeffs=dict(zip(names, map(float, c))), units="us", fit_rmse=float(np.sqrt(np.mean(res ** 2))), n=len(y), max_abs_res=float(np.abs(res).max()), mean_components=comp,
                   range=dict(T=[1, 256], H=[1024, 2048], KS=[4, 24], pdelay_ns=1500), jobs=jobs)
    print(nm, {k: round(v, 3) for k, v in out[nm]["coeffs"].items()}, "rmse %.3f maxres %.3f n %d" % (out[nm]["fit_rmse"], out[nm]["max_abs_res"], len(y)), comp)
# empty-arm floor (ticket observe + fence), var 1, by placement
E = {}
for pl, nm in ((1, "a_idle"), (2, "b_lastprod"), (3, "c_all")):
    R = [r for r in rows if r["var"] == 1 and r["place"] == pl and r["pdelay_ns"] == 1500 and (r["T"] <= 128)]
    E[nm] = float(np.mean([r["t_fin_mean"] for r in R])); print("empty floor", nm, round(E[nm], 3), "n", len(R))
out["t_empty_floor"] = dict(formula="t_fin of the empty finalizer arm (poll/ticket-observe + fence), T <= 128, mean over H, KS", coeffs=E, units="us", fit_rmse=None, range=dict(T=[1, 128]), jobs=jobs)
json.dump(out, open("params.json", "w"), indent=1)
