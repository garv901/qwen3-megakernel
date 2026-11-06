"""B6 analysis. Usage: analyze.py JSONL [JSONL ...]  (rows from several jobs are pooled per config; also prints per-job gate lines).
Tables: t_fin / obs / rc / wf by (place, var=0) and T,H,KS; decomposition by arms; gate; fit t_fin = a + b*T + c*T*(H/1024) per placement."""
import json, sys, collections
import numpy as np
rows = []
for f in sys.argv[1:]:
    for l in open(f):
        try: rows.append(json.loads(l))
        except Exception: pass
key = lambda r: (r["place"], r["var"], r["T"], r["H"], r["KS"], r["pdelay_ns"])
by = collections.defaultdict(list)
for r in rows: by[key(r)].append(r)
def m(k, f):
    v = [r[f] for r in by.get(k, [])]
    return float(np.mean(v)) if v else float("nan")
bad = [r for r in rows if not r["pass"]]
print(f"rows {len(rows)}, FAIL rows {len(bad)}", [(r['arm'], r['var'], r['T'], r['H'], r['KS']) for r in bad][:10])
PD = 1500
print("\n== t_fin (var 0) us: place, H, KS, then T = 1 8 16 32 64 128 256 ==")
Ts = [1, 8, 16, 32, 64, 128, 256]
fit = {}
for pl, nm in ((1, "a_idle"), (2, "b_lastprod"), (3, "c_all")):
    xs, ys = [], []
    for H in (1024, 2048):
        for KS in (4, 12, 16, 24):
            v = [m((pl, 0, T, H, KS, PD), "t_fin_mean") for T in Ts]
            if all(x != x for x in v): continue
            print(f"{nm:10s} H{H} KS{KS:2d} " + " ".join(f"{x:6.2f}" for x in v))
            for T, x in zip(Ts, v):
                if x == x: xs.append((1, T, T * H / 1024, KS)); ys.append(x)
    if xs:
        X = np.array(xs, float); y = np.array(ys)
        A = X[:, :3]; c, *_ = np.linalg.lstsq(A, y, rcond=None); res = y - A @ c
        fit[nm] = dict(coeffs=dict(a_us=c[0], b_us_per_T=c[1], c_us_per_T_per_H1024=c[2]), fit_rmse=float(np.sqrt(np.mean(res ** 2))), n=len(y))
        print(nm, "fit t_fin = a + b*T + c*T*(H/1024):", {k: round(v, 4) for k, v in fit[nm]["coeffs"].items()}, "rmse", round(fit[nm]["fit_rmse"], 3), "n", len(y))
print("\n== decomposition (mean over phases, crit CTA): obs / read+convert+store-issue / fence ; arms: full(0), empty(1), noStore(2) t_fin ==")
for pl, nm in ((1, "a_idle"), (2, "b_lastprod"), (3, "c_all")):
    for H, KS in ((1024, 16), (1024, 12), (2048, 4)):
        print(f"{nm} H{H} KS{KS}")
        for T in Ts:
            k0, k1, k2 = (pl, 0, T, H, KS, PD), (pl, 1, T, H, KS, PD), (pl, 2, T, H, KS, PD)
            if k0 not in by: continue
            print(f"  T{T:3d} full {m(k0,'t_fin_mean'):5.2f} [obs {m(k0,'t_obs_mean'):5.2f} rc {m(k0,'t_rc_mean'):5.2f} wf {m(k0,'t_wf_mean'):5.2f}] empty {m(k1,'t_fin_mean'):5.2f} nostore {m(k2,'t_fin_mean'):5.2f}"
                  f"  => floor(empty) {m(k1,'t_fin_mean'):.2f}, read+conv {m(k2,'t_fin_mean')-m(k1,'t_fin_mean'):.2f}, write {m(k0,'t_fin_mean')-m(k2,'t_fin_mean'):.2f}")
print("\n== gate: H1024 NB8 KS16 T8, place a vs base: us_work_last (mk 'last' = max over CTAs of work) ==")
for pd in sorted({r["pdelay_ns"] for r in rows}):
    for jb in sorted({r["job"] for r in rows}):
        b = [r for r in rows if r["job"] == jb and r["pdelay_ns"] == pd and r["place"] == 0 and r["T"] == 8 and r["H"] == 1024 and r["KS"] == 16]
        a = [r for r in rows if r["job"] == jb and r["pdelay_ns"] == pd and r["place"] == 1 and r["var"] == 0 and r["T"] == 8 and r["H"] == 1024 and r["KS"] == 16]
        if a and b:
            la = np.mean([r["us_work_last"] for r in a]); lb = np.mean([r["us_work_last"] for r in b]); ta = np.mean([r["t_fin_mean"] for r in a])
            print(f"job {jb} pdelay {pd}: last(a) {la:.2f} last(base) {lb:.2f} delta {la-lb:+.2f} (target +1.8 +- 0.4)  t_fin {ta:.2f}  lastfin_frac {np.mean([r['lastfin_frac'] for r in a]):.2f}  n={len(a)}/{len(b)}")
json.dump(fit, open("fit_raw.json", "w"), indent=1)
