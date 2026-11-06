#!/usr/bin/env python3
"""why: summary + E1 attribution + E2 die/home. Usage: $PY analyze.py <job>   (reads results/2026-10-28-why/{why,dump,lat}_<job>.*)"""
import json, sys, os, numpy as np, statistics as st
J = sys.argv[1]; R = os.path.join(os.path.dirname(__file__), "../../../results/2026-10-28-why")
rows = [json.loads(l) for l in open(f"{R}/why_{J}.jsonl")]
arms = []
for r in rows:
    if r["arm"] not in arms: arms.append(r["arm"])
print(f"{'arm':12} {'make us':>8} {'tail':>6} {'sprd':>6} {'p50':>7} {'GB/s':>6} {'B/SM MB':>13}")
for a in arms:
    rs = [r for r in rows if r["arm"] == a]; m = lambda k: st.median(r[k] for r in rs)
    print(f"{a:12} {m('makespan_ns')/1e3:8.1f} {m('tail_ns')/1e3:6.2f} {m('spread_ns')/1e3:6.2f} {m('p50_ns')/1e3:7.1f} {m('bw_gbs'):6.0f} {m('bytes_min')/1e6:6.2f}-{m('bytes_max')/1e6:.2f}  n={len(rs)}")

meta, reps = {}, {}
for l in open(f"{R}/dump_{J}.jsonl"):
    d = json.loads(l)
    if d["type"] == "meta": meta[d["arm"]] = d
    else: reps.setdefault(d["arm"], []).append({k: np.array(v) for k, v in d.items() if isinstance(v, list)})

def r2(y, groups_list):
    """R^2 of additive least squares y ~ sum of categorical factors (sparse-free dense dummies)."""
    cols = []
    for g in groups_list:
        u, inv = np.unique(g, return_inverse=True)
        D = np.zeros((len(y), len(u))); D[np.arange(len(y)), inv] = 1; cols.append(D)
    X = np.hstack(cols + [np.ones((len(y), 1))])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()

print("\n== per-SM normalized throughput (SM bytes / SM finish; imbalance-free): median over SMs and reps ==")
for a in arms:
    if a not in reps: continue
    ln = np.array(meta[a]["len"]) * 16.0; v = []; agg = []
    for rp in reps[a]:
        fin = {}; by = {}
        for i in range(len(ln)):
            s_ = rp["smid"][i]; fin[s_] = max(fin.get(s_, 0), rp["t1"][i]); by[s_] = by.get(s_, 0) + ln[i]
        v += [by[k] / fin[k] for k in fin]; agg.append(sum(by.values()) / max(fin.values()))
    print(f"{a:20} per-SM rate median {np.median(v):.2f} GB/s p5 {np.percentile(v,5):.2f} p95 {np.percentile(v,95):.2f}  => x148 = {np.median(v)*148/1e3:.2f} TB/s")
print("\n== E1 attribution: chunk duration ~ SM + address region (pooled reps; R^2) ==")
for a in arms:
    if a not in reps or meta[a]["n"] < 1000 or a.endswith(("_sh1","_sh2","_sh8")): continue
    off = np.array(meta[a]["off"]) * 16.0  # bytes
    Y, S, P, G, Rg = [], [], [], [], []
    for i, rp in enumerate(reps[a]):
        Y.append(rp["t1"] - rp["t0"]); S.append(rp["smid"]); P.append((off // (2 << 20)).astype(int)); G.append((off // (32 << 20)).astype(int))
    Y, S, P, G = map(np.concatenate, (Y, S, P, G)); Y = Y.astype(float)
    # per-SM rate spread: bytes/time per SM
    ln = np.tile(np.array(meta[a]["len"]) * 16.0, len(reps[a]))
    T0 = np.concatenate([rp["t0"] for rp in reps[a]]); TB = (T0 // 8000).astype(int)
    print(f"{a:12} R2 start-time(8us bin) {r2(Y,[TB]):.3f}  time+SM+2M {r2(Y,[TB,S,P]):.3f}")
    print(f"{a:12} R2 SM {r2(Y,[S]):.3f}  page2M {r2(Y,[P]):.3f}  region32M {r2(Y,[G]):.3f}  SM+32M {r2(Y,[S,G]):.3f}  SM+2M {r2(Y,[S,P]):.3f}")
    # per-SM mean duration spread (coefficient of variation of SM-mean duration)
    sm_means = np.array([Y[S == s].mean() for s in np.unique(S)])
    print(f"{'':12} per-SM mean chunk dur: min {sm_means.min()/1e3:.2f} max {sm_means.max()/1e3:.2f} cv {sm_means.std()/sm_means.mean()*100:.2f}%  overall mean {Y.mean()/1e3:.2f} us")

print("\n== E1 rotation: static share s on CTA (s+R)%148; does slowness follow share or SM? ==")
Rots = [("static", 0), ("static_rot37", 37), ("static_rot74", 74), ("static_rot111", 111)]
T, SM = [], []
for a, R_ in Rots:
    if a not in reps: continue
    T.append(np.mean([rp["t1"] for rp in reps[a]], axis=0)); SM.append(np.array([st.mode(rp["smid"][s] for rp in reps[a]) for s in range(148)]))
T = np.array(T)  # [rot][share] finish ns (t1 rel to min t0, t0~0)
print("rep-averaged finish per share: mean %.1f us  sd %.2f us" % (T.mean()/1e3, T.std()/1e3))
print("corr(share finish) between rotations (follow-address -> high):")
print(np.round(np.corrcoef(T), 2))
# SM follow: reindex by SM
Tsm = np.zeros_like(T)
for k in range(len(T)):
    for s in range(148): Tsm[k, SM[k][s]] = T[k][s]
print("corr(SM finish) between rotations (follow-SM -> high):")
print(np.round(np.corrcoef(Tsm), 2))
y = T.ravel(); sh = np.tile(np.arange(148), len(T)); sm = np.concatenate([SM[k] for k in range(len(T))])
print("R2 finish ~ share: %.3f  ~ SM: %.3f  ~ share+SM: %.3f" % (r2(y, [sh]), r2(y, [sm]), r2(y, [sh, sm])))

# E2 -----------------------------------------------------------------
lf = f"{R}/lat_{J}.u16"
if os.path.exists(lf):
    L = np.fromfile(lf, dtype=np.uint16).reshape(3, 256, -1).astype(float)
    sms = [s for s in range(256) if L[1, s].max() > 0]; M = L[:, sms]
    print("\n== E2 latency probe (SM clock cycles) ==  active SMs:", len(sms), " regions:", M.shape[2])
    names = ["pass0 same-addr cold race", "pass1 same-addr L2-warm", "pass2 SM-specific line (DRAM)"]
    for p in range(3):
        X = M[p]; h, e = np.histogram(X, bins=[0, 300, 450, 700, 850, 1100, 1400, 70000])
        print(f"{names[p]}: median {np.median(X):.0f} p5 {np.percentile(X,5):.0f} p95 {np.percentile(X,95):.0f} sd {X.std():.0f}; share by bin {dict(zip(e[:-1].astype(int), np.round(h/X.size,3)))}")
        tot = X.var(); print(f"   variance share: SM {X.mean(1).var()/tot:.4f}  region(64KB) {X.mean(0).var()/tot:.4f}  interaction+noise {1-(X.mean(1).var()+X.mean(0).var())/tot:.4f}")
        Z = X - X.mean(0, keepdims=True) - X.mean(1, keepdims=True) + X.mean()
        nr = Z.shape[1] // 32 * 32; Zs = Z[:, :nr].reshape(Z.shape[0], -1, 32).mean(2)
        Sv = np.linalg.svd(Zs, compute_uv=False); Sr = np.linalg.svd(np.random.default_rng(0).permutation(Zs.ravel()).reshape(Zs.shape), compute_uv=False)
        print("   interaction SVD (2MB-smoothed) top5", np.round(Sv[:5], 1), " shuffled-null top5", np.round(Sr[:5], 1))
        print("   per-SM mean min/max %.0f/%.0f" % (X.mean(1).min(), X.mean(1).max()))
    X = M[1]; Z = X - X.mean(0, keepdims=True) - X.mean(1, keepdims=True) + X.mean()
    # rank-1 structure at 64KB resolution on the warm same-address pass
    U, Sv, Vt = np.linalg.svd(Z, full_matrices=False)
    print("pass1 full-res interaction SVD top5", np.round(Sv[:5], 1))
    grp = (U[:, 0] > 0).astype(int); home = (Vt[0] > 0).astype(int)
    print("SM groups:", np.bincount(grp), " region groups:", np.bincount(home))
    for g in (0, 1):
        print("  SMgrp%d: mean latency to homegrp0 %.1f  homegrp1 %.1f" % (g, X[grp == g][:, home == 0].mean(), X[grp == g][:, home == 1].mean()))
# E4 -----------------------------------------------------------------
import glob, csv, io
fs = sorted(glob.glob(f"{R}/ncu_{J}_*.csv"))
if fs:
    print("\n== E4 ncu (sum over instances unless noted) ==")
    tab = {}
    for f in fs:
        a = f.split(f"ncu_{J}_")[1][:-4]; txt = open(f).read(); i = txt.find('"ID"')
        for r in csv.DictReader(io.StringIO(txt[i:])):
            m = r["Metric Name"]; v = r["Metric Value"].replace(",", "")
            try: v = float(v)
            except: continue
            tab.setdefault(a, {}).setdefault(m, []).append(v)
    mets = sorted({m for a in tab for m in tab[a]})
    print(f"{'metric':60}" + "".join(f"{a:>14}" for a in tab))
    for m in mets:
        print(f"{m:60}" + "".join(f"{(tab[a].get(m,[float('nan')])[0] if len(tab[a].get(m,[]))==1 else np.sum(tab[a].get(m,[np.nan]))):14.4g}" for a in tab))
    print("per-instance dram__bytes_read imbalance (max/mean, n instances):")
    for a in tab:
        v = np.array(tab[a].get("dram__bytes_read.sum", [])) if False else None

# E2b + H3 test ---------------------------------------------------------
l1 = f"{R}/lat1_{J}.u16"
if os.path.exists(l1) and os.path.exists(lf):
    D = np.fromfile(l1, dtype=np.uint16).reshape(-1, 16384).astype(float)   # [smid][region] DRAM-miss latency, same addresses
    W = L[1, :D.shape[0]]                                                    # L2-warm same-address latency, same smid rows
    Zw = W - W.mean(0) - W.mean(1)[:, None] + W.mean(); U, Sv, Vt = np.linalg.svd(Zw, full_matrices=False)
    grp = (U[:, 0] > 0).astype(int); home = (Vt[0] > 0).astype(int)
    if W[grp == 0][:, home == 0].mean() < W[grp == 0][:, home == 1].mean(): near = lambda g, h: g == h        # near = same-group
    else: near = lambda g, h: g != h
    print("\n== E2b DRAM-miss latency, identical addresses, one SM at a time ==")
    print("L2-warm groups: SM", np.bincount(grp), "region", np.bincount(home))
    for g in (0, 1):
        for h in (0, 1):
            print(f"  SMgrp{g} x home{h} ({'near' if near(g,h) else 'far '}): L2-hit {W[grp==g][:, home==h].mean():6.1f}  DRAM-miss {D[grp==g][:, home==h].mean():7.1f}  p50 {np.median(D[grp==g][:, home==h]):.0f}")
    Zd = D - D.mean(0) - D.mean(1)[:, None] + D.mean(); Sd = np.linalg.svd(Zd, compute_uv=False)
    print("DRAM-miss: var share SM %.4f region %.4f; interaction SVD top5 %s" % (D.mean(1).var() / D.var(), D.mean(0).var() / D.var(), np.round(Sd[:5], 0)))
    h2, e2 = np.histogram(D.ravel(), bins=[0, 450, 700, 850, 1100, 70000]); print("DRAM latency bins", dict(zip(e2[:-1].astype(int), np.round(h2 / D.size, 3))))
    # does the DRAM bimodality follow the home group?
    for h in (0, 1): print(f"  frac of probes >750 cycles for home{h}: {(D[:, home==h] > 750).mean():.3f}")
    # share homing granularity: near-fraction of each static share (7.26 MB = 111 x 64 KB) per SM group
    R64 = 16384; edges = np.linspace(0, R64, 149).astype(int)
    for g in (0, 1):
        fr = np.array([np.mean(near(g, home[edges[s]:edges[s+1]])) for s in range(148)])
        print(f"SM group {g}: near-homed fraction of the 148 shares: mean {fr.mean():.3f} min {fr.min():.3f} max {fr.max():.3f}")
    # static per-SM finish vs SM die (static arm, SM s runs share s)
    if "static" in reps:
        fin = np.mean([rp["t1"] for rp in reps["static"]], axis=0); smv = np.array([st.mode(rp["smid"][s] for rp in reps["static"]) for s in range(148)])
        gsm = grp[smv]
        print("static finish by SM die group: %.2f us (n=%d) vs %.2f us (n=%d)" % (fin[gsm == 0].mean() / 1e3, (gsm == 0).sum(), fin[gsm == 1].mean() / 1e3, (gsm == 1).sum()))
        sf = np.array([np.mean(near(grp[smv[s]], home[edges[s]:edges[s+1]])) for s in range(148)])
        print("corr(static share finish, near-homed fraction of the share on its own SM): %.3f" % np.corrcoef(fin, sf)[0, 1])
