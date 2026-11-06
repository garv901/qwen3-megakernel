"""B7 analysis. Usage: analyze.py RESULTS_DIR JOB   (rows: RESULTS_DIR/*_<JOB>.jsonl; raw stamps *.st of QKV).
Prints: gate, T tables (redadd / ws, pf 1 / pf 0), paired same-max-steps differences, fits (redadd and ws), oracle arms, stamps cross-check.
Writes params.json next to this file. T = us_Tmean (telescoping mean over phases 2..nph, median over launches), medians over the passes of the job."""
import glob, json, os, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "../common"))
import stamps, tiles
R, JOB = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(f"{R}/*_{JOB}.jsonl")):
    for l in open(f):
        if l.startswith("{"):
            x = json.loads(l); x["file"] = os.path.basename(f); rows.append(x)
print("rows", len(rows), "FAIL rows", sum(1 for x in rows if x.get("pass") == 0))
main = [x for x in rows if x["orc"] == 0 and x.get("pass") == 1]
D = collections.defaultdict(list)
for x in main: D[(x["arm"], x["shape"], x["T"], x["KS"], x["pf"])].append(x)
def G(k, f="us_Tmean"): return float(np.median([x[f] for x in D[k]])) if k in D else float("nan")
SH = ["QKV", "O", "GU", "D"]; TS = [8, 32, 128, 256]; KSL = [1, 2, 3, 4, 6, 8, 12, 24]

print("\n== GATE QKV T128 TT128 (redadd): KS4 vs KS8, us_Tmean per pass file")
gate_rows = {}
for pf in (1, 0):
    for ks in (4, 8):
        for x in D[("redadd", "QKV", 128, ks, pf)]: print(f"pf{pf} KS{ks} {x['file']:28s} T {x['us_Tmean']:.3f} (reps {x['us_T_min']:.3f}..{x['us_T_max']:.3f}) mhz {x['mhz']:.0f}")
    d = G(("redadd", "QKV", 128, 8, pf)) - G(("redadd", "QKV", 128, 4, pf)); gate_rows[pf] = d
    print(f"pf{pf}: KS8 - KS4 = {d:.3f} us (median over passes: KS4 {G(('redadd','QKV',128,4,pf)):.3f}, KS8 {G(('redadd','QKV',128,8,pf)):.3f}); gate >= 0.8: {'PASS' if d >= 0.8 else 'FAIL'}")
per = []
for p in sorted({x["file"] for x in D[("redadd", "QKV", 128, 4, 1)]}):
    a = [x["us_Tmean"] for x in D[("redadd", "QKV", 128, 4, 1)] if x["file"] == p]; b = [x["us_Tmean"] for x in D[("redadd", "QKV", 128, 8, 1)] if x["file"] == p]
    if a and b: per.append(b[0] - a[0])
print("per-pass diffs (pf 1):", np.round(per, 3))

print("\n== T tables: us_Tmean (pf1 / pf0) | tiles busy max_steps tiles_max | ws arm pf1 (gemm work / reduce work, busy CTAs)")
for sh in SH:
    for T in TS:
        print(f"{sh} T{T}")
        for ks in KSL:
            k = ("redadd", sh, T, ks, 1)
            if k not in D: continue
            x = D[k][0]; w = ("ws", sh, T, ks, 1)
            s = f"  KS{ks:2d} TT{x['TT']:3d} ring{x['ring']} tiles {x['tiles']:4d} busy {x['busy_ctas']:3d} steps {x['max_steps']:2d} tiles_max {x['tiles_max']} | redadd {G(k):6.2f} / {G(('redadd',sh,T,ks,0)):6.2f} (busy work {G(k,'us_busy'):.2f} idle {G(k,'us_idle'):.2f} last {G(k,'us_last'):.2f} lat {G(k,'us_lat'):.2f})"
            if w in D: s += f" | ws {G(w):6.2f} / {G(('ws',sh,T,ks,0)):6.2f} (gemm {G(w,'us_busy'):.2f}, reduce {G(w,'us_work2'):.2f})"
            print(s)

print("\n== paired: same (shape, T), same max_steps, different KS (redadd pf1): d T and d (tiles_max, busy)")
for sh in SH:
    for T in TS:
        ks_av = [ks for ks in KSL if ("redadd", sh, T, ks, 1) in D]
        for i, a in enumerate(ks_av):
            for b in ks_av[i + 1:]:
                xa, xb = D[("redadd", sh, T, a, 1)][0], D[("redadd", sh, T, b, 1)][0]
                if xa["max_steps"] == xb["max_steps"] and a > 1:
                    print(f"  {sh} T{T} steps {xa['max_steps']}: KS{a} (busy {xa['busy_ctas']}, tiles_max {xa['tiles_max']}) {G(('redadd',sh,T,a,1)):.2f} -> KS{b} (busy {xb['busy_ctas']}, tiles_max {xb['tiles_max']}) {G(('redadd',sh,T,b,1)):.2f}: {G(('redadd',sh,T,b,1))-G(('redadd',sh,T,a,1)):+.2f} us")

# ---- fits ----
def feats(x):
    KS, TT, T = x["KS"], x["TT"], x["T"]; v = min(T, TT) / 128
    t = tiles.ts(x["N"], x["K"], T, TT, KS)
    assert t["max_steps"] == x["max_steps"] and t["busy"] == x["busy_ctas"], (x["shape"], T, KS)
    return dict(one=1.0, ms=x["max_steps"], mt=x["tiles_max"], busy=x["busy_ctas"], KS=KS, lgks=np.log2(KS), red_t=x["tiles_max"] * v * (KS > 1), pl_t=x["tiles_max"] * v * (KS == 1),
                rdMB=KS * T * x["N"] * 4 / 1e6, wsw_t=x["tiles_max"] * v, T=T, TT=TT)
def lsq(pts, cols):
    A = np.array([[f[c] for c in cols] for f, _ in pts]); y = np.array([t for _, t in pts]); co = np.linalg.lstsq(A, y, rcond=None)[0]; e = A @ co - y
    return co, float(np.sqrt((e ** 2).mean())), float(abs(e).max())
P_r = [(feats(D[k][0]), G(k)) for k in D if k[0] == "redadd" and k[4] == 1]
P_w = [(feats(D[k][0]), G(k)) for k in D if k[0] == "ws" and k[4] == 1]
P_r0 = [(feats(D[k][0]), G(k)) for k in D if k[0] == "redadd" and k[4] == 0]
fits = {}
print("\n== fits, redadd pf1 (n=%d): T = t0 + a*max_steps + c_red*red_tiles + c_pl*plain_tiles [+ g*log2(KS)]; red_tiles = tiles_max * min(T,TT)/128 for KS>1 (one unit = 128 x 128 fp32 = 64 KB per CTA), plain_tiles same for KS=1" % len(P_r))
for name, cols in [("lin_steps", ["one", "ms"]), ("F1", ["one", "ms", "red_t", "pl_t"]), ("F2", ["one", "ms", "red_t", "pl_t", "lgks"]), ("F3", ["one", "ms", "mt", "red_t", "pl_t"]), ("F4_busy", ["one", "ms", "red_t", "pl_t", "busy"])]:
    co, rm, mx = lsq(P_r, cols); fits[name] = dict(cols=cols, coeffs=[round(float(c), 4) for c in co], rmse=round(rm, 3), maxres=round(mx, 3)); print(f"  {name:10s} {dict(zip(cols, np.round(co, 3)))} rmse {rm:.3f} max {mx:.3f}")
for TTc in (16, 32, 128):
    sub = [p for p in P_r if p[0]["TT"] == TTc]; co, rm, mx = lsq(sub, ["one", "ms", "red_t", "pl_t"]); print(f"  F1 TT{TTc} only (n={len(sub)}): {dict(zip(['one','ms','red_t','pl_t'], np.round(co, 3)))} rmse {rm:.3f}")
co0, rm0, mx0 = lsq(P_r0, ["one", "ms", "red_t", "pl_t"]); print(f"  F1 on pf0 rows: {np.round(co0, 3)} rmse {rm0:.3f}")
fits["F1_pf0"] = dict(cols=["one", "ms", "red_t", "pl_t"], coeffs=[round(float(c), 4) for c in co0], rmse=round(rm0, 3), maxres=round(mx0, 3))
print("== fit ws pf1 (n=%d): T = t0 + a*max_steps + c_pl*partial_tiles + r*reduce_MB (KS*T*N*4 read bytes)" % len(P_w))
wfit = {}
for name, cols in [("W1", ["one", "ms", "wsw_t", "rdMB"]), ("W2", ["one", "ms", "wsw_t", "rdMB", "lgks"])]:
    co, rm, mx = lsq(P_w, cols); wfit[name] = dict(cols=cols, coeffs=[round(float(c), 4) for c in co], rmse=round(rm, 3), maxres=round(mx, 3)); print(f"  {name} {dict(zip(cols, np.round(co, 3)))} rmse {rm:.3f} max {mx:.3f}")
# reduce stage alone: work2 (median over CTAs of arrive - exit, reduce stage) vs read MB
Pq = [(feats(D[k][0]), G(k, "us_work2")) for k in D if k[0] == "ws" and k[4] == 1]
co, rm, mx = lsq(Pq, ["one", "rdMB"]); print(f"  reduce-stage work = {co[0]:.3f} + {co[1]*1e3:.3f} us/GB-read... i.e. {co[1]:.4f} us/MB, rmse {rm:.3f}"); wfit["reduce_stage"] = dict(cols=["one", "rdMB"], coeffs=[round(float(c), 4) for c in co], rmse=round(rm, 3))
print("\n== ws minus redadd at the same (shape, T, KS), pf1")
dd = []
for k in sorted(D):
    if k[0] == "ws" and k[4] == 1 and ("redadd",) + k[1:] in D:
        dd.append((k, G(k) - G(("redadd",) + k[1:]), D[k][0]["KS"] * k[2] * D[k][0]["N"] * 4 / 1e6))
for sh in SH:
    print(" ", sh, " ".join(f"T{k[2]}KS{k[3]}:{d:+.1f}" for k, d, mb in dd if k[1] == sh))

print("\n== oracle arms (QKV T128, pf1, us_Tmean): none / no-MMA(1) / no-epi-stores(2) / no-B(4) / no-A(8)")
orcs = collections.defaultdict(dict)
for x in rows:
    if x["shape"] == "QKV" and x["T"] == 128 and x["arm"] == "redadd" and x["pf"] == 1 and x["KS"] in (2, 4, 8) and x["orc"] in (1, 2, 4, 8): orcs[x["KS"]][x["orc"]] = x["us_Tmean"]
for ks in (2, 4, 8): print(f"  KS{ks}: none {G(('redadd','QKV',128,ks,1)):.2f}  " + "  ".join(f"orc{o}: {orcs[ks].get(o, float('nan')):.2f}" for o in (1, 2, 4, 8)))

print("\n== stamps.py cross-check (QKV, raw .st, last timed launch of each config): Tmean from stamps.metrics vs jsonl")
mx_d = 0; n = 0
for x in rows:
    if x.get("st_n") and x["arm"] == "redadd" and x["orc"] == 0:
        st = stamps.load(f"{R}/{x['file'].replace('.jsonl', '.st')}", x["grid"], x["nstmax"])[x["st_off"]:x["st_off"] + 1]
        m = stamps.metrics(st, x["nph"], 2); mx_d = max(mx_d, abs(m["Tmean"] - x["us_Tmean"])); n += 1
print(f"  {n} configs, max |Tmean(stamps.py, last launch) - jsonl (median of launches)| = {mx_d:.3f} us")
ld = [abs(x["us_launch_per_phase"] / x["us_Tmean"] - 1) for x in main]
print(f"  launch/nph vs Tmean: median ratio-1 {np.median([x['us_launch_per_phase']/x['us_Tmean']-1 for x in main]):.3f}, max |.| {max(ld):.3f}, n>5%: {sum(1 for v in ld if v > .05)} of {len(ld)}")
print("  mhz range over rows: %.0f .. %.0f" % (min(x["mhz"] for x in main), max(x["mhz"] for x in main)))
json.dump({"b7_splitk": {
    "redadd_T": {"formula": "T = t0 + a*max_steps + c_red*red_tiles + c_pl*plain_tiles; max_steps = tiles_max*steps_per_tile (common/tiles.py ts); red_tiles = tiles_max*min(T,TT)/128 if KS>1 else 0; plain_tiles = same if KS==1 (128x128 fp32 tile per CTA = 64 KB); window prefetch on",
                 "coeffs": {"cols": fits["F1"]["cols"], "values": fits["F1"]["coeffs"]}, "units": "us", "fit_rmse": fits["F1"]["rmse"], "range": {"shapes": SH, "T": TS, "TT": "min(T,128) rounded to 16", "KS": KSL, "n": len(P_r)}, "jobs": [int(JOB)], "variants": fits},
    "ws_T": {"formula": "T = t0 + a*max_steps + c_pl*partial_tiles + r*read_MB, read_MB = KS*T*N*4/1e6; two barriers per phase; window prefetch at the end barrier", "coeffs": wfit, "units": "us", "fit_rmse": wfit["W1"]["rmse"], "range": {"n": len(P_w)}, "jobs": [int(JOB)]},
    "gate": {"KS8_minus_KS4_us_pf1": gate_rows[1], "KS8_minus_KS4_us_pf0": gate_rows[0], "target": ">= 0.8"}}}, open(os.path.join(HERE, "params.json"), "w"), indent=1)

print("\n== pf1 - pf0 (arm r, us_Tmean) by busy CTAs")
bb = collections.defaultdict(list)
for k in D:
    if k[0] == "redadd" and k[4] == 1 and ("redadd",) + k[1:4] + (0,) in D:
        x = D[k][0]; bb[x["busy_ctas"] if x["busy_ctas"] in (8, 16, 24, 32, 48, 64, 96, 128, 148) else x["busy_ctas"]].append(G(k) - G(("redadd",) + k[1:4] + (0,)))
for b in sorted(bb): print(f"  busy {b:3d}: n {len(bb[b]):2d} median {np.median(bb[b]):+.2f} min {min(bb[b]):+.2f} max {max(bb[b]):+.2f}")
