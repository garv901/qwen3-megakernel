"""B3 analysis. Usage: analyze.py RESULTS_DIR JOB [JOB2 ...]  -> tables, fit t_tile = a + SPT * t_step(N), gate vs hv2_tc 9425, params.json (next to this file).
Reads check/check2/sweep{1,2}/dep{1,2}/single/ring8 _JOB.jsonl. Cycles = clock64 across the timed pass / tiles; us = globaltimer / tiles; mhz = cycles / us."""
import json, sys, os, collections
import numpy as np
R = sys.argv[1]; jobs = sys.argv[2:]; here = os.path.dirname(os.path.abspath(__file__))
def rows(f): return [json.loads(l) for l in open(f) if l.startswith("{")] if os.path.exists(f) else []
def load(name):
    out = []
    for j in jobs:
        for nm in (name, name.replace("_F0", "") if name.endswith("_F0") else name):   # job 14290 ran before the --fence flag existed (fence 0, files without _F0)
            for x in rows(f"{R}/{nm}_{j}.jsonl"):
                x["jobid"] = j; x.setdefault("fence_tile", 0); out.append(x)
            if nm == name and out and name.endswith("_F0") and any(o["jobid"] == j for o in out): break
    return out
GATE = {16: 58.1, 128: 80.6, 256: 128.2}
chk = [x for L in (0, 1) for F in (0, 1) for x in load(f"check_L{L}_F{F}") + load(f"check2_L{L}_F{F}")]
print("== correctness (tcgen05.ld -> global vs CUDA-core)"); print(f"{len(chk)} configs, fail {sum(1 for x in chk if not x['pass'])}, max abs err {max(x['maxerr'] for x in chk) if chk else None}")
for x in chk:
    if not x["pass"]: print("FAIL", {k: x[k] for k in ("job", "TT", "SPT", "nacc", "commit_tile", "nbad")})
sw = [x for L in (0, 1) for F in (0, 1) for n in ("sweep1", "sweep2", "dep1", "dep2") for x in load(f"{n}_L{L}_F{F}") if x["pass"]]
for x in sw: x["tag"] = x["tag"][:5] if x["tag"].startswith("sweep") else x["tag"][:4]
key = lambda x: (x["TT"], x["nacc"], x["nact"], x["commit_tile"], x["dep"], x["lean"], x["fence_tile"])
G = collections.defaultdict(lambda: collections.defaultdict(list))
for x in sw: G[(key(x), x["jobid"], x["tag"][-1:] if x["tag"].startswith(("sweep", "dep")) else "")][x["SPT"]].append(x)
fits = {}
for k, d in sorted(G.items()):
    S = sorted(d); 
    if len(S) < 3: continue
    c = np.array([np.median([x["cyc_tile"] for x in d[s]]) for s in S]); u = np.array([np.median([x["us_tile"] for x in d[s]]) for s in S]); m = np.median([x["mhz"] for s in S for x in d[s]])
    A = np.vstack([np.ones(len(S)), S]).T
    (a, ts), *_ = np.linalg.lstsq(A, c, rcond=None); (au, tsu), *_ = np.linalg.lstsq(A, u, rcond=None)
    rm = float(np.sqrt(np.mean((A @ [a, ts] - c) ** 2)))
    fits[k] = dict(a_cyc=a, step_cyc=ts, mma_cyc=ts / 8, a_us=au, step_us=tsu, rmse_cyc=rm, mhz=m)
def hdr(): print(f"{'N':>4} {'acc':>3} {'act':>3} {'cm':>2} {'dep':>3} {'ln':>2} {'fn':>2} {'run':>8} | a_cyc step_cyc mma_cyc | a_us step_us | mhz rmse")
print("\n== fit t_tile = a + SPT*t_step per (N, nacc, nact, commit_tile, dep, job/pass)"); hdr()
for k, f in sorted(fits.items()):
    (TT, na, nact, cm, dep, ln, fn), job, p = k
    print(f"{TT:>4} {na:>3} {nact:>3} {cm:>2} {dep:>3} {ln:>2} {fn:>2} {job + p:>8} | {f['a_cyc']:6.0f} {f['step_cyc']:8.1f} {f['mma_cyc']:7.1f} | {f['a_us']:5.3f} {f['step_us']:6.3f} | {f['mhz']:6.0f} {f['rmse_cyc']:.1f}")
print("\n== gate: per-MMA cycles (t_step/8) vs hv2_tc 9425 (N16 58.1, N128 80.6, N256 128.2); pooled over runs, nact 1 and 148")
res = {}
for TT, ref in GATE.items():
    for (na, cm, dep, ln, fn) in [(na, cm, dep, ln, fn) for fn in (0, 1) for ln in (1, 0) for dep in (1, 2, 4) for na in (1, 2) for cm in (0, 1)]:
        for nact in (1, 148):
            v = [f["mma_cyc"] for (kk, j, p), f in fits.items() if kk == (TT, na, nact, cm, dep, ln, fn)]
            if v: res[(TT, na, nact, cm, dep, ln, fn)] = (np.median(v), min(v), max(v)); print(f"N{TT:<4} acc{na} act{nact:<3} commit_tile{cm} dep{dep} lean{ln} fence_tile{fn}: mma {np.median(v):7.1f} [{min(v):.1f},{max(v):.1f}] ref {ref}  ratio {np.median(v) / ref:.3f}  {'PASS' if abs(np.median(v) / ref - 1) <= 0.10 else 'FAIL'}")
print("\n== single tile (ntile 1) and ring8 rows"); 
for nm in ("single_L0_F0", "single_L0_F1", "single_L1_F0", "single_L1_F1", "ring8", "tw_F0", "tw_F1"):
    for x in load(nm): print(nm, {k: x[k] for k in ("TT", "SPT", "nacc", "nact", "commit_tile", "lean", "fence_tile", "ring", "cyc_tile", "us_tile", "mhz")})
# params.json
coef = {"N": [], "lean": [], "fence_tile": [], "dep": [], "nacc": [], "nact": [], "commit_tile": [], "a_cyc": [], "step_cyc": [], "a_us": [], "step_us": [], "mhz": []}
byk = collections.defaultdict(list)
for (kk, j, p), f in fits.items(): byk[kk].append(f)
for kk, fl in sorted(byk.items()):
    for n, v in zip(("N", "nacc", "nact", "commit_tile", "dep", "lean", "fence_tile"), kk): coef[n if n != "N" else "N"].append(v)
    for n in ("a_cyc", "step_cyc", "a_us", "step_us", "mhz"): coef[n].append(round(float(np.median([f[n] for f in fl])), 3))
rm = max(f["rmse_cyc"] for f in fits.values()) if fits else None
P = {"b3_step": {"t_chain": {"formula": "t_tile = a + SPT * t_step; cycles (clock64) and us (globaltimer); one entry per (N, nacc, nact, commit_tile, dep); t_step/8 = per-MMA cycles",
      "coeffs": coef, "units": "cycles (a_cyc, step_cyc), us (a_us, step_us), MHz", "fit_rmse": rm, "range": {"N": [16, 256], "SPT": [1, 2, 4, 8, 16]}, "jobs": [int(j) for j in jobs]}}}
json.dump(P, open(f"{here}/params.json", "w"), indent=1)
