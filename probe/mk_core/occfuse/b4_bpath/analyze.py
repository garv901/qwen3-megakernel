"""B4 analysis. Usage: analyze.py RESULTS_DIR JOB [JOB2 ...]  -> tables, fits, gate, params.json (next to this file).
t_B = work_mean(mode) - work_mean(z, same loaded)  [us]; work = per-CTA (barrier-arrive stamp - previous barrier exit), mean over CTAs and phases 2..nph (resolves < 0.25 us).
us_Tmean (telescoping mean of median exit stamps) is shown as a cross-check."""
import glob, json, sys, os
import numpy as np
R = sys.argv[1]; jobs = sys.argv[2:]
here = os.path.dirname(os.path.abspath(__file__))
rows = []
for j in jobs:
    for f in sorted(glob.glob(f"{R}/r*_l*_m*_{j}.jsonl")):
        for l in open(f):
            if l.startswith("{"):
                x = json.loads(l); x["jobid"] = j; x["pass_id"] = os.path.basename(f).split("_")[0]; rows.append(x)
MN = {"a1": "a1 staged/k-step", "a2": "a2 staged/batch", "b": "b TMA bf16", "c": "c TMA fp32"}
def key(x): return (x["jobid"], x["pass_id"], x["loaded"], x["T"], x["kb"])
z = {key(x): x for x in rows if x["arm"] == "z"}
for x in rows:
    b = z.get(key(x)); x["tB"] = x["us_work_mean"] - b["us_work_mean"] if b else float("nan")
    x["tB_T"] = x["us_Tmean"] - b["us_Tmean"] if b else float("nan")
fails = [x for x in rows if not x["pass"]]
print("FAIL rows:", [(x["arm"], x["T"], x["kb"], x["loaded"], x["jobid"]) for x in fails]); print("checked rows (chk=1):", sum(1 for x in rows if x["chk"] == 1), " chk=-1:", sum(1 for x in rows if x["chk"] == -1))
good = [x for x in rows if x["pass"] and x["arm"] != "z"]
Ts = sorted({x["T"] for x in rows}); kbs = sorted({x["kb"] for x in rows})
def avg(arm, ld, T, kb, f="tB"):
    v = [x[f] for x in good if x["arm"] == arm and x["loaded"] == ld and x["T"] == T and x["kb"] == kb]; return float(np.mean(v)) if v else float("nan")
for ld in (0, 1):
    print(f"\n==== loaded={ld}: t_B (us) = work(mode) - work(z); mean over passes/jobs; rows T, cols kb")
    for arm in ("a1", "a2", "b", "c"):
        print(f"-- {MN[arm]}"); print("T\\kb " + " ".join(f"{k:>8}" for k in kbs))
        for T in Ts: print(f"{T:<5} " + " ".join(f"{avg(arm, ld, T, k):8.2f}" for k in kbs))
    print("z baseline work (us): " + ", ".join(f"T{T}/kb{kbs[0]}={np.mean([x['us_work_mean'] for x in rows if x['arm']=='z' and x['loaded']==ld and x['T']==T and x['kb']==kbs[0]] or [np.nan]):.2f}" for T in Ts))
print("\n==== repeat spread (max-min of tB across passes/jobs), worst 8")
sp = []
for arm in MN:
    for ld in (0, 1):
        for T in Ts:
            for k in kbs:
                v = [x["tB"] for x in good if x["arm"] == arm and x["loaded"] == ld and x["T"] == T and x["kb"] == k]
                if len(v) > 1: sp.append((max(v) - min(v), arm, ld, T, k, [round(a, 2) for a in v]))
for s in sorted(sp, reverse=True)[:8]: print(s)
print("\n==== gate: T=256, staged vs TMA (kb, loaded): a1, a2, b, c, ratio a1/b, a2/b")
for ld in (0, 1):
    for k in kbs:
        a1, a2, b, c = (avg(m, ld, 256, k) for m in ("a1", "a2", "b", "c")); print(f"loaded={ld} kb={k}: a1={a1:.2f} a2={a2:.2f} b={b:.2f} c={c:.2f}  a1/b={a1/b:.2f} a2/b={a2/b:.2f}")
print("\n==== T<=16 gap (staged - TMA bf16), us")
for ld in (0, 1):
    for T in (1, 8, 16):
        print(f"loaded={ld} T={T}: " + "  ".join(f"kb{k}: a1-b={avg('a1',ld,T,k)-avg('b',ld,T,k):.2f} a2-b={avg('a2',ld,T,k)-avg('b',ld,T,k):.2f}" for k in kbs))
print("\n==== clocks: mhz median per arm/loaded")
for arm in ("z", "a1", "a2", "b", "c"):
    for ld in (0, 1):
        v = [x["mhz"] for x in rows if x["arm"] == arm and x["loaded"] == ld]
        if v: print(f"{arm} loaded={ld}: mhz med {np.median(v):.0f} min {min(v):.0f} max {max(v):.0f}")
bad = [x for x in rows if abs(x["us_launch_per_phase"] / x["us_T"] - 1) > 0.05]
print(f"\nlaunch/phase vs T differ >5%: {len(bad)}/{len(rows)}; worst:", sorted([(round(abs(x['us_launch_per_phase']/x['us_T']-1),3), x['arm'], x['T'], x['kb'], x['loaded']) for x in bad], reverse=True)[:5])

# fits: t_B = lat + bytes/thr. bytes: staged (a1, a2) = T*kb*4 fp32 bytes read from L2 (the unit used for all modes' "L2 bytes"); also smem-bytes version TT*kb*2.
def fit(arm, ld, xf, label):
    pts = [(xf(x), x["tB"]) for x in good if x["arm"] == arm and x["loaded"] == ld and np.isfinite(x["tB"])]
    if len(pts) < 3: return None
    b = np.array([p[0] for p in pts], float); t = np.array([p[1] for p in pts]); A = np.vstack([np.ones_like(b), b]).T
    co, *_ = np.linalg.lstsq(A, t, rcond=None); rm = float(np.sqrt(np.mean((A @ co - t) ** 2)))
    return {"lat_us": float(co[0]), "thr_B_per_us": float(1 / co[1]) if co[1] > 0 else float("inf"), "us_per_KB": float(co[1] * 1024), "rmse_us": rm, "n": len(pts), "max_resid_us": float(np.max(np.abs(A @ co - t)))}
bytes_of = {"a1": lambda x: x["bytes_l2"], "a2": lambda x: x["bytes_l2"], "b": lambda x: x["bytes_smem"], "c": lambda x: x["bytes_l2"]}
print("\n==== fits t_B = lat + bytes/thr   (x-axis bytes: a: T*kb*4 fp32 read; b: TT*kb*2 smem tile (box incl. pad); c: T*kb*4 TMA'd)")
out = {}
for arm in ("a1", "a2", "b", "c"):
    for ld in (0, 1):
        f = fit(arm, ld, bytes_of[arm], arm)
        if f:
            print(arm, "loaded", ld, f)
            out[f"t_B_{arm}_{'loaded' if ld else 'idle'}"] = {"formula": "t_B = lat_us + bytes / thr_B_per_us (bytes: " + ("T*kb*2 [TT*kb*2 incl. pad rows]" if arm == "b" else "T*kb*4 fp32 bytes") + ")", "coeffs": {k: round(v, 5) if isinstance(v, float) else v for k, v in f.items() if k in ("lat_us", "thr_B_per_us", "us_per_KB")},
                "units": "us; bytes in B", "fit_rmse": round(f["rmse_us"], 4), "range": {"T": [min(Ts), max(Ts)], "kb": [min(kbs), max(kbs)]}, "jobs": [int(j) for j in jobs]}
print("\n-- fits restricted to T>=64 (throughput regime) and per-T small-T table")
for arm in ("a1", "a2", "b"):
    for ld in (0, 1):
        pts = [(bytes_of[arm](x), x["tB"]) for x in good if x["arm"] == arm and x["loaded"] == ld and x["T"] >= 64]
        if len(pts) > 3:
            b = np.array([p[0] for p in pts], float); t = np.array([p[1] for p in pts]); A = np.vstack([np.ones_like(b), b]).T; co, *_ = np.linalg.lstsq(A, t, rcond=None)
            print(arm, ld, f"T>=64: lat={co[0]:.2f} us, thr={1/co[1]/1e3:.2f} GB/s-equiv (B/us/1e3), rmse={np.sqrt(np.mean((A@co-t)**2)):.2f}")

print("\n==== 3-term fits t_B = lat + s*steps + bytes/thr  (steps: a1 kb/128 k-steps, others 0; chunk term for staged = TT*kb*2 smem bytes written incl. pad rows; b/c: boxes*? use bytes_smem / L2 bytes)")
for arm in ("a1", "a2", "b", "c"):
    for ld in (0, 1):
        pts = [x for x in good if x["arm"] == arm and x["loaded"] == ld and np.isfinite(x["tB"])]
        cols = [np.ones(len(pts)), np.array([x["kb"] / 128 if arm == "a1" else x["nbox"] for x in pts], float), np.array([x["bytes_smem"] if arm in ("a1", "a2", "b") else x["bytes_l2"] for x in pts], float)]
        A = np.vstack(cols).T; t = np.array([x["tB"] for x in pts]); co, *_ = np.linalg.lstsq(A, t, rcond=None); rm = float(np.sqrt(np.mean((A @ co - t) ** 2)))
        print(f"{arm} loaded={ld}: lat={co[0]:.3f} us, per-{'kstep' if arm=='a1' else 'box'}={co[1]:.3f} us, {co[2]*1024:.4f} us/KB (smem tile bytes {'' if arm!='c' else '(L2 bytes)'}) rmse={rm:.3f} max={np.max(np.abs(A@co-t)):.2f}")
        out[f"t_B3_{arm}_{'loaded' if ld else 'idle'}"] = {"formula": "t_B = lat_us + per_unit_us*(kb/128 k-steps for a1, TMA boxes else) + us_per_KB*KB (bytes_smem = TT*kb*2; c: T*kb*4)", "coeffs": {"lat_us": round(float(co[0]), 4), "per_unit_us": round(float(co[1]), 4), "us_per_KB": round(float(co[2]) * 1024, 5)}, "units": "us", "fit_rmse": round(rm, 4), "range": {"T": [min(Ts), max(Ts)], "kb": [min(kbs), max(kbs)]}, "jobs": [int(j) for j in jobs]}
json.dump(out, open(os.path.join(here, "params.json"), "w"), indent=1)
