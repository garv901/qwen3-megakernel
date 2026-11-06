"""B8 analysis. Usage: analyze.py RESULTS_DIR [mk_stamps_DIR]   -> tables (bench per phase, mk per phase, term-sum prediction).
Phase metrics as smallt/analyze.py (layers 4..23, median over layers and launches): T, work, busy, idle, last; plus Tmean = mean of the per-phase T (no median grid)."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
import stamps as S

PH = ("QKV", "RoPE", "attn", "O", "GU", "D")
G = 148

def counts(total, g=G):
    return np.array([(c + 1) * total // g - c * total // g for c in range(g)])

def phase_metrics(st, masks, layers=range(4, 24)):
    """st [L][G][>=338]; masks {phase: bool[G] or None}. Returns {phase: dict}, layer_us."""
    L = st.shape[0]; NBAR = 1 + 6 * 28
    arr, ex = st[:, :, 0:2 * NBAR:2].astype(np.int64), st[:, :, 1:2 * NBAR:2].astype(np.int64)
    rows = {}
    for p in range(6):
        r = {k: [] for k in ("T", "work", "busy", "idle", "last", "lastbusy", "lat", "spread")}
        m = masks.get(p)
        for l in range(L):
            for layer in layers:
                b = 1 + 6 * layer + p
                wk = (arr[l, :, b] - ex[l, :, b - 1]) / 1e3
                r["T"].append((np.median(ex[l, :, b]) - np.median(ex[l, :, b - 1])) / 1e3)
                r["work"].append(np.median(wk)); r["last"].append(wk.max())
                r["lat"].append((np.median(ex[l, :, b]) - arr[l, :, b].max()) / 1e3)
                r["spread"].append((arr[l, :, b].max() - np.median(arr[l, :, b])) / 1e3)
                if m is not None:
                    r["busy"].append(np.median(wk[m])); r["idle"].append(np.median(wk[~m]) if (~m).any() else np.nan)
                    r["lastbusy"].append(float(m[int(arr[l, :, b].argmax())]))
        d = {k: (float(np.nanmedian(v)) if len(v) else np.nan) for k, v in r.items() if k != "T"}
        d["T"] = float(np.median(r["T"])); d["Tmean"] = float(np.mean(r["T"]))
        rows[PH[p]] = d
    lay = float(np.mean([(np.median(ex[l, :, 1 + 6 * 23 + 5]) - np.median(ex[l, :, 1 + 6 * 3 + 5])) for l in range(L)])) / 1e3 / 20
    return rows, lay

def mask_of(work):
    return counts(work) > 0 if work else None

def bench_rows(d):
    out = []
    for jf in sorted(glob.glob(f"{d}/*.jsonl")):
        base = jf[:-6]
        for line in open(jf):
            line = line.strip()
            if not line.startswith("{"): continue
            r = json.loads(line); r["_file"] = os.path.basename(jf)
            out.append(r)
    return out

def load_st(d, r):
    f = os.path.join(d, r["_file"].replace(".jsonl", ".st"))
    n, g, off = r["st_n"], r["grid"], r["st_off"]
    a = np.fromfile(f, dtype=np.uint64, count=n * g * r["nstmax"], offset=off * g * r["nstmax"] * 8).astype(np.int64)
    return a.reshape(n, g, r["nstmax"])

def bench_metrics(d, r):
    st = load_st(d, r)
    masks = {}
    for p, nm in ((0, "QKV"), (3, "O"), (4, "GU"), (5, "D")): masks[p] = mask_of(r[nm + "_work"])
    return phase_metrics(st, masks)

# mk stamps: npz sets; masks from the engine / bucket tables (mk_common.cuh), checked against smallt/analyze.py's TILES for NPAD <= 128
def mk_masks(npad, ts):
    if not ts: return {0: mask_of(256), 3: mask_of(128), 4: mask_of(384), 5: mask_of(192)}
    ntt = lambda tt: -(-npad // tt) if npad >= tt else 1
    ttq = min(npad, 128); ksq = 2 if npad == 256 else 4
    ttg = 96 if npad == 256 else 48 if npad == 128 else npad
    ksd = 8 if npad == 256 else 12
    return {0: mask_of(32 * ntt(ttq) * ksq), 3: mask_of(128), 4: mask_of(48 * ntt(ttg)), 5: mask_of(8 * ntt(ttq) * ksd)}

def mk_metrics(path):
    z = np.load(path); st = z["st"]; kn = json.loads(str(z["knobs"]))
    b = int(os.path.basename(path).split("-b")[1].split("-")[0]); npad = next(x for x in (16, 32, 64, 128, 256) if x >= b)
    mts = kn.get("MK_TS", 128); ts = mts != 0 and (128 if mts == 1 else mts) <= npad
    return phase_metrics(st, mk_masks(npad, ts)), ts

if __name__ == "__main__":
    d = sys.argv[1]
    for r in bench_rows(d):
        if r.get("row") != "time" or "error" in r: continue
        rows, lay = bench_metrics(d, r)
        print(f"\n{r['_file']} design {r['design']} T {r['T']} pf {r['pf']} null {r['null']} fin {r['fin_place']}  layer {lay:.2f} us (launch/layer {r['us_layer_launch']:.2f}) mhz {r['mhz']:.0f}")
        for k, v in rows.items():
            print(f"  {k:5s} T {v['T']:6.2f} Tmean {v['Tmean']:6.2f} work {v['work']:6.2f} busy {v['busy']:6.2f} idle {v['idle']:6.2f} last {v['last']:6.2f}")


# ---------------------------------------------------------------- B8 report (bench vs mk, gates, term-sum)
DESN = {"classic": "C", "ts_ks1": "TS1", "ts_split": "TS3"}
MKDIRS = ["../../../../results/2026-10-29-smallt/st_14155", "../../../../results/2026-10-29-occfuse/b8_compose/mk_stamps_14303"]

def collect_bench(d, tag, null=0, pf=1):
    out = {}
    for r in bench_rows(d):
        if r.get("row") != "time" or "error" in r or not r["_file"].startswith(tag): continue
        if r["null"] != null or r["pf"] != pf: continue
        rows, lay = bench_metrics(d, r)
        out[(r["design"], r["T"], r["fin_place"])] = (rows, lay, r)
    return out

def collect_mk():
    here = os.path.dirname(os.path.abspath(__file__)); res = {}
    for md in MKDIRS:
        for f in sorted(glob.glob(os.path.join(here, md, "*.npz"))):
            nm = os.path.basename(f)[:-4]; b = int(nm.split("-b")[1].split("-")[0]); st = nm.split("__")[1]
            if st not in ("C", "T", "C_null", "T_null"): continue
            (rows, lay), ts = mk_metrics(f)
            res[(b, st)] = (rows, ts, lay)
    return res

if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "report":
    here = os.path.dirname(os.path.abspath(__file__))
    mk = collect_mk(); B = collect_bench(d, "main_")
    print("\n=== bench (pf 1, no null) vs mk, per phase: T (us), %diff; busy ===")
    for T in (1, 8, 32, 64, 128, 256):
        for des, st in (("classic", "C"), ("ts_split", "T")):
            k = next((x for x in B if x[0] == des and x[1] == T), None)
            if k is None or (T, st) not in mk: continue
            br = B[k][0]; mr = mk[(T, st)][0]
            s = f"T={T:3d} {des:9s}vs mk {st} (mk ts={mk[(T, st)][1]}): "
            for p in ("QKV", "O", "GU", "D"):
                s += f"{p} {br[p]['T']:5.2f}/{mr[p]['T']:5.2f} ({100 * (br[p]['T'] / mr[p]['T'] - 1):+4.0f}%)  "
            sb = sum(br[p]['T'] for p in ("QKV", "O", "GU", "D")); sm = sum(mr[p]['T'] for p in ("QKV", "O", "GU", "D"))
            print(s + f"sum4 {sb:5.2f}/{sm:5.2f} ({100 * (sb / sm - 1):+4.0f}%)")
    print("\n=== detail: work / busy / idle / last / lat / spread  (bench | mk) ===")
    for T in (8, 128, 256):
        for des, st in (("classic", "C"), ("ts_split", "T")):
            k = next((x for x in B if x[0] == des and x[1] == T), None)
            if k is None or (T, st) not in mk: continue
            for p in ("QKV", "O", "GU", "D"):
                f = lambda v: " ".join(f"{v[q]:5.2f}" for q in ("T", "work", "busy", "idle", "last", "lat", "spread"))
                print(f"T={T:3d} {des:9s}{p:4s} bench {f(B[k][0][p])} | mk {f(mk[(T, st)][0][p])}")
    R = os.path.dirname(os.path.abspath(d + "/x"))
    print("\n=== arms (phase T us): tag design T pf null fin ===")
    for tag in ("main_", "pf0_", "pf2_", "null_", "fin0_", "fin2_"):
        for r in bench_rows(d):
            if r.get("row") != "time" or "error" in r or not r["_file"].startswith(tag): continue
            if tag == "main_" and r["T"] not in (8, 64, 256): continue
            rows, lay = bench_metrics(d, r)
            print(f"{tag:6s}{r['design']:9s} T {r['T']:3d} pf {r['pf']} null {r['null']} fin {r['fin_place']} layer {lay:6.2f} | " + " ".join(f"{p} {rows[p]['T']:5.2f}/b{rows[p]['busy']:5.2f}" for p in ("QKV", "O", "GU", "D")))
    print("\n=== check rows ===")
    for r in bench_rows(d):
        if r.get("row") == "check" and r["_file"].startswith("main_14306"): pass
    ck = [r for r in bench_rows(d) if r.get("row") == "check" and r["_file"].startswith("main_14306")]
    print(len(ck), "checks,", sum(r["pass"] for r in ck), "pass; max maxerr", max(r["maxerr"] for r in ck))
    print("\n=== open item (a): null arms. bench null (pf 1) vs mk C_null / T_null: T / busy (idle) ===")
    BN = collect_bench(d, "null_", null=1)
    for T in (1, 8, 32, 64, 128, 256):
        for des, st in (("classic", "C_null"), ("ts_split", "T_null")):
            k = next((x for x in BN if x[0] == des and x[1] == T), None)
            if k is None or (T, st) not in mk: continue
            br, mr = BN[k][0], mk[(T, st)][0]
            print(f"T={T:3d} {st:7s} " + "  ".join(f"{p} {br[p]['T']:5.2f}/{br[p]['busy']:5.2f} | mk {mr[p]['T']:5.2f}/{mr[p]['busy']:5.2f}" for p in ("QKV", "O", "GU", "D")))

# ---------------------------------------------------------------- term-sum prediction (a sum of the fitted terms; no refit)
def predict(r, p, T, pr):
    """Predicted phase time (us) for phase p in row r from params.json terms. overlap = max(stream side, chain) + rest."""
    b1, b2, b3, b7, b6 = pr["b1_stream"]["t_stream"]["coeffs"], pr["b2_prefetch"], pr["b3_step"], pr["b7_splitk"], pr["b6_fin"]
    eng, work, spt = r[p + "_eng"], r[p + "_work"], r[p + "_spt"]
    TT, KS, R, fin = r[p + "_TT"], r[p + "_KS"], r[p + "_R"], r[p + "_fin"]
    nact = min(work, G); tmax = -(-work // G)
    ms = tmax * spt
    Rn = 2 if eng == 0 else max(2, min(R, 7))
    i = b1["R"].index(Rn) if Rn in b1["R"] else min(range(len(b1["R"])), key=lambda j: abs(b1["R"][j] - Rn))
    rate = min(b1["r_sm_GBps"][i], b1["BW_agg_GBps"][i] / nact) * 1e3      # bytes / us
    S = ms * 32768.0
    hid = min(0.408 * min(Rn, ms) * 32768.0 / rate, 0.5 * S / rate)          # classic_win_lt_1.3 slope (B2)
    stream = b1["t0_us"][i] + S / rate - hid
    ncol = min(T, 128 if eng == 0 else TT)
    if eng == 0:   # staged B (B4 a1_idle: lat + per k-step + per KB), once per unit
        bpath = tmax * (0.3726 + 0.4394 * spt + 0.05603 * (ncol * 128 * 2 * spt / 1024.0))
    else:          # TMA B (B4 b_idle form, boxes = 2 per k-step)
        bpath = tmax * (0.3514 + 0.0604 * 2 * spt + 0.00242 * (TT * 128 * 2 * spt / 1024.0))
    nct = -(-T // 128) if eng == 0 else 1      # classic: ceil(T/128) column passes over the units (chain, B path, piece scale; A stream is re-read)
    bpath *= nct; S *= nct
    stream = b1["t0_us"][i] + S / rate - hid
    chain = 0.525 * ms * nct
    tb = 1.2
    red = tmax * ncol / 128.0 if (eng == 0 or KS > 1) else 0.0
    pl = tmax * ncol / 128.0 if (eng == 1 and KS <= 1) else 0.0
    piece = (1.382 * red + 0.9563 * pl) * nct
    nit = -(-min(T, 128) // 32)
    f = 0.0
    if fin: f = (1.705 + 0.00222 * min(T, 128)) if r["fin_place"] == 2 or (r["fin_place"] == -1 and T >= 128) else (1.009 + 0.6238 * nit)
    ov = max(stream + bpath, chain)
    return dict(pred=tb + ov + piece + f, stream=stream, bpath=bpath, chain=chain, piece=piece, fin=f, t_bar=tb)

if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "terms":
    pr = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "params.json")))
    print("design T phase | pred meas (err%) | stream+bpath chain piece fin")
    for r in bench_rows(d):
        if r.get("row") != "time" or "error" in r or not r["_file"].startswith("main_") or r["null"] or r["pf"] != 1: continue
        rows, lay = bench_metrics(d, r)
        for p in ("QKV", "O", "GU", "D"):
            q = predict(r, p, r["T"], pr); m = rows[p]["T"]
            print(f"{r['design']:9s} {r['T']:3d} {p:3s} | {q['pred']:6.2f} {m:6.2f} ({100 * (q['pred'] / m - 1):+4.0f}%) | {q['stream']:5.2f}+{q['bpath']:5.2f} {q['chain']:5.2f} {q['piece']:5.2f} {q['fin']:5.2f}")

    import collections
    agg = collections.defaultdict(list); n20 = [0, 0]
    for r in bench_rows(d):
        if r.get("row") != "time" or "error" in r or not r["_file"].startswith("main_") or r["null"] or r["pf"] != 1: continue
        rows, lay = bench_metrics(d, r)
        for p in ("QKV", "O", "GU", "D"):
            e = 100 * (predict(r, p, r["T"], pr)["pred"] / rows[p]["T"] - 1); agg[(r["design"], p, "T<=64" if r["T"] <= 64 else "T>=96")].append(e)
            n20[0] += abs(e) <= 20; n20[1] += 1
    print("\nmean signed err % (pred vs measured), count within 20%:", n20)
    for k, v in sorted(agg.items()): print(f"  {k[0]:9s} {k[1]:3s} {k[2]:6s} mean {np.mean(v):+5.0f}%  min {min(v):+4.0f} max {max(v):+4.0f}")
