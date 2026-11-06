"""B1 analysis. Usage: analyze.py RESULTS_DIR JOB [JOB2 ...]  -> tables, fit t = t0 + S/rate(N), rate = min(r_sm, BW/N), gate; writes params.json.
Rows: run{1,2}_JOB.jsonl. Time used: us_busymean (mean over phases 2..nph of the median-over-active-CTAs work = arrive - previous exit), per launch, median over launches.
S in KB (1 KB = 1024 B); rates in GB/s = 1e9 B/s."""
import glob, json, sys, os
import numpy as np
R = sys.argv[1]; jobs = sys.argv[2:]
here = os.path.dirname(os.path.abspath(__file__))
rows = []
for j in jobs:
    for r in (1, 2):
        f = f"{R}/run{r}_{j}.jsonl"
        if os.path.exists(f):
            for l in open(f):
                if l.startswith("{"): x = json.loads(l); x["pass_id"] = f"{j}.{r}"; rows.append(x)
print("rows", len(rows), "jobs", jobs)
def agg(sel, key):
    v = [x[key] for x in sel]; return float(np.median(v)) if v else float("nan")
def get(place, N, S, Rr, key="us_busymean"):
    return agg([x for x in rows if x["place"] == place and x["N"] == N and x["S_kb"] == S and x["R"] == Rr], key)
Ns = sorted({x["N"] for x in rows}); Ss = sorted({x["S_kb"] for x in rows}); Rs = sorted({x["R"] for x in rows})
for key in ("us_busymean", "us_lastmean", "us_Tmean"):
    for place in (0, 1):
        for Rr in Rs:
            if Rr not in (2, 4, 7): continue
            print(f"\n== {key} place={'spread' if place==0 else 'first'} R={Rr}: N (rows) x S KB (cols), us")
            print("N\\S " + " ".join(f"{s:>7}" for s in Ss))
            for n in Ns: print(f"{n:<4}" + " ".join(f"{get(place,n,s,Rr,key):7.2f}" for s in Ss))
# pass-to-pass spread
pp = {}
for x in rows: pp.setdefault((x["place"], x["N"], x["S_kb"], x["R"]), {})[x["pass_id"]] = x["us_busymean"]
dev = [abs(max(v.values()) - min(v.values())) / np.mean(list(v.values())) for v in pp.values() if len(v) > 1]
print("\npass-to-pass relative spread of busymean: median %.3f max %.3f (n=%d)" % (np.median(dev), np.max(dev), len(dev)))

# aggregate rate: N*S / time (GB/s), using last (slowest CTA) and busy
print("\n== aggregate GB/s = N_active*S/us_lastmean, spread placement, R=4")
Nact = lambda place, n: int(sum(1 for c in range(148) if ((c+1)*n)//148 > (c*n)//148)) if place == 0 else n
print("N\\S " + " ".join(f"{s:>7}" for s in Ss))
for n in Ns: print(f"{n:<4}" + " ".join(f"{Nact(0,n)*s*1024/get(0,n,s,4,'us_lastmean')/1e3:7.0f}" for s in Ss))

# fit per (place, R): t = t0 + S/min(r_sm, BW/N), S in bytes -> t in us; S/rate with rate in GB/s: S[KB]*1.024/rate[GB/s] us ... S*1024 B / (rate*1e9 B/s) = us*1e-6 -> S*1.024e-3/rate*1e3 us
def model(p, N, S): t0, r, bw = p; rate = np.minimum(r, bw / N); return t0 + S * 1024 / (rate * 1e3)   # rate GB/s -> B/us = rate*1e3
fitsout = {}
print("\n== fit t = t0 + S/min(r_sm, BW/N)  (key=us_busymean | us_lastmean), all N, S >= 64 KB")
for key in ("us_busymean", "us_lastmean"):
    for place in (0, 1):
        for Rr in Rs:
            d = [(Nact(place, x["N"]), x["S_kb"], x[key]) for x in rows if x["place"] == place and x["R"] == Rr and x["S_kb"] >= 64]
            if not d: continue
            N_, S_, t_ = map(np.array, zip(*d))
            w = 1 / np.maximum(t_, 0.5) ** 0.5
            def sse(r, bw):
                m = model([0, r, bw], N_, S_); t0 = np.sum(w * w * (t_ - m)) / np.sum(w * w); return np.sum((w * (t_ - m - t0)) ** 2), t0
            best = None
            for r in np.arange(10, 150, 1.0):
                for bw in np.arange(2000, 9000, 50.0):
                    e, t0 = sse(r, bw)
                    if best is None or e < best[0]: best = (e, r, bw, t0)
            for r in np.arange(best[1] - 1, best[1] + 1.01, 0.1):
                for bw in np.arange(best[2] - 50, best[2] + 50.1, 5.0):
                    e, t0 = sse(r, bw)
                    if e < best[0]: best = (e, r, bw, t0)
            class res: x = [best[3], best[1], best[2]]
            rm = float(np.sqrt(np.mean((model(res.x, N_, S_) - t_) ** 2)))
            fitsout[(key, place, Rr)] = dict(t0_us=float(res.x[0]), r_sm_GBps=float(res.x[1]), BW_agg_GBps=float(res.x[2]), rmse_us=rm, n=len(d))
            print(f"{key:12} place={place} R={Rr}: t0={res.x[0]:.3f} us  r_sm={res.x[1]:.1f} GB/s  BW_agg={res.x[2]:.0f} GB/s  rmse={rm:.3f} us (n={len(d)})")
json.dump({f"{k[0]}|place{k[1]}|R{k[2]}": v for k, v in fitsout.items()}, open(f"{R}/fits_{'_'.join(jobs)}.json", "w"), indent=1)

# gate: N=148, S>=512: aggregate rate vs why (static 5.7-5.95 TB/s; il/queue/contig 6.3-6.6 TB/s; per-SM 40-45 GB/s)
print("\n== gate: N=148 aggregate rate (GB/s) = 148*S/lastmean and /busymean, per R, per pass")
for Rr in Rs:
    for S in (512, 768, 1024):
        for key in ("us_lastmean", "us_busymean", "us_Tmean"):
            v = {}
            for x in rows:
                if x["N"] == 148 and x["place"] == 0 and x["S_kb"] == S and x["R"] == Rr: v[x["pass_id"]] = 148 * S * 1024 / x[key] / 1e3
            print(f"R={Rr} S={S} {key:12}", {k: round(a) for k, a in v.items()})
print("\n== N=48 S=256 busy work (us): place0 / place1 per R   [H until B2]")
for Rr in Rs: print(f"R={Rr} spread {get(0,48,256,Rr):.3f} first {get(1,48,256,Rr):.3f} last {get(0,48,256,Rr,'us_lastmean'):.3f}  T {get(0,48,256,Rr,'us_Tmean'):.3f}")
print("\nmhz median over rows:", np.median([x["mhz"] for x in rows]), "range", min(x["mhz"] for x in rows), max(x["mhz"] for x in rows))
print("launch/phase vs Tmean check (N=148,R=4,S=256,512):", [(x["pass_id"], x["S_kb"], round(x["us_launch_per_phase"],2), round(x["us_Tmean"],2)) for x in rows if x["N"]==148 and x["R"]==4 and x["place"]==0 and x["S_kb"] in (256,512)])
