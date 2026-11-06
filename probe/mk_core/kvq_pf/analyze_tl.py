"""kvq_diag timeline analysis: $PY analyze_tl.py file.npz [...]  (KVQ_TL=1 runs: clock64 events of the last layer of the last launch; per-CTA cycles, never compared across CTAs / SMs).
Per CTA: stages i (producer issue time I_i = end of produce(); consumer warps: wait start tw0, full seen tw1, released tw2; last stage of a unit: EOI enter te0 / leave te1).
cycles -> us per CTA with the CTA's own attend span (globaltimer cstart..cend). Outputs medians over CTAs:
  period_int / period_bnd : mean time between 'full seen' (min over warps) of consecutive stages, inside a unit / across a unit boundary (stage L last of a unit -> next stage)
  bubble/unit = period_bnd - period_int (us): what a unit boundary adds to the stream period
  eoi: enter(min over warps) -> leave(max over warps) of the end-of-item step; prod_bnd: produce() duration of a unit-first stage (global unit fetch), prod_int: of an interior stage
  idle: time (per CTA, us) with no stage in flight, union of [I_i, F_i] (F = min over warps of 'full seen': a lower bound of idle), split by whether the next issue is the stage after a unit end
  pfull_bnd / pfull_int: produce() incl. the prefetch step (what thread 0 spends per call; I = end of the TMA issue excludes it)
  wait_first / wait_int: consumer (warp-mean) wait at full for the first stage of a unit / other stages."""
import sys
import numpy as np


def analyze(path):
    z = np.load(path)
    tl, nct, tln, nl = z["tl"], int(z["nct"]), int(z["tln"]), int(z["nl"])
    cons = tl[:nct * 16 * tln * 8].reshape(nct, 16, tln, 8).astype(np.int64)
    prod = tl[nct * 16 * tln * 8:].reshape(nct, tln, 8).astype(np.int64)
    cs, ce = z["cstart"].reshape(nl, nct)[-1], z["cend"].reshape(nl, nct)[-1]
    out = {k: [] for k in ("stages", "units", "period_int", "period_bnd", "bubble", "eoi", "prod_bnd", "prod_int", "idle", "idle_bnd", "wait_first", "wait_int", "span", "pfull_bnd", "pfull_int")}
    for c in range(nct):
        w0 = cons[c, 0, :, 5]
        ns = int((w0 & 0xffff > 0).sum())
        if ns < 4:
            continue
        sl = slice(0, ns)
        tw0, tw1, tw2, te0, te1, w0 = (cons[c, :, sl, k] for k in (0, 1, 2, 3, 4, 5))
        w0 = w0[0]
        last = (w0 & 0x20000) > 0; first = (w0 & 0x10000) > 0
        span_cyc = max(tw2.max(), te1.max())
        us = (ce[c] - cs[c]) / 1e3 / span_cyc                              # us per cycle for this CTA
        F = tw1.min(0); Fmax = tw1.max(0)
        I = prod[c, sl, 1]
        per = np.diff(F)
        bnd = last[:-1]                                                     # the step i -> i+1 crosses a unit boundary
        pi, pb = per[~bnd], per[bnd]
        if len(pb) == 0:
            continue
        e_in = np.where(te0 > 0, te0, 1 << 60).min(0); e_out = te1.max(0)
        eoi_d = (e_out - e_in)[last]
        pdur = prod[c, sl, 1] - prod[c, sl, 0]
        newu = (prod[c, sl, 4] & 1) > 0
        pfd = prod[c, sl, 7] - prod[c, sl, 0]                                # produce() incl. the record prefetch step (KVQ_RPF)
        # idle: union complement of [I_i, F_i] between the first issue and the last F
        iv = sorted(zip(I.tolist(), F.tolist()))
        idle = 0; idle_b = 0; cur_end = iv[0][1]
        prev_last = np.concatenate([[False], last[:-1]])                    # stage i directly follows a unit end
        order = np.argsort(I)
        cur_end = F[order[0]]
        for i in order[1:]:
            if I[i] > cur_end:
                idle += I[i] - cur_end
                if prev_last[i] or first[i]:
                    idle_b += I[i] - cur_end
            cur_end = max(cur_end, F[i])
        wait = (tw1 - tw0).mean(0)
        out["stages"].append(ns); out["units"].append(int(last.sum()))
        out["period_int"].append(pi.mean() * us); out["period_bnd"].append(pb.mean() * us); out["bubble"].append((pb.mean() - pi.mean()) * us)
        out["eoi"].append(eoi_d.mean() * us if len(eoi_d) else np.nan)
        out["prod_bnd"].append(pdur[newu].mean() * us if newu.any() else np.nan); out["prod_int"].append(pdur[~newu].mean() * us if (~newu).any() else np.nan)
        out["idle"].append(idle * us); out["idle_bnd"].append(idle_b * us)
        out["wait_first"].append(wait[first].mean() * us); out["wait_int"].append(wait[~first & ~last].mean() * us if (~first & ~last).any() else np.nan)
        out["span"].append(span_cyc * us)
        out["pfull_bnd"].append(pfd[newu].mean() * us if newu.any() else np.nan); out["pfull_int"].append(pfd[~newu].mean() * us if (~newu).any() else np.nan)
    return {k: (float(np.nanmedian(v)) if v else float("nan")) for k, v in out.items()}, len(out["stages"])


if __name__ == "__main__":
    keys = ("stages", "units", "period_int", "period_bnd", "bubble", "eoi", "prod_bnd", "prod_int", "pfull_bnd", "pfull_int", "idle", "idle_bnd", "wait_first", "wait_int", "span")
    print("file".ljust(48), *[k.rjust(10) for k in keys], "nCTA")
    for p in sys.argv[1:]:
        r, n = analyze(p)
        print(p.split("/")[-1][:47].ljust(48), *[f"{r[k]:10.3f}" for k in keys], n)
