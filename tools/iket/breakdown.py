import numpy as np, sys, json
z = np.load(sys.argv[1]); T = z["T"].copy(); life = z["life"]
nb, nl = T.shape[:2]; TOL = 32
# per (cta, warp): sequence of 4*4*28 markers must be non-decreasing (tol); else mark the warp from the first violation on as bad
T[:, :, :, 0, :] = np.nan  # BWE: warp 0 records one extra Q_BWE in the prologue wait in some CTAs, which shifts its layer index; not used below
seq = T.transpose(0, 4, 1, 2, 3).reshape(nb, 16, -1)  # [cta, warp, layer*16]
bad = np.zeros_like(seq, bool)
for c in range(nb):
    for w in range(16):
        s = seq[c, w]; d = np.diff(s)
        v = np.where(np.isfinite(d) & (d < -TOL))[0]
        if v.size: bad[c, w, v[0]:] = True
print("warps with an order violation:", int(bad.any(-1).sum()), "of", nb * 16, "; markers masked:", int(bad.sum()))
seq[bad] = np.nan
T = seq.reshape(nb, 16, nl, 4, 4).transpose(0, 2, 3, 4, 1)
print("warp-0 bad:", int(bad[:, 0].any(-1).sum()), "CTAs")
W0 = T[..., 0]  # [cta, layer, phase, local]
BWX, SAL, SAE = W0[..., 1], W0[..., 3], np.nanmax(T[..., 2, :], axis=-1)
PH = "QFGD"
res = {}
for p in range(4):
    nxt = (lambda l: BWX[:, l, p + 1]) if p < 3 else (lambda l: BWX[:, l + 1, 0])
    L = range(nl) if p < 3 else range(nl - 1)
    work = (SAL[:, :, p] - BWX[:, :, p])
    hop, skew, span, lastin = [], [], [], []
    for l in L:
        last = np.nanmax(SAL[:, l, p]); med = np.nanmedian(SAL[:, l, p])
        hop.append(np.nanmedian(nxt(l)) - last); skew.append(last - med)
        span.append(np.nanmedian(nxt(l)) - np.nanmedian(BWX[:, l, p]))
        lastin.append(int(np.nanargmax(SAL[:, l, p])))
    res[PH[p]] = dict(work=float(np.nanmedian(work)), skew=float(np.median(skew)), hop=float(np.median(hop)), span=float(np.median(span)),
                      hop_p10=float(np.percentile(hop, 10)), hop_p90=float(np.percentile(hop, 90)), last_cta_mode=int(np.bincount(lastin).argmax()))
for k, v in res.items(): print(k, {a: round(b) if isinstance(b, float) else b for a, b in v.items()})
start = np.nanmin(life[..., 0]); end = np.nanmax(life[..., 1])
q0 = np.nanmedian(BWX[:, 0, 0]); dlast = np.nanmax(SAL[:, nl - 1, 3])
print("kernel span", end - start, "prologue (start -> median Q BWX L0)", q0 - start, "layers (Q BWX L0 -> last D arrive)", dlast - q0, "LM head + argmax (last D arrive -> end)", end - dlast)
lay = [np.nanmedian(BWX[:, l + 1, 0]) - np.nanmedian(BWX[:, l, 0]) for l in range(nl - 1)]
print("layer time median", np.median(lay), "min", min(lay), "max", max(lay))
json.dump(dict(phases=res, kernel_span=float(end - start), prologue=float(q0 - start), layers=float(dlast - q0), lm_head=float(end - dlast),
               layer_median=float(np.median(lay))), open(sys.argv[2], "w"), indent=1)
