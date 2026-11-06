"""Tables from dump.py output. Usage: analyze.py DIR.
Per phase (layers 4..23, median over layers and launches), times in us:
  T     = phase time = median exit of this barrier - median exit of the previous barrier
  work  = median over CTAs of (own arrive - own previous exit)
  busy  = median work of CTAs WITH tiles / units;  idle = median work of CTAs without (finalizer CTAs in TS O / D)
  last  = max over CTAs of work (the CTA that sets the barrier);  lastgrp = group (busy / idle) of the last arriver
Tile / unit counts: classic split-K units of 128 x 128 k; tile stream tiles (TT = NPAD, KS 4 / 1 / 12)."""
import glob
import json
import sys

import numpy as np

PH = ("QKV", "RoPE", "attn", "O", "GU", "D")
NB = 1 + 6 * 28
UNITS = {0: 256, 3: 128, 4: 384, 5: 192}
TILES = {0: 128, 3: 128, 4: 48, 5: 96}   # O stays classic in both engines


def counts(total, G):
    return np.array([(c + 1) * total // G - c * total // G for c in range(G)])


def table(path):
    z = np.load(path)
    st, kn = z["st"], json.loads(str(z["knobs"]))
    L, G, _ = st.shape
    b = int(path.split("-b")[1].split("-")[0])
    npad = next(x for x in (16, 32, 64, 128, 256) if x >= b)
    mts = kn.get("MK_TS", 128)
    ts = mts != 0 and (128 if mts == 1 else mts) <= npad
    rows = {}
    for p in range(6):
        res = {k: [] for k in ("T", "work", "busy", "idle", "last", "lastbusy")}
        n = None
        if p in UNITS:
            n = counts((TILES if ts and p != 3 else UNITS)[p], G)
        for l in range(L):
            arr, wex = st[l, :, 0:2 * NB:2], st[l, :, 1:2 * NB:2]
            for layer in range(4, 24):
                b = 1 + 6 * layer + p
                wk = (arr[:, b] - wex[:, b - 1]) / 1e3
                res["T"].append((np.median(wex[:, b]) - np.median(wex[:, b - 1])) / 1e3)
                res["work"].append(np.median(wk)); res["last"].append(wk.max())
                if n is not None:
                    res["busy"].append(np.median(wk[n > 0]))
                    res["idle"].append(np.median(wk[n == 0]) if (n == 0).any() else np.nan)
                    res["lastbusy"].append(float(n[int(arr[:, b].argmax())] > 0))
        rows[PH[p]] = {k: (float(np.nanmedian(v)) if len(v) else np.nan) for k, v in res.items()}
        rows[PH[p]]["n"] = (int((n > 0).sum()), int(n.max())) if n is not None else None
    wx = st[:, :, 1:2 * NB:2]
    tot = float(np.mean([np.median(wx[l, :, 1 + 6 * 23 + 5]) - np.median(wx[l, :, 1 + 6 * 3 + 5]) for l in range(L)])) / 1e3 / 20
    return rows, tot, ts


for path in sorted(glob.glob(f"{sys.argv[1]}/*.npz")):
    rows, tot, ts = table(path)
    print(f"\n{path.split('/')[-1][:-4]}  (engine {'TS' if ts else 'classic'}; layer time {tot:.2f} us)")
    print("  phase |     T |  work |  busy |  idle |  last | last=busy | CTAs w/ work, max")
    for k, r in rows.items():
        f = lambda x: "    - " if x != x else f"{x:6.2f}"
        print(f"  {k:5s} |{f(r['T'])} |{f(r['work'])} |{f(r['busy'])} |{f(r['idle'])} |{f(r['last'])} |   {f(r['lastbusy'])}  | {r['n']}")
