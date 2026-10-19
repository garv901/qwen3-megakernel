"""CPU: decode the IKET fx1 28-layer trace (job 6357, build ik1a) into a small npz + json for the README charts.

Picks, of the last 20 marker launches, the launch with the median kernel span. Saves T[cta, layer, phase, local, warp] (ns from the
earliest warp start) and the warp lifetimes, plus a summary of per-layer phase timings.
"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/raid/garv901/vibe-megakernel")
from megabench.mkprobe.timeline import iket_adapter as A

RUN = Path("/raid/garv901/vibe-megakernel/megabench/mkprobe/runs/20261012T220000Z-IKET-fx1/out-6357/ik1a_iket/iket")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

p = A.find_trace(RUN)
print("trace", p, flush=True)
d = A.load(p)
lns = A.launches(d)[-20:]
spans = []
for ln in lns:
    wl = ln["warpLifetimes"]
    spans.append(max(w["endTs"] for w in wl) - min(w["startTs"] for w in wl))
order = np.argsort(spans)
pick = int(order[len(order) // 2])
print("spans (ns)", sorted(spans), "pick", pick, spans[pick], flush=True)
T, info = A.array_of_launch(d, lns[pick], 28)
print("info", info, flush=True)
caus = A.causality(T)
print("causality", json.dumps(caus), flush=True)
t0 = info["t0"]
life = np.full((A.NB, A.NWARP, 2), np.nan)
for w in lns[pick]["warpLifetimes"]:
    loc = A._get(d, w, "locIdx", "warpLocation", "locationTable")
    c, wi = int(loc["ctaId"][0]), int(loc["warpId"])
    if c < A.NB and wi < A.NWARP:
        life[c, wi] = (w["startTs"] - t0, w["endTs"] - t0)
np.savez_compressed(OUT / "ik1a_launch.npz", T=T, life=life)
json.dump({"spans_ns": spans, "pick": pick, "span_ns": spans[pick], "info": {k: v for k, v in info.items() if k != "t0"},
           "causality": caus}, open(OUT / "ik1a_launch.json", "w"), indent=1)
print("saved", flush=True)
