import json, sys, numpy as np
sys.path.insert(0, "/raid/garv901/vibe-megakernel")
from megabench.mkprobe.timeline import iket_adapter as A
d = A.load(A.find_trace(sys.argv[1])); ln = A.launches(d)[-20:][5]
sm = {}
for w in ln["warpLifetimes"]:
    loc = A._get(d, w, "locIdx", "warpLocation", "locationTable")
    sm[int(loc["ctaId"][0])] = int(loc["smId"])
json.dump(sm, open(sys.argv[2], "w"))
print(len(sm))
