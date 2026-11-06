"""Per-CTA stamp metrics with mk's definitions (as probe/mk_core/smallt/analyze.py). Library + CLI.
Raw file: concatenated launches of uint64 [grid][NSTMAX]; stamp 2p = arrive at barrier p, 2p+1 = exit of barrier p (barrier 0 = start).
Phase p (1-based) = the work between barrier p-1 exit and barrier p arrive.
  T    = median exit(p) - median exit(p-1)            (phase time, what a sequence of phases costs)
  work = median over CTAs of arrive(p) - exit(p-1)
  busy / idle = median work of CTAs in / not in `mask`  (e.g. CTAs with tiles)
  last = max over CTAs of work; spread = max arrive - median arrive; lat = median exit(p) - max arrive(p)
Usage: stamps.py FILE GRID NSTMAX NPH [FIRST_PHASE=2] [busy CTA list "0-47" or "" ]"""
import sys

import numpy as np


def load(path, grid, nstmax):
    a = np.fromfile(path, dtype=np.uint64).astype(np.int64)
    return a.reshape(-1, grid, nstmax)


def metrics(st, nph, first=2, mask=None):
    """st [launches][grid][nstmax]; phases first..nph (skip warm-up phases); returns dict of medians in us."""
    out = {k: [] for k in ("T", "work", "busy", "idle", "last", "spread", "lat")}
    tmean = []
    for L in st:
        arr, ex = L[:, 0::2], L[:, 1::2]
        # telescoping mean T: medians of stamps sit on a 0.256 us grid, this mean resolves ~ns (B0)
        tmean.append((np.median(ex[:, nph]) - np.median(ex[:, first - 1])) / 1e3 / (nph - first + 1))
        for p in range(first, nph + 1):
            wk = (arr[:, p] - ex[:, p - 1]) / 1e3
            out["T"].append((np.median(ex[:, p]) - np.median(ex[:, p - 1])) / 1e3)
            out["work"].append(np.median(wk)); out["last"].append(wk.max())
            out["spread"].append((arr[:, p].max() - np.median(arr[:, p])) / 1e3)
            out["lat"].append((np.median(ex[:, p]) - arr[:, p].max()) / 1e3)
            if mask is not None:
                out["busy"].append(np.median(wk[mask]) if mask.any() else np.nan)
                out["idle"].append(np.median(wk[~mask]) if (~mask).any() else np.nan)
    r = {k: (float(np.nanmedian(v)) if len(v) else float("nan")) for k, v in out.items()}
    r["Tmean"] = float(np.median(tmean)) if tmean else float("nan")
    return r


def parse_mask(s, grid):
    m = np.zeros(grid, bool)
    for part in filter(None, s.split(",")):
        a, _, b = part.partition("-")
        m[int(a):int(b or a) + 1] = True
    return m


if __name__ == "__main__":
    f, g, n, nph = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    first = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    mask = parse_mask(sys.argv[6], g) if len(sys.argv) > 6 and sys.argv[6] else None
    print(metrics(load(f, g, n), nph, first, mask))
