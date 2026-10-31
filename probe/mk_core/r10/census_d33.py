"""R10 step-shape census: decode-only steps with 33-256 requests, conv/code trace, stock arm step logs.
Prints per (trace, load): GPU-time-weighted distributions of n_req, mean kv, max kv, total kv tokens,
and a 2-D histogram (n bucket x mean-kv bucket) of stock GPU time share. Read-only."""
import json, glob, sys, statistics as st
from collections import defaultdict
ROOT = '/raid/garv901/regime-mk-serve/runs/e2e'
def wq(xs, ws, f):
    z = sorted(zip(xs, ws)); tot = sum(ws); a = 0
    for x, w in z:
        a += w
        if a >= f * tot: return x
    return z[-1][0]
NB = [(33, 48), (49, 64), (65, 96), (97, 128), (129, 192), (193, 256)]
KB = [(0, 2048), (2048, 4096), (4096, 6144), (6144, 8192), (8192, 1 << 30)]
for tr in sys.argv[1:] or ('conv', 'code'):
    for L in ('L3', 'L4'):
        rows = []
        for f in sorted(glob.glob(f'{ROOT}/*-m2_{tr}_*_{L}/stock/steps.jsonl')):
            for line in open(f):
                r = json.loads(line)
                if r.get('fwd_us') is None: continue
                q, kv = r['q'], r['kv']
                if not all(x == 1 and k > 0 for x, k in zip(q, kv)): continue
                if not (33 <= len(q) <= 256): continue
                rows.append((len(q), st.mean(kv), max(kv), sum(kv), r['fwd_us'], kv))
        if not rows: continue
        w = [r[4] for r in rows]
        print(f'== {tr} {L}: {len(rows)} steps, stock GPU us total {sum(w)/1e6:.2f} s; fwd_us p50 {wq([r[4] for r in rows],[1]*len(rows),.5):.0f}')
        for name, i in (('n_req', 0), ('mean kv', 1), ('max kv', 2), ('total kv tok', 3)):
            xs = [r[i] for r in rows]
            print(f'  {name:13s} time-weighted p10/p50/p90/max {wq(xs,w,.1):.0f}/{wq(xs,w,.5):.0f}/{wq(xs,w,.9):.0f}/{max(xs):.0f}')
        # cv of kv within a step
        cv = [st.pstdev(r[5]) / max(1, r[1]) for r in rows]
        print(f'  kv CV within step p50/p90 {wq(cv,w,.5):.2f}/{wq(cv,w,.9):.2f}')
        h = defaultdict(float)
        for r in rows:
            nb = next(i for i, (a, b) in enumerate(NB) if a <= r[0] <= b)
            kb = next(i for i, (a, b) in enumerate(KB) if a <= r[1] < b)
            h[nb, kb] += r[4]
        T = sum(w)
        print('  time share % by n_req (rows) x mean kv (cols: <2K 2-4K 4-6K 6-8K >=8K)')
        for i, (a, b) in enumerate(NB):
            print(f'   n {a:3d}-{b:3d}: ' + ' '.join(f'{100*h[i,j]/T:5.1f}' for j in range(len(KB))))
