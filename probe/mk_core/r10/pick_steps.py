"""Pick real decode steps (33-256 reqs) from the stock step logs nearest to target (n, mean kv), with total
cache tokens under a cap. Prints candidates with source file and line number. Read-only."""
import json, glob, statistics as st, sys
ROOT = '/raid/garv901/regime-mk-serve/runs/e2e'
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 337_000
targets = [('conv', 'L3', 54, 2930), ('conv', 'L4', 87, 2790), ('conv', 'L4', 150, 2200), ('code', 'L3', 40, 2440)]
for tr, L, n0, k0 in targets:
    best = []
    for f in sorted(glob.glob(f'{ROOT}/*-m2_{tr}_*_{L}/stock/steps.jsonl')):
        for i, line in enumerate(open(f)):
            r = json.loads(line)
            if r.get('fwd_us') is None: continue
            q, kv = r['q'], r['kv']
            if not (33 <= len(q) <= 256) or not all(x == 1 and k > 0 for x, k in zip(q, kv)): continue
            tot = sum(kv) + len(kv)
            if tot > CAP: continue
            d = abs(len(q) - n0) / n0 + abs(st.mean(kv) - k0) / k0
            best.append((d, f.split('/')[-3], i, len(q), round(st.mean(kv)), max(kv), tot, round(st.pstdev(kv) / st.mean(kv), 2), r['fwd_us']))
    best.sort()
    print(f'== target {tr} {L} n {n0} mean kv {k0}')
    for b in best[:3]: print('  ', b[1:])
