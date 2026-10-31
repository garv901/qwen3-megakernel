"""Per-point table of a regimebench run: mk vs best fi-graph variant vs best vLLM variant (steady median us/step).
Usage: table.py RUN_DIR [RUN_DIR...]"""
import json, sys
from collections import defaultdict
d = defaultdict(dict); acc = defaultdict(list); order = []
for rd in sys.argv[1:]:
    for l in open(f'{rd}/records.jsonl'):
        r = json.loads(l)
        if r.get('record_type') == 'timing':
            wid = r['workload']['id']
            if wid not in order: order.append(wid)
            d[wid][f"{r['arm']}:{r['variant']}"] = r['steady']['median'] * 1e3
        elif r.get('record_type') == 'accuracy' and r.get('arm') != 'reference':
            acc[r['workload']['id']].append(f"{r['arm']}:{r.get('verdict')}")
print('| point | mk | best fi-graph | best vLLM | mk / best | win |')
print('|---|---|---|---|---|---|')
wins = 0; ratios = []
for w in order:
    x = d[w]
    mk = min((v for k, v in x.items() if k.startswith('mk:')), default=None)
    fi = min(((v, k) for k, v in x.items() if k.startswith('fi-graph:')), default=(None, ''))
    vl = min(((v, k) for k, v in x.items() if k.startswith('vllm:')), default=(None, ''))
    best = min(v for v in (fi[0], vl[0]) if v is not None)
    r = mk / best if mk else None
    ratios.append(r); wins += bool(r and r < 1)
    f = lambda v: f'{v:.1f}' if v else '-'
    print(f"| {w} | {f(mk)} | {f(fi[0])} ({fi[1].split(':')[-1]}) | {f(vl[0])} | {r:.3f} | {'yes' if r < 1 else 'no'} |")
import math
print(f'\nwins {wins}/{len(order)}; geomean mk/best {math.exp(sum(math.log(r) for r in ratios)/len(ratios)):.3f}')
bad = {w: a for w, a in acc.items() if any(not s.endswith(('pass', 'PASS')) for s in a)}
print('accuracy:', 'all pass' if not bad else bad, f'({sum(len(a) for a in acc.values())} records)')
