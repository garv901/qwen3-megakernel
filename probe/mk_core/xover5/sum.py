# xover5 summary: us per phase per (shape, T, bin) = median over (job, round) samples; pair deltas with their min..max over samples.
# Usage: python3 sum.py logs/slurm-A.log [logs/slurm-B.log ...]
import json, re, sys, statistics as st
from collections import defaultdict
v = defaultdict(dict); fails = []
for path in sys.argv[1:]:
    text = open(path).read()   # stderr check lines can land inside a block-buffered JSON line (run.sh before the fix): cut them out first
    chk = re.compile(r'(PASS|FAIL|SKIP) \S+ \S+ T=\d+ max\|d\|=\S+ bad=\d+(?: \(4096-row sample\))?\n')
    fails += [path + ': ' + m.group(0).strip() for m in chk.finditer(text) if m.group(1) == 'FAIL']
    for line in chk.sub('', text).splitlines():
        if not line.startswith('{') or '"floor"' in line: continue
        d = json.loads(line)
        if d.get('pass') is False: fails.append(path + ': ' + line.strip())
        v[(d['shape'], d['T'], d['bin'])][(path, d['round'])] = d['us_phase']
PAIRS = [('c0', 'a0', 'X cp.async vs LDG (ring 2)'), ('c3', 'x3', 'X cp.async vs LDG (ring max)'), ('c3', 'a3', 'X cp.async vs TMA'),
         ('w3', 'a3', 'W cp.async vs TMA'), ('wc3', 'c3', 'W cp.async vs TMA (X cp.async)'), ('wc3', 'a3', 'all cp.async vs all TMA')]
BINS = ['a0', 'c0', 'x3', 'c3', 'a3', 'w3', 'wc3']
shapes = ['QKV', 'O', 'GU', 'D', 'LM']; Ts = sorted({k[1] for k in v})
print('FAIL lines:', len(fails)); [print('  ', f) for f in fails[:20]]
print('\n## us per phase (median of samples)\n| shape | T | ' + ' | '.join(BINS) + ' |\n|' + '---|' * (len(BINS) + 2))
for s in shapes:
    for t in Ts:
        row = [('%.2f' % st.median(v[(s, t, b)].values())) if v.get((s, t, b)) else '-' for b in BINS]
        if any(r != '-' for r in row): print(f'| {s} | {t} | ' + ' | '.join(row) + ' |')
print('\n## deltas (us, median [min..max] over paired samples; negative = first arm faster)')
for a, b, name in PAIRS:
    print(f'\n### {a} - {b}: {name}\n| shape | ' + ' | '.join(f'T={t}' for t in Ts) + ' |\n|' + '---|' * (len(Ts) + 1))
    for s in shapes:
        cells = []
        for t in Ts:
            A, B = v.get((s, t, a), {}), v.get((s, t, b), {})
            ks = sorted(set(A) & set(B))
            if not ks: cells.append('-'); continue
            d = [A[k] - B[k] for k in ks]
            cells.append('%+.2f [%+.2f..%+.2f]' % (st.median(d), min(d), max(d)))
        print(f'| {s} | ' + ' | '.join(cells) + ' |')
