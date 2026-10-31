#!/usr/bin/env python3
"""Tables from an xover6 slurm log: python3 sum.py runs/slurm-<job>.log"""
import sys, json, re, statistics as st
from collections import defaultdict
lines = open(sys.argv[1]).read().split('\n')
SH = ['QKV', 'O', 'GU', 'D']
recs = []; ctx = ''; npass = nfail = 0; fails = []
for l in lines:
    if l.startswith('## '): ctx = l[3:]; continue
    if l.startswith('PASS'): npass += 1
    if l.startswith('FAIL'): nfail += 1; fails.append(l)
    if l.startswith('{'):
        try: r = json.loads(l)
        except Exception: continue
        r['_ctx'] = ctx; recs.append(r)
def med(x): return st.median(x) if x else float('nan')
kv = [r for r in recs if r.get('mode') == 'kv']
print(f'GEMM PASS lines: {npass}, FAIL lines: {nfail}'); [print(' ', f) for f in fails[:10]]
bad_sink = [r for r in kv + [r for r in recs if r.get('mode') == 'both'] if not r.get('sink_ok')]
print(f'kv sink_ok failures: {len(bad_sink)}')
# ---- table A: first 2 records of kv are the drift check (start); exclude the first and last of the 148/4 configs from the grid by using the sweep block
A = defaultdict(list)
for r in kv: A[(r['k_a'], r['depth'])].append(r)
print('\nTable A: kv alone, aggregate GB/s (bytes_per_iter 512 MiB, 20 its). total = all iterations / (last end - first start); ss = steady state (skips first 4 its); it = median of per-iteration (max end - min start) (no grid barrier, so CTA drift lowers it)')
print('k_a  | ' + ' | '.join(f'depth {d}: total / ss / it   per-CTA' for d in (2, 4, 6)))
for ka in (148, 128, 116, 104, 92, 74):
    row = []
    for d in (2, 4, 6):
        rs = A.get((ka, d), [])
        if not rs: row.append('-'); continue
        r = rs[1] if (ka, d) == (148, 4) and len(rs) > 2 else rs[0]   # 148/4: [start drift, grid, end drift]
        row.append(f"{r['agg_gbs_total']:7.0f} / {r['agg_gbs_ss']:6.0f} / {r['agg_gbs']:6.0f}  {r['cta_gbs']:6.1f}")
    print(f'{ka:4d} | ' + ' | '.join(row))
base = A[(148, 4)][1]['agg_gbs_total'] if len(A.get((148, 4), [])) > 2 else float('nan')
print('ratio to 148-SM depth-4 total rate (%.0f):' % base)
for ka in (148, 128, 116, 104, 92, 74):
    print(f'  k_a {ka:3d}: ' + '  '.join(f"d{d} {A[(ka, d)][1 if (ka, d) == (148, 4) and len(A[(ka, d)]) > 2 else 0]['agg_gbs_total'] / base:5.3f}" for d in (2, 4, 6) if A.get((ka, d))))
if len(A.get((148, 4), [])) >= 3: print('drift kv 148/4 total GB/s: start %.0f, grid %.0f, end %.0f' % tuple(A[(148, 4)][i]['agg_gbs_total'] for i in range(3)))

# ---- table B: gemm best per (k_c, shape, T)
G = defaultdict(list)
for r in recs:
    if r.get('mode') == 'gemm' and r['_ctx'].startswith('gemm') and r['pass'] is not False: G[(r['k_c'], r['shape'], r['T'])].append(r)
best = {k: min(v, key=lambda r: r['us_phase']) for k, v in G.items()}
print('\nTable B: gemm alone, best us/phase (TT/KS) per shape; sum = QKV+O+GU+D')
for T in (32, 64, 128):
    print(f'T={T}:')
    print('  k_c  | ' + ' | '.join(f'{s:>14}' for s in SH) + ' |   sum')
    for kc in (20, 32, 44, 56, 74, 111, 148):
        if not all((kc, s, T) in best for s in SH): continue
        print(f'  {kc:4d} | ' + ' | '.join(f"{best[(kc, s, T)]['us_phase']:6.2f} ({best[(kc, s, T)]['TT']}/{best[(kc, s, T)]['KS']})".rjust(14) for s in SH) + f" | {sum(best[(kc, s, T)]['us_phase'] for s in SH):6.2f}")
# ---- reproduction vs GRID_N
print('\nReproduction: run-time k_c vs compile-time GRID_N build (xover4.cu), best us/phase over the common configs (TT in {T, min(T,64)})')
R = defaultdict(list)
for r in recs:
    if r.get('bin') == 'ts' and r['_ctx'].startswith('ref'):
        g = int(r['_ctx'].split()[0][3:]); T = r['T']
        if r['TT'] in (T, min(T, 64)): R[(g, r['shape'], T)].append(r['us_phase'])
mx = 0
for T in (64, 128):
    for kc in (148, 111, 74):
        out = []; sa = sb = 0
        for s in SH:
            a = best.get((kc, s, T)); b = min(R[(kc, s, T)]) if R.get((kc, s, T)) else None
            if a and b: out.append(f"{s} {a['us_phase']:.2f}/{b:.2f}"); sa += a['us_phase']; sb += b; mx = max(mx, abs(a['us_phase'] / b - 1))
        if sb: print(f'  T={T} k_c={kc:3d}: runtime/GRID_N  ' + '  '.join(out) + f'   sum {sa:.2f}/{sb:.2f} = {sa / sb:.3f}')
print(f'  max per-shape deviation: {mx * 100:.1f} %')
d0 = [r for r in recs if r.get('mode') == 'gemm_best' and r['shape'] == 'GU' and r['T'] == 64 and r['k_c'] == 74]
if len(d0) >= 2: print('drift gemm k_c 74 T64 GU best us/phase: ' + ' ... '.join(f"{r['us_phase']:.3f}" for r in (d0[0], d0[-1])))

# ---- table C
Bt = defaultdict(list)
for r in recs:
    if r.get('mode') == 'both' and r['pass'] is not False: Bt[(r['k_a'], r['depth'], r['T'], r['shape'])].append(r)
print('\nTable C: both modes. gemm sum = sum over QKV/O/GU/D of us/phase (median of 3 reps) with the stream running vs alone (same config, same process); KV GB/s = mean over the 4 shapes of the rate inside the overlap window vs the kv-alone steady-state rate of the same process')
print(' k_a/k_c dp   T | gemm both / alone  ratio | KV win  / alone   ratio | min overlap | ts ms | kv iters (min)')
for ka in (128, 116, 104, 92):
    for dp in (4, 6):
        for T in (32, 64, 128):
            if not all(Bt.get((ka, dp, T, s)) for s in SH): continue
            gb = sum(med([r['gemm_us_both'] for r in Bt[(ka, dp, T, s)]]) for s in SH)
            ga = sum(Bt[(ka, dp, T, s)][0]['gemm_us_alone'] for s in SH)
            kw = st.mean(med([r['kv_gbs_win'] for r in Bt[(ka, dp, T, s)]]) for s in SH)
            kl = st.mean(Bt[(ka, dp, T, s)][0]['kv_gbs_alone'] for s in SH)
            ov = min(r['ovl_min'] for s in SH for r in Bt[(ka, dp, T, s)])
            tm = min(r['ts_ms'] for s in SH for r in Bt[(ka, dp, T, s)])
            ki = min(r['kv_iters'] for s in SH for r in Bt[(ka, dp, T, s)])
            print(f' {ka:3d}/{148 - ka:3d} {dp:2d} {T:4d} | {gb:7.2f} / {ga:6.2f}  {gb / ga:5.2f} | {kw:6.0f} / {kl:6.0f}  {kw / kl:5.3f} | {ov:.4f} | {tm:5.1f} | {ki}')
print('\nPer shape (k_a/depth/T -> gemm both vs alone us/phase, cfg TT/KS):')
for ka in (128, 116, 104, 92):
    for T in (32, 64, 128):
        dp = 4
        if not all(Bt.get((ka, dp, T, s)) for s in SH): continue
        print(f'  k_a {ka} d{dp} T{T}: ' + '  '.join(f"{s} {med([r['gemm_us_both'] for r in Bt[(ka, dp, T, s)]]):.2f}/{Bt[(ka, dp, T, s)][0]['gemm_us_alone']:.2f} ({Bt[(ka, dp, T, s)][0]['TT']}/{Bt[(ka, dp, T, s)][0]['KS']})" for s in SH))
info = [r for r in recs if r.get('mode') == 'info']
if info: print(f"\nregs/spill: k_ts {info[0]['regs_ts']} / {info[0]['spill_ts']} B local, k_kv {info[0]['regs_kv']} / {info[0]['spill_kv']} B local")
