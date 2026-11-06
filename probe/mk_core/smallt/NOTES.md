# smallt: classic vs tile stream at NPAD 16 / 32 / 64 (job 14155, c2bc494, placeholder attention)
Run: `sbatch -p main --time=00:35:00 probe/mk_core/smallt/smallt.sbatch`; tables: `$PY probe/mk_core/smallt/analyze.py results/2026-10-29-smallt/st_14155`.
Sets: C = MK_TS 128 (classic at these buckets), T = MK_TS 16 (full tile stream), Tq = MK_TS 16 + MK_TSB 0 (QKV only), *_null = MK_NULL 0x39.
Phase time T = median exit - previous median exit (layers 4..23, 5 launches); globaltimer steps are 0.256 us, so per-phase T is +-0.13.

Phase delta T(TS) - T(C), us/layer:  b1 / b8 / b32 / b64
  QKV -0.77 / -0.52 / -0.51 / -1.28   O +1.53 / +1.79 / +1.79 / +2.82   GU +1.02 / +1.03 / +1.28 / +2.31   D +0.77 / +0.52 / +0.51 / -1.02
  layer +2.48 / +2.84 / +3.00 / +1.92;  e2e A/B (7 rounds) T +79.0 / +90.5 / +114.3 / (b64 lost: Tq hang), Tq +34.5 / +34.3 / +62.7
- O (same engine): last arriver is a finalizer CTA in 100 % of layers; finalizer tail 3.54 vs 1.73 (b8); busy CTAs +0.5 (tickets + fence).
- GU: 48 tiles (KS 1); NULL floor of the busy CTAs 2.62-3.20 vs classic 1.46-1.50 (256 KB A per CTA vs <= 96 KB, 2 units prefetched).
- Tq: D +2.31 (b8): the xb ticket finalizer on a busy CTA costs more than the QKV gain. Tq hangs at NPAD 64 (dump timeout; A/B b64 lost).
