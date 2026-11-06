# why: static vs dynamic split at 148 CTAs x 1 GiB (jobs 11630 11636 11648 11659 11678 final; chk 11667/11671/11675)
Run: `sbatch -p priority --time=00:25:00 probe/mk_core/why/why.sbatch`; table: `$PY analyze.py 11678`. Results: results/2026-10-28-why/ (analysis_11678.txt).
Arm suffixes: _al (4096 aligned 256 KB chunks), _shK (+K*16 B), _syncK (CTA sync every K x 64 KB), _wrot1 (warp slot rotation per batch).
Caveat: queue arms ran 36 us slow in jobs 11659/11667/11671 (a codegen change from a runtime wrot branch in stream(); fixed with a template, job 11675/11678). Use 11678.
