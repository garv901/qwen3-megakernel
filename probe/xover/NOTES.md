# xover notes (CPU-side checks, not yet run on a GPU)

## Build (nvcc 12.x, sm_100a, -O3, -Xptxas -v; `bash run.sh --build-only`)
| binary | ENG | regs | spill | stack |
|---|---|---|---|---|
| bin/xover_em1 | -1 oracle | 112 | 0 | 0 |
| bin/xover_e0 | 0 FFMA | 112 | 0 | 0 |
| bin/xover_e1 | 1 tcgen05 | 116 | 0 | 0 |
| bin/xover_e2 | 2/3 both paths | 119 | 0 | 0 |
512 thr x 1 CTA/SM caps regs at 128. X staging is done in two halves of 4 x 16 B per thread; with 8 in flight, e0/e2 spilled 8/32 B.
The JSON "spill" field is cudaFuncAttributes.localSizeBytes (0 here).

## SASS (cuobjdump -sass, kernel + k_ref/k_init/hash FFMAs are 15 of the FFMA counts)
- tcgen05 MMA mnemonic: `UTCHMMA` (8 per binary = the 8 unrolled K16 steps), plus `UTCBAR` (commit), `UTCATOMSWS.*` (alloc/dealloc), `LDTM.` (tcgen05.ld).
- ENG=0 (e0) and ENG=-1: no UTCHMMA / UTCBAR / UTCATOMSWS / LDTM. e0 has 143 FFMA (128 in the TB=4 token tile + 15 elsewhere).
- ENG=1 (e1): 8 UTCHMMA, 2 LDTM, 32 REDG.ADD.F32 (flush, 2x16).
- ENG=2 (e2): 143 FFMA and 8 UTCHMMA and 2 LDTM, i.e. both paths compiled (eng flag chooses at runtime).
- FFMA count of the ENG0 engine: exactly T*N*K FFMA per phase (+ one shift/and ALU op per X element: 32 per 32 FFMA).

## Notes
- ENG0 red.adds: every unit's 128 x T sums; ENG1: once per row-block segment (D: up to ~2x fewer). Oracle (-1): same count as ENG0, adding 0.
- ENG1 reads back only ceil(T/16) 16-column chunks of TMEM, not all Npad columns.
- Correctness only for T in {1,8,64,256} (PASS/FAIL on stderr, `pass` in JSON; null = not checked, oracle never checked).
