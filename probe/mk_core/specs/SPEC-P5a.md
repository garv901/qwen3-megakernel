# SPEC-P5a: tile-stream engine in mk; QKV on it with B = bf16 xb by TMA (NPAD 128 / 256)

## Why (PASS-04.md)
- At R1-b256, mk's GEMM phases take 70.7 us/layer (job 9593); the xover3 tile-stream (ts) engine does them in 28.5 (job 9592).
- ts only wins with B by TMA from bf16 (job 9595: fused fp32 staging in ts costs +5 us at QKV T = 256).
- P5a builds the engine and the bf16 B path for QKV. P5b does O, GU and D.

## Scope
- Files: `regimebench/kernels/mk/mk_step.cu`, `mk_common.cuh`, `build.py`, `NOTES.md`, `regimebench/arms/mk.py`, `tests/test_mk.py`.
- Knob `MK_TS` (default 0). `constexpr bool TS = MK_TS && NPAD >= 128 && MK_KU == 128;`.
- With MK_TS = 0: 0-line SASS diff vs HEAD at all 5 buckets.
- With MK_TS = 1 at NPAD <= 64: SASS identical to MK_TS = 0.
- Port the engine from `probe/mk_core/xover3/xover3.cu` `k_ts` (SB = 0 path, verified on GPU in jobs 9592 and 9595: 260 checks PASS).
  Use mk's helpers (`mk_common.cuh`), and the `lm_phase` structure in `mk_step.cu`.

## Data
- New global buffer `xb`: bf16 [MAXTOK][HID]. Allocate it in `arms/mk.py` with torch.zeros. It is a new kernel argument after `gu`.
  - It holds bf16(x[t][k] * ln1[l][k]): the same value the v0 QKV `get` stages, before rr1.
  - QKV still red.adds unscaled sums into qkv, and the RoPE phase still applies rr1.
- New tensor map 8 = `xb`: dims {HID, MAXTOK}, box {64, TT}, SW128. NMAP 9 in the kernel.
  - `build.py`: `WEIGHT_MAPS` stays for the weights. Add an encoder for the workspace map, and append its 128 B to the maps blob in the arm.
- New global `tk`: u32 [8] tickets, zero at allocation. It is a kernel argument. The last user resets it.

## Who writes xb
1. **Prologue:** after x = embed[token], each thread writes xb = bf16(x * ln1[0]) for its own share of x.
   This runs before the first grid barrier, so QKV of layer 0 reads it after that barrier.
2. **D phase of layer l** (current engine, `gemm_phase<PhD>`; D unit u = rb * 24 + kc, rb = 128 x columns):
   - After a CTA's flush of row block rb (its red.adds of the kc units it owns), add a hook `fin(rb, nk)`.
     nk = number of kc units of rb flushed by this CTA. Implement it with `if constexpr`, so it is a no-op unless TS.
   - The hook runs: `__syncthreads`, then tid 0 does `__threadfence()` and `old = atomicAdd(&tk[rb], nk)`. last = (old + nk == 24).
     Broadcast last via smem.
   - If last (and l + 1 < NLAY): `__threadfence()`, then all 512 threads read x[t][rb*128 .. +127] for t < ntok
     with `__ldcg` and write xb[t][...] = bf16(x * ln1[l+1][...]). Then tid 0 sets tk[rb] = 0.
   - Then every writer thread runs `fence.proxy.async.global` (generic writes read later by TMA in other CTAs).
     The grid barrier arrive follows (release).
   - Explain the memory-model argument in NOTES.md: red.adds of other CTAs, then their threadfence and ticket, then this CTA's ticket,
     threadfence and ld.cg.
3. **Rows t in [ntok, NPAD):** xb keeps its zeros from allocation. Nothing writes them. The QKV epilogue ignores columns >= ntok anyway.

## QKV phase with TS: `ts_phase<PhQkv, TT, KS, NUL>(g, l, epi)`
- **Configs:** NPAD 256: TT 128, KS 2. NPAD 128: TT 128, KS 4 (job 9592 best). Write them as a constexpr table.
- **Tiles:** t = (rb * NTT + tt) * KS + ks, NRB = 32, contiguous ranges per CTA (`k_ts`).
  - A = weight rows from `PhQkv::src(l, rb, m, row)`, k-chunk at kc = ks*K/KS + j*128.
  - B = map 8 boxes (k, tt*TT).
- **Ring:** slot = A 32 KB + B TT*256 B. Ring depth = as many slots as fit in DYN_SMEM, overlaying the GEMM ring, B tiles and attention regions.
  - DYN_SMEM = max over the layouts. Mirror it in `build.py:dyn_smem`, and keep `MK_EXPECT_DYN` passing.
  - Add a static_assert that dyn + static <= 227 KB.
- **Barriers:** a new TS barrier block in the bar region: full[R], empty[R], accfull[2], accempty[2]. Grow BARS_BYTES.
  - full count 2: the A expect_tx (window) and the B expect_tx (after the grid barrier), as `k_ts`.
  - Re-init all TS barriers at the start of every `ts_first` (see below). Their parities then count from 0 per phase.
  - Explain in NOTES.md why re-init is legal: no thread waits on them and no TMA or commit is pending then.
- **ts_first<PhQkv>(g, l)** replaces `prefetch_next<PhQkv>(g, l)` in the barrier window, in both MK_PF_ARRIVE uses: the prologue and D of layer l-1.
  - tid 0 re-inits the TS barriers, runs `fence.mbarrier_init`, then issues the A boxes of the first min(R, steps) steps.
  - The first fill of the CTA's ring slots overlays smem that the previous phase used. Argue as in "LM head, pipelined" that no TMA
    into it is still in flight, and that all reads of it are done. Note: the D-phase finalize uses no dynamic smem.
- **Roles:** after the grid barrier, the roles are as in `k_ts`.
  - The producer issues the B boxes of the prefetched steps first (after `fence.proxy.async.global`), then the rest.
  - MMA issuer tid 32. Epilogue warps 4..15: TMEM to `epi(row, c0, v)`, the same v0 QKV red.add math into qkv, for columns < ntok only.
  - 2 TMEM accumulators x TT columns. Check that the allocation is >= 2*TT, i.e. 256 columns.
- **After the roles join:** `__syncthreads`, `tc_fence_after`. Then the QKV `extra()` (rr_of_x(x, rr1, ntok); zero_f32(gu, ...))
  runs as today, by all threads.
- **MK_NULL bit 0 with TS:** producer plus full waits only (A and B loads), no MMA and no epilogue.

## Checks (CPU only; never run on a GPU)
- All 5 buckets compile with MK_TS 0 and 1, with 0 spill. Report regs, dyn smem and ring R per bucket.
- 0-line SASS diff: MK_TS=0 vs HEAD (all buckets), and MK_TS=1 vs MK_TS=0 at NPAD 16, 32, 64.
- SASS counts at NPAD 256 with MK_TS=1: UTCHMMA, UTMALDG, SYNCS, REDG and the proxy fences.
- `tests/test_mk.py`: compile tests for TS. The 0-diff tests. A host test that the xb map encodes with the right dims
  (CPU: only `map_specs`-style dims; encoding needs a GPU, so skip that part on CPU).
- Run the CPU suite: `CUDA_CACHE_DISABLE=1 PYTHONPATH=/raid/garv901/regime-mk-core /raid/garv901/vibe-megakernel/.venv/bin/python -m pytest -q tests`.
  Check its exit code; do not pipe it through tail.
- Do not commit.
- Report: the files, the build table, the SASS counts, all the arguments (parity, re-init, overlap, memory model / proxy) and every deviation.
