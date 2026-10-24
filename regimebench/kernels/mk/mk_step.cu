// mk v0: Qwen3-0.6B decode step as ONE persistent cooperative kernel (148 CTAs x 512 threads) for 1..NPAD tokens, tcgen05 GEMM phases.
// Skeleton and glue: hv2g MTN (6 grid barriers per layer). GEMM phase: probe/xover ENG=1 (swap-AB: weights = A on M = 128, staged token
// tile = B on N = NPAD, K-major SW128, TMEM accumulator, red.add of the split-K partial sums). Attention: placeholder behind the A1
// interface (attn_placeholder.cuh). See NOTES.md for the design as built and the deviations from SPEC-v0.
//
// Phase map per layer (grid barrier after each):
//   1 QKV GEMM (B = bf16(x * ln1), red.add -> qkv unscaled; rr1) | 2 q/k-norm + RoPE + KV write (qb bf16; v * rr1) | 3 attention -> ao bf16
//   4 O GEMM (B = ao, red.add -> x; zero qkv) | 5 GU GEMM (B = bf16(x * ln2), red.add -> gu unscaled; rr2) | 6 D GEMM (B = bf16(silu * up))
// then the LM head (B = bf16(x[lm_tok] * fnorm), full row blocks, logits + per-slot argmax).

// ---------------------------------------------------------------- phase descriptors
// src(layer, row block, map index, first TMA row): maps 0..7 = wq, wk, wv, wo, wg, wu, wd, embed.
struct PhQkv {
    static constexpr int ID = 0, NRB = 32, K = HID;
    static constexpr bool WHOLE = false;
    __device__ static void src(int l, int rb, int& m, int& row) {
        if (rb < 16) { m = 0; row = l * 2048 + rb * 128; }
        else if (rb < 24) { m = 1; row = l * 1024 + (rb - 16) * 128; }
        else { m = 2; row = l * 1024 + (rb - 24) * 128; }
    }
};
struct PhO {
    static constexpr int ID = 3, NRB = 8, K = QDIM;
    static constexpr bool WHOLE = false;
    __device__ static void src(int l, int rb, int& m, int& row) { m = 3; row = l * HID + rb * 128; }
};
struct PhGu {
    static constexpr int ID = 4, NRB = 48, K = HID;
    static constexpr bool WHOLE = false;
    __device__ static void src(int l, int rb, int& m, int& row) {
        if (TSB) { m = 10; row = l * 6144 + rb * 128; }   // P5b: wgu, row block rb = 64 gate rows + the 64 matching up rows (rb = 0..47)
        else if (rb < 24) { m = 4; row = l * INTER + rb * 128; }
        else { m = 5; row = l * INTER + (rb - 24) * 128; }
    }
};
struct PhD {
    static constexpr int ID = 5, NRB = 8, K = INTER;
    static constexpr bool WHOLE = false;
    __device__ static void src(int l, int rb, int& m, int& row) { m = 6; row = l * HID + rb * 128; }
};
struct PhLm {
    static constexpr int ID = 6, NRB = VOCAB / 128, K = HID;
    static constexpr bool WHOLE = true;   // units are full row blocks (no split-K)
    __device__ static void src(int l, int rb, int& m, int& row) { (void)l; m = 7; row = rb * 128; }
};
static_assert(VOCAB % 128 == 0, "LM row blocks");
#ifdef MK_SKV
static_assert(NW == skv::kBlock / 32, "attention header assumes 16 warps");
static_assert(!TS || TSB, "MK_ATTN=splitkv with MK_TS on needs MK_TSB 1: the P5a-only layout (TS, TSB 0) hangs at NPAD 256 in mk barriers (job 9662, A1 debug waits silent; open, see LEDGER)");
static_assert(MK_KU == 128, "MK_ATTN=splitkv: the O phase stages one head per 128-k unit (combine_head); MK_KU 64 stays on the placeholder");
static_assert(SKV_TAIL == 64 && (OFF_BARS % 1024) == 512 && (OFF_BARS - (int)skv::kBarOff) >= 0 && ((OFF_BARS - (int)skv::kBarOff) % 1024) == 0,
              "splitkv scratch: mbarrier tail at [OFF_BARS, OFF_BARS + 64), scratch base 1024 B aligned");
static_assert(OFF_X + XREG <= OFF_BARS && LM_BYTES <= OFF_BARS && TS_BYTES <= OFF_BARS, "all overlays (GEMM ring, B tiles, LM ring, TS ring + exchange) stay below OFF_BARS");
static_assert(OFF_BARS - (int)skv::kBarOff + ATTN_BYTES == OFF_MB, "scratch end == start of mk's barrier block");
static_assert(QDIM / KU == NQH && PhO::NRB * (PhO::K / KU) == 128, "O units: u = rb * 16 + kc, kc = q-head");
#else
static_assert(NW == SKV_NW, "attention header assumes 16 warps");
static_assert(XREG >= ATTN_BYTES, "attention smem overlay");
#endif
static_assert(DYN_SMEM + 11280 <= 227 * 1024, "dynamic + static smem (11280 B: LM tables) exceed 227 KB (MK_LMR too large for this NPAD)");
static_assert(!TS || (2 * TS_TT <= 256 && NPAD % TS_TT == 0 && TS_R >= 2), "TS: two accumulators of TS_TT columns in the 256-column TMEM allocation; ring >= 2");
static_assert(!TS || (PhQkv::K % (TS_KS * 128) == 0), "TS: split-K factor");
static_assert(!TSB || (2 * GU_TT <= 256 && 2 * D_TT <= 256 && GU_TT % 16 == 0 && NPAD % D_TT == 0 && ts_ringn(GU_TT) >= 2 && ts_ringn(D_TT) >= 2), "TSB: TMEM columns / ring depth per phase");
static_assert(!TSB || (PhGu::K % (GU_KS * 128) == 0 && PhD::K % (D_KS * 128) == 0 && PhGu::NRB == 48 && PhD::NRB == 8), "TSB: split-K factors");
static_assert(!TSB || ts_ringn(D_TT) == ts_ringn(TS_TT), "TSB: D shares the QKV ring depth (TS_RING_A)");
static_assert(!TSB || NB > PhO::NRB * (PhO::K / KU), "TSB: the idle-CTA finalizers need CTAs without an O unit (grid > 128); no last-arriver fallback is built");
static_assert(!TS || DYN_SMEM + 11280 <= 227 * 1024, "TS: dynamic + static smem exceed 227 KB");
#if defined(MK_EXPECT_DYN) && MK_EXPECT_DYN
static_assert(MK_EXPECT_DYN == DYN_SMEM, "host dyn_smem (build.py) != kernel DYN_SMEM");
#endif

// next-phase weight prefetch vs grid-barrier arrive: MK_PFA 0 = prefetch, then arrive (v0); 1 = arrive, then prefetch (the TMA issue
// overlaps the barrier latency and cannot delay the arrive's release fence). The prefetch writes only ring slots whose MMAs completed.
#if MK_PFA
#define MK_PF_ARRIVE(...) do { bar_arrive(bar_cnt); __VA_ARGS__; } while (0)
#else
#define MK_PF_ARRIVE(...) do { __VA_ARGS__; bar_arrive(bar_cnt); } while (0)
#endif

// Unit numbering and row-block ownership live ONLY here (geom) and in PH::src: u = rb * KC + kc, CTA b owns units [u0, u0 + n).
template <class PH> struct Geom { int u0, n; };
template <class PH>
__device__ __forceinline__ Geom<PH> geom() {
    constexpr int KC = PH::K / KU, U = PH::NRB * KC;
    Geom<PH> g;
    if (PH::WHOLE) {
        const int r0 = (int)(((long long)blockIdx.x * PH::NRB) / gridDim.x);
        const int r1 = (int)(((long long)(blockIdx.x + 1) * PH::NRB) / gridDim.x);
        g.u0 = r0 * KC; g.n = (r1 - r0) * KC;
    } else {
        g.u0 = (int)(((long long)blockIdx.x * U) / gridDim.x);
        g.n = (int)(((long long)(blockIdx.x + 1) * U) / gridDim.x) - g.u0;
    }
    return g;
}

// per-CTA GEMM state
struct GS {
    u32 base;                     // shared address of the 1024-aligned smem base
    unsigned char* sm;            // generic pointer to it
    u32 bars;                     // tma bars [0, RING) at +8 s, mma bars at +64 + 8 s
    u32 tb;                       // TMEM base (256 columns)
    const unsigned char* maps;    // 8 tensor maps x 128 B in global memory
    u32 F;                        // global unit counter of this CTA (ring slot f % RING, mbarrier parity (f / RING) & 1)
};

// tid 0: TMA fill number f (ring slot f % RING) of unit u of phase PH, layer l
template <class PH>
__device__ __forceinline__ void fill_unit(const GS& g, int l, int u, u32 f) {
    constexpr int KC = PH::K / KU;
    const int kc = u % KC, rb = u / KC;
    int m, row;
    PH::src(l, rb, m, row);
    const u32 s = f % RING, bar = g.bars + 8 * s;
    mbar_expect_tx(bar, UB);
    mk_tma_unit(g.base + s * UB, g.maps + m * 128, kc * KU, row, bar);
}
// barrier-window prefetch: the first min(RING, n) units of the NEXT GEMM phase PH (g.F already counts the finished phase)
template <class PH>
__device__ __forceinline__ void prefetch_next(const GS& g, int l) {
    if (threadIdx.x == 0) {
        const Geom<PH> ge = geom<PH>();
        for (int i = 0; i < RING && i < ge.n; ++i) fill_unit<PH>(g, l, ge.u0 + i, g.F + i);
    }
}

// all threads: B tile of one unit (token rows 0..NPAD-1 x KU k starting at kcol) into the SW128 K-major slot xs (k-atom a at a * NPAD * 128 B,
// row r at r * 128 B, 16-byte chunk ch ^ (r & 7)). get(t, k, j) returns the 8 bf16 of token row t, k .. k + 7 (zero for t >= rows).
template <class GetF>
__device__ __forceinline__ void stage_tile(unsigned char* xs, int kcol, int j, GetF& get) {
    constexpr int CPR = KU / 8, CH = NPAD * CPR, ITER = MK_CDIV(CH, NT);
    const int tid = threadIdx.x;
#pragma unroll
    for (int g0 = 0; g0 < ITER; g0 += MK_STG) {
        uint4 v[MK_STG];
#pragma unroll
        for (int jj = 0; jj < MK_STG; ++jj) {
            const int i = g0 + jj, c = tid + NT * i;
            if (i < ITER && c < CH) v[jj] = get(c / CPR, kcol + (c % CPR) * 8, j);
        }
#pragma unroll
        for (int jj = 0; jj < MK_STG; ++jj) {
            const int i = g0 + jj, c = tid + NT * i;
            if (i < ITER && c < CH) {
                const int r = c / CPR, ch = c % CPR;
                *reinterpret_cast<uint4*>(xs + (ch >> 3) * (NPAD * 128) + r * 128 + (((ch & 7) ^ (r & 7)) << 4)) = v[jj];
            }
        }
    }
    fence_proxy_async();
}

// B-tile staging of a unit: a staging functor (stage_tile) or, under MK_SKV, nothing (SkvB: the A1 combine of the unit's head runs in the phase body before gemm_phase:
// inlined into gemm_phase its 20-float4 merge state spills; the unit is the CTA's only one, see the static_assert on NB)
template <class GetF>
__device__ __forceinline__ void stage_any(unsigned char* xs, int kcol, int j, GetF& get) { stage_tile(xs, kcol, j, get); }
#ifdef MK_SKV
struct SkvB { int comb; const bf16_t* ao; int ntok; };   // comb == 0 (plan.s_max == 1): attend wrote every ao row, stage_any stages the B tile from ao like the placeholder build
// Opaque pointer: the A1 calls' smem addresses are not loop-invariant for the compiler, so the invariants derived from them are not hoisted out of the layer loop (kept live across every phase)
__device__ __forceinline__ unsigned char* mk_opaque(unsigned char* p) { asm volatile("" : "+l"(p)); return p; }
__device__ __forceinline__ void stage_any(unsigned char* xs, int kcol, int j, SkvB& b) {
    if (b.comb) return;   // block-uniform kernel argument: the combine_head of the unit's head staged the slot
    auto get = [&](int t, int k, int) -> uint4 {   // = the placeholder O-phase get (ld.global.cg of ao after the grid barrier)
        if (t >= b.ntok) return make_uint4(0u, 0u, 0u, 0u);
        return __ldcg(reinterpret_cast<const uint4*>(b.ao + (size_t)t * QDIM + k));
    };
    stage_tile(xs, kcol, j, get);
}
static_assert(NB >= PhO::NRB * (PhO::K / KU), "MK_ATTN=splitkv: at most one O unit per CTA (grid >= 128): the combine is done once, before gemm_phase");
#endif

// One GEMM phase. Precondition: the first min(RING, n) units of this CTA were already issued (prologue / previous window prefetch).
// get: staging functor (see stage_tile); epi(row, c0, v, rb): output of 16 token columns c0.. of accumulator row `row` (v = raw TMEM words);
// extra(): called once by all threads while unit 0's MMA runs (small per-phase side work). nrows = token rows of the B tile.
// NUL: null-work build of the phase (TMA + ring waits only).
struct NoFin { __device__ __forceinline__ void operator()(int, int) const {} };
// fin(rb, nk): all threads, after the epilogue of a row-block segment (nk = units of row block rb this CTA flushed); block-uniform (MK_TS: D phase xb hook).
template <class PH, bool NUL, class GetF, class EpiF, class ExtraF, class FinF = NoFin>
__device__ __forceinline__ void gemm_phase(GS& g, int l, int nrows, GetF get, EpiF epi, ExtraF extra, FinF fin = FinF()) {
    constexpr int KC = PH::K / KU;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, q = warp & 3;
    const Geom<PH> ge = geom<PH>();
    const int n = ge.n, u0 = ge.u0;
    const u32 F0 = g.F;
    g.F = F0 + n;
    if (NUL) {
        for (int j = 0; j < n; ++j) {
            if (tid == 0) {
                const u32 f = F0 + j;
                mbar_wait(g.bars + 8 * (f % RING), (f / RING) & 1);
                if (j + RING < n) fill_unit<PH>(g, l, u0 + j + RING, f + RING);   // nothing reads the slot: refill at once
            }
            __syncthreads();
        }
        return;
    }
    if (n == 0) { extra(); return; }
    const u32 idesc = tc_idesc(128, NPAD);
    unsigned char* const xbase = g.sm + OFF_X;
    stage_any(xbase, (u0 % KC) * KU, 0, get);
    __syncthreads();
    int jseg = 0;   // first unit (j) of the current row-block segment
    for (int j = 0; j < n; ++j) {
        const u32 f = F0 + j, s = f % RING, par = (f / RING) & 1;
        const int u = u0 + j;
        if (tid == 0) {
            mbar_wait(g.bars + 8 * s, par);
            tc_fence_after();
            const bool first = (j == 0) || (u % KC == 0);
            const u32 sr = g.base + s * UB, xb = g.base + OFF_X + (j & 1) * XS;
            mk_mma_unit(g.tb, sr, xb, idesc, first);
            mk_commit(g.bars + 64 + 8 * s);
        }
        if (j == 0) extra();
        if (j + 1 < n) {
            if (j >= 1) {   // MMA(j-1) done reading ring slot (f-1) % RING and B slot (j+1) & 1
                mbar_wait(g.bars + 64 + 8 * ((f - 1) % RING), ((f - 1) / RING) & 1);
                if (tid == 0 && j + RING - 1 < n) fill_unit<PH>(g, l, u0 + j + RING - 1, f + RING - 1);
            }
            stage_any(xbase + ((j + 1) & 1) * XS, ((u0 + j + 1) % KC) * KU, j + 1, get);
        }
        if (j == n - 1 || (u + 1) % KC == 0) {   // row-block segment ends: TMEM -> output
            mbar_wait(g.bars + 64 + 8 * s, par);
            tc_fence_after();
            const int rb = u / KC;
            const int nch = (nrows + 15) / 16;
            for (int c = warp >> 2; c < nch; c += (MK_EPI2 ? 8 : 4)) {
                u32 v0[16], v1[16];
                const bool two = MK_EPI2 && (c + 4 < nch);
                mk_tmem_read16(v0, g.tb, q, c * 16);
                if (two) mk_tmem_read16(v1, g.tb, q, (c + 4) * 16);
                tc_wait_ld();
                const int row = rb * 128 + mk_row_of(q, lane);
                epi(row, c * 16, v0, rb);
                if (two) epi(row, (c + 4) * 16, v1, rb);
            }
            tc_fence_before();
            fin(rb, j - jseg + 1);
            jseg = j + 1;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------- pipelined LM head (MK_LMP, NPAD <= 64; see NOTES.md "LM head, pipelined")
// LM unit j of this CTA (rb = r0 + j / 8, kc = j % 8) lives in LM ring slot j % LMR, full barrier full[j % LMR].
__device__ __forceinline__ void lm_fill(const GS& g, int r0, int j) {   // one thread
    constexpr int KCL = HID / KU;
    const int s = j % LMR;
    int m, row;
    PhLm::src(0, r0 + j / KCL, m, row);
    const u32 bar = g.bars + LMB_FULL + 8 * s;
    mbar_expect_tx(bar, UB);
    mk_tma_unit(g.base + LM_RING_OFF + s * UB, g.maps + m * 128, (j % KCL) * KU, row, bar);
}
// barrier window after the last D phase: first LM units (LMP) or the v0 ring prefetch
__device__ __forceinline__ void lm_first(const GS& g) {
    if constexpr (LMP) {
        if (threadIdx.x == 0) {
            const Geom<PhLm> ge = geom<PhLm>();
            for (int i = 0; i < LMR && i < ge.n; ++i) lm_fill(g, ge.u0 / (HID / KU), i);
        }
    } else {
        prefetch_next<PhLm>(g, 0);
    }
}
// All threads. Precondition: lm_first() ran in the barrier window and the grid barrier has been passed. Returns with the roles still
// divergent (the caller's __syncthreads joins them).
template <bool NUL, class GetF, class EpiF>
__device__ __forceinline__ void lm_phase(GS& g, GetF get, EpiF epi) {
    constexpr int KCL = HID / KU, NCH = NPAD / 16;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, q = warp & 3;
    const Geom<PhLm> ge = geom<PhLm>();
    const int n = ge.n, r0 = ge.u0 / KCL, nrb = n / KCL;
    const u32 fullb = g.bars + LMB_FULL, emptyb = g.bars + LMB_EMPTY, accf = g.bars + LMB_ACCF, acce = g.bars + LMB_ACCE;
    if (NUL) {   // producer + full waits only (nothing reads the slot: refill at once)
        if (tid == 0)
            for (int j = 0; j < n; ++j) {
                mbar_wait(fullb + 8 * (j % LMR), (j / LMR) & 1);
                if (j + LMR < n) lm_fill(g, r0, j + LMR);
            }
        return;
    }
    // X (all 8 k-chunks of B) staged once: chunk c -> (kc, row r, 16-byte chunk ch), same SW128 K-major layout as stage_tile
    {
        constexpr int PER = NPAD * 16, ITER = 8 * PER / NT;
#pragma unroll
        for (int g0 = 0; g0 < ITER; g0 += MK_STG) {
            uint4 v[MK_STG];
#pragma unroll
            for (int jj = 0; jj < MK_STG; ++jj) {
                const int i = g0 + jj, c = tid + NT * i;
                if (i < ITER) v[jj] = get((c % PER) >> 4, (c / PER) * KU + (c & 15) * 8, 0);
            }
#pragma unroll
            for (int jj = 0; jj < MK_STG; ++jj) {
                const int i = g0 + jj, c = tid + NT * i;
                if (i < ITER) {
                    const int kc = c / PER, r = (c % PER) >> 4, ch = c & 15;
                    *reinterpret_cast<uint4*>(g.sm + kc * XS + (ch >> 3) * (NPAD * 128) + r * 128 + (((ch & 7) ^ (r & 7)) << 4)) = v[jj];
                }
            }
        }
        fence_proxy_async();
    }
    __syncthreads();
    if (tid == 0) {   // producer: units LMR.. (the first min(LMR, n) were issued in the barrier window)
        for (int j = LMR; j < n; ++j) {
            const int s = j % LMR;
            mbar_wait(emptyb + 8 * s, ((j / LMR) - 1) & 1);   // previous use of the slot (use number j / LMR - 1) finished
            lm_fill(g, r0, j);
        }
    } else if (tid == 32) {   // MMA issuer
        const u32 idesc = tc_idesc(128, NPAD);
        for (int j = 0; j < n; ++j) {
            const int i = j / KCL, kc = j % KCL, b = i & 1, s = j % LMR;
            if (kc == 0 && i >= 2) mbar_wait(acce + 8 * b, ((i / 2) - 1) & 1);   // epilogue done with buffer b (use i / 2 - 1)
            mbar_wait(fullb + 8 * s, (j / LMR) & 1);
            tc_fence_after();
            mk_mma_unit(g.tb + b * NPAD, g.base + LM_RING_OFF + s * UB, g.base + kc * XS, idesc, kc == 0);
            mk_commit(emptyb + 8 * s);
            if (kc == KCL - 1) mk_commit(accf + 8 * b);
        }
    } else if (warp >= 4 && (warp >> 2) - 1 < NCH) {   // epilogue: warp w = 4 .. 4 + 4 min(NCH, 3) - 1, quadrant q = w & 3, column chunks (w >> 2) - 1 + 3 k
        for (int i = 0; i < nrb; ++i) {
            const int b = i & 1;
            mbar_wait(accf + 8 * b, (i / 2) & 1);
            tc_fence_after();
            for (int c = (warp >> 2) - 1; c < NCH; c += 3) {   // chunks are disjoint slots: still one writer per (quadrant, slot)
                u32 v[16];
                mk_tmem_read16(v, g.tb + b * NPAD, q, c * 16);
                tc_wait_ld();
                epi((r0 + i) * 128 + mk_row_of(q, lane), c * 16, v, r0 + i);
            }
            tc_fence_before();
            __syncwarp();
            if (lane == 0) mbar_arrive(acce + 8 * b);
        }
    }
}

// ---------------------------------------------------------------- tile-stream engine (MK_TS, NPAD >= 128; ported from probe/mk_core/xover3 k_ts, SB = 0; NOTES.md "Tile-stream QKV")
// Tile t = (rb * NTT + tt) * KS + ks: W rows 128 rb .., tokens TT tt .., k range ks * K / KS ..; CTA b owns [b NTILE / grid, (b + 1) NTILE / grid). A step = 128 k = one ring
// slot [A: 128 rows x 128 k | B: TT tokens x 128 k]; the B boxes come from the bf16 xb map (map 8). Barrier indices are LOCAL to the phase (re-initialised in ts_first).
template <class PH, int TT, int KS> struct TsC {
    static constexpr int NTT = NPAD >= TT ? (NPAD + TT - 1) / TT : 1, NTILE = PH::NRB * NTT * KS, SPT = PH::K / (KS * 128), KSL = PH::K / KS, NCH = TT / 16;
    static constexpr int R = ts_ringn(TT), SLOT = ts_slot(TT);   // ring depth and slot bytes of this phase (QKV: TS_R, TS_SLOT)
};
struct TsGeom { int t0, nt, nst; };
template <class PH, int TT, int KS>
__device__ __forceinline__ TsGeom ts_geom() {
    constexpr int NTILE = TsC<PH, TT, KS>::NTILE;
    TsGeom ge;
    ge.t0 = (int)(((long long)blockIdx.x * NTILE) / gridDim.x);
    ge.nt = (int)(((long long)(blockIdx.x + 1) * NTILE) / gridDim.x) - ge.t0;
    ge.nst = ge.nt * TsC<PH, TT, KS>::SPT;
    return ge;
}
// tid 0: A boxes of step j of the phase into ring slot s (full[s] arrival 1: expect_tx of the A bytes)
template <class PH, int TT, int KS>
__device__ __forceinline__ void ts_fill_a(const GS& g, int l, int t0, int j, int s) {
    using C = TsC<PH, TT, KS>;
    const int t = t0 + j / C::SPT, kc = (t % KS) * C::KSL + (j % C::SPT) * 128, rb = (t / KS) / C::NTT;
    int m, row;
    PH::src(l, rb, m, row);
    const u32 bar = g.bars + LMB_FULL + 8 * s;
    mbar_expect_tx(bar, UB);
    mk_tma_unit(g.base + s * C::SLOT, g.maps + m * 128, kc, row, bar);
}
// tid 0: B boxes of step j (map BM = 8 xb | 9 xb2 | 11 act, k kc.., tokens TT tt..) into ring slot s (full[s] arrival 2); only after the grid barrier + fence.proxy.async.global
template <class PH, int TT, int KS, int BM>
__device__ __forceinline__ void ts_fill_b(const GS& g, int t0, int j, int s) {
    using C = TsC<PH, TT, KS>;
    const int t = t0 + j / C::SPT, kc = (t % KS) * C::KSL + (j % C::SPT) * 128, tt = (t / KS) % C::NTT;
    const u32 bar = g.bars + LMB_FULL + 8 * s;
    mbar_expect_tx(bar, TT * 256);
#pragma unroll
    for (int a = 0; a < 2; ++a) tma2(g.base + s * C::SLOT + 32768 + a * (TT * 128), g.maps + BM * 128, kc + a * 64, tt * TT, bar);
}
// barrier window before the tile-stream phase (replaces prefetch_next<PhQkv>): tid 0 re-initialises the TS mbarriers and issues the A boxes of the first min(R, nst) steps
template <class PH, int TT, int KS>
__device__ __forceinline__ void ts_first(const GS& g, int l) {
    if (threadIdx.x == 0) {
        const u32 fullb = g.bars + LMB_FULL, emptyb = g.bars + LMB_EMPTY, accf = g.bars + LMB_ACCF, acce = g.bars + LMB_ACCE;
        for (int i = 0; i < TsC<PH, TT, KS>::R; ++i) { mbar_init(fullb + 8 * i, 2); mbar_init(emptyb + 8 * i, 1); }
        for (int b = 0; b < 2; ++b) { mbar_init(accf + 8 * b, 1); mbar_init(acce + 8 * b, 12); }   // accempty: the 12 epilogue warps (4 quadrants x 3 chunk groups)
        mbar_fence_init();
        const TsGeom ge = ts_geom<PH, TT, KS>();
        for (int i = 0; i < TsC<PH, TT, KS>::R && i < ge.nst; ++i) ts_fill_a<PH, TT, KS>(g, l, ge.t0, i, i);
    }
}
// All threads. Precondition: ts_first ran in the barrier window and the grid barrier has been passed. epi(row, c0, v, rb): as gemm_phase (c0 = first token column).
// Returns after a __syncthreads that joins the roles. NUL: producer + full waits only.
template <class PH, int TT, int KS, int BM, bool NUL, class EpiF>
__device__ __forceinline__ void ts_phase(GS& g, int l, int ntok, EpiF epi) {
    using C = TsC<PH, TT, KS>;
    constexpr int SPT = C::SPT, NCH = C::NCH, R = C::R;
    constexpr bool PAIR = TSB && PH::ID == PhGu::ID;   // P5b GU: g / u quadrant pairs, epi(q, lane, group, rb, cb, v) exchanges through smem
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, q = warp & 3;
    const TsGeom ge = ts_geom<PH, TT, KS>();
    const int t0 = ge.t0, nt = ge.nt, nst = ge.nst;
    const u32 fullb = g.bars + LMB_FULL, emptyb = g.bars + LMB_EMPTY, accf = g.bars + LMB_ACCF, acce = g.bars + LMB_ACCE;
    if (tid == 0) {   // ---- producer
        fence_proxy_async_global();   // xb was written with generic stores (D phase / prologue of other CTAs, released by the grid barrier)
        for (int i = 0; i < R && i < nst; ++i) ts_fill_b<PH, TT, KS, BM>(g, t0, i, i);   // B of the window-prefetched steps
        if (NUL) {
            for (int j = 0; j < nst; ++j) {
                mbar_wait(fullb + 8 * (j % R), (j / R) & 1);
                if (j + R < nst) { ts_fill_a<PH, TT, KS>(g, l, t0, j + R, j % R); ts_fill_b<PH, TT, KS, BM>(g, t0, j + R, j % R); }
            }
        } else {
            for (int j = R; j < nst; ++j) {
                const int s = j % R;
                mbar_wait(emptyb + 8 * s, ((j / R) - 1) & 1);   // step j - R (use j / R - 1 of the slot) has been read by its MMAs
                ts_fill_a<PH, TT, KS>(g, l, t0, j, s);
                ts_fill_b<PH, TT, KS, BM>(g, t0, j, s);
            }
            for (int j = nst > R ? nst - R : 0; j < nst; ++j) mbar_wait(emptyb + 8 * (j % R), (j / R) & 1);   // drain: last use of every slot is complete
        }
    } else if (!NUL && tid == 32) {   // ---- MMA issuer
        const u32 idesc = tc_idesc(128, TT);
        for (int ti = 0; ti < nt; ++ti) {
            const int b = ti & 1;
            if (ti >= 2) mbar_wait(acce + 8 * b, ((ti >> 1) - 1) & 1);   // epilogue done with accumulator b (use ti / 2 - 1)
            for (int k = 0; k < SPT; ++k) {
                const int f = ti * SPT + k, s = f % R;
                mbar_wait(fullb + 8 * s, (f / R) & 1);
                tc_fence_after();
                const u32 sa = g.base + s * C::SLOT, sb = sa + 32768;
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) {
                    const int a = kk >> 2, off = (kk & 3) * 32;
                    tc_mma(g.tb + b * TT, sdesc(sa + a * 16384 + off), sdesc(sb + a * (TT * 128) + off), idesc, (k > 0 || kk > 0) ? 1u : 0u);
                }
                mk_commit(emptyb + 8 * s);
                if (k == SPT - 1) mk_commit(accf + 8 * b);
            }
        }
    } else if (!NUL && warp >= 4) {   // ---- epilogue: quadrant q = warp & 3, column chunks (warp >> 2) - 1 + 3 m (disjoint per (quadrant, chunk))
        if constexpr (PAIR) {   // GU: epi(-1, ..) = per-token setup by the 12 epilogue warps while the MMAs run (r2 into smem; ends with a 384-thread named barrier)
            u32 none[16];
            if (nt > 0) epi(-1, lane, 0, 0, 0, none);
        }
        for (int ti = 0; ti < nt; ++ti) {
            const int b = ti & 1, t = t0 + ti, rb = (t / KS) / C::NTT, tt = (t / KS) % C::NTT;
            mbar_wait(accf + 8 * b, (ti >> 1) & 1);
            tc_fence_after();
            const int row = rb * 128 + mk_row_of(q, lane);
            if constexpr (PAIR) {   // one chunk per iteration: warps w and w + 2 of a group (quadrants q, q + 2) hold g and u of the same channel and run the same chunks
                const int grp = (warp >> 2) - 1;
                for (int c = grp; c < NCH; c += 3) {
                    const int cb = tt * TT + c * 16;
                    if (cb >= ntok) break;   // warp-uniform and equal for both warps of the pair
                    u32 v0[16];
                    mk_tmem_read16(v0, g.tb + b * TT, q, c * 16);
                    tc_wait_ld();
                    epi(q, lane, grp, rb, cb, v0);
                }
            } else {
            for (int c = (warp >> 2) - 1; c < NCH; c += 6) {
                const int cb = tt * TT + c * 16;
                if (cb >= ntok) break;   // warp-uniform: nothing stored at columns >= ntok (chunks c + 3, c + 6.. are further right)
                const bool two = (c + 3 < NCH) && (cb + 48 < ntok);
                u32 v0[16], v1[16];
                mk_tmem_read16(v0, g.tb + b * TT, q, c * 16);
                if (two) mk_tmem_read16(v1, g.tb + b * TT, q, (c + 3) * 16);
                tc_wait_ld();
                epi(row, cb, v0, rb);
                if (two) epi(row, cb + 48, v1, rb);
            }
            }
            tc_fence_before();
            __syncwarp();
            if (lane == 0) mbar_arrive(acce + 8 * b);
        }
    }
    __syncthreads();
    tc_fence_after();
}

__device__ __forceinline__ void zero_f32(float* p, int total) {   // total floats (multiple of 4), whole grid
    for (int i = (blockIdx.x * NT + threadIdx.x) * 4; i < total; i += gridDim.x * NT * 4)
        __stcg(reinterpret_cast<float4*>(p + i), make_float4(0.f, 0.f, 0.f, 0.f));
}
// rr[t] = rsqrt(mean(x[t]^2) + eps) for t = blockIdx.x + gridDim.x * (warp - 8), warps 8..15 (one token per warp)
__device__ __forceinline__ void rr_of_x(const float* x, float* rr, int ntok) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp < 8) return;
    const int t = blockIdx.x + gridDim.x * (warp - 8);
    if (t >= ntok) return;
    float4 a[8];
#pragma unroll
    for (int m = 0; m < 8; ++m) a[m] = __ldcg(reinterpret_cast<const float4*>(x + (size_t)t * HID) + lane + 32 * m);
    float ss = 0.f;
#pragma unroll
    for (int m = 0; m < 8; ++m) ss += a[m].x * a[m].x + a[m].y * a[m].y + a[m].z * a[m].z + a[m].w * a[m].w;
    ss = warp_sum(ss);
    if (lane == 0) __stcg(rr + t, rsqrtf(ss * (1.f / HID) + EPS));
}
// pack 8 fp32 (x * w) into 8 bf16
__device__ __forceinline__ uint4 pack_xw(const float4 a, const float4 b, const uint4 w) {
    return make_uint4(pack_bf16(a.x * bflo(w.x), a.y * bfhi(w.x)), pack_bf16(a.z * bflo(w.y), a.w * bfhi(w.y)),
                      pack_bf16(b.x * bflo(w.z), b.y * bfhi(w.z)), pack_bf16(b.z * bflo(w.w), b.w * bfhi(w.w)));
}

// ---------------------------------------------------------------- P5b: idle-CTA finalizers (SPEC-P5b; NOTES.md "GU and D on the tile stream")
// Ticket layout in tk[64]: O tickets tk[0 .. 8 NTT) (index rb * NTT + tt, target 16 = the units of a row block), D tickets tk[TK_D ..] (same index, target D_KS).
constexpr int TK_D = 16;
// CTA b of an NTILE-unit phase owns [lo(b), lo(b + 1)) (the formula of geom / ts_geom); idle = owns none. rank = index of this CTA among the idle CTAs in blockIdx order, nidle = their count.
template <int NTILE>
__device__ __forceinline__ void idle_rank(int& rank, int& nidle) {
    // NTILE <= grid: every working CTA owns exactly one unit, so the working CTAs below b are lo(b) and the idle ones b - lo(b).
    // Closed form (a loop over the grid with 64-bit divisions cost ~20 us per phase, job 9613).
    static_assert(NTILE <= NB, "idle_rank: closed form needs NTILE <= grid");
    const int b = (int)blockIdx.x, G = (int)gridDim.x;
    const int lo = (b * NTILE) / G, hi = ((b + 1) * NTILE) / G;
    nidle = G - NTILE;
    rank = hi == lo ? b - lo : -1;
}
// All threads of an idle CTA (block-uniform; other CTAs return at once). Jobs = (rb, tt) with TT = 128 tokens: job j = rb * NTT + tt, idle CTA r takes j = r, r + nidle, ...
// IS_O: wait tk[j] == 16, xb2[t][rb 128 ..] = bf16(x * ln2[l]) and ss2[t] += sum of x^2 over the 128 columns; else: wait tk[TK_D + j] == D_KS, xb = bf16(x * ln1[l + 1]).
// Ends with fence.proxy.async.global in every writing thread (the caller then arrives at the grid barrier).
template <bool IS_O, int NTILE, bool SS = IS_O>
__device__ __forceinline__ void idle_finalize(const float* x, const bf16_t* lnw, bf16_t* out, float* ss2, unsigned* tk, int ntok) {
    constexpr int NTT = NPAD / 128, TARGET = IS_O ? PhO::K / KU : D_KS;
    int rank, nidle;
    idle_rank<NTILE>(rank, nidle);
    if (rank < 0) return;   // not an idle CTA (block-uniform)
    const int tid = threadIdx.x;
    if constexpr (MK_FIN == 0) {
    for (int job = rank; job < 8 * NTT; job += nidle) {
        const int rb = job / NTT, tt = job % NTT;
        unsigned* tick = tk + (IS_O ? 0 : TK_D) + job;
        if (tid == 0) {
            unsigned v;
            do {
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(tick) : "memory");
                if (v != TARGET) __nanosleep(64);
            } while (v != TARGET);
            atomicExch(tick, 0u);   // reset: every worker add of this layer has landed (the target is reached); the next use is a layer later, behind grid barriers
        }
        __syncthreads();   // broadcast: tid 0's acquire orders the other threads' reads of x below
        const int ntl = min(ntok - tt * 128, 128), items = ntl > 0 ? ntl * 16 : 0;
        const bf16_t* lw = lnw + rb * 128;
        for (int i0 = 0; i0 < items; i0 += NT) {   // block-uniform trip count (the shuffles below need all lanes)
            const int i = i0 + tid, t = tt * 128 + (i >> 4), cg = i & 15;
            const bool ok = i < items;
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = a;
            if (ok) {
                const float* xp = x + (size_t)t * HID + rb * 128 + cg * 8;
                a = __ldcg(reinterpret_cast<const float4*>(xp)); b = __ldcg(reinterpret_cast<const float4*>(xp + 4));
            }
            if (ok) *reinterpret_cast<uint4*>(out + (size_t)t * HID + rb * 128 + cg * 8) = pack_xw(a, b, __ldg(reinterpret_cast<const uint4*>(lw + cg * 8)));
            if constexpr (SS) {
                float sq = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
#pragma unroll
                for (int o = 8; o; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);   // the 16 lanes of one token (16 consecutive i)
                if (ok && cg == 0) atomicAdd(ss2 + t, sq);
            }
        }
    }
    } else {
    // MK_FIN 1: each job is split over nsub = nidle / NJOB idle CTAs by token rows (part p: tokens tt 128 + p pr ..); the last of the nsub CTAs that saw
    // the target resets the ticket and the seen counter tk[32 + (IS_O ? 0 : 16) + job]. (Issuing all loads before the stores deadlocked at b128, job 9766
    // and srun bisect 2026-10-23; not root-caused, not kept.)
    constexpr int NJOB = 8 * NTT;
    const int nsub = max(1, nidle / NJOB), pr = (128 + nsub - 1) / nsub;
    for (int w = rank; w < NJOB * nsub; w += nidle) {
        const int job = w % NJOB, part = w / NJOB, rb = job / NTT, tt = job % NTT;
        unsigned* tick = tk + (IS_O ? 0 : TK_D) + job;
        if (tid == 0) {
            unsigned v;
            do {
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(tick) : "memory");
                if (v != TARGET) __nanosleep(64);
            } while (v != TARGET);
            if (nsub == 1) atomicExch(tick, 0u);
            else {
                unsigned* seen = tk + 32 + (IS_O ? 0 : 16) + job;
                if (atomicAdd(seen, 1u) == (unsigned)nsub - 1) { atomicExch(seen, 0u); atomicExch(tick, 0u); }   // every sub CTA has seen the target
            }
        }
        __syncthreads();
        const int ntl = min(ntok - tt * 128, 128), p0 = part * pr, np = max(0, min(ntl - p0, pr)), items = np * 16;
        const bf16_t* lw = lnw + rb * 128;
        for (int i0 = 0; i0 < items; i0 += NT) {   // block-uniform trip count (the shuffles below need all lanes)
            const int i = i0 + tid, t = tt * 128 + p0 + (i >> 4), cg = i & 15;
            const bool ok = i < items;
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = a;
            if (ok) {
                const float* xp = x + (size_t)t * HID + rb * 128 + cg * 8;
                a = __ldcg(reinterpret_cast<const float4*>(xp)); b = __ldcg(reinterpret_cast<const float4*>(xp + 4));
            }
            if (ok) *reinterpret_cast<uint4*>(out + (size_t)t * HID + rb * 128 + cg * 8) = pack_xw(a, b, __ldg(reinterpret_cast<const uint4*>(lw + cg * 8)));
            if constexpr (SS) {
                float sq = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
#pragma unroll
                for (int o = 8; o; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
                if (ok && cg == 0) atomicAdd(ss2 + t, sq);
            }
        }
    }
    }
    fence_proxy_async_global();   // generic writes of xb2 / xb -> TMA reads after the grid barrier
}

#ifdef MK_SKV
// A1 v1 parameters, rebuilt at each use from the kernel arguments (constant bank) so that no Params copy stays live across the phases
// (q = qb as the placeholder reads it; ao: S == 1 pairs are written by attend, combine_head reads it)
#define MK_SKV_PARAMS(SP) \
    skv::Params SP; \
    SP.q = qb; SP.kc = kcache; SP.vc = vcache; SP.cache_rows = cache_rows; SP.n_layers = NLAY; \
    SP.items = skv_items; SP.items_stride = skv_items_stride; SP.item_off = nullptr; SP.pair_info = skv_pair_info; SP.part = skv_part; \
    SP.ao = ao; SP.tmaps = skv_maps; SP.engine = skv_engine; SP.total_q = ntok
#endif
extern "C" __global__ void __launch_bounds__(NT, 1) mk_step(
    const long long* __restrict__ token,
    const bf16_t* __restrict__ ln1, const bf16_t* __restrict__ qn, const bf16_t* __restrict__ kn,
    const bf16_t* __restrict__ ln2, const bf16_t* __restrict__ fnorm, const bf16_t* __restrict__ embed,
    const unsigned char* __restrict__ maps,
    bf16_t* kcache, bf16_t* vcache,
    float* __restrict__ logits, long long* __restrict__ next_token,
    float* x, float* qkv, bf16_t* qb, bf16_t* ao, float* gu, float* rr,
    unsigned* sync, unsigned* pkey, int* pidx,
    const float* __restrict__ rope, const int* __restrict__ t_base, const int* __restrict__ t_vis,
    const int* __restrict__ lm_tok, const int* __restrict__ lm_seq,
    const int ntok, const int n_lm, const int cache_rows,
    bf16_t* xb, unsigned* tk,   // MK_TS: xb = bf16(x * ln1) [MAXTOK][HID] (TMA B of QKV), tk = tickets [64] (P5b; [8] in P5a); last, so that the other argument offsets do not change
    bf16_t* xb2, float* ss2, bf16_t* act   // MK_TSB: xb2 = bf16(x * ln2) [MAXTOK][HID] (B of GU), ss2 = sum x^2 per token [MAXTOK], act = bf16(silu * up) [MAXTOK][INTER] (B of D); tk is [64]
#ifdef MK_SKV
    , const int4* skv_items, int skv_items_stride, const int* skv_pair_info, float* skv_part, const void* skv_maps, int skv_engine, int skv_comb   // A1 (plan.py): item records, pair_info, partials, 2 * kNumMaps tensor maps (null: LDG engine), engine, skv_comb = plan.s_max > 1 (0: ao complete after attend, no combine)
#endif
    )
{
    extern __shared__ unsigned char smraw[];
    __shared__ float s_ss[MAXTOK];        // LM: sum of squares of the staged x rows (lm slot)
    __shared__ int s_lmtok[MAXTOK], s_lmseq[MAXTOK];
    __shared__ u32 s_bk[4][MAXTOK];       // LM: per quadrant best key / row per lm slot
    __shared__ int s_bi[4][MAXTOK];
    __shared__ int s_flag;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int gwarp = blockIdx.x * NW + warp, nwarps = gridDim.x * NW;
    const u32 raw = smem_u32(smraw), base = (raw + 1023) & ~1023u;
    unsigned char* sm = smraw + (base - raw);
    volatile u32* tslot = reinterpret_cast<volatile u32*>(sm + OFF_MB + 128);
    unsigned* bar_cnt = sync;
    unsigned* ticket = sync + 2;
    unsigned bar_target = 0;
    float* rr1 = rr;
    float* rr2 = rr + MAXTOK;

    // ---- prologue: TMEM, mbarriers, small smem state ----
    if (tid == 0) {   // the host's dyn smem must cover the layout (build.py dyn_smem mirrors DYN_SMEM)
        u32 dyn;
        asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn));
        if (dyn < (u32)DYN_SMEM) __trap();
    }
    __syncthreads();
    if (warp == 0) mk_tmem_alloc(smem_u32((void*)tslot));
    if (tid == 0) {
        for (int i = 0; i < RING; ++i) { mbar_init(base + OFF_MB + 8 * i, 1); mbar_init(base + OFF_MB + 64 + 8 * i, 1); }
        if constexpr (LMP) {
            for (int i = 0; i < LMR; ++i) { mbar_init(base + OFF_MB + LMB_FULL + 8 * i, 1); mbar_init(base + OFF_MB + LMB_EMPTY + 8 * i, 1); }
            for (int b = 0; b < 2; ++b) { mbar_init(base + OFF_MB + LMB_ACCF + 8 * b, 1); mbar_init(base + OFF_MB + LMB_ACCE + 8 * b, LM_EPI_WARPS); }
        }
        mbar_fence_init();
        for (int m = 0; m < NMAP; ++m) map_acquire(maps + 128 * m);
#if MK_STAMPS
        mk_sk_n = 0;
#endif  // MK_STAMPS
    }
    bool lm_ident = false;   // LMTS: n_lm == ntok and lm_tok[s] == s (the tile-stream LM reads B row s = token s of xb); block- and grid-uniform
    bool lmi_ok = true;
    for (int i = tid; i < MAXTOK; i += NT) {
        s_ss[i] = 0.f;
        s_lmtok[i] = i < n_lm ? __ldg(lm_tok + i) : 0;
        if constexpr (LMTS) lmi_ok = lmi_ok && (i >= n_lm || s_lmtok[i] == i);
        s_lmseq[i] = i < n_lm ? __ldg(lm_seq + i) : 0;
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) { s_bk[qq][i] = 0u; s_bi[qq][i] = 0x7fffffff; }
    }
    if constexpr (LMTS) lm_ident = __syncthreads_and(lmi_ok) && n_lm == ntok;
    else __syncthreads();
    tc_fence_after();
    GS g;
    g.base = base; g.sm = sm; g.bars = base + OFF_MB; g.tb = *tslot; g.maps = maps; g.F = 0;
    if (MK_DBG_EXIT == 1) return;
#ifdef MK_SKV
    unsigned ring_phase = 0;   // A1: bit 31 = its mbarriers are initialised, low bits = TMA stages consumed; per CTA, the same variable for every layer
    if (tid == 0 && skv_maps != nullptr)
        for (int m = 0; m < 2 * (int)skv::kNumMaps; ++m) map_acquire(reinterpret_cast<const unsigned char*>(skv_maps) + 128 * m);
#endif

    // x[t] = embed[token[t]]; qkv and gu zeroed (their red.add targets)
    if constexpr (TS) {
        for (int i = blockIdx.x * NT + tid; i < ntok * HID; i += gridDim.x * NT) {
            const float xv = bf2f(embed[(size_t)token[i / HID] * HID + (i % HID)]);
            __stcg(x + i, xv);
            xb[i] = (bf16_t)(pack_bf16(xv * bf2f(ln1[i % HID]), 0.f) & 0xffffu);   // xb[t][k] = bf16(x * ln1[0][k]), the value the v0 QKV get stages
        }
    } else {
        for (int i = blockIdx.x * NT + tid; i < ntok * HID; i += gridDim.x * NT)
            __stcg(x + i, bf2f(embed[(size_t)token[i / HID] * HID + (i % HID)]));
    }
    zero_f32(qkv, ntok * QKVN);
    if constexpr (!TSB) zero_f32(gu, ntok * GUN);   // TSB: no gu buffer (GU accumulates in TMEM, ss2 is zeroed in the QKV extra())
    if (MK_DBG_EXIT == 2) return;
    if constexpr (TS) fence_proxy_async_global();   // xb generic stores -> TMA reads of every CTA after the grid barrier
    MK_PF_ARRIVE(if constexpr (TS) { ts_first<PhQkv, TS_TT, TS_KS>(g, 0); } else if (tid == 0) {   // layer 0 QKV: first weight units (landing during the barrier)
        const Geom<PhQkv> ge = geom<PhQkv>();
        for (int i = 0; i < RING && i < ge.n; ++i) fill_unit<PhQkv>(g, 0, ge.u0 + i, i);
    });
    bar_wait(bar_cnt, bar_target);

    for (int l = 0; l < NLAY; ++l) {
        // ---- Phase 1: QKV GEMM, B = bf16(x * ln1) (deferred norm), red.add -> qkv; rr1; zero gu ----
        {
            const bf16_t* lnw = ln1 + (size_t)l * HID;
            auto get = [&](int t, int k, int) -> uint4 {
                if (t >= ntok) return make_uint4(0u, 0u, 0u, 0u);
                const float4 a = __ldcg(reinterpret_cast<const float4*>(x + (size_t)t * HID + k));
                const float4 b = __ldcg(reinterpret_cast<const float4*>(x + (size_t)t * HID + k + 4));
                return pack_xw(a, b, __ldg(reinterpret_cast<const uint4*>(lnw + k)));
            };
            auto epi = [&](int row, int c0, const u32 (&v)[16], int) {
#pragma unroll
                for (int i = 0; i < 16; ++i)
                    if (c0 + i < ntok) red_add(qkv + (size_t)(c0 + i) * QKVN + row, __uint_as_float(v[i]));
            };
            const bool zssf = LMTS && l == NLAY - 1;   // ssf: sums of squares of the last layer's x (MAXTOK-sized, MAXTOK % 4 == 0); constant false (not captured) without LMTS
            auto extra = [&]() { rr_of_x(x, rr1, ntok); if constexpr (TSB) { zero_f32(ss2, ntok); if (zssf) zero_f32(ss2 + MAXTOK, (ntok + 3) & ~3); } else zero_f32(gu, ntok * GUN); };
#ifndef MK_SKV
            skv_prefetch(l, blockIdx.x, gridDim.x);   // splitkv: A1 has no prefetch (L2 prefetch measured net <= 0)
#endif
            if constexpr (TS) {
                ts_phase<PhQkv, TS_TT, TS_KS, 8, mk_null(0)>(g, l, ntok, epi);   // joined by a __syncthreads inside
                if constexpr (!mk_null(0)) extra();
            } else {
                gemm_phase<PhQkv, mk_null(0)>(g, l, ntok, get, epi, extra);
            }
        }
#ifdef MK_SKV
        bar_arrive(bar_cnt);   // splitkv: attention's TMA ring overlays the weight ring: O's first units are prefetched after attend (phase 3 window)
#else
        MK_PF_ARRIVE(prefetch_next<PhO>(g, l));   // phases 2 and 3 do not use the ring: O's first units land meanwhile
#endif
        bar_wait(bar_cnt, bar_target);

        // ---- Phase 2: q/k-norm + RoPE + KV write; one warp per (token, head); head < 16: q (-> qb), else kv head ----
        if constexpr (!mk_null(1)) {
            for (int it = gwarp; it < ntok * (NQH + NKVH); it += nwarps) {
                const int t = it / (NQH + NKVH), hh = it - t * (NQH + NKVH);
                const bool isq = hh < NQH;
                const int kvh = hh - NQH;
                const float r1 = __ldcg(rr1 + t);
                const int pos = __ldg(t_vis + t) - 1;
                const float4 a = __ldcg(reinterpret_cast<const float4*>(qkv + (size_t)t * QKVN + (isq ? hh * HDIM : QDIM + kvh * HDIM)) + lane);
                const uint2 nw = __ldg(reinterpret_cast<const uint2*>((isq ? qn : kn) + l * HDIM + lane * 4));
                float e[4] = {a.x * r1, a.y * r1, a.z * r1, a.w * r1}, o[4];
                const float rn = rsqrtf(warp_sum(e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + e[3] * e[3]) / (float)HDIM + EPS);
#pragma unroll
                for (int i = 0; i < 4; ++i) e[i] = e[i] * rn * bf2f((i < 2 ? nw.x : nw.y) >> ((i & 1) * 16) & 0xffffu);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float pr = __shfl_xor_sync(0xffffffffu, e[i], 16);
                    const int j = (lane * 4 + i) & 63;
                    const float cs = __ldg(rope + (size_t)pos * 128 + j), sn = __ldg(rope + (size_t)pos * 128 + 64 + j);
                    o[i] = e[i] * cs + (lane < 16 ? -pr : pr) * sn;
                }
                const uint2 ob = make_uint2(pack_bf16(o[0], o[1]), pack_bf16(o[2], o[3]));
                if (isq) {
                    *reinterpret_cast<uint2*>(qb + ((size_t)t * NQH + hh) * HDIM + lane * 4) = ob;
                } else {
                    const size_t orow = (((size_t)l * cache_rows + __ldg(t_base + t) + pos) * NKVH + kvh) * HDIM + lane * 4;
                    *reinterpret_cast<uint2*>(kcache + orow) = ob;
                    const float4 v = __ldcg(reinterpret_cast<const float4*>(qkv + (size_t)t * QKVN + QDIM + HID + kvh * HDIM) + lane);
                    *reinterpret_cast<uint2*>(vcache + orow) = make_uint2(pack_bf16(v.x * r1, v.y * r1), pack_bf16(v.z * r1, v.w * r1));
                }
            }
        }
#ifdef MK_SKV
        if constexpr (!mk_null(1)) fence_proxy_async_global();   // the new K / V row (generic stores) -> attend's TMA reads in other CTAs, after the grid barrier
#endif
        bar_arrive(bar_cnt);
        bar_wait(bar_cnt, bar_target);

        // ---- Phase 3: attention (A1 interface; placeholder: one CTA per (token, q-head) item) -> ao ----
#ifdef MK_SKV
        if constexpr (!mk_null(2)) if (!MK_DBG_NOATT || ntok < 0) { MK_SKV_PARAMS(SP); skv::attend<NT>(SP, l, mk_opaque(sm + OFF_BARS - skv::kBarOff), ring_phase); }   // CTAs without items return early (no block barrier): bar_arrive below starts with __syncthreads
        bar_arrive(bar_cnt);   // always arrive first (MK_PFA 0 order would race the TMA writes with other warps' last reads of attend's ring)
        prefetch_next<PhO>(g, l);   // attend is done with the ring bytes (all its reads precede the arrive's __syncthreads): O's first units land during the barrier
#else
        if constexpr (!mk_null(2))
            skv_attention(l, reinterpret_cast<const unsigned short*>(qb), reinterpret_cast<const unsigned short*>(kcache),
                          reinterpret_cast<const unsigned short*>(vcache), reinterpret_cast<unsigned short*>(ao), ntok, t_base, t_vis,
                          cache_rows, sm + OFF_X, blockIdx.x, gridDim.x);
        bar_arrive(bar_cnt);
#endif
        bar_wait(bar_cnt, bar_target);

        // ---- Phase 4: O GEMM (K = 2048), B = ao, red.add -> x (residual); zero qkv ----
        {
#ifdef MK_SKV
            SkvB get{skv_comb, ao, ntok};   // the B tile of unit (rb, kc) is q-head kc: staged here by combine_head (writes the SW128 K-major slot at OFF_X, ends with fence.proxy.async), or (skv_comb 0) by stage_any from ao inside gemm_phase; its __syncthreads precedes the MMA
            if constexpr (!mk_null(2) && !mk_null(3)) {   // MK_NULL bit 2: attention (and so the combine) skipped; bit 3: the O phase has no staging
                const Geom<PhO> ge = geom<PhO>();
                MK_SKV_PARAMS(SP);
                if (skv_comb && ge.n > 0) skv::combine_head<NPAD, NT>(SP, l, ge.u0 % (PhO::K / KU), mk_opaque(sm + OFF_X), ntok);
            }
#else
            skv_combine(l, ntok, reinterpret_cast<unsigned short*>(ao), blockIdx.x, gridDim.x);   // placeholder: no splits
            auto get = [&](int t, int k, int) -> uint4 {
                if (t >= ntok) return make_uint4(0u, 0u, 0u, 0u);
                return __ldcg(reinterpret_cast<const uint4*>(ao + (size_t)t * QDIM + k));
            };
#endif
            auto epi = [&](int row, int c0, const u32 (&v)[16], int) {
#pragma unroll
                for (int i = 0; i < 16; ++i)
                    if (c0 + i < ntok) red_add(x + (size_t)(c0 + i) * HID + row, __uint_as_float(v[i]));
            };
            auto extra = [&]() { zero_f32(qkv, ntok * QKVN); };
            if constexpr (TSB) {
                // P5b: after a CTA's flush of row block rb its O tickets (one per token tile) get nk units; the idle CTAs (no O unit) convert x -> xb2, ss2 once a ticket reaches 16
                auto fin = [&](int rb, int nk) {
                    __syncthreads();   // every thread's red.adds of this row block are issued
                    if (tid == 0) {
                        __threadfence();   // release: the CTA's red.adds (ordered before this by the barrier) precede the ticket adds
                        for (int tt = 0; tt < NPAD / 128; ++tt) atomicAdd(tk + rb * (NPAD / 128) + tt, (unsigned)nk);
                    }
                };
                gemm_phase<PhO, mk_null(3)>(g, l, ntok, get, epi, extra, fin);
                if constexpr (!mk_null(3))
                    idle_finalize<true, PhO::NRB * (PhO::K / KU)>(x, ln2 + (size_t)l * HID, xb2, ss2, tk, ntok);
            } else {
                gemm_phase<PhO, mk_null(3)>(g, l, ntok, get, epi, extra);
            }
        }
        MK_PF_ARRIVE(if constexpr (TSB) ts_first<PhGu, GU_TT, GU_KS>(g, l); else prefetch_next<PhGu>(g, l));
        bar_wait(bar_cnt, bar_target);

        // ---- Phase 5: GU GEMM, B = bf16(x * ln2), red.add -> gu (rows 0..3071 gate, 3072.. up); rr2 ----
        if constexpr (TSB) {
            // P5b: GU on the tile stream (A = wgu row block rb: 64 gate + 64 up rows, B = xb2 by TMA, KS 1): the epilogue turns the TMEM accumulator into act = bf16(silu(r2 g) * (r2 u))
            auto epi = [&](int q, int lane, int grp, int rb, int cb, const u32 (&v)[16]) {
                if (q < 0) {   // setup: s_ss[t] = r2 of token t (s_ss is the LM head's table; re-zeroed after this phase). One load per token, not one per (token, chunk) in the loop below (job 9618: those loads cost ~8.6 us/layer at b256)
                    for (int t = tid - 128; t < ntok; t += 384) s_ss[t] = rsqrtf(__ldcg(ss2 + t) * (1.f / HID) + EPS);
                    named_bar_sync(7, 384);
                    return;
                }
                const int pid = grp * 2 + (q & 1);   // the pair (quadrants q & 1 and (q & 1) + 2 of group grp): named barrier id 1 + pid, 64 threads
                float* ex = reinterpret_cast<float*>(sm + TS_XOFF) + pid * 512;   // [16 tokens][32 lanes] fp32
                if (q >= 2) {   // u warp: lanes 64..127 of the tile = up of channel 64 rb + 32 (q - 2) + lane
#pragma unroll
                    for (int i = 0; i < 16; ++i) ex[i * 32 + lane] = __uint_as_float(v[i]);
                    named_bar_sync(1 + pid, 64);   // u written
                    named_bar_sync(1 + pid, 64);   // g warp has read them (slot reusable)
                } else {        // g warp: lanes 0..63 = gate of channel 64 rb + 32 q + lane
                    float ua[16];
                    named_bar_sync(1 + pid, 64);
#pragma unroll
                    for (int i = 0; i < 16; ++i) ua[i] = ex[i * 32 + lane];
                    named_bar_sync(1 + pid, 64);
                    const int ch = 64 * rb + 32 * q + lane;
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const int t = cb + i;
                        if (t < ntok) {   // warp-uniform
                            const float r = s_ss[t];
                            const float gg = r * __uint_as_float(v[i]);
                            const float h = MK_SILU(gg) * (r * ua[i]);
                            act[(size_t)t * INTER + ch] = (bf16_t)(pack_bf16(h, 0.f) & 0xffffu);
                        }
                    }
                }
            };
            ts_phase<PhGu, GU_TT, GU_KS, 9, mk_null(4)>(g, l, ntok, epi);
            if constexpr (!mk_null(4)) {
                if (warp >= 4) fence_proxy_async_global();   // act: generic stores -> D's TMA reads (after the grid barrier)
                for (int t = tid; t < ntok; t += NT) s_ss[t] = 0.f;   // the LM head accumulates into s_ss (zero since the prologue); the join barrier above ordered the epilogue reads
            }   // act: generic stores -> D's TMA reads (after the grid barrier)
        } else {
            const bf16_t* lnw = ln2 + (size_t)l * HID;
            auto get = [&](int t, int k, int) -> uint4 {
                if (t >= ntok) return make_uint4(0u, 0u, 0u, 0u);
                const float4 a = __ldcg(reinterpret_cast<const float4*>(x + (size_t)t * HID + k));
                const float4 b = __ldcg(reinterpret_cast<const float4*>(x + (size_t)t * HID + k + 4));
                return pack_xw(a, b, __ldg(reinterpret_cast<const uint4*>(lnw + k)));
            };
            auto epi = [&](int row, int c0, const u32 (&v)[16], int) {
#pragma unroll
                for (int i = 0; i < 16; ++i)
                    if (c0 + i < ntok) red_add(gu + (size_t)(c0 + i) * GUN + row, __uint_as_float(v[i]));
            };
            auto extra = [&]() { rr_of_x(x, rr2, ntok); };
            gemm_phase<PhGu, mk_null(4)>(g, l, ntok, get, epi, extra);
        }
        MK_PF_ARRIVE(if constexpr (TSB) ts_first<PhD, D_TT, D_KS>(g, l); else prefetch_next<PhD>(g, l));
        bar_wait(bar_cnt, bar_target);

        // ---- Phase 6: D GEMM (K = 3072), B = bf16(silu(rr2 * gate) * (rr2 * up)), red.add -> x ----
        {
            auto get = [&](int t, int k, int) -> uint4 {
                if (t >= ntok) return make_uint4(0u, 0u, 0u, 0u);
                const float r = __ldcg(rr2 + t);
                const float* gp = gu + (size_t)t * GUN + k;
                const float4 g0 = __ldcg(reinterpret_cast<const float4*>(gp)), g1 = __ldcg(reinterpret_cast<const float4*>(gp + 4));
                const float4 u0 = __ldcg(reinterpret_cast<const float4*>(gp + INTER)), u1 = __ldcg(reinterpret_cast<const float4*>(gp + INTER + 4));
                const float ga[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
                const float ua[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
                float h[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) { const float gg = r * ga[i]; h[i] = MK_SILU(gg) * (r * ua[i]); }
                return make_uint4(pack_bf16(h[0], h[1]), pack_bf16(h[2], h[3]), pack_bf16(h[4], h[5]), pack_bf16(h[6], h[7]));
            };
            auto epi = [&](int row, int c0, const u32 (&v)[16], int) {
#pragma unroll
                for (int i = 0; i < 16; ++i)
                    if (c0 + i < ntok) red_add(x + (size_t)(c0 + i) * HID + row, __uint_as_float(v[i]));
            };
            auto extra = [&]() {};
            if constexpr (TSB) {
                // P5b: D on the tile stream (A = wd, B = act by TMA, KS = D_KS); each tile's CTA adds 1 to its D ticket after the join, the idle CTAs convert x -> xb for the next QKV
                ts_phase<PhD, D_TT, D_KS, 11, mk_null(5)>(g, l, ntok, epi);   // epi: the v0 D red.add into x; joined by a __syncthreads inside
                if constexpr (!mk_null(5)) {
                    if (l + 1 < NLAY || (LMTS && lm_ident)) {   // block-uniform; the last layer has no next QKV: no tickets, no conversion (LMTS: the finalizer writes xb = bf16(x * fnorm) and ssf for the LM head)
                        if (tid == 0) {
                            const TsGeom ge = ts_geom<PhD, D_TT, D_KS>();
                            if (ge.nt > 0) {
                                __threadfence();   // release: this CTA's red.adds (ordered before tid 0 by the join barrier) precede the ticket adds
                                for (int ti = 0; ti < ge.nt; ++ti) atomicAdd(tk + TK_D + (ge.t0 + ti) / D_KS, 1u);
                            }
                        }
                        if constexpr (LMTS) {
                            if (l + 1 < NLAY) idle_finalize<false, TsC<PhD, D_TT, D_KS>::NTILE>(x, ln1 + (size_t)(l + 1) * HID, xb, ss2, tk, ntok);
                            else idle_finalize<false, TsC<PhD, D_TT, D_KS>::NTILE, true>(x, fnorm, xb, ss2 + MAXTOK, tk, ntok);
                        } else {
                            idle_finalize<false, TsC<PhD, D_TT, D_KS>::NTILE>(x, ln1 + (size_t)(l + 1) * HID, xb, ss2, tk, ntok);
                        }
                    }
                }
            } else if constexpr (TS) {
                // MK_TS: after a CTA's flush of row block rb, the last of the 24 flushing units' CTAs (ticket) writes xb[t][rb*128 ..] = bf16(x * ln1[l+1]) for the next QKV
                auto fin = [&](int rb, int nk) {
                    if constexpr (TS) {
                        if (l + 1 < NLAY) {   // block-uniform
                            __syncthreads();   // every thread's red.adds of this row block are issued
                            if (tid == 0) {
                                __threadfence();   // release: the CTA's red.adds (ordered before this by the barrier) precede the ticket
                                const unsigned old = atomicAdd(tk + rb, (unsigned)nk);
                                s_flag = (old + (unsigned)nk == (unsigned)(PhD::K / KU));
                            }
                            __syncthreads();
                            if (s_flag) {   // last CTA of this row block: all 24 units have been flushed
                                __threadfence();   // acquire side of the ticket
                                const bf16_t* lnw = ln1 + (size_t)(l + 1) * HID + rb * 128;
                                for (int i = tid; i < ntok * 16; i += NT) {
                                    const int t = i >> 4, cg = i & 15;
                                    const float* xp = x + (size_t)t * HID + rb * 128 + cg * 8;
                                    const float4 a = __ldcg(reinterpret_cast<const float4*>(xp)), b = __ldcg(reinterpret_cast<const float4*>(xp + 4));
                                    *reinterpret_cast<uint4*>(xb + (size_t)t * HID + rb * 128 + cg * 8) = pack_xw(a, b, __ldg(reinterpret_cast<const uint4*>(lnw + cg * 8)));
                                }
                                fence_proxy_async_global();   // generic writes -> TMA reads in other CTAs (after the grid barrier release / acquire)
                                if (tid == 0) atomicExch(tk + rb, 0u);
                            }
                        }
                    } else {
                        (void)rb; (void)nk;
                    }
                };
                gemm_phase<PhD, mk_null(5)>(g, l, ntok, get, epi, extra, fin);
            } else {
                gemm_phase<PhD, mk_null(5)>(g, l, ntok, get, epi, extra);
            }
        }
        MK_PF_ARRIVE(if (l + 1 < NLAY) { if constexpr (TS) ts_first<PhQkv, TS_TT, TS_KS>(g, l + 1); else prefetch_next<PhQkv>(g, l + 1); } else { if constexpr (LMTS) { if (lm_ident) ts_first<PhLm, 128, 1>(g, 0); else lm_first(g); } else lm_first(g); });
        bar_wait(bar_cnt, bar_target);
    }

    // ---- LM head: B = bf16(x[lm_tok] * fnorm) (slot s = row s), full row blocks (K = 1024 accumulated in TMEM), logits = rrf * acc,
    // per-quadrant best (key, row) per slot; rrf comes from the sums of squares collected while staging the CTA's first row block ----
    {
        constexpr int CPR = KU / 8, KCL = HID / KU;
        auto get = [&](int t, int k, int j) -> uint4 {
            const bool valid = t < n_lm;
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = a;
            if (valid) {
                const size_t xo = (size_t)s_lmtok[t] * HID + k;
                a = __ldcg(reinterpret_cast<const float4*>(x + xo));
                b = __ldcg(reinterpret_cast<const float4*>(x + xo + 4));
            }
            if (j < KCL) {   // first row block of this CTA: x^2 sums of the whole row pass through here exactly once
                float sq = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
#pragma unroll
                for (int o = CPR / 2; o; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
                if (valid && (threadIdx.x & (CPR - 1)) == 0) atomicAdd(&s_ss[t], sq);
            }
            return pack_xw(a, b, __ldg(reinterpret_cast<const uint4*>(fnorm + k)));
        };
        auto epi = [&](int row, int c0, const u32 (&v)[16], int) {
            u32 rk = 0u;
            int ri = 0;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int s = c0 + i;
                if (s < n_lm) {   // warp-uniform
                    const float val = rsqrtf(s_ss[s] * (1.f / HID) + EPS) * __uint_as_float(v[i]);
                    logits[(size_t)s_lmseq[s] * VOCAB + row] = val;
                    const u32 ub = __float_as_uint(val);
                    const u32 key = (ub & 0x80000000u) ? ~ub : (ub | 0x80000000u);   // order-preserving float -> uint
                    const u32 mx = __reduce_max_sync(0xffffffffu, key);
                    const u32 eq = __ballot_sync(0xffffffffu, key == mx);
                    if (lane == i) { rk = mx; ri = row - lane + (__ffs(eq) - 1); }   // lowest row among equal maxima
                }
            }
            if (lane < 16 && c0 + lane < n_lm) {   // this warp is the only writer of (quadrant, slot)
                const int s = c0 + lane, qq = warp & 3;
                if (rk > s_bk[qq][s]) { s_bk[qq][s] = rk; s_bi[qq][s] = ri; }   // row blocks ascend: strict > keeps the lowest index
            }
        };
        auto extra = [&]() {};
        if constexpr (LMTS) {
            if (lm_ident) {   // SPEC-P3c: B = xb (fnorm rows, written by the last D finalize), ssf = raw sums of squares; epi as above (it applies rsqrt(s_ss / HID + EPS))
                for (int t = tid; t < n_lm; t += NT) s_ss[t] = __ldcg(ss2 + MAXTOK + t);   // s_ss is zero for t >= n_lm (prologue, GU re-zero)
                __syncthreads();
                ts_phase<PhLm, 128, 1, 8, mk_null(6)>(g, 0, n_lm, epi);
            } else {
                if constexpr (LMP) lm_phase<mk_null(6)>(g, get, epi);
                else gemm_phase<PhLm, mk_null(6)>(g, 0, n_lm, get, epi, extra);
            }
        } else {
            if constexpr (LMP) lm_phase<mk_null(6)>(g, get, epi);
            else gemm_phase<PhLm, mk_null(6)>(g, 0, n_lm, get, epi, extra);
        }
    }
    __syncthreads();
    tc_fence_after();
    if (warp == 0) mk_tmem_dealloc(g.tb);
#if MK_STAMPS
    if (tid == 0) mk_stamp();   // end of the LM phase
    __syncthreads();
    for (int i = tid; i < MK_NSTAMP; i += NT) mk_stamps[blockIdx.x * MK_NSTAMP + i] = i < (int)mk_sk_n ? mk_sk_s[i] : 0ull;
#endif  // MK_STAMPS

    // ---- argmax: per-CTA best per slot -> last CTA (ticket) reduces; ties -> lowest index ----
    if (tid < n_lm) {
        u32 bk = s_bk[0][tid];
        int bi = s_bi[0][tid];
#pragma unroll
        for (int qq = 1; qq < 4; ++qq) {
            const u32 k = s_bk[qq][tid];
            const int i = s_bi[qq][tid];
            if (k > bk || (k == bk && i < bi)) { bk = k; bi = i; }
        }
        __stcg(pkey + (size_t)blockIdx.x * MAXTOK + tid, bk);
        __stcg(pidx + (size_t)blockIdx.x * MAXTOK + tid, bi);
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned tk = atomicAdd(ticket, 1u);
        s_flag = (tk == gridDim.x - 1);
    }
    __syncthreads();
    if (s_flag) {   // last CTA: a warp per slot
        __threadfence();
        for (int slot = warp; slot < n_lm; slot += NW) {
            u32 bk = 0u;
            int bi = 0x7fffffff;
            for (int b = lane; b < (int)gridDim.x; b += 32) {
                const u32 k = __ldcg(pkey + (size_t)b * MAXTOK + slot);
                const int i = __ldcg(pidx + (size_t)b * MAXTOK + slot);
                if (k > bk || (k == bk && i < bi)) { bk = k; bi = i; }
            }
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                const u32 ok = __shfl_xor_sync(0xffffffffu, bk, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ok > bk || (ok == bk && oi < bi)) { bk = ok; bi = oi; }
            }
            if (lane == 0) next_token[__ldg(lm_seq + slot)] = (long long)bi;
        }
        __syncthreads();
        if (tid == 0) {
            atomicExch(ticket, 0u);
            atomicExch(bar_cnt, 0u);   // every CTA has passed its last barrier poll
        }
    }
}
