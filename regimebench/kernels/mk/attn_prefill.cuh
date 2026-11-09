// mk prefill attention (MK_BIGT step 3): causal GQA attention of a large-T step on tcgen05, owned by MK-core (kernels/mk/), so it does not touch
// common/splitkv (MK-attn). Only the MK_BIGT build compiles it: the bucket builds see an empty file (SASS unchanged).
//
// Work item = (sequence, kv head, 64-token query block): one host-planned int4 {tok0, nq | kvh << 8, pos0, row_base} (build.py pf_plan).
//   tok0     = step token index of the first query, nq = queries in the block (1..64), pos0 = cache position of the first query (kv_len + 64 b),
//   row_base = first cache row of the sequence. Query i (0..nq-1) sees positions 0 .. pos0 + i (causal), so the item reads (pos0 + nq) / 128 KV blocks.
// Tile: M = 128 rows = 2 q heads (2 kvh, 2 kvh + 1) x 64 queries (row r: head r >> 6, query r & 63), so both heads of the GQA pair share each K / V block.
//   S  = Q K^T   A = Q   [128 rows][128 d] K-major SW128 (4 TMA boxes {64 d, 1 head, 64 tokens}),  B = K block [128 pos][128 d] K-major SW128, D = TMEM cols 0..127
//   PV = P V     A = P   [128 rows][128 pos] K-major SW128 (bf16, written by the softmax threads),     B = V block [128 pos][128 d] MN-major SW128 (idesc bit 16;
//                LBO = 16 KB between the two 64-dim boxes, SBO = 1 KB per 8 positions: the layout splitkv's SKV_TC consumer uses for V as A), D = TMEM cols 128..255
// Softmax: all 16 warps. Warp w: TMEM quadrant w & 3 (row r = 32 (w & 3) + lane), column group cg = w >> 2 (score columns / output dims 32 cg .. 32 cg + 31). The row max
// is combined over the 4 column groups through smem; each thread keeps its own partial row sum and its 32 output dims in registers (rescaled by the common max).
// The issuing thread (tid 0) also does its softmax share: per block it issues PV(j), then S(j + 1), then the K / V loads of block j + 2 into the freed stage.
// Work distribution: a per-layer item counter (ctr[l], zeroed in the prologue); items are planned longest first.
// mbarriers: the tile-stream block (LMB_*), invalidated and re-initialised here (all 20, count 1), so the O phase's ts_first finds valid objects:
//   full[0] = Q, full[1 + s] = K stage s, full[3 + s] = V stage s, accf[0] = S done, accf[1] = PV done.
#if MK_BIGT
constexpr int PF_QOFF = 0, PF_KOFF = 32768, PF_VOFF = 65536, PF_STAGE = 65536;   // Q 32 KB | stage s: K 32 KB, V 32 KB at 32 KB + 64 KB s
constexpr int PF_POFF = PF_KOFF + 2 * PF_STAGE;                                // P 32 KB
constexpr int PF_RED = PF_POFF + 32768;                                         // [4 column groups][128 rows] fp32: partial row max, then partial row sum
constexpr int PF_ITEM = PF_RED + 4 * 128 * 4;                                   // next item index (broadcast)
constexpr int PF_BYTES = PF_ITEM + 16;
constexpr int PF_MAP0 = 13;                                                     // tensor maps 13, 14, 15 = qb, kcache, vcache (build.py pf_map_specs)
constexpr int PF_QBLK = 64;                                                     // queries per item (build.py PF_QBLK)
constexpr float PF_SC = 0.12751743074133604f;                                   // softmax scale 1 / sqrt(128) times log2(e) (scores in the exp2 domain)
static_assert(PF_BYTES <= OFF_BARS, "prefill attention smem overlays the tile-stream ring (idle between the QKV and O phases)");

__device__ __forceinline__ void pf_tma3(u32 dst, const void* map, int c0, int c1, int c2, u32 bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
                 :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ float pf_ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
// SW128 MN-major smem descriptor (V as the B operand): LBO = 16 KB (second 64-dim box), SBO = 1 KB (8 positions x 128 B)
__device__ __forceinline__ ull pf_vdesc(u32 addr) { return (ull)((addr >> 4) & 0x3fff) | ((ull)(16384 >> 4) << 16) | ((ull)(1024 >> 4) << 32) | (1ull << 46) | (2ull << 61); }

// tid 0: K and V of KV block gj of this CTA (stage gj & 1), cache row `row` (layer-major), kv head kvh
__device__ __forceinline__ void pf_load_kv(u32 base, u32 fullb, const unsigned char* maps, u32 gj, int kvh, int row) {
    const u32 st = gj & 1, kb = fullb + 8 * (1 + st), vb = fullb + 8 * (3 + st);
    const u32 kd = base + PF_KOFF + st * PF_STAGE, vd = base + PF_VOFF + st * PF_STAGE;
    mbar_expect_tx(kb, 32768);
    pf_tma3(kd, maps + 128, 0, kvh, row, kb);
    pf_tma3(kd + 16384, maps + 128, 64, kvh, row, kb);
    mbar_expect_tx(vb, 32768);
    pf_tma3(vd, maps + 256, 0, kvh, row, vb);
    pf_tma3(vd + 16384, maps + 256, 64, kvh, row, vb);
}
// tid 0: S = Q K^T of the block in stage st into TMEM columns 0..127
__device__ __forceinline__ void pf_mma_s(u32 base, u32 tb, u32 st) {
    constexpr u32 id = tc_idesc(128, 128);
    const u32 qa = base + PF_QOFF, ka = base + PF_KOFF + st * PF_STAGE;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
        const int a = kk >> 2, off = (kk & 3) * 32;
        tc_mma(tb, sdesc(qa + a * 16384 + off), sdesc(ka + a * 16384 + off), id, kk > 0 ? 1u : 0u);
    }
}
// tid 0: PV = P V of the block in stage st into TMEM columns 128..255 (overwrite)
__device__ __forceinline__ void pf_mma_pv(u32 base, u32 tb, u32 st) {
    constexpr u32 id = tc_idesc(128, 128) | (1u << 16);   // B (V) MN-major
    const u32 pa = base + PF_POFF, va = base + PF_VOFF + st * PF_STAGE;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) tc_mma(tb + 128, sdesc(pa + (kk >> 2) * 16384 + (kk & 3) * 32), pf_vdesc(va + kk * 2048), id, kk > 0 ? 1u : 0u);
}

// All threads of every CTA, between grid barriers (the tile-stream ring and TMEM are idle). base / sm: the 1024-aligned smem base, bars: mk's barrier block
// (OFF_MB), tb: TMEM base (256 columns), maps: the 3 attention maps (qb, K, V; 128 B each, acquired in the prologue). ao: bf16 [ntok][16 * 128].
// Every issued TMA and MMA is waited before return (the O phase's ts_first invalidates these mbarriers).
__device__ __forceinline__ void pf_attention(u32 base, unsigned char* sm, u32 bars, u32 tb, const unsigned char* maps, int l, int cache_rows,
                                             const int4* __restrict__ items, int n_items, unsigned* ctr, bf16_t* ao) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, q = warp & 3, cg = warp >> 2;
    const int r = 32 * q + lane, h2 = r >> 6, ri = r & 63;
    const u32 fullb = bars + LMB_FULL, accf = bars + LMB_ACCF;
    float* red = reinterpret_cast<float*>(sm + PF_RED);
    volatile int* s_item = reinterpret_cast<volatile int*>(sm + PF_ITEM);
    const float NEG = __int_as_float(0xff800000);
    if (tid == 0) {
        for (int i = 0; i < 8; ++i) { mbar_inval(fullb + 8 * i); mbar_inval(bars + LMB_EMPTY + 8 * i); }
        for (int b = 0; b < 2; ++b) { mbar_inval(accf + 8 * b); mbar_inval(bars + LMB_ACCE + 8 * b); }
        for (int i = 0; i < 8; ++i) { mbar_init(fullb + 8 * i, 1); mbar_init(bars + LMB_EMPTY + 8 * i, 1); }
        for (int b = 0; b < 2; ++b) { mbar_init(accf + 8 * b, 1); mbar_init(bars + LMB_ACCE + 8 * b, 1); }
        mbar_fence_init();
        s_item[0] = (int)atomicAdd(ctr + l, 1u);
    }
    __syncthreads();
    u32 nq_use = 0, gb = 0, s_cnt = 0, pv_cnt = 0;   // Q loads, KV blocks loaded, S / PV commits (uniform over the CTA)
    int it = s_item[0];
    while (it < n_items) {
        const int4 rec = __ldg(items + it);
        int nxt = 0;
        if (tid == 0) nxt = (int)atomicAdd(ctr + l, 1u);   // the next item (used at the end of this one)
        const int tok0 = rec.x, nq = rec.y & 255, kvh = rec.y >> 8, pos0 = rec.z;
        const int nblk = (pos0 + nq + 127) >> 7;
        const int kvrow = l * cache_rows + rec.w;
        if (tid == 0) {
            mbar_expect_tx(fullb, 32768);
#pragma unroll
            for (int a = 0; a < 2; ++a)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) pf_tma3(base + PF_QOFF + a * 16384 + hh * 8192, maps, a * 64, 2 * kvh + hh, tok0, fullb);
            for (int j = 0; j < 2 && j < nblk; ++j) pf_load_kv(base, fullb, maps, gb + j, kvh, kvrow + 128 * j);
            mbar_wait(fullb, nq_use & 1);
            mbar_wait(fullb + 8 * (1 + (gb & 1)), (gb >> 1) & 1);
            tc_fence_after();
            pf_mma_s(base, tb, gb & 1);
            tc_commit(accf);
        }
        ++nq_use;
        float o[32];
#pragma unroll
        for (int c = 0; c < 32; ++c) o[c] = 0.f;
        float m = NEG, lsum = 0.f;
        for (int j = 0; j < nblk; ++j) {
            const u32 gj = gb + j, st = gj & 1;
            // ---- scores of block j: row r, positions 128 j + 32 cg + c
            mbar_wait(accf, s_cnt & 1);
            ++s_cnt;
            tc_fence_after();
            // pass 1: masked row max over this thread's 32 columns (S re-read from TMEM in pass 2 instead of kept: register pressure)
            u32 va[16], vb[16];
            mk_tmem_read16(va, tb, q, 32 * cg);
            mk_tmem_read16(vb, tb, q, 32 * cg + 16);
            tc_wait_ld();
            const int lim = ri < nq ? pos0 + ri - (128 * j + 32 * cg) : -1;   // column c is visible iff c <= lim
            float pm = NEG;
#pragma unroll
            for (int c = 0; c < 32; ++c) pm = fmaxf(pm, c <= lim ? __uint_as_float(c < 16 ? va[c] : vb[c - 16]) * PF_SC : NEG);
            red[cg * 128 + r] = pm;
            __syncthreads();
            float mn = fmaxf(fmaxf(red[r], red[128 + r]), fmaxf(red[256 + r], red[384 + r]));
            mn = fmaxf(m, mn);
            const float mu = mn == NEG ? 0.f : mn;   // rows with nothing visible (padding rows ri >= nq): p = 0
            const float alpha = pf_ex2(m - mu);
            m = mn;
            // pass 2: p = 2^(s - mu) as bf16 into P row r, positions 32 cg ..: K-major SW128 atom cg >> 1, 16-byte chunks (cg & 1) * 4 + k (k = 0..3)
            float ps = 0.f;
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                mk_tmem_read16(va, tb, q, 32 * cg + 16 * hf);
                tc_wait_ld();
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    u32 pk[4];
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const int c0 = 8 * k + 2 * c, cc = 16 * hf + c0;
                        const float p0 = cc <= lim ? pf_ex2(__uint_as_float(va[c0]) * PF_SC - mu) : 0.f;
                        const float p1 = cc + 1 <= lim ? pf_ex2(__uint_as_float(va[c0 + 1]) * PF_SC - mu) : 0.f;
                        ps += p0 + p1;
                        pk[c] = pack_bf16(p0, p1);
                    }
                    *reinterpret_cast<uint4*>(sm + PF_POFF + (cg >> 1) * 16384 + r * 128 + ((((cg & 1) * 4 + 2 * hf + k) ^ (r & 7)) << 4)) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
                }
            }
            lsum = lsum * alpha + ps;
            fence_proxy_async();
            tc_fence_before();
            __syncthreads();   // P complete, every thread's S read done (red reads too)
            if (tid == 0) {
                tc_fence_after();
                mbar_wait(fullb + 8 * (3 + st), (gj >> 1) & 1);   // V of block j
                pf_mma_pv(base, tb, st);
                tc_commit(accf + 8);
                if (j + 1 < nblk) {   // S of block j + 1 runs while the threads rescale o
                    const u32 g1 = gj + 1;
                    mbar_wait(fullb + 8 * (1 + (g1 & 1)), (g1 >> 1) & 1);
                    pf_mma_s(base, tb, g1 & 1);
                    tc_commit(accf);
                }
            }
            mbar_wait(accf + 8, pv_cnt & 1);
            ++pv_cnt;
            tc_fence_after();
            mk_tmem_read16(va, tb + 128, q, 32 * cg);
            mk_tmem_read16(vb, tb + 128, q, 32 * cg + 16);
            tc_wait_ld();
            tc_fence_before();
#pragma unroll
            for (int c = 0; c < 32; ++c) o[c] = o[c] * alpha + __uint_as_float(c < 16 ? va[c] : vb[c - 16]);
            if (tid == 0 && j + 2 < nblk) pf_load_kv(base, fullb, maps, gj + 2, kvh, kvrow + 128 * (j + 2));   // stage st: S(j) and PV(j) are done
        }
        gb += nblk;
        red[cg * 128 + r] = lsum;   // every thread's red reads of the last block precede the last __syncthreads above
        __syncthreads();
        if (ri < nq) {
            const float inv = 1.f / (red[r] + red[128 + r] + red[256 + r] + red[384 + r]);
            uint4* dst = reinterpret_cast<uint4*>(ao + (size_t)(tok0 + ri) * QDIM + (2 * kvh + h2) * HDIM + 32 * cg);
#pragma unroll
            for (int k = 0; k < 4; ++k)
                dst[k] = make_uint4(pack_bf16(o[8 * k] * inv, o[8 * k + 1] * inv), pack_bf16(o[8 * k + 2] * inv, o[8 * k + 3] * inv),
                                    pack_bf16(o[8 * k + 4] * inv, o[8 * k + 5] * inv), pack_bf16(o[8 * k + 6] * inv, o[8 * k + 7] * inv));
        }
        if (tid == 0) s_item[0] = nxt;
        __syncthreads();   // red reads done before the next item's writes; s_item visible
        it = s_item[0];
    }
}
#else
// names for the discarded MK_BIGT branches of mk_step.cu (never called in the bucket builds)
constexpr int PF_MAP0 = 0;
__device__ __forceinline__ void pf_attention(u32, unsigned char*, u32, u32, const unsigned char*, int, int, const int4*, int, unsigned*, bf16_t*) {}
#endif  // MK_BIGT
