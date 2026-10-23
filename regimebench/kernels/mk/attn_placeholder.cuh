// Attention placeholder behind the A1 interface (kernels/common/README.md). Until `splitkv-vN` exists, mk builds with this header.
// Content = hv2g MTN phase 3 (kernels/hv2g/hv2g_mtn.cu:194-262) + attn_chunk / attn_merge (hv2g_common.cuh), one item per
// (token, q-head), 128-position chunks, 16 warps x 8 positions, with ONE change: q is read from `qb` (bf16, already q-normed and
// RoPE'd by the caller) instead of recomputed, and the output is bf16 [t][qh * 128 + d] (the O-proj B operand).
// Interface rules: only SKV_-prefixed names, no dependence on mk's macros or helper names, model sizes as constexpr.
// Requires: 512 threads per CTA (16 warps), all CTAs of the grid call it (item loop is strided by nct), cache rows are bf16
// [L, cache_rows, 8, 128] with SKV_CTX zero rows allocated after the last layer, the new K/V rows are written before the call.
#ifndef SKV_ATTN_PLACEHOLDER_CUH
#define SKV_ATTN_PLACEHOLDER_CUH

constexpr int SKV_QH = 16;     // q heads
constexpr int SKV_KVH = 8;     // kv heads
constexpr int SKV_HD = 128;    // head dim
constexpr int SKV_CTX = 128;   // positions per chunk (NOT the context length)
constexpr int SKV_NW = 16;     // warps per CTA
constexpr int SKV_PPW = SKV_CTX / SKV_NW;   // positions per warp per chunk
// dynamic / static smem the caller must provide for skv_attention (16-byte aligned): V chunk | per-warp o | per-warp m, l
constexpr int SKV_SMEM_BYTES = SKV_CTX * SKV_HD * 2 + SKV_NW * SKV_HD * 4 + 2 * SKV_NW * 4;
#define SKV_ATTN_SCALE 0.08838834764831845f
#define SKV_EXPF(x) __expf(x)

__device__ __forceinline__ float skv_neginf() { return __int_as_float(0xff800000); }
__device__ __forceinline__ float skv_bflo(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float skv_bfhi(unsigned u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ unsigned short skv_f2bf(float f) {
    unsigned u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}
__device__ __forceinline__ float skv_warp_max(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Chunk state of ONE (token, q-head) for the warp's 8 positions of a 128-position chunk. K of the positions is in registers kk[j]
// (lane: dims lane*4..+3), V in sVc rows (warp * 8 + j). `nvalid` = number of positions of the CHUNK that are visible (>= SKV_CTX:
// all); the others get probability 0. Returns m (warp-uniform max score, -inf if none visible), lsum = sum e^(s - m), o = this lane's
// dims lane*4..+3 of sum e^(s - m) * v.
__device__ __forceinline__ void skv_attn_chunk(const uint2 (&kk)[SKV_PPW], const float4 q, const int nvalid,
                                               const unsigned short* sVc, float& m_out, float& l_out, float4& o_out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float v[SKV_PPW];
#pragma unroll
    for (int j = 0; j < SKV_PPW; ++j) {
        const uint2 kj = kk[j];
        v[j] = skv_bflo(kj.x) * q.x + skv_bfhi(kj.x) * q.y + skv_bflo(kj.y) * q.z + skv_bfhi(kj.y) * q.w;
    }
    // transpose butterfly: the lane ends with the score of position lane >> 2
#pragma unroll
    for (int lv = 0; lv < 3; ++lv) {
        const int hh = SKV_PPW >> (lv + 1), o = 16 >> lv;
        const bool hi = lane & o;
#pragma unroll
        for (int i = 0; i < hh; ++i) {
            float send = hi ? v[i] : v[i + hh], keep = hi ? v[i + hh] : v[i];
            v[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
    }
    v[0] += __shfl_xor_sync(0xffffffffu, v[0], 2);
    v[0] += __shfl_xor_sync(0xffffffffu, v[0], 1);
    float sc = v[0] * SKV_ATTN_SCALE;
    const bool valid = (warp * SKV_PPW + (lane >> 2)) < nvalid;
    if (!valid) sc = skv_neginf();
    const float m = skv_warp_max(sc);
    const float ee = valid ? SKV_EXPF(sc - (m == skv_neginf() ? 0.f : m)) : 0.f;
    float lsum = 0.f;
    float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
    asm volatile("cp.async.wait_all;" ::: "memory");
#pragma unroll
    for (int j = 0; j < SKV_PPW; ++j) {
        const uint2 vj = *(const uint2*)(sVc + (warp * SKV_PPW + j) * SKV_HD + lane * 4);
        const float e = __shfl_sync(0xffffffffu, ee, j * 4);
        lsum += e;
        o.x += e * skv_bflo(vj.x); o.y += e * skv_bfhi(vj.x);
        o.z += e * skv_bflo(vj.y); o.w += e * skv_bfhi(vj.y);
    }
    m_out = m; l_out = lsum; o_out = o;
}

// online-softmax merge of a chunk state (m2, l2, o2) into the running state (m, l, o)
__device__ __forceinline__ void skv_attn_merge(float& m, float& l, float4& o, const float m2, const float l2, const float4 o2) {
    const float mn = fmaxf(m, m2);
    const float a = SKV_EXPF(m - mn), b = SKV_EXPF(m2 - mn);   // m is finite whenever a merge happens (chunk 0 of a multi-chunk context is full)
    l = l * a + l2 * b;
    o.x = o.x * a + o2.x * b; o.y = o.y * a + o2.y * b; o.z = o.z * a + o2.z * b; o.w = o.w * a + o2.w * b;
    m = mn;
}

// Called by every CTA once per layer right after its weight TMAs are issued (the A1 hook for L2 prefetch of KV). Placeholder: nothing.
__device__ __forceinline__ void skv_prefetch(int layer, int cta, int nct) { (void)layer; (void)cta; (void)nct; }

// Attention of all tokens of the step for layer `layer`. qb: bf16 [ntok][SKV_QH][SKV_HD] (post q-norm + RoPE); kcache / vcache: bf16
// [L, cache_rows, SKV_KVH, SKV_HD]; t_base[t] first cache row of the token's sequence, t_vis[t] visible rows (= position + 1, the token's
// own K/V row is t_base + t_vis - 1 and is already written); ao: bf16 [ntok][SKV_QH * SKV_HD]. smem: SKV_SMEM_BYTES, 16-byte aligned.
// Positions past the last visible row of the last chunk are clamped to it (reads stay inside rows written before the caller's barrier,
// so the non-coherent L1 path of the V cp.async never holds a stale copy of a row a later layer of the launch writes) and masked.
__device__ __forceinline__ void skv_attention(const int layer, const unsigned short* __restrict__ qb,
                                              const unsigned short* kcache, const unsigned short* vcache,
                                              unsigned short* ao, const int ntok, const int* __restrict__ t_base,
                                              const int* __restrict__ t_vis, const int cache_rows, void* smem,
                                              const int cta, const int nct) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    unsigned short* sVc = (unsigned short*)smem;
    float* sow = (float*)((char*)smem + SKV_CTX * SKV_HD * 2);   // SKV_NW x 128 per-warp unnormalized o
    float* red = sow + SKV_NW * SKV_HD;                          // [0, NW) m, [NW, 2 NW) l
    for (int item = cta; item < ntok * SKV_QH; item += nct) {
        const int t = item / SKV_QH, qh = item - t * SKV_QH, kvh = qh >> 1;
        const int base = __ldg(t_base + t), vis = __ldg(t_vis + t);
        const int nch = (vis + SKV_CTX - 1) / SKV_CTX;
        uint2 kk[SKV_PPW];
        const unsigned dstv = (unsigned)__cvta_generic_to_shared(sVc + warp * SKV_PPW * SKV_HD + lane * 4);
        auto issue_chunk = [&](int c) {
#pragma unroll
            for (int j = 0; j < SKV_PPW; ++j) {
                const int pp = min(c * SKV_CTX + warp * SKV_PPW + j, vis - 1);
                const size_t off = (((size_t)layer * cache_rows + base + pp) * SKV_KVH + kvh) * SKV_HD + lane * 4;
                kk[j] = __ldcg((const uint2*)(kcache + off));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(dstv + j * SKV_HD * 2), "l"(vcache + off) : "memory");
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        };
        issue_chunk(0);
        float4 q;
        {
            const uint2 qw = __ldcg((const uint2*)(qb + ((size_t)t * SKV_QH + qh) * SKV_HD + lane * 4));
            q = make_float4(skv_bflo(qw.x), skv_bfhi(qw.x), skv_bflo(qw.y), skv_bfhi(qw.y));
        }
        float m, lsum;
        float4 o;
        skv_attn_chunk(kk, q, vis, sVc, m, lsum, o);
        if (nch > 1) issue_chunk(1);
#pragma unroll 1
        for (int c = 1; c < nch; ++c) {
            float m2, l2;
            float4 o2;
            skv_attn_chunk(kk, q, vis - c * SKV_CTX, sVc, m2, l2, o2);
            if (c + 1 < nch) issue_chunk(c + 1);
            skv_attn_merge(m, lsum, o, m2, l2, o2);
        }
        *(float4*)(sow + warp * SKV_HD + lane * 4) = o;
        if (lane == 0) { red[warp] = m; red[SKV_NW + warp] = lsum; }
        __syncthreads();
        if (tid < SKV_HD) {
            float M = red[0];
#pragma unroll
            for (int w = 1; w < SKV_NW; ++w) M = fmaxf(M, red[w]);
            float L = 0.f, ov = 0.f;
#pragma unroll
            for (int w = 0; w < SKV_NW; ++w) {
                const float c = SKV_EXPF(red[w] - M);
                L += c * red[SKV_NW + w];
                ov += c * sow[w * SKV_HD + tid];
            }
            __stcg(ao + (size_t)t * (SKV_QH * SKV_HD) + qh * SKV_HD + tid, skv_f2bf(ov / L));
        }
        __syncthreads();   // sow / red reused by the next item
    }
}

// Merge of the splits of one (token, q-head) into `ao`. The placeholder has no splits: nothing to do.
__device__ __forceinline__ void skv_combine(int layer, int ntok, unsigned short* ao, int cta, int nct) {
    (void)layer; (void)ntok; (void)ao; (void)cta; (void)nct;
}
#endif  // SKV_ATTN_PLACEHOLDER_CUH
