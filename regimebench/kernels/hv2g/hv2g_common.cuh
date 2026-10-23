// hv2g common part: model constants, device helpers (copied unchanged from the frozen hv2 kernel with the
// default macro set resolved: R10_PF 14, R10_GUE / DNE 1, R10_DEF 2, R10_LMH 2, R11_FEXP 1; every diagnostic and
// experiment switch of hv2 is gone), plus the shared attention chunk.
// Compile with -DNB=<SM count> and one of -DHV2G_MT1=1 (ntok == 1 kernel) / -DHV2G_MT1=0 (general kernel).

#ifndef NB
#error "define NB (grid = SM count)"
#endif
#ifndef HV2G_MT1
#error "define HV2G_MT1 (1: MT1 kernel, 0: MTN kernel)"
#endif
#ifndef MAXTOK
#define MAXTOK 256  // MTN: tokens per launch (scratch and pval are sized for it)
#endif

#define R8_STAGE_BYTES 5632  // MT1: warp 0's window stage, 11 units x 32 lanes x 16 B
#define R8_SPIN_CYC 490
#define R8_SPIN_PRO R8_SPIN_CYC
#define R8_SPIN_QKV R8_SPIN_CYC
#define R8_SPIN_FUSED 0
#define R8_SPIN_GU 735
#define R8_SPIN_DOWN 245
#define R11_EXPF(x) __expf(x)
#define R11_SILU(g) ((g) / (1.f + expf(-(g))))
#define H 1024
#define NL 28
#define QH 16
#define KVH 8
#define HD 128
#define INTER 3072
#define VOCAB 151936
#define CTX 128  // positions per attention chunk (NOT the context length: that is runtime)
#define NT 512
#define NW 16
#define EPS 1e-6f
#define NSL 9  // MT1: O-proj row slices: fused block b < QH * NSL owns q-head b % QH, slice b / QH
#define NQD (H / 4)
#define PPW (CTX / NW)  // cached positions per warp per chunk
#define WOU 4
#define INFINITY __int_as_float(0x7f800000)
#define ATTN_SCALE 0.08838834764831845f

typedef unsigned short bf16_t;

__device__ __forceinline__ float bf2f(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bflo(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bfhi(unsigned u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ bf16_t f2bf(float f) {
    unsigned u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16_t)(u >> 16);
}

__device__ __forceinline__ int ctaid_opaque() {  // defeats loop-invariant hoisting (register pressure)
    int v;
    asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(v));
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// GEMV activation vectors in smem use a permuted layout: within each 256-element
// segment, element lane*8+j lives at lane*4+j (j<4) or 128+lane*4+(j-4), so a
// lane's two float4 reads at lane*4 and 128+lane*4 are bank-conflict free.
__device__ __forceinline__ int sperm(int e) {
    const int o = e & 255;
    return (e & ~255) + ((o & 4) << 5) + (o >> 3) * 4 + (o & 3);
}
// float4 index variant: logical float4 f (elements 4f..4f+3) -> physical float4.
__device__ __forceinline__ int sperm4(int f) {
    const int o = f & 63;
    return (f & ~63) + (o & 1) * 32 + (o >> 1);
}

// Grid barrier on a monotonic arrival counter (requires co-residency: cooperative launch). Each block arrives once
// with a release reduction and thread 0 relaxed-polls (then one acquire fence) until all blocks of this epoch have
// arrived; the counter is zeroed at the end of the launch by the last argmax ticket holder.
__device__ __forceinline__ void bar_arrive(unsigned* cnt) {
    __syncthreads();
    if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(cnt) : "memory");
}
template <unsigned SPIN>
__device__ __forceinline__ void bar_wait(const unsigned* cnt, unsigned& target) {
    target += gridDim.x;
    if (threadIdx.x == 0) {
        if (SPIN > 0) {
            unsigned t0, t;
            asm volatile("mov.u32 %0, %%clock;" : "=r"(t0)::"memory");
            do { asm volatile("mov.u32 %0, %%clock;" : "=r"(t)::"memory"); } while (t - t0 < SPIN);
        }
        unsigned v;
        do {
            asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(cnt) : "memory");
        } while (v < target);
        asm volatile("fence.acquire.gpu;" ::: "memory");  // one acquire after the relaxed poll succeeds
    }
    __syncthreads();
}

// same from a smem vector in the sperm layout, segment seg
__device__ __forceinline__ void smem_act(const float* s, int seg, float (&a)[8]) {
    const float* xs = s + seg * 256 + (threadIdx.x & 31) * 4;
    const float4 x0 = *(const float4*)xs, x1 = *(const float4*)(xs + 128);
    a[0] = x0.x; a[1] = x0.y; a[2] = x0.z; a[3] = x0.w; a[4] = x1.x; a[5] = x1.y; a[6] = x1.z; a[7] = x1.w;
}

extern __shared__ __align__(16) uint4 r8_stage[];  // dynamic smem (MT1: stage + sGU + sD; MTN: the activation staging buffer)
__device__ __forceinline__ void cpa16(void* sdst, const void* g) {   // 16 B, L2 only
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((unsigned)__cvta_generic_to_shared(sdst)), "l"(g) : "memory"); }
__device__ __forceinline__ void cpa8(void* sdst, const void* g) {    // 8 B (V-cache copy)
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"((unsigned)__cvta_generic_to_shared(sdst)), "l"(g) : "memory"); }

// Block-cooperative GEMV. A block owns rows [r0, r0+n); each row is split into
// 256-element segments ("units", one uint4 per lane). Warp w handles units
// w, w+NW, ...; all of a warp's weight loads are issued before any use.
#define CDIV(a, b) (((a) + (b) - 1) / (b))
#define UMAX(rows, nseg) CDIV(CDIV(rows, NB) * (nseg), NW)
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define PART_N (NW * MAX2(UMAX(4096, 4), MAX2(UMAX(INTER, 8), UMAX(H, 12))))

__device__ __forceinline__ void row_range(int nrows, int& r0, int& n) {
    r0 = (int)(((long long)blockIdx.x * nrows) / gridDim.x);
    n = (int)(((long long)(blockIdx.x + 1) * nrows) / gridDim.x) - r0;
}

// F(u) -> pointer to the unit's 256-element bf16 weight segment.
template <int U, class F>
__device__ __forceinline__ void gemv_load(uint4 (&buf)[U], int nunits, F unit_ptr) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int k = 0; k < U; ++k) {
        int u = min(warp + k * NW, nunits - 1);  // tail warps re-load a valid unit (unused)
        buf[k] = __ldg((const uint4*)unit_ptr(u) + lane);
    }
}

// gemv_dot with the activations in registers: act[j][e] = logical element e of the warp's j-th segment
// (lane*8 .. +7 of it). Unit k uses act[k % NSEG]. All U units are computed unconditionally (tail warps redo a
// valid unit); the U per-lane sums are reduced together by a transpose butterfly (P-1+(5-log2 P) shuffles).
template <int U, int NSEG>
__device__ __forceinline__ void gemv_dot_reg(const uint4 (&buf)[U], int nunits, const float (&act)[NSEG][8],
                                             float* part) {
    constexpr int LG = U <= 1 ? 0 : U <= 2 ? 1 : U <= 4 ? 2 : U <= 8 ? 3 : 4;
    constexpr int P = 1 << LG;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float v[P];
#pragma unroll
    for (int k = 0; k < U; ++k) {
        const uint4 w = buf[k];
        const float* a = act[k % NSEG];
        v[k] = bflo(w.x) * a[0] + bfhi(w.x) * a[1] + bflo(w.y) * a[2] + bfhi(w.y) * a[3]
             + bflo(w.z) * a[4] + bfhi(w.z) * a[5] + bflo(w.w) * a[6] + bfhi(w.w) * a[7];
    }
#pragma unroll
    for (int k = U; k < P; ++k) v[k] = 0.f;
#pragma unroll
    for (int lv = 0; lv < LG; ++lv) {
        const int h = P >> (lv + 1), o = 16 >> lv;
        const bool hi = lane & o;
#pragma unroll
        for (int i = 0; i < h; ++i) {
            float send = hi ? v[i] : v[i + h], keep = hi ? v[i + h] : v[i];
            v[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
    }
#pragma unroll
    for (int o = 16 >> LG; o; o >>= 1) v[0] += __shfl_xor_sync(0xffffffffu, v[0], o);
    const int k = lane >> (5 - LG), u = warp + k * NW;
    if ((lane & ((32 >> LG) - 1)) == 0 && k < U && u < nunits) part[u] = v[0];
}

// r of a deferred RMSNorm from the NW per-warp partial sums of x^2 in red (read after the dot's __syncthreads)
__device__ __forceinline__ float rmsd_r(const float* red) {
    float ss = red[0];
#pragma unroll
    for (int i = 1; i < NW; ++i) ss += red[i];
    return rsqrtf(ss / (float)H + EPS);
}

// ---- attention over one chunk of CTX = 128 positions for ONE (token, q-head) ----
// Warp w owns the chunk's positions w*PPW .. +PPW-1: K of those positions in registers kk[j] (lane: dims lane*4..+3),
// V in sVc rows (w*PPW + j) (the lane's own dims, copied by its own cp.async). A transpose butterfly (as gemv_dot_reg)
// leaves the score of position (lane >> 2) in each lane. `nvalid` = number of positions of the CHUNK that are visible
// (>= CTX: all); the others get probability 0. Returns the warp's chunk state: m (warp-uniform max score, -inf when the warp
// has no visible position), lsum (sum of e^(s - m)), o (this lane's dims lane*4..+3 of sum e^(s - m) * v).
// NEWTOK (MT1 chunk 0, warp 0 only): the new token (k in sk, v in sv, smem) is folded in exactly as in hv2.
template <bool NEWTOK>
__device__ __forceinline__ void attn_chunk(const uint2 (&kk)[PPW], const float4 q, const int nvalid,
                                           const bf16_t* sVc, const float* sk, const float* sv,
                                           float& m_out, float& l_out, float4& o_out) {
    static_assert(PPW == 8, "transpose butterfly below assumes 8 positions per warp");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float v[PPW];
#pragma unroll
    for (int j = 0; j < PPW; ++j) {
        const uint2 kj = kk[j];
        v[j] = bflo(kj.x) * q.x + bfhi(kj.x) * q.y + bflo(kj.y) * q.z + bfhi(kj.y) * q.w;
    }
    // transpose butterfly (as gemv_dot): lane ends with the score of position lane >> 2
#pragma unroll
    for (int lv = 0; lv < 3; ++lv) {
        const int hh = PPW >> (lv + 1), o = 16 >> lv;
        const bool hi = lane & o;
#pragma unroll
        for (int i = 0; i < hh; ++i) {
            float send = hi ? v[i] : v[i + hh], keep = hi ? v[i + hh] : v[i];
            v[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
    }
    v[0] += __shfl_xor_sync(0xffffffffu, v[0], 2);
    v[0] += __shfl_xor_sync(0xffffffffu, v[0], 1);
    float sc = v[0] * ATTN_SCALE;
    const bool valid = (warp * PPW + (lane >> 2)) < nvalid;
    if (!valid) sc = -INFINITY;
    float m = warp_max(sc), snew = 0.f;
    if (NEWTOK && warp == 0) {  // new token (k, v from smem)
        const float4 k = *(const float4*)(sk + lane * 4);
        snew = warp_sum(k.x * q.x + k.y * q.y + k.z * q.z + k.w * q.w) * ATTN_SCALE;
        m = fmaxf(m, snew);
    }
    const float ee = valid ? R11_EXPF(sc - (m == -INFINITY ? 0.f : m)) : 0.f;
    float lsum = 0.f;
    float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
    asm volatile("cp.async.wait_all;" ::: "memory");
#pragma unroll
    for (int j = 0; j < PPW; ++j) {
        const uint2 vj = *(const uint2*)(sVc + (warp * PPW + j) * HD + lane * 4);
        const float e = __shfl_sync(0xffffffffu, ee, j * 4);
        lsum += e;
        o.x += e * bflo(vj.x); o.y += e * bfhi(vj.x);
        o.z += e * bflo(vj.y); o.w += e * bfhi(vj.y);
    }
    if (NEWTOK && warp == 0) {
        const float e = R11_EXPF(snew - m);
        const float4 nv = *(const float4*)(sv + lane * 4);
        lsum += e;
        o.x += e * nv.x; o.y += e * nv.y; o.z += e * nv.z; o.w += e * nv.w;
    }
    m_out = m; l_out = lsum; o_out = o;
}

// online-softmax merge of a chunk state (m2, l2, o2) into the running state (m, l, o)
__device__ __forceinline__ void attn_merge(float& m, float& l, float4& o, const float m2, const float l2, const float4 o2) {
    const float mn = fmaxf(m, m2);
    const float a = R11_EXPF(m - mn), b = R11_EXPF(m2 - mn);  // m is finite whenever a merge happens (chunk 0 of a multi-chunk context is full)
    l = l * a + l2 * b;
    o.x = o.x * a + o2.x * b; o.y = o.y * a + o2.y * b; o.z = o.z * a + o2.z * b; o.w = o.w * a + o2.w * b;
    m = mn;
}
