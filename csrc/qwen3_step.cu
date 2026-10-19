// Qwen3-0.6B decode step, batch 1, position 128, as one persistent cooperative kernel.
// Main ideas:
//  - One launch for the whole step. One 512-thread CTA per SM. Grid barriers separate the dependent phases.
//  - A grid barrier is a release-add on a counter plus a relaxed poll. The weight loads of the next phase are issued between arrive and wait.
//  - Activations stay in fp32. The GEMV keeps its activations in registers and sums the lane results with a transpose butterfly.
//  - Fused attention: one CTA per q-head slice computes the softmax over all 129 positions and a split-K O-projection that red.adds into x.
//  - The gate / up rows and the down rows of a layer are TMA bulk copies into dynamic shared memory, ahead of use.
//  - RMSNorm of the next GEMV input is deferred. The row sums are scaled after the dot, so one __syncthreads suffices.
//  - The LM head keeps two rows in flight per warp. The argmax is a per-CTA reduction plus a last-arriver reduction.
// %clock cycles that thread 0 waits before it polls a grid barrier (one value per barrier)
#define SPIN_CYC 490
#define SPIN_PRO SPIN_CYC
#define SPIN_QKV SPIN_CYC
#define SPIN_FUSED 0
#define SPIN_GU 735
#define SPIN_DOWN 245
// LM-head rows in flight per warp
#define LM_ROWS 2
#define STAGE_BYTES 5632  // gate/up: 11 units x 32 lanes x 16 B
#define H 1024
#define NL 28
#define QH 16
#define KVH 8
#define HD 128
#define INTER 3072
#define VOCAB 151936
#define CTX 128
#define NT 512
#define NW 16
#define EPS 1e-6f
#define NSL 9  // O-proj row slices: fused block b < QH * NSL owns q-head b % QH, slice b / QH
#define NQD (H / 4)  // Wo row quads: slices start and end on quads (16-byte aligned red.add.v4)
#define PPW (CTX / NW)  // cached positions per warp in the fused attention
#define WOU 4  // Wo uint4 per thread: slice rows (<= 116) x 16 uint4 <= WOU * NT
#define INFINITY __int_as_float(0x7f800000)

typedef unsigned short bf16_t;

// scratch layout (floats)
#define S_X 0
#define S_QKV (S_X + H)
#define S_HID (S_QKV + 4096)

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

__device__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float t = (l < NW) ? red[l] : 0.f;
    return warp_sum(t);
}
__device__ float block_max(float v, float* red) {
    v = warp_max(v);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float t = (l < NW) ? red[l] : -INFINITY;
    return warp_max(t);
}

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
__device__ __forceinline__ void grid_barrier(unsigned* cnt, unsigned& target) {
    bar_arrive(cnt);
    bar_wait<0>(cnt, target);
}

// Warp dot product of a bf16 row (length K, multiple of 256) with an fp32 smem vector.
template <int K>
__device__ __forceinline__ float row_dot(const bf16_t* __restrict__ w, const float* s) {
    const uint4* wr = (const uint4*)w;
    int lane = threadIdx.x & 31;
    float acc = 0.f;
#pragma unroll
    for (int it = 0; it < K / 256; ++it) {
        int idx = it * 32 + lane;
        uint4 u = __ldg(wr + idx);
        const float* xs = s + it * 256 + lane * 4;  // permuted layout (sperm)
        float4 a = *(const float4*)xs, b = *(const float4*)(xs + 128);
        acc += bflo(u.x) * a.x + bfhi(u.x) * a.y + bflo(u.y) * a.z + bfhi(u.y) * a.w
             + bflo(u.z) * b.x + bfhi(u.z) * b.y + bflo(u.w) * b.z + bfhi(u.w) * b.w;
    }
    return warp_sum(acc);
}

// N rows in flight per warp: rows r0 + j*step (j < N), valid when < lim. All N*4 loads are issued first (invalid rows read
// row 0 and are dropped), then N dots (same sum order as row_dot) and N warp_sums; lane 0 writes the logits and updates
// (best, besti) in ascending row order.
template <int N>
__device__ __forceinline__ void lm_group(const bf16_t* __restrict__ embed, int r0, int step, int lim, const float* s,
                                         float* logits, float& best, int& besti) {
    const int lane = threadIdx.x & 31;
    uint4 u[N][H / 256];
#pragma unroll
    for (int j = 0; j < N; ++j) {
        const int row = r0 + j * step;
        const uint4* wr = (const uint4*)(embed + (size_t)(row < lim ? row : 0) * H);
#pragma unroll
        for (int it = 0; it < H / 256; ++it) u[j][it] = __ldcs(wr + it * 32 + lane);
    }
#pragma unroll
    for (int j = 0; j < N; ++j) {
        float acc = 0.f;
#pragma unroll
        for (int it = 0; it < H / 256; ++it) {
            const float* xs = s + it * 256 + lane * 4;  // permuted layout (sperm)
            float4 a = *(const float4*)xs, b = *(const float4*)(xs + 128);
            const uint4 v = u[j][it];
            acc += bflo(v.x) * a.x + bfhi(v.x) * a.y + bflo(v.y) * a.z + bfhi(v.y) * a.w
                 + bflo(v.z) * b.x + bfhi(v.z) * b.y + bflo(v.w) * b.z + bfhi(v.w) * b.w;
        }
        const float d = warp_sum(acc);
        const int row = r0 + j * step;
        if (lane == 0 && row < lim) {
            logits[row] = d;
            if (d > best) { best = d; besti = row; }
        }
    }
}

template <int K>
__device__ __forceinline__ float row_dot_reg(const bf16_t* __restrict__ w, const float (&a)[K / 256][8]) {
    const uint4* wr = (const uint4*)w;
    int lane = threadIdx.x & 31;
    float acc = 0.f;
#pragma unroll
    for (int it = 0; it < K / 256; ++it) {
        int idx = it * 32 + lane;
        uint4 u = __ldg(wr + idx);
        acc += bflo(u.x) * a[it][0] + bfhi(u.x) * a[it][1] + bflo(u.y) * a[it][2] + bfhi(u.y) * a[it][3]
             + bflo(u.z) * a[it][4] + bfhi(u.z) * a[it][5] + bflo(u.w) * a[it][6] + bfhi(u.w) * a[it][7];
    }
    return warp_sum(acc);
}

// a[j] = x[j] * g[j] for the lane's 8 logical elements (xp, gp point at element lane*8 of the segment);
// returns the lane's sum of x[j]^2. x is read with ld.cg (written by other CTAs).
__device__ __forceinline__ float norm_act(const float* xp, const bf16_t* __restrict__ gp, float (&a)[8]) {
    const float4 x0 = __ldcg((const float4*)xp), x1 = __ldcg((const float4*)(xp + 4));
    const uint4 g = __ldg((const uint4*)gp);
    const float ss = x0.x * x0.x + x0.y * x0.y + x0.z * x0.z + x0.w * x0.w
                   + x1.x * x1.x + x1.y * x1.y + x1.z * x1.z + x1.w * x1.w;
    a[0] = x0.x * bflo(g.x); a[1] = x0.y * bfhi(g.x); a[2] = x0.z * bflo(g.y); a[3] = x0.w * bfhi(g.y);
    a[4] = x1.x * bflo(g.z); a[5] = x1.y * bfhi(g.z); a[6] = x1.z * bflo(g.w); a[7] = x1.w * bfhi(g.w);
    return ss;
}
// a[j] = the lane's 8 logical elements of fp32 vector xp (no norm).
__device__ __forceinline__ void load_act(const float* xp, float (&a)[8]) {
    const float4 x0 = __ldcg((const float4*)xp), x1 = __ldcg((const float4*)(xp + 4));
    a[0] = x0.x; a[1] = x0.y; a[2] = x0.z; a[3] = x0.w; a[4] = x1.x; a[5] = x1.y; a[6] = x1.z; a[7] = x1.w;
}
// same from a smem vector in the sperm layout, segment seg
__device__ __forceinline__ void smem_act(const float* s, int seg, float (&a)[8]) {
    const float* xs = s + seg * 256 + (threadIdx.x & 31) * 4;
    const float4 x0 = *(const float4*)xs, x1 = *(const float4*)(xs + 128);
    a[0] = x0.x; a[1] = x0.y; a[2] = x0.z; a[3] = x0.w; a[4] = x1.x; a[5] = x1.y; a[6] = x1.z; a[7] = x1.w;
}

extern __shared__ __align__(16) uint4 stage[];  // dynamic smem: warp 0's window loads, issued by warps 1-15
__device__ __forceinline__ void cpa16(void* sdst, const void* g) {   // 16 B, L2 only
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((unsigned)__cvta_generic_to_shared(sdst)), "l"(g) : "memory"); }
__device__ __forceinline__ void cpa8(void* sdst, const void* g) {    // 8 B
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"((unsigned)__cvta_generic_to_shared(sdst)), "l"(g) : "memory"); }

template <bool WAIT = false>
__device__ void rms_to_smem(const float* x, const bf16_t* __restrict__ w, float* s, float* red) {
    int t = threadIdx.x;
    float a = __ldcg(x + t), b = __ldcg(x + t + NT);
    float ss = block_sum(a * a + b * b, red);
    float r = rsqrtf(ss / (float)H + EPS);
    s[sperm(t)] = a * r * bf2f(w[t]);
    s[sperm(t + NT)] = b * r * bf2f(w[t + NT]);
    if (WAIT) asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
}

template <bool WAIT = false>
__device__ void rmsd_to_smem(const float* x, const bf16_t* __restrict__ w, float* s, float* red) {
    int t = threadIdx.x;
    float a = __ldcg(x + t), b = __ldcg(x + t + NT);
    const float w0 = bf2f(w[t]), w1 = bf2f(w[t + NT]);  // issued before any use of a / b
    s[sperm(t)] = a * w0;
    s[sperm(t + NT)] = b * w1;
    float v = warp_sum(a * a + b * b);
    if ((t & 31) == 0) red[t >> 5] = v;
    if (WAIT) asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
}
// r of the deferred norm from the NW per-warp partials in red (read after the dot's __syncthreads)
__device__ __forceinline__ float rmsd_r(const float* red) {
    float ss = red[0];
#pragma unroll
    for (int i = 1; i < NW; ++i) ss += red[i];
    return rsqrtf(ss / (float)H + EPS);
}

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

template <int U, class F>
__device__ __forceinline__ void gemv_load_lf(uint4 (&buf)[U], int nunits, F unit_ptr) {
    static_assert(U * 32 * 16 <= STAGE_BYTES, "stage fits");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp != 0) {
#pragma unroll
        for (int k = 0; k < U; ++k) {
            int u = min(warp + k * NW, nunits - 1);
            buf[k] = __ldg((const uint4*)unit_ptr(u) + lane);
        }
        const int t = threadIdx.x - 32;
        static_assert(U * 32 <= NT - 32, "one stage entry per helper thread");
        if (t < U * 32) {
            const int k = t >> 5, ln = t & 31, u = min(k * NW, nunits - 1);
            cpa16(&stage[t], (const uint4*)unit_ptr(u) + ln);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    } else {  // warp 0: defined (overwritten by stage_to_buf); without this ptxas spills 320 B
#pragma unroll
        for (int k = 0; k < U; ++k) buf[k] = make_uint4(0, 0, 0, 0);
    }
}
template <int U>
__device__ __forceinline__ void stage_to_buf(uint4 (&buf)[U]) {
    if ((threadIdx.x >> 5) == 0) {
#pragma unroll
        for (int k = 0; k < U; ++k) buf[k] = stage[k * 32 + (threadIdx.x & 31)];
    }
}

// part[u] = dot(unit u weights, s[voff(u) ...]) (s in sperm layout). All U units
// are computed unconditionally (tail warps redo a valid unit); the U per-lane
// sums are reduced together by a transpose butterfly: each level halves the
// values a lane holds, so the warp needs P-1+(5-log2 P) shuffles instead of 5U.
template <int U, class V>
__device__ __forceinline__ void gemv_dot(const uint4 (&buf)[U], int nunits, V voff, const float* s,
                                         float* part) {
    constexpr int LG = U <= 1 ? 0 : U <= 2 ? 1 : U <= 4 ? 2 : U <= 8 ? 3 : 4;
    constexpr int P = 1 << LG;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float v[P];
#pragma unroll
    for (int k = 0; k < U; ++k) {
        const int u = min(warp + k * NW, nunits - 1);
        const uint4 w = buf[k];
        const float* xs = s + voff(u) + lane * 4;
        float4 a = *(const float4*)xs, b = *(const float4*)(xs + 128);
        v[k] = bflo(w.x) * a.x + bfhi(w.x) * a.y + bflo(w.y) * a.z + bfhi(w.y) * a.w
             + bflo(w.z) * b.x + bfhi(w.z) * b.y + bflo(w.w) * b.z + bfhi(w.w) * b.w;
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
    // lane holds the full sum of value k = lane >> (5 - LG)
    const int k = lane >> (5 - LG), u = warp + k * NW;
    if ((lane & ((32 >> LG) - 1)) == 0 && k < U && u < nunits) part[u] = v[0];
}

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

extern "C" __global__ void __launch_bounds__(NT, 1) qwen3_step(
    const long long* __restrict__ token,
    const bf16_t* __restrict__ ln1, const bf16_t* __restrict__ wq,
    const bf16_t* __restrict__ wk, const bf16_t* __restrict__ wv,
    const bf16_t* __restrict__ qn, const bf16_t* __restrict__ kn,
    const bf16_t* __restrict__ wo, const bf16_t* __restrict__ ln2,
    const bf16_t* __restrict__ wg, const bf16_t* __restrict__ wu,
    const bf16_t* __restrict__ wd, const bf16_t* __restrict__ fnorm,
    const bf16_t* __restrict__ embed, const bf16_t* __restrict__ kcache,
    const bf16_t* __restrict__ vcache,
    float* __restrict__ logits, long long* __restrict__ next_token,
    bf16_t* __restrict__ k_write, bf16_t* __restrict__ v_write,
    float* scratch, unsigned* sync, float* pval, int* pidx, const float* __restrict__ rope)
{
    __shared__ __align__(16) float sbuf[INTER];
    __shared__ float red[32];
    __shared__ int redi[32];
    __shared__ float part[PART_N];
    __shared__ __align__(16) bf16_t sVc[CTX * HD];  // fused attention: V-cache rows of kv-head fg
    __shared__ __align__(8) unsigned long long gu_mbar;  // completes (phase l) when layer l's gate / up rows have landed in sGU
    __shared__ __align__(8) unsigned long long d_mbar;   // completes (phase l) when layer l's down rows have landed in sD

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int gwarp = blockIdx.x * NW + warp;
    const int nwarps = gridDim.x * NW;
    unsigned* bar_cnt = sync;
    unsigned bar_target = 0;
    unsigned* ticket = sync + 2;

    float* x = scratch + S_X;
    float* qkv = scratch + S_QKV;
    float* hid = scratch + S_HID;

    // Per-phase row ownership and weight register buffers. Each phase's weights
    // are issued between the arrival and the wait of the grid barrier that
    // precedes the phase, so their DRAM round trip overlaps barrier latency.
    int rq0, nq, rh0, nh, ri0, ni;
    row_range(4096, rq0, nq);
    row_range(H, rh0, nh);
    row_range(INTER, ri0, ni);
    uint4 bq[UMAX(4096, 4)], bgu[UMAX(INTER, 8)], bd[UMAX(H, 12)];
    // fused attention + split-K O-proj ownership: q-head fh (kv-head fg), Wo rows [fr0, fr0+fn)
    const bool fused = blockIdx.x < QH * NSL;
    // recomputed from blockIdx at each use (as const ints held across the layer loop they spill)
#define fh ((int)blockIdx.x % QH)
#define fs ((int)blockIdx.x / QH)
#define fg (fh >> 1)
#define fr0 (fs * NQD / NSL * 4)
#define fn ((fs + 1) * NQD / NSL * 4 - fr0)
    static_assert((NQD + NSL - 1) / NSL * 4 * (HD / 8) <= WOU * NT, "Wo slice fits WOU uint4 per thread");
    static_assert((WOU - 1) * NT / 16 <= NQD / NSL * 4, "only the last Wo uint4 can fall past a slice");
    static_assert(PPW * NW == CTX, "positions split evenly over warps");

    auto load_qkv = [&](int l) {
        gemv_load_lf(bq, nq * 4, [&](int u) -> const bf16_t* {
            int r = rq0 + (u >> 2), sg = (u & 3) * 256;
            if (r < 2048) return wq + ((size_t)l * 2048 + r) * H + sg;
            if (r < 3072) return wk + ((size_t)l * 1024 + (r - 2048)) * H + sg;
            return wv + ((size_t)l * 1024 + (r - 3072)) * H + sg;
        });
    };

    static_assert(2 * CDIV(INTER, NB) * H * 2 <= 86016, "sGU fits the 86016 B of dynamic smem");
    bf16_t* const sGU = (bf16_t*)(stage + STAGE_BYTES / 16);
    auto issue_gue = [&](int ll) {
        if (tid == 128) {
            const unsigned mb = (unsigned)__cvta_generic_to_shared(&gu_mbar);
            const unsigned dst_gate = (unsigned)__cvta_generic_to_shared(sGU);
            const unsigned dst_up = (unsigned)__cvta_generic_to_shared(sGU + CDIV(INTER, NB) * H);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(mb), "r"(2 * ni * H * 2) : "memory");
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                         :: "r"(dst_gate), "l"(wg + ((size_t)ll * INTER + ri0) * H), "r"(ni * H * 2), "r"(mb) : "memory");
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                         :: "r"(dst_up), "l"(wu + ((size_t)ll * INTER + ri0) * H), "r"(ni * H * 2), "r"(mb) : "memory");
        }
    };
    // GU phase, every thread, just before the dot: wait for layer l's copy, then bgu[k] from sGU with the unit mapping of gemv_load_lf
    // (u = min(warp + k * NW, ni * 8 - 1): row u >> 3, matrix (u >> 2) & 1, segment u & 3; the lane's uint4 at + lane). Warp 0 does the same (no stage for GU).
    // The tail clamp reads valid rows only; rows past ni in sGU are never read.
    auto gu_fill = [&](int ll) {
        unsigned ok;
        do {
            asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                         : "=r"(ok) : "r"((unsigned)__cvta_generic_to_shared(&gu_mbar)), "r"(ll & 1) : "memory");
        } while (!ok);
#pragma unroll
        for (int k = 0; k < UMAX(INTER, 8); ++k) {
            const int u = min(warp + k * NW, ni * 8 - 1);
            bgu[k] = ((const uint4*)(sGU + ((u >> 2) & 1) * (CDIV(INTER, NB) * H) + (size_t)(u >> 3) * H + (u & 3) * 256))[lane];
        }
    };
    static_assert(CDIV(H, NB) * INTER * 2 <= 43008, "sD fits 43008 B of dynamic smem");
    bf16_t* const sD = (bf16_t*)(stage + (STAGE_BYTES + 86016) / 16);
    auto issue_dne = [&](int ll) {
        if (tid == 128) {
            const unsigned mb = (unsigned)__cvta_generic_to_shared(&d_mbar);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(mb), "r"(nh * INTER * 2) : "memory");
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                         :: "r"((unsigned)__cvta_generic_to_shared(sD)), "l"(wd + ((size_t)ll * H + rh0) * INTER), "r"(nh * INTER * 2), "r"(mb) : "memory");
        }
    };
    // down phase, every thread, just before the dot: wait, then bd[k] from sD with the unit mapping of the down gemv_load_lf call
    // (u = min(warp + k * NW, nh * 12 - 1): row u / 12, segment u % 12; the lane's uint4 at + lane). Warp 0 fills its own units (no stage).
    auto d_fill = [&](int ll) {
        unsigned ok;
        do {
            asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                         : "=r"(ok) : "r"((unsigned)__cvta_generic_to_shared(&d_mbar)), "r"(ll & 1) : "memory");
        } while (!ok);
#pragma unroll
        for (int k = 0; k < UMAX(H, 12); ++k) {
            const int u = min(warp + k * NW, nh * 12 - 1);
            bd[k] = ((const uint4*)(sD + (size_t)(u / 12) * INTER + (u % 12) * 256))[lane];
        }
    };

    __shared__ float s_rope[2 * 64];  // rope cos / sin, constant over layers; ordered by the prologue bar_arrive
    if (tid < 2 * 64) s_rope[tid] = rope[tid];
    auto pf_row = [&](const bf16_t* p) {
        if (warp == 1 && lane < 16) asm volatile("prefetch.global.L2 [%0];" :: "l"(p + lane * 64));
    };
    auto pf_qkn = [&](int l) {
        if (warp == 1 && lane >= 16 && lane < 20) {
            const bf16_t* p = (lane < 18 ? qn : kn) + l * HD + (lane & 1) * 64;
            asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
        }
    };

    // x = embed[token]
    {
        long long tok = token[0];
        const bf16_t* e = embed + tok * H;
        for (int i = blockIdx.x * NT + tid; i < H; i += gridDim.x * NT)
            __stcg(x + i, bf2f(e[i]));
    }
    if (tid == 0) {  // before the prologue bar_arrive, whose __syncthreads makes the init visible to every thread
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"((unsigned)__cvta_generic_to_shared(&gu_mbar)) : "memory");
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"((unsigned)__cvta_generic_to_shared(&d_mbar)) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    bar_arrive(bar_cnt);
    load_qkv(0);
    pf_row(ln1);
    pf_qkn(0);
    bar_wait<SPIN_PRO>(bar_cnt, bar_target);

    for (int l = 0; l < NL; ++l) {
        float act[1][8];  // logical elements lane*8..+7 of segment warp & 3 (u & 3 == warp & 3 as NW = 16)
        rmsd_to_smem<true>(x, ln1 + l * H, sbuf, red);  // waits for the cp.async group of the QKV window
        stage_to_buf(bq);
        smem_act(sbuf, warp & 3, act[0]);
        gemv_dot_reg(bq, nq * 4, act, part);
        __syncthreads();
        if (tid < nq) {  // red[0 .. NW) is read here, before this CTA's bar_arrive; its next writer is after that barrier (rmsd_to_smem)
            const float r = rmsd_r(red);
            __stcg(qkv + rq0 + tid, r * (part[tid * 4] + part[tid * 4 + 1] + part[tid * 4 + 2] + part[tid * 4 + 3]));
        }
        bar_arrive(bar_cnt);
        // defined on every path: a fused-only definition keeps them live across the layer loop
        uint4 bw[WOU] = {};
        uint2 kk[PPW] = {};
        if (fused && warp != 0) {  // group A: warp 0's Wo / K / V entries (virtual thread vt = lane of warp 0)
            static_assert(WOU * 32 * 16 + PPW * 32 * 8 <= STAGE_BYTES, "stage fits");
            for (int e = tid - 32; e < (WOU + 2 * PPW) * 32; e += NT - 32) {
                if (e < WOU * 32) {
                    const int k = e >> 5, vt = e & 31;
                    const bf16_t* wb0 = wo + ((size_t)l * H + fr0 + (vt >> 4)) * 2048 + fh * HD + (vt & 15) * 8;
                    cpa16(&stage[e], (const uint4*)(k < WOU - 1 || (vt >> 4) + k * (NT / 16) < fn ? wb0 + k * (NT / 16) * 2048 : wb0));
                } else if (e < (WOU + PPW) * 32) {
                    const int e2 = e - WOU * 32, j = e2 >> 5, vt = e2 & 31;
                    const size_t off0 = (((size_t)l * CTX) * KVH + fg) * HD + vt * 4;
                    cpa8(&((uint2*)(stage + WOU * 32))[e2], kcache + off0 + j * KVH * HD);
                } else {
                    const int e2 = e - (WOU + PPW) * 32, j = e2 >> 5, vt = e2 & 31;
                    const size_t off0 = (((size_t)l * CTX) * KVH + fg) * HD + vt * 4;
                    cpa8((char*)sVc + (vt * 4 + j * HD) * 2, vcache + off0 + j * KVH * HD);
                }
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        if (fused && warp != 0) {  // own Wo slice, K and V-cache loads (group B)
            const bf16_t* wb = wo + ((size_t)l * H + fr0 + (tid >> 4)) * 2048 + fh * HD + (tid & 15) * 8;
#pragma unroll
            for (int k = 0; k < WOU; ++k)
                bw[k] = __ldg((const uint4*)(k < WOU - 1 || (tid >> 4) + k * (NT / 16) < fn ? wb + k * (NT / 16) * 2048 : wb));
            // warp w's cached positions w*PPW..+PPW-1: K into registers, V into smem by cp.async
            // (each lane copies, and later reads, only its own dims lane*4..+3)
            const size_t off = (((size_t)l * CTX + warp * PPW) * KVH + fg) * HD + lane * 4;
            const unsigned dst = (unsigned)__cvta_generic_to_shared(sVc + warp * PPW * HD + lane * 4);
#pragma unroll
            for (int j = 0; j < PPW; ++j) {
                kk[j] = __ldg((const uint2*)(kcache + off + j * KVH * HD));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(dst + j * HD * 2),
                             "l"(vcache + off + j * KVH * HD) : "memory");
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        uint2 nwq = make_uint2(0, 0);
        if (fused && warp < 2) nwq = __ldg((const uint2*)((warp == 0 ? qn : kn) + l * HD + lane * 4));
        bar_wait<SPIN_QKV>(bar_cnt, bar_target);
        if (!fused) issue_gue(l);  // layer l's gate / up rows: non-fused CTAs issue at the start of the phase

        // ---- Phase 2: fused attention (q-head fh, all 129 positions, exact in-block
        // softmax) + split-K O-proj of Wo rows [fr0, fr0+fn) x head-fh columns. Each
        // block red.adds its partial row sums into x; every block read x for ln1
        // before the barrier above, and the barrier below orders the adds before ln2. ----
        if (fused) {
            float* sq = sbuf;            // 128: post-norm, post-RoPE q of head fh
            float* sk = sbuf + HD;       // 128: new k
            float* sv = sbuf + 2 * HD;   // 128: new v
            float* soh = sbuf + 3 * HD;  // 128: normalized o of head fh
            float* sow = sbuf + 4 * HD;  // NW x 128 per-warp unnormalized o
            static_assert(4 * HD + NW * HD <= INTER, "attention smem");
            const bool wr = (fh & 1) == 0 && fs == 0;  // writes k_write/v_write of kv-head fg
            // prep roles: warp 0: q head fh, warp 1: k of kv-head fg, warp 2: v copy
            if (warp < 2) {
                // q head fh / k of kv-head fg. lane holds dims lane*4..+3
                const float* src = warp == 0 ? qkv + fh * HD : qkv + 2048 + fg * HD;
                float4 a = __ldcg((const float4*)src + lane);
                float r = rsqrtf(warp_sum(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w) / (float)HD + EPS);
                float e[4] = {a.x, a.y, a.z, a.w}, o[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    e[i] = e[i] * r * bf2f((i < 2 ? nwq.x : nwq.y) >> ((i & 1) * 16) & 0xffffu);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float pr = __shfl_xor_sync(0xffffffffu, e[i], 16);
                    int j = (lane * 4 + i) & 63;
                    float cs = s_rope[j], sn = s_rope[64 + j];
                    o[i] = e[i] * cs + (lane < 16 ? -pr : pr) * sn;
                }
                *(float4*)((warp == 0 ? sq : sk) + lane * 4) = make_float4(o[0], o[1], o[2], o[3]);
                if (warp == 1 && wr) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) k_write[(l * KVH + fg) * HD + lane * 4 + i] = f2bf(o[i]);
                }
            } else if (warp == 2) {
                float4 a = __ldcg((const float4*)(qkv + 3072 + fg * HD) + lane);
                *(float4*)(sv + lane * 4) = a;
                if (wr) {
                    v_write[(l * KVH + fg) * HD + lane * 4 + 0] = f2bf(a.x);
                    v_write[(l * KVH + fg) * HD + lane * 4 + 1] = f2bf(a.y);
                    v_write[(l * KVH + fg) * HD + lane * 4 + 2] = f2bf(a.z);
                    v_write[(l * KVH + fg) * HD + lane * 4 + 3] = f2bf(a.w);
                }
            }
            asm volatile("cp.async.wait_group 1;" ::: "memory");
            __syncthreads();
            issue_gue(l);  // fused CTAs: right after the prep __syncthreads (the fused phase's own qkv read has arrived)
            if (warp == 0) {
#pragma unroll
                for (int k = 0; k < WOU; ++k) bw[k] = stage[k * 32 + lane];
#pragma unroll
                for (int j = 0; j < PPW; ++j) kk[j] = ((const uint2*)(stage + WOU * 32))[j * 32 + lane];
            }
            // warp w: cached positions w*PPW..+PPW-1 (K/V in registers); warp 0 also the new token
            {
                const float4 q = *(const float4*)(sq + lane * 4);
                static_assert(PPW == 8, "transpose butterfly below assumes 8 positions per warp");
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
                const float sc = v[0] * 0.08838834764831845f;
                float m = warp_max(sc), snew = 0.f;
                if (warp == 0) {  // new token (k, v from smem)
                    const float4 k = *(const float4*)(sk + lane * 4);
                    snew = warp_sum(k.x * q.x + k.y * q.y + k.z * q.z + k.w * q.w) * 0.08838834764831845f;
                    m = fmaxf(m, snew);
                }
                const float ee = __expf(sc - m);
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
                if (warp == 0) {
                    const float e = __expf(snew - m);
                    const float4 nv = *(const float4*)(sv + lane * 4);
                    lsum += e;
                    o.x += e * nv.x; o.y += e * nv.y; o.z += e * nv.z; o.w += e * nv.w;
                }
                *(float4*)(sow + warp * HD + lane * 4) = o;
                if (lane == 0) { red[warp] = m; red[NW + warp] = lsum; }
            }
            __syncthreads();
            if (tid < HD) {
                float M = red[0];
#pragma unroll
                for (int w = 1; w < NW; ++w) M = fmaxf(M, red[w]);
                float L = 0.f, o = 0.f;
#pragma unroll
                for (int w = 0; w < NW; ++w) {
                    const float c = __expf(red[w] - M);
                    L += c * red[NW + w];
                    o += c * sow[w * HD + tid];
                }
                soh[tid] = o / L;
            }
            __syncthreads();
            const float4 a = *(const float4*)(soh + (tid & 15) * 8), b = *(const float4*)(soh + (tid & 15) * 8 + 4);
            float* srow = sow;
#pragma unroll
            for (int k = 0; k < WOU; ++k) {
                const uint4 w = bw[k];
                float d = bflo(w.x) * a.x + bfhi(w.x) * a.y + bflo(w.y) * a.z + bfhi(w.y) * a.w
                        + bflo(w.z) * b.x + bfhi(w.z) * b.y + bflo(w.w) * b.z + bfhi(w.w) * b.w;
#pragma unroll
                for (int o = 8; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                if ((tid & 15) == 0) srow[(tid >> 4) + k * (NT / 16)] = d;
            }
            __syncthreads();
            const int q0 = ctaid_opaque() / QH * NQD / NSL, nq4 = (ctaid_opaque() / QH + 1) * NQD / NSL - q0;
            if (tid < nq4) {
                const float4 d = *(const float4*)(srow + tid * 4);
                asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(
                             x
                             + (q0 + tid) * 4),
                             "f"(d.x), "f"(d.y), "f"(d.z), "f"(d.w) : "memory");
            }
        }
        bar_arrive(bar_cnt);
        pf_row(ln2 + l * H);
        bar_wait<SPIN_FUSED>(bar_cnt, bar_target);

        rmsd_to_smem<false>(x, ln2 + l * H, sbuf, red);
        issue_dne(l);  // layer l's down rows: right after the GU norm's __syncthreads (x has arrived for every thread)
        smem_act(sbuf, warp & 3, act[0]);
        gu_fill(l);
        gemv_dot_reg(bgu, ni * 8, act, part);
        __syncthreads();
        if (tid < ni) {
            const float* pp = part + tid * 8;
            const float r = rmsd_r(red);  // red[0 .. NW) is read before this CTA's bar_arrive; the next writer (fused phase / next QKV norm) is after it
            float gt = r * (pp[0] + pp[1] + pp[2] + pp[3]);
            float up = r * (pp[4] + pp[5] + pp[6] + pp[7]);
            __stcg(hid + ri0 + tid, (gt / (1.f + expf(-gt))) * up);
        }
        bar_arrive(bar_cnt);
        static_assert(CDIV(H, NB) <= 32, "nh rows fit warp 1");
        float xres = 0.f;
        if (tid >= 32 && tid < 32 + nh) xres = __ldcg(x + rh0 + tid - 32);
        bar_wait<SPIN_GU>(bar_cnt, bar_target);

        // hid is 768 float4: issue both loads of a thread before any store
        {
            float4 h0 = __ldcg((const float4*)hid + tid);
            float4 h1 = tid < INTER / 4 - NT ? __ldcg((const float4*)hid + NT + tid) : make_float4(0.f, 0.f, 0.f, 0.f);
            *((float4*)sbuf + sperm4(tid)) = h0;
            if (tid < INTER / 4 - NT) *((float4*)sbuf + sperm4(NT + tid)) = h1;
        }
        __syncthreads();
        d_fill(l);  // just before the dot
        float act3[3][8];
#pragma unroll
        for (int j = 0; j < 3; ++j) smem_act(sbuf, (warp + 4 * j) % 12, act3[j]);
        gemv_dot_reg(bd, nh * 12, act3, part);
        __syncthreads();
        if (tid >= 32 && tid < 32 + nh) {  // warp 1 holds the old x rows (loaded in the window)
            float d = 0.f;
#pragma unroll
            for (int j = 0; j < 12; ++j) d += part[(tid - 32) * 12 + j];
            __stcg(x + rh0 + tid - 32, xres + d);
        }
        bar_arrive(bar_cnt);
        if (l + 1 < NL) load_qkv(l + 1);
        if (l + 1 < NL) { pf_row(ln1 + (l + 1) * H); pf_qkn(l + 1); }
        bar_wait<SPIN_DOWN>(bar_cnt, bar_target);
    }

    rms_to_smem(x, fnorm, sbuf, red);
    float best = -INFINITY;
    int besti = 0x7fffffff;
    for (int r = gwarp; r < VOCAB; r += nwarps * LM_ROWS)
        lm_group<LM_ROWS>(embed, r, nwarps, VOCAB, sbuf, logits, best, besti);
    __syncthreads();
    if (lane == 0) { red[warp] = best; redi[warp] = besti; }
    __syncthreads();
    if (tid == 0) {
        float bv = red[0]; int bi = redi[0];
        for (int w = 1; w < NW; ++w)
            if (red[w] > bv || (red[w] == bv && redi[w] < bi)) { bv = red[w]; bi = redi[w]; }
        __stcg(pval + blockIdx.x, bv);
        __stcg(pidx + blockIdx.x, bi);
        __threadfence();
        unsigned t = atomicAdd(ticket, 1u);
        redi[0] = (t == gridDim.x - 1);
    }
    __syncthreads();
    if (redi[0]) {
        __threadfence();
        float bv = -INFINITY; int bi = 0x7fffffff;
        for (int b = tid; b < (int)gridDim.x; b += NT) {
            const float v = __ldcg(pval + b); const int i = __ldcg(pidx + b);
            if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        __syncthreads();                       // every thread has read redi[0] before it is overwritten
        if (lane == 0) { red[warp] = bv; redi[warp] = bi; }
        __syncthreads();
        if (tid == 0) {
            float v0 = red[0]; int i0 = redi[0];
            for (int w = 1; w < NW; ++w)
                if (red[w] > v0 || (red[w] == v0 && redi[w] < i0)) { v0 = red[w]; i0 = redi[w]; }
            next_token[0] = (long long)i0;
            atomicExch(ticket, 0u);
            atomicExch(bar_cnt, 0u);  // every block has passed its last barrier poll
        }
    }
}
