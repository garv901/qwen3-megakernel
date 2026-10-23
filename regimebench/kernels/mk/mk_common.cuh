// mk common part: model constants, knobs, helpers. Device helpers copied from probe/xover/xover.cu (verified on GPU, job 9521) and
// kernels/hv2g/hv2g_common.cuh (grid barrier, bf16 helpers). The attention header (attn_placeholder.cuh or common/splitkv/splitkv.cuh)
// is prepended to this text by build.py. Compile with -DNPAD=<16|32|64|128|256> [-DNB=<SM count>] and the knobs below.
//
// Knobs (all default to the verified configuration; with MK_STAMPS = 0 and MK_NULL = 0 the SASS is that of the source without
// the knob code, see NOTES.md):
//   MK_KU     k per weight unit (128 or 64); a unit is 128 rows x MK_KU k of W, TMA boxes of 64 k.
//   MK_RING   weight ring slots (units in flight), 2..4.
//   MK_XTMA   0 only (reserved: O-phase B tile by TMA from ao; not implemented in v0).
//   MK_STAMPS 1: thread 0 stores %globaltimer at every bar_arrive and bar_wait exit into smem, one dump to mk_stamps at the end.
//   MK_NULL   bitmask, bit p-1 = phase p (1..6), bit 6 = LM head: the phase keeps its barrier(s), weight TMAs and ring waits and skips
//             staging, MMA, TMEM read and output (wrong outputs by design; null-work oracle).
#ifndef NPAD
#error "define NPAD (token bucket: 16, 32, 64, 128 or 256)"
#endif
#ifndef NB
#define NB 148   // grid = SM count (the arm passes the real count)
#endif
#ifndef MK_KU
#define MK_KU 128
#endif
#ifndef MK_RING
#define MK_RING 2
#endif
#ifndef MK_XTMA
#define MK_XTMA 0
#endif
#ifndef MK_STAMPS
#define MK_STAMPS 0
#endif
#ifndef MK_NULL
#define MK_NULL 0
#endif
#ifndef MK_DBG_EXIT
#define MK_DBG_EXIT 0   // debug: return after prologue stage n (1: smem/TMEM/mbarrier init, 2: + embed copy and zeroing)
#endif
#ifndef MK_EPI2
#define MK_EPI2 0   // TMEM read: 2 chunks of 16 columns per warp iteration (1) or one (0, default: 1 spills at NPAD 16..256)
#endif
#ifndef MK_STG
#define MK_STG 4   // staging chunks (16 B) in flight per thread
#endif
static_assert(NPAD == 16 || NPAD == 32 || NPAD == 64 || NPAD == 128 || NPAD == 256, "NPAD bucket");
static_assert(MK_KU == 128 || MK_KU == 64, "MK_KU");
static_assert(MK_RING >= 2 && MK_RING <= 4, "MK_RING");
static_assert(MK_XTMA == 0, "MK_XTMA = 1 (O-phase B tile by TMA) is not implemented in v0");

typedef unsigned int u32;
typedef unsigned long long ull;
typedef unsigned short bf16_t;

constexpr int NT = 512, NW = 16;
constexpr int HID = 1024, NLAY = 28, NQH = 16, NKVH = 8, HDIM = 128, INTER = 3072, VOCAB = 151936;
constexpr int QKVN = 4096, GUN = 6144, QDIM = 2048;
constexpr int MAXTOK = 256;
constexpr float EPS = 1e-6f;
constexpr int NMAP = 8;   // tensor maps: wq, wk, wv, wo, wg, wu, wd, embed (128 B each, in global memory)

constexpr int KU = MK_KU, NBOX = KU / 64, RING = MK_RING;
constexpr int UB = NBOX * 16384;              // bytes of one weight unit in the ring (128 rows x KU k bf16)
constexpr int XS = NBOX * NPAD * 128;         // bytes of one B-tile slot (NBOX k-atoms of NPAD rows x 128 B)
constexpr int ATTN_BYTES = SKV_SMEM_BYTES;    // attention scratch, overlaid on the B-tile slots (idle in phases 2 and 3)
constexpr int XREG = (2 * XS > ATTN_BYTES) ? 2 * XS : ATTN_BYTES;
constexpr int OFF_X = RING * UB;              // B-tile slots (offset from the 1024-aligned smem base)
constexpr int OFF_BARS = OFF_X + XREG;        // tma bars [0, RING), mma bars at +64, tmem slot at +128
constexpr int DYN_SMEM = 1024 + OFF_BARS + 256;
constexpr int N_BARRIERS = 1 + 6 * NLAY;      // grid barriers per launch
#if MK_STAMPS
constexpr int MK_NSTAMP = 2 * N_BARRIERS + 8; // arrive + wait-exit per barrier, then the end of the LM phase
#endif  // MK_STAMPS

constexpr bool mk_null(int p) { return ((MK_NULL >> p) & 1) != 0; }   // p = phase - 1 (0..5), 6 = LM

#define MK_CDIV(a, b) (((a) + (b) - 1) / (b))

// ---------------------------------------------------------------- bf16 helpers (hv2g_common.cuh)
__device__ __forceinline__ float bf2f(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bflo(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bfhi(unsigned u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ bf16_t f2bf(float f) {
    unsigned u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16_t)(u >> 16);
}
// two fp32 -> packed bf16x2 (lo = a, hi = b), round to nearest even: the form of cuda_bf16.hpp __floats2bfloat162_rn (sm_80+)
__device__ __forceinline__ u32 pack_bf16(float a, float b) {
    u32 r;
    asm("{ cvt.rn.bf16x2.f32 %0, %2, %1;}\n" : "=r"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
#define MK_SILU(g) ((g) / (1.f + expf(-(g))))   // hv2g R11_SILU

// ---------------------------------------------------------------- xover.cu helpers (verified, job 9521)
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ ull gt() { ull t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void mbar_init(u32 a, u32 cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(cnt) : "memory"); }
__device__ __forceinline__ void mbar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_wait(u32 a, u32 parity) {
    u32 ok;
    while (true) {
        asm volatile("{\n\t.reg .pred P;\n\tmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2;\n\tselp.u32 %0, 1, 0, P;\n\t}" : "=r"(ok) : "r"(a), "r"(parity) : "memory");
        if (ok) break;
    }
}
__device__ __forceinline__ void mbar_expect_tx(u32 a, u32 bytes) { asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(bytes) : "memory"); }
__device__ __forceinline__ void tma2(u32 dst, const void* map, int c0, int c1, u32 mbar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" :: "r"(dst), "l"(map), "r"(mbar), "r"(c0), "r"(c1) : "memory");
}
// tensor map in global memory (written by the host before the launch): one acquire of the tensormap proxy per map (form of dense_tc map_patch)
__device__ __forceinline__ void map_acquire(const void* slot) {
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"(slot) : "memory");
}
__device__ __forceinline__ void tmem_alloc(u32 smem_slot, u32 ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_slot), "r"(ncols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(u32 taddr, u32 ncols) { asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(ncols) : "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void tc_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__host__ __device__ constexpr u32 tc_idesc(int M, int N) { return (1u << 4) | (1u << 7) | (1u << 10) | ((u32)(N >> 3) << 17) | ((u32)(M >> 4) << 24); }
// UMMA smem descriptor: K-major, SWIZZLE_128B, SBO = 1024 B (8 rows x 128 B), LBO unused, sm100 version bit
__device__ __forceinline__ ull sdesc(u32 addr) { return (ull)((addr >> 4) & 0x3fff) | (1ull << 16) | (64ull << 32) | (1ull << 46) | (2ull << 61); }
__device__ __forceinline__ void tc_mma(u32 d_tmem, ull a_desc, ull b_desc, u32 idesc, u32 acc) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\ttcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}" :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(acc) : "memory");
}
__device__ __forceinline__ void tc_commit(u32 mbar) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mbar) : "memory"); }
// TMEM -> regs: 32 lanes (warp quadrant) x 16 fp32 columns. M = 128: TMEM lane = accumulator row (32 q + lane).
__device__ __forceinline__ void tmem_ld16(u32 (&v)[16], u32 ta) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]),
                   "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]) : "r"(ta));
}
__device__ __forceinline__ void red_add(float* p, float v) { asm volatile("red.global.add.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory"); }

// ---------------------------------------------------------------- grid barrier (hv2g_common.cuh) + optional stamps
// Monotonic arrival counter (requires co-residency: cooperative launch). Each CTA arrives once with a release reduction and thread 0
// relaxed-polls (then one acquire fence) until all CTAs of this epoch have arrived; the counter is zeroed at the end of the launch by
// the last argmax ticket holder.
#if MK_STAMPS
__device__ ull mk_stamps[NB * MK_NSTAMP];   // [cta * MK_NSTAMP + k]: %globaltimer (ns); even 2b = arrive of barrier b, 2b + 1 = wait exit, then LM end
__shared__ ull mk_sk_s[MK_NSTAMP];
__shared__ unsigned mk_sk_n;
__device__ __forceinline__ void mk_stamp() {   // thread 0 only
    const ull t = gt();
    const unsigned n = mk_sk_n;
    mk_sk_n = n + 1;
    if (n < MK_NSTAMP) mk_sk_s[n] = t;
}
#endif  // MK_STAMPS
__device__ __forceinline__ void bar_arrive(unsigned* cnt) {
    __syncthreads();
#if MK_STAMPS
    if (threadIdx.x == 0) mk_stamp();
#endif  // MK_STAMPS
    if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(cnt) : "memory");
}
__device__ __forceinline__ void bar_wait(const unsigned* cnt, unsigned& target) {
    target += gridDim.x;
    if (threadIdx.x == 0) {
        unsigned v;
        do {
            asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(cnt) : "memory");
        } while (v < target);
        asm volatile("fence.acquire.gpu;" ::: "memory");   // one acquire after the relaxed poll succeeds
#if MK_STAMPS
        mk_stamp();
#endif  // MK_STAMPS
    }
    __syncthreads();
}

// ---------------------------------------------------------------- engine wrappers (cta_group::1 today; a later MK_CG=2 build changes only these,
// geom() / mk_row_of() in mk_step.cu and the launch attribute in coop.py / build.py; the phase code never touches tcgen05 / TMA asm directly)
constexpr int MK_CG = 1;   // cta_group of the MMA engine (only 1 is implemented)
__device__ __forceinline__ void mk_tmem_alloc(u32 smem_slot) { tmem_alloc(smem_slot, 256); }          // whole warp
__device__ __forceinline__ void mk_tmem_dealloc(u32 taddr) { tmem_dealloc(taddr, 256); }              // whole warp
// issuing thread: all K steps of one weight unit (A = ring slot at sr, B = staged tile at xb), accumulate into TMEM column 0 (first: overwrite)
__device__ __forceinline__ void mk_mma_unit(u32 tb, u32 sr, u32 xb, u32 idesc, bool first) {
#pragma unroll
    for (int kk = 0; kk < KU / 16; ++kk) {
        const int a = kk >> 2, off = (kk & 3) * 32;
        tc_mma(tb, sdesc(sr + a * 16384 + off), sdesc(xb + a * (NPAD * 128) + off), idesc, (kk > 0 || !first) ? 1u : 0u);
    }
}
__device__ __forceinline__ void mk_commit(u32 mbar) { tc_commit(mbar); }
// warp quadrant q reads TMEM lanes 32 q.. : 16 token columns c0.. of the accumulator
__device__ __forceinline__ void mk_tmem_read16(u32 (&v)[16], u32 tb, int q, int c0) { tmem_ld16(v, tb + ((u32)(q * 32) << 16) + c0); }
// TMEM lane (quadrant q, lane) -> accumulator row within the 128-row block (M = 128: identity layout)
__device__ __forceinline__ int mk_row_of(int q, int lane) { return 32 * q + lane; }
// one weight unit's TMA: NBOX boxes of 64 k x 128 rows into ring slot s, completing on mbarrier bar
__device__ __forceinline__ void mk_tma_unit(u32 dst, const void* map, int k0, int row, u32 bar) {
#pragma unroll
    for (int a = 0; a < NBOX; ++a) tma2(dst + a * 16384, map, k0 + a * 64, row, bar);
}
