// occfuse common device header. Helpers copied verbatim from probe/mk_core/xover4/xover4.cu (cta_group::1 only; verified in xover2 / xover4 / mk).
// Additions below the copied block: per-CTA phase stamps (mk MK_STAMPS definitions), clock sampling, hash init, L2 flush, spin.
#pragma once
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
typedef unsigned int u32;
typedef unsigned long long ull;
typedef __nv_bfloat16 bf16;
#ifndef GRID_N
#define GRID_N 148   // grid size = CTAs, one per SM (mk geometry)
#endif
constexpr int NT = 512, GRID = GRID_N, MAXRING = 16, TMEM_COLS = 512;
constexpr int SMEM_MAX = 227 * 1024, SMEM_FIXED = 1024 + 2048;   // 1 KB alignment slack + 2 KB (barriers, tmem slot)
constexpr int A_BYTES = 128 * 128 * 2;                           // one k-step (128 k) of a 128-row weight block: 2 SW128 boxes of 64 k x 128 rows

// ---- helpers copied from xover4.cu (lines 36-107) ----
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ ull gt() { ull t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void mbar_init(u32 a, u32 cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(cnt) : "memory"); }
__device__ __forceinline__ void mbar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_wait(u32 a, u32 parity) {
  u32 ok;
  while (true) {
#ifndef TW
#define TW 0
#endif
#if TW   // spin on the non-blocking test_wait (no try_wait suspend window)
    asm volatile("{\n\t.reg .pred P;\n\tmbarrier.test_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2;\n\tselp.u32 %0, 1, 0, P;\n\t}" : "=r"(ok) : "r"(a), "r"(parity) : "memory");
#else
    asm volatile("{\n\t.reg .pred P;\n\tmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2;\n\tselp.u32 %0, 1, 0, P;\n\t}" : "=r"(ok) : "r"(a), "r"(parity) : "memory");
#endif
    if (ok) break;
  }
}
__device__ __forceinline__ void mbar_arrive(u32 a) { asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(a) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(u32 a, u32 bytes) { asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(bytes) : "memory"); }
__device__ __forceinline__ void tma2(u32 dst, const void* map, int c0, int c1, u32 mbar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" :: "r"(dst), "l"(map), "r"(mbar), "r"(c0), "r"(c1) : "memory");
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
__host__ __device__ constexpr u32 tc_idesc(int M, int N) { return (1u << 4) | (1u << 7) | (1u << 10) | ((u32)(N >> 3) << 17) | ((u32)(M >> 4) << 24); }   // M = total M of the (pair) instruction
// UMMA smem descriptor: K-major, SWIZZLE_128B, SBO = 1024 B (8 rows x 128 B), LBO unused, sm100 version bit (A: tc_a_desc, B: gemm_tc sdesc)
__device__ __forceinline__ ull sdesc(u32 addr) { return (ull)((addr >> 4) & 0x3fff) | (1ull << 16) | (64ull << 32) | (1ull << 46) | (2ull << 61); }
__device__ __forceinline__ void tc_mma(u32 d_tmem, ull a_desc, ull b_desc, u32 idesc, u32 acc) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\ttcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}" :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(acc) : "memory");
}
__device__ __forceinline__ void tc_commit(u32 mbar) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mbar) : "memory"); }
// TMEM -> regs: 32 lanes (warp quadrant) x 16 fp32 columns. TMEM lane = accumulator row (M=128 per CTA; for cta_group::2 M=256 CTA r's own 128 rows, h_bench RPC=128 notes)
__device__ __forceinline__ void tmem_ld16(u32 (&v)[16], u32 ta) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]),
                 "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]) : "r"(ta));
}
__device__ __forceinline__ void gbar(unsigned* cnt, unsigned epoch) {   // gemm_tc.cu grid barrier
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned old;
    asm volatile("atom.add.release.gpu.global.u32 %0, [%1], 1;" : "=r"(old) : "l"(cnt) : "memory");
    while ((int)(ld_acquire(cnt) - epoch * gridDim.x) < 0) {}
  }
  __syncthreads();
}
__device__ __forceinline__ void red_add(float* p, float v) { asm volatile("red.global.add.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory"); }


// ---- additions (occfuse) ----
// Per-CTA phase stamps, same definitions as mk MK_STAMPS / probe/mk_core/smallt/analyze.py:
//   st[(cta * NSTMAX) + 2 * p]     = globaltimer when CTA arrives at barrier p (end of its work in phase p)
//   st[(cta * NSTMAX) + 2 * p + 1] = globaltimer when CTA leaves barrier p (start of phase p + 1)
// Barrier 0 is the start barrier (before phase 1). Host analysis: common/stamps.py.
#ifndef NSTMAX
#define NSTMAX 512   // stamps per CTA (2 per barrier): up to 255 phases per launch
#endif
__device__ __forceinline__ void stamp(ull* st, int idx) { if (st && threadIdx.x == 0 && idx < NSTMAX) st[blockIdx.x * NSTMAX + idx] = gt(); }
// Grid barrier with stamps: arrive stamp, gbar, exit stamp. epoch = 1-based barrier count since the counter was zeroed.
__device__ __forceinline__ void gbar_st(unsigned* cnt, unsigned epoch, ull* st, int b) {
  __syncthreads();
  if (threadIdx.x == 0) {
    stamp(st, 2 * b);
    unsigned old;
    asm volatile("atom.add.release.gpu.global.u32 %0, [%1], 1;" : "=r"(old) : "l"(cnt) : "memory");
    while ((int)(ld_acquire(cnt) - epoch * gridDim.x) < 0) {}
    stamp(st, 2 * b + 1);
  }
  __syncthreads();
}
// mk's split grid barrier (regimebench/kernels/mk/mk_common.cuh bar_arrive / bar_wait, with the stamps at the same points):
//   bar_arrive(cnt, st, b); <thread 0 may issue the next phase's prefetch here: the barrier window, as MK_PF_ARRIVE>; bar_wait(cnt, target, st, b);
// target starts at 0 (counter zeroed before the launch) and grows by gridDim.x per barrier. Use these (not gbar) in every phase bench.
__device__ __forceinline__ void bar_arrive(unsigned* cnt, ull* st, int b) {
  __syncthreads();
  if (threadIdx.x == 0) { stamp(st, 2 * b); asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(cnt) : "memory"); }
}
__device__ __forceinline__ void bar_wait(const unsigned* cnt, unsigned& target, ull* st, int b) {
  target += gridDim.x;
  if (threadIdx.x == 0) {
    unsigned v;
    do { asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(cnt) : "memory"); } while (v < target);
    asm volatile("fence.acquire.gpu;" ::: "memory");
    stamp(st, 2 * b + 1);
  }
  __syncthreads();
}
// Clock sample: clock64 and globaltimer at kernel start / end (thread 0 of each CTA) -> MHz on the host = dclk / dns * 1e3.
__device__ __forceinline__ void clk_sample(ull* ck, int which) { if (ck && threadIdx.x == 0) { ck[blockIdx.x * 4 + 2 * which] = clock64(); ck[blockIdx.x * 4 + 2 * which + 1] = gt(); } }
// Busy spin of about ns nanoseconds (globaltimer, 32 ns steps on B200; B0 job 14277).
__device__ __forceinline__ void spin_ns(ull ns) { const ull t0 = gt(); while (gt() - t0 < ns) {} }
__device__ __forceinline__ float hashf(u32 i) { u32 h = i * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16; return ((float)(h & 0xffff) - 32768.f) * (1.f / 32768.f); }   // [-1, 1)
__global__ void k_init_bf16(bf16* p, size_t n, u32 seed) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = __float2bfloat16(hashf((u32)i + seed)); }
__global__ void k_init_f32(float* p, size_t n, u32 seed) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = hashf((u32)i + seed); }
// Evict L2 between launches when a run must start cold: stream-write a buffer larger than L2.
__global__ void k_l2flush(uint4* b, size_t n, u32 v) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) b[i] = make_uint4(v, v, v, v); }
