// xover2: tcgen05 GEMM phase  Y[T][N] += X[T][K] * W[N][K]^T  per Qwen3-0.6B shape, T = 1..256; tile / ingress / cta_group arms.
// Extension of probe/xover/xover.cu (ENG=1 tcgen05 engine and ENG=-1 oracle kept, FFMA engines dropped). Same skeleton: 148 CTAs x 512 thr,
// cooperative launch, nph phases per launch with a counter grid barrier, cold weights (fresh slice of a 2 GiB pool per phase, wrapped),
// swap-AB (A = weight unit, M rows; B = X tile, N = Npad tokens), D fp32 in TMEM, Y by red.global.add.f32 at row-block segment ends.
// Compile-time knobs (one binary per arm, see run.sh): KU (k per unit: 128 or 64), XTMA (0: all-thread LDG+STS X staging, 1: X by TMA from a
// [256][K] bf16 map whose outer dim is T, rows >= T zero-filled by TMA OOB), RING (weight ring slots, 0 = max that fits 227 KB, capped at the
// units one CTA owns), CG (1, or 2 = cluster of 2 CTAs, tcgen05.mma.cta_group::2 M = 256), ORACLE (1: same loads / barrier / zero red.add, no MMA).
// Weight unit: 128 rows x KU k per CTA (CG=2: 256 rows per pair, rank r holds rows 128r..128r+127; B = X, rank r loads tokens Npad/2*r ..).
// u = rb * (K/KU) + kc; CTA b (CG=2: pair b/2 of 74) owns units [b*U/148, (b+1)*U/148).
// Run: ./bin/x [nph=0 (0 = 112, LM 4)] [reps=5] [shape=QKV|O|GU|D|LM|all] [T=0 (all)]
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)
#ifndef KU
#define KU 128
#endif
#ifndef XTMA
#define XTMA 0
#endif
#ifndef RING
#define RING 2
#endif
#ifndef CG
#define CG 1
#endif
#ifndef ORACLE
#define ORACLE 0
#endif
#ifndef BIN
#define BIN "x"
#endif
static_assert(KU == 128 || KU == 64, "KU");
static_assert(CG == 1 || CG == 2, "CG");
static_assert(CG == 1 || XTMA == 1, "CG=2 requires XTMA=1");
static_assert(!(CG == 2 && ORACLE), "oracle is CG=1 only");
typedef unsigned int u32;
typedef unsigned long long ull;
typedef __nv_bfloat16 bf16;
constexpr int NT = 512, NW = 16, GRID = 148;
constexpr int NB = KU / 64;              // 64-k TMA boxes per unit
constexpr int UBW = 128 * KU * 2;        // weight bytes per CTA per unit (32768 / 16384)
constexpr int KCH = KU / 8;              // 16 B chunks per X row of a unit
constexpr int SMEM_MAX = 227 * 1024, SMEM_FIXED = 1024 + 2048;   // 1 KB alignment slack + 2 KB (barriers, tmem slot)
constexpr bool HAS_TC = !ORACLE;
__host__ __device__ constexpr int npad_of(int T) { return (T + 15) / 16 * 16 < (CG == 2 ? 32 : 16) ? (CG == 2 ? 32 : 16) : (T + 15) / 16 * 16; }   // N of the MMA

// ---- helpers copied from xover.cu (dense_tc tc_common.HELPERS_SRC / gemm_tc.cu, verified) ----
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ ull gt() { ull t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
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

// ---- cta_group::2 helpers: tma2d2 / mma_tc2 / commit2 VERBATIM from pi05 h_bench.cu (lines 32-45); alloc/free = tmem_alloc64c2/tmem_free64c2 with
// the column count made an operand (64 -> ncols; 256 needed for Npad up to 256) and cluster_sync_all from fg_k1.cu ----
__device__ __forceinline__ void tma2d2(uint32_t dst, const CUtensorMap* m, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               ::"r"(dst), "l"((unsigned long long)m), "r"(bar & 0xFEFFFFFFu), "r"(c0), "r"(c1) : "memory"); }
__device__ __forceinline__ void mma_tc2(uint32_t d, unsigned long long da, unsigned long long db, uint32_t id, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}"
               ::"r"(d), "l"(da), "l"(db), "r"(id), "r"(acc) : "memory"); }
__device__ __forceinline__ void commit2(uint32_t bar) { asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(bar), "h"((unsigned short)3) : "memory"); }
__device__ __forceinline__ void tmem_alloc_c2(uint32_t slot, uint32_t ncols) { asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(slot), "r"(ncols) : "memory"); }
__device__ __forceinline__ void tmem_free_c2(uint32_t tb, uint32_t ncols) { asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(tb), "r"(ncols) : "memory"); }
__device__ __forceinline__ void cluster_sync_all() { asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory"); }

__device__ __forceinline__ void tma_ld(u32 dst, const CUtensorMap* map, int c0, int c1, u32 mbar) {
  if (CG == 2) tma2d2(dst, map, mbar, c0, c1); else tma2(dst, map, c0, c1, mbar);
}

__global__ void __launch_bounds__(NT, 1) k_xover(const __grid_constant__ CUtensorMap mW, const __grid_constant__ CUtensorMap mX, const bf16* __restrict__ X,
                                                  float* Y, int N, int K, int T, int nph, int phase0, long long rowsAvail, int ring_arg, unsigned* cnt, ull* tm) {
  extern __shared__ uint8_t smraw[];
  const u32 raw = smem_u32(smraw), base = (raw + 1023) & ~1023u;
  uint8_t* sm = smraw + (base - raw);
  const int ring = RING != 0 ? RING : ring_arg;                        // compile-time slot count for the fixed-ring arms (as xover: mask, no div)
  const int Npad = npad_of(T), Rows = CG == 2 ? Npad / 2 : Npad;
  const u32 XSB = NB * Rows * 128;                                     // bytes per X slot (NB k-atoms of 64)
  const int nX = XTMA ? ring : 2;                                      // X slots (XTMA: one per weight slot)
  const u32 xbase = base + ring * UBW;
  const u32 bars = xbase + nX * XSB;                                   // full[s] at +8s, mma[s] at +256+8s, xfull[s] at +512+8s, tmem slot at +1024
  const u32 mbs = bars + 256, xbs = bars + 512;
  volatile u32* tslot = reinterpret_cast<volatile u32*>(sm + (bars - base) + 1024);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, q = warp & 3;
  const int rank = CG == 2 ? (blockIdx.x & 1) : 0;
  const ull c0 = clock64(), g0 = gt();
  if (HAS_TC && warp == 0) { if (CG == 2) tmem_alloc_c2(smem_u32((void*)tslot), 256); else tmem_alloc(smem_u32((void*)tslot), 256); }
  if (HAS_TC && CG == 2 && warp == 0) tc_fence_before();
  if (tid == 0) { for (int i = 0; i < ring; ++i) { mbar_init(bars + 8 * i, 1); mbar_init(mbs + 8 * i, 1); mbar_init(xbs + 8 * i, 1); } mbar_fence_init(); }
  if (!XTMA) for (int i = tid; i < (int)(nX * XSB) / 16; i += NT) reinterpret_cast<uint4*>(sm + (xbase - base))[i] = make_uint4(0, 0, 0, 0);   // pad rows stay zero
  fence_proxy_async();
  __syncthreads();
  if (CG == 2) cluster_sync_all();   // both CTAs' barriers initialised and TMEM allocated before any cross-CTA TMA / MMA traffic
  tc_fence_after();
  const u32 tb = HAS_TC ? *tslot : 0;
  const int KC = K / KU;
  int u0, u1;
  if (CG == 1) { const int U = (N / 128) * KC; u0 = blockIdx.x * U / GRID; u1 = (blockIdx.x + 1) * U / GRID; }
  else { const int U = ((N + 255) / 256) * KC, pp = blockIdx.x >> 1; u0 = pp * U / (GRID / 2); u1 = (pp + 1) * U / (GRID / 2); }
  const int n = u1 - u0;
  const u32 idesc = tc_idesc(CG == 2 ? 256 : 128, Npad);
  auto prow = [&](int p) { return (int)(((long long)(phase0 + p) * N) % rowsAvail); };
  // tid 0 (both ranks for CG=2): TMA fill number f (global counter, slot f % ring) of unit i of phase p.
  // fill_w: weight boxes on full[s] (independent of the previous phase, may run in the barrier window).
  // fill_x (XTMA only): X boxes on xfull[s]; X is the previous phase's output in the megakernel, so it is issued only after the grid barrier (or at an in-phase refill).
  auto fill_w = [&](int p, int i, u32 f) {
    const int u = u0 + i, kc = u % KC, s = f % ring;
    const int row = prow(p) + (u / KC) * (CG == 2 ? 256 : 128) + (CG == 2 ? 128 * rank : 0);
    const u32 full = bars + 8 * s;
    if (CG == 2) { if (rank == 0) mbar_expect_tx(full, 2 * UBW); }   // rank 0's barrier counts both CTAs' bytes (peer bit cleared in tma2d2)
    else mbar_expect_tx(full, UBW);
    for (int a = 0; a < NB; ++a) tma_ld(base + s * UBW + a * 16384, &mW, kc * KU + a * 64, row, full);
  };
  auto fill_x = [&](int i, u32 f) {
    if (!XTMA) return;
    const int u = u0 + i, kc = u % KC, s = f % ring;
    const u32 xfull = xbs + 8 * s;
    if (CG == 2) { if (rank == 0) mbar_expect_tx(xfull, 2 * XSB); }
    else mbar_expect_tx(xfull, XSB);
    for (int a = 0; a < NB; ++a) tma_ld(xbase + s * XSB + a * (Rows * 128), &mX, kc * KU + a * 64, CG == 2 ? rank * Rows : 0, xfull);
  };
  // all threads (XTMA=0): X columns of unit i -> X slot i&1, SW128 K-major
  auto stage = [&](int i) {
    const int kc = (u0 + i) % KC;
    uint8_t* xs = sm + (xbase - base) + (i & 1) * XSB;
#pragma unroll
    for (int h = 0; h < NB; ++h) {          // NB halves of 4 x 16 B per thread (register pressure: 128-reg cap at 512 thr)
      uint4 v[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) { const int c = tid + NT * (4 * h + j); if (c < T * KCH) v[j] = *reinterpret_cast<const uint4*>(X + (size_t)(c / KCH) * K + kc * KU + (c % KCH) * 8); }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int c = tid + NT * (4 * h + j), r = c / KCH, ch = c % KCH;
        if (c < T * KCH) *reinterpret_cast<uint4*>(xs + (ch >> 3) * (Npad * 128) + r * 128 + (((ch & 7) ^ (r & 7)) << 4)) = v[j];
      }
    }
    fence_proxy_async();
  };
  if (tid == 0) for (int i = 0; i < min(ring, n); ++i) { fill_w(0, i, i); fill_x(i, i); }
  u32 F = 0;                                                            // global unit counter (ring slot f % ring, mbarrier parity (f / ring) & 1)
  for (int p = 0; p < nph; ++p) {
    const u32 F0 = F;
    if (p > 0 && tid == 0) for (int i = 0; i < min(ring, n); ++i) fill_x(i, F0 + i);   // X of the prefetched units, after the grid barrier
    if (!XTMA && n > 0) stage(0);
    __syncthreads();
    for (int j = 0; j < n; ++j) {
      const u32 f = F0 + j, s = f % ring, par = (f / ring) & 1;
      const int u = u0 + j;
      const int n0 = (u / KC) * (CG == 2 ? 256 : 128) + (CG == 2 ? 128 * rank : 0);
      if (!ORACLE) {
        if (tid == 0 && (CG == 1 || rank == 0)) {
          mbar_wait(bars + 8 * s, par);
          if (XTMA) mbar_wait(xbs + 8 * s, par);
          tc_fence_after();
          const bool first = (j == 0) || (u % KC == 0);
          const u32 sr = base + s * UBW, xb = xbase + (XTMA ? s : (j & 1)) * XSB;
#pragma unroll
          for (int kk = 0; kk < KU / 16; ++kk) {
            const int a = kk >> 2, off = (kk & 3) * 32;
            const u32 acc = (kk > 0 || !first) ? 1u : 0u;
            if (CG == 2) mma_tc2(tb, sdesc(sr + a * 16384 + off), sdesc(xb + a * (Rows * 128) + off), idesc, acc);
            else tc_mma(tb, sdesc(sr + a * 16384 + off), sdesc(xb + a * (Rows * 128) + off), idesc, acc);
          }
          if (CG == 2) commit2(mbs + 8 * s); else tc_commit(mbs + 8 * s);
        }
        if (XTMA) {
          if (j >= 1 && j - 1 + ring < n && tid == 0) {      // MMA(j-1) done reading ring slot (f-1) % ring: refill it with unit j-1+ring
            mbar_wait(mbs + 8 * ((f - 1) % ring), ((f - 1) / ring) & 1);
            fill_w(p, j - 1 + ring, f - 1 + ring); fill_x(j - 1 + ring, f - 1 + ring);
          }
        } else if (j + 1 < n) {
          if (j >= 1) {      // MMA(j-1) done reading ring slot (f-1) % ring and X slot (j+1)&1
            mbar_wait(mbs + 8 * ((f - 1) % ring), ((f - 1) / ring) & 1);
            if (tid == 0 && j - 1 + ring < n) fill_w(p, j - 1 + ring, f - 1 + ring);   // (XTMA=0: no X TMA)
          }
          stage(j + 1);
        }
        if (j == n - 1 || (u + 1) % KC == 0) {   // row-block segment ends: TMEM -> Y
          mbar_wait(mbs + 8 * s, par);
          tc_fence_after();
          const int nch = (T + 15) / 16;
          for (int c = warp >> 2; c < nch; c += 8) {
            u32 v0[16], v1[16];
            const bool two = c + 4 < nch;
            tmem_ld16(v0, tb + ((u32)(q * 32) << 16) + c * 16);
            if (two) tmem_ld16(v1, tb + ((u32)(q * 32) << 16) + (c + 4) * 16);
            tc_wait_ld();
            const int nrow = n0 + 32 * q + lane;
            float* yp = Y + nrow;
            if (nrow < N) {
#pragma unroll
              for (int i = 0; i < 16; ++i) {
                if (c * 16 + i < T) red_add(yp + (size_t)(c * 16 + i) * N, __uint_as_float(v0[i]));
                if (two && (c + 4) * 16 + i < T) red_add(yp + (size_t)((c + 4) * 16 + i) * N, __uint_as_float(v1[i]));
              }
            }
          }
          tc_fence_before();
        }
        __syncthreads();
      } else {
        // oracle (xover ENG=-1): wait the unit's loads, red.add of zeros as the FFMA engine's per-unit pattern, no MMA / TMEM
        mbar_wait(bars + 8 * s, par);
        if (XTMA) mbar_wait(xbs + 8 * s, par);
        const int r = 8 * warp + (lane >> 2), qk = lane & 3;
        float* yp = Y + n0 + r;
        if (qk == 0) for (int t = 0; t < T; ++t) red_add(yp + (size_t)t * N, 0.f);
        if (!XTMA && j + 1 < n) stage(j + 1);
        __syncthreads();
        if (tid == 0 && j + ring < n) { fill_w(p, j + ring, f + ring); fill_x(j + ring, f + ring); }   // everyone finished reading ring slot s
      }
    }
    F = F0 + n;
    if (p + 1 < nph && tid == 0) for (int i = 0; i < min(ring, n); ++i) fill_w(p + 1, i, F + i);   // barrier-window prefetch: weights only
    __threadfence();
    gbar(cnt, p + 1);
  }
  __syncthreads();
  if (CG == 2) cluster_sync_all();   // no CTA frees TMEM / exits while its peer's MMAs or TMAs may still touch it
  tc_fence_after();
  if (HAS_TC && warp == 0) { if (CG == 2) tmem_free_c2(tb, 256); else tmem_dealloc(tb, 256); }
  if (tid == 0) { const ull c1 = clock64(), g1 = gt(); ull* o = tm + blockIdx.x * 4; o[0] = c0; o[1] = c1; o[2] = g0; o[3] = g1; }
}

__device__ __forceinline__ float hashf(u32 i) { u32 h = i * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16; return ((float)(h & 0xffff) - 32768.f) * (1.f / 32768.f); }   // [-1, 1)
__global__ void k_init(bf16* p, size_t n, u32 seed) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = __float2bfloat16(hashf((u32)i + seed)); }
// CUDA-core reference for phase 0 (pool rows 0..N-1), one thread per (t, n); counts violations of |d| <= 2e-3 * max(1, |ref|).
// sampled: only 4096 rows of N (the first 2048 and the last 2048), entry index = t * 4096 + idx.
__global__ void k_ref(const bf16* pool, const bf16* X, const float* Y, int N, int K, int T, int sampled, unsigned* maxerr, unsigned* nbad) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const int NS = sampled ? 4096 : N;
  if (i >= (size_t)T * NS) return;
  const int t = (int)(i / NS), idx = (int)(i % NS), n = sampled ? (idx < 2048 ? idx : N - 4096 + idx) : idx;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += __bfloat162float(X[(size_t)t * K + k]) * __bfloat162float(pool[(size_t)n * K + k]);
  const float d = fabsf(Y[(size_t)t * N + n] - acc);
  atomicMax(maxerr, __float_as_uint(d));
  if (!(d <= 2e-3f * fmaxf(1.f, fabsf(acc)))) atomicAdd(nbad, 1u);
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  const int nph_arg = argc > 1 ? atoi(argv[1]) : 0, reps = argc > 2 ? atoi(argv[2]) : 5;
  const char* shsel = argc > 3 ? argv[3] : "all";
  const int Tsel = argc > 4 ? atoi(argv[4]) : 0;
  int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  if (nsm != GRID) fprintf(stderr, "warning: %d SMs (expected %d)\n", nsm, GRID);
  const size_t POOL = 2ull << 30;
  bf16 *pool, *X; float* Y; unsigned *cnt, *ver; ull* tm;
  CK(cudaMalloc(&pool, POOL)); CK(cudaMalloc(&X, 256ull * 3072 * 2)); CK(cudaMalloc(&Y, 256ull * 151936 * 4)); CK(cudaMalloc(&cnt, 1024));
  CK(cudaMalloc(&ver, 64)); CK(cudaMalloc(&tm, GRID * 4 * 8));
  k_init<<<4096, 256>>>(pool, POOL / 2, 12345u); CK(cudaDeviceSynchronize());
  PFN_cuTensorMapEncodeTiled_v12000 enc; cudaDriverEntryPointQueryResult qr;
  CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&enc, 12000, cudaEnableDefault, &qr));
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, k_xover));
  const int regs = fa.numRegs, spill = (int)fa.localSizeBytes;   // local bytes (= spill when the source has no local arrays)
  CK(cudaFuncSetAttribute(k_xover, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
  struct Sh { const char* name; int N, K; } shapes[5] = {{"QKV", 4096, 1024}, {"O", 1024, 2048}, {"GU", 6144, 1024}, {"D", 1024, 3072}, {"LM", 151936, 1024}};
  const int Ts[9] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  for (auto& sh : shapes) {
    if (strcmp(shsel, "all") && strcmp(shsel, sh.name)) continue;
    const int N = sh.N, K = sh.K, KC = K / KU;
    const int nph = nph_arg ? nph_arg : (!strcmp(sh.name, "LM") ? 4 : 112);
    k_init<<<256, 256>>>(X, 256ull * K, 777u); CK(cudaDeviceSynchronize());
    const long long rows = (long long)(POOL / (2ull * K)), rowsAvail = rows - (long long)((N + 255) / 256) * 256;
    CUtensorMap mW;
    { cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)rows}, strides[1] = {(cuuint64_t)K * 2};
      cuuint32_t box[2] = {64, 128}, es[2] = {1, 1};
      CUresult r = enc(&mW, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, pool, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                       CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (r != CUDA_SUCCESS) { fprintf(stderr, "tensor map encode failed %d\n", (int)r); return 1; } }
    // units per CTA (max) for this binary's geometry
    const long long U = CG == 1 ? (long long)(N / 128) * KC : (long long)((N + 255) / 256) * KC;
    int upc = 0;
    if (CG == 1) for (int b = 0; b < GRID; ++b) upc = std::max(upc, (int)((b + 1) * U / GRID - b * U / GRID));
    else for (int b = 0; b < GRID / 2; ++b) upc = std::max(upc, (int)((b + 1) * U / (GRID / 2) - b * U / (GRID / 2)));
    for (int T : Ts) {
      if (Tsel && T != Tsel) continue;
      const int Npad = npad_of(T), Rows = CG == 2 ? Npad / 2 : Npad, XSB = NB * Rows * 128;
      const int nXs = XTMA ? 0 : 2 * XSB, slotB = UBW + (XTMA ? XSB : 0);
      const int fit = (SMEM_MAX - SMEM_FIXED - nXs) / slotB;
      const int ring = RING == 0 ? std::max(1, std::min(fit, upc)) : RING;
      const int smem = SMEM_FIXED + nXs + ring * slotB;
      if (smem > SMEM_MAX || ring > 32) { fprintf(stderr, "%s %s T=%d: ring %d needs %d B smem (> %d), skipped\n", BIN, sh.name, T, ring, smem, SMEM_MAX); continue; }
      CUtensorMap mX; memset(&mX, 0, sizeof(mX));
      if (XTMA) {
        cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)T}, strides[1] = {(cuuint64_t)K * 2};
        cuuint32_t box[2] = {64, (cuuint32_t)Rows}, es[2] = {1, 1};
        CUresult r = enc(&mX, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, X, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                         CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) { fprintf(stderr, "X tensor map encode failed %d\n", (int)r); return 1; }
      }
      cudaLaunchConfig_t lc = {}; lc.gridDim = dim3(GRID); lc.blockDim = dim3(NT); lc.dynamicSmemBytes = smem; lc.stream = 0;
      cudaLaunchAttribute at[2]; at[0].id = cudaLaunchAttributeCooperative; at[0].val.cooperative = 1;
      at[1].id = cudaLaunchAttributeClusterDimension; at[1].val.clusterDim.x = 2; at[1].val.clusterDim.y = 1; at[1].val.clusterDim.z = 1;
      lc.attrs = at; lc.numAttrs = CG == 2 ? 2 : 1;
      int occupancy = 0;   // CG=1: resident CTAs per SM (need >= 1); CG=2: max co-resident clusters of 2 (need 74)
      if (CG == 1) CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, k_xover, NT, smem));
      else CK(cudaOccupancyMaxActiveClusters(&occupancy, k_xover, &lc));
      if (occupancy < (CG == 1 ? 1 : GRID / 2)) { fprintf(stderr, "%s %s T=%d: occupancy %d too low (smem %d), skipped\n", BIN, sh.name, T, occupancy, smem); continue; }
      // floors: weight stream, X bytes of the phase (T*K*2) and X bytes staged into smem summed over CTAs (units on the grid x KU x T x 2)
      const double xph = (double)T * K * 2, xstaged = (double)U * KU * T * 2;
      printf("{\"floor\":1,\"bin\":\"%s\",\"shape\":\"%s\",\"T\":%d,\"us_weight_stream\":%.2f,\"x_bytes_phase\":%.0f,\"us_x_phase\":%.3f,\"x_bytes_staged\":%.0f,\"us_x_staged\":%.3f}\n",
             BIN, sh.name, T, (double)N * K * 2 / 6.5e12 * 1e6, xph, xph / 6.5e12 * 1e6, xstaged, xstaged / 6.5e12 * 1e6);
      auto launch = [&](int np, int p0) {
        CK(cudaMemset(cnt, 0, 1024));
        CK(cudaLaunchKernelEx(&lc, k_xover, mW, mX, (const bf16*)X, Y, N, K, T, np, p0, rowsAvail, ring, cnt, tm));
      };
      int pass = -1;   // -1 = not checked
      if (T == 1 || T == 8 || T == 64 || T == 256) {
        CK(cudaMemset(Y, 0, (size_t)T * N * 4)); CK(cudaMemset(ver, 0, 64));
        launch(1, 0); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        const int sampled = N > 8192;
        const size_t cntr = (size_t)T * (sampled ? 4096 : N);
        k_ref<<<(cntr + 255) / 256, 256>>>(pool, X, Y, N, K, T, sampled, ver, ver + 1); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        unsigned hv[2]; CK(cudaMemcpy(hv, ver, 8, cudaMemcpyDeviceToHost)); float me; memcpy(&me, &hv[0], 4);
        pass = ORACLE ? -1 : (hv[1] == 0);   // the oracle adds zeros: nothing to compare
        fprintf(stderr, "%s %s %s T=%d max|d|=%.3e bad=%u%s\n", ORACLE ? "SKIP" : (pass ? "PASS" : "FAIL"), BIN, sh.name, T, me, hv[1], sampled ? " (4096-row sample)" : "");
      }
      int p0 = 0;
      for (int w = 0; w < 2; ++w) { launch(nph, p0); p0 += nph; }
      CK(cudaDeviceSynchronize());
      cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
      std::vector<double> us, mhz;
      for (int r = 0; r < reps; ++r) {
        CK(cudaEventRecord(e0)); launch(nph, p0); p0 += nph; CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); us.push_back(ms * 1000.0 / nph);
        ull h[GRID * 4]; CK(cudaMemcpy(h, tm, sizeof(h), cudaMemcpyDeviceToHost));
        std::vector<double> m; for (int b = 0; b < GRID; ++b) m.push_back((double)(h[b * 4 + 1] - h[b * 4]) / (double)(h[b * 4 + 3] - h[b * 4 + 2]) * 1000.0);
        mhz.push_back(median(m));
      }
      printf("{\"bin\":\"%s\",\"shape\":\"%s\",\"T\":%d,\"us_phase\":%.3f,\"us_reps\":[", BIN, sh.name, T, median(us));
      for (size_t i = 0; i < us.size(); ++i) printf("%s%.3f", i ? "," : "", us[i]);
      printf("],\"mhz\":%.0f,\"pass\":%s,\"regs\":%d,\"spill\":%d,\"ring\":%d,\"units_per_cta_max\":%d,\"occupancy\":%d}\n", median(mhz),
             pass < 0 ? "null" : (pass ? "true" : "false"), regs, spill, ring, upc, occupancy);
      fflush(stdout);
      CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    }
  }
  return 0;
}
