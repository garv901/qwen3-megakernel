// xover6: (copy of xover4.cu, CG = 1 only, run-time grid size, + k_kv TMA KV stream and concurrent driver, SPEC-xover6)
// xover4: tile-stream tcgen05 GEMM engine "ts" with cta_group::2 pairs (-DCG=2) or cta_group::1 (-DCG=1) and ragged token tiles (SPEC-xover4).
// Copy of probe/mk_core/xover3/xover3.cu with the SB = 1 (staged-B) arm removed; helpers of cta_group::2 copied from probe/mk_core/xover2/xover2.cu.
// Tile (rb, tt, ks): W rows 128 CG rb.., tokens TT tt.. (NTT = ceil(T / TT), rows >= T are TMA OOB zero fill), k range ks*K/KS ..; t = (rb*NTT + tt)*KS + ks.
// CG = 1: CTA b owns [b*NTILE/148, (b+1)*NTILE/148).  CG = 2: pair p = blockIdx.x >> 1 (cluster of 2, 74 pairs) owns [p*NTILE/74, (p+1)*NTILE/74); both ranks run the same loops.
// Per k-step (128 k) one ring slot = A (128 x 128 bf16, 2 SW128 boxes 64x128, per CTA) + B (TT/CG rows x 128 bf16, 2 SW128 boxes 64 x TT/CG per CTA), full barrier count 2
// (arrival 1: A expect_tx, may be issued in the previous phase's barrier window; arrival 2: B expect_tx, issued after the grid barrier).
// CG = 2: rank r loads W rows 128 r.. of the 256-row pair block and tokens (TT/2) r..; all TMAs complete on rank 0's full[s]; the MMA (M = 256, N = TT) is issued by rank 0 tid 32,
// commits multicast to both CTAs' empty[s] / accf[b]; both ranks' epilogue warps read their own TMEM; rank 1's epilogue warps arrive remotely on rank 0's acce[b].
// Run: ./bin/ts{1,2} [nph=0 (112)] [reps=5] [shape=QKV|O|GU|D|all] [T=0 (32,64,128,256)] [TT=0 (all)] [KS=0 (all)]
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
#ifndef CG
#define CG 1
#endif
static_assert(CG == 1, "xover6: CG = 1 only");
typedef unsigned int u32;
typedef unsigned long long ull;
typedef __nv_bfloat16 bf16;
#ifndef GRID_N
#define GRID_N 148   // grid size (CTAs, one per SM); < 148 = the GEMM on a subset of the SMs (nano-batch overlap question, MK-attn 2026-10-23)
#endif
constexpr int NT = 512, GRID = GRID_N, MAXRING = 16, TMEM_COLS = 512;
constexpr int SMEM_MAX = 227 * 1024, SMEM_FIXED = 1024 + 2048;   // 1 KB alignment slack + 2 KB (barriers, tmem slot)
constexpr int A_BYTES = 128 * 128 * 2;                           // 32768 per k-step and CTA

// ---- helpers copied from xover2.cu (verified there / in mk) ----
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
#ifndef PROWH
#define PROWH 0
#endif
#if TW   // spin on the non-blocking test_wait (no try_wait suspend window)
    asm volatile("{\n\t.reg .pred P;\n\tmbarrier.test_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2;\n\tselp.u32 %0, 1, 0, P;\n\t}" : "=r"(ok) : "r"(a), "r"(parity) : "memory");
#else
    asm volatile("{\n\t.reg .pred P;\n\tmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2;\n\tselp.u32 %0, 1, 0, P;\n\t}" : "=r"(ok) : "r"(a), "r"(parity) : "memory");
#endif
    if (ok) break;
  }
}
// CG = 2: acce[b] receives remote (cluster-scope release) arrivals: wait with acquire at cluster scope
__device__ __forceinline__ void mbar_wait_cluster(u32 a, u32 parity) {
  u32 ok;
  while (true) {
    asm volatile("{\n\t.reg .pred P;\n\tmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%1], %2;\n\tselp.u32 %0, 1, 0, P;\n\t}" : "=r"(ok) : "r"(a), "r"(parity) : "memory");
    if (ok) break;
  }
}
__device__ __forceinline__ void mbar_arrive(u32 a) { asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(a) : "memory"); }
// arrive on the same-offset barrier of cluster rank 0 (a = local shared::cta address of the barrier)
__device__ __forceinline__ void mbar_arrive_rank0(u32 a) {
  asm volatile("{\n\t.reg .b32 ra;\n\tmapa.shared::cluster.u32 ra, %0, 0;\n\tmbarrier.arrive.release.cluster.shared::cluster.b64 _, [ra];\n\t}" :: "r"(a) : "memory");
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

// ---- cta_group::2 helpers: tma2d2 / mma_tc2 / commit2 VERBATIM from pi05 h_bench.cu; alloc/free with the column count as an operand; cluster_sync_all from fg_k1.cu (xover2.cu lines 105-116) ----
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
__device__ __forceinline__ u32 cluster_rank() { u32 r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }

// TMEM layout: accumulator b (= global tile counter & 1) at columns b*TT .. b*TT+TT-1 of the 512-column allocation.
#ifndef WU
#define WU 0   // 1: producer / issuer roles as whole warps (lane 0 works, lanes 1..31 wait at __syncwarp) instead of single threads
#endif
#ifndef EV
#define EV 0
#endif
#if EV   // per-role globaltimer events of phase nph / 2 (CTA b: g_ev[b * 32 + i]); 0: issuer after full wait k (1 + k), producer after empty wait j (12 + j), epilogue (24, 25), tid 0 before / after gbar (26, 27), phase start (0)
__device__ ull g_ev[GRID_N * 32];
#define EVT(i) do { if (cur_p == nph / 2) g_ev[blockIdx.x * 32 + (i)] = gt(); } while (0)
#else
#define EVT(i) do { } while (0)
#endif
__global__ void __launch_bounds__(NT, 1) k_ts(const __grid_constant__ CUtensorMap mW, const __grid_constant__ CUtensorMap mX, float* Y, int N, int K, int T,
                                              int TT, int KS, int nph, int phase0, long long rowsAvail, int ring, unsigned* cnt, ull* tm, int orc, unsigned* stop) {   // stop: optional flag set by CTA 0 after the last grid barrier (stream kernel exit). orc (oracle): bit 0 no MMA (commits kept), bit 1 no epilogue stores (TMEM reads kept), bit 2 no B loads, bit 3 no A loads, bit 4 no TMEM reads, bit 5 plain arrives for commits, bit 6 grid barrier only
  extern __shared__ uint8_t smraw[];
  const u32 raw = smem_u32(smraw), base = (raw + 1023) & ~1023u;
  uint8_t* sm = smraw + (base - raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, q = warp & 3;
  const int rank = CG == 2 ? (int)cluster_rank() : 0, own = blockIdx.x / CG;       // own: CTA (CG 1) or pair (CG 2) index
  const int BR = TT / CG;                                             // B rows (tokens) per CTA
  const u32 BB = (u32)BR * 256, SLOT = A_BYTES + BB;                  // B bytes / ring slot bytes per CTA (multiples of 1024)
  const u32 bars = base + ring * SLOT;
  const u32 fullb = bars, emptyb = bars + 128, accf = bars + 256, acce = bars + 272;   // full[s] +8s, empty[s] +8s (ring <= 16), accfull[2], accempty[2]
  volatile u32* tslot = reinterpret_cast<volatile u32*>(sm + (bars - base) + 512);
  const int NCH = TT / 16, NEPI = NCH < 3 ? NCH : 3;                  // column chunks; epilogue warps per quadrant that have work
  const ull c0 = clock64(), g0 = gt();
  if (warp == 0) { if (CG == 2) tmem_alloc_c2(smem_u32((void*)tslot), TMEM_COLS); else tmem_alloc(smem_u32((void*)tslot), TMEM_COLS); }
  if (CG == 2 && warp == 0) tc_fence_before();
  if (tid == 0) {
    for (int i = 0; i < ring; ++i) { mbar_init(fullb + 8 * i, 2); mbar_init(emptyb + 8 * i, 1); }
    for (int b = 0; b < 2; ++b) { mbar_init(accf + 8 * b, 1); mbar_init(acce + 8 * b, CG * 4 * NEPI); }   // acce: rank 0's barrier counts both CTAs' epilogue warps
    mbar_fence_init();
  }
  __syncthreads();
  if (CG == 2) cluster_sync_all();   // both CTAs' barriers initialised and TMEM allocated before any cross-CTA TMA / MMA traffic
  tc_fence_after();
  const u32 tb = *tslot;
  const int NRB = N / (128 * CG), NTT = (T + TT - 1) / TT, NTILE = NRB * NTT * KS, SPT = K / (KS * 128), KSL = K / KS;
  const int t0 = (int)((long long)own * NTILE / ((int)gridDim.x / CG)), t1 = (int)((long long)(own + 1) * NTILE / ((int)gridDim.x / CG));
  const int nt = t1 - t0, nst = nt * SPT;                             // tiles / k-steps of this CTA per phase
  const u32 idesc = tc_idesc(128 * CG, TT);
#if PROWH   // prow of phases p and p + 1 computed once per phase (the 64-bit modulo is not on the fill path)
  int prow_c[2] = {(int)(((long long)phase0 * N) % rowsAvail), 0}, prow_p = 0;
  auto prow = [&](int p) { if (p != prow_p) { prow_p = p; prow_c[p & 1] = (int)(((long long)(phase0 + p) * N) % rowsAvail); } return prow_c[p & 1]; };
#else
  auto prow = [&](int p) { return (int)(((long long)(phase0 + p) * N) % rowsAvail); };
#endif
  // fill_a: A boxes of step j (global step number f, slot f % ring) of phase p on full[s] (arrival 1). Independent of the previous phase: may run in the barrier window.
  // CG = 2: both ranks load their 128 rows; only rank 0 expects the bytes of both (all TMAs complete on rank 0's barrier).
  auto fill_a = [&](int p, int j, u32 f) {
    const int t = t0 + j / SPT, kc = (t % KS) * KSL + (j % SPT) * 128, rb = (t / KS) / NTT, s = f % ring;
    const u32 full = fullb + 8 * s;
    if (orc & 8) { if (CG == 1 || rank == 0) mbar_arrive(full); return; }   // oracle: no A loads
    if (CG == 1 || rank == 0) mbar_expect_tx(full, CG * A_BYTES);
    for (int a = 0; a < 2; ++a) {
      const int row = prow(p) + rb * (128 * CG) + 128 * rank;
      if (CG == 2) tma2d2(base + s * SLOT + a * 16384, &mW, full, kc + a * 64, row);
      else tma2(base + s * SLOT + a * 16384, &mW, kc + a * 64, row, full);
    }
  };
  // fill_b: B boxes (X, the previous phase's output in the megakernel) on full[s] (arrival 2): issued after the grid barrier. Box {64, TT / CG}; rows >= T are OOB (zero fill).
  auto fill_b = [&](int j, u32 f) {
    const int t = t0 + j / SPT, kc = (t % KS) * KSL + (j % SPT) * 128, tt = (t / KS) % NTT, s = f % ring;
    const u32 full = fullb + 8 * s;
    if (orc & 4) { if (CG == 1 || rank == 0) mbar_arrive(full); return; }   // oracle: no B loads
    if (CG == 1 || rank == 0) mbar_expect_tx(full, (u32)TT * 256);
    for (int a = 0; a < 2; ++a) {
      const int row = tt * TT + rank * BR;
      if (CG == 2) tma2d2(base + s * SLOT + A_BYTES + a * (BR * 128), &mX, full, kc + a * 64, row);
      else tma2(base + s * SLOT + A_BYTES + a * (BR * 128), &mX, kc + a * 64, row, full);
    }
  };
  // epi_tile: epilogue of tile ti of the phase (accumulator b = G & 1), TMEM -> Y; CTA r's TMEM lane 32 q + l = W row rb * 128 CG + 128 rank + 32 q + l, column c = token tt * TT + c.
  int cur_p = 0;
  auto epi_tile = [&](int ti, u32 G0) {
      const u32 G = G0 + ti, b = G & 1;
      const int t = t0 + ti, rb = (t / KS) / NTT, tt = (t / KS) % NTT;
      mbar_wait(accf + 8 * b, (G >> 1) & 1);
      if (warp == 4 && lane == 0 && ti == 0) EVT(24);
      tc_fence_after();
      float* yp = Y + (size_t)(tt * TT) * N + rb * (128 * CG) + 128 * rank + 32 * q + lane;
      const u32 ta = tb + ((u32)(q * 32) << 16) + b * TT;
      for (int c = (warp >> 2) - 1; c < NCH; c += 6) {
        if (orc & 16) break;                                          // oracle: no TMEM reads
        const int cb = tt * TT + c * 16;                              // first token of the chunk
        if (cb >= T) break;                                           // ragged tile: chunk past T
        const bool two = c + 3 < NCH && cb + 48 < T;
        const int nv0 = min(16, T - cb), nv1 = min(16, T - cb - 48);  // valid tokens of the two chunks (T < 16 or T % 16 != 0)
        u32 v0[16], v1[16];
        tmem_ld16(v0, ta + c * 16);
        if (two) tmem_ld16(v1, ta + (c + 3) * 16);
        tc_wait_ld();
        float* y0 = yp + (size_t)(c * 16) * N;
        if (orc & 2) { if (__uint_as_float(v0[0]) == 1234.5f) y0[0] = 0.f; continue; }   // oracle: no stores (one never-taken store keeps the TMEM reads live)
        if (KS == 1) {
#pragma unroll
          for (int i = 0; i < 16; ++i) if (i < nv0) y0[(size_t)i * N] = __uint_as_float(v0[i]);
          if (two) {
#pragma unroll
            for (int i = 0; i < 16; ++i) if (i < nv1) y0[(size_t)(48 + i) * N] = __uint_as_float(v1[i]);
          }
        } else {
#pragma unroll
          for (int i = 0; i < 16; ++i) if (i < nv0) red_add(y0 + (size_t)i * N, __uint_as_float(v0[i]));
          if (two) {
#pragma unroll
            for (int i = 0; i < 16; ++i) if (i < nv1) red_add(y0 + (size_t)(48 + i) * N, __uint_as_float(v1[i]));
          }
        }
      }
      tc_fence_before();
      __syncwarp();
      if (warp == 4 && lane == 0 && ti == 0) EVT(25);
      if (lane == 0) { if (CG == 2 && rank == 1) mbar_arrive_rank0(acce + 8 * b); else mbar_arrive(acce + 8 * b); }
  };
  if (tid == 0) for (int i = 0; i < min(ring, nst); ++i) { fill_a(0, i, i); fill_b(i, i); }
  for (int p = 0; p < nph; ++p) {
    cur_p = p;
    if (tid == 0) EVT(0);
    const u32 F0 = (u32)p * nst, G0 = (u32)p * nt;                    // global step / tile counters at the start of this phase
    if (orc & 64) {                                                   // oracle: grid barrier only (the window prefetch of phase 0 is never consumed)
    } else if (WU ? warp == 0 : tid == 0) { if (!WU || lane == 0) {   // ---- producer (both ranks for CG = 2, own empty[s])
      if (p > 0) for (int i = 0; i < min(ring, nst); ++i) fill_b(i, F0 + i);   // B of the window-prefetched steps, after the grid barrier
      for (int j = ring; j < nst; ++j) {
        const u32 f = F0 + j, s = f % ring;
        mbar_wait(emptyb + 8 * s, ((f / ring) - 1) & 1); if (j < 12) EVT(12 + j);              // previous use (step f - ring, use number f / ring - 1) of the slot has been read by its MMAs
        fill_a(p, j, f); fill_b(j, f);
      }
      if (p + 1 < nph) {                                              // barrier window: A of the next phase's first steps
        const u32 F1 = F0 + nst;
        for (int i = 0; i < min(ring, nst); ++i) {
          const u32 f = F1 + i, s = f % ring;
          if (f >= (u32)ring) mbar_wait(emptyb + 8 * s, ((f / ring) - 1) & 1);   // step f - ring (this or an earlier phase) must be done with the slot
          fill_a(p + 1, i, f);
        }
      }
    } if (WU) __syncwarp(); } else if (WU ? (warp == 1 && rank == 0) : (tid == 32 && rank == 0)) { if (!WU || lane == 0) {   // ---- MMA issuer (rank 0 only for CG = 2: pair MMA, M = 256)
      for (int ti = 0; ti < nt; ++ti) {
        const u32 G = G0 + ti, b = G & 1;
        if (G >= 2) { if (CG == 2) mbar_wait_cluster(acce + 8 * b, ((G >> 1) - 1) & 1); else mbar_wait(acce + 8 * b, ((G >> 1) - 1) & 1); }   // epilogue(s) done with accumulator b (use G / 2 - 1)
        for (int k = 0; k < SPT; ++k) {
          const u32 f = F0 + ti * SPT + k, s = f % ring;
          mbar_wait(fullb + 8 * s, (f / ring) & 1);
          if (ti == 0 && k < 11) EVT(1 + k);
          tc_fence_after();
          const u32 sa = base + s * SLOT, sb = sa + A_BYTES;
#pragma unroll
          for (int kk = 0; kk < 8; ++kk) {
            if (orc & 1) break;
            const int a = kk >> 2, off = (kk & 3) * 32;
            const u32 acc = (k > 0 || kk > 0) ? 1u : 0u;
            if (CG == 2) mma_tc2(tb + b * TT, sdesc(sa + a * 16384 + off), sdesc(sb + a * (BR * 128) + off), idesc, acc);
            else tc_mma(tb + b * TT, sdesc(sa + a * 16384 + off), sdesc(sb + a * (BR * 128) + off), idesc, acc);
          }
          if (CG == 2) { commit2(emptyb + 8 * s); if (k == SPT - 1) commit2(accf + 8 * b); }
          else if (orc & 32) { mbar_arrive(emptyb + 8 * s); if (k == SPT - 1) mbar_arrive(accf + 8 * b); }   // oracle: plain arrives (no tcgen05.commit)
          else { tc_commit(emptyb + 8 * s); if (k == SPT - 1) tc_commit(accf + 8 * b); }
        }
      }
    } if (WU) __syncwarp(); } else if (warp >= 4 && (warp >> 2) - 1 < NCH) {                  // ---- epilogue (both ranks): quadrant q = warp & 3, chunks (warp >> 2) - 1 + 3 m
      for (int ti = 0; ti < nt; ++ti) epi_tile(ti, G0);
    }
    if (tid == 0) EVT(26);
    __threadfence();
    gbar(cnt, p + 1);
    if (tid == 0) EVT(27);
  }
  if (stop && blockIdx.x == 0 && tid == 0) asm volatile("st.release.gpu.global.u32 [%0], 1;" :: "l"(stop) : "memory");
  __syncthreads();
  if (CG == 2) cluster_sync_all();   // no CTA frees TMEM / exits while its peer's MMAs, TMAs or remote arrivals may still be in flight
  tc_fence_after();
  if (warp == 0) { if (CG == 2) tmem_free_c2(tb, TMEM_COLS); else tmem_dealloc(tb, TMEM_COLS); }
  if (tid == 0) { const ull c1 = clock64(), g1 = gt(); ull* o = tm + blockIdx.x * 4; o[0] = c0; o[1] = c1; o[2] = g0; o[3] = g1; }
}

__device__ __forceinline__ float hashf(u32 i) { u32 h = i * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16; return ((float)(h & 0xffff) - 32768.f) * (1.f / 32768.f); }   // [-1, 1)
__global__ void k_init(bf16* p, size_t n, u32 seed) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = __float2bfloat16(hashf((u32)i + seed)); }
// CUDA-core reference for phase 0 (pool rows 0..N-1), one thread per (t, n); counts violations of |d| <= 2e-3 * max(1, |ref|).
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

// ======================= k_kv: TMA KV stream (SPEC-xover6) =======================
// Pool viewed as rows of 2048 B (8 KV heads x 128 dims bf16). Stage = one 32 KB slot = 128 token rows x 128 dims of ONE head = two SWIZZLE_128B boxes {64 dims, 128 rows}.
// Global stage g -> row block g / 8, head g % 8: 8 consecutive stages cover 128 contiguous rows (256 KB contiguous). CTA b of nb owns stages [b*S/nb, (b+1)*S/nb) of each iteration;
// iteration it reads window (it % nwin) of the pool (no re-read of the previous iteration's bytes; pool >> L2).
constexpr int KV_STAGE = 32768, KV_NCONS = 12;
__device__ __forceinline__ ull kv_policy() {   // copied from splitkv.cuh (SKV_KV_EVICT_FIRST 1)
  ull pol;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void tma2_kv(u32 dst, const void* map, int c0, int c1, u32 mbar, ull pol) {   // 2D form of splitkv.cuh tma3d (evict_first hint)
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(dst), "l"(map), "r"(mbar), "r"(c0), "r"(c1), "l"(pol) : "memory");
}
// stamps[(it * nb + b) * 2 + {0, 1}] = iteration start / end (globaltimer); nitdone[b] = completed iterations; sink[b * 384 + tc] = per-thread sum.
__global__ void __launch_bounds__(512, 1) k_kv(const __grid_constant__ CUtensorMap mK, int S, int nwin, int depth, int cap, ull* stamps, int* nitdone, u32* sink, unsigned* stop) {
  extern __shared__ uint8_t smraw[];
  const u32 raw = smem_u32(smraw), base = (raw + 1023) & ~1023u;
  uint8_t* sm = smraw + (base - raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const u32 bars = base + depth * KV_STAGE, fullb = bars, emptyb = bars + 128;
  volatile u32* done = reinterpret_cast<volatile u32*>(sm + depth * KV_STAGE + 256);   // done[s]: slot s carries the sentinel (producer end)
  if (tid == 0) {
    for (int i = 0; i < depth; ++i) { mbar_init(fullb + 8 * i, 1); mbar_init(emptyb + 8 * i, KV_NCONS); done[i] = 0; }
    mbar_fence_init();
  }
  __syncthreads();
  const int b = blockIdx.x, nb = gridDim.x;
  const long long s0 = (long long)b * S / nb, s1 = (long long)(b + 1) * S / nb;
  const int nst = (int)(s1 - s0);
  if (tid == 0) {   // ---- producer
    const ull pol = kv_policy();
    u32 f = 0;
    for (int it = 0; it < cap; ++it) {
      if (stop && it > 0 && ld_acquire(stop)) break;
      const long long gb = (long long)(it % nwin) * S + s0;
      for (int j = 0; j < nst; ++j) {
        const long long g = gb + j;
        const int row = (int)(g >> 3) * 128, head = (int)(g & 7);
        const u32 s = f % depth;
        if (f >= (u32)depth) mbar_wait(emptyb + 8 * s, ((f / depth) - 1) & 1);
        mbar_expect_tx(fullb + 8 * s, KV_STAGE);
        for (int a = 0; a < 2; ++a) tma2_kv(base + s * KV_STAGE + a * 16384, &mK, head * 128 + a * 64, row, fullb + 8 * s, pol);
        ++f;
      }
    }
    const u32 s = f % depth;   // sentinel: plain arrive (no tx) with done[s] set
    if (f >= (u32)depth) mbar_wait(emptyb + 8 * s, ((f / depth) - 1) & 1);
    done[s] = 1;
    mbar_arrive(fullb + 8 * s);
  } else if (warp >= 4) {   // ---- consumers: warps 4..15 (384 threads), 16 B each of every slot
    const int tc = tid - 128;
    u32 f = 0, acc = 0; int it = 0; bool fin = false;
    ull tprev = gt();
    for (; !fin; ++it) {
      for (int j = 0; j < nst; ++j) {
        const u32 s = f % depth;
        mbar_wait(fullb + 8 * s, (f / depth) & 1);
        if (done[s]) { fin = true; break; }
        uint4 v;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(base + s * KV_STAGE + tc * 16) : "memory");
        acc += v.x + v.y + v.z + v.w;
        __syncwarp();
        if (lane == 0) mbar_arrive(emptyb + 8 * s);
        ++f;
      }
      if (fin) break;
      if (tid == 128) { const ull te = gt(); stamps[((size_t)it * nb + b) * 2] = tprev; stamps[((size_t)it * nb + b) * 2 + 1] = te; tprev = te; }
    }
    sink[b * 384 + tc] = acc;
    if (tid == 128) nitdone[b] = it;
  }
}

// ======================= host =======================
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
struct Shp { const char* name; int N, K; };
static Shp shapes[4] = {{"QKV", 4096, 1024}, {"O", 1024, 2048}, {"GU", 6144, 1024}, {"D", 1024, 3072}};
static const int KSs[9] = {1, 2, 3, 4, 6, 8, 12, 16, 24};
static PFN_cuTensorMapEncodeTiled_v12000 enc;
static const size_t POOL = 2ull << 30, KVPOOL = 8ull << 30;

static CUtensorMap mkmap(void* ptr, int dim0, long long dim1, int box0, int box1, CUtensorMapL2promotion prom) {
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)dim0, (cuuint64_t)dim1}, strides[1] = {(cuuint64_t)dim0 * 2};
  cuuint32_t box[2] = {(cuuint32_t)box0, (cuuint32_t)box1}, es[2] = {1, 1};
  CUresult r = enc(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, prom, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "tensor map encode failed %d\n", (int)r); exit(1); }
  return m;
}

struct Dev {   // device state
  bf16 *pool, *X, *kvpool = nullptr; float* Y; unsigned *cnt, *ver, *stop; ull* tm; ull* kst = nullptr; int* nitd; u32* sink;
  int regs_ts, spill_ts, regs_kv, spill_kv; int kcap = 0;
};
struct Cfg { int sh, T, TT, KS; };
struct Gemm { CUtensorMap mW, mX; int N, K, T, TT, KS, ring, smem, ntile; long long rowsAvail; };

static bool make_gemm(Dev& d, const Cfg& c, int kc, Gemm& g) {   // false if the config is not valid / has too little work / low occupancy
  const Shp& sh = shapes[c.sh];
  if (c.TT % 16 || c.TT < 16 || c.TT > 256 || c.TT > c.T || (sh.K / 128) % c.KS) return false;
  g.N = sh.N; g.K = sh.K; g.T = c.T; g.TT = c.TT; g.KS = c.KS;
  const long long rows = (long long)(POOL / (2ull * sh.K)); g.rowsAvail = rows - sh.N;
  g.mW = mkmap(d.pool, sh.K, rows, 64, 128, CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  g.mX = mkmap(d.X, sh.K, c.T, 64, c.TT, CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  const int slotB = A_BYTES + c.TT * 256;
  g.ring = std::min(MAXRING, (SMEM_MAX - SMEM_FIXED) / slotB); g.smem = SMEM_FIXED + g.ring * slotB;
  const int NTT = (c.T + c.TT - 1) / c.TT; g.ntile = (sh.N / 128) * NTT * c.KS;
  if (g.ntile < std::min(kc, 37)) return false;   // xover4 rule (NTILE < 37 skipped); fewer tiles than CTAs is allowed (idle CTAs)
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_ts, NT, g.smem));
  return occ >= 1;
}
static void launch_ts(Dev& d, Gemm& g, int kc, int np, int p0, cudaStream_t st, bool coop, unsigned* stop) {
  cudaLaunchConfig_t lc = {}; lc.gridDim = dim3(kc); lc.blockDim = dim3(NT); lc.dynamicSmemBytes = g.smem; lc.stream = st;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeCooperative; at[0].val.cooperative = 1; lc.attrs = at; lc.numAttrs = coop ? 1 : 0;
  CK(cudaMemsetAsync(d.cnt, 0, 1024, st));
  CK(cudaLaunchKernelEx(&lc, k_ts, g.mW, g.mX, d.Y, g.N, g.K, g.T, g.TT, g.KS, np, p0, g.rowsAvail, g.ring, d.cnt, d.tm, 0, stop));
}
static void init_x(Dev& d, int sh) { k_init<<<256, 256>>>(d.X, 256ull * shapes[sh].K, 777u); CK(cudaDeviceSynchronize()); }
// PASS check: 1 phase (phase 0), compared with the CUDA-core reference
static int check_gemm(Dev& d, Gemm& g, int kc, bool coop) {
  CK(cudaMemset(d.Y, g.KS == 1 ? 0xFF : 0, (size_t)g.T * g.N * 4)); CK(cudaMemset(d.ver, 0, 64));
  launch_ts(d, g, kc, 1, 0, 0, coop, nullptr); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  k_ref<<<((size_t)g.T * g.N + 255) / 256, 256>>>(d.pool, d.X, d.Y, g.N, g.K, g.T, 0, d.ver, d.ver + 1); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  unsigned hv[2]; CK(cudaMemcpy(hv, d.ver, 8, cudaMemcpyDeviceToHost));
  return hv[1] == 0;
}
static double time_gemm(Dev& d, Gemm& g, int kc, int nph, int reps, bool coop) {   // median us / phase
  int p0 = 0;
  for (int w = 0; w < 2; ++w) { launch_ts(d, g, kc, nph, p0, 0, coop, nullptr); p0 += nph; }
  CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  std::vector<double> us;
  for (int r = 0; r < reps; ++r) {
    CK(cudaEventRecord(e0)); launch_ts(d, g, kc, nph, p0, 0, coop, nullptr); p0 += nph; CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); us.push_back(ms * 1000.0 / nph);
  }
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  return median(us);
}
// config list for (shape, T): TT in {T, min(T, 64)}, KS in the xover4 list
static std::vector<Cfg> cfgs(int sh, int T, int TTsel, int KSsel) {
  std::vector<int> tts = {T}; if (std::min(T, 64) != T) tts.push_back(std::min(T, 64));
  std::vector<Cfg> v;
  for (int TT : tts) for (int KS : KSs) { if (TTsel && TT != TTsel) continue; if (KSsel && KS != KSsel) continue; v.push_back({sh, T, TT, KS}); }
  return v;
}
// sweep: returns best config (us < 0 if none valid); print = emit one JSON line per config
static Cfg sweep_gemm(Dev& d, int sh, int T, int TTsel, int KSsel, int kc, int nph, bool print, double* best_us) {
  init_x(d, sh);
  Cfg best = {sh, T, 0, 0}; *best_us = 1e30;
  for (const Cfg& c : cfgs(sh, T, TTsel, KSsel)) {
    Gemm g;
    if (!make_gemm(d, c, kc, g)) { if (print) printf("{\"mode\":\"gemm_skip\",\"k_c\":%d,\"shape\":\"%s\",\"T\":%d,\"TT\":%d,\"KS\":%d}\n", kc, shapes[sh].name, T, c.TT, c.KS); continue; }
    int pass = -1;
    if (T <= 128) { pass = check_gemm(d, g, kc, true); fprintf(stderr, "%s ts k_c=%d %s T=%d TT=%d KS=%d\n", pass ? "PASS" : "FAIL", kc, shapes[sh].name, T, c.TT, c.KS); }
    const double us = time_gemm(d, g, kc, nph, 5, true);
    if (print) { printf("{\"mode\":\"gemm\",\"k_a\":%d,\"k_c\":%d,\"shape\":\"%s\",\"T\":%d,\"TT\":%d,\"KS\":%d,\"us_phase\":%.3f,\"pass\":%s,\"ring\":%d,\"ntile\":%d}\n",
                        148 - kc, kc, shapes[sh].name, T, c.TT, c.KS, us, pass < 0 ? "null" : (pass ? "true" : "false"), g.ring, g.ntile); fflush(stdout); }
    if (pass != 0 && us < *best_us) { *best_us = us; best = c; }
  }
  return best;
}

struct KvRes { double agg_med, agg_tot, agg_ss, cta_med; int sink_ok, nit; };
static void alloc_kv(Dev& d, int ka_max, int cap) {
  if (!d.kvpool) { CK(cudaMalloc(&d.kvpool, KVPOOL)); CK(cudaMemset(d.kvpool, 0x3C, KVPOOL)); }
  if (d.kst && d.kcap >= cap) return;
  if (d.kst) CK(cudaFree(d.kst));
  CK(cudaMalloc(&d.kst, (size_t)cap * 148 * 2 * 8)); d.kcap = cap;
}
static void launch_kv(Dev& d, const CUtensorMap& mK, int ka, int depth, ull bytes, int cap, cudaStream_t st, unsigned* stop) {
  const int S = (int)(bytes / KV_STAGE), nwin = (int)(KVPOOL / ((ull)S * KV_STAGE));
  CK(cudaMemsetAsync(d.nitd, 0, 148 * 4, st)); CK(cudaMemsetAsync(d.sink, 0, 148 * 384 * 4, st));
  cudaLaunchConfig_t lc = {}; lc.gridDim = dim3(ka); lc.blockDim = dim3(512); lc.dynamicSmemBytes = SMEM_MAX; lc.stream = st;
  CK(cudaLaunchKernelEx(&lc, k_kv, mK, S, nwin, depth, cap, d.kst, d.nitd, d.sink, stop));
}
static KvRes kv_stats(Dev& d, int ka, ull bytes) {
  const int S = (int)(bytes / KV_STAGE);
  std::vector<int> nit(148); CK(cudaMemcpy(nit.data(), d.nitd, 148 * 4, cudaMemcpyDeviceToHost));
  int n = 1 << 30; for (int b = 0; b < ka; ++b) n = std::min(n, nit[b]);
  std::vector<ull> st((size_t)n * ka * 2); CK(cudaMemcpy(st.data(), d.kst, st.size() * 8, cudaMemcpyDeviceToHost));
  std::vector<u32> sk(148 * 384); CK(cudaMemcpy(sk.data(), d.sink, sk.size() * 4, cudaMemcpyDeviceToHost));
  KvRes r = {}; r.nit = n; r.sink_ok = 1;
  for (int b = 0; b < ka; ++b) { u32 any = 0; for (int t = 0; t < 384; ++t) any |= sk[b * 384 + t]; if (!any) r.sink_ok = 0; }
  const double B = (double)S * KV_STAGE;
  std::vector<double> agg, cta; ull mn0 = ~0ull, mxl = 0;
  for (int i = 0; i < n; ++i) {
    ull mn = ~0ull, mx = 0;
    for (int b = 0; b < ka; ++b) {
      const ull a = st[((size_t)i * ka + b) * 2], e = st[((size_t)i * ka + b) * 2 + 1]; mn = std::min(mn, a); mx = std::max(mx, e);
      const double bb = (double)((long long)(b + 1) * S / ka - (long long)b * S / ka) * KV_STAGE; cta.push_back(bb / (double)(e - a));
      if (i == 0) mn0 = std::min(mn0, a); if (i == n - 1) mxl = std::max(mxl, e);
    }
    agg.push_back(B / (double)(mx - mn));
  }
  { const int w = std::min(4, n / 4); ull e0 = ~0ull; for (int b = 0; b < ka; ++b) e0 = std::min(e0, st[((size_t)(w - 1) * ka + b) * 2 + 1]);   // steady state: skip the first w iterations (ramp)
    r.agg_ss = B * (n - w) / (double)(mxl - e0); }
  r.agg_med = median(agg); r.agg_tot = B * n / (double)(mxl - mn0); r.cta_med = median(cta);
  return r;
}
static KvRes run_kv_alone(Dev& d, const CUtensorMap& mK, int ka, int depth, ull bytes, int iters) {
  alloc_kv(d, ka, std::max(iters, 8));
  launch_kv(d, mK, ka, depth, bytes, 4, 0, nullptr); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());   // warm-up
  launch_kv(d, mK, ka, depth, bytes, iters, 0, nullptr); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  return kv_stats(d, ka, bytes);
}

int main(int argc, char** argv) {
  if (argc < 10) { fprintf(stderr, "usage: x6 mode(kv|gemm|both) k_a depth bytes_per_iter shape(QKV|O|GU|D) T TT(0=sweep/best) KS(0=sweep/best) iters(kv: iterations; gemm: phases, 0 = 112)\n"); return 1; }
  setvbuf(stdout, 0, _IOLBF, 0);
  const char* mode = argv[1]; const int ka = atoi(argv[2]), depth = atoi(argv[3]); const ull bytes = strtoull(argv[4], 0, 10);
  const char* shn = argv[5]; const int T = atoi(argv[6]), TTsel = atoi(argv[7]), KSsel = atoi(argv[8]), iters = atoi(argv[9]);
  const int kc = 148 - ka;
  int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  if (nsm != 148) { fprintf(stderr, "warning: %d SMs (expected 148)\n", nsm); }
  int sh = 0; for (int i = 0; i < 4; ++i) if (!strcmp(shn, shapes[i].name)) sh = i;
  Dev d;
  CK(cudaMalloc(&d.pool, POOL)); CK(cudaMalloc(&d.X, 256ull * 3072 * 2)); CK(cudaMalloc(&d.Y, 256ull * 6144 * 4)); CK(cudaMalloc(&d.cnt, 1024));
  CK(cudaMalloc(&d.ver, 64)); CK(cudaMalloc(&d.stop, 64)); CK(cudaMalloc(&d.tm, 148 * 4 * 8)); CK(cudaMalloc(&d.nitd, 148 * 4)); CK(cudaMalloc(&d.sink, 148 * 384 * 4));
  CK(cudaMemset(d.stop, 0, 64));
  k_init<<<4096, 256>>>(d.pool, POOL / 2, 12345u); CK(cudaDeviceSynchronize());
  cudaDriverEntryPointQueryResult qr; CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&enc, 12000, cudaEnableDefault, &qr));
  { cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, k_ts)); d.regs_ts = fa.numRegs; d.spill_ts = (int)fa.localSizeBytes;
    CK(cudaFuncGetAttributes(&fa, k_kv)); d.regs_kv = fa.numRegs; d.spill_kv = (int)fa.localSizeBytes; }
  CK(cudaFuncSetAttribute(k_ts, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
  CK(cudaFuncSetAttribute(k_kv, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
  printf("{\"mode\":\"info\",\"cmd\":\"%s k_a=%d depth=%d bytes=%llu shape=%s T=%d TT=%d KS=%d iters=%d\",\"regs_ts\":%d,\"spill_ts\":%d,\"regs_kv\":%d,\"spill_kv\":%d}\n",
         mode, ka, depth, bytes, shn, T, TTsel, KSsel, iters, d.regs_ts, d.spill_ts, d.regs_kv, d.spill_kv);
  if (!strcmp(mode, "kv")) {
    if (ka < 1 || ka > 148 || depth < 2 || depth > 6) { fprintf(stderr, "bad k_a / depth\n"); return 1; }
    alloc_kv(d, ka, std::max(iters, 8));
    const CUtensorMap mKv = mkmap(d.kvpool, 1024, KVPOOL / 2048, 64, 128, CU_TENSOR_MAP_L2_PROMOTION_NONE);
    int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_kv, 512, SMEM_MAX));
    KvRes r = run_kv_alone(d, mKv, ka, depth, bytes, iters);
    printf("{\"mode\":\"kv\",\"k_a\":%d,\"depth\":%d,\"bytes\":%llu,\"iters\":%d,\"agg_gbs\":%.1f,\"agg_gbs_total\":%.1f,\"agg_gbs_ss\":%.1f,\"cta_gbs\":%.2f,\"sink_ok\":%d,\"occ\":%d}\n", ka, depth, bytes, r.nit, r.agg_med, r.agg_tot, r.agg_ss, r.cta_med, r.sink_ok, occ);
    return 0;
  }
  if (!strcmp(mode, "gemm")) {
    double bu; const int nph = iters ? iters : 112;
    Cfg b = sweep_gemm(d, sh, T, TTsel, KSsel, kc, nph, true, &bu);
    printf("{\"mode\":\"gemm_best\",\"k_a\":%d,\"k_c\":%d,\"shape\":\"%s\",\"T\":%d,\"TT\":%d,\"KS\":%d,\"us_phase\":%.3f}\n", ka, kc, shn, T, b.TT, b.KS, bu);
    return 0;
  }
  if (!strcmp(mode, "both")) {
    if (ka < 1 || kc < 1 || depth < 2 || depth > 6) { fprintf(stderr, "bad k_a / depth\n"); return 1; }
    const int cap = 20000;
    alloc_kv(d, ka, cap);
    const CUtensorMap mKv = mkmap(d.kvpool, 1024, KVPOOL / 2048, 64, 128, CU_TENSOR_MAP_L2_PROMOTION_NONE);
    Cfg c = {sh, T, TTsel, KSsel}; double us_alone = 0;
    if (!TTsel || !KSsel) { double bu; c = sweep_gemm(d, sh, T, TTsel, KSsel, kc, 112, false, &bu); us_alone = bu; }
    Gemm g; if (!c.TT || !make_gemm(d, c, kc, g)) { fprintf(stderr, "no valid gemm config\n"); return 1; }
    init_x(d, sh);
    const int pass = T <= 128 ? check_gemm(d, g, kc, true) : -1;
    fprintf(stderr, "%s ts(both cfg, alone) k_c=%d %s T=%d TT=%d KS=%d\n", pass ? "PASS" : "FAIL", kc, shn, T, c.TT, c.KS);
    if (!us_alone) us_alone = time_gemm(d, g, kc, 112, 5, true);
    const KvRes kv_alone = run_kv_alone(d, mKv, ka, depth, bytes, 100);
    int occ_ts = 0, occ_kv = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_ts, k_ts, NT, g.smem)); CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_kv, k_kv, 512, SMEM_MAX));
    const int nph = std::max(112, (int)(60000.0 / us_alone) + 1);
    cudaStream_t sA, sB; CK(cudaStreamCreateWithFlags(&sA, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&sB, cudaStreamNonBlocking));
    for (int rep = 0; rep < 4; ++rep) {   // rep 0 = warm-up (not reported)
      CK(cudaMemset(d.stop, 0, 64)); CK(cudaDeviceSynchronize());
      launch_ts(d, g, kc, nph, 0, sA, false, d.stop);   // non-cooperative (all k_c CTAs are resident: k_c + k_a <= 148, 1 CTA / SM for both kernels)
      launch_kv(d, mKv, ka, depth, bytes, cap, sB, d.stop);
      CK(cudaStreamSynchronize(sA)); CK(cudaStreamSynchronize(sB)); CK(cudaGetLastError());
      if (!rep) continue;
      ull h[148 * 4]; CK(cudaMemcpy(h, d.tm, sizeof(h), cudaMemcpyDeviceToHost));
      ull tsS = ~0ull, tsE = 0; for (int b = 0; b < kc; ++b) { tsS = std::min(tsS, h[b * 4 + 2]); tsE = std::max(tsE, h[b * 4 + 3]); }
      std::vector<int> nit(148); CK(cudaMemcpy(nit.data(), d.nitd, 148 * 4, cudaMemcpyDeviceToHost));
      int n = 1 << 30; for (int b = 0; b < ka; ++b) n = std::min(n, nit[b]);
      std::vector<ull> st((size_t)n * ka * 2); CK(cudaMemcpy(st.data(), d.kst, st.size() * 8, cudaMemcpyDeviceToHost));
      std::vector<u32> sk(148 * 384); CK(cudaMemcpy(sk.data(), d.sink, sk.size() * 4, cudaMemcpyDeviceToHost));
      int sink_ok = 1; for (int b = 0; b < ka; ++b) { u32 any = 0; for (int t = 0; t < 384; ++t) any |= sk[b * 384 + t]; if (!any) sink_ok = 0; }
      ull kvS = ~0ull, kvE = 0; for (int b = 0; b < ka; ++b) { kvS = std::min(kvS, st[(size_t)b * 2]); kvE = std::max(kvE, st[((size_t)(n - 1) * ka + b) * 2 + 1]); }
      const ull ws = std::max(tsS, kvS), we = std::min(tsE, kvE);
      const double W = we > ws ? (double)(we - ws) : 0.0;
      // KV bytes inside [ws, we], prorating iterations that straddle the window edges
      const int S = (int)(bytes / KV_STAGE); double inb = 0;
      for (int i = 0; i < n; ++i) for (int b = 0; b < ka; ++b) {
        const double a = (double)st[((size_t)i * ka + b) * 2], e = (double)st[((size_t)i * ka + b) * 2 + 1];
        const double ov = std::min(e, (double)we) - std::max(a, (double)ws); if (ov <= 0) continue;
        inb += ov / (e - a) * (double)((long long)(b + 1) * S / ka - (long long)b * S / ka) * KV_STAGE;
      }
      const double ts_us = (double)(tsE - tsS) / 1000.0 / nph, ts_in = W > 0 ? (W / 1000.0) / nph * ((double)(tsE - tsS) / W) : 0;
      (void)ts_in;
      printf("{\"mode\":\"both\",\"rep\":%d,\"k_a\":%d,\"k_c\":%d,\"depth\":%d,\"shape\":\"%s\",\"T\":%d,\"TT\":%d,\"KS\":%d,\"nph\":%d,\"gemm_us_both\":%.3f,\"gemm_us_alone\":%.3f,"
             "\"kv_gbs_win\":%.1f,\"kv_gbs_alone\":%.1f,\"ts_ms\":%.2f,\"kv_ms\":%.2f,\"ovl_ts\":%.4f,\"ovl_kv\":%.4f,\"ovl_min\":%.4f,\"kv_iters\":%d,\"sink_ok\":%d,\"pass\":%s,\"occ_ts\":%d,\"occ_kv\":%d}\n",
             rep, ka, kc, depth, shn, T, c.TT, c.KS, nph, ts_us, us_alone, inb / W, kv_alone.agg_ss, (tsE - tsS) / 1e6, (double)(kvE - kvS) / 1e6, W / (double)(tsE - tsS), W / (double)(kvE - kvS),
             W / (double)std::min(tsE - tsS, kvE - kvS), n, sink_ok, pass < 0 ? "null" : (pass ? "true" : "false"), occ_ts, occ_kv);
      fflush(stdout);
    }
    return 0;
  }
  fprintf(stderr, "unknown mode\n"); return 1;
}
