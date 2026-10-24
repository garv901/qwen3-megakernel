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
static_assert(CG == 1 || CG == 2, "CG");
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
    asm volatile("{\n\t.reg .pred P;\n\tmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2;\n\tselp.u32 %0, 1, 0, P;\n\t}" : "=r"(ok) : "r"(a), "r"(parity) : "memory");
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
__global__ void __launch_bounds__(NT, 1) k_ts(const __grid_constant__ CUtensorMap mW, const __grid_constant__ CUtensorMap mX, float* Y, int N, int K, int T,
                                              int TT, int KS, int nph, int phase0, long long rowsAvail, int ring, unsigned* cnt, ull* tm, int orc) {   // orc (oracle): bit 0 no MMA (commits kept), bit 1 no epilogue stores (TMEM reads kept)
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
  const int t0 = (int)((long long)own * NTILE / (GRID / CG)), t1 = (int)((long long)(own + 1) * NTILE / (GRID / CG));
  const int nt = t1 - t0, nst = nt * SPT;                             // tiles / k-steps of this CTA per phase
  const u32 idesc = tc_idesc(128 * CG, TT);
  auto prow = [&](int p) { return (int)(((long long)(phase0 + p) * N) % rowsAvail); };
  // fill_a: A boxes of step j (global step number f, slot f % ring) of phase p on full[s] (arrival 1). Independent of the previous phase: may run in the barrier window.
  // CG = 2: both ranks load their 128 rows; only rank 0 expects the bytes of both (all TMAs complete on rank 0's barrier).
  auto fill_a = [&](int p, int j, u32 f) {
    const int t = t0 + j / SPT, kc = (t % KS) * KSL + (j % SPT) * 128, rb = (t / KS) / NTT, s = f % ring;
    const u32 full = fullb + 8 * s;
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
    if (CG == 1 || rank == 0) mbar_expect_tx(full, (u32)TT * 256);
    for (int a = 0; a < 2; ++a) {
      const int row = tt * TT + rank * BR;
      if (CG == 2) tma2d2(base + s * SLOT + A_BYTES + a * (BR * 128), &mX, full, kc + a * 64, row);
      else tma2(base + s * SLOT + A_BYTES + a * (BR * 128), &mX, kc + a * 64, row, full);
    }
  };
  // epi_tile: epilogue of tile ti of the phase (accumulator b = G & 1), TMEM -> Y; CTA r's TMEM lane 32 q + l = W row rb * 128 CG + 128 rank + 32 q + l, column c = token tt * TT + c.
  auto epi_tile = [&](int ti, u32 G0) {
      const u32 G = G0 + ti, b = G & 1;
      const int t = t0 + ti, rb = (t / KS) / NTT, tt = (t / KS) % NTT;
      mbar_wait(accf + 8 * b, (G >> 1) & 1);
      tc_fence_after();
      float* yp = Y + (size_t)(tt * TT) * N + rb * (128 * CG) + 128 * rank + 32 * q + lane;
      const u32 ta = tb + ((u32)(q * 32) << 16) + b * TT;
      for (int c = (warp >> 2) - 1; c < NCH; c += 6) {
        const int cb = tt * TT + c * 16;                              // first token of the chunk
        if (cb >= T) break;                                           // ragged tile: chunk past T (T is a multiple of 16 here: no per-element guard needed)
        const bool two = c + 3 < NCH && cb + 48 < T;
        u32 v0[16], v1[16];
        tmem_ld16(v0, ta + c * 16);
        if (two) tmem_ld16(v1, ta + (c + 3) * 16);
        tc_wait_ld();
        float* y0 = yp + (size_t)(c * 16) * N;
        if (orc & 2) { if (__uint_as_float(v0[0]) == 1234.5f) y0[0] = 0.f; continue; }   // oracle: no stores (one never-taken store keeps the TMEM reads live)
        if (KS == 1) {
#pragma unroll
          for (int i = 0; i < 16; ++i) y0[(size_t)i * N] = __uint_as_float(v0[i]);
          if (two) {
#pragma unroll
            for (int i = 0; i < 16; ++i) y0[(size_t)(48 + i) * N] = __uint_as_float(v1[i]);
          }
        } else {
#pragma unroll
          for (int i = 0; i < 16; ++i) red_add(y0 + (size_t)i * N, __uint_as_float(v0[i]));
          if (two) {
#pragma unroll
            for (int i = 0; i < 16; ++i) red_add(y0 + (size_t)(48 + i) * N, __uint_as_float(v1[i]));
          }
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) { if (CG == 2 && rank == 1) mbar_arrive_rank0(acce + 8 * b); else mbar_arrive(acce + 8 * b); }
  };
  if (tid == 0) for (int i = 0; i < min(ring, nst); ++i) { fill_a(0, i, i); fill_b(i, i); }
  for (int p = 0; p < nph; ++p) {
    const u32 F0 = (u32)p * nst, G0 = (u32)p * nt;                    // global step / tile counters at the start of this phase
    if (tid == 0) {                                                   // ---- producer (both ranks for CG = 2, own empty[s])
      if (p > 0) for (int i = 0; i < min(ring, nst); ++i) fill_b(i, F0 + i);   // B of the window-prefetched steps, after the grid barrier
      for (int j = ring; j < nst; ++j) {
        const u32 f = F0 + j, s = f % ring;
        mbar_wait(emptyb + 8 * s, ((f / ring) - 1) & 1);              // previous use (step f - ring, use number f / ring - 1) of the slot has been read by its MMAs
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
    } else if (tid == 32 && rank == 0) {                              // ---- MMA issuer (rank 0 only for CG = 2: pair MMA, M = 256)
      for (int ti = 0; ti < nt; ++ti) {
        const u32 G = G0 + ti, b = G & 1;
        if (G >= 2) { if (CG == 2) mbar_wait_cluster(acce + 8 * b, ((G >> 1) - 1) & 1); else mbar_wait(acce + 8 * b, ((G >> 1) - 1) & 1); }   // epilogue(s) done with accumulator b (use G / 2 - 1)
        for (int k = 0; k < SPT; ++k) {
          const u32 f = F0 + ti * SPT + k, s = f % ring;
          mbar_wait(fullb + 8 * s, (f / ring) & 1);
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
          else { tc_commit(emptyb + 8 * s); if (k == SPT - 1) tc_commit(accf + 8 * b); }
        }
      }
    } else if (warp >= 4 && (warp >> 2) - 1 < NCH) {                  // ---- epilogue (both ranks): quadrant q = warp & 3, chunks (warp >> 2) - 1 + 3 m
      for (int ti = 0; ti < nt; ++ti) epi_tile(ti, G0);
    }
    __threadfence();
    gbar(cnt, p + 1);
  }
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

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  const int nph_arg = argc > 1 ? atoi(argv[1]) : 0, reps = argc > 2 ? atoi(argv[2]) : 5;
  const char* shsel = argc > 3 ? argv[3] : "all";
  const int Tsel = argc > 4 ? atoi(argv[4]) : 0, TTsel = argc > 5 ? atoi(argv[5]) : 0, KSsel = argc > 6 ? atoi(argv[6]) : 0;
  const int orc = argc > 7 ? atoi(argv[7]) : 0, ringcap = argc > 8 ? atoi(argv[8]) : 0;   // oracle bits (k_ts orc); ring depth cap (0: as many slots as fit)
  int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  if (nsm != GRID) fprintf(stderr, "warning: %d SMs (expected %d)\n", nsm, GRID);
  const size_t POOL = 2ull << 30;
  bf16 *pool, *X; float* Y; unsigned *cnt, *ver; ull* tm;
  CK(cudaMalloc(&pool, POOL)); CK(cudaMalloc(&X, 256ull * 3072 * 2)); CK(cudaMalloc(&Y, 256ull * 6144 * 4)); CK(cudaMalloc(&cnt, 1024));
  CK(cudaMalloc(&ver, 64)); CK(cudaMalloc(&tm, GRID * 4 * 8));
  k_init<<<4096, 256>>>(pool, POOL / 2, 12345u); CK(cudaDeviceSynchronize());
  PFN_cuTensorMapEncodeTiled_v12000 enc; cudaDriverEntryPointQueryResult qr;
  CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&enc, 12000, cudaEnableDefault, &qr));
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, k_ts));
  const int regs = fa.numRegs, spill = (int)fa.localSizeBytes;
  CK(cudaFuncSetAttribute(k_ts, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
  struct Sh { const char* name; int N, K; } shapes[4] = {{"QKV", 4096, 1024}, {"O", 1024, 2048}, {"GU", 6144, 1024}, {"D", 1024, 3072}};
  const int Ts[4] = {32, 64, 128, 256}, KSs[9] = {1, 2, 3, 4, 6, 8, 12, 16, 24};
  const std::vector<int> TTbase = CG == 1 ? std::vector<int>{48, 64, 96, 128, 256} : std::vector<int>{48, 64, 96, 128, 192, 256};
  for (auto& sh : shapes) {
    if (strcmp(shsel, "all") && strcmp(shsel, sh.name)) continue;
    const int N = sh.N, K = sh.K, NRB = N / (128 * CG);
    const int nph = nph_arg ? nph_arg : 112;
    k_init<<<256, 256>>>(X, 256ull * K, 777u); CK(cudaDeviceSynchronize());
    const long long rows = (long long)(POOL / (2ull * K)), rowsAvail = rows - N;
    CUtensorMap mW;
    { cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)rows}, strides[1] = {(cuuint64_t)K * 2};
      cuuint32_t box[2] = {64, 128}, es[2] = {1, 1};
      CUresult r = enc(&mW, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, pool, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                       CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (r != CUDA_SUCCESS) { fprintf(stderr, "tensor map encode failed %d\n", (int)r); return 1; } }
    for (int T : Ts) {
      if (Tsel && T != Tsel) continue;
      std::vector<int> tts = TTbase; if (std::find(tts.begin(), tts.end(), T) == tts.end()) tts.push_back(T);   // TT = T is always valid
      for (int TT : tts) {
        if (TT > T || (TTsel && TT != TTsel)) continue;
        if (TT % 16 || TT < 16 || TT > 256 || (CG == 2 && (TT / 2) % 8)) continue;
        // B map: X [T][K] bf16, box 64 k x TT/CG tokens, SW128; T % TT != 0: the last tile's rows >= T are TMA OOB (zero fill)
        CUtensorMap mX;
        { cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)T}, strides[1] = {(cuuint64_t)K * 2};
          cuuint32_t box[2] = {64, (cuuint32_t)(TT / CG)}, es[2] = {1, 1};
          CUresult r = enc(&mX, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, X, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                           CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
          if (r != CUDA_SUCCESS) { fprintf(stderr, "X tensor map encode failed %d\n", (int)r); return 1; } }
        const int slotB = A_BYTES + (TT / CG) * 256;
        const int ring = std::min(ringcap ? ringcap : MAXRING, std::min(MAXRING, (SMEM_MAX - SMEM_FIXED) / slotB));
        const int smem = SMEM_FIXED + ring * slotB;
        for (int KS : KSs) {
          if (KSsel && KS != KSsel) continue;
          if ((K / 128) % KS) continue;
          const int NTT = (T + TT - 1) / TT, NTILE = NRB * NTT * KS, SPT = K / (KS * 128);
          if (NTILE < 37) { printf("{\"skipped\":\"NT<37\",\"cg\":%d,\"shape\":\"%s\",\"T\":%d,\"TT\":%d,\"KS\":%d,\"nt\":%d,\"ntile\":%d}\n", CG, sh.name, T, TT, KS, NTILE, NTILE); continue; }
          const int NOWN = GRID / CG;
          int tmax = 0; for (int b = 0; b < NOWN; ++b) tmax = std::max(tmax, (int)((long long)(b + 1) * NTILE / NOWN - (long long)b * NTILE / NOWN));
          cudaLaunchConfig_t lc = {}; lc.gridDim = dim3(GRID); lc.blockDim = dim3(NT); lc.dynamicSmemBytes = smem; lc.stream = 0;
          cudaLaunchAttribute at[2]; at[0].id = cudaLaunchAttributeCooperative; at[0].val.cooperative = 1;
          at[1].id = cudaLaunchAttributeClusterDimension; at[1].val.clusterDim.x = 2; at[1].val.clusterDim.y = 1; at[1].val.clusterDim.z = 1;
          lc.attrs = at; lc.numAttrs = CG == 2 ? 2 : 1;
          int occupancy = 0;   // CG = 1: resident CTAs per SM (need >= 1); CG = 2: max co-resident clusters of 2 (need 74)
          if (CG == 1) CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, k_ts, NT, smem));
          else CK(cudaOccupancyMaxActiveClusters(&occupancy, k_ts, &lc));
          if (occupancy < (CG == 1 ? 1 : GRID / 2)) { fprintf(stderr, "ts%d %s T=%d TT=%d KS=%d: occupancy %d too low (smem %d), skipped\n", CG, sh.name, T, TT, KS, occupancy, smem); continue; }
          auto launch = [&](int np, int p0) {
            CK(cudaMemset(cnt, 0, 1024));
            CK(cudaLaunchKernelEx(&lc, k_ts, mW, mX, Y, N, K, T, TT, KS, np, p0, rowsAvail, ring, cnt, tm, orc));
          };
          int pass = -1;
          if (!orc && (T == 64 || T == 256)) {
            CK(cudaMemset(Y, KS == 1 ? 0xFF : 0, (size_t)T * N * 4)); CK(cudaMemset(ver, 0, 64));   // KS == 1 writes every element: NaN fill catches misses
            launch(1, 0); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            k_ref<<<((size_t)T * N + 255) / 256, 256>>>(pool, X, Y, N, K, T, 0, ver, ver + 1); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            unsigned hv[2]; CK(cudaMemcpy(hv, ver, 8, cudaMemcpyDeviceToHost)); float me; memcpy(&me, &hv[0], 4);
            pass = (hv[1] == 0);
            fprintf(stderr, "%s ts cg=%d %s T=%d TT=%d KS=%d max|d|=%.3e bad=%u\n", pass ? "PASS" : "FAIL", CG, sh.name, T, TT, KS, me, hv[1]);
          }
          int p0 = 0;
          for (int w = 0; w < 2; ++w) { launch(nph, p0); p0 += nph; }   // warm-up (Y is not re-zeroed for KS > 1, as xover2: values grow, timing unaffected)
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
          printf("{\"bin\":\"ts\",\"orc\":%d,\"cg\":%d,\"shape\":\"%s\",\"T\":%d,\"TT\":%d,\"KS\":%d,\"us_phase\":%.3f,\"us_reps\":[", orc, CG, sh.name, T, TT, KS, median(us));
          for (size_t i = 0; i < us.size(); ++i) printf("%s%.3f", i ? "," : "", us[i]);
          printf("],\"mhz\":%.0f,\"pass\":%s,\"regs\":%d,\"spill\":%d,\"ring\":%d,\"nt\":%d,\"ntile\":%d,\"tiles_max\":%d,\"tiles_mean\":%.3f,\"steps_per_tile\":%d,\"smem\":%d}\n",
                 median(mhz), pass < 0 ? "null" : (pass ? "true" : "false"), regs, spill, ring, NTILE, NTILE, tmax, (double)NTILE / NOWN, SPT, smem);
          fflush(stdout);
          CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
        }
      }
    }
  }
  return 0;
}
