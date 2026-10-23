// xover: FFMA vs tcgen05 engine for ONE GEMM phase  Y[T][N] += X[T][K] * W[N][K]^T  over tokens T = 1..256 (B200).
// Shapes: GU (N=6144, K=1024), D (N=1024, K=3072). W bf16 row-major (K contiguous), X bf16 [T][K], Y fp32 [T][N].
// Skeleton (identical for every engine): 148 CTAs x 512 thr, 1 CTA/SM, cooperative launch, nph phases per launch with a counter
// grid barrier between phases (copied from gemm_tc.cu U10), cold weights (phase p reads a fresh slice of a 2 GiB pool, wrapped).
// Work unit = 128 rows x 128 k of W (32 KB). u = rb*(K/128)+kc; CTA b owns units [b*U/148, (b+1)*U/148) (1..3 units).
// Weights: TMA 2D (SW128, box 64k x 128 rows, 2 boxes/unit) into a 2-slot ring, one mbarrier (expect_tx) per slot. The next
// phase's first two units are issued after this phase's math finished reading the ring, before the grid barrier. A 3rd unit
// (GU) is issued into the ring slot of unit 0 as soon as unit 0 is done reading it.
// X: per unit all threads stage the unit's X columns (128 k x T) from global into one of 2 smem X slots in the SW128 K-major
// layout (row = token, rows T..Npad-1 zero, Npad = max(16, roundup16(T)), k-atom a at a*Npad*128 B). FFMA reads the same slot.
// Y: red.global.add.f32 (t < T). ENG0 adds every unit's 128 x T sums from registers (up to ~2x the red.adds of ENG1 at D);
// ENG1 keeps a row block's sum in TMEM and flushes it when the CTA's range leaves the row block or ends (a flush stalls the next
// MMA: one 256-column TMEM accumulator only).
// Engines (build with -DENG=..): 0 = FFMA (warp w owns rows 8w..8w+7, lane = row 8w+l/4 x k quarter l%4, 32 weights in fp32
//   regs, token tiles of TB (default 4), 4 LDS.128 of X per token (a lane's quarter is 32 bf16 = 64 B), 32 FFMA, 2 shuffles/token);
//   1 = tcgen05 swap-AB (A = weight unit, M=128 K-major SW128; B = X tile, N = Npad; D fp32 TMEM, 8 x K16 MMAs per unit, commit ->
//   mbarrier, warps q=w%4 read TMEM lanes 32q.. with tcgen05.ld 32x32b.x16 = row 32q+lane, 16 token columns, warp w handles
//   column chunks w/4, w/4+4, ..); 2 = both paths compiled, runtime flag picks FFMA (reported as eng 2) or tcgen05 (reported as eng
//   3), TMEM alloc/dealloc happen in both; -1 = no-math oracle (same TMA, staging, barrier, red.add of zeros per unit as ENG0).
// Run: ./xover_eN [nph=112] [reps=5] [shape=GU|D|all] [T=0 (all)]    Output: JSON lines, see run.sh.
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -DENG=1 xover.cu -o xover_e1
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
#ifndef ENG
#define ENG 0
#endif
#ifndef TB
#define TB 4
#endif
typedef unsigned int u32;
typedef unsigned long long ull;
typedef __nv_bfloat16 bf16;
constexpr int NT = 512, NW = 16, GRID = 148, RINGB = 65536, UB = 32768;
constexpr bool HAS_TC = (ENG == 1 || ENG == 2);

// ---- helpers copied from dense_tc tc_common.HELPERS_SRC / gemm_tc.cu (verified) ----
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
__host__ __device__ constexpr u32 tc_idesc(int M, int N) { return (1u << 4) | (1u << 7) | (1u << 10) | ((u32)(N >> 3) << 17) | ((u32)(M >> 4) << 24); }
// UMMA smem descriptor: K-major, SWIZZLE_128B, SBO = 1024 B (8 rows x 128 B), LBO unused, sm100 version bit (A: tc_a_desc, B: gemm_tc sdesc)
__device__ __forceinline__ ull sdesc(u32 addr) { return (ull)((addr >> 4) & 0x3fff) | (1ull << 16) | (64ull << 32) | (1ull << 46) | (2ull << 61); }
__device__ __forceinline__ void tc_mma(u32 d_tmem, ull a_desc, ull b_desc, u32 idesc, u32 acc) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\ttcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}" :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(acc) : "memory");
}
__device__ __forceinline__ void tc_commit(u32 mbar) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mbar) : "memory"); }
// TMEM -> regs: 32 lanes (warp quadrant) x 16 fp32 columns. M=128: TMEM lane = accumulator row (tc_row_of(128,q,l) = 32q + l).
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
__device__ __forceinline__ float dot8(float a, const float* w, uint4 x) {   // 8 FFMA, w[2m] pairs with the low bf16 of word m
  a = fmaf(w[0], __uint_as_float(x.x << 16), a); a = fmaf(w[1], __uint_as_float(x.x & 0xffff0000u), a);
  a = fmaf(w[2], __uint_as_float(x.y << 16), a); a = fmaf(w[3], __uint_as_float(x.y & 0xffff0000u), a);
  a = fmaf(w[4], __uint_as_float(x.z << 16), a); a = fmaf(w[5], __uint_as_float(x.z & 0xffff0000u), a);
  a = fmaf(w[6], __uint_as_float(x.w << 16), a); a = fmaf(w[7], __uint_as_float(x.w & 0xffff0000u), a);
  return a;
}

__global__ void __launch_bounds__(NT, 1) k_xover(const __grid_constant__ CUtensorMap mW, const bf16* __restrict__ X, float* Y, int N, int K, int T,
                                                  int nph, int phase0, long long rowsAvail, int eng_arg, unsigned* cnt, ull* tm) {
  extern __shared__ uint8_t smraw[];
  const u32 raw = smem_u32(smraw), base = (raw + 1023) & ~1023u;
  uint8_t* sm = smraw + (base - raw);
  const int Npad = max(16, (T + 15) & ~15), XS = 2 * Npad * 128;       // bytes per X slot (2 k-atoms)
  const u32 bars = base + RINGB + 2 * XS;                              // tma bars [0..1], mma bars at +16, tmem slot at +32
  volatile u32* tslot = reinterpret_cast<volatile u32*>(sm + (bars - base) + 32);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, q = warp & 3;
  const bool use_tc = HAS_TC && (ENG == 1 || eng_arg == 1);
  const ull c0 = clock64(), g0 = gt();
  if (HAS_TC && warp == 0) tmem_alloc(smem_u32((void*)tslot), 256);
  if (tid == 0) { for (int i = 0; i < 4; ++i) mbar_init(bars + 8 * i, 1); mbar_fence_init(); }
  for (int i = tid; i < XS / 8; i += NT) reinterpret_cast<uint4*>(sm + RINGB)[i] = make_uint4(0, 0, 0, 0);   // both X slots: pad rows stay zero
  fence_proxy_async();
  __syncthreads();
  tc_fence_after();
  const u32 tb = HAS_TC ? *tslot : 0;
  const int KC = K / 128, U = (N / 128) * KC, u0 = blockIdx.x * U / GRID, u1 = (blockIdx.x + 1) * U / GRID, n = u1 - u0;
  const u32 idesc = tc_idesc(128, Npad);
  auto prow = [&](int p) { return (int)(((long long)(phase0 + p) * N) % rowsAvail); };
  // tid 0: TMA fill number f (global counter, slot f&1) of unit i of phase p
  auto fill = [&](int p, int i, u32 f) {
    const int u = u0 + i, kc = u % KC, row = prow(p) + (u / KC) * 128, s = f & 1;
    mbar_expect_tx(bars + 8 * s, UB);
    for (int a = 0; a < 2; ++a) tma2(base + s * UB + a * 16384, &mW, kc * 128 + a * 64, row, bars + 8 * s);
  };
  // all threads: X columns of unit i -> X slot i&1, SW128 K-major
  auto stage = [&](int i) {
    const int kc = (u0 + i) % KC;
    uint8_t* xs = sm + RINGB + (i & 1) * XS;
#pragma unroll
    for (int h = 0; h < 2; ++h) {           // two halves of 4 x 16 B per thread (register pressure: 128-reg cap at 512 thr)
      uint4 v[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) { const int c = tid + NT * (4 * h + j); if (c < T * 16) v[j] = *reinterpret_cast<const uint4*>(X + (size_t)(c >> 4) * K + kc * 128 + (c & 15) * 8); }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int c = tid + NT * (4 * h + j), r = c >> 4, ch = c & 15;
        if (c < T * 16) *reinterpret_cast<uint4*>(xs + (ch >> 3) * (Npad * 128) + r * 128 + (((ch & 7) ^ (r & 7)) << 4)) = v[j];
      }
    }
    fence_proxy_async();
  };
  if (tid == 0) for (int i = 0; i < min(2, n); ++i) fill(0, i, i);
  u32 F = 0;                                                            // global unit counter (ring slot, mbarrier parity)
  for (int p = 0; p < nph; ++p) {
    const u32 F0 = F;
    stage(0);
    __syncthreads();
    for (int j = 0; j < n; ++j) {
      const u32 f = F0 + j, s = f & 1, par = (f >> 1) & 1;
      const int u = u0 + j, n0 = (u / KC) * 128;
      if (use_tc) {
        if (tid == 0) {
          mbar_wait(bars + 8 * s, par);
          tc_fence_after();
          const bool first = (j == 0) || (u % KC == 0);
          const u32 sr = base + s * UB, xb = base + RINGB + (j & 1) * XS;
#pragma unroll
          for (int kk = 0; kk < 8; ++kk) {
            const int a = kk >> 2, off = (kk & 3) * 32;
            tc_mma(tb, sdesc(sr + a * 16384 + off), sdesc(xb + a * (Npad * 128) + off), idesc, (kk > 0 || !first) ? 1u : 0u);
          }
          tc_commit(bars + 16 + 8 * s);
        }
      } else {
        mbar_wait(bars + 8 * s, par);
        const int r = 8 * warp + (lane >> 2), qk = lane & 3, a = qk >> 1, ch0 = (qk & 1) * 4;
        float* yp = Y + n0 + r;
        if (ENG == -1) {
          if (qk == 0) for (int t = 0; t < T; ++t) red_add(yp + (size_t)t * N, 0.f);
        } else {
          float w[32];
          const uint8_t* wr = sm + s * UB + a * 16384 + r * 128;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const uint4 v = *reinterpret_cast<const uint4*>(wr + (((ch0 + i) ^ (r & 7)) << 4));
            w[8 * i + 0] = __uint_as_float(v.x << 16); w[8 * i + 1] = __uint_as_float(v.x & 0xffff0000u);
            w[8 * i + 2] = __uint_as_float(v.y << 16); w[8 * i + 3] = __uint_as_float(v.y & 0xffff0000u);
            w[8 * i + 4] = __uint_as_float(v.z << 16); w[8 * i + 5] = __uint_as_float(v.z & 0xffff0000u);
            w[8 * i + 6] = __uint_as_float(v.w << 16); w[8 * i + 7] = __uint_as_float(v.w & 0xffff0000u);
          }
          const uint8_t* xa = sm + RINGB + (j & 1) * XS + a * (Npad * 128);
          for (int tb0 = 0; tb0 < T; tb0 += TB) {
            float acc[TB];
#pragma unroll
            for (int ti = 0; ti < TB; ++ti) {
              acc[ti] = 0.f;
              const int t = tb0 + ti;
              if (t < T) {
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[ti] = dot8(acc[ti], w + 8 * i, *reinterpret_cast<const uint4*>(xa + t * 128 + (((ch0 + i) ^ (t & 7)) << 4)));
              }
            }
#pragma unroll
            for (int ti = 0; ti < TB; ++ti) {
              const int t = tb0 + ti;
              if (t < T) {
                float v = acc[ti];
                v += __shfl_xor_sync(0xffffffffu, v, 1);
                v += __shfl_xor_sync(0xffffffffu, v, 2);
                if (qk == 0) red_add(yp + (size_t)t * N, v);
              }
            }
          }
        }
      }
      if (j + 1 < n) {
        if (use_tc && j >= 1) {      // MMA(j-1) done reading ring slot (f-1)&1 and X slot (j+1)&1
          mbar_wait(bars + 16 + 8 * ((f - 1) & 1), ((f - 1) >> 1) & 1);
          if (tid == 0) fill(p, j + 1, f + 1);
        }
        stage(j + 1);
      }
      if (use_tc && (j == n - 1 || (u + 1) % KC == 0)) {   // row-block segment ends: TMEM -> Y
        mbar_wait(bars + 16 + 8 * s, par);
        tc_fence_after();
        const int nch = (T + 15) / 16;
        for (int c = warp >> 2; c < nch; c += 8) {
          u32 v0[16], v1[16];
          const bool two = c + 4 < nch;
          tmem_ld16(v0, tb + ((u32)(q * 32) << 16) + c * 16);
          if (two) tmem_ld16(v1, tb + ((u32)(q * 32) << 16) + (c + 4) * 16);
          tc_wait_ld();
          float* yp = Y + n0 + 32 * q + lane;
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            if (c * 16 + i < T) red_add(yp + (size_t)(c * 16 + i) * N, __uint_as_float(v0[i]));
            if (two && (c + 4) * 16 + i < T) red_add(yp + (size_t)((c + 4) * 16 + i) * N, __uint_as_float(v1[i]));
          }
        }
        tc_fence_before();
      }
      __syncthreads();
      if (!use_tc && tid == 0 && j + 2 < n) fill(p, j + 2, f + 2);   // everyone finished reading ring slot j&1
    }
    F = F0 + n;
    if (p + 1 < nph && tid == 0) for (int i = 0; i < min(2, n); ++i) fill(p + 1, i, F + i);   // barrier-window prefetch
    __threadfence();
    gbar(cnt, p + 1);
  }
  __syncthreads();
  tc_fence_after();
  if (HAS_TC && warp == 0) tmem_dealloc(tb, 256);
  if (tid == 0) { const ull c1 = clock64(), g1 = gt(); ull* o = tm + blockIdx.x * 4; o[0] = c0; o[1] = c1; o[2] = g0; o[3] = g1; }
}

__device__ __forceinline__ float hashf(u32 i) { u32 h = i * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16; return ((float)(h & 0xffff) - 32768.f) * (1.f / 32768.f); }   // [-1, 1)
__global__ void k_init(bf16* p, size_t n, u32 seed) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = __float2bfloat16(hashf((u32)i + seed)); }
// CUDA-core reference for phase 0 (pool rows 0..N-1), one thread per (t, n); counts violations of |d| <= 2e-3 * max(1, |ref|)
__global__ void k_ref(const bf16* pool, const bf16* X, const float* Y, int N, int K, int T, unsigned* maxerr, unsigned* nbad) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)T * N) return;
  const int t = (int)(i / N), n = (int)(i % N);
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += __bfloat162float(X[(size_t)t * K + k]) * __bfloat162float(pool[(size_t)n * K + k]);
  const float d = fabsf(Y[i] - acc);
  atomicMax(maxerr, __float_as_uint(d));
  if (!(d <= 2e-3f * fmaxf(1.f, fabsf(acc)))) atomicAdd(nbad, 1u);
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  const int nph = argc > 1 ? atoi(argv[1]) : 112, reps = argc > 2 ? atoi(argv[2]) : 5;
  const char* shsel = argc > 3 ? argv[3] : "all";
  const int Tsel = argc > 4 ? atoi(argv[4]) : 0;
  int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  if (nsm != GRID) fprintf(stderr, "warning: %d SMs (expected %d)\n", nsm, GRID);
  const size_t POOL = 2ull << 30;
  bf16 *pool, *X; float* Y; unsigned *cnt, *ver; ull* tm; float* Yd;
  CK(cudaMalloc(&pool, POOL)); CK(cudaMalloc(&X, 256ull * 3072 * 2)); CK(cudaMalloc(&Y, 256ull * 6144 * 4)); CK(cudaMalloc(&cnt, 1024));
  CK(cudaMalloc(&ver, 64)); CK(cudaMalloc(&tm, GRID * 4 * 8)); Yd = Y;
  k_init<<<4096, 256>>>(pool, POOL / 2, 12345u); CK(cudaDeviceSynchronize());
  PFN_cuTensorMapEncodeTiled_v12000 enc; cudaDriverEntryPointQueryResult qr;
  CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&enc, 12000, cudaEnableDefault, &qr));
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, k_xover));
  const int regs = fa.numRegs, spill = (int)fa.localSizeBytes;   // local bytes (= spill when the source has no local arrays)
  const int maxsmem = 1024 + RINGB + 4 * 256 * 128 + 256;
  CK(cudaFuncSetAttribute(k_xover, cudaFuncAttributeMaxDynamicSharedMemorySize, maxsmem));
  std::vector<int> engs; if (ENG == 2) { engs = {2, 3}; } else engs = {ENG};
  struct Sh { const char* name; int N, K; } shapes[2] = {{"GU", 6144, 1024}, {"D", 1024, 3072}};
  const int Ts[9] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  for (auto& sh : shapes) {
    if (strcmp(shsel, "all") && strcmp(shsel, sh.name)) continue;
    const int N = sh.N, K = sh.K;
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
      const int Npad = std::max(16, (T + 15) & ~15), smem = 1024 + RINGB + 4 * Npad * 128 + 256;
      printf("{\"floor\":1,\"shape\":\"%s\",\"T\":%d,\"us_weight_stream\":%.2f,\"us_ffma_math\":%.2f}\n", sh.name, T,
             (double)N * K * 2 / 6.5e12 * 1e6, (double)T * N * K / (148.0 * 128 * 1.964e9) * 1e6);
      for (int eng : engs) {
        const int ea = eng == 3 ? 1 : 0;
        auto launch = [&](int np, int p0) {
          CK(cudaMemset(cnt, 0, 1024));
          int Ni = N, Ki = K, Ti = T, npi = np, p0i = p0, eai = ea; long long ra = rowsAvail;
          void* args[] = {&mW, &X, &Yd, &Ni, &Ki, &Ti, &npi, &p0i, &ra, &eai, &cnt, &tm};
          CK(cudaLaunchCooperativeKernel((void*)k_xover, dim3(GRID), dim3(NT), args, smem, 0));
        };
        int pass = -1;   // -1 = not checked
        if (T == 1 || T == 8 || T == 64 || T == 256) {
          CK(cudaMemset(Y, 0, (size_t)T * N * 4)); CK(cudaMemset(ver, 0, 64));
          launch(1, 0); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
          k_ref<<<((size_t)T * N + 255) / 256, 256>>>(pool, X, Y, N, K, T, ver, ver + 1); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
          unsigned hv[2]; CK(cudaMemcpy(hv, ver, 8, cudaMemcpyDeviceToHost)); float me; memcpy(&me, &hv[0], 4);
          pass = eng == -1 ? -1 : (hv[1] == 0);   // the oracle adds zeros: nothing to compare
          fprintf(stderr, "%s eng=%d %s T=%d max|d|=%.3e bad=%u\n", eng == -1 ? "SKIP" : (pass ? "PASS" : "FAIL"), eng, sh.name, T, me, hv[1]);
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
        printf("{\"eng\":%d,\"shape\":\"%s\",\"T\":%d,\"us_phase\":%.3f,\"us_reps\":[", eng, sh.name, T, median(us));
        for (size_t i = 0; i < us.size(); ++i) printf("%s%.3f", i ? "," : "", us[i]);
        printf("],\"mhz\":%.0f,\"pass\":%s,\"regs\":%d,\"spill\":%d}\n", median(mhz), pass < 0 ? "null" : (pass ? "true" : "false"), regs, spill);
        fflush(stdout);
      }
    }
  }
  return 0;
}
