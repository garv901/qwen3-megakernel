// B8 b8_compose: whole GEMM phases of complete designs, run as phase sequences in mk order (QKV, RoPE, attn, O, GU, D) x nlay,
// with mk's split barrier + per-CTA stamps (same barrier indexing as mk: barrier 0 = start, 1 + 6 l + p).
// One generic phase engine, parameterized per phase by the Ph table (rows N, K, T bucket, TT, KS, ring R, epilogue mode, finalizer):
//   classic (eng 0): port of mk gemm_phase (128 x 128 units, contiguous unit ranges over all CTAs, B staged by all threads, red.add, ring 2)
//   TS      (eng 1): port of mk ts_phase / ts_first / ts_geom (tile stream, B by TMA from a bf16 buffer, split-K red.add / plain / GU pair act)
//   finalizer: port of mk idle_finalize (placement 0 idle CTAs, 2 all CTAs) fed by tickets, writes the next phase's bf16 B copy
// Designs (host side tables): 1 classic everywhere; 2 TS KS 1 everywhere (GU pair act, plain stores, no finalizer);
//                              3 mk tile-stream tables per bucket (QKV TS KS_q, O classic + finalizer, GU TS pair, D TS KS_d + finalizer).
// pf modes: 0 none (first A issued at phase start), 1 barrier window after bar_arrive (mk), 2 producer side after the CTA's own last step (xover3 style).
// RoPE / attention are spin fillers (--fill rope_ns,attn_ns), no weight traffic.
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -I../common b8_compose.cu
#include "ub.cuh"
#include "host.h"

// ------------------------------------------------------------------ small helpers (mk_common.cuh)
__device__ __forceinline__ void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
__device__ __forceinline__ void named_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ u32 pack_bf16(float a, float b) { u32 r; asm("{ cvt.rn.bf16x2.f32 %0, %2, %1;}\n" : "=r"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float bflo(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bfhi(unsigned u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ float warp_sum(float v) { for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o); return v; }
#define SILU(g) ((g) / (1.f + expf(-(g))))
constexpr float EPS = 1e-6f;
#ifdef B8_DBG   // watchdog build (used for the pf 2 arm): a wait that spins > 3e8 times prints its place and traps
__device__ __forceinline__ void mbw_dbg(u32 a, u32 parity, int line) {
  u32 ok; ull n = 0;
  while (true) {
    asm volatile("{\n\t.reg .pred P;\n\tmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2;\n\tselp.u32 %0, 1, 0, P;\n\t}" : "=r"(ok) : "r"(a), "r"(parity) : "memory");
    if (ok) break;
    if (++n > 300000000ull) { printf("STUCK mbar blk %d tid %d line %d addr %u par %u\n", blockIdx.x, threadIdx.x, line, a, parity); __trap(); }
  }
}
#define MBW(a, p) mbw_dbg((a), (p), __LINE__)
__device__ __forceinline__ void bar_wait_l(const unsigned* cnt, unsigned& target, ull* st, int b) {
  target += gridDim.x;
  if (threadIdx.x == 0) {
    unsigned v; ull n = 0;
    do { asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(cnt) : "memory"); if (++n > 300000000ull) { printf("STUCK grid barrier blk %d bi %d cnt %u target %u\n", blockIdx.x, b, v, target); __trap(); } } while (v < target);
    asm volatile("fence.acquire.gpu;" ::: "memory");
    stamp(st, 2 * b + 1);
  }
  __syncthreads();
}
#define BWAIT(c, t, s, b) bar_wait_l((c), (t), (s), (b))
#define POLL_GUARD(n, tag, v, tgt) do { if (++(n) > 300000000ull) { printf("STUCK poll %s blk %d v %u target %u\n", tag, blockIdx.x, v, tgt); __trap(); } } while (0)
#else
#define MBW(a, p) mbar_wait((a), (p))
#define BWAIT(c, t, s, b) bar_wait((c), (t), (s), (b))
#define POLL_GUARD(n, tag, v, tgt) do { } while (0)
#endif
constexpr int MAXTOK = 256, UB = 32768, RING = 2, STG = 4;
constexpr int LMB_FULL = 192, LMB_EMPTY = 256, LMB_ACCF = 320, LMB_ACCE = 336, TK_D = 16, EXCH_BYTES = 12288;
__device__ __forceinline__ uint4 pack_xw(const float4 a, const float4 b, const uint4 w) {
  return make_uint4(pack_bf16(a.x * bflo(w.x), a.y * bfhi(w.x)), pack_bf16(a.z * bflo(w.y), a.w * bfhi(w.y)),
                    pack_bf16(b.x * bflo(w.z), b.y * bfhi(w.z)), pack_bf16(b.z * bflo(w.w), b.w * bfhi(w.w)));
}
__device__ __forceinline__ void zero_f32(float* p, int total) {   // whole grid, total multiple of 4
  for (int i = (blockIdx.x * NT + threadIdx.x) * 4; i < total; i += gridDim.x * NT * 4) __stcg(reinterpret_cast<float4*>(p + i), make_float4(0.f, 0.f, 0.f, 0.f));
}
__device__ __forceinline__ void rr_of_x(const float* x, float* rr, int ntok) {   // x [T][1024], one token per warp 8..15
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (warp < 8) return;
  const int t = blockIdx.x + gridDim.x * (warp - 8);
  if (t >= ntok) return;
  float4 a[8];
#pragma unroll
  for (int m = 0; m < 8; ++m) a[m] = __ldcg(reinterpret_cast<const float4*>(x + (size_t)t * 1024) + lane + 32 * m);
  float ss = 0.f;
#pragma unroll
  for (int m = 0; m < 8; ++m) ss += a[m].x * a[m].x + a[m].y * a[m].y + a[m].z * a[m].z + a[m].w * a[m].w;
  ss = warp_sum(ss);
  if (lane == 0) __stcg(rr + t, rsqrtf(ss * (1.f / 1024) + EPS));
}

// ------------------------------------------------------------------ tables
enum { K_QKV = 0, K_O = 1, K_GU = 2, K_D = 3, K_FILL = 4 };
struct Ph {
  int kind, eng;            // eng 0 classic, 1 TS
  int N, K, npad, TT, KS, R;  // rows, k, token bucket (classic B tile N; TS NTT), TS token tile, split-K, TS ring depth
  int epi;                  // TS epilogue: 0 red.add, 1 plain, 2 GU pair act
  int fin;                  // finalizer after this phase (O: classic tickets + finalizer, D: TS tickets + finalizer)
  int wmap, bmap, bget;     // W map (K 1024/2048/3072), TS B map, classic staging kind (0 x*ln fp32, 1 bf16 ao, 2 gu silu)
  int slot;                 // phase number within the layer for the pool row (QKV 0, O 1, GU 2, D 3)
  long long rowsAvail;
  int fill_ns;
};
struct Cfg { Ph ph[6]; int nlay, ntok, pf, check, wu, fin_place, p0, xoff, null; };
struct Bufs { float *x0, *xO, *xD, *qkv, *gu, *rr1, *rr2, *ss2; bf16 *xb, *xb2, *act, *ao, *ln1, *ln2, *xbn; unsigned* tk; };

struct GS { u32 base; uint8_t* sm; u32 bars; u32 tb; u32 F; };
struct Maps { const CUtensorMap* w[3]; const CUtensorMap* b[4]; };

__device__ __forceinline__ int prow(const Ph& p, int gp) { return (int)(((long long)gp * p.N) % p.rowsAvail); }
__device__ __forceinline__ int cdiv(int a, int b) { return (a + b - 1) / b; }

// ------------------------------------------------------------------ classic engine (mk gemm_phase)
__device__ __forceinline__ void cl_geom(const Ph& p, int& u0, int& n) {
  const int U = (p.N / 128) * (p.K / 128);
  u0 = (int)(((long long)blockIdx.x * U) / gridDim.x);
  n = (int)(((long long)(blockIdx.x + 1) * U) / gridDim.x) - u0;
}
__device__ __forceinline__ void cl_fill(const GS& g, const Maps& M, const Ph& p, int gp, int u, u32 f) {   // tid 0
  const int KC = p.K / 128, kc = u % KC, rb = u / KC, row = prow(p, gp) + rb * 128;
  const u32 s = f % RING, bar = g.bars + 8 * s;
  mbar_expect_tx(bar, UB);
  tma2(g.base + s * UB, M.w[p.wmap], kc * 128, row, bar);
  tma2(g.base + s * UB + 16384, M.w[p.wmap], kc * 128 + 64, row, bar);
}
__device__ __forceinline__ void cl_prefetch(const GS& g, const Maps& M, const Ph& p, int gp) {   // tid 0; g.F counts the finished classic phase
  int u0, n; cl_geom(p, u0, n);
  for (int i = 0; i < RING && i < n; ++i) cl_fill(g, M, p, gp, u0 + i, g.F + i);
}
// B getter of the classic phases. kind 0: bf16(x * ln) from fp32 x [T][1024]; 1: bf16 ao [T][2048]; 2: bf16(silu(r gate) * (r up)) from gu [T][6144], r = rr2[t]
__device__ __forceinline__ uint4 cl_get(const Ph& p, const Bufs& B, int l, int t, int k, int ntok) {
  if (t >= ntok) return make_uint4(0u, 0u, 0u, 0u);
  if (p.bget == 1) return __ldcg(reinterpret_cast<const uint4*>(B.ao + (size_t)t * p.K + k));
  if (p.bget == 0) {
    const float* x = p.kind == K_QKV ? B.x0 : B.xO;
    const bf16* lnw = (p.kind == K_QKV ? B.ln1 : B.ln2) + (size_t)l * 1024;
    const float4 a = __ldcg(reinterpret_cast<const float4*>(x + (size_t)t * 1024 + k)), b = __ldcg(reinterpret_cast<const float4*>(x + (size_t)t * 1024 + k + 4));
    return pack_xw(a, b, __ldg(reinterpret_cast<const uint4*>(lnw + k)));
  }
  const float r = __ldcg(B.rr2 + t);
  const float* gp = B.gu + (size_t)t * 6144 + k;
  const float4 g0 = __ldcg(reinterpret_cast<const float4*>(gp)), g1 = __ldcg(reinterpret_cast<const float4*>(gp + 4));
  const float4 u0 = __ldcg(reinterpret_cast<const float4*>(gp + 3072)), u1 = __ldcg(reinterpret_cast<const float4*>(gp + 3072 + 4));
  const float ga[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w}, ua[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
  float h[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) { const float gg = r * ga[i]; h[i] = SILU(gg) * (r * ua[i]); }
  return make_uint4(pack_bf16(h[0], h[1]), pack_bf16(h[2], h[3]), pack_bf16(h[4], h[5]), pack_bf16(h[6], h[7]));
}
__device__ __forceinline__ void cl_stage(unsigned char* xs, int kcol, const Ph& p, const Bufs& B, int l, int ntok) {
  const int NP = p.npad, CH = NP * 16, ITER = cdiv(CH, NT), tid = threadIdx.x;
  for (int g0 = 0; g0 < ITER; g0 += STG) {
    uint4 v[STG];
#pragma unroll
    for (int jj = 0; jj < STG; ++jj) { const int i = g0 + jj, c = tid + NT * i; if (i < ITER && c < CH) v[jj] = cl_get(p, B, l, c / 16, kcol + (c % 16) * 8, ntok); }
#pragma unroll
    for (int jj = 0; jj < STG; ++jj) {
      const int i = g0 + jj, c = tid + NT * i;
      if (i < ITER && c < CH) { const int r = c / 16, ch = c % 16; *reinterpret_cast<uint4*>(xs + (ch >> 3) * (NP * 128) + r * 128 + (((ch & 7) ^ (r & 7)) << 4)) = v[jj]; }
    }
  }
  fence_proxy_async();
}
// `first_next`: producer-side issue of the next phase's first A (pf 2), called by tid 0 once all MMAs of this CTA are complete (or at phase start without work)
template <class FN>
__device__ __forceinline__ void cl_phase(GS& g, const Maps& M, const Ph& p, int gp, int l, int ntok, const Bufs& B, bool check, bool do_fin, bool C_null, FN first_next) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, q = warp & 3;
  const int KC = p.K / 128, N = p.N;
  int u0, n; cl_geom(p, u0, n);
  const u32 F0 = g.F; g.F = F0 + n;
  float* out = p.kind == K_QKV ? B.qkv : p.kind == K_O ? B.xO : p.kind == K_GU ? B.gu : B.xD;
  auto extra = [&]() {
    if (p.kind == K_QKV) { rr_of_x(B.x0, B.rr1, ntok); if (!check) zero_f32(B.gu, ntok * 6144); }
    else if (p.kind == K_O) { if (!check) zero_f32(B.qkv, ntok * 4096); }
    else if (p.kind == K_GU) rr_of_x(B.xO, B.rr2, ntok);
  };
  if (C_null) {   // mk NUL: weight TMAs + ring waits only
    for (int j = 0; j < n; ++j) {
      if (tid == 0) { const u32 f = F0 + j; MBW(g.bars + 8 * (f % RING), (f / RING) & 1); if (j + RING < n) cl_fill(g, M, p, gp, u0 + j + RING, f + RING); }
      __syncthreads();
    }
    if (tid == 0) first_next();
    return;
  }
  if (n == 0) { if (tid == 0) first_next(); extra(); return; }
  const u32 idesc = tc_idesc(128, p.npad);
  const int XS = 2 * p.npad * 128;
  unsigned char* const xbase = g.sm + 2 * UB;
  cl_stage(xbase, (u0 % KC) * 128, p, B, l, ntok);
  __syncthreads();
  int jseg = 0;
  for (int j = 0; j < n; ++j) {
    const u32 f = F0 + j, s = f % RING, par = (f / RING) & 1;
    const int u = u0 + j;
    if (tid == 0) {
      MBW(g.bars + 8 * s, par);
      tc_fence_after();
      const bool first = (j == 0) || (u % KC == 0);
      const u32 sr = g.base + s * UB, xb = g.base + 2 * UB + (j & 1) * XS;
      for (int kk = 0; kk < 8; ++kk) { const int a = kk >> 2, off = (kk & 3) * 32; tc_mma(g.tb, sdesc(sr + a * 16384 + off), sdesc(xb + a * (p.npad * 128) + off), idesc, (kk > 0 || !first) ? 1u : 0u); }
      tc_commit(g.bars + 64 + 8 * s);
    }
    if (j == 0) extra();
    if (j + 1 < n) {
      if (j >= 1) {
        MBW(g.bars + 64 + 8 * ((f - 1) % RING), ((f - 1) / RING) & 1);
        if (tid == 0 && j + RING - 1 < n) cl_fill(g, M, p, gp, u0 + j + RING - 1, f + RING - 1);
      }
      cl_stage(xbase + ((j + 1) & 1) * XS, ((u0 + j + 1) % KC) * 128, p, B, l, ntok);
    }
    if (j == n - 1 || (u + 1) % KC == 0) {
      MBW(g.bars + 64 + 8 * s, par);
      tc_fence_after();
      if (j == n - 1 && tid == 0) first_next();   // pf 2: every MMA of this CTA is complete, the ring is free
      const int rb = u / KC, nch = (ntok + 15) / 16;
      for (int c = warp >> 2; c < nch; c += 4) {
        u32 v0[16];
        tmem_ld16(v0, g.tb + ((u32)(q * 32) << 16) + c * 16);
        tc_wait_ld();
        const int row = rb * 128 + 32 * q + lane;
#pragma unroll
        for (int i = 0; i < 16; ++i) if (c * 16 + i < ntok) red_add(out + (size_t)(c * 16 + i) * N + row, __uint_as_float(v0[i]));
      }
      tc_fence_before();
      if (do_fin) {   // O tickets: one per token tile, nk units of row block rb
        __syncthreads();
        if (tid == 0) {
          __threadfence();
          const int ntt = (p.npad + 127) / 128;
          for (int tt = 0; tt < ntt; ++tt) atomicAdd(B.tk + rb * ntt + tt, (unsigned)(j - jseg + 1));
        }
      }
      jseg = j + 1;
    }
    __syncthreads();
  }
}

// ------------------------------------------------------------------ tile-stream engine (mk ts_phase)
__device__ __forceinline__ int ts_ntt(const Ph& p) { return p.npad >= p.TT ? (p.npad + p.TT - 1) / p.TT : 1; }
__device__ __forceinline__ int ts_ntile(const Ph& p) { return (p.N / 128) * ts_ntt(p) * p.KS; }
__device__ __forceinline__ void ts_geom(const Ph& p, int& t0, int& nt) {
  const int NT_ = ts_ntile(p);
  t0 = (int)(((long long)blockIdx.x * NT_) / gridDim.x);
  nt = (int)(((long long)(blockIdx.x + 1) * NT_) / gridDim.x) - t0;
}
__device__ __forceinline__ void ts_fill_a(const GS& g, const Maps& M, const Ph& p, int gp, int t0, int j, int s) {
  const int spt = p.K / (p.KS * 128), kl = p.K / p.KS, ntt = ts_ntt(p);
  const int t = t0 + j / spt, kc = (t % p.KS) * kl + (j % spt) * 128, rb = (t / p.KS) / ntt;
  const int row = prow(p, gp) + rb * 128;
  const u32 bar = g.bars + LMB_FULL + 8 * s, slot = 32768 + p.TT * 256;
  mbar_expect_tx(bar, UB);
  tma2(g.base + s * slot, M.w[p.wmap], kc, row, bar);
  tma2(g.base + s * slot + 16384, M.w[p.wmap], kc + 64, row, bar);
}
__device__ __forceinline__ void ts_fill_b(const GS& g, const Maps& M, const Ph& p, int t0, int j, int s) {
  const int spt = p.K / (p.KS * 128), kl = p.K / p.KS, ntt = ts_ntt(p);
  const int t = t0 + j / spt, kc = (t % p.KS) * kl + (j % spt) * 128, tt = (t / p.KS) % ntt;
  const u32 bar = g.bars + LMB_FULL + 8 * s, slot = 32768 + p.TT * 256;
  mbar_expect_tx(bar, p.TT * 256);
  tma2(g.base + s * slot + 32768, M.b[p.bmap], kc, tt * p.TT, bar);
  tma2(g.base + s * slot + 32768 + p.TT * 128, M.b[p.bmap], kc + 64, tt * p.TT, bar);
}
// tid 0: mbarriers of the phase (full / empty re-init: part 1; accumulator barriers: part 2) and the A boxes of the first min(R, nst) steps (part 3)
__device__ __forceinline__ void ts_init_fe(const GS& g, const Ph& p) {
  const u32 fullb = g.bars + LMB_FULL, emptyb = g.bars + LMB_EMPTY;
  for (int i = 0; i < p.R; ++i) { mbar_init(fullb + 8 * i, 2); mbar_init(emptyb + 8 * i, 1); }
}
__device__ __forceinline__ void ts_init_acc(const GS& g) {
  const u32 accf = g.bars + LMB_ACCF, acce = g.bars + LMB_ACCE;
  for (int b = 0; b < 2; ++b) { mbar_init(accf + 8 * b, 1); mbar_init(acce + 8 * b, 12); }
}
__device__ __forceinline__ void ts_fills(const GS& g, const Maps& M, const Ph& p, int gp) {
  int t0, nt; ts_geom(p, t0, nt);
  const int spt = p.K / (p.KS * 128), nst = nt * spt;
  for (int i = 0; i < p.R && i < nst; ++i) ts_fill_a(g, M, p, gp, t0, i, i);
}
__device__ __forceinline__ void ts_first(const GS& g, const Maps& M, const Ph& p, int gp) {   // mk ts_first (pf 1 / 0)
  ts_init_fe(g, p); ts_init_acc(g); mbar_fence_init(); ts_fills(g, M, p, gp);
}

// GU pair epilogue, as mk P5b (q < 0: per-token setup)
__device__ __forceinline__ void gu_pair_epi(const GS& g, const Cfg& C, const Bufs& B, const Ph& p, float* s_ss, int q, int lane, int grp, int rb, int cb, const u32 (&v)[16], int ntok) {
  const int tid = threadIdx.x;
  if (q < 0) {
    for (int t = tid - 128; t < ntok; t += 384) s_ss[t] = rsqrtf(__ldcg(B.ss2 + t) * (1.f / p.K) + EPS);
    named_bar_sync(7, 384);
    return;
  }
  const int pid = grp * 2 + (q & 1);
  float* ex = reinterpret_cast<float*>(g.sm + C.xoff) + pid * 512;
  if (q >= 2) {
#pragma unroll
    for (int i = 0; i < 16; ++i) ex[i * 32 + lane] = __uint_as_float(v[i]);
    named_bar_sync(1 + pid, 64);
    named_bar_sync(1 + pid, 64);
  } else {
    float ua[16];
    named_bar_sync(1 + pid, 64);
#pragma unroll
    for (int i = 0; i < 16; ++i) ua[i] = ex[i * 32 + lane];
    named_bar_sync(1 + pid, 64);
    const int ch = 64 * rb + 32 * q + lane;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int t = cb + i;
      if (t < ntok) {
        const float r = s_ss[t], gg = r * __uint_as_float(v[i]), h = SILU(gg) * (r * ua[i]);
        B.act[(size_t)t * (p.N / 2) + ch] = (bf16)__float2bfloat16_rn(h);
      }
    }
  }
}

// FN: producer-side first issue of the next phase (pf 2), run by the producer after its drain
template <class FN>
__device__ __forceinline__ void ts_phase(GS& g, const Maps& M, const Ph& p, int gp, const Cfg& C, const Bufs& B, int ntok, float* s_ss, FN first_next) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, q = warp & 3;
  const int TT = p.TT, R = p.R, KS = p.KS, N = p.N;
  const int spt = p.K / (KS * 128), NCH = TT / 16, ntt = ts_ntt(p), slotb = 32768 + TT * 256;
  int t0, nt; ts_geom(p, t0, nt);
  const int nst = nt * spt;
  const u32 fullb = g.bars + LMB_FULL, emptyb = g.bars + LMB_EMPTY, accf = g.bars + LMB_ACCF, acce = g.bars + LMB_ACCE;
  const bool wu = C.wu != 0;
  float* out = p.kind == K_QKV ? B.qkv : p.kind == K_O ? B.xO : B.xD;
  if (wu ? warp == 0 : tid == 0) {
    if (!wu || lane == 0) {   // producer
      fence_proxy_async_global();
      for (int i = 0; i < R && i < nst; ++i) ts_fill_b(g, M, p, t0, i, i);
      if (C.null) {   // mk NUL: producer + full waits only
        for (int j = 0; j < nst; ++j) {
          MBW(fullb + 8 * (j % R), (j / R) & 1);
          if (j + R < nst) { ts_fill_a(g, M, p, gp, t0, j + R, j % R); ts_fill_b(g, M, p, t0, j + R, j % R); }
        }
      } else {
        for (int j = R; j < nst; ++j) {
          const int s = j % R;
          MBW(emptyb + 8 * s, ((j / R) - 1) & 1);
          ts_fill_a(g, M, p, gp, t0, j, s);
          ts_fill_b(g, M, p, t0, j, s);
        }
        for (int j = nst > R ? nst - R : 0; j < nst; ++j) MBW(emptyb + 8 * (j % R), (j / R) & 1);   // drain
      }
      first_next();
    }
    if (wu) __syncwarp();
  } else if (!C.null && (wu ? warp == 1 : tid == 32)) {
    if (!wu || lane == 0) {   // MMA issuer
      const u32 idesc = tc_idesc(128, TT);
      for (int ti = 0; ti < nt; ++ti) {
        const int b = ti & 1;
        if (ti >= 2) MBW(acce + 8 * b, ((ti >> 1) - 1) & 1);
        for (int k = 0; k < spt; ++k) {
          const int f = ti * spt + k, s = f % R;
          MBW(fullb + 8 * s, (f / R) & 1);
          tc_fence_after();
          const u32 sa = g.base + s * slotb, sb = sa + 32768;
#pragma unroll
          for (int kk = 0; kk < 8; ++kk) { const int a = kk >> 2, off = (kk & 3) * 32; tc_mma(g.tb + b * TT, sdesc(sa + a * 16384 + off), sdesc(sb + a * (TT * 128) + off), idesc, (k > 0 || kk > 0) ? 1u : 0u); }
          tc_commit(emptyb + 8 * s);
          if (k == spt - 1) tc_commit(accf + 8 * b);
        }
      }
    }
    if (wu) __syncwarp();
  } else if (!C.null && warp >= 4) {   // epilogue warps
    if (p.epi == 2) { u32 none[16]; if (nt > 0) gu_pair_epi(g, C, B, p, s_ss, -1, lane, 0, 0, 0, none, ntok); }
    for (int ti = 0; ti < nt; ++ti) {
      const int b = ti & 1, t = t0 + ti, rb = (t / KS) / ntt, tt = (t / KS) % ntt;
      MBW(accf + 8 * b, (ti >> 1) & 1);
      tc_fence_after();
      const int row = rb * 128 + 32 * q + lane;
      if (p.epi == 2) {
        const int grp = (warp >> 2) - 1;
        for (int c = grp; c < NCH; c += 3) {
          const int cb = tt * TT + c * 16;
          if (cb >= ntok) break;
          u32 v0[16];
          tmem_ld16(v0, g.tb + b * TT + ((u32)(q * 32) << 16) + c * 16);
          tc_wait_ld();
          gu_pair_epi(g, C, B, p, s_ss, q, lane, grp, rb, cb, v0, ntok);
        }
      } else {
        for (int c = (warp >> 2) - 1; c < NCH; c += 6) {
          const int cb = tt * TT + c * 16;
          if (cb >= ntok) break;
          const bool two = (c + 3 < NCH) && (cb + 48 < ntok);
          u32 v0[16], v1[16];
          tmem_ld16(v0, g.tb + b * TT + ((u32)(q * 32) << 16) + c * 16);
          if (two) tmem_ld16(v1, g.tb + b * TT + ((u32)(q * 32) << 16) + (c + 3) * 16);
          tc_wait_ld();
          if (p.epi == 0) {
#pragma unroll
            for (int i = 0; i < 16; ++i) if (cb + i < ntok) red_add(out + (size_t)(cb + i) * N + row, __uint_as_float(v0[i]));
            if (two) {
#pragma unroll
              for (int i = 0; i < 16; ++i) if (cb + 48 + i < ntok) red_add(out + (size_t)(cb + 48 + i) * N + row, __uint_as_float(v1[i]));
            }
          } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) if (cb + i < ntok) out[(size_t)(cb + i) * N + row] = __uint_as_float(v0[i]);
            if (two) {
#pragma unroll
              for (int i = 0; i < 16; ++i) if (cb + 48 + i < ntok) out[(size_t)(cb + 48 + i) * N + row] = __uint_as_float(v1[i]);
            }
          }
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(acce + 8 * b);
    }
  }
  __syncthreads();
  tc_fence_after();
}

// ------------------------------------------------------------------ finalizer (mk idle_finalize, FIN 0 and 2)
// is_o: x = xO, target = K_O / 128 units per row block, xb2 + ss2 (O); else x = xD, target = KS (D), no ss. ntile = tiles / units of the producing phase.
__device__ __forceinline__ void finalize(const Cfg& C, const Bufs& B, const Ph& p, bool is_o, int l, int ntok) {
  const int ntt = (p.npad + 127) / 128, target = is_o ? p.K / 128 : p.KS, ntile = is_o ? (p.N / 128) * (p.K / 128) : ts_ntile(p);
  const float* x = is_o ? B.xO : B.xD;
  const bf16* lnw = is_o ? B.ln2 + (size_t)l * 1024 : B.ln1 + (size_t)(l + 1) * 1024;
  bf16* outp = is_o ? B.xb2 : B.xbn;
  float* ss2 = is_o ? B.ss2 : nullptr;
  unsigned* tk = B.tk;
  const int NB_ = p.N / 128, H = p.N;
  int rank, nidle;
  if (C.fin_place == 2) { rank = blockIdx.x; nidle = gridDim.x; }
  else {
    const int b = blockIdx.x, G = gridDim.x, lo = (b * ntile) / G, hi = ((b + 1) * ntile) / G;
    nidle = G - ntile; rank = hi == lo ? b - lo : -1;
    if (rank < 0) return;
  }
  const int tid = threadIdx.x;
  if (C.fin_place == 0) {
    for (int job = rank; job < NB_ * ntt; job += nidle) {
      const int rb = job / ntt, tt = job % ntt;
      unsigned* tick = tk + (is_o ? 0 : TK_D) + job;
      if (tid == 0) {
        unsigned v; ull pg = 0;
        do { asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(tick) : "memory"); if (v != (unsigned)target) __nanosleep(64); POLL_GUARD(pg, "fin", v, (unsigned)target); } while (v != (unsigned)target);
        atomicExch(tick, 0u);
      }
      __syncthreads();
      const int ntl = min(ntok - tt * 128, 128), items = ntl > 0 ? ntl * 16 : 0;
      const bf16* lw = lnw + rb * 128;
      for (int i0 = 0; i0 < items; i0 += NT) {
        const int i = i0 + tid, t = tt * 128 + (i >> 4), cg = i & 15;
        const bool ok = i < items;
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = a;
        if (ok) { const float* xp = x + (size_t)t * H + rb * 128 + cg * 8; a = __ldcg(reinterpret_cast<const float4*>(xp)); b = __ldcg(reinterpret_cast<const float4*>(xp + 4)); }
        if (ok) *reinterpret_cast<uint4*>(outp + (size_t)t * H + rb * 128 + cg * 8) = pack_xw(a, b, __ldg(reinterpret_cast<const uint4*>(lw + cg * 8)));
        if (ss2) {
          float sq = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
          for (int o = 8; o; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
          if (ok && cg == 0) atomicAdd(ss2 + t, sq);
        }
      }
    }
  } else {
    const int NJOB = NB_ * ntt, nsub = max(1, nidle / NJOB), pr = (128 + nsub - 1) / nsub;
    for (int w = rank; w < NJOB * nsub; w += nidle) {
      const int job = w % NJOB, part = w / NJOB, rb = job / ntt, tt = job % ntt;
      unsigned* tick = tk + (is_o ? 0 : TK_D) + job;
      if (tid == 0) {
        unsigned v; ull pg = 0;
        do { asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(tick) : "memory"); if (v != (unsigned)target) __nanosleep(64); POLL_GUARD(pg, "fin", v, (unsigned)target); } while (v != (unsigned)target);
        if (nsub == 1) atomicExch(tick, 0u);
        else { unsigned* seen = tk + 32 + (is_o ? 0 : 16) + job; if (atomicAdd(seen, 1u) == (unsigned)nsub - 1) { atomicExch(seen, 0u); atomicExch(tick, 0u); } }
      }
      __syncthreads();
      const int ntl = min(ntok - tt * 128, 128), p0 = part * pr, np = max(0, min(ntl - p0, pr)), items = np * 16;
      const bf16* lw = lnw + rb * 128;
      for (int i0 = 0; i0 < items; i0 += NT) {
        const int i = i0 + tid, t = tt * 128 + p0 + (i >> 4), cg = i & 15;
        const bool ok = i < items;
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = a;
        if (ok) { const float* xp = x + (size_t)t * H + rb * 128 + cg * 8; a = __ldcg(reinterpret_cast<const float4*>(xp)); b = __ldcg(reinterpret_cast<const float4*>(xp + 4)); }
        if (ok) *reinterpret_cast<uint4*>(outp + (size_t)t * H + rb * 128 + cg * 8) = pack_xw(a, b, __ldg(reinterpret_cast<const uint4*>(lw + cg * 8)));
        if (ss2) {
          float sq = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
          for (int o = 8; o; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
          if (ok && cg == 0) atomicAdd(ss2 + t, sq);
        }
      }
    }
  }
  fence_proxy_async_global();
}

// ------------------------------------------------------------------ kernel
__global__ void __launch_bounds__(NT, 1) k_b8(const __grid_constant__ CUtensorMap mW0, const __grid_constant__ CUtensorMap mW1, const __grid_constant__ CUtensorMap mW2,
                                              const __grid_constant__ CUtensorMap mB0, const __grid_constant__ CUtensorMap mB1, const __grid_constant__ CUtensorMap mB2,
                                              const __grid_constant__ CUtensorMap mB3, Bufs B, unsigned* cnt, ull* st, ull* ck, const __grid_constant__ Cfg C) {
  extern __shared__ uint8_t smraw[];
  __shared__ float s_ss[MAXTOK];
  __shared__ u32 s_tslot;
  const int tid = threadIdx.x, warp = tid >> 5;
  const u32 raw = smem_u32(smraw), base = (raw + 1023) & ~1023u;
  uint8_t* sm = smraw + (base - raw);
  const u32 bars = base + C.xoff + EXCH_BYTES;
  Maps M = {{&mW0, &mW1, &mW2}, {&mB0, &mB1, &mB2, &mB3}};
  clk_sample(ck, 0);
  if (warp == 0) tmem_alloc(smem_u32((void*)&s_tslot), 256);
  if (tid == 0) {
    for (int i = 0; i < RING; ++i) { mbar_init(bars + 8 * i, 1); mbar_init(bars + 64 + 8 * i, 1); }
    mbar_fence_init();
  }
  for (int i = tid; i < MAXTOK; i += NT) s_ss[i] = 0.f;
  __syncthreads();
  tc_fence_after();
  GS g; g.base = base; g.sm = sm; g.bars = bars; g.tb = *(volatile u32*)&s_tslot; g.F = 0;
  const int ntok = C.ntok, nlay = C.nlay, pf = C.pf;
  unsigned target = 0; int bi = 0;
  const Ph* P = C.ph;
  // issue the first A of phase P[idx] (global phase number gp): classic prefetch / TS first. `full` = with mbarrier init + acc init (window), `fe` = full / empty only
  auto first = [&](int idx, int gp) {
    if (tid != 0) return;
    if (P[idx].eng == 0) cl_prefetch(g, M, P[idx], gp); else ts_first(g, M, P[idx], gp);
  };
  auto first_pf2 = [&](int idx, int gp) {   // producer side: full / empty init + A boxes, acc barriers are re-initialised in the window
    if (P[idx].eng == 0) cl_prefetch(g, M, P[idx], gp); else { ts_init_fe(g, P[idx]); mbar_fence_init(); ts_fills(g, M, P[idx], gp); }
  };
  auto gpn = [&](int l, int slot) { return C.p0 + 4 * l + slot; };
  // phase 0 window (layer 0 QKV)
  first(0, gpn(0, 0));
  bar_arrive(cnt, st, bi);
  BWAIT(cnt, target, st, bi); ++bi;
  // window helper after a phase: arrive, (thread 0) first of the next phase, wait
  for (int l = 0; l < nlay; ++l) {
    for (int ph = 0; ph < 6; ++ph) {
      const Ph& p = P[ph];
      int nidx = -1, ngp = 0;   // next GEMM phase for the prefetch
      if (ph == 0) { nidx = 3; ngp = gpn(l, 1); }
      else if (ph == 3) { nidx = 4; ngp = gpn(l, 2); }
      else if (ph == 4) { nidx = 5; ngp = gpn(l, 3); }
      else if (ph == 5 && l + 1 < nlay) { nidx = 0; ngp = gpn(l + 1, 0); }
      const int gp = ph == 0 ? gpn(l, 0) : ph == 3 ? gpn(l, 1) : ph == 4 ? gpn(l, 2) : gpn(l, 3);
      if (p.kind == K_FILL) {
        if (tid == 0) spin_ns((ull)p.fill_ns);
        bar_arrive(cnt, st, bi); BWAIT(cnt, target, st, bi); ++bi;
        continue;
      }
      auto fn = [&]() { if (pf == 2 && nidx >= 0) first_pf2(nidx, ngp); };   // producer-side (tid 0 / producer thread)
      if (pf == 0 && tid == 0) first(ph, gp);   // no prefetch: this phase's first A at its start
      if (pf == 0) __syncthreads();
      if (p.eng == 0) {
        cl_phase(g, M, p, gp, l, ntok, B, C.check != 0, p.fin != 0 && !C.null, C.null != 0, fn);
        if (p.fin && !C.null) finalize(C, B, p, true, l, ntok);
      } else {
        ts_phase(g, M, p, gp, C, B, ntok, s_ss, fn);
        if (C.null) {}
        else if (p.kind == K_QKV) {
          rr_of_x(B.x0, B.rr1, ntok);
          if (!C.check && P[3].fin) zero_f32(B.ss2, ntok);
        } else if (p.kind == K_O) { if (!C.check) zero_f32(B.qkv, ntok * 4096); }
        else if (p.kind == K_GU) { for (int t = tid; t < ntok; t += NT) s_ss[t] = 0.f; if (warp >= 4) fence_proxy_async_global(); }
        else if (p.kind == K_D && p.fin && (l + 1 < nlay || C.check)) {
          if (tid == 0) {
            int t0, nt; ts_geom(p, t0, nt);
            if (nt > 0) { __threadfence(); for (int ti = 0; ti < nt; ++ti) atomicAdd(B.tk + TK_D + (t0 + ti) / p.KS, 1u); }
          }
          finalize(C, B, p, false, l, ntok);
        }
      }
      bar_arrive(cnt, st, bi);
      if (pf == 1 && nidx >= 0) first(nidx, ngp);              // barrier window (mk)
      if (pf == 2 && tid == 0 && nidx >= 0 && P[nidx].eng == 1) ts_init_acc(g);   // acc barriers of the next TS phase (first_pf2 did full / empty + A)
      BWAIT(cnt, target, st, bi); ++bi;
    }
  }
  __syncthreads();
  tc_fence_after();
  if (warp == 0) tmem_dealloc(g.tb, 256);
  clk_sample(ck, 1);
}

// ------------------------------------------------------------------ CUDA-core references and comparisons (rule 8)
__global__ void k_cvt(const float* x, const bf16* ln, bf16* out, int T, int K) {   // out = bf16_rn(x * ln[k])
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)T * K) return;
  out[i] = __float2bfloat16_rn(x[i] * __bfloat162float(ln[i % K]));
}
__global__ void k_cvt_d(const float* gu, const float* rr2, bf16* out, int T) {   // classic D staging: bf16(silu(r g) * (r u))
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)T * 3072) return;
  const int t = (int)(i / 3072), k = (int)(i % 3072); const float r = rr2[t];
  const float gg = r * gu[(size_t)t * 6144 + k];
  out[i] = __float2bfloat16_rn(SILU(gg) * (r * gu[(size_t)t * 6144 + 3072 + k]));
}
__global__ void k_refg(const bf16* pool, int row0, int K, const bf16* Bm, int T, int N, float* R, float* S) {   // R[t][n] = sum_k B[t][k] W[row0 + n][k]
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)T * N) return;
  const int t = (int)(i % T), n = (int)(i / T);
  float acc = 0.f, sa = 0.f;
  for (int k = 0; k < K; ++k) { const float pr = __bfloat162float(Bm[(size_t)t * K + k]) * __bfloat162float(pool[(size_t)(row0 + n) * K + k]); acc += pr; sa += fabsf(pr); }
  R[(size_t)t * N + n] = acc; S[(size_t)t * N + n] = sa;
}
__global__ void k_cmp(const float* a, const float* r, const float* S, const float* base, size_t n, float rtol, unsigned* maxerr, unsigned* nbad) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= n) return;
  const float ref = r[i] + (base ? base[i] : 0.f), d = fabsf(a[i] - ref);
  atomicMax(maxerr, __float_as_uint(d));
  if (!(d <= rtol * S[i] + 1e-4f * fmaxf(1.f, fabsf(base ? base[i] : 0.f)))) atomicAdd(nbad, 1u);   // tolerance relative to sum |b w| (fp32 accumulation order)
}
__global__ void k_cmp_gu(const bf16* act, const float* R, const float* ss2, int T, unsigned* maxerr, unsigned* nbad) {   // act[t][64 rb + c] vs silu(r g) * (r u)
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)T * 3072) return;
  const int t = (int)(i / 3072), ch = (int)(i % 3072), rb = ch / 64, c = ch % 64;
  const float r = rsqrtf(ss2[t] * (1.f / 1024) + EPS), gg = r * R[(size_t)t * 6144 + 128 * rb + c], uu = r * R[(size_t)t * 6144 + 128 * rb + 64 + c];
  const float ref = SILU(gg) * uu, d = fabsf(__bfloat162float(act[i]) - ref);
  atomicMax(maxerr, __float_as_uint(d));
  if (!(d <= 8e-3f * fabsf(ref) + 1e-3f)) atomicAdd(nbad, 1u);
}
__global__ void k_cmp_bf(const bf16* a, const bf16* b, size_t n, unsigned* nbad) { const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n && __bfloat16_as_ushort(a[i]) != __bfloat16_as_ushort(b[i])) atomicAdd(nbad, 1u); }
__global__ void k_cmp_ss(const float* x, const float* ss2, int T, unsigned* nbad) {   // ss2[t] vs sum_k x[t][k]^2
  const int t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= T) return;
  float s = 0.f; for (int k = 0; k < 1024; ++k) s += x[(size_t)t * 1024 + k] * x[(size_t)t * 1024 + k];
  if (!(fabsf(ss2[t] - s) <= 1e-3f * fmaxf(1.f, s))) atomicAdd(nbad, 1u);
}

// ------------------------------------------------------------------ host
static std::vector<double> dlist(const char* s) { std::vector<double> v; char* d = strdup(s); for (char* t = strtok(d, ","); t; t = strtok(nullptr, ",")) v.push_back(atof(t)); return v; }
static const char* garg(int c, char** v, const char* k, const char* d) { for (int i = 1; i + 1 < c; ++i) if (!strcmp(v[i], k)) return v[i + 1]; return d; }
static int npad_of(int T) { return T <= 16 ? 16 : T <= 32 ? 32 : T <= 64 ? 64 : T <= 128 ? 128 : 256; }
static const int TS_AVAIL = 227 * 1024 - 11280 - 8 * (2 * (1 + 6 * 28) + 8) - 1024 - 384 - EXCH_BYTES;   // mk TS_SMEM_AVAIL (204704)
static int ringn(int tt) { return std::min(TS_AVAIL / (32768 + tt * 256), 8); }

struct Opt { int ksq = 0, ksd = 0, fin = -1, wu = -1, nlay = 28; long long fill_rope = 1100, fill_attn = 3700; };
static Cfg make_cfg(int design, int T, int pf, int null, const Opt& o, long long rows1024, long long rows2048, long long rows3072, int check) {
  Cfg C; memset(&C, 0, sizeof C);
  const int np = npad_of(T);
  const int TTq = std::min(np, 128), KSq = o.ksq ? o.ksq : (np == 256 ? 2 : 4), TTg = np == 256 ? 96 : np == 128 ? 48 : np, TTd = TTq, KSd = o.ksd ? o.ksd : (np == 256 ? 8 : 12);
  auto mk = [&](int kind, int eng, int N, int K, int TT, int KS, int epi, int fin, int bmap, int bget, int slot) {
    Ph p; memset(&p, 0, sizeof p);
    p.kind = kind; p.eng = eng; p.N = N; p.K = K; p.npad = np; p.TT = TT; p.KS = KS; p.R = eng ? ringn(TT) : 0; p.epi = epi; p.fin = fin;
    p.wmap = K == 1024 ? 0 : K == 2048 ? 1 : 2; p.bmap = bmap; p.bget = bget; p.slot = slot;
    p.rowsAvail = (K == 1024 ? rows1024 : K == 2048 ? rows2048 : rows3072) - N;
    return p;
  };
  Ph fl; memset(&fl, 0, sizeof fl); fl.kind = K_FILL;
  C.ph[1] = fl; C.ph[1].fill_ns = (int)o.fill_rope; C.ph[2] = fl; C.ph[2].fill_ns = (int)o.fill_attn;
  if (design == 1) {
    C.ph[0] = mk(K_QKV, 0, 4096, 1024, 0, 0, 0, 0, 0, 0, 0);
    C.ph[3] = mk(K_O, 0, 1024, 2048, 0, 0, 0, 0, 1, 1, 1);
    C.ph[4] = mk(K_GU, 0, 6144, 1024, 0, 0, 0, 0, 2, 0, 2);
    C.ph[5] = mk(K_D, 0, 1024, 3072, 0, 0, 0, 0, 3, 2, 3);
  } else if (design == 2) {
    C.ph[0] = mk(K_QKV, 1, 4096, 1024, TTq, 1, 1, 0, 0, 0, 0);
    C.ph[3] = mk(K_O, 1, 1024, 2048, TTq, 1, 1, 0, 1, 0, 1);
    C.ph[4] = mk(K_GU, 1, 6144, 1024, TTg, 1, 2, 0, 2, 0, 2);
    C.ph[5] = mk(K_D, 1, 1024, 3072, TTd, 1, 1, 0, 3, 0, 3);
  } else {
    C.ph[0] = mk(K_QKV, 1, 4096, 1024, TTq, KSq, KSq > 1 ? 0 : 1, 0, 0, 0, 0);
    C.ph[3] = mk(K_O, 0, 1024, 2048, 0, 0, 0, 1, 1, 1, 1);
    C.ph[4] = mk(K_GU, 1, 6144, 1024, TTg, 1, 2, 0, 2, 0, 2);
    C.ph[5] = mk(K_D, 1, 1024, 3072, TTd, KSd, KSd > 1 ? 0 : 1, 1, 3, 0, 3);
  }
  // smem: region below the exchange area = max(classic 2 ring slots + 2 B slots, largest TS ring)
  int reg = 0;
  for (int i : {0, 3, 4, 5}) { const Ph& p = C.ph[i]; reg = std::max(reg, p.eng ? p.R * (32768 + p.TT * 256) : 2 * UB + 2 * (2 * p.npad * 128)); }
  C.xoff = (reg + 1023) / 1024 * 1024;
  C.nlay = check ? 1 : o.nlay; C.ntok = T; C.pf = pf; C.check = check; C.null = null;
  C.wu = o.wu >= 0 ? o.wu : (np == 128);
  C.fin_place = o.fin >= 0 ? o.fin : (np >= 128 ? 2 : 0);
  return C;
}
static int ts_ntile_host(const Ph& p) { const int ntt = p.npad >= p.TT ? (p.npad + p.TT - 1) / p.TT : 1; return (p.N / 128) * ntt * p.KS; }
static int smem_of(const Cfg& C) { return 1024 + C.xoff + EXCH_BYTES + 512; }

int main(int argc, char** argv) {
  const char* outst = garg(argc, argv, "--out", nullptr);
  const int nl = atoi(garg(argc, argv, "--launches", "5")), warm = atoi(garg(argc, argv, "--warm", "2"));
  const char* tag = garg(argc, argv, "--tag", ""); const char* job = getenv("SLURM_JOB_ID");
  Opt o; o.ksq = atoi(garg(argc, argv, "--ksq", "0")); o.ksd = atoi(garg(argc, argv, "--ksd", "0")); o.fin = atoi(garg(argc, argv, "--fin", "-1")); o.wu = atoi(garg(argc, argv, "--wu", "-1")); o.nlay = atoi(garg(argc, argv, "--nlay", "28"));
  const std::vector<double> Ts = dlist(garg(argc, argv, "--T", "1,8,16,32,64,96,128,192,256")), designs = dlist(garg(argc, argv, "--design", "1,2,3")), pfs = dlist(garg(argc, argv, "--pf", "1")), nulls = dlist(garg(argc, argv, "--null", "0"));
  const std::vector<double> checkT = dlist(garg(argc, argv, "--checkT", "8,96,192"));
  const int docheck = atoi(garg(argc, argv, "--check", "1")), dotime = atoi(garg(argc, argv, "--time", "1"));
  // filler durations "T:rope_ns:attn_ns;..." (default o.fill_*); the part before the first ':' is T
  std::vector<std::vector<double>> fills; { char* d = strdup(garg(argc, argv, "--fill", "")); char *s1 = nullptr, *s2 = nullptr; for (char* t = strtok_r(d, ";", &s1); t; t = strtok_r(nullptr, ";", &s1)) { std::vector<double> v; for (char* u = strtok_r(t, ":", &s2); u; u = strtok_r(nullptr, ":", &s2)) v.push_back(atof(u)); fills.push_back(v); } /* fixed after job 14306: nested strtok applied only the first entry */ }
  int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  if (nsm != GRID) fprintf(stderr, "warning: %d SMs (expected %d)\n", nsm, GRID);
  Pool P = pool_alloc();
  const long long r1 = P.bytes / 2 / 1024, r2 = P.bytes / 2 / 2048, r3 = P.bytes / 2 / 3072;
  CUtensorMap mW0 = tmap_bf16(P.p, 1024, r1, 64, 128), mW1 = tmap_bf16(P.p, 2048, r2, 64, 128), mW2 = tmap_bf16(P.p, 3072, r3, 64, 128);
  Bufs B; memset(&B, 0, sizeof B);
  float *xinit, *xO, *xD, *R, *R2; unsigned *cnt, *tk, *ver; ull *st, *ck; bf16 *Bref, *aoI, *xb2I;
  CK(cudaMalloc(&B.x0, 256 * 1024 * 4)); CK(cudaMalloc(&xinit, 256 * 1024 * 4)); CK(cudaMalloc(&xO, 256 * 1024 * 4)); CK(cudaMalloc(&xD, 256 * 1024 * 4));
  CK(cudaMalloc(&B.qkv, 256 * 4096 * 4)); CK(cudaMalloc(&B.gu, 256 * 6144 * 4)); CK(cudaMalloc(&B.rr1, 256 * 4)); CK(cudaMalloc(&B.rr2, 256 * 4)); CK(cudaMalloc(&B.ss2, 256 * 4));
  CK(cudaMalloc(&B.xb, 256 * 1024 * 2)); CK(cudaMalloc(&B.xb2, 256 * 1024 * 2)); CK(cudaMalloc(&B.xbn, 256 * 1024 * 2)); CK(cudaMalloc(&B.act, 256 * 3072 * 2)); CK(cudaMalloc(&B.ao, 256 * 2048 * 2));
  CK(cudaMalloc(&B.ln1, 30 * 1024 * 2)); CK(cudaMalloc(&B.ln2, 30 * 1024 * 2)); CK(cudaMalloc(&Bref, 256 * 3072 * 2)); CK(cudaMalloc(&R, 256 * 6144 * 4)); CK(cudaMalloc(&R2, 256 * 6144 * 4));
  CK(cudaMalloc(&aoI, 256 * 2048 * 2)); CK(cudaMalloc(&xb2I, 256 * 1024 * 2));
  CK(cudaMalloc(&cnt, 1024)); CK(cudaMalloc(&tk, 4096)); CK(cudaMalloc(&ver, 64)); B.tk = tk;
  CK(cudaMalloc(&st, (size_t)GRID * NSTMAX * 8)); CK(cudaMalloc(&ck, GRID * 4 * 8));
  k_init_f32<<<64, 256>>>(xinit, 256 * 1024, 11u); k_init_bf16<<<64, 256>>>(B.ao, 256 * 2048, 22u); k_init_bf16<<<64, 256>>>(aoI, 256 * 2048, 22u);
  k_init_bf16<<<64, 256>>>(B.ln1, 30 * 1024, 33u); k_init_bf16<<<64, 256>>>(B.ln2, 30 * 1024, 44u); k_init_bf16<<<64, 256>>>(xb2I, 256 * 1024, 55u);
  CK(cudaDeviceSynchronize());
  CK(cudaFuncSetAttribute(k_b8, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX - 2048));
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, k_b8));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  long long st_off = 0; int p0 = 0;
  const char* dn[4] = {"", "classic", "ts_ks1", "ts_split"};
  auto maps_for = [&](const Cfg& C, CUtensorMap* mb) {
    const void* bufs[4] = {B.xb, B.ao, B.xb2, B.act}; const int Ks[4] = {1024, 2048, 1024, 3072};
    for (int i = 0; i < 4; ++i) { const Ph& p = C.ph[i == 0 ? 0 : i == 1 ? 3 : i == 2 ? 4 : 5]; mb[i] = tmap_bf16(bufs[i], Ks[i], 256, 64, p.eng ? p.TT : 16); }
  };
  auto launch = [&](Cfg C) {
    CUtensorMap mb[4]; maps_for(C, mb);
    CK(cudaMemset(cnt, 0, 1024)); CK(cudaMemset(st, 0, (size_t)GRID * NSTMAX * 8)); CK(cudaMemset(ck, 0, GRID * 4 * 8));
    launch_coop(k_b8, GRID, smem_of(C), mW0, mW1, mW2, mb[0], mb[1], mb[2], mb[3], B, cnt, st, ck, C);
  };
  auto reset = [&](const Cfg& C, bool chk) {   // inputs for one launch
    const bool d2 = C.ph[3].eng == 1;   // design 2: static B copies, ss2 = constant
    CK(cudaMemcpy(B.x0, xinit, 256 * 1024 * 4, cudaMemcpyDeviceToDevice));
    if (chk) { CK(cudaMemcpy(xO, xinit, 256 * 1024 * 4, cudaMemcpyDeviceToDevice)); CK(cudaMemset(xD, 0, 256 * 1024 * 4)); }
    k_cvt<<<(256 * 1024 + 255) / 256, 256>>>(xinit, B.ln1, B.xb, 256, 1024);
    CK(cudaMemset(B.qkv, 0, 256 * 4096 * 4)); CK(cudaMemset(B.gu, 0, 256 * 6144 * 4)); CK(cudaMemset(B.act, 0, 256 * 3072 * 2)); CK(cudaMemset(tk, 0, 4096));
    CK(cudaMemset(B.xbn, 0, 256 * 1024 * 2));
    if (d2) { CK(cudaMemcpy(B.xb2, xb2I, 256 * 1024 * 2, cudaMemcpyDeviceToDevice)); k_init_f32<<<1, 256>>>(B.ss2, 256, 1u); CK(cudaDeviceSynchronize()); std::vector<float> h(256, 1024.f); CK(cudaMemcpy(B.ss2, h.data(), 1024, cudaMemcpyHostToDevice)); }
    else { CK(cudaMemset(B.xb2, 0, 256 * 1024 * 2)); CK(cudaMemset(B.ss2, 0, 1024)); }
    CK(cudaDeviceSynchronize());
  };
  // ---------------------------------------------------------------- correctness
  if (docheck) for (double dd : designs) for (double Td : checkT) {
    const int design = (int)dd, T = (int)Td;
    Cfg C = make_cfg(design, T, 1, 0, o, r1, r2, r3, 1); C.p0 = 0;
    Bufs sv = B;
    B.xO = xO; B.xD = xD;
    reset(C, true);
    launch(C); cudaError_t er = cudaDeviceSynchronize(); if (er == cudaSuccess) er = cudaGetLastError();
    B = sv;
    if (er != cudaSuccess) { fprintf(stderr, "check launch error %s (design %d T %d)\n", cudaGetErrorString(er), design, T); J j; j.s("bench", "b8_compose").s("row", "check").i("design", design).i("T", T).s("error", cudaGetErrorString(er)); j.print(); return 2; }
    auto zero = [&]() { CK(cudaMemset(ver, 0, 64)); };
    auto rd = [&](double& me) { unsigned hv[2]; CK(cudaMemcpy(hv, ver, 8, cudaMemcpyDeviceToHost)); float f; memcpy(&f, &hv[0], 4); me = f; return (int)hv[1]; };
    auto blocks = [](size_t n) { return (unsigned)((n + 255) / 256); };
    const Cfg& c = C; auto rowof = [&](int idx, int slot) { return (int)(((long long)slot * c.ph[idx].N) % c.ph[idx].rowsAvail); };
    const bool ts = design != 1;
    struct Res { const char* name; int bad; double me; }; std::vector<Res> res;
    // QKV
    k_cvt<<<blocks((size_t)T * 1024), 256>>>(xinit, B.ln1, Bref, T, 1024);
    k_refg<<<blocks((size_t)T * 4096), 256>>>(P.p, rowof(0, 0), 1024, Bref, T, 4096, R, R2);
    zero(); k_cmp<<<blocks((size_t)T * 4096), 256>>>(B.qkv, R, R2, nullptr, (size_t)T * 4096, 2e-5f, ver, ver + 1); { double me; int bd = rd(me); res.push_back({"QKV", bd, me}); }
    // O: xO = x0 + ao @ Wo
    k_refg<<<blocks((size_t)T * 1024), 256>>>(P.p, rowof(3, 1), 2048, aoI, T, 1024, R, R2);
    zero(); k_cmp<<<blocks((size_t)T * 1024), 256>>>(xO, R, R2, design == 2 ? nullptr : xinit, (size_t)T * 1024, 2e-5f, ver, ver + 1); { double me; int bd = rd(me); res.push_back({"O", bd, me}); }
    // O finalizer (design 3): xb2 == bf16(xO * ln2), ss2 == sum xO^2
    if (design == 3) {
      k_cvt<<<blocks((size_t)T * 1024), 256>>>(xO, B.ln2, Bref, T, 1024);
      zero(); k_cmp_bf<<<blocks((size_t)T * 1024), 256>>>(B.xb2, Bref, (size_t)T * 1024, ver + 1); k_cmp_ss<<<(T + 63) / 64, 64>>>(xO, B.ss2, T, ver + 1); { double me; int bd = rd(me); res.push_back({"O_fin", bd, me}); }
    }
    // GU
    if (!ts) k_cvt<<<blocks((size_t)T * 1024), 256>>>(xO, B.ln2, Bref, T, 1024);
    const bf16* gB = ts ? B.xb2 : Bref;
    k_refg<<<blocks((size_t)T * 6144), 256>>>(P.p, rowof(4, 2), 1024, gB, T, 6144, R, R2);
    zero();
    if (!ts) k_cmp<<<blocks((size_t)T * 6144), 256>>>(B.gu, R, R2, nullptr, (size_t)T * 6144, 2e-5f, ver, ver + 1);
    else k_cmp_gu<<<blocks((size_t)T * 3072), 256>>>(B.act, R, B.ss2, T, ver, ver + 1);
    { double me; int bd = rd(me); res.push_back({"GU", bd, me}); }
    // D
    const bf16* dB;
    if (!ts) { k_cvt_d<<<blocks((size_t)T * 3072), 256>>>(B.gu, B.rr2, Bref, T); dB = Bref; } else dB = B.act;
    k_refg<<<blocks((size_t)T * 1024), 256>>>(P.p, rowof(5, 3), 3072, dB, T, 1024, R, R2);
    zero(); k_cmp<<<blocks((size_t)T * 1024), 256>>>(xD, R, R2, nullptr, (size_t)T * 1024, 2e-5f, ver, ver + 1); { double me; int bd = rd(me); res.push_back({"D", bd, me}); }
    if (design == 3) {
      k_cvt<<<blocks((size_t)T * 1024), 256>>>(xD, B.ln1 + 1024, Bref, T, 1024);
      zero(); k_cmp_bf<<<blocks((size_t)T * 1024), 256>>>(B.xbn, Bref, (size_t)T * 1024, ver + 1); { double me; int bd = rd(me); res.push_back({"D_fin", bd, me}); }
    }
    CK(cudaDeviceSynchronize());
    for (auto& r : res) {
      J j; j.s("bench", "b8_compose").s("row", "check").s("design", dn[design]).i("T", T).s("phase", r.name).i("bad", r.bad).d("maxerr", r.me).i("pass", r.bad == 0).s("job", job ? job : "local"); j.print();
      if (r.bad) fprintf(stderr, "FAIL %s design %d T %d bad=%d max|d|=%.3e\n", r.name, design, T, r.bad, r.me);
    }
  }
  // ---------------------------------------------------------------- timing
  if (dotime) for (double Td : Ts) for (double dd : designs) for (double pfd : pfs) for (double nud : nulls) {
    const int design = (int)dd, T = (int)Td, pf = (int)pfd, nu = (int)nud;
    Opt oo = o; for (auto& f : fills) if ((int)f[0] == T && f.size() >= 3) { oo.fill_rope = (long long)f[1]; oo.fill_attn = (long long)f[2]; }
    Cfg C = make_cfg(design, T, pf, nu, oo, r1, r2, r3, 0);
    B.xO = B.x0; B.xD = B.x0;
    std::vector<ull> keep; int stored = 0; const long long off0 = st_off;
    std::vector<double> rLP, rM; double dummy = 0; (void)dummy;
    std::vector<std::vector<ull>> stl;
    for (int l = 0; l < warm + nl; ++l) {
      reset(C, false);
      C.p0 = p0; p0 += 4 * C.nlay + 7;
      CK(cudaEventRecord(e0)); launch(C); CK(cudaEventRecord(e1));
      cudaError_t er = cudaEventSynchronize(e1); if (er == cudaSuccess) er = cudaGetLastError();
      if (er != cudaSuccess) { fprintf(stderr, "launch error %s (design %d T %d pf %d)\n", cudaGetErrorString(er), design, T, pf); J j; j.s("bench", "b8_compose").s("row", "time").s("design", dn[design]).i("T", T).i("pf", pf).i("null", nu).s("error", cudaGetErrorString(er)); j.print(); return 2; }
      if (l < warm) continue;
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
      std::vector<ull> h((size_t)GRID * NSTMAX), hc(GRID * 4);
      CK(cudaMemcpy(h.data(), st, h.size() * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hc.data(), ck, hc.size() * 8, cudaMemcpyDeviceToHost));
      rLP.push_back(ms * 1e3 / C.nlay); rM.push_back(mhz_of(hc, GRID));
      if (outst) { stamps_append(outst, h); ++stored; ++st_off; }
    }
    J j; j.s("bench", "b8_compose").s("row", "time").s("design", dn[design]).s("tag", tag).i("T", T).i("npad", npad_of(T)).i("pf", pf).i("null", nu).i("nlay", C.nlay).i("nl", nl)
        .i("fin_place", C.fin_place).i("wu", C.wu).i("fill_rope", oo.fill_rope).i("fill_attn", oo.fill_attn);
    const int idx[4] = {0, 3, 4, 5}; const char* nm[4] = {"QKV", "O", "GU", "D"};
    for (int i = 0; i < 4; ++i) {
      const Ph& p = C.ph[idx[i]]; char key[32];
      snprintf(key, sizeof key, "%s_eng", nm[i]); j.i(key, p.eng);
      snprintf(key, sizeof key, "%s_work", nm[i]); j.i(key, p.eng ? ts_ntile_host(p) : (p.N / 128) * (p.K / 128));
      snprintf(key, sizeof key, "%s_spt", nm[i]); j.i(key, p.eng ? p.K / (p.KS * 128) : 1);
      snprintf(key, sizeof key, "%s_TT", nm[i]); j.i(key, p.TT);
      snprintf(key, sizeof key, "%s_KS", nm[i]); j.i(key, p.KS);
      snprintf(key, sizeof key, "%s_R", nm[i]); j.i(key, p.R);
      snprintf(key, sizeof key, "%s_fin", nm[i]); j.i(key, p.fin);
    }
    j.d("us_layer_launch", med(rLP)).arr("us_layer_launch_reps", rLP).d("mhz", med(rM)).i("regs", fa.numRegs).i("spill", (int)fa.localSizeBytes).i("smem", smem_of(C)).s("job", job ? job : "local")
        .i("st_off", off0).i("st_n", stored).i("grid", GRID).i("nstmax", NSTMAX);
    j.print();
  }
  return 0;
}
