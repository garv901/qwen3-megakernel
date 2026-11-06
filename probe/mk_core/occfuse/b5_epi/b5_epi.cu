// B5 b5_epi: epilogue output modes. A full TMEM accumulator (M128 x TT fp32, one K=64 MMA from prefilled smem operands) -> output written and visible.
// Modes: 1 plain fp32 store, 2 red.add fp32 (KS CTAs share a tile), 3 bf16 store, 4 GU pair act (mk ts_phase PAIR epilogue), 5 argmax over rows (mk lm_phase epilogue: logits store + warp reduce).
// Epilogue warps 4..4+nw-1 (nw = 4, 8, 12), q = w & 3, grp = (w >> 2) - 1, G = nw / 4 groups; chunk map as mk (non-pair: chunks grp, grp + G, two in flight, stride 2G; pair: stride G).
// Per phase: MMA fill (tid 32) -> accfull -> epilogue (stamped per warp) -> join -> mk split barrier (B0 harness, bar_arrive / bar_wait). No A stream, no prefetch (epilogue only).
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -DNSTMAX=232 -I../common b5_epi.cu
#ifndef NSTMAX
#define NSTMAX 232
#endif
#include "ub.cuh"
#include "host.h"

constexpr int EVS = 40;         // u64 per (cta, phase): 12 warps x {s, e1, e2}, then e3
constexpr int TTMAX = 128;      // tokens per output slab
constexpr int OFF_A = 0, OFF_B = 16384, OFF_EX = 32768, OFF_SS = 45056, OFF_BK = 45568, OFF_BI = 47616, OFF_BAR = 49664;
constexpr int SMEM_B5 = OFF_BAR + 256 + SMEM_FIXED;

__device__ __forceinline__ void named_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void fence_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
__device__ __forceinline__ float rt_of(int t) { return 1.f + 0.01f * (float)t; }   // per-token scale (stands for r2 / rsqrt(ss))
__device__ __forceinline__ float silu_f(float x) { return x / (1.f + __expf(-x)); }
// operand values, bf16-exact in smem and in the reference (CTA-dependent so mis-addressed tiles are detected)
__device__ __forceinline__ float aval(int b, int r, int k) { return __bfloat162float(__float2bfloat16(hashf((u32)(r * 64 + k) + 1000u * (u32)(b + 1)) * 0.5f)); }
__device__ __forceinline__ float bval(int b, int t, int k) { return __bfloat162float(__float2bfloat16(hashf((u32)(t * 64 + k) + 77777u + 1000u * (u32)(b + 1)) * 0.5f)); }

template <int MODE, int TT>
__global__ void __launch_bounds__(NT, 1) k_epi(unsigned* cnt, ull* st, ull* ck, ull* ev, float* out32, bf16* outbf, u32* pkey, int* pidx, int nph, int T, int nw, int KS, int orc, int ld) {
  extern __shared__ uint8_t smraw[];
  const u32 raw = smem_u32(smraw), base = (raw + 1023) & ~1023u;
  uint8_t* sm = smraw + (base - raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, q = warp & 3, grp = (warp >> 2) - 1, G = nw >> 2;
  constexpr int NCH = TT / 16;
  const u32 accb = base + OFF_BAR;
  volatile u32* tslot = reinterpret_cast<volatile u32*>(sm + OFF_BAR + 64);
  float* s_ss = reinterpret_cast<float*>(sm + OFF_SS);
  u32* s_bk = reinterpret_cast<u32*>(sm + OFF_BK);   // [4][128]
  int* s_bi = reinterpret_cast<int*>(sm + OFF_BI);
  float* s_ex = reinterpret_cast<float*>(sm + OFF_EX);
  if (warp == 0) tmem_alloc(smem_u32((void*)tslot), 128);
  // operands: A [128 rows][64 k], B [TT rows][64 k], SW128 K-major (byte = r * 128 + ((chunk ^ (r & 7)) << 4) + (k & 7) * 2)
  for (int i = tid; i < 128 * 64; i += NT) { const int r = i >> 6, k = i & 63; *reinterpret_cast<bf16*>(sm + OFF_A + r * 128 + ((((k >> 3) ^ (r & 7)) << 4)) + (k & 7) * 2) = __float2bfloat16(aval(blockIdx.x, r, k)); }
  for (int i = tid; i < TT * 64; i += NT) { const int r = i >> 6, k = i & 63; *reinterpret_cast<bf16*>(sm + OFF_B + r * 128 + ((((k >> 3) ^ (r & 7)) << 4)) + (k & 7) * 2) = __float2bfloat16(bval(blockIdx.x, r, k)); }
  for (int i = tid; i < 128; i += NT) s_ss[i] = rt_of(i);
  for (int i = tid; i < 512; i += NT) { s_bk[i] = 0u; s_bi[i] = 0; }
  fence_proxy_async();
  if (tid == 0) { mbar_init(accb, 1); mbar_fence_init(); }
  __syncthreads();
  tc_fence_after();
  const u32 tb = *tslot;
  const int g = blockIdx.x / KS;                              // tile id shared by the KS CTAs of a group
  const int row = q * 32 + lane;
  float* o32 = out32 + (size_t)(g >> 3) * TTMAX * ld + (size_t)(g & 7) * 128 + row;
  bf16* obf = outbf + (size_t)(g >> 3) * TTMAX * ld + (size_t)(g & 7) * 128 + row;
  bf16* act = outbf + (size_t)(g >> 4) * TTMAX * ld + 64 * (g & 15) + 32 * q + lane;   // mode 4: channel 64 rb + 32 q + lane
  const bool epi_warp = warp >= 4 && warp < 4 + nw;
  const u32 idesc = tc_idesc(128, TT);
  unsigned target = 0; u32 par = 0;
  clk_sample(ck, 0);
  bar_arrive(cnt, st, 0); bar_wait(cnt, target, st, 0);
  for (int p = 1; p <= nph; ++p) {
    if (tid == 32) {
      tc_fence_after();
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) tc_mma(tb, sdesc(base + OFF_A + kk * 32), sdesc(base + OFF_B + kk * 32), idesc, kk > 0 ? 1u : 0u);
      tc_commit(accb);
    }
    if (epi_warp) {
      mbar_wait(accb, par);
      tc_fence_after();
      const ull t_s = gt();
      if (!(orc & 1)) {
        if (MODE == 4) {
          for (int c = grp; c < NCH; c += G) {
            const int cb = c * 16;
            if (cb >= T) break;   // warp-uniform, equal for both warps of a pair
            u32 v[16];
            tmem_ld16(v, tb + ((u32)(q * 32) << 16) + cb);
            tc_wait_ld();
            const int pid = grp * 2 + (q & 1);
            float* ex = s_ex + pid * 512;
            if (q >= 2) {   // u warp
#pragma unroll
              for (int i = 0; i < 16; ++i) ex[i * 32 + lane] = __uint_as_float(v[i]);
              named_bar_sync(1 + pid, 64);
              named_bar_sync(1 + pid, 64);
            } else {        // g warp
              float ua[16];
              named_bar_sync(1 + pid, 64);
#pragma unroll
              for (int i = 0; i < 16; ++i) ua[i] = ex[i * 32 + lane];
              named_bar_sync(1 + pid, 64);
#pragma unroll
              for (int i = 0; i < 16; ++i) {
                const int t = cb + i;
                if (t < T) {
                  const float r = s_ss[t];
                  const float h = silu_f(r * __uint_as_float(v[i])) * (r * ua[i]);
                  act[(size_t)t * ld] = __float2bfloat16(h);
                }
              }
            }
          }
        } else {
          auto emit = [&](int cb, const u32 (&v)[16]) {
            if (MODE == 5) {
              u32 rk = 0u; int ri = 0;
#pragma unroll
              for (int i = 0; i < 16; ++i) {
                const int s = cb + i;
                if (s < T) {   // warp-uniform
                  const float val = s_ss[s] * __uint_as_float(v[i]);
                  o32[(size_t)s * ld] = val;
                  const u32 ub = __float_as_uint(val);
                  const u32 key = (ub & 0x80000000u) ? ~ub : (ub | 0x80000000u);
                  const u32 mx = __reduce_max_sync(0xffffffffu, key);
                  const u32 eq = __ballot_sync(0xffffffffu, key == mx);
                  if (lane == i) { rk = mx; ri = row - lane + (__ffs(eq) - 1); }
                }
              }
              if (lane < 16 && cb + lane < T) { const int s = cb + lane; if (rk > s_bk[q * 128 + s]) { s_bk[q * 128 + s] = rk; s_bi[q * 128 + s] = ri; } }
            } else {
#pragma unroll
              for (int i = 0; i < 16; ++i) {
                const int t = cb + i;
                if (t < T) {   // warp-uniform
                  const float f = __uint_as_float(v[i]);
                  if (MODE == 1) { if (orc & 2) { if (f == 1234.5f) o32[(size_t)t * ld] = 0.f; } else o32[(size_t)t * ld] = f; }
                  else if (MODE == 2) red_add(o32 + (size_t)t * ld, f);
                  else obf[(size_t)t * ld] = __float2bfloat16(f);
                }
              }
            }
          };
          for (int c = grp; c < NCH; c += 2 * G) {
            const int cb = c * 16;
            if (cb >= T) break;
            const bool two = (c + G < NCH) && (cb + 16 * G < T);
            u32 v0[16], v1[16];
            tmem_ld16(v0, tb + ((u32)(q * 32) << 16) + cb);
            if (two) tmem_ld16(v1, tb + ((u32)(q * 32) << 16) + cb + 16 * G);
            tc_wait_ld();
            emit(cb, v0);
            if (two) emit(cb + 16 * G, v1);
          }
        }
      }
      tc_fence_before();
      __syncwarp();
      const ull t_1 = gt();
      if (MODE == 4) fence_proxy_async_global();
      fence_gpu();
      const ull t_2 = gt();
      if (lane == 0) { ull* e = ev + ((size_t)blockIdx.x * (nph + 1) + p) * EVS + (warp - 4) * 3; e[0] = t_s; e[1] = t_1; e[2] = t_2; }
    }
    par ^= 1;
    __syncthreads();   // join (mk ts_phase)
    if (MODE == 5) {   // per-CTA best per slot over the 4 quadrants -> pkey / pidx (the cross-CTA reduction of mk is not part of this bench)
      if (tid < T && !(orc & 1)) {
        u32 bk = s_bk[tid]; int bi = s_bi[tid];
#pragma unroll
        for (int qq = 1; qq < 4; ++qq) { const u32 k = s_bk[qq * 128 + tid]; const int i = qq * 0 + s_bi[qq * 128 + tid]; if (k > bk || (k == bk && i < bi)) { bk = k; bi = i; } }
        __stcg(pkey + (size_t)blockIdx.x * 128 + tid, bk); __stcg(pidx + (size_t)blockIdx.x * 128 + tid, bi);
      }
      __syncthreads();
    }
    if (tid == 0) ev[((size_t)blockIdx.x * (nph + 1) + p) * EVS + 39] = gt();
    if (MODE == 5) for (int i = tid; i < 512; i += NT) s_bk[i] = 0u;
    bar_arrive(cnt, st, p);
    bar_wait(cnt, target, st, p);
  }
  clk_sample(ck, 1);
  __syncthreads();
  tc_fence_after();
  if (warp == 0) tmem_dealloc(tb, 128);
}

// CUDA-core reference + check. thread = (tile g, row r, token t). Counts mismatches; maxerr = max |got - exp| / (tol scale).
__device__ __forceinline__ float acc_of(int b, int r, int t) { float a = 0.f; for (int k = 0; k < 64; ++k) a += aval(b, r, k) * bval(b, t, k); return a; }
__global__ void k_check(int mode, int TT, int T, int KS, int nph, int ld, const float* out32, const bf16* outbf, const u32* pkey, const int* pidx, unsigned* bad, unsigned* nchk, unsigned* maxerr) {
  const int ngroups = (GRID + KS - 1) / KS;
  const long long n = (long long)ngroups * 128 * T, i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < n) {
    const int t = (int)(i % T), r = (int)((i / T) % 128), g = (int)(i / ((long long)T * 128));
    const int b0 = g * KS, b1 = min(GRID, b0 + KS);
    float got = 0.f, ex = 0.f, tol = 0.f; bool skip = false;
    if (mode == 1 || mode == 2 || mode == 3) {
      float s = 0.f, sa = 0.f;
      for (int b = b0; b < b1; ++b) { const float a = acc_of(b, r, t); s += a; sa += fabsf(a); }
      if (mode == 2) { ex = s * nph; tol = 2e-3f * sa * nph + 1e-3f; } else if (mode == 1) { ex = s; tol = 1e-4f * sa + 1e-5f; } else { ex = s; tol = 1.6e-2f * sa + 1e-4f; }
      const size_t idx = (size_t)(g >> 3) * TTMAX * ld + (size_t)t * ld + (g & 7) * 128 + r;
      got = mode == 3 ? __bfloat162float(outbf[idx]) : out32[idx];
    } else if (mode == 4) {
      if (r >= 64) skip = true;
      else {
        const float gg = acc_of(b0, r, t), uu = acc_of(b0, r + 64, t), rr = rt_of(t);
        ex = silu_f(rr * gg) * (rr * uu); tol = 2e-2f * fabsf(ex) + 2e-3f;
        got = __bfloat162float(outbf[(size_t)(g >> 4) * TTMAX * ld + (size_t)t * ld + 64 * (g & 15) + r]);
      }
    } else {
      ex = rt_of(t) * acc_of(b0, r, t); tol = 1e-4f * fabsf(ex) + 1e-5f;
      got = out32[(size_t)(g >> 3) * TTMAX * ld + (size_t)t * ld + (g & 7) * 128 + r];
    }
    if (!skip) {
      atomicAdd(nchk, 1u);
      const float e = fabsf(got - ex);
      if (!(e <= tol)) atomicAdd(bad, 1u);
      atomicMax(maxerr, __float_as_uint(fminf(e / tol, 1e6f)));
    }
  }
  if (mode == 5 && i < (long long)GRID * T) {   // argmax: chosen row's reference value must be within tolerance of the maximum
    const int b = (int)(i / T), t = (int)(i % T);
    float mx = -1e30f; for (int r = 0; r < 128; ++r) mx = fmaxf(mx, acc_of(b, r, t));
    const int ri = pidx[b * 128 + t];
    const float v = (ri >= 0 && ri < 128) ? acc_of(b, ri, t) : -1e30f;
    atomicAdd(nchk, 1u);
    if (!(v >= mx - 1e-4f * fabsf(mx) - 1e-5f)) atomicAdd(bad, 1u);
  }
}

// ---------------- host ----------------
struct Args { unsigned* cnt; ull *st, *ck, *ev; float* out32; bf16* outbf; u32* pkey; int* pidx; int nph, T, nw, KS, orc, ld; };
static int g_regs = 0, g_spill = 0;
template <int M, int TT> static void launch_mt(const Args& a) {
  static bool init = false;
  if (!init) { CK(cudaFuncSetAttribute(k_epi<M, TT>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX)); init = true; }
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, k_epi<M, TT>)); g_regs = fa.numRegs; g_spill = (int)fa.localSizeBytes;
  launch_coop(k_epi<M, TT>, GRID, SMEM_B5, a.cnt, a.st, a.ck, a.ev, a.out32, a.outbf, a.pkey, a.pidx, a.nph, a.T, a.nw, a.KS, a.orc, a.ld);
}
template <int M> static void launch_m(int TT, const Args& a) {
  switch (TT) { case 16: launch_mt<M, 16>(a); break; case 32: launch_mt<M, 32>(a); break; case 48: launch_mt<M, 48>(a); break; case 64: launch_mt<M, 64>(a); break;
    case 96: launch_mt<M, 96>(a); break; case 128: launch_mt<M, 128>(a); break; default: fprintf(stderr, "bad TT %d\n", TT); exit(1); }
}
static void launch_any(int mode, int TT, const Args& a) {
  switch (mode) { case 1: launch_m<1>(TT, a); break; case 2: launch_m<2>(TT, a); break; case 3: launch_m<3>(TT, a); break; case 4: launch_m<4>(TT, a); break; case 5: launch_m<5>(TT, a); break; default: exit(1); }
}
static std::vector<int> ilist(const char* s) { std::vector<int> v; char* d = strdup(s); for (char* t = strtok(d, ","); t; t = strtok(nullptr, ",")) v.push_back(atoi(t)); return v; }
static const char* garg(int c, char** v, const char* k, const char* d) { for (int i = 1; i + 1 < c; ++i) if (!strcmp(v[i], k)) return v[i + 1]; return d; }
static double mean(const std::vector<double>& v) { if (v.empty()) return -1; double s = 0; for (double x : v) s += x; return s / v.size(); }
static double pct(std::vector<double> v, double p) { if (v.empty()) return -1; std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t)(p * v.size()))]; }

int main(int argc, char** argv) {
  const char* outst = garg(argc, argv, "--out", nullptr); const char* tag = garg(argc, argv, "--tag", ""); const char* job = getenv("SLURM_JOB_ID");
  const int nph = atoi(garg(argc, argv, "--nph", "112")), nl = atoi(garg(argc, argv, "--launches", "7")), warm = atoi(garg(argc, argv, "--warm", "2"));
  const int ld = atoi(garg(argc, argv, "--ld", "1024")), stall = atoi(garg(argc, argv, "--stall", "0"));
  const std::vector<int> modes = ilist(garg(argc, argv, "--modes", "1,2,3,4,5")), TTs = ilist(garg(argc, argv, "--TT", "16,32,48,64,96,128")),
      nws = ilist(garg(argc, argv, "--nw", "4,8,12")), KSs = ilist(garg(argc, argv, "--KS", "1,2,4,8,12,24")), Tsel = ilist(garg(argc, argv, "--T", "1,8,16,0")),   // 0 = TT
      orcs = ilist(garg(argc, argv, "--orc", "0"));
  if (nph + 1 > NSTMAX / 2 - 1) { fprintf(stderr, "nph too large\n"); return 1; }
  int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0)); if (nsm != GRID) fprintf(stderr, "warning: %d SMs\n", nsm);
  const size_t slabs = (GRID + 7) / 8 + 1, o32n = slabs * TTMAX * ld, obfn = o32n;
  unsigned* cnt; ull *st, *ck, *ev; float* out32; bf16* outbf; u32* pkey; int* pidx; unsigned* ctr;
  CK(cudaMalloc(&cnt, 1024)); CK(cudaMalloc(&st, (size_t)GRID * NSTMAX * 8)); CK(cudaMalloc(&ck, GRID * 4 * 8)); CK(cudaMalloc(&ev, (size_t)GRID * (nph + 1) * EVS * 8));
  CK(cudaMalloc(&out32, o32n * 4)); CK(cudaMalloc(&outbf, obfn * 2)); CK(cudaMalloc(&pkey, GRID * 128 * 4)); CK(cudaMalloc(&pidx, GRID * 128 * 4)); CK(cudaMalloc(&ctr, 64));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  long long st_off = 0;
  std::vector<ull> hst((size_t)GRID * NSTMAX), hck(GRID * 4), hev((size_t)GRID * (nph + 1) * EVS);

  auto run_cfg = [&](int mode, int TT, int T, int nw, int KS, int orc, int storest) {
    Args a{cnt, st, ck, ev, out32, outbf, pkey, pidx, nph, T, nw, KS, orc, ld};
    std::vector<char> elig(GRID, 1);
    if (KS > 1) for (int b = 0; b < GRID; ++b) elig[b] = ((b / KS + 1) * KS <= GRID);
    std::vector<double> rT, rTm, rLoop, rVis, rJoin, rP90, rLP, rM, rW, rWarpMax, rWarpMean;
    int ok = 1, stored = 0; const long long off0 = st_off; unsigned h_bad = 0, h_n = 0, h_me = 0;
    for (int l = 0; l < warm + nl; ++l) {
      CK(cudaMemset(cnt, 0, 1024)); CK(cudaMemset(st, 0, (size_t)GRID * NSTMAX * 8)); CK(cudaMemset(ck, 0, GRID * 4 * 8));
      CK(cudaMemset(out32, 0, o32n * 4)); CK(cudaMemset(outbf, 0, obfn * 2)); CK(cudaMemset(pkey, 0, GRID * 512)); CK(cudaMemset(pidx, 0, GRID * 512));
      CK(cudaEventRecord(e0)); launch_any(mode, TT, a); CK(cudaEventRecord(e1)); cudaError_t er = cudaEventSynchronize(e1);
      if (er == cudaSuccess) er = cudaGetLastError();
      if (er != cudaSuccess) { fprintf(stderr, "launch error %s\n", cudaGetErrorString(er)); ok = 0; break; }
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
      if (l == warm + nl - 1) {   // correctness on the last launch (outputs of the last launch)
        CK(cudaMemset(ctr, 0, 64));
        const long long n = std::max<long long>((long long)((GRID + KS - 1) / KS) * 128 * T, (long long)GRID * T);
        k_check<<<(unsigned)((n + 255) / 256), 256>>>(mode, TT, T, KS, nph, ld, out32, outbf, pkey, pidx, ctr, ctr + 1, ctr + 2); CK(cudaDeviceSynchronize());
        unsigned h[3]; CK(cudaMemcpy(h, ctr, 12, cudaMemcpyDeviceToHost)); h_bad = h[0]; h_n = h[1]; h_me = h[2];
      }
      CK(cudaMemcpy(hst.data(), st, hst.size() * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hck.data(), ck, hck.size() * 8, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(hev.data(), ev, hev.size() * 8, cudaMemcpyDeviceToHost));
      if (l < warm) continue;
      // barrier-stamp metrics (B0 definitions): T median per phase, Tmean telescoping, work mean over eligible CTAs
      auto arr = [&](int c, int p) { return (double)hst[(size_t)c * NSTMAX + 2 * p]; }; auto exq = [&](int c, int p) { return (double)hst[(size_t)c * NSTMAX + 2 * p + 1]; };
      std::vector<double> Ts, wk_all; std::vector<double> exs(GRID), exsp(GRID);
      auto medex = [&](int p) { std::vector<double> v; for (int c = 0; c < GRID; ++c) v.push_back(exq(c, p)); return med(v); };
      for (int p = 2; p <= nph; ++p) {
        Ts.push_back((medex(p) - medex(p - 1)) / 1e3);
        for (int c = 0; c < GRID; ++c) if (elig[c]) wk_all.push_back((arr(c, p) - exq(c, p - 1)) / 1e3);
      }
      rT.push_back(med(Ts)); rTm.push_back((medex(nph) - medex(1)) / 1e3 / (nph - 1)); rW.push_back(mean(wk_all));
      // epilogue events
      std::vector<double> lo, vi, jo, wmx, wmn;
      for (int c = 0; c < GRID; ++c) {
        if (!elig[c]) continue;
        for (int p = 2; p <= nph; ++p) {
          const ull* e = &hev[((size_t)c * (nph + 1) + p) * EVS];
          ull smin = ~0ull, e1m = 0, e2m = 0; double wm = 0, ws = 0;
          for (int w = 0; w < nw; ++w) { smin = std::min(smin, e[w * 3]); e1m = std::max(e1m, e[w * 3 + 1]); e2m = std::max(e2m, e[w * 3 + 2]); const double d = (double)(e[w * 3 + 1] - e[w * 3]); wm = std::max(wm, d); ws += d; }
          lo.push_back((double)(e1m - smin) / 1e3); vi.push_back((double)(e2m - smin) / 1e3); jo.push_back((double)(e[39] - smin) / 1e3); wmx.push_back(wm / 1e3); wmn.push_back(ws / nw / 1e3);
        }
      }
      rLoop.push_back(mean(lo)); rVis.push_back(mean(vi)); rJoin.push_back(mean(jo)); rP90.push_back(pct(lo, 0.9)); rWarpMax.push_back(mean(wmx)); rWarpMean.push_back(mean(wmn));
      rLP.push_back(ms * 1e3 / nph); rM.push_back(mhz_of(hck, GRID));
      if (storest && stored < 1) { stamps_append(outst, hst); ++stored; ++st_off; }
    }
    float maxerr_f; memcpy(&maxerr_f, &h_me, 4);
    const bool pass = ok && h_bad == 0 && h_n > 0;
    J j; j.s("bench", "b5_epi").s("arm", "epi").s("tag", tag).i("mode", mode).i("TT", TT).i("T", T).i("nw", nw).i("KS", KS).i("orc", orc).i("nph", nph).i("nl", nl)
        .d("us_loop", med(rLoop)).d("us_loop_min", vmin(rLoop)).d("us_loop_max", vmax(rLoop)).d("us_loop_p90", med(rP90)).d("us_vis", med(rVis)).d("us_join", med(rJoin))
        .d("us_warpmax", med(rWarpMax)).d("us_warpmean", med(rWarpMean)).d("us_T", med(rT)).d("us_Tmean", med(rTm)).d("us_Tmean_min", vmin(rTm)).d("us_Tmean_max", vmax(rTm)).d("us_work", med(rW))
        .d("us_busy", med(rW)).d("us_idle", -1).d("us_last", -1).d("us_launch_per_phase", med(rLP)).arr("us_reps", rLoop).arr("Tmean_reps", rTm).d("mhz", med(rM))
        .i("pass", pass).i("nbad", h_bad).i("nchk", h_n).d("maxerr_over_tol", (double)maxerr_f).i("regs", g_regs).i("spill", g_spill).i("smem", SMEM_B5).s("job", job ? job : "local")
        .i("st_off", off0).i("st_n", stored).i("grid", GRID).i("nstmax", NSTMAX).i("neligible", (int)std::count(elig.begin(), elig.end(), 1));
    j.print();
  };

  for (int mode : modes) for (int TT : TTs) {
    std::vector<int> Tl; for (int t : Tsel) { const int tv = t == 0 ? TT : t; if (tv <= TT && std::find(Tl.begin(), Tl.end(), tv) == Tl.end()) Tl.push_back(tv); }
    for (int T : Tl) for (int nw : nws) for (int orc : orcs) {
      if (mode == 2) { for (int KS : KSs) run_cfg(mode, TT, T, nw, KS, orc, stall || (T == TT && nw == 12)); }
      else run_cfg(mode, TT, T, nw, 1, orc, stall || (T == TT && nw == 12));
    }
  }
  return 0;
}
