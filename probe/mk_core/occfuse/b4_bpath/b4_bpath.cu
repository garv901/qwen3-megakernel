// B4 b4_bpath: getting the token operand B (T rows x kb, bf16, SW128 K-major) into smem.
// modes: 0 z (no B work), 1 a1 staged per k-step (128 k), 2 a2 staged, all kb in one batch, 3 b TMA of bf16 copy, 4 c TMA of fp32 copy (bytes only).
// loaded: thread 0 streams 32 KB A weight slots (2 slot ring, R=2) from the Pool into separate smem; it re-arms a finished slot at the points where it is
//   free (between k-steps / load groups / while waiting for the B TMA), never blocking (mk gemm_phase thread 0 has the same duty).
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -DNSTMAX=232 -I../common b4_bpath.cu
#ifndef NSTMAX
#define NSTMAX 232
#endif
#ifndef STG
#define STG 4   // chunks (16 B bf16 = 2 x float4 fp32) in flight per thread per load group (register cap, mk MK_STG)
#endif
#include "ub.cuh"
#include "host.h"

constexpr int LDX = 1024;                 // x row length (fp32 elements), max kb
constexpr int ABYTES = 32768, NA = 2;     // A ring: 2 slots x (2 boxes of 64 k x 128 rows)
constexpr int BREG = SMEM_MAX - 1024 - NA * ABYTES - 2048;   // 163840 B for the B tiles
constexpr float EPS = 1e-6f;

__device__ __forceinline__ bool mbar_test(u32 a, u32 parity) {
  u32 ok;
  asm volatile("{\n\t.reg .pred P;\n\tmbarrier.test_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2;\n\tselp.u32 %0, 1, 0, P;\n\t}" : "=r"(ok) : "r"(a), "r"(parity) : "memory");
  return ok;
}
__device__ __forceinline__ float rsc(const float* ss, int r) { return rsqrtf(__ldcg(ss + r) * (1.f / LDX) + EPS); }
__device__ __forceinline__ float nrm(float x, float rr, float w) { return __fmul_rn(__fmul_rn(x, rr), w); }
__device__ __forceinline__ uint4 pack8(float4 a, float4 b, float rr, float4 wa, float4 wb) {
  uint4 o; __nv_bfloat162 h;
  h = __floats2bfloat162_rn(nrm(a.x, rr, wa.x), nrm(a.y, rr, wa.y)); o.x = *reinterpret_cast<u32*>(&h);
  h = __floats2bfloat162_rn(nrm(a.z, rr, wa.z), nrm(a.w, rr, wa.w)); o.y = *reinterpret_cast<u32*>(&h);
  h = __floats2bfloat162_rn(nrm(b.x, rr, wb.x), nrm(b.y, rr, wb.y)); o.z = *reinterpret_cast<u32*>(&h);
  h = __floats2bfloat162_rn(nrm(b.z, rr, wb.z), nrm(b.w, rr, wb.w)); o.w = *reinterpret_cast<u32*>(&h);
  return o;
}
__global__ void k_ref(const float* x, const float* ss, const float* w, bf16* xb, int rows) {   // CUDA-core reference of the normalized activation = the bf16 copy for TMA
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * LDX; i += gridDim.x * blockDim.x) {
    const int r = i / LDX, k = i % LDX;
    xb[i] = __float2bfloat16(nrm(x[i], rsqrtf(ss[r] * (1.f / LDX) + EPS), w[k]));
  }
}

struct AStream {   // thread 0 only
  u32 full[NA], sm[NA]; unsigned par[NA]; bool out[NA]; long long cnt; const CUtensorMap* mW; long long rows;
  __device__ void issue(int s) {
    mbar_expect_tx(full[s], ABYTES);
    const long long row = (((cnt++ * GRID) + blockIdx.x) * 128) % (rows - 128);
    tma2(sm[s], mW, 0, (int)row, full[s]); tma2(sm[s] + 16384, mW, 64, (int)row, full[s]); out[s] = true;
  }
  __device__ void poll() { for (int s = 0; s < NA; ++s) if (out[s] && mbar_test(full[s], par[s])) { par[s] ^= 1; issue(s); } }
  __device__ void drain() { for (int s = 0; s < NA; ++s) if (out[s]) { mbar_wait(full[s], par[s]); out[s] = false; } }
};

// Stage kw columns starting at k0 for all TT rows (rows >= T zero). SW128 layout as mk stage_tile; sub-box (64 k) j goes to slot j % nslots.
// poll: thread 0 re-arms the A ring after each load group's stores.
__device__ __forceinline__ void stage(uint8_t* xs, int k0, int kw, int T, int TT, int nslots, const float* x, const float* ss, const float* lnw, bool poll, AStream& as) {
  const int tid = threadIdx.x, lgc = 31 - __clz(kw >> 3), CPR = kw >> 3, CH = TT * CPR, ITER = (CH + NT - 1) / NT;
  for (int g0 = 0; g0 < ITER; g0 += STG) {
    uint4 v[STG];
#pragma unroll
    for (int jj = 0; jj < STG; ++jj) {
      const int i = g0 + jj, c = tid + NT * i;
      v[jj] = make_uint4(0, 0, 0, 0);
      if (i < ITER && c < CH) {
        const int r = c >> lgc, k = k0 + (c & (CPR - 1)) * 8;
        if (r < T) {
          const float* p = x + (size_t)r * LDX + k;
          const float4 a = __ldcg(reinterpret_cast<const float4*>(p)), b = __ldcg(reinterpret_cast<const float4*>(p) + 1);
          const float4 wa = __ldg(reinterpret_cast<const float4*>(lnw + k)), wb = __ldg(reinterpret_cast<const float4*>(lnw + k) + 1);
          v[jj] = pack8(a, b, rsc(ss, r), wa, wb);
        }
      }
    }
#pragma unroll
    for (int jj = 0; jj < STG; ++jj) {
      const int i = g0 + jj, c = tid + NT * i;
      if (i < ITER && c < CH) {
        const int r = c >> lgc, k = k0 + (c & (CPR - 1)) * 8, j = k >> 6, ch = (k >> 3) & 7;
        *reinterpret_cast<uint4*>(xs + (size_t)(j % nslots) * (TT * 128) + r * 128 + ((ch ^ (r & 7)) << 4)) = v[jj];
      }
    }
    if (poll && tid == 0) as.poll();
  }
}

__global__ void __launch_bounds__(NT, 1) k_b4(const __grid_constant__ CUtensorMap mW, const __grid_constant__ CUtensorMap mB, const __grid_constant__ CUtensorMap mF,
                                              unsigned* cnt, ull* st, ull* ck, uint8_t* dump, int nph, int mode, int loaded, int T, int TT, int kb, int nslots, int nbox,
                                              const float* x, const float* ss, const float* lnw, long long rows) {
  extern __shared__ uint8_t smraw[];
  const u32 raw = smem_u32(smraw), base = (raw + 1023) & ~1023u;
  uint8_t* sm = smraw + (base - raw);
  const int tid = threadIdx.x;
  uint8_t* xs = sm + NA * ABYTES;
  const u32 xsa = base + NA * ABYTES, bars = xsa + BREG;   // bars: [0,1] A full, [2,3] B tma
  AStream as;
  if (tid == 0) {
    for (int s = 0; s < NA; ++s) { as.full[s] = bars + 8 * s; as.sm[s] = base + s * ABYTES; as.par[s] = 0; as.out[s] = false; mbar_init(as.full[s], 1); }
    mbar_init(bars + 16, 1); mbar_init(bars + 24, 1); mbar_fence_init();
    as.cnt = 0; as.mW = &mW; as.rows = rows;
  }
  __syncthreads();
  if (tid == 0 && loaded) for (int s = 0; s < NA; ++s) as.issue(s);
  unsigned target = 0; u32 bpar[2] = {0, 0};
  clk_sample(ck, 0);
  bar_arrive(cnt, st, 0); bar_wait(cnt, target, st, 0);
  const CUtensorMap* mp = (mode == 4) ? &mF : &mB;
  for (int p = 1; p <= nph; ++p) {
    if (mode == 1) {
      for (int s = 0; s < (kb >> 7); ++s) {
        stage(xs, s * 128, 128, T, TT, nslots, x, ss, lnw, false, as);
        fence_proxy_async(); __syncthreads();
        if (tid == 0 && loaded) as.poll();
      }
    } else if (mode == 2) {
      stage(xs, 0, kb, T, TT, nslots, x, ss, lnw, loaded != 0, as);
      fence_proxy_async(); __syncthreads();
    } else if (mode >= 3) {
      if (tid == 0) {
        const u32 per = (u32)TT * 128;
        const int n0 = (nbox + 1) >> 1, n1 = nbox >> 1;
        mbar_expect_tx(bars + 16, n0 * per); if (n1) mbar_expect_tx(bars + 24, n1 * per);
        for (int j = 0; j < nbox; ++j) tma2(xsa + (j % nslots) * per, mp, 64 * j, 0, bars + 16 + 8 * (j & 1));
        bool d0 = false, d1 = (n1 == 0);
        while (!(d0 && d1)) {
          if (!d0) d0 = mbar_test(bars + 16, bpar[0]);
          if (!d1) d1 = mbar_test(bars + 24, bpar[1]);
          if (loaded) as.poll();
        }
        bpar[0] ^= 1; if (n1) bpar[1] ^= 1;
      }
    }
    bar_arrive(cnt, st, p);
    bar_wait(cnt, target, st, p);
  }
  if (dump && blockIdx.x == gridDim.x - 1) for (int i = tid; i < BREG / 16; i += NT) reinterpret_cast<uint4*>(dump)[i] = reinterpret_cast<uint4*>(xs)[i];
  if (tid == 0 && loaded) as.drain();
  clk_sample(ck, 1);
  __syncthreads();
}

static std::vector<int> ilist(const char* s) { std::vector<int> v; for (char* t = strtok(strdup(s), ","); t; t = strtok(nullptr, ",")) v.push_back(atoi(t)); return v; }
static const char* garg(int c, char** v, const char* k, const char* d) { for (int i = 1; i + 1 < c; ++i) if (!strcmp(v[i], k)) return v[i + 1]; return d; }
static float bf2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

struct Met { double T, work, lat, Tmean, wmean; };
static Met metrics(const std::vector<ull>& st, int nph) {   // phases 2..nph, us
  std::vector<double> T, W, A;
  auto arr = [&](int c, int p) { return (double)st[(size_t)c * NSTMAX + 2 * p]; };
  auto ex = [&](int c, int p) { return (double)st[(size_t)c * NSTMAX + 2 * p + 1]; };
  double wsum = 0; long long wn = 0;
  std::vector<double> e1;
  for (int p = 2; p <= nph; ++p) {
    std::vector<double> e, ep, w; double amax = 0;
    for (int c = 0; c < GRID; ++c) { e.push_back(ex(c, p)); ep.push_back(ex(c, p - 1)); w.push_back(arr(c, p) - ex(c, p - 1)); amax = std::max(amax, arr(c, p)); wsum += w.back(); ++wn; }
    const double me = med(e);
    T.push_back((me - med(ep)) / 1e3); W.push_back(med(w) / 1e3); A.push_back((me - amax) / 1e3);
  }
  std::vector<double> e1v, enp; for (int c = 0; c < GRID; ++c) { e1v.push_back(ex(c, 1)); enp.push_back(ex(c, nph)); }
  return {med(T), med(W), med(A), (med(enp) - med(e1v)) / 1e3 / (nph - 1), wsum / wn / 1e3};
}

int main(int argc, char** argv) {
  const char* outst = garg(argc, argv, "--out", nullptr);
  const int nph = atoi(garg(argc, argv, "--nph", "112")), nl = atoi(garg(argc, argv, "--launches", "7")), warm = atoi(garg(argc, argv, "--warm", "2"));
  const int stl = atoi(garg(argc, argv, "--stlaunches", "1"));
  const char* tag = garg(argc, argv, "--tag", ""); const char* job = getenv("SLURM_JOB_ID");
  std::vector<int> Ts = ilist(garg(argc, argv, "--T", "1,8,16,32,64,128,256")), kbs = ilist(garg(argc, argv, "--kb", "128,256,512,1024"));
  std::vector<int> modes = ilist(garg(argc, argv, "--mode", "0,1,2,3,4")), lds = ilist(garg(argc, argv, "--loaded", "0,1"));
  if (nph + 1 > NSTMAX / 2 - 1) { fprintf(stderr, "nph too large\n"); return 1; }
  int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  if (nsm != GRID) fprintf(stderr, "warning: %d SMs (expected %d)\n", nsm, GRID);
  Pool P = pool_alloc(); const long long rows = (long long)(P.bytes / (2ull * 1024));
  CUtensorMap mW = tmap_bf16(P.p, 1024, rows, 64, 128);
  const int TMAX = 256;
  float *x, *ss, *lnw; bf16* xb; uint8_t* dump;
  CK(cudaMalloc(&x, TMAX * LDX * 4)); CK(cudaMalloc(&ss, TMAX * 4)); CK(cudaMalloc(&lnw, LDX * 4)); CK(cudaMalloc(&xb, TMAX * LDX * 2)); CK(cudaMalloc(&dump, BREG));
  k_init_f32<<<64, 256>>>(x, TMAX * LDX, 77u); k_init_f32<<<4, 256>>>(lnw, LDX, 99u);
  CK(cudaDeviceSynchronize());
  std::vector<float> hx(TMAX * LDX), hw(LDX), hss(TMAX);
  CK(cudaMemcpy(hx.data(), x, hx.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hw.data(), lnw, LDX * 4, cudaMemcpyDeviceToHost));
  for (int i = 0; i < LDX; ++i) hw[i] = 0.75f + 0.25f * hw[i];
  CK(cudaMemcpy(lnw, hw.data(), LDX * 4, cudaMemcpyHostToDevice));
  for (int r = 0; r < TMAX; ++r) { double s = 0; for (int k = 0; k < LDX; ++k) s += (double)hx[r * LDX + k] * hx[r * LDX + k]; hss[r] = (float)s; }
  CK(cudaMemcpy(ss, hss.data(), TMAX * 4, cudaMemcpyHostToDevice));
  k_ref<<<256, 256>>>(x, ss, lnw, xb, TMAX); CK(cudaDeviceSynchronize());
  std::vector<uint16_t> hxb(TMAX * LDX); CK(cudaMemcpy(hxb.data(), xb, hxb.size() * 2, cudaMemcpyDeviceToHost));
  { // host double reference vs device reference (sample all): relative error bound of bf16
    double mx = 0; for (int r = 0; r < TMAX; ++r) { const double rr = 1.0 / sqrt((double)hss[r] / LDX + EPS); for (int k = 0; k < LDX; ++k) { double ref = hx[r * LDX + k] * rr * hw[k]; double got = bf2f(hxb[r * LDX + k]); mx = std::max(mx, fabs(got - ref) / (fabs(ref) + 1e-3)); } }
    fprintf(stderr, "ref check: max rel err device-ref vs host double = %.4g\n", mx); if (mx > 1e-2) { fprintf(stderr, "REF FAIL\n"); return 2; }
  }
  const int smem = 1024 + NA * ABYTES + BREG + 2048;
  CK(cudaFuncSetAttribute(k_b4, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, k_b4));
  unsigned* cnt; ull *st, *ck; CK(cudaMalloc(&cnt, 1024)); CK(cudaMalloc(&st, (size_t)GRID * NSTMAX * 8)); CK(cudaMalloc(&ck, GRID * 4 * 8 * 64));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  long long st_off = 0;
  const char* mname[] = {"z", "a1", "a2", "b", "c"};

  for (int T : Ts) {
    const int TT = (T + 15) / 16 * 16;
    CUtensorMap mB = tmap_bf16(xb, LDX, T, 64, TT);
    CUtensorMap mF = tmap_bf16(x, 2 * LDX, T, 64, TT);   // fp32 copy viewed as bf16 pairs: same bytes, 2x boxes
    for (int kb : kbs) for (int md : modes) for (int ld : lds) {
      const int nbox = md == 4 ? kb / 32 : kb / 64;
      const int cap = BREG / (TT * 128), nslots = std::min(nbox, cap);
      int chk = -1;   // -1 not checked (wraps / z / c), 1 pass, 0 fail
      if ((md == 1 || md == 2 || md == 3) && nslots == nbox) {
        CK(cudaMemset(cnt, 0, 1024)); CK(cudaMemset(dump, 0xEE, BREG));
        launch_coop(k_b4, GRID, smem, mW, mB, mF, cnt, (ull*)nullptr, ck, dump, 2, md, ld, T, TT, kb, nslots, nbox, x, ss, lnw, rows);
        cudaError_t er = cudaDeviceSynchronize(); if (er != cudaSuccess) { fprintf(stderr, "verify launch error %s\n", cudaGetErrorString(er)); return 3; }
        std::vector<uint8_t> hd(BREG); CK(cudaMemcpy(hd.data(), dump, BREG, cudaMemcpyDeviceToHost));
        long long bad = 0;
        for (int j = 0; j < nbox; ++j) for (int r = 0; r < TT; ++r) for (int ch = 0; ch < 8; ++ch) {
          const uint8_t* g = hd.data() + (size_t)j * TT * 128 + r * 128 + ((ch ^ (r & 7)) << 4);
          for (int e = 0; e < 8; ++e) { uint16_t want = r < T ? hxb[r * LDX + 64 * j + ch * 8 + e] : 0, got; memcpy(&got, g + 2 * e, 2); bad += got != want; }
        }
        chk = bad == 0; if (!chk) fprintf(stderr, "CORRECTNESS FAIL mode %s T %d kb %d loaded %d: %lld mismatching bf16\n", mname[md], T, kb, ld, bad);
      }
      std::vector<double> rT, rW, rL, rTm, rWm, rLP, rM;
      int stored = 0; const long long off0 = st_off; int ok = 1;
      for (int l = 0; l < warm + nl; ++l) {
        CK(cudaMemset(cnt, 0, 1024)); CK(cudaMemset(st, 0, (size_t)GRID * NSTMAX * 8)); CK(cudaMemset(ck, 0, GRID * 4 * 8));
        CK(cudaEventRecord(e0));
        launch_coop(k_b4, GRID, smem, mW, mB, mF, cnt, st, ck, (uint8_t*)nullptr, nph, md, ld, T, TT, kb, nslots, nbox, x, ss, lnw, rows);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        cudaError_t er = cudaGetLastError(); if (er != cudaSuccess) { fprintf(stderr, "launch error %s\n", cudaGetErrorString(er)); ok = 0; break; }
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        std::vector<ull> h((size_t)GRID * NSTMAX), hc(GRID * 4);
        CK(cudaMemcpy(h.data(), st, h.size() * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hc.data(), ck, hc.size() * 8, cudaMemcpyDeviceToHost));
        const double mhz = mhz_of(hc, GRID);
        if (l < warm) continue;
        Met m = metrics(h, nph);
        rT.push_back(m.T); rW.push_back(m.work); rL.push_back(m.lat); rTm.push_back(m.Tmean); rWm.push_back(m.wmean); rLP.push_back(ms * 1e3 / nph); rM.push_back(mhz);
        if (stored < stl) { stamps_append(outst, h); ++stored; ++st_off; }
      }
      J j; j.s("bench", "b4_bpath").s("arm", mname[md]).s("tag", tag).i("mode", md).i("T", T).i("TT", TT).i("kb", kb).i("loaded", ld).i("nslots", nslots).i("nbox", nbox)
          .i("bytes_smem", (long long)TT * kb * 2).i("bytes_l2", (long long)(md == 0 ? 0 : T * kb * (md == 3 ? 2 : 4)))
          .i("nph", nph).i("nl", nl).d("us_T", med(rT)).d("us_T_min", vmin(rT)).d("us_T_max", vmax(rT)).d("us_Tmean", med(rTm)).d("us_work", med(rW)).d("us_work_mean", med(rWm))
          .d("us_lat", med(rL)).d("us_launch_per_phase", med(rLP)).arr("us_reps", rT).arr("us_Tmean_reps", rTm).d("mhz", med(rM)).d("mhz_min", vmin(rM)).d("mhz_max", vmax(rM))
          .i("chk", chk).i("pass", ok && chk != 0).i("regs", fa.numRegs).i("spill", (int)fa.localSizeBytes).i("smem", smem).s("job", job ? job : "local")
          .i("st_off", off0).i("st_n", stored).i("grid", GRID).i("nstmax", NSTMAX);
      j.print();
    }
  }
  return 0;
}
