// B6 b6_fin: serial materialization (finalizer) chain. After a split-K red.add reduction into fp32 x [T][H], time from the last producer's ticket to the
// complete bf16 copy xb (+ ss = sum of squares per token). mk semantics (regimebench/kernels/mk/mk_step.cu idle_finalize, MK_FIN 0 / 2):
//   place 0 = base (producers red.add, no tickets, no finalizer; the "classic" arm for the gate)
//   place 1 = (a) idle CTAs, one job (rb, tt) per idle CTA (job = rb * NTT + tt, idle rank r takes r, r + nidle, ...)   [MK_FIN 0]
//   place 2 = (b) the CTA whose ticket add completes a block (old + nk == KS) finalizes that job itself (no polling)
//   place 3 = (c) all CTAs after their own tickets; each job split over nsub = max(1, G / NJOB) CTAs by token rows, seen counters  [MK_FIN 2]
//   var: 0 full, 1 empty (poll + fence only), 2 read + convert + ss but no xb store (store guarded by an impossible value)
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -DNSTMAX=232 -I../common b6_fin.cu
#ifndef NSTMAX
#define NSTMAX 232
#endif
#include "ub.cuh"
#include "host.h"

constexpr int NBMAX = 16, MAXT = 256, JMAX = 64;
struct Prm {
  int NB, H, T, KS, NTT, place, var, Pc, nph, nsleep, nunits;
  ull pdelay;
  float* x[2]; float* ss[2]; bf16* xb; const bf16* lnw; unsigned* tk; unsigned* seen; ull* tp; ull* fe;
};
__device__ __forceinline__ float pval(int p, int kc, int rb, int t, int row) { return hashf((u32)(((p * 32 + kc) * 16 + rb) * 65536u + (u32)(t * 128 + row))) * 0.25f; }
__device__ __forceinline__ u32 pk2(float a, float b) { __nv_bfloat162 v = __floats2bfloat162_rn(a, b); return *reinterpret_cast<u32*>(&v); }
__device__ __forceinline__ float bflo(u32 u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bfhi(u32 u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ uint4 pack_xw(const float4 a, const float4 b, const uint4 w) {   // as mk
  return make_uint4(pk2(a.x * bflo(w.x), a.y * bfhi(w.x)), pk2(a.z * bflo(w.y), a.w * bfhi(w.y)), pk2(b.x * bflo(w.z), b.y * bfhi(w.z)), pk2(b.z * bflo(w.w), b.w * bfhi(w.w)));
}
__device__ __forceinline__ void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }

// the item loop of mk idle_finalize for tokens [tt * 128 + p0, + np) of row block rb (all threads)
__device__ __forceinline__ void fin_body(const Prm& P, const float* x, float* ss, int rb, int tt, int p0, int np) {
  const int tid = threadIdx.x, H = P.H, items = np * 16;
  const bf16* lw = P.lnw + rb * 128;
  for (int i0 = 0; i0 < items; i0 += NT) {   // block-uniform trip count
    const int i = i0 + tid, t = tt * 128 + p0 + (i >> 4), cg = i & 15;
    const bool ok = i < items;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = a;
    if (ok) { const float* xp = x + (size_t)t * H + rb * 128 + cg * 8; a = __ldcg(reinterpret_cast<const float4*>(xp)); b = __ldcg(reinterpret_cast<const float4*>(xp + 4)); }
    const uint4 pk = pack_xw(a, b, __ldg(reinterpret_cast<const uint4*>(lw + cg * 8)));
    if (P.var == 0) { if (ok) *reinterpret_cast<uint4*>(P.xb + (size_t)t * H + rb * 128 + cg * 8) = pk; }
    else if (pk.x == 0x7fc07fc1u && pk.y == 0x7fc17fc2u && pk.z == 0x12345678u && ok) P.xb[0] = (bf16)0;   // keep the converts alive in the no-store arm
    float sq = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
#pragma unroll
    for (int o = 8; o; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    if (ok && cg == 0) atomicAdd(ss + t, sq);
  }
}
__device__ __forceinline__ void poll(unsigned* tick, unsigned target, int nsleep) {   // tid 0, as mk
  unsigned v;
  do { asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(tick) : "memory"); if (v != target && nsleep) __nanosleep(nsleep); } while (v != target);
}

__global__ void __launch_bounds__(NT, 1) k_fin(const __grid_constant__ Prm P, unsigned* cnt, ull* st, ull* ck) {
  __shared__ int s_jobs[JMAX], s_nj;
  const int tid = threadIdx.x, G = gridDim.x, b = blockIdx.x, NB = P.NB, H = P.H, T = P.T, KS = P.KS, NTT = P.NTT, Pc = P.Pc, NTILE = P.nunits, NJOB = NB * NTT;
  const int lo = (b * Pc) / G, hi = ((b + 1) * Pc) / G;
  const bool prod = hi > lo;
  const int u0 = prod ? (lo * NTILE) / Pc : 0, u1 = prod ? ((lo + 1) * NTILE) / Pc : 0;
  unsigned target = 0;
  clk_sample(ck, 0);
  bar_arrive(cnt, st, 0); bar_wait(cnt, target, st, 0);
  for (int p = 1; p <= P.nph; ++p) {
    float* xc = P.x[p & 1]; float* xn = P.x[(p + 1) & 1]; float* sc = P.ss[p & 1]; float* sn = P.ss[(p + 1) & 1];
    for (int i = b * NT + tid; i < T * H / 4; i += G * NT) reinterpret_cast<float4*>(xn)[i] = make_float4(0.f, 0.f, 0.f, 0.f);   // next phase's buffers (last read a phase ago)
    if (b == 0) for (int i = tid; i < T; i += NT) sn[i] = 0.f;
    if (tid == 0 && prod && P.pdelay) spin_ns(P.pdelay);
    __syncthreads();
    ull* fe = P.fe + ((size_t)p * G + b) * 4;
    if (tid == 0) s_nj = 0;
    // ---- producer: red.add of [128 rows][T] per unit; after the last unit of a row block: fence + tickets ----
    if (prod) {
      for (int u = u0; u < u1;) {
        const int rb = u / KS; int ue = min(u1, (rb + 1) * KS);
        for (int uu = u; uu < ue; ++uu) {
          const int kc = uu % KS;
          for (int idx = tid; idx < 128 * T; idx += NT) { const int row = idx & 127, t = idx >> 7; red_add(xc + (size_t)t * H + rb * 128 + row, pval(p, kc, rb, t, row)); }
        }
        if (P.place != 0) {
          const unsigned nk = ue - u;
          __syncthreads();   // every thread's red.adds of this row block are issued
          if (tid == 0) {
            __threadfence();
            const ull ts = gt();
            for (int tt = 0; tt < NTT; ++tt) {
              unsigned* tick = P.tk + rb * NTT + tt;
              if (P.place == 2) { const unsigned old = atomicAdd(tick, nk); if (old + nk == (unsigned)KS) { __threadfence(); if (s_nj < JMAX) s_jobs[s_nj++] = rb * NTT + tt; } }
              else atomicAdd(tick, nk);
            }
            atomicMax(P.tp + p * NBMAX + rb, ts);
          }
        }
        u = ue;
      }
    }
    // ---- finalizer ----
    if (P.place == 1) {
      const bool idle = !prod; const int nidle = G - Pc, rank = b - lo;   // mk idle_rank (Pc <= G)
      if (idle) {
        for (int job = rank; job < NJOB; job += nidle) {
          const int rb = job / NTT, tt = job % NTT;
          unsigned* tick = P.tk + job;
          if (tid == 0) { poll(tick, KS, P.nsleep); atomicExch(tick, 0u); fe[0] = gt(); fe[3] = job + 1; }
          __syncthreads();
          const int ntl = min(T - tt * 128, 128);
          if (P.var != 1) fin_body(P, xc, sc, rb, tt, 0, ntl);
        }
        __syncthreads();
        if (tid == 0 && rank < NJOB) fe[1] = gt();
        fence_proxy_async_global();
        __syncthreads();
        if (tid == 0 && rank < NJOB) fe[2] = gt();
      }
    } else if (P.place == 3) {
      const int nsub = max(1, G / NJOB), pr = (128 + nsub - 1) / nsub;
      for (int w = b; w < NJOB * nsub; w += G) {
        const int job = w % NJOB, part = w / NJOB, rb = job / NTT, tt = job % NTT;
        unsigned* tick = P.tk + job;
        if (tid == 0) {
          poll(tick, KS, P.nsleep);
          if (nsub == 1) atomicExch(tick, 0u);
          else { unsigned* seen = P.seen + job; if (atomicAdd(seen, 1u) == (unsigned)nsub - 1) { atomicExch(seen, 0u); atomicExch(tick, 0u); } }
          fe[0] = gt(); fe[3] = job + 1;
        }
        __syncthreads();
        const int ntl = min(T - tt * 128, 128), p0 = part * pr, np = max(0, min(ntl - p0, pr));
        if (P.var != 1) fin_body(P, xc, sc, rb, tt, p0, np);
      }
      __syncthreads();
      if (tid == 0 && b < NJOB * nsub) fe[1] = gt();
      fence_proxy_async_global();
      __syncthreads();
      if (tid == 0 && b < NJOB * nsub) fe[2] = gt();
    } else if (P.place == 2) {
      __syncthreads();   // s_jobs list complete
      const int nj = s_nj;
      if (nj > 0) {
        for (int k = 0; k < nj; ++k) {
          const int job = s_jobs[k], rb = job / NTT, tt = job % NTT;
          if (tid == 0) { atomicExch(P.tk + job, 0u); fe[0] = gt(); fe[3] = job + 1; }
          __syncthreads();
          const int ntl = min(T - tt * 128, 128);
          if (P.var != 1) fin_body(P, xc, sc, rb, tt, 0, ntl);
        }
        __syncthreads();
        if (tid == 0) fe[1] = gt();
        fence_proxy_async_global();
        __syncthreads();
        if (tid == 0) fe[2] = gt();
      }
    }
    bar_arrive(cnt, st, p);
    bar_wait(cnt, target, st, p);
  }
  clk_sample(ck, 1);
}

// reference: x vs the sum of the partials (double), xb vs bf16(x * w) (bit exact), ss vs sum x^2
__global__ void k_check(Prm P, int p, int* err) {
  const float* x = P.x[p & 1]; const float* ss = P.ss[p & 1];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < P.T * P.H; i += gridDim.x * blockDim.x) {
    const int t = i / P.H, col = i % P.H, rb = col / 128, row = col & 127;
    double e = 0; for (int kc = 0; kc < P.KS; ++kc) e += pval(p, kc, rb, t, row);
    if (fabs((double)x[i] - e) > 2e-5) atomicAdd(err + 0, 1);
    if (P.place != 0 && P.var == 0) {
      const bf16 r = __float2bfloat16_rn(x[i] * __bfloat162float(P.lnw[col]));
      if (*reinterpret_cast<const unsigned short*>(&r) != *reinterpret_cast<const unsigned short*>(P.xb + i)) atomicAdd(err + 1, 1);
    }
  }
  if (P.place != 0 && P.var != 1) {
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < P.T; t += gridDim.x * blockDim.x) {
      double s = 0; for (int c = 0; c < P.H; ++c) { double v = x[(size_t)t * P.H + c]; s += v * v; }
      if (fabs(s - ss[t]) > 1e-3 * s + 1e-6) atomicAdd(err + 2, 1);
    }
  }
}

static std::vector<int> ilist(const char* s) { std::vector<int> v; char* c = strdup(s); for (char* t = strtok(c, ","); t; t = strtok(nullptr, ",")) v.push_back(atoi(t)); return v; }
static const char* garg(int c, char** v, const char* k, const char* d) { for (int i = 1; i + 1 < c; ++i) if (!strcmp(v[i], k)) return v[i + 1]; return d; }
static double mean(const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return v.empty() ? -1 : s / v.size(); }

int main(int argc, char** argv) {
  const char* outst = garg(argc, argv, "--out", nullptr); const char* tag = garg(argc, argv, "--tag", ""); const char* job = getenv("SLURM_JOB_ID");
  const int nph = atoi(garg(argc, argv, "--nph", "112")), nl = atoi(garg(argc, argv, "--launches", "7")), warm = atoi(garg(argc, argv, "--warm", "2"));
  const int nsleep = atoi(garg(argc, argv, "--nsleep", "64")), pcmax = atoi(garg(argc, argv, "--pc", "128"));
  const int stsel = atoi(garg(argc, argv, "--stsel", "1"));   // 1: keep stamps of one launch for var 0 / base rows
  std::vector<int> Ts = ilist(garg(argc, argv, "--T", "1,8,16,32,64,128,256")), Hs = ilist(garg(argc, argv, "--H", "1024,2048")), KSs = ilist(garg(argc, argv, "--KS", "4,12,16,24"));
  std::vector<int> places = ilist(garg(argc, argv, "--place", "0,1,2,3")), vars = ilist(garg(argc, argv, "--var", "0,1,2")), pds = ilist(garg(argc, argv, "--pdelay", "1500"));
  const int reps_cfg = atoi(garg(argc, argv, "--rep", "1"));   // repeat the whole config list (second pass)
  if (nph + 1 > NSTMAX / 2 - 1) { fprintf(stderr, "nph too large\n"); return 1; }
  int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  if (nsm != GRID) fprintf(stderr, "warning: %d SMs (expected %d)\n", nsm, GRID);
  const int smem = 160 * 1024;
  CK(cudaFuncSetAttribute(k_fin, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, k_fin));
  const size_t XN = (size_t)MAXT * 2048;
  float *x0, *x1, *s0, *s1; bf16 *xb, *lnw; unsigned *cnt, *tk; ull *st, *ck, *tp, *fe; int* err;
  CK(cudaMalloc(&x0, XN * 4)); CK(cudaMalloc(&x1, XN * 4)); CK(cudaMalloc(&s0, MAXT * 4)); CK(cudaMalloc(&s1, MAXT * 4)); CK(cudaMalloc(&xb, XN * 2)); CK(cudaMalloc(&lnw, 2048 * 2));
  CK(cudaMalloc(&cnt, 1024)); CK(cudaMalloc(&tk, 4096)); CK(cudaMalloc(&st, (size_t)GRID * NSTMAX * 8)); CK(cudaMalloc(&ck, GRID * 4 * 8));
  CK(cudaMalloc(&tp, (size_t)(nph + 2) * NBMAX * 8)); CK(cudaMalloc(&fe, (size_t)(nph + 2) * GRID * 4 * 8)); CK(cudaMalloc(&err, 16));
  k_init_bf16<<<64, 256>>>(lnw, 2048, 99u); CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  long long st_off = 0;

  auto run_cfg = [&](int place, int var, int T, int H, int KS, int pd) {
    Prm P = {}; P.NB = H / 128; P.H = H; P.T = T; P.KS = KS; P.NTT = (T + 127) / 128; P.place = place; P.var = var; P.nph = nph; P.nsleep = nsleep; P.nunits = P.NB * KS; P.pdelay = (ull)pd;
    P.Pc = std::min(P.nunits, std::min(pcmax, GRID)); if (place == 1 && P.Pc >= GRID) P.Pc = GRID - 1;
    P.x[0] = x0; P.x[1] = x1; P.ss[0] = s0; P.ss[1] = s1; P.xb = xb; P.lnw = lnw; P.tk = tk; P.seen = tk + 512; P.tp = tp; P.fe = fe;
    std::vector<char> mask(GRID); for (int c = 0; c < GRID; ++c) mask[c] = ((c + 1) * P.Pc) / GRID > (c * P.Pc) / GRID;
    const int NJOB = P.NB * P.NTT;
    std::vector<double> rT, rTm, rL, rLb, rLat, rBusy, rIdle, rF, rFm, rO, rR, rW, rLP, rM, rFl, fall, oall, rall, wall;
    long long off0 = st_off; int stored = 0, ok = 1, perr[3] = {0, 0, 0};
    for (int l = 0; l < warm + nl; ++l) {
      CK(cudaMemset(cnt, 0, 1024)); CK(cudaMemset(tk, 0, 4096)); CK(cudaMemset(st, 0, (size_t)GRID * NSTMAX * 8)); CK(cudaMemset(ck, 0, GRID * 4 * 8));
      CK(cudaMemset(tp, 0, (size_t)(nph + 2) * NBMAX * 8)); CK(cudaMemset(fe, 0, (size_t)(nph + 2) * GRID * 4 * 8)); CK(cudaMemset(err, 0, 16));
      CK(cudaMemset(x0, 0, XN * 4)); CK(cudaMemset(x1, 0, XN * 4)); CK(cudaMemset(s0, 0, MAXT * 4)); CK(cudaMemset(s1, 0, MAXT * 4)); CK(cudaMemset(xb, 0xff, XN * 2));
      CK(cudaDeviceSynchronize());
      CK(cudaEventRecord(e0));
      launch_coop(k_fin, GRID, smem, P, cnt, st, ck);
      CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      cudaError_t er = cudaGetLastError(); if (er != cudaSuccess) { fprintf(stderr, "launch error %s\n", cudaGetErrorString(er)); ok = 0; break; }
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
      k_check<<<148, 256>>>(P, nph, err); CK(cudaDeviceSynchronize());
      int he[4]; CK(cudaMemcpy(he, err, 16, cudaMemcpyDeviceToHost)); for (int i = 0; i < 3; ++i) { perr[i] += he[i]; if (he[i]) ok = 0; }
      std::vector<ull> h((size_t)GRID * NSTMAX), hc(GRID * 4), htp((size_t)(nph + 2) * NBMAX), hfe((size_t)(nph + 2) * GRID * 4);
      CK(cudaMemcpy(h.data(), st, h.size() * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hc.data(), ck, hc.size() * 8, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(htp.data(), tp, htp.size() * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hfe.data(), fe, hfe.size() * 8, cudaMemcpyDeviceToHost));
      const double mhz = mhz_of(hc, GRID);
      if (l < warm) continue;
      auto arr = [&](int c, int p) { return (double)h[(size_t)c * NSTMAX + 2 * p]; };
      auto ex = [&](int c, int p) { return (double)h[(size_t)c * NSTMAX + 2 * p + 1]; };
      std::vector<double> T_, L_, Lat_, Bz, Iz, F_, O_, R_, W_, Fl_;
      for (int p = 2; p <= nph; ++p) {
        std::vector<double> e, ep, wb, wi; double wmax = 0, amax = 0; int lastc = 0;
        for (int c = 0; c < GRID; ++c) {
          e.push_back(ex(c, p)); ep.push_back(ex(c, p - 1)); const double wk = arr(c, p) - ex(c, p - 1); (mask[c] ? wb : wi).push_back(wk);
          if (wk > wmax) { wmax = wk; lastc = c; } amax = std::max(amax, arr(c, p));
        }
        T_.push_back((med(e) - med(ep)) / 1e3); L_.push_back(wmax / 1e3); Lat_.push_back((med(e) - amax) / 1e3); Bz.push_back(med(wb) / 1e3); Iz.push_back(wi.empty() ? -1 : med(wi) / 1e3);
        Fl_.push_back(mask[lastc] ? 0.0 : 1.0);
        if (place != 0) {
          double Tp = 0; for (int rb = 0; rb < P.NB; ++rb) Tp = std::max(Tp, (double)htp[(size_t)p * NBMAX + rb]);
          double Tf = 0; int cc = -1;
          for (int c = 0; c < GRID; ++c) { const double v = (double)hfe[((size_t)p * GRID + c) * 4 + 2]; if (v > Tf) { Tf = v; cc = c; } }
          if (cc >= 0 && Tp > 0) {
            const ull* f = &hfe[((size_t)p * GRID + cc) * 4]; const int jb = (int)f[3] - 1, rbj = jb / P.NTT;
            F_.push_back((Tf - Tp) / 1e3); O_.push_back(((double)f[0] - (double)htp[(size_t)p * NBMAX + rbj]) / 1e3); R_.push_back(((double)f[1] - (double)f[0]) / 1e3); W_.push_back(((double)f[2] - (double)f[1]) / 1e3);
          }
        }
      }
      rT.push_back(med(T_)); rL.push_back(med(L_)); rLat.push_back(med(Lat_)); rBusy.push_back(med(Bz)); rIdle.push_back(med(Iz)); rFl.push_back(mean(Fl_));
      rTm.push_back((med([&] { std::vector<double> v; for (int c = 0; c < GRID; ++c) v.push_back(ex(c, nph)); return v; }()) - med([&] { std::vector<double> v; for (int c = 0; c < GRID; ++c) v.push_back(ex(c, 1)); return v; }())) / 1e3 / (nph - 1));
      rLb.push_back(mean(L_));
      if (!F_.empty()) { rF.push_back(med(F_)); rFm.push_back(mean(F_)); rO.push_back(med(O_)); rR.push_back(med(R_)); rW.push_back(med(W_)); }
      for (double v : F_) fall.push_back(v); for (double v : O_) oall.push_back(v); for (double v : R_) rall.push_back(v); for (double v : W_) wall.push_back(v);
      rLP.push_back(ms * 1e3 / nph); rM.push_back(mhz);
      if (stored < 1 && stsel && (var == 0 || place == 0)) { stamps_append(outst, h); ++stored; ++st_off; }
    }
    static const char* an[] = {"base", "a_idle", "b_lastprod", "c_all"};
    J j; j.s("bench", "b6_fin").s("arm", an[place]).s("tag", tag).i("place", place).i("var", var).i("T", T).i("H", H).i("NB", P.NB).i("KS", KS).i("nunits", P.nunits).i("Pc", P.Pc).i("njob", NJOB)
        .i("pdelay_ns", pd).i("nph", nph).i("nl", nl).d("us_T", med(rT)).d("us_Tmean", med(rTm)).d("us_work_last", med(rL)).d("us_work_last_mean", med(rLb)).d("us_lat", med(rLat))
        .d("us_busy", med(rBusy)).d("us_idle", med(rIdle)).d("lastfin_frac", med(rFl)).d("us_launch_per_phase", med(rLP)).arr("us_reps", rT).arr("last_reps", rL)
        .d("t_fin_med", med(rF)).d("t_fin_mean", mean(rFm)).d("t_fin_p10", [&] { auto v = fall; std::sort(v.begin(), v.end()); return v.empty() ? -1. : v[v.size() / 10]; }()).d("t_fin_p90", [&] { auto v = fall; std::sort(v.begin(), v.end()); return v.empty() ? -1. : v[v.size() * 9 / 10]; }())
        .d("t_obs_med", med(rO)).d("t_rc_med", med(rR)).d("t_wf_med", med(rW)).d("t_obs_mean", mean(oall)).d("t_rc_mean", mean(rall)).d("t_wf_mean", mean(wall)).arr("tfin_reps", rF)
        .d("mhz", med(rM)).i("pass", ok).i("err_x", perr[0]).i("err_xb", perr[1]).i("err_ss", perr[2]).i("regs", fa.numRegs).i("spill", (int)fa.localSizeBytes).i("smem", smem)
        .s("job", job ? job : "local").i("st_off", off0).i("st_n", stored).i("grid", GRID).i("nstmax", NSTMAX);
    j.print();
  };

  for (int rep = 0; rep < reps_cfg; ++rep)
    for (int pd : pds) for (int H : Hs) for (int KS : KSs) for (int T : Ts) for (int pl : places) {
      if (pl == 0) { run_cfg(0, 0, T, H, KS, pd); continue; }
      for (int v : vars) run_cfg(pl, v, T, H, KS, pd);
    }
  return 0;
}
