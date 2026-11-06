// B0 b0_calib: harness check. (a) empty phases: bar_arrive / bar_wait, N CTAs spin W; (b) + window TMA loads (32 KB each) between arrive and wait;
// (c) clock: idle phase sequence vs tcgen05-busy phase sequence; plus the globaltimer step.
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -DNSTMAX=232 -I../common b0_calib.cu
#ifndef NSTMAX
#define NSTMAX 232
#endif
#include "ub.cuh"
#include "host.h"

constexpr int NLD_MAX = 4, SLOT_B = 32768;   // TMA window loads: up to 4 slots of 32 KB (2 boxes of 64 k x 128 rows)
__device__ __forceinline__ void spin_clk(ull cyc) { const ull t0 = clock64(); while ((ull)(clock64() - t0) < cyc) {} }

// mode 0: phase = optional spin (active CTAs) ; mode 1: phase = `reps` tcgen05 MMAs (M128 N256 K16, smem operands) on every CTA.
__global__ void __launch_bounds__(NT, 1) k_b0(const __grid_constant__ CUtensorMap mW, unsigned* cnt, ull* st, ull* ck, int nph, int N, int place, ull w_ns, ull w_cyc,
                                              int spinmode, int nld, long long rows, int mode, int reps) {
  extern __shared__ uint8_t smraw[];
  const u32 raw = smem_u32(smraw), base = (raw + 1023) & ~1023u;
  uint8_t* sm = smraw + (base - raw);
  const int tid = threadIdx.x, warp = tid >> 5;
  const u32 bars = base + NLD_MAX * SLOT_B, tmab = bars, accb = bars + 8;
  volatile u32* tslot = reinterpret_cast<volatile u32*>(sm + (bars - base) + 512);
  if (mode == 1) {
    if (warp == 0) tmem_alloc(smem_u32((void*)tslot), TMEM_COLS);
    for (int i = tid; i < 48 * 1024 / 2; i += NT) ((bf16*)sm)[i] = __float2bfloat16(hashf(i + 7u) * 0.25f);
    fence_proxy_async();
  }
  if (tid == 0) { mbar_init(tmab, 1); mbar_init(accb, 1); mbar_fence_init(); }
  __syncthreads();
  tc_fence_after();
  const u32 tb = mode == 1 ? *tslot : 0;
  const bool active = place ? (blockIdx.x < (unsigned)N) : (((blockIdx.x + 1) * N) / GRID > (blockIdx.x * N) / GRID);
  const u32 idesc = tc_idesc(128, 256);
  unsigned target = 0; u32 tpar = 0, apar = 0; bool pending = false;
  clk_sample(ck, 0);
  bar_arrive(cnt, st, 0); bar_wait(cnt, target, st, 0);   // start barrier
  for (int p = 1; p <= nph; ++p) {
    // ---- phase work ----
    if (mode == 0) {
      if (tid == 0 && active && (spinmode ? w_cyc : w_ns)) { if (spinmode) spin_clk(w_cyc); else spin_ns(w_ns); }
    } else {
      if (tid == 32) {
        tc_fence_after();
        for (int i = 0; i < reps; ++i) tc_mma(tb, sdesc(base + (i & 3) * 32), sdesc(base + 16384 + (i & 3) * 32), idesc, i > 0 ? 1u : 0u);
        tc_commit(accb);
      }
      if (tid == 0) { mbar_wait(accb, apar); apar ^= 1; }
    }
    if (tid == 0 && pending) { mbar_wait(tmab, tpar); tpar ^= 1; pending = false; }   // window loads of the previous phase landed
    // ---- split barrier with window prefetch ----
    bar_arrive(cnt, st, p);
    if (tid == 0 && nld > 0) {
      mbar_expect_tx(tmab, nld * SLOT_B);
      for (int i = 0; i < nld; ++i) {
        const long long row = ((((long long)p * GRID + blockIdx.x) * NLD_MAX + i) * 128) % (rows - 128);
        tma2(base + i * SLOT_B, &mW, 0, (int)row, tmab); tma2(base + i * SLOT_B + 16384, &mW, 64, (int)row, tmab);
      }
      pending = true;
    }
    bar_wait(cnt, target, st, p);
  }
  if (tid == 0 && pending) mbar_wait(tmab, tpar);
  clk_sample(ck, 1);
  __syncthreads();
  if (mode == 1) { tc_fence_after(); if (warp == 0) tmem_dealloc(tb, TMEM_COLS); }
}

__global__ void k_gtstep(ull* out, int n) {   // thread 0 of each CTA: min nonzero globaltimer delta and number of distinct values in n reads
  if (threadIdx.x) return;
  ull prev = gt(), mn = ~0ull, distinct = 1;
  for (int i = 0; i < n; ++i) { ull t = gt(); if (t != prev) { if (t - prev < mn) mn = t - prev; ++distinct; prev = t; } }
  out[blockIdx.x * 2] = mn; out[blockIdx.x * 2 + 1] = distinct;
}

static std::vector<int> ilist(const char* s) { std::vector<int> v; for (char* t = strtok(strdup(s), ","); t; t = strtok(nullptr, ",")) v.push_back(atoi(t)); return v; }
static const char* garg(int c, char** v, const char* k, const char* d) { for (int i = 1; i + 1 < c; ++i) if (!strcmp(v[i], k)) return v[i + 1]; return d; }

struct Met { double T, work, busy, idle, last, spread, lat; };
static Met metrics(const std::vector<ull>& st, int nph, const std::vector<char>& mask) {   // one launch, phases 2..nph, us
  std::vector<double> T, W, B, I, L, S, A;
  auto arr = [&](int c, int p) { return (double)st[(size_t)c * NSTMAX + 2 * p]; };
  auto ex = [&](int c, int p) { return (double)st[(size_t)c * NSTMAX + 2 * p + 1]; };
  for (int p = 2; p <= nph; ++p) {
    std::vector<double> e, ep, a, w, wb, wi;
    double amax = 0, wmax = 0;
    for (int c = 0; c < GRID; ++c) {
      e.push_back(ex(c, p)); ep.push_back(ex(c, p - 1)); a.push_back(arr(c, p)); double wk = arr(c, p) - ex(c, p - 1); w.push_back(wk);
      (mask[c] ? wb : wi).push_back(wk); amax = std::max(amax, arr(c, p)); wmax = std::max(wmax, wk);
    }
    const double me = med(e);
    T.push_back((me - med(ep)) / 1e3); W.push_back(med(w) / 1e3); B.push_back(med(wb) / 1e3); I.push_back(med(wi) / 1e3); L.push_back(wmax / 1e3);
    S.push_back((amax - med(a)) / 1e3); A.push_back((me - amax) / 1e3);
  }
  return {med(T), med(W), med(B), med(I), med(L), med(S), med(A)};
}

int main(int argc, char** argv) {
  const char* arm = garg(argc, argv, "--arm", "all"); const char* outst = garg(argc, argv, "--out", nullptr);
  const int nph = atoi(garg(argc, argv, "--nph", "112")), nl = atoi(garg(argc, argv, "--launches", "7")), warm = atoi(garg(argc, argv, "--warm", "2"));
  const int stl = atoi(garg(argc, argv, "--stlaunches", "2")), reps = atoi(garg(argc, argv, "--reps", "48"));
  const double dur_s = atof(garg(argc, argv, "--dur", "4"));
  const char* tag = garg(argc, argv, "--tag", ""); const char* job = getenv("SLURM_JOB_ID");
  std::vector<int> Ns = ilist(garg(argc, argv, "--N", "8,16,32,64,96,128,148")), Ws = ilist(garg(argc, argv, "--W", "0,500,1000,2000,4000"));
  std::vector<int> NsB = ilist(garg(argc, argv, "--NB", "8,48,148")), nlds = ilist(garg(argc, argv, "--nld", "1,2,3,4"));
  if (nph + 1 > NSTMAX / 2 - 1) { fprintf(stderr, "nph too large for NSTMAX\n"); return 1; }
  int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  if (nsm != GRID) fprintf(stderr, "warning: %d SMs (expected %d)\n", nsm, GRID);
  Pool P = pool_alloc(); const int K = 1024; const long long rows = (long long)(P.bytes / (2ull * K));
  CUtensorMap mW = tmap_bf16(P.p, K, rows, 64, 128);
  const int smem = NLD_MAX * SLOT_B + SMEM_FIXED;
  CK(cudaFuncSetAttribute(k_b0, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, k_b0));
  unsigned* cnt; ull *st, *ck; CK(cudaMalloc(&cnt, 1024)); CK(cudaMalloc(&st, (size_t)GRID * NSTMAX * 8)); CK(cudaMalloc(&ck, GRID * 4 * 8 * 64));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  long long st_off = 0;
  double mhz_cur = 1965.;

  auto run_cfg = [&](const char* armname, int N, int place, int W, int spinmode, int nld, int mode, int rp) {
    std::vector<char> mask(GRID, 0);
    for (int b = 0; b < GRID; ++b) mask[b] = mode == 1 ? 1 : (place ? b < N : ((b + 1) * N) / GRID > (b * N) / GRID);
    std::vector<double> rT, rW, rB, rI, rL, rS, rA, rLP, rM;
    int stored = 0; const long long off0 = st_off; int ok = 1;
    for (int l = 0; l < warm + nl; ++l) {
      CK(cudaMemset(cnt, 0, 1024)); CK(cudaMemset(st, 0, (size_t)GRID * NSTMAX * 8)); CK(cudaMemset(ck, 0, GRID * 4 * 8));
      const ull wcyc = (ull)(W * mhz_cur / 1e3);
      CK(cudaEventRecord(e0));
      launch_coop(k_b0, GRID, smem, mW, cnt, st, ck, nph, N, place, (ull)W, wcyc, spinmode, nld, rows, mode, rp);
      CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      cudaError_t er = cudaGetLastError(); if (er != cudaSuccess) { fprintf(stderr, "launch error %s\n", cudaGetErrorString(er)); ok = 0; break; }
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
      std::vector<ull> h((size_t)GRID * NSTMAX), hc(GRID * 4);
      CK(cudaMemcpy(h.data(), st, h.size() * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hc.data(), ck, hc.size() * 8, cudaMemcpyDeviceToHost));
      const double mhz = mhz_of(hc, GRID); mhz_cur = mhz;
      if (l < warm) continue;
      Met m = metrics(h, nph, mask);
      rT.push_back(m.T); rW.push_back(m.work); rB.push_back(m.busy); rI.push_back(m.idle); rL.push_back(m.last); rS.push_back(m.spread); rA.push_back(m.lat);
      rLP.push_back(ms * 1e3 / nph); rM.push_back(mhz);
      if (stored < stl) { stamps_append(outst, h); ++stored; ++st_off; }
    }
    J j; j.s("bench", "b0_calib").s("arm", armname).s("tag", tag).i("N", N).i("place", place).i("W_ns", W).i("spin", spinmode).i("nld", nld).i("mode", mode).i("reps", rp)
        .i("nph", nph).i("nl", nl).d("us_T", med(rT)).d("us_T_min", vmin(rT)).d("us_T_max", vmax(rT)).d("us_work", med(rW)).d("us_busy", med(rB)).d("us_idle", med(rI))
        .d("us_last", med(rL)).d("us_spread", med(rS)).d("us_lat", med(rA)).d("us_lat_min", vmin(rA)).d("us_lat_max", vmax(rA)).d("us_launch_per_phase", med(rLP)).arr("us_reps", rT)
        .arr("lat_reps", rA).d("mhz", med(rM)).d("mhz_min", vmin(rM)).d("mhz_max", vmax(rM)).i("pass", ok).i("regs", fa.numRegs).i("spill", (int)fa.localSizeBytes).i("smem", smem)
        .s("job", job ? job : "local").i("st_off", off0).i("st_n", stored).i("grid", GRID).i("nstmax", NSTMAX);
    j.print();
  };

  const bool all = !strcmp(arm, "all");
  if (all || !strcmp(arm, "gtstep")) {
    ull* d; CK(cudaMalloc(&d, GRID * 16)); k_gtstep<<<GRID, 32>>>(d, 200000); CK(cudaDeviceSynchronize());
    std::vector<ull> h(GRID * 2); CK(cudaMemcpy(h.data(), d, GRID * 16, cudaMemcpyDeviceToHost));
    std::vector<double> mn, dc; for (int b = 0; b < GRID; ++b) { mn.push_back((double)h[2 * b]); dc.push_back((double)h[2 * b + 1]); }
    J j; j.s("bench", "b0_calib").s("arm", "gtstep").d("step_ns_med", med(mn)).d("step_ns_min", vmin(mn)).d("step_ns_max", vmax(mn)).d("distinct_per_200k_reads", med(dc)).s("job", job ? job : "local"); j.print();
  }
  if (all || !strcmp(arm, "a")) {
    for (int sp = 0; sp < 2; ++sp) for (int N : Ns) for (int W : Ws) run_cfg("a", N, 0, W, sp, 0, 0, 0);
    for (int N : Ns) run_cfg("a_first", N, 1, 1000, 0, 0, 0, 0);
  }
  if (all || !strcmp(arm, "b")) for (int nld : nlds) for (int N : NsB) for (int W : Ws) run_cfg("b", N, 0, W, 0, nld, 0, 0);
  if (!strcmp(arm, "c_idle") || !strcmp(arm, "c_busy")) {   // long run of batches of 32 launches; stamps only on launch 0 of the batch (phase T of the sequence)
    const int mode = !strcmp(arm, "c_busy"); const int W = atoi(garg(argc, argv, "--cw", "3000"));
    std::vector<double> mh, Tv; std::vector<char> mask(GRID, 1);
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    double elapsed = 0; int batches = 0;
    while (elapsed < dur_s * 1e3) {
      CK(cudaMemset(ck, 0, GRID * 4 * 8 * 32));
      CK(cudaEventRecord(t0));
      for (int l = 0; l < 32; ++l) {
        CK(cudaMemsetAsync(cnt + 0, 0, 1024));
        launch_coop(k_b0, GRID, smem, mW, cnt, l == 0 ? st : (ull*)nullptr, ck + (size_t)l * GRID * 4, nph, GRID, 0, (ull)W, (ull)0, 0, 0, rows, mode, reps);
      }
      CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1)); float ms; CK(cudaEventElapsedTime(&ms, t0, t1)); elapsed += ms; ++batches;
      cudaError_t er = cudaGetLastError(); if (er != cudaSuccess) { fprintf(stderr, "launch error %s\n", cudaGetErrorString(er)); break; }
      std::vector<ull> hc(GRID * 4 * 32); CK(cudaMemcpy(hc.data(), ck, hc.size() * 8, cudaMemcpyDeviceToHost));
      for (int l = 0; l < 32; ++l) mh.push_back(mhz_of(std::vector<ull>(hc.begin() + (size_t)l * GRID * 4, hc.begin() + (size_t)(l + 1) * GRID * 4), GRID));
      std::vector<ull> h((size_t)GRID * NSTMAX); CK(cudaMemcpy(h.data(), st, h.size() * 8, cudaMemcpyDeviceToHost));
      Tv.push_back(metrics(h, nph, mask).T);
    }
    const size_t n = mh.size(), q = std::max<size_t>(1, n / 10);
    std::vector<double> first(mh.begin(), mh.begin() + q), last(mh.end() - q, mh.end());
    J j; j.s("bench", "b0_calib").s("arm", arm).s("tag", tag).i("N", GRID).i("W_ns", W).i("mode", mode).i("reps", mode ? reps : 0).i("nph", nph).i("launches", (long long)n)
        .d("wall_ms", elapsed).d("us_T", med(Tv)).d("us_launch_per_phase", elapsed * 1e3 / (n * nph)).d("mhz", med(mh)).d("mhz_min", vmin(mh)).d("mhz_max", vmax(mh))
        .d("mhz_first10pct", med(first)).d("mhz_last10pct", med(last)).i("pass", 1).i("regs", fa.numRegs).i("spill", (int)fa.localSizeBytes).i("smem", smem).s("job", job ? job : "local"); j.print();
  }
  return 0;
}
