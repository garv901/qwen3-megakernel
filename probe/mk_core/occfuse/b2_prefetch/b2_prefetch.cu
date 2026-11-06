// B2 b2_prefetch: bytes of the weight stream hidden by the barrier-window prefetch. Built on b1_stream (copied, not edited).
// Phases alternate: odd = spin phase (all CTAs spin W, CTA 0 spins W + spread), even = stream phase (measured).
// Barrier window of a spin phase (mk: bar_arrive -> tid 0 prefetch -> bar_wait): tid 0 issues the first P ring slots of the NEXT (stream) phase.
// Stream phase = mk's MK_NULL structure: tid 0 waits full[j] and refills step j+R at once (no MMA, no epilogue); classic (mode 0) syncs all threads per step.
// mode 0 classic: slot = A (32 KB), full count 1, ring state continues across phases (mk prefetch_next).
// mode 1 TS: slot = A 32 KB + B (TT x 128 k bf16, 2 boxes from an L2-resident map), full count 2; mbarriers re-initialised in the window (mk ts_first),
//   B boxes are issued only after the barrier exit (after fence.proxy.async.global), as mk ts_phase.
// CTA c has nt_c = floor((c+1)*tot/148) - floor(c*tot/148) tiles of spt steps (mk geom / ts_geom).
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -DNSTMAX=232 -I../common b2_prefetch.cu
#ifndef NSTMAX
#define NSTMAX 232
#endif
#include "ub.cuh"
#include "host.h"

constexpr int KDIM = 1024, KSTEPS = KDIM / 128, SLOT_A = 32768;
__device__ __forceinline__ void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }

struct Cfg { int mode, R, tot, spt, P, TTB, noB, hot; ull spin_ns, spread_ns; long long nrb; };

__global__ void __launch_bounds__(NT, 1) k_b2(const __grid_constant__ CUtensorMap mW, const __grid_constant__ CUtensorMap mB, unsigned* cnt, ull* st, ull* ck, int nph, Cfg c) {
  extern __shared__ uint8_t smraw[];
  const u32 raw = smem_u32(smraw), base = (raw + 1023) & ~1023u;
  const int tid = threadIdx.x, R = c.R, mode = c.mode;
  const u32 slotb = SLOT_A + (mode ? c.TTB : 0), fullb = base + R * slotb;
  const int nt = ((blockIdx.x + 1) * c.tot) / GRID - (blockIdx.x * c.tot) / GRID, nst = nt * c.spt;
  const int pre = min(min(c.P, R), nst);          // steps whose A is issued in the barrier window
  const bool useB = mode && !c.noB;   // diagnostic: noB drops the B boxes (A only, full count 1)
  const u32 fcnt = useB ? 2 : 1;
  auto init_ring = [&]() { for (int i = 0; i < R; ++i) mbar_init(fullb + 8 * i, fcnt); mbar_fence_init(); };
  if (tid == 0) init_ring();
  __syncthreads();
  unsigned target = 0; unsigned gb = 0;            // gb = ring position at the start of the next stream phase (classic: continues; TS: always 0)
  auto fillA = [&](int s, int pidx, unsigned pos) {   // step s of stream phase pidx into ring position pos
    const int slot = pos % R; const u32 bar = fullb + 8 * slot;
    const long long u = ((long long)(c.hot ? 0 : pidx) * GRID + blockIdx.x) * nst + s;   // hot: same window every phase (A hits L2)
    const long long row = ((u / KSTEPS) % c.nrb) * 128; const int kc = (int)(u % KSTEPS) * 128;
    mbar_expect_tx(bar, SLOT_A);
    tma2(base + slot * slotb, &mW, kc, (int)row, bar); tma2(base + slot * slotb + 16384, &mW, kc + 64, (int)row, bar);
  };
  auto fillB = [&](int s, int pidx, unsigned pos) {
    const int slot = pos % R; const u32 bar = fullb + 8 * slot;
    const long long u = ((long long)pidx * GRID + blockIdx.x) * nst + s; const int kc = (int)(u % KSTEPS) * 128; const int rows = c.TTB / 256;
    mbar_expect_tx(bar, c.TTB);
    for (int a = 0; a < 2; ++a) tma2(base + slot * slotb + SLOT_A + a * (rows * 128), &mB, kc + a * 64, 0, bar);
  };
  clk_sample(ck, 0);
  bar_arrive(cnt, st, 0); bar_wait(cnt, target, st, 0);
  for (int p = 1; p <= nph; ++p) {
    if (p & 1) {                                   // spin phase; barrier window prefetch for stream phase p + 1
      if (tid == 0) spin_ns(c.spin_ns + (blockIdx.x == 0 ? c.spread_ns : 0));
      bar_arrive(cnt, st, p);
      if (tid == 0 && pre > 0) {
        if (mode) init_ring();
        for (int i = 0; i < pre; ++i) fillA(i, p + 1, gb + i);
      }
      bar_wait(cnt, target, st, p);
    } else {                                       // stream phase
      if (nst > 0) {
        if (tid == 0) {
          if (mode) { fence_proxy_async_global(); if (pre == 0) init_ring(); }
          for (int i = 0; i < R && i < nst; ++i) { if (i >= pre) fillA(i, p, gb + i); if (useB) fillB(i, p, gb + i); }
        }
        for (int j = 0; j < nst; ++j) {
          if (tid == 0) {
            const unsigned f = gb + j;
            mbar_wait(fullb + 8 * (f % R), (f / R) & 1);
            if (j + R < nst) { fillA(j + R, p, f + R); if (useB) fillB(j + R, p, f + R); }
          }
          if (mode == 0) __syncthreads();
        }
        __syncthreads();
        if (mode == 0) gb += nst;
      }
      bar_arrive(cnt, st, p); bar_wait(cnt, target, st, p);
    }
  }
  clk_sample(ck, 1);
}

static std::vector<double> dlist(const char* s) { std::vector<double> v; char* d = strdup(s); for (char* t = strtok(d, ","); t; t = strtok(nullptr, ",")) v.push_back(atof(t)); return v; }
static const char* garg(int c, char** v, const char* k, const char* d) { for (int i = 1; i + 1 < c; ++i) if (!strcmp(v[i], k)) return v[i + 1]; return d; }

struct Met { double busy, win, T, Tspin, last, spread; double busymean, winmean, Tmean, Tspinmean, lastmean, lat; };
// stream phases p = 2, 4, .. nph. busy = arrive(p) - exit(p-1) (mk "busy work"), win = exit(p-1) - arrive(p-1) of the spin barrier before it.
static Met metrics(const std::vector<ull>& st, int nph, const std::vector<char>& mask) {
  auto arr = [&](int c, int p) { return (double)st[(size_t)c * NSTMAX + 2 * p]; };
  auto ex = [&](int c, int p) { return (double)st[(size_t)c * NSTMAX + 2 * p + 1]; };
  std::vector<double> B, Wn, T, Ts, L, S, A; double bs = 0, ws = 0, ls = 0; int n = 0;
  std::vector<double> e(GRID), ep(GRID);
  for (int p = 2; p <= nph; p += 2) {
    std::vector<double> wb, ww; double wmax = 0, amax = 0; std::vector<double> a;
    for (int c = 0; c < GRID; ++c) {
      const double wk = arr(c, p) - ex(c, p - 1); amax = std::max(amax, arr(c, p)); a.push_back(arr(c, p)); wmax = std::max(wmax, wk);
      if (mask[c]) { wb.push_back(wk); ww.push_back(ex(c, p - 1) - arr(c, p - 1)); }
      e[c] = ex(c, p); ep[c] = ex(c, p - 1);
    }
    std::vector<double> e2(GRID); for (int c = 0; c < GRID; ++c) e2[c] = ex(c, p - 2 >= 0 ? p - 2 : 0);
    B.push_back(med(wb) / 1e3); Wn.push_back(med(ww) / 1e3); T.push_back((med(e) - med(ep)) / 1e3); Ts.push_back((med(ep) - med(e2)) / 1e3); L.push_back(wmax / 1e3);
    S.push_back((amax - med(a)) / 1e3); A.push_back((med(e) - amax) / 1e3);
    double s = 0; for (double x : wb) s += x; bs += s / wb.size() / 1e3; s = 0; for (double x : ww) s += x; ws += s / ww.size() / 1e3; ls += wmax / 1e3; ++n;
  }
  std::vector<double> e1, en; for (int c = 0; c < GRID; ++c) { e1.push_back(ex(c, 1)); en.push_back(ex(c, nph)); }
  const double tm = (med(en) - med(e1)) / 1e3 / (nph - 1) * 2;   // mean pair time (spin + stream phase)
  return {med(B), med(Wn), med(T), med(Ts), med(L), med(S), bs / n, ws / n, tm, 0, ls / n, med(A)};
}

struct Named { std::string name; int mode, tot, spt, R; };

int main(int argc, char** argv) {
  const char* outst = garg(argc, argv, "--out", nullptr);
  const int nph = atoi(garg(argc, argv, "--nph", "100")), nl = atoi(garg(argc, argv, "--launches", "5")), warm = atoi(garg(argc, argv, "--warm", "2"));
  const char* tag = garg(argc, argv, "--tag", ""); const char* job = getenv("SLURM_JOB_ID");
  const int gate = atoi(garg(argc, argv, "--gate", "0")), mode = atoi(garg(argc, argv, "--mode", "0")), TT = atoi(garg(argc, argv, "--TT", "16"));
  const std::vector<double> Ns = dlist(garg(argc, argv, "--N", "48,96,128,148")), Ss = dlist(garg(argc, argv, "--S", "32,64,128,256,512,1024")), Rs = dlist(garg(argc, argv, "--R", "2,3,4"));
  const std::vector<double> Ps = dlist(garg(argc, argv, "--P", "0,1,2,-1"));   // -1 = R
  const std::vector<double> stKeep = dlist(garg(argc, argv, "--keepW", "0,2"));   // gate rows: raw stamps for these W (us) at spread in {0,1,2}
  std::vector<std::pair<double, double>> WS;   // (W all-CTA spin, spread of CTA 0), us
  { char* d = strdup(garg(argc, argv, "--ws", "0:0,0.5:0,1:0,2:0,4:0,6:0,0:0.5,0:1,0:2,0:4,0:6")); for (char* t = strtok(d, ","); t; t = strtok(nullptr, ",")) { double w, s; sscanf(t, "%lf:%lf", &w, &s); WS.push_back({w, s}); } }
  const std::vector<double> gR = dlist(garg(argc, argv, "--gR", "0"));   // gate: R override list (0 = native)
  const int noB = atoi(garg(argc, argv, "--noB", "0")), hot = atoi(garg(argc, argv, "--hot", "0"));
  const int floorarm = atoi(garg(argc, argv, "--floor", "1"));
  if (nph + 1 > NSTMAX / 2 - 1 || (nph & 1)) { fprintf(stderr, "nph must be even and fit NSTMAX\n"); return 1; }
  int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  if (nsm != GRID) fprintf(stderr, "warning: %d SMs (expected %d)\n", nsm, GRID);
  Pool P = pool_alloc(); const long long rows = (long long)(P.bytes / (2ull * KDIM)), nrb = rows / 128 - 1;
  CUtensorMap mW = tmap_bf16(P.p, KDIM, rows, 64, 128);
  bf16* xb; CK(cudaMalloc(&xb, 256ull * KDIM * 2)); k_init_bf16<<<64, 256>>>(xb, 256ull * KDIM, 777u); CK(cudaDeviceSynchronize());
  CUtensorMap mB = tmap_bf16(xb, KDIM, 256, 64, TT);
  CK(cudaFuncSetAttribute(k_b2, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, k_b2));
  unsigned* cnt; ull *st, *ck; CK(cudaMalloc(&cnt, 1024)); CK(cudaMalloc(&st, (size_t)GRID * NSTMAX * 8)); CK(cudaMalloc(&ck, GRID * 4 * 8 * 64));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  long long st_off = 0;

  std::vector<Named> cfgs;
  if (gate) {   // mk MK_NULL b8 configurations (tiles.py / smallt analyze.py): tot = units or tiles over 148 CTAs, spt = steps per tile, R = mk ring depth
    cfgs = {{"cQKV", 0, 256, 1, 2}, {"cGU", 0, 384, 1, 2}, {"cD", 0, 192, 1, 2}, {"tQKV", 1, 128, 2, 5}, {"tGU", 1, 48, 8, 5}, {"tD", 1, 96, 2, 5}};
  } else {
    for (double N : Ns) for (double S : Ss) for (double R : Rs) cfgs.push_back({"gen", mode, (int)N, (int)S / 32, (int)R});
  }
  for (const Named& nc : cfgs) for (double gr : (gate ? gR : std::vector<double>{0})) {
    const int R = (gate && gr > 0) ? (int)gr : nc.R;
    const int slot = SLOT_A + (nc.mode ? TT * 256 : 0), smem = R * slot + SMEM_FIXED;
    if (smem > SMEM_MAX) { fprintf(stderr, "skip %s R=%d: smem %d > max\n", nc.name.c_str(), R, smem); continue; }
    std::vector<char> mask(GRID, 0); int nbusy = 0, maxnt = 0;
    for (int b = 0; b < GRID; ++b) { const int n = ((b + 1) * nc.tot) / GRID - (b * nc.tot) / GRID; mask[b] = n > 0; nbusy += n > 0; maxnt = std::max(maxnt, n); }
    std::vector<int> pl; for (double pp : Ps) { int pe = pp < 0 ? R : std::min((int)pp, R); if (std::find(pl.begin(), pl.end(), pe) == pl.end()) pl.push_back(pe); }
    // stream-free floor: spt = 0 (same barriers, spin phases, no TMA)
    for (int fl = floorarm ? 1 : 0; fl >= 0; --fl) for (int Pq : pl) for (auto& ws : WS) {
      if (fl && (Pq != 0)) continue;     // floor row once per W/spread (P irrelevant)
      Cfg c; c.mode = nc.mode; c.R = R; c.tot = nc.tot; c.spt = fl ? 0 : nc.spt; c.P = Pq; c.TTB = TT * 256; c.spin_ns = (ull)(ws.first * 1e3 + 0.5); c.spread_ns = (ull)(ws.second * 1e3 + 0.5); c.nrb = nrb; c.noB = noB; c.hot = hot;
      std::vector<double> rB, rBm, rW, rWm, rT, rTm, rL, rLm, rM, rLP; int stored = 0, ok = 1; const long long off0 = st_off;
      bool keep = gate && !fl && std::find(stKeep.begin(), stKeep.end(), ws.first) != stKeep.end() && (ws.second == 0 || ws.second == 1 || ws.second == 2) && (ws.first == 0 || ws.second == 0);
      for (int l = 0; l < warm + nl && ok; ++l) {
        CK(cudaMemset(cnt, 0, 1024)); CK(cudaMemset(st, 0, (size_t)GRID * NSTMAX * 8)); CK(cudaMemset(ck, 0, GRID * 4 * 8));
        CK(cudaEventRecord(e0));
        launch_coop(k_b2, GRID, smem, mW, mB, cnt, st, ck, nph, c);
        CK(cudaEventRecord(e1));
        cudaError_t er = cudaEventSynchronize(e1); if (er == cudaSuccess) er = cudaGetLastError();
        if (er != cudaSuccess) { fprintf(stderr, "launch error %s\n", cudaGetErrorString(er)); ok = 0; break; }
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        std::vector<ull> h((size_t)GRID * NSTMAX), hc(GRID * 4);
        CK(cudaMemcpy(h.data(), st, h.size() * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hc.data(), ck, hc.size() * 8, cudaMemcpyDeviceToHost));
        if (l < warm) continue;
        Met m = metrics(h, nph, mask);
        rB.push_back(m.busy); rBm.push_back(m.busymean); rW.push_back(m.win); rWm.push_back(m.winmean); rT.push_back(m.T); rTm.push_back(m.Tmean);
        rL.push_back(m.last); rLm.push_back(m.lastmean); rLP.push_back(ms * 1e3 / nph); rM.push_back(mhz_of(hc, GRID));
        if (keep && stored < 1) { stamps_append(outst, h); ++stored; ++st_off; }
      }
      if (!ok) CK(cudaDeviceReset());   // not reached in normal runs; a launch error is fatal for the context
      J j; j.s("bench", "b2_prefetch").s("arm", fl ? "floor" : (gate ? "gate" : "stream")).s("name", nc.name).s("tag", tag).i("mode", nc.mode).i("N", nbusy).i("tot", nc.tot).i("spt", nc.spt)
          .i("nt_max", maxnt).i("S_kb", fl ? 0 : nc.spt * 32).i("R", R).i("P", Pq).d("W_us", ws.first).d("spread_us", ws.second).i("TT", nc.mode ? TT : 0).i("noB", noB).i("hot", hot).i("nph", nph).i("nl", nl)
          .d("us_busy", med(rB)).d("us_busymean", med(rBm)).d("us_win", med(rW)).d("us_winmean", med(rWm)).d("us_T", med(rT)).d("us_pair", med(rTm)).d("us_last", med(rL)).d("us_lastmean", med(rLm))
          .d("us_launch_per_phase", med(rLP)).arr("us_busymean_reps", rBm).d("mhz", med(rM)).i("pass", ok).i("regs", fa.numRegs).i("spill", (int)fa.localSizeBytes).i("smem", smem)
          .s("job", job ? job : "local").i("st_off", off0).i("st_n", stored).i("grid", GRID).i("nstmax", NSTMAX);
      j.print();
    }
  }
  return 0;
}
