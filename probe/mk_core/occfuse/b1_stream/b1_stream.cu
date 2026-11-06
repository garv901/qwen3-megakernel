// B1 b1_stream: weight stream vs active SMs. N active CTAs each stream S bytes of A (Pool, HBM) by TMA into an R-slot ring of 32 KB slots
// (2 boxes of 64 k x 128 rows per slot, as mk). Consumer = one thread: wait full, arrive empty (no math). No prefetch. Inactive CTAs only do the barrier.
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -DNSTMAX=232 -I../common b1_stream.cu
#ifndef NSTMAX
#define NSTMAX 232
#endif
#include "ub.cuh"
#include "host.h"

constexpr int SLOT_B = 32768, KDIM = 1024, KSTEPS = KDIM / 128;   // pool matrix [rows][1024] bf16: 8 k-steps per 128-row block

// place 0: spread (tiles.py floor ranges), place 1: first N blockIdx. R runtime (<= 7: smem). nsteps = S / 32 KB (runtime).
__global__ void __launch_bounds__(NT, 1) k_b1(const __grid_constant__ CUtensorMap mW, unsigned* cnt, ull* st, ull* ck, int nph, int N, int place, int R, int nsteps, long long nrb) {
  extern __shared__ uint8_t smraw[];
  const u32 raw = smem_u32(smraw), base = (raw + 1023) & ~1023u;
  const int tid = threadIdx.x;
  const u32 fullb = base + R * SLOT_B, emptyb = fullb + 8 * MAXRING;
  if (tid == 0) { for (int i = 0; i < R; ++i) { mbar_init(fullb + 8 * i, 1); mbar_init(emptyb + 8 * i, 1); } mbar_fence_init(); }
  __syncthreads();
  const bool active = place ? (blockIdx.x < (unsigned)N) : (((blockIdx.x + 1) * N) / GRID > (blockIdx.x * N) / GRID);
  unsigned target = 0; unsigned g = 0;   // g = steps issued/consumed so far (ring position continues across phases)
  clk_sample(ck, 0);
  bar_arrive(cnt, st, 0); bar_wait(cnt, target, st, 0);
  for (int p = 1; p <= nph; ++p) {
    if (active && tid == 0) {            // producer
      unsigned gg = g;
      for (int s = 0; s < nsteps; ++s, ++gg) {
        const int slot = gg % R; const unsigned j = gg / R;
        if (j > 0) mbar_wait(emptyb + 8 * slot, (j - 1) & 1);
        mbar_expect_tx(fullb + 8 * slot, SLOT_B);
        const long long u = ((long long)p * GRID + blockIdx.x) * nsteps + s;   // unique window per (phase, CTA): 8 consecutive k-steps per row block
        const long long row = ((u / KSTEPS) % nrb) * 128; const int kc = (int)(u % KSTEPS) * 128;
        tma2(base + slot * SLOT_B, &mW, kc, (int)row, fullb + 8 * slot);
        tma2(base + slot * SLOT_B + 16384, &mW, kc + 64, (int)row, fullb + 8 * slot);
      }
    } else if (active && tid == 32) {    // consumer: wait full, free at once
      unsigned gg = g;
      for (int s = 0; s < nsteps; ++s, ++gg) { const int slot = gg % R; mbar_wait(fullb + 8 * slot, (gg / R) & 1); mbar_arrive(emptyb + 8 * slot); }
    }
    if (active) g += nsteps;
    bar_arrive(cnt, st, p); bar_wait(cnt, target, st, p);
  }
  clk_sample(ck, 1);
}

static std::vector<int> ilist(const char* s) { std::vector<int> v; for (char* t = strtok(strdup(s), ","); t; t = strtok(nullptr, ",")) v.push_back(atoi(t)); return v; }
static const char* garg(int c, char** v, const char* k, const char* d) { for (int i = 1; i + 1 < c; ++i) if (!strcmp(v[i], k)) return v[i + 1]; return d; }

struct Met { double T, work, busy, idle, last, spread, lat, Tmean, busymean, lastmean; };
static Met metrics(const std::vector<ull>& st, int nph, const std::vector<char>& mask) {   // one launch, phases 2..nph, us
  std::vector<double> T, W, B, I, L, S, A; double bsum = 0, lsum = 0;
  auto arr = [&](int c, int p) { return (double)st[(size_t)c * NSTMAX + 2 * p]; };
  auto ex = [&](int c, int p) { return (double)st[(size_t)c * NSTMAX + 2 * p + 1]; };
  for (int p = 2; p <= nph; ++p) {
    std::vector<double> e, ep, a, w, wb, wi; double amax = 0, wmax = 0;
    for (int c = 0; c < GRID; ++c) {
      e.push_back(ex(c, p)); ep.push_back(ex(c, p - 1)); a.push_back(arr(c, p)); double wk = arr(c, p) - ex(c, p - 1); w.push_back(wk);
      (mask[c] ? wb : wi).push_back(wk); amax = std::max(amax, arr(c, p)); wmax = std::max(wmax, wk);
    }
    const double me = med(e);
    T.push_back((me - med(ep)) / 1e3); W.push_back(med(w) / 1e3); B.push_back(med(wb) / 1e3); I.push_back(med(wi) / 1e3); L.push_back(wmax / 1e3);
    S.push_back((amax - med(a)) / 1e3); A.push_back((me - amax) / 1e3);
    double s = 0; for (double x : wb) s += x; bsum += s / wb.size() / 1e3; lsum += wmax / 1e3;
  }
  double tm = 0; { std::vector<double> e1, en; for (int c = 0; c < GRID; ++c) { e1.push_back(ex(c, 1)); en.push_back(ex(c, nph)); } tm = (med(en) - med(e1)) / 1e3 / (nph - 1); }
  return {med(T), med(W), med(B), med(I), med(L), med(S), med(A), tm, bsum / (nph - 1), lsum / (nph - 1)};
}

int main(int argc, char** argv) {
  const char* outst = garg(argc, argv, "--out", nullptr);
  const int nph = atoi(garg(argc, argv, "--nph", "100")), nl = atoi(garg(argc, argv, "--launches", "7")), warm = atoi(garg(argc, argv, "--warm", "2"));
  const char* tag = garg(argc, argv, "--tag", ""); const char* job = getenv("SLURM_JOB_ID");
  const std::vector<int> Ns = ilist(garg(argc, argv, "--N", "8,16,32,48,64,96,128,148")), Ss = ilist(garg(argc, argv, "--S", "32,64,96,128,192,256,384,512,768,1024"));
  const std::vector<int> Rs = ilist(garg(argc, argv, "--R", "2,3,4,6,7")), Ps = ilist(garg(argc, argv, "--place", "0,1"));
  const std::vector<int> stS = ilist(garg(argc, argv, "--stS", "128,256,1024"));   // store raw stamps only for these S (KB)
  if (nph + 1 > NSTMAX / 2 - 1) { fprintf(stderr, "nph too large for NSTMAX\n"); return 1; }
  int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  if (nsm != GRID) fprintf(stderr, "warning: %d SMs (expected %d)\n", nsm, GRID);
  Pool P = pool_alloc(); const long long rows = (long long)(P.bytes / (2ull * KDIM)), nrb = rows / 128 - 1;
  CUtensorMap mW = tmap_bf16(P.p, KDIM, rows, 64, 128);
  CK(cudaFuncSetAttribute(k_b1, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, k_b1));
  unsigned* cnt; ull *st, *ck; CK(cudaMalloc(&cnt, 1024)); CK(cudaMalloc(&st, (size_t)GRID * NSTMAX * 8)); CK(cudaMalloc(&ck, GRID * 4 * 8 * 64));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  long long st_off = 0;

  for (int place : Ps) for (int N : Ns) for (int R : Rs) for (int SK : Ss) {
    const int nsteps = SK / 32, smem = R * SLOT_B + SMEM_FIXED;
    if (smem > SMEM_MAX) { fprintf(stderr, "skip R=%d: smem %d > max\n", R, smem); continue; }
    std::vector<char> mask(GRID, 0);
    for (int b = 0; b < GRID; ++b) mask[b] = place ? b < N : ((b + 1) * N) / GRID > (b * N) / GRID;
    std::vector<double> rT, rW, rB, rI, rL, rS, rA, rLP, rM, rTm, rBm, rLm;
    int stored = 0; const long long off0 = st_off; int ok = 1; const bool keep = std::find(stS.begin(), stS.end(), SK) != stS.end();
    for (int l = 0; l < warm + nl; ++l) {
      CK(cudaMemset(cnt, 0, 1024)); CK(cudaMemset(st, 0, (size_t)GRID * NSTMAX * 8)); CK(cudaMemset(ck, 0, GRID * 4 * 8));
      CK(cudaEventRecord(e0));
      launch_coop(k_b1, GRID, smem, mW, cnt, st, ck, nph, N, place, R, nsteps, nrb);
      CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      cudaError_t er = cudaGetLastError(); if (er != cudaSuccess) { fprintf(stderr, "launch error %s\n", cudaGetErrorString(er)); ok = 0; break; }
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
      std::vector<ull> h((size_t)GRID * NSTMAX), hc(GRID * 4);
      CK(cudaMemcpy(h.data(), st, h.size() * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hc.data(), ck, hc.size() * 8, cudaMemcpyDeviceToHost));
      if (l < warm) continue;
      Met m = metrics(h, nph, mask);
      rT.push_back(m.T); rW.push_back(m.work); rB.push_back(m.busy); rI.push_back(m.idle); rL.push_back(m.last); rS.push_back(m.spread); rA.push_back(m.lat);
      rTm.push_back(m.Tmean); rBm.push_back(m.busymean); rLm.push_back(m.lastmean);
      rLP.push_back(ms * 1e3 / nph); rM.push_back(mhz_of(hc, GRID));
      if (keep && stored < 1) { stamps_append(outst, h); ++stored; ++st_off; }
    }
    J j; j.s("bench", "b1_stream").s("arm", "stream").s("tag", tag).i("N", N).i("place", place).i("S_kb", SK).i("R", R).i("nsteps", nsteps).i("pf", 0)
        .i("nph", nph).i("nl", nl).d("us_T", med(rT)).d("us_Tmean", med(rTm)).d("us_work", med(rW)).d("us_busy", med(rB)).d("us_busymean", med(rBm)).d("us_idle", med(rI))
        .d("us_last", med(rL)).d("us_lastmean", med(rLm)).d("us_spread", med(rS)).d("us_lat", med(rA)).d("us_launch_per_phase", med(rLP)).arr("us_reps", rT).arr("us_Tmean_reps", rTm)
        .arr("us_busymean_reps", rBm).d("mhz", med(rM)).i("pass", ok).i("regs", fa.numRegs).i("spill", (int)fa.localSizeBytes).i("smem", smem)
        .s("job", job ? job : "local").i("st_off", off0).i("st_n", stored).i("grid", GRID).i("nstmax", NSTMAX);
    j.print();
  }
  return 0;
}
