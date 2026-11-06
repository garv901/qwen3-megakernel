// B7 b7_splitk: price of splitting K for a whole phase. TS engine of xover4 (k_ts, CG = 1) with the occfuse harness:
//   mk split barrier (bar_arrive -> thread 0 window prefetch of the next phase's first ring slots' A -> bar_wait) and per-CTA stamps.
// Arm r (redadd): KS == 1 plain fp32 store, KS > 1 red.add fp32 into Y [T][N].
// Arm w (ws):     partial tiles stored plain to WS [KS][T][N] fp32; barrier; one reduce pass by all CTAs (sum over KS -> Y [T][N], float4); barrier.
//                 Two barriers per phase (stamps barrier 2p+1 = after GEMM, 2p+2 = after reduce). The window prefetch is issued at the end barrier only.
// Tile (rb, tt, ks), t = (rb*NTT + tt)*KS + ks; CTA b owns [b*NTILE/148, (b+1)*NTILE/148) (mk ts_geom). Per k-step one ring slot: A (2 SW128 boxes 64 k x 128 rows) + B (TT rows x 128 k, 2 boxes), full count 2.
// A of the first min(ring, nst) steps of phase p+1 is issued in the barrier window (pf 1) or after the barrier together with B (pf 0). B is always issued after the barrier (fence.proxy.async.global first).
// Shapes, T, KS, TT come from the command line. orc bits (as xover4): 1 no MMA, 2 no epilogue stores, 4 no B loads, 8 no A loads, 16 no TMEM reads.
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -I../common b7_splitk.cu
#include "ub.cuh"
#include "host.h"

__device__ __forceinline__ void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }

struct P7 { int N, K, T, TT, KS, nph, phase0, ring, arm, pf, orc; long long rowsAvail; };

__global__ void __launch_bounds__(NT, 1) k_b7(const __grid_constant__ CUtensorMap mW, const __grid_constant__ CUtensorMap mX, float* Y, float* WS, unsigned* cnt, ull* st, ull* ck, P7 q7) {
  extern __shared__ uint8_t smraw[];
  const u32 raw = smem_u32(smraw), base = (raw + 1023) & ~1023u;
  uint8_t* sm = smraw + (base - raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, q = warp & 3;
  const int N = q7.N, K = q7.K, T = q7.T, TT = q7.TT, KS = q7.KS, nph = q7.nph, ring = q7.ring, orc = q7.orc, arm = q7.arm, pf = q7.pf;
  const u32 BB = (u32)TT * 256, SLOT = A_BYTES + BB;
  const u32 bars = base + ring * SLOT;
  const u32 fullb = bars, emptyb = bars + 128, accf = bars + 256, acce = bars + 272;
  volatile u32* tslot = reinterpret_cast<volatile u32*>(sm + (bars - base) + 512);
  const int NCH = TT / 16, NEPI = NCH < 3 ? NCH : 3;
  clk_sample(ck, 0);
  if (warp == 0) tmem_alloc(smem_u32((void*)tslot), TMEM_COLS);
  if (tid == 0) {
    for (int i = 0; i < ring; ++i) { mbar_init(fullb + 8 * i, 2); mbar_init(emptyb + 8 * i, 1); }
    for (int b = 0; b < 2; ++b) { mbar_init(accf + 8 * b, 1); mbar_init(acce + 8 * b, 4 * NEPI); }
    mbar_fence_init();
  }
  __syncthreads();
  tc_fence_after();
  const u32 tb = *tslot;
  const int NRB = N / 128, NTT = (T + TT - 1) / TT, NTILE = NRB * NTT * KS, SPT = K / (KS * 128), KSL = K / KS;
  const int t0 = (int)((long long)blockIdx.x * NTILE / GRID), t1 = (int)((long long)(blockIdx.x + 1) * NTILE / GRID);
  const int nt = t1 - t0, nst = nt * SPT;
  const u32 idesc = tc_idesc(128, TT);
  auto prow = [&](int p) { return (int)(((long long)(q7.phase0 + p) * N) % q7.rowsAvail); };
  auto fill_a = [&](int p, int j, u32 f) {
    const int t = t0 + j / SPT, kc = (t % KS) * KSL + (j % SPT) * 128, rb = (t / KS) / NTT, s = f % ring;
    const u32 full = fullb + 8 * s;
    if (orc & 8) { mbar_arrive(full); return; }
    mbar_expect_tx(full, A_BYTES);
    for (int a = 0; a < 2; ++a) tma2(base + s * SLOT + a * 16384, &mW, kc + a * 64, prow(p) + rb * 128, full);
  };
  auto fill_b = [&](int j, u32 f) {
    const int t = t0 + j / SPT, kc = (t % KS) * KSL + (j % SPT) * 128, tt = (t / KS) % NTT, s = f % ring;
    const u32 full = fullb + 8 * s;
    if (orc & 4) { mbar_arrive(full); return; }
    mbar_expect_tx(full, BB);
    for (int a = 0; a < 2; ++a) tma2(base + s * SLOT + A_BYTES + a * (TT * 128), &mX, kc + a * 64, tt * TT, full);
  };
  auto epi_tile = [&](int ti, u32 G0) {
    const u32 G = G0 + ti, b = G & 1;
    const int t = t0 + ti, rb = (t / KS) / NTT, tt = (t / KS) % NTT, ks = t % KS;
    mbar_wait(accf + 8 * b, (G >> 1) & 1);
    tc_fence_after();
    float* out = arm ? WS + (size_t)ks * T * N : Y;
    float* yp = out + (size_t)(tt * TT) * N + rb * 128 + 32 * q + lane;
    const bool plain = arm || KS == 1;
    const u32 ta = tb + ((u32)(q * 32) << 16) + b * TT;
    for (int c = (warp >> 2) - 1; c < NCH; c += 6) {
      if (orc & 16) break;
      const int cb = tt * TT + c * 16;
      if (cb >= T) break;
      const bool two = c + 3 < NCH && cb + 48 < T;
      const int nv0 = min(16, T - cb), nv1 = min(16, T - cb - 48);
      u32 v0[16], v1[16];
      tmem_ld16(v0, ta + c * 16);
      if (two) tmem_ld16(v1, ta + (c + 3) * 16);
      tc_wait_ld();
      float* y0 = yp + (size_t)(c * 16) * N;
      if (orc & 2) { if (__uint_as_float(v0[0]) == 1234.5f) y0[0] = 0.f; continue; }
      if (plain) {
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
    if (lane == 0) mbar_arrive(acce + 8 * b);
  };
  unsigned target = 0; int bi = 0;   // bi = barrier index for stamps
  if (pf && tid == 0) for (int i = 0; i < min(ring, nst); ++i) fill_a(0, i, i);   // phase 0 window prefetch (before the start barrier)
  bar_arrive(cnt, st, bi); bar_wait(cnt, target, st, bi); ++bi;
  for (int p = 0; p < nph; ++p) {
    const u32 F0 = (u32)p * nst, G0 = (u32)p * nt;
    if (tid == 0) {                                                  // ---- producer
      fence_proxy_async_global();
      for (int i = 0; i < min(ring, nst); ++i) { if (!pf) fill_a(p, i, F0 + i); fill_b(i, F0 + i); }
      for (int j = ring; j < nst; ++j) {
        const u32 f = F0 + j, s = f % ring;
        mbar_wait(emptyb + 8 * s, ((f / ring) - 1) & 1);
        fill_a(p, j, f); fill_b(j, f);
      }
    } else if (tid == 32) {                                          // ---- MMA issuer
      for (int ti = 0; ti < nt; ++ti) {
        const u32 G = G0 + ti, b = G & 1;
        if (G >= 2) mbar_wait(acce + 8 * b, ((G >> 1) - 1) & 1);
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
            tc_mma(tb + b * TT, sdesc(sa + a * 16384 + off), sdesc(sb + a * (TT * 128) + off), idesc, acc);
          }
          tc_commit(emptyb + 8 * s);
          if (k == SPT - 1) tc_commit(accf + 8 * b);
        }
      }
    } else if (warp >= 4 && (warp >> 2) - 1 < NCH) {                 // ---- epilogue warps 4-15
      for (int ti = 0; ti < nt; ++ti) epi_tile(ti, G0);
    }
    __threadfence();
    if (arm) {                                                       // workspace arm: barrier, then the reduce pass by all CTAs
      bar_arrive(cnt, st, bi); bar_wait(cnt, target, st, bi); ++bi;
      const size_t tot4 = (size_t)T * N / 4, ws4 = (size_t)T * N;
      for (size_t i = (size_t)blockIdx.x * NT + tid; i < tot4; i += (size_t)GRID * NT) {
        float4 s = __ldcg(reinterpret_cast<const float4*>(WS) + i);
        for (int k = 1; k < KS; ++k) { const float4 v = __ldcg(reinterpret_cast<const float4*>(WS + k * ws4) + i); s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w; }
        reinterpret_cast<float4*>(Y)[i] = s;
      }
      __threadfence();
    }
    bar_arrive(cnt, st, bi);
    if (pf && tid == 0 && p + 1 < nph) {                             // barrier window: A of the next phase's first steps
      const u32 F1 = F0 + nst;
      for (int i = 0; i < min(ring, nst); ++i) {
        const u32 f = F1 + i, s = f % ring;
        if (f >= (u32)ring) mbar_wait(emptyb + 8 * s, ((f / ring) - 1) & 1);
        fill_a(p + 1, i, f);
      }
    }
    bar_wait(cnt, target, st, bi); ++bi;
  }
  __syncthreads();
  tc_fence_after();
  if (warp == 0) tmem_dealloc(tb, TMEM_COLS);
  clk_sample(ck, 1);
}

// CUDA-core reference for phase 0 (pool rows 0..N-1): |d| <= 2e-3 * max(1, |ref|).
__global__ void k_ref(const bf16* pool, const bf16* X, const float* Y, int N, int K, int T, unsigned* maxerr, unsigned* nbad) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)T * N) return;
  const int t = (int)(i / N), n = (int)(i % N);
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += __bfloat162float(X[(size_t)t * K + k]) * __bfloat162float(pool[(size_t)n * K + k]);
  const float d = fabsf(Y[(size_t)t * N + n] - acc);
  atomicMax(maxerr, __float_as_uint(d));
  if (!(d <= 2e-3f * fmaxf(1.f, fabsf(acc)))) atomicAdd(nbad, 1u);
}

static std::vector<double> dlist(const char* s) { std::vector<double> v; char* d = strdup(s); for (char* t = strtok(d, ","); t; t = strtok(nullptr, ",")) v.push_back(atof(t)); return v; }
static const char* garg(int c, char** v, const char* k, const char* d) { for (int i = 1; i + 1 < c; ++i) if (!strcmp(v[i], k)) return v[i + 1]; return d; }

struct Met { double T, Tmean, work, work2, busy, idle, last, spread, lat; };
// sb = barriers per phase (1 redadd, 2 ws). Phase p (1-based) ends at barrier sb*p; its GEMM part ends at barrier sb*p - sb + 1. Skips phase 1.
static Met metrics(const std::vector<ull>& st, int nph, int sb, const std::vector<char>& mask) {
  auto arr = [&](int c, int b) { return (double)st[(size_t)c * NSTMAX + 2 * b]; };
  auto ex = [&](int c, int b) { return (double)st[(size_t)c * NSTMAX + 2 * b + 1]; };
  std::vector<double> T, W, W2, B, I, L, S, A; std::vector<double> e(GRID), ep(GRID), a(GRID);
  for (int p = 2; p <= nph; ++p) {
    const int eb = sb * p, gb = eb - sb + 1, pb = sb * (p - 1);
    std::vector<double> wb, wi, wall; double amax = 0, am2 = 0, wmax = 0;
    for (int c = 0; c < GRID; ++c) {
      const double wk = arr(c, gb) - ex(c, pb); wall.push_back(wk); wmax = std::max(wmax, wk); a[c] = arr(c, gb); amax = std::max(amax, a[c]);
      (mask[c] ? wb : wi).push_back(wk); e[c] = ex(c, eb); ep[c] = ex(c, pb); am2 = std::max(am2, arr(c, eb));
    }
    T.push_back((med(e) - med(ep)) / 1e3); W.push_back(med(wall) / 1e3); B.push_back(wb.empty() ? -1 : med(wb) / 1e3); I.push_back(wi.empty() ? -1 : med(wi) / 1e3);
    L.push_back(wmax / 1e3); S.push_back((amax - med(a)) / 1e3); A.push_back((med(e) - am2) / 1e3);
    if (sb == 2) { std::vector<double> w2; for (int c = 0; c < GRID; ++c) w2.push_back(arr(c, eb) - ex(c, eb - 1)); W2.push_back(med(w2) / 1e3); } else W2.push_back(0);
  }
  std::vector<double> e1, en; for (int c = 0; c < GRID; ++c) { e1.push_back(ex(c, sb)); en.push_back(ex(c, sb * nph)); }
  return {med(T), (med(en) - med(e1)) / 1e3 / (nph - 1), med(W), med(W2), med(B), med(I), med(L), med(S), med(A)};
}

int main(int argc, char** argv) {
  const char* outst = garg(argc, argv, "--out", nullptr);
  const int nph = atoi(garg(argc, argv, "--nph", "112")), nl = atoi(garg(argc, argv, "--launches", "7")), warm = atoi(garg(argc, argv, "--warm", "2"));
  const char* tag = garg(argc, argv, "--tag", ""); const char* job = getenv("SLURM_JOB_ID");
  const int orc = atoi(garg(argc, argv, "--orc", "0")), ringcap = atoi(garg(argc, argv, "--ring", "0")), TTforce = atoi(garg(argc, argv, "--TT", "0"));
  const std::vector<double> Ts = dlist(garg(argc, argv, "--T", "8,32,128,256")), KSs = dlist(garg(argc, argv, "--KS", "1,2,3,4,6,8,12,24")), pfs = dlist(garg(argc, argv, "--pf", "1")), arms = dlist(garg(argc, argv, "--arm", "0,1"));
  const char* shsel = garg(argc, argv, "--shapes", "QKV:4096:1024,O:1024:2048,GU:6144:1024,D:1024:3072");
  const char* keepst = garg(argc, argv, "--keepst", "");   // shape names whose last timed launch's raw stamps go to --out
  if (2 * (2 * nph + 1) > NSTMAX) { fprintf(stderr, "nph too large for NSTMAX\n"); return 1; }
  int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  if (nsm != GRID) fprintf(stderr, "warning: %d SMs (expected %d)\n", nsm, GRID);
  Pool P = pool_alloc();
  bf16* X; float *Y, *WS; unsigned *cnt, *ver; ull *st, *ck;
  CK(cudaMalloc(&X, 256ull * 3072 * 2)); CK(cudaMalloc(&Y, 256ull * 6144 * 4)); CK(cudaMalloc(&WS, 24ull * 256 * 6144 * 4)); CK(cudaMalloc(&cnt, 1024)); CK(cudaMalloc(&ver, 64));
  CK(cudaMalloc(&st, (size_t)GRID * NSTMAX * 8)); CK(cudaMalloc(&ck, GRID * 4 * 8 * 2));
  CK(cudaFuncSetAttribute(k_b7, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, k_b7));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  long long st_off = 0;
  struct Sh { std::string name; int N, K; }; std::vector<Sh> shapes;
  { char* d = strdup(shsel); for (char* t = strtok(d, ","); t; t = strtok(nullptr, ",")) { char nm[32]; int n, k; sscanf(t, "%31[^:]:%d:%d", nm, &n, &k); shapes.push_back({nm, n, k}); } }
  for (auto& sh : shapes) {
    const int N = sh.N, K = sh.K;
    k_init_bf16<<<256, 256>>>(X, 256ull * K, 777u); CK(cudaDeviceSynchronize());
    const long long rows = (long long)(P.bytes / (2ull * K)), rowsAvail = rows - N;
    CUtensorMap mW = tmap_bf16(P.p, K, rows, 64, 128);
    for (double Td : Ts) {
      const int T = (int)Td, TT = TTforce ? TTforce : std::min(128, (T + 15) / 16 * 16);
      CUtensorMap mX = tmap_bf16(X, K, std::max(T, 16), 64, TT);   // T < 16: rows T..15 are real X (outputs not stored); T % TT != 0: OOB rows zero fill
      const int slotB = A_BYTES + TT * 256;
      const int ring = std::min(ringcap ? ringcap : MAXRING, std::min(MAXRING, (SMEM_MAX - SMEM_FIXED) / slotB)), smem = SMEM_FIXED + ring * slotB;
      for (double ksd : KSs) for (double armd : arms) for (double pfd : pfs) {
        const int KS = (int)ksd, arm = (int)armd, pf = (int)pfd;
        if ((K / 128) % KS) continue;
        if (arm && KS == 1) continue;
        const int NTT = (T + TT - 1) / TT, NTILE = N / 128 * NTT * KS, SPT = K / (KS * 128), sb = arm ? 2 : 1;
        std::vector<char> mask(GRID, 0); int nbusy = 0, tmax = 0;
        for (int b = 0; b < GRID; ++b) { const int n = (int)((long long)(b + 1) * NTILE / GRID - (long long)b * NTILE / GRID); mask[b] = n > 0; nbusy += n > 0; tmax = std::max(tmax, n); }
        P7 pp = {N, K, T, TT, KS, nph, 0, ring, arm, pf, orc, rowsAvail};
        auto launch = [&](int np, int p0) {
          P7 c = pp; c.nph = np; c.phase0 = p0;
          CK(cudaMemset(cnt, 0, 1024)); CK(cudaMemset(st, 0, (size_t)GRID * NSTMAX * 8)); CK(cudaMemset(ck, 0, GRID * 4 * 8));
          launch_coop(k_b7, GRID, smem, mW, mX, Y, WS, cnt, st, ck, c);
        };
        int pass = -1; double maxerr = 0;
        if (!orc) {
          CK(cudaMemset(Y, (arm || KS == 1) ? 0xFF : 0, (size_t)T * N * 4)); CK(cudaMemset(ver, 0, 64));
          launch(1, 0); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
          k_ref<<<((size_t)T * N + 255) / 256, 256>>>(P.p, X, Y, N, K, T, ver, ver + 1); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
          unsigned hv[2]; CK(cudaMemcpy(hv, ver, 8, cudaMemcpyDeviceToHost)); float me; memcpy(&me, &hv[0], 4); maxerr = me; pass = hv[1] == 0;
          if (!pass) fprintf(stderr, "FAIL %s T=%d KS=%d arm=%d bad=%u max|d|=%.3e\n", sh.name.c_str(), T, KS, arm, hv[1], me);
        }
        std::vector<double> rT, rTm, rW, rW2, rB, rI, rL, rS, rA, rLP, rM; int p0 = 1000, stored = 0; const long long off0 = st_off;
        for (int l = 0; l < warm + nl; ++l) {
          CK(cudaMemset(Y, 0, (size_t)T * N * 4));
          CK(cudaEventRecord(e0)); launch(nph, p0); CK(cudaEventRecord(e1));
          cudaError_t er = cudaEventSynchronize(e1); if (er == cudaSuccess) er = cudaGetLastError();
          if (er != cudaSuccess) { fprintf(stderr, "launch error %s (%s T=%d KS=%d arm=%d)\n", cudaGetErrorString(er), sh.name.c_str(), T, KS, arm); J j; j.s("bench", "b7_splitk").s("arm", arm ? "ws" : "redadd").s("shape", sh.name).i("T", T).i("KS", KS).i("pf", pf).s("error", cudaGetErrorString(er)); j.print(); return 2; }
          p0 += nph;
          if (l < warm) continue;
          float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
          std::vector<ull> h((size_t)GRID * NSTMAX), hc(GRID * 4);
          CK(cudaMemcpy(h.data(), st, h.size() * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hc.data(), ck, hc.size() * 8, cudaMemcpyDeviceToHost));
          Met m = metrics(h, nph, sb, mask);
          rT.push_back(m.T); rTm.push_back(m.Tmean); rW.push_back(m.work); rW2.push_back(m.work2); rB.push_back(m.busy); rI.push_back(m.idle); rL.push_back(m.last); rS.push_back(m.spread); rA.push_back(m.lat);
          rLP.push_back(ms * 1e3 / nph); rM.push_back(mhz_of(hc, GRID));
          if (l == warm + nl - 1 && outst && strstr(keepst, sh.name.c_str())) { stamps_append(outst, h); ++stored; ++st_off; }
        }
        J j; j.s("bench", "b7_splitk").s("arm", arm ? "ws" : "redadd").s("shape", sh.name).s("tag", tag).i("N", N).i("K", K).i("T", T).i("TT", TT).i("KS", KS).i("pf", pf).i("orc", orc).i("nph", nph).i("nl", nl)
            .i("ring", ring).i("tiles", NTILE).i("busy_ctas", nbusy).i("tiles_max", tmax).i("spt", SPT).i("max_steps", tmax * SPT).i("sb", sb)
            .d("us_T", med(rT)).d("us_Tmean", med(rTm)).d("us_work", med(rW)).d("us_work2", med(rW2)).d("us_busy", med(rB)).d("us_idle", med(rI)).d("us_last", med(rL)).d("us_spread", med(rS)).d("us_lat", med(rA))
            .d("us_launch_per_phase", med(rLP)).arr("us_reps", rTm).d("us_T_min", vmin(rTm)).d("us_T_max", vmax(rTm)).d("mhz", med(rM)).i("pass", pass).d("maxerr", maxerr)
            .i("regs", fa.numRegs).i("spill", (int)fa.localSizeBytes).i("smem", smem).s("job", job ? job : "local").i("st_off", off0).i("st_n", stored).i("grid", GRID).i("nstmax", NSTMAX);
        j.print();
      }
    }
  }
  return 0;
}
