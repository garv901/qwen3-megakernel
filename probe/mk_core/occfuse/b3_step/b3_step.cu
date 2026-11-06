// B3 b3_step: compute chain of one tile with operands already in smem. SPT k-steps x 8 MMAs (M128 x N(TT) x K16, cta_group::1, bf16 -> fp32, SW128 K-major descriptors as k_ts),
// mbarrier handoff as k_ts: producer (warp 0) arrives `full` without loading (ring pre-filled once); MMA warp (warp 1) waits full, issues 8 MMAs, commits `empty`
// (per k-step) or arrives `empty` plainly and commits once per tile (commit mode 1); last step commits accfull[b]; epilogue warp (warp 4) waits accfull and arrives accempty[b].
// No phase barrier (a launch = npass x ntile back-to-back tiles); timed region = pass npass-1 only: MMA warp stamps (clock64, globaltimer) before its first wait,
// the epilogue warp stamps after its last accfull wait. Check mode: warps 4-7 read TMEM (tcgen05.ld) and dump fp32 to global; CUDA-core reference compares.
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -I../common b3_step.cu
#include "ub.cuh"
#include "host.h"

constexpr int RBMAX = 16;
__device__ __forceinline__ int swz(int row, int kc) { return row * 128 + ((((kc >> 3) ^ (row & 7))) << 4) + ((kc & 7) << 1); }   // SW128 byte offset of (row, k within a 64-k box)
__device__ __forceinline__ float vA(int s, int m, int k) { return hashf((u32)(s * 40503 + m * 977 + k * 31 + 1)) * 0.5f; }
__device__ __forceinline__ float vB(int s, int n, int k) { return hashf((u32)(s * 70001 + n * 1031 + k * 17 + 5)) * 0.5f; }

template <int DEP, int LEAN>
__global__ void __launch_bounds__(NT, 1) k_b3(int TT, int SPT, int NACC, int ntile, int npass, int ring, int commit_tile, int nact, int check, int fence_tile, ull* ck, float* dump, bf16* Aref, bf16* Bref) {
  if ((int)blockIdx.x >= nact) return;
  extern __shared__ uint8_t smraw[];
  const u32 raw = smem_u32(smraw), base = (raw + 1023) & ~1023u;
  uint8_t* sm = smraw + (base - raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, q = warp & 3;
  const u32 SLOT = A_BYTES + (u32)TT * 256;
  const int RB = commit_tile ? RBMAX : ring;                       // number of full / empty barriers
  const u32 bars = base + ring * SLOT;
  const u32 fullb = bars, emptyb = bars + 128, accf = bars + 256, acce = bars + 272;
  volatile u32* tslot = reinterpret_cast<volatile u32*>(sm + (bars - base) + 512);
  if (warp == 0) tmem_alloc(smem_u32((void*)tslot), TMEM_COLS);
  // fill the ring once (plus plain row-major copies in check mode)
  for (int s = 0; s < ring; ++s)
    for (int i = tid; i < 128 * 128; i += NT) {
      const int m = i >> 7, k = i & 127, a = k >> 6, kc = k & 63;
      const bf16 v = __float2bfloat16(vA(s, m, k));
      *(bf16*)(sm + s * SLOT + a * 16384 + swz(m, kc)) = v;
      if (check) Aref[((size_t)s * 128 + m) * 128 + k] = v;
    }
  for (int s = 0; s < ring; ++s)
    for (int i = tid; i < TT * 128; i += NT) {
      const int n = i >> 7, k = i & 127, a = k >> 6, kc = k & 63;
      const bf16 v = __float2bfloat16(vB(s, n, k));
      *(bf16*)(sm + s * SLOT + A_BYTES + a * (TT * 128) + swz(n, kc)) = v;
      if (check) Bref[((size_t)s * TT + n) * 128 + k] = v;
    }
  fence_proxy_async();
  const int NEPI = check ? 4 : 1;
  if (tid == 0) {
    for (int i = 0; i < RB; ++i) { mbar_init(fullb + 8 * i, 1); mbar_init(emptyb + 8 * i, 1); }
    for (int b = 0; b < 2; ++b) { mbar_init(accf + 8 * b, 1); mbar_init(acce + 8 * b, NEPI); }
    mbar_fence_init();
  }
  __syncthreads();
  tc_fence_after();
  const u32 tb = *tslot, idesc = tc_idesc(128, TT);
  const int ntot = ntile * npass, nstot = ntot * SPT, tstart = (npass - 1) * ntile;
  if (warp == 0) {                                                  // ---- producer: arrive full (no load), back-pressure on empty
    if (lane == 0)
      for (int f = 0; f < nstot; ++f) {
        const int s = f % RB;
        if (f >= RB) mbar_wait(emptyb + 8 * s, ((f / RB) - 1) & 1);
        mbar_arrive(fullb + 8 * s);
      }
  } else if (warp == 1) {                                           // ---- MMA warp
    if (lane == 0) {
      for (int G = 0; G < ntot; ++G) {
        if (G == tstart) { ck[blockIdx.x * 4] = clock64(); ck[blockIdx.x * 4 + 1] = gt(); }
        const int b = NACC == 2 ? (G & 1) : 0;
        if (G >= NACC) mbar_wait(acce + 8 * b, ((G / NACC) - 1) & 1);
        for (int k = 0; k < SPT; ++k) {
          const int f = G * SPT + k, s = f % RB, ps = f % ring;
          mbar_wait(fullb + 8 * s, (f / RB) & 1);
          if (!fence_tile || k == 0) tc_fence_after();   // fence_tile 1: tcgen05.fence::after_thread_sync once per tile (after the first full wait) instead of per k-step
          const u32 sa = base + ps * SLOT, sb = sa + A_BYTES;
          // LEAN 0: descriptors rebuilt from addresses for every MMA (as k_ts / mk). LEAN 1: sdesc once per operand per step, per-MMA offsets added as constants.
          // DEP independent accumulator chains inside the tile (DEP 1: mk's single chain); D column = b * TT * DEP + (kk % DEP) * TT.
          const ull dA = sdesc(sa), dB = sdesc(sb); const u32 bstep = (u32)(TT * 8);   // (TT * 128) >> 4
#pragma unroll
          for (int kk = 0; kk < 8; ++kk) {
            const int a = kk >> 2, off = (kk & 3) * 32;
            const u32 acc = (k > 0 || kk >= DEP) ? 1u : 0u, dcol = tb + b * TT * DEP + (kk % DEP) * TT;
            if (LEAN) tc_mma(dcol, dA + (ull)(a * 1024 + (off >> 4)), dB + (ull)(a * bstep + (off >> 4)), idesc, acc);
            else tc_mma(dcol, sdesc(sa + a * 16384 + off), sdesc(sb + a * (TT * 128) + off), idesc, acc);
          }
          if (commit_tile) { mbar_arrive(emptyb + 8 * s); if (k == SPT - 1) tc_commit(accf + 8 * b); }
          else { tc_commit(emptyb + 8 * s); if (k == SPT - 1) tc_commit(accf + 8 * b); }
        }
      }
    }
  } else if (warp >= 4 && (check ? warp < 8 : warp == 4)) {         // ---- epilogue
    for (int G = 0; G < ntot; ++G) {
      const int b = NACC == 2 ? (G & 1) : 0;
      mbar_wait(accf + 8 * b, (G / NACC) & 1);
      if (G == ntot - 1 && lane == 0 && warp == 4) { ck[blockIdx.x * 4 + 2] = clock64(); ck[blockIdx.x * 4 + 3] = gt(); }
      tc_fence_after();
      if (check && G < 2) {
        for (int c = 0; c < TT / 16; ++c) {
          u32 v[16];
          tmem_ld16(v, tb + ((u32)(q * 32) << 16) + b * TT + c * 16);
          tc_wait_ld();
#pragma unroll
          for (int i = 0; i < 16; ++i) dump[((size_t)G * 128 + q * 32 + lane) * TT + c * 16 + i] = __uint_as_float(v[i]);
        }
        tc_fence_before();
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(acce + 8 * b);
    }
  }
  __syncthreads();
  tc_fence_after();
  if (warp == 0) tmem_dealloc(tb, TMEM_COLS);
}

// CUDA-core reference: tile G in {0,1}; steps G*SPT + k use slot (G * SPT + k) % ring.
__global__ void k_ref(const bf16* Aref, const bf16* Bref, const float* dump, int TT, int SPT, int ring, int ntile, unsigned* nbad, unsigned* maxerr_bits) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)ntile * 128 * TT) return;
  const int n = i % TT, m = (i / TT) % 128, G = (int)(i / ((size_t)TT * 128));
  float acc = 0.f;
  for (int k = 0; k < SPT; ++k) {
    const int s = (G * SPT + k) % ring;
    for (int kk = 0; kk < 128; ++kk) acc += __bfloat162float(Aref[((size_t)s * 128 + m) * 128 + kk]) * __bfloat162float(Bref[((size_t)s * TT + n) * 128 + kk]);
  }
  const float d = fabsf(dump[i] - acc), tol = 2e-3f * fmaxf(1.f, fabsf(acc));
  if (!(d <= tol)) atomicAdd(nbad, 1u);
  atomicMax(maxerr_bits, __float_as_uint(d));
}

static std::vector<int> ilist(const char* s) { std::vector<int> v; for (char* t = strtok(strdup(s), ","); t; t = strtok(nullptr, ",")) v.push_back(atoi(t)); return v; }
static const char* garg(int c, char** v, const char* k, const char* d) { for (int i = 1; i + 1 < c; ++i) if (!strcmp(v[i], k)) return v[i + 1]; return d; }

int main(int argc, char** argv) {
  const char* arm = garg(argc, argv, "--arm", "sweep"); const char* tag = garg(argc, argv, "--tag", ""); const char* job = getenv("SLURM_JOB_ID");
  const int nl = atoi(garg(argc, argv, "--launches", "5")), warm = atoi(garg(argc, argv, "--warm", "2")), nsteps = atoi(garg(argc, argv, "--nsteps", "2048"));
  const int ringcap = atoi(garg(argc, argv, "--ring", "4")), ntile_fix = atoi(garg(argc, argv, "--ntile", "0"));
  std::vector<int> TTs = ilist(garg(argc, argv, "--N", "16,32,48,64,96,128,192,256")), SPTs = ilist(garg(argc, argv, "--SPT", "1,2,4,8,16"));
  const int fence_tile = atoi(garg(argc, argv, "--fence", "0")), lean = atoi(garg(argc, argv, "--lean", "0"));
  std::vector<int> deps = ilist(garg(argc, argv, "--dep", "1")), accs = ilist(garg(argc, argv, "--acc", "1,2")), acts = ilist(garg(argc, argv, "--act", "1,148")), cms = ilist(garg(argc, argv, "--commit", "0,1"));
  int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  if (nsm != GRID) fprintf(stderr, "warning: %d SMs (expected %d)\n", nsm, GRID);
  auto kfn = [&](int dep, int lean) -> void (*)(int, int, int, int, int, int, int, int, int, int, ull*, float*, bf16*, bf16*) {
    if (dep == 1) return lean ? k_b3<1, 1> : k_b3<1, 0>;
    if (dep == 2) return lean ? k_b3<2, 1> : k_b3<2, 0>;
    return lean ? k_b3<4, 1> : k_b3<4, 0>;
  };
  for (int d : {1, 2, 4}) for (int l : {0, 1}) CK(cudaFuncSetAttribute(kfn(d, l), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, kfn(1, 0)));
  ull* ck; CK(cudaMalloc(&ck, GRID * 4 * 8));
  float* dump; bf16 *Aref, *Bref; unsigned* red;
  CK(cudaMalloc(&dump, 2 * 128 * 256 * 4)); CK(cudaMalloc(&Aref, 16 * 128 * 128 * 2)); CK(cudaMalloc(&Bref, 16 * 256 * 128 * 2)); CK(cudaMalloc(&red, 8));
  auto ringfor = [&](int TT) { const int slot = A_BYTES + TT * 256; return std::min(ringcap, (SMEM_MAX - SMEM_FIXED) / slot); };

  auto run_cfg = [&](const char* armname, int TT, int SPT, int NACC, int nact, int cm, int check, int dep, int lean, int fence_tile) {
    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, kfn(dep, lean)));
    const int ring = ringfor(TT), smem = ring * (A_BYTES + TT * 256) + SMEM_FIXED;
    const int ntile = check ? 2 : (ntile_fix ? ntile_fix : std::max(1, nsteps / SPT)), npass = check ? 1 : 2;
    std::vector<double> rc, rcm, ru, rm; int ok = 1; unsigned nbad = 0, mxe = 0;
    for (int l = 0; l < (check ? 1 : warm + nl); ++l) {
      CK(cudaMemset(ck, 0, GRID * 4 * 8)); CK(cudaMemset(dump, 0, 2 * 128 * 256 * 4));
      launch_coop(kfn(dep, lean), GRID, smem, TT, SPT, NACC, ntile, npass, ring, cm, nact, check, fence_tile, ck, dump, Aref, Bref);
      cudaError_t er = cudaDeviceSynchronize(); if (er != cudaSuccess) { fprintf(stderr, "launch error %s\n", cudaGetErrorString(er)); ok = 0; break; }
      if (check) {
        CK(cudaMemset(red, 0, 8));
        k_ref<<<(ntile * 128 * TT + 255) / 256, 256>>>(Aref, Bref, dump, TT, SPT, ring, ntile, red, red + 1); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(&nbad, red, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&mxe, red + 1, 4, cudaMemcpyDeviceToHost));
        if (nbad) ok = 0;
        continue;
      }
      if (l < warm) continue;
      std::vector<ull> h(GRID * 4); CK(cudaMemcpy(h.data(), ck, h.size() * 8, cudaMemcpyDeviceToHost));
      std::vector<double> c, u, m;
      for (int b = 0; b < nact; ++b) {
        const double dc = (double)(h[b * 4 + 2] - h[b * 4]), dn = (double)(h[b * 4 + 3] - h[b * 4 + 1]);
        c.push_back(dc / ntile); u.push_back(dn / 1e3 / ntile); m.push_back(dn > 0 ? dc / dn * 1e3 : 0);
      }
      rc.push_back(med(c)); rcm.push_back(vmax(c)); ru.push_back(med(u)); rm.push_back(med(m));
    }
    J j; j.s("bench", "b3_step").s("arm", armname).s("tag", tag).i("TT", TT).i("SPT", SPT).i("nacc", NACC).i("nact", nact).i("dep", dep).i("lean", lean).i("fence_tile", fence_tile).i("commit_tile", cm).i("ring", ring).i("ntile", ntile)
        .d("cyc_tile", med(rc)).d("cyc_tile_min", vmin(rc)).d("cyc_tile_max", vmax(rc)).d("cyc_tile_ctamax", med(rcm)).arr("cyc_reps", rc)
        .d("us_tile", med(ru)).arr("us_reps", ru).d("mhz", med(rm)).d("mhz_min", vmin(rm)).d("mhz_max", vmax(rm))
        .i("pass", ok).i("nbad", nbad).d("maxerr", *(float*)&mxe).i("regs", fa.numRegs).i("spill", (int)fa.localSizeBytes).i("smem", smem).s("job", job ? job : "local").i("grid", GRID); j.print();
  };

  if (!strcmp(arm, "check")) {
    for (int TT : TTs) for (int SPT : SPTs) for (int na : accs) for (int cm : cms) run_cfg("check", TT, SPT, na, 1, cm, 1, 1, lean, fence_tile);
  } else {
    for (int TT : TTs) for (int SPT : SPTs) for (int na : accs) for (int nact : acts) for (int cm : cms) for (int dp : deps) if (TT * dp * na <= 512) run_cfg("sweep", TT, SPT, na, nact, cm, 0, dp, lean, fence_tile);
  }
  return 0;
}
