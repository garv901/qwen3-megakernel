// clc: finish-time spread of a 1 GiB streaming read under 8 work-assignment arms (static split, atomic queue,
// prefetched queue, guided queue, hw block scheduler, CLC try_cancel with/without prefetch, static interleaved). 1 CTA/SM in every arm.
// static_il: CTA b streams chunks b, b + grid, ...; nchunks = grid * round(N / C / grid) so every CTA gets equal bytes (no atomics).
// JSON lines (one per arm x C x rep) on stdout; PASS/FAIL per arm x C on stderr.
// Usage: ./clc [--validate] [--arms a,b] [--chunks bytes,bytes] [--reps 10] [--warmup 2]
// Build: -DUNROLL=8 (loads in flight per thread). CLC is raw PTX (same sequence as cuda::for_each_canceled_block).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#ifndef UNROLL
#define UNROLL 8
#endif
#define THREADS 512
#define SMEM_DYN (160 * 1024)
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

enum { A_STATIC, A_QUEUE, A_QUEUE_PF, A_GUIDED, A_HW, A_CLC, A_CLC_NP, A_STATIC_IL, NARMS };
static const char* ARM_NAME[NARMS] = {"static", "queue", "queue_pf", "guided", "hw", "clc", "clc_np", "static_il"};

struct Rec { uint32_t smid, cta, chunk, pad; unsigned long long t0, t1; };
static const unsigned long long N_U4 = 1ull << 26;  // 1 GiB of uint4

__host__ __device__ inline uint32_t pat(uint64_t w) { uint32_t x = (uint32_t)w * 2654435761u + (uint32_t)(w >> 32); x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; return x; }

__global__ void k_fill(uint4* b, uint64_t n) {
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x)
    b[i] = make_uint4(pat(4 * i), pat(4 * i + 1), pat(4 * i + 2), pat(4 * i + 3));
}
__global__ void k_flush(uint4* b, uint64_t n, uint32_t v) {
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) b[i] = make_uint4(v, v, v, v);
}

__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint32_t smid() { uint32_t s; asm("mov.u32 %0, %smid;" : "=r"(s)); return s; }
__device__ __forceinline__ uint4 ldnc(const uint4* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
// stream uint4 range [off, off+len) with the CTA; returns the per-thread XOR
__device__ __forceinline__ uint32_t stream(const uint4* buf, uint32_t off, uint32_t len) {
  uint32_t x = 0, end = off + len;
  for (uint32_t base = off; base < end; base += THREADS * UNROLL) {
    uint4 v[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; u++) { uint32_t i = base + u * THREADS + threadIdx.x; v[u] = (i < end) ? ldnc(buf + i) : make_uint4(0, 0, 0, 0); }
#pragma unroll
    for (int u = 0; u < UNROLL; u++) x ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  return x;
}
// one chunk with timing record (thread 0); the leading __syncthreads is the CTA start fence
__device__ __forceinline__ uint32_t do_chunk(const uint4* buf, const uint2* list, Rec* rec, uint32_t* visited, int c) {
  __syncthreads();
  unsigned long long t0 = 0;
  if (threadIdx.x == 0) t0 = gt();
  uint2 ch = list[c];
  uint32_t x = stream(buf, ch.x, ch.y);
  __syncthreads();
  if (threadIdx.x == 0) {
    Rec r; r.smid = smid(); r.cta = blockIdx.x; r.chunk = c; r.pad = 0; r.t0 = t0; r.t1 = gt();
    rec[c] = r; atomicAdd(&visited[c], 1u);
  }
  return x;
}
__device__ __forceinline__ void cta_sink(uint32_t x, uint32_t* csum) {
  __shared__ uint32_t red[THREADS / 32];
  for (int o = 16; o; o >>= 1) x ^= __shfl_xor_sync(0xffffffffu, x, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = x;
  __syncthreads();
  if (threadIdx.x == 0) { uint32_t y = 0; for (int i = 0; i < THREADS / 32; i++) y ^= red[i]; atomicXor(csum, y); }
}

__device__ __forceinline__ void clc_issue(uint32_t resp, uint32_t bar) {
  asm volatile("fence.proxy.async.shared::cta;\n\t"
               "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];\n\t"
               "mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%1], 16;" :: "r"(resp), "r"(bar) : "memory");
}
__device__ __forceinline__ void clc_wait(uint32_t bar, uint32_t ph) {
  asm volatile("{\n\t.reg .pred p;\n\tW%=:\n\t"
               "mbarrier.try_wait.parity.relaxed.cta.shared::cta.b64 p, [%0], %1;\n\t@!p bra W%=;\n\t}" :: "r"(bar), "r"(ph) : "memory");
}
// returns canceled-ctaid.x, or -1 if the cancel failed
__device__ __forceinline__ int clc_query(const ulonglong2* resp) {
  unsigned long long r0, r1;
  asm volatile("ld.volatile.shared.v2.u64 {%0,%1}, [%2];" : "=l"(r0), "=l"(r1) : "r"((uint32_t)__cvta_generic_to_shared(resp)) : "memory");
  int ok, x;
  asm volatile("{\n\t.reg .b128 q;\n\t.reg .pred p;\n\t"
               "mov.b128 q, {%2, %3};\n\t"
               "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, q;\n\t"
               "selp.b32 %0, 1, 0, p;\n\t"
               "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %1, q;\n\t}"
               : "=r"(ok), "=r"(x) : "l"(r0), "l"(r1) : "memory");
  return ok ? x : -1;
}

template <int ARM>
__global__ void __launch_bounds__(THREADS, 1)
k_stream(const uint4* buf, const uint2* list, int nchunks, uint32_t* ctr, Rec* rec, uint32_t* visited, uint32_t* csum) {
  extern __shared__ __align__(16) uint8_t dyn[];  // 160 KB held only to force 1 CTA/SM
  __shared__ int s_c;
  __shared__ __align__(16) ulonglong2 s_resp;
  __shared__ __align__(8) uint64_t s_bar;
  uint32_t x = 0;
  if constexpr (ARM == A_STATIC) {
    x = do_chunk(buf, list, rec, visited, blockIdx.x);
  } else if constexpr (ARM == A_HW) {
    x = do_chunk(buf, list, rec, visited, blockIdx.x);
  } else if constexpr (ARM == A_STATIC_IL) {
    for (int c = blockIdx.x; c < nchunks; c += gridDim.x) x ^= do_chunk(buf, list, rec, visited, c);
  } else if constexpr (ARM == A_QUEUE || ARM == A_GUIDED) {  // guided = queue over a shrinking list; pf below
    for (;;) {
      if (threadIdx.x == 0) s_c = (int)atomicAdd(ctr, 1u);
      __syncthreads();
      int c = s_c;
      __syncthreads();
      if (c >= nchunks) break;
      x ^= do_chunk(buf, list, rec, visited, c);
    }
  } else if constexpr (ARM == A_QUEUE_PF) {
    int nxt = 0;
    if (threadIdx.x == 0) nxt = (int)atomicAdd(ctr, 1u);
    for (;;) {
      if (threadIdx.x == 0) s_c = nxt;
      __syncthreads();
      int c = s_c;
      if (c >= nchunks) break;
      if (threadIdx.x == 0) nxt = (int)atomicAdd(ctr, 1u);  // issued before streaming chunk c
      x ^= do_chunk(buf, list, rec, visited, c);
      __syncthreads();
    }
  } else {  // CLC; PF = try_cancel for the next block is issued before streaming the current one
    constexpr bool PF = (ARM == A_CLC);
    uint32_t resp = (uint32_t)__cvta_generic_to_shared(&s_resp), bar = (uint32_t)__cvta_generic_to_shared(&s_bar);
    uint32_t ph = 0;
    int c = blockIdx.x;
    if (threadIdx.x == 0) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar) : "memory");
      if (PF) asm volatile("clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];\n\t"
                           "mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%1], 16;" :: "r"(resp), "r"(bar) : "memory");
    }
    for (;;) {
      x ^= do_chunk(buf, list, rec, visited, c);
      if (threadIdx.x == 0) {
        if (!PF) clc_issue(resp, bar);
        clc_wait(bar, ph);
      }
      ph ^= 1;
      __syncthreads();
      int nx = clc_query(&s_resp);
      __syncthreads();  // all reads of the response done before it is reused
      if (nx < 0) break;
      c = nx;
      if (PF && threadIdx.x == 0) clc_issue(resp, bar);
    }
  }
  cta_sink(x, csum);
}

// ---------------- host ----------------
struct Arm { int id; };
static double med(std::vector<double> v) { if (v.empty()) return -1; std::sort(v.begin(), v.end()); size_t n = v.size(); return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]); }
static double pct(std::vector<double> v, double p) { if (v.empty()) return -1; std::sort(v.begin(), v.end()); return v[(size_t)(p * (v.size() - 1) + 0.5)]; }

static std::vector<std::string> split(const std::string& s) { std::vector<std::string> o; size_t i = 0; while (i <= s.size()) { size_t j = s.find(',', i); if (j == std::string::npos) j = s.size(); if (j > i) o.push_back(s.substr(i, j - i)); i = j + 1; } return o; }

template <int ARM> static void* kptr() { return (void*)k_stream<ARM>; }
static void* kernel_of(int a) {
  switch (a) { case A_STATIC: return kptr<A_STATIC>(); case A_QUEUE: return kptr<A_QUEUE>(); case A_QUEUE_PF: return kptr<A_QUEUE_PF>();
    case A_GUIDED: return kptr<A_QUEUE_PF>(); case A_HW: return kptr<A_HW>(); case A_CLC: return kptr<A_CLC>(); case A_STATIC_IL: return kptr<A_STATIC_IL>(); default: return kptr<A_CLC_NP>(); }
}

int main(int argc, char** argv) {
  bool validate_only = false; int reps = 10, warm = 2;
  std::string arms_s = "static,queue,queue_pf,guided,hw,clc,clc_np,static_il", chunks_s = "65536,262144,1048576,4194304";
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--validate")) validate_only = true;
    else if (!strcmp(argv[i], "--arms") && i + 1 < argc) arms_s = argv[++i];
    else if (!strcmp(argv[i], "--chunks") && i + 1 < argc) chunks_s = argv[++i];
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warm = atoi(argv[++i]);
    else { fprintf(stderr, "bad arg %s\n", argv[i]); return 2; }
  }
  int dev = 0; CK(cudaSetDevice(dev));
  int nsm, coop; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev)); CK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev));
  const size_t bytes = N_U4 * 16, FLUSH = 256ull << 20;
  uint4 *buf, *flush; uint2* d_list; uint32_t *d_ctr, *d_vis, *d_csum; Rec* d_rec;
  CK(cudaMalloc(&buf, bytes)); CK(cudaMalloc(&flush, FLUSH));
  const int MAXCH = (int)(bytes / 65536) + 8192;
  CK(cudaMalloc(&d_list, MAXCH * sizeof(uint2))); CK(cudaMalloc(&d_rec, MAXCH * sizeof(Rec))); CK(cudaMalloc(&d_vis, MAXCH * 4));
  CK(cudaMalloc(&d_ctr, 4)); CK(cudaMalloc(&d_csum, 4));
  k_fill<<<nsm * 8, 1024>>>(buf, N_U4); CK(cudaDeviceSynchronize());
  uint32_t host_x = 0;
  for (uint64_t i = 0; i < 4 * N_U4; i++) host_x ^= pat(i);

  static void* kfn[NARMS];
  int regs[NARMS];
  for (int a = 0; a < NARMS; a++) {
    kfn[a] = kernel_of(a);
    CK(cudaFuncSetAttribute(kfn[a], cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_DYN));
    int occ; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kfn[a], THREADS, SMEM_DYN));
    if (occ != 1) { fprintf(stderr, "FAIL occupancy arm %s = %d\n", ARM_NAME[a], occ); return 1; }
    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, kfn[a])); regs[a] = fa.numRegs;
  }

  int nfail = 0;
  for (auto& an : split(arms_s)) {
    int a = -1; for (int i = 0; i < NARMS; i++) if (an == ARM_NAME[i]) a = i;
    if (a < 0) { fprintf(stderr, "unknown arm %s\n", an.c_str()); return 2; }
    std::vector<unsigned long long> Cs;
    if (a == A_STATIC) Cs.push_back(0);
    else if (a == A_GUIDED) Cs.push_back(0);
    else for (auto& s : split(chunks_s)) Cs.push_back(strtoull(s.c_str(), 0, 10));
    bool armfail = false;
    for (unsigned long long C : Cs) {
      if (armfail) break;
      // chunk list (uint4 units)
      std::vector<uint2> list;
      if (a == A_STATIC) {
        for (int b = 0; b < nsm; b++) { uint64_t o = N_U4 * b / nsm, e = N_U4 * (b + 1) / nsm; list.push_back(make_uint2((uint32_t)o, (uint32_t)(e - o))); }
      } else if (a == A_GUIDED) {
        uint64_t big = (N_U4 / nsm / 4) / 4096 * 4096, o = 0;  // 4096 u4 = 64 KB
        while (o + big <= N_U4 * 85 / 100) { list.push_back(make_uint2((uint32_t)o, (uint32_t)big)); o += big; }
        while (o + 16384 <= N_U4 * 97 / 100) { list.push_back(make_uint2((uint32_t)o, 16384)); o += 16384; }  // 256 KB
        while (o < N_U4) { uint64_t l = std::min<uint64_t>(4096, N_U4 - o); list.push_back(make_uint2((uint32_t)o, (uint32_t)l)); o += l; }
      } else if (a == A_STATIC_IL) {
        uint64_t n = std::max<uint64_t>(1, (N_U4 * 16 / C + nsm / 2) / nsm) * nsm;  // multiple of grid: equal bytes per CTA (+-1 uint4)
        for (uint64_t c = 0; c < n; c++) { uint64_t o = N_U4 * c / n, e = N_U4 * (c + 1) / n; list.push_back(make_uint2((uint32_t)o, (uint32_t)(e - o))); }
      } else {
        uint64_t cu = C / 16; for (uint64_t o = 0; o < N_U4; o += cu) list.push_back(make_uint2((uint32_t)o, (uint32_t)std::min<uint64_t>(cu, N_U4 - o)));
      }
      int nch = (int)list.size();
      if (nch > MAXCH) { fprintf(stderr, "too many chunks\n"); return 2; }
      CK(cudaMemcpy(d_list, list.data(), nch * sizeof(uint2), cudaMemcpyHostToDevice));
      uint64_t Creport = (a == A_STATIC) ? N_U4 * 16 / nsm : (a == A_GUIDED ? 0 : C);
      bool persistent = (a == A_STATIC || a == A_QUEUE || a == A_QUEUE_PF || a == A_GUIDED || a == A_STATIC_IL);
      int grid = persistent ? nsm : nch;
      cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
      auto launch = [&]() {
        CK(cudaMemsetAsync(d_ctr, 0, 4)); CK(cudaMemsetAsync(d_csum, 0, 4)); CK(cudaMemsetAsync(d_vis, 0, nch * 4));
        k_flush<<<nsm * 4, 1024>>>(flush, FLUSH / 16, 0x5a5a0000u);
        CK(cudaEventRecord(e0));
        const uint4* bp = buf; const uint2* lp = d_list; int n = nch;
        void* args[] = {&bp, &lp, &n, &d_ctr, &d_rec, &d_vis, &d_csum};
        if (persistent && coop) CK(cudaLaunchCooperativeKernel(kfn[a], grid, THREADS, args, SMEM_DYN, 0));
        else CK(cudaLaunchKernel(kfn[a], grid, THREADS, args, SMEM_DYN, 0));
        CK(cudaEventRecord(e1)); CK(cudaDeviceSynchronize());
      };
      // validate
      launch();
      {
        std::vector<uint32_t> vis(nch); uint32_t cs;
        CK(cudaMemcpy(vis.data(), d_vis, nch * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&cs, d_csum, 4, cudaMemcpyDeviceToHost));
        int bad = 0; for (int c = 0; c < nch; c++) bad += (vis[c] != 1);
        bool ok = (bad == 0 && cs == host_x);
        fprintf(stderr, "%s %s C=%llu nchunks=%d visited_bad=%d csum=%08x host=%08x\n", ok ? "PASS" : "FAIL", ARM_NAME[a], (unsigned long long)Creport, nch, bad, cs, host_x);
        if (!ok) { nfail++; armfail = true; continue; }
      }
      if (validate_only) continue;
      for (int w = 0; w < warm; w++) launch();
      for (int r = 0; r < reps; r++) {
        launch();
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        std::vector<Rec> rc(nch); CK(cudaMemcpy(rc.data(), d_rec, nch * sizeof(Rec), cudaMemcpyDeviceToHost));
        unsigned long long tmin = ~0ull, tmax = 0; for (auto& q : rc) { tmin = std::min(tmin, q.t0); tmax = std::max(tmax, q.t1); }
        std::vector<std::vector<int>> bysm(512);
        for (int c = 0; c < nch; c++) bysm[rc[c].smid & 511].push_back(c);
        std::vector<double> fin; std::vector<double> bsm, gaps; double mnb = 1e30, mxb = 0; int nact = 0;
        for (auto& v : bysm) {
          if (v.empty()) continue; nact++;
          std::sort(v.begin(), v.end(), [&](int x, int y) { return rc[x].t0 < rc[y].t0; });
          double f = 0, b = 0; for (int c : v) { f = std::max(f, (double)(rc[c].t1 - tmin)); b += 16.0 * list[c].y; }
          fin.push_back(f); mnb = std::min(mnb, b); mxb = std::max(mxb, b);
          for (size_t k = 1; k < v.size(); k++) gaps.push_back((double)rc[v[k]].t0 - (double)rc[v[k - 1]].t1);
        }
        double mk = (double)(tmax - tmin), mf = med(fin);
        double spread = *std::max_element(fin.begin(), fin.end()) - *std::min_element(fin.begin(), fin.end());
        printf("{\"arm\":\"%s\",\"C\":%llu,\"rep\":%d,\"nchunks\":%d,\"grid\":%d,\"nsm_active\":%d,\"makespan_ns\":%.0f,\"tail_ns\":%.0f,\"spread_ns\":%.0f,"
               "\"p50_ns\":%.0f,\"p90_ns\":%.0f,\"bw_gbs\":%.2f,\"ev_ms\":%.4f,\"ev_gbs\":%.2f,\"bytes_min\":%.0f,\"bytes_max\":%.0f,\"ovh_ns\":%.0f,"
               "\"unroll\":%d,\"threads\":%d,\"smem\":%d,\"regs\":%d}\n",
               ARM_NAME[a], (unsigned long long)Creport, r, nch, grid, nact, mk, mk - mf, spread, mf, pct(fin, 0.9), (double)bytes / mk, ms, (double)bytes / (ms * 1e6),
               mnb, mxb, gaps.empty() ? -1.0 : med(gaps), UNROLL, THREADS, SMEM_DYN, regs[a]);
        fflush(stdout);
      }
      CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    }
  }
  fprintf(stderr, nfail ? "SOME FAIL (%d)\n" : "ALL PASS\n", nfail);
  return nfail ? 1 : 0;
}
