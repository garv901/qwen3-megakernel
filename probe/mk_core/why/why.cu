// why: mechanism of static-vs-dynamic work split when 148 CTAs (1/SM) stream 1 GiB. Derived from clc/clc.cu.
// One kernel, three assignment modes over a chunk list (chunk id -> [off,len)):
//   mode 0 table by blockIdx, mode 1 table by %smid (rate-proportional split), mode 2 atomic queue over tbl (an order).
// Arms (chunk lists): static / static_rotR (148 shares, share s -> CTA (s+R)%148), static_il, il_shiftK, static_perm, queue, queue_perm, static_lpt.
// All 256 KB-class arms share one list of n = 148*28 = 4144 equal chunks. Every arm validated (visited once + XOR checksum).
// Also: E2 latency probe (--lat file): per-SM x per-64KB-region uncached single-load latency; ncu mode (--ncu arm): one launch.
// stdout: JSON summary per arm x rep. --dump file: JSONL per-chunk records (meta line + rep lines). stderr: PASS/FAIL.
// Usage: ./why [--arms a,b] [--reps 10] [--warmup 2] [--dump f] [--lat f] [--ncu arm] [--validate]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#ifndef UNROLL
#define UNROLL 8
#endif
#define THREADS 512
#define SMEM_DYN (160 * 1024)
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

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
// wrot > 0: in batch k warp w reads slot (w + wrot*k) mod 16 (still 512 B/warp contiguous), so no warp keeps a fixed address phase
template <bool W>
__device__ __forceinline__ uint32_t stream(const uint4* buf, uint32_t off, uint32_t len, int sync_k, int wrot) {
  uint32_t x = 0, end = off + len; int nb = 0; uint32_t bi = 0;
  for (uint32_t base = off; base < end; base += THREADS * UNROLL) {
    if (sync_k && nb++ == sync_k) { __syncthreads(); nb = 1; }  // optional CTA re-sync every sync_k batches (64 KB each)
    uint4 v[UNROLL]; bi++;
    const uint32_t tx = W ? ((threadIdx.x + 32u * (uint32_t)wrot * bi) & (THREADS - 1)) : threadIdx.x;
#pragma unroll
    for (int u = 0; u < UNROLL; u++) { uint32_t i = base + u * THREADS + tx; v[u] = (i < end) ? ldnc(buf + i) : make_uint4(0, 0, 0, 0); }
#pragma unroll
    for (int u = 0; u < UNROLL; u++) x ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  return x;
}
__device__ __forceinline__ uint32_t do_chunk(const uint4* buf, const uint2* list, Rec* rec, uint32_t* visited, unsigned long long* wend, int sync_k, int wrot, int c) {
  __syncthreads();
  unsigned long long t0 = 0;
  if (threadIdx.x == 0) t0 = gt();
  uint2 ch = list[c];
  uint32_t x = wrot ? stream<true>(buf, ch.x, ch.y, sync_k, wrot) : stream<false>(buf, ch.x, ch.y, sync_k, 0);
  if ((threadIdx.x & 31) == 0) wend[blockIdx.x * (THREADS / 32) + (threadIdx.x >> 5)] = gt();  // per-warp end of the (last) chunk
  __syncthreads();
  if (threadIdx.x == 0) {
    Rec r; r.smid = smid(); r.cta = blockIdx.x; r.chunk = c; r.pad = 0; r.t0 = t0; r.t1 = gt();
    rec[c] = r; atomicAdd(&visited[c], 1u);
  }
  return x;
}
__global__ void __launch_bounds__(THREADS, 1)
k_stream(const uint4* buf, const uint2* list, const int* tbl, const int2* tinfo, int mode, int sync_k, int wrot, uint32_t* ctr, Rec* rec, uint32_t* visited, unsigned long long* wend, uint32_t* csum) {
  extern __shared__ __align__(16) uint8_t dyn[];  // 160 KB held only to force 1 CTA/SM
  __shared__ int s_c;
  __shared__ uint32_t red[THREADS / 32];
  uint32_t x = 0;
  if (mode == 2) {
    for (;;) {
      if (threadIdx.x == 0) s_c = (int)atomicAdd(ctr, 1u);
      __syncthreads();
      int p = s_c;
      __syncthreads();
      if (p >= tinfo[0].y) break;  // tinfo[0].y = number of queue items
      x ^= do_chunk(buf, list, rec, visited, wend, sync_k, wrot, tbl[p]);
    }
  } else {
    if (threadIdx.x == 0) s_c = (mode == 1) ? (int)smid() : (int)blockIdx.x;
    __syncthreads();
    int2 ti = tinfo[s_c];
    for (int k = 0; k < ti.y; k++) x ^= do_chunk(buf, list, rec, visited, wend, sync_k, wrot, tbl[ti.x + k]);
  }
  for (int o = 16; o; o >>= 1) x ^= __shfl_xor_sync(0xffffffffu, x, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = x;
  __syncthreads();
  if (threadIdx.x == 0) { uint32_t y = 0; for (int i = 0; i < THREADS / 32; i++) y ^= red[i]; atomicXor(csum, y); }
}

// E2: block (1 thread) per SM, timed dependent ld.cg, one probe per 64 KB region. pass 0/1: identical addresses on all SMs (cold race / L2-hit, TLB warm); pass 2: SM-specific line.
__global__ void __launch_bounds__(32, 1) k_lat(const uint8_t* buf, uint16_t* out, int nreg) {
  extern __shared__ uint8_t dyn[];
  if (threadIdx.x) return;
  uint32_t sm = smid();
  for (int pass = 0; pass < 3; pass++) {
    uint32_t acc = 0;
#pragma unroll 1
    for (int r = 0; r < nreg; r++) {
      uint32_t line = pass == 2 ? ((blockIdx.x * 7 + 3) & 511) : 3;  // pass 0/1: same address for every SM (0 cold race, 1 L2-warm); pass 2: SM-specific line (DRAM miss)
      uint32_t dep; asm volatile("and.b32 %0, %1, 0;" : "=r"(dep) : "r"(acc));  // address depends on the previous load: no overlap
      const uint32_t* p = (const uint32_t*)(buf + (size_t)r * 65536 + line * 128 + dep);
      uint32_t v; long long a, b;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(a) : "r"(dep));
      asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
      asm volatile("xor.b32 %0, %0, %1;" : "+r"(acc) : "r"(v));
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(b) : "r"(acc));
      long long d = b - a; out[((size_t)pass * 256 + sm) * nreg + r] = (uint16_t)(d > 65535 ? 65535 : d);
    }
    if (acc == 0x12345678u) out[0] = 1;
  }
}

// E2b: only the block that lands on SM `target` probes (identical addresses for every target, L2 flushed between launches): pure DRAM-miss latency per SM x 64 KB region.
__global__ void __launch_bounds__(32, 1) k_lat1(const uint8_t* buf, uint16_t* out, int nreg, int target) {
  extern __shared__ uint8_t dyn[];
  if (threadIdx.x || (int)smid() != target) return;
  uint32_t acc = 0;
#pragma unroll 1
  for (int r = 0; r < nreg; r++) {
    uint32_t dep; asm volatile("and.b32 %0, %1, 0;" : "=r"(dep) : "r"(acc));
    const uint32_t* p = (const uint32_t*)(buf + (size_t)r * 65536 + 3 * 128 + dep);
    uint32_t v; long long a, b;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(a) : "r"(dep));
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
    asm volatile("xor.b32 %0, %0, %1;" : "+r"(acc) : "r"(v));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(b) : "r"(acc));
    long long d = b - a; out[(size_t)target * nreg + r] = (uint16_t)(d > 65535 ? 65535 : d);
  }
  if (acc == 0x12345678u) out[0] = 1;
}

// ---------------- host ----------------
struct Cfg { int sync_k = 0, wrot = 0; std::string name; std::vector<uint2> list; std::vector<int> tbl; std::vector<int2> tinfo; int mode; };
static double med(std::vector<double> v) { if (v.empty()) return -1; std::sort(v.begin(), v.end()); size_t n = v.size(); return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]); }
static double pct(std::vector<double> v, double p) { if (v.empty()) return -1; std::sort(v.begin(), v.end()); return v[(size_t)(p * (v.size() - 1) + 0.5)]; }
static std::vector<std::string> split(const std::string& s) { std::vector<std::string> o; size_t i = 0; while (i <= s.size()) { size_t j = s.find(',', i); if (j == std::string::npos) j = s.size(); if (j > i) o.push_back(s.substr(i, j - i)); i = j + 1; } return o; }

static std::vector<uint2> equal_list(uint64_t n) { std::vector<uint2> l; for (uint64_t c = 0; c < n; c++) { uint64_t o = N_U4 * c / n, e = N_U4 * (c + 1) / n; l.push_back(make_uint2((uint32_t)o, (uint32_t)(e - o))); } return l; }
static std::vector<double> g_rate;  // per-smid bytes from queue runs (for static_lpt)

// chunk list: nominal count n; al = true -> n = 4096 boundaries at multiples of 256 KB (+ sh uint4 units, sh=0 aligned); else n equal (misaligned) chunks
static std::vector<uint2> shifted_list(uint64_t n, uint64_t sh) {
  std::vector<uint64_t> b(n + 1); const uint64_t C = N_U4 / n;  // n | 4096 -> C = 16384 uint4
  for (uint64_t c = 0; c <= n; c++) b[c] = (c == 0) ? 0 : (c == n ? N_U4 : c * C + sh);
  std::vector<uint2> l; for (uint64_t c = 0; c < n; c++) l.push_back(make_uint2((uint32_t)b[c], (uint32_t)(b[c + 1] - b[c]))); return l;
}
// Arm names: base[_al|_shK]. Bases: static, static_rotR, static_il, il_shiftK, static_perm, static_contig, static_contig_rj (random phase in share),
// static_contig_ro (random order in share), queue, queue_perm, static_lpt.  _al: 4096 aligned 256 KB chunks (CTAs get 27 or 28 chunks);
// _shK: aligned list with every interior boundary moved +K*16 B.  No suffix: 4144 equal chunks (all boundaries 16 B-granular, i.e. misaligned).
static Cfg build(const std::string& an0, int nsm) {
  std::string an = an0; bool al = false; uint64_t sh = 0; int sync_k = 0, wrot = 0;
  { size_t q = an.rfind("_wrot"); if (q != std::string::npos) { wrot = atoi(an.c_str() + q + 5); an = an.substr(0, q); } }
  { size_t q = an.rfind("_sync"); if (q != std::string::npos) { sync_k = atoi(an.c_str() + q + 5); an = an.substr(0, q); } }
  size_t p = an.rfind("_sh"); if (p != std::string::npos && an.find_first_not_of("0123456789", p + 3) == std::string::npos && p + 3 < an.size()) { sh = atoi(an.c_str() + p + 3); an = an.substr(0, p); al = true; }
  if (an.size() > 3 && an.substr(an.size() - 3) == "_al") { an = an.substr(0, an.size() - 3); al = true; }
  Cfg c; c.sync_k = sync_k; c.wrot = wrot; c.name = an0; c.tinfo.assign(512, make_int2(0, 0)); c.mode = 0;
  const uint64_t n = al ? 4096 : (uint64_t)nsm * 28;
  auto mk = [&](uint64_t nn) { return al ? shifted_list(nn, sh) : equal_list(nn); };
  auto table_by_cta = [&](const std::vector<std::vector<int>>& per) { for (int b = 0; b < (int)per.size(); b++) { c.tinfo[b] = make_int2((int)c.tbl.size(), (int)per[b].size()); for (int x : per[b]) c.tbl.push_back(x); } };
  std::vector<int> perm(n); for (uint64_t i = 0; i < n; i++) perm[i] = (int)i; { std::mt19937 g(12345); std::shuffle(perm.begin(), perm.end(), g); }
  auto lo = [&](int b) { return (int)(n * b / nsm); };  // contiguous slice [lo(b), lo(b+1)) of n items
  if (an == "static" || an.rfind("static_rot", 0) == 0) {
    int R = an == "static" ? 0 : atoi(an.c_str() + 10);
    // 148 shares; aligned: share boundaries at chunk multiples (27 or 28 chunks)
    if (al) { for (int s = 0; s < nsm; s++) { uint64_t o = (uint64_t)lo(s) * (N_U4 / n) + (s ? sh : 0), e = (s + 1 == nsm) ? N_U4 : (uint64_t)lo(s + 1) * (N_U4 / n) + sh; c.list.push_back(make_uint2((uint32_t)o, (uint32_t)(e - o))); } }
    else c.list = equal_list(nsm);
    std::vector<std::vector<int>> per(nsm); for (int s = 0; s < nsm; s++) per[(s + R) % nsm].push_back(s); table_by_cta(per);
  } else if (an == "static_il" || an.rfind("il_shift", 0) == 0) {
    int k = an == "static_il" ? 1 : atoi(an.c_str() + 8);
    c.list = mk(n); std::vector<std::vector<int>> per(nsm);
    for (int b = 0; b < nsm; b++) for (uint64_t i = 0; (uint64_t)(((uint64_t)b * k) % nsm) + nsm * i < n; i++) per[b].push_back((int)(((uint64_t)b * k) % nsm) + nsm * (int)i);
    table_by_cta(per);
  } else if (an == "static_perm") {
    c.list = mk(n); std::vector<std::vector<int>> per(nsm);
    for (int b = 0; b < nsm; b++) for (int i = lo(b); i < lo(b + 1); i++) per[b].push_back(perm[i]);
    table_by_cta(per);
  } else if (an == "static_contig" || an == "static_contig_rj" || an == "static_contig_ro") {
    c.list = mk(n); std::vector<std::vector<int>> per(nsm); std::mt19937 g(777);
    for (int b = 0; b < nsm; b++) {
      int a = lo(b), z = lo(b + 1), len = z - a; std::vector<int> v(len); for (int i = 0; i < len; i++) v[i] = a + i;
      if (an == "static_contig_rj") std::rotate(v.begin(), v.begin() + (int)(g() % len), v.end());
      if (an == "static_contig_ro") std::shuffle(v.begin(), v.end(), g);
      per[b] = v;
    }
    table_by_cta(per);
  } else if (an == "queue" || an == "queue_perm") {
    c.list = mk(n); c.mode = 2;
    for (uint64_t i = 0; i < n; i++) c.tbl.push_back(an == "queue" ? (int)i : perm[i]);
    c.tinfo[0] = make_int2(0, (int)n);
  } else if (an == "static_lpt") {
    c.list = mk(n); c.mode = 1;
    std::vector<double> rate(512, 0); double tot = 0; for (int s = 0; s < 512 && s < (int)g_rate.size(); s++) { rate[s] = g_rate[s]; tot += rate[s]; }
    if (tot <= 0) { fprintf(stderr, "static_lpt: no rates\n"); exit(2); }
    std::vector<std::vector<int>> per(512); std::vector<double> asg(512, 0);
    for (uint64_t ch = 0; ch < n; ch++) {  // next chunk to the SM with the smallest (assigned+1)/rate: interleaved, count ~ rate
      int best = -1; double bv = 1e300; for (int s = 0; s < 512; s++) if (rate[s] > 0) { double v = (asg[s] + 1) / rate[s]; if (v < bv) { bv = v; best = s; } }
      per[best].push_back((int)ch); asg[best] += 1;
    }
    table_by_cta(per);  // indexed by smid (mode 1)
  } else { fprintf(stderr, "unknown arm %s\n", an0.c_str()); exit(2); }
  return c;
}

int main(int argc, char** argv) {
  bool validate_only = false; int reps = 10, warm = 2; std::string ncu_arm, dump_f, lat_f, lat1_f;
  std::string arms_s = "static,static_rot37,static_rot74,static_rot111,static_il,il_shift3,il_shift5,il_shift7,static_perm,queue,queue_perm,static_lpt";
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--validate")) validate_only = true;
    else if (!strcmp(argv[i], "--arms") && i + 1 < argc) arms_s = argv[++i];
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warm = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--ncu") && i + 1 < argc) ncu_arm = argv[++i];
    else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump_f = argv[++i];
    else if (!strcmp(argv[i], "--lat") && i + 1 < argc) lat_f = argv[++i];
    else if (!strcmp(argv[i], "--lat1") && i + 1 < argc) lat1_f = argv[++i];
    else { fprintf(stderr, "bad arg %s\n", argv[i]); return 2; }
  }
  int dev = 0; CK(cudaSetDevice(dev));
  int nsm; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  const size_t bytes = N_U4 * 16, FLUSH = 256ull << 20;
  uint4 *buf, *flush; uint2* d_list; int* d_tbl; int2* d_ti; uint32_t *d_ctr, *d_vis, *d_csum; Rec* d_rec; unsigned long long* d_wend;
  CK(cudaMalloc(&buf, bytes)); CK(cudaMalloc(&flush, FLUSH));
  const int MAXCH = 8192;
  CK(cudaMalloc(&d_list, MAXCH * sizeof(uint2))); CK(cudaMalloc(&d_tbl, MAXCH * 4)); CK(cudaMalloc(&d_ti, 512 * sizeof(int2)));
  CK(cudaMalloc(&d_rec, MAXCH * sizeof(Rec))); CK(cudaMalloc(&d_vis, MAXCH * 4)); CK(cudaMalloc(&d_wend, 512 * 16 * 8)); CK(cudaMalloc(&d_ctr, 4)); CK(cudaMalloc(&d_csum, 4));
  k_fill<<<nsm * 8, 1024>>>(buf, N_U4); CK(cudaDeviceSynchronize());
  uint32_t host_x = 0; for (uint64_t i = 0; i < 4 * N_U4; i++) host_x ^= pat(i);
  CK(cudaFuncSetAttribute(k_stream, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_DYN));
  CK(cudaFuncSetAttribute(k_lat, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_DYN));
  { int occ; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_stream, THREADS, SMEM_DYN)); if (occ != 1) { fprintf(stderr, "FAIL occupancy %d\n", occ); return 1; } }

  if (!lat1_f.empty()) {  // E2b
    CK(cudaFuncSetAttribute(k_lat1, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_DYN));
    const int NREG = (int)(bytes / 65536), NL = 3; uint16_t* d_out; CK(cudaMalloc(&d_out, (size_t)256 * NREG * 2));
    std::vector<std::vector<uint16_t>> all(NL, std::vector<uint16_t>((size_t)nsm * NREG));
    for (int l = 0; l < NL; l++) for (int t = 0; t < nsm; t++) {
      CK(cudaMemset(d_out, 0, (size_t)256 * NREG * 2));
      k_flush<<<nsm * 4, 1024>>>(flush, FLUSH / 16, 0x5a5a0000u + t);
      const uint8_t* bp = (const uint8_t*)buf; int tt = t; void* args[] = {&bp, &d_out, (void*)&NREG, &tt};
      CK(cudaLaunchCooperativeKernel((void*)k_lat1, nsm, 32, args, SMEM_DYN, 0)); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(all[l].data() + (size_t)t * NREG, d_out + (size_t)t * NREG, NREG * 2, cudaMemcpyDeviceToHost));
    }
    std::vector<uint16_t> m(all[0].size()); std::vector<uint16_t> tv(NL);
    for (size_t i = 0; i < m.size(); i++) { for (int l = 0; l < NL; l++) tv[l] = all[l][i]; std::sort(tv.begin(), tv.end()); m[i] = tv[NL / 2]; }
    FILE* f = fopen(lat1_f.c_str(), "wb"); fwrite(m.data(), 2, m.size(), f); fclose(f);
    fprintf(stderr, "lat1: wrote %s [%d][%d] uint16 cycles\n", lat1_f.c_str(), nsm, NREG);
    return 0;
  }
  if (!lat_f.empty()) {  // E2
    const int NREG = (int)(bytes / 65536); uint16_t* d_out; CK(cudaMalloc(&d_out, (size_t)3 * 256 * NREG * 2));
    const int NL = 5; std::vector<std::vector<uint16_t>> all(NL, std::vector<uint16_t>((size_t)3 * 256 * NREG));
    for (int l = 0; l < NL; l++) {
      CK(cudaMemset(d_out, 0, (size_t)3 * 256 * NREG * 2));
      k_flush<<<nsm * 4, 1024>>>(flush, FLUSH / 16, 0x5a5a0000u + l);
      const uint8_t* bp = (const uint8_t*)buf; void* args[] = {&bp, &d_out, (void*)&NREG};
      CK(cudaLaunchCooperativeKernel((void*)k_lat, nsm, 32, args, SMEM_DYN, 0)); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(all[l].data(), d_out, all[l].size() * 2, cudaMemcpyDeviceToHost));
    }
    std::vector<uint16_t> m(all[0].size()); std::vector<uint16_t> t(NL);
    for (size_t i = 0; i < m.size(); i++) { for (int l = 0; l < NL; l++) t[l] = all[l][i]; std::sort(t.begin(), t.end()); m[i] = t[NL / 2]; }
    FILE* f = fopen(lat_f.c_str(), "wb"); fwrite(m.data(), 2, m.size(), f); fclose(f);
    fprintf(stderr, "lat: wrote %s [3][256][%d] uint16 cycles (median of %d launches)\n", lat_f.c_str(), NREG, NL);
    return 0;
  }

  if (std::find(split(arms_s).begin(), split(arms_s).end(), std::string("static_lpt")) != split(arms_s).end() || ncu_arm == "static_lpt") {
    Cfg q = build("queue", nsm);  // gather per-SM rates (bytes per SM over 10 queue launches)
    CK(cudaMemcpy(d_list, q.list.data(), q.list.size() * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_tbl, q.tbl.data(), q.tbl.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ti, q.tinfo.data(), 512 * sizeof(int2), cudaMemcpyHostToDevice));
    g_rate.assign(512, 0); int nch = (int)q.list.size();
    for (int r = 0; r < 12; r++) {
      CK(cudaMemset(d_ctr, 0, 4)); CK(cudaMemset(d_csum, 0, 4)); CK(cudaMemset(d_vis, 0, nch * 4));
      k_flush<<<nsm * 4, 1024>>>(flush, FLUSH / 16, 0x5a5a0000u);
      const uint4* bp = buf; const uint2* lp = d_list; const int* tp = d_tbl; const int2* ip = d_ti; int mode = q.mode, sk = 0, wr = 0;
      void* args[] = {&bp, &lp, &tp, &ip, &mode, &sk, &wr, &d_ctr, &d_rec, &d_vis, &d_wend, &d_csum};
      CK(cudaLaunchCooperativeKernel((void*)k_stream, nsm, THREADS, args, SMEM_DYN, 0)); CK(cudaDeviceSynchronize());
      if (r < 2) continue;
      std::vector<Rec> rc(nch); CK(cudaMemcpy(rc.data(), d_rec, nch * sizeof(Rec), cudaMemcpyDeviceToHost));
      for (auto& x : rc) g_rate[x.smid & 511] += q.list[x.chunk].y;
    }
  }

  std::vector<std::string> arms = ncu_arm.empty() ? split(arms_s) : std::vector<std::string>{ncu_arm};
  FILE* dump = dump_f.empty() ? nullptr : fopen(dump_f.c_str(), "w");
  int nfail = 0;
  for (auto& an : arms) {
    Cfg c = build(an, nsm); int nch = (int)c.list.size();
    CK(cudaMemcpy(d_list, c.list.data(), nch * sizeof(uint2), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tbl, c.tbl.data(), c.tbl.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_ti, c.tinfo.data(), 512 * sizeof(int2), cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto launch = [&]() {
      CK(cudaMemset(d_ctr, 0, 4)); CK(cudaMemset(d_csum, 0, 4)); CK(cudaMemset(d_vis, 0, nch * 4)); CK(cudaMemset(d_rec, 0, nch * sizeof(Rec)));
      k_flush<<<nsm * 4, 1024>>>(flush, FLUSH / 16, 0x5a5a0000u);
      CK(cudaEventRecord(e0));
      const uint4* bp = buf; const uint2* lp = d_list; const int* tp = d_tbl; const int2* ip = d_ti; int mode = c.mode, sk = c.sync_k, wr = c.wrot;
      void* args[] = {&bp, &lp, &tp, &ip, &mode, &sk, &wr, &d_ctr, &d_rec, &d_vis, &d_wend, &d_csum};
      CK(cudaLaunchCooperativeKernel((void*)k_stream, nsm, THREADS, args, SMEM_DYN, 0));
      CK(cudaEventRecord(e1)); CK(cudaDeviceSynchronize());
    };
    if (!ncu_arm.empty()) { launch(); fprintf(stderr, "ncu launch done %s\n", an.c_str()); continue; }
    launch();
    {
      std::vector<uint32_t> vis(nch); uint32_t cs;
      CK(cudaMemcpy(vis.data(), d_vis, nch * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&cs, d_csum, 4, cudaMemcpyDeviceToHost));
      int bad = 0; for (int i = 0; i < nch; i++) bad += (vis[i] != 1);
      bool ok = (bad == 0 && cs == host_x);
      fprintf(stderr, "%s %s nchunks=%d visited_bad=%d csum=%08x host=%08x\n", ok ? "PASS" : "FAIL", an.c_str(), nch, bad, cs, host_x);
      if (!ok) { nfail++; continue; }
    }
    if (validate_only) continue;
    if (dump) {
      fprintf(dump, "{\"type\":\"meta\",\"arm\":\"%s\",\"n\":%d,\"off\":[", an.c_str(), nch);
      for (int i = 0; i < nch; i++) fprintf(dump, "%s%u", i ? "," : "", c.list[i].x);
      fprintf(dump, "],\"len\":[");
      for (int i = 0; i < nch; i++) fprintf(dump, "%s%u", i ? "," : "", c.list[i].y);
      fprintf(dump, "]}\n");
    }
    for (int w = 0; w < warm; w++) launch();
    for (int r = 0; r < reps; r++) {
      launch();
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
      std::vector<Rec> rc(nch); CK(cudaMemcpy(rc.data(), d_rec, nch * sizeof(Rec), cudaMemcpyDeviceToHost));
      unsigned long long tmin = ~0ull, tmax = 0; for (auto& q : rc) { tmin = std::min(tmin, q.t0); tmax = std::max(tmax, q.t1); }
      std::vector<std::vector<int>> bysm(512);
      for (int i = 0; i < nch; i++) bysm[rc[i].smid & 511].push_back(i);
      std::vector<double> fin; double mnb = 1e30, mxb = 0; int nact = 0;
      for (auto& v : bysm) {
        if (v.empty()) continue; nact++;
        double f = 0, b = 0; for (int i : v) { f = std::max(f, (double)(rc[i].t1 - tmin)); b += 16.0 * c.list[i].y; }
        fin.push_back(f); mnb = std::min(mnb, b); mxb = std::max(mxb, b);
      }
      std::vector<unsigned long long> we(nsm * (THREADS / 32)); CK(cudaMemcpy(we.data(), d_wend, we.size() * 8, cudaMemcpyDeviceToHost));
      std::vector<double> wsp; for (int b = 0; b < nsm; b++) { auto mm = std::minmax_element(we.begin() + b * 16, we.begin() + b * 16 + 16); wsp.push_back((double)(*mm.second - *mm.first)); }
      double mk = (double)(tmax - tmin), mf = med(fin);
      double spread = *std::max_element(fin.begin(), fin.end()) - *std::min_element(fin.begin(), fin.end());
      printf("{\"arm\":\"%s\",\"rep\":%d,\"nchunks\":%d,\"nsm_active\":%d,\"makespan_ns\":%.0f,\"tail_ns\":%.0f,\"spread_ns\":%.0f,\"p50_ns\":%.0f,\"p90_ns\":%.0f,"
             "\"bw_gbs\":%.2f,\"ev_ms\":%.4f,\"bytes_min\":%.0f,\"bytes_max\":%.0f,\"warp_spread_med_ns\":%.0f,\"warp_spread_p90_ns\":%.0f}\n",
             an.c_str(), r, nch, nact, mk, mk - mf, spread, mf, pct(fin, 0.9), (double)bytes / mk, ms, mnb, mxb, med(wsp), pct(wsp, 0.9));
      fflush(stdout);
      if (dump) {
        fprintf(dump, "{\"type\":\"rep\",\"arm\":\"%s\",\"rep\":%d,\"smid\":[", an.c_str(), r);
        for (int i = 0; i < nch; i++) fprintf(dump, "%s%u", i ? "," : "", rc[i].smid);
        fprintf(dump, "],\"cta\":[");
        for (int i = 0; i < nch; i++) fprintf(dump, "%s%u", i ? "," : "", rc[i].cta);
        fprintf(dump, "],\"t0\":[");
        for (int i = 0; i < nch; i++) fprintf(dump, "%s%llu", i ? "," : "", rc[i].t0 - tmin);
        fprintf(dump, "],\"t1\":[");
        for (int i = 0; i < nch; i++) fprintf(dump, "%s%llu", i ? "," : "", rc[i].t1 - tmin);
        fprintf(dump, "]}\n");
      }
    }
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  }
  if (dump) fclose(dump);
  fprintf(stderr, nfail ? "SOME FAIL (%d)\n" : "ALL PASS\n", nfail);
  return nfail ? 1 : 0;
}
