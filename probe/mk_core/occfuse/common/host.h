// occfuse common host helpers (C++17). Include after ub.cuh. Tensor-map encode and pool setup follow probe/mk_core/xover4/xover4.cu main().
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <string>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)

static double med(std::vector<double> v) { if (v.empty()) return -1; std::sort(v.begin(), v.end()); size_t n = v.size(); return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]); }
static double vmin(const std::vector<double>& v) { return v.empty() ? -1 : *std::min_element(v.begin(), v.end()); }
static double vmax(const std::vector<double>& v) { return v.empty() ? -1 : *std::max_element(v.begin(), v.end()); }

// cuTensorMapEncodeTiled through the runtime entry point (no -lcuda needed).
static PFN_cuTensorMapEncodeTiled_v12000 tmap_enc() {
  static PFN_cuTensorMapEncodeTiled_v12000 enc = nullptr;
  if (!enc) { cudaDriverEntryPointQueryResult qr; CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&enc, 12000, cudaEnableDefault, &qr)); }
  return enc;
}
// 2D bf16 row-major [rows][inner] map, box {box_inner, box_rows}, SW128 (box_inner * 2 B must be 128 for SW128 K-major boxes as in mk: box_inner = 64).
static CUtensorMap tmap_bf16(const void* base, unsigned long long inner, unsigned long long rows, unsigned box_inner, unsigned box_rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)inner, (cuuint64_t)rows}, strides[1] = {(cuuint64_t)inner * 2};
  cuuint32_t box[2] = {box_inner, box_rows}, es[2] = {1, 1};
  CUresult r = tmap_enc()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                          CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "tensor map encode failed %d\n", (int)r); exit(1); }
  return m;
}

// Weight pool larger than 2x L2 (A always from HBM): phases read consecutive row windows, wrapping (xover4 prow).
struct Pool { bf16* p = nullptr; size_t bytes = 0; };
static Pool pool_alloc(size_t bytes = 2ull << 30, unsigned seed = 12345u) {
  Pool P; P.bytes = bytes; CK(cudaMalloc(&P.p, bytes));
  k_init_bf16<<<4096, 256>>>(P.p, bytes / 2, seed); CK(cudaDeviceSynchronize());
  int l2 = 0; CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
  if (bytes < 2ull * (size_t)l2) fprintf(stderr, "warning: pool %zu B < 2 x L2 (%d B)\n", bytes, l2);
  return P;
}

// Cooperative launch, 1 CTA per SM (grid barrier needs all CTAs resident).
template <class K, class... A>
static void launch_coop(K kern, int grid, int smem, A... args) {
  cudaLaunchConfig_t lc = {}; lc.gridDim = dim3(grid); lc.blockDim = dim3(NT); lc.dynamicSmemBytes = smem; lc.stream = 0;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeCooperative; at[0].val.cooperative = 1; lc.attrs = at; lc.numAttrs = 1;
  CK(cudaLaunchKernelEx(&lc, kern, args...));
}

// SM clock in MHz from a clk_sample buffer (median over CTAs).
static double mhz_of(const std::vector<unsigned long long>& ck, int grid) {
  std::vector<double> m;
  for (int b = 0; b < grid; ++b) { double dc = (double)(ck[b * 4 + 2] - ck[b * 4]), dn = (double)(ck[b * 4 + 3] - ck[b * 4 + 1]); if (dn > 0) m.push_back(dc / dn * 1e3); }
  return med(m);
}

// Raw stamps: append one launch [grid][NSTMAX] uint64 to a binary file; common/stamps.py reads it with the JSONL row's "stamps" field.
static void stamps_append(const char* path, const std::vector<unsigned long long>& st) {
  if (!path) return;
  FILE* f = fopen(path, "ab"); if (!f) { perror(path); exit(1); }
  fwrite(st.data(), sizeof(unsigned long long), st.size(), f); fclose(f);
}

// JSONL row builder: J j; j.s("bench","b1").i("N",148).d("us",1.23).arr("us_reps",v); j.print();
struct J {
  std::string b = "{";
  J& key(const char* k) { if (b.size() > 1) b += ","; b += "\""; b += k; b += "\":"; return *this; }
  J& s(const char* k, const std::string& v) { key(k); b += "\"" + v + "\""; return *this; }
  J& i(const char* k, long long v) { key(k); b += std::to_string(v); return *this; }
  J& d(const char* k, double v) { key(k); char t[64]; snprintf(t, sizeof t, "%.4f", v); b += t; return *this; }
  J& arr(const char* k, const std::vector<double>& v) { key(k); b += "["; for (size_t n = 0; n < v.size(); ++n) { char t[64]; snprintf(t, sizeof t, "%s%.4f", n ? "," : "", v[n]); b += t; } b += "]"; return *this; }
  void print() { b += "}"; printf("%s\n", b.c_str()); fflush(stdout); b = "{"; }
};
