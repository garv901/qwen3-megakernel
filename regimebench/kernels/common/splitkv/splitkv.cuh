// A1 v1: split-KV decode attention (q_len == 1 per sequence) + one-pass combine, for the bespoke megakernel `mk` (MK-attn owns this file).
// Packaging: the consumer prepends this text to its NVRTC source. No includes, no global macro dependencies: model sizes are constexpr in
// namespace skv (named kXxx so they cannot collide with a global macro H/NL/QH/KVH/HD/NT/...), every macro starts with SKV_.
// Contract: regimebench/kernels/common/README.md + splitkv/CHANGELOG.md (interface v1). Planner: splitkv/plan.py (item encoding documented there).
//
// Math domain: q is pre-scaled by (1/sqrt(128)) * log2(e), so scores live in log2 units and every exp is a bare ex2.approx.
// Partial slots hold (o unnormalised w.r.t. m, m, l) in that domain; only combine_head / combine_all_to_ao read them.
// Memory notes: the LDG engine reads KV rows with a coherent ld.global.L1::no_allocate (the new token's row is written by the caller earlier
// in the same kernel, so .nc would be outside the PTX contract; SKV_KV_NC=1 selects it, same speed). TMA reads go through L2.
// q / ao / partials are read with ld.global.cg (L1 bypass).
#ifndef SKV_CUH_
#define SKV_CUH_

#ifndef SKV_LDG_DB
#define SKV_LDG_DB 2   // LDG engine: 1 = register double buffer (step i+1's loads before step i), 2 = half buffer (after step i's scores), 0 = single
#endif

#ifndef SKV_KV_NC
#define SKV_KV_NC 0    // LDG engine KV loads: 0 = coherent weak load (default; = nc within 0.05 us, P2 job 9601), 1 = ld.global.nc (outside PTX contract)
#endif

#ifndef SKV_FORCE_ENGINE
#define SKV_FORCE_ENGINE (-1)   // -1: runtime Params::engine; 0 / 1: compile only the LDG / TMA engine (smaller code)
#endif

#ifndef SKV_STAGE_ROWS
#define SKV_STAGE_ROWS 128  // TMA engine: positions per ring stage (32 / 64 / 128; K and V are rows * 256 B each). v1 = 32; 128 x 2 best in P2b (job 9610)
#endif
#ifndef SKV_STAGES
#define SKV_STAGES 2        // TMA engine: ring stages (2..8). Ring bytes = stages * rows * 512. v1 = 4
#endif
#ifndef SKV_TMA_RELEASE
#define SKV_TMA_RELEASE 1   // 0 = v1 (__syncthreads after every stage, then thread 0 refills); 1 = per-slot "empty" mbarrier (16 warp arrivals), no CTA barrier in the stage loop
#endif
#ifndef SKV_PROF_NOCOMP
#define SKV_PROF_NOCOMP 0   // 1 = profiling only: no dots / softmax / PV, loaded K/V words are xored into st.o0[0]; results are wrong by design
#endif
#ifndef SKV_ITEM_PREFETCH
#define SKV_ITEM_PREFETCH 0 // (default 0: no gain, job 9610) 1 = item k+1's record + q are loaded during item k, and the LDG engine issues its first K/V loads before item k's merge
#endif

namespace skv {

typedef unsigned short bf16;   // raw bf16 bits (hv2g does the same; no cuda_bf16 dependency)

constexpr int kKvHeads = 8, kDim = 128, kQHeads = 16;
constexpr int kBlock = 512;                       // threads per CTA (16 warps)
constexpr int kPartStride = 132;                  // floats per partial slot: o[128], m, l, pad (528 B)
constexpr int kStageRows = SKV_STAGE_ROWS, kStages = SKV_STAGES;   // TMA ring: kStages stages, a stage = kStageRows positions of K + V (256 B each)
static_assert(kStageRows == 32 || kStageRows == 64 || kStageRows == 128, "SKV_STAGE_ROWS must be 32, 64 or 128");
static_assert(kStages >= 2 && kStages <= 8, "SKV_STAGES must be 2..8");
constexpr int kU = kStageRows / 32;               // positions per row slot per stage (one online-softmax rescale per kU positions)
constexpr int kStageBytes = 2 * kStageRows * 256; // K half then V half
constexpr int kNumMaps = (kStageRows == 32 ? 6 : kStageRows == 64 ? 7 : 8);   // box rows kStageRows, /2, .., 1 per K / V: tmaps[kv * kNumMaps + log2(kStageRows / rows)]
constexpr int kMergeChunk = 20;                   // combine: partial float4 loads in flight per lane
constexpr float kQScale = 0.08838834764831845f * 1.4426950408889634f;   // 1/sqrt(128) * log2(e)
constexpr int kRingOff = 0, kMrgOff = kStages * kStageBytes, kMrgBytes = 16 * 2 * kPartStride * 4;
constexpr int kBarBytes = (16 * kStages > 64 ? 16 * kStages : 64);   // mbarriers: full[s] at +8 s, empty[s] at +8 (kStages + s)
constexpr int kBarOff = kMrgOff + kMrgBytes, kSmemBytes = kBarOff + kBarBytes;

struct Params {                    // filled by the host planner (plan.py) + caller pointers; plain device pointers
  const bf16* q;                   // [total_q, 16, 128] bf16, post q-norm + RoPE
  const bf16* kc; const bf16* vc;  // [L, cache_rows, 8, 128]
  int cache_rows, n_layers;
  const int4* items; int items_stride;      // CTA b's item records: items + b * items_stride * 2 (int4 units); record = 8 ints, see plan.py
  const int* item_off;             // unused by attend (kept for callers)
  const int* pair_info;            // per (seq * 8 + kvh): 4 ints {part_base, S, tok0, vis_len} (combine only)
  float* part;                     // partials scratch (plan.part_floats)
  bf16* ao;                        // fallback output [total_q, 2048] bf16 (always written for S == 1 pairs)
  const void* tmaps;               // 2 * kNumMaps CUtensorMap (128 B each, 64 B aligned): [K|V][box rows kStageRows..1]; null if engine LDG
  int engine;                      // 0 LDG, 1 TMA (planner decides per launch)
  int total_q;
};

// ------------------------------------------------------------------------------------------------ helpers
__device__ __forceinline__ float bflo(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bfhi(unsigned u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ unsigned short f2bf(float f) {   // RNE
  unsigned u = __float_as_uint(f);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (unsigned short)(u >> 16);
}
__device__ __forceinline__ unsigned pack2(float a, float b) { return (unsigned)f2bf(a) | ((unsigned)f2bf(b) << 16); }
__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ uint4 ldg_nc(const void* p) {   // KV stream load; SKV_KV_NC 1: non-coherent (LDG...CONSTANT), 0: coherent weak load
  uint4 v;
#if SKV_KV_NC
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
#else
  asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
#endif
  return v;
}
__device__ __forceinline__ uint4 ldg_cg4(const void* p) { return __ldcg(reinterpret_cast<const uint4*>(p)); }
__device__ __forceinline__ unsigned smaddr(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint4 lds128(unsigned a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ void mb_init(unsigned b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n)); }
__device__ __forceinline__ void mb_expect(unsigned b, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mb_wait(unsigned b, unsigned parity) {
  asm volatile("{\n .reg .pred p;\n SKV_WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra SKV_WAIT_%=;\n}\n" ::"r"(b), "r"(parity) : "memory");
}
// arrive (count 1) on an mbarrier. Ordering of this thread's earlier shared loads before the arrive comes from the arrive's default .release.cta
// semantics (the CUTLASS consumer-release pattern). `dep` appears only in a PTX comment: it does not create a hardware dependency.
__device__ __forceinline__ void mb_arrive_dep(unsigned b, unsigned dep) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0]; // %1" ::"r"(b), "r"(dep) : "memory");
}
#ifdef SKV_DEBUG_WAIT   // debug: bounded wait; thread 0 prints a stuck wait, then every thread gives up on it (results wrong, kernel ends)
__device__ __forceinline__ bool mb_try(unsigned b, unsigned parity) {
  unsigned ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(b), "r"(parity) : "memory");
  return ok;
}
__device__ __forceinline__ void mb_wait_dbg(unsigned b, unsigned parity, int layer, unsigned g, int item, int j) {
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  while (!mb_try(b, parity)) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    if (t - t0 > (threadIdx.x == 0 ? 50000000ull : 100000000ull)) {   // 50 / 100 ms; thread 0 reports first
      if (threadIdx.x == 0) printf("SKV_DEBUG_WAIT cta %d layer %d g %u slot %u parity %u item %d stage %d\n", blockIdx.x, layer, g, g % kStages, parity, item, j);
      return;   // give up (results are wrong) so the kernel ends and the printf buffer is flushed
    }
  }
}
#endif
__device__ __forceinline__ void tma3d(unsigned dst, const void* map, int c0, int c1, int c2, unsigned b) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
               ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(b) : "memory");
}

// per-slot online-softmax state: two q-heads, m / l replicated over the 16 lanes of a half-warp, o = this lane's 8 dims
struct St { float m0, m1, l0, l1, o0[8], o1[8]; };
__device__ __forceinline__ void st_init(St& s) {
  s.m0 = s.m1 = neg_inf(); s.l0 = s.l1 = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) s.o0[j] = s.o1[j] = 0.f;
}
__device__ __forceinline__ void unpack8(const uint4& x, float (&f)[8]) {
  f[0] = bflo(x.x); f[1] = bfhi(x.x); f[2] = bflo(x.y); f[3] = bfhi(x.y);
  f[4] = bflo(x.z); f[5] = bfhi(x.z); f[6] = bflo(x.w); f[7] = bfhi(x.w);
}
// score of one position for both heads: 16 FFMA over this lane's 8 dims (two chains per head), then xor 8,4,2,1 over the half-warp
__device__ __forceinline__ void score2(const uint4& k, const float (&q0)[8], const float (&q1)[8], float& s0, float& s1) {
  float kf[8];
  unpack8(k, kf);
  float a0 = kf[0] * q0[0], a1 = kf[1] * q0[1], b0 = kf[0] * q1[0], b1 = kf[1] * q1[1];
#pragma unroll
  for (int j = 2; j < 8; j += 2) {
    a0 = fmaf(kf[j], q0[j], a0); a1 = fmaf(kf[j + 1], q0[j + 1], a1);
    b0 = fmaf(kf[j], q1[j], b0); b1 = fmaf(kf[j + 1], q1[j + 1], b1);
  }
  s0 = a0 + a1; s1 = b0 + b1;
#pragma unroll
  for (int o = 8; o; o >>= 1) { s0 += __shfl_xor_sync(0xffffffffu, s0, o); s1 += __shfl_xor_sync(0xffffffffu, s1, o); }
}
// one online-softmax step over U positions (one rescale per step); invalid positions (valid[u] false) score -inf and must carry v == 0.
// Split in two halves so the LDG engine can issue the next step's loads between them (K registers are dead after step_score).
// SKV_PROF_NOCOMP: no arithmetic; step_score folds the K words of each position into s0[u], step_acc xors s0[u] and the V words into st.o0[0].
#if SKV_PROF_NOCOMP
__device__ __forceinline__ unsigned xor4(const uint4& x) { return x.x ^ x.y ^ x.z ^ x.w; }
template <int U>
__device__ __forceinline__ void step_score(const float (&q0)[8], const float (&q1)[8], const uint4 (&k)[U], const bool (&valid)[U], float (&s0)[U], float (&s1)[U]) {
  (void)q0; (void)q1; (void)valid;
#pragma unroll
  for (int u = 0; u < U; ++u) { s0[u] = __uint_as_float(xor4(k[u])); s1[u] = 0.f; }
}
template <int U>
__device__ __forceinline__ void step_acc(St& s, const float (&s0)[U], const float (&s1)[U], const uint4 (&v)[U]) {
  (void)s1;
  unsigned x = __float_as_uint(s.o0[0]);
#pragma unroll
  for (int u = 0; u < U; ++u) x ^= __float_as_uint(s0[u]) ^ xor4(v[u]);
  s.o0[0] = __uint_as_float(x);
}
#else
template <int U>
__device__ __forceinline__ void step_score(const float (&q0)[8], const float (&q1)[8], const uint4 (&k)[U], const bool (&valid)[U], float (&s0)[U], float (&s1)[U]) {
#pragma unroll
  for (int u = 0; u < U; ++u) {
    score2(k[u], q0, q1, s0[u], s1[u]);
    if (!valid[u]) { s0[u] = neg_inf(); s1[u] = neg_inf(); }
  }
}
template <int U>
__device__ __forceinline__ void step_acc(St& s, const float (&s0)[U], const float (&s1)[U], const uint4 (&v)[U]) {
  float n0 = s.m0, n1 = s.m1;
#pragma unroll
  for (int u = 0; u < U; ++u) { n0 = fmaxf(n0, s0[u]); n1 = fmaxf(n1, s1[u]); }
  const float c0 = n0 == neg_inf() ? 0.f : n0, c1 = n1 == neg_inf() ? 0.f : n1;
  const float a0 = ex2(s.m0 - c0), a1 = ex2(s.m1 - c1);
  s.m0 = n0; s.m1 = n1;
  s.l0 *= a0; s.l1 *= a1;
#pragma unroll
  for (int j = 0; j < 8; ++j) { s.o0[j] *= a0; s.o1[j] *= a1; }
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const float p0 = ex2(s0[u] - c0), p1 = ex2(s1[u] - c1);
    s.l0 += p0; s.l1 += p1;
    float vf[8];
    unpack8(v[u], vf);
#pragma unroll
    for (int j = 0; j < 8; ++j) { s.o0[j] = fmaf(p0, vf[j], s.o0[j]); s.o1[j] = fmaf(p1, vf[j], s.o1[j]); }
  }
}
#endif
template <int U>
__device__ __forceinline__ void step(St& s, const float (&q0)[8], const float (&q1)[8], const uint4 (&k)[U], const uint4 (&v)[U], const bool (&valid)[U]) {
  float s0[U], s1[U];
  step_score<U>(q0, q1, k, valid, s0, s1);
  step_acc<U>(s, s0, s1, v);
}
// merge two chunk states of one head (m == -inf allowed on either side, not both)
__device__ __forceinline__ void merge_state(float& m, float& l, float (&o)[8], float m2, float l2, const float (&o2)[8]) {
  const float n = fmaxf(m, m2), c = n == neg_inf() ? 0.f : n;
  const float a = ex2(m - c), b = ex2(m2 - c);
  l = l * a + l2 * b;
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = o[j] * a + o2[j] * b;
  m = n;
}

constexpr size_t attend_smem_bytes() { return (size_t)kSmemBytes; }   // dynamic smem attend needs: TMA ring + merge buffer 16.5 KB + mbarriers

// ------------------------------------------------------------------------------------------------ attend
// The CTA's item loop for one engine (TMA = compile-time engine, so the LDG-only registers do not exist in the TMA loop and vice versa).
// rec = this CTA's item records (2 int4 each: {row_base, y, p0, p1}, {part_slot, S, tok0, n_items}); g = TMA stages consumed so far (in/out).
template <bool TMA>
__device__ __forceinline__ void run_items(const Params& p, int layer, unsigned char* smem, unsigned& g, const int4* rec, int ni) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, l16 = lane & 15;
  const int r = warp * 2 + (lane >> 4);                       // row slot 0..31
  const unsigned sbase = smaddr(smem);
  const unsigned bar0 = sbase + kBarOff;
  float* mrg = reinterpret_cast<float*>(smem + kMrgOff);
  const size_t layer_row = (size_t)layer * p.cache_rows;
  const int4 zero4 = make_int4(0, 0, 0, 0);

  // ---- TMA producer state (thread 0 only)
  int pk = -1, pp = 0, pend = 0, pkvh = 0;
  size_t prow = 0;
  int4 pnxt = zero4;
  auto produce = [&](int slot) {                               // issue the next stage of the CTA's flat stage sequence into `slot`
    if (pp >= pend) {
      if (pk + 1 >= ni) return;
      ++pk;
      const int4 it = pnxt;
      pnxt = (pk + 1 < ni) ? rec[2 * (pk + 1)] : zero4;
      pp = it.z; pend = it.w; pkvh = it.y & 15; prow = layer_row + (size_t)it.x;
    }
    const int n = min(kStageRows, pend - pp);
    const unsigned dst = sbase + kRingOff + slot * kStageBytes, bar = bar0 + 8 * slot;
    mb_expect(bar, (unsigned)n * 512u);
    int off = 0, rc = (int)(prow + pp);
#pragma unroll
    for (int rs = 0; rs < kNumMaps; ++rs) {
      const int rows = kStageRows >> rs;
      if (n & rows) {
        tma3d(dst + off * 256, (const char*)p.tmaps + (0 * kNumMaps + rs) * 128, 0, pkvh, rc, bar);
        tma3d(dst + kStageRows * 256 + off * 256, (const char*)p.tmaps + (1 * kNumMaps + rs) * 128, 0, pkvh, rc, bar);
        off += rows; rc += rows;
      }
    }
    pp += n;
  };
  if (TMA && tid == 0) {
    pnxt = rec[0];
#pragma unroll
    for (int s = 0; s < kStages; ++s) produce((g + s) % kStages);
  }

  // ---- item records and q: item k's are loaded by iteration k-1 (SKV_ITEM_PREFETCH 1) or at the end of iteration k-1 (0)
  uint4 ka[4], va[4];                                          // LDG engine: first-step K / V registers of the current item
  // byte offset of (row_base, kvh, this lane's 16 B) in a [L, rows, 8, 128] bf16 cache; the cache pointers stay in the (constant-bank) params
  auto item_off = [&](const int4& a) { return ((layer_row + (size_t)a.x) * kKvHeads + (a.y & 15)) * 256 + l16 * 16; };
  auto ldg_first = [&](const int4& a) {                        // LDG engine: item `a`'s step-0 loads (4 positions of K and V per thread)
    const size_t o = item_off(a);
    const char* kb = (const char*)p.kc + o;
    const char* vb = (const char*)p.vc + o;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int pos = a.z + r + 32 * u;
      ka[u] = pos < a.w ? ldg_nc(kb + (size_t)pos * 2048) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int pos = a.z + r + 32 * u;
      va[u] = pos < a.w ? ldg_nc(vb + (size_t)pos * 2048) : make_uint4(0, 0, 0, 0);
    }
  };
  auto load_q = [&](const int4& a, int tok, uint4& x, uint4& y) {   // raw bf16 q of both heads of the pair (tok = record word 6)
    const bf16* qp = p.q + (size_t)(tok * kQHeads + 2 * (a.y & 15)) * kDim + l16 * 8;
    x = ldg_cg4(qp); y = ldg_cg4(qp + kDim);
  };
  constexpr bool kPre = (SKV_LDG_DB != 0);                     // single-buffered LDG loads inside the step loop: nothing to issue ahead
  // Register budget (LDG engine at 128 regs): only the NEXT item's first record half (4 words) and its tok0 are carried through an item; the current item's
  // second record half (part_slot, S) is re-read from the record (L1 hit, issued before the merge) when the merge needs it.
  int4 ra = rec[0];
  uint4 qa, qb;
  if (SKV_ITEM_PREFETCH) {                                     // (prefetch 0: every item loads its own record, q and first K / V at the top of its iteration)
    load_q(ra, rec[1].z, qa, qb);
    if (!TMA && kPre) ldg_first(ra);
  }

  for (int k = 0; k < ni; ++k) {
    if (!SKV_ITEM_PREFETCH) {
      ra = rec[2 * k];
      load_q(ra, rec[2 * k + 1].z, qa, qb);
      if (!TMA && kPre) ldg_first(ra);
    }
    const int kvh = ra.y & 15, p0 = ra.z, p1 = ra.w;
    int4 na = zero4;
    int ntok = 0;
    uint4 nqa = make_uint4(0, 0, 0, 0), nqb = nqa;
    if (SKV_ITEM_PREFETCH && k + 1 < ni) { na = rec[2 * k + 2]; ntok = rec[2 * k + 3].z; }
    float q0[8], q1[8];
    unpack8(qa, q0); unpack8(qb, q1);
#pragma unroll
    for (int j = 0; j < 8; ++j) { q0[j] *= kQScale; q1[j] *= kQScale; }
    St st;
    st_init(st);
    const int len = p1 - p0;
    if (!TMA) {
      // ---- LDG engine: slot r takes positions p0 + r + 32 j; 4 positions per step -> 8 LDG.128 in flight
      const size_t ioff = item_off(ra);
      const int nsteps = (len + 127) >> 7;
      auto load_k = [&](int i, uint4 (&kk)[4]) {
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int pos = p0 + r + 32 * (4 * i + u);
          kk[u] = pos < p1 ? ldg_nc((const char*)p.kc + ioff + (size_t)pos * 2048) : make_uint4(0, 0, 0, 0);
        }
      };
      auto load_v = [&](int i, uint4 (&vv)[4]) {
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int pos = p0 + r + 32 * (4 * i + u);
          vv[u] = pos < p1 ? ldg_nc((const char*)p.vc + ioff + (size_t)pos * 2048) : make_uint4(0, 0, 0, 0);
        }
      };
      auto valid_of = [&](int i, bool (&valid)[4]) {
#pragma unroll
        for (int u = 0; u < 4; ++u) valid[u] = (p0 + r + 32 * (4 * i + u)) < p1;
      };
#if SKV_LDG_DB == 1
      // full register double buffer: step i+1's 8 loads are issued before step i is computed (64 regs of K/V); step 0 was issued ahead (ldg_first)
      uint4 kb2[4], vb2[4];
      for (int i = 0; i < nsteps; i += 2) {
        if (i + 1 < nsteps) { load_k(i + 1, kb2); load_v(i + 1, vb2); }
        { bool valid[4]; valid_of(i, valid); step<4>(st, q0, q1, ka, va, valid); }
        if (i + 1 < nsteps) {
          if (i + 2 < nsteps) { load_k(i + 2, ka); load_v(i + 2, va); }
          bool valid[4]; valid_of(i + 1, valid); step<4>(st, q0, q1, kb2, vb2, valid);
        }
      }
#elif SKV_LDG_DB == 2
      // half buffer: step i+1's K and V loads are issued right after step i's scores (K regs dead), before its softmax / PV (48 regs of K/V)
      uint4 vb2[4];
      for (int i = 0; i < nsteps; i += 2) {
        float s0[4], s1[4];
        bool valid[4];
        valid_of(i, valid); step_score<4>(q0, q1, ka, valid, s0, s1);
        if (i + 1 < nsteps) { load_k(i + 1, ka); load_v(i + 1, vb2); }
        step_acc<4>(st, s0, s1, va);
        if (i + 1 < nsteps) {
          valid_of(i + 1, valid); step_score<4>(q0, q1, ka, valid, s0, s1);
          if (i + 2 < nsteps) { load_k(i + 2, ka); load_v(i + 2, va); }
          step_acc<4>(st, s0, s1, vb2);
        }
      }
#else
      for (int i = 0; i < nsteps; ++i) {      // single-buffered: all 8 loads issued before use
        load_k(i, ka); load_v(i, va);
        bool valid[4]; valid_of(i, valid); step<4>(st, q0, q1, ka, va, valid);
      }
#endif
      // K / V registers are dead: item k+1's first loads (and its q) go out before this item's merge
      if (SKV_ITEM_PREFETCH && k + 1 < ni) { load_q(na, ntok, nqa, nqb); if (kPre) ldg_first(na); }
    } else {
      // ---- TMA engine: stage = kStageRows positions (K, V); row slot r takes positions r + 32 u, u < kU, of the stage
      const int ns = (len + kStageRows - 1) / kStageRows;
      for (int j = 0; j < ns; ++j, ++g) {
        const int slot = g % kStages;
        const unsigned par = (g / kStages) & 1;
#ifdef SKV_DEBUG_WAIT
        mb_wait_dbg(bar0 + 8 * slot, par, layer, g, k, j);
#else
        mb_wait(bar0 + 8 * slot, par);
#endif
        const int n = min(kStageRows, len - j * kStageRows);
        uint4 kk[kU], vv[kU];
        bool valid[kU];
        const unsigned sa = sbase + kRingOff + slot * kStageBytes + r * 256 + l16 * 16;
        unsigned dep = 0;
#pragma unroll
        for (int u = 0; u < kU; ++u) {
          valid[u] = (r + 32 * u) < n;
          if (valid[u]) { kk[u] = lds128(sa + u * 8192); vv[u] = lds128(sa + kStageRows * 256 + u * 8192); }
          else { kk[u] = make_uint4(0, 0, 0, 0); vv[u] = make_uint4(0, 0, 0, 0); }
          dep ^= kk[u].x ^ vv[u].w;
        }
#if SKV_TMA_RELEASE
        // slot release without a CTA barrier: __syncwarp orders the other lanes' shared loads before lane 0's arrive, whose .release.cta semantics
        // order them before the producer's acquire (try_wait) and refill
        __syncwarp();
        if (lane == 0) mb_arrive_dep(bar0 + 8 * (kStages + slot), dep);
#endif
        step<kU>(st, q0, q1, kk, vv, valid);
#if SKV_TMA_RELEASE
        if (tid == 0) { mb_wait(bar0 + 8 * (kStages + slot), par); produce(slot); }   // all 16 warps have read stage g: slot is free for stage g + kStages
        __syncwarp();
#else
        __syncthreads();                                         // every thread is done with the slot
        if (tid == 0) produce(slot);
#endif
      }
    }
    if (TMA && SKV_ITEM_PREFETCH && k + 1 < ni) load_q(na, ntok, nqa, nqb);
    const int4 rb = rec[2 * k + 1];                             // {part_slot, S, tok0, n_items}
    // ---- end of item: xor-16 merge inside the warp, 16 warp states through smem, partial / normalised store
    {
      float m, l, o[8], m2, l2, o2[8];
      m = st.m0; l = st.l0; m2 = __shfl_xor_sync(0xffffffffu, m, 16); l2 = __shfl_xor_sync(0xffffffffu, l, 16);
#pragma unroll
      for (int j = 0; j < 8; ++j) { o[j] = st.o0[j]; o2[j] = __shfl_xor_sync(0xffffffffu, o[j], 16); }
      merge_state(m, l, o, m2, l2, o2);
      float* d0 = mrg + (warp * 2 + 0) * kPartStride;
      if (lane < 16) {
        *reinterpret_cast<float4*>(d0 + l16 * 8) = make_float4(o[0], o[1], o[2], o[3]);
        *reinterpret_cast<float4*>(d0 + l16 * 8 + 4) = make_float4(o[4], o[5], o[6], o[7]);
        if (lane == 0) { d0[128] = m; d0[129] = l; }
      }
      m = st.m1; l = st.l1; m2 = __shfl_xor_sync(0xffffffffu, m, 16); l2 = __shfl_xor_sync(0xffffffffu, l, 16);
#pragma unroll
      for (int j = 0; j < 8; ++j) { o[j] = st.o1[j]; o2[j] = __shfl_xor_sync(0xffffffffu, o[j], 16); }
      merge_state(m, l, o, m2, l2, o2);
      float* d1 = mrg + (warp * 2 + 1) * kPartStride;
      if (lane < 16) {
        *reinterpret_cast<float4*>(d1 + l16 * 8) = make_float4(o[0], o[1], o[2], o[3]);
        *reinterpret_cast<float4*>(d1 + l16 * 8 + 4) = make_float4(o[4], o[5], o[6], o[7]);
        if (lane == 0) { d1[128] = m; d1[129] = l; }
      }
    }
    __syncthreads();
    if (tid < 256) {
      const int hq = tid >> 7, d = tid & 127;
      const float* b = mrg + hq * kPartStride;
      float M = neg_inf();
#pragma unroll
      for (int w = 0; w < 16; ++w) M = fmaxf(M, b[w * 2 * kPartStride + 128]);
      const float Mc = M == neg_inf() ? 0.f : M;
      float L = 0.f, O = 0.f;
#pragma unroll
      for (int w = 0; w < 16; ++w) {
        const float a = ex2(b[w * 2 * kPartStride + 128] - Mc);
        L = fmaf(a, b[w * 2 * kPartStride + 129], L);
        O = fmaf(a, b[w * 2 * kPartStride + d], O);
      }
      if (rb.y == 1) {
        p.ao[(size_t)rb.z * (kQHeads * kDim) + (2 * kvh + hq) * kDim + d] = f2bf(O / L);
      } else {
        float* dst = p.part + ((size_t)rb.x * 2 + hq) * kPartStride;     // rb.x = part_base + split
        __stcg(dst + d, O);
        if (d == 0) { __stcg(dst + 128, M); __stcg(dst + 129, L); }
      }
    }
    __syncthreads();                                            // merge buffer free for the next item
    if (SKV_ITEM_PREFETCH) { ra = na; qa = nqa; qb = nqb; }
  }
}

// All 512 threads. Runs the CTA's items one after another. ring_phase: bit 31 = mbarriers initialised, low bits = TMA stages consumed so far;
// start every CTA with ring_phase = 0 and pass the same variable to every layer's call.
// smem: attend_smem_bytes() bytes, 128 B aligned, owned by attend during the call. Between calls the caller may use bytes [0, kBarOff)
// (TMA ring + merge buffer; e.g. GEMM stages or the combine_head xs slot) but must keep the last kBarBytes bytes (the mbarriers) unchanged.
// Ends with __syncthreads (CTAs with no items return early without it).
template <int TNT>
__device__ void attend(const Params& p, int layer, unsigned char* smem, unsigned& ring_phase) {
  static_assert(TNT == kBlock, "skv::attend is written for 512 threads");
  const int tid = threadIdx.x;
  const unsigned bar0 = smaddr(smem) + kBarOff;
  const bool tma = SKV_FORCE_ENGINE < 0 ? (p.engine != 0) : (SKV_FORCE_ENGINE != 0);
  // mbarriers (last kBarBytes of the smem: full[s] at +8 s, empty[s] at +8 (kStages + s)) are initialised once per kernel (first TMA call) and then
  // carried with their phase in ring_phase; re-initialising them on every call hung the TMA ring (P2 debug, job 9599). The fence orders the caller's
  // generic-proxy use of the ring bytes between calls before this call's TMA (async-proxy) writes.
  if (tma && !(ring_phase >> 31)) {
    if (tid == 0) {
      for (int s = 0; s < kStages; ++s) { mb_init(bar0 + 8 * s, 1); mb_init(bar0 + 8 * (kStages + s), 16); }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    ring_phase |= 0x80000000u;
  }
  if (tma) { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); __syncthreads(); }
  unsigned g = ring_phase & 0x7fffffffu;
  const int4* rec = p.items + (size_t)blockIdx.x * p.items_stride * 2;
  const int ni = rec[1].w;                                      // n_items is stored in every record of the CTA
  if (ni == 0) return;
  if (tma) run_items<true>(p, layer, smem, g, rec, ni);
  else run_items<false>(p, layer, smem, g, rec, ni);
  ring_phase = (ring_phase & 0x80000000u) | (g & 0x7fffffffu);
}

// ------------------------------------------------------------------------------------------------ combine
// One warp: normalised attention output of (token tok, q-head head) for this lane's dims lane*4 .. +3, merged from the S > 1 partials in ONE pass:
// the o loads (kMergeChunk float4 per lane, issued with the m/l loads) are in flight together; lanes 0..S-1 hold m/l, warp max, weights by shfl.
__device__ __forceinline__ float4 merge_th(const Params& p, int tok, int head, int lane) {
  const int4 pi = *reinterpret_cast<const int4*>(p.pair_info + (tok * kKvHeads + (head >> 1)) * 4);
  const int S = pi.y;
  const float* base = p.part + ((size_t)pi.x * 2 + (head & 1)) * kPartStride;     // split s at base + s * 2 * kPartStride
  constexpr int SS = 2 * kPartStride;
  float4 x[kMergeChunk];
#pragma unroll
  for (int c = 0; c < kMergeChunk; ++c) x[c] = c < S ? __ldcg(reinterpret_cast<const float4*>(base + c * SS) + lane) : make_float4(0.f, 0.f, 0.f, 0.f);
  float2 ml = lane < S ? __ldcg(reinterpret_cast<const float2*>(base + lane * SS + 128)) : make_float2(neg_inf(), 0.f);
  float M = ml.x;
#pragma unroll
  for (int o = 16; o; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
  const float w = lane < S ? ex2(ml.x - M) : 0.f;
  float L = w * ml.y;
#pragma unroll
  for (int o = 16; o; o >>= 1) L += __shfl_xor_sync(0xffffffffu, L, o);
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int c = 0; c < kMergeChunk; ++c) {
    if (c < S) {
      const float wc = __shfl_sync(0xffffffffu, w, c);
      a.x = fmaf(wc, x[c].x, a.x); a.y = fmaf(wc, x[c].y, a.y); a.z = fmaf(wc, x[c].z, a.z); a.w = fmaf(wc, x[c].w, a.w);
    }
  }
  for (int s0 = kMergeChunk; s0 < S; s0 += kMergeChunk) {          // S > kMergeChunk (rare): further rounds
    float4 y[kMergeChunk];
#pragma unroll
    for (int c = 0; c < kMergeChunk; ++c) y[c] = s0 + c < S ? __ldcg(reinterpret_cast<const float4*>(base + (s0 + c) * SS) + lane) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int c = 0; c < kMergeChunk; ++c) {
      if (s0 + c < S) {
        const float wc = __shfl_sync(0xffffffffu, w, s0 + c);
        a.x = fmaf(wc, y[c].x, a.x); a.y = fmaf(wc, y[c].y, a.y); a.z = fmaf(wc, y[c].z, a.z); a.w = fmaf(wc, y[c].w, a.w);
      }
    }
  }
  const float inv = 1.f / L;
  return make_float4(a.x * inv, a.y * inv, a.z * inv, a.w * inv);
}
// S pieces of a pair can exceed kMergeChunk only up to 32 (planner smax); lanes >= S must not be read as partials (guarded above).

// All 512 threads: q-head `head`, tokens 0..ntok-1 -> bf16 into the SW128 K-major B slot xs (2 * TPAD * 128 bytes), rows ntok..TPAD-1 zero.
// element (t, k) -> xs + (k/64) * TPAD * 128 + t * 128 + ((((k%64)/8) ^ (t&7)) << 4), position k%8 in the chunk. One warp per token row.
// No __syncthreads inside; ends with fence.proxy.async (generic -> async proxy) so the caller only needs its own __syncthreads before the MMA.
template <int TPAD, int TNT>
__device__ void combine_head(const Params& p, int layer, int head, unsigned char* xs, int ntok) {
  static_assert(TNT == kBlock && TPAD % 16 == 0, "combine_head: 512 threads, TPAD multiple of 16");
  (void)layer;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int c = lane >> 1;                                         // 16-byte chunk (8 dims) of the row; lanes 2c, 2c+1 hold its two halves
  for (int t = warp; t < TPAD; t += 16) {
    uint2 v = make_uint2(0u, 0u);
    if (t < ntok) {
      const int4 pi = *reinterpret_cast<const int4*>(p.pair_info + (t * kKvHeads + (head >> 1)) * 4);
      if (pi.y == 1) {
        v = __ldcg(reinterpret_cast<const uint2*>(p.ao + (size_t)t * (kQHeads * kDim) + head * kDim) + lane);
      } else {
        const float4 o = merge_th(p, t, head, lane);
        v.x = pack2(o.x, o.y); v.y = pack2(o.z, o.w);
      }
    }
    const int off = (c >> 3) * (TPAD * 128) + t * 128 + ((((c & 7) ^ (t & 7))) << 4) + (lane & 1) * 8;
    *reinterpret_cast<uint2*>(xs + off) = v;
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// Fallback / tests: grid-strided over (token, q-head) pairs with S > 1 (S == 1 pairs were written by attend); writes ao.
__device__ void combine_all_to_ao(const Params& p, int layer, int cta, int ncta) {
  (void)layer;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int total = p.total_q * kQHeads;
  for (int idx = cta * 16 + warp; idx < total; idx += ncta * 16) {
    const int tok = idx >> 4, head = idx & 15;
    const int4 pi = *reinterpret_cast<const int4*>(p.pair_info + (tok * kKvHeads + (head >> 1)) * 4);
    if (pi.y == 1) continue;
    const float4 o = merge_th(p, tok, head, lane);
    uint2 v; v.x = pack2(o.x, o.y); v.y = pack2(o.z, o.w);
    *reinterpret_cast<uint2*>(p.ao + (size_t)tok * (kQHeads * kDim) + head * kDim + lane * 4) = v;
  }
}

}  // namespace skv
#endif  // SKV_CUH_
