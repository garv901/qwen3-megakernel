// A1 v1: split-KV decode attention (q_len == 1 per sequence) + one-pass combine, for the bespoke megakernel `mk` (MK-attn owns this file).
// Packaging: the consumer prepends this text to its NVRTC source. No includes, no global macro dependencies: model sizes are constexpr in
// namespace skv (named kXxx so they cannot collide with a global macro H/NL/QH/KVH/HD/NT/...), every macro starts with SKV_.
// Contract: regimebench/kernels/common/README.md + splitkv/CHANGELOG.md (interface v1; v2 = -DSKV_IFACE=2, attend does q/k norm + RoPE + the new-row KV write). Planner: splitkv/plan.py (item encoding documented there).
//
// Math domain: q is pre-scaled by (1/sqrt(128)) * log2(e), so scores live in log2 units and every exp is a bare ex2.approx.
// Partial slots hold (o unnormalised w.r.t. m, m, l) in that domain; only combine_head / combine_all_to_ao read them.
// Memory notes: the LDG engine reads KV rows with a coherent ld.global.L1::no_allocate (the new token's row is written by the caller earlier
// in the same kernel, so .nc would be outside the PTX contract; SKV_KV_NC=1 selects it, same speed). TMA reads go through L2.
// q / ao / partials are read with ld.global.cg (L1 bypass).
#ifndef SKV_CUH_
#define SKV_CUH_

#define SKV_IFACE_MAX 2   // highest interface version this header implements (consumers static_assert it)
#ifndef SKV_IFACE
#define SKV_IFACE 1       // 1 = q is bf16 post-norm / RoPE (default); 2 = attend does q/k RMS-norm + RoPE + the new-row KV write from the fp32 QKV sums (SPEC-A1v2)
#endif
#if SKV_IFACE != 1 && SKV_IFACE != 2
#error "SKV_IFACE must be 1 or 2"
#endif

#ifndef SKV_LDG_DB
#define SKV_LDG_DB 2   // LDG engine: 1 = register double buffer (step i+1's loads before step i), 2 = half buffer (after step i's scores), 0 = single
#endif

#ifndef SKV_KV_NC
#define SKV_KV_NC 0    // LDG engine KV loads: 0 = coherent weak load (default; = nc within 0.05 us, P2 job 9601), 1 = ld.global.nc (outside PTX contract)
#endif

#ifndef SKV_FORCE_ENGINE
#define SKV_FORCE_ENGINE 1      // 1 (default, P8 job 9708): compile only the TMA engine (smaller code, 120-122 regs; TMA-only <= both + 0.08 us at 17 points).
                                // 0: only the LDG engine (best at b1 kv <= 2K by 0.1-0.2, b256 kv128 by 1.4). -1: both, runtime Params::engine
#endif

#ifndef SKV_QMAX
#define SKV_QMAX 1          // max q_len per sequence. 1 = the q_len == 1 consumers (default build, unchanged). 2..16 = R6 multi-token causal decode: the two-phase mma.sync consumer
                            // (TMA engine only, interface v1, SKV_TMA_MMA 1, 64-position stages; the planner puts q_len > 1 on the TMA engine). mk's R6 bucket builds with -DSKV_QMAX=16
#endif
#if SKV_QMAX < 1 || SKV_QMAX > 16
#error "SKV_QMAX must be 1..16"
#endif
#ifndef SKV_STAGE_ROWS
#if SKV_QMAX > 1
#define SKV_STAGE_ROWS 64   // R6 consumer: 4 position tiles of 16 per stage
#else
#define SKV_STAGE_ROWS 128  // TMA engine: positions per ring stage (32 / 64 / 128; K and V are rows * 256 B each). v1 = 32; 128 x 2 best in P2b (job 9610)
#endif
#endif
#ifndef SKV_STAGES
#if SKV_QMAX > 1
#define SKV_STAGES 4        // R6: 4 x 64 rows = the bytes of the default 2 x 128 ring (same attend_smem_bytes), finer-grained refills
#else
#define SKV_STAGES 2        // TMA engine: ring stages (2..8). Ring bytes = stages * rows * 512. v1 = 4
#endif
#endif
#ifndef SKV_TMA_RELEASE
#define SKV_TMA_RELEASE 1   // 0 = v1 (__syncthreads after every stage, then thread 0 refills); 1 = per-slot "empty" mbarrier (16 warp arrivals), no CTA barrier in the stage loop
#endif
#ifndef SKV_PROF_NOCOMP
#define SKV_PROF_NOCOMP 0   // 1 = profiling only: no dots / softmax / PV, loaded K/V words are xored into st.o0[0]; results are wrong by design
#endif
#ifndef SKV_TMA_MMA
#define SKV_TMA_MMA 1       // (default 1: faster than FFMA at every point, jobs 9657, 9658) 0 = FFMA consumer; 1 = TMA engine consumer on mma.sync (bf16 m16n8k16, ldmatrix from 128B-swizzled TMA boxes, movmatrix); host maps must be SWIZZLE_128B, box 64 x 1 x rows (CHANGELOG)
#endif
#ifndef SKV_MERGE_CHUNK
#define SKV_MERGE_CHUNK 18  // (default 18: 20 spills with both engines + SKV_TMA_MMA) combine: partial float4 loads in flight per lane (80 registers at 20). 18 frees 8 registers for builds that spill with both engines + SKV_TMA_MMA (S > 18 then needs a second round)
#endif
#ifndef SKV_COMBINE_HALF
#define SKV_COMBINE_HALF 0  // (default 0: job 9673, +0.13..0.29 us C at b1 S 19 but -0.35 at b1kv128 S 16, -0.06..0.11 at b8) 1 = combine_head with ntok <= 4: rows with S > SKV_MERGE_CHUNK use two warps (merge_th2, float2: one load round for S <= 20); 0 = one warp per row
#endif
#ifndef SKV_TICKET
#define SKV_TICKET 0        // (default 0) 1 = deferred last-arriver merge inside attend: ao is complete after attend + the caller's grid barrier (no combine_head / merge_th call). Needs Params::tickets (CHANGELOG)
#endif
#ifndef SKV_ITEM_PREFETCH
#if SKV_IFACE == 1 && SKV_QMAX == 1
#define SKV_ITEM_PREFETCH 1 // (default 1 since P9b job 9711: b256 kv128 -5.2, b128 -1.2, b32 kv1K -0.4 us/layer; b1 +0.05..+0.08) item k+1's record + q are loaded during item k
#else                       // (and the LDG engine issues its first K/V loads before item k's merge)
#define SKV_ITEM_PREFETCH 0 // iface 2 and the R6 consumer do not support it
#endif
#endif

#ifndef SKV_KV_EVICT_FIRST
#define SKV_KV_EVICT_FIRST 0 // (default 0, opt-in for MK-core's A/B) 1 = every KV load (TMA boxes of both consumers, LDG engine) carries an L2 evict_first cache policy:
#endif                       // KV is read once per step, so it should not displace the caller's weights / activations in L2
#ifndef SKV_TAIL_LDG
#define SKV_TAIL_LDG 0      // (default 0, opt-in; P9d) 1 = q_len 1 TMA mma path: an item whose length is == 1 (mod kStageRows) (and > 1) streams only its first len - 1 positions by TMA
#endif                      // (whole stages); the last position is read by LDG (warp 0) and merged as a parked single-position state, as the iface 2 new row
#if SKV_TAIL_LDG && !(SKV_IFACE == 1 && SKV_QMAX == 1 && SKV_TMA_MMA)
#error "SKV_TAIL_LDG needs SKV_IFACE 1, SKV_QMAX 1, SKV_TMA_MMA 1"
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
constexpr int kMergeChunk = SKV_MERGE_CHUNK;                   // combine: partial float4 loads in flight per lane
constexpr float kQScale = 0.08838834764831845f * 1.4426950408889634f;   // 1/sqrt(128) * log2(e)
constexpr int kRingOff = 0, kMrgOff = kStages * kStageBytes, kMrgBytes = 16 * 2 * kPartStride * 4;
constexpr int kBarBytes = (16 * kStages > 64 ? 16 * kStages : 64);   // mbarriers: full[s] at +8 s, empty[s] at +8 (kStages + s)
constexpr int kBarOff = kMrgOff + kMrgBytes, kSmemBytes = kBarOff + kBarBytes;

constexpr bool kMmaOn = (SKV_TMA_MMA != 0);
// mma consumer geometry: warp w takes 16-position tile (w % kMmaT) of the stage and the dim slice dh = w / kMmaT (kMmaDW dims = kMmaMt m-tiles of 16)
constexpr int kMmaT = kStageRows / 16, kMmaWPT = 16 / kMmaT, kMmaDW = 128 / kMmaWPT, kMmaMt = kMmaDW / 16, kMmaMS = kMmaDW + 4;   // kMmaMS: floats of one (warp, q-head) merge slot: O[kMmaDW], m, l, pad
constexpr int kMmaNew = 32 * kMmaMS, kMmaQs = kMmaNew + 132;   // merge buffer (float offsets): 32 slots | new-row state (v[128], m[2], pad) | per-item q scratch (2 x 128 bf16 = 128 floats)
constexpr int kMmaG = 1;                        // V m-tiles loaded / multiplied together
constexpr int kHalfBytes = kStageRows * 128;                   // one 64-dim half of K or V in a stage (SKV_TMA_MMA: [rows][64] bf16, 128 B swizzled rows)
static_assert(!kMmaOn || (kStageRows >= 32 && kMmaDW % 16 == 0 && (kMmaQs + 128) * 4 <= kMrgBytes), "SKV_TMA_MMA geometry");

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
#if SKV_TICKET
  unsigned* tickets;               // [n_pairs = 8 * n_seq] u32, zeroed once per plan, never reset: tickets[seq * 8 + kvh] gains S per layer (SKV_TICKET)
#endif
#if SKV_IFACE == 2
  const float* qkv;                // [256][4096] fp32 row t = token t: q head h at h*128, k head g at 2048 + g*128, v head g at 3072 + g*128; NOT rms-scaled
  const float* rr1;                // [256] fp32 per-token rms scale (q, k, v = rr1[t] * qkv)
  const bf16* qn; const bf16* kn;  // [n_layers][128] bf16 q-norm / k-norm weights (layer l at l * 128)
  const float* rope;               // [max_pos][128] fp32: row p = cos(p f_j) j < 64, then sin(p f_j)
#endif
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
__device__ __forceinline__ unsigned long long kv_policy() {   // SKV_KV_EVICT_FIRST: L2 evict_first policy for KV loads (non-volatile: the compiler may hoist / CSE it)
  unsigned long long pol;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ uint4 ldg_nc(const void* p) {   // KV stream load; SKV_KV_NC 1: non-coherent (LDG...CONSTANT), 0: coherent weak load
  uint4 v;
#if SKV_KV_EVICT_FIRST
  const unsigned long long pol = kv_policy();
#if SKV_KV_NC
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(pol));
#else
  asm volatile("ld.global.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(pol));
#endif
#elif SKV_KV_NC
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
#if SKV_KV_EVICT_FIRST
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;"
               ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(b), "l"(kv_policy()) : "memory");
#else
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
               ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(b) : "memory");
#endif
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
#if SKV_IFACE == 2
__device__ __forceinline__ float bfround(float x) { return __uint_as_float((unsigned)f2bf(x) << 16); }
__device__ __forceinline__ uint4 pack8(const float (&f)[8]) { return make_uint4(pack2(f[0], f[1]), pack2(f[2], f[3]), pack2(f[4], f[5]), pack2(f[6], f[7])); }
__device__ __forceinline__ void ldg_f8(const float* p, float (&f)[8]) {   // 2 x 16 B, L1 bypass (qkv is written earlier in the same kernel by other CTAs)
  const float4 a = __ldcg(reinterpret_cast<const float4*>(p)), b = __ldcg(reinterpret_cast<const float4*>(p) + 1);
  f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w; f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}
// RMS-norm + weights + RoPE rotate-half of one head, 8 dims per lane (lane16 holds d = 8 lane16 .. +7; the RoPE partner d ^ 64 sits in lane16 ^ 8).
// Norm sum order: e_j = x_j * rr; ss = fma chain over j = 0..7 (per lane), then xor-butterfly 8, 4, 2, 1 over the half-warp. Full-warp mask: all 32 lanes call it.
__device__ __forceinline__ void norm_rope8(const float (&x)[8], float rr, const uint4& wb, const float (&cs)[8], const float (&sn)[8], bool lo, float (&o)[8]) {
  float e[8], ss = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) { e[j] = x[j] * rr; ss = fmaf(e[j], e[j], ss); }
#pragma unroll
  for (int d = 8; d; d >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, d);
  const float rn = rsqrtf(ss * 0.0078125f + 1e-6f);
  float w[8];
  unpack8(wb, w);                                              // bf16 weights stay packed (4 regs) until the norm sum is done
#pragma unroll
  for (int j = 0; j < 8; ++j) e[j] = e[j] * rn * w[j];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float pj = __shfl_xor_sync(0xffffffffu, e[j], 8);
    o[j] = e[j] * cs[j] + (lo ? -pj : pj) * sn[j];
  }
}
#endif

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

// ---- mma consumer helpers (SKV_TMA_MMA). Shared-memory layout of a stage: K lo | K hi | V lo | V hi, each [rows][64] bf16 = rows * 128 B, 1024 B aligned;
// 16 B chunk c (8 dims) of row r of a half sits at r * 128 + ((c ^ (r & 7)) << 4) (TMA SWIZZLE_128B). The offset functions are __host__ __device__ constexpr so
// tests/test_splitkv_plan.py can evaluate the real address math on the CPU and check it against a numpy model of ldmatrix / movmatrix / mma.
__host__ __device__ constexpr unsigned sw128_off(unsigned row, unsigned chunk) { return row * 128u + ((chunk ^ (row & 7u)) << 4); }
// ldmatrix.x4 (non-trans) row address of `lane` for the K A fragment (16 positions x 16 dims) of k-step ks: matrix mi = lane / 8: a0 = (rows 0-7, dims 0-7), a1 = (rows 8-15, dims 0-7),
// a2 = (rows 0-7, dims 8-15), a3 = (rows 8-15, dims 8-15); lane % 8 selects the row inside the matrix. row0 = first position of the tile (multiple of 16).
__host__ __device__ constexpr unsigned mma_k_off(unsigned lane, unsigned row0, unsigned ks, unsigned half_bytes) {
  return (((ks * 2u + (lane >> 4)) >> 3) * half_bytes) + sw128_off(row0 + ((lane >> 3) & 1u) * 8u + (lane & 7u), (ks * 2u + (lane >> 4)) & 7u);
}
// ldmatrix.x4.trans row address of `lane` for the V^T A fragment (16 dims dim0.. x 16 positions): a0 = (dims 0-7, pos 0-7), a1 = (dims 8-15, pos 0-7), a2 = (dims 0-7, pos 8-15),
// a3 = (dims 8-15, pos 8-15); the stored matrices are [pos][8 dims] rows. Offset includes the 2 * half_bytes of V.
__host__ __device__ constexpr unsigned mma_v_off(unsigned lane, unsigned row0, unsigned dim0, unsigned half_bytes) {
  return (2u + ((dim0 / 8u + ((lane >> 3) & 1u)) >> 3)) * half_bytes + sw128_off(row0 + (lane >> 4) * 8u + (lane & 7u), (dim0 / 8u + ((lane >> 3) & 1u)) & 7u);
}
// Cheap forms of the same addresses (what the stage loop uses: 2 per-lane base registers + immediates / one LOP per k-step; written out in the CPU test against the plain forms above).
// With i = lane % 8, j32 = (i / 2) << 5: (c ^ i) << 4 for c = 2 u + v is ((u ^ i / 2) << 5) + ((v ^ i % 2) << 4).
__host__ __device__ constexpr unsigned mma_j32(unsigned lane) { return ((lane & 7u) >> 1) << 5; }
__host__ __device__ constexpr unsigned mma_k_base(unsigned lane, unsigned row0) { return (row0 + ((lane >> 3) & 1u) * 8u + (lane & 7u)) * 128u + (((lane >> 4) ^ (lane & 1u)) << 4); }
__host__ __device__ constexpr unsigned mma_k_term(unsigned ks, unsigned j32, unsigned half_bytes) { return (((ks & 3u) << 5) ^ j32) + (ks >> 2) * half_bytes; }
__host__ __device__ constexpr unsigned mma_v_base(unsigned lane, unsigned row0) { return (row0 + (lane >> 4) * 8u + (lane & 7u)) * 128u + ((((lane >> 3) & 1u) ^ (lane & 1u)) << 4); }
__host__ __device__ constexpr unsigned mma_v_term(unsigned y, unsigned j32, unsigned half_bytes) { return (2u + (y >> 2)) * half_bytes + (((y & 3u) << 5) ^ j32); }   // y = dim0 / 16
// Q^T B fragment in the per-item q scratch (head g at g * 256 B): thread (g = lane / 4, c = lane % 4), k-step ks reads one 8 B pair {b0, b1} = bf16 pairs of dims ks * 16 + 2c, +1 and
// ks * 16 + 8 + 2c, +1. Scratch word index inside a head = ks * 8 + c * 2 + h; the word of 16 B chunk j = 2 ks + h (dims 8 j .. 8 j + 7), pair c, is stored at that index (mma_q_word).
__host__ __device__ constexpr unsigned mma_q_off(unsigned lane, unsigned ks) { return (lane >> 2) * 256u + ks * 32u + (lane & 3u) * 8u; }
__host__ __device__ constexpr unsigned mma_q_word(unsigned chunk, unsigned c) { return (chunk >> 1) * 8u + c * 2u + (chunk & 1u); }
// ---- R6 (SKV_QMAX > 1) address functions (always defined; used by the R6 consumer and evaluated by tests/test_splitkv_plan.py).
// q staging [N8][128] bf16: row n = 256 B, 16 B chunk c (8 dims) of row n at n * 256 + (((c & 8) | ((c & 7) ^ (n & 7))) << 4): the 8 rows of an ldmatrix matrix hit 8 distinct bank groups.
// P^T staging [N8][64] bf16: row n = 128 B, 16 B chunk c (8 positions) at n * 128 + ((c ^ (n & 7)) << 4) (the TMA 128B swizzle pattern).
__host__ __device__ constexpr unsigned r6_q_off(unsigned n, unsigned c) { return n * 256u + (((c & 8u) | ((c & 7u) ^ (n & 7u))) << 4); }
__host__ __device__ constexpr unsigned r6_p_off(unsigned n, unsigned c) { return n * 128u + ((c ^ (n & 7u)) << 4); }
// ldmatrix.x4 (non-trans) row address of `lane` for the q B fragments of n-tile nt, k-step pair jj (matrix mi = lane / 8 = chunk 4 jj + mi; regs 0, 1 = k-step 2 jj {b0, b1}, regs 2, 3 = k-step 2 jj + 1)
__host__ __device__ constexpr unsigned r6_q_ld(unsigned lane, unsigned nt, unsigned jj) { return r6_q_off(nt * 8u + (lane & 7u), 4u * jj + (lane >> 3)); }
__host__ __device__ constexpr unsigned r6_q_base(unsigned lane, unsigned nt) { return (nt * 8u + (lane & 7u)) * 256u; }
__host__ __device__ constexpr unsigned r6_q_xv(unsigned lane) { return (((lane >> 3) ^ (lane & 7u)) << 4); }
__host__ __device__ constexpr unsigned r6_q_term(unsigned jj, unsigned xv) { return (jj >> 1) * 128u + (xv ^ ((jj & 1u) * 64u)); }
// ldmatrix row address of `lane` for the P^T B fragment of k-step ks (positions 16 ks .. +15 = chunks 2 ks, 2 ks + 1): matrix mi = lane / 8 = (n-tile nb0 + (ntb == 2 ? mi / 2 : 0), chunk 2 ks + (mi & 1)); regs (b0, b1) per n-tile
__host__ __device__ constexpr unsigned r6_p_ld(unsigned lane, unsigned nb0, unsigned ntb, unsigned ks) { return r6_p_off((nb0 + (ntb == 2u ? (lane >> 4) : 0u)) * 8u + (lane & 7u), 2u * ks + ((lane >> 3) & 1u)); }
__host__ __device__ constexpr unsigned r6_p_base(unsigned lane, unsigned nb0, unsigned ntb) { return ((nb0 + (ntb == 2u ? (lane >> 4) : 0u)) * 8u + (lane & 7u)) * 128u; }
__host__ __device__ constexpr unsigned r6_p_xp(unsigned lane) { return ((((lane >> 3) & 1u) ^ (lane & 7u)) << 4); }
__host__ __device__ constexpr unsigned r6_p_term(unsigned ks, unsigned xp) { return xp ^ (ks * 32u); }

__device__ __forceinline__ void ldsm4(unsigned (&r)[4], unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm4t(unsigned (&r)[4], unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ uint2 lds64(unsigned a) {
  uint2 v;
  asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ unsigned movm_t(unsigned x) {   // 8x8 b16 transpose across the warp (SASS MOVM.16.MT88)
  unsigned y;
  asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(y) : "r"(x));
  return y;
}
__device__ __forceinline__ unsigned pack_bf16x2(float lo, float hi) {   // RNE, lo in the low half
  unsigned d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
__device__ __forceinline__ void hmma16816(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm2(unsigned& r0, unsigned& r1, unsigned a) {   // x2: matrices 0, 1 (row addresses of lanes 0..15; the other lanes' addresses are ignored)
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(a) : "memory");
}
__device__ __forceinline__ void sts32(unsigned a, unsigned v) { asm volatile("st.shared.b32 [%0], %1;" ::"r"(a), "r"(v) : "memory"); }

#if SKV_TICKET
__device__ __forceinline__ unsigned ld_acquire_gpu(const unsigned* a) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ void red_release_gpu(unsigned* a) { asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(a) : "memory"); }
// pair = seq * 8 + kvh of a record's y word (q_len > 1 plans keep q_len - 1 in y bits 24..27)
__device__ __forceinline__ int ticket_pair(int y) { return (SKV_QMAX > 1 ? ((y >> 12) & 0xFFF) : (y >> 12)) * kKvHeads + (y & 15); }
__device__ __forceinline__ int rec_splits(const int4& rv) { return SKV_IFACE == 2 ? (rv.y & 255) : rv.y; }
#endif

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
  constexpr bool kMma = TMA && kMmaOn;
  constexpr bool kTail = (SKV_TAIL_LDG != 0) && kMma;           // P9d: 1-row item tails by LDG
#if SKV_TMA_MMA && (defined(SKV_DEBUG_WAIT) || defined(SKV_DEBUG_ALIGN))
  if (TMA && threadIdx.x == 0 && (sbase & 1023u)) { printf("SKV_TMA_MMA: smem base %u is not 1024 B aligned\n", sbase); __trap(); }
#endif

  // ---- TMA producer state (thread 0 only)
  int pk = -1, pp = 0, pend = 0, pkvh = 0;
  size_t prow = 0;
  int4 pnxt = zero4;
  auto produce = [&](int slot) {                               // issue the next stage of the CTA's flat stage sequence into `slot`
#if SKV_IFACE == 2
    while (pp >= pend) {                                       // (an empty range, only the owner of a 1-position piece, is skipped)
#else
    if (pp >= pend) {
#endif
      if (pk + 1 >= ni) return;
      ++pk;
      const int4 it = pnxt;
      pnxt = (pk + 1 < ni) ? rec[2 * (pk + 1)] : zero4;
      pp = it.z; pend = it.w; pkvh = it.y & 15; prow = layer_row + (size_t)it.x;
#if SKV_IFACE == 2
      if (it.w == rec[2 * pk + 1].w) --pend;                   // the piece ending at vis owns the new row: not written yet, the stream stops at vis - 1
#endif
      if (kTail && it.w - it.z > 1 && (it.w - it.z) % kStageRows == 1) --pend;   // P9d tail item (same rule as the consumer): the stream stops at p1 - 1, the last row is read by LDG
    }
    const int n = min(kStageRows, pend - pp);
    const unsigned dst = sbase + kRingOff + slot * kStageBytes, bar = bar0 + 8 * slot;
    mb_expect(bar, (unsigned)n * 512u);
    int off = 0, rc = (int)(prow + pp);
#pragma unroll
    for (int rs = 0; rs < kNumMaps; ++rs) {
      const int rows = kStageRows >> rs;
      if (n & rows) {
#if SKV_TMA_MMA
        const char* mk = (const char*)p.tmaps + (0 * kNumMaps + rs) * 128;   // swizzled maps, box 64 dims x 1 x rows: lo half (dims 0-63), hi half (64-127)
        const char* mv = (const char*)p.tmaps + (1 * kNumMaps + rs) * 128;
        tma3d(dst + off * 128, mk, 0, pkvh, rc, bar);
        tma3d(dst + kHalfBytes + off * 128, mk, 64, pkvh, rc, bar);
        tma3d(dst + 2 * kHalfBytes + off * 128, mv, 0, pkvh, rc, bar);
        tma3d(dst + 3 * kHalfBytes + off * 128, mv, 64, pkvh, rc, bar);
#else
        tma3d(dst + off * 256, (const char*)p.tmaps + (0 * kNumMaps + rs) * 128, 0, pkvh, rc, bar);
        tma3d(dst + kStageRows * 256 + off * 256, (const char*)p.tmaps + (1 * kNumMaps + rs) * 128, 0, pkvh, rc, bar);
#endif
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
  // P9d: the tail row (position p1 - 1) of item `a`, warp 0 only: lanes 0..15 = 16 B of the K row each, lanes 16..31 = V row (same address math as the LDG engine)
  auto load_tail = [&](const int4& a, uint4& t) {
    t = make_uint4(0, 0, 0, 0);
    if (warp == 0 && a.w - a.z > 1 && (a.w - a.z) % kStageRows == 1)
      t = ldg_nc((const char*)(lane < 16 ? p.kc : p.vc) + item_off(a) + (size_t)(a.w - 1) * 2048);
  };
  uint4 tra = make_uint4(0, 0, 0, 0), trb = tra;                // current item's / next item's tail row (kTail only)
  constexpr bool kPre = (SKV_LDG_DB != 0);                     // single-buffered LDG loads inside the step loop: nothing to issue ahead
  // Register budget (LDG engine at 128 regs): only the NEXT item's first record half (4 words) and its tok0 are carried through an item; the current item's
  // second record half (part_slot, S) is re-read from the record (L1 hit, issued before the merge) when the merge needs it.
  int4 ra = rec[0];
  uint4 qa, qb;
#if SKV_IFACE == 2
  static_assert(SKV_ITEM_PREFETCH == 0, "SKV_IFACE 2 does not support SKV_ITEM_PREFETCH");
#endif
  if (SKV_ITEM_PREFETCH) {                                     // (prefetch 0: every item loads its own record, q and first K / V at the top of its iteration)
    load_q(ra, rec[1].z, qa, qb);
    if (!TMA && kPre) ldg_first(ra);
    if (kTail) load_tail(ra, tra);
  }

  for (int k = 0; k < ni; ++k) {
#if SKV_IFACE == 2
    bool own;                                                  // this item ends at vis: it computes, uses and writes the new row (position vis - 1)
    int tok, pos_new;
    {
      ra = rec[2 * k];
      const int4 rv = rec[2 * k + 1];                           // {part_slot, S | n_items << 8, tok0, vis}
      tok = rv.z; pos_new = rv.w - 1;
      own = ra.w == rv.w;
      if (own) --ra.w;                                         // stream range [p0, vis - 1)
      if (!TMA && kPre) ldg_first(ra);
    }
#else
    if (!SKV_ITEM_PREFETCH) {
      ra = rec[2 * k];
      load_q(ra, rec[2 * k + 1].z, qa, qb);
      if (!TMA && kPre) ldg_first(ra);
      if (kTail) load_tail(ra, tra);
    }
#endif
    const int kvh = ra.y & 15, p0 = ra.z, p1 = ra.w;
    const bool tail = kTail && p1 - p0 > 1 && (p1 - p0) % kStageRows == 1;   // P9d (warp- and CTA-uniform): the last position is the LDG tail row
    int4 na = zero4;
    int ntok = 0;
    uint4 nqa = make_uint4(0, 0, 0, 0), nqb = nqa;
    if (SKV_ITEM_PREFETCH && k + 1 < ni) { na = rec[2 * k + 2]; ntok = rec[2 * k + 3].z; }
    float q0[8], q1[8];
    uint4 qpa = make_uint4(0, 0, 0, 0), qpb = qpa;                // mma consumer: raw bf16 q (dims l16 * 8 .. +7) of both heads, scale not applied
#if SKV_IFACE == 2
    uint4 vn1 = make_uint4(0, 0, 0, 0);                        // the new row's bf16 v and its two scores, in row slot 0's 16 lanes (warp 0), unused elsewhere
    float sn0 = 0.f, sn1 = 0.f;
    {
      // q prep: all loads first, then both heads; q = bf16(norm + RoPE) * kQScale. cs / sn = rope row pos_new, dims (lane16 & 7) * 8 .. +7 (the partner dims share them)
      const bool lo = l16 < 8;
      const float* qrow = p.qkv + (size_t)tok * 4096 + (2 * kvh) * kDim + l16 * 8;
      const float* rrow = p.rope + (size_t)pos_new * 128 + (l16 & 7) * 8;
      float x0[8], x1[8], cs[8], sn[8];
      ldg_f8(qrow, x0); ldg_f8(qrow + kDim, x1); ldg_f8(rrow, cs); ldg_f8(rrow + 64, sn);
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(p.qn + layer * kDim + l16 * 8));
      const float rr = __ldcg(p.rr1 + tok);
      norm_rope8(x0, rr, w, cs, sn, lo, q0);
      norm_rope8(x1, rr, w, cs, sn, lo, q1);
      if (kMma) { qpa = pack8(q0); qpb = pack8(q1); }              // == bf16 bits of bfround(q) (pack8 rounds RNE like bfround)
#pragma unroll
      for (int j = 0; j < 8; ++j) { q0[j] = bfround(q0[j]) * kQScale; q1[j] = bfround(q1[j]) * kQScale; }
      if (own && warp == 0) {                                  // new row (warp-uniform branch): lanes 0..15 = k (norm + RoPE), lanes 16..31 = v; each writes its 16 B of the cache row
        const bool kl = lane < 16;
        float xx[8], ko[8];
        ldg_f8(p.qkv + (size_t)tok * 4096 + (kl ? 2048 : 3072) + kvh * kDim + l16 * 8, xx);
        const uint4 kw = __ldcg(reinterpret_cast<const uint4*>(p.kn + layer * kDim + l16 * 8));
        norm_rope8(xx, rr, kw, cs, sn, lo, ko);                 // (the v lanes' norm result is ignored)
#pragma unroll
        for (int j = 0; j < 8; ++j) xx[j] = kl ? ko[j] : xx[j] * rr;
        const uint4 row = pack8(xx);
        const size_t o = (((layer_row + (size_t)ra.x + pos_new) * kKvHeads + kvh) * kDim + l16 * 8) * 2;
        *reinterpret_cast<uint4*>(const_cast<char*>(reinterpret_cast<const char*>(kl ? p.kc : p.vc)) + o) = row;
        vn1.x = __shfl_xor_sync(0xffffffffu, row.x, 16); vn1.y = __shfl_xor_sync(0xffffffffu, row.y, 16);   // slot 0 (lanes 0..15) gets v from lane + 16
        vn1.z = __shfl_xor_sync(0xffffffffu, row.z, 16); vn1.w = __shfl_xor_sync(0xffffffffu, row.w, 16);
        score2(kl ? row : make_uint4(0, 0, 0, 0), q0, q1, sn0, sn1);   // all 32 lanes shuffle; lanes >= 16 are ignored
        if (kMma && lane < 16) {                                 // mma consumer: park the new row's state (v, m = score per head, l = 1) for the end-of-item merge
          float vf[8];
          unpack8(vn1, vf);
          float* nb = mrg + kMmaNew;
          *reinterpret_cast<float4*>(nb + l16 * 8) = make_float4(vf[0], vf[1], vf[2], vf[3]);
          *reinterpret_cast<float4*>(nb + l16 * 8 + 4) = make_float4(vf[4], vf[5], vf[6], vf[7]);
          if (lane == 0) { nb[128] = sn0; nb[129] = sn1; }
        }
      }
    }
#else
    qpa = qa; qpb = qb;
    unpack8(qa, q0); unpack8(qb, q1);
#pragma unroll
    for (int j = 0; j < 8; ++j) { q0[j] *= kQScale; q1[j] *= kQScale; }
    if (tail && warp == 0) {
      // P9d: park the tail row's single-position state (v, m = score * kQScale per head, l = 1) for the end-of-item merge. Scores use the raw bf16 q and the fp32 dot
      // then * kQScale, exactly like the mma consumer (which scales the fp32 scores). Ordering: the previous item's merge reads of this slot (tid < 256, after its first
      // __syncthreads) are complete because every thread passed that item's second __syncthreads before reaching this point; this slot is read by the end-of-item merge
      // only after the first __syncthreads of THIS item, which follows these writes by every thread of warp 0. Exactly one merge: the slot is consumed once, by `tail` of this item.
      float qr0[8], qr1[8], kf[8], vf[8];
      unpack8(qa, qr0); unpack8(qb, qr1);
      unpack8(tra, kf);                                          // lanes < 16: K row dims l16 * 8 ..; lanes >= 16: V row (their dot is ignored)
      float d0 = 0.f, d1 = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) { d0 = fmaf(kf[j], qr0[j], d0); d1 = fmaf(kf[j], qr1[j], d1); }
#pragma unroll
      for (int o = 1; o <= 8; o <<= 1) { d0 += __shfl_xor_sync(0xffffffffu, d0, o); d1 += __shfl_xor_sync(0xffffffffu, d1, o); }   // within each half-warp
      uint4 vn;
      vn.x = __shfl_xor_sync(0xffffffffu, tra.x, 16); vn.y = __shfl_xor_sync(0xffffffffu, tra.y, 16);
      vn.z = __shfl_xor_sync(0xffffffffu, tra.z, 16); vn.w = __shfl_xor_sync(0xffffffffu, tra.w, 16);   // lanes < 16 get the V row
      if (lane < 16) {
        unpack8(vn, vf);
        float* nb = mrg + kMmaNew;
        *reinterpret_cast<float4*>(nb + l16 * 8) = make_float4(vf[0], vf[1], vf[2], vf[3]);
        *reinterpret_cast<float4*>(nb + l16 * 8 + 4) = make_float4(vf[4], vf[5], vf[6], vf[7]);
        if (lane == 0) { nb[128] = d0 * kQScale; nb[129] = d1 * kQScale; }
      }
    }
#endif
    St st;
    st_init(st);
    // ---- mma consumer state: S^T = K Q^T (positions x q-rows n = 2 * tok_local + hq; q_len == 1: n = 0, 1 real, rest zero), O^T = V^T P^T
    float mr0 = neg_inf(), mr1 = neg_inf(), lp0 = 0.f, lp1 = 0.f;   // running max (cols 2c, 2c+1 of this thread) and per-thread partial column sums of p
    float oacc[kMmaMt][4];
    if (kMma) {
#pragma unroll
      for (int i = 0; i < kMmaMt; ++i) oacc[i][0] = oacc[i][1] = oacc[i][2] = oacc[i][3] = 0.f;
      // q -> per-item scratch (all warps store identical bytes; each warp reads back after its own __syncwarp), then B fragments: thread (g, c), g < 2 = q-head g
      const unsigned qsa = smaddr(mrg + kMmaQs);
      if (lane < 16) {
        unsigned* qw = reinterpret_cast<unsigned*>(mrg + kMmaQs);
        qw[mma_q_word(l16, 0)] = qpa.x; qw[mma_q_word(l16, 1)] = qpa.y; qw[mma_q_word(l16, 2)] = qpa.z; qw[mma_q_word(l16, 3)] = qpa.w;
        qw[64 + mma_q_word(l16, 0)] = qpb.x; qw[64 + mma_q_word(l16, 1)] = qpb.y; qw[64 + mma_q_word(l16, 2)] = qpb.z; qw[64 + mma_q_word(l16, 3)] = qpb.w;
      }
      __syncwarp();
    }
#if SKV_IFACE == 2
    if (own && !kMma) {                                                  // the new position first (online softmax is order-free): row slot 0 starts from that single position
      float vf[8];                                              // (= one step<1> from the empty state: m = score, l = 1, o = v; slot 0 keeps the result)
      unpack8(vn1, vf);
      if (r == 0) {
        st.m0 = sn0; st.m1 = sn1; st.l0 = st.l1 = 1.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) st.o0[j] = st.o1[j] = vf[j];
      }
    }
#endif
    const int len = p1 - p0 - (tail ? 1 : 0);                   // (P9d: the TMA stream of a tail item stops at p1 - 1; the producer applies the same rule)
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
        if (kMma) {
          // warp: tile (warp % kMmaT) = positions tb .. tb + 15 of the stage, dim slice dh. K loads + QK^T, softmax, V loads (slot released after the last), PV.
          const int tb = (warp % kMmaT) * 16, dh = warp / kMmaT;
          const unsigned sb = sbase + kRingOff + slot * kStageBytes;
          const bool live = tb < n, partial = tb + 16 > n;       // warp-uniform: tile has >= 1 valid position / has invalid tail positions
          unsigned kdep = 0, pb0 = 0, pb1 = 0;
          const unsigned qsa = smaddr(mrg + kMmaQs) + mma_q_off(lane, 0);
          const unsigned j32 = mma_j32(lane), kb = sb + mma_k_base(lane, tb), vb = sb + mma_v_base(lane, tb);
          if (live) {
            float sa[4] = {0.f, 0.f, 0.f, 0.f}, sb2[4] = {0.f, 0.f, 0.f, 0.f};   // QK^T: two accumulation chains (even / odd k-steps)
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {                     // K fragments just in time (register pressure: 128-reg cap)
              unsigned k0[4];
              ldsm4(k0, kb + mma_k_term(ks, j32, kHalfBytes));
              uint2 qb0 = make_uint2(0u, 0u);                    // Q^T B fragment re-read from the scratch every stage (16 fewer live registers); q-rows >= 2 are zero
              if ((lane >> 2) < 2) qb0 = lds64(qsa + ks * 32);
              hmma16816((ks & 1) ? sb2 : sa, k0, qb0.x, qb0.y);  // two accumulation chains (even / odd k-steps)
              kdep ^= k0[0];
            }
            float s0 = sa[0] + sb2[0], s1 = sa[1] + sb2[1], s2 = sa[2] + sb2[2], s3 = sa[3] + sb2[3];   // (row g: s0, s1; row g + 8: s2, s3) x (cols 2c, 2c + 1)
            if (partial) {
              // TODO(R6, P5): causal mask per q column (pos > vis_t - 1 -> -inf, column n = 2 * tok + hq) goes here as well; q_len == 1 has only the tail mask.
              const int pos = tb + (lane >> 2);
              if (pos >= n) { s0 = neg_inf(); s1 = neg_inf(); }
              if (pos + 8 >= n) { s2 = neg_inf(); s3 = neg_inf(); }
            }
            // online softmax per column (q-row): max over the tile = own 2 rows, then xor 4, 8, 16 (lanes sharing c); kQScale (> 0) is applied to the fp32 scores
            float c0 = fmaxf(s0, s2), c1 = fmaxf(s1, s3);
#pragma unroll
            for (int o = 4; o <= 16; o <<= 1) { c0 = fmaxf(c0, __shfl_xor_sync(0xffffffffu, c0, o)); c1 = fmaxf(c1, __shfl_xor_sync(0xffffffffu, c1, o)); }
            const float n0 = fmaxf(mr0, c0 * kQScale), n1 = fmaxf(mr1, c1 * kQScale);   // finite: the tile has a valid position
            const float a0 = ex2(mr0 - n0), a1 = ex2(mr1 - n1);
            mr0 = n0; mr1 = n1;
            const float p0 = ex2(fmaf(s0, kQScale, -n0)), p1 = ex2(fmaf(s1, kQScale, -n1)), p2 = ex2(fmaf(s2, kQScale, -n0)), p3 = ex2(fmaf(s3, kQScale, -n1));
            // l: per-thread partial column sums of fp32 p (rescaled with the max), reduced over the 8 row groups once per item (cheaper than a ones-column HMMA or 6 shuffles per tile)
            lp0 = fmaf(lp0, a0, p0 + p2); lp1 = fmaf(lp1, a1, p1 + p3);
            // one O rescale per tile (each warp has exactly one tile per stage), skipped when no real column (c == 0 lanes: q-rows 0, 1) changed its max
            if (__any_sync(0xffffffffu, (lane & 3) == 0 && (a0 != 1.f || a1 != 1.f))) {
#pragma unroll
              for (int mt = 0; mt < kMmaMt; ++mt) { oacc[mt][0] *= a0; oacc[mt][1] *= a1; oacc[mt][2] *= a0; oacc[mt][3] *= a1; }
            }
            pb0 = movm_t(pack_bf16x2(p0, p1)); pb1 = movm_t(pack_bf16x2(p2, p3));   // C (rows pos g / g + 8) -> B (k = pos 2c.., n = g)
          }
          // PV: V A-fragments in groups of kMmaG m-tiles (register pressure); the slot is released after the last group's loads
#pragma unroll
          for (int g0 = 0; g0 < kMmaMt; g0 += kMmaG) {
            unsigned vf[kMmaG][4];
            if (live) {
#pragma unroll
              for (int gi = 0; gi < kMmaG; ++gi) ldsm4t(vf[gi], vb + mma_v_term(dh * kMmaMt + g0 + gi, j32, kHalfBytes));
              if (partial) {
                // V rows beyond n are stale / uninitialised smem: p is exactly 0 there but 0 * NaN would not be: zero the A-fragment halves (k = positions 2c, 2c + 1 (+ 8))
                const int k0 = tb + 2 * (lane & 3);
                const unsigned m01 = (k0 < n ? 0xffffu : 0u) | (k0 + 1 < n ? 0xffff0000u : 0u), m23 = (k0 + 8 < n ? 0xffffu : 0u) | (k0 + 9 < n ? 0xffff0000u : 0u);
#pragma unroll
                for (int gi = 0; gi < kMmaG; ++gi) { vf[gi][0] &= m01; vf[gi][1] &= m01; vf[gi][2] &= m23; vf[gi][3] &= m23; }
              }
            }
#if SKV_TMA_RELEASE
            if (g0 + kMmaG >= kMmaMt) {
              __syncwarp();                                      // all of this warp's ldmatrix are issued above (program order); the arrive's release orders them before the refill
              if (lane == 0) mb_arrive_dep(bar0 + 8 * (kStages + slot), kdep ^ (live ? vf[0][0] : 0u));
            }
#endif
            if (live) {
#pragma unroll
              for (int gi = 0; gi < kMmaG; ++gi) hmma16816(oacc[g0 + gi], vf[gi], pb0, pb1);
            }
          }
        } else {
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
        }
#if SKV_TMA_RELEASE
        if (tid == 0) { mb_wait(bar0 + 8 * (kStages + slot), par); produce(slot); }   // all 16 warps have read stage g: slot is free for stage g + kStages
        __syncwarp();
#else
        __syncthreads();                                         // every thread is done with the slot
        if (tid == 0) produce(slot);
#endif
      }
    }
    if (TMA && SKV_ITEM_PREFETCH && k + 1 < ni) { load_q(na, ntok, nqa, nqb); if (kTail) load_tail(na, trb); }
    const int4 rb = rec[2 * k + 1];                             // {part_slot, S [| n_items << 8 in iface 2], tok0, n_items | vis}
    // ---- end of item: xor-16 merge inside the warp, 16 warp states through smem, partial / normalised store
    {
      if (kMma) {
        float l0 = lp0, l1 = lp1;                                // column sums over the 8 row groups (lanes sharing c)
#pragma unroll
        for (int o = 4; o <= 16; o <<= 1) { l0 += __shfl_xor_sync(0xffffffffu, l0, o); l1 += __shfl_xor_sync(0xffffffffu, l1, o); }
        float* ws = mrg + (warp * 2) * kMmaMS;                   // slot (warp, hq): O^T dims of this warp (local index), m, l
        if ((lane & 3) == 0) {                                   // lanes holding q-rows 0 / 1 (cols 2c, 2c + 1 with c == 0)
#pragma unroll
          for (int mt = 0; mt < kMmaMt; ++mt) {
            const int ld = mt * 16 + (lane >> 2);
            ws[ld] = oacc[mt][0]; ws[kMmaMS + ld] = oacc[mt][1]; ws[ld + 8] = oacc[mt][2]; ws[kMmaMS + ld + 8] = oacc[mt][3];
          }
        }
        if (lane == 0) { ws[kMmaDW] = mr0; ws[kMmaDW + 1] = l0; ws[kMmaMS + kMmaDW] = mr1; ws[kMmaMS + kMmaDW + 1] = l1; }
      } else {
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
    }
    __syncthreads();
    if (tid < 256) {
      const int hq = tid >> 7, d = tid & 127;
      float M, L, O;
      if (kMma) {
        // dims d are owned by the kMmaT warps dh * kMmaT + t (tiles t = disjoint position sets); their m / l / O slots are merged, plus the new row (iface 2 owner)
        const int dh = d / kMmaDW;
        const float* b = mrg + (dh * kMmaT * 2 + hq) * kMmaMS;
        const float* nb = mrg + kMmaNew;
        M = neg_inf();
#if SKV_IFACE == 2
        if (own) M = nb[128 + hq];
#elif SKV_TAIL_LDG
        if (tail) M = nb[128 + hq];
#endif
#pragma unroll
        for (int t = 0; t < kMmaT; ++t) M = fmaxf(M, b[t * 2 * kMmaMS + kMmaDW]);
        const float Mc = M == neg_inf() ? 0.f : M;
        L = 0.f; O = 0.f;
#pragma unroll
        for (int t = 0; t < kMmaT; ++t) {
          const float a = ex2(b[t * 2 * kMmaMS + kMmaDW] - Mc);
          L = fmaf(a, b[t * 2 * kMmaMS + kMmaDW + 1], L);
          O = fmaf(a, b[t * 2 * kMmaMS + (d - dh * kMmaDW)], O);
        }
#if SKV_IFACE == 2
        if (own) { const float a = ex2(nb[128 + hq] - Mc); L += a; O = fmaf(a, nb[d], O); }
#elif SKV_TAIL_LDG
        if (tail) { const float a = ex2(nb[128 + hq] - Mc); L += a; O = fmaf(a, nb[d], O); }
#endif
      } else {
      const float* b = mrg + hq * kPartStride;
      M = neg_inf();
#pragma unroll
      for (int w = 0; w < 16; ++w) M = fmaxf(M, b[w * 2 * kPartStride + 128]);
      const float Mc = M == neg_inf() ? 0.f : M;
      L = 0.f; O = 0.f;
#pragma unroll
      for (int w = 0; w < 16; ++w) {
        const float a = ex2(b[w * 2 * kPartStride + 128] - Mc);
        L = fmaf(a, b[w * 2 * kPartStride + 129], L);
        O = fmaf(a, b[w * 2 * kPartStride + d], O);
      }
      }
#if SKV_IFACE == 2
      if ((rb.y & 255) == 1) {
#else
      if (rb.y == 1) {
#endif
        p.ao[(size_t)rb.z * (kQHeads * kDim) + (2 * kvh + hq) * kDim + d] = f2bf(O / L);
      } else {
        float* dst = p.part + ((size_t)rb.x * 2 + hq) * kPartStride;     // rb.x = part_base + split
        __stcg(dst + d, O);
        if (d == 0) { __stcg(dst + 128, M); __stcg(dst + 129, L); }
      }
    }
    __syncthreads();                                            // merge buffer free for the next item
#if SKV_TICKET
    // S > 1: the partial stores of every thread precede this barrier; thread 0's fence + release add publish them (cumulatively) on the pair's ticket. Nothing waits here.
    if (tid == 0 && rec_splits(rb) > 1) { __threadfence(); red_release_gpu(p.tickets + ticket_pair(ra.y)); }
#endif
    if (SKV_ITEM_PREFETCH) { ra = na; qa = nqa; qb = nqb; if (kTail) tra = trb; }
  }
}

#if SKV_QMAX > 1
// ------------------------------------------------------------------------------------------------ R6: multi-token causal decode (SKV_QMAX > 1)
// Two-phase mma.sync consumer for the TMA ring (64-position stages, R = 64, T = R / 16 = 4 position tiles). q-rows (columns) n = 2 t + hq (token t of the sequence, q-head hq of the
// GQA pair), N = 2 q_len <= 2 QMAX, kRNT n-tiles of 8 columns (compile time; columns >= 2 q_len hold q = 0 and see no position: never stored).
//   Phase A (S^T = K Q^T), task (position tile tt, n-tile nt): warp w -> tt = w % 4, nt = w / 4 when kRNT == 4 (one task per warp); kRNT 2: warps with w % 8 < 4 (nt = w / 8);
//           kRNT 1: warps 0..3. (Deviation from "idle warps >= T * NT" for kRNT < 4: the task's n-tile must lie in the same warp's phase-B n-tile set so that running m / l need no
//           cross-warp traffic.) 8 k-steps of ldmatrix.x4 K x q B fragment (ldmatrix of the swizzled q staging), 2 accumulation chains. Column max over the tile -> colmax[g & 1][tt][n].
//   barrier 0; every phase-B warp: stage max over the 4 tiles, running m_n, alpha_n; phase-A warps: p = ex2(s kQScale - m_n), fp32 column sums -> lsum[tt][n], P^T (bf16, via movmatrix) -> P[n][pos].
//   barrier 0; Phase B (O^T += V^T P^T): warp w owns dims (w % 8) * 16 .. +15 and n-tiles nb0 .. nb0 + kRNTB - 1 (nb0 = (w / 8) kRNTB; kRNT 1: warps 8..15 idle); rescale by alpha, l_n = l_n alpha_n +
//           sum_tt lsum[tt][n], 4 k-steps of V^T (ldmatrix.x4.trans) x P^T (ldmatrix from P[n][pos]). The ring slot is released (16 warp arrivals, idle warps too) after the last V ldmatrix.
//   Single-buffered P and lsum: stage s's writes (after barrier 1 of s) are read only in phase B of s (after barrier 2 of s); stage s + 1 writes them only after barrier 1 of s + 1, which no
//   warp passes before every warp has finished phase B of s (program order + barrier). colmax is written before barrier 1 and read between barriers 1 and 2; parity g & 1 double-buffers it.
//   q staging is per item (all 512 threads, one 16 B chunk each, then one __syncthreads); it is rewritten after the last stage of the previous item, whose barrier 2 follows every read of q (phase A).
//   No cross-warp merge at the end of an item: each B warp stores its 16 dims x its columns (S == 1: normalised bf16 into ao, else partial o (unnormalised w.r.t. m), m, l at part slot + t S).
constexpr int kRNT = SKV_QMAX <= 4 ? 1 : (SKV_QMAX <= 8 ? 2 : 4);   // n-tiles of 8 columns (ceil(2 QMAX / 8), rounded up to a power of 2)
constexpr int kRN8 = kRNT * 8;
constexpr int kRNTB = kRNT >= 2 ? kRNT / 2 : 1;                      // n-tiles per phase-B warp
constexpr int kRT = 4;                                               // position tiles per stage
constexpr int kRQo = 0, kRCMo = kRN8 * 256, kRLSo = kRCMo + 2 * kRT * kRN8 * 4, kRPo = kRLSo + kRT * kRN8 * 4, kREnd = kRPo + kRN8 * 128;   // byte offsets in the merge buffer
static_assert(kRNT * 8 >= 2 * SKV_QMAX, "R6 n-tiles");
static_assert(kREnd <= kMrgBytes, "R6 staging must fit in the merge buffer (kMrgBytes)");
static_assert(kStageRows == 64 && kMmaOn && SKV_IFACE == 1 && SKV_TMA_RELEASE == 1 && SKV_PROF_NOCOMP == 0 && SKV_ITEM_PREFETCH == 0 && SKV_FORCE_ENGINE != 0,
              "SKV_QMAX > 1 needs SKV_STAGE_ROWS 64, SKV_TMA_MMA 1, SKV_IFACE 1, SKV_TMA_RELEASE 1, the TMA engine");
static_assert(kStages * kStageBytes + kMrgBytes <= 147968, "R6 smem must not grow past the default");

__device__ __forceinline__ void run_items_r6(const Params& p, int layer, unsigned char* smem, unsigned& g, const int4* rec, int ni) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, c4 = lane & 3;
  const unsigned sbase = smaddr(smem), bar0 = sbase + kBarOff, mrga = sbase + kMrgOff;
  unsigned char* mrgb = smem + kMrgOff;
  const size_t layer_row = (size_t)layer * p.cache_rows;
  const int4 zero4 = make_int4(0, 0, 0, 0);

  // ---- TMA producer state (thread 0 only): as the mma consumer of run_items (interface v1)
  int pk = -1, pp = 0, pend = 0, pkvh = 0;
  size_t prow = 0;
  int4 pnxt = zero4;
  auto produce = [&](int slot) {
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
        const char* mk = (const char*)p.tmaps + (0 * kNumMaps + rs) * 128;
        const char* mv = (const char*)p.tmaps + (1 * kNumMaps + rs) * 128;
        tma3d(dst + off * 128, mk, 0, pkvh, rc, bar);
        tma3d(dst + kHalfBytes + off * 128, mk, 64, pkvh, rc, bar);
        tma3d(dst + 2 * kHalfBytes + off * 128, mv, 0, pkvh, rc, bar);
        tma3d(dst + 3 * kHalfBytes + off * 128, mv, 64, pkvh, rc, bar);
        off += rows; rc += rows;
      }
    }
    pp += n;
  };
  if (tid == 0) {
    pnxt = rec[0];
#pragma unroll
    for (int s = 0; s < kStages; ++s) produce((g + s) % kStages);
  }

  // ---- warp roles (see the header comment)
  const int tt = warp & 3, tb = tt * 16;
  const int nb0 = kRNT >= 2 ? (warp >> 3) * kRNTB : 0;               // first n-tile of this warp's phase B
  const int la = kRNTB == 2 ? ((warp >> 2) & 1) : 0;                 // index of the phase-A n-tile inside the phase-B set
  const int nta = nb0 + la;                                          // phase-A n-tile
  const bool aact = kRNT == 4 ? true : (kRNT == 2 ? (warp & 7) < 4 : warp < 4);
  const bool bact = kRNT >= 2 ? true : warp < 8;
  const unsigned j32 = mma_j32(lane);
  const unsigned qa = mrga + kRQo + r6_q_base(lane, nta), qxv = r6_q_xv(lane);
  const unsigned pa = mrga + kRPo + r6_p_base(lane, nb0, kRNTB), pxp = r6_p_xp(lane);
  const unsigned pst = mrga + kRPo + r6_p_off(nta * 8 + g8, 2 * tt) + c4 * 4;     // P^T store of this thread (second 8 positions: r6_p_off chunk 2 tt + 1)
  const unsigned pst1 = mrga + kRPo + r6_p_off(nta * 8 + g8, 2 * tt + 1) + c4 * 4;
  float* cmb = reinterpret_cast<float*>(mrgb + kRCMo);
  float* lsb = reinterpret_cast<float*>(mrgb + kRLSo);

  for (int k = 0; k < ni; ++k) {
    const int4 ra = rec[2 * k], rv = rec[2 * k + 1];               // {row_base, y, p0, p1}, {part_slot, S, tok0, n_items}
    const int kvh = ra.y & 15, p0 = ra.z, len = ra.w - ra.z, ql = ((ra.y >> 24) & 15) + 1, tok0 = rv.z, S = rv.y;
    const int vis0 = p.pair_info[(tok0 * kKvHeads + kvh) * 4 + 3];  // vis of token 0 (kv_len + 1); token t sees kv_len + t + 1 positions
    // q staging: row n = 2 t + hq, 16 chunks of 8 dims; rows >= 2 ql are zero
    for (int i = tid; i < kRN8 * 16; i += kBlock) {
      const int n = i >> 4, c = i & 15;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (n < 2 * ql) v = ldg_cg4(p.q + ((size_t)(tok0 + (n >> 1)) * kQHeads + 2 * kvh + (n & 1)) * kDim + c * 8);
      *reinterpret_cast<uint4*>(mrgb + kRQo + r6_q_off(n, c)) = v;
    }
    __syncthreads();
    float mrun[kRNTB][2], lrun[kRNTB][2], oacc[kRNTB][4], alpha[kRNTB][2];
#pragma unroll
    for (int i = 0; i < kRNTB; ++i) {
      mrun[i][0] = mrun[i][1] = neg_inf(); lrun[i][0] = lrun[i][1] = 0.f; alpha[i][0] = alpha[i][1] = 1.f;
      oacc[i][0] = oacc[i][1] = oacc[i][2] = oacc[i][3] = 0.f;
    }
    const int ns = (len + kStageRows - 1) / kStageRows;
    for (int j = 0; j < ns; ++j, ++g) {
      const int slot = g % kStages;
      const unsigned par = (g / kStages) & 1;
#ifdef SKV_DEBUG_WAIT
      mb_wait_dbg(bar0 + 8 * slot, par, layer, g, k, j);
#else
      mb_wait(bar0 + 8 * slot, par);
#endif
      const int n = min(kStageRows, len - j * kStageRows);          // valid positions of the stage (stage-relative positions >= n are stale smem)
      const unsigned sb = sbase + kRingOff + slot * kStageBytes;
      float* cm = cmb + (g & 1) * (kRT * kRN8);
      unsigned kdep = 0;
      float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
      // ---- phase A: scores of tile tt x n-tile nta (rows g8 / g8 + 8 positions, columns 2 c4, 2 c4 + 1 = tokens nta * 4 + c4, heads 0 / 1)
      if (aact) {
        float sa[4] = {0.f, 0.f, 0.f, 0.f}, sb2[4] = {0.f, 0.f, 0.f, 0.f};
        const unsigned kb = sb + mma_k_base(lane, tb);
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          unsigned qf[4];
          ldsm4(qf, qa + r6_q_term(jj, qxv));
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int ks = 2 * jj + h;
            unsigned k0[4];
            ldsm4(k0, kb + mma_k_term(ks, j32, kHalfBytes));
            hmma16816((ks & 1) ? sb2 : sa, k0, qf[2 * h], qf[2 * h + 1]);
            kdep ^= k0[0];
          }
        }
        s0 = sa[0] + sb2[0]; s1 = sa[1] + sb2[1]; s2 = sa[2] + sb2[2]; s3 = sa[3] + sb2[3];
        // mask: tail (stage position >= n) and causal (absolute position >= vis_t); columns of tokens >= q_len see nothing
        const int ta = nta * 4 + c4;
        const int cut = min(n, (ta < ql ? vis0 + ta : 0) - (p0 + j * kStageRows));
        const int pos = tb + g8;
        if (pos >= cut) { s0 = neg_inf(); s1 = neg_inf(); }
        if (pos + 8 >= cut) { s2 = neg_inf(); s3 = neg_inf(); }
        float c0 = fmaxf(s0, s2), c1 = fmaxf(s1, s3);               // raw (unscaled) scores: kQScale > 0 commutes with max
#pragma unroll
        for (int o = 4; o <= 16; o <<= 1) { c0 = fmaxf(c0, __shfl_xor_sync(0xffffffffu, c0, o)); c1 = fmaxf(c1, __shfl_xor_sync(0xffffffffu, c1, o)); }
        if (lane < 4) *reinterpret_cast<float2*>(cm + tt * kRN8 + nta * 8 + 2 * c4) = make_float2(c0, c1);
      }
      __syncthreads();
      // ---- running max / alpha of the phase-B columns (all-masked columns keep m = -inf with alpha = 1)
      if (bact) {
#pragma unroll
        for (int i = 0; i < kRNTB; ++i) {
          float mx0 = neg_inf(), mx1 = neg_inf();
#pragma unroll
          for (int u = 0; u < kRT; ++u) {
            const float2 v = *reinterpret_cast<const float2*>(cm + u * kRN8 + (nb0 + i) * 8 + 2 * c4);
            mx0 = fmaxf(mx0, v.x); mx1 = fmaxf(mx1, v.y);
          }
          const float n0 = fmaxf(mrun[i][0], mx0 * kQScale), n1 = fmaxf(mrun[i][1], mx1 * kQScale);
          alpha[i][0] = n0 == neg_inf() ? 1.f : ex2(mrun[i][0] - n0);
          alpha[i][1] = n1 == neg_inf() ? 1.f : ex2(mrun[i][1] - n1);
          mrun[i][0] = n0; mrun[i][1] = n1;
        }
      }
      // ---- phase A, second half: p, column sums, P^T
      if (aact) {
        float ma0 = mrun[0][0], ma1 = mrun[0][1];
        if (kRNTB == 2 && la) { ma0 = mrun[1][0]; ma1 = mrun[1][1]; }
        const float f0 = ma0 == neg_inf() ? 0.f : ma0, f1 = ma1 == neg_inf() ? 0.f : ma1;
        const float e0 = ex2(fmaf(s0, kQScale, -f0)), e1 = ex2(fmaf(s1, kQScale, -f1)), e2 = ex2(fmaf(s2, kQScale, -f0)), e3 = ex2(fmaf(s3, kQScale, -f1));
        float l0 = e0 + e2, l1 = e1 + e3;
#pragma unroll
        for (int o = 4; o <= 16; o <<= 1) { l0 += __shfl_xor_sync(0xffffffffu, l0, o); l1 += __shfl_xor_sync(0xffffffffu, l1, o); }
        if (lane < 4) *reinterpret_cast<float2*>(lsb + tt * kRN8 + nta * 8 + 2 * c4) = make_float2(l0, l1);
        sts32(pst, movm_t(pack_bf16x2(e0, e1)));                      // C (rows pos g8, columns 2 c4..) -> (n = g8, positions 2 c4, 2 c4 + 1) of P[n][pos]
        sts32(pst1, movm_t(pack_bf16x2(e2, e3)));                     // positions tb + 8 ..
      }
      __syncthreads();
      // ---- phase B
      unsigned vdep = 0;
      if (bact) {
#pragma unroll
        for (int i = 0; i < kRNTB; ++i) {
          float t0 = 0.f, t1 = 0.f;
#pragma unroll
          for (int u = 0; u < kRT; ++u) {
            const float2 v = *reinterpret_cast<const float2*>(lsb + u * kRN8 + (nb0 + i) * 8 + 2 * c4);
            t0 += v.x; t1 += v.y;
          }
          lrun[i][0] = fmaf(lrun[i][0], alpha[i][0], t0); lrun[i][1] = fmaf(lrun[i][1], alpha[i][1], t1);
          oacc[i][0] *= alpha[i][0]; oacc[i][1] *= alpha[i][1]; oacc[i][2] *= alpha[i][0]; oacc[i][3] *= alpha[i][1];
        }
#pragma unroll
        for (int ks = 0; ks < kRT; ++ks) {
          unsigned vf[4], pf[4] = {0u, 0u, 0u, 0u};
          ldsm4t(vf, sb + mma_v_base(lane, ks * 16) + mma_v_term(warp & 7, j32, kHalfBytes));
          if (kRNTB == 2) ldsm4(pf, pa + r6_p_term(ks, pxp)); else ldsm2(pf[0], pf[1], pa + r6_p_term(ks, pxp));
          if (n < kStageRows) {                                       // V rows beyond n are stale smem (0 * NaN): zero the A-fragment halves of positions >= n
            const int kk = ks * 16 + 2 * c4;
            const unsigned m01 = (kk < n ? 0xffffu : 0u) | (kk + 1 < n ? 0xffff0000u : 0u), m23 = (kk + 8 < n ? 0xffffu : 0u) | (kk + 9 < n ? 0xffff0000u : 0u);
            vf[0] &= m01; vf[1] &= m01; vf[2] &= m23; vf[3] &= m23;
          }
          vdep ^= vf[0];
#pragma unroll
          for (int i = 0; i < kRNTB; ++i) hmma16816(oacc[i], vf, pf[2 * i], pf[2 * i + 1]);
        }
      }
      __syncwarp();                                                   // this warp's ldmatrix of the slot (K in phase A, V in phase B) are all issued above
      if (lane == 0) mb_arrive_dep(bar0 + 8 * (kStages + slot), kdep ^ vdep);
      if (tid == 0) { mb_wait(bar0 + 8 * (kStages + slot), par); produce(slot); }   // all 16 warps released stage g: refill the slot with stage g + kStages
      __syncwarp();
    }
    // ---- end of item: each phase-B warp stores its 16 dims x its columns. S == 1: normalised bf16 into ao; else (o, m, l) at slot part_slot + t S
    if (bact) {
      const int dm = (warp & 7) * 16 + g8;
#pragma unroll
      for (int i = 0; i < kRNTB; ++i) {
        const int t = (nb0 + i) * 4 + c4;                             // token of this thread's columns 2 c4 (head 0), 2 c4 + 1 (head 1)
        if (t < ql) {
#pragma unroll
          for (int hq = 0; hq < 2; ++hq) {
            if (S == 1) {
              bf16* dst = p.ao + (size_t)(tok0 + t) * (kQHeads * kDim) + (2 * kvh + hq) * kDim + dm;
              const float inv = 1.f / lrun[i][hq];
              dst[0] = f2bf(oacc[i][hq] * inv); dst[8] = f2bf(oacc[i][2 + hq] * inv);
            } else {
              float* dst = p.part + ((size_t)(rv.x + t * S) * 2 + hq) * kPartStride;
              __stcg(dst + dm, oacc[i][hq]); __stcg(dst + dm + 8, oacc[i][2 + hq]);
              if ((warp & 7) == 0 && g8 == 0) { __stcg(dst + 128, mrun[i][hq]); __stcg(dst + 129, lrun[i][hq]); }
            }
          }
        }
      }
    }
#if SKV_TICKET
    if (S > 1) {                                                      // S is CTA-uniform: all warps' partial stores precede the barrier, then thread 0 publishes them
      __syncthreads();
      if (tid == 0) { __threadfence(); red_release_gpu(p.tickets + ticket_pair(ra.y)); }
    }
#endif
  }
}
#endif  // SKV_QMAX > 1

#if SKV_TICKET
// ------------------------------------------------------------------------------------------------ SKV_TICKET: deferred last-arriver merge
// Contract: tickets[pair] (pair = seq * 8 + kvh) is zeroed once per plan and never reset. Every S > 1 item adds 1 (release, gpu scope) after its partial store, so a layer adds exactly S
// to its pair's counter. Layers (and steps) are separated by the caller's grid barrier, so a read at the end of layer l sees no adds of layer l + 1: count % S == 0 iff all S items of
// layer l have arrived (count > 0 because the reader's own CTA added before reading). After its last item every CTA re-reads its S > 1 records, and one thread per item does ld.acquire.gpu
// of the ticket: complete iff count % S == 0. A pair may be merged by several CTAs: they compute identical values (same inputs, same order), so the duplicate bf16 stores are benign.
// At least one CTA sees a pair complete: each CTA's read follows its own add (barrier + program order) and RMWs on one address are totally ordered, so the CTA that issued the last add of the
// layer observes all S. (count is u32 and wraps after 2^32 / S layer-steps; 2^32 % S != 0 for S not a power of 2, hence re-zero per plan.) Zero the tickets again whenever the plan changes.
// Merge: 256 threads per (pair, token row): hq = t >> 7, dim d = t & 127; the two 256-thread halves of the CTA take alternate units. Same math as merge_th (max, weights, sum).
constexpr int kTkChunk = 16;                                          // o loads in flight per round (S <= 32 = planner smax: at most 2 rounds)
__device__ __forceinline__ void ticket_merge_unit(const Params& p, int slot0, int S, int hq, int d, bf16* dst) {
  constexpr int SS = 2 * kPartStride;
  const float* base = p.part + ((size_t)slot0 * 2 + hq) * kPartStride;     // split s at base + s * SS
  float2 ml[32];
#pragma unroll
  for (int c = 0; c < 32; ++c) ml[c] = c < S ? __ldcg(reinterpret_cast<const float2*>(base + c * SS + 128)) : make_float2(neg_inf(), 0.f);
  float o[kTkChunk];
#pragma unroll
  for (int c = 0; c < kTkChunk; ++c) o[c] = c < S ? __ldcg(base + c * SS + d) : 0.f;
  float M = neg_inf();
#pragma unroll
  for (int c = 0; c < 32; ++c) M = fmaxf(M, ml[c].x);
  float w[32], L = 0.f;
#pragma unroll
  for (int c = 0; c < 32; ++c) { w[c] = ex2(ml[c].x - M); L = fmaf(w[c], ml[c].y, L); }
  float a = 0.f;
#pragma unroll
  for (int c = 0; c < kTkChunk; ++c) a = fmaf(w[c], o[c], a);
  if (S > kTkChunk) {                                                  // S 17..32 (planner smax 32): second round (compile-time indices: no local memory)
#pragma unroll
    for (int c = 0; c < kTkChunk; ++c) o[c] = kTkChunk + c < S ? __ldcg(base + (kTkChunk + c) * SS + d) : 0.f;
#pragma unroll
    for (int c = 0; c < kTkChunk; ++c) a = fmaf(w[kTkChunk + c], o[c], a);   // (w of c >= S is 0)
  }
  *dst = f2bf(a / L);
}
// All 512 threads, after the CTA's item loop. Uses the first two words of the merge buffer (free: the item loop is done) for the per-chunk "pair complete" masks.
__device__ __forceinline__ void ticket_merge(const Params& p, unsigned char* smem, const int4* rec, int ni) {
  const int tid = threadIdx.x;
  volatile unsigned* flag = reinterpret_cast<volatile unsigned*>(smem + kMrgOff);
  const int h = tid >> 8, hq = (tid >> 7) & 1, d = tid & 127;
  __syncthreads();                                                     // the item loop's last stores / reads of the merge buffer are done; our own red is before this barrier
  unsigned unit = 0;
  for (int c0 = 0, ch = 0; c0 < ni; c0 += 32, ++ch) {
    if (tid < 32) {                                                    // warp 0: lane j checks item c0 + j (S > 1 only)
      unsigned m = 0;
      if (c0 + tid < ni) {
        const int4 ra = rec[2 * (c0 + tid)], rv = rec[2 * (c0 + tid) + 1];
        const int S = rec_splits(rv);
        if (S > 1) m = (ld_acquire_gpu(p.tickets + ticket_pair(ra.y)) % (unsigned)S) == 0u;
      }
      const unsigned b = __ballot_sync(0xffffffffu, m);
      if (tid == 0) flag[ch & 1] = b;                                  // two words alternate: the write of chunk ch + 2 follows the barrier of chunk ch + 1
    }
    __syncthreads();
    unsigned mask = flag[ch & 1];
    while (mask) {
      const int i = c0 + (__ffs(mask) - 1);
      mask &= mask - 1;
      const int4 ra = rec[2 * i], rv = rec[2 * i + 1];
      const int S = rec_splits(rv), kvh = ra.y & 15;
      const int ql = SKV_QMAX > 1 ? ((ra.y >> 24) & 15) + 1 : 1;
      for (int t = 0; t < ql; ++t, ++unit) {
        if ((unit & 1) != (unsigned)h) continue;
        const int4 pi = *reinterpret_cast<const int4*>(p.pair_info + ((rv.z + t) * kKvHeads + kvh) * 4);   // {part_base + t S, S, tok0, vis_t}
        ticket_merge_unit(p, pi.x, S, hq, d, p.ao + (size_t)(rv.z + t) * (kQHeads * kDim) + (2 * kvh + hq) * kDim + d);
      }
    }
  }
}
#endif

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
#if SKV_QMAX > 1
  constexpr bool tma = true;                                    // R6: TMA engine only (the planner never emits engine 0 for q_len > 1)
#else
  const bool tma = SKV_FORCE_ENGINE < 0 ? (p.engine != 0) : (SKV_FORCE_ENGINE != 0);
#endif
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
#if SKV_IFACE == 2
  const int ni = rec[1].y >> 8;                                 // iface 2: n_items = S word >> 8 (word 7 is vis)
#else
  const int ni = rec[1].w;                                      // n_items is stored in every record of the CTA
#endif
  if (ni == 0) return;
#if SKV_QMAX > 1
  run_items_r6(p, layer, smem, g, rec, ni);
#else
  if (tma) run_items<true>(p, layer, smem, g, rec, ni);
  else run_items<false>(p, layer, smem, g, rec, ni);
#endif
#if SKV_TICKET
  ticket_merge(p, smem, rec, ni);
#endif
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

// Half-row form for few tokens (combine_head with ntok <= 8: two warps per token row): dims half * 64 + lane * 2, +1. float2 loads -> 2 registers per
// split, so kMergeChunk2 = 20 covers the b1 plans (S <= 20) in one round without spills (the float4 form needs SKV_MERGE_CHUNK = 18 for 0 spills).
constexpr int kMergeChunk2 = 20;
__device__ __forceinline__ float2 merge_th2(const Params& p, int tok, int head, int lane, int half) {
  const int4 pi = *reinterpret_cast<const int4*>(p.pair_info + (tok * kKvHeads + (head >> 1)) * 4);
  const int S = pi.y;
  const float* base = p.part + ((size_t)pi.x * 2 + (head & 1)) * kPartStride + half * 64;
  constexpr int SS = 2 * kPartStride;
  float2 x[kMergeChunk2];
#pragma unroll
  for (int c = 0; c < kMergeChunk2; ++c) x[c] = c < S ? __ldcg(reinterpret_cast<const float2*>(base + c * SS) + lane) : make_float2(0.f, 0.f);
  float2 ml = lane < S ? __ldcg(reinterpret_cast<const float2*>(base - half * 64 + lane * SS + 128)) : make_float2(neg_inf(), 0.f);
  float M = ml.x;
#pragma unroll
  for (int o = 16; o; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
  const float w = lane < S ? ex2(ml.x - M) : 0.f;
  float L = w * ml.y;
#pragma unroll
  for (int o = 16; o; o >>= 1) L += __shfl_xor_sync(0xffffffffu, L, o);
  float2 a = make_float2(0.f, 0.f);
#pragma unroll
  for (int c = 0; c < kMergeChunk2; ++c) {
    if (c < S) { const float wc = __shfl_sync(0xffffffffu, w, c); a.x = fmaf(wc, x[c].x, a.x); a.y = fmaf(wc, x[c].y, a.y); }
  }
  for (int s0 = kMergeChunk2; s0 < S; s0 += kMergeChunk2) {         // S > kMergeChunk2 (rare): further rounds
#pragma unroll
    for (int c = 0; c < kMergeChunk2; ++c) x[c] = s0 + c < S ? __ldcg(reinterpret_cast<const float2*>(base + (s0 + c) * SS) + lane) : make_float2(0.f, 0.f);
#pragma unroll
    for (int c = 0; c < kMergeChunk2; ++c) {
      if (s0 + c < S) { const float wc = __shfl_sync(0xffffffffu, w, s0 + c); a.x = fmaf(wc, x[c].x, a.x); a.y = fmaf(wc, x[c].y, a.y); }
    }
  }
  const float inv = 1.f / L;
  return make_float2(a.x * inv, a.y * inv);
}

// All 512 threads: q-head `head`, tokens 0..ntok-1 -> bf16 into the SW128 K-major B slot xs (2 * TPAD * 128 bytes), rows ntok..TPAD-1 zero.
// element (t, k) -> xs + (k/64) * TPAD * 128 + t * 128 + ((((k%64)/8) ^ (t&7)) << 4), position k%8 in the chunk. One warp per token row.
// No __syncthreads inside; ends with fence.proxy.async (generic -> async proxy) so the caller only needs its own __syncthreads before the MMA.
template <int TPAD, int TNT>
__device__ void combine_head(const Params& p, int layer, int head, unsigned char* xs, int ntok) {
  static_assert(TNT == kBlock && TPAD % 16 == 0, "combine_head: 512 threads, TPAD multiple of 16");
  (void)layer;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (SKV_COMBINE_HALF && ntok <= 4) {                             // few tokens (b1 .. b4): two warps per token row; rows with S > kMergeChunk use merge_th2
    const int half = warp & 1;                                     // (one float2 round, 64 dims per warp), the others the float4 form on warp half 0 (job 9663:
    for (int t = warp >> 1; t < TPAD; t += 8) {                    // half rows are slower when S <= kMergeChunk or when all 16 warps have a row)
      const int4 pi = t < ntok ? *reinterpret_cast<const int4*>(p.pair_info + (t * kKvHeads + (head >> 1)) * 4) : make_int4(0, 0, 0, 0);
      if (pi.y > kMergeChunk) {
        const float2 o = merge_th2(p, t, head, lane, half);
        const int c = half * 8 + (lane >> 2);                      // 16-byte chunk of the row; lanes 4c' .. 4c'+3 hold its four 4-byte quarters
        const int off = (c >> 3) * (TPAD * 128) + t * 128 + ((((c & 7) ^ (t & 7))) << 4) + (lane & 3) * 4;
        *reinterpret_cast<unsigned*>(xs + off) = pack2(o.x, o.y);
      } else if (half == 0) {
        uint2 v = make_uint2(0u, 0u);
        if (pi.y == 1) {
          v = __ldcg(reinterpret_cast<const uint2*>(p.ao + (size_t)t * (kQHeads * kDim) + head * kDim) + lane);
        } else if (pi.y > 1) {
          const float4 o = merge_th(p, t, head, lane);
          v.x = pack2(o.x, o.y); v.y = pack2(o.z, o.w);
        }
        const int c = lane >> 1;
        const int off = (c >> 3) * (TPAD * 128) + t * 128 + ((((c & 7) ^ (t & 7))) << 4) + (lane & 1) * 8;
        *reinterpret_cast<uint2*>(xs + off) = v;
      }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    return;
  }
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
