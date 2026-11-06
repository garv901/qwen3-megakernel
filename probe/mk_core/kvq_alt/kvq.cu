// kvq: work-assignment arms on mk's real KV access pattern (NVRTC, sm_100a). Standalone: helpers copied from splitkv.cuh @ HEAD (tma3d, kv_policy, mb_*), no header prepended.
// Persistent cooperative kernel, 148 x 512, dynamic smem as mk NPAD 16 (165248). Per layer l = 0..nl-1:
//   grid barrier, stamp0 (block 0; every CTA stamps its start) | attend | every CTA stamps its end | grid barrier, stamp1 (block 0)
// attend = mk's run_items<true> ring without math: 2 stages x 128 rows, a stage = K + V of 128 positions of one (seq, kvh) at row stride 2 KB (256 B per kvh per tensor).
//   Producer = thread 0 (as mk: it refills slot s after all 16 warps arrived on empty[s]); consumers = 16 warps, wait full[s], read the stage from smem
//   (XOR into a register, as SKV_PROF_NOCOMP), arrive on empty[s] (lane 0 of each warp, count 16 = SKV_TMA_RELEASE 1).
//   Partial stages use 8 maps per tensor (box rows 128, 64, .., 1) like kNumMaps. KVQ_SWIZ 0: SWIZZLE_NONE maps, box 128 dims x 1 x rows, 2 TMA ops per full stage (the a1ub `ld` build);
//   KVQ_SWIZ 1: SWIZZLE_128B maps, box 64 dims x 1 x rows, 4 TMA ops per full stage (K lo/hi, V lo/hi; the a1ub `mk` build's access shape).
// Work: units[] = int4 {row_base, kvh, p0, p1} (words 0-3 of a plan record). CTA c runs the static list units[sbeg[c] .. sbeg[c] + scnt[c]) first, then (qcnt > 0) fetches units
//   units[qbeg + t] with t = atomicAdd(ctr[l], 1) until t >= qcnt. pf = 1: the ticket for unit k+1 is fetched when unit k is started (only once the static list is exhausted).
//   static = scnt only; queue = qcnt only; hybrid = both. ctr[l] is monotonic per launch (zeroed by the host before the launch).
// Consumers do not know the stage sequence, so the producer writes {n rows | first << 16 | last << 17, unit id} to a per-slot smem word before arming full[s]; an n = 0
//   sentinel stage ends the layer (it takes a slot and a barrier phase like any stage, everyone arrives on its empty barrier, so phases stay aligned across layers).
// COMB (mk's combine; KVQ_COMB 1): the end-of-item step stores the unit's partial to slot uinfo[u].x ({slot, S, tok, kvh}) in mk's layout part[(slot * 2 + hq) * 132 + {o[128], m, l, pad}]
//   (o = the 16-warp sum, m / l synthetic from the unit id; S == 1 pairs store bf16 ao instead, as mk's attend; check = 1 stores the partial too). comb = 1: after the attend barrier, `combine_head` of
//   HEAD splitkv.cuh (merge_th, kMergeChunk 20) as a1ub --comb head: CTA b owns O unit floor(b * 128 / 148) (head = u % 16) iff it starts a new one, merges all ntok rows of that head into an smem xs slot.
//   Every CTA stamps cend2 at the end of the combine (C = max cend2 - block-0 stamp after the attend barrier, as a1ub). comb = 0: no combine phase (mk: s_max == 1).
//   tick = 1 (SKV_TICKET): per-pair last-arriver merge after the CTA's item loop (ticket_merge), ao complete after attend; no combine phase. Needs S <= 32.
// kvq_alt additions (Agent B):
//   KVQ_ACC 1 / 2 / 3 / 4 (R1, fixed-reference accumulation): no per-piece slots / S-way merge. Each S > 1 piece scales its (o, l) by w = ex2(m - M_REF) (the synthetic m against the fixed reference, M_REF = 1.0 >= max m) and ADDS them into
//     one fp32 accumulator per (token, q head): acc[((tok * 8 + kvh) * 2 + hq) * 132 + {o[128], l, 0, 0, 0}] (pair = 1056 B contiguous). 1: red.global.add.f32 (256 threads, scalar); 2 / 3: red.global.add.v2 / .v4.f32 (132 / 66 threads, each sums
//     2 / 4 dims over the 16 warp states); 4: staging buffer in smem (kStgOff) + one cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 of 1056 B (thread 0; fence.proxy.async before, wait_group.read 0 before the next staging write, wait_group 0 at layer end).
//     The "combine" phase becomes normalize_head: o / l from acc (S independent, same CTA / row mapping as combine_head). acc is double-buffered by layer parity (buffer l & 1 in layer l); normalize zeroes the other buffer's rows (read in the previous layer); the host zeroes both for layer 0 / 1. S == 1 pieces store ao (bf16) as before.
//     check = 1 additionally stores the scaled (o, l) partial to the slot (m field = M_REF) so numpy can sum the same per-piece values.
//   tick = 2 (R3, merge queue): pieces release a per-pair ticket (as tick = 1) but nothing is listed; a warp whose CTA has no stream work left pulls (pair, hq) tasks mtask[ctr] from mctr[l] (tasks sorted by the unit id of the pair's last piece), spins until
//     ticket >= S * (l + 1), merge_th, writes bf16 ao. No combine phase. S <= 32.
// check = 1: visited[l][unit] += 1 per fetch; chk[l] ^= every 32-bit word of every stage read (any bijective thread -> chunk map gives the same XOR).
// rec = 1: urec[(l * n_units + unit) * 4] = {cta, smid, t first stage seen full, t last stage seen full} (thread 0, globaltimer).
typedef unsigned long long u64;
#ifndef KVQ_SWIZ
#define KVQ_SWIZ 0
#endif
#ifndef KVQ_MERGE
#define KVQ_MERGE 1    // per-unit end-of-item emulation (mk's merge): 16 warp states through smem (3 CTA barriers), 1 KB partial store per unit
#endif
#ifndef KVQ_MK
#define KVQ_MK 20      // SKV_MERGE_CHUNK of mk's combine
#endif
#ifndef KVQ_NOREAD
#define KVQ_NOREAD 0   // debug: consumers skip the smem reads (arrive only)
#endif
#ifndef KVQ_ACC
#define KVQ_ACC 0      // 0: per-piece slots + combine (mk); 1 red.f32, 2 red.v2.f32, 3 red.v4.f32, 4 TMA bulk reduce add.f32 (R1)
#endif
#ifndef KVQ_EF
#define KVQ_EF 1   // L2 evict_first policy on the KV loads (SKV_KV_EVICT_FIRST 1)
#endif
constexpr int kStageRows = 128, kStages = 2, kNumMaps = 8;
constexpr int kStageBytes = kStageRows * 512;                   // K + V of 128 positions
constexpr int kHalf = kStageRows * 128;                         // KVQ_SWIZ 1: one 64-dim half of K or V
constexpr int kMetaOff = kStages * kStageBytes;                 // per stage (4 entries of 16 B) {w0, unit id, part slot, S | tok << 16 | kvh << 24}
constexpr int kMrgOff = kMetaOff + 1024, kMrgBytes = 16 * 256 * 4;   // KVQ_MERGE scratch (16.4 KB; mk: 16.5 KB merge buffer)
constexpr int kPS = 132, kULim = 512;                             // partial slot stride (floats), per-CTA unit list capacity (ticket)
constexpr int kStgOff = kMrgOff + kMrgBytes;                    // KVQ_ACC 4: 2 x 132 floats staging for the bulk reduce
constexpr float kMRef = 1.0f;                                    // R1 fixed reference (log2 domain)
constexpr int kBarOff = kMetaOff + 64;                          // full[s] at +8 s, empty[s] at +8 (kStages + s)

__device__ __forceinline__ u64 gt() { u64 t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned smaddr(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ u64 kv_policy() { u64 pol; asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol)); return pol; }
__device__ __forceinline__ uint4 lds128(unsigned a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ void mb_init(unsigned b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n)); }
__device__ __forceinline__ void mb_expect(unsigned b, unsigned bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mb_wait(unsigned b, unsigned parity) {
  asm volatile("{\n .reg .pred p;\n KVQ_WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra KVQ_WAIT_%=;\n}\n" ::"r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void mb_arrive(unsigned b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory"); }
__device__ __forceinline__ void tma3d(unsigned dst, const void* map, int c0, int c1, int c2, unsigned b, u64 pol) {
#if KVQ_EF
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;"
               ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(b), "l"(pol) : "memory");
#else
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
               ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(b) : "memory"); (void)pol;
#endif
}

__device__ __forceinline__ float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float neginf() { return __int_as_float(0xff800000); }
__device__ __forceinline__ unsigned pack2(float a, float b) { unsigned r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r; }
__device__ __forceinline__ float bf2f(unsigned short h) { return __uint_as_float((unsigned)h << 16); }
__device__ __forceinline__ unsigned short f2bf(float a) { unsigned r; asm("cvt.rn.bf16x2.f32 %0, 0f00000000, %1;" : "=r"(r) : "f"(a)); return (unsigned short)(r & 0xffffu); }
__device__ __forceinline__ unsigned ld_acq(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void red_f32(float* p, float v) { asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory"); }
__device__ __forceinline__ void red_v2(float* p, float a, float b) { asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" ::"l"(p), "f"(a), "f"(b) : "memory"); }
__device__ __forceinline__ void red_v4(float* p, float a, float b, float c, float d) { asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory"); }
__device__ __forceinline__ void bulk_red_add(float* g, unsigned s, unsigned bytes) {
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;" ::"l"(g), "r"(s), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
__device__ __forceinline__ void bulk_wait_read0() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait0() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ float ldcg(const float* p) { return __ldcg(p); }
// merge_th of HEAD splitkv.cuh: one warp, (token, q-head) -> 4 dims per lane; the S partials in one pass (kMergeChunk float4 per lane in flight), further rounds if S > kMergeChunk
__device__ __forceinline__ float4 merge_th(const float* part, const int* pinfo, int tok, int head, int lane) {
  constexpr int kMergeChunk = KVQ_MK;
  const int4 pi = *reinterpret_cast<const int4*>(pinfo + (tok * 8 + (head >> 1)) * 4);
  const int S = pi.y;
  const float* base = part + ((size_t)pi.x * 2 + (head & 1)) * kPS;
  constexpr int SS = 2 * kPS;
  float4 x[kMergeChunk];
#pragma unroll
  for (int c = 0; c < kMergeChunk; ++c) x[c] = c < S ? __ldcg(reinterpret_cast<const float4*>(base + c * SS) + lane) : make_float4(0.f, 0.f, 0.f, 0.f);
  float2 ml = lane < S ? __ldcg(reinterpret_cast<const float2*>(base + lane * SS + 128)) : make_float2(neginf(), 0.f);
  float M = ml.x;
#pragma unroll
  for (int o = 16; o; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
  const float w = lane < S ? ex2f(ml.x - M) : 0.f;
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
  for (int s0 = kMergeChunk; s0 < S; s0 += kMergeChunk) {
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
// S > 32 (HEAD's merge_th keeps m / l of split s in lane s, so it is valid for S <= 32 = planner smax only): M first over all splits (ceil(S / 32) m loads per lane), then rounds of
// kMergeChunk splits: o loads in flight, w of split s0 + c = ex2(m - M) from lane c (loaded per round), L accumulated per lane. Used by the S-cap "none" arms.
__device__ __forceinline__ float4 merge_th_big(const float* part, const int* pinfo, int tok, int head, int lane) {
  constexpr int kMergeChunk = KVQ_MK, SS = 2 * kPS;
  const int4 pi = *reinterpret_cast<const int4*>(pinfo + (tok * 8 + (head >> 1)) * 4);
  const int S = pi.y;
  const float* base = part + ((size_t)pi.x * 2 + (head & 1)) * kPS;
  float M = neginf();
#pragma unroll 4
  for (int s = lane; s < S; s += 32) M = fmaxf(M, __ldcg(base + s * SS + 128));
#pragma unroll
  for (int o = 16; o; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
  float L = 0.f;
  for (int s0 = 0; s0 < S; s0 += kMergeChunk) {
    float4 y[kMergeChunk];
#pragma unroll
    for (int c = 0; c < kMergeChunk; ++c) y[c] = s0 + c < S ? __ldcg(reinterpret_cast<const float4*>(base + (s0 + c) * SS) + lane) : make_float4(0.f, 0.f, 0.f, 0.f);
    const float2 ml = (lane < kMergeChunk && s0 + lane < S) ? __ldcg(reinterpret_cast<const float2*>(base + (s0 + lane) * SS + 128)) : make_float2(neginf(), 0.f);
    const float w = (lane < kMergeChunk && s0 + lane < S) ? ex2f(ml.x - M) : 0.f;
    L = fmaf(w, ml.y, L);
#pragma unroll
    for (int c = 0; c < kMergeChunk; ++c) {
      if (s0 + c < S) {
        const float wc = __shfl_sync(0xffffffffu, w, c);
        a.x = fmaf(wc, y[c].x, a.x); a.y = fmaf(wc, y[c].y, a.y); a.z = fmaf(wc, y[c].z, a.z); a.w = fmaf(wc, y[c].w, a.w);
      }
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) L += __shfl_xor_sync(0xffffffffu, L, o);
  const float inv = 1.f / L;
  return make_float4(a.x * inv, a.y * inv, a.z * inv, a.w * inv);
}
// combine_head of HEAD (float4 form, SKV_COMBINE_HALF 0): all 512 threads, q-head `head`, rows 0..ntok-1 -> bf16 into the SW128 K-major xs slot (2 * TPAD * 128 B), rows ntok..TPAD-1 zero; one warp per row.
// cout != null (check): the fp32 merge result also goes to cout[(tok * 16 + head) * 128 + dim].
__device__ void combine_head(const float* part, const int* pinfo, const unsigned short* ao, float* cout, int head, unsigned char* xs, int ntok, int TPAD) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int c = lane >> 1;
  for (int t = warp; t < TPAD; t += 16) {
    uint2 v = make_uint2(0u, 0u);
    if (t < ntok) {
      const int4 pi = *reinterpret_cast<const int4*>(pinfo + (t * 8 + (head >> 1)) * 4);
      if (pi.y == 1) {
        v = __ldcg(reinterpret_cast<const uint2*>(ao + (size_t)t * 2048 + head * 128) + lane);
        if (cout) *reinterpret_cast<float4*>(cout + ((size_t)t * 16 + head) * 128 + lane * 4) =
            make_float4(bf2f(v.x & 0xffffu), bf2f(v.x >> 16), bf2f(v.y & 0xffffu), bf2f(v.y >> 16));
      } else {
        const float4 o = pi.y > 32 ? merge_th_big(part, pinfo, t, head, lane) : merge_th(part, pinfo, t, head, lane);
        v.x = pack2(o.x, o.y); v.y = pack2(o.z, o.w);
        if (cout) *reinterpret_cast<float4*>(cout + ((size_t)t * 16 + head) * 128 + lane * 4) = o;
      }
    }
    const int off = (c >> 3) * (TPAD * 128) + t * 128 + ((((c & 7) ^ (t & 7))) << 4) + (lane & 1) * 8;
    *reinterpret_cast<uint2*>(xs + off) = v;
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
// R1 normalize (replaces combine_head): same mapping; row (t, head) -> o / l from acc (one float4 per lane + l), S == 1 pairs read ao. acc is double-buffered by layer parity; accz (non-null in one of the 8 CTAs that read a head) zeroes the same row of the other buffer.
__device__ void normalize_head(float* acc, float* accz, const int* pinfo, const unsigned short* ao, float* cout, int head, unsigned char* xs, int ntok, int TPAD) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int c = lane >> 1;
  for (int t = warp; t < TPAD; t += 16) {
    uint2 v = make_uint2(0u, 0u);
    if (t < ntok) {
      const int4 pi = *reinterpret_cast<const int4*>(pinfo + (t * 8 + (head >> 1)) * 4);
      if (pi.y == 1) {
        v = __ldcg(reinterpret_cast<const uint2*>(ao + (size_t)t * 2048 + head * 128) + lane);
        if (cout) *reinterpret_cast<float4*>(cout + ((size_t)t * 16 + head) * 128 + lane * 4) =
            make_float4(bf2f(v.x & 0xffffu), bf2f(v.x >> 16), bf2f(v.y & 0xffffu), bf2f(v.y >> 16));
      } else {
        float* row = acc + ((size_t)(t * 8 + (head >> 1)) * 2 + (head & 1)) * kPS;
        const float4 o = __ldcg(reinterpret_cast<const float4*>(row) + lane);
        const float l = __ldcg(row + 128);
        const float inv = 1.f / l;
        const float4 r = make_float4(o.x * inv, o.y * inv, o.z * inv, o.w * inv);
        v.x = pack2(r.x, r.y); v.y = pack2(r.z, r.w);
        if (cout) *reinterpret_cast<float4*>(cout + ((size_t)t * 16 + head) * 128 + lane * 4) = r;
        if (accz) {                                              // the other buffer (read in the previous layer) is zeroed for the next layer; only the CTA that owns O unit u0 < 16 (the 8 units of a head read the same rows)
          float* rz = accz + ((size_t)(t * 8 + (head >> 1)) * 2 + (head & 1)) * kPS;
          __stcg(reinterpret_cast<float4*>(rz) + lane, make_float4(0.f, 0.f, 0.f, 0.f));
          if (lane == 0) __stcg(reinterpret_cast<float4*>(rz + 128), make_float4(0.f, 0.f, 0.f, 0.f));
        }
      }
    }
    const int off = (c >> 3) * (TPAD * 128) + t * 128 + ((((c & 7) ^ (t & 7))) << 4) + (lane & 1) * 8;
    *reinterpret_cast<uint2*>(xs + off) = v;
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
// SKV_TICKET: HEAD ticket_merge_unit (256 threads per (pair, token row): hq = tid >> 7, dim d); S <= 32. Writes bf16 ao.
__device__ __forceinline__ void ticket_merge_unit(const float* part, int slot0, int S, int hq, int d, unsigned short* dst) {
  constexpr int kTkChunk = 16, SS = 2 * kPS;
  const float* base = part + ((size_t)slot0 * 2 + hq) * kPS;
  float2 ml[32];
#pragma unroll
  for (int c = 0; c < 32; ++c) ml[c] = c < S ? __ldcg(reinterpret_cast<const float2*>(base + c * SS + 128)) : make_float2(neginf(), 0.f);
  float o[kTkChunk];
#pragma unroll
  for (int c = 0; c < kTkChunk; ++c) o[c] = c < S ? __ldcg(base + c * SS + d) : 0.f;
  float M = neginf();
#pragma unroll
  for (int c = 0; c < 32; ++c) M = fmaxf(M, ml[c].x);
  float w[32], L = 0.f;
#pragma unroll
  for (int c = 0; c < 32; ++c) { w[c] = ex2f(ml[c].x - M); L = fmaf(w[c], ml[c].y, L); }
  float a = 0.f;
#pragma unroll
  for (int c = 0; c < kTkChunk; ++c) a = fmaf(w[c], o[c], a);
  if (S > kTkChunk) {
#pragma unroll
    for (int c = 0; c < kTkChunk; ++c) o[c] = kTkChunk + c < S ? __ldcg(base + (kTkChunk + c) * SS + d) : 0.f;
#pragma unroll
    for (int c = 0; c < kTkChunk; ++c) a = fmaf(w[kTkChunk + c], o[c], a);
  }
  *dst = f2bf(a / L);
}
// All 512 threads after the CTA's item loop (HEAD ticket_merge): per chunk of 32 listed units (S > 1) warp 0 checks "pair complete" (ticket % S == 0, ld.acquire.gpu), the complete ones are merged by the two 256-thread halves alternately.
__device__ __forceinline__ void ticket_merge(const float* part, const int* pinfo, const int4* uinfo, const unsigned* tickets, unsigned short* ao, const int* ulist, const int* nlist, unsigned char* flagbuf) {
  const int tid = threadIdx.x, h = tid >> 8, hq = (tid >> 7) & 1, d = tid & 127;
  volatile unsigned* flag = reinterpret_cast<volatile unsigned*>(flagbuf);
  __syncthreads();
  const int ni = *reinterpret_cast<const volatile int*>(nlist);
  unsigned unit = 0;
  for (int c0 = 0, ch = 0; c0 < ni; c0 += 32, ++ch) {
    if (tid < 32) {
      unsigned m = 0;
      if (c0 + tid < ni) {
        const int4 ui = uinfo[ulist[c0 + tid]];
        if (ui.y > 1) m = (ld_acq(tickets + ui.z * 8 + ui.w) % (unsigned)ui.y) == 0u;
      }
      const unsigned b = __ballot_sync(0xffffffffu, m);
      if (tid == 0) flag[ch & 1] = b;
    }
    __syncthreads();
    unsigned mask = flag[ch & 1];
    while (mask) {
      const int i = c0 + (__ffs(mask) - 1);
      mask &= mask - 1;
      const int4 ui = uinfo[ulist[i]];
      if ((unit++ & 1) != (unsigned)h) continue;
      ticket_merge_unit(part, pinfo[(ui.z * 8 + ui.w) * 4], ui.y, hq, d, ao + (size_t)ui.z * 2048 + (2 * ui.w + hq) * 128 + d);
    }
  }
}
__device__ __forceinline__ void bar_arrive(unsigned* cnt) {   // production grid barrier (a1ub a1_bar_arrive / a1_bar_wait)
  __syncthreads();
  if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(cnt) : "memory");
}
__device__ __forceinline__ void bar_wait(const unsigned* cnt, unsigned& target) {
  target += gridDim.x;
  if (threadIdx.x == 0) {
    unsigned v;
    do { asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(cnt) : "memory"); } while (v < target);
    asm volatile("fence.acquire.gpu;" ::: "memory");
  }
  __syncthreads();
}

extern "C" __global__ void __launch_bounds__(512, 1) kvq(
    const void* tmaps, const int4* units, const int* sbeg, const int* scnt, int qbeg, int qcnt, unsigned* ctr, int pf, int nl, int cache_rows, int n_units,
    unsigned* bar, u64* stamps, u64* cstart, u64* cend, unsigned* cstat, unsigned* chk, unsigned* visited, u64* urec, int check, int rec, unsigned* sink, float* part,
    const char* pool, int fill_vec, size_t fill_adv, size_t pool_mod, size_t pool_off0,
    const int4* uinfo, const int* pinfo, unsigned short* ao, float* cout, u64* cend2, int comb, int tick, int ntok, int tpad, unsigned* tickets, int* ulist, float* accb, unsigned* mctr, const int* mtask, int nmt) {
  extern __shared__ __align__(1024) unsigned char sm[];
  const int tid = threadIdx.x, cta = blockIdx.x, nct = gridDim.x, lane = tid & 31;
  const unsigned sbase = smaddr(sm), bar0 = sbase + kBarOff;
  const u64 pol = kv_policy();                                   // one policy per CTA (the compiler CSEs it in mk)
  if (tid == 0 && (sbase & 1023u)) __trap();                     // the swizzled ring needs a 1024 B aligned base
  if (tid == 0) {
    for (int s = 0; s < kStages; ++s) { mb_init(bar0 + 8 * s, 1); mb_init(bar0 + 8 * (kStages + s), 16); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int sb = sbeg[cta], sc = scnt[cta];
  unsigned target = 0, acc = 0, g = 0, pg = 0;                   // g: consumer stage counter, pg: producer stage counter (thread 0), both carried across layers
  for (int l = 0; l < nl; ++l) {
    if (fill_vec) {                                              // a1ub's F phase: cold rotating-pool weight read (LDG.128), the L2 / DRAM state the attention layers follow in mk
      const size_t off = ((pool_off0 + (size_t)l * fill_adv) % pool_mod) & ~(size_t)4095;
      const char* fb = pool + off + (size_t)cta * fill_vec * 16;
      for (int i = tid; i < fill_vec; i += 512 * 4) {
        uint4 v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int idx = i + u * 512; v[u] = make_uint4(0, 0, 0, 0);
          if (idx < fill_vec) asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(fb + (size_t)idx * 16));
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
      }
    }
    float* accl = accb + (size_t)(l & 1) * ntok * 16 * kPS;
    bar_arrive(bar); bar_wait(bar, target);
    if (tid == 0) { const u64 t = gt(); cstart[(size_t)l * nct + cta] = t; if (cta == 0) stamps[l * 2] = t; }
    if (tid == 0) *reinterpret_cast<volatile int*>(sm + kMetaOff + 128) = 0;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    // ---- producer state (thread 0)
    int sidx = sb; const int send = sb + sc;
    unsigned qpre = 0; bool qhave = false, done = false, pfirst = false;
    int pp = 0, pend = 0, pkvh = 0, uid = 0;
    int4 uin = make_int4(0, 0, 0, 0);                            // uinfo of the unit being streamed (fetched with its record, so the end-of-item step does no global load)
    size_t prow = 0;
    unsigned npos = 0, nunit = 0;
    bool prepped = false;                                        // the next stage's meta word is written (done before the empty wait when the unit continues)
    int pn = 0;
    auto prep = [&]() {                                          // meta word of stage pg (4 entries: stage g + 2 never shares one with a stage still being read)
      volatile unsigned* meta = reinterpret_cast<volatile unsigned*>(sm + kMetaOff + (pg & 3) * 16);
      pn = min(kStageRows, pend - pp);
      meta[0] = (unsigned)pn | (pfirst ? 1u << 16 : 0u) | ((pp + pn >= pend) ? 1u << 17 : 0u); meta[1] = (unsigned)uid;
      meta[2] = (unsigned)uin.x; meta[3] = (unsigned)uin.y | ((unsigned)uin.z << 16) | ((unsigned)uin.w << 24);
      prepped = true;
    };
    auto produce = [&]() {
      if (done) return;
      const int slot = pg & 1;
      if (!prepped) {
        if (pp >= pend) {
          bool got = false;
          if (sidx < send) { uid = sidx++; got = true; }
          else if (qcnt > 0) {
            unsigned t;
            if (qhave) { t = qpre; qhave = false; } else t = atomicAdd(ctr + l, 1u);
            if (t < (unsigned)qcnt) { uid = qbeg + (int)t; got = true; }
          }
          if (!got) {                                            // sentinel: ends the layer
            done = true;
            reinterpret_cast<volatile unsigned*>(sm + kMetaOff + (pg & 3) * 16)[0] = 0;
            mb_expect(bar0 + 8 * slot, 0u); ++pg; return;
          }
          const int4 it = units[uid]; uin = uinfo[uid];
          pp = it.z; pend = it.w; pkvh = it.y; prow = (size_t)l * cache_rows + it.x; pfirst = true;
          ++nunit;
          if (check) atomicAdd(visited + (size_t)l * n_units + uid, 1u);
          if (pf && qcnt > 0 && sidx >= send && !qhave) { qpre = atomicAdd(ctr + l, 1u); qhave = true; }   // ticket of the next unit, fetched while this one streams
        }
        prep();
      }
      const int n = pn;
      const unsigned dst = sbase + slot * kStageBytes, bar = bar0 + 8 * slot;
      mb_expect(bar, (unsigned)n * 512u);
      int off = 0, rc = (int)(prow + pp);
#pragma unroll
      for (int rs = 0; rs < kNumMaps; ++rs) {
        const int rows = kStageRows >> rs;
        if (n & rows) {
          const char* mk = (const char*)tmaps + (0 * kNumMaps + rs) * 128;
          const char* mv = (const char*)tmaps + (1 * kNumMaps + rs) * 128;
#if KVQ_SWIZ
          tma3d(dst + off * 128, mk, 0, pkvh, rc, bar, pol);
          tma3d(dst + kHalf + off * 128, mk, 64, pkvh, rc, bar, pol);
          tma3d(dst + 2 * kHalf + off * 128, mv, 0, pkvh, rc, bar, pol);
          tma3d(dst + 3 * kHalf + off * 128, mv, 64, pkvh, rc, bar, pol);
#else
          tma3d(dst + off * 256, mk, 0, pkvh, rc, bar, pol);
          tma3d(dst + kStageRows * 256 + off * 256, mv, 0, pkvh, rc, bar, pol);
#endif
          off += rows; rc += rows;
        }
      }
      pp += n; pfirst = false; npos += n; ++pg; prepped = false;
    };
    if (tid == 0) { produce(); produce(); }
    unsigned lacc = 0;
    for (;;) {
      const int slot = g & 1;
      const unsigned par = (g >> 1) & 1;
      mb_wait(bar0 + 8 * slot, par);
      const unsigned w0 = *reinterpret_cast<volatile unsigned*>(sm + kMetaOff + (g & 3) * 16);
      const int n = (int)(w0 & 0xffffu);
      if (n == 0) {                                              // sentinel
        __syncwarp();
        if (lane == 0) mb_arrive(bar0 + 8 * (kStages + slot));
        ++g;
        break;
      }
      if (rec && tid == 0 && (w0 & 0x30000u)) {
        const unsigned u = *reinterpret_cast<volatile unsigned*>(sm + kMetaOff + (g & 3) * 16 + 4);
        u64* r = urec + ((size_t)l * n_units + u) * 4;
        const u64 t = gt();
        if (w0 & 0x10000u) { unsigned smid; asm("mov.u32 %0, %%smid;" : "=r"(smid)); r[0] = cta; r[1] = smid; r[2] = t; }
        if (w0 & 0x20000u) r[3] = t;
      }
#if KVQ_SWIZ
      const int per = n * 8, nreg = 4, rstride = kHalf;
#else
      const int per = n * 16, nreg = 2, rstride = kStageRows * 256;
#endif
      uint4 v[8];                                                // all 8 chunks of this thread are loaded before the slot is released (as mk's consumer: lds, then arrive, then math)
      const int tot = per * nreg;
#pragma unroll
      for (int k = 0; k < 8; ++k) {                              // flat 16 B chunks of the stage's regions (K, V [x lo / hi])
        const int c = tid + 512 * k;
        v[k] = make_uint4(0, 0, 0, 0);
        if (!KVQ_NOREAD && c < tot) { const int rg = (c >= per) + (c >= 2 * per) + (c >= 3 * per), i = c - rg * per; v[k] = lds128(sbase + slot * kStageBytes + rg * rstride + i * 16); }
      }
      __syncwarp();                                              // as mk: the slot is released right after the loads are issued (the arrive's .release.cta orders them); the XOR of the loaded words follows
      if (lane == 0) mb_arrive(bar0 + 8 * (kStages + slot));
      unsigned a = 0;
#pragma unroll
      for (int k = 0; k < 8; ++k) a ^= v[k].x ^ v[k].y ^ v[k].z ^ v[k].w;
      lacc ^= a;
      if (tid == 0) {
        if (!done && !prepped && pp < pend) prep();                // stage g + 2 continues the current unit: its meta word is written before the empty wait
        mb_wait(bar0 + 8 * (kStages + slot), par); produce();
      }   // all 16 warps released stage g: refill the slot with stage g + 2
      __syncwarp();
#if KVQ_MERGE
      if (w0 & 0x20000u) {                                       // last stage of a unit: warp states -> smem -> merged partial -> global (mk's end-of-item xor-16 / 16-warp merge)
        float* mrg = reinterpret_cast<float*>(sm + kMrgOff);
#if KVQ_ACC == 4
        if (tid == 0) bulk_wait_read0();                         // the previous unit's bulk reduce has read the staging buffer (long done)
#endif
        const float fa = (float)(a & 0xfffu) * 0.01f;
#pragma unroll
        for (int j = 0; j < 8; ++j) mrg[(tid >> 5) * 256 + lane * 8 + j] = fa + (float)j;
        __syncthreads();
        float sum = 0.f;
        const unsigned u = *reinterpret_cast<volatile unsigned*>(sm + kMetaOff + (g & 3) * 16 + 4);
        const unsigned w3 = *reinterpret_cast<volatile unsigned*>(sm + kMetaOff + (g & 3) * 16 + 12);
        const int4 ui = make_int4(*reinterpret_cast<volatile int*>(sm + kMetaOff + (g & 3) * 16 + 8), (int)(w3 & 0xffffu), (int)((w3 >> 16) & 0xffu), (int)(w3 >> 24));   // {slot, S, tok, kvh}
#if KVQ_ACC >= 2 && KVQ_ACC <= 3
        constexpr int V = KVQ_ACC == 3 ? 4 : 2, NT = 2 * (128 / V + 1);            // threads per hq: 128 / V dim groups + 1 for l
        if (ui.y > 1) {
          if (tid < NT) {
            const int hq = tid / (NT / 2), j = tid % (NT / 2);
            const float m = 0.1f * (float)((u * 37u + hq * 11u) % 19u) - 1.f, lsum = 1.f + 0.25f * (float)((u + hq) % 7u);
            const float wt = ex2f(m - kMRef);
            float* row = accl + ((size_t)(ui.z * 8 + ui.w) * 2 + hq) * kPS;
            if (j < 128 / V) {
              float s4[V];
#pragma unroll
              for (int c = 0; c < V; ++c) s4[c] = 0.f;
#pragma unroll
              for (int w = 0; w < 16; ++w) {
#pragma unroll
                for (int c = 0; c < V; ++c) s4[c] += mrg[w * 256 + hq * 128 + j * V + c];
              }
              sum = s4[0];
              if (V == 4) red_v4(row + j * 4, s4[0] * wt, s4[1] * wt, s4[2] * wt, s4[3] * wt); else red_v2(row + j * 2, s4[0] * wt, s4[1] * wt);
            } else {
              if (V == 4) red_v4(row + 128, lsum * wt, 0.f, 0.f, 0.f); else red_v2(row + 128, lsum * wt, 0.f);
            }
          }
        }
        if ((check || ui.y == 1) && tid < 256) {                  // check / S == 1 stores as the scalar path
          const int hq = tid >> 7, d = tid & 127;
          float sm_ = 0.f;
#pragma unroll
          for (int w = 0; w < 16; ++w) sm_ += mrg[w * 256 + tid];
          sum = sm_;
          const float m = 0.1f * (float)((u * 37u + hq * 11u) % 19u) - 1.f, lsum = 1.f + 0.25f * (float)((u + hq) % 7u);
          float* pp_ = part + ((size_t)ui.x * 2 + hq) * kPS;
          if (check) {
            const float wt = ex2f(m - kMRef);
            pp_[d] = sum * wt;
            if (d == 0) *reinterpret_cast<float2*>(pp_ + 128) = make_float2(kMRef, lsum * wt);
          }
          if (ui.y == 1) ao[(size_t)ui.z * 2048 + (2 * ui.w + hq) * 128 + d] = f2bf(sum / lsum);
        }
        __syncthreads();
#else
        if (tid < 256) {
#pragma unroll
          for (int w = 0; w < 16; ++w) sum += mrg[w * 256 + tid];
          const int hq = tid >> 7, d = tid & 127;
          const float m = 0.1f * (float)((u * 37u + hq * 11u) % 19u) - 1.f, lsum = 1.f + 0.25f * (float)((u + hq) % 7u);   // synthetic m, l of the unit
          float* pp_ = part + ((size_t)ui.x * 2 + hq) * kPS;
#if KVQ_ACC == 0
          if (ui.y > 1 || check) {                                // mk: S > 1 items store the partial (o, m, l)
            pp_[d] = sum;
            if (d == 0) *reinterpret_cast<float2*>(pp_ + 128) = make_float2(m, lsum);
          }
#else
          const float wt = ex2f(m - kMRef), ow = sum * wt, lw = lsum * wt;                  // piece (o, l) against the fixed reference
          if (check) {
            pp_[d] = ow;
            if (d == 0) *reinterpret_cast<float2*>(pp_ + 128) = make_float2(kMRef, lw);
          }
          if (ui.y > 1) {
            float* row = accl + ((size_t)(ui.z * 8 + ui.w) * 2 + hq) * kPS;
#if KVQ_ACC == 1
            red_f32(row + d, ow);
            if (d == 0) red_f32(row + 128, lw);
#else
            float* stg = reinterpret_cast<float*>(sm + kStgOff) + hq * kPS;
            stg[d] = ow;
            if (d == 0) *reinterpret_cast<float4*>(stg + 128) = make_float4(lw, 0.f, 0.f, 0.f);
#endif
          }
#endif
          if (ui.y == 1) ao[(size_t)ui.z * 2048 + (2 * ui.w + hq) * 128 + d] = f2bf(sum / lsum);   // mk: S == 1 items write ao (normalised)
          if (tick && ui.y > 1) __threadfence();
        }
        __syncthreads();
#if KVQ_ACC == 4
        if (tid == 0 && ui.y > 1) {                               // staging -> global add (async proxy reads what the generic proxy wrote)
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          bulk_red_add(accl + (size_t)(ui.z * 8 + ui.w) * 2 * kPS, smaddr(sm + kStgOff), 2 * kPS * 4);
        }
#endif
#endif
        if (tick && tid == 0 && ui.y > 1) {                       // last-arriver ticket: release after the partial stores, then record the unit for the end-of-CTA merge (tick 1)
          asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(tickets + ui.z * 8 + ui.w) : "memory");
          if (tick == 1) {
            volatile int* nl_ = reinterpret_cast<volatile int*>(sm + kMetaOff + 128);
            const int n_ = *nl_; if (n_ >= kULim) __trap();
            ulist[cta * kULim + n_] = (int)u; *nl_ = n_ + 1;
          }
        }
        acc ^= __float_as_uint(sum) & 1u;
      }
#endif
      ++g;
    }
    acc ^= lacc;
#if KVQ_ACC == 4
    if (tid == 0) bulk_wait0();                                  // all bulk reductions of this CTA have completed (before the release in bar_arrive)
#endif
    if (tick == 2) {                                             // R3: merge tasks (pair, hq), one per warp
      const unsigned wl = (unsigned)l + 1u;
      for (;;) {
        unsigned t = 0;
        if (lane == 0) t = atomicAdd(mctr + l, 1u);
        t = __shfl_sync(0xffffffffu, t, 0);
        if (t >= (unsigned)nmt) break;
        const int task = mtask[t], pr = task >> 1, hq = task & 1;
        const int4 pi = *reinterpret_cast<const int4*>(pinfo + pr * 4);
        const unsigned target = (unsigned)pi.y * wl;
        const u64 tw0 = rec ? gt() : 0;
        while (ld_acq(tickets + pr) < target) __nanosleep(32);
        __syncwarp();
        const u64 tw1 = rec ? gt() : 0;
        const int tok = pr >> 3, head = 2 * (pr & 7) + hq;
        const float4 o = merge_th(part, pinfo, tok, head, lane);
        *reinterpret_cast<uint2*>(ao + (size_t)tok * 2048 + head * 128 + lane * 4) = make_uint2(pack2(o.x, o.y), pack2(o.z, o.w));
        if (rec && lane == 0) {                                  // R3 stats at urec[nl * n_units * 4 + l * 3 + {spin ns, merge ns, tasks}], summed over warps
          const u64 tw2 = gt();
          u64* st = urec + (size_t)nl * n_units * 4 + l * 3;
          atomicAdd(st, tw1 - tw0); atomicAdd(st + 1, tw2 - tw1); atomicAdd(st + 2, 1ull);
        }
      }
    }
    if (tick == 1) ticket_merge(part, pinfo, uinfo, tickets, ao, ulist + cta * kULim, reinterpret_cast<const int*>(sm + kMetaOff + 128), sm + kMrgOff);
    if (check) {
#pragma unroll
      for (int o = 16; o; o >>= 1) lacc ^= __shfl_xor_sync(0xffffffffu, lacc, o);
      if (lane == 0) atomicXor(chk + l, lacc);
    }
    __syncthreads();
    if (tid == 0) {
      cend[(size_t)l * nct + cta] = gt();
      cstat[((size_t)l * nct + cta) * 2] = npos; cstat[((size_t)l * nct + cta) * 2 + 1] = nunit;
    }
    bar_arrive(bar); bar_wait(bar, target);
    if (cta == 0 && tid == 0) stamps[l * 2 + 1] = gt();
    if (comb) {                                                  // a1ub --comb head: CTA b owns O unit u0 = floor(b * 128 / ncta) iff floor((b + 1) * 128 / ncta) > u0
      const int u0 = (cta * 128) / nct, u1 = ((cta + 1) * 128) / nct;
      if (u1 > u0) {
#if KVQ_ACC
        normalize_head(accb + (size_t)(l & 1) * ntok * 16 * kPS, u0 < 16 ? accb + (size_t)((l + 1) & 1) * ntok * 16 * kPS : nullptr, pinfo, ao, check ? cout : nullptr, u0 % 16, sm, ntok, tpad);
#else
        combine_head(part, pinfo, ao, check ? cout : nullptr, u0 % 16, sm, ntok, tpad);
#endif
        __syncthreads();
        acc ^= reinterpret_cast<const unsigned*>(sm)[tid];
        __syncthreads();
      }
    }
    if (tid == 0) cend2[(size_t)l * nct + cta] = gt();
  }
  sink[cta * 512 + tid] = acc;
}
