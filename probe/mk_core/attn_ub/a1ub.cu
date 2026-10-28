// P2 (MK-attn PASS-02) bench kernel for A1 v1 (splitkv.cuh is prepended by a1bench.py; NVRTC, sm_100a). Persistent cooperative kernel, 148 x 512.
// Per layer l = 0..n_layers-1:
//   F: filler (cold rotating-pool weights, LDG.128 nc, as kv_stream.cu)  | grid barrier, stamp0 (block 0)
//   A: skv::attend                                                       | grid barrier, stamp1 (block 0)
//   C: mode & 1: O-proj emulation, CTA b owns unit u = floor(b*128/148) iff floor((b+1)*128/148) > floor(b*128/148), head = u % 16:
//        skv::combine_head<TPAD> into smem (the ring region), __syncthreads, dummy read of xs
//      mode & 2: skv::combine_all_to_ao + grid barrier
//      mode & 4: (with 1) layer n_layers-1: CTAs with unit u < 16 dump xs (2*TPAD*128 B) to dbg + u * 2*TPAD*128
//   every CTA stamps its end of C into cend[l * ncta + cta] (C time = max over CTAs - stamp1 of block 0).
// Layer-dependent output: ao + l * ao_stride (ao_stride = 0 in timing mode, total_q * 2048 in check mode so every layer's ao survives).
// One instantiation per TPAD (xs rows): a1_k16 .. a1_k256.
// -DSKV_IFACE=2 (a1bench.py --iface 2): the five trailing kernel arguments (qkv, rr1, qn, kn, rope) are the v2 inputs; q is not read (null-equivalent).
// a1bench.py passes -DA1_EXPECT_SMEM=<its own formula of skv::attend_smem_bytes()> so a mismatch with the header fails the build.
#ifdef A1_EXPECT_SMEM
static_assert(skv::attend_smem_bytes() == (size_t)A1_EXPECT_SMEM, "a1bench.py attend smem formula != skv::attend_smem_bytes()");
#endif
__device__ __forceinline__ unsigned long long a1_gt() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void a1_bar_arrive(unsigned* cnt) {   // production grid barrier (hv2g_common.cuh bar_arrive / bar_wait)
  __syncthreads();
  if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(cnt) : "memory");
}
__device__ __forceinline__ void a1_bar_wait(const unsigned* cnt, unsigned& target) {
  target += gridDim.x;
  if (threadIdx.x == 0) {
    unsigned v;
    do { asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(cnt) : "memory"); } while (v < target);
    asm volatile("fence.acquire.gpu;" ::: "memory");
  }
  __syncthreads();
}

template <int TPAD>
__device__ __forceinline__ void a1_body(const skv::bf16* q, const skv::bf16* kc, const skv::bf16* vc, int cache_rows, int n_layers, const int4* items,
                                        int items_stride, const int* item_off, const int* pair_info, float* part, skv::bf16* ao, const void* tmaps, int engine, int total_q,
                                        const char* pool, int fill_vec, size_t fill_adv, size_t pool_mod, size_t pool_off0, unsigned* bar,
                                        unsigned long long* stamps, unsigned long long* cend, unsigned* sink, int mode, unsigned char* dbg,
                                        size_t ao_stride, const float* qkv, const float* rr1, const skv::bf16* qn, const skv::bf16* kn, const float* rope, unsigned* tickets) {
  extern __shared__ __align__(1024) unsigned char sm[];
  const int tid = threadIdx.x, cta = blockIdx.x, nct = gridDim.x;
#if SKV_TMA_MMA
  if (tid == 0 && (skv::smaddr(sm) & 1023u)) __trap();   // SKV_TMA_MMA: the swizzled ring needs a 1024 B aligned scratch base (offset 0 of the dynamic smem)
#endif
  skv::Params p;
#if SKV_TC
  __shared__ unsigned a1_tm;                                         // SKV_TC: the caller (mk: g.tb) owns TMEM; this bench allocates 256 columns like mk and passes the base
  if ((tid >> 5) == 0) skv::tc_alloc(skv::smaddr(&a1_tm), 256);
  __syncthreads();
  p.tmem = *reinterpret_cast<volatile unsigned*>(&a1_tm);
#endif
  p.q = q; p.kc = kc; p.vc = vc; p.cache_rows = cache_rows; p.n_layers = n_layers; p.items = items; p.items_stride = items_stride; p.item_off = item_off; p.pair_info = pair_info;
  p.part = part; p.ao = ao; p.tmaps = tmaps; p.engine = engine; p.total_q = total_q;
#if SKV_IFACE == 2
  p.qkv = qkv; p.rr1 = rr1; p.qn = qn; p.kn = kn; p.rope = rope;     // iface 2: attend does q/k norm + RoPE + the new-row KV write
#else
  (void)qkv; (void)rr1; (void)qn; (void)kn; (void)rope;
#endif
#if SKV_TICKET
  p.tickets = tickets;                                               // SKV_TICKET: [8 * n_seq] u32, zeroed once per plan by the host, never reset (ao is complete after attend)
#else
  (void)tickets;
#endif
  // O-proj unit of this CTA (128 units over nct CTAs; some CTAs idle)
  const int u0 = (cta * 128) / nct, u1 = ((cta + 1) * 128) / nct;
  const bool own = u1 > u0;
  unsigned ring_phase = 0, target = 0, acc = 0;
  for (int l = 0; l < n_layers; ++l) {
    if (fill_vec) {
      const size_t off = ((pool_off0 + (size_t)l * fill_adv) % pool_mod) & ~(size_t)4095;
      const char* fb = pool + off + (size_t)cta * fill_vec * 16;
      for (int i = tid; i < fill_vec; i += 512 * 4) {
        uint4 v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) { const int idx = i + u * 512; v[u] = idx < fill_vec ? skv::ldg_nc(fb + (size_t)idx * 16) : make_uint4(0, 0, 0, 0); }
#pragma unroll
        for (int u = 0; u < 4; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
      }
    }
    a1_bar_arrive(bar); a1_bar_wait(bar, target);
    if (cta == 0 && tid == 0) stamps[l * 4 + 0] = a1_gt();
    skv::Params pl = p;
    pl.ao = ao + (size_t)l * ao_stride;
    skv::attend<512>(pl, l, sm, ring_phase);
    if (mode & 8) {                                                   // a1ub: per-CTA attend end (all 512 threads out of attend; = the bar_arrive's own __syncthreads)
      __syncthreads();
      if (tid == 0) cend[(size_t)l * nct + cta] = a1_gt();
    }
    a1_bar_arrive(bar); a1_bar_wait(bar, target);
    if (cta == 0 && tid == 0) stamps[l * 4 + 1] = a1_gt();
    if (mode & 1) {
      if (own) {
        skv::combine_head<TPAD, 512>(pl, l, u0 % 16, sm, total_q);
        __syncthreads();
        acc ^= reinterpret_cast<const unsigned*>(sm)[tid];
        if ((mode & 4) && l == n_layers - 1 && u0 < 16) {
          uint4* d = reinterpret_cast<uint4*>(dbg + (size_t)u0 * 2 * TPAD * 128);
          for (int i = tid; i < 2 * TPAD * 8; i += 512) d[i] = reinterpret_cast<const uint4*>(sm)[i];
        }
        __syncthreads();
      }
    }
    if (mode & 2) {
      skv::combine_all_to_ao(pl, l, cta, nct);
      a1_bar_arrive(bar); a1_bar_wait(bar, target);
    }
    if (tid == 0 && !(mode & 8)) cend[(size_t)l * nct + cta] = a1_gt();
  }
  sink[cta * 512 + tid] = acc;
#if SKV_TC
  skv::tc_fence_before_();
  __syncthreads();
  if ((tid >> 5) == 0) { skv::tc_fence_after_(); skv::tc_dealloc(p.tmem, 256); }
#endif
}

#define A1_KERNEL(NP)                                                                                                                                     \
  extern "C" __global__ void __launch_bounds__(512, 1) a1_k##NP(                                                                                          \
      const skv::bf16* q, const skv::bf16* kc, const skv::bf16* vc, int cache_rows, int n_layers, const int4* items, int items_stride, const int* item_off,  \
      const int* pair_info, float* part, skv::bf16* ao, const void* tmaps, int engine, int total_q, const char* pool, int fill_vec, size_t fill_adv,      \
      size_t pool_mod, size_t pool_off0, unsigned* bar, unsigned long long* stamps, unsigned long long* cend, unsigned* sink, int mode,                   \
      unsigned char* dbg, size_t ao_stride, const float* qkv, const float* rr1, const skv::bf16* qn, const skv::bf16* kn, const float* rope, unsigned* tickets) {  \
    a1_body<NP>(q, kc, vc, cache_rows, n_layers, items, items_stride, item_off, pair_info, part, ao, tmaps, engine, total_q, pool, fill_vec, fill_adv, pool_mod,        \
                pool_off0, bar, stamps, cend, sink, mode, dbg, ao_stride, qkv, rr1, qn, kn, rope, tickets);                                                    \
  }
A1_KERNEL(16)
A1_KERNEL(32)
A1_KERNEL(64)
A1_KERNEL(128)
A1_KERNEL(256)
