// A0 unit test of the SKV_TC tcgen05 consumer's descriptors (prepended with splitkv.cuh, -DSKV_TC=1). One CTA, 128 threads.
// K / V tile: two real SWIZZLE_128B TMA boxes each (128 rows x 64 dims) from the [rows, 8, 128] caches; Q and P^T [16][128] bf16 (host) are staged with the production address functions
// (tc_q_off / tc_p_off). Outputs: S^T = K Q^T as fp32 [128 pos][16], O^T = V^T P^T as fp32 [128 dims][16] (TMEM -> registers -> global).
// vlbo / vsbo: byte strides of the MN-major V descriptor (production: 16384 / 1024); swap = 1 exchanges them (descriptor semantics probe).
extern "C" __global__ void __launch_bounds__(128, 1) tc_unit(const void* kmap, const void* vmap, const skv::bf16* q, const skv::bf16* pt, float* sout, float* oout,
                                                              int row0, int kvh, unsigned vlbo, unsigned vsbo, int nk_steps) {
  extern __shared__ __align__(1024) unsigned char sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const unsigned sb = skv::smaddr(sm);
  if (tid == 0 && (sb & 1023u)) __trap();
  const unsigned kA = sb, vA = sb + 32768, qA = sb + 65536, pA = sb + 69632, bars = sb + 73728, slot = sb + 73792;
  if (tid == 0) {
    skv::mb_init(bars, 1); skv::mb_init(bars + 8, 1); skv::mb_init(bars + 16, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 0) skv::tc_alloc(slot, 64);
  // Q / P^T staging (swizzled): thread t < 128: row n = t / 8, 8 chunks of 16 B each... all 16 rows x 16 chunks = 256 chunks (2 per thread)
  for (int i = tid; i < 256; i += 128) {
    const int n = i >> 4, c = i & 15;
    skv::sts128(qA + skv::tc_q_off(n, c), *reinterpret_cast<const uint4*>(q + n * 128 + c * 8));
    // P^T: chunk c of 8 positions of row n
    const uint4 v = *reinterpret_cast<const uint4*>(pt + n * 128 + c * 8);
    const unsigned short* e = reinterpret_cast<const unsigned short*>(&v);
    for (int u = 0; u < 8; ++u) skv::sts16(pA + skv::tc_p_off(n, c * 8 + u), e[u]);
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  const unsigned tm = *reinterpret_cast<volatile unsigned*>(sm + 73792);
  if (tid == 0) {
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"(kmap) : "memory");
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"(vmap) : "memory");
    skv::mb_expect(bars, 65536u);
    skv::tma3d(kA, kmap, 0, kvh, row0, bars);
    skv::tma3d(kA + 16384, kmap, 64, kvh, row0, bars);
    skv::tma3d(vA, vmap, 0, kvh, row0, bars);
    skv::tma3d(vA + 16384, vmap, 64, kvh, row0, bars);
    skv::mb_wait(bars, 0);
    skv::tc_fence_after_();
    constexpr unsigned idS = skv::tc_idesc_(128, 16, false), idV = skv::tc_idesc_(128, 16, true);
    for (int kk = 0; kk < nk_steps; ++kk)
      skv::tc_mma_(tm, skv::tc_kdesc(kA + (kk >> 2) * 16384 + (kk & 3) * 32), skv::tc_kdesc(qA + (kk >> 2) * 2048 + (kk & 3) * 32), idS, kk > 0);
    skv::tc_commit_(bars + 8);
    for (int kk = 0; kk < nk_steps; ++kk)
      skv::tc_mma_(tm + 32, skv::tc_desc_(vA + kk * 2048, vlbo, vsbo), skv::tc_kdesc(pA + (kk >> 2) * 2048 + (kk & 3) * 32), idV, kk > 0);
    skv::tc_commit_(bars + 16);
  }
  __syncwarp();
  skv::mb_wait(bars + 8, 0);
  skv::mb_wait(bars + 16, 0);
  skv::tc_fence_after_();
  unsigned v[16];
  skv::tc_ld16(v, tm + ((unsigned)(warp * 32) << 16));
  skv::tc_wait_ld_();
  for (int c = 0; c < 16; ++c) sout[(warp * 32 + lane) * 16 + c] = __uint_as_float(v[c]);
  skv::tc_ld16(v, tm + 32 + ((unsigned)(warp * 32) << 16));
  skv::tc_wait_ld_();
  for (int c = 0; c < 16; ++c) oout[(warp * 32 + lane) * 16 + c] = __uint_as_float(v[c]);
  skv::tc_fence_before_();
  __syncthreads();
  if (warp == 0) { skv::tc_fence_after_(); skv::tc_dealloc(tm, 64); }
}
