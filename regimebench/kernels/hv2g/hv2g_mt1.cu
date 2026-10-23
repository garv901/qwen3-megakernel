// hv2g MT1: the hv2 kernel for exactly one new token (total_q == 1), generalized from "b1, context 128" to
// "one sequence, runtime kv length, runtime position". Structure, phases, barriers, weight staging and the
// QKV / GU / down code are those of hv2 (default macro set, resolved by hand). What changed:
//  * kv length / cache row are runtime kernel arguments (kvlen, cache_rows, base_row); the cache is the fixture's packed
//    layout [L, cache_rows, KVH, D] with this sequence's rows starting at base_row.
//  * RoPE: `rope` points at the cos/sin row (64 cos, 64 sin) of this token's position (the host adds position * 128).
//  * Fused attention (phase 2) loops over the kv positions in chunks of CTX = 128: chunk 0 is prefetched in the
//    QKV -> fused window exactly as hv2 did for its single chunk; chunk c + 1 is issued by every warp right after
//    it has consumed chunk c's V (V rows of a warp live only in that warp's own cp.async lanes: no __syncthreads).
//    Each warp keeps an online-softmax state (m, l, o) across chunks; the 16 warp states are merged as in hv2.
//    The new token (k, v from smem) is folded into warp 0 in chunk 0 as in hv2. Positions >= kvlen are masked.
//  * the new k / v are written into the cache row base_row + kvlen (instead of k_write / v_write outputs).
// With kvlen == 128 the executed path is hv2's (plus the masks, which are no-ops there).
#if HV2G_MT1
__device__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float t = (l < NW) ? red[l] : 0.f;
    return warp_sum(t);
}

// RMSNorm of the global fp32 residual x (H) with bf16 weight w into smem s (final norm).
__device__ void rms_to_smem(const float* x, const bf16_t* __restrict__ w, float* s, float* red) {
    int t = threadIdx.x;
    float a = __ldcg(x + t), b = __ldcg(x + t + NT);
    float ss = block_sum(a * a + b * b, red);
    float r = rsqrtf(ss / (float)H + EPS);
    s[sperm(t)] = a * r * bf2f(w[t]);
    s[sperm(t + NT)] = b * r * bf2f(w[t + NT]);
    __syncthreads();
}

// Deferred RMSNorm with ONE sync. s = x * g (no r), per-warp partial sums of x^2 in red[0 .. NW); the caller applies
// r = rsqrt(sum(red[0 .. NW)) / H + eps) to the row sums in its store (rmsd_r), after the dot's __syncthreads. The two norm-weight loads are issued
// before any use of a / b, so they overlap the x loads.
// WAR on red[0 .. NW): the writers here come after the barrier of the phase. The previous readers of red[0 .. 32) are the fused phase (attention
// reads, combine) and the previous QKV / GU store; each is followed by this CTA's bar_arrive __syncthreads (and more) before the next phase's code
// runs. The readers of this phase are the stores after the dot's __syncthreads, and the next writer of red (the fused phase, the next norm) comes after
// this CTA's next bar_arrive __syncthreads, so no thread still reads red when it is written. The sync at the end orders the writes before the reads.
template <bool WAIT>
__device__ void rmsd_to_smem(const float* x, const bf16_t* __restrict__ w, float* s, float* red) {
    int t = threadIdx.x;
    float a = __ldcg(x + t), b = __ldcg(x + t + NT);
    const float w0 = bf2f(w[t]), w1 = bf2f(w[t + NT]);  // issued before any use of a / b
    s[sperm(t)] = a * w0;
    s[sperm(t + NT)] = b * w1;
    float v = warp_sum(a * a + b * b);
    if ((t & 31) == 0) red[t >> 5] = v;
    if (WAIT) asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
}

// Load-free poller variant: warp 0 issues nothing; warps 1-15 load their own units (as gemv_load) and
// also cp.async warp 0's units into r8_stage. After the barrier warp 0 runs stage_to_buf.
template <int U, class F>
__device__ __forceinline__ void gemv_load_lf(uint4 (&buf)[U], int nunits, F unit_ptr) {
    static_assert(U * 32 * 16 <= R8_STAGE_BYTES, "stage fits");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp != 0) {
#pragma unroll
        for (int k = 0; k < U; ++k) {
            int u = min(warp + k * NW, nunits - 1);
            buf[k] = __ldg((const uint4*)unit_ptr(u) + lane);
        }
        const int t = threadIdx.x - 32;
        static_assert(U * 32 <= NT - 32, "one stage entry per helper thread");
        if (t < U * 32) {
            const int k = t >> 5, ln = t & 31, u = min(k * NW, nunits - 1);
            cpa16(&r8_stage[t], (const uint4*)unit_ptr(u) + ln);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    } else {  // warp 0: defined (overwritten by stage_to_buf); without this ptxas spills 320 B
#pragma unroll
        for (int k = 0; k < U; ++k) buf[k] = make_uint4(0, 0, 0, 0);
    }
}
template <int U>
__device__ __forceinline__ void stage_to_buf(uint4 (&buf)[U]) {
    if ((threadIdx.x >> 5) == 0) {
#pragma unroll
        for (int k = 0; k < U; ++k) buf[k] = r8_stage[k * 32 + (threadIdx.x & 31)];
    }
}

// N rows in flight per warp: rows r0 + j*step (j < N), valid when < lim. All N*4 loads are issued first (invalid rows read
// row 0 and are dropped), then N dots (same sum order as row_dot) and N warp_sums; lane 0 writes the logits and updates
// (best, besti) in ascending row order.
template <int N>
__device__ __forceinline__ void lm_group(const bf16_t* __restrict__ embed, int r0, int step, int lim, const float* s,
                                         float* logits, float& best, int& besti) {
    const int lane = threadIdx.x & 31;
    uint4 u[N][H / 256];
#pragma unroll
    for (int j = 0; j < N; ++j) {
        const int row = r0 + j * step;
        const uint4* wr = (const uint4*)(embed + (size_t)(row < lim ? row : 0) * H);
#pragma unroll
        for (int it = 0; it < H / 256; ++it) u[j][it] = __ldcs(wr + it * 32 + lane);
    }
#pragma unroll
    for (int j = 0; j < N; ++j) {
        float acc = 0.f;
#pragma unroll
        for (int it = 0; it < H / 256; ++it) {
            const float* xs = s + it * 256 + lane * 4;  // permuted layout (sperm)
            float4 a = *(const float4*)xs, b = *(const float4*)(xs + 128);
            const uint4 v = u[j][it];
            acc += bflo(v.x) * a.x + bfhi(v.x) * a.y + bflo(v.y) * a.z + bfhi(v.y) * a.w
                 + bflo(v.z) * b.x + bfhi(v.z) * b.y + bflo(v.w) * b.z + bfhi(v.w) * b.w;
        }
        const float d = warp_sum(acc);
        const int row = r0 + j * step;
        if (lane == 0 && row < lim) {
            logits[row] = d;
            if (d > best) { best = d; besti = row; }
        }
    }
}

extern "C" __global__ void __launch_bounds__(NT, 1) qwen3_step_mt1(
    const long long* __restrict__ token,
    const bf16_t* __restrict__ ln1, const bf16_t* __restrict__ wq,
    const bf16_t* __restrict__ wk, const bf16_t* __restrict__ wv,
    const bf16_t* __restrict__ qn, const bf16_t* __restrict__ kn,
    const bf16_t* __restrict__ wo, const bf16_t* __restrict__ ln2,
    const bf16_t* __restrict__ wg, const bf16_t* __restrict__ wu,
    const bf16_t* __restrict__ wd, const bf16_t* __restrict__ fnorm,
    const bf16_t* __restrict__ embed, bf16_t* kcache, bf16_t* vcache,
    float* __restrict__ logits, long long* __restrict__ next_token,
    float* scratch, unsigned* sync, float* pval, int* pidx, const float* __restrict__ rope,
    const int kvlen, const int cache_rows, const int base_row)
{
    __shared__ __align__(16) float sbuf[INTER];
    __shared__ float red[32];
    __shared__ int redi[32];
    __shared__ float part[PART_N];
    __shared__ __align__(16) bf16_t sVc[CTX * HD];  // fused attention: V-cache rows of kv-head fg (one chunk)
    __shared__ __align__(8) unsigned long long gu_mbar;  // completes (phase l) when layer l's gate / up rows have landed in sGU
    __shared__ __align__(8) unsigned long long d_mbar;   // completes (phase l) when layer l's down rows have landed in sD

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int gwarp = blockIdx.x * NW + warp;
    const int nwarps = gridDim.x * NW;
    unsigned* bar_cnt = sync;
    unsigned bar_target = 0;
    unsigned* ticket = sync + 2;

    float* x = scratch;
    float* qkv = scratch + H;
    float* hid = scratch + H + 4096;

    // Per-phase row ownership and weight register buffers. Each phase's weights
    // are issued between the arrival and the wait of the grid barrier that
    // precedes the phase, so their DRAM round trip overlaps barrier latency.
    int rq0, nq, rh0, nh, ri0, ni;
    row_range(4096, rq0, nq);
    row_range(H, rh0, nh);
    row_range(INTER, ri0, ni);
    uint4 bq[UMAX(4096, 4)], bgu[UMAX(INTER, 8)], bd[UMAX(H, 12)];
    // fused attention + split-K O-proj ownership: q-head fh (kv-head fg), Wo rows [fr0, fr0+fn)
    const bool fused = blockIdx.x < QH * NSL;
    // recomputed from blockIdx at each use (as const ints held across the layer loop they spill)
#define fh ((int)blockIdx.x % QH)
#define fs ((int)blockIdx.x / QH)
#define fg (fh >> 1)
#define fr0 (fs * NQD / NSL * 4)
#define fn ((fs + 1) * NQD / NSL * 4 - fr0)
    // element offset of (layer l, kv position p, kv head fg, dim 0). Positions >= kvlen of the last chunk are read too (no clamp: it costs registers,
    // hv2 has none to spare) and masked: the host pads every layer of the cache with >= CTX finite rows (zeros) after the last row, so these
    // reads stay in bounds and see finite values (0 * finite = 0 in the masked softmax terms).
#define KVE(l, p) (((size_t)(l) * cache_rows + base_row + (int)(p)) * (KVH * HD) + fg * HD)
    static_assert((NQD + NSL - 1) / NSL * 4 * (HD / 8) <= WOU * NT, "Wo slice fits WOU uint4 per thread");
    static_assert((WOU - 1) * NT / 16 <= NQD / NSL * 4, "only the last Wo uint4 can fall past a slice");
    static_assert(PPW * NW == CTX, "positions split evenly over warps");

    auto load_qkv = [&](int l) {
        gemv_load_lf(bq, nq * 4, [&](int u) -> const bf16_t* {
            int r = rq0 + (u >> 2), sg = (u & 3) * 256;
            if (r < 2048) return wq + ((size_t)l * 2048 + r) * H + sg;
            if (r < 3072) return wk + ((size_t)l * 1024 + (r - 2048)) * H + sg;
            return wv + ((size_t)l * 1024 + (r - 3072)) * H + sg;
        });
    };

    // Gate / up rows [ri0, ri0 + ni) of layer ll as 2 TMA bulk copies into dynamic smem behind the stage: sGU[0 .. ni * H) gate rows, sGU[NIMAX * H ..) up
    // rows (NIMAX = 21: 2 * 21 * 1024 * 2 B = 86016 B). Issued by thread 128 of every CTA in the fused phase (right after the prep __syncthreads for fused
    // CTAs, at the fused start for the others). Layer l is mbarrier phase l (parity l & 1); the GU phase waits for it before its first sGU read.
    // WAR: layer l + 1's copy overwrites sGU. The last generic reads of layer l's rows are in the GU phase of layer l (the bgu fill, before the dot). At
    // least 3 __syncthreads (dot, GU -> down barrier, down phase, down -> QKV barrier) come before layer l + 1's issue, then fence.proxy.async.
    static_assert(2 * CDIV(INTER, NB) * H * 2 <= 86016, "sGU fits the 86016 B of dynamic smem");
#define R10_NIMAX CDIV(INTER, NB)
    bf16_t* const sGU = (bf16_t*)(r8_stage + R8_STAGE_BYTES / 16);
    auto issue_gue = [&](int ll) {
        if (tid == 128) {
            const unsigned mb = (unsigned)__cvta_generic_to_shared(&gu_mbar);
            const unsigned dst_gate = (unsigned)__cvta_generic_to_shared(sGU);
            const unsigned dst_up = (unsigned)__cvta_generic_to_shared(sGU + R10_NIMAX * H);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(mb), "r"(2 * ni * H * 2) : "memory");
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                         :: "r"(dst_gate), "l"(wg + ((size_t)ll * INTER + ri0) * H), "r"(ni * H * 2), "r"(mb) : "memory");
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                         :: "r"(dst_up), "l"(wu + ((size_t)ll * INTER + ri0) * H), "r"(ni * H * 2), "r"(mb) : "memory");
        }
    };
    // GU phase, every thread, just before the dot: wait for layer l's copy, then bgu[k] from sGU (u = min(warp + k * NW, ni * 8 - 1):
    // row u >> 3, matrix (u >> 2) & 1, segment u & 3; the lane's uint4 at + lane).
    auto gu_fill = [&](int ll) {
        unsigned ok;
        do {
            asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                         : "=r"(ok) : "r"((unsigned)__cvta_generic_to_shared(&gu_mbar)), "r"(ll & 1) : "memory");
        } while (!ok);
#pragma unroll
        for (int k = 0; k < UMAX(INTER, 8); ++k) {
            const int u = min(warp + k * NW, ni * 8 - 1);
            bgu[k] = ((const uint4*)(sGU + ((u >> 2) & 1) * (R10_NIMAX * H) + (size_t)(u >> 3) * H + (u & 3) * 256))[lane];
        }
    };
    // Down rows [rh0, rh0 + nh) of layer ll (contiguous in wd: INTER bf16 per row) as ONE bulk copy into sD (dynamic smem behind sGU, NHMAX = 7 rows,
    // 43008 B), by thread 128, in the GU phase right after the GU norm's __syncthreads. d_mbar phase l = layer l (wait parity l & 1).
    static_assert(CDIV(H, NB) * INTER * 2 <= 43008, "sD fits 43008 B");
    bf16_t* const sD = (bf16_t*)(r8_stage + (R8_STAGE_BYTES + 86016) / 16);
    auto issue_dne = [&](int ll) {
        if (tid == 128) {
            const unsigned mb = (unsigned)__cvta_generic_to_shared(&d_mbar);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(mb), "r"(nh * INTER * 2) : "memory");
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                         :: "r"((unsigned)__cvta_generic_to_shared(sD)), "l"(wd + ((size_t)ll * H + rh0) * INTER), "r"(nh * INTER * 2), "r"(mb) : "memory");
        }
    };
    // down phase, every thread, just before the dot: wait, then bd[k] from sD (u = min(warp + k * NW, nh * 12 - 1): row u / 12, segment u % 12)
    auto d_fill = [&](int ll) {
        unsigned ok;
        do {
            asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                         : "=r"(ok) : "r"((unsigned)__cvta_generic_to_shared(&d_mbar)), "r"(ll & 1) : "memory");
        } while (!ok);
#pragma unroll
        for (int k = 0; k < UMAX(H, 12); ++k) {
            const int u = min(warp + k * NW, nh * 12 - 1);
            bd[k] = ((const uint4*)(sD + (size_t)(u / 12) * INTER + (u % 12) * 256))[lane];
        }
    };

    __shared__ float s_rope[2 * 64];  // rope cos / sin of this token's position, constant over layers; ordered by the prologue bar_arrive
    if (tid < 2 * 64) s_rope[tid] = rope[tid];
    // L2 prefetch of the next norm rows (16 x 128 B lines each) and of q/k norm weights (2 x 256 B), warp 1 only; warp 0 is the poller
    auto pf_row = [&](const bf16_t* p) {
        if (warp == 1 && lane < 16) asm volatile("prefetch.global.L2 [%0];" :: "l"(p + lane * 64));
    };
    auto pf_qkn = [&](int l) {
        if (warp == 1 && lane >= 16 && lane < 20) {
            const bf16_t* p = (lane < 18 ? qn : kn) + l * HD + (lane & 1) * 64;
            asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
        }
    };

    // x = embed[token]
    {
        long long tok = token[0];
        const bf16_t* e = embed + tok * H;
        for (int i = blockIdx.x * NT + tid; i < H; i += gridDim.x * NT)
            __stcg(x + i, bf2f(e[i]));
    }
    if (tid == 0) {  // before the prologue bar_arrive, whose __syncthreads makes the init visible to every thread
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"((unsigned)__cvta_generic_to_shared(&gu_mbar)) : "memory");
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"((unsigned)__cvta_generic_to_shared(&d_mbar)) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    bar_arrive(bar_cnt);
    load_qkv(0);
    pf_row(ln1);
    pf_qkn(0);
    bar_wait<R8_SPIN_PRO>(bar_cnt, bar_target);

    for (int l = 0; l < NL; ++l) {
        // ---- Phase 1: RMSNorm(ln1) + fused QKV GEMV (4096 rows) ----
        float act[1][8];  // logical elements lane*8..+7 of segment warp & 3 (u & 3 == warp & 3 as NW = 16)
        rmsd_to_smem<true>(x, ln1 + l * H, sbuf, red);
        stage_to_buf(bq);
        smem_act(sbuf, warp & 3, act[0]);
        gemv_dot_reg(bq, nq * 4, act, part);
        __syncthreads();
        if (tid < nq) {  // red[0 .. NW) is read here, before this CTA's bar_arrive; its next writer is after that barrier (rmsd_to_smem)
            const float r = rmsd_r(red);
            __stcg(qkv + rq0 + tid, r * (part[tid * 4] + part[tid * 4 + 1] + part[tid * 4 + 2] + part[tid * 4 + 3]));
        }
        bar_arrive(bar_cnt);
        // defined on every path: a fused-only definition keeps them live across the layer loop
        uint4 bw[WOU] = {};
        uint2 kk[PPW] = {};
        if (fused && warp != 0) {  // group A: warp 0's Wo / K / V entries (virtual thread vt = lane of warp 0; chunk 0 positions 0 .. PPW - 1)
            static_assert(WOU * 32 * 16 + PPW * 32 * 8 <= R8_STAGE_BYTES, "stage fits");
            for (int e = tid - 32; e < (WOU + 2 * PPW) * 32; e += NT - 32) {
                if (e < WOU * 32) {
                    const int k = e >> 5, vt = e & 31;
                    const bf16_t* wb0 = wo + ((size_t)l * H + fr0 + (vt >> 4)) * 2048 + fh * HD + (vt & 15) * 8;
                    cpa16(&r8_stage[e], (const uint4*)(k < WOU - 1 || (vt >> 4) + k * (NT / 16) < fn ? wb0 + k * (NT / 16) * 2048 : wb0));
                } else if (e < (WOU + PPW) * 32) {
                    const int e2 = e - WOU * 32, j = e2 >> 5, vt = e2 & 31;
                    cpa8(&((uint2*)(r8_stage + WOU * 32))[e2], kcache + KVE(l, j) + vt * 4);
                } else {
                    const int e2 = e - (WOU + PPW) * 32, j = e2 >> 5, vt = e2 & 31;
                    cpa8((char*)sVc + (vt * 4 + j * HD) * 2, vcache + KVE(l, j) + vt * 4);
                }
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        if (fused && warp != 0) {  // own Wo slice, K and V-cache loads as in V1q (group B)
            // thread t, uint4 k: slice row t/16 + k*NT/16, head columns (t%16)*8..+7 (only the
            // last k can fall past the slice; it then re-loads row t/16, unused)
            const bf16_t* wb = wo + ((size_t)l * H + fr0 + (tid >> 4)) * 2048 + fh * HD + (tid & 15) * 8;
#pragma unroll
            for (int k = 0; k < WOU; ++k)
                bw[k] = __ldg((const uint4*)(k < WOU - 1 || (tid >> 4) + k * (NT / 16) < fn ? wb + k * (NT / 16) * 2048 : wb));
            // warp w's cached positions w*PPW..+PPW-1: K into registers, V into smem by cp.async
            // (each lane copies, and later reads, only its own dims lane*4..+3)
            const unsigned dst = (unsigned)__cvta_generic_to_shared(sVc + warp * PPW * HD + lane * 4);
#pragma unroll
            for (int j = 0; j < PPW; ++j) {
                const size_t off = KVE(l, warp * PPW + j) + lane * 4;
                kk[j] = __ldg((const uint2*)(kcache + off));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(dst + j * HD * 2),
                             "l"(vcache + off) : "memory");
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        // warp 0: qn, warp 1: kn of this layer, lane's dims lane*4..+3 as one uint2; used after the barrier in the prep
        uint2 nwq = make_uint2(0, 0);
        if (fused && (unsigned)warp < 2u) nwq = __ldg((const uint2*)((warp == 0 ? qn : kn) + l * HD + lane * 4));
        bar_wait<R8_SPIN_QKV>(bar_cnt, bar_target);
        if (!fused) issue_gue(l);  // layer l's gate / up rows (fused CTAs: after their prep __syncthreads)

        // ---- Phase 2: fused attention (q-head fh, all kvlen + 1 positions, in-block softmax over 128-position chunks) + split-K O-proj of
        // Wo rows [fr0, fr0+fn) x head-fh columns. Each block red.adds its partial row sums into x; every block read x for ln1
        // before the barrier above, and the barrier below orders the adds before ln2. ----
        if (fused) {
            float* sq = sbuf;            // 128: post-norm, post-RoPE q of head fh
            float* sk = sbuf + HD;       // 128: new k
            float* sv = sbuf + 2 * HD;   // 128: new v
            float* soh = sbuf + 3 * HD;  // 128: normalized o of head fh
            float* sow = sbuf + 4 * HD;  // NW x 128 per-warp unnormalized o
            static_assert(4 * HD + NW * HD <= INTER, "attention smem");
            const bool wr = (fh & 1) == 0 && fs == 0;  // writes the new k / v of kv-head fg into the cache
            const size_t newrow = (((size_t)l * cache_rows + base_row + kvlen) * KVH + fg) * HD;
            // prep roles: warp 0: q head fh, 1: k of kv-head fg, 2: v copy
            if ((unsigned)warp < 2u) {
                // q head fh / k of kv-head fg. lane holds dims lane*4..+3
                const float* src = warp == 0 ? qkv + fh * HD : qkv + 2048 + fg * HD;
                float4 a = __ldcg((const float4*)src + lane);
                float r = rsqrtf(warp_sum(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w) / (float)HD + EPS);
                float e[4] = {a.x, a.y, a.z, a.w}, o[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    e[i] = e[i] * r * bf2f((i < 2 ? nwq.x : nwq.y) >> ((i & 1) * 16) & 0xffffu);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float pr = __shfl_xor_sync(0xffffffffu, e[i], 16);
                    int j = (lane * 4 + i) & 63;
                    float cs = s_rope[j], sn = s_rope[64 + j];
                    o[i] = e[i] * cs + (lane < 16 ? -pr : pr) * sn;
                }
                *(float4*)((warp == 0 ? sq : sk) + lane * 4) = make_float4(o[0], o[1], o[2], o[3]);
                if (warp == 1 && wr) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) kcache[newrow + lane * 4 + i] = f2bf(o[i]);
                }
            } else if (warp == 2) {
                float4 a = __ldcg((const float4*)(qkv + 3072 + fg * HD) + lane);
                *(float4*)(sv + lane * 4) = a;
                if (wr) {
                    vcache[newrow + lane * 4 + 0] = f2bf(a.x);
                    vcache[newrow + lane * 4 + 1] = f2bf(a.y);
                    vcache[newrow + lane * 4 + 2] = f2bf(a.z);
                    vcache[newrow + lane * 4 + 3] = f2bf(a.w);
                }
            }
            asm volatile("cp.async.wait_group 1;" ::: "memory");
            __syncthreads();
            issue_gue(l);  // fused CTAs: right after the prep __syncthreads (the fused phase's own qkv read has arrived)
            if (warp == 0) {
#pragma unroll
                for (int k = 0; k < WOU; ++k) bw[k] = r8_stage[k * 32 + lane];
#pragma unroll
                for (int j = 0; j < PPW; ++j) kk[j] = ((const uint2*)(r8_stage + WOU * 32))[j * 32 + lane];
            }
            // warp w: cached positions c*CTX + w*PPW..+PPW-1 (K in registers, V in smem); warp 0 also the new token (chunk 0)
            {
                const float4 q = *(const float4*)(sq + lane * 4);
                float m, lsum;
                float4 o;
                attn_chunk<true>(kk, q, kvlen < CTX ? kvlen : CTX, sVc, sk, sv, m, lsum, o);
                // chunk c = 1 .. : K into registers, V by cp.async (own lanes only), issued right after chunk c - 1's V was consumed
                auto issue_chunk = [&](int c) {
                    const unsigned dstv = (unsigned)__cvta_generic_to_shared(sVc + warp * PPW * HD + lane * 4);
#pragma unroll
                    for (int j = 0; j < PPW; ++j) {
                        const size_t off = KVE(l, c * CTX + warp * PPW + j) + lane * 4;
                        kk[j] = __ldg((const uint2*)(kcache + off));
                        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(dstv + j * HD * 2),
                                     "l"(vcache + off) : "memory");
                    }
                    asm volatile("cp.async.commit_group;" ::: "memory");
                };
#define NCH ((kvlen + CTX - 1) / CTX)
                if (NCH > 1) issue_chunk(1);
#pragma unroll 1
                for (int c = 1; c < NCH; ++c) {
                    float m2, l2;
                    float4 o2;
                    attn_chunk<false>(kk, q, kvlen - c * CTX, sVc, sk, sv, m2, l2, o2);
                    if (c + 1 < NCH) issue_chunk(c + 1);
                    attn_merge(m, lsum, o, m2, l2, o2);
                }
                *(float4*)(sow + warp * HD + lane * 4) = o;
                if (lane == 0) { red[warp] = m; red[NW + warp] = lsum; }
            }
            __syncthreads();
            if (tid < HD) {
                float M = red[0];
#pragma unroll
                for (int w = 1; w < NW; ++w) M = fmaxf(M, red[w]);
                float L = 0.f, o = 0.f;
#pragma unroll
                for (int w = 0; w < NW; ++w) {
                    const float c = R11_EXPF(red[w] - M);
                    L += c * red[NW + w];
                    o += c * sow[w * HD + tid];
                }
                soh[tid] = o / L;
            }
            __syncthreads();
            // split-K O-proj: bw[k] is row fr0 + tid/16 + k*NT/16, head columns (tid%16)*8..+7;
            // 16 consecutive lanes share a row and reduce by shuffles into the row partial
            // (slice-local row index, in the dead sow region); then one 16-byte red.add per quad
            const float4 a = *(const float4*)(soh + (tid & 15) * 8), b = *(const float4*)(soh + (tid & 15) * 8 + 4);
            float* srow = sow;
#pragma unroll
            for (int k = 0; k < WOU; ++k) {
                const uint4 w = bw[k];
                float d = bflo(w.x) * a.x + bfhi(w.x) * a.y + bflo(w.y) * a.z + bfhi(w.y) * a.w
                        + bflo(w.z) * b.x + bfhi(w.z) * b.y + bflo(w.w) * b.z + bfhi(w.w) * b.w;
#pragma unroll
                for (int o = 8; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                if ((tid & 15) == 0) srow[(tid >> 4) + k * (NT / 16)] = d;
            }
            __syncthreads();
            const int q0 = ctaid_opaque() / QH * NQD / NSL, nq4 = (ctaid_opaque() / QH + 1) * NQD / NSL - q0;
            if (tid < nq4) {
                const float4 d = *(const float4*)(srow + tid * 4);
                asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(
                             x + (q0 + tid) * 4),
                             "f"(d.x), "f"(d.y), "f"(d.z), "f"(d.w) : "memory");
            }
        }
        bar_arrive(bar_cnt);
        // no GU weight load and no cp.async group in this window (the rows were issued as bulk copies in the fused phase; bgu is filled in the GU phase)
        pf_row(ln2 + l * H);
        bar_wait<R8_SPIN_FUSED>(bar_cnt, bar_target);

        // ---- Phase 4: RMSNorm(ln2) + gate/up GEMV + SiLU*up ----
        rmsd_to_smem<false>(x, ln2 + l * H, sbuf, red);
        issue_dne(l);  // layer l's down rows: right after the GU norm's __syncthreads (x has arrived for every thread)
        smem_act(sbuf, warp & 3, act[0]);
        gu_fill(l);
        gemv_dot_reg(bgu, ni * 8, act, part);
        __syncthreads();
        if (tid < ni) {
            const float* pp = part + tid * 8;
            const float r = rmsd_r(red);  // red[0 .. NW) is read before this CTA's bar_arrive; the next writer (fused phase / next QKV norm) is after it
            float gt = r * (pp[0] + pp[1] + pp[2] + pp[3]);
            float up = r * (pp[4] + pp[5] + pp[6] + pp[7]);
            __stcg(hid + ri0 + tid, R11_SILU(gt) * up);
        }
        bar_arrive(bar_cnt);
        // no down load and no cp.async group in the GU -> down window (the rows are a bulk copy into sD; bd is filled in the down phase)
        // x[rh0 ..] is final here (fused red.adds are before the fused->GU barrier; the down store is this CTA's own, after this barrier)
        static_assert(CDIV(H, NB) <= 32, "nh rows fit warp 1");
        float xres = 0.f;
        if (tid >= 32 && tid < 32 + nh) xres = __ldcg(x + rh0 + tid - 32);
        bar_wait<R8_SPIN_GU>(bar_cnt, bar_target);

        // ---- Phase 5: down GEMV + residual ----
        // hid is 768 float4: issue both loads of a thread before any store
        {
            float4 h0 = __ldcg((const float4*)hid + tid);
            float4 h1 = tid < INTER / 4 - NT ? __ldcg((const float4*)hid + NT + tid) : make_float4(0.f, 0.f, 0.f, 0.f);
            *((float4*)sbuf + sperm4(tid)) = h0;
            if (tid < INTER / 4 - NT) *((float4*)sbuf + sperm4(NT + tid)) = h1;
        }
        __syncthreads();
        d_fill(l);  // just before the dot
        float act3[3][8];
#pragma unroll
        for (int j = 0; j < 3; ++j) smem_act(sbuf, (warp + 4 * j) % 12, act3[j]);
        gemv_dot_reg(bd, nh * 12, act3, part);
        __syncthreads();
        if (tid >= 32 && tid < 32 + nh) {  // warp 1 holds the old x rows (loaded in the window)
            float d = 0.f;
#pragma unroll
            for (int j = 0; j < 12; ++j) d += part[(tid - 32) * 12 + j];
            __stcg(x + rh0 + tid - 32, xres + d);
        }
        bar_arrive(bar_cnt);
        if (l + 1 < NL) load_qkv(l + 1);
        if (l + 1 < NL) { pf_row(ln1 + (l + 1) * H); pf_qkn(l + 1); }
        bar_wait<R8_SPIN_DOWN>(bar_cnt, bar_target);
    }

    // ---- final norm + LM head (tied embed) + argmax ----
    rms_to_smem(x, fnorm, sbuf, red);
    float best = -INFINITY;
    int besti = 0x7fffffff;
    for (int r = gwarp; r < VOCAB; r += nwarps * 2)
        lm_group<2>(embed, r, nwarps, VOCAB, sbuf, logits, best, besti);
    __syncthreads();
    if (lane == 0) { red[warp] = best; redi[warp] = besti; }
    __syncthreads();
    if (tid == 0) {
        float bv = red[0]; int bi = redi[0];
        for (int w = 1; w < NW; ++w)
            if (red[w] > bv || (red[w] == bv && redi[w] < bi)) { bv = red[w]; bi = redi[w]; }
        __stcg(pval + blockIdx.x, bv);
        __stcg(pidx + blockIdx.x, bi);
        __threadfence();
        unsigned t = atomicAdd(ticket, 1u);
        redi[0] = (t == gridDim.x - 1);
    }
    __syncthreads();
    if (redi[0]) {
        __threadfence();
        float bv = -INFINITY; int bi = 0x7fffffff;
        for (int b = tid; b < (int)gridDim.x; b += NT) {
            const float v = __ldcg(pval + b); const int i = __ldcg(pidx + b);
            if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        __syncthreads();                       // every thread has read redi[0] before it is overwritten
        if (lane == 0) { red[warp] = bv; redi[warp] = bi; }
        __syncthreads();
        if (tid == 0) {
            float v0 = red[0]; int i0 = redi[0];
            for (int w = 1; w < NW; ++w)
                if (red[w] > v0 || (red[w] == v0 && redi[w] < i0)) { v0 = red[w]; i0 = redi[w]; }
            next_token[0] = (long long)i0;
            atomicExch(ticket, 0u);
            atomicExch(bar_cnt, 0u);  // every block has passed its last barrier poll
        }
    }
}
#undef fh
#undef fs
#undef fg
#undef fr0
#undef fn
#undef KVE
#endif  // HV2G_MT1
