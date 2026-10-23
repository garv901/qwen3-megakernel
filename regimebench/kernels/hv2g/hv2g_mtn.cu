// hv2g MTN: general kernel for ntok >= 1 new tokens of any number of sequences (also correct for ntok == 1, just slower than MT1).
// Same persistent cooperative skeleton as hv2 / MT1 (148 CTAs x 512 threads, monotonic grid barrier, register-resident weights loaded in the
// window between a barrier's arrive and wait, deferred RMSNorm, transpose-butterfly dots) with these differences:
//  * every GEMV streams each weight tile ONCE per launch (into registers) and applies it to all ntok tokens, in token blocks (the
//    activations of a block are staged in dynamic smem: TB = 8 tokens for QKV / GU, 4 for O-proj, 2 for down);
//  * 6 grid barriers per layer: QKV | K-norm + RoPE + cache write | attention (token, q-head) items | O-proj (own GEMV, K = 2048) | GU | down;
//  * all new K / V rows are in the cache before the attention barrier, so attention treats every visible position (including the token's own
//    row and the earlier new tokens of the same sequence: causal) uniformly from the cache;
//  * per-token descriptors: t_base (first cache row of the token's sequence), t_vis (number of visible rows = position + 1; the token's K/V row
//    is t_base + t_vis - 1 and its RoPE position t_vis - 1);
//  * LM head + argmax only for the tokens listed in lm_tok (local token index) / lm_seq (output slot).
// Positions past the visible range (last attention chunk) are clamped to the last visible row and masked.
#if !HV2G_MT1
#define TBQ 8   // token block of QKV / GU / LM head
#define TBO 4   // token block of O-proj (K = 2048)
#define TBD 2   // token block of down (K = 3072)
#define DYN_SMEM_MTN (TBQ * H * 4)
static_assert(TBO * 2048 * 4 <= DYN_SMEM_MTN && TBD * INTER * 4 <= DYN_SMEM_MTN, "token blocks fit the dynamic smem");

// x * g (deferred norm: r applied after the dot) of nt token rows (global fp32 rows x + t * H) into sbuf rows tb * H, partial sums of x^2 into red[tb * NW + warp]
__device__ __forceinline__ void stage_norm(const float* x, const bf16_t* __restrict__ g, int t0, int nt, float* sbuf, float* red) {
    const int tid = threadIdx.x;
    const float w0 = bf2f(g[tid]), w1 = bf2f(g[tid + NT]);
    for (int tb = 0; tb < nt; ++tb) {
        const float a = __ldcg(x + (size_t)(t0 + tb) * H + tid), b = __ldcg(x + (size_t)(t0 + tb) * H + tid + NT);
        sbuf[tb * H + sperm(tid)] = a * w0;
        sbuf[tb * H + sperm(tid + NT)] = b * w1;
        const float v = warp_sum(a * a + b * b);
        if ((tid & 31) == 0) red[tb * NW + (tid >> 5)] = v;
    }
}
// plain copy of nt rows of K floats (src + (t0 + tb) * K) into sbuf rows tb * K in the sperm layout
template <int K>
__device__ __forceinline__ void stage_plain(const float* src, int t0, int nt, float* sbuf) {
    for (int tb = 0; tb < nt; ++tb)
        for (int i = threadIdx.x; i < K / 4; i += NT)
            *((float4*)(sbuf + tb * K) + sperm4(i)) = __ldcg((const float4*)(src + (size_t)(t0 + tb) * K) + i);
}

// several tokens per weight row group: N embed rows in flight per warp, SB tokens each
template <int N>
__device__ __forceinline__ void lm_group_n(const bf16_t* __restrict__ embed, int r0, int step, int lim, const float* s, int nt,
                                           const float* lmr, const int* lseq, float* logits, float (&best)[TBQ], int (&besti)[TBQ]) {
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
        const int row = r0 + j * step;
#pragma unroll
        for (int tb = 0; tb < TBQ; ++tb) {
            if (tb < nt) {
                float acc = 0.f;
#pragma unroll
                for (int it = 0; it < H / 256; ++it) {
                    const float* xs = s + tb * H + it * 256 + lane * 4;
                    const float4 a = *(const float4*)xs, b = *(const float4*)(xs + 128);
                    const uint4 v = u[j][it];
                    acc += bflo(v.x) * a.x + bfhi(v.x) * a.y + bflo(v.y) * a.z + bfhi(v.y) * a.w
                         + bflo(v.z) * b.x + bfhi(v.z) * b.y + bflo(v.w) * b.z + bfhi(v.w) * b.w;
                }
                const float d = warp_sum(acc) * lmr[tb];
                if (lane == 0 && row < lim) {
                    logits[(size_t)lseq[tb] * VOCAB + row] = d;
                    if (d > best[tb]) { best[tb] = d; besti[tb] = row; }
                }
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(NT, 1) qwen3_step_mtn(
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
    const int* __restrict__ t_base, const int* __restrict__ t_vis,
    const int* __restrict__ lm_tok, const int* __restrict__ lm_seq,
    const int ntok, const int n_lm, const int cache_rows)
{
    __shared__ float red[TBQ * NW];
    __shared__ float part[TBQ * PART_N];
    __shared__ __align__(16) bf16_t sVc[CTX * HD];
    __shared__ float lmr[TBQ];
    __shared__ int lseq_s[TBQ];
    __shared__ int s_flag;
    float* const sbuf = (float*)r8_stage;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int gwarp = blockIdx.x * NW + warp;
    const int nwarps = gridDim.x * NW;
    unsigned* bar_cnt = sync;
    unsigned bar_target = 0;
    unsigned* ticket = sync + 2;

    float* x = scratch;
    float* qkv = x + (size_t)MAXTOK * H;
    float* hid = qkv + (size_t)MAXTOK * 4096;
    float* ao = hid + (size_t)MAXTOK * INTER;

    int rq0, nq, rh0, nh, ri0, ni;
    row_range(4096, rq0, nq);
    row_range(H, rh0, nh);
    row_range(INTER, ri0, ni);
    uint4 bq[UMAX(4096, 4)], bo[UMAX(H, 8)], bgu[UMAX(INTER, 8)], bd[UMAX(H, 12)];

    auto load_q = [&](int l) {
        gemv_load(bq, nq * 4, [&](int u) -> const bf16_t* {
            int r = rq0 + (u >> 2), sg = (u & 3) * 256;
            if (r < 2048) return wq + ((size_t)l * 2048 + r) * H + sg;
            if (r < 3072) return wk + ((size_t)l * 1024 + (r - 2048)) * H + sg;
            return wv + ((size_t)l * 1024 + (r - 3072)) * H + sg;
        });
    };
    auto load_o = [&](int l) {
        gemv_load(bo, nh * 8, [&](int u) -> const bf16_t* { return wo + ((size_t)l * H + rh0 + (u >> 3)) * 2048 + (u & 7) * 256; });
    };
    auto load_gu = [&](int l) {
        gemv_load(bgu, ni * 8, [&](int u) -> const bf16_t* {
            return ((u >> 2) & 1 ? wu : wg) + ((size_t)l * INTER + ri0 + (u >> 3)) * H + (u & 3) * 256;
        });
    };
    auto load_d = [&](int l) {
        gemv_load(bd, nh * 12, [&](int u) -> const bf16_t* { return wd + ((size_t)l * H + rh0 + u / 12) * INTER + (u % 12) * 256; });
    };

    // x[t] = embed[token[t]]
    for (int i = blockIdx.x * NT + tid; i < ntok * H; i += gridDim.x * NT)
        __stcg(x + i, bf2f(embed[(size_t)token[i / H] * H + (i % H)]));
    bar_arrive(bar_cnt);
    load_q(0);
    bar_wait<0>(bar_cnt, bar_target);

    for (int l = 0; l < NL; ++l) {
        // ---- Phase 1: RMSNorm(ln1) + QKV GEMV, all tokens ----
        for (int t0 = 0; t0 < ntok; t0 += TBQ) {
            const int nt = min(TBQ, ntok - t0);
            stage_norm(x, ln1 + l * H, t0, nt, sbuf, red);
            __syncthreads();
            for (int tb = 0; tb < nt; ++tb) {
                float act[1][8];
                smem_act(sbuf + tb * H, warp & 3, act[0]);
                gemv_dot_reg(bq, nq * 4, act, part + tb * PART_N);
            }
            __syncthreads();
            for (int i = tid; i < nt * nq; i += NT) {
                const int tb = i / nq, rr = i - tb * nq;
                const float* p = part + tb * PART_N + rr * 4;
                __stcg(qkv + (size_t)(t0 + tb) * 4096 + rq0 + rr, rmsd_r(red + tb * NW) * (p[0] + p[1] + p[2] + p[3]));
            }
            __syncthreads();
        }
        bar_arrive(bar_cnt);
        bar_wait<0>(bar_cnt, bar_target);

        // ---- Phase 2: k-norm + RoPE of k, cache write of the new k / v; one warp per (token, kv-head) ----
        for (int it = gwarp; it < ntok * KVH; it += nwarps) {
            const int t = it / KVH, kvh = it - t * KVH;
            const int pos = __ldg(t_vis + t) - 1;
            const size_t orow = (((size_t)l * cache_rows + __ldg(t_base + t) + pos) * KVH + kvh) * HD + lane * 4;
            const float4 a = __ldcg((const float4*)(qkv + (size_t)t * 4096 + 2048 + kvh * HD) + lane);
            const uint2 nw = __ldg((const uint2*)(kn + l * HD + lane * 4));
            const float r = rsqrtf(warp_sum(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w) / (float)HD + EPS);
            float e[4] = {a.x, a.y, a.z, a.w}, o[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) e[i] = e[i] * r * bf2f((i < 2 ? nw.x : nw.y) >> ((i & 1) * 16) & 0xffffu);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float pr = __shfl_xor_sync(0xffffffffu, e[i], 16);
                const int j = (lane * 4 + i) & 63;
                const float cs = __ldg(rope + (size_t)pos * 128 + j), sn = __ldg(rope + (size_t)pos * 128 + 64 + j);
                o[i] = e[i] * cs + (lane < 16 ? -pr : pr) * sn;
            }
            *(uint2*)(kcache + orow) = make_uint2((unsigned)f2bf(o[0]) | ((unsigned)f2bf(o[1]) << 16), (unsigned)f2bf(o[2]) | ((unsigned)f2bf(o[3]) << 16));
            const float4 v = __ldcg((const float4*)(qkv + (size_t)t * 4096 + 3072 + kvh * HD) + lane);
            *(uint2*)(vcache + orow) = make_uint2((unsigned)f2bf(v.x) | ((unsigned)f2bf(v.y) << 16), (unsigned)f2bf(v.z) | ((unsigned)f2bf(v.w) << 16));
        }
        bar_arrive(bar_cnt);
        bar_wait<0>(bar_cnt, bar_target);

        // ---- Phase 3: attention item (token t, q-head qh) per CTA, 128-position chunks, 16 warps x 8 positions ----
        {
            float* sow = sbuf;  // NW x 128 per-warp unnormalized o
            for (int item = blockIdx.x; item < ntok * QH; item += gridDim.x) {
                const int t = item / QH, qh = item - t * QH, kvh = qh >> 1;
                const int base = __ldg(t_base + t), vis = __ldg(t_vis + t), pos = vis - 1;
                const int nch = (vis + CTX - 1) / CTX;
                uint2 kk[PPW];
                const unsigned dstv = (unsigned)__cvta_generic_to_shared(sVc + warp * PPW * HD + lane * 4);
                auto issue_chunk = [&](int c) {
#pragma unroll
                    for (int j = 0; j < PPW; ++j) {
                        // positions past the last visible row are clamped to it (masked later): reads stay inside rows written before the attention barrier, so the
                        // non-coherent L1 path of the V cp.async can never hold a stale copy of a row that a later layer of this launch writes
                        const int pp = min(c * CTX + warp * PPW + j, vis - 1);
                        const size_t off = (((size_t)l * cache_rows + base + pp) * KVH + kvh) * HD + lane * 4;
                        kk[j] = __ldcg((const uint2*)(kcache + off));
                        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(dstv + j * HD * 2), "l"(vcache + off) : "memory");
                    }
                    asm volatile("cp.async.commit_group;" ::: "memory");
                };
                issue_chunk(0);
                // q: head qh, q-norm + RoPE (each warp redundantly; lane holds dims lane*4..+3)
                float4 q;
                {
                    const float4 a = __ldcg((const float4*)(qkv + (size_t)t * 4096 + qh * HD) + lane);
                    const uint2 nw = __ldg((const uint2*)(qn + l * HD + lane * 4));
                    const float r = rsqrtf(warp_sum(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w) / (float)HD + EPS);
                    float e[4] = {a.x, a.y, a.z, a.w}, o[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) e[i] = e[i] * r * bf2f((i < 2 ? nw.x : nw.y) >> ((i & 1) * 16) & 0xffffu);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float pr = __shfl_xor_sync(0xffffffffu, e[i], 16);
                        const int j = (lane * 4 + i) & 63;
                        const float cs = __ldg(rope + (size_t)pos * 128 + j), sn = __ldg(rope + (size_t)pos * 128 + 64 + j);
                        o[i] = e[i] * cs + (lane < 16 ? -pr : pr) * sn;
                    }
                    q = make_float4(o[0], o[1], o[2], o[3]);
                }
                float m, lsum;
                float4 o;
                attn_chunk<false>(kk, q, vis, sVc, nullptr, nullptr, m, lsum, o);
                if (nch > 1) issue_chunk(1);
#pragma unroll 1
                for (int c = 1; c < nch; ++c) {
                    float m2, l2;
                    float4 o2;
                    attn_chunk<false>(kk, q, vis - c * CTX, sVc, nullptr, nullptr, m2, l2, o2);
                    if (c + 1 < nch) issue_chunk(c + 1);
                    attn_merge(m, lsum, o, m2, l2, o2);
                }
                *(float4*)(sow + warp * HD + lane * 4) = o;
                if (lane == 0) { red[warp] = m; red[NW + warp] = lsum; }
                __syncthreads();
                if (tid < HD) {
                    float M = red[0];
#pragma unroll
                    for (int w = 1; w < NW; ++w) M = fmaxf(M, red[w]);
                    float L = 0.f, ov = 0.f;
#pragma unroll
                    for (int w = 0; w < NW; ++w) {
                        const float c = R11_EXPF(red[w] - M);
                        L += c * red[NW + w];
                        ov += c * sow[w * HD + tid];
                    }
                    __stcg(ao + (size_t)t * 2048 + qh * HD + tid, ov / L);
                }
                __syncthreads();  // sow / red reused by the next item
            }
        }
        bar_arrive(bar_cnt);
        load_o(l);
        bar_wait<0>(bar_cnt, bar_target);

        // ---- Phase 4: O-proj (K = 2048) + residual: x[t][row] += sum, rows owned by this CTA ----
        for (int t0 = 0; t0 < ntok; t0 += TBO) {
            const int nt = min(TBO, ntok - t0);
            stage_plain<2048>(ao, t0, nt, sbuf);
            __syncthreads();
            for (int tb = 0; tb < nt; ++tb) {
                float act[1][8];
                smem_act(sbuf + tb * 2048, warp & 7, act[0]);
                gemv_dot_reg(bo, nh * 8, act, part + tb * PART_N);
            }
            __syncthreads();
            for (int i = tid; i < nt * nh; i += NT) {
                const int tb = i / nh, rr = i - tb * nh;
                const float* p = part + tb * PART_N + rr * 8;
                float d = 0.f;
#pragma unroll
                for (int s = 0; s < 8; ++s) d += p[s];
                float* xp = x + (size_t)(t0 + tb) * H + rh0 + rr;
                __stcg(xp, __ldcg(xp) + d);
            }
            __syncthreads();
        }
        bar_arrive(bar_cnt);
        load_gu(l);
        bar_wait<0>(bar_cnt, bar_target);

        // ---- Phase 5: RMSNorm(ln2) + gate / up + SiLU * up ----
        for (int t0 = 0; t0 < ntok; t0 += TBQ) {
            const int nt = min(TBQ, ntok - t0);
            stage_norm(x, ln2 + l * H, t0, nt, sbuf, red);
            __syncthreads();
            for (int tb = 0; tb < nt; ++tb) {
                float act[1][8];
                smem_act(sbuf + tb * H, warp & 3, act[0]);
                gemv_dot_reg(bgu, ni * 8, act, part + tb * PART_N);
            }
            __syncthreads();
            for (int i = tid; i < nt * ni; i += NT) {
                const int tb = i / ni, rr = i - tb * ni;
                const float* pp = part + tb * PART_N + rr * 8;
                const float r = rmsd_r(red + tb * NW);
                const float gt = r * (pp[0] + pp[1] + pp[2] + pp[3]);
                const float up = r * (pp[4] + pp[5] + pp[6] + pp[7]);
                __stcg(hid + (size_t)(t0 + tb) * INTER + ri0 + rr, R11_SILU(gt) * up);
            }
            __syncthreads();
        }
        bar_arrive(bar_cnt);
        load_d(l);
        bar_wait<0>(bar_cnt, bar_target);

        // ---- Phase 6: down (K = 3072) + residual ----
        for (int t0 = 0; t0 < ntok; t0 += TBD) {
            const int nt = min(TBD, ntok - t0);
            stage_plain<INTER>(hid, t0, nt, sbuf);
            __syncthreads();
            for (int tb = 0; tb < nt; ++tb) {
                float act3[3][8];
#pragma unroll
                for (int j = 0; j < 3; ++j) smem_act(sbuf + tb * INTER, (warp + 4 * j) % 12, act3[j]);
                gemv_dot_reg(bd, nh * 12, act3, part + tb * PART_N);
            }
            __syncthreads();
            for (int i = tid; i < nt * nh; i += NT) {
                const int tb = i / nh, rr = i - tb * nh;
                const float* p = part + tb * PART_N + rr * 12;
                float d = 0.f;
#pragma unroll
                for (int s = 0; s < 12; ++s) d += p[s];
                float* xp = x + (size_t)(t0 + tb) * H + rh0 + rr;
                __stcg(xp, __ldcg(xp) + d);
            }
            __syncthreads();
        }
        bar_arrive(bar_cnt);
        if (l + 1 < NL) load_q(l + 1);
        bar_wait<0>(bar_cnt, bar_target);
    }

    // ---- final norm + LM head (tied embed) + argmax for the n_lm listed tokens, TBQ at a time ----
    for (int s0 = 0; s0 < n_lm; s0 += TBQ) {
        const int nt = min(TBQ, n_lm - s0);
        {
            const float w0 = bf2f(fnorm[tid]), w1 = bf2f(fnorm[tid + NT]);
            for (int tb = 0; tb < nt; ++tb) {
                const int t = __ldg(lm_tok + s0 + tb);
                const float a = __ldcg(x + (size_t)t * H + tid), b = __ldcg(x + (size_t)t * H + tid + NT);
                sbuf[tb * H + sperm(tid)] = a * w0;
                sbuf[tb * H + sperm(tid + NT)] = b * w1;
                const float v = warp_sum(a * a + b * b);
                if (lane == 0) red[tb * NW + warp] = v;
            }
        }
        __syncthreads();
        if (tid < nt) { lmr[tid] = rmsd_r(red + tid * NW); lseq_s[tid] = __ldg(lm_seq + s0 + tid); }
        __syncthreads();
        float best[TBQ];
        int besti[TBQ];
#pragma unroll
        for (int tb = 0; tb < TBQ; ++tb) { best[tb] = -INFINITY; besti[tb] = 0x7fffffff; }
        for (int r = gwarp; r < VOCAB; r += nwarps * 2)
            lm_group_n<2>(embed, r, nwarps, VOCAB, sbuf, nt, lmr, lseq_s, logits, best, besti);
        __syncthreads();  // red reads above are done
        if (lane == 0) {
#pragma unroll
            for (int tb = 0; tb < TBQ; ++tb) { red[tb * NW + warp] = best[tb]; ((int*)part)[tb * NW + warp] = besti[tb]; }
        }
        __syncthreads();
        if (tid < nt) {
            float bv = red[tid * NW]; int bi = ((int*)part)[tid * NW];
            for (int w = 1; w < NW; ++w) {
                const float v = red[tid * NW + w]; const int i = ((int*)part)[tid * NW + w];
                if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
            }
            __stcg(pval + (size_t)blockIdx.x * MAXTOK + s0 + tid, bv);
            __stcg(pidx + (size_t)blockIdx.x * MAXTOK + s0 + tid, bi);
        }
        __syncthreads();
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned tk = atomicAdd(ticket, 1u);
        s_flag = (tk == gridDim.x - 1);
    }
    __syncthreads();
    if (s_flag) {  // last CTA: reduce the per-CTA bests of each slot (a warp per slot)
        __threadfence();
        for (int slot = warp; slot < n_lm; slot += NW) {
            float bv = -INFINITY; int bi = 0x7fffffff;
            for (int b = lane; b < (int)gridDim.x; b += 32) {
                const float v = __ldcg(pval + (size_t)b * MAXTOK + slot); const int i = __ldcg(pidx + (size_t)b * MAXTOK + slot);
                if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
            }
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) next_token[__ldg(lm_seq + slot)] = (long long)bi;
        }
        __syncthreads();
        if (tid == 0) {
            atomicExch(ticket, 0u);
            atomicExch(bar_cnt, 0u);  // every block has passed its last barrier poll
        }
    }
}
#endif  // !HV2G_MT1
