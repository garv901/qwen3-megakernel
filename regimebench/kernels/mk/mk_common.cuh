// mk common part: model constants, knobs, helpers. Device helpers copied from probe/xover/xover.cu (verified on GPU, job 9521) and
// kernels/hv2g/hv2g_common.cuh (grid barrier, bf16 helpers). The attention header (attn_placeholder.cuh or common/splitkv/splitkv.cuh)
// is prepended to this text by build.py. Compile with -DNPAD=<16|32|64|128|256> [-DNB=<SM count>] and the knobs below.
//
// Knobs (all default to the verified configuration; with MK_STAMPS = 0 and MK_NULL = 0 the SASS is that of the source without
// the knob code, see NOTES.md):
//   MK_KU     k per weight unit (128 or 64); a unit is 128 rows x MK_KU k of W, TMA boxes of 64 k.
//   MK_RING   weight ring slots (units in flight), 2..4.
//   MK_XTMA   0 only (reserved: O-phase B tile by TMA from ao; not implemented in v0).
//   MK_STAMPS 1: thread 0 stores %globaltimer at every bar_arrive and bar_wait exit into smem, one dump to mk_stamps at the end.
//   MK_LMP    1: pipelined LM head (NPAD <= 64; NOTES.md "LM head, pipelined"); MK_LMR LM ring slots.
//   MK_TS     N: tile-stream engine at NPAD >= N (0 off, 1 = 128, default 32) (SPEC-P5a; NOTES.md "Tile-stream QKV"): QKV phase with B = bf16 xb by TMA (MK_KU 128 only; else no effect).
//   MK_TSB    1: P5b (SPEC-P5b) on top of MK_TS: GU and D on the tile stream (B = bf16 xb2 / act by TMA, A = interleaved wg|wu), act written by the GU epilogue,
//             bf16 conversions (xb2 + ss2 after O, xb after D) done by idle CTAs on tickets. Default 0 = the P5a build (SASS identical).
//   MK_SKV    (defined by build.py for attn = splitkv, SPEC-P2): attention = skv::attend + skv::combine_head (O-phase B tile); needs MK_KU 128.
//   MK_NULL   bitmask, bit p-1 = phase p (1..6), bit 6 = LM head: the phase keeps its barrier(s), weight TMAs and ring waits and skips
//             staging, MMA, TMEM read and output (wrong outputs by design; null-work oracle).
#ifndef NPAD
#error "define NPAD (token bucket: 16, 32, 64, 128 or 256)"
#endif
#ifndef NB
#define NB 148   // grid = SM count (the arm passes the real count)
#endif
#ifndef MK_KU
#define MK_KU 128
#endif
#ifndef MK_RING
#define MK_RING 2
#endif
#ifndef MK_XTMA
#define MK_XTMA 0
#endif
#ifndef MK_STAMPS
#define MK_STAMPS 0
#endif
#ifndef MK_NULL
#define MK_NULL 0
#endif
#ifndef MK_PFA
#define MK_PFA 1   // 0: v0 order (prefetch, then arrive); 1 (default, job 9582: -66 us at b1): next-phase weight prefetch after the barrier arrive (see MK_PF_ARRIVE in mk_step.cu)
#endif
#ifndef MK_DBG_EXIT
#define MK_DBG_EXIT 0   // debug: return after prologue stage n (1: smem/TMEM/mbarrier init, 2: + embed copy and zeroing)
#endif
#ifndef MK_EPI2
#define MK_EPI2 0   // TMEM read: 2 chunks of 16 columns per warp iteration (1) or one (0, default: 1 spills at NPAD 16..256)
#endif
#ifndef MK_LMP
#define MK_LMP 1   // 1 (default, job 9586): pipelined LM head (SPEC-lm): X staged once, LM ring of MK_LMR units, 2 TMEM accumulators, warp-specialized (NPAD <= 64 only; larger NPAD keeps the v0 LM path)
#endif
#ifndef MK_LMR
#define MK_LMR 0   // LM ring slots (32 KB each) of the MK_LMP path; 0 = per bucket: 4 at NPAD 16, 3 at NPAD 32 (job 9586), 2 at NPAD 64 (smem limit)
#endif
#ifndef MK_TS
#define MK_TS 128  // 0 off; N (16 / 32 / 64 / 128 / 256): tile-stream engine (ts_phase) at NPAD >= N (1 = 128; with MK_TSB 1: job 9622 b128 -167, b256 -957 us; P5a alone was b128 +54, job 9602; NPAD 16 / 32 / 64 opt-in only: e2e b1 +90, b32 +104, b64 +57 us, jobs 10294 / 10308 / 10309); the D phase writes xb = bf16(x * ln1[l+1]) (see NOTES.md)
#endif
#ifndef MK_LMTS
#define MK_LMTS 1   // (default 1, job 9701: b128 -96, b256 -216 us) 1: SPEC-P3c, the LM head on the tile stream (ts_phase<PhLm, 128, 1>, A = embed, B = bf16(x * fnorm) from the last D finalize) when TSB and every lm slot s holds token s; else the v0 LM path
#endif
#ifndef MK_TSB
#define MK_TSB 1   // (default 1, job 9622) 1: GU and D on the tile stream + idle-CTA finalizers (SPEC-P5b); needs MK_TS on at the bucket (else no effect)
#endif
#ifndef MK_WU
#define MK_WU -1  // P14 (-1 = per bucket: 1 at NPAD 128 only; job 9803: b128 -10.4, b256 +8.9): 1 = tile-stream producer / MMA issuer as whole warps 0 / 1 (lane 0 works, lanes 1..31 wait at __syncwarp) instead of tid 0 / 32 with the rest of the warp at the join barrier (xover4 job 9802: GU / D / QKV T 256 -0.55 / -0.58 / -0.65 us/phase)
#endif
#ifndef MK_L2PF
#define MK_L2PF 0   // P11: L2 prefetch (cp.async.bulk.prefetch.tensor) of this CTA's GU (bit 0) / D (bit 1) weight boxes before attend (TSB + splitkv)
#endif
#ifndef MK_QR
#define MK_QR 0   // P10: 1 = QKV -> q/k-norm + RoPE without a grid barrier (ticket-gated rope jobs on every CTA; TSB + splitkv only)
#endif
#ifndef MK_PFE
#define MK_PFE 0   // P9: 1 = issue the next phase's first weight units (ts_first) before the O / D finalizer instead of after the barrier arrive
#endif
#ifndef MK_FIN
#define MK_FIN -1   // P7: idle-CTA finalizers (O -> xb2 / ss2, D -> xb): 0 = one CTA per (rb, tt) job; 1 = each job split by token rows over nidle / njobs idle CTAs (seen counters tk[32, 64)); 2 = 1 over all CTAs after their own tickets;
                     // -1 = per bucket: 2 at NPAD >= 128 (job 9778 on g4a: b128 -37.9 vs 1, b256 -30.0 vs 0), else 0 (job 9771: 1 at b64 +5.7)
#endif
#ifndef MK_GU_TT
#define MK_GU_TT 0   // P6: GU token tile override (0 = 96 at NPAD 256 (job 9757: -59.6 us vs 128), 48 at 128 (job 9758: -42.3 us vs 64)); multiple of 16, <= 128 (two accumulators in 256 TMEM columns). xover4 job 9756: GU T=256 TT 96 8.45 vs TT 128 8.85 us
#endif
#ifndef MK_KVE
#define MK_KVE 0       // lead (a): skv::attend_pre after the CTA's QKV tiles (needs SKV_EARLY in splitkv); 0 = off
#endif
#ifndef MK_DBG_WAIT
#define MK_DBG_WAIT 0   // debug: bounded spins (grid barrier, mbarriers, tickets): after ~2^31 cycles print the site and give up (results wrong, the kernel ends and printf flushes)
#endif
#if MK_DBG_WAIT
#define MK_DBG_SPIN(t0, site, a, b) if (clock64() - (t0) > (1ll << 31)) { printf("MK_DBG_WAIT site %d cta %d tid %d a %u b %u\n", (site), (int)blockIdx.x, (int)threadIdx.x, (unsigned)(a), (unsigned)(b)); break; }
#define MK_DBG_T0(t0) const long long t0 = clock64()
#else
#define MK_DBG_SPIN(t0, site, a, b)
#define MK_DBG_T0(t0)
#endif
#ifndef MK_DBG_NOATT
#define MK_DBG_NOATT 0   // debug (MK_SKV builds): 1 = attend is compiled in but skipped at run time (guard ntok < 0): separates its code-gen cost from its run-time cost; output is wrong
#endif
#ifndef MK_CMB
#define MK_CMB -1   // R10 P20 (jobs 14986 / 14987): -1 = 1 at NPAD >= 64 (default; stamps: one more barrier per layer when the plan splits, stamps_spread.py reads it); 1 = (MK_SKV, plans with S > 1) combine all split rows into ao on every CTA (skv::combine_all_to_ao) + one grid barrier after attend, instead of combine_head of the unit's head in the O phase (8 O units per head repeat it)
#endif
constexpr int CMB = MK_CMB >= 0 ? MK_CMB : (NPAD >= 64 ? 1 : 0);   // default measured at NPAD >= 64 only
#ifndef MK_STG
#define MK_STG 4   // staging chunks (16 B) in flight per thread
#endif
#ifndef MK_BIGT
#define MK_BIGT 0   // 1: large-T build (branch mk-bigt, step 1). Smem layout of the NPAD 128 bucket; the token count is a run-time value 1..MK_MAXT; every GEMM
                    // (QKV, O, GU, D, LM) runs on the tile stream with a run-time token-tile count; tickets are laid out by that count; the last D finalizer writes
                    // the LM rows compacted (lm_inv), so the LM tables stay MAXLM (n_lm <= 256). 0 (default): the bucket builds, SASS unchanged.
#endif
#ifndef MK_MAXT
#define MK_MAXT 8192   // MK_BIGT: token capacity of the global per-token buffers (x, qkv, xb, xb2, act, ao, ss2, rr; the host allocates them with this size)
#endif
#ifndef MK_BIG_QKS
#define MK_BIG_QKS 2   // MK_BIGT: split-K of the QKV tile stream (K 1024: 1, 2, 4 or 8)
#endif
#ifndef MK_BIG_OKS
#define MK_BIG_OKS 2   // MK_BIGT: split-K of the O tile stream (K 2048: 1, 2, 4, 8 or 16)
#endif
#ifndef MK_BIG_DKS
#define MK_BIG_DKS 4   // MK_BIGT: split-K of the D tile stream (K 3072: 1, 2, 3, 4, 6, 8, 12 or 24)
#endif
static_assert(NPAD == 16 || NPAD == 32 || NPAD == 64 || NPAD == 128 || NPAD == 256, "NPAD bucket");
static_assert(MK_KU == 128 || MK_KU == 64, "MK_KU");
static_assert(MK_RING >= 2 && MK_RING <= 4, "MK_RING");
static_assert(MK_TS == 0 || MK_TS == 1 || MK_TS == 16 || MK_TS == 32 || MK_TS == 64 || MK_TS == 128 || MK_TS == 256, "MK_TS");
static_assert(MK_TSB == 0 || MK_TSB == 1, "MK_TSB");
static_assert(MK_LMTS == 0 || MK_LMTS == 1, "MK_LMTS");
static_assert(MK_LMP == 0 || MK_LMP == 1, "MK_LMP");
static_assert(MK_LMR == 0 || (MK_LMR >= 2 && MK_LMR <= 8), "MK_LMR");
static_assert(MK_XTMA == 0, "MK_XTMA = 1 (O-phase B tile by TMA) is not implemented in v0");

typedef unsigned int u32;
typedef unsigned long long ull;
typedef unsigned short bf16_t;

constexpr int NT = 512, NW = 16;
constexpr int HID = 1024, NLAY = 28, NQH = 16, NKVH = 8, HDIM = 128, INTER = 3072, VOCAB = 151936;
constexpr int QKVN = 4096, GUN = 6144, QDIM = 2048;
constexpr bool BIGT = MK_BIGT != 0;
constexpr int MAXLM = 256;                        // LM slots: smem LM tables, pkey / pidx stride (n_lm <= MAXLM in every build)
constexpr int MAXT = BIGT ? MK_MAXT : 256;        // token capacity of the global per-token buffers (ss2 / rr sections); 256 in the bucket builds
constexpr float EPS = 1e-6f;
constexpr int TS_MIN = MK_TS == 1 ? 128 : MK_TS;   // smallest NPAD bucket on the tile stream (keep in sync with build.py:ts_min)
constexpr bool TS = MK_TS && NPAD >= TS_MIN && MK_KU == 128;   // tile-stream QKV (SPEC-P5a)
constexpr bool TSB = TS && MK_TSB;
constexpr bool LMTS = TSB && MK_LMTS;   // SPEC-P3c: tile-stream LM head (run time: lm_ident)
// tensor maps: wq, wk, wv, wo, wg, wu, wd, embed (+ xb when TS) (+ xb2, wgu, act when TSB: maps 9, 10, 11) (+ ao when BIGT: map 12; + the prefill attention's qb, kcache,
// vcache: maps 13, 14, 15, attn_prefill.cuh), 128 B each, in global memory
constexpr int NMAP = BIGT ? 16 : TSB ? 12 : TS ? 9 : 8;

constexpr int KU = MK_KU, NBOX = KU / 64, RING = MK_RING;
constexpr int UB = NBOX * 16384;              // bytes of one weight unit in the ring (128 rows x KU k bf16)
constexpr int XS = NBOX * NPAD * 128;         // bytes of one B-tile slot (NBOX k-atoms of NPAD rows x 128 B)
#ifdef MK_SKV   // attn = splitkv (A1 v1, SPEC-P2): the scratch is skv::kSmemBytes ending at OFF_BARS + kBarBytes (its mbarrier tail), see OFF_BARS below
constexpr int ATTN_BYTES = (int)skv::attend_smem_bytes();
constexpr int SKV_TAIL = (int)skv::kBarBytes;   // A1's mbarriers [OFF_BARS, OFF_BARS + SKV_TAIL): never written by mk; mk's own barrier block starts behind them
constexpr int XREG = 2 * XS;                    // the attention scratch is not part of the overlay regions: it is anchored at OFF_BARS - kBarOff
#else
constexpr int ATTN_BYTES = SKV_SMEM_BYTES;    // attention scratch, overlaid on the B-tile slots (idle in phases 2 and 3)
constexpr int XREG = (2 * XS > ATTN_BYTES) ? 2 * XS : ATTN_BYTES;
#endif
constexpr int OFF_X = RING * UB;              // B-tile slots (offset from the 1024-aligned smem base)
// LM head, pipelined path (SPEC-lm), NPAD <= 64: smem from the base = [XL: 8 k-chunks x XS][LM ring: LMR x UB]; overlays every other region
// (no other phase runs after the LM phase); the barriers stay behind the larger of the two layouts.
constexpr bool LMP = MK_LMP && NPAD <= 64 && MK_KU == 128;   // KU 64 keeps the v0 LM path
constexpr int LMR = MK_LMR ? MK_LMR : (NPAD == 16 ? 4 : NPAD == 32 ? 3 : 2);   // keep in sync with build.py:lm_ring
constexpr int LM_EPI_WARPS = 4 * (NPAD / 16 < 3 ? NPAD / 16 : 3);   // epilogue warps of the LM head (accempty count)
constexpr int LM_RING_OFF = 8 * XS;
constexpr int LM_BYTES = LMP ? 8 * XS + LMR * UB : 0;
// LM mbarriers (LMP only): full[LMR] at +LMB_FULL, empty[LMR] at +LMB_EMPTY, accfull[2] at +LMB_ACCF, accempty[2] at +LMB_ACCE
constexpr int LMB_FULL = 192, LMB_EMPTY = 256, LMB_ACCF = 320, LMB_ACCE = 336;
// Tile-stream QKV (TS): ring of TS_R slots [A 32 KB | B TT x 256 B] from the base, overlaying every other region; its mbarriers reuse the LM block:
// full[TS_R] at +LMB_FULL, empty[TS_R] at +LMB_EMPTY, accfull[2] at +LMB_ACCF, accempty[2] at +LMB_ACCE. With LMP and TS (NPAD <= 64) every ts_first re-initialises the block
// for its phase and lm_first re-initialises it for the LM head (LMP path).
constexpr int TS_TT = NPAD < 128 ? NPAD : 128;              // tokens per tile (B box rows); NPAD 256: 2 token tiles, NPAD <= 128: 1
constexpr int TS_KS = BIGT ? MK_BIG_QKS : (NPAD == 256) ? 2 : 4;   // split-K factor of the tile stream (job 9592 best per bucket; MK_BIGT: knob)
constexpr int TS_SLOT = 32768 + TS_TT * 256;                // ring slot bytes: A (128 rows x 128 k bf16) + B (TS_TT x 128 k bf16)
// P5b phase configs (TT tokens per tile / KS split-K): GU 96 / 1 (NPAD 256, ragged: 96 + 96 + 64), 48 / 1 (NPAD 128, 48 + 48 + 32), NPAD / 1 (NPAD <= 64); D TS_TT / 8 (256), TS_TT / 12 (<= 128)
// (xover4 job 10214 at T 32 / 64: QKV TT T KS 4, GU TT T KS 1, D TT T KS 12). QKV = (TS_TT, TS_KS).
constexpr int FIN = MK_FIN >= 0 ? MK_FIN : (NPAD >= 128 ? 2 : 0);
constexpr bool WU = MK_WU >= 0 ? MK_WU != 0 : NPAD == 128;   // P14 per bucket
constexpr int GU_TT = MK_GU_TT ? MK_GU_TT : (NPAD == 256 ? 96 : NPAD == 128 ? 48 : NPAD), GU_KS = 1;   // MK_GU_TT: ragged tiles allowed (NTT = ceil(NPAD / TT); B rows past MAXTOK are TMA OOB zero fill)
constexpr int D_TT = TS_TT, D_KS = BIGT ? MK_BIG_DKS : NPAD == 256 ? 8 : 12;
constexpr int O_TT = 128, O_KS = MK_BIG_OKS;                // MK_BIGT: O on the tile stream (B = ao by TMA, map 12)
constexpr int TK_NTT = (NPAD + 127) / 128;                  // 128-token tiles of the O / D tickets and the idle-CTA finalizer jobs (job = rb * TK_NTT + tt); MK_BIGT: run time (tk_ntt)
static_assert(!BIGT || (NPAD == 128 && MK_KU == 128 && MK_TS != 0 && MK_TSB && MK_LMTS && MK_QR == 0 && MK_PFE == 0 && MK_FIN == -1 && MK_LMP == 1),
              "MK_BIGT: the NPAD 128 tile-stream layout (TS + TSB + LMTS, FIN 2) without QR / PFE");
static_assert(!BIGT || (MK_MAXT % 128 == 0 && MK_MAXT >= 256), "MK_MAXT: multiple of 128, >= 256");
static_assert(!BIGT || (HID % (MK_BIG_QKS * 128) == 0 && QDIM % (MK_BIG_OKS * 128) == 0 && INTER % (MK_BIG_DKS * 128) == 0), "MK_BIG_*KS: split-K must divide K into 128-k steps");
constexpr int TS_EXCH = TSB ? 12288 : 0;                    // GU exchange region: [3 groups][2 quadrant pairs][16 tokens][32 lanes] fp32
constexpr int TS_SMEM_AVAIL = 227 * 1024 - 11280 - (MK_STAMPS ? 8 * (2 * (1 + 6 * NLAY) + 8) : 0) - 1024 - 384 - TS_EXCH;   // dyn smem budget after static smem (+ mk_sk_s with MK_STAMPS), alignment slack, barriers and the exchange region
constexpr int ts_slot(int tt) { return 32768 + tt * 256; }
constexpr int ts_ringn(int tt) { return TS_SMEM_AVAIL / ts_slot(tt) < 8 ? TS_SMEM_AVAIL / ts_slot(tt) : 8; }   // ring depth of a phase with TT = tt: as many slots as fit
constexpr int TS_R = TS ? ts_ringn(TS_TT) : 1;              // QKV ring depth
constexpr int TS_RING_A = TS ? ts_ringn(TS_TT) * ts_slot(TS_TT) : 0, TS_RING_B = TSB ? ts_ringn(GU_TT) * ts_slot(GU_TT) : 0;
constexpr int TS_RING_BYTES = TS_RING_A > TS_RING_B ? TS_RING_A : TS_RING_B;   // largest ring over the phases (D shares QKV's TT)
constexpr int TS_XOFF = TS_RING_BYTES;                      // GU exchange region offset (after the ring, below OFF_BARS)
constexpr int TS_BYTES = TS ? TS_RING_BYTES + TS_EXCH : 0;
constexpr int mk_max3(int a, int b, int c) { return a > b ? (a > c ? a : c) : (b > c ? b : c); }
#ifdef MK_SKV
// OFF_BARS: >= kBarOff (the scratch base OFF_BARS - kBarOff must be >= 0) and >= every layout; OFF_BARS % 1024 == 512 (= kBarOff % 1024), so the scratch base is 1024 B aligned
constexpr int OFF_LAY = mk_max3(OFF_X + XREG, LM_BYTES, TS_BYTES);
constexpr int OFF_MIN = OFF_LAY > (int)skv::kBarOff ? OFF_LAY : (int)skv::kBarOff;
constexpr int OFF_BARS = (OFF_MIN - 512 + 1023) / 1024 * 1024 + 512;
constexpr int OFF_MB = OFF_BARS + SKV_TAIL;   // mk's barrier block: tma bars [0, RING), mma bars at +64, tmem slot at +128 (+ LM / TS barriers)
constexpr int BARS_BYTES = ((LMP || TS) ? 384 : 256) + SKV_TAIL;
#else
constexpr int OFF_BARS = mk_max3(OFF_X + XREG, LM_BYTES, TS_BYTES);   // tma bars [0, RING), mma bars at +64, tmem slot at +128
constexpr int OFF_MB = OFF_BARS;
constexpr int BARS_BYTES = (LMP || TS) ? 384 : 256;
#endif
constexpr int DYN_SMEM = 1024 + OFF_BARS + BARS_BYTES;
constexpr int N_BARRIERS = 1 + (6 + CMB) * NLAY;   // grid barriers per launch (max: CMB adds one per layer when the plan splits)
#if MK_STAMPS
constexpr int MK_NSTAMP = 2 * N_BARRIERS + 8; // arrive + wait-exit per barrier, then the end of the LM phase
#endif  // MK_STAMPS

constexpr bool mk_null(int p) { return ((MK_NULL >> p) & 1) != 0; }   // p = phase - 1 (0..5), 6 = LM

#define MK_CDIV(a, b) (((a) + (b) - 1) / (b))

// ---------------------------------------------------------------- bf16 helpers (hv2g_common.cuh)
__device__ __forceinline__ float bf2f(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bflo(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bfhi(unsigned u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ bf16_t f2bf(float f) {
    unsigned u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16_t)(u >> 16);
}
// two fp32 -> packed bf16x2 (lo = a, hi = b), round to nearest even: the form of cuda_bf16.hpp __floats2bfloat162_rn (sm_80+)
__device__ __forceinline__ u32 pack_bf16(float a, float b) {
    u32 r;
    asm("{ cvt.rn.bf16x2.f32 %0, %2, %1;}\n" : "=r"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
#define MK_SILU(g) ((g) / (1.f + expf(-(g))))   // hv2g R11_SILU

// ---------------------------------------------------------------- xover.cu helpers (verified, job 9521)
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ ull gt() { ull t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void mbar_init(u32 a, u32 cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(cnt) : "memory"); }
__device__ __forceinline__ void mbar_inval(u32 a) { asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(a) : "memory"); }
__device__ __forceinline__ void mbar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_wait(u32 a, u32 parity) {
    u32 ok;
    MK_DBG_T0(t0);
    while (true) {
        asm volatile("{\n\t.reg .pred P;\n\tmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2;\n\tselp.u32 %0, 1, 0, P;\n\t}" : "=r"(ok) : "r"(a), "r"(parity) : "memory");
        if (ok) break;
        MK_DBG_SPIN(t0, 1, a, parity)
    }
}
__device__ __forceinline__ void mbar_arrive(u32 a) { asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(a) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(u32 a, u32 bytes) { asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(bytes) : "memory"); }
__device__ __forceinline__ void tma2(u32 dst, const void* map, int c0, int c1, u32 mbar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" :: "r"(dst), "l"(map), "r"(mbar), "r"(c0), "r"(c1) : "memory");
}
// tensor map in global memory (written by the host before the launch): one acquire of the tensormap proxy per map (form of dense_tc map_patch)
__device__ __forceinline__ void map_acquire(const void* slot) {
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"(slot) : "memory");
}
__device__ __forceinline__ void tmem_alloc(u32 smem_slot, u32 ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_slot), "r"(ncols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(u32 taddr, u32 ncols) { asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(ncols) : "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }   // generic writes to global -> later TMA reads (TS: xb)
// named barrier (bar.sync id, n threads): ids 1..6 = the GU exchange pairs, 7 = the GU epilogue setup (384 threads) (mk_step.cu); id 0 is __syncthreads
__device__ __forceinline__ void named_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void tc_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__host__ __device__ constexpr u32 tc_idesc(int M, int N) { return (1u << 4) | (1u << 7) | (1u << 10) | ((u32)(N >> 3) << 17) | ((u32)(M >> 4) << 24); }
// UMMA smem descriptor: K-major, SWIZZLE_128B, SBO = 1024 B (8 rows x 128 B), LBO unused, sm100 version bit
__device__ __forceinline__ ull sdesc(u32 addr) { return (ull)((addr >> 4) & 0x3fff) | (1ull << 16) | (64ull << 32) | (1ull << 46) | (2ull << 61); }
__device__ __forceinline__ void tc_mma(u32 d_tmem, ull a_desc, ull b_desc, u32 idesc, u32 acc) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\ttcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}" :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(acc) : "memory");
}
__device__ __forceinline__ void tc_commit(u32 mbar) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mbar) : "memory"); }
// TMEM -> regs: 32 lanes (warp quadrant) x 16 fp32 columns. M = 128: TMEM lane = accumulator row (32 q + lane).
__device__ __forceinline__ void tmem_ld16(u32 (&v)[16], u32 ta) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]),
                   "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]) : "r"(ta));
}
__device__ __forceinline__ void red_add(float* p, float v) { asm volatile("red.global.add.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory"); }

// ---------------------------------------------------------------- grid barrier (hv2g_common.cuh) + optional stamps
// Monotonic arrival counter (requires co-residency: cooperative launch). Each CTA arrives once with a release reduction and thread 0
// relaxed-polls (then one acquire fence) until all CTAs of this epoch have arrived; the counter is zeroed at the end of the launch by
// the last argmax ticket holder.
#if MK_STAMPS
__device__ ull mk_stamps[NB * MK_NSTAMP];   // [cta * MK_NSTAMP + k]: %globaltimer (ns); even 2b = arrive of barrier b, 2b + 1 = wait exit, then LM end
__shared__ ull mk_sk_s[MK_NSTAMP];
__shared__ unsigned mk_sk_n;
__device__ __forceinline__ void mk_stamp() {   // thread 0 only
    const ull t = gt();
    const unsigned n = mk_sk_n;
    mk_sk_n = n + 1;
    if (n < MK_NSTAMP) mk_sk_s[n] = t;
}
#endif  // MK_STAMPS
__device__ __forceinline__ void bar_arrive(unsigned* cnt) {
    __syncthreads();
#if MK_STAMPS
    if (threadIdx.x == 0) mk_stamp();
#endif  // MK_STAMPS
    if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(cnt) : "memory");
}
__device__ __forceinline__ void bar_wait(const unsigned* cnt, unsigned& target) {
    target += gridDim.x;
    if (threadIdx.x == 0) {
        unsigned v;
        MK_DBG_T0(t0);
        do {
            asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(cnt) : "memory");
            MK_DBG_SPIN(t0, 2, v, target)
        } while (v < target);
        asm volatile("fence.acquire.gpu;" ::: "memory");   // one acquire after the relaxed poll succeeds
#if MK_STAMPS
        mk_stamp();
#endif  // MK_STAMPS
    }
    __syncthreads();
}

// ---------------------------------------------------------------- engine wrappers (cta_group::1 today; a later MK_CG=2 build changes only these,
// geom() / mk_row_of() in mk_step.cu and the launch attribute in coop.py / build.py; the phase code never touches tcgen05 / TMA asm directly)
constexpr int MK_CG = 1;   // cta_group of the MMA engine (only 1 is implemented)
__device__ __forceinline__ void mk_tmem_alloc(u32 smem_slot) { tmem_alloc(smem_slot, 256); }          // whole warp
__device__ __forceinline__ void mk_tmem_dealloc(u32 taddr) { tmem_dealloc(taddr, 256); }              // whole warp
// issuing thread: all K steps of one weight unit (A = ring slot at sr, B = staged tile at xb), accumulate into TMEM column 0 (first: overwrite)
__device__ __forceinline__ void mk_mma_unit(u32 tb, u32 sr, u32 xb, u32 idesc, bool first) {
#pragma unroll
    for (int kk = 0; kk < KU / 16; ++kk) {
        const int a = kk >> 2, off = (kk & 3) * 32;
        tc_mma(tb, sdesc(sr + a * 16384 + off), sdesc(xb + a * (NPAD * 128) + off), idesc, (kk > 0 || !first) ? 1u : 0u);
    }
}
__device__ __forceinline__ void mk_commit(u32 mbar) { tc_commit(mbar); }
// warp quadrant q reads TMEM lanes 32 q.. : 16 token columns c0.. of the accumulator
__device__ __forceinline__ void mk_tmem_read16(u32 (&v)[16], u32 tb, int q, int c0) { tmem_ld16(v, tb + ((u32)(q * 32) << 16) + c0); }
// TMEM lane (quadrant q, lane) -> accumulator row within the 128-row block (M = 128: identity layout)
__device__ __forceinline__ int mk_row_of(int q, int lane) { return 32 * q + lane; }
// one weight unit's TMA: NBOX boxes of 64 k x 128 rows into ring slot s, completing on mbarrier bar
__device__ __forceinline__ void mk_tma_unit(u32 dst, const void* map, int k0, int row, u32 bar) {
#pragma unroll
    for (int a = 0; a < NBOX; ++a) tma2(dst + a * 16384, map, k0 + a * 64, row, bar);
}
