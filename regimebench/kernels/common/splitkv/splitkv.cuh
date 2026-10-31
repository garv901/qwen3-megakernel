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
#ifndef SKV_PAGED
#define SKV_PAGED 0         // 1 = serving mode (mk MK_SERVE): KV in the vLLM paged cache [blocks, 8, kPageRows rows, 256] (K | V per row, 512 B), read through a device block table (Params::bt).
#endif                      // Item word 0 = the request index (not a packed row base). 4D (5D with SKV_TMA_FUSE) tensor maps (build.py skv_map_specs paged=True). Items start at multiples of the
                            // stage rows (plan.py align), so a stage never crosses a block. q_len 1 / R6 TMA engines only (no LDG engine, no SKV_IFACE 2).
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
#if SKV_QMAX == 1
#define SKV_KV_EVICT_FIRST 1 // (default 1 for q_len 1 builds since MK-core job 9725: in mk b128 -106, b256 -145 us/step, the GEMM phases recover; standalone neutral, job 9720)
#else
#define SKV_KV_EVICT_FIRST 0 // (default 0 for the R6 consumer: MK-core job 9748, in mk q4 +10.8, q16 +4.7 us/step with it)
#endif
#endif                       // L2 evict_first cache policy on KV loads: KV is read once per step, so it should not displace the caller's weights / activations
                             // in L2. 0 = none, 1 = TMA boxes only (both consumers), 2 = TMA + LDG engine (LDG-only b1 builds lose in mk: kv16K +14.7)
#ifndef SKV_TAIL_LDG
#define SKV_TAIL_LDG 0      // (default 0, opt-in; P9d) 1 = q_len 1 TMA mma path: an item whose length is == 1 (mod kStageRows) (and > 1) streams only its first len - 1 positions by TMA
#endif                      // (whole stages); the last position is read by LDG (warp 0) and merged as a parked single-position state, as the iface 2 new row
#if SKV_TAIL_LDG && !(SKV_IFACE == 1 && SKV_QMAX == 1 && SKV_TMA_MMA)
#error "SKV_TAIL_LDG needs SKV_IFACE 1, SKV_QMAX 1, SKV_TMA_MMA 1"
#endif
#ifndef SKV_HALVES
#define SKV_HALVES 0        // (default 0, opt-in; P10) 1 = two independent half-CTA consumers (8 warps each): half h takes the CTA's items k = h, h + 2, .. with its own half of the ring,
#endif                      // its own producer thread, merge and named barrier (bar.sync 1 + h, 256). For mk's NPAD >= 64 buckets (every CTA has >= 2 items); q_len 1 TMA mma path only
#if SKV_HALVES && !(SKV_TMA_MMA && SKV_IFACE == 1 && SKV_QMAX == 1 && SKV_FORCE_ENGINE == 1 && SKV_TICKET == 0 && SKV_TMA_RELEASE == 1 && SKV_PROF_NOCOMP == 0 && SKV_ITEM_PREFETCH == 1)
#error "SKV_HALVES needs SKV_TMA_MMA 1, SKV_IFACE 1, SKV_QMAX 1, SKV_FORCE_ENGINE 1, SKV_TICKET 0, SKV_TMA_RELEASE 1, SKV_PROF_NOCOMP 0, SKV_ITEM_PREFETCH 1"
#endif

#ifndef SKV_TMA_FUSE
#define SKV_TMA_FUSE 0      // (P13, default 0, opt-in) 1 = a full stage's K (and V) is one 4D TMA box (64 dims, rows, 2 halves, 1) instead of two 3D half boxes; partial stages keep the per-half boxes. Needs SKV_HALVES 1; host adds 2 fused maps
#endif
#if SKV_TMA_FUSE && !SKV_HALVES
#error "SKV_TMA_FUSE needs SKV_HALVES 1 (and so SKV_TMA_MMA 1)"
#endif
#ifndef SKV_GROUPS
#define SKV_GROUPS 2        // (P11) number of independent consumer groups of SKV_HALVES mode: 2 (half CTAs, P10), 4 (quarter CTAs) or 8 (P12: 2 warps per group, 16-row stages, no merge; needs SKV_STAGE_ROWS 16, SKV_STAGES 16)
#endif
#if SKV_GROUPS != 2 && SKV_GROUPS != 4 && SKV_GROUPS != 8
#error "SKV_GROUPS must be 2, 4 or 8"
#endif
#if SKV_GROUPS != 2 && !SKV_HALVES
#error "SKV_GROUPS != 2 needs SKV_HALVES 1"
#endif

#ifndef SKV_SI
#define SKV_SI 0            // (P21, default 0, opt-in) 1 = stage-interleaved groups: the CTA's items form ONE stage stream, global stage s goes to group s % G (balanced to 1 stage whatever the item sizes);
#endif                      // each group keeps a running (m, l, o) per item it touches and parks it in smem; the last arrival of an item merges the parked partials. Needs SKV_HALVES 1, SKV_GROUPS 8; plan groups=1
#if SKV_SI && !(SKV_HALVES && SKV_GROUPS == 8)
#error "SKV_SI needs SKV_HALVES 1 and SKV_GROUPS 8"
#endif
#ifndef SKV_SI_P
#define SKV_SI_P 5          // SKV_SI: park slots (items in flight); 5 x 8448 B
#endif
#ifndef SKV_SI_CAP
#define SKV_SI_CAP 64       // SKV_SI: max items per CTA (the records are copied to smem; more traps)
#endif

#ifndef SKV_POOL
#define SKV_POOL 0          // (P25, default 0, opt-in) 1 = guided static list + atomic pool: the plan (plan(pool=f)) gives each CTA a static list (items row b) and puts the smallest items in a global pool (items row gridDim.x,
#endif                      // count npool in the ni word of its record 0); after its static list a CTA claims pool items one at a time (atomicAdd on p.pool_ctr[layer], the next claim is issued one item ahead).
                            // The counter resets itself (the CTA that gets ticket npool + grid - 1 writes 0); the host zeroes it once. Every CTA makes exactly one failing claim. Not with SKV_TC / SKV_TICKET
#if SKV_POOL && (SKV_TC || SKV_TICKET)
#error "SKV_POOL does not combine with SKV_TC or SKV_TICKET"
#endif
// (P25b) SKV_POOL 2 (only with SKV_SI) = the pool claim happens INSIDE the SI stage stream (see run_items_si): the static list and the pool are one call, no drain / refill per claimed item.
#if SKV_POOL > 2 || SKV_POOL < 0
#error "SKV_POOL must be 0, 1 or 2"
#endif
#if SKV_POOL == 2 && !SKV_SI
#error "SKV_POOL 2 (in-stream claim) is only implemented for the SI consumer: build with SKV_SI=1 (g8 / ring builds: use SKV_POOL 1)"
#endif
#ifndef SKV_LOOKAHEAD
#define SKV_LOOKAHEAD 48    // SKV_POOL 2: the claimer (group 0's producer thread) claims the next pool item while (known stages - its next stage to produce) < SKV_LOOKAHEAD (about 6 per group)
#endif
#ifndef SKV_TC
#define SKV_TC 0            // (default 0, opt-in; MK-core tcgen05 pass) 1 = q_len 1 tcgen05 consumer: S^T = K Q^T and O_t^T = V^T P^T on the tensor core (M = 128 positions / dims, N = 16 = 2 real q heads),
#endif                      // separate K / V TMA rings, softmax warpgroup (warps 4-7) in registers + TMEM, O accumulated in registers. Needs Params::tmem (64 TMEM columns); TMA engine only
#ifndef SKV_TC_KST
#define SKV_TC_KST 2        // SKV_TC: K ring stages (128 rows x 256 B = 32 KB each)
#endif
#ifndef SKV_TC_VST
#define SKV_TC_VST 2        // SKV_TC: V ring stages
#endif
#if SKV_TC && !(SKV_TMA_MMA && SKV_IFACE == 1 && SKV_QMAX == 1 && SKV_FORCE_ENGINE == 1 && SKV_TICKET == 0 && SKV_PROF_NOCOMP == 0 && SKV_HALVES == 0 && SKV_TAIL_LDG == 0 && SKV_STAGE_ROWS == 128 && SKV_TMA_FUSE == 0)
#error "SKV_TC needs SKV_TMA_MMA 1, SKV_IFACE 1, SKV_QMAX 1, SKV_FORCE_ENGINE 1, SKV_TICKET 0, SKV_PROF_NOCOMP 0, SKV_HALVES 0, SKV_TAIL_LDG 0, SKV_STAGE_ROWS 128, SKV_TMA_FUSE 0"
#endif
#if SKV_TC && (SKV_TC_KST < 2 || SKV_TC_KST > 3 || SKV_TC_VST < 2 || SKV_TC_VST > 3)
#error "SKV_TC_KST / SKV_TC_VST must be 2 or 3"
#endif
#if SKV_PAGED && !(SKV_IFACE == 1 && SKV_TMA_MMA && SKV_FORCE_ENGINE == 1 && SKV_TICKET == 0)
#error "SKV_PAGED needs SKV_IFACE 1, SKV_TMA_MMA 1, SKV_FORCE_ENGINE 1 and SKV_TICKET 0"
#endif
#if SKV_PAGED && SKV_TC
#error "SKV_PAGED is not ported to the SKV_TC consumer (splitkv-v15): use the mma.sync consumers at NPAD 16 / 32 in serving"
#endif

namespace skv {

typedef unsigned short bf16;   // raw bf16 bits (hv2g does the same; no cuda_bf16 dependency)

constexpr int kKvHeads = 8, kDim = 128, kQHeads = 16;
constexpr int kBlock = 512;                       // threads per CTA (16 warps)
constexpr int kPartStride = 132;                  // floats per partial slot: o[128], m, l, pad (528 B)
constexpr int kStageRows = SKV_STAGE_ROWS, kStages = SKV_STAGES;   // TMA ring: kStages stages, a stage = kStageRows positions of K + V (256 B each)
#if SKV_HALVES && SKV_GROUPS == 8
static_assert(kStageRows == 16 && kStages == 16, "SKV_GROUPS 8 needs SKV_STAGE_ROWS 16 and SKV_STAGES 16");   // P12
#else
static_assert(kStageRows == 32 || kStageRows == 64 || kStageRows == 128, "SKV_STAGE_ROWS must be 32, 64 or 128");
static_assert(kStages >= 2 && kStages <= 8, "SKV_STAGES must be 2..8");
#endif
constexpr int kU = (kStageRows < 32 ? 32 : kStageRows) / 32;               // positions per row slot per stage (one online-softmax rescale per kU positions)
constexpr int kStageBytes = 2 * kStageRows * 256; // K half then V half
constexpr int kNumMaps = (kStageRows == 16 ? 5 : kStageRows == 32 ? 6 : kStageRows == 64 ? 7 : 8);   // box rows kStageRows, /2, .., 1 per K / V: tmaps[kv * kNumMaps + log2(kStageRows / rows)]
// P13: with SKV_TMA_FUSE the table has 2 more maps after the 2 * kNumMaps: [2 * kNumMaps] = K fused 4D, [2 * kNumMaps + 1] = V fused 4D
// (dims (64, L * cache_rows, 2, 8), strides of dims 1..3 (2048, 128, 256) B, box (64, kStageRows, 2, 1), SWIZZLE_128B; coordinates {0, row, 0, kvh})
constexpr int kNumTmaps = 2 * kNumMaps + (SKV_TMA_FUSE ? 2 : 0);
constexpr int kMergeChunk = SKV_MERGE_CHUNK;                   // combine: partial float4 loads in flight per lane
constexpr float kQScale = 0.08838834764831845f * 1.4426950408889634f;   // 1/sqrt(128) * log2(e)
constexpr int kRingOff = 0, kMrgOff = kStages * kStageBytes, kMrgBytes = 16 * 2 * kPartStride * 4;
#if SKV_TC
// SKV_TC smem map (bytes from the 1024 B aligned scratch base): K ring kTcK x 32 KB | V ring kTcV x 32 KB | Q 2 x 4 KB | P^T 2 x 4 KB | exchange 512 B | mbarriers (256 B).
// A K / V stage = two 64-dim halves of [128 rows][128 B] (SWIZZLE_128B TMA boxes, 16 KB each). Q / P^T buffer = two halves of [16 rows][128 B] (2 KB each).
constexpr int kTcK = SKV_TC_KST, kTcV = SKV_TC_VST;
constexpr int kTcKOff = 0, kTcVOff = kTcK * 32768, kTcQOff = (kTcK + kTcV) * 32768, kTcPOff = kTcQOff + 8192, kTcXOff = kTcPOff + 8192;
constexpr int kBarBytes = 256;
constexpr int kBarOff = kTcXOff + 512, kSmemBytes = kBarOff + kBarBytes;   // kBarOff % 1024 == 512 (mk_step.cu static_assert)
static_assert(kBarOff % 1024 == 512, "SKV_TC kBarOff");
#else
constexpr int kBarBytes0 = (16 * kStages > 64 ? 16 * kStages : 64);   // mbarriers: full[s] at +8 s, empty[s] at +8 (kStages + s)
#if SKV_POOL
constexpr int kBarBytes = kBarBytes0 + 64, kPoolSlot = kBarBytes0;    // SKV_POOL: 64 B after the mbarriers (offset kBarOff + kPoolSlot): the claimed ticket (2 words, alternating by iteration)
#else
constexpr int kBarBytes = kBarBytes0;
#endif
#if SKV_SI
// P21 SI region between the merge buffer and the mbarriers: item records (CAP x 2 int4) | prefix of stages gs[CAP + 1] (padded to 16 B) | ctr (arrivals [P][2], gens [P][2]; padded to 16 B) |
// park area P slots x 16 entries (dh * 8 + group) x kPartStride floats (o[2 heads][64 dims], m[2], l[2]).
constexpr int kSiP = SKV_SI_P, kSiCap = SKV_SI_CAP;
constexpr int kSiTabOff = kMrgOff + kMrgBytes, kSiGsOff = kSiTabOff + kSiCap * 32;
constexpr int kSiCtrOff = kSiGsOff + ((kSiCap + 1 + 3) / 4) * 16, kSiParkOff = kSiCtrOff + ((4 * kSiP + 3) / 4) * 16;
#if SKV_POOL == 2
constexpr int kSiStOff = kSiParkOff + kSiP * 16 * kPartStride * 4, kSiStBytes = 16;   // SKV_POOL 2: 16 B behind the park area: [0] = T_known (stages known), [1] = done
#else
constexpr int kSiStBytes = 0;
#endif
constexpr int kSiBytes = ((kSiParkOff + kSiP * 16 * kPartStride * 4 + kSiStBytes - kSiTabOff + 1023) / 1024) * 1024;   // padded to 1 KB: keeps kBarOff % 1024 (mk_step.cu static_assert)
constexpr int kBarOff = kMrgOff + kMrgBytes + kSiBytes, kSmemBytes = kBarOff + kBarBytes;
#else
constexpr int kBarOff = kMrgOff + kMrgBytes, kSmemBytes = kBarOff + kBarBytes;
#endif
#endif

constexpr bool kMmaOn = (SKV_TMA_MMA != 0);
// mma consumer geometry: warp w takes 16-position tile (w % kMmaT) of the stage and the dim slice dh = w / kMmaT (kMmaDW dims = kMmaMt m-tiles of 16)
constexpr int kGeoRows = (kStageRows < 32 ? 32 : kStageRows);   // P12: the 16-row stages exist only in SKV_GROUPS 8 (own consumer); the other consumers' geometry constants use 32 rows then (never instantiated)
constexpr int kMmaT = kGeoRows / 16, kMmaWPT = 16 / kMmaT, kMmaDW = 128 / kMmaWPT, kMmaMt = kMmaDW / 16, kMmaMS = kMmaDW + 4;   // kMmaMS: floats of one (warp, q-head) merge slot: O[kMmaDW], m, l, pad
constexpr int kMmaNew = 32 * kMmaMS, kMmaQs = kMmaNew + 132;   // merge buffer (float offsets): 32 slots | new-row state (v[128], m[2], pad) | per-item q scratch (2 x 128 bf16 = 128 floats)
constexpr int kMmaG = 1;                        // V m-tiles loaded / multiplied together
constexpr int kHalfBytes = kStageRows * 128;                   // one 64-dim half of K or V in a stage (SKV_TMA_MMA: [rows][64] bf16, 128 B swizzled rows)
static_assert(!kMmaOn || (kGeoRows >= 32 && kMmaDW % 16 == 0 && (kMmaQs + 128) * 4 <= kMrgBytes), "SKV_TMA_MMA geometry");
#if SKV_HALVES && SKV_GROUPS == 8
// P12 geometry (SKV_GROUPS 8): group q = warp >> 1 (2 warps, 64 threads) owns ring slots 2 q, 2 q + 1 (16-row stages); warp lw = warp & 1 = dims half dh (64 dims = kHMt m-tiles), the one tile of
// the stage is computed by both warps (full 128-dim QK^T), so each warp's online-softmax state is the group's: no merge. Merge buffer (floats): 16 x tail park (64 dims of v) | 16 x q scratch (128 floats).
constexpr int kG = 8, kHW = 2, kHT = 64, kHS = kStages / kG, kHDW = 64, kHMt = kHDW / 16;
static_assert(kHS == 2 && kStageRows == 16, "SKV_GROUPS 8 geometry");
constexpr int kP8Park = 0, kP8Q = 16 * 64, kP8Floats = kP8Q + 16 * 128;
static_assert(kP8Floats * 4 <= kMrgBytes, "SKV_GROUPS 8 merge buffer budget");
#elif SKV_HALVES
// P10 half-CTA geometry: half h = warps 8h .. 8h + 7 owns ring slots h * kHS .. h * kHS + kHS - 1. Warp lw = warp % 8: dim half dh = lw / 4 (kHDW = 64 dims = kHMt m-tiles), tile owner tw = lw % 4 takes
// the stage's 16-position tiles tw, tw + 4, .. (kHTP of them). Merge slot (warp, hq) = O[64], m, l, pad (kHMS floats), indexed by the global warp; per-half new-row (tail park) state and q scratch.
// P11: G = SKV_GROUPS groups (G = 2: the halves above; G = 4: quarters of 4 warps / 128 threads, items k = q, q + G, ..): group q = warp / kHW, kHW = 16 / G warps, kHT = 512 / G threads.
// The warps of a group split into 2 dims halves of kHO = kHW / 2 tile owners each (tile owner tw takes the stage's tiles tw, tw + kHO, ..).
constexpr int kG = SKV_GROUPS, kHW = 16 / kG, kHT = 512 / kG, kHO = kHW / 2;
constexpr int kHTs = (kG == 2 ? 8 : 7), kHOs = (kG == 2 ? 2 : 1), kGs = (kG == 2 ? 1 : 2);   // log2(kHT), log2(kHO), log2(kG)
static_assert(kStages % kG == 0 && kStages / kG >= 1, "SKV_HALVES needs SKV_STAGES % SKV_GROUPS == 0");
static_assert(kG == 2 ? kStageRows >= 64 : kStageRows >= 32, "SKV_HALVES needs SKV_STAGE_ROWS 64 / 128 (SKV_GROUPS 2) or 32 / 64 / 128 (SKV_GROUPS 4)");
constexpr int kHS = kStages / kG, kHDW = 64, kHMt = kHDW / 16, kHMS = kHDW + 4, kHTP = kMmaT / kHO;
static_assert(kMmaT % kHO == 0 && kHTP >= 1, "SKV_HALVES tile owners must divide the stage's tiles");
constexpr int kHNew = 32 * kHMS, kHQs = kHNew + kG * 132;     // float offsets in the merge buffer: 32 slots | G x tail park (v[128], m[2], pad) | G x q scratch (128 floats each)
constexpr int kHFloats = kHQs + kG * 128;                      // merge buffer floats used (of kMrgBytes / 4 = 4224)
static_assert(kHFloats * 4 <= kMrgBytes, "SKV_HALVES merge buffer budget");
constexpr int kHShift = (kG == 2 ? 16 : 8), kHMask = (kG == 2 ? 0x7fff : 0x7f);   // ring_phase: counter of group q = (g >> (q * kHShift)) & kHMask
#endif

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
#if SKV_TC
  unsigned tmem;                   // SKV_TC: TMEM base address (column 0 of >= 64 allocated columns, warp-aligned lane 0) owned by the caller; attend uses columns 0..63 only during the call
#endif
#if SKV_PAGED
  const int* bt; int bt_stride;    // block table [requests][bt_stride] int32 (vLLM block ids)
  int blk_per_layer;               // blocks between layers: the maps' last dim is layer * blk_per_layer + block
#endif
#if SKV_POOL
  unsigned* pool_ctr;              // SKV_POOL: [n_layers] u32, zeroed once by the host; attend resets counter [layer] itself
#endif
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
#if SKV_KV_EVICT_FIRST >= 2
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

#if SKV_TMA_FUSE || SKV_PAGED
__device__ __forceinline__ void tma4d(unsigned dst, const void* map, int c0, int c1, int c2, int c3, unsigned b) {
#if SKV_KV_EVICT_FIRST
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
               ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(b), "l"(kv_policy()) : "memory");
#else
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];"
               ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(b) : "memory");
#endif
}
#endif

#if SKV_PAGED && SKV_TMA_FUSE
__device__ __forceinline__ void tma5d(unsigned dst, const void* map, int c0, int c1, int c2, int c3, int c4, unsigned b) {
#if SKV_KV_EVICT_FIRST
  asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
               ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4), "r"(b), "l"(kv_policy()) : "memory");
#else
  asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
               ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4), "r"(b) : "memory");
#endif
}
#endif

#if SKV_PAGED
constexpr int kPageRows = 128;   // rows per cache block (the serving cache uses block size 128)
static_assert(kPageRows == 128 && (kPageRows % kStageRows) == 0, "paged: 128-row blocks, stage rows divide the block");
// block coordinate (last map dim) of position pos of request req in layer `layer`
__device__ __forceinline__ int page_cb(const Params& p, int layer, int req, int pos) {
  return layer * p.blk_per_layer + __ldg(p.bt + (size_t)req * p.bt_stride + (pos / kPageRows));
}
// address of this lane's 16 B of row `pos` of item `a` (a.x = request, a.y & 15 = kv head): lanes 0..15 the K row, lanes 16..31 the V row (K | V per 512 B cache row)
__device__ __forceinline__ const char* page_row_ptr(const Params& p, int layer, const int4& a, int pos, int lane) {
  const size_t blk = (size_t)page_cb(p, layer, a.x, pos);
  return (const char*)p.kc + (((blk * kKvHeads + (a.y & 15)) * kPageRows + (pos % kPageRows)) * 512) + (lane < 16 ? 0 : 256) + (lane & 15) * 16;
}
#define SKV_TAIL_ADDR(a) page_row_ptr(p, layer, (a), (a).w - 1, lane)
#define SKV_BOX(dst, map, c0) tma4d(dst, map, c0, rc, pkvh, cb, bar)
#define SKV_FBOX(dst, map) tma5d(dst, map, 0, rc, 0, pkvh, cb, bar)
#else
#define SKV_TAIL_ADDR(a) ((const char*)(lane < 16 ? p.kc : p.vc) + item_off(a) + (size_t)((a).w - 1) * 2048)
#define SKV_BOX(dst, map, c0) tma3d(dst, map, c0, pkvh, rc, bar)
#define SKV_FBOX(dst, map) tma4d(dst, map, 0, rc, 0, pkvh, bar)
#endif

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
#if SKV_PAGED
  int pcb = 0;                                                 // block coordinate of position pp
#endif
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
#if SKV_PAGED
      pcb = page_cb(p, layer, it.x, pp);
#endif
#if SKV_IFACE == 2
      if (it.w == rec[2 * pk + 1].w) --pend;                   // the piece ending at vis owns the new row: not written yet, the stream stops at vis - 1
#endif
      if (kTail && it.w - it.z > 1 && (it.w - it.z) % kStageRows == 1) --pend;   // P9d tail item (same rule as the consumer): the stream stops at p1 - 1, the last row is read by LDG
    }
    const int n = min(kStageRows, pend - pp);
    const unsigned dst = sbase + kRingOff + slot * kStageBytes, bar = bar0 + 8 * slot;
    mb_expect(bar, (unsigned)n * 512u);
#if SKV_PAGED
    int off = 0, rc = pp & (kPageRows - 1);
    const int cb = pcb;
#else
    int off = 0, rc = (int)(prow + pp);
#endif
#pragma unroll
    for (int rs = 0; rs < kNumMaps; ++rs) {
      const int rows = kStageRows >> rs;
      if (n & rows) {
#if SKV_TMA_MMA
        const char* mk = (const char*)p.tmaps + (0 * kNumMaps + rs) * 128;   // swizzled maps, box 64 dims x 1 x rows: lo half (dims 0-63), hi half (64-127)
        const char* mv = (const char*)p.tmaps + (1 * kNumMaps + rs) * 128;
        SKV_BOX(dst + off * 128, mk, 0);
        SKV_BOX(dst + kHalfBytes + off * 128, mk, 64);
        SKV_BOX(dst + 2 * kHalfBytes + off * 128, mv, 0);
        SKV_BOX(dst + 3 * kHalfBytes + off * 128, mv, 64);
#else
        tma3d(dst + off * 256, (const char*)p.tmaps + (0 * kNumMaps + rs) * 128, 0, pkvh, rc, bar);
        tma3d(dst + kStageRows * 256 + off * 256, (const char*)p.tmaps + (1 * kNumMaps + rs) * 128, 0, pkvh, rc, bar);
#endif
        off += rows; rc += rows;
      }
    }
    pp += n;
#if SKV_PAGED
    if (pp < pend && !(pp & (kPageRows - 1))) pcb = page_cb(p, layer, (int)prow, pp);   // next stage starts a new block
#endif
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
      t = ldg_nc(SKV_TAIL_ADDR(a));
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

#if SKV_HALVES && SKV_GROUPS != 8
// ------------------------------------------------------------------------------------------------ P10: half-CTA consumers (SKV_HALVES 1)
// P11: the code is parameterised by G = SKV_GROUPS (2 = halves, as described here with 8 warps / 256 threads / 4 tile owners; 4 = quarters: 4 warps / 128 threads / 2 tile owners, merge in
// 2 passes, bar.sync 1 + h, 128, ring_phase counters of 7 bits); `h` below is the group index and `half` means group.
// Half h (threads 256 h .. 256 h + 255) runs the CTA's items k = h, h + 2, .. as an independent q_len 1 mma consumer: its own ring slots (h * kHS ..), producer thread (256 h), item prefetch,
// q scratch, tail park slot, merge slots (indexed by the global warp) and named barrier (bar.sync 1 + h, 256). While one half runs its serial chain (merge, store, q staging) the other
// half computes or waits for its loads. Stage counters: gh0 / gh1 in g bits 0..14 / 16..30; the stage of counter c of half h uses slot h * kHS + c % kHS, parity (c / kHS) & 1.
// Named barrier rules (MK-core): (a) every bar.sync 1 + h a half starts is completed by all its 256 threads (the item loop bound nih is uniform inside a half and there is no exit inside it);
// (b) a half with no items executes no bar.sync 1 + h and goes straight to the final __syncthreads. attend's contract (ends with one __syncthreads) is kept by that last barrier.
// Ordering argument per half (all of it is inside one half, whose 256 threads share barrier 1 + h):
//   park slot (kHNew + 132 h): written by warp 8 h (all 32 lanes execute, lanes < 16 store) after the previous item's second bar.sync, by which every thread of the half has finished the previous
//     item's merge reads of it; read by the merge only after this item's first bar.sync, which follows the writes by every thread of warp 8 h. Consumed once, by `tail` of this item.
//   q scratch (kHQs + 128 h): each warp stores identical bytes (lanes < 16) then reads after its own __syncwarp; the stores of item k + 1 follow the second bar.sync of item k, which
//     follows every warp's last read of it (stage loop of item k) in program order, so no read of item k sees item k + 1's bytes.
//   merge slots (warp, hq): written by their warp before the first bar.sync, read by the half's merge threads between the two barriers, rewritten only after the second one.
//   The two halves touch disjoint ring slots, park slots, q scratch and merge slots; the ring-slot empty barriers (count 8) are arrived on by the 8 warps of the owning half only.
__device__ __forceinline__ void bar_sync_grp(int id) { asm volatile("bar.sync %0, %1;" ::"r"(id), "n"(kHT) : "memory"); }

__device__ __forceinline__ void run_items_halves(const Params& p, int layer, unsigned char* smem, unsigned& g, const int4* rec, int ni) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, l16 = lane & 15;
  const int h = tid >> kHTs, t = tid & (kHT - 1), lw = warp & (kHW - 1), dh = lw >> kHOs, tw = lw & (kHO - 1);   // h = group index
  const unsigned sbase = smaddr(smem);
  const unsigned bar0 = sbase + kBarOff;
  float* mrg = reinterpret_cast<float*>(smem + kMrgOff);
  const size_t layer_row = (size_t)layer * p.cache_rows;
  const int4 zero4 = make_int4(0, 0, 0, 0);
  constexpr bool kTail = (SKV_TAIL_LDG != 0);

  // ---- both halves' stage totals of this call (a pure function of the records; every thread computes them, so every thread stores the same packed counters at the end)
  int tot0 = 0, tot1 = 0, tot2 = 0, tot3 = 0;                  // stage totals of groups 0..G-1 (tot2 / tot3 only for G = 4; dead code for G = 2)
  for (int k = 0; k < ni; ++k) {
    const int4 it = rec[2 * k];
    const int ln = it.w - it.z;
    const int ns = (ln - ((kTail && ln > 1 && ln % kStageRows == 1) ? 1 : 0) + kStageRows - 1) / kStageRows;
    if (kG == 2) { if (k & 1) tot1 += ns; else tot0 += ns; }
    else { const int kq = k & 3; if (kq == 0) tot0 += ns; else if (kq == 1) tot1 += ns; else if (kq == 2) tot2 += ns; else tot3 += ns; }
  }

  const int nih = (ni - h + kG - 1) >> kGs;                     // items of this group: k = h, h + G, ..
  const int4* rh = rec + 2 * h;                                // item kk of the group: records rh[2 G kk], rh[2 G kk + 1]
  const unsigned g0c = g & kHMask, g1c = (g >> kHShift) & kHMask;
  const unsigned g2c = (g >> (2 * kHShift)) & kHMask, g3c = (g >> (3 * kHShift)) & kHMask;   // (g2c / g3c only for G = 4)
  const int gh0 = (int)(kG == 2 ? (h ? g1c : g0c) : (h == 0 ? g0c : h == 1 ? g1c : h == 2 ? g2c : g3c));   // this group's stage counter at entry

  // ---- TMA producer (thread 256 h): the half's stage sequence into its own slots
  int pk = -1, pp = 0, pend = 0, pkvh = 0;
  size_t prow = 0;
#if SKV_PAGED
  int pcb = 0;                                                 // block coordinate of position pp
#endif
  int4 pnxt = zero4;
  auto produce = [&](int slot) {
    if (pp >= pend) {
      if (pk + 1 >= nih) return;
      ++pk;
      const int4 it = pnxt;
      pnxt = (pk + 1 < nih) ? rh[2 * kG * (pk + 1)] : zero4;
      pp = it.z; pend = it.w; pkvh = it.y & 15; prow = layer_row + (size_t)it.x;
#if SKV_PAGED
      pcb = page_cb(p, layer, it.x, pp);
#endif
      if (kTail && it.w - it.z > 1 && (it.w - it.z) % kStageRows == 1) --pend;   // P9d tail rule (as the consumer)
    }
    const int n = min(kStageRows, pend - pp);
    const unsigned dst = sbase + kRingOff + slot * kStageBytes, bar = bar0 + 8 * slot;
    mb_expect(bar, (unsigned)n * 512u);
#if SKV_PAGED
    int off = 0, rc = pp & (kPageRows - 1);
    const int cb = pcb;
#else
    int off = 0, rc = (int)(prow + pp);
#endif
#if SKV_TMA_FUSE
    if (n == kStageRows) {   // P13: one 4D box per tensor covers both 64-dim halves (K lo | K hi, V lo | V hi)
      SKV_FBOX(dst, (const char*)p.tmaps + (2 * kNumMaps + 0) * 128);
      SKV_FBOX(dst + 2 * kHalfBytes, (const char*)p.tmaps + (2 * kNumMaps + 1) * 128);
    } else
#endif
#pragma unroll
    for (int rs = 0; rs < kNumMaps; ++rs) {
      const int rows = kStageRows >> rs;
      if (n & rows) {
        const char* mk = (const char*)p.tmaps + (0 * kNumMaps + rs) * 128;
        const char* mv = (const char*)p.tmaps + (1 * kNumMaps + rs) * 128;
        SKV_BOX(dst + off * 128, mk, 0);
        SKV_BOX(dst + kHalfBytes + off * 128, mk, 64);
        SKV_BOX(dst + 2 * kHalfBytes + off * 128, mv, 0);
        SKV_BOX(dst + 3 * kHalfBytes + off * 128, mv, 64);
        off += rows; rc += rows;
      }
    }
    pp += n;
#if SKV_PAGED
    if (pp < pend && !(pp & (kPageRows - 1))) pcb = page_cb(p, layer, (int)prow, pp);   // next stage starts a new block
#endif
  };
  if (t == 0 && nih > 0) {
    pnxt = rh[0];
#pragma unroll
    for (int s = 0; s < kHS; ++s) produce(h * kHS + (gh0 + s) % kHS);
  }

  auto item_off = [&](const int4& a) { return ((layer_row + (size_t)a.x) * kKvHeads + (a.y & 15)) * 256 + l16 * 16; };
  auto load_q = [&](const int4& a, int tok, uint4& x, uint4& y) {
    const bf16* qp = p.q + (size_t)(tok * kQHeads + 2 * (a.y & 15)) * kDim + l16 * 8;
    x = ldg_cg4(qp); y = ldg_cg4(qp + kDim);
  };
  auto load_tail = [&](const int4& a, uint4& tr) {             // the half's first warp (warp 8 h): lanes 0..15 = K row, lanes 16..31 = V row of position p1 - 1
    tr = make_uint4(0, 0, 0, 0);
    if (lw == 0 && a.w - a.z > 1 && (a.w - a.z) % kStageRows == 1)
      tr = ldg_nc(SKV_TAIL_ADDR(a));
  };
  uint4 tra = make_uint4(0, 0, 0, 0), trb = tra;
  int4 ra = nih > 0 ? rh[0] : zero4;                           // (a half without items must not read past the CTA's records)
  uint4 qa = make_uint4(0, 0, 0, 0), qb = qa;
  if (nih > 0) {
    load_q(ra, rh[1].z, qa, qb);
    if (kTail) load_tail(ra, tra);
  }
  float* nbp = mrg + kHNew + 132 * h;                          // this half's tail park slot
  float* qsp = mrg + kHQs + 128 * h;                           // this half's q scratch
  int gh = gh0;

  for (int k = 0; k < nih; ++k) {
    const int kvh = ra.y & 15, p0 = ra.z, p1 = ra.w;
    const bool tail = kTail && p1 - p0 > 1 && (p1 - p0) % kStageRows == 1;   // half-uniform
    int4 na = zero4;
    int ntok = 0;
    uint4 nqa = make_uint4(0, 0, 0, 0), nqb = nqa;
    if (k + 1 < nih) { na = rh[2 * kG * (k + 1)]; ntok = rh[2 * kG * (k + 1) + 1].z; }
    if (tail && lw == 0) {
      float qr0[8], qr1[8], kf[8], vf[8];
      unpack8(qa, qr0); unpack8(qb, qr1);
      unpack8(tra, kf);
      float d0 = 0.f, d1 = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) { d0 = fmaf(kf[j], qr0[j], d0); d1 = fmaf(kf[j], qr1[j], d1); }
#pragma unroll
      for (int o = 1; o <= 8; o <<= 1) { d0 += __shfl_xor_sync(0xffffffffu, d0, o); d1 += __shfl_xor_sync(0xffffffffu, d1, o); }
      uint4 vn;
      vn.x = __shfl_xor_sync(0xffffffffu, tra.x, 16); vn.y = __shfl_xor_sync(0xffffffffu, tra.y, 16);
      vn.z = __shfl_xor_sync(0xffffffffu, tra.z, 16); vn.w = __shfl_xor_sync(0xffffffffu, tra.w, 16);
      if (lane < 16) {
        unpack8(vn, vf);
        *reinterpret_cast<float4*>(nbp + l16 * 8) = make_float4(vf[0], vf[1], vf[2], vf[3]);
        *reinterpret_cast<float4*>(nbp + l16 * 8 + 4) = make_float4(vf[4], vf[5], vf[6], vf[7]);
        if (lane == 0) { nbp[128] = d0 * kQScale; nbp[129] = d1 * kQScale; }
      }
    }
    float mr0 = neg_inf(), mr1 = neg_inf(), lp0 = 0.f, lp1 = 0.f;
    float oacc[kHMt][4];
#pragma unroll
    for (int i = 0; i < kHMt; ++i) oacc[i][0] = oacc[i][1] = oacc[i][2] = oacc[i][3] = 0.f;
    {
      unsigned* qw = reinterpret_cast<unsigned*>(qsp);
      if (lane < 16) {
        qw[mma_q_word(l16, 0)] = qa.x; qw[mma_q_word(l16, 1)] = qa.y; qw[mma_q_word(l16, 2)] = qa.z; qw[mma_q_word(l16, 3)] = qa.w;
        qw[64 + mma_q_word(l16, 0)] = qb.x; qw[64 + mma_q_word(l16, 1)] = qb.y; qw[64 + mma_q_word(l16, 2)] = qb.z; qw[64 + mma_q_word(l16, 3)] = qb.w;
      }
      __syncwarp();
    }
    const int len = p1 - p0 - (tail ? 1 : 0);
    const int ns = (len + kStageRows - 1) / kStageRows;
    const unsigned qsa = smaddr(qsp) + mma_q_off(lane, 0);
    for (int j = 0; j < ns; ++j, ++gh) {
      const int slot = h * kHS + gh % kHS;
      const unsigned par = (gh / kHS) & 1;
      mb_wait(bar0 + 8 * slot, par);
      const int n = min(kStageRows, len - j * kStageRows);
      const unsigned sb = sbase + kRingOff + slot * kStageBytes;
      const unsigned j32 = mma_j32(lane);
      unsigned kdep = 0;
#pragma unroll 1
      for (int ti = 0; ti < kHTP; ++ti) {                      // tiles tw, tw + kHO, ..: the per-stage code of run_items, one tile after another
        const int tb = (tw + kHO * ti) * 16;
        const unsigned kb = sb + mma_k_base(lane, tb), vb = sb + mma_v_base(lane, tb);
        const bool live = tb < n, partial = tb + 16 > n;
        unsigned pb0 = 0, pb1 = 0;
        if (live) {
          float sa[4] = {0.f, 0.f, 0.f, 0.f}, sb2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) {
            unsigned k0[4];
            ldsm4(k0, kb + mma_k_term(ks, j32, kHalfBytes));
            uint2 qb0 = make_uint2(0u, 0u);
            if ((lane >> 2) < 2) qb0 = lds64(qsa + ks * 32);
            hmma16816((ks & 1) ? sb2 : sa, k0, qb0.x, qb0.y);
            kdep ^= k0[0];
          }
          float s0 = sa[0] + sb2[0], s1 = sa[1] + sb2[1], s2 = sa[2] + sb2[2], s3 = sa[3] + sb2[3];
          if (partial) {
            const int pos = tb + (lane >> 2);
            if (pos >= n) { s0 = neg_inf(); s1 = neg_inf(); }
            if (pos + 8 >= n) { s2 = neg_inf(); s3 = neg_inf(); }
          }
          float c0 = fmaxf(s0, s2), c1 = fmaxf(s1, s3);
#pragma unroll
          for (int o = 4; o <= 16; o <<= 1) { c0 = fmaxf(c0, __shfl_xor_sync(0xffffffffu, c0, o)); c1 = fmaxf(c1, __shfl_xor_sync(0xffffffffu, c1, o)); }
          const float n0 = fmaxf(mr0, c0 * kQScale), n1 = fmaxf(mr1, c1 * kQScale);
          const float a0 = ex2(mr0 - n0), a1 = ex2(mr1 - n1);
          mr0 = n0; mr1 = n1;
          const float p0v = ex2(fmaf(s0, kQScale, -n0)), p1v = ex2(fmaf(s1, kQScale, -n1)), p2v = ex2(fmaf(s2, kQScale, -n0)), p3v = ex2(fmaf(s3, kQScale, -n1));
          lp0 = fmaf(lp0, a0, p0v + p2v); lp1 = fmaf(lp1, a1, p1v + p3v);
          if (__any_sync(0xffffffffu, (lane & 3) == 0 && (a0 != 1.f || a1 != 1.f))) {
#pragma unroll
            for (int mt = 0; mt < kHMt; ++mt) { oacc[mt][0] *= a0; oacc[mt][1] *= a1; oacc[mt][2] *= a0; oacc[mt][3] *= a1; }
          }
          pb0 = movm_t(pack_bf16x2(p0v, p1v)); pb1 = movm_t(pack_bf16x2(p2v, p3v));
        }
#pragma unroll
        for (int g0 = 0; g0 < kHMt; g0 += kMmaG) {
          unsigned vf[kMmaG][4];
          if (live) {
#pragma unroll
            for (int gi = 0; gi < kMmaG; ++gi) ldsm4t(vf[gi], vb + mma_v_term(dh * kHMt + g0 + gi, j32, kHalfBytes));
            if (partial) {
              const int k0 = tb + 2 * (lane & 3);
              const unsigned m01 = (k0 < n ? 0xffffu : 0u) | (k0 + 1 < n ? 0xffff0000u : 0u), m23 = (k0 + 8 < n ? 0xffffu : 0u) | (k0 + 9 < n ? 0xffff0000u : 0u);
#pragma unroll
              for (int gi = 0; gi < kMmaG; ++gi) { vf[gi][0] &= m01; vf[gi][1] &= m01; vf[gi][2] &= m23; vf[gi][3] &= m23; }
            }
          }
          if (ti == kHTP - 1 && g0 + kMmaG >= kHMt) {          // after this warp's last V ldmatrix of the stage: one arrive on the owning half's empty barrier
            __syncwarp();
            if (lane == 0) mb_arrive_dep(bar0 + 8 * (kStages + slot), kdep ^ (live ? vf[0][0] : 0u));
          }
          if (live) {
#pragma unroll
            for (int gi = 0; gi < kMmaG; ++gi) hmma16816(oacc[g0 + gi], vf[gi], pb0, pb1);
          }
        }
      }
      if (t == 0) { mb_wait(bar0 + 8 * (kStages + slot), par); produce(slot); }   // all 8 warps of the half have read the stage
      __syncwarp();
    }
    if (k + 1 < nih) { load_q(na, ntok, nqa, nqb); if (kTail) load_tail(na, trb); }
    const int4 rb = rh[2 * kG * k + 1];
    {
      float l0 = lp0, l1 = lp1;
#pragma unroll
      for (int o = 4; o <= 16; o <<= 1) { l0 += __shfl_xor_sync(0xffffffffu, l0, o); l1 += __shfl_xor_sync(0xffffffffu, l1, o); }
      float* ws = mrg + (warp * 2) * kHMS;
      if ((lane & 3) == 0) {
#pragma unroll
        for (int mt = 0; mt < kHMt; ++mt) {
          const int ld = mt * 16 + (lane >> 2);
          ws[ld] = oacc[mt][0]; ws[kHMS + ld] = oacc[mt][1]; ws[ld + 8] = oacc[mt][2]; ws[kHMS + ld + 8] = oacc[mt][3];
        }
      }
      if (lane == 0) { ws[kHDW] = mr0; ws[kHDW + 1] = l0; ws[kHMS + kHDW] = mr1; ws[kHMS + kHDW + 1] = l1; }
    }
    bar_sync_grp(1 + h);
#pragma unroll
    for (int ps = 0; ps < 256 / kHT; ++ps) {                    // 256 / kHT passes of (hq, d) = (idx >> 7, idx & 127), idx = t + ps kHT
      const int idx = t + ps * kHT, hq = idx >> 7, d = idx & 127, dd = d >> 6;   // dims d are owned by the kHO warps kHW h + kHO dd + t' (disjoint position tiles)
      const float* b = mrg + ((kHW * h + kHO * dd) * 2 + hq) * kHMS;
      float M = neg_inf();
      if (tail) M = nbp[128 + hq];
#pragma unroll
      for (int tt = 0; tt < kHO; ++tt) M = fmaxf(M, b[tt * 2 * kHMS + kHDW]);
      const float Mc = M == neg_inf() ? 0.f : M;
      float L = 0.f, O = 0.f;
#pragma unroll
      for (int tt = 0; tt < kHO; ++tt) {
        const float a = ex2(b[tt * 2 * kHMS + kHDW] - Mc);
        L = fmaf(a, b[tt * 2 * kHMS + kHDW + 1], L);
        O = fmaf(a, b[tt * 2 * kHMS + (d - dd * kHDW)], O);
      }
      if (tail) { const float a = ex2(nbp[128 + hq] - Mc); L += a; O = fmaf(a, nbp[d], O); }
      if (rb.y == 1) {
        p.ao[(size_t)rb.z * (kQHeads * kDim) + (2 * kvh + hq) * kDim + d] = f2bf(O / L);
      } else {
        float* dst = p.part + ((size_t)rb.x * 2 + hq) * kPartStride;
        __stcg(dst + d, O);
        if (d == 0) { __stcg(dst + 128, M); __stcg(dst + 129, L); }
      }
    }
    bar_sync_grp(1 + h);                                        // merge slots / park slot free for the group's next item
    ra = na; qa = nqa; qb = nqb; if (kTail) tra = trb;
  }
  __syncthreads();                                              // both halves done (a half without items arrives here directly); attend ends with this barrier
  g = (((g1c + (unsigned)tot1) & kHMask) << kHShift) | ((g0c + (unsigned)tot0) & kHMask);
  if (kG == 4) g |= (((g2c + (unsigned)tot2) & kHMask) << (2 * kHShift)) | (((g3c + (unsigned)tot3) & kHMask) << (3 * kHShift));
}
#endif

#if SKV_HALVES && SKV_GROUPS == 8
// ------------------------------------------------------------------------------------------------ P12: 8 merge-free consumer groups (SKV_HALVES 1, SKV_GROUPS 8)
// Group q = warp >> 1 (2 warps, threads 64 q ..) runs the CTA's items k = q, q + 8, .. with its own ring slots 2 q, 2 q + 1 (16-row stages: one 16-position tile per stage), its own producer
// (lane 0 of warp 2 q) and item prefetch. Both warps of a group compute the same scores (full 128-dim QK^T of the stage's tile) and the same online-softmax state (m, l per q-head); warp
// lw = warp & 1 = dims half dh does the PV product for its 64 dims. So there is no cross-warp merge, no merge slot and no group barrier (no named barrier in this mode): each warp normalises and
// stores its own 64 dims (S == 1) or its partial (O for its dims; m, l written by the dh == 0 warp). Everything a warp touches in smem is its own: q scratch (128 floats, written by lanes < 16,
// read after __syncwarp) and the tail park (the 64 dims of the tail V row this warp needs, floats); the park of item k + 1 is written after the __syncwarp that follows the fold of item k.
// Ring slots: the producer of group q refills a slot after both warps of the group arrived on its empty barrier (count 2); ring_phase: counter of group q = (g >> 3 q) & 7 (wrap mod 8 is a
// multiple of 2 kHS = 4, so slot and parity stay consistent). The final __syncthreads is kept (a group without items goes straight to it).
__device__ __forceinline__ void run_items_g8(const Params& p, int layer, unsigned char* smem, unsigned& g, const int4* rec, int ni) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, l16 = lane & 15;
  const int h = tid >> 6, t = tid & 63, dh = warp & 1;          // h = group index
  const unsigned sbase = smaddr(smem);
  const unsigned bar0 = sbase + kBarOff;
  float* mrg = reinterpret_cast<float*>(smem + kMrgOff);
  const size_t layer_row = (size_t)layer * p.cache_rows;
  const int4 zero4 = make_int4(0, 0, 0, 0);
  constexpr bool kTail = (SKV_TAIL_LDG != 0);

  const int nih = (ni - h + 7) >> 3;                            // items of this group: k = h, h + 8, ..
  const int4* rh = rec + 2 * h;                                 // item kk of the group: records rh[16 kk], rh[16 kk + 1]
  const int gh0 = (int)((g >> (3 * h)) & 7u);                   // this group's stage counter at entry

  // ---- TMA producer (lane 0 of warp 2 h): the group's stage sequence into its own slots
  int pk = -1, pp = 0, pend = 0, pkvh = 0;
  size_t prow = 0;
#if SKV_PAGED
  int pcb = 0;                                                 // block coordinate of position pp
#endif
  int4 pnxt = zero4;
  auto produce = [&](int slot) {
    if (pp >= pend) {
      if (pk + 1 >= nih) return;
      ++pk;
      const int4 it = pnxt;
      pnxt = (pk + 1 < nih) ? rh[16 * (pk + 1)] : zero4;
      pp = it.z; pend = it.w; pkvh = it.y & 15; prow = layer_row + (size_t)it.x;
#if SKV_PAGED
      pcb = page_cb(p, layer, it.x, pp);
#endif
      if (kTail && it.w - it.z > 1 && (it.w - it.z) % kStageRows == 1) --pend;   // P9d tail rule (as the consumer)
    }
    const int n = min(kStageRows, pend - pp);
    const unsigned dst = sbase + kRingOff + slot * kStageBytes, bar = bar0 + 8 * slot;
    mb_expect(bar, (unsigned)n * 512u);
#if SKV_PAGED
    int off = 0, rc = pp & (kPageRows - 1);
    const int cb = pcb;
#else
    int off = 0, rc = (int)(prow + pp);
#endif
#if SKV_TMA_FUSE
    if (n == kStageRows) {   // P13: one 4D box per tensor covers both 64-dim halves (K lo | K hi, V lo | V hi)
      SKV_FBOX(dst, (const char*)p.tmaps + (2 * kNumMaps + 0) * 128);
      SKV_FBOX(dst + 2 * kHalfBytes, (const char*)p.tmaps + (2 * kNumMaps + 1) * 128);
    } else
#endif
#pragma unroll
    for (int rs = 0; rs < kNumMaps; ++rs) {
      const int rows = kStageRows >> rs;
      if (n & rows) {
        const char* mk = (const char*)p.tmaps + (0 * kNumMaps + rs) * 128;
        const char* mv = (const char*)p.tmaps + (1 * kNumMaps + rs) * 128;
        SKV_BOX(dst + off * 128, mk, 0);
        SKV_BOX(dst + kHalfBytes + off * 128, mk, 64);
        SKV_BOX(dst + 2 * kHalfBytes + off * 128, mv, 0);
        SKV_BOX(dst + 3 * kHalfBytes + off * 128, mv, 64);
        off += rows; rc += rows;
      }
    }
    pp += n;
#if SKV_PAGED
    if (pp < pend && !(pp & (kPageRows - 1))) pcb = page_cb(p, layer, (int)prow, pp);   // next stage starts a new block
#endif
  };
  if (t == 0 && nih > 0) {
    pnxt = rh[0];
#pragma unroll
    for (int s = 0; s < kHS; ++s) produce(h * kHS + (gh0 + s) % kHS);
  }

  auto item_off = [&](const int4& a) { return ((layer_row + (size_t)a.x) * kKvHeads + (a.y & 15)) * 256 + l16 * 16; };
  auto load_q = [&](const int4& a, int tok, uint4& x, uint4& y) {
    const bf16* qp = p.q + (size_t)(tok * kQHeads + 2 * (a.y & 15)) * kDim + l16 * 8;
    x = ldg_cg4(qp); y = ldg_cg4(qp + kDim);
  };
  auto load_tail = [&](const int4& a, uint4& tr) {              // every warp: lanes 0..15 = K row, lanes 16..31 = V row of position p1 - 1
    tr = make_uint4(0, 0, 0, 0);
    if (a.w - a.z > 1 && (a.w - a.z) % kStageRows == 1)
      tr = ldg_nc(SKV_TAIL_ADDR(a));
  };
  uint4 tra = make_uint4(0, 0, 0, 0), trb = tra;
  int4 ra = nih > 0 ? rh[0] : zero4;                            // (a group without items must not read past the CTA's records)
  uint4 qa = make_uint4(0, 0, 0, 0), qb = qa;
  if (nih > 0) {
    load_q(ra, rh[1].z, qa, qb);
    if (kTail) load_tail(ra, tra);
  }
  float* nbp = mrg + kP8Park + 64 * warp;                       // this warp's tail park (64 dims of the tail V row)
  float* qsp = mrg + kP8Q + 128 * warp;                         // this warp's q scratch
  int gh = gh0;

  for (int k = 0; k < nih; ++k) {
    const int kvh = ra.y & 15, p0 = ra.z, p1 = ra.w;
    const bool tail = kTail && p1 - p0 > 1 && (p1 - p0) % kStageRows == 1;   // warp-uniform
    int4 na = zero4;
    int ntok = 0;
    uint4 nqa = make_uint4(0, 0, 0, 0), nqb = nqa;
    if (k + 1 < nih) { na = rh[16 * (k + 1)]; ntok = rh[16 * (k + 1) + 1].z; }
    float st0 = 0.f, st1 = 0.f;                                 // tail scores of the two q-heads (log2 units)
    if (tail) {
      float qr0[8], qr1[8], kf[8], vf[8];
      unpack8(qa, qr0); unpack8(qb, qr1);
      unpack8(tra, kf);
      float d0 = 0.f, d1 = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) { d0 = fmaf(kf[j], qr0[j], d0); d1 = fmaf(kf[j], qr1[j], d1); }
#pragma unroll
      for (int o = 1; o <= 8; o <<= 1) { d0 += __shfl_xor_sync(0xffffffffu, d0, o); d1 += __shfl_xor_sync(0xffffffffu, d1, o); }
      st0 = __shfl_sync(0xffffffffu, d0, 0) * kQScale; st1 = __shfl_sync(0xffffffffu, d1, 0) * kQScale;
      uint4 vn;
      vn.x = __shfl_xor_sync(0xffffffffu, tra.x, 16); vn.y = __shfl_xor_sync(0xffffffffu, tra.y, 16);
      vn.z = __shfl_xor_sync(0xffffffffu, tra.z, 16); vn.w = __shfl_xor_sync(0xffffffffu, tra.w, 16);
      if (lane < 16 && (l16 >> 3) == dh) {                      // lanes holding dims 64 dh .. 64 dh + 63 of the V row
        unpack8(vn, vf);
        *reinterpret_cast<float4*>(nbp + (l16 & 7) * 8) = make_float4(vf[0], vf[1], vf[2], vf[3]);
        *reinterpret_cast<float4*>(nbp + (l16 & 7) * 8 + 4) = make_float4(vf[4], vf[5], vf[6], vf[7]);
      }
    }
    float mr0 = neg_inf(), mr1 = neg_inf(), lp0 = 0.f, lp1 = 0.f;
    float oacc[kHMt][4];
#pragma unroll
    for (int i = 0; i < kHMt; ++i) oacc[i][0] = oacc[i][1] = oacc[i][2] = oacc[i][3] = 0.f;
    {
      unsigned* qw = reinterpret_cast<unsigned*>(qsp);
      if (lane < 16) {
        qw[mma_q_word(l16, 0)] = qa.x; qw[mma_q_word(l16, 1)] = qa.y; qw[mma_q_word(l16, 2)] = qa.z; qw[mma_q_word(l16, 3)] = qa.w;
        qw[64 + mma_q_word(l16, 0)] = qb.x; qw[64 + mma_q_word(l16, 1)] = qb.y; qw[64 + mma_q_word(l16, 2)] = qb.z; qw[64 + mma_q_word(l16, 3)] = qb.w;
      }
      __syncwarp();
    }
    const int len = p1 - p0 - (tail ? 1 : 0);
    const int ns = (len + kStageRows - 1) / kStageRows;
    const unsigned qsa = smaddr(qsp) + mma_q_off(lane, 0);
    for (int j = 0; j < ns; ++j, ++gh) {
      const int slot = h * kHS + gh % kHS;
      const unsigned par = (gh / kHS) & 1;
      mb_wait(bar0 + 8 * slot, par);
      const int n = min(kStageRows, len - j * kStageRows);      // valid positions of the stage (>= 1)
      const unsigned sb = sbase + kRingOff + slot * kStageBytes;
      const unsigned j32 = mma_j32(lane);
      unsigned kdep = 0;
      const unsigned kb = sb + mma_k_base(lane, 0), vb = sb + mma_v_base(lane, 0);
      const bool partial = n < 16;
      unsigned pb0, pb1;
      {
        float sa[4] = {0.f, 0.f, 0.f, 0.f}, sb2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          unsigned k0[4];
          ldsm4(k0, kb + mma_k_term(ks, j32, kHalfBytes));
          uint2 qb0 = make_uint2(0u, 0u);
          if ((lane >> 2) < 2) qb0 = lds64(qsa + ks * 32);
          hmma16816((ks & 1) ? sb2 : sa, k0, qb0.x, qb0.y);
          kdep ^= k0[0];
        }
        float s0 = sa[0] + sb2[0], s1 = sa[1] + sb2[1], s2 = sa[2] + sb2[2], s3 = sa[3] + sb2[3];
        if (partial) {
          const int pos = lane >> 2;
          if (pos >= n) { s0 = neg_inf(); s1 = neg_inf(); }
          if (pos + 8 >= n) { s2 = neg_inf(); s3 = neg_inf(); }
        }
        float c0 = fmaxf(s0, s2), c1 = fmaxf(s1, s3);
#pragma unroll
        for (int o = 4; o <= 16; o <<= 1) { c0 = fmaxf(c0, __shfl_xor_sync(0xffffffffu, c0, o)); c1 = fmaxf(c1, __shfl_xor_sync(0xffffffffu, c1, o)); }
        const float n0 = fmaxf(mr0, c0 * kQScale), n1 = fmaxf(mr1, c1 * kQScale);
        const float a0 = ex2(mr0 - n0), a1 = ex2(mr1 - n1);
        mr0 = n0; mr1 = n1;
        const float p0v = ex2(fmaf(s0, kQScale, -n0)), p1v = ex2(fmaf(s1, kQScale, -n1)), p2v = ex2(fmaf(s2, kQScale, -n0)), p3v = ex2(fmaf(s3, kQScale, -n1));
        lp0 = fmaf(lp0, a0, p0v + p2v); lp1 = fmaf(lp1, a1, p1v + p3v);
        if (__any_sync(0xffffffffu, (lane & 3) == 0 && (a0 != 1.f || a1 != 1.f))) {
#pragma unroll
          for (int mt = 0; mt < kHMt; ++mt) { oacc[mt][0] *= a0; oacc[mt][1] *= a1; oacc[mt][2] *= a0; oacc[mt][3] *= a1; }
        }
        pb0 = movm_t(pack_bf16x2(p0v, p1v)); pb1 = movm_t(pack_bf16x2(p2v, p3v));
      }
#pragma unroll
      for (int g0 = 0; g0 < kHMt; g0 += kMmaG) {
        unsigned vf[kMmaG][4];
#pragma unroll
        for (int gi = 0; gi < kMmaG; ++gi) ldsm4t(vf[gi], vb + mma_v_term(dh * kHMt + g0 + gi, j32, kHalfBytes));
        if (partial) {
          const int k0 = 2 * (lane & 3);
          const unsigned m01 = (k0 < n ? 0xffffu : 0u) | (k0 + 1 < n ? 0xffff0000u : 0u), m23 = (k0 + 8 < n ? 0xffffu : 0u) | (k0 + 9 < n ? 0xffff0000u : 0u);
#pragma unroll
          for (int gi = 0; gi < kMmaG; ++gi) { vf[gi][0] &= m01; vf[gi][1] &= m01; vf[gi][2] &= m23; vf[gi][3] &= m23; }
        }
        if (g0 + kMmaG >= kHMt) {                               // after this warp's last V ldmatrix of the stage: one arrive on the group's empty barrier
          __syncwarp();
          if (lane == 0) mb_arrive_dep(bar0 + 8 * (kStages + slot), kdep ^ vf[0][0]);
        }
#pragma unroll
        for (int gi = 0; gi < kMmaG; ++gi) hmma16816(oacc[g0 + gi], vf[gi], pb0, pb1);
      }
      if (t == 0) { mb_wait(bar0 + 8 * (kStages + slot), par); produce(slot); }   // both warps of the group have read the stage
      __syncwarp();
    }
    if (k + 1 < nih) { load_q(na, ntok, nqa, nqb); if (kTail) load_tail(na, trb); }
    const int4 rb = rh[16 * k + 1];
    float l0 = lp0, l1 = lp1;
#pragma unroll
    for (int o = 4; o <= 16; o <<= 1) { l0 += __shfl_xor_sync(0xffffffffu, l0, o); l1 += __shfl_xor_sync(0xffffffffu, l1, o); }
    if (tail) {                                                 // fold the tail position (single-position state) into this warp's state
      const float M0 = fmaxf(mr0, st0), M1 = fmaxf(mr1, st1);
      const float a0 = ex2(mr0 - M0), a1 = ex2(mr1 - M1), e0 = ex2(st0 - M0), e1 = ex2(st1 - M1);
      mr0 = M0; mr1 = M1;
      l0 = fmaf(l0, a0, e0); l1 = fmaf(l1, a1, e1);
      const int r = lane >> 2;
#pragma unroll
      for (int mt = 0; mt < kHMt; ++mt) {
        const float va = nbp[mt * 16 + r], vb8 = nbp[mt * 16 + r + 8];
        oacc[mt][0] = fmaf(oacc[mt][0], a0, e0 * va); oacc[mt][1] = fmaf(oacc[mt][1], a1, e1 * va);
        oacc[mt][2] = fmaf(oacc[mt][2], a0, e0 * vb8); oacc[mt][3] = fmaf(oacc[mt][3], a1, e1 * vb8);
      }
      __syncwarp();                                             // park free for this warp's next item
    }
    if ((lane & 3) == 0) {
      const int d0 = dh * kHDW + (lane >> 2);
      if (rb.y == 1) {
        bf16* ao0 = p.ao + (size_t)rb.z * (kQHeads * kDim) + (2 * kvh) * kDim;
#pragma unroll
        for (int mt = 0; mt < kHMt; ++mt) {
          ao0[d0 + mt * 16] = f2bf(oacc[mt][0] / l0); ao0[kDim + d0 + mt * 16] = f2bf(oacc[mt][1] / l1);
          ao0[d0 + mt * 16 + 8] = f2bf(oacc[mt][2] / l0); ao0[kDim + d0 + mt * 16 + 8] = f2bf(oacc[mt][3] / l1);
        }
      } else {
        float* dst0 = p.part + ((size_t)rb.x * 2) * kPartStride;
        float* dst1 = dst0 + kPartStride;
#pragma unroll
        for (int mt = 0; mt < kHMt; ++mt) {
          __stcg(dst0 + d0 + mt * 16, oacc[mt][0]); __stcg(dst1 + d0 + mt * 16, oacc[mt][1]);
          __stcg(dst0 + d0 + mt * 16 + 8, oacc[mt][2]); __stcg(dst1 + d0 + mt * 16 + 8, oacc[mt][3]);
        }
        if (dh == 0 && lane == 0) { __stcg(dst0 + 128, mr0); __stcg(dst0 + 129, l0); __stcg(dst1 + 128, mr1); __stcg(dst1 + 129, l1); }
      }
    }
    ra = na; qa = nqa; qb = nqb; if (kTail) tra = trb;
  }
  __syncthreads();                                              // all groups done (a group without items arrives here directly); attend ends with this barrier
  unsigned gout = 0;                                            // every thread stores the same packed counters: group q's counter + its stage total of this call
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    int tot = 0;
    for (int k = q; k < ni; k += 8) {
      const int4 it = rec[2 * k];
      const int ln = it.w - it.z;
      tot += (ln - ((kTail && ln > 1 && ln % kStageRows == 1) ? 1 : 0) + kStageRows - 1) / kStageRows;
    }
    gout |= (((g >> (3 * q)) + (unsigned)tot) & 7u) << (3 * q);
  }
  g = gout;
}
#endif

#if SKV_SI
// ------------------------------------------------------------------------------------------------ P21: stage-interleaved groups (SKV_SI 1; SKV_HALVES 1, SKV_GROUPS 8)
// The CTA's items (record order) form one stream of 16-row stages; item i has ns_i stages (as run_items_g8: tail rule, ceil(len / 16)), gs[i] = stages before item i. Global stage s goes to group
// s % 8, so every group has floor / ceil(T / 8) stages whatever the item sizes (the plan only balances CTAs: plan groups=1). Group q = warp >> 1 (2 warps; warp & 1 = dims half dh) runs its stages
// s = q, q + 8, .. with its own ring slots 2 q, 2 q + 1 and its own producer (lane 0 of warp 2 q); per item it touches ("visit": its stages of the item, consecutive in its own stream) it keeps a fresh
// (m, l, o) as in run_items_g8 and at the end of the visit PARKS it (o of the warp's 64 dims x 2 q-heads, m[2], l[2]: kPartStride floats) in park slot (item % P), entry dh * 8 + group.
// Contributing groups of item i: (gs[i] + j) % 8, j < min(8, ns_i). Per (slot, dh) there is an arrival counter and a generation (items of the slot already merged). A warp parks item k of slot
// sl = k % P only when gen == k / P (the previous item of the slot has been merged and its entries read), writes its entry, fences, atomicAdd(arrivals); the arrival that makes min(8, ns) merges
// the entries of its dh (online-softmax combine over the contributing groups, one float4 per lane), stores like run_items_g8 (S == 1: normalised bf16 ao; else partial o / m / l at part slot rb.x), then
// resets arrivals, fences and publishes gen + 1. No CTA barrier per item.
// Deadlock freedom: a warp blocks only (a) on ring barriers, whose producers are the group's own warp 2 q (no cross-group wait) and (b) in the park wait. At the park point both warps of the group have consumed all
// stages of the visit (the producer thread waited the group's empty barrier of the last stage and already refilled it: the next visit's two stages are in flight), so a parked-waiting group blocks nothing but itself.
// Let k* be the lowest item not fully merged: every contributor either has parked k* or will reach its park (items < k* are merged, so it never waited on them; its ring data does not depend on any other
// group), and gen(sl*) == k* / P since all earlier items of the slot are merged: nobody waits for k*, the last arrival merges it, so progress. The merge itself waits for nothing.
// P25b, SKV_POOL 2 (in-stream claim): the item list GROWS while the stream runs; attend passes the first pool ticket and the call claims pool items itself (no per-item call, no CTA barrier, no ring drain).
// Shared state: T_known = stages known (gs[nt]), done. Only thread 0 (group 0's producer, the claimer) writes them, after the record (tab) and prefix gs[nt + 1] are written and a __threadfence_block. An item is
// appended whole. The claimer claims (one atomicAdd on p.pool_ctr[layer], always one claim in flight) while T_known - 8 pc < SKV_LOOKAHEAD (pc = group 0's produced stages): in setup (first ticket), after each consumed
// stage of group 0 and in its wait loop. Success: append, publish T_known, issue the next claim. Failure: done = 1 (after the last publish), the ticket npool + grid - 1 resets the counter. Every CTA makes exactly one failing
// claim (a success with kSiCap items traps: the claim cannot be returned and the counter reset needs it). Readers read done BEFORE T_known (done = 1 then makes T_known final), then fence, then read tab / gs.
// Producers never wait for an unknown stage: try_produce (thread 0 of each group) produces the stages pc < consumed + kHS that are known and defers the rest; the deferred ones are produced when thread 0
// pumps again (after each consumption, at the top of each visit, in the wait loop). A consumer waits for an unknown next stage only at the top of a visit, i.e. AFTER it parked and merged the previous visit
// (the next visit's q is prefetched only when its stage was already known at the visit start). Deadlock argument (extends the one above): the new waits are (c) a group waiting for an unknown stage: it holds no
// park entry and no open item, so no merge needs it; and (d) the claimer's progress, which every such wait needs. The claimer is blocked only by what blocks any group (ring barriers of known stages, park wait).
// Let k* be the lowest item not fully merged (all appended items are whole and known to every contributor): items < k* are merged, so no warp waits on them; every contributor reaches park of k* using only stages
// of known items, which its group's thread 0 produces without waiting (produce needs the partner warp's read of stage - kHS: an earlier stage, whose consumer is not blocked, as before; a warp parked-waiting on
// an item >= k* has consumed everything it was asked to read, and every stage the partner still needs was produced before thread 0 consumed it); so k* merges. When all appended items are merged nobody holds
// a park wait, the claimer is free: its lookahead test holds once group 0 produced all known stages (8 pc >= T_known - 7), so it claims until the failure sets done and every group's wait ends.
// Records and prefix are copied to smem once per call (warp 0 scan, one __syncthreads); counters / gens are zeroed per call. ring_phase: group q's counter + its stage total (T - q + 7) / 8.
__device__ __forceinline__ void run_items_si(const Params& p, int layer, unsigned char* smem, unsigned& g, const int4* rec, int ni
#if SKV_POOL == 2
                                             , unsigned tk0      // the first pool ticket (issued by attend before the call, valid in thread 0 only)
#endif
                                             ) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, l16 = lane & 15;
  const int h = tid >> 6, t = tid & 63, dh = warp & 1;          // h = group index
  const unsigned sbase = smaddr(smem);
  const unsigned bar0 = sbase + kBarOff;
  float* mrg = reinterpret_cast<float*>(smem + kMrgOff);
  int4* tab = reinterpret_cast<int4*>(smem + kSiTabOff);        // tab[2 k], tab[2 k + 1] = record k
  int* gs = reinterpret_cast<int*>(smem + kSiGsOff);
  int* ctr = reinterpret_cast<int*>(smem + kSiCtrOff);          // arrivals [slot][dh] at 2 sl + dh, gens at 2 kSiP + 2 sl + dh
  float* park = reinterpret_cast<float*>(smem + kSiParkOff);
  const size_t layer_row = (size_t)layer * p.cache_rows;
  const int4 zero4 = make_int4(0, 0, 0, 0);
  constexpr bool kTail = (SKV_TAIL_LDG != 0);

#if SKV_POOL == 2
  // ---- P25b in-stream pool: shared state T_known / done, the claimer (thread 0) and the producer position of group 0 (pc), see the header comment
  volatile int* tkp = reinterpret_cast<volatile int*>(smem + kSiStOff);
  volatile int* donep = tkp + 1;
  const int4* pool = p.items + (size_t)gridDim.x * p.items_stride * 2;
#if SKV_IFACE == 2
  const int npool = pool[1].y >> 8;
#else
  const int npool = pool[1].w;
#endif
  int pc = 0, pk = 0;                                           // producer position of this group (stages produced, item of the next one)
  unsigned tkc = tk0;                                           // claimer (tid 0 only): the ticket in flight
  int ntc = ni, Tc = 0;                                         // claimer: items known, stages known
  bool cdone = false;
  auto claim_step = [&](bool force) {                           // tid 0 only. One claim at most: resolve the ticket in flight; success = append + publish + issue the next one, failure = done
    if (cdone) return;
    if (!force && Tc - 8 * pc >= SKV_LOOKAHEAD) return;
    const unsigned tt = tkc;
    if (tt < (unsigned)npool) {
      if (ntc >= kSiCap) __trap();                              // no room for a claimed item: cannot be given back (the counter reset needs every CTA's failing claim)
      const int4 ra = pool[2 * (size_t)tt], rb = pool[2 * (size_t)tt + 1];
      tab[2 * ntc] = ra; tab[2 * ntc + 1] = rb;
      const int ln = ra.w - ra.z;
      Tc += (ln - ((kTail && ln > 1 && ln % kStageRows == 1) ? 1 : 0) + kStageRows - 1) / kStageRows;
      gs[ntc + 1] = Tc; ++ntc;
      __threadfence_block();                                    // record + prefix before the publish
      *tkp = Tc;
      tkc = atomicAdd(&p.pool_ctr[layer], 1u);                  // the next claim, in flight until the next step
    } else {
      cdone = true;
      if (tt == (unsigned)npool + gridDim.x - 1) p.pool_ctr[layer] = 0;   // the last failing claim of the layer: every other claim is done (as SKV_POOL 1)
      __threadfence_block();                                    // every append is published before done
      *donep = 1;
    }
  };
#endif

  // ---- per-call setup: item records + stage prefix (warp 0), counters
  if (ni > kSiCap) __trap();
  if (tid < 4 * kSiP) ctr[tid] = 0;
  if (warp == 0) {
    int carry = 0;
    if (lane == 0) gs[0] = 0;
    for (int b0 = 0; b0 < ni; b0 += 32) {
      const int i = b0 + lane;
      int ns = 0;
      if (i < ni) {
        const int4 a = rec[2 * i], b = rec[2 * i + 1];
        tab[2 * i] = a; tab[2 * i + 1] = b;
        const int ln = a.w - a.z;
        ns = (ln - ((kTail && ln > 1 && ln % kStageRows == 1) ? 1 : 0) + kStageRows - 1) / kStageRows;
      }
      int incl = ns;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) { const int v = __shfl_up_sync(0xffffffffu, incl, o); if (lane >= o) incl += v; }
      if (i < ni) gs[i + 1] = carry + incl;
      carry += __shfl_sync(0xffffffffu, incl, 31);
    }
#if SKV_POOL == 2
    __syncwarp();
    if (lane == 0) { Tc = carry; *donep = 0; *tkp = Tc; claim_step(true); }   // resolve the first claim now (its latency hid behind attend's setup)
#endif
  }
  __syncthreads();
#if SKV_POOL != 2
  const int T = gs[ni];                                         // total stages of the CTA
  const int nstg = (T - h + 7) >> 3;                            // this group's stages: s = h + 8 c, c < nstg
#endif
  const int gh0 = (int)((g >> (3 * h)) & 7u);                   // this group's ring counter at entry

  // ---- TMA producer (lane 0 of warp 2 h): the group's stage sequence into its own slots
#if SKV_POOL != 2
  int pc = 0, pk = 0;
#endif
  auto produce = [&](int slot) {
#if SKV_POOL != 2
    if (pc >= nstg) return;
#endif
    const int sp = h + 8 * pc;
    while (gs[pk + 1] <= sp) ++pk;
    const int4 it = tab[2 * pk];
    const int ln = it.w - it.z;
    const int len = ln - ((kTail && ln > 1 && ln % kStageRows == 1) ? 1 : 0);
    const int j = sp - gs[pk];
    const int n = min(kStageRows, len - j * kStageRows);
    const int pkvh = it.y & 15;
#if SKV_PAGED
    const int pos0 = it.z + j * kStageRows;                     // a stage never crosses a cache block (items start at multiples of the stage rows: plan align)
    int rc = pos0 & (kPageRows - 1);
    const int cb = page_cb(p, layer, it.x, pos0);
#else
    int rc = (int)(layer_row + (size_t)it.x) + it.z + j * kStageRows;
#endif
    const unsigned dst = sbase + kRingOff + slot * kStageBytes, bar = bar0 + 8 * slot;
    mb_expect(bar, (unsigned)n * 512u);
    int off = 0;
#if SKV_TMA_FUSE
    if (n == kStageRows) {   // P13: one 4D box per tensor covers both 64-dim halves (K lo | K hi, V lo | V hi)
      SKV_FBOX(dst, (const char*)p.tmaps + (2 * kNumMaps + 0) * 128);
      SKV_FBOX(dst + 2 * kHalfBytes, (const char*)p.tmaps + (2 * kNumMaps + 1) * 128);
    } else
#endif
#pragma unroll
    for (int rs = 0; rs < kNumMaps; ++rs) {
      const int rows = kStageRows >> rs;
      if (n & rows) {
        const char* mk = (const char*)p.tmaps + (0 * kNumMaps + rs) * 128;
        const char* mv = (const char*)p.tmaps + (1 * kNumMaps + rs) * 128;
        SKV_BOX(dst + off * 128, mk, 0);
        SKV_BOX(dst + kHalfBytes + off * 128, mk, 64);
        SKV_BOX(dst + 2 * kHalfBytes + off * 128, mv, 0);
        SKV_BOX(dst + 3 * kHalfBytes + off * 128, mv, 64);
        off += rows; rc += rows;
      }
    }
    ++pc;
  };
#if SKV_POOL == 2
  // known-stage count with acquire semantics (volatile read, then a fence: the table / prefix reads that follow see the publisher's writes)
  auto rd_known = [&]() -> int { const int v = *tkp; __threadfence_block(); return v; };
  // t == 0 only. Produce the group's next stages while a ring slot is free (pc < cons + kHS: stage pc - kHS was consumed by both warps, its empty barrier is waited) and the stage is known.
  // Never waits for a stage to become known: a deferred stage is produced by a later call.
  auto try_produce = [&](int cons) {
    while (pc < cons + kHS) {
      if (h + 8 * pc >= rd_known()) break;
      produce(h * kHS + (gh0 + pc) % kHS);
    }
  };
  // All threads of the group. true = stage sp is known; false = no such stage (done and sp >= T_known). done is read BEFORE T_known: done = 1 implies the T_known read after it is final.
  // Thread 0 pumps while it waits: group 0's thread 0 claims, every group's thread 0 produces the stages that became known.
  auto wait_stage = [&](int sp, int cons) -> bool {
#if defined(SKV_DEBUG_WAIT) && (SKV_DEBUG_WAIT + 0)
    unsigned long long polls = 0;
#endif
    for (;;) {
      const int d = *donep;
      __threadfence_block();
      if (sp < *tkp) { __threadfence_block(); return true; }
      if (d) return false;
      if (t == 0) { if (h == 0) claim_step(false); try_produce(cons); }
      __nanosleep(32);
#if defined(SKV_DEBUG_WAIT) && (SKV_DEBUG_WAIT + 0)
      if (++polls == (1ull << 26) && lane == 0) printf("SKV_DEBUG_WAIT si-stream cta %d layer %d warp %d sp %d T_known %d done %d\n", blockIdx.x, layer, warp, sp, (int)*tkp, (int)*donep);
#endif
    }
  };
  if (t == 0) { try_produce(0); if (h == 0) claim_step(false); }
#else
  if (t == 0 && nstg > 0) {
#pragma unroll
    for (int s = 0; s < kHS; ++s) produce(h * kHS + (gh0 + s) % kHS);
  }
#endif

  auto load_q = [&](const int4& a, int tok, uint4& x, uint4& y) {
    const bf16* qp = p.q + (size_t)(tok * kQHeads + 2 * (a.y & 15)) * kDim + l16 * 8;
    x = ldg_cg4(qp); y = ldg_cg4(qp + kDim);
  };
  auto load_tail = [&](const int4& a, uint4& tr) {              // every warp: lanes 0..15 = K row, lanes 16..31 = V row of position p1 - 1
    tr = make_uint4(0, 0, 0, 0);
    if (a.w - a.z > 1 && (a.w - a.z) % kStageRows == 1)
#if SKV_PAGED
      tr = ldg_nc(page_row_ptr(p, layer, a, a.w - 1, lane));
#else
      tr = ldg_nc((const char*)(lane < 16 ? p.kc : p.vc) + ((((layer_row + (size_t)a.x) * kKvHeads + (a.y & 15)) * 256 + l16 * 16)) + (size_t)(a.w - 1) * 2048);
#endif
  };
  uint4 tra = make_uint4(0, 0, 0, 0), trb = tra;
  uint4 qa = make_uint4(0, 0, 0, 0), qb = qa;
  int kc = 0;                                                   // item of this group's next stage
#if SKV_POOL == 2
  bool pre = false;                                             // q of the next visit prefetched (its first stage was known at the start of the previous visit)
#else
  if (nstg > 0) {
    while (gs[kc + 1] <= h) ++kc;
    load_q(tab[2 * kc], tab[2 * kc + 1].z, qa, qb);
    if (kTail && h == gs[kc]) load_tail(tab[2 * kc], tra);
  }
#endif
  float* nbp = mrg + kP8Park + 64 * warp;                       // this warp's tail park (64 dims of the tail V row)
  float* qsp = mrg + kP8Q + 128 * warp;                         // this warp's q scratch
  int gh = gh0;

#if SKV_POOL == 2
  for (int c = 0;;) {
    const int s0 = h + 8 * c;
    if (!pre) {                                                 // nothing parked or pending here: wait for the stage to be known (or for the end)
      if (!wait_stage(s0, c)) break;
      __syncwarp();
      if (c > 0) ++kc;                                          // the next stage's item is after the visited one
      while (gs[kc + 1] <= s0) ++kc;
      load_q(tab[2 * kc], tab[2 * kc + 1].z, qa, qb);
      if (kTail && s0 == gs[kc]) load_tail(tab[2 * kc], tra);
    }
    if (t == 0) try_produce(c);                                 // stage s0 is known: produced (deferred refills of the previous visit)
#else
  for (int c = 0; c < nstg;) {
    const int s0 = h + 8 * c;
#endif
    const int4 ra = tab[2 * kc], rb = tab[2 * kc + 1];
    const int kvh = ra.y & 15, p0 = ra.z, p1 = ra.w;
    const int gsk = gs[kc], gse = gs[kc + 1];
    const bool tail = kTail && s0 == gsk && p1 - p0 > 1 && (p1 - p0) % kStageRows == 1;   // warp-uniform: the owner of stage 0 folds the tail position
    const int len = p1 - p0 - ((kTail && p1 - p0 > 1 && (p1 - p0) % kStageRows == 1) ? 1 : 0);
    const int nv = (gse - 1 - s0) / 8 + 1;                      // stages of this visit
    const int cn = c + nv;
#if SKV_POOL == 2
    const bool has_next = __shfl_sync(0xffffffffu, rd_known(), 0) > h + 8 * cn;   // snapshot (lane 0's read, so the warp agrees), never waits: an unknown next stage is waited for after this visit's park
#else
    const bool has_next = cn < nstg;
#endif
    int kn = kc, ntok = 0;
    int4 na = zero4;
    bool nown = false;
    uint4 nqa = make_uint4(0, 0, 0, 0), nqb = nqa;
    if (has_next) {
      const int sn = h + 8 * cn;
      kn = kc + 1;
      while (gs[kn + 1] <= sn) ++kn;
      na = tab[2 * kn]; ntok = tab[2 * kn + 1].z; nown = (sn == gs[kn]);
    }
    float st0 = 0.f, st1 = 0.f;                                 // tail scores of the two q-heads (log2 units)
    if (tail) {
      float qr0[8], qr1[8], kf[8], vf[8];
      unpack8(qa, qr0); unpack8(qb, qr1);
      unpack8(tra, kf);
      float d0 = 0.f, d1 = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) { d0 = fmaf(kf[j], qr0[j], d0); d1 = fmaf(kf[j], qr1[j], d1); }
#pragma unroll
      for (int o = 1; o <= 8; o <<= 1) { d0 += __shfl_xor_sync(0xffffffffu, d0, o); d1 += __shfl_xor_sync(0xffffffffu, d1, o); }
      st0 = __shfl_sync(0xffffffffu, d0, 0) * kQScale; st1 = __shfl_sync(0xffffffffu, d1, 0) * kQScale;
      uint4 vn;
      vn.x = __shfl_xor_sync(0xffffffffu, tra.x, 16); vn.y = __shfl_xor_sync(0xffffffffu, tra.y, 16);
      vn.z = __shfl_xor_sync(0xffffffffu, tra.z, 16); vn.w = __shfl_xor_sync(0xffffffffu, tra.w, 16);
      if (lane < 16 && (l16 >> 3) == dh) {                      // lanes holding dims 64 dh .. 64 dh + 63 of the V row
        unpack8(vn, vf);
        *reinterpret_cast<float4*>(nbp + (l16 & 7) * 8) = make_float4(vf[0], vf[1], vf[2], vf[3]);
        *reinterpret_cast<float4*>(nbp + (l16 & 7) * 8 + 4) = make_float4(vf[4], vf[5], vf[6], vf[7]);
      }
    }
    float mr0 = neg_inf(), mr1 = neg_inf(), lp0 = 0.f, lp1 = 0.f;
    float oacc[kHMt][4];
#pragma unroll
    for (int i = 0; i < kHMt; ++i) oacc[i][0] = oacc[i][1] = oacc[i][2] = oacc[i][3] = 0.f;
    {
      unsigned* qw = reinterpret_cast<unsigned*>(qsp);
      if (lane < 16) {
        qw[mma_q_word(l16, 0)] = qa.x; qw[mma_q_word(l16, 1)] = qa.y; qw[mma_q_word(l16, 2)] = qa.z; qw[mma_q_word(l16, 3)] = qa.w;
        qw[64 + mma_q_word(l16, 0)] = qb.x; qw[64 + mma_q_word(l16, 1)] = qb.y; qw[64 + mma_q_word(l16, 2)] = qb.z; qw[64 + mma_q_word(l16, 3)] = qb.w;
      }
      __syncwarp();
    }
    const unsigned qsa = smaddr(qsp) + mma_q_off(lane, 0);
    for (int jj = 0; jj < nv; ++jj, ++gh) {
      if (jj == nv - 1 && has_next) { load_q(na, ntok, nqa, nqb); if (kTail && nown) load_tail(na, trb); }   // next visit's q (and tail) in flight during the last stage
      const int j = (s0 + 8 * jj - gsk);                        // stage index within the item
      const int slot = h * kHS + gh % kHS;
      const unsigned par = (gh / kHS) & 1;
      mb_wait(bar0 + 8 * slot, par);
      const int n = min(kStageRows, len - j * kStageRows);      // valid positions of the stage (>= 1)
      const unsigned sb = sbase + kRingOff + slot * kStageBytes;
      const unsigned j32 = mma_j32(lane);
      unsigned kdep = 0;
      const unsigned kb = sb + mma_k_base(lane, 0), vb = sb + mma_v_base(lane, 0);
      const bool partial = n < 16;
      unsigned pb0, pb1;
      {
        float sa[4] = {0.f, 0.f, 0.f, 0.f}, sb2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          unsigned k0[4];
          ldsm4(k0, kb + mma_k_term(ks, j32, kHalfBytes));
          uint2 qb0 = make_uint2(0u, 0u);
          if ((lane >> 2) < 2) qb0 = lds64(qsa + ks * 32);
          hmma16816((ks & 1) ? sb2 : sa, k0, qb0.x, qb0.y);
          kdep ^= k0[0];
        }
        float s0v = sa[0] + sb2[0], s1v = sa[1] + sb2[1], s2v = sa[2] + sb2[2], s3v = sa[3] + sb2[3];
        if (partial) {
          const int pos = lane >> 2;
          if (pos >= n) { s0v = neg_inf(); s1v = neg_inf(); }
          if (pos + 8 >= n) { s2v = neg_inf(); s3v = neg_inf(); }
        }
        float c0 = fmaxf(s0v, s2v), c1 = fmaxf(s1v, s3v);
#pragma unroll
        for (int o = 4; o <= 16; o <<= 1) { c0 = fmaxf(c0, __shfl_xor_sync(0xffffffffu, c0, o)); c1 = fmaxf(c1, __shfl_xor_sync(0xffffffffu, c1, o)); }
        const float n0 = fmaxf(mr0, c0 * kQScale), n1 = fmaxf(mr1, c1 * kQScale);
        const float a0 = ex2(mr0 - n0), a1 = ex2(mr1 - n1);
        mr0 = n0; mr1 = n1;
        const float p0v = ex2(fmaf(s0v, kQScale, -n0)), p1v = ex2(fmaf(s1v, kQScale, -n1)), p2v = ex2(fmaf(s2v, kQScale, -n0)), p3v = ex2(fmaf(s3v, kQScale, -n1));
        lp0 = fmaf(lp0, a0, p0v + p2v); lp1 = fmaf(lp1, a1, p1v + p3v);
        if (__any_sync(0xffffffffu, (lane & 3) == 0 && (a0 != 1.f || a1 != 1.f))) {
#pragma unroll
          for (int mt = 0; mt < kHMt; ++mt) { oacc[mt][0] *= a0; oacc[mt][1] *= a1; oacc[mt][2] *= a0; oacc[mt][3] *= a1; }
        }
        pb0 = movm_t(pack_bf16x2(p0v, p1v)); pb1 = movm_t(pack_bf16x2(p2v, p3v));
      }
#pragma unroll
      for (int g0 = 0; g0 < kHMt; g0 += kMmaG) {
        unsigned vf[kMmaG][4];
#pragma unroll
        for (int gi = 0; gi < kMmaG; ++gi) ldsm4t(vf[gi], vb + mma_v_term(dh * kHMt + g0 + gi, j32, kHalfBytes));
        if (partial) {
          const int k0 = 2 * (lane & 3);
          const unsigned m01 = (k0 < n ? 0xffffu : 0u) | (k0 + 1 < n ? 0xffff0000u : 0u), m23 = (k0 + 8 < n ? 0xffffu : 0u) | (k0 + 9 < n ? 0xffff0000u : 0u);
#pragma unroll
          for (int gi = 0; gi < kMmaG; ++gi) { vf[gi][0] &= m01; vf[gi][1] &= m01; vf[gi][2] &= m23; vf[gi][3] &= m23; }
        }
        if (g0 + kMmaG >= kHMt) {                               // after this warp's last V ldmatrix of the stage: one arrive on the group's empty barrier
          __syncwarp();
          if (lane == 0) mb_arrive_dep(bar0 + 8 * (kStages + slot), kdep ^ vf[0][0]);
        }
#pragma unroll
        for (int gi = 0; gi < kMmaG; ++gi) hmma16816(oacc[g0 + gi], vf[gi], pb0, pb1);
      }
#if SKV_POOL == 2
      if (t == 0) {
        if (h == 0) claim_step(false);                          // before the empty wait: the claim hides behind the partner warp's read of the stage
        mb_wait(bar0 + 8 * (kStages + slot), par);              // both warps of the group have read the stage
        try_produce(c + jj + 1);
      }
#else
      if (t == 0) { mb_wait(bar0 + 8 * (kStages + slot), par); produce(slot); }   // both warps of the group have read the stage
#endif
      __syncwarp();
    }
    float l0 = lp0, l1 = lp1;
#pragma unroll
    for (int o = 4; o <= 16; o <<= 1) { l0 += __shfl_xor_sync(0xffffffffu, l0, o); l1 += __shfl_xor_sync(0xffffffffu, l1, o); }
    if (tail) {                                                 // fold the tail position (single-position state) into this warp's state
      const float M0 = fmaxf(mr0, st0), M1 = fmaxf(mr1, st1);
      const float a0 = ex2(mr0 - M0), a1 = ex2(mr1 - M1), e0 = ex2(st0 - M0), e1 = ex2(st1 - M1);
      mr0 = M0; mr1 = M1;
      l0 = fmaf(l0, a0, e0); l1 = fmaf(l1, a1, e1);
      const int r = lane >> 2;
#pragma unroll
      for (int mt = 0; mt < kHMt; ++mt) {
        const float va = nbp[mt * 16 + r], vb8 = nbp[mt * 16 + r + 8];
        oacc[mt][0] = fmaf(oacc[mt][0], a0, e0 * va); oacc[mt][1] = fmaf(oacc[mt][1], a1, e1 * va);
        oacc[mt][2] = fmaf(oacc[mt][2], a0, e0 * vb8); oacc[mt][3] = fmaf(oacc[mt][3], a1, e1 * vb8);
      }
      __syncwarp();                                             // park free for this warp's next item
    }
    // ---- park (wait for the slot's previous item to be merged), arrive, merge if last
    const int sl = kc % kSiP;
    const int tk = kc / kSiP;
    const int cnt = min(8, gse - gsk);                          // contributing groups = expected arrivals per dh
    int* arrp = ctr + 2 * sl + dh;
    volatile int* genp = ctr + 2 * kSiP + 2 * sl + dh;
#if defined(SKV_DEBUG_WAIT) && (SKV_DEBUG_WAIT + 0)
    if (lane == 0) {                                            // debug: count the polls, report once after 2^26, keep spinning
      unsigned long long polls = 0;
      while (*genp != tk) {
        __nanosleep(20);
        if (++polls == (1ull << 26)) printf("SKV_DEBUG_WAIT si-park cta %d layer %d warp %d item %d slot %d gen %d want %d\n", blockIdx.x, layer, warp, kc, sl, (int)*genp, tk);
      }
    }
#else
    if (lane == 0) { while (*genp != tk) __nanosleep(20); }
#endif
    __syncwarp();
    __threadfence_block();
    float* ent = park + ((size_t)sl * 16 + dh * 8 + h) * kPartStride;
    if ((lane & 3) == 0) {
      const int d0 = lane >> 2;
#pragma unroll
      for (int mt = 0; mt < kHMt; ++mt) {
        ent[mt * 16 + d0] = oacc[mt][0]; ent[64 + mt * 16 + d0] = oacc[mt][1];
        ent[mt * 16 + d0 + 8] = oacc[mt][2]; ent[64 + mt * 16 + d0 + 8] = oacc[mt][3];
      }
    }
    if (lane == 0) { ent[128] = mr0; ent[129] = mr1; ent[130] = l0; ent[131] = l1; }
    __syncwarp();
    int last = 0;
    if (lane == 0) { __threadfence_block(); last = (atomicAdd(arrp, 1) + 1 == cnt); __threadfence_block(); }
    last = __shfl_sync(0xffffffffu, last, 0);
    if (last) {
      const int hq = lane >> 4, d4 = (lane & 15) * 4;
      float M = neg_inf(), L = 0.f, o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
      for (int i = 0; i < cnt; ++i) {
        const float* e = park + ((size_t)sl * 16 + dh * 8 + ((gsk + i) & 7)) * kPartStride;
        const float me = e[128 + hq], le = e[130 + hq];
        const float4 oe = *reinterpret_cast<const float4*>(e + hq * 64 + d4);
        const float Mn = fmaxf(M, me), a = ex2(M - Mn), b = ex2(me - Mn);
        L = fmaf(L, a, le * b);
        o0 = fmaf(o0, a, oe.x * b); o1 = fmaf(o1, a, oe.y * b); o2 = fmaf(o2, a, oe.z * b); o3 = fmaf(o3, a, oe.w * b);
        M = Mn;
      }
      if (rb.y == 1) {
        bf16* ao0 = p.ao + (size_t)rb.z * (kQHeads * kDim) + (2 * kvh + hq) * kDim + dh * 64 + d4;
        unsigned* dst = reinterpret_cast<unsigned*>(ao0);
        dst[0] = pack2(o0 / L, o1 / L); dst[1] = pack2(o2 / L, o3 / L);
      } else {
        float* dst = p.part + ((size_t)rb.x * 2 + hq) * kPartStride;
        __stcg(reinterpret_cast<float4*>(dst + dh * 64 + d4), make_float4(o0, o1, o2, o3));
        if (dh == 0 && (lane & 15) == 0) { __stcg(dst + 128, M); __stcg(dst + 129, L); }
      }
      __syncwarp();
      __threadfence_block();
      if (lane == 0) { *(volatile int*)arrp = 0; __threadfence_block(); *(volatile int*)genp = tk + 1; }
    }
    qa = nqa; qb = nqb; if (kTail) tra = trb;
    kc = kn; c = cn;
#if SKV_POOL == 2
    pre = has_next;
#endif
  }
  __syncthreads();                                              // all groups done (a group without stages arrives here directly); attend ends with this barrier
#if SKV_POOL == 2
  const int T = *tkp;                                           // final: done was set after the last append, every group left the loop only after reading done
#endif
  unsigned gout = 0;                                            // every thread stores the same packed counters: group q's counter + its stage total of this call
#pragma unroll
  for (int q = 0; q < 8; ++q) gout |= (((g >> (3 * q)) + (unsigned)((T - q + 7) >> 3)) & 7u) << (3 * q);
  g = gout;
}
#endif

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
#if SKV_PAGED
  int pcb = 0;                                                 // block coordinate of position pp
#endif
  int4 pnxt = zero4;
  auto produce = [&](int slot) {
    if (pp >= pend) {
      if (pk + 1 >= ni) return;
      ++pk;
      const int4 it = pnxt;
      pnxt = (pk + 1 < ni) ? rec[2 * (pk + 1)] : zero4;
      pp = it.z; pend = it.w; pkvh = it.y & 15; prow = layer_row + (size_t)it.x;
#if SKV_PAGED
      pcb = page_cb(p, layer, it.x, pp);
#endif
    }
    const int n = min(kStageRows, pend - pp);
    const unsigned dst = sbase + kRingOff + slot * kStageBytes, bar = bar0 + 8 * slot;
    mb_expect(bar, (unsigned)n * 512u);
#if SKV_PAGED
    int off = 0, rc = pp & (kPageRows - 1);
    const int cb = pcb;
#else
    int off = 0, rc = (int)(prow + pp);
#endif
#pragma unroll
    for (int rs = 0; rs < kNumMaps; ++rs) {
      const int rows = kStageRows >> rs;
      if (n & rows) {
        const char* mk = (const char*)p.tmaps + (0 * kNumMaps + rs) * 128;
        const char* mv = (const char*)p.tmaps + (1 * kNumMaps + rs) * 128;
        SKV_BOX(dst + off * 128, mk, 0);
        SKV_BOX(dst + kHalfBytes + off * 128, mk, 64);
        SKV_BOX(dst + 2 * kHalfBytes + off * 128, mv, 0);
        SKV_BOX(dst + 3 * kHalfBytes + off * 128, mv, 64);
        off += rows; rc += rows;
      }
    }
    pp += n;
#if SKV_PAGED
    if (pp < pend && !(pp & (kPageRows - 1))) pcb = page_cb(p, layer, (int)prow, pp);   // next stage starts a new block
#endif
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

#if SKV_TC
// ------------------------------------------------------------------------------------------------ SKV_TC: tcgen05 consumer (q_len 1, TMA engine)
// SwapAB: per 128-position tile  S^T[pos][n] = K[pos][:] . Q[n][:]   (A = K tile, K-major SW128, B = Q [16 rows][128 dims] K-major SW128, D = TMEM 128 lanes x 16 columns fp32; rows n >= 2 of Q are zero)
//         O_t^T[dim][n] = sum_pos V[pos][dim] P^T[n][pos]  (A = V tile, MN-major SW128 (idesc bit 15; LBO = 16 KB dim-half stride, SBO = 1 KB), B = P^T [16 rows][128 positions] K-major SW128)
// Roles (512 threads): warp 0 lane 0 = TMA producer, warp 1 lane 0 = MMA issuer, warps 4-7 = softmax / correction / epilogue (TMEM quadrant = warp % 4: lane = position resp. dim),
// the other warps idle until the final __syncthreads. Every role walks the same flat tile sequence (counter g, persisted in ring_phase bits 0..23; the item counter in bits 24..30).
// mbarriers (8 B each from bar0): kfull[KST] kempty[KST] vfull[VST] vempty[VST] qfull[2] sfull[2] sfree[2] pfull[2] ofull[2] ofree[2]. "empty" style waits use parity ((n / depth) & 1) ^ 1 (a fresh barrier passes).
// TMEM columns (from Params::tmem): S buffer b at 16 b, O_t buffer b at 32 + 16 b (b = tile counter & 1).
typedef unsigned long long ull;
constexpr int kTcBarK = 0, kTcBarKE = kTcK, kTcBarV = 2 * kTcK, kTcBarVE = 2 * kTcK + kTcV, kTcBarQ = 2 * (kTcK + kTcV), kTcBarS = kTcBarQ + 2, kTcBarSF = kTcBarS + 2,
              kTcBarP = kTcBarSF + 2, kTcBarO = kTcBarP + 2, kTcBarOF = kTcBarO + 2, kTcNBar = kTcBarOF + 2;
static_assert(kTcNBar * 8 <= kBarBytes, "SKV_TC mbarriers");
constexpr int kTcNamedBar = 8;                                   // named barrier of the softmax warpgroup (mk: ids 1..7 are the GU exchange pairs)
constexpr unsigned kTcTileMask = 0xffffffu;                      // ring_phase bits of the tile counter

// Q (B operand of S): row n (0..15) = one q head, chunk c = 16 B = dims 8 c .. 8 c + 7. Half c >> 3 at +2048, row n at 128 n, 16 B chunk (c & 7) ^ (n & 7).
__host__ __device__ constexpr unsigned tc_q_off(unsigned n, unsigned c) { return (c >> 3) * 2048u + n * 128u + (((c & 7u) ^ (n & 7u)) << 4); }
// P^T (B operand of PV): row n, K index = position pos (0..127): half pos >> 6, 16 B chunk ((pos & 63) >> 3) ^ (n & 7), 2 B element pos & 7.
__host__ __device__ constexpr unsigned tc_p_off(unsigned n, unsigned pos) { return (pos >> 6) * 2048u + n * 128u + (((((pos & 63u) >> 3)) ^ (n & 7u)) << 4) + (pos & 7u) * 2u; }

// instruction descriptor, kind::f16: D f32, A / B bf16, M x N, bit 15 = A is MN-major (bit 16 = B MN-major: unused)
__host__ __device__ constexpr unsigned tc_idesc_(int M, int N, bool a_mn) { return (1u << 4) | (1u << 7) | (1u << 10) | (a_mn ? (1u << 15) : 0u) | ((unsigned)(N >> 3) << 17) | ((unsigned)(M >> 4) << 24); }
// smem descriptor, SWIZZLE_128B, sm100 version bit. K-major: LBO unused (1), SBO = 1024 B (8 rows x 128 B). MN-major (V as A): LBO = byte stride of the 64-element MN atoms (dim halves, 16 KB), SBO = stride of the 8-row K groups (1 KB)
__device__ __forceinline__ ull tc_desc_(unsigned addr, unsigned lbo, unsigned sbo) { return (ull)((addr >> 4) & 0x3fff) | ((ull)((lbo >> 4) & 0x3fff) << 16) | ((ull)((sbo >> 4) & 0x3fff) << 32) | (1ull << 46) | (2ull << 61); }
__device__ __forceinline__ ull tc_kdesc(unsigned addr) { return tc_desc_(addr, 16u, 1024u); }
__device__ __forceinline__ ull tc_vdesc(unsigned addr) { return tc_desc_(addr, 16384u, 1024u); }
__device__ __forceinline__ void tc_mma_(unsigned d, ull a, ull b, unsigned idesc, unsigned acc) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\ttcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}" ::"r"(d), "l"(a), "l"(b), "r"(idesc), "r"(acc) : "memory");
}
__device__ __forceinline__ void tc_commit_(unsigned mbar) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(mbar) : "memory"); }
__device__ __forceinline__ void tc_fence_before_() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after_() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_wait_ld_() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tc_ld2(unsigned& a, unsigned& b, unsigned ta) { asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0,%1}, [%2];" : "=r"(a), "=r"(b) : "r"(ta)); }
__device__ __forceinline__ void tc_ld16(unsigned (&v)[16], unsigned ta) {   // unit test: 16 columns
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15])
               : "r"(ta));
}
__device__ __forceinline__ void tc_alloc(unsigned smem_slot, unsigned ncols) {   // whole warp (unit test / standalone benches; mk allocates its own 256 columns)
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_slot), "r"(ncols) : "memory");
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tc_dealloc(unsigned taddr, unsigned ncols) { asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory"); }
__device__ __forceinline__ void mb_arrive1(unsigned b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory"); }
__device__ __forceinline__ void sts16(unsigned a, unsigned short v) { asm volatile("st.shared.b16 [%0], %1;" ::"r"(a), "h"(v) : "memory"); }
__device__ __forceinline__ void sts128(unsigned a, const uint4& v) { asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ void sts64f(unsigned a, float x, float y) { asm volatile("st.shared.v2.f32 [%0], {%1,%2};" ::"r"(a), "f"(x), "f"(y) : "memory"); }
__device__ __forceinline__ float2 lds64f(unsigned a) {
  float2 v;
  asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(a) : "memory");
  return v;
}

// number of 128-position tiles of item record a (positions p0 .. p1 - 1)
__device__ __forceinline__ int tc_tiles(const int4& a) { return (a.w - a.z + 127) >> 7; }

// ---- TMA producers: warp 0 lane 0 = K (kV false), warp 2 lane 0 = V. K(g) into K slot g % kTcK, V(g) into V slot g % kTcV; always full 128-row boxes (rows past the item / the map are other data /
// TMA zero fill: masked by the softmax). Separate threads so that K (slot freed right after the S MMAs) never waits for a V slot (freed after the PV MMAs).
template <bool kV>
__device__ __forceinline__ void tc_producer(const Params& p, int layer, unsigned sbase, unsigned bar0, unsigned g, const int4* rec, int ni) {
  const size_t layer_row = (size_t)layer * p.cache_rows;
  const char* mp = (const char*)p.tmaps + (kV ? kNumMaps * 128 : 0);   // map [0] = K box (64 dims, 1, 128 rows), [kNumMaps] = V
  constexpr unsigned D = kV ? kTcV : kTcK;
  constexpr unsigned bf = kV ? kTcBarV : kTcBarK, be = kV ? kTcBarVE : kTcBarKE, off = kV ? kTcVOff : kTcKOff;
  for (int k = 0; k < ni; ++k) {
    const int4 a = rec[2 * k];
    const int nt = tc_tiles(a), kvh = a.y & 15;
    int rc = (int)(layer_row + (size_t)a.x) + a.z;
    for (int j = 0; j < nt; ++j, ++g, rc += 128) {
      const unsigned s = g % D, bar = bar0 + 8 * (bf + s);
      mb_wait(bar0 + 8 * (be + s), ((g / D) & 1) ^ 1);
      mb_expect(bar, 32768u);
      const unsigned dst = sbase + off + s * 32768;
      tma3d(dst, mp, 0, kvh, rc, bar);
      tma3d(dst + 16384, mp, 64, kvh, rc, bar);
    }
  }
}

__device__ __forceinline__ bool tc_try(unsigned b, unsigned parity) {
  unsigned ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(b), "r"(parity) : "memory");
  return ok != 0;
}

// ---- warp 1 lane 0: MMA issuer. Whichever of "S(gs): K(gs), Q, S buffer ready" and "PV(gp), gp < gs: P(gp), V(gp), O buffer ready" is ready first is issued (PV first when both are).
__device__ __forceinline__ void tc_mma_issuer(const Params& p, unsigned sbase, unsigned bar0, unsigned g, unsigned ic, const int4* rec, int ni) {
  constexpr unsigned idS = tc_idesc_(128, 16, false), idV = tc_idesc_(128, 16, true);
  unsigned tot = 0;
  for (int k = 0; k < ni; ++k) tot += (unsigned)tc_tiles(rec[2 * k]);
  const unsigned end = g + tot;
  unsigned gs = g, gp = g;
  int k = 0, j = 0, nt = tc_tiles(rec[0]);
  while (gp < end) {
    if (gp < gs) {
      const unsigned b = gp & 1, vs = gp % kTcV;
      if (tc_try(bar0 + 8 * (kTcBarP + b), (gp >> 1) & 1) && tc_try(bar0 + 8 * (kTcBarV + vs), (gp / kTcV) & 1) && tc_try(bar0 + 8 * (kTcBarOF + b), ((gp >> 1) & 1) ^ 1)) {
        tc_fence_after_();
        const unsigned va = sbase + kTcVOff + vs * 32768, pa = sbase + kTcPOff + b * 4096, d = p.tmem + 32 + 16 * b;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) tc_mma_(d, tc_vdesc(va + kk * 2048), tc_kdesc(pa + (kk >> 2) * 2048 + (kk & 3) * 32), idV, kk > 0);
        tc_commit_(bar0 + 8 * (kTcBarO + b));
        tc_commit_(bar0 + 8 * (kTcBarVE + vs));
        ++gp;
        continue;
      }
    }
    if (gs < end) {
      const unsigned b = gs & 1, ks = gs % kTcK;
      const unsigned iq = ic + (unsigned)k;
      if ((j > 0 || tc_try(bar0 + 8 * (kTcBarQ + (iq & 1)), (iq >> 1) & 1)) && tc_try(bar0 + 8 * (kTcBarK + ks), (gs / kTcK) & 1) && tc_try(bar0 + 8 * (kTcBarSF + b), ((gs >> 1) & 1) ^ 1)) {
        tc_fence_after_();
        const unsigned qb = sbase + kTcQOff + (iq & 1) * 4096, ka = sbase + kTcKOff + ks * 32768, d = p.tmem + 16 * b;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) tc_mma_(d, tc_kdesc(ka + (kk >> 2) * 16384 + (kk & 3) * 32), tc_kdesc(qb + (kk >> 2) * 2048 + (kk & 3) * 32), idS, kk > 0);
        tc_commit_(bar0 + 8 * (kTcBarS + b));
        tc_commit_(bar0 + 8 * (kTcBarKE + ks));
        ++gs;
        if (++j == nt) { j = 0; ++k; if (k < ni) nt = tc_tiles(rec[2 * k]); }
      }
    }
  }
}

// ---- warps 4-7: softmax, O correction, epilogue. Thread (wq, lane): S^T position / O^T dim = 32 wq + lane (the TMEM lane); two q heads in two columns.
__device__ __forceinline__ void tc_softmax(const Params& p, int layer, unsigned sbase, unsigned bar0, unsigned g, unsigned ic, const int4* rec, int ni) {
  (void)layer;
  const int wt = threadIdx.x - 128, lane = wt & 31, wq = wt >> 5;
  const unsigned tq = (unsigned)(wq * 32) << 16;
  const unsigned xm = sbase + kTcXOff, xl = xm + 64;            // max exchange [2 buffers][4 warps][2 heads], l exchange [4 warps][2]
  {                                                               // rows 2..15 of the Q / P^T buffers (8 half blocks of 2 KB at kTcQOff, kTcPOff = kTcQOff + 8 KB): zero once per call
    const uint4 z = make_uint4(0, 0, 0, 0);
    for (int i = wt; i < 8 * 112; i += 128) sts128(sbase + kTcQOff + (i / 112) * 2048 + 256 + (i % 112) * 16, z);
  }
  auto ldq = [&](int k) {                                         // warp 4 only: lane = (head hq = lane >> 4, 16 B chunk lane & 15) of item k's q
    const int4 a = rec[2 * k], b = rec[2 * k + 1];
    return ldg_cg4(p.q + (size_t)(b.z * kQHeads + 2 * (a.y & 15) + (lane >> 4)) * kDim + (lane & 15) * 8);
  };
  auto stq = [&](unsigned ib, const uint4& x) { sts128(sbase + kTcQOff + ib * 4096 + tc_q_off(lane >> 4, lane & 15), x); };
  uint4 qn = make_uint4(0, 0, 0, 0);
  if (wq == 0) { qn = ldq(0); stq(ic & 1, qn); }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("bar.sync %0, %1;" ::"r"(kTcNamedBar), "r"(128) : "memory");
  if (wt == 0) mb_arrive1(bar0 + 8 * (kTcBarQ + (ic & 1)));

#ifdef SKV_TC_PROF
  long long pw_s = 0, pw_o = 0, pt0 = clock64(), pfirst = 0, plast = 0;
  int ptiles = 0;
#define PROF_BEGIN long long pq_ = clock64();
#define PROF_END(v) v += clock64() - pq_;
#else
#define PROF_BEGIN
#define PROF_END(v)
#endif
  for (int k = 0; k < ni; ++k, ++ic) {
    const int4 ra = rec[2 * k], rb = rec[2 * k + 1];
    const int len = ra.w - ra.z, nt = tc_tiles(ra), kvh = ra.y & 15;
    if (wq == 0 && k + 1 < ni) qn = ldq(k + 1);
    float m0 = neg_inf(), m1 = neg_inf(), l0 = 0.f, l1 = 0.f, o0 = 0.f, o1 = 0.f, ap0 = 0.f, ap1 = 0.f;
    for (int j = 0; j < nt; ++j, ++g) {
      const unsigned b = g & 1;
      { PROF_BEGIN mb_wait(bar0 + 8 * (kTcBarS + b), (g >> 1) & 1); PROF_END(pw_s) }
#ifdef SKV_TC_PROF
      if (ptiles == 0) pfirst = clock64() - pt0;
      ++ptiles; plast = clock64();
#endif
      tc_fence_after_();
      unsigned u0, u1;
      tc_ld2(u0, u1, p.tmem + tq + 16 * b);
      if (j == 0 && wq == 0 && k + 1 < ni) {                       // S MMAs of this item's first tile are done => the other Q buffer (item k - 1) is free: stage item k + 1's q
        stq((ic + 1) & 1, qn);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
        if (lane == 0) mb_arrive1(bar0 + 8 * (kTcBarQ + ((ic + 1) & 1)));
      }
      tc_wait_ld_();
      tc_fence_before_();
      __syncwarp();
      if (lane == 0) mb_arrive1(bar0 + 8 * (kTcBarSF + b));
      float s0 = __uint_as_float(u0), s1 = __uint_as_float(u1);
      if (j == nt - 1 && wq * 32 + lane >= len - j * 128) { s0 = neg_inf(); s1 = neg_inf(); }   // positions >= n (only the last tile can be partial)
      float c0, c1;
      asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(c0) : "f"(s0));
      asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(c1) : "f"(s1));
      const unsigned xa = xm + (b * 8 + wq * 2) * 4;
      if (lane == 0) sts64f(xa, c0, c1);
      asm volatile("bar.sync %0, %1;" ::"r"(kTcNamedBar), "r"(128) : "memory");
      {
        const unsigned xb = xm + b * 32;
        const float2 e0 = lds64f(xb), e1 = lds64f(xb + 8), e2 = lds64f(xb + 16), e3 = lds64f(xb + 24);
        c0 = fmaxf(fmaxf(e0.x, e1.x), fmaxf(e2.x, e3.x)); c1 = fmaxf(fmaxf(e0.y, e1.y), fmaxf(e2.y, e3.y));
      }
      const float n0 = fmaxf(m0, c0 * kQScale), n1 = fmaxf(m1, c1 * kQScale);      // finite: every tile has a valid position
      const float a0 = ex2(m0 - n0), a1 = ex2(m1 - n1);
      m0 = n0; m1 = n1;
      const float pp0 = ex2(fmaf(s0, kQScale, -n0)), pp1 = ex2(fmaf(s1, kQScale, -n1));
      l0 = fmaf(l0, a0, pp0); l1 = fmaf(l1, a1, pp1);
      {
        const unsigned pa = sbase + kTcPOff + b * 4096, pos = wq * 32 + lane;
        sts16(pa + tc_p_off(0, pos), f2bf(pp0));
        sts16(pa + tc_p_off(1, pos), f2bf(pp1));
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncwarp();
      if (lane == 0) mb_arrive1(bar0 + 8 * (kTcBarP + b));
      if (j > 0) {                                                  // O_t of the previous tile: O = O * alpha(g - 1) + O_t(g - 1)  (O is relative to m of tile g - 2 before)
        const unsigned gg = g - 1, bb = gg & 1;
        { PROF_BEGIN mb_wait(bar0 + 8 * (kTcBarO + bb), (gg >> 1) & 1); PROF_END(pw_o) }
        tc_fence_after_();
        unsigned v0, v1;
        tc_ld2(v0, v1, p.tmem + tq + 32 + 16 * bb);
        tc_wait_ld_();
        tc_fence_before_();
        __syncwarp();
        if (lane == 0) mb_arrive1(bar0 + 8 * (kTcBarOF + bb));
        o0 = fmaf(o0, ap0, __uint_as_float(v0)); o1 = fmaf(o1, ap1, __uint_as_float(v1));
      }
      ap0 = a0; ap1 = a1;
    }
    // l: per-position partial sums -> warp -> 4 warps
#pragma unroll
    for (int o = 16; o; o >>= 1) { l0 += __shfl_xor_sync(0xffffffffu, l0, o); l1 += __shfl_xor_sync(0xffffffffu, l1, o); }
    if (lane == 0) sts64f(xl + wq * 8, l0, l1);
    asm volatile("bar.sync %0, %1;" ::"r"(kTcNamedBar), "r"(128) : "memory");
    float L0, L1;
    {
      const float2 e0 = lds64f(xl), e1 = lds64f(xl + 8), e2 = lds64f(xl + 16), e3 = lds64f(xl + 24);
      L0 = (e0.x + e1.x) + (e2.x + e3.x); L1 = (e0.y + e1.y) + (e2.y + e3.y);
    }
    {                                                               // drain: the last tile's O_t
      const unsigned gg = g - 1, bb = gg & 1;
      mb_wait(bar0 + 8 * (kTcBarO + bb), (gg >> 1) & 1);
      tc_fence_after_();
      unsigned v0, v1;
      tc_ld2(v0, v1, p.tmem + tq + 32 + 16 * bb);
      tc_wait_ld_();
      tc_fence_before_();
      __syncwarp();
      if (lane == 0) mb_arrive1(bar0 + 8 * (kTcBarOF + bb));
      o0 = fmaf(o0, ap0, __uint_as_float(v0)); o1 = fmaf(o1, ap1, __uint_as_float(v1));
    }
    const int d = wt;
    if (rb.y == 1) {
      p.ao[(size_t)rb.z * (kQHeads * kDim) + (2 * kvh) * kDim + d] = f2bf(o0 / L0);
      p.ao[(size_t)rb.z * (kQHeads * kDim) + (2 * kvh + 1) * kDim + d] = f2bf(o1 / L1);
    } else {
      float* d0 = p.part + ((size_t)rb.x * 2) * kPartStride;
      float* d1 = d0 + kPartStride;
      __stcg(d0 + d, o0); __stcg(d1 + d, o1);
      if (d == 0) { __stcg(d0 + 128, m0); __stcg(d0 + 129, L0); __stcg(d1 + 128, m1); __stcg(d1 + 129, L1); }
    }
  }
#ifdef SKV_TC_PROF
  if (wt == 0 && layer == 5 && (blockIdx.x == 0 || blockIdx.x == 77)) printf("TCPROF first_s %lld since_last_s %lld cta %d tiles %d total %lld wait_s %lld wait_o %lld per-tile total %lld wait_s %lld wait_o %lld\n", pfirst, clock64() - plast, blockIdx.x, ptiles, clock64() - pt0, pw_s, pw_o, (clock64() - pt0) / ptiles, pw_s / ptiles, pw_o / ptiles);
#endif
}

// All roles of one attend call. ring_phase: bit 31 init, bits 24..30 item counter, bits 0..23 tile counter (identical in every thread at return).
__device__ __forceinline__ void run_items_tc(const Params& p, int layer, unsigned char* smem, unsigned& ring_phase, const int4* rec, int ni) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const unsigned sbase = smaddr(smem), bar0 = sbase + kBarOff;
  const unsigned g = ring_phase & kTcTileMask, ic = (ring_phase >> 24) & 0x7fu;
  if (warp == 0) {
    if (lane == 0) tc_producer<false>(p, layer, sbase, bar0, g, rec, ni);
    __syncwarp();
  } else if (warp == 2) {
    if (lane == 0) tc_producer<true>(p, layer, sbase, bar0, g, rec, ni);
    __syncwarp();
  } else if (warp == 1) {
#ifdef SKV_TC_NOMATH   // profiling only: the K / V rings are drained as fast as they fill (no MMA, no softmax, no output): the load pipeline of the swizzled 128 B-box access pattern
    if (lane == 0) {
      unsigned tot = 0;
      for (int k = 0; k < ni; ++k) tot += (unsigned)tc_tiles(rec[2 * k]);
      for (unsigned gg = g; gg < g + tot; ++gg) {
        mb_wait(bar0 + 8 * (kTcBarK + gg % kTcK), (gg / kTcK) & 1); mb_arrive1(bar0 + 8 * (kTcBarKE + gg % kTcK));
        mb_wait(bar0 + 8 * (kTcBarV + gg % kTcV), (gg / kTcV) & 1); mb_arrive1(bar0 + 8 * (kTcBarVE + gg % kTcV));
      }
    }
    __syncwarp();
#else
    if (lane == 0) tc_mma_issuer(p, sbase, bar0, g, ic, rec, ni);
    __syncwarp();
  } else if (warp >= 4 && warp < 8) {
    tc_softmax(p, layer, sbase, bar0, g, ic, rec, ni);
#endif
  }
  unsigned nt = 0;
  for (int k = 0; k < ni; ++k) nt += (unsigned)tc_tiles(rec[2 * k]);
  tc_fence_before_();
  __syncthreads();
  tc_fence_after_();
  ring_phase = (ring_phase & 0x80000000u) | ((((ring_phase >> 24) & 0x7fu) + (unsigned)ni) & 0x7fu) << 24 | ((g + nt) & kTcTileMask);
}
#endif  // SKV_TC

// All 512 threads. Runs the CTA's items one after another. ring_phase: bit 31 = mbarriers initialised, low bits = TMA stages consumed so far;
// start every CTA with ring_phase = 0 and pass the same variable to every layer's call.
// SKV_HALVES: stage counters of the groups (identical in every thread at return): G = 2: bits 0..14 / 16..30; G = 4: 7 bits each at bits 0, 8, 16, 24; G = 8: 3 bits each at bits 0, 3, .., 21.
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
#if SKV_TC
      for (int i = 0; i < kTcNBar; ++i) mb_init(bar0 + 8 * i, (i >= kTcBarSF && i < kTcBarSF + 2) || (i >= kTcBarP && i < kTcBarP + 2) || (i >= kTcBarOF && i < kTcBarOF + 2) ? 4 : 1);   // sfree / pfull / ofree: one arrive per softmax warp
#elif SKV_HALVES
      for (int s = 0; s < kStages; ++s) { mb_init(bar0 + 8 * s, 1); mb_init(bar0 + 8 * (kStages + s), kHW); }   // P10: one arrive per warp of the owning group (kHW = 16 / SKV_GROUPS)
#else
      for (int s = 0; s < kStages; ++s) { mb_init(bar0 + 8 * s, 1); mb_init(bar0 + 8 * (kStages + s), 16); }
#endif
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    ring_phase |= 0x80000000u;
  }
  if (tma) { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); __syncthreads(); }
#if !SKV_TC
  unsigned g = ring_phase & 0x7fffffffu;
#endif
  const int4* rec = p.items + (size_t)blockIdx.x * p.items_stride * 2;
#if SKV_IFACE == 2
  const int ni = rec[1].y >> 8;                                 // iface 2: n_items = S word >> 8 (word 7 is vis)
#else
  const int ni = rec[1].w;                                      // n_items is stored in every record of the CTA
#endif
#if SKV_POOL
  unsigned tk = 0;
  if (tid == 0) tk = atomicAdd(&p.pool_ctr[layer], 1u);        // first claim, before the static run (its latency hides behind it)
  if (SKV_POOL == 2 || ni > 0) {                               // a CTA without static items still runs the pool loop (POOL 1) / claims at once inside run_items_si (POOL 2)
#else
  if (ni == 0) return;
#endif
#if SKV_TC
  run_items_tc(p, layer, smem, ring_phase, rec, ni);
  return;
#elif SKV_SI
#if SKV_POOL == 2
  run_items_si(p, layer, smem, g, rec, ni, tk);                // static list + pool in one stage stream
#else
  run_items_si(p, layer, smem, g, rec, ni);
#endif
#elif SKV_HALVES && SKV_GROUPS == 8
  run_items_g8(p, layer, smem, g, rec, ni);
#elif SKV_HALVES
  run_items_halves(p, layer, smem, g, rec, ni);
#elif SKV_QMAX > 1
  run_items_r6(p, layer, smem, g, rec, ni);
#else
  if (tma) run_items<true>(p, layer, smem, g, rec, ni);
  else run_items<false>(p, layer, smem, g, rec, ni);
#endif
#if SKV_POOL
  }
#if SKV_POOL == 1
  {
    const int4* pool = p.items + (size_t)gridDim.x * p.items_stride * 2;
#if SKV_IFACE == 2
    const int npool = pool[1].y >> 8;
#else
    const int npool = pool[1].w;
#endif
    volatile unsigned* slot = reinterpret_cast<volatile unsigned*>(smem + kBarOff + kPoolSlot);
    for (int it = 0;; ++it) {
      __syncthreads();                                         // the previous call's groups are done with the smem its setup rewrites (SI table / counters)
      if (tid == 0) slot[it & 1] = tk;
      __syncthreads();
      const unsigned t = slot[it & 1];                         // slot parity: the next iteration's write cannot race this read (a sync sits between)
      if (t >= (unsigned)npool) {
        if (tid == 0 && t == (unsigned)npool + gridDim.x - 1) p.pool_ctr[layer] = 0;   // the last failing claim of the layer: every other claim is done
        break;
      }
      if (tid == 0) tk = atomicAdd(&p.pool_ctr[layer], 1u);    // next claim, one item ahead
      const int4* pr = pool + 2 * (size_t)t;
#if SKV_SI
      run_items_si(p, layer, smem, g, pr, 1);
#elif SKV_HALVES && SKV_GROUPS == 8
      run_items_g8(p, layer, smem, g, pr, 1);
#elif SKV_HALVES
      run_items_halves(p, layer, smem, g, pr, 1);
#elif SKV_QMAX > 1
      run_items_r6(p, layer, smem, g, pr, 1);
#else
      if (tma) run_items<true>(p, layer, smem, g, pr, 1);
      else run_items<false>(p, layer, smem, g, pr, 1);
#endif
    }
  }
#endif
#endif
#if SKV_TICKET
  ticket_merge(p, smem, rec, ni);
#endif
#if !SKV_TC
  ring_phase = (ring_phase & 0x80000000u) | (g & 0x7fffffffu);
#endif
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
