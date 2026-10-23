"""Qwen3-0.6B decode step (b1, context 128) as one persistent cooperative CUDA kernel.

r10 = r9a + compile-time switch R10_NF (see README.md): the gate/up rows on the CTAs that do no fused work.
R10_NF 0 compiles to the same SASS as r9a (R9_REGACT 1, R9_DIRECT 0, R9_LMH 0; R8_* as in dense_r8).

The kernel is compiled with NVRTC at build() time and launched once per run()
with cuLaunchCooperativeKernel. Activations stay fp32; dependent phases are
separated by a grid barrier on a monotonic counter reset at kernel end.
"""

from __future__ import annotations

import ctypes

import torch

KERNEL_SRC = r"""
#ifndef R8_ARGMAX
#define R8_ARGMAX 1
#endif
#ifndef R8_LF
#define R8_LF 1
#endif
#ifndef R8_SPIN_CYC
#define R8_SPIN_CYC 490
#endif
#ifndef R8_SPIN_PRO
#define R8_SPIN_PRO R8_SPIN_CYC
#endif
#ifndef R8_SPIN_QKV
#define R8_SPIN_QKV R8_SPIN_CYC
#endif
#ifndef R8_SPIN_FUSED
#define R8_SPIN_FUSED 0  // J3 sweep (jobs 5430/5431): best on both GPUs
#endif
#ifndef R8_SPIN_GU
#define R8_SPIN_GU 735  // J3 sweep (jobs 5430/5431): best on both GPUs
#endif
#ifndef R8_SPIN_DOWN
#define R8_SPIN_DOWN 245  // J3 sweep (jobs 5430/5431): best on both GPUs
#endif
#ifndef R9_REGACT
#define R9_REGACT 1  // gemv activations in registers (gemv_dot_reg) instead of smem reads per weight unit
#endif
#ifndef R9_DIRECT
#define R9_DIRECT 0  // no smem staging of activations; RMSNorm applied after the dot (needs R9_REGACT)
#endif
#ifndef R9_LMH
#define R9_LMH 0  // LM head: norm + activations per warp in registers, no block sync
#endif
#ifndef R9_PROBE
#define R9_PROBE 0  // diagnostic: per-phase %clock deltas of warp 1 lane 0 into r9_probe
#endif
#ifndef R11_SKEW
#define R11_SKEW 0  // 1: thread 0 stores %globaltimer into a smem array at each bar_arrive (just before the red), one dump to r11_skew at the kernel end; no other global traffic
#endif
#ifndef R11_SKD
#define R11_SKD 0  // 1: split-K down projection. The CTA that computes hid[ri0 .. ri0 + ni) multiplies it by the matching 32 columns of W_d (TMA 3D tensor map, 4 copies of a 256 x 32 box) and red.adds 1024 partials
                   // into the other x buffer (x ping-pong); no hid in global memory, no GU -> down grid barrier (3 barriers per layer)
#endif
#ifndef R11_SKD_RI
#define R11_SKD_RI 0  // R11_SKD red issue: 0 = lanes that hold a quad (2 per warp) issue 16 red.v4 each; 1 = they stage the quad sums in smem (sdn), then threads 0 .. 255 issue ONE red.v4 each (after the residual add)
#endif
#ifndef R11_SKD_NX
#define R11_SKD_NX 1  // R11_SKD: x is the SUM of NX (1, 2 or 4) partial vectors per ping-pong side; O-proj reds of CTA b -> partial b & (NX - 1), down reds of CTA b -> partial b & (NX - 1) of the next x
#endif
#ifndef R11_SKD_AT
#define R11_SKD_AT 0  // R11_SKD copy issue point: 0 in the GU phase right after the norm __syncthreads; 1 right after the fused bar_arrive; 2 in the fused phase right after the GU copy issue (R10_GUE_AT 3)
#endif
#ifndef R11_FEXP
#define R11_FEXP 1  // 1: the fused-phase softmax expf (scores, new token) and the combine expf become __expf; 2: also the GU SiLU gt / (1 + expf(-gt)) becomes __fdividef(gt, 1 + __expf(-gt)) (all four SiLU sites)
#endif
#ifndef R11_CMB
#define R11_CMB 0  // 1: the fused-phase combine runs on all 512 threads (dim t & 127, group t >> 7 = 4 warps); the 4 partial o / L are summed by the O-proj readers (no extra __syncthreads)
#endif
#ifndef R11_GRPQ
#define R11_GRPQ 0  // 1: the QKV -> fused grid barrier becomes a per-kv-group wait: a CTA that computes qkv rows of kv-head group g (Q rows of heads 2g, 2g + 1, K row block g, V row block g) red.releases counter g
                    // (sync[16 + 8 g]); a fused CTA of group fg = (b % 16) / 2 waits for counter fg only; the O-proj red.adds go to xo (scratch S_XO) instead of x (see the comments at the sites)
#endif
#ifndef R11_LMS_KB
#define R11_LMS_KB 0  // > 0: per-CTA L2 prefetch size in KB (one cp.async.bulk.prefetch.L2 by thread 128 per staged layer) of the first rows of the LM-head matrix `embed`, in the last R11_LMS_L layers
#endif
#ifndef R11_LMS_L
#define R11_LMS_L 3  // R11_LMS: number of last layers that stage (staging step s = l - (NL - R11_LMS_L); CTA b prefetches embed bytes [(s * NB + b) * KB * 1024, + KB * 1024))
#endif
#ifndef R11_LMS_POL
#define R11_LMS_POL 0  // R11_LMS: 1 = L2 evict_last cache hint (createpolicy.fractional.L2::evict_last.b64, .L2::cache_hint form), 0 = no hint
#endif
#ifndef R11_LMS_NULL
#define R11_LMS_NULL 0  // R11_LMS: 1 = the same instruction with size 16 (code cost only)
#endif
#ifndef R10_NF
#define R10_NF 0  // 0 r9a; 1 no gate/up rows on CTAs >= 144; 2 their gate/up loads issue late; 3 after the barrier
#endif
#ifndef R10_NF_DELAY
#define R10_NF_DELAY 4000  // R10_NF 2: %clock cycles (about 2.0 us at 1.96 GHz) before the non-fused CTAs issue
#endif
#ifndef R10_LMH
#define R10_LMH 2  // LM head: 0 one row in flight per warp; 2 or 4 rows in flight per warp (activations from smem)
#endif
#ifndef R10_LMH_DYN
#define R10_LMH_DYN 0  // 1 (needs R10_LMH >= 2): static rows [0, VS), the rest in chunks from the counter sync[4]
#endif
#ifndef R10_LMH_STATIC_FRAC_PCT
#define R10_LMH_STATIC_FRAC_PCT 85
#endif
#ifndef R10_LMH_CHUNK
#define R10_LMH_CHUNK 32  // rows per dynamic chunk (multiple of R10_LMH)
#endif
#ifndef R10_LMH_CONSEC
#define R10_LMH_CONSEC 0  // 1 (R10_LMH >= 2, R10_LMH_DYN 0): the N rows of a group are consecutive, groups nwarps*N rows apart
#endif
#ifndef R10_LMH_LD
#define R10_LMH_LD 0  // LM-head row load: 0 __ldcs, 1 __ldg, 2 ld.global.nc.L1::no_allocate + L2 evict_first policy (inline asm)
#endif
#ifndef R10_PROBE2
#define R10_PROBE2 0  // diagnostic: per-CTA phase / barrier / activation-arrival %clock sums into r10_probe (slot map below)
#endif
#ifndef R10_PROBE2_W
#define R10_PROBE2_W 0  // with R10_PROBE2: thread 96 times the arrival of the last weight register of each gemv phase
#endif
#ifndef R10_STRICT
#define R10_STRICT 0  // diagnostic: every thread that issued window loads waits for all of them before the barrier wait
#endif
#ifndef R10_ORACLE
#define R10_ORACLE 0  // timing ceilings, wrong outputs: 1 regconst 2 L2 3 ring 4 no barrier wait 5 = 1+4 6 no LM head 7 no layers 8 constants; see README
#endif
#ifndef R10_PF
#define R10_PF 14  // bit mask, small per-layer constants loaded in the barrier windows: 1 norm weights in registers, 2 L2 prefetch of norm rows, 4 rope in smem + q/k norm weights in registers, 8 down residual in a register, 16 norm weights in registers with warp 0 load-free (warp 1 loads warp 0's pair, hand-off through smem), 32 fused prep on warps 1-3
#endif
#if R9_DIRECT && !R9_REGACT
#error "R9_DIRECT requires R9_REGACT"
#endif
#if R10_PF & 32
#define R10_PFS 1  // fused prep on warps 1-3
#else
#define R10_PFS 0
#endif
#ifndef R10_W0LF
#define R10_W0LF 0  // gemv_load_lf weight loads of every phase: 0 plain loads; 1 ld_v4_nc predicated on %tid.x >= 32 (warp 0 load-free); 2 the same inline-asm load, NOT predicated (warp 0 loads as in r9a)
#endif
#ifndef R10_PROBE3
#define R10_PROBE3 0  // needs R10_PROBE2: fine sub-marks in the QKV / GU / down / fused phases, slots 26..51, 54..56 (map below)
#endif
#if R10_W0LF < 0 || R10_W0LF > 2
#error "R10_W0LF is 0, 1 or 2"
#endif
#if R10_W0LF && (R9_DIRECT || !R9_REGACT || !R8_LF || R10_NF)
#error "R10_W0LF is written for R9_REGACT 1, R9_DIRECT 0, R8_LF 1, R10_NF 0"
#endif
#if R10_W0LF == 2 && (R10_PF & 16)
#error "R10_W0LF 2 excludes R10_PF bit 16 (bit 16 forces the predicated load in GU / down)"
#endif
#if R10_PROBE3 && !R10_PROBE2
#error "R10_PROBE3 needs R10_PROBE2"
#endif
// load mode of gemv_load_lf: 0 plain, 1 ld_v4_nc predicated (warp 0 load-free), 2 ld_v4_nc not predicated
#define R10_PFG ((R10_PF & 16) ? 1 : R10_W0LF)  // GU and down loads
#define R10_LFQ R10_W0LF                        // QKV loads
#ifndef R10_DEF
#define R10_DEF 2  // 1 (needs R9_DIRECT 1): deferred RMS without the load gate; the stage hand-off to warp 0 is a named barrier (barrier 1) inside the phase
                   // 2 (needs R9_DIRECT 0, the c0 read path): one-sync deferred RMS (rmsd_to_smem): x * g into smem, r applied in the QKV / GU stores
#endif
#ifndef R10_GUE
#define R10_GUE 1  // 1: the gate / up weights of layer l are 2 TMA bulk copies into dynamic smem (sGU, behind the stage), issued by thread 128 in the fused phase
#endif
#ifndef R10_GUE_AT
#define R10_GUE_AT 3  // R10_GUE issue point: 0 first statement of the fused phase (after the QKV -> fused bar_wait); 1 fused CTAs: after the __syncthreads that ends the attention
                      // 2 paced: 4 row chunks (8 bulk copies); fused CTAs issue chunk 0 at fused start, 1 after the prep sync, 2 after the attention sync, 3 after the combine sync; non-fused CTAs all 4 at fused start
                      // 3 fused CTAs: one issue right after the prep __syncthreads (the fused phase's own qkv read has arrived); non-fused CTAs at fused start (as 1)
#endif
#ifndef R10_DNE
#define R10_DNE 1  // 1 (needs R10_GUE 1): the down weights of layer l are one TMA bulk copy into dynamic smem (sD, behind sGU), issued by thread 128; no down load in the GU -> down window
#endif
#ifndef R10_DNE_AT
#define R10_DNE_AT 2  // R10_DNE issue point: 0 first statement of the GU phase (after the fused -> GU bar_wait); 1 with the GU copies (first statement of the fused phase)
                      // 2 (needs R10_DEF 2): in the GU phase right after rmsd_to_smem returns (its __syncthreads: x has arrived for every thread)
#endif
#ifndef R10_QKE
#define R10_QKE 0  // 1 (needs R10_GUE 1, R10_DNE 1, R10_DEF 2): the QKV weights of layer l are TMA bulk copies into sGU (shared with the gate / up rows), issued in the down phase of l - 1
#endif
#ifndef R10_QKE_AT
#define R10_QKE_AT 0  // R10_QKE issue point in the down phase: 0 first statement; 1 right after the __syncthreads that follows the hid smem store (hid has arrived)
#endif
#if R10_PF && ((R9_DIRECT && !R10_DEF) || !R9_REGACT || !R8_LF || R10_NF)
#error "R10_PF is written for R9_REGACT 1, R9_DIRECT 0 (or R10_DEF 1), R8_LF 1, R10_NF 0"
#endif
#if R10_PF < 0 || R10_PF > 63
#error "R10_PF is a mask of bits 1, 2, 4, 8, 16, 32"
#endif
#if (R10_PF & 16) && (R10_PF & 3)
#error "R10_PF bit 16 excludes bits 1 and 2 (no L2 prefetch of qn / kn with 16: not built)"
#endif
#if (R10_PF & 1) && (R10_PF & 2)
#error "R10_PF bits 1 and 2 are exclusive"
#endif
#ifndef R10_KVE
#define R10_KVE 0  // 1: K / V / Wo of layer l are cp.async-ed into smem (sK, sWo, sVc) in the window before the QKV phase of layer l (one group after the QKV stage group)
#endif
#ifndef R10_KVM
#define R10_KVM 0  // 1: K / V / Wo of layer l go to smem (sK, sWo, sVc, as R10_KVE) but are issued INSIDE the QKV phase of layer l (R10_KVM_AT), not in the window before it. Excludes R10_KVE
#endif
#ifndef R10_KVM_AT
#define R10_KVM_AT 0  // 0: right after rmsd_to_smem / rms_to_smem returns in the QKV phase (x has arrived); 1: right after the QKV dot's __syncthreads
#endif
#ifndef R10_NSM
#define R10_NSM 0  // 1 (needs R10_KVE 1): ln1, ln2, qn, kn rows of layer l in smem (dynamic smem + 4608 B) (cp.async in the GU -> down window of layer l - 1), rope in smem
#endif
#if (R10_KVE || R10_KVM) && (R9_DIRECT || !R9_REGACT || !R8_LF || R10_NF || R10_ORACLE || R10_STRICT || (R10_PF & 17))
#error "R10_KVE / R10_KVM are written for R9_REGACT 1, R9_DIRECT 0, R8_LF 1, R10_NF 0, no oracle / strict, R10_PF bits 1 and 16 off"
#endif
#if R10_KVM && (R10_KVE || R10_NSM)
#error "R10_KVM excludes R10_KVE and R10_NSM"
#endif
#if R10_KVM_AT < 0 || R10_KVM_AT > 1
#error "R10_KVM_AT is 0 or 1"
#endif
#if R10_KVM_AT && !R10_KVM
#error "R10_KVM_AT needs R10_KVM"
#endif
#if R10_NSM && (!R10_KVE || (R10_PF & 23))
#error "R10_NSM needs R10_KVE 1 and R10_PF bits 1, 2, 4 and 16 off"
#endif
#if R10_DEF < 0 || R10_DEF > 2
#error "R10_DEF is 0, 1 or 2"
#endif
#if R10_DEF == 2 && (R9_DIRECT || !R9_REGACT || !R8_LF || R10_NF || R10_KVE || R10_NSM || R10_ORACLE || R10_STRICT || (R10_PF & 49))
#error "R10_DEF 2 is written for R9_DIRECT 0, R9_REGACT 1, R8_LF 1, R10_NF 0, R10_KVE 0, R10_NSM 0, no oracle / strict, R10_PF bits 1, 16 and 32 off"
#endif
#if R10_DEF == 1 && (!R9_DIRECT || !R9_REGACT || !R8_LF || R10_NF || R10_KVE || R10_NSM || R10_ORACLE || R10_STRICT || (R10_PF & 49))
#error "R10_DEF is written for R9_DIRECT 1, R9_REGACT 1, R8_LF 1, R10_NF 0, R10_KVE 0, R10_NSM 0, no oracle / strict, R10_PF bits 1, 16 and 32 off"
#endif
#if R10_GUE && (!R9_REGACT || !R8_LF || R10_NF || R10_KVE || R10_NSM || R10_ORACLE || R10_STRICT || (R10_PF & 16))
#error "R10_GUE is written for R9_REGACT 1, R8_LF 1, R10_NF 0, R10_KVE 0, R10_NSM 0, no oracle / strict, R10_PF bit 16 off"
#endif
#if R10_GUE_AT < 0 || R10_GUE_AT > 3
#error "R10_GUE_AT is 0, 1, 2 or 3"
#endif
#if R10_DNE && (!R10_GUE || R9_DIRECT)
#error "R10_DNE needs R10_GUE 1 and R9_DIRECT 0"
#endif
#if R10_DNE_AT < 0 || R10_DNE_AT > 2
#error "R10_DNE_AT is 0, 1 or 2"
#endif
#if R10_DNE_AT == 2 && R10_DEF != 2
#error "R10_DNE_AT 2 needs R10_DEF 2"
#endif
#if R10_QKE && (!R10_DNE || R10_DEF != 2 || (R10_PF & 17))
#error "R10_QKE needs R10_GUE 1, R10_DNE 1, R10_DEF 2 and R10_PF bits 1 and 16 off"
#endif
#if R10_KVM && R10_DNE
#error "R10_KVM with R10_DNE does not fit: static 47568 + dynamic 5632 + 86016 + 43008 + 65536 B is over 232448 B"
#endif
#if R10_QKE_AT < 0 || R10_QKE_AT > 1 || (R10_QKE_AT && !R10_QKE)
#error "R10_QKE_AT is 0 or 1 and needs R10_QKE"
#endif
#if R11_SKD && (!R10_GUE || !R10_DNE || R10_DEF != 2 || R9_DIRECT || !R9_REGACT || R10_QKE || R10_NF || R10_KVE || R10_KVM || R10_ORACLE || R10_STRICT || R11_SKEW || R10_PROBE2 || R9_PROBE || R10_NSM)
#error "R11_SKD needs R10_GUE 1, R10_DNE 1, R10_DEF 2, R9_DIRECT 0, R9_REGACT 1 and R10_QKE, R10_NF, R10_KVE, R10_KVM, R10_ORACLE, R10_STRICT, R11_SKEW, R10_PROBE2, R9_PROBE, R10_NSM all 0"
#endif
#if R11_SKD_RI < 0 || R11_SKD_RI > 1
#error "R11_SKD_RI is 0 or 1"
#endif
#if R11_SKD_NX != 1 && R11_SKD_NX != 2 && R11_SKD_NX != 4
#error "R11_SKD_NX is 1, 2 or 4"
#endif
#if (R11_SKD_RI || R11_SKD_NX > 1) && !R11_SKD
#error "R11_SKD_RI and R11_SKD_NX > 1 need R11_SKD 1"
#endif
#if R11_SKD_AT < 0 || R11_SKD_AT > 2
#error "R11_SKD_AT is 0, 1 or 2"
#endif
#if R11_SKD && R11_SKD_AT == 2 && R10_GUE_AT != 3
#error "R11_SKD_AT 2 needs R10_GUE_AT 3"
#endif
#if (R11_GRPQ || R11_LMS_KB) && (R11_SKD || R11_SKEW || R10_PROBE2 || R9_PROBE || R10_ORACLE || R10_STRICT || R10_NF || R10_KVE || R10_KVM || R10_QKE || R10_NSM)
#error "R11_GRPQ / R11_LMS_KB need R11_SKD, R11_SKEW, R10_PROBE2, R9_PROBE, R10_ORACLE, R10_STRICT, R10_NF, R10_KVE, R10_KVM, R10_QKE, R10_NSM all 0"
#endif
#if (R11_GRPQ || R11_LMS_KB) && (!R10_GUE || R10_GUE_AT != 3 || !R10_DNE || R10_DEF != 2 || R9_DIRECT || !R9_REGACT || !R8_LF)
#error "R11_GRPQ / R11_LMS_KB are written for R10_GUE 1, R10_GUE_AT 3, R10_DNE 1, R10_DEF 2, R9_DIRECT 0, R9_REGACT 1, R8_LF 1 (the g3d2 macros)"
#endif
#if R11_FEXP < 0 || R11_FEXP > 2 || R11_CMB < 0 || R11_CMB > 1
#error "R11_FEXP is 0, 1 or 2; R11_CMB is 0 or 1"
#endif
#if R11_CMB && (R11_SKD || R11_GRPQ || R10_KVE || R10_KVM || R10_QKE || R10_PROBE2 || R9_PROBE || R10_ORACLE || R10_STRICT)
#error "R11_CMB is written for the g3d2 fused phase: R11_SKD, R11_GRPQ, R10_KVE, R10_KVM, R10_QKE, R10_PROBE2, R9_PROBE, R10_ORACLE, R10_STRICT all 0 (R11_FEXP has no such restriction)"
#endif
#if R11_FEXP >= 1
#define R11_EXPF(x) __expf(x)
#else
#define R11_EXPF(x) expf(x)
#endif
#if R11_FEXP >= 2
#define R11_SILU(g) __fdividef((g), 1.f + __expf(-(g)))
#else
#define R11_SILU(g) ((g) / (1.f + expf(-(g))))
#endif
#if R11_GRPQ < 0 || R11_GRPQ > 1
#error "R11_GRPQ is 0 or 1"
#endif
#if R11_GRPQ && (!(R10_PF & 8) || (R10_PF & 17) || R9_LMH)
#error "R11_GRPQ needs R10_PF bit 8 (the down residual register x + xo) and R10_PF bits 1 and 16 off (the other norm paths read x only), R9_LMH 0"
#endif
#if R11_LMS_KB < 0 || R11_LMS_L < 1 || R11_LMS_L > 28 || R11_LMS_POL < 0 || R11_LMS_POL > 1 || R11_LMS_NULL < 0 || R11_LMS_NULL > 1
#error "R11_LMS_KB >= 0, R11_LMS_L 1 .. 28, R11_LMS_POL 0 or 1, R11_LMS_NULL 0 or 1"
#endif
#if R11_LMS_NULL && !R11_LMS_KB
#error "R11_LMS_NULL needs R11_LMS_KB > 0"
#endif
#if R11_LMS_KB && (R10_LMH < 2 || (R11_LMS_KB * 1024) % 16)
#error "R11_LMS_KB needs R10_LMH >= 2 (the LM head rows ascend by iteration) and a size that is a multiple of 16 B"
#endif
#define R10_KVS (R10_KVE || R10_KVM)          // K / V / Wo of the fused phase are in smem (sK, sWo, sVc); KVE issues them in the window before the QKV phase, KVM inside it
#define R10_PF4 ((R10_PF & 4) && (!R10_KVE || R10_KVM))    // qn / kn of the layer in registers (nwq), loaded in the QKV -> fused window
#define R10_ROPES ((R10_PF & 4) || R10_NSM)   // rope in smem
#if R10_KVE
#define R10_KVQ_T , bool KVQ = false
#define R10_KVQ_A , true
#else
#define R10_KVQ_T
#define R10_KVQ_A
#endif
#if (R10_PROBE2 || R10_STRICT || R10_ORACLE) && ((R9_DIRECT && !R10_DEF) || !R9_REGACT || !R8_LF)
#error "R10 diagnostics are written for R9_REGACT 1, R9_DIRECT 0 (or R10_DEF 1), R8_LF 1"
#endif
#if R10_PROBE2_W && !R10_PROBE2
#error "R10_PROBE2_W needs R10_PROBE2"
#endif
#if R10_ORACLE == 8
#define R10_O_CONST 1  // norm weights (ln1, ln2, qn, kn) = 1, rope cs = 1 / sn = 0, no down residual read
#else
#define R10_O_CONST 0
#endif
#if R10_ORACLE == 1 || R10_ORACLE == 5
#define R10_O_REG 1  // weights are per-thread register values, no memory traffic
#else
#define R10_O_REG 0
#endif
#if R10_ORACLE == 3
#define R10_O_RING 1  // weights come from a smem ring
#else
#define R10_O_RING 0
#endif
#if R10_ORACLE == 4 || R10_ORACLE == 5
#define R10_O_NOBAR 1
#else
#define R10_O_NOBAR 0
#endif
#if R10_ORACLE == 2
#define R10_WL(l) ((l) & 1)  // weight layer index: layers 0-1 only (about 63 MB, L2 resident)
#define R10_LDW(p) __ldcg(p)
#else
#define R10_WL(l) (l)
#define R10_LDW(p) __ldg(p)
#endif
#if R10_ORACLE == 6
#define R10_LM_END 0  // no LM-head rows
#else
#define R10_LM_END VOCAB
#endif
#if R10_ORACLE == 7
#define R10_NL 0  // no layers
#else
#define R10_NL NL
#endif
#define R10_RING_BYTES (UMAX(INTER, 8) * NT * 16)
#if R10_NF && R9_DIRECT
#error "R10_NF is written for R9_DIRECT 0"
#endif
#define R8_STAGE_BYTES 5632  // gate/up: 11 units x 32 lanes x 16 B
#define H 1024
#define NL 28
#define QH 16
#define KVH 8
#define HD 128
#define INTER 3072
#define VOCAB 151936
#define CTX 128
#define NT 512
#define NW 16
#define EPS 1e-6f
#define NSL 9  // O-proj row slices: fused block b < QH * NSL owns q-head b % QH, slice b / QH
#define NQD (H / 4)  // Wo row quads: slices start and end on quads (16-byte aligned red.add.v4)
#define PPW (CTX / NW)  // cached positions per warp in the fused attention
#define WOU 4  // Wo uint4 per thread: slice rows (<= 116) x 16 uint4 <= WOU * NT
#define INFINITY __int_as_float(0x7f800000)

typedef unsigned short bf16_t;

// scratch layout (floats)
#define S_X 0
#define S_QKV (S_X + H)
#define S_HID (S_QKV + 4096)
#define S_END (S_HID + INTER)
#define S_XO S_END  // R11_GRPQ: 1024 floats, the O-proj red.add target (zero at the start of every fused phase); scratch is S_END + H floats
#if R11_SKD
// R11_SKD_NX x partials: partial 0 of side s is S_X (s = 0) / S_HID (s = 1), partial p >= 1 of side s is at S_END + ((p - 1) * 2 + s) * H (behind the qkv region; scratch is S_END + 2 * (NX - 1) * H floats)
#define S_SCR_END (S_END + 2 * (R11_SKD_NX - 1) * H)
// NX > 1: every x reader is converted in the QKV norm (rmsd_to_smem_p), the GU norm (same), the residual load (sres) and the final norm (rms_to_smem_p); these branches read x differently and are not converted
#if R11_SKD_NX > 1 && (R9_LMH || (R10_PF & 17) || R10_O_CONST || R10_NL != NL || R10_DEF != 2 || R10_PROBE3)
#error "R11_SKD_NX > 1 needs R9_LMH 0, R10_PF & 17 == 0, R10_O_CONST 0, R10_NL == NL, R10_DEF 2 (the readers of x that are converted)"
#endif
__device__ __forceinline__ float* r11_xp(float* scratch, int s, int p) {
    return p == 0 ? scratch + (s ? S_HID : S_X) : scratch + S_END + ((p - 1) * 2 + s) * H;
}
#endif

__device__ __forceinline__ float bf2f(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bflo(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bfhi(unsigned u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ bf16_t f2bf(float f) {
    unsigned u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16_t)(u >> 16);
}

__device__ __forceinline__ int ctaid_opaque() {  // defeats loop-invariant hoisting (register pressure)
    int v;
    asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(v));
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// GEMV activation vectors in smem use a permuted layout: within each 256-element
// segment, element lane*8+j lives at lane*4+j (j<4) or 128+lane*4+(j-4), so a
// lane's two float4 reads at lane*4 and 128+lane*4 are bank-conflict free.
__device__ __forceinline__ int sperm(int e) {
    const int o = e & 255;
    return (e & ~255) + ((o & 4) << 5) + (o >> 3) * 4 + (o & 3);
}
// float4 index variant: logical float4 f (elements 4f..4f+3) -> physical float4.
__device__ __forceinline__ int sperm4(int f) {
    const int o = f & 63;
    return (f & ~63) + (o & 1) * 32 + (o >> 1);
}

__device__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float t = (l < NW) ? red[l] : 0.f;
    return warp_sum(t);
}
__device__ float block_max(float v, float* red) {
    v = warp_max(v);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float t = (l < NW) ? red[l] : -INFINITY;
    return warp_max(t);
}

// Grid barrier on a monotonic arrival counter (requires co-residency:
// cooperative launch). Each block arrives once with a release reduction and
// thread 0 relaxed-polls (then one acquire fence) until all blocks of this epoch have arrived; the
// counter is zeroed at the end of the launch by the last argmax ticket holder.
#if R11_SKEW
#define R11_NBAR 128  // 1 prologue + 4 per layer x 28 = 113 arrivals; slots 113..127 stay 0
__device__ unsigned long long r11_skew[NB * R11_NBAR];  // [cta * R11_NBAR + k]: %globaltimer (ns) of CTA's k-th bar_arrive, written once at the kernel end
__shared__ unsigned long long r11_sk_s[R11_NBAR];
__shared__ unsigned r11_sk_n;
#endif
#if R11_SKD
// 128-byte tensor map of wd ({INTER, H, NL} bf16, box {32, 256, 1}); written by the host (cuMemcpyHtoDAsync on the launch stream) before the launch. CUtensorMap is not available under NVRTC.
struct __align__(64) r11_tmap_t { unsigned long long opaque[16]; };
__device__ r11_tmap_t r11_wd_map;
#endif
__device__ __forceinline__ void bar_arrive(unsigned* cnt) {
    __syncthreads();
#if R11_SKEW
    if (threadIdx.x == 0) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
        const unsigned n = r11_sk_n;  // zeroed by thread 0 before the prologue arrive
        r11_sk_n = n + 1;
        if (n < R11_NBAR) r11_sk_s[n] = t;
    }
#endif
    if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(cnt) : "memory");
}
template <unsigned SPIN>
__device__ __forceinline__ void bar_wait(const unsigned* cnt, unsigned& target) {
    target += gridDim.x;
#if !R10_O_NOBAR
    if (threadIdx.x == 0) {
        if (SPIN > 0) {
            unsigned t0, t;
            asm volatile("mov.u32 %0, %%clock;" : "=r"(t0)::"memory");
            do { asm volatile("mov.u32 %0, %%clock;" : "=r"(t)::"memory"); } while (t - t0 < SPIN);
        }
        unsigned v;
        do {
            asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(cnt) : "memory");
        } while (v < target);
        asm volatile("fence.acquire.gpu;" ::: "memory");  // one acquire after the relaxed poll succeeds
    }
    __syncthreads();
#endif
}
__device__ __forceinline__ void grid_barrier(unsigned* cnt, unsigned& target) {
    bar_arrive(cnt);
    bar_wait<0>(cnt, target);
}
#if R11_GRPQ
// R11_GRPQ: per-kv-group counters. Group g (0 .. KVH - 1) = kv head g; its qkv rows are Q rows [256 g, 256 g + 256) (heads 2g, 2g + 1), K rows [2048 + 128 g, + 128), V rows [3072 + 128 g, + 128).
// P(g) = the CTAs whose row range [rq0, rq0 + nq) = row_range(4096) meets one of the three ranges. Counter g is sync[R11_GC(g)] (8 words = 32 B apart; sync has 128 words),
// monotonic like bar_cnt: every member of P(g) adds 1 per layer, so layer l is complete when it reaches (l + 1) * |P(g)|. Zeroed at the kernel end next to bar_cnt.
#define R11_GC(g) (16 + 8 * (g))
__device__ __forceinline__ bool r11_meets(int a, int n, int lo, int len) { return a < lo + len && a + n > lo; }
__device__ __forceinline__ bool r11_member(int g, int a, int n) {
    return r11_meets(a, n, 256 * g, 256) || r11_meets(a, n, 2048 + 128 * g, 128) || r11_meets(a, n, 3072 + 128 * g, 128);
}
// CTA that owns qkv row r (the largest b with floor(b * 4096 / G) <= r, the inverse of row_range(4096))
__device__ __forceinline__ int r11_owner(int r, int G) { return (int)((((long long)(r + 1) * G + 4095) >> 12) - 1); }
// |P(g)|: the three row ranges of one group are >= 1792 rows apart and a CTA owns about 28 rows (G >= 64), so no CTA meets two of them: |P(g)| = the sum of the three owner spans
__device__ __forceinline__ unsigned r11_pcnt(int g) {
    const int G = gridDim.x;
    return (unsigned)(r11_owner(256 * g + 255, G) - r11_owner(256 * g, G) + r11_owner(2048 + 128 * g + 127, G) - r11_owner(2048 + 128 * g, G)
                      + r11_owner(3072 + 128 * g + 127, G) - r11_owner(3072 + 128 * g, G) + 3);
}
// producer side, end of the QKV phase (replaces bar_arrive of the QKV -> fused barrier): the __syncthreads orders every thread's qkv stores before thread 0's releases
__device__ __forceinline__ void r11_grp_arrive(unsigned* sync, int rq0, int nq) {
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int g = 0; g < KVH; ++g)
            if (r11_member(g, rq0, nq)) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(sync + R11_GC(g)) : "memory");
    }
}
// consumer side (thread 0 only, the caller follows with a __syncthreads): relaxed poll, then one acquire fence (as bar_wait)
template <unsigned SPIN>
__device__ __forceinline__ void r11_grp_wait(const unsigned* cnt, unsigned goal) {
    if (threadIdx.x == 0) {
        if (SPIN > 0) {
            unsigned t0, t;
            asm volatile("mov.u32 %0, %%clock;" : "=r"(t0)::"memory");
            do { asm volatile("mov.u32 %0, %%clock;" : "=r"(t)::"memory"); } while (t - t0 < SPIN);
        }
        unsigned v;
        do {
            asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(cnt) : "memory");
        } while (v < goal);
        asm volatile("fence.acquire.gpu;" ::: "memory");
    }
}
#endif

// Warp dot product of a bf16 row (length K, multiple of 256) with an fp32 smem vector.
template <int K>
__device__ __forceinline__ float row_dot(const bf16_t* __restrict__ w, const float* s) {
    const uint4* wr = (const uint4*)w;
    int lane = threadIdx.x & 31;
    float acc = 0.f;
#pragma unroll
    for (int it = 0; it < K / 256; ++it) {
        int idx = it * 32 + lane;
        uint4 u = __ldg(wr + idx);
        const float* xs = s + it * 256 + lane * 4;  // permuted layout (sperm)
        float4 a = *(const float4*)xs, b = *(const float4*)(xs + 128);
        acc += bflo(u.x) * a.x + bfhi(u.x) * a.y + bflo(u.y) * a.z + bfhi(u.y) * a.w
             + bflo(u.z) * b.x + bfhi(u.z) * b.y + bflo(u.w) * b.z + bfhi(u.w) * b.w;
    }
    return warp_sum(acc);
}

#if R10_LMH >= 2
#if R10_LMH_LD == 2
__device__ __forceinline__ uint4 lmh_ld(const uint4* p) {
    // .v4.u32 with .L2::evict_first directly is rejected by ptxas for this target; use a createpolicy evict_first hint instead
    uint4 v;
    unsigned long long pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(pol));
    return v;
}
#define LMH_LOAD(p) lmh_ld(p)
#elif R10_LMH_LD == 1
#define LMH_LOAD(p) __ldg(p)
#else
#define LMH_LOAD(p) __ldcs(p)
#endif
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
        for (int it = 0; it < H / 256; ++it) u[j][it] = LMH_LOAD(wr + it * 32 + lane);
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
#endif

#if R9_REGACT
// row_dot with the activations in registers: a[it][j] = logical element it*256 + lane*8 + j.
template <int K>
__device__ __forceinline__ float row_dot_reg(const bf16_t* __restrict__ w, const float (&a)[K / 256][8]) {
    const uint4* wr = (const uint4*)w;
    int lane = threadIdx.x & 31;
    float acc = 0.f;
#pragma unroll
    for (int it = 0; it < K / 256; ++it) {
        int idx = it * 32 + lane;
        uint4 u = __ldg(wr + idx);
        acc += bflo(u.x) * a[it][0] + bfhi(u.x) * a[it][1] + bflo(u.y) * a[it][2] + bfhi(u.y) * a[it][3]
             + bflo(u.z) * a[it][4] + bfhi(u.z) * a[it][5] + bflo(u.w) * a[it][6] + bfhi(u.w) * a[it][7];
    }
    return warp_sum(acc);
}

// a[j] = x[j] * g[j] for the lane's 8 logical elements (xp, gp point at element lane*8 of the segment);
// returns the lane's sum of x[j]^2. x is read with ld.cg (written by other CTAs).
__device__ __forceinline__ float norm_act(const float* xp, const bf16_t* __restrict__ gp, float (&a)[8]) {
    const float4 x0 = __ldcg((const float4*)xp), x1 = __ldcg((const float4*)(xp + 4));
    const uint4 g = __ldg((const uint4*)gp);
    const float ss = x0.x * x0.x + x0.y * x0.y + x0.z * x0.z + x0.w * x0.w
                   + x1.x * x1.x + x1.y * x1.y + x1.z * x1.z + x1.w * x1.w;
    a[0] = x0.x * bflo(g.x); a[1] = x0.y * bfhi(g.x); a[2] = x0.z * bflo(g.y); a[3] = x0.w * bfhi(g.y);
    a[4] = x1.x * bflo(g.z); a[5] = x1.y * bfhi(g.z); a[6] = x1.z * bflo(g.w); a[7] = x1.w * bfhi(g.w);
    return ss;
}
// a[j] = the lane's 8 logical elements of fp32 vector xp (no norm).
__device__ __forceinline__ void load_act(const float* xp, float (&a)[8]) {
    const float4 x0 = __ldcg((const float4*)xp), x1 = __ldcg((const float4*)(xp + 4));
    a[0] = x0.x; a[1] = x0.y; a[2] = x0.z; a[3] = x0.w; a[4] = x1.x; a[5] = x1.y; a[6] = x1.z; a[7] = x1.w;
}
// same from a smem vector in the sperm layout, segment seg
__device__ __forceinline__ void smem_act(const float* s, int seg, float (&a)[8]) {
    const float* xs = s + seg * 256 + (threadIdx.x & 31) * 4;
    const float4 x0 = *(const float4*)xs, x1 = *(const float4*)(xs + 128);
    a[0] = x0.x; a[1] = x0.y; a[2] = x0.z; a[3] = x0.w; a[4] = x1.x; a[5] = x1.y; a[6] = x1.z; a[7] = x1.w;
}
#endif

#if R9_PROBE
__device__ unsigned long long r9_probe[NB * 16];
__device__ __forceinline__ unsigned r9_clk() {
    unsigned v;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(v) :: "memory");
    return v;
}
#endif

#if R10_PROBE2
// r10_probe[blockIdx.x * 64 + slot], %clock cycles summed over layers (p2_s is flushed by thread 32 at kernel end):
//  [0..3]  phase work p = 0 QKV, 1 fused, 2 GU, 3 down: A_p - E_p       (thread 32; E_p = exit of the barrier before phase p,
//  [4..7]  barrier after phase p: E_(p+1) - A_p, the last layer's down barrier is skipped   A_p = just before its bar_arrive)
//  [8+2q], [9+2q]  activation arrival after the barrier exit, q = 0 QKV, 1 GU, 2 down: thread 0 (no loads of its own), thread 64
//  [14+q]  (R10_PROBE2_W) thread 96: arrival of the last weight register buf[U-1] after the barrier exit
//  [20..24] fused sub-marks, delay from E_1 (thread 32, CTAs < 144): after q/k/v prep sync, attention sync, combine sync,
//          O-proj dot sync, red.add block
//  (R10_PROBE3) fine sub-marks, delay from the exit of the barrier before the phase, cycles summed; thread 32 unless noted. A mark that
//          follows a load stores a value that depends on it into the volatile p2_dep[3] first, so the clock is read after the data arrived.
//  [26..33] QKV: 26 x value arrived (rms_to_smem*, after __ldcg), 27 after block_sum, 28 norm weight available, 29 after the final
//          __syncthreads of rms_to_smem* (incl. cp.async.wait_all), 30 after smem_act, 31 after gemv_dot_reg (plain: no exported
//          part value), 32 after the __syncthreads that follows the dot, 33 after the qkv store (thread 32 stores nothing: nq < 32)
//  [34..41] GU: the same 8 points (41 after the hid store; ni < 32, thread 32 stores nothing)
//  [42..47] down: 42 hid value arrived, 43 after cp.async.wait_all + __syncthreads, 44 after smem_act x3, 45 after gemv_dot_reg,
//          46 after the __syncthreads after the dot, 47 after the residual store (nh < 32: thread 32 stores nothing)
//  [48..51] fused (CTAs < 144): 48 qkv value of thread 32 (warp 1 lane 0: its prep `a`, both with and without R10_PF 32) arrived,
//          49 after the prep warp_sum, 50 after the sq / sk stores and the k_write stores, 51 after cp.async.wait_group 1. (52 = slot 20)
//  [54..56] thread 0 (warp 0), delay from its clock after bar_wait to the end of stage_to_buf: QKV, GU, down
//  (R10_DEF) the DEF paths write 26 + 8q (activation arrived: act[0][0] / act3[0][0] after the loads, before the hand-off), 31 / 39 / 45 (after
//          gemv_dot_reg), 32 / 40 / 46 (after the dot __syncthreads), 33 / 41 / 47 (after the store), 54..56 (thread 0: end of the hand-off + stage_to_buf)
//          and the new slots 57 / 59 / 60: thread 32 after the hand-off (bar.arrive 1) in QKV / GU / down. Slots 27..30, 34..38, 42..44 stay 0 (no block_sum,
//          no norm weight wait, no smem_act on these paths); the probe harness skips them.
//  (R10_GUE) slot 58: thread 32 after the mbarrier wait for the GU weights, delay from the exit of the barrier before the GU phase.
//  (R10_DNE) slot 61: thread 32 after the mbarrier wait for the down weights, delay from the exit of the barrier before the down phase.
//  (R10_QKE) slot 62: thread 32 after the mbarrier wait for the QKV weights, delay from the exit of the barrier before the QKV phase.
//  [63]    layer count
// GU slots 10, 11 stay 0 on CTAs without gate/up rows (R10_NF 1, CTAs >= 144).
__device__ unsigned long long r10_probe[NB * 64];
__shared__ unsigned long long p2_s[64];
__shared__ unsigned p2_t0[2], p2_dep[4];
#if R10_PROBE3
__shared__ unsigned p2_eb;  // thread 32's clock at the last P2_E (the helpers below cannot see the kernel local p2_e)
// thread 32: mark i, delay from the barrier exit, after the value v (an unsigned) is available
#define P3_M(i, v) do { if (threadIdx.x == 32) { ((volatile unsigned*)p2_dep)[3] = (unsigned)(v); p2_s[i] += p2_clk() - p2_eb; } } while (0)
#define P3_P(i) do { if (threadIdx.x == 32) p2_s[i] += p2_clk() - p2_eb; } while (0)
// thread 0: mark i after stage_to_buf of b (the volatile store of its last word depends on the stage read)
#define P3_W0(i, b) do { if (threadIdx.x == 0) { ((volatile unsigned*)p2_dep)[0] = (b)[sizeof(b) / sizeof(uint4) - 1].x; \
    p2_s[i] += p2_clk() - p2_t0[0]; } } while (0)
#else
#define P3_M(i, v)
#define P3_P(i)
#define P3_W0(i, b)
#endif
__device__ __forceinline__ unsigned p2_clk() {
    unsigned v;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(v) :: "memory");
    return v;
}
// thread 0 / 64: clock right after bar_wait; P2_ACT: after a smem store of a value that depends on the first activation load
#define P2_ACT0() do { if (threadIdx.x == 0) p2_t0[0] = p2_clk(); else if (threadIdx.x == 64) p2_t0[1] = p2_clk(); } while (0)
#define P2_ACT(q, dep) do { if (threadIdx.x == 0 || threadIdx.x == 64) { const int i_ = threadIdx.x >> 6; \
    ((volatile unsigned*)p2_dep)[i_] = __float_as_uint(dep); p2_s[8 + 2 * (q) + i_] += p2_clk() - p2_t0[i_]; } } while (0)
#define R10_PQ(q) , q
#if R10_PROBE2_W
// thread 96: the store cannot issue before the load of buf[U-1] has returned
#define P2_W(q, b) do { if (threadIdx.x == 96) { const unsigned t_ = p2_clk(); \
    ((volatile unsigned*)p2_dep)[2] = (b)[sizeof(b) / sizeof(uint4) - 1].x; p2_s[14 + (q)] += p2_clk() - t_; } } while (0)
#else
#define P2_W(q, b)
#endif
#else
#define P2_ACT0()
#define P2_ACT(q, dep)
#define R10_PQ(q)
#define P2_W(q, b)
#define P3_M(i, v)
#define P3_P(i)
#define P3_W0(i, b)
#endif

#if R8_LF
#if R11_SKD
extern __shared__ __align__(128) uint4 r8_stage[];  // R11_SKD: sD is a cp.async.bulk.tensor destination (128-byte aligned smem); offsets in it are multiples of 128
#else
extern __shared__ __align__(16) uint4 r8_stage[];  // dynamic smem: warp 0's window loads, issued by warps 1-15
#endif
__device__ __forceinline__ void cpa16(void* sdst, const void* g) {   // 16 B, L2 only
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((unsigned)__cvta_generic_to_shared(sdst)), "l"(g) : "memory"); }
__device__ __forceinline__ void cpa8(void* sdst, const void* g) {    // 8 B (as V1q's V-cache copy)
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"((unsigned)__cvta_generic_to_shared(sdst)), "l"(g) : "memory"); }
#endif
#if R10_O_REG
// oracle 1: a weight uint4 that costs no memory traffic and that the compiler cannot fold (each word is its own volatile asm;
// bf16 pairs of normal values in [0.5, 1))
__device__ __forceinline__ uint4 r10_oreg(unsigned k) {
    uint4 r;
    asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; xor.b32 t, t, %1; and.b32 t, t, 0x007f007f; or.b32 %0, t, 0x3f003f00; }" : "=r"(r.x) : "r"(4 * k + 0));
    asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; xor.b32 t, t, %1; and.b32 t, t, 0x007f007f; or.b32 %0, t, 0x3f003f00; }" : "=r"(r.y) : "r"(4 * k + 1));
    asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; xor.b32 t, t, %1; and.b32 t, t, 0x007f007f; or.b32 %0, t, 0x3f003f00; }" : "=r"(r.z) : "r"(4 * k + 2));
    asm volatile("{ .reg .u32 t; mov.u32 t, %%tid.x; xor.b32 t, t, %1; and.b32 t, t, 0x007f007f; or.b32 %0, t, 0x3f003f00; }" : "=r"(r.w) : "r"(4 * k + 3));
    return r;
}
#endif
#if R10_O_RING
// oracle 3: weights from a smem ring behind the stage (dynamic smem = R8_STAGE_BYTES + R10_RING_BYTES), element k * NT + tid
#define R10_RING_REF(k) (r8_stage + R8_STAGE_BYTES / 16)[(k) * NT + threadIdx.x]
#endif
#if R10_O_REG || R10_O_RING
#define R10_O_WO 1  // Wo window comes from the oracle (no cp.async stage entries, no global loads)
#else
#define R10_O_WO 0
#endif
#if R10_STRICT
// all words of a register buffer XORed into one unsigned (the STS of it cannot issue before every load has returned)
template <int U>
__device__ __forceinline__ unsigned r10_fold(const uint4 (&b)[U]) {
    unsigned f = 0;
#pragma unroll
    for (int k = 0; k < U; ++k) f ^= b[k].x ^ b[k].y ^ b[k].z ^ b[k].w;
    return f;
}
template <int U>
__device__ __forceinline__ unsigned r10_foldk(const uint2 (&b)[U]) {
    unsigned f = 0;
#pragma unroll
    for (int k = 0; k < U; ++k) f ^= b[k].x ^ b[k].y;
    return f;
}
__shared__ unsigned s_sink[NT];
// before a bar_wait that has window loads in front of it: helpers drain their cp.async, then store the fold; warp 0 does nothing
#define R10_SK(f) do { if (warp != 0) { asm volatile("cp.async.wait_all;" ::: "memory"); ((volatile unsigned*)s_sink)[tid] = (f); } } while (0)
#else
#define R10_SK(f)
#endif

// RMSNorm of the global fp32 residual x (H) with bf16 weight w into smem s.
// WAIT (LF build only): cp.async.wait_all before the final __syncthreads.
#if R10_PROBE2
template <bool WAIT = false, int PQ = -1 R10_KVQ_T>
#else
template <bool WAIT = false R10_KVQ_T>
#endif
__device__ void rms_to_smem(const float* x, const bf16_t* __restrict__ w, float* s, float* red) {
    int t = threadIdx.x;
    float a = __ldcg(x + t), b = __ldcg(x + t + NT);
#if R10_PROBE2
    if (PQ >= 0) { P2_ACT(PQ, a); P3_M(26 + 8 * PQ, __float_as_uint(a)); }
#endif
    float ss = block_sum(a * a + b * b, red);
#if R10_PROBE2
    if (PQ >= 0) P3_M(27 + 8 * PQ, __float_as_uint(ss));
#endif
    float r = rsqrtf(ss / (float)H + EPS);
#if R10_PROBE3
    const float w0 = bf2f(w[t]), w1 = bf2f(w[t + NT]);
    if (PQ >= 0) P3_M(28 + 8 * PQ, __float_as_uint(w0) ^ __float_as_uint(w1));
    s[sperm(t)] = a * r * w0;
    s[sperm(t + NT)] = b * r * w1;
#else
    s[sperm(t)] = a * r * bf2f(w[t]);
    s[sperm(t + NT)] = b * r * bf2f(w[t + NT]);
#endif
#if R10_KVE
    if (WAIT && KVQ) {  // QKV phase: the group of the K / V / Wo entries (fused CTAs, last committed) stays outstanding
        if (blockIdx.x < QH * NSL) asm volatile("cp.async.wait_group 1;" ::: "memory");
        else asm volatile("cp.async.wait_all;" ::: "memory");
    } else
#endif
    if (WAIT) asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
#if R10_PROBE2
    if (PQ >= 0) P3_P(29 + 8 * PQ);
#endif
}

#if R10_DEF == 2
// R10_DEF 2: deferred RMSNorm with ONE sync. s = x * g (no r), per-warp partial sums of x^2 in red[0 .. NW); the caller applies
// r = rsqrt(sum(red[0 .. NW)) / H + eps) to the row sums in its store (rmsd_r), after the dot's __syncthreads. The two norm-weight loads are issued
// before any use of a / b, so they overlap the x loads.
// WAR on red[0 .. NW): the writers here come after the barrier of the phase. The previous readers of red[0 .. 32) are the fused phase (attention
// reads, combine) and the previous QKV / GU store; each is followed by this CTA's bar_arrive __syncthreads (and more) before the next phase's code
// runs. The readers of this phase are the stores after the dot's __syncthreads, and the next writer of red (the fused phase, the next norm) comes after
// this CTA's next bar_arrive __syncthreads, so no thread still reads red when it is written. The sync at the end orders the writes before the reads.
#if R10_PROBE2
template <bool WAIT = false, int PQ = -1>
#else
template <bool WAIT = false>
#endif
__device__ void rmsd_to_smem(const float* x, const bf16_t* __restrict__ w, float* s, float* red) {
    int t = threadIdx.x;
    float a = __ldcg(x + t), b = __ldcg(x + t + NT);
    const float w0 = bf2f(w[t]), w1 = bf2f(w[t + NT]);  // issued before any use of a / b
#if R10_PROBE2
    if (PQ >= 0) { P2_ACT(PQ, a); P3_M(26 + 8 * PQ, __float_as_uint(a) ^ __float_as_uint(b)); }
#endif
    s[sperm(t)] = a * w0;
    s[sperm(t + NT)] = b * w1;
    float v = warp_sum(a * a + b * b);
    if ((t & 31) == 0) red[t >> 5] = v;
#if R10_PROBE2
    if (PQ >= 0) { P3_M(27 + 8 * PQ, __float_as_uint(v)); P3_M(28 + 8 * PQ, __float_as_uint(w0) ^ __float_as_uint(w1)); }
#endif
    if (WAIT) asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
#if R10_PROBE2
    if (PQ >= 0) P3_P(29 + 8 * PQ);
#endif
}
#if R11_GRPQ
// R11_GRPQ: rmsd_to_smem on x + xo (the O-proj sum is in xo); all 4 global loads and both weight loads are issued before the first use. Same WAR argument on red / s as rmsd_to_smem.
template <bool WAIT = false>
__device__ void rmsd2_to_smem(const float* x, const float* xo, const bf16_t* __restrict__ w, float* s, float* red) {
    int t = threadIdx.x;
    const float xa = __ldcg(x + t), xb = __ldcg(x + t + NT), oa = __ldcg(xo + t), ob = __ldcg(xo + t + NT);
    const float w0 = bf2f(w[t]), w1 = bf2f(w[t + NT]);
    const float a = xa + oa, b = xb + ob;
    s[sperm(t)] = a * w0;
    s[sperm(t + NT)] = b * w1;
    float v = warp_sum(a * a + b * b);
    if ((t & 31) == 0) red[t >> 5] = v;
    if (WAIT) asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
}
#endif
// r of the deferred norm from the NW per-warp partials in red (read after the dot's __syncthreads)
__device__ __forceinline__ float rmsd_r(const float* red) {
    float ss = red[0];
#pragma unroll
    for (int i = 1; i < NW; ++i) ss += red[i];
    return rsqrtf(ss / (float)H + EPS);
}
#endif

#if R11_SKD_NX > 1
// R11_SKD_NX > 1: x = sum over the NX partials of side `side` (r11_xp). All 2 * NX loads of a thread are issued before the first use (independent L2 round trips).
// rmsd_to_smem_p = rmsd_to_smem on that sum (same WAR argument on red / s); rms_to_smem_p = rms_to_smem (non-probe, non-WAIT) on it (the final norm).
__device__ __forceinline__ void r11_ldx(float* scratch, int side, int t, float& a, float& b) {
    float xa[R11_SKD_NX], xb[R11_SKD_NX];
#pragma unroll
    for (int p = 0; p < R11_SKD_NX; ++p) {
        const float* xp = r11_xp(scratch, side, p);
        xa[p] = __ldcg(xp + t);
        xb[p] = __ldcg(xp + t + NT);
    }
    a = xa[0];
    b = xb[0];
#pragma unroll
    for (int p = 1; p < R11_SKD_NX; ++p) { a += xa[p]; b += xb[p]; }
}
template <bool WAIT = false>
__device__ void rmsd_to_smem_p(float* scratch, int side, const bf16_t* __restrict__ w, float* s, float* red) {
    int t = threadIdx.x;
    float a, b;
    r11_ldx(scratch, side, t, a, b);
    const float w0 = bf2f(w[t]), w1 = bf2f(w[t + NT]);
    s[sperm(t)] = a * w0;
    s[sperm(t + NT)] = b * w1;
    float v = warp_sum(a * a + b * b);
    if ((t & 31) == 0) red[t >> 5] = v;
    if (WAIT) asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
}
__device__ void rms_to_smem_p(float* scratch, int side, const bf16_t* __restrict__ w, float* s, float* red) {
    int t = threadIdx.x;
    float a, b;
    r11_ldx(scratch, side, t, a, b);
    float ss = block_sum(a * a + b * b, red);
    float r = rsqrtf(ss / (float)H + EPS);
    s[sperm(t)] = a * r * bf2f(w[t]);
    s[sperm(t + NT)] = b * r * bf2f(w[t + NT]);
    __syncthreads();
}
#endif

#if (R10_PF & 17) || R10_O_CONST
// rms_to_smem with the two norm-weight bf16 w[t], w[t + NT] already in wp (low / high half): no load after block_sum.
#if R10_PROBE2
template <bool WAIT = false, int PQ = -1>
#else
template <bool WAIT = false>
#endif
__device__ void rms_to_smem_r(const float* x, unsigned wp, float* s, float* red, unsigned wp0 = 0) {
    int t = threadIdx.x;
    float a = __ldcg(x + t), b = __ldcg(x + t + NT);
#if R10_PROBE2
    if (PQ >= 0) { P2_ACT(PQ, a); P3_M(26 + 8 * PQ, __float_as_uint(a)); }
#endif
#if R10_PF & 16
    // R10_PF 16: warp 0 loads nothing. Warp 1 holds warp 0's pair in wp0 and writes it before block_sum; warp 0 reads it after
    // block_sum (its first __syncthreads orders the write). The next write is in the next norm, after many __syncthreads.
    __shared__ unsigned s_w0[32];
    if ((t >> 5) == 1) s_w0[t & 31] = wp0;
#endif
    float ss = block_sum(a * a + b * b, red);
#if R10_PROBE2
    if (PQ >= 0) P3_M(27 + 8 * PQ, __float_as_uint(ss));
#endif
#if R10_PF & 16
    if ((t >> 5) == 0) wp = s_w0[t & 31];
#endif
#if R10_PROBE2
    if (PQ >= 0) P3_M(28 + 8 * PQ, wp);  // thread 32 (warp 1): its own register pair, loaded in the window
#endif
    float r = rsqrtf(ss / (float)H + EPS);
    s[sperm(t)] = a * r * bf2f(wp & 0xffffu);
    s[sperm(t + NT)] = b * r * bf2f(wp >> 16);
    if (WAIT) asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
#if R10_PROBE2
    if (PQ >= 0) P3_P(29 + 8 * PQ);
#endif
}
// both halves of a thread's norm weights (ln row, 1024 bf16) as one register
__device__ __forceinline__ unsigned ld_norm2(const bf16_t* __restrict__ w) {
    const int t = threadIdx.x;
    return (unsigned)__ldg(w + t) | ((unsigned)__ldg(w + t + NT) << 16);
}
#if R10_PF & 16
// predicated in the PTX itself: with a plain `if` the compiler may drop the guard (own is overwritten in warp 0 later) and load in warp 0
__device__ __forceinline__ unsigned ld_u16_nc(const bf16_t* p, bool on) {
    unsigned short v;
    asm volatile("{ .reg .pred q; setp.ne.u32 q, %2, 0; mov.u16 %0, 0; @q ld.global.nc.u16 %0, [%1]; }"
                 : "=h"(v) : "l"(p), "r"((unsigned)on));
    return v;
}
// R10_PF 16: own = this thread's pair, loaded by warps 1.. only; w0 = warp 0 lane's pair, loaded by warp 1 only (0 elsewhere)
__device__ __forceinline__ void ld_norm16(const bf16_t* __restrict__ w, unsigned& own, unsigned& w0) {
    const int t = threadIdx.x, wp = t >> 5, ln = t & 31;
    own = ld_u16_nc(w + t, wp != 0) | (ld_u16_nc(w + t + NT, wp != 0) << 16);
    w0 = ld_u16_nc(w + ln, wp == 1) | (ld_u16_nc(w + ln + NT, wp == 1) << 16);
}
#endif
#endif

// Block-cooperative GEMV. A block owns rows [r0, r0+n); each row is split into
// 256-element segments ("units", one uint4 per lane). Warp w handles units
// w, w+NW, ...; all of a warp's weight loads are issued before any use.
#define CDIV(a, b) (((a) + (b) - 1) / (b))
#define UMAX(rows, nseg) CDIV(CDIV(rows, NB) * (nseg), NW)
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define PART_N (NW * MAX2(UMAX(4096, 4), MAX2(UMAX(INTER, 8), UMAX(H, 12))))

__device__ __forceinline__ void row_range(int nrows, int& r0, int& n) {
    r0 = (int)(((long long)blockIdx.x * nrows) / gridDim.x);
    n = (int)(((long long)(blockIdx.x + 1) * nrows) / gridDim.x) - r0;
}

// F(u) -> pointer to the unit's 256-element bf16 weight segment.
template <int U, class F>
__device__ __forceinline__ void gemv_load(uint4 (&buf)[U], int nunits, F unit_ptr) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int k = 0; k < U; ++k) {
        int u = min(warp + k * NW, nunits - 1);  // tail warps re-load a valid unit (unused)
        buf[k] = R10_LDW((const uint4*)unit_ptr(u) + lane);
    }
}

#if R8_LF && ((R10_PF & 16) || R10_W0LF)
// R10_PF 16 / R10_W0LF 1: the weight loads of gemv_load_lf, predicated in the PTX. Warp 0's registers are dead (stage_to_buf overwrites them), so
// with a plain `if (warp != 0)` ptxas runs the loads in warp 0 too (seen in the r9a SASS), and the exit of warp 0's barrier poll waits for them.
// The predicate is computed from %tid.x inside the asm: a C bool is constant-folded inside the `if (warp != 0)` and ptxas then drops it
template <int M>  // M 1: predicated on %tid.x >= 32; M 2 (R10_W0LF 2): the same load without the predicate, to isolate the cost of the asm form
__device__ __forceinline__ uint4 ld_v4_nc(const uint4* p) {
    uint4 v;
    if (M == 1)
        asm("{ .reg .pred q; .reg .u32 tt; mov.u32 tt, %%tid.x; setp.ge.u32 q, tt, 32; mov.u32 %0, 0; mov.u32 %1, 0; mov.u32 %2, 0; mov.u32 %3, 0;\n"
                     "  @q ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4]; }"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    else
        asm("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
#endif
#if R8_LF
// Load-free poller variant: warp 0 issues nothing; warps 1-15 load their own units (as gemv_load) and
// also cp.async warp 0's units into r8_stage. After the barrier warp 0 runs stage_to_buf.
template <int G = 0, int U, class F>  // G: 0 plain loads, 1 (R10_PF 16, R10_W0LF 1) ld_v4_nc predicated on warp != 0, 2 (R10_W0LF 2) ld_v4_nc unpredicated
__device__ __forceinline__ void gemv_load_lf(uint4 (&buf)[U], int nunits, F unit_ptr) {
    static_assert(U * 32 * 16 <= R8_STAGE_BYTES, "stage fits");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#if R10_O_RING
    static_assert(U * NT * 16 <= R10_RING_BYTES, "ring fits");
#pragma unroll
    for (int k = 0; k < U; ++k) buf[k] = R10_RING_REF(k);
#else
    if (warp != 0) {
#pragma unroll
        for (int k = 0; k < U; ++k) {
#if R10_O_REG
            buf[k] = r10_oreg(k);
#else
            int u = min(warp + k * NW, nunits - 1);
#if R8_LF && ((R10_PF & 16) || R10_W0LF)
            if (G) buf[k] = ld_v4_nc<G>((const uint4*)unit_ptr(u) + lane);
            else
#endif
            buf[k] = R10_LDW((const uint4*)unit_ptr(u) + lane);
#endif
        }
#if !R10_O_REG
        const int t = threadIdx.x - 32;
        static_assert(U * 32 <= NT - 32, "one stage entry per helper thread");
        if (t < U * 32) {
            const int k = t >> 5, ln = t & 31, u = min(k * NW, nunits - 1);
            cpa16(&r8_stage[t], (const uint4*)unit_ptr(u) + ln);
        }
#endif
        asm volatile("cp.async.commit_group;" ::: "memory");
    } else {  // warp 0: defined (overwritten by stage_to_buf); without this ptxas spills 320 B
#pragma unroll
        for (int k = 0; k < U; ++k) buf[k] = make_uint4(0, 0, 0, 0);
    }
#endif
}
template <int U>
__device__ __forceinline__ void stage_to_buf(uint4 (&buf)[U]) {
#if !R10_O_RING
    if ((threadIdx.x >> 5) == 0) {
#pragma unroll
#if R10_O_REG
        for (int k = 0; k < U; ++k) buf[k] = r10_oreg(k);
#else
        for (int k = 0; k < U; ++k) buf[k] = r8_stage[k * 32 + (threadIdx.x & 31)];
#endif
    }
#endif
}
#if R10_DEF == 1
// Stage hand-off of the DEF paths on named barrier 1 (512 threads: the 480 helpers arrive, warp 0 syncs). A helper drains its cp.async group (the
// stage entries it issued in the window) and arrives; warp 0 syncs, then copies the stage into its registers. Barrier 0 is __syncthreads; the kernel
// uses no other named barrier. Each use of barrier 1 completes before the next one: the dot's __syncthreads (which needs warp 0) lies between them.
// WAR on r8_stage: the next cp.async into r8_stage (fused phase group A, fused -> GU window, GU -> down window, down -> QKV window) is issued after
// the bar_arrive that follows this phase's dot, and that bar_arrive's __syncthreads comes after warp 0's stage_to_buf.
template <int U>
__device__ __forceinline__ void def_handoff(uint4 (&buf)[U]) {
    if ((threadIdx.x >> 5) != 0) asm volatile("cp.async.wait_all;\n\tbar.arrive 1, 512;" ::: "memory");
    else {
        asm volatile("bar.sync 1, 512;" ::: "memory");
        stage_to_buf(buf);
    }
}
#endif
#endif

// part[u] = dot(unit u weights, s[voff(u) ...]) (s in sperm layout). All U units
// are computed unconditionally (tail warps redo a valid unit); the U per-lane
// sums are reduced together by a transpose butterfly: each level halves the
// values a lane holds, so the warp needs P-1+(5-log2 P) shuffles instead of 5U.
template <int U, class V>
__device__ __forceinline__ void gemv_dot(const uint4 (&buf)[U], int nunits, V voff, const float* s,
                                         float* part) {
    constexpr int LG = U <= 1 ? 0 : U <= 2 ? 1 : U <= 4 ? 2 : U <= 8 ? 3 : 4;
    constexpr int P = 1 << LG;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float v[P];
#pragma unroll
    for (int k = 0; k < U; ++k) {
        const int u = min(warp + k * NW, nunits - 1);
        const uint4 w = buf[k];
        const float* xs = s + voff(u) + lane * 4;
        float4 a = *(const float4*)xs, b = *(const float4*)(xs + 128);
        v[k] = bflo(w.x) * a.x + bfhi(w.x) * a.y + bflo(w.y) * a.z + bfhi(w.y) * a.w
             + bflo(w.z) * b.x + bfhi(w.z) * b.y + bflo(w.w) * b.z + bfhi(w.w) * b.w;
    }
#pragma unroll
    for (int k = U; k < P; ++k) v[k] = 0.f;
#pragma unroll
    for (int lv = 0; lv < LG; ++lv) {
        const int h = P >> (lv + 1), o = 16 >> lv;
        const bool hi = lane & o;
#pragma unroll
        for (int i = 0; i < h; ++i) {
            float send = hi ? v[i] : v[i + h], keep = hi ? v[i + h] : v[i];
            v[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
    }
#pragma unroll
    for (int o = 16 >> LG; o; o >>= 1) v[0] += __shfl_xor_sync(0xffffffffu, v[0], o);
    // lane holds the full sum of value k = lane >> (5 - LG)
    const int k = lane >> (5 - LG), u = warp + k * NW;
    if ((lane & ((32 >> LG) - 1)) == 0 && k < U && u < nunits) part[u] = v[0];
}

#if R9_REGACT
// gemv_dot with the activations in registers: act[j][e] = logical element e of the warp's j-th segment
// (lane*8 .. +7 of it). Unit k uses act[k % NSEG] (QKV, gate/up: one segment; down: three).
template <int U, int NSEG>
__device__ __forceinline__ void gemv_dot_reg(const uint4 (&buf)[U], int nunits, const float (&act)[NSEG][8],
                                             float* part) {
    constexpr int LG = U <= 1 ? 0 : U <= 2 ? 1 : U <= 4 ? 2 : U <= 8 ? 3 : 4;
    constexpr int P = 1 << LG;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float v[P];
#pragma unroll
    for (int k = 0; k < U; ++k) {
        const uint4 w = buf[k];
        const float* a = act[k % NSEG];
        v[k] = bflo(w.x) * a[0] + bfhi(w.x) * a[1] + bflo(w.y) * a[2] + bfhi(w.y) * a[3]
             + bflo(w.z) * a[4] + bfhi(w.z) * a[5] + bflo(w.w) * a[6] + bfhi(w.w) * a[7];
    }
#pragma unroll
    for (int k = U; k < P; ++k) v[k] = 0.f;
#pragma unroll
    for (int lv = 0; lv < LG; ++lv) {
        const int h = P >> (lv + 1), o = 16 >> lv;
        const bool hi = lane & o;
#pragma unroll
        for (int i = 0; i < h; ++i) {
            float send = hi ? v[i] : v[i + h], keep = hi ? v[i + h] : v[i];
            v[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
    }
#pragma unroll
    for (int o = 16 >> LG; o; o >>= 1) v[0] += __shfl_xor_sync(0xffffffffu, v[0], o);
    const int k = lane >> (5 - LG), u = warp + k * NW;
    if ((lane & ((32 >> LG) - 1)) == 0 && k < U && u < nunits) part[u] = v[0];
}
#endif

extern "C" __global__ void __launch_bounds__(NT, 1) qwen3_step(
    const long long* __restrict__ token,
    const bf16_t* __restrict__ ln1, const bf16_t* __restrict__ wq,
    const bf16_t* __restrict__ wk, const bf16_t* __restrict__ wv,
    const bf16_t* __restrict__ qn, const bf16_t* __restrict__ kn,
    const bf16_t* __restrict__ wo, const bf16_t* __restrict__ ln2,
    const bf16_t* __restrict__ wg, const bf16_t* __restrict__ wu,
    const bf16_t* __restrict__ wd, const bf16_t* __restrict__ fnorm,
    const bf16_t* __restrict__ embed, const bf16_t* __restrict__ kcache,
    const bf16_t* __restrict__ vcache,
    float* __restrict__ logits, long long* __restrict__ next_token,
    bf16_t* __restrict__ k_write, bf16_t* __restrict__ v_write,
    float* scratch, unsigned* sync, float* pval, int* pidx, const float* __restrict__ rope)
{
    __shared__ __align__(16) float sbuf[INTER];
    __shared__ float red[32];
    __shared__ int redi[32];
    __shared__ float part[PART_N];
    __shared__ __align__(16) bf16_t sVc[CTX * HD];  // fused attention: V-cache rows of kv-head fg
#if R10_GUE
    __shared__ __align__(8) unsigned long long gu_mbar;  // completes (phase l) when layer l's gate / up rows have landed in sGU
#endif
#if R10_DNE
    __shared__ __align__(8) unsigned long long d_mbar;   // completes (phase l) when layer l's down rows have landed in sD
#endif
#if R11_SKD
    __shared__ __align__(16) float shid[32];  // hid[c0 .. c0 + 32) of this CTA's columns (c0 = ri0 & ~7): slots off .. off + ni hold hid, the rest 0
    __shared__ float sres[8];                 // the residual x[rh0 .. rh0 + nh) (final after the fused -> GU barrier)
#if R11_SKD_RI
    // R11_SKD_RI: the 1024 down sums of this CTA (quad q = rows 4q .. 4q + 3 at sdn[4q], residual not yet added) live in the first 4 KB of sbuf. sbuf is dead in the down part with R11_SKD: its only
    // readers in a layer (smem_act of the GU norm; the QKV smem_act, fused attention) are before the dot's __syncthreads / after the next bar_arrive (see the WAR comment at the sdn reads).
    float* const sdn = sbuf;
#endif
#endif
#if R10_QKE
    __shared__ __align__(8) unsigned long long q_mbar;   // completes (phase l) when layer l's QKV rows have landed in sGU
#endif

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
#if R9_PROBE
    // Warp 1 lane 0 (tid 32) times each gemv phase: probe_s[4p + i] sums TA/TW/TD/TR minus T0 for phase p
    // (0 QKV, 1 gate/up, 2 down); slot 12 counts layers. Only tid 32 touches probe_s (no sync needed).
    __shared__ unsigned long long probe_s[16];
    unsigned pb_t0 = 0;
    if (tid == 32) for (int i = 0; i < 16; ++i) probe_s[i] = 0;
#define PB_COUNT() do { if (tid == 32) probe_s[12] += 1; } while (0)
#define PB_T0() do { if (tid == 32) pb_t0 = r9_clk(); } while (0)
#define PB_ACC(p, i) do { if (tid == 32) probe_s[(p) * 4 + (i)] += (unsigned)(r9_clk() - pb_t0); } while (0)
    // TA: activations ready (dep is a value that depends on them); TW: weight buf[U-1] arrived (the STS of a
    // value derived from it cannot issue before the load returns, the clock read comes after the STS)
#define PB_TA(p, dep) do { if (tid == 32) { probe_s[14] = __float_as_uint(dep); PB_ACC(p, 0); } } while (0)
#define PB_TW(p, b) do { if (tid == 32) { probe_s[15] = (b)[sizeof(b) / sizeof(uint4) - 1].x; PB_ACC(p, 1); } } while (0)
#else
#define PB_COUNT()
#define PB_T0()
#define PB_ACC(p, i)
#define PB_TA(p, dep)
#define PB_TW(p, b)
#endif
#if R10_PROBE2
    // thread 32 owns the phase / barrier / sub-mark slots (E: barrier exit, A: before bar_arrive); threads 0, 64, 96 their own slots
    unsigned p2_e = 0, p2_a = 0;
    bool p2_have = false;
    if (tid == 32) for (int i = 0; i < 64; ++i) p2_s[i] = 0;  // visible to threads 0, 64, 96 after the first bar_arrive
#if R10_PROBE3
#define P3_EB(t) p2_eb = (t)
#else
#define P3_EB(t)
#endif
#define P2_E(p) do { if (tid == 32) { const unsigned t_ = p2_clk(); if (p2_have) p2_s[4 + ((p) + 3) % 4] += t_ - p2_a; p2_e = t_; P3_EB(t_); } } while (0)
#define P2_A(p) do { if (tid == 32) { const unsigned t_ = p2_clk(); p2_s[p] += t_ - p2_e; p2_a = t_; p2_have = true; } } while (0)
#define P2_SUB(i) do { if (tid == 32) p2_s[i] += p2_clk() - p2_e; } while (0)
#define P2_COUNT() do { if (tid == 32) p2_s[63] += 1; } while (0)
#else
#define P2_E(p)
#define P2_A(p)
#define P2_SUB(i)
#define P2_COUNT()
#endif
#if R10_O_RING
    for (int i = tid; i < R10_RING_BYTES / 16; i += NT) r8_stage[R8_STAGE_BYTES / 16 + i] = make_uint4(0, 0, 0, 0);  // synced by the first bar_arrive
#endif
    const int gwarp = blockIdx.x * NW + warp;
    const int nwarps = gridDim.x * NW;
    unsigned* bar_cnt = sync;
    unsigned bar_target = 0;
    unsigned* ticket = sync + 2;

    float* x = scratch + S_X;
    float* qkv = scratch + S_QKV;
    float* hid = scratch + S_HID;
#if R11_GRPQ
    float* const xo = scratch + S_XO;  // the O-proj red.add target, see the R11_GRPQ comment at the fused phase
#endif
#if R11_SKD
    // x ping-pong: layer l reads and red.adds xa and builds xn = xa + (O-proj sum already in xa) + down partials. hid is not used with R11_SKD (xb1 is its scratch slot, 1024 floats of the 3072).
    static_assert(NL % 2 == 0, "the final x (after layer NL - 1) is in xb0");
    static_assert(S_HID % 4 == 0 && S_X % 4 == 0, "x buffers 16-byte aligned (red.add.v4)");
    float* const xb0 = scratch + S_X;
    float* const xb1 = scratch + S_HID;
#endif

    // Per-phase row ownership and weight register buffers. Each phase's weights
    // are issued between the arrival and the wait of the grid barrier that
    // precedes the phase, so their DRAM round trip overlaps barrier latency.
    int rq0, nq, rh0, nh, ri0, ni;
    row_range(4096, rq0, nq);
    row_range(H, rh0, nh);
#if R10_NF == 1
    // gate/up rows split over the QH * NSL fused CTAs only; the other CTAs own none (ni = 0)
    static_assert(UMAX(INTER, 8) >= CDIV(CDIV(INTER, QH * NSL) * 8, NW), "144-way gate/up split fits the register buffer");
    static_assert(CDIV(CDIV(INTER, QH * NSL) * 8, NW) * 32 * 16 <= R8_STAGE_BYTES, "144-way gate/up split fits the stage");
    ri0 = blockIdx.x < QH * NSL ? (int)(((long long)blockIdx.x * INTER) / (QH * NSL)) : 0;
    ni = blockIdx.x < QH * NSL ? (int)(((long long)(blockIdx.x + 1) * INTER) / (QH * NSL)) - ri0 : 0;
#else
    row_range(INTER, ri0, ni);
#endif
    uint4 bq[UMAX(4096, 4)], bgu[UMAX(INTER, 8)], bd[UMAX(H, 12)];
    // fused attention + split-K O-proj ownership: q-head fh (kv-head fg), Wo rows [fr0, fr0+fn)
    const bool fused = blockIdx.x < QH * NSL;
    // recomputed from blockIdx at each use (as const ints held across the layer loop they spill)
#define fh ((int)blockIdx.x % QH)
#define fs ((int)blockIdx.x / QH)
#define fg (fh >> 1)
#define fr0 (fs * NQD / NSL * 4)
#define fn ((fs + 1) * NQD / NSL * 4 - fr0)
    static_assert((NQD + NSL - 1) / NSL * 4 * (HD / 8) <= WOU * NT, "Wo slice fits WOU uint4 per thread");
    static_assert((WOU - 1) * NT / 16 <= NQD / NSL * 4, "only the last Wo uint4 can fall past a slice");
    static_assert(PPW * NW == CTX, "positions split evenly over warps");

    auto load_qkv = [&](int l) {
#if R8_LF
        gemv_load_lf<R10_LFQ>(bq, nq * 4, [&](int u) -> const bf16_t* {
#else
        gemv_load(bq, nq * 4, [&](int u) -> const bf16_t* {
#endif
            int r = rq0 + (u >> 2), sg = (u & 3) * 256;
            if (r < 2048) return wq + ((size_t)R10_WL(l) * 2048 + r) * H + sg;
            if (r < 3072) return wk + ((size_t)R10_WL(l) * 1024 + (r - 2048)) * H + sg;
            return wv + ((size_t)R10_WL(l) * 1024 + (r - 3072)) * H + sg;
        });
    };

#if R10_KVS
    // sWo / sK in dynamic smem behind the stage (R8_STAGE_BYTES + 65536 B); sVc is the static array of today. R10_KVM: behind sGU (and sD) when those exist
#if R10_KVM
#define R10_KVOFF ((R10_GUE ? 86016 : 0) + (R10_DNE ? 43008 : 0))
#else
#define R10_KVOFF 0
#endif
    uint4* const sWo = r8_stage + (R8_STAGE_BYTES + R10_KVOFF) / 16;  // WOU * NT uint4: entry k * NT + t is what thread t held in bw[k]
    uint2* const sK = (uint2*)(sWo + WOU * NT);         // PPW * NW * 32 uint2: entry (j * NW + w) * 32 + lane is what thread (w, lane) held in kk[j]
    static_assert(WOU * NT * 16 + PPW * NW * 32 * 8 == 65536, "sWo + sK are 64 KB");
    // Wo / K / V of layer ll for fused CTAs, issued by threads 32.. (warp 0 issues nothing) as ONE cp.async group, committed right after the QKV
    // stage group (load_qkv) of the same window. They depend on no data of the step. Consumed after the fused prep's cp.async.wait_all + __syncthreads.
    // WAR: layer ll + 1's writes into sWo / sK / sVc start in the down -> QKV window of layer ll, after layer ll's fused phase (the last reads of
    // these buffers). That phase is followed by at least two __syncthreads (the bar_arrive of the fused -> GU barrier and the one of the GU -> down
    // barrier, plus those inside the GU and down phases) that every thread passes before it issues.
    // R10_KVM: the same body, issued in the QKV phase of layer ll itself (R10_KVM_AT 0: right after the norm call returns; 1: right after the dot's
    // __syncthreads). WAR: the last reads of sWo / sK / sVc of layer ll - 1 are in the fused phase of ll - 1 (attention, O-proj). After them come the
    // fused bar_arrive and bar_wait (2 __syncthreads), the GU phase's norm and dot __syncthreads, the GU bar_arrive, the down phase's __syncthreads,
    // the down bar_arrive, the bar_wait and the QKV norm __syncthreads (more than 2 in all), all passed by every thread, before the issue.
    // cp.async group audit (KVM, R10_DEF 2 or 0): the QKV stage group (load_qkv, issued in the down -> QKV window) is drained by the QKV norm's
    // cp.async.wait_all (rmsd_to_smem<true> / rms_to_smem<true>, WITHOUT the KVQ keep-one-group variant), which returns BEFORE the KVM group is committed.
    // After the KVM issue in the QKV phase no other wait_all / wait_group runs until the fused prep's wait_all: stage_to_buf, smem_act, the dot and the
    // store hold none; R10_SK is empty (no R10_STRICT); R9_LF_WAIT is empty (R9_DIRECT 0); the window loads of R10_PF 4 (nwq) are plain __ldg. Non-fused
    // CTAs issue no group. R10_GUE / R10_DNE / R10_QKE copies use mbarriers, not cp.async groups. The prep's wait_all therefore waits for exactly the KVM group.
    auto issue_kve = [&](int ll) {
        if (fused && warp != 0) {
#pragma unroll 1
            for (int e = tid - 32; e < WOU * NT; e += NT - 32) {  // Wo: sWo[k * NT + t], row fr0 + t / 16 + k * NT / 16, columns fh * HD + (t % 16) * 8
                const int k = e / NT, t = e % NT;
                const bf16_t* wb0 = wo + ((size_t)R10_WL(ll) * H + fr0 + (t >> 4)) * 2048 + fh * HD + (t & 15) * 8;
                cpa16(&sWo[e], (const uint4*)(k < WOU - 1 || (t >> 4) + k * (NT / 16) < fn ? wb0 + k * (NT / 16) * 2048 : wb0));
            }
#pragma unroll 1
            for (int e = tid - 32; e < PPW * NW * 32; e += NT - 32) {  // K into sK, V into sVc: position w * PPW + j, dims lane * 4..+3
                const int j = e / (NW * 32), w = (e >> 5) % NW, ln = e & 31;
                const size_t off = (((size_t)ll * CTX + w * PPW) * KVH + fg) * HD + ln * 4 + j * KVH * HD;
                cpa8(&sK[e], kcache + off);
                cpa8((char*)sVc + ((w * PPW + j) * HD + ln * 4) * 2, vcache + off);
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
    };
#endif

#if R10_GUE
    // Gate / up rows [ri0, ri0 + ni) of layer ll as 2 TMA bulk copies into dynamic smem behind the stage: sGU[0 .. ni * H) gate rows, sGU[NIMAX * H ..) up
    // rows (NIMAX = 21: 2 * 21 * 1024 * 2 B = 86016 B). Issued by thread 128 (warp 4 lane 0) of every CTA in the fused phase (R10_GUE_AT), when HBM is mostly
    // idle, so the fused -> GU window carries no GU bytes. Layer l is mbarrier phase l (parity l & 1); the GU phase waits for it before its first sGU read.
    // R10_GUE_AT 2: the same copies in 4 row chunks (issue_gue_chunk); expect_tx once, for all bytes, with chunk 0.
    // WAR: layer l + 1's copy overwrites sGU. The last generic reads of layer l's rows are in the GU phase of layer l (the bgu fill, before the dot). The
    // dot's __syncthreads, the bar_arrive and bar_wait of the GU -> down barrier, the down phase's __syncthreads and the bar_arrive and bar_wait of the
    // down -> QKV barrier (at least 3 __syncthreads, all passed by every thread) come before layer l + 1's issue; the issuing thread then runs
    // fence.proxy.async.shared::cta (generic reads before async-proxy writes) as the first statement of the issue.
    // R10_QKE: sGU also holds the QKV rows of layer l (read by the QKV phase of l, before the GU copy of l is issued at the fused start of l). Between that
    // last read (bq fill, before the QKV dot) and the issue: the QKV dot's __syncthreads, the bar_arrive and the bar_wait of the QKV -> fused barrier
    // (3 __syncthreads, passed by every thread), then the fence. And the QKV copy for l + 1 (issued in the first statement of the down phase of l)
    // overwrites the GU rows of l, whose last reads are the bgu fill of the GU phase of l: the GU dot's __syncthreads and the bar_arrive and bar_wait of
    // the GU -> down barrier lie between (3 __syncthreads). The QKV copy of l + 1 and the GU copy of l + 1 therefore never overlap in time: the GU copy is
    // issued after the QKV phase of l + 1 has read its rows.
    // R10_GUE_AT 3: the issue is right after the prep __syncthreads of the fused phase (fused CTAs), later than with AT 0 in the same phase, so the WAR
    // argument above holds as is (non-fused CTAs issue at fused start, as AT 0 / 1). With R10_QKE_AT 1 the QKV copy of l + 1 is issued in the down phase of l
    // after the hid __syncthreads: the GU phase of l has read sGU (bgu fill) before the GU dot's __syncthreads, the GU bar_arrive and bar_wait; and the GU copy of
    // l + 1 is issued after the QKV phase of l + 1 has read the QKV rows (QKV dot sync, 2 barrier syncs, the prep sync).
    static_assert(2 * CDIV(INTER, NB) * H * 2 <= 86016, "sGU fits the 86016 B of dynamic smem (_dyn_smem)");
#define R10_NIMAX CDIV(INTER, NB)
    bf16_t* const sGU = (bf16_t*)(r8_stage + R8_STAGE_BYTES / 16);
#if R10_GUE_AT == 2
    // chunk ck of 4: rows [ck * ni / 4, (ck + 1) * ni / 4) of the gate and of the up block (2 bulk copies, each a contiguous row block). One expect_tx for the
    // full 2 * ni * H * 2 B goes with chunk 0 (the barrier's only arrival; the later chunks only add complete_tx). The WAR argument above holds for every
    // chunk (each is issued later than chunk 0 in the same phase); the fence.proxy.async is repeated, which is harmless.
    auto issue_gue_chunk = [&](int ll, int ck) {
        if (tid == 128) {
            const int r0 = ck * ni / 4, nr = (ck + 1) * ni / 4 - r0;
            const unsigned mb = (unsigned)__cvta_generic_to_shared(&gu_mbar);
            const unsigned dst_gate = (unsigned)__cvta_generic_to_shared(sGU + r0 * H);
            const unsigned dst_up = (unsigned)__cvta_generic_to_shared(sGU + R10_NIMAX * H + r0 * H);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (ck == 0)
                asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(mb), "r"(2 * ni * H * 2) : "memory");
            if (nr > 0) {
                asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                             :: "r"(dst_gate), "l"(wg + ((size_t)R10_WL(ll) * INTER + ri0 + r0) * H), "r"(nr * H * 2), "r"(mb) : "memory");
                asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                             :: "r"(dst_up), "l"(wu + ((size_t)R10_WL(ll) * INTER + ri0 + r0) * H), "r"(nr * H * 2), "r"(mb) : "memory");
            }
        }
    };
#else
    auto issue_gue = [&](int ll) {
        if (tid == 128) {
            const unsigned mb = (unsigned)__cvta_generic_to_shared(&gu_mbar);
            const unsigned dst_gate = (unsigned)__cvta_generic_to_shared(sGU);
            const unsigned dst_up = (unsigned)__cvta_generic_to_shared(sGU + R10_NIMAX * H);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(mb), "r"(2 * ni * H * 2) : "memory");
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                         :: "r"(dst_gate), "l"(wg + ((size_t)R10_WL(ll) * INTER + ri0) * H), "r"(ni * H * 2), "r"(mb) : "memory");
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                         :: "r"(dst_up), "l"(wu + ((size_t)R10_WL(ll) * INTER + ri0) * H), "r"(ni * H * 2), "r"(mb) : "memory");
        }
    };
#endif
    // GU phase, every thread, just before the dot: wait for layer l's copy, then bgu[k] from sGU with the unit mapping of gemv_load_lf
    // (u = min(warp + k * NW, ni * 8 - 1): row u >> 3, matrix (u >> 2) & 1, segment u & 3; the lane's uint4 at + lane). Warp 0 does the same (no stage for GU).
    // The tail clamp reads valid rows only; rows past ni in sGU are never read.
    auto gu_fill = [&](int ll) {
        unsigned ok;
        do {
            asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                         : "=r"(ok) : "r"((unsigned)__cvta_generic_to_shared(&gu_mbar)), "r"(ll & 1) : "memory");
        } while (!ok);
        P3_P(58);
#pragma unroll
        for (int k = 0; k < UMAX(INTER, 8); ++k) {
            const int u = min(warp + k * NW, ni * 8 - 1);
            bgu[k] = ((const uint4*)(sGU + ((u >> 2) & 1) * (R10_NIMAX * H) + (size_t)(u >> 3) * H + (u & 3) * 256))[lane];
        }
    };
#endif
#if R11_LMS_KB
    // R11_LMS: thread 128 (the thread that issues the TMA copies) issues ONE cp.async.bulk.prefetch.L2 per staged layer ll >= NL - R11_LMS_L: embed bytes
    // [(s * NB + b) * KB * 1024, + KB * 1024) with s = ll - (NL - R11_LMS_L), so the slices of all CTAs and staging steps tile the prefix of R11_LMS_L * NB * KB KB of embed (row-major
    // [VOCAB][H] bf16, 2 KB per row). The LM head (R10_LMH >= 2, no DYN / CONSEC) reads rows [k * 2 * nwarps, (k + 1) * 2 * nwarps) in its iteration k, i.e. that prefix first.
    // The prefetch is a hint (no data is written, no ordering with anything is needed). R11_LMS_NULL: the same instruction with size 16.
    auto issue_lms = [&](int ll) {
        if (tid == 128 && ll >= NL - R11_LMS_L) {
            const int st = ll - (NL - R11_LMS_L);
            const unsigned long long addr = (unsigned long long)embed + ((unsigned long long)(st * NB + (int)blockIdx.x)) * (unsigned long long)(R11_LMS_KB * 1024);
            const unsigned sz = R11_LMS_NULL ? 16u : (unsigned)(R11_LMS_KB * 1024);
#if R11_LMS_POL
            unsigned long long pol;
            asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
            asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" :: "l"(addr), "r"(sz), "l"(pol) : "memory");
#else
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(addr), "r"(sz) : "memory");
#endif
        }
    };
#endif
#if R11_SKD
    // R11_SKD: this CTA's down COLUMNS [ri0, ri0 + ni) of layer ll (the columns that multiply the hid values it computes) as 4 TMA tensor copies of a
    // 256-row x 32-column box into sD (65536 B, dynamic smem behind sGU): row r of the 1024 is at sD + r * 32 (64 B per row, no swizzle). Columns start at
    // c0 = ri0 & ~7 (16-byte aligned global address), off = ri0 - c0 (0..7), off + ni <= 28 <= 32; box columns past 3072 (c0 + 32 > INTER) are zero-filled
    // by TMA and still count in the transaction bytes. The tensor map r11_wd_map (3D: INTER, H, NL) is a __device__ variable written by the host before the
    // launch (no in-kernel modification, so no tensormap fence); its generic address is the tensormap operand.
    // Thread 128: fence.proxy.async.shared::cta, then expect_tx 65536 B on d_mbar, then the 4 copies. d_mbar phase l = layer l (wait parity l & 1).
    // WAR on sD: the last generic reads of sD for layer l - 1 are in its down part (inside its GU phase), before the down -> QKV bar_arrive. The issue for layer l
    // follows at least 3 __syncthreads of every thread, for each point:
    //  AT 0 (GU phase of l, after the norm __syncthreads): the down bar_arrive sync and bar_wait sync of l - 1, the QKV norm sync, the QKV dot sync, the QKV -> fused
    //       bar_arrive and bar_wait syncs, the fused -> GU bar_arrive and bar_wait syncs, the GU norm sync (9 or more).
    //  AT 1 (right after the fused bar_arrive of l): l - 1's down bar_arrive and bar_wait syncs, QKV norm, QKV dot, QKV bar_arrive and bar_wait, fused bar_arrive (7).
    //  AT 2 (fused phase after the GU copy issue; fused CTAs after the prep __syncthreads, others at the fused start): l - 1's down bar_arrive and bar_wait,
    //       QKV norm, QKV dot, QKV bar_arrive, bar_wait (6 or more).
    // All of these are passed by every thread before the issuing thread runs the proxy fence (generic reads before async-proxy writes).
    // The mbarrier phases cannot alias: phase l + 1 completes only after the issue for l + 1, which is after every thread of this CTA has left its wait for phase l
    // (they are before the down bar_arrive of l, and the issue is after the bar_wait that follows).
    static_assert(32 * 2 * 1024 == 65536, "sD is 1024 rows x 64 B");
    bf16_t* const sD = (bf16_t*)(r8_stage + (R8_STAGE_BYTES + 86016) / 16);
    const int skd_c0 = ri0 & ~7, skd_off = ri0 - skd_c0;
    auto issue_skd = [&](int ll) {
        if (tid == 128) {
            const unsigned mb = (unsigned)__cvta_generic_to_shared(&d_mbar);
            const unsigned dst = (unsigned)__cvta_generic_to_shared(sD);
            const unsigned long long tm = (unsigned long long)&r11_wd_map;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(mb), "r"(65536) : "memory");
#pragma unroll
            for (int j = 0; j < 4; ++j)
                asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                             :: "r"(dst + j * 256 * 32 * 2), "l"(tm), "r"(skd_c0), "r"(j * 256), "r"(R10_WL(ll)), "r"(mb) : "memory");
        }
    };
    auto skd_wait = [&](int ll) {
        unsigned ok;
        do {
            asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                         : "=r"(ok) : "r"((unsigned)__cvta_generic_to_shared(&d_mbar)), "r"(ll & 1) : "memory");
        } while (!ok);
    };
#endif
#if R10_DNE && !R11_SKD
    // R10_DNE: down rows [rh0, rh0 + nh) of layer ll (contiguous in wd: INTER bf16 per row) as ONE bulk copy into sD (dynamic smem behind sGU, NHMAX = 7 rows,
    // 43008 B), by thread 128, fence.proxy.async first, then expect_tx, then the copy. Layer l is d_mbar phase l (one completion per layer, wait parity l & 1).
    // R10_DNE_AT 0 issues it as the first statement of the GU phase of l, 1 with the GU copies at the fused start, 2 in the GU phase right after rmsd_to_smem returns.
    // WAR: the last generic reads of sD for layer l are the bd fill in the down phase of l (before the dot). After them come the dot's __syncthreads, the
    // down bar_arrive and bar_wait (2 more), the QKV phase's dot __syncthreads and the bar_arrive and bar_wait of the QKV -> fused barrier: at least 3
    // __syncthreads, passed by every thread, before the issue for layer l + 1 (at the earliest the fused start of l + 1); then the proxy fence.
    // R10_DNE_AT 2: the issue for layer l + 1 is in the GU phase of l + 1; the last reads of sD for layer l (d_fill in the down phase of l) are followed by at
    // least 3 __syncthreads (the down dot sync, the down bar_arrive and bar_wait, the QKV norm and dot syncs, the QKV barrier, the fused syncs, ...) before it.
    // With R10_DNE and R10_QKE together no phase uses r8_stage as a stage any more (the fused phase still uses it for group A); it stays allocated (simpler).
    static_assert(CDIV(H, NB) * INTER * 2 <= 43008, "sD fits 43008 B (_dyn_smem)");
    bf16_t* const sD = (bf16_t*)(r8_stage + (R8_STAGE_BYTES + 86016) / 16);
    auto issue_dne = [&](int ll) {
        if (tid == 128) {
            const unsigned mb = (unsigned)__cvta_generic_to_shared(&d_mbar);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(mb), "r"(nh * INTER * 2) : "memory");
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                         :: "r"((unsigned)__cvta_generic_to_shared(sD)), "l"(wd + ((size_t)R10_WL(ll) * H + rh0) * INTER), "r"(nh * INTER * 2), "r"(mb) : "memory");
        }
    };
    // down phase, every thread, just before the dot: wait, then bd[k] from sD with the unit mapping of the down gemv_load_lf call
    // (u = min(warp + k * NW, nh * 12 - 1): row u / 12, segment u % 12; the lane's uint4 at + lane). Warp 0 fills its own units (no stage).
    auto d_fill = [&](int ll) {
        unsigned ok;
        do {
            asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                         : "=r"(ok) : "r"((unsigned)__cvta_generic_to_shared(&d_mbar)), "r"(ll & 1) : "memory");
        } while (!ok);
        P3_P(61);
#pragma unroll
        for (int k = 0; k < UMAX(H, 12); ++k) {
            const int u = min(warp + k * NW, nh * 12 - 1);
            bd[k] = ((const uint4*)(sD + (size_t)(u / 12) * INTER + (u % 12) * 256))[lane];
        }
    };
#endif
#if R10_QKE
    // R10_QKE: QKV rows [rq0, rq0 + nq) of layer ll into sGU (57344 B at most). q, k and v are SEPARATE matrices (wq 2048 rows, wk 1024, wv 1024 per layer), so a
    // CTA whose rows cross row 2048 or 3072 issues one copy per contiguous piece (at most 2: the boundaries are 1024 rows apart, nq <= 28). One expect_tx for
    // nq * H * 2 B, then the pieces. Row formula as load_qkv: wq + (WL * 2048 + r) * H, wk + (WL * 1024 + r - 2048) * H, wv + (WL * 1024 + r - 3072) * H.
    // Layer l + 1: first statement of the down phase of l. Layer 0: the prologue, after a __syncthreads that makes the mbarrier inits visible, before the
    // prologue bar_arrive. Layer l is q_mbar phase l (wait parity l & 1). WAR: see the R10_GUE comment above.
    static_assert(CDIV(4096, NB) * H * 2 <= 86016, "QKV rows fit sGU");
    auto issue_qke = [&](int ll) {
        if (tid == 128) {
            const unsigned mb = (unsigned)__cvta_generic_to_shared(&q_mbar);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(mb), "r"(nq * H * 2) : "memory");
#pragma unroll
            for (int m = 0; m < 3; ++m) {
                const int ms = m == 0 ? 0 : m == 1 ? 2048 : 3072, me = m == 0 ? 2048 : m == 1 ? 3072 : 4096;
                const int lo = max(rq0, ms), hi = min(rq0 + nq, me);
                if (hi > lo) {
                    const bf16_t* w = m == 0 ? wq : m == 1 ? wk : wv;
                    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                                 :: "r"((unsigned)__cvta_generic_to_shared(sGU + (lo - rq0) * H)), "l"(w + ((size_t)R10_WL(ll) * (me - ms) + (lo - ms)) * H),
                                    "r"((hi - lo) * H * 2), "r"(mb) : "memory");
                }
            }
        }
    };
    // QKV phase, every thread, just before the dot: wait, then bq[k] from sGU with the unit mapping of load_qkv (u = min(warp + k * NW, nq * 4 - 1): row u >> 2,
    // segment u & 3; the lane's uint4 at + lane). Warp 0 fills its own units.
    auto q_fill = [&](int ll) {
        unsigned ok;
        do {
            asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                         : "=r"(ok) : "r"((unsigned)__cvta_generic_to_shared(&q_mbar)), "r"(ll & 1) : "memory");
        } while (!ok);
        P3_P(62);
#pragma unroll
        for (int k = 0; k < UMAX(4096, 4); ++k) {
            const int u = min(warp + k * NW, nq * 4 - 1);
            bq[k] = ((const uint4*)(sGU + (size_t)(u >> 2) * H + (u & 3) * 256))[lane];
        }
    };
#endif

#if R10_NF
    // unit u: row ri0 + u/8, matrix (u>>2)&1 (0 gate, 1 up), segment u&3
    auto load_gu = [&](int l) {
#if R8_LF
        gemv_load_lf(bgu, ni * 8, [&](int u) {
#else
        gemv_load(bgu, ni * 8, [&](int u) {
#endif
            const bf16_t* w = ((u >> 2) & 1) ? wu : wg;
            return w + ((size_t)R10_WL(l) * INTER + ri0 + (u >> 3)) * H + (u & 3) * 256;
        });
    };
#endif

#if R9_DIRECT
    // Norm-weight prefetch (warp 1 lanes 0-15: 16 x 128 B lines of the 2 KB weight); warp 0 is the poller.
    auto pf_norm = [&](const bf16_t* p) {
        if (warp == 1 && lane < 16) asm volatile("prefetch.global.L2 [%0];" :: "l"(p + lane * 64));
    };
#if R10_DEF == 1 && (R10_PF & 2)
#define R10_PFN(p) do {} while (0)  // pf_row(...) of R10_PF bit 2 already prefetches the same row
#else
#define R10_PFN(p) pf_norm(p)
#endif
#if R10_DEF == 1
    // R10_DEF: no gate. The stage of the coming phase is waited for inside the phase (def_handoff), not before the window's bar_wait.
#define R9_LF_WAIT() do {} while (0)
#elif R8_LF
    // The stage of the next phase is no longer waited for inside the phase: helpers drain their cp.async
    // before the window's bar_wait, whose exit __syncthreads makes the stage visible to warp 0.
#define R9_LF_WAIT() do { if (warp != 0) asm volatile("cp.async.wait_all;" ::: "memory"); } while (0)
#else
#define R9_LF_WAIT() do {} while (0)
#endif
#endif

#if R10_ROPES
    __shared__ float s_rope[2 * 64];  // rope cos / sin, constant over layers; ordered by the prologue bar_arrive
    if (tid < 2 * 64) s_rope[tid] = rope[tid];
#endif
#if R10_NSM
    // ln1 / ln2 rows (H bf16 each) and qn, kn (HD each) of the coming layer: dynamic smem behind sK (4608 B; with them static the kernel
    // is over the 48 KB static limit), R8_STAGE_BYTES + 65536 + 4608 in all
    bf16_t* const s_ln1 = (bf16_t*)(r8_stage + (R8_STAGE_BYTES + 65536) / 16);
    bf16_t* const s_ln2 = s_ln1 + H;
    bf16_t* const s_qkn = s_ln2 + H;
    // Rows of layer ll, 4.5 KB = 288 x 16 B: one cp.async per helper thread, committed as ONE group (before the down stage group of the same
    // window, or before the QKV stage group in the prologue). Layers >= 1: the down phase's cp.async.wait_all + __syncthreads makes them visible.
    // WAR: layer ll's rows are issued in the GU -> down window of layer ll - 1. The last reads of the rows of layer ll - 1 are ln1 (QKV phase),
    // qn / kn (fused prep) and ln2 (GU phase); each is followed by at least one __syncthreads (the one ending rms_to_smem, the one after the
    // prep, the ones of the fused phase) that every thread passes before this CTA's GU bar_arrive, after which a thread issues.
    auto issue_nsm = [&](int ll) {
        if (warp != 0) {
            const int e = tid - 32;
            if (e < 128) cpa16(s_ln1 + e * 8, ln1 + ll * H + e * 8);
            else if (e < 256) cpa16(s_ln2 + (e - 128) * 8, ln2 + ll * H + (e - 128) * 8);
            else if (e < 272) cpa16(s_qkn + (e - 256) * 8, qn + ll * HD + (e - 256) * 8);
            else if (e < 288) cpa16(s_qkn + HD + (e - 272) * 8, kn + ll * HD + (e - 272) * 8);
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
    };
#define R10_LN1(l) s_ln1
#define R10_LN2(l) s_ln2
#else
#define R10_LN1(l) (ln1 + (l) * H)
#define R10_LN2(l) (ln2 + (l) * H)
#endif
#if R10_PF & 2
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
#endif
#if R10_PF & 17
    unsigned wln = 0;  // ln1 norm weights of the coming QKV phase (this thread's w[t], w[t + NT]; 0 in warp 0 with bit 16)
#endif
#if R10_PF & 16
    unsigned wln0 = 0;  // warp 1 only: warp 0's pair of the same row
#endif

    // x = embed[token]
    {
        long long tok = token[0];
        const bf16_t* e = embed + tok * H;
        for (int i = blockIdx.x * NT + tid; i < H; i += gridDim.x * NT)
            __stcg(x + i, bf2f(e[i]));
    }
#if R11_SKD_NX > 1
    // partials p >= 1 of side 0 (x of layer 0) start at 0: this CTA zeroes its rows [rh0, rh0 + nh) (the row sets of all CTAs cover [0, H)), ordered by the prologue bar_arrive below
    // (the first reader / writer of them is after the prologue barrier; the side-1 partials are zeroed in layer 0 as xn).
    if (tid < nh) {
#pragma unroll
        for (int p = 1; p < R11_SKD_NX; ++p) __stcg(r11_xp(scratch, 0, p) + rh0 + tid, 0.f);
    }
#endif
#if R11_GRPQ
    // xo starts every launch at zero (the previous launch's last down phase leaves it zero; this makes a launch independent of that): this CTA zeroes xo[rh0 .. rh0 + nh) (the row sets of all CTAs cover [0, H)).
    // Ordered before the first O-proj red.add into xo (layer 0, fused phase) by the prologue bar_arrive / bar_wait: a fused CTA reaches it only after the group wait, whose
    // producers have passed the prologue bar_wait (every CTA arrived after these stores; chain: store -> bar_arrive release -> producer's poll + acquire -> its counter release -> consumer's acquire).
    if (tid < nh) __stcg(xo + rh0 + tid, 0.f);
#endif
#if R10_GUE
    if (tid == 0) {  // before the prologue bar_arrive, whose __syncthreads makes the init visible to every thread
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"((unsigned)__cvta_generic_to_shared(&gu_mbar)) : "memory");
#if R10_DNE
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"((unsigned)__cvta_generic_to_shared(&d_mbar)) : "memory");
#endif
#if R10_QKE
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"((unsigned)__cvta_generic_to_shared(&q_mbar)) : "memory");
#endif
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
#endif
#if R10_QKE
    __syncthreads();  // the mbarrier inits are visible to thread 128
    issue_qke(0);
#endif
#if R11_SKEW
    if (tid == 0) r11_sk_n = 0;  // thread 0 is the only reader / writer of the counter
#endif
    bar_arrive(bar_cnt);
#if R10_NSM
    issue_nsm(0);  // before the QKV stage group
#endif
#if !R10_QKE
    load_qkv(0);
#endif
#if R10_KVE
    issue_kve(0);  // after the QKV stage group (R10_KVM: layer 0 is issued in the QKV phase of the loop)
#endif
#if R10_PF & 16
    ld_norm16(ln1, wln, wln0);
#elif R10_PF & 1
    wln = ld_norm2(ln1);
#endif
#if R10_PF & 2
    pf_row(ln1);
    pf_qkn(0);
#endif
#if R9_DIRECT
    R10_PFN(ln1);
    R9_LF_WAIT();
#endif
    R10_SK(r10_fold(bq));
#if R10_NSM
    // layer 0: the rows (oldest group) must be complete before the QKV phase reads s_ln1; the exit __syncthreads of bar_wait makes them visible
    if (warp != 0) {
        if (fused) asm volatile("cp.async.wait_group 2;" ::: "memory");  // groups: rows, QKV stage, K / V / Wo
        else asm volatile("cp.async.wait_group 1;" ::: "memory");        // groups: rows, QKV stage
    }
#endif
    bar_wait<R8_SPIN_PRO>(bar_cnt, bar_target);

    for (int l = 0; l < R10_NL; ++l) {
#if R11_SKD
        // shadows the outer x (= xb0, used after the loop by the final norm): in layer l every read and red.add of x is xa, the zeroed buffer is xn
        float* const x = (l & 1) ? xb1 : xb0;
        float* const xn = (l & 1) ? xb0 : xb1;
#if R11_SKD_NX > 1
        const int sa = l & 1, sn = sa ^ 1;  // side of x of this layer / of the next x; partial p of a side is r11_xp(scratch, side, p) (x above is partial 0 of side sa: only the sum is valid)
#endif
#endif
        PB_T0();
        P2_E(0);  // the exit of the previous layer's down barrier (of the prologue barrier: no barrier slot, p2_have is false)
        P2_COUNT();
        P2_ACT0();
        P2_W(0, bq);
        // ---- Phase 1: RMSNorm(ln1) + fused QKV GEMV (4096 rows) ----
#if R9_REGACT
        float act[1][8];  // logical elements lane*8..+7 of segment warp & 3 (u & 3 == warp & 3 as NW = 16)
#endif
#if R9_DIRECT
        // RMSNorm deferred: the dot runs on x * g, the scalar r = rsqrt(mean(x^2) + eps) scales the row sums.
        // Warps 0-3 hold the 4 segments, so their warp sums of x^2 add up to the full sum. red[] is free here:
        // its last use before this phase (Phase 2 / the previous Phase 5 combine reads) is followed by at
        // least 3 __syncthreads (bar_arrive and bar_wait of the barriers in between).
        // x is read after the barrier and consumed before this CTA's next bar_arrive; other CTAs' fused
        // red.adds into x come after the QKV->fused barrier, so the WAR order is the same as in r8d.
#if R10_DEF == 1
        // R10_DEF: the stage hand-off is inside the phase: activation loads, then helpers drain their stage group and arrive on barrier 1, warp 0 syncs
        // on it and copies the stage (def_handoff), then the dot. Nothing waits for the stage before the barrier exit.
        float ssq = norm_act(x + (warp & 3) * 256 + lane * 8, ln1 + l * H + (warp & 3) * 256 + lane * 8, act[0]);
        P2_ACT(0, act[0][0]);
        P3_M(26, __float_as_uint(act[0][0]));
        def_handoff(bq);
        P3_P(57);
        P3_W0(54, bq);
        if (warp < 4) {
            ssq = warp_sum(ssq);
            if (lane == 0) red[warp] = ssq;
        }
        PB_TA(0, act[0][7]);
        PB_TW(0, bq);
        gemv_dot_reg(bq, nq * 4, act, part);
        P3_P(31);
        __syncthreads();
        P3_P(32);
        PB_ACC(0, 2);
        if (tid < nq) {
            const float r = rsqrtf((red[0] + red[1] + red[2] + red[3]) / (float)H + EPS);
            __stcg(qkv + rq0 + tid, r * (part[tid * 4] + part[tid * 4 + 1] + part[tid * 4 + 2] + part[tid * 4 + 3]));
        }
        P3_P(33);
#else
#if R8_LF
        stage_to_buf(bq);
#endif
        {
            float ss = norm_act(x + (warp & 3) * 256 + lane * 8, ln1 + l * H + (warp & 3) * 256 + lane * 8, act[0]);
            if (warp < 4) {
                ss = warp_sum(ss);
                if (lane == 0) red[warp] = ss;
            }
        }
        PB_TA(0, act[0][7]);
        PB_TW(0, bq);
        gemv_dot_reg(bq, nq * 4, act, part);
        __syncthreads();
        PB_ACC(0, 2);
        if (tid < nq) {
            const float r = rsqrtf((red[0] + red[1] + red[2] + red[3]) / (float)H + EPS);
            __stcg(qkv + rq0 + tid, r * (part[tid * 4] + part[tid * 4 + 1] + part[tid * 4 + 2] + part[tid * 4 + 3]));
        }
#endif
#else
#if R8_LF
#if (R10_PF & 16)
        rms_to_smem_r<true R10_PQ(0)>(x, wln, sbuf, red, wln0);
#elif (R10_PF & 1)
        rms_to_smem_r<true R10_PQ(0)>(x, wln, sbuf, red);
#elif R10_O_CONST
        rms_to_smem_r<true R10_PQ(0)>(x, 0x3f803f80u, sbuf, red);
#elif R10_DEF == 2 && R11_SKD_NX > 1
        rmsd_to_smem_p<true>(scratch, sa, R10_LN1(l), sbuf, red);  // x = sum of the NX partials of side sa
#elif R10_DEF == 2
        rmsd_to_smem<!R10_QKE R10_PQ(0)>(x, R10_LN1(l), sbuf, red);  // R10_QKE: no QKV cp.async group in the down -> QKV window (and none left over), so no wait_all
#else
        rms_to_smem<true R10_PQ(0) R10_KVQ_A>(x, R10_LN1(l), sbuf, red);
#endif
#if R10_KVM && R10_KVM_AT == 0
        issue_kve(l);  // R10_KVM: x has arrived (the norm's __syncthreads); K / V / Wo of this layer into smem, one cp.async group after the QKV stage group was drained
#endif
#if !R10_QKE
        stage_to_buf(bq);
        P3_W0(54, bq);
#endif
#else
        rms_to_smem(x, ln1 + l * H, sbuf, red);
#endif
#if R9_REGACT
        smem_act(sbuf, warp & 3, act[0]);
        P3_M(30, __float_as_uint(act[0][7]));
#endif
#if R10_QKE
        q_fill(l);  // just before the dot
#endif
        PB_TA(0, 0.f);
        PB_TW(0, bq);
#if R9_REGACT
        gemv_dot_reg(bq, nq * 4, act, part);
#else
        gemv_dot(bq, nq * 4, [](int u) { return (u & 3) * 256; }, sbuf, part);
#endif
        P3_P(31);
        __syncthreads();
        P3_P(32);
#if R10_KVM && R10_KVM_AT == 1
        issue_kve(l);  // R10_KVM: right after the QKV dot's __syncthreads (before the store)
#endif
        PB_ACC(0, 2);
#if R10_DEF == 2
        if (tid < nq) {  // red[0 .. NW) is read here, before this CTA's bar_arrive; its next writer is after that barrier (rmsd_to_smem)
            const float r = rmsd_r(red);
            __stcg(qkv + rq0 + tid, r * (part[tid * 4] + part[tid * 4 + 1] + part[tid * 4 + 2] + part[tid * 4 + 3]));
        }
#else
        if (tid < nq)
            __stcg(qkv + rq0 + tid, part[tid * 4] + part[tid * 4 + 1] + part[tid * 4 + 2] + part[tid * 4 + 3]);
#endif
        P3_P(33);
#endif
        PB_ACC(0, 3);
#if R11_SKD
        // Zero this CTA's rows [rh0, rh0 + nh) of xn (the row sets of all CTAs cover [0, H)). xn was x of layer l - 1: its last readers (QKV norm, O-proj red.adds,
        // GU norm of l - 1) are before the down -> QKV barrier of l - 1, which every CTA has passed (the layer 0 buffer xb1 is not read before). The first red.add into
        // xn in layer l is in the down part of the GU phase, after the fused -> GU barrier of l, whose release (this CTA's bar_arrive below follows these stores) orders them.
#if R11_SKD_NX > 1
        // NX partials: all of side sn (the same argument per partial: every reader / writer of side sn in layer l - 1 is before the down -> QKV barrier of l - 1; the first red.add in
        // layer l is in the down part, after the fused -> GU barrier of l, whose release orders these stores)
        if (tid < nh) {
#pragma unroll
            for (int p = 0; p < R11_SKD_NX; ++p) __stcg(r11_xp(scratch, sn, p) + rh0 + tid, 0.f);
        }
#else
        if (tid < nh) __stcg(xn + rh0 + tid, 0.f);
#endif
#endif
        PB_COUNT();
        P2_A(0);
#if R11_GRPQ
        // R11_GRPQ: no QKV -> fused grid barrier. Producer: the counters of this CTA's kv groups (r11_grp_arrive: __syncthreads, then thread 0 red.release per group).
        // This CTA still passes the same two __syncthreads here and after the wait below, and bar_cnt / bar_target see one arrival / wait less per layer (each CTA arrives and waits
        // 3 times per layer instead of 4: the grid barrier count per layer drops by one; bar_target counts bar_wait calls only, so it stays consistent).
        r11_grp_arrive(sync, rq0, nq);
#else
        bar_arrive(bar_cnt);
#endif
#if R10_KVS
        // Wo / K / V of this layer are in smem (issue_kve: KVE one phase earlier, KVM inside the QKV phase): the QKV -> fused window holds no load and no cp.async group
#else
        // defined on every path: a fused-only definition keeps them live across the layer loop
        uint4 bw[WOU] = {};
        uint2 kk[PPW] = {};
#if R10_O_WO
        if (fused) {  // oracle: the Wo registers at the program point of the loads
#pragma unroll
            for (int k = 0; k < WOU; ++k)
#if R10_O_REG
                bw[k] = r10_oreg(k);
#else
                bw[k] = R10_RING_REF(k);
#endif
        }
#endif
#if R8_LF
        if (fused && warp != 0) {  // group A: warp 0's Wo / K / V entries (virtual thread vt = lane of warp 0)
            static_assert(WOU * 32 * 16 + PPW * 32 * 8 <= R8_STAGE_BYTES, "stage fits");
            for (int e = tid - 32; e < (WOU + 2 * PPW) * 32; e += NT - 32) {
                if (e < WOU * 32) {
#if !R10_O_WO
                    const int k = e >> 5, vt = e & 31;
                    const bf16_t* wb0 = wo + ((size_t)R10_WL(l) * H + fr0 + (vt >> 4)) * 2048 + fh * HD + (vt & 15) * 8;
                    cpa16(&r8_stage[e], (const uint4*)(k < WOU - 1 || (vt >> 4) + k * (NT / 16) < fn ? wb0 + k * (NT / 16) * 2048 : wb0));
#endif
                } else if (e < (WOU + PPW) * 32) {
                    const int e2 = e - WOU * 32, j = e2 >> 5, vt = e2 & 31;
                    const size_t off0 = (((size_t)l * CTX) * KVH + fg) * HD + vt * 4;
                    cpa8(&((uint2*)(r8_stage + WOU * 32))[e2], kcache + off0 + j * KVH * HD);
                } else {
                    const int e2 = e - (WOU + PPW) * 32, j = e2 >> 5, vt = e2 & 31;
                    const size_t off0 = (((size_t)l * CTX) * KVH + fg) * HD + vt * 4;
                    cpa8((char*)sVc + (vt * 4 + j * HD) * 2, vcache + off0 + j * KVH * HD);
                }
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
#endif
#if R8_LF
        if (fused && warp != 0) {  // own Wo slice, K and V-cache loads as in V1q (group B)
#else
        if (fused) {  // Wo slice and K/V-cache rows are inputs: issue them during the barrier
#endif
            // thread t, uint4 k: slice row t/16 + k*NT/16, head columns (t%16)*8..+7 (only the
            // last k can fall past the slice; it then re-loads row t/16, unused)
#if !R10_O_WO
            const bf16_t* wb = wo + ((size_t)R10_WL(l) * H + fr0 + (tid >> 4)) * 2048 + fh * HD + (tid & 15) * 8;
#pragma unroll
            for (int k = 0; k < WOU; ++k)
                bw[k] = R10_LDW((const uint4*)(k < WOU - 1 || (tid >> 4) + k * (NT / 16) < fn ? wb + k * (NT / 16) * 2048 : wb));
#endif
            // warp w's cached positions w*PPW..+PPW-1: K into registers, V into smem by cp.async
            // (each lane copies, and later reads, only its own dims lane*4..+3)
            const size_t off = (((size_t)l * CTX + warp * PPW) * KVH + fg) * HD + lane * 4;
            const unsigned dst = (unsigned)__cvta_generic_to_shared(sVc + warp * PPW * HD + lane * 4);
#pragma unroll
            for (int j = 0; j < PPW; ++j) {
                kk[j] = __ldg((const uint2*)(kcache + off + j * KVH * HD));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(dst + j * HD * 2),
                             "l"(vcache + off + j * KVH * HD) : "memory");
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
#endif
#if R10_PF4
        // warp 0: qn, warp 1: kn of this layer, lane's dims lane*4..+3 as one uint2; used after the barrier in the prep
        // (R10_PF 32: warp 1 loads qn, warp 2 loads kn, warp 0 loads nothing)
        uint2 nwq = make_uint2(0, 0);
        if (fused && (unsigned)(warp - R10_PFS) < 2u) nwq = __ldg((const uint2*)((warp == R10_PFS ? qn : kn) + l * HD + lane * 4));
#endif
        R10_SK(r10_fold(bw) ^ r10_foldk(kk));
#if R11_GRPQ
        // Consumer: a fused CTA (b < 144, head fh = b % 16, kv head fg = fh / 2) reads qkv Q rows of head fh (in group fg's Q range), K and V rows of head fg: all in the row set of group fg,
        // so it waits for counter fg to reach (l + 1) * |P(fg)| (thread 0 polls with ld.relaxed.gpu, fence.acquire.gpu, then the __syncthreads below publishes to the CTA). CTAs 144 .. 147 do no
        // fused work (the whole fused phase is `if (fused)`; they only issue copies / prefetches that read weights) and read nothing the QKV phase wrote: no wait, only the __syncthreads.
        // What the removed grid barrier ordered, and why each still holds:
        //  1. qkv stores of layer l before the attention reads: the group wait (producer: stores -> __syncthreads -> red.release; consumer: acquire -> __syncthreads -> ldcg).
        //  2. x: the O-proj red.adds of layer l (below) must not start before every CTA has read x in its QKV norm of layer l. Without the barrier this is false, so the reds go to xo, not x
        //     (QKV norm reads x only). x + xo is read by the GU norm and the down residual (after the fused -> GU barrier, which still orders all reds before those reads).
        //  3. xo is zero when the first red.add of layer l lands: the owner zeroes xo[rh0 ..) in the down phase of l - 1 after its xo reads (the GU norm reads of every CTA are before the
        //     GU -> down barrier of l - 1, the owner's own xres read is program-ordered before its zero); that zero is before the owner's down -> QKV arrive of l - 1. A fused CTA's red in
        //     layer l is after its group wait, whose producers have all passed the down -> QKV barrier of l - 1 (chain as for the prologue), so after the zeros. Layer 0: the prologue zero.
        //  4. WAR on qkv: layer l + 1's qkv stores are after the down -> QKV barrier of l, which is after the fused -> GU barrier of l, which every CTA arrives at after its attention reads of qkv.
        //  5. k_write / v_write: written in the fused phase (wr CTAs), read by nobody in the kernel (kcache / vcache are separate inputs). No ordering was provided by the barrier.
        //  6. sVc / bw / kk window loads (cp.async groups A / B, R10_PF4 nwq): per-CTA; the same __syncthreads after the wait, and the same wait_group points inside the fused phase.
        //  7. red[] / part / sbuf reuse and the smem WARs of sGU / sD (the issue-point comments): per-CTA; the two __syncthreads (arrive, post-wait) are kept on every CTA, so every count of
        //     "bar_arrive + bar_wait syncs" in those comments is unchanged.
        //  8. Counter reuse: the counters are monotonic. A layer l + 1 increment cannot satisfy a layer l wait early: layer l + 1's producers write only after the down -> QKV barrier of l, which every
        //     consumer has passed its layer l wait before arriving at (fused -> GU barrier); the goal (l + 1) * |P| is reached by layer l's own increments exactly.
        if (fused) r11_grp_wait<R8_SPIN_QKV>(sync + R11_GC(fg), (unsigned)(l + 1) * r11_pcnt(fg));
        __syncthreads();
#else
        bar_wait<R8_SPIN_QKV>(bar_cnt, bar_target);
#endif
        P2_E(1);
#if R10_GUE
#if R10_GUE_AT == 2
        // paced: every CTA chunk 0 here; non-fused CTAs chunks 1..3 too, fused CTAs after the prep / attention / combine __syncthreads
        issue_gue_chunk(l, 0);
        if (!fused) { issue_gue_chunk(l, 1); issue_gue_chunk(l, 2); issue_gue_chunk(l, 3); }
#else
        if (R10_GUE_AT == 0 || !fused) issue_gue(l);  // layer l's gate / up rows: first statement of the phase (non-fused CTAs also with GUE_AT 1)
#if R11_LMS_KB
        if (!fused) issue_lms(l);  // non-fused CTAs: right after their GU copy issue (fused CTAs: next to theirs, after the prep __syncthreads)
#endif
#if R11_SKD && R11_SKD_AT == 2
        if (!fused) issue_skd(l);  // fused CTAs issue after the prep __syncthreads, next to their GU copy
#endif
#endif
#if R10_DNE && !R11_SKD && R10_DNE_AT == 1
        issue_dne(l);  // layer l's down rows, with the GU copies
#endif
#endif

        // ---- Phase 2: fused attention (q-head fh, all 129 positions, exact in-block
        // softmax) + split-K O-proj of Wo rows [fr0, fr0+fn) x head-fh columns. Each
        // block red.adds its partial row sums into x; every block read x for ln1
        // before the barrier above, and the barrier below orders the adds before ln2. ----
        if (fused) {
            float* sq = sbuf;            // 128: post-norm, post-RoPE q of head fh
            float* sk = sbuf + HD;       // 128: new k
            float* sv = sbuf + 2 * HD;   // 128: new v
            float* soh = sbuf + 3 * HD;  // 128: normalized o of head fh
            float* sow = sbuf + 4 * HD;  // NW x 128 per-warp unnormalized o
            static_assert(4 * HD + NW * HD <= INTER, "attention smem");
            const bool wr = (fh & 1) == 0 && fs == 0;  // writes k_write/v_write of kv-head fg
            // prep roles: warp R10_PFS + 0: q head fh, + 1: k of kv-head fg, + 2: v copy (R10_PFS is 1 with R10_PF 32: warp 0 is idle)
            if ((unsigned)(warp - R10_PFS) < 2u) {
                // q head fh / k of kv-head fg. lane holds dims lane*4..+3
                const float* src = warp == R10_PFS ? qkv + fh * HD : qkv + 2048 + fg * HD;
#if !R10_PF4 && !R10_NSM && !R10_O_CONST
                const bf16_t* nw = warp == R10_PFS ? qn + l * HD : kn + l * HD;
#endif
                float4 a = __ldcg((const float4*)src + lane);
                P3_M(48, __float_as_uint(a.x));  // thread 32 = warp 1 lane 0: holds `a` with and without R10_PF 32
                float r = rsqrtf(warp_sum(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w) / (float)HD + EPS);
                P3_M(49, __float_as_uint(r));
                float e[4] = {a.x, a.y, a.z, a.w}, o[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
#if R10_PF4
                    e[i] = e[i] * r * bf2f((i < 2 ? nwq.x : nwq.y) >> ((i & 1) * 16) & 0xffffu);
#elif R10_O_CONST
                    e[i] = e[i] * r;
#elif R10_NSM
                    e[i] = e[i] * r * bf2f(s_qkn[(warp == R10_PFS ? 0 : HD) + lane * 4 + i]);
#else
                    e[i] = e[i] * r * bf2f(nw[lane * 4 + i]);
#endif
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float pr = __shfl_xor_sync(0xffffffffu, e[i], 16);
                    int j = (lane * 4 + i) & 63;
#if R10_ROPES
                    float cs = s_rope[j], sn = s_rope[64 + j];
#elif R10_O_CONST
                    float cs = 1.f, sn = 0.f;
#else
                    float cs = rope[j], sn = rope[64 + j];
#endif
                    o[i] = e[i] * cs + (lane < 16 ? -pr : pr) * sn;
                }
                *(float4*)((warp == R10_PFS ? sq : sk) + lane * 4) = make_float4(o[0], o[1], o[2], o[3]);
                if (warp == R10_PFS + 1 && wr) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) k_write[(l * KVH + fg) * HD + lane * 4 + i] = f2bf(o[i]);
                }
                P3_P(50);
            } else if (warp == R10_PFS + 2) {
                float4 a = __ldcg((const float4*)(qkv + 3072 + fg * HD) + lane);
                *(float4*)(sv + lane * 4) = a;
                if (wr) {
                    v_write[(l * KVH + fg) * HD + lane * 4 + 0] = f2bf(a.x);
                    v_write[(l * KVH + fg) * HD + lane * 4 + 1] = f2bf(a.y);
                    v_write[(l * KVH + fg) * HD + lane * 4 + 2] = f2bf(a.z);
                    v_write[(l * KVH + fg) * HD + lane * 4 + 3] = f2bf(a.w);
                }
            }
#if R10_KVS
            asm volatile("cp.async.wait_all;" ::: "memory");  // the K / V / Wo group (the only one outstanding): the __syncthreads below makes sWo, sK, sVc visible
#elif R8_LF
            asm volatile("cp.async.wait_group 1;" ::: "memory");
#endif
            P3_P(51);
            __syncthreads();
            P2_SUB(20);
#if R10_GUE && R10_GUE_AT == 2
            issue_gue_chunk(l, 1);  // fused CTAs: right after the prep __syncthreads
#elif R10_GUE && R10_GUE_AT == 3
            issue_gue(l);  // fused CTAs: right after the prep __syncthreads (the fused phase's own qkv read has arrived)
#if R11_LMS_KB
            issue_lms(l);
#endif
#if R11_SKD && R11_SKD_AT == 2
            issue_skd(l);
#endif
#endif
#if R10_KVS
            // no stage copy: K and Wo are read from sK / sWo
#elif R8_LF
            if (warp == 0) {
#if !R10_O_WO
#pragma unroll
                for (int k = 0; k < WOU; ++k) bw[k] = r8_stage[k * 32 + lane];
#endif
#pragma unroll
                for (int j = 0; j < PPW; ++j) kk[j] = ((const uint2*)(r8_stage + WOU * 32))[j * 32 + lane];
            }
#endif
            // warp w: cached positions w*PPW..+PPW-1 (K/V in registers); warp 0 also the new token
            {
                const float4 q = *(const float4*)(sq + lane * 4);
                static_assert(PPW == 8, "transpose butterfly below assumes 8 positions per warp");
                float v[PPW];
#pragma unroll
                for (int j = 0; j < PPW; ++j) {
#if R10_KVS
                    const uint2 kj = sK[(j * NW + warp) * 32 + lane];
#else
                    const uint2 kj = kk[j];
#endif
                    v[j] = bflo(kj.x) * q.x + bfhi(kj.x) * q.y + bflo(kj.y) * q.z + bfhi(kj.y) * q.w;
                }
                // transpose butterfly (as gemv_dot): lane ends with the score of position lane >> 2
#pragma unroll
                for (int lv = 0; lv < 3; ++lv) {
                    const int hh = PPW >> (lv + 1), o = 16 >> lv;
                    const bool hi = lane & o;
#pragma unroll
                    for (int i = 0; i < hh; ++i) {
                        float send = hi ? v[i] : v[i + hh], keep = hi ? v[i + hh] : v[i];
                        v[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
                    }
                }
                v[0] += __shfl_xor_sync(0xffffffffu, v[0], 2);
                v[0] += __shfl_xor_sync(0xffffffffu, v[0], 1);
                const float sc = v[0] * 0.08838834764831845f;
                float m = warp_max(sc), snew = 0.f;
                if (warp == 0) {  // new token (k, v from smem)
                    const float4 k = *(const float4*)(sk + lane * 4);
                    snew = warp_sum(k.x * q.x + k.y * q.y + k.z * q.z + k.w * q.w) * 0.08838834764831845f;
                    m = fmaxf(m, snew);
                }
                const float ee = R11_EXPF(sc - m);
                float lsum = 0.f;
                float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
#if !R10_KVS
                asm volatile("cp.async.wait_all;" ::: "memory");
#endif
#pragma unroll
                for (int j = 0; j < PPW; ++j) {
                    const uint2 vj = *(const uint2*)(sVc + (warp * PPW + j) * HD + lane * 4);
                    const float e = __shfl_sync(0xffffffffu, ee, j * 4);
                    lsum += e;
                    o.x += e * bflo(vj.x); o.y += e * bfhi(vj.x);
                    o.z += e * bflo(vj.y); o.w += e * bfhi(vj.y);
                }
                if (warp == 0) {
                    const float e = R11_EXPF(snew - m);
                    const float4 nv = *(const float4*)(sv + lane * 4);
                    lsum += e;
                    o.x += e * nv.x; o.y += e * nv.y; o.z += e * nv.z; o.w += e * nv.w;
                }
                *(float4*)(sow + warp * HD + lane * 4) = o;
                if (lane == 0) { red[warp] = m; red[NW + warp] = lsum; }
            }
            __syncthreads();
            P2_SUB(21);
#if R10_GUE && R10_GUE_AT == 1
            issue_gue(l);  // fused CTAs: right after the __syncthreads that ends the attention
#elif R10_GUE && R10_GUE_AT == 2
            issue_gue_chunk(l, 2);  // right after the attention __syncthreads
#endif
#if R11_CMB
            static_assert(NT == 4 * HD && NW == 16, "R11_CMB: 4 groups of 4 warps over 128 dims");
            // soh4[4][HD] aliases sq / sk / sv / soh (4 * HD floats): sq, sk, sv are last read inside the attention (before the __syncthreads above, which every thread has passed), soh is unused here.
            // sL[4] is after the sow rows (sbuf + 4 * HD + NW * HD + 4 <= INTER). WAR: no thread writes soh4 / sL before the attention __syncthreads; sow rows are only read here (the O-proj overwrites them as
            // srow after the combine __syncthreads below); soh4 / sL are read by the O-proj between the combine __syncthreads and the O-proj __syncthreads, and the next sbuf writer (the GU norm) is after that.
            float* soh4 = sbuf;
            float* sL = sbuf + 4 * HD + NW * HD;
            {
                const int d = tid & (HD - 1), g = tid >> 7;
                float M = red[0];
#pragma unroll
                for (int w = 1; w < NW; ++w) M = fmaxf(M, red[w]);
                float L = 0.f, o = 0.f;
#pragma unroll
                for (int w = 4 * g; w < 4 * g + 4; ++w) {
                    const float c = R11_EXPF(red[w] - M);
                    L += c * red[NW + w];
                    o += c * sow[w * HD + d];
                }
                soh4[g * HD + d] = o;
                if (d == 0) sL[g] = L;
            }
#else
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
#endif
            __syncthreads();
            P2_SUB(22);
#if R10_GUE && R10_GUE_AT == 2
            issue_gue_chunk(l, 3);  // right after the combine __syncthreads
#endif
            // split-K O-proj: bw[k] is row fr0 + tid/16 + k*NT/16, head columns (tid%16)*8..+7;
            // 16 consecutive lanes share a row and reduce by shuffles into the row partial
            // (slice-local row index, in the dead sow region); then one 16-byte red.add per quad
#if R11_CMB
            const float rL = 1.f / (sL[0] + sL[1] + sL[2] + sL[3]);
            float4 a, b;
            {
                const int i0 = (tid & 15) * 8;
                const float4 a0 = *(const float4*)(soh4 + i0), a1 = *(const float4*)(soh4 + HD + i0), a2 = *(const float4*)(soh4 + 2 * HD + i0), a3 = *(const float4*)(soh4 + 3 * HD + i0);
                const float4 b0 = *(const float4*)(soh4 + i0 + 4), b1 = *(const float4*)(soh4 + HD + i0 + 4), b2 = *(const float4*)(soh4 + 2 * HD + i0 + 4), b3 = *(const float4*)(soh4 + 3 * HD + i0 + 4);
                a = make_float4(((a0.x + a1.x) + (a2.x + a3.x)) * rL, ((a0.y + a1.y) + (a2.y + a3.y)) * rL, ((a0.z + a1.z) + (a2.z + a3.z)) * rL, ((a0.w + a1.w) + (a2.w + a3.w)) * rL);
                b = make_float4(((b0.x + b1.x) + (b2.x + b3.x)) * rL, ((b0.y + b1.y) + (b2.y + b3.y)) * rL, ((b0.z + b1.z) + (b2.z + b3.z)) * rL, ((b0.w + b1.w) + (b2.w + b3.w)) * rL);
            }
#else
            const float4 a = *(const float4*)(soh + (tid & 15) * 8), b = *(const float4*)(soh + (tid & 15) * 8 + 4);
#endif
            float* srow = sow;
#pragma unroll
            for (int k = 0; k < WOU; ++k) {
#if R10_KVS
                const uint4 w = sWo[k * NT + tid];
#else
                const uint4 w = bw[k];
#endif
                float d = bflo(w.x) * a.x + bfhi(w.x) * a.y + bflo(w.y) * a.z + bfhi(w.y) * a.w
                        + bflo(w.z) * b.x + bfhi(w.z) * b.y + bflo(w.w) * b.z + bfhi(w.w) * b.w;
#pragma unroll
                for (int o = 8; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                if ((tid & 15) == 0) srow[(tid >> 4) + k * (NT / 16)] = d;
            }
            __syncthreads();
            P2_SUB(23);
            const int q0 = ctaid_opaque() / QH * NQD / NSL, nq4 = (ctaid_opaque() / QH + 1) * NQD / NSL - q0;
            if (tid < nq4) {
                const float4 d = *(const float4*)(srow + tid * 4);
                asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(
#if R11_SKD_NX > 1
                             // O-proj reds of CTA b -> partial b & (NX - 1) of x of this layer (16 adds per address for NX 1, 16 / NX otherwise); read by the GU norm after the fused -> GU barrier
                             r11_xp(scratch, sa, (int)blockIdx.x & (R11_SKD_NX - 1))
#else
#if R11_GRPQ
                             xo  // R11_GRPQ: the O-proj sum goes to xo (zero here, see the comment at the QKV -> fused wait), not x
#else
                             x
#endif
#endif
                             + (q0 + tid) * 4),
                             "f"(d.x), "f"(d.y), "f"(d.z), "f"(d.w) : "memory");
            }
            P2_SUB(24);
        }
        P2_A(1);
        bar_arrive(bar_cnt);
#if R11_SKD && R11_SKD_AT == 1
        issue_skd(l);  // right after the fused bar_arrive (its __syncthreads has been passed by every thread)
#endif
#if R10_NF == 0 && R10_GUE
        // R10_GUE: no GU weight load and no cp.async group in this window (the rows were issued as bulk copies in the fused phase; bgu is filled in the GU phase)
#elif R10_NF == 0
        // unit u: row ri0 + u/8, matrix (u>>2)&1 (0 gate, 1 up), segment u&3
#if R8_LF
        gemv_load_lf<R10_PFG>(bgu, ni * 8, [&](int u) {
#else
        gemv_load(bgu, ni * 8, [&](int u) {
#endif
            const bf16_t* w = ((u >> 2) & 1) ? wu : wg;
            return w + ((size_t)R10_WL(l) * INTER + ri0 + (u >> 3)) * H + (u & 3) * 256;
        });
#elif R10_NF == 1
        // CTAs with ni == 0 issue no loads and no cp.async group: the cp.async.wait_all of the phase returns at once
        if (ni > 0) load_gu(l);
        else {  // defined on every path (else ptxas spills)
#pragma unroll
            for (int k = 0; k < UMAX(INTER, 8); ++k) bgu[k] = make_uint4(0, 0, 0, 0);
        }
#elif R10_NF == 2
        if (!fused) {  // spin, then issue (no sync needed: every thread spins on its own clock)
            unsigned t0, t;
            asm volatile("mov.u32 %0, %%clock;" : "=r"(t0)::"memory");
            do { asm volatile("mov.u32 %0, %%clock;" : "=r"(t)::"memory"); } while (t - t0 < R10_NF_DELAY);
        }
        load_gu(l);
#else
        if (fused) load_gu(l);
        else {  // defined on every path (else ptxas spills); load_gu fills it after the barrier
#pragma unroll
            for (int k = 0; k < UMAX(INTER, 8); ++k) bgu[k] = make_uint4(0, 0, 0, 0);
        }
#endif
#if R10_PF & 16
        unsigned wln2, wln2_0;
        ld_norm16(ln2 + l * H, wln2, wln2_0);
#elif R10_PF & 1
        const unsigned wln2 = ld_norm2(ln2 + l * H);
#endif
#if R10_PF & 2
        pf_row(ln2 + l * H);
#endif
#if R9_DIRECT
        R10_PFN(ln2 + l * H);
        R9_LF_WAIT();
#endif
        R10_SK(r10_fold(bgu));
        bar_wait<R8_SPIN_FUSED>(bar_cnt, bar_target);
        PB_T0();
        P2_E(2);
#if R10_DNE && !R11_SKD && R10_DNE_AT == 0
        issue_dne(l);  // layer l's down rows: first statement of the GU phase
#endif
        P2_ACT0();
        P2_W(1, bgu);
#if R10_NF == 3
        if (!fused) load_gu(l);  // non-fused CTAs issue after the barrier
#endif

        // ---- Phase 4: RMSNorm(ln2) + gate/up GEMV + SiLU*up ----
#if R9_DIRECT
        // as Phase 1: dot on x * g, r applied to the row sums. red[] is free: its last use (Phase 2 reads,
        // or the Phase 1 combine for CTAs that are not fused) is followed by at least 3 __syncthreads.
        // x is read after the barrier; down's stores to x come after the GU->down barrier (same WAR order as r8d).
#if R10_DEF == 1
        // R10_DEF: as Phase 1 (hand-off on barrier 1 inside the phase); with R10_GUE there is no stage: every thread fills bgu from sGU after the mbarrier wait
        float ssq2 = norm_act(x + (warp & 3) * 256 + lane * 8, ln2 + l * H + (warp & 3) * 256 + lane * 8, act[0]);
        P2_ACT(1, act[0][0]);
        P3_M(34, __float_as_uint(act[0][0]));
#if !R10_GUE
        def_handoff(bgu);
        P3_P(59);
        P3_W0(55, bgu);
#endif
        if (warp < 4) {
            ssq2 = warp_sum(ssq2);
            if (lane == 0) red[warp] = ssq2;
        }
#if R10_GUE
        gu_fill(l);  // just before the dot
#endif
        PB_TA(1, act[0][7]);
        PB_TW(1, bgu);
        gemv_dot_reg(bgu, ni * 8, act, part);
        P3_P(39);
        __syncthreads();
        P3_P(40);
        PB_ACC(1, 2);
        if (tid < ni) {
            const float r = rsqrtf((red[0] + red[1] + red[2] + red[3]) / (float)H + EPS);
            const float* pp = part + tid * 8;
            float gt = r * (pp[0] + pp[1] + pp[2] + pp[3]);
            float up = r * (pp[4] + pp[5] + pp[6] + pp[7]);
            __stcg(hid + ri0 + tid, R11_SILU(gt) * up);
        }
        P3_P(41);
#else
#if R10_GUE
        gu_fill(l);
#elif R8_LF
        stage_to_buf(bgu);
#endif
        {
            float ss = norm_act(x + (warp & 3) * 256 + lane * 8, ln2 + l * H + (warp & 3) * 256 + lane * 8, act[0]);
            if (warp < 4) {
                ss = warp_sum(ss);
                if (lane == 0) red[warp] = ss;
            }
        }
        PB_TA(1, act[0][7]);
        PB_TW(1, bgu);
        gemv_dot_reg(bgu, ni * 8, act, part);
        __syncthreads();
        PB_ACC(1, 2);
        if (tid < ni) {
            const float r = rsqrtf((red[0] + red[1] + red[2] + red[3]) / (float)H + EPS);
            const float* pp = part + tid * 8;
            float gt = r * (pp[0] + pp[1] + pp[2] + pp[3]);
            float up = r * (pp[4] + pp[5] + pp[6] + pp[7]);
            __stcg(hid + ri0 + tid, R11_SILU(gt) * up);
        }
#endif
#else
#if R10_NF == 1
#define R10_OWN if (ni > 0)  // ni is block-uniform: skipped code holds no __syncthreads that another path needs
#else
#define R10_OWN
#endif
#if R8_LF
// R10_GUE: no GU stage group in this window (and none left over from the fused phase: its groups A / B were drained there), so the norm does not
// wait on cp.async and there is no stage_to_buf; bgu is filled from sGU below
#if R10_GUE
#define R10_GUW false
#define R10_GU_STAGE() do {} while (0)
#else
#define R10_GUW true
#define R10_GU_STAGE() do { stage_to_buf(bgu); P3_W0(55, bgu); } while (0)
#endif
#if (R10_PF & 16)
        R10_OWN { rms_to_smem_r<R10_GUW R10_PQ(1)>(x, wln2, sbuf, red, wln2_0); R10_GU_STAGE(); }
#elif (R10_PF & 1)
        R10_OWN { rms_to_smem_r<R10_GUW R10_PQ(1)>(x, wln2, sbuf, red); R10_GU_STAGE(); }
#elif R10_O_CONST
        R10_OWN { rms_to_smem_r<R10_GUW R10_PQ(1)>(x, 0x3f803f80u, sbuf, red); R10_GU_STAGE(); }
#elif R10_DEF == 2 && R11_SKD_NX > 1
        R10_OWN { rmsd_to_smem_p<R10_GUW>(scratch, sa, R10_LN2(l), sbuf, red); R10_GU_STAGE(); }
#elif R10_DEF == 2 && R11_GRPQ
        // R11_GRPQ: x + xo (the O-proj reds of layer l are in xo; all of them precede the fused -> GU barrier this CTA has passed)
        R10_OWN { rmsd2_to_smem<R10_GUW>(x, xo, R10_LN2(l), sbuf, red); R10_GU_STAGE(); }
#elif R10_DEF == 2
        R10_OWN { rmsd_to_smem<R10_GUW R10_PQ(1)>(x, R10_LN2(l), sbuf, red); R10_GU_STAGE(); }
#else
        R10_OWN { rms_to_smem<R10_GUW R10_PQ(1)>(x, R10_LN2(l), sbuf, red); R10_GU_STAGE(); }
#endif
#else
        R10_OWN rms_to_smem(x, ln2 + l * H, sbuf, red);
#endif
#if R10_DNE && !R11_SKD && R10_DNE_AT == 2
        issue_dne(l);  // layer l's down rows: right after the GU norm's __syncthreads (x has arrived for every thread)
#endif
#if R11_SKD
#if R11_SKD_AT == 0
        issue_skd(l);  // layer l's down columns: right after the GU norm's __syncthreads (x has arrived for every thread)
#endif
        // shid slots other than off .. off + ni stay 0 and sres holds the residual; written here, read after the dot's __syncthreads and the shid __syncthreads (next writers: next layer, after
        // this CTA's down bar_arrive __syncthreads, which every thread passes after its reads)
        if (tid < 32) shid[tid] = 0.f;
#if R11_SKD_NX > 1
        if (tid >= 32 && tid < 32 + nh) {  // sum of the NX partials (all loads issued before the first use); final: the fused red.adds precede the fused -> GU barrier
            float xr[R11_SKD_NX];
#pragma unroll
            for (int p = 0; p < R11_SKD_NX; ++p) xr[p] = __ldcg(r11_xp(scratch, sa, p) + rh0 + tid - 32);
            float sum = xr[0];
#pragma unroll
            for (int p = 1; p < R11_SKD_NX; ++p) sum += xr[p];
            sres[tid - 32] = sum;
        }
#else
        if (tid >= 32 && tid < 32 + nh) sres[tid - 32] = __ldcg(x + rh0 + tid - 32);  // x is final: the fused red.adds precede the fused -> GU barrier
#endif
#endif
#if R9_REGACT
        R10_OWN smem_act(sbuf, warp & 3, act[0]);
        P3_M(38, __float_as_uint(act[0][7]));
#endif
#if R10_GUE
        gu_fill(l);
#endif
        PB_TA(1, 0.f);
        PB_TW(1, bgu);
#if R9_REGACT
        R10_OWN gemv_dot_reg(bgu, ni * 8, act, part);
#else
        gemv_dot(bgu, ni * 8, [](int u) { return (u & 3) * 256; }, sbuf, part);
#endif
        P3_P(39);
        __syncthreads();
        P3_P(40);
        PB_ACC(1, 2);
        if (tid < ni) {
            const float* pp = part + tid * 8;
#if R10_DEF == 2
            const float r = rmsd_r(red);  // red[0 .. NW) is read before this CTA's bar_arrive; the next writer (fused phase / next QKV norm) is after it
            float gt = r * (pp[0] + pp[1] + pp[2] + pp[3]);
            float up = r * (pp[4] + pp[5] + pp[6] + pp[7]);
#else
            float gt = pp[0] + pp[1] + pp[2] + pp[3];
            float up = pp[4] + pp[5] + pp[6] + pp[7];
#endif
#if R11_SKD
            shid[skd_off + tid] = R11_SILU(gt) * up;  // the zero stores of the other slots are before the dot's __syncthreads
#else
            __stcg(hid + ri0 + tid, R11_SILU(gt) * up);
#endif
        }
        P3_P(41);
#endif
        PB_ACC(1, 3);
#if R11_SKD
        // ---- down part: x_next[r] += sum_k W_d[r][c0 + k] * shid[k] for all 1024 rows r (this CTA's 32 columns), residual added by the owner of the row ----
        __syncthreads();  // shid complete
        skd_wait(l);      // the 4 tensor copies of layer l have landed in sD (parity l & 1)
        {
            const int c = lane & 3, rr = lane >> 2;
            const float4 h0 = *(const float4*)(shid + c * 8), h1 = *(const float4*)(shid + c * 8 + 4);
            const uint4* const sDr = (const uint4*)sD;  // row r at uint4 r * 4 (64 B); the 32 lanes of a group read 512 contiguous bytes: no bank conflict
#pragma unroll
            for (int it = 0; it < 8; ++it) {
                const int rb = warp * 64 + it * 8;
                const uint4 w = sDr[(rb + rr) * 4 + c];
                float d = bflo(w.x) * h0.x + bfhi(w.x) * h0.y + bflo(w.y) * h0.z + bfhi(w.y) * h0.w
                        + bflo(w.z) * h1.x + bfhi(w.z) * h1.y + bflo(w.w) * h1.z + bfhi(w.w) * h1.w;
                d += __shfl_xor_sync(0xffffffffu, d, 1);
                d += __shfl_xor_sync(0xffffffffu, d, 2);
                const int qb = lane & 16;  // quad 0: rows rb .. rb + 3 (lanes 0, 4, 8, 12), quad 1: rows rb + 4 .. rb + 7 (lanes 16, 20, 24, 28)
                float v0 = __shfl_sync(0xffffffffu, d, qb), v1 = __shfl_sync(0xffffffffu, d, qb + 4);
                float v2 = __shfl_sync(0xffffffffu, d, qb + 8), v3 = __shfl_sync(0xffffffffu, d, qb + 12);
#if R11_SKD_RI
                // staged: the quad holders store the 4 sums (residual not added) into sdn; thread q < 256 issues the red of quad q below. Each quad index is written by exactly one lane.
                if ((lane & 15) == 0) *(float4*)(sdn + rb + (qb >> 2)) = make_float4(v0, v1, v2, v3);
#else
                if ((lane & 15) == 0) {
                    const int r0 = rb + (qb >> 2);
                    const int o0 = r0 - rh0;
                    if ((unsigned)o0 < (unsigned)nh) v0 += sres[o0];
                    if ((unsigned)(o0 + 1) < (unsigned)nh) v1 += sres[o0 + 1];
                    if ((unsigned)(o0 + 2) < (unsigned)nh) v2 += sres[o0 + 2];
                    if ((unsigned)(o0 + 3) < (unsigned)nh) v3 += sres[o0 + 3];
                    asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(
#if R11_SKD_NX > 1
                                     r11_xp(scratch, sn, (int)blockIdx.x & (R11_SKD_NX - 1))  // down reds of CTA b -> partial b & (NX - 1) of the next x
#else
                                     xn
#endif
                                     + r0), "f"(v0), "f"(v1), "f"(v2), "f"(v3) : "memory");
                }
#endif
            }
        }
#if R11_SKD_RI
        // sdn WAR: written above (after the shid __syncthreads, in this CTA's down part of layer l); read below after this __syncthreads; the next writer is the down part of layer l + 1,
        // after this CTA's down -> QKV bar_arrive __syncthreads, which every thread passes after the reads below. The writes of sdn come after the shid __syncthreads, which follows the
        // GU dot's __syncthreads, and every thread's last sbuf read (smem_act before the dot) is before that dot sync, so no thread still reads sbuf when it is overwritten. The first writer of sbuf
        // after the reads is the next layer's QKV norm (after the next bar_wait).
        __syncthreads();
        if (tid < 256) {
            const int r0 = tid * 4, o0 = r0 - rh0;
            float4 v = *(const float4*)(sdn + r0);
            if ((unsigned)o0 < (unsigned)nh) v.x += sres[o0];
            if ((unsigned)(o0 + 1) < (unsigned)nh) v.y += sres[o0 + 1];
            if ((unsigned)(o0 + 2) < (unsigned)nh) v.z += sres[o0 + 2];
            if ((unsigned)(o0 + 3) < (unsigned)nh) v.w += sres[o0 + 3];
            asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(
#if R11_SKD_NX > 1
                             r11_xp(scratch, sn, (int)blockIdx.x & (R11_SKD_NX - 1))
#else
                             xn
#endif
                             + r0), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
        }
#endif
        // the down -> QKV bar_arrive below: its __syncthreads and red.release order the red.adds (as the O-proj)
#else
        P2_A(2);
        bar_arrive(bar_cnt);
#if R10_NSM
        if (l + 1 < NL) issue_nsm(l + 1);  // one group, before the down stage group
#endif
#if R10_DNE
        // R10_DNE: no down load and no cp.async group in the GU -> down window (the rows are a bulk copy into sD; bd is filled in the down phase)
#elif R8_LF
        gemv_load_lf<R10_PFG>(bd, nh * 12, [&](int u) { return wd + ((size_t)R10_WL(l) * H + rh0 + u / 12) * INTER + (u % 12) * 256; });
#else
        gemv_load(bd, nh * 12, [&](int u) { return wd + ((size_t)R10_WL(l) * H + rh0 + u / 12) * INTER + (u % 12) * 256; });
#endif
#if R10_PF & 8
        // x[rh0 ..] is final here (fused red.adds are before the fused->GU barrier; the down store is this CTA's own, after this barrier)
        static_assert(CDIV(H, NB) <= 32, "nh rows fit warp 1");
        float xres = 0.f;
#if R11_GRPQ
        // R11_GRPQ: the residual is x + xo (both final here: the fused red.adds into xo precede the fused -> GU barrier); the two loads are independent
        if (tid >= 32 && tid < 32 + nh) {
            const float xa = __ldcg(x + rh0 + tid - 32), xb = __ldcg(xo + rh0 + tid - 32);
            xres = xa + xb;
        }
#else
        if (tid >= 32 && tid < 32 + nh) xres = __ldcg(x + rh0 + tid - 32);
#endif
#endif
#if R9_DIRECT
        R9_LF_WAIT();
#endif
        R10_SK(r10_fold(bd));
        bar_wait<R8_SPIN_GU>(bar_cnt, bar_target);
        PB_T0();
        P2_E(3);
#if R10_QKE && R10_QKE_AT == 0
        if (l + 1 < NL) issue_qke(l + 1);  // first statement of the down phase: the GU phase of l has read sGU (see the WAR comment of R10_GUE)
#endif
        P2_ACT0();
        P2_W(2, bd);

        // ---- Phase 5: down GEMV + residual ----
#if R9_DIRECT
        // Each lane reads its 3 hid segments (warp + 4j) % 12 straight from global memory (hid is written by
        // other CTAs before the GU->down barrier; this CTA's next hid write is a layer later). No smem pass.
        float act3[3][8];
#pragma unroll
        for (int j = 0; j < 3; ++j) load_act(hid + ((warp + 4 * j) % 12) * 256 + lane * 8, act3[j]);
#if R10_DEF == 1
        P2_ACT(2, act3[0][0]);
        P3_M(42, __float_as_uint(act3[0][0]));
        def_handoff(bd);  // the stage hand-off on barrier 1, as in Phase 1
        P3_P(60);
        P3_W0(56, bd);
#elif R8_LF
        stage_to_buf(bd);
#endif
        PB_TA(2, act3[0][7] + act3[1][7] + act3[2][7]);
        PB_TW(2, bd);
        gemv_dot_reg(bd, nh * 12, act3, part);
#if R10_DEF == 1
        P3_P(45);
#endif
        __syncthreads();
#if R10_DEF == 1
        P3_P(46);
#endif
        PB_ACC(2, 2);
#else
        // hid is 768 float4: issue both loads of a thread before any store
        {
            float4 h0 = __ldcg((const float4*)hid + tid);
            float4 h1 = tid < INTER / 4 - NT ? __ldcg((const float4*)hid + NT + tid) : make_float4(0.f, 0.f, 0.f, 0.f);
#if R10_PROBE2
            P2_ACT(2, h0.x);
            P3_M(42, __float_as_uint(h0.x));
#endif
            *((float4*)sbuf + sperm4(tid)) = h0;
            if (tid < INTER / 4 - NT) *((float4*)sbuf + sperm4(NT + tid)) = h1;
        }
#if R8_LF && !R10_DNE
        asm volatile("cp.async.wait_all;" ::: "memory");  // R10_DNE: no cp.async group is outstanding here (the fused groups A / B were drained in the fused phase, the GU and down windows issue none)
#endif
        __syncthreads();
        P3_P(43);
#if R10_QKE && R10_QKE_AT == 1
        if (l + 1 < NL) issue_qke(l + 1);  // right after the __syncthreads that follows the hid smem store (hid has arrived); the sGU WAR is in the R10_GUE comment
#endif
#if R10_DNE
        d_fill(l);  // just before the dot
#elif R8_LF
        stage_to_buf(bd);
        P3_W0(56, bd);
#endif
#if R9_REGACT
        float act3[3][8];
#pragma unroll
        for (int j = 0; j < 3; ++j) smem_act(sbuf, (warp + 4 * j) % 12, act3[j]);
        P3_M(44, __float_as_uint(act3[0][7] + act3[1][7] + act3[2][7]));
#endif
        PB_TA(2, 0.f);
        PB_TW(2, bd);
#if R9_REGACT
        gemv_dot_reg(bd, nh * 12, act3, part);
#else
        gemv_dot(bd, nh * 12, [](int u) { return (u % 12) * 256; }, sbuf, part);
#endif
        P3_P(45);
        __syncthreads();
        P3_P(46);
        PB_ACC(2, 2);
#endif
#if R10_PF & 8
        if (tid >= 32 && tid < 32 + nh) {  // warp 1 holds the old x rows (loaded in the window)
            float d = 0.f;
#pragma unroll
            for (int j = 0; j < 12; ++j) d += part[(tid - 32) * 12 + j];
            __stcg(x + rh0 + tid - 32, xres + d);
#if R11_GRPQ
            // xo[r] was read (xres, above) by this same thread and, in the GU norm, by every CTA before the GU -> down barrier this CTA has passed: zeroing it now cannot race with a reader.
            // The next red.add into xo (layer l + 1, fused phase) is ordered after this store by the down -> QKV barrier (see the comment at the QKV -> fused wait, point 3). The final layer
            // zeroes it too, so xo is zero at the end of every launch; the LM head's final norm reads x (= x + xo + d), not xo.
            __stcg(xo + rh0 + tid - 32, 0.f);
#endif
        }
#else
        if (tid < nh) {
            float d = 0.f;
#pragma unroll
            for (int j = 0; j < 12; ++j) d += part[tid * 12 + j];
#if R10_O_CONST
            __stcg(x + rh0 + tid, d);
#else
            __stcg(x + rh0 + tid, __ldcg(x + rh0 + tid) + d);
#endif
        }
#endif
        P3_P(47);
        PB_ACC(2, 3);
        P2_A(3);
#endif
        bar_arrive(bar_cnt);
#if !R10_QKE
        if (l + 1 < NL) load_qkv(l + 1);
#endif
#if R10_KVE
        if (l + 1 < NL) issue_kve(l + 1);  // after the QKV stage group
#endif
#if R10_PF & 16
        if (l + 1 < NL) ld_norm16(ln1 + (l + 1) * H, wln, wln0);
#elif R10_PF & 1
        if (l + 1 < NL) wln = ld_norm2(ln1 + (l + 1) * H);
#endif
#if R10_PF & 2
        if (l + 1 < NL) { pf_row(ln1 + (l + 1) * H); pf_qkn(l + 1); }
#endif
#if R9_DIRECT
        if (l + 1 < NL) R10_PFN(ln1 + (l + 1) * H);
        R9_LF_WAIT();
#endif
        if (l + 1 < NL) R10_SK(r10_fold(bq));
        bar_wait<R8_SPIN_DOWN>(bar_cnt, bar_target);
    }


    // ---- final norm + LM head (tied embed) + argmax ----
#if R9_LMH
    // every warp covers all 1024 elements: lane holds elements it*256 + lane*8..+7; no block sync
    float la[H / 256][8];
    float lss = 0.f;
#pragma unroll
    for (int it = 0; it < H / 256; ++it) lss += norm_act(x + it * 256 + lane * 8, fnorm + it * 256 + lane * 8, la[it]);
    const float lrs = rsqrtf(warp_sum(lss) / (float)H + EPS);
#else
#if R11_SKD_NX > 1
    rms_to_smem_p(scratch, 0, fnorm, sbuf, red);  // final x = side NL & 1 = 0, the sum of its NX partials
#else
    rms_to_smem(x, fnorm, sbuf, red);
#endif
#endif
    float best = -INFINITY;
    int besti = 0x7fffffff;
#if R10_LMH >= 2
#if R9_LMH
#error R10_LMH needs R9_LMH 0 (activations from smem)
#endif
#if R10_LMH_CONSEC && R10_LMH_DYN
#error "R10_LMH_CONSEC needs R10_LMH_DYN 0"
#endif
#if R10_LMH_DYN
    // static rows [0, VS) with the stride, then chunks of R10_LMH_CHUNK rows from the counter sync[4]
    const int VS = ((R10_LMH_STATIC_FRAC_PCT * R10_LM_END / 100) / (nwarps * R10_LMH)) * (nwarps * R10_LMH);
    for (int r = gwarp; r < VS; r += nwarps * R10_LMH) lm_group<R10_LMH>(embed, r, nwarps, VS, sbuf, logits, best, besti);
    for (;;) {
        unsigned c = 0;
        if (lane == 0) c = atomicAdd(sync + 4, 1u);
        c = __shfl_sync(0xffffffffu, c, 0);
        const int base = VS + (int)c * R10_LMH_CHUNK;  // chunk ids of a warp increase: its rows ascend
        if (base >= R10_LM_END) break;
        const int end = min(base + R10_LMH_CHUNK, R10_LM_END);
        for (int r = base; r < end; r += R10_LMH) lm_group<R10_LMH>(embed, r, 1, end, sbuf, logits, best, besti);
    }
#elif R10_LMH_CONSEC
    // a warp's R10_LMH rows are consecutive; its groups are nwarps * R10_LMH rows apart (rows of a warp still ascend)
    for (int r = gwarp * R10_LMH; r < R10_LM_END; r += nwarps * R10_LMH)
        lm_group<R10_LMH>(embed, r, 1, R10_LM_END, sbuf, logits, best, besti);
#else
    for (int r = gwarp; r < R10_LM_END; r += nwarps * R10_LMH)
        lm_group<R10_LMH>(embed, r, nwarps, R10_LM_END, sbuf, logits, best, besti);
#endif
#else
    for (int r = gwarp; r < R10_LM_END; r += nwarps) {
#if R9_LMH
        float d = lrs * row_dot_reg<H>(embed + (size_t)r * H, la);
#else
        float d = row_dot<H>(embed + (size_t)r * H, sbuf);
#endif
        if (lane == 0) {
            logits[r] = d;
            if (d > best) { best = d; besti = r; }  // rows ascend per warp
        }
    }
#endif
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
#if R8_ARGMAX
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
#if R11_GRPQ
            // the group counters: every fused CTA's last poll (layer NL - 1, QKV phase) is before its later barrier arrivals, which precede its ticket; every red.release on them is
            // before the same CTA's later bar_arrive release, which the polls of the other CTAs acquired: so all increments and polls precede this ticket holder's zeroing
#pragma unroll
            for (int g = 0; g < KVH; ++g) atomicExch(sync + R11_GC(g), 0u);
#endif
#if R10_LMH_DYN
            atomicExch(sync + 4, 0u);  // every warp has finished its rows
#endif
        }
#else
        __threadfence();
        if (tid == 0) {
            float bv = __ldcg(pval); int bi = __ldcg(pidx);
            for (int b = 1; b < (int)gridDim.x; ++b) {
                float v = __ldcg(pval + b); int i = __ldcg(pidx + b);
                if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
            }
            next_token[0] = (long long)bi;
            atomicExch(ticket, 0u);
            atomicExch(bar_cnt, 0u);  // every block has passed its last barrier poll
#if R11_GRPQ
            // the group counters: every fused CTA's last poll (layer NL - 1, QKV phase) is before its later barrier arrivals, which precede its ticket; every red.release on them is
            // before the same CTA's later bar_arrive release, which the polls of the other CTAs acquired: so all increments and polls precede this ticket holder's zeroing
#pragma unroll
            for (int g = 0; g < KVH; ++g) atomicExch(sync + R11_GC(g), 0u);
#endif
#if R10_LMH_DYN
            atomicExch(sync + 4, 0u);  // every warp has finished its rows
#endif
        }
#endif
    }
#if R9_PROBE
    if (tid == 32)
        for (int i = 0; i < 13; ++i) atomicAdd(&r9_probe[blockIdx.x * 16 + i], probe_s[i]);
#endif
#if R11_SKEW
    __syncthreads();  // all arrivals of this CTA are in smem (thread 0 wrote them); one dump per launch
    for (int i = tid; i < R11_NBAR; i += NT) r11_skew[blockIdx.x * R11_NBAR + i] = (i < (int)r11_sk_n) ? r11_sk_s[i] : 0ull;
#endif
#if R10_PROBE2
    if (tid == 32)
        for (int i = 0; i < 64; ++i) atomicAdd(&r10_probe[blockIdx.x * 64 + i], p2_s[i]);
#endif
}
"""

_INPUT_ORDER = ("token", "ln1", "wq", "wk", "wv", "qn", "kn", "wo", "ln2",
                "wg", "wu", "wd", "fnorm", "embed", "kcache", "vcache")


def _dyn_smem() -> int:
    """Dynamic smem the default build needs: R8_STAGE_BYTES when its default R8_LF is 1, else 0; plus sWo + sK (64 KB) when R10_KVE or R10_KVM defaults to 1, plus 4608 B (ln1, ln2, qn, kn rows) when R10_NSM defaults to 1, plus 86016 B (sGU: 2 * 21 rows of the gate / up weights) when R10_GUE defaults to 1, plus 43008 B (sD: 7 rows of the down weights) when R10_DNE defaults to 1."""
    import re
    m = re.search(r"#ifndef R8_LF\s*\n#define R8_LF (\d)", KERNEL_SRC)
    k = re.search(r"#ifndef R10_KVE\s*\n#define R10_KVE (\d)", KERNEL_SRC)
    n = re.search(r"#ifndef R10_NSM\s*\n#define R10_NSM (\d)", KERNEL_SRC)
    g = re.search(r"#ifndef R10_GUE\s*\n#define R10_GUE (\d)", KERNEL_SRC)
    d = re.search(r"#ifndef R10_DNE\s*\n#define R10_DNE (\d)", KERNEL_SRC)
    km = re.search(r"#ifndef R10_KVM\s*\n#define R10_KVM (\d)", KERNEL_SRC)
    sk = re.search(r"#ifndef R11_SKD\s*\n#define R11_SKD (\d)", KERNEL_SRC)
    return ((5632 if m and m.group(1) == "1" else 0) + (65536 if (k and k.group(1) == "1") or (km and km.group(1) == "1") else 0) + (4608 if n and n.group(1) == "1" else 0)
            + (86016 if g and g.group(1) == "1" else 0) + (((65536 if sk and sk.group(1) == "1" else 43008)) if d and d.group(1) == "1" else 0))


def _skd_nx() -> int:
    """R11_SKD_NX default of the compiled source (1 when absent)."""
    import re
    m = re.search(r"#ifndef R11_SKD_NX\s*\n#define R11_SKD_NX (\d)", KERNEL_SRC)
    return int(m.group(1)) if m else 1


def _scratch_floats() -> int:
    """Scratch floats the default build needs: 1024 + 4096 + 3072 (x, qkv, hid) plus 2 * (NX - 1) * 1024 for the extra x partials of R11_SKD_NX."""
    return 1024 + 4096 + 3072 + (2 * (_skd_nx() - 1) * 1024 if _skd_on() else 0) + (1024 if _grpq_on() else 0)


def _grpq_on() -> bool:
    """True when the compiled source has R11_GRPQ 1 by default: xo (1024 floats, S_XO = S_END) is added to the scratch."""
    import re
    m = re.search(r"#ifndef R11_GRPQ\s*\n#define R11_GRPQ (\d)", KERNEL_SRC)
    return bool(m and m.group(1) == "1")


_SYNC_INTS = 128  # sync words: [0] bar_cnt, [2] ticket, [4] LM-head chunk counter, [16 + 8 g] R11_GRPQ group counters (g < 8); older kernels use the first 16


def _skd_on() -> bool:
    """True when the compiled source has R11_SKD 1 by default (parsed like _dyn_smem)."""
    import re
    m = re.search(r"#ifndef R11_SKD\s*\n#define R11_SKD (\d)", KERNEL_SRC)
    return bool(m and m.group(1) == "1")


def _encode_wd_map(wd_ptr: int, driver=None) -> bytes:
    """The 128-byte tensor map of wd ([NL][H][INTER] bf16, row-major): 3D dims {INTER, H, NL}, byte strides {INTER * 2, H * INTER * 2},
    box {32, 256, 1}, element strides 1, bf16, no interleave, no swizzle, L2 promotion 128 B, no OOB fill (zeros). `driver` is cuda.bindings.driver (injectable for tests)."""
    import struct
    if driver is None:
        from cuda.bindings import driver
    res = driver.cuTensorMapEncodeTiled(
        driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, int(wd_ptr),
        [driver.cuuint64_t(v) for v in (3072, 1024, 28)], [driver.cuuint64_t(v) for v in (3072 * 2, 1024 * 3072 * 2)],
        [driver.cuuint32_t(v) for v in (32, 256, 1)], [driver.cuuint32_t(v) for v in (1, 1, 1)],
        driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
        driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B, driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    if int(res[0]) != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled: {res[0]}")
    return struct.pack("<16Q", *[int(v) for v in res[1].opaque])


def _write_wd_map(module, wd_ptr: int, stream, cache=None, driver=None):
    """Write r11_wd_map of `module` on `stream` (ordered before the next launch on it) when wd_ptr differs from the cached one. `cache` is a dict that keeps the
    pinned host buffer alive and the last pointer ({} on first use). Returns the cache."""
    if driver is None:
        from cuda.bindings import driver
    cache = {} if cache is None else cache
    if cache.get("ptr") == int(wd_ptr):
        return cache
    buf = ctypes.create_string_buffer(_encode_wd_map(wd_ptr, driver), 128)
    dptr, size = _check(driver.cuModuleGetGlobal(module, b"r11_wd_map"))
    if int(size) != 128:
        raise RuntimeError(f"r11_wd_map is {size} B, expected 128")
    _check(driver.cuMemcpyHtoDAsync(dptr, ctypes.addressof(buf), 128, stream))
    cache["buf"] = buf  # the async copy reads it later: keep it alive (a new write replaces it only after a later launch is queued on the same stream)
    cache["ptr"] = int(wd_ptr)
    return cache


def _check(res):
    err = res[0]
    if int(err) != 0:
        raise RuntimeError(f"CUDA error: {err}")
    return res[1:] if len(res) > 2 else (res[1] if len(res) == 2 else None)


def _compile(device: torch.device, n_blocks: int) -> bytes:
    from cuda.bindings import nvrtc

    major, minor = torch.cuda.get_device_capability(device)
    err, prog = nvrtc.nvrtcCreateProgram(KERNEL_SRC.encode(), b"qwen3_step.cu", 0, [], [])
    if int(err) != 0:
        raise RuntimeError(f"nvrtcCreateProgram: {err}")
    opts = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"-std=c++17",
            b"--fmad=true", b"-default-device", f"-DNB={n_blocks}".encode()]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if int(err) != 0:
        _, n = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * n
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError("NVRTC compile failed:\n" + log.decode(errors="replace"))
    _, n = nvrtc.nvrtcGetCUBINSize(prog)
    cubin = b" " * n
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    nvrtc.nvrtcDestroyProgram(prog)
    return cubin


def build(case: dict):
    from cuda.bindings import driver

    p = case["params"]
    assert (p["batch"], p["context"], p["layers"], p["hidden"], p["q_heads"],
            p["kv_heads"], p["head_dim"], p["intermediate"], p["vocab"]) == (
        1, 128, 28, 1024, 16, 8, 128, 3072, 151936), "kernel is specialized to this case"

    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.init()
    torch.zeros(1, device=device)  # make torch's primary context current
    n_sm = torch.cuda.get_device_properties(device).multi_processor_count
    grid = n_sm  # one 512-thread block per SM; co-residency guaranteed by cooperative launch
    cubin = _compile(device, grid)
    module = _check(driver.cuModuleLoadData(cubin))
    func = _check(driver.cuModuleGetFunction(module, b"qwen3_step"))
    dyn = _dyn_smem()
    if dyn:
        _check(driver.cuFuncSetAttribute(
            func, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dyn))

    nblocks_per_sm = _check(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(func, 512, dyn))
    if nblocks_per_sm < 1:
        raise RuntimeError("kernel cannot be resident")
    scratch = torch.zeros(_scratch_floats(),
                          dtype=torch.float32, device=device)
    sync = torch.zeros(_SYNC_INTS, dtype=torch.int32, device=device)
    # RoPE cos/sin at position 128 (theta 1e6, head_dim 128), fp64 -> fp32
    inv = 1e6 ** (-torch.arange(0, 128, 2, dtype=torch.float64) / 128)
    ang = 128.0 * inv
    rope = torch.cat([torch.cos(ang), torch.sin(ang)]).float().to(device)
    pval = torch.zeros(grid, dtype=torch.float32, device=device)
    pidx = torch.zeros(grid, dtype=torch.int32, device=device)
    torch.cuda.synchronize(device)

    n_args = len(_INPUT_ORDER) + 4 + 5
    arg_vals = (ctypes.c_void_p * n_args)()
    arg_ptrs = (ctypes.c_void_p * n_args)(
        *[ctypes.addressof(arg_vals) + i * ctypes.sizeof(ctypes.c_void_p) for i in range(n_args)])
    tail = [scratch.data_ptr(), sync.data_ptr(), pval.data_ptr(), pidx.data_ptr(), rope.data_ptr()]
    state = {"module": module, "keep": (scratch, sync, pval, pidx, rope, arg_vals, arg_ptrs)}
    skd = _skd_on()
    wd_cache: dict = {}

    # Host V2: the 4 output tensors of the NEXT call are allocated right after the launch (hidden behind the
    # running kernel); a call uses the tensors made at the end of the previous call (first call: allocates them).
    # Each call still returns fresh tensors that no later call writes.
    nxt = [None]

    def _alloc():
        return (torch.empty(151936, dtype=torch.float32, device=device),
                torch.empty((), dtype=torch.int64, device=device),
                torch.empty((28, 8, 128), dtype=torch.bfloat16, device=device),
                torch.empty((28, 8, 128), dtype=torch.bfloat16, device=device))

    def run(inputs: dict) -> dict:
        logits, next_token, k_write, v_write = nxt[0] if nxt[0] is not None else _alloc()
        ptrs = [inputs[name].data_ptr() for name in _INPUT_ORDER]
        ptrs += [logits.data_ptr(), next_token.data_ptr(), k_write.data_ptr(), v_write.data_ptr()]
        ptrs += tail
        for i, v in enumerate(ptrs):
            arg_vals[i] = v
        stream = torch.cuda.current_stream(device).cuda_stream
        if skd:  # R11_SKD: the wd tensor map (cached on the wd pointer), ordered before the launch on the same stream
            _write_wd_map(module, inputs["wd"].data_ptr(), stream, wd_cache)
        _check(driver.cuLaunchCooperativeKernel(
            func, grid, 1, 1, 512, 1, 1, dyn, stream, ctypes.addressof(arg_ptrs)))
        nxt[0] = _alloc()  # Host V2: next call's outputs, allocated while the kernel runs
        return {"logits": logits, "next_token": next_token,
                "k_write": k_write, "v_write": v_write}

    run._state = state  # keep module and buffers alive
    return run
