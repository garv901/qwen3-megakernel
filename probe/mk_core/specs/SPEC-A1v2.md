# SPEC-A1v2: attention interface v2 (MK-core P2 <-> MK-attn A1)

Goal: remove mk's phase 2 (q/k-norm + RoPE + KV write) and its grid barrier (~2 us/layer, V0.md). A1 v2 does that work.
v1 (post-RoPE bf16 q `qb`) stays behind the flag; mk selects v2 with `-DSKV_IFACE=2` (A1 header defines `SKV_IFACE_MAX`
>= 2 when it supports it; mk static_asserts it).

## Inputs at the start of the attention phase (after the QKV barrier; all written by mk's QKV phase, final)
All pointers global, layer `l` given. `ntok` tokens, token t in [0, ntok).
| name | type / layout | meaning |
|---|---|---|
| qkv | fp32 [MAXTOK = 256][4096] row-major, row t = token t | raw x * ln1 GEMM sums, NOT rms-scaled. cols 0..2047 = q (head h at h*128), 2048..3071 = k (kv head g at 2048 + g*128), 3072..4095 = v (3072 + g*128) |
| rr1 | fp32 [MAXTOK] | per-token rms scale: q, k and v are rr1[t] * qkv[t][...] |
| qn, kn | bf16 [28][128] | q-norm / k-norm weights of layer l at l*128 |
| rope | fp32 [max_pos][128] | row p = cos(p f_j) j = 0..63, then sin(p f_j) j = 0..63 (hv2g rope_table, theta 1e6) |
| t_base, t_vis | int32 [ntok] | cache row base of t's sequence; t's position pos = t_vis[t] - 1 |
| kcache, vcache | bf16 [(28 * cache_rows + CTX)][8][128] | row index l * cache_rows + t_base[t] + pos |
| eps | 1e-6 (EPS in mk_common.cuh) | |

## Exact math (v0 phase 2, mk_step.cu @2839ced lines ~315-345; must stay bit-identical to keep the A/B clean)
Per (t, head) with 128 dims e_d, d = 0..127 (q head h: qkv[t][h*128 + d]; k head g: qkv[t][2048 + g*128 + d]):
1. e_d = qkv * rr1[t]
2. rn = rsqrtf(sum_d e_d^2 / 128 + eps); e_d = e_d * rn * w_d (w = qn or kn of layer l, bf16 -> fp32)
3. RoPE (rotate-half): j = d & 63; partner p_d = e_{d ^ 64}; o_d = e_d * cos_j + (d < 64 ? -p_d : p_d) * sin_j, cos/sin from rope row pos
4. q: use o (bf16-rounded, as v1's qb) for scores. k: write bf16(o) to kcache row; v: write bf16(qkv[t][3072 + g*128 + d] * rr1[t]) to vcache row.
v0 does the norm sum as a warp_sum over 4 dims per lane (lane L holds d = 4L..4L+3); any summation order is acceptable
for v2 (the A/B is graded against the band, not bit-exactness), but say which you use.

## Ownership (MK-attn's planner rule, agreed)
All new-token rows of a (seq, kv head) are in ONE piece (the last split). That item computes them, uses them from smem, and is
the only writer of those cache rows. No other split reads them in this phase.

## Outputs (unchanged from v1)
`skv_combine_head(layer, head, xs, ntok)` into the O-proj B slot (LEDGER 2026-10-23), global ao bf16 [total_q, 2048] fallback.
`outputs()` of the arm reads k_new / v_new from the cache rows above: they must be written by the end of the step.

## mk side
Phase 2 code and its barrier compile out when SKV_IFACE == 2; `qb` is not written. QKV epilogue / rr1 unchanged.
Barrier count per layer: 6 -> 5.
