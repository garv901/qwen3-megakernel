"""Speed-of-light (SOL) model per Workload: HBM-bytes and tensor-core-FLOP floors.

Optimistic useful-work bound, not a latency prediction. Bytes count each weight
once (tied embedding / LM head counted ONCE, 151936x1024 bf16 = 311,164,928 B,
as upstream megabench ``sol.py`` does), the KV prefix read, the new K/V write,
token ids, fp32 logits out and norm weights. Scratch traffic, synchronisation
and instruction cost are omitted. For R0 (decode b1 kv128) the bytes equal the
upstream megabench floor 1,207,502,352 B exactly (150.94 us at 8 TB/s).

FLOPs: 2 * (linear params per layer * L) * total_q
     + 4 * L * QH * D * sum over new tokens of attended positions
     + 2 * V * H * n_seq (LM head on the last token of each sequence).
Token j of a sequence with kv_len K attends K + j + 1 positions.

T_bw = bytes / BW, T_fl = flops / PEAK, T_sol = max(T_bw, T_fl).
"""

from __future__ import annotations

from .workload import Workload

BF16, FP32, INT64 = 2, 4, 8

# Spec constants. TODO(main agent): verify both against the B200 datasheet / probes.
SPEC_BW = 8.0e12          # bytes/s, B200 HBM3e (NVIDIA DGX B200: 64 TB/s / 8 GPUs)
SPEC_BW_SOURCE = "https://www.nvidia.com/en-eu/data-center/dgx-b200/"
# DGX B200 datasheet: 72 PFLOPS FP8 (sparse) / 8 GPUs = 9 PF; BF16 dense = FP8 sparse / 4 = 2.25 PF.
SPEC_PEAK = 2.25e15       # FLOP/s, BF16 dense tensor core
SPEC_PEAK_SOURCE = "NVIDIA DGX B200 datasheet: 72 PFLOPS FP8 sparse / 8 GPUs, /4 -> BF16 dense (unverified)"
FP32_SMS, FP32_LANES = 148, 128


def fp32_ffma_peak(clock_hz: float) -> float:
    """FP32 FFMA peak FLOP/s: 148 SMs x 128 lanes x 2 FLOP x clock (record the clock)."""
    return FP32_SMS * FP32_LANES * 2 * clock_hz


def byte_breakdown(w: Workload) -> dict[str, int]:
    g = w.model
    layers, h, d = g.layers, g.hidden, g.head_dim
    qd, kd, inter = g.q_dim, g.kv_dim, g.intermediate
    proj = layers * (qd * h + 2 * kd * h + h * qd)
    mlp = layers * 3 * inter * h
    norms = layers * (2 * h + 2 * d) + h
    return {
        "linear_weights": (proj + mlp) * BF16,
        "norm_weights": norms * BF16,
        "embed_lm_head_once": g.vocab * h * BF16,
        "kv_read": g.kv_bytes_per_token * w.total_kv,
        "kv_write": g.kv_bytes_per_token * w.total_q,
        "token_io": INT64 * (w.total_q + w.n_seq),
        "logits_out": w.n_seq * g.vocab * FP32,
    }


def linear_params_per_layer(w: Workload) -> int:
    g = w.model
    return (g.q_dim * g.hidden + 2 * g.kv_dim * g.hidden + g.hidden * g.q_dim
            + 3 * g.intermediate * g.hidden)


def flop_breakdown(w: Workload) -> dict[str, int]:
    g = w.model
    attended = sum(s.q_len * s.kv_len + s.q_len * (s.q_len + 1) // 2 for s in w.seqs)
    return {
        "linear": 2 * linear_params_per_layer(w) * g.layers * w.total_q,
        "attention": 4 * g.layers * g.q_heads * g.head_dim * attended,
        "lm_head": 2 * g.vocab * g.hidden * w.n_seq,
    }


def sol(w: Workload, bw: float = SPEC_BW, peak: float = SPEC_PEAK) -> dict:
    """SOL for one workload at bandwidth ``bw`` (B/s) and tensor peak ``peak`` (FLOP/s)."""
    b, f = byte_breakdown(w), flop_breakdown(w)
    nbytes, nflops = sum(b.values()), sum(f.values())
    t_bw, t_fl = nbytes / bw, nflops / peak
    # Tokens per step at which T_fl == T_bw, scaling only the linear FLOPs with total_q
    # (bytes, attention and LM-head FLOPs held at this point's values).
    per_token = 2 * linear_params_per_layer(w) * w.model.layers
    ridge = max(0.0, (t_bw * peak - f["attention"] - f["lm_head"]) / per_token)
    return {"bytes": nbytes, "bytes_breakdown": b, "flops": nflops, "flops_breakdown": f,
            "bw": bw, "peak": peak, "t_bw_us": t_bw * 1e6, "t_fl_us": t_fl * 1e6,
            "t_sol_us": max(t_bw, t_fl) * 1e6,
            "bound": "memory" if t_bw >= t_fl else "compute",
            "ridge_tokens": ridge}


def sol_both(w: Workload, measured_bw: float | None = None,
             measured_peak: float | None = None) -> dict:
    """``{"spec": sol(...), "measured": sol(...) or None}`` (measured needs both probes)."""
    out = {"spec": sol(w), "measured": None}
    if measured_bw and measured_peak:
        out["measured"] = sol(w, measured_bw, measured_peak)
    return out
