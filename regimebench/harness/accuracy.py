"""Accuracy metrics and grading against the FP32 oracle band.

Rule: an arm passes a workload when, for each of logits / k_new / v_new, its
rel-L2 to the FP32 reference is <= NOISE_FACTOR (2x) the BF16 reference's own
rel-L2 to the FP32 reference ("oracle band"). ``next_token`` agreement with the
FP32 reference is recorded (rate over sequences), not gated.
"""

from __future__ import annotations

from ..arms.base import OUTPUT_NAMES

NOISE_FACTOR = 2.0
BAND_KEYS = ("logits", "k_new", "v_new")
BAND_FLOOR = 1e-12


def _err(got, ref) -> dict:
    g, r = got.double().flatten(), ref.double().flatten()
    diff = (g - r).norm().item()
    rn = r.norm().item()
    return {"max_abs": float((g - r).abs().max().item()) if g.numel() else 0.0,
            "rel_l2": diff / rn if rn > 0 else float(diff),
            "cosine": float((g @ r).item() / max(g.norm().item() * rn, 1e-300))}


def compare(got: dict, ref: dict) -> dict:
    """Per-output metrics of ``got`` against ``ref`` (float64 on CPU)."""
    for key in OUTPUT_NAMES:
        if key not in got or key not in ref:
            raise ValueError(f"missing output {key!r}")
    out: dict = {}
    for key in BAND_KEYS:
        g, r = got[key].detach().float().cpu(), ref[key].detach().float().cpu()
        if tuple(g.shape) != tuple(r.shape):
            raise ValueError(f"{key}: shape {tuple(g.shape)} vs reference {tuple(r.shape)}")
        out[key] = _err(g, r)
    gl, rl = got["logits"].detach().float().cpu(), ref["logits"].detach().float().cpu()
    out["logits"]["argmax_match_rate"] = float(
        (gl.argmax(-1) == rl.argmax(-1)).float().mean().item())
    gt, rt = got["next_token"].detach().cpu().long(), ref["next_token"].detach().cpu().long()
    if tuple(gt.shape) != tuple(rt.shape):
        raise ValueError(f"next_token: shape {tuple(gt.shape)} vs reference {tuple(rt.shape)}")
    out["next_token"] = {"match_rate": float((gt == rt).float().mean().item()),
                         "match_all": bool((gt == rt).all().item())}
    return out


def oracle_band(ref_bf16: dict, ref_fp32: dict) -> dict:
    """Rel-L2 of the BF16 reference against the FP32 reference, per output."""
    m = compare(ref_bf16, ref_fp32)
    return {k: m[k]["rel_l2"] for k in BAND_KEYS}


def grade_vs_oracle(metrics_fp32: dict, band: dict) -> dict:
    """Verdict dict: ``verdict`` pass/fail and per-output ratio to the band."""
    ratios = {k: metrics_fp32[k]["rel_l2"] / max(band[k], BAND_FLOOR) for k in BAND_KEYS}
    worst = max(ratios.values())
    return {"verdict": "pass" if worst <= NOISE_FACTOR else "fail",
            "ratios": ratios, "worst_ratio": worst, "noise_factor": NOISE_FACTOR}


def snapshot_inputs(inputs: dict) -> dict:
    import torch
    return {k: v.detach().cpu().clone() for k, v in inputs.items()
            if isinstance(v, torch.Tensor)}


def mutated_inputs(inputs: dict, snapshot: dict) -> list[str]:
    import torch
    return [k for k, v in snapshot.items()
            if k not in inputs or not torch.equal(inputs[k].detach().cpu(), v)]
