"""Accuracy metrics, grading, and reference helpers for the SOTA harness."""

from __future__ import annotations

# FP32 arms must match the FP32 reference tightly (calibrated on the first GPU
# smoke). BF16 arms are graded by the oracle-band rule: their distance from the
# FP32 reference may be at most NOISE_FACTOR times the BF16 MegaBench
# reference's own distance from it. On the b1-s128 fixture that band is ~5%
# rel-L2, so independent BF16 stacks differ from each other by about as much as
# BF16 differs from FP32.
PASS_REL_L2 = 1e-3
WARN_REL_L2 = 6e-2
KV_PASS_REL_L2 = 1e-2
NOISE_FACTOR = 2.0

OUTPUT_KEYS = ("logits", "next_token", "k_write", "v_write")


def _match_shapes(name: str, got, ref):
    while got.ndim > ref.ndim and got.shape[0] == 1:
        got = got.squeeze(0)
    while ref.ndim > got.ndim and ref.shape[0] == 1:
        ref = ref.squeeze(0)
    if tuple(got.shape) != tuple(ref.shape):
        raise ValueError(f"{name}: shape {tuple(got.shape)} vs reference {tuple(ref.shape)}")
    return got, ref


def _err(got, ref) -> dict:
    g, r = got.double().flatten(), ref.double().flatten()
    diff = (g - r).norm().item()
    rn = r.norm().item()
    return {"max_abs": float((g - r).abs().max().item()),
            "rel_l2": diff / rn if rn > 0 else float(diff),
            "cosine": float((g @ r).item() / max(g.norm().item() * rn, 1e-300))}


def compare(got: dict, ref: dict) -> dict:
    """Per-output metrics of ``got`` against ``ref`` (everything compared in float64)."""
    import torch
    for key in OUTPUT_KEYS:
        if key not in got or key not in ref:
            raise ValueError(f"missing output {key!r}")
    out: dict = {}
    g, r = _match_shapes("logits", got["logits"].detach().float().cpu(),
                         ref["logits"].detach().float().cpu())
    m = _err(g, r)
    gv, rv = g.flatten(), r.flatten()
    k = min(5, rv.numel())
    m["argmax_match"] = bool(gv.argmax().item() == rv.argmax().item())
    m["top5_overlap"] = len(set(gv.topk(k).indices.tolist()) &
                            set(rv.topk(k).indices.tolist())) / 5
    out["logits"] = m
    g, r = _match_shapes("next_token", got["next_token"].detach().cpu(),
                         ref["next_token"].detach().cpu())
    out["next_token"] = {"match": bool(torch.equal(g.long(), r.long()))}
    for name in ("k_write", "v_write"):
        g, r = _match_shapes(name, got[name].detach().float().cpu(),
                             ref[name].detach().float().cpu())
        m = _err(g, r)
        out[name] = {"max_abs": m["max_abs"], "rel_l2": m["rel_l2"]}
    return out


def grade(metrics_own: dict) -> str:
    """"pass", "warn" or "fail" from metrics against the step's own-precision reference."""
    rel = metrics_own["logits"]["rel_l2"]
    kv = max(metrics_own["k_write"]["rel_l2"], metrics_own["v_write"]["rel_l2"])
    if rel <= PASS_REL_L2 and metrics_own["next_token"]["match"] and kv <= KV_PASS_REL_L2:
        return "pass"
    if rel <= WARN_REL_L2:
        return "warn"
    return "fail"


def oracle_band(ref_bf16: dict, ref_fp32: dict) -> dict:
    """Rel-L2 of the BF16 reference against the FP32 reference, per output."""
    m = compare(ref_bf16, ref_fp32)
    return {k: m[k]["rel_l2"] for k in ("logits", "k_write", "v_write")}


def grade_vs_oracle(metrics_fp32: dict, precision: str, band: dict) -> str:
    """Grade against the FP32 reference: tight for FP32 arms, banded for BF16 arms."""
    if precision == "fp32":
        return grade(metrics_fp32)
    worst = max(metrics_fp32[k]["rel_l2"] / max(band[k], 1e-12)
                for k in ("logits", "k_write", "v_write"))
    if worst <= NOISE_FACTOR:
        return "pass"
    return "warn" if worst <= 2 * NOISE_FACTOR else "fail"


def fp32_reference(shape, inputs: dict, device: str) -> dict:
    """FP32 eager reference via ``qwen3_torch`` (raises RuntimeError if unavailable)."""
    import torch
    from ..tasks.dense import QWEN3_WEIGHT_NAMES
    try:
        from . import qwen3_torch
        w32 = {n: inputs[n].float() for n in QWEN3_WEIGHT_NAMES}
        state = qwen3_torch.build_state(shape, inputs, torch.float32, device)
        with torch.no_grad():
            out = qwen3_torch.decode_step(w32, state, torch.float32)
    except (NotImplementedError, ImportError, AttributeError) as exc:
        raise RuntimeError(f"fp32 reference unavailable: {exc!r}") from exc
    return {k: (v.squeeze(0) if shape.batch == 1 and v.ndim and v.shape[0] == 1 else v)
            for k, v in out.items()}


def megabench_verdict(expected_bf16: dict, got: dict, shape, device: str) -> dict:
    """MegaBench's own tolerance check against the BF16 reference (informational only)."""
    from ..harness.correctness import _compare
    from .shapes import megabench_case
    try:
        detail = _compare(expected_bf16, got, megabench_case(shape), device)
        return {"pass": True, "detail": detail}
    except AssertionError as exc:
        return {"pass": False, "error": str(exc)[:500]}
    except Exception as exc:
        return {"pass": False, "error": f"{type(exc).__name__}: {exc}"[:500]}


def snapshot_inputs(inputs: dict) -> dict:
    import torch
    return {k: v.detach().cpu().clone() for k, v in inputs.items()
            if isinstance(v, torch.Tensor)}


def mutated_inputs(inputs: dict, snapshot: dict) -> list[str]:
    import torch
    return [k for k, v in snapshot.items()
            if k not in inputs or not torch.equal(inputs[k].detach().cpu(), v)]
