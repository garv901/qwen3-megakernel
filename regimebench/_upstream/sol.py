"""Optimistic one-read HBM traffic floors for ready MegaBench model steps.

This is a useful-byte bound, not a prediction of achievable kernel latency.
It counts each needed weight once, the old KV values needed by the step, and
required outputs. It omits scratch traffic, synchronization, and instruction
cost. P0 model weights are larger than cache, so weight streaming is a useful
first-order bound for those cases.
"""

from __future__ import annotations

import math

from .cases import Case


DEFAULT_B200_BANDWIDTH_TB_S = 8.0
DEFAULT_B200_BANDWIDTH_SOURCE = "https://www.nvidia.com/en-eu/data-center/dgx-b200/"
BF16_BYTES = 2
FP32_BYTES = 4
INT64_BYTES = 8


def estimate_case(case: Case, bandwidth_tb_s: float = DEFAULT_B200_BANDWIDTH_TB_S,
                  measured_ms: float | None = None) -> dict:
    """Return a transparent HBM floor for one ready, batch-one case."""
    if not math.isfinite(bandwidth_tb_s) or bandwidth_tb_s <= 0:
        raise ValueError("bandwidth_tb_s must be finite and positive")
    if measured_ms is not None and (not math.isfinite(measured_ms) or measured_ms <= 0):
        raise ValueError("measured_ms must be finite and positive")
    result: dict = {"case_id": case.id, "status": "estimated"}
    if not case.ready:
        return result | {"status": "unavailable", "reason": case.note}
    if case.family not in {"dense_step", "moe_step", "quant_step",
                           "spec_target_step"}:
        return result | {"status": "unavailable",
                         "reason": f"no traffic model for {case.family}"}
    p = case.params
    if p["batch"] != 1:
        return result | {"status": "unavailable",
                         "reason": "traffic model currently supports batch one"}

    layers, hidden, head_dim = p["layers"], p["hidden"], p["head_dim"]
    qdim, kdim = p["q_heads"] * head_dim, p["kv_heads"] * head_dim
    inter, vocab, context = p["intermediate"], p["vocab"], p["context"]
    steps = p["draft_depth"] + 1 if case.family == "spec_target_step" else 1
    projection_elements = layers * (
        qdim * hidden + 2 * kdim * hidden + hidden * qdim)
    mlp_elements = layers * 3 * inter * hidden
    components: list[dict] = []

    def add(name: str, bytes_: int, detail: str) -> None:
        components.append({"name": name, "bytes": bytes_, "detail": detail})

    if case.family == "quant_step":
        bits = p["bits"]
        if bits not in (4, 8):
            return result | {"status": "unavailable",
                             "reason": f"unsupported quantization width {bits}"}
        weight_elements = projection_elements + mlp_elements
        if weight_elements * bits % 8:
            raise ValueError(f"packed weights do not contain whole bytes: {case.id}")
        add("quantized_projection_and_mlp_weights", weight_elements * bits // 8,
            f"all seven projection stacks, packed {bits}-bit weights")
        output_rows = layers * (qdim + 2 * kdim + 2 * hidden + 2 * inter)
        add("quantization_scales", output_rows * FP32_BYTES,
            "one FP32 scale per projection output row")
        add("normalization_weights",
            (layers * (4 * hidden + 2 * head_dim) + hidden) * BF16_BYTES,
            "four layer norms, Q/K norms, and final norm in BF16")
        add("tied_embedding_and_head", vocab * hidden * BF16_BYTES,
            "full BF16 vocabulary head, including the token embedding row")
        local_layers = layers - layers // 6
        local_tokens = min(context, max(0, p["local_window"] - 1))
        cache_positions = local_layers * local_tokens + (layers // 6) * context
    else:
        add("attention_projection_weights", projection_elements * BF16_BYTES,
            "Q/K/V/output projection weights, read once")
        cache_positions = layers * context
        if case.family == "moe_step":
            experts, topk = p["experts"], p["topk"]
            if topk > experts:
                raise ValueError(f"topk exceeds expert count: {case.id}")
            add("router_weights", layers * experts * hidden * BF16_BYTES,
                "all router rows in each layer")
            add("active_expert_weights", mlp_elements * topk * BF16_BYTES,
                f"only the {topk} selected experts per layer")
            add("normalization_weights",
                (layers * (2 * hidden + 2 * head_dim) + hidden) * BF16_BYTES,
                "two layer norms, Q/K norms, and final norm in BF16")
            add("input_embedding_row", hidden * BF16_BYTES,
                "one BF16 embedding row")
            add("lm_head", vocab * hidden * BF16_BYTES,
                "separate BF16 vocabulary head")
        elif case.family == "dense_step":
            add("mlp_weights", mlp_elements * BF16_BYTES,
                "gate/up/down weights in every layer")
            add("normalization_weights",
                (layers * (2 * hidden + 2 * head_dim) + hidden) * BF16_BYTES,
                "two layer norms, Q/K norms, and final norm in BF16")
            add("tied_embedding_and_head", vocab * hidden * BF16_BYTES,
                "full BF16 vocabulary head, including the token embedding row")
        else:
            add("mlp_weights", mlp_elements * BF16_BYTES,
                "gate/up/down weights shared across verification positions")
            add("normalization_weights",
                (layers * 2 * hidden + hidden) * BF16_BYTES,
                "two layer norms and final norm in BF16")
            add("input_embedding_rows", steps * hidden * BF16_BYTES,
                "one BF16 embedding row per verified position")
            add("lm_head", vocab * hidden * BF16_BYTES,
                "separate BF16 vocabulary head read once across positions")

    add("old_kv_cache_reads", 2 * cache_positions * kdim * BF16_BYTES,
        "one read of the needed old K and V positions")
    add("logits_writes", steps * vocab * FP32_BYTES,
        "all required FP32 logits")
    add("new_kv_writes", 2 * layers * steps * kdim * BF16_BYTES,
        "required BF16 K and V outputs, including masked speculative slots")
    if case.family == "spec_target_step":
        depth = p["draft_depth"]
        add("token_and_feature_io",
            (steps + 3 + steps) * INT64_BYTES + 3 * hidden * BF16_BYTES,
            "input token/drafts, accepted/committed/cache scalars, "
            "committed tokens, and three target features")
    else:
        add("token_io", 2 * INT64_BYTES, "input and next-token IDs")

    minimum_bytes = sum(item["bytes"] for item in components)
    floor_ms = minimum_bytes / (bandwidth_tb_s * 1e12) * 1e3
    result.update(
        minimum_bytes=minimum_bytes,
        minimum_gb=minimum_bytes / 1e9,
        hbm_floor_ms=floor_ms,
        components=components,
    )
    if measured_ms is not None:
        result.update(measured_ms=measured_ms,
                      gap_to_hbm_floor_x=measured_ms / floor_ms,
                      percent_of_hbm_sol=100 * floor_ms / measured_ms)
    return result
