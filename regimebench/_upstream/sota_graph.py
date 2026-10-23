"""SOTA decode baseline: a vLLM-style Qwen3 step from FlashInfer + cuBLAS kernels.

The step mirrors vLLM's Qwen3 decode: fused QKV / gate-up GEMMs, FlashInfer
RMSNorm (with the fused add+norm between layers), per-head QK-norm, in-place
neox RoPE from a [cos|sin] cache, paged-KV append, paged decode attention,
SwiGLU, tied LM head, and an argmax. Variants form a factorial ablation over
CUDA graph (yes/no), PDL (on/off) and GEMM source (cuBLAS via torch or
``flashinfer.mm_bf16``). The attention backend is autotuned once per shape under
PDL and then held fixed; ``mm_bf16`` backends are chosen once per GEMM shape
under PDL and reused with PDL off.

Self-test (needs a GPU)::

    python -m megabench.sota.arms.sota_graph --selftest [--variants graph-pdl ...]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from typing import Any, Callable

from ..shapes import QWEN3_06B, Shape

PAGE = 16
VARIANTS = ("graph-pdl", "graph-nopdl", "graph-figemm-pdl", "graph-figemm-nopdl",
            "eager-pdl", "eager-nopdl")
# (graph, pdl, flashinfer GEMM) per variant
_FACTORS = {
    "graph-pdl": (True, True, False),
    "graph-nopdl": (True, False, False),
    "graph-figemm-pdl": (True, True, True),
    "graph-figemm-nopdl": (True, False, True),
    "eager-pdl": (False, True, False),
    "eager-nopdl": (False, False, False),
}
_PDL_GEMM_BACKENDS = ("tgv", "cudnn", "tinygemm", "cute-dsl")
_REL_L2_TOL = 1e-2

# Autotune results, keyed by shape id (attention) and (m, n, k) (GEMM).
_ATTN_CACHE: dict[str, tuple[dict, dict]] = {}
_GEMM_CACHE: dict[tuple[int, int, int], tuple[str, dict]] = {}


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {str(e).splitlines()[0][:240] if str(e) else ''}"


def _rel_l2(a, b) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _time_graph(fn: Callable[[], None], calls: int, reps: int = 20) -> float:
    """Median ms per call of ``fn`` (which issues ``calls`` calls), via a CUDA graph."""
    import torch
    fn()
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / calls)
    del g
    return statistics.median(times)


# --------------------------------------------------------------------------- KV
class _KV:
    """Static paged KV cache (HND, page 16) plus the int32 index tensors."""

    def __init__(self, shape: Shape, inputs: dict, device: str) -> None:
        import torch
        m, B, ctx = shape.model, shape.batch, shape.context
        self.pages = -(-(ctx + 1) // PAGE)
        L, KVH, D = m.layers, m.kv_heads, m.head_dim
        # [layer, page, k|v, kv_head, slot, dim]
        self.kv = torch.zeros(L, B * self.pages, 2, KVH, PAGE, D, dtype=torch.bfloat16,
                              device=device)
        for j, name in enumerate(("kcache", "vcache")):
            pad = torch.zeros(L, self.pages * PAGE, KVH, D, dtype=torch.bfloat16,
                              device=device)
            pad[:, :ctx] = inputs[name]
            pg = pad.view(L, self.pages, PAGE, KVH, D).permute(0, 1, 3, 2, 4)
            for b in range(B):  # batch>1 fixtures do not exist; replicate the prefix
                self.kv[:, b * self.pages:(b + 1) * self.pages, j] = pg
        i32 = dict(dtype=torch.int32, device=device)
        self.indptr = torch.arange(B + 1, **i32) * self.pages
        self.indices = torch.arange(B * self.pages, **i32)
        self.last_page_len = torch.full((B,), (ctx + 1 - 1) % PAGE + 1, **i32)
        self.seq_lens = torch.full((B,), ctx + 1, **i32)
        self.block_tables = self.indices.view(B, self.pages).contiguous()
        self.positions = torch.full((B,), ctx, **i32)
        self.batch_idx = torch.arange(B, **i32)


def _make_attn(cfg: dict, shape: Shape, kv: _KV, device: str) -> Callable:
    """Build ``run(kv_layer, q[B,16,128], out[B,16,128], pdl)`` for an attention config."""
    import torch
    import flashinfer
    m, B = shape.model, shape.batch
    scale = 1.0 / math.sqrt(m.head_dim)
    ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    if cfg["kind"] == "direct":
        def run_direct(kv_l, q, out, pdl):
            flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                q, kv_l, ws, kv.block_tables, kv.seq_lens, shape.attended,
                bmm1_scale=scale, bmm2_scale=1.0, out=out, kv_layout="HND",
                enable_pdl=pdl)
        run_direct.ws = ws
        return run_direct
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        ws, "HND", use_cuda_graph=True, use_tensor_cores=cfg["tc"],
        paged_kv_indptr_buffer=kv.indptr, paged_kv_indices_buffer=kv.indices,
        paged_kv_last_page_len_buffer=kv.last_page_len, backend=cfg["backend"])
    w.plan(kv.indptr, kv.indices, kv.last_page_len, m.q_heads, m.kv_heads, m.head_dim,
           PAGE, pos_encoding_mode="NONE", q_data_type=torch.bfloat16,
           kv_data_type=torch.bfloat16, sm_scale=scale)

    def run(kv_l, q, out, pdl):
        w.run(q, kv_l, out=out, enable_pdl=pdl)
    run.wrapper, run.ws = w, ws  # keep alive
    run.resolved_backend = getattr(w, "_backend", cfg["backend"])
    return run


def _ref_attn(kv_l, q, n: int):
    """fp32-softmax GQA attention over the first ``n`` cached tokens, B=1."""
    import torch
    k = kv_l[:, 0].permute(0, 2, 1, 3).reshape(-1, kv_l.shape[2], kv_l.shape[4])[:n]
    v = kv_l[:, 1].permute(0, 2, 1, 3).reshape(-1, kv_l.shape[2], kv_l.shape[4])[:n]
    g = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(g, 1).float(), v.repeat_interleave(g, 1).float()
    s = torch.einsum("hd,thd->ht", q[0].float(), k) / math.sqrt(q.shape[-1])
    return torch.einsum("ht,thd->hd", s.softmax(-1), v).unsqueeze(0)


def _autotune_attention(shape: Shape, inputs: dict, device: str) -> tuple[dict, dict]:
    if shape.id in _ATTN_CACHE:
        return _ATTN_CACHE[shape.id]
    import torch
    m = shape.model
    kv = _KV(shape, inputs, device)
    g = torch.Generator(device=device).manual_seed(11)
    # Fill the new-token slot and make a random query so validation is meaningful.
    kv.kv[:, shape.context // PAGE, :, :, shape.context % PAGE] = (
        torch.randn(m.layers, 2, m.kv_heads, m.head_dim, generator=g, device=device) * 0.1
    ).to(torch.bfloat16)
    q = torch.randn(shape.batch, m.q_heads, m.head_dim, generator=g, device=device).to(
        torch.bfloat16)
    cands: list[dict] = [{"kind": "wrapper", "backend": be, "tc": tc}
                         for be in ("fa2", "trtllm-gen", "auto", "cute-dsl")
                         for tc in (False, True)]
    cands.append({"kind": "direct"})
    record: dict[str, Any] = {}
    best: tuple[float, dict] | None = None
    for cfg in cands:
        key = ("trtllm-direct" if cfg["kind"] == "direct"
               else f"{cfg['backend']}/tc={cfg['tc']}")
        if cfg["kind"] == "wrapper" and cfg["backend"] in ("trtllm-gen", "cute-dsl") \
                and not cfg["tc"]:
            record[key] = "skipped: backend forces tensor cores (same as tc=True)"
            continue
        try:
            run = _make_attn(cfg, shape, kv, device)
            out = torch.zeros(shape.batch, m.q_heads, m.head_dim, dtype=torch.bfloat16,
                              device=device)
            run(kv.kv[0], q, out, True)
            torch.cuda.synchronize()
            rel = _rel_l2(out[:1], _ref_attn(kv.kv[0], q, shape.attended))
            if not rel <= _REL_L2_TOL:
                record[key] = f"invalid: rel_l2={rel:.3e}"
                continue

            def many():
                for l in range(m.layers):
                    run(kv.kv[l], q, out, True)
            ms = _time_graph(many, m.layers) * m.layers
            record[key] = {"ms_per_28_layers": ms, "rel_l2": rel,
                           "resolved": getattr(run, "resolved_backend", "trtllm-gen")}
            if best is None or ms < best[0]:
                best = (ms, cfg)
        except Exception as e:  # noqa: BLE001 - record and move on
            record[key] = _err(e)
    if best is None:
        raise RuntimeError(f"no valid attention backend: {record}")
    chosen = dict(best[1], name=next(k for k, v in record.items()
                                     if isinstance(v, dict) and v["ms_per_28_layers"] == best[0]))
    _ATTN_CACHE[shape.id] = (chosen, record)
    return chosen, record


def _autotune_gemm(tag: str, wts: list, m_rows: int, device: str) -> tuple[str, dict]:
    """Pick the fastest PDL-capable ``mm_bf16`` backend for weights ``wts`` ([n,k] each)."""
    n, k = wts[0].shape
    key = (m_rows, n, k)
    if key in _GEMM_CACHE:
        return _GEMM_CACHE[key]
    import torch
    import flashinfer
    g = torch.Generator(device=device).manual_seed(5)
    a = torch.randn(m_rows, k, generator=g, device=device).to(torch.bfloat16)
    out = torch.empty(m_rows, n, dtype=torch.bfloat16, device=device)
    ref = torch.mm(a, wts[0].t())
    record: dict[str, Any] = {}
    best: tuple[float, str] | None = None
    for be in _PDL_GEMM_BACKENDS:
        try:
            flashinfer.mm_bf16(a, wts[0].t(), pdl=True, out=out, backend=be)
            torch.cuda.synchronize()
            rel = _rel_l2(out, ref)
            if not rel <= _REL_L2_TOL:
                record[be] = f"invalid: rel_l2={rel:.3e}"
                continue

            def many():
                for w in wts:
                    flashinfer.mm_bf16(a, w.t(), pdl=True, out=out, backend=be)
            ms = _time_graph(many, len(wts))
            record[be] = {"ms_per_call": ms, "rel_l2": rel}
            if best is None or ms < best[0]:
                best = (ms, be)
        except Exception as e:  # noqa: BLE001
            record[be] = _err(e)
    if best is None:
        raise RuntimeError(f"no usable mm_bf16 PDL backend for {tag} {key}: {record}")
    _GEMM_CACHE[key] = (best[1], record)
    return _GEMM_CACHE[key]


# --------------------------------------------------------------------------- Step
class SotaGraphStep:
    arm = "sota-graph"
    precision = "bf16"

    def __init__(self, arm: "SotaGraphArm", shape: Shape, inputs: dict, variant: str,
                 device: str) -> None:
        import torch
        import flashinfer
        self._fi = flashinfer
        self.variant = variant
        graph, pdl, fi_gemm = _FACTORS[variant]
        self.shape, self.pdl, self.fi_gemm, self.graph = shape, pdl, fi_gemm, graph
        self.config: dict[str, Any] = {}
        m, B = shape.model, shape.batch
        self.m, self.B, self.eps = m, B, m.eps
        bf16 = torch.bfloat16
        dev = dict(device=device)
        self.inputs = inputs  # read-only references; keeps weights alive

        # Fused weights are shared across variants of one arm instance.
        w = arm._fused_weights(shape, inputs)
        self.w_qkv, self.w_gu = w["w_qkv"], w["w_gu"]
        L = m.layers
        self.layers = [dict(
            ln1=inputs["ln1"][l], qn=inputs["qn"][l], kn=inputs["kn"][l],
            ln2=inputs["ln2"][l],
            qkv=self.w_qkv[l].t(), o=inputs["wo"][l].t(), gu=self.w_gu[l].t(),
            down=inputs["wd"][l].t()) for l in range(L)]
        self.fnorm = inputs["fnorm"]
        self.embed = inputs["embed"]
        self.lm_head = self.embed.t()

        # RoPE cos|sin cache (fp32, [max_pos, head_dim]; NOT the duplicated HF layout).
        D = m.head_dim
        max_pos = max(shape.attended, 4096)
        inv = m.rope_theta ** (-torch.arange(0, D, 2, dtype=torch.float64, **dev) / D)
        ang = torch.outer(torch.arange(max_pos, dtype=torch.float64, **dev), inv)
        self.cos_sin = torch.cat((ang.cos(), ang.sin()), -1).float().contiguous()

        self.kvs = _KV(shape, inputs, device)
        # Static activation buffers.
        self.tok = inputs["token"].reshape(1).expand(B).clone().to(torch.int64)
        self.next = torch.zeros(B, dtype=torch.int64, **dev)
        self.res = torch.empty(B, m.hidden, dtype=bf16, **dev)
        self.h = torch.empty(B, m.hidden, dtype=bf16, **dev)
        self.qkv = torch.empty(B, m.q_dim + 2 * m.kv_dim, dtype=bf16, **dev)
        self.q_n = torch.empty(B, m.q_dim, dtype=bf16, **dev)
        self.k_n = torch.empty(B, m.kv_dim, dtype=bf16, **dev)
        self.attn = torch.empty(B, m.q_heads, D, dtype=bf16, **dev)
        self.gu = torch.empty(B, 2 * m.intermediate, dtype=bf16, **dev)
        self.act = torch.empty(B, m.intermediate, dtype=bf16, **dev)
        self.logits_bf16 = torch.empty(B, m.vocab, dtype=bf16, **dev)
        self.logits = torch.empty(B, m.vocab, dtype=torch.float32, **dev)
        # Views (strided slices of the fused QKV output; the kernels take row strides).
        self.q_in = self.qkv[:, :m.q_dim].view(B, m.q_heads, D)
        self.k_in = self.qkv[:, m.q_dim:m.q_dim + m.kv_dim].view(B, m.kv_heads, D)
        self.v_in = self.qkv[:, m.q_dim + m.kv_dim:].view(B, m.kv_heads, D)
        self.q_out3 = self.q_n.view(B, m.q_heads, D)
        self.k_out3 = self.k_n.view(B, m.kv_heads, D)
        self.attn2 = self.attn.view(B, m.q_dim)
        self._gemm_out = {"qkv": self.qkv, "o": self.h, "gu": self.gu, "down": self.h,
                          "lm_head": self.logits_bf16}

        # Attention backend: autotuned once per shape (PDL on), fixed for every variant.
        self.attn_cfg, attn_rec = _autotune_attention(shape, inputs, device)
        self._attn_run = _make_attn(self.attn_cfg, shape, self.kvs, device)

        # GEMM backends for the FlashInfer-GEMM variants.
        self._gemm_be: dict[str, str] = {}
        gemm_rec: dict[str, Any] = {}
        if fi_gemm:
            try:
                for tag, wts in (("qkv", [x["qkv"].t() for x in self.layers]),
                                 ("o", [x["o"].t() for x in self.layers]),
                                 ("gu", [x["gu"].t() for x in self.layers]),
                                 ("down", [x["down"].t() for x in self.layers]),
                                 ("lm_head", [self.embed])):
                    be, rec = _autotune_gemm(tag, wts, B, device)
                    self._gemm_be[tag], gemm_rec[tag] = be, rec
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"flashinfer mm_bf16 unavailable: {_err(e)}") from e

        # PDL accounting, filled during the first (warmup) step.
        self._ops: dict[str, list] = {}
        self._recording = True
        self._chained = False
        self._graphs: dict[bool, Any] = {}
        self._warm_and_capture()
        self._recording = False
        self._fill_config(attn_rec, gemm_rec)

    # -- op helpers --------------------------------------------------------
    def _note(self, op: str, supported: bool) -> None:
        if self._recording:
            e = self._ops.setdefault(op, [0, supported])
            e[0] += 1

    def _mm(self, tag: str, a, wt, out) -> None:
        self._note(f"gemm_{tag}", self.fi_gemm)
        if self.fi_gemm:
            self._fi.mm_bf16(a, wt, pdl=self.pdl, out=out, backend=self._gemm_be[tag])
        else:
            import torch
            torch.mm(a, wt, out=out)

    def _step(self, src) -> None:
        """Enqueue one decode step (token read from ``src``). No sync, no allocation
        (beyond the caching allocator in eager mode)."""
        import torch
        fi, pdl, kvs, m = self._fi, self.pdl, self.kvs, self.m
        res, h, eps = self.res, self.h, self.eps
        self._note("embedding", False)
        torch.index_select(self.embed, 0, src, out=res)
        for l, w in enumerate(self.layers):
            if l == 0:
                self._note("rmsnorm", True)
                fi.rmsnorm(res, w["ln1"], eps, out=h, enable_pdl=pdl)
            else:
                self._note("fused_add_rmsnorm", True)
                fi.fused_add_rmsnorm(h, res, w["ln1"], eps, enable_pdl=pdl)
            self._mm("qkv", h, w["qkv"], self.qkv)
            self._note("qk_rmsnorm", True)
            fi.rmsnorm(self.q_in, w["qn"], eps, out=self.q_out3, enable_pdl=pdl)
            fi.rmsnorm(self.k_in, w["kn"], eps, out=self.k_out3, enable_pdl=pdl)
            self._note("rope", False)
            fi.apply_rope_with_cos_sin_cache_inplace(
                kvs.positions, self.q_n, self.k_n, m.head_dim, self.cos_sin, True)
            self._note("append_paged_kv", False)
            fi.append_paged_kv_cache(
                self.k_out3, self.v_in, kvs.batch_idx, kvs.positions, kvs.kv[l],
                kvs.indices, kvs.indptr, kvs.last_page_len, kv_layout="HND")
            self._note("attention", True)
            self._attn_run(kvs.kv[l], self.q_out3, self.attn, pdl)
            self._mm("o", self.attn2, w["o"], h)
            self._note("fused_add_rmsnorm", True)
            fi.fused_add_rmsnorm(h, res, w["ln2"], eps, enable_pdl=pdl)
            self._mm("gu", h, w["gu"], self.gu)
            self._note("silu_and_mul", True)
            fi.silu_and_mul(self.gu, out=self.act, enable_pdl=pdl)
            self._mm("down", self.act, w["down"], h)
        self._note("fused_add_rmsnorm", True)
        fi.fused_add_rmsnorm(h, res, self.fnorm, eps, enable_pdl=pdl)
        self._mm("lm_head", h, self.lm_head, self.logits_bf16)
        self._note("logits_to_fp32", False)
        self.logits.copy_(self.logits_bf16)
        self._note("argmax", False)
        torch.argmax(self.logits, -1, out=self.next)

    def _warm_and_capture(self) -> None:
        import torch
        cur = torch.cuda.current_stream()
        s = torch.cuda.Stream()
        s.wait_stream(cur)
        with torch.cuda.stream(s):  # warmup also triggers every JIT build
            self._step(self.tok)  # first step records the PDL op accounting
            self._recording = False
            self._step(self.tok)
            self._step(self.next)
        cur.wait_stream(s)
        torch.cuda.synchronize()
        if not self.graph:
            return
        for chained, src in ((False, self.tok), (True, self.next)):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                self._step(src)
            self._graphs[chained] = g
        torch.cuda.synchronize()

    def _fill_config(self, attn_rec: dict, gemm_rec: dict) -> None:
        import torch
        fi = self._fi
        per_op = {op: {"count": n, "pdl_supported": sup,
                       "pdl_requested": bool(sup and self.pdl)}
                  for op, (n, sup) in self._ops.items()}
        total = sum(v["count"] for v in per_op.values())
        with_pdl = sum(v["count"] for v in per_op.values() if v["pdl_requested"])
        try:
            norm_impl = ("cuda-jit" if fi.norm._use_cuda_norm(torch.device("cuda"))
                         else "cute-dsl")
        except Exception:  # noqa: BLE001
            norm_impl = "unknown"
        import os
        self.config.update({
            "graph": self.graph, "pdl_enabled": self.pdl,
            "gemm": "flashinfer-mm_bf16" if self.fi_gemm else "cublas",
            "figemm_backend": dict(self._gemm_be) if self.fi_gemm else None,
            "figemm_autotune": gemm_rec or None,
            "attn_backend": self.attn_cfg["name"],
            "attn_autotune": attn_rec,
            "pdl": {"launches_with_pdl": with_pdl, "launches_total": total,
                    "per_op": per_op},
            "env_FLASHINFER_USE_CUDA_NORM": os.environ.get("FLASHINFER_USE_CUDA_NORM"),
            "flashinfer_version": getattr(fi, "__version__", "unknown"),
            "norm_impl": norm_impl,
            "torch_version": torch.__version__,
        })

    # -- Step protocol -------------------------------------------------------
    def launch(self) -> None:
        if self.graph:
            self._graphs[self._chained].replay()
        else:
            self._step(self.next if self._chained else self.tok)

    def outputs(self) -> dict[str, Any]:
        ctx = self.shape.context
        page, slot = ctx // PAGE, ctx % PAGE
        kv = self.kvs.kv
        k_w = kv[:, page, 0, :, slot].contiguous()  # [L, KVH, D] (B=1)
        v_w = kv[:, page, 1, :, slot].contiguous()
        if self.B != 1:
            raise NotImplementedError("batched outputs are a sweep TODO")
        return {"logits": self.logits[0], "next_token": self.next[0],
                "k_write": k_w, "v_write": v_w}

    def set_chained(self, on: bool) -> None:
        self._chained = bool(on)

    def close(self) -> None:
        self._graphs.clear()
        for name in ("layers", "w_qkv", "w_gu", "kvs", "_attn_run", "inputs"):
            setattr(self, name, None)


# --------------------------------------------------------------------------- Arm
class SotaGraphArm:
    name = "sota-graph"

    def __init__(self, **options: Any) -> None:
        self.options = options
        self._weights: dict[tuple, dict] = {}

    def variants(self) -> tuple[str, ...]:
        return VARIANTS

    def supports(self, shape: Shape, variant: str) -> str | None:
        if variant not in _FACTORS:
            return f"unknown variant {variant!r}"
        if shape.batch != 1:
            return "only batch 1 has fixtures"
        try:
            import torch
            if not torch.cuda.is_available():
                return "CUDA device required"
            major, _ = torch.cuda.get_device_capability()
        except Exception as e:  # noqa: BLE001
            return f"torch/CUDA unavailable: {_err(e)}"
        if _FACTORS[variant][1] and major < 9:
            return f"PDL needs compute capability >= 9 (got {major}.x)"
        return None

    def _fused_weights(self, shape: Shape, inputs: dict) -> dict:
        import torch
        key = (shape.id, inputs["wq"].data_ptr(), inputs["wg"].data_ptr())
        if key not in self._weights:
            self._weights[key] = {
                "w_qkv": torch.cat((inputs["wq"], inputs["wk"], inputs["wv"]), 1).contiguous(),
                "w_gu": torch.cat((inputs["wg"], inputs["wu"]), 1).contiguous(),
                "_keep": (inputs["wq"], inputs["wg"])}
        return self._weights[key]

    def prepare(self, shape: Shape, inputs: dict, variant: str, device: str):
        reason = self.supports(shape, variant)
        if reason:
            raise RuntimeError(f"unsupported: {reason}")
        return SotaGraphStep(self, shape, inputs, variant, device)


# --------------------------------------------------------------------------- self-test
def _selftest(variants: list[str] | None, seed: int = 7301) -> int:
    import torch
    from .. import shapes as S
    shape = Shape(1, 128)
    dev = "cuda"
    inputs = S.make_shape_inputs(shape, seed, dev)
    ref = S.reference_outputs(shape, inputs)
    arm = SotaGraphArm()
    failures = 0
    for i, variant in enumerate(variants or list(arm.variants())):
        print(f"=== {variant}", flush=True)
        try:
            step = arm.prepare(shape, inputs, variant, dev)
        except Exception as e:  # noqa: BLE001
            print(f"  PREPARE FAILED: {_err(e)}")
            failures += 1
            continue
        step.set_chained(False)
        step.launch()
        torch.cuda.synchronize()
        out = step.outputs()
        lg, rl = out["logits"].float(), ref["logits"].float()
        print(f"  logits rel_l2={_rel_l2(lg, rl):.3e} max_abs={float((lg - rl).abs().max()):.3e}"
              f" argmax={int(out['next_token'])} ref_argmax={int(ref['next_token'])}")
        for k in ("k_write", "v_write"):
            print(f"  {k} max_abs={float((out[k].float() - ref[k].float()).abs().max()):.3e}")
        # Idempotence: a second unchained replay must reproduce the same logits.
        step.launch()
        torch.cuda.synchronize()
        print(f"  repeat-launch max_abs={float((step.outputs()['logits'] - lg).abs().max()):.3e}")
        step.set_chained(True)
        for _ in range(20):
            step.launch()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(200):
            step.launch()
        b.record()
        torch.cuda.synchronize()
        print(f"  chained: {a.elapsed_time(b) / 200 * 1e3:.1f} us/step (200 launches)")
        step.set_chained(False)
        c = step.config
        print(f"  attn_backend={c['attn_backend']} gemm={c['gemm']} "
              f"figemm_backend={c['figemm_backend']} norm_impl={c['norm_impl']} "
              f"pdl={c['pdl']['launches_with_pdl']}/{c['pdl']['launches_total']} "
              f"(FlashInfer {c['flashinfer_version']})")
        if i == 0:
            print("  attn_autotune:", json.dumps(c["attn_autotune"], indent=2, default=str))
        if c["figemm_autotune"]:
            print("  figemm_autotune:", json.dumps(c["figemm_autotune"], indent=2, default=str))
        step.close()
    return 1 if failures else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--variants", nargs="*", default=None)
    args = p.parse_args()
    if not args.selftest:
        p.error("only --selftest is supported")
    raise SystemExit(_selftest(args.variants))


if __name__ == "__main__":
    main()
