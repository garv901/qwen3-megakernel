"""fi-graph arm: the best CUDA-graph + PDL multi-kernel Qwen3 step built from FlashInfer + cuBLAS.

Ported from ``_upstream/sota_graph.py`` (``Shape(batch, context)``) to ``Workload`` (ragged
sequences, ``q_len >= 1`` per sequence). The step mirrors a vLLM-style Qwen3 forward:
fused QKV / gate-up GEMMs, FlashInfer RMSNorm (fused add+norm between layers), per-head
QK-norm, neox RoPE from a [cos|sin] cache at per-token positions, paged-KV append, paged
attention, SwiGLU, tied LM head over the last token of every sequence, argmax.

Design points
  * Paged KV cache, page 16, HND, ``[L, pages, 2, KVH, 16, D]``. Every sequence owns its own
    pages (capacity kv_len + q_len); the fixture prefix is copied in ``prepare``.
  * Attention backend is autotuned once per workload with PDL on (candidates that raise or
    disagree with a torch attention on the same inputs are dropped and recorded); the winner
    is reused by every variant.
  * GEMM backend per group (qkv / o / gu / down / lm_head) is autotuned once per (M, N, K)
    under PDL: ``torch.mm`` (cuBLAS) vs ``flashinfer.mm_bf16`` backends. The ``*-cublas``
    variants always use ``torch.mm``. The PDL-off variants reuse the PDL-on winners.
  * Timed ``launch`` = replay of the captured graph (or the eager step). Logits stay in BF16
    as the LM-head GEMM produces them and argmax runs on them (timed); the fp32 cast, and the
    gather of k_new / v_new from the paged cache, happen in ``outputs()`` (untimed).

Variants: graph-pdl-best (headline), graph-pdl-cublas, graph-nopdl-best, graph-nopdl-cublas,
eager-pdl-best (R0 only).
"""

from __future__ import annotations

import hashlib
import math
import os
import statistics
from pathlib import Path
from typing import Any, Callable

import torch

from ..workload import Workload

PAGE = 16
VARIANTS = ("graph-pdl-best", "graph-pdl-cublas", "graph-nopdl-best", "graph-nopdl-cublas",
            "eager-pdl-best")
# variant -> (graph, pdl, gemm mode)
_FACTORS = {
    "graph-pdl-best": (True, True, "best"),
    "graph-pdl-cublas": (True, True, "cublas"),
    "graph-nopdl-best": (True, False, "best"),
    "graph-nopdl-cublas": (True, False, "cublas"),
    "eager-pdl-best": (False, True, "best"),
}
_FI_GEMM_BACKENDS = ("tgv", "cudnn", "tinygemm", "cute-dsl", "cutlass", "cublaslt")
_PDL_GEMM_BACKENDS = ("tgv", "tinygemm")   # cute-dsl: PDL only for M <= 32 (see _gemm_pdl_capable)
_REL_L2_TOL = 1e-2
_DECODE_BACKENDS = ("fa2", "trtllm-gen", "auto", "cute-dsl")
_PREFILL_BACKENDS = ("fa2", "fa3", "trtllm-gen")
_N_CHECK_SEQS = 8

# Autotune results: attention keyed by workload shape, GEMM keyed by (m, n, k).
_ATTN_CACHE: dict[tuple, tuple[dict, dict]] = {}
_GEMM_CACHE: dict[tuple[int, int, int], tuple[str, dict]] = {}


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {str(e).splitlines()[0][:240] if str(e) else ''}"


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _time_graph(fn: Callable[[], None], calls: int, reps: int = 20) -> float:
    """Median ms per call of ``fn`` (which issues ``calls`` calls) replayed from a CUDA graph."""
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


# --------------------------------------------------------------------------- paged KV
class Batch:
    """Index tensors of a (sub-)batch of sequences over a paged cache (no KV data).

    Sequence i holds ``kv_lens[i] + q_lens[i]`` cache slots (the step appends the new
    tokens before attention), in ``ceil(total / 16)`` consecutive pages starting at
    ``page0[i]``. ``page_base`` is the first global page id of this (sub-)batch.
    """

    def __init__(self, q_lens: list[int], kv_lens: list[int], device: str,
                 page_base: int = 0) -> None:
        self.device = device
        self.q_lens, self.kv_lens = list(q_lens), list(kv_lens)
        self.n_seq = n = len(q_lens)
        self.total_q = sum(q_lens)
        self.totals = [k + q for k, q in zip(kv_lens, q_lens)]
        self.pages = [_cdiv(t, PAGE) for t in self.totals]
        self.page_base = page_base
        self.page0 = []
        acc = page_base
        for p in self.pages:
            self.page0.append(acc)
            acc += p
        self.n_pages = acc - page_base
        self.max_q, self.max_kv = max(q_lens), max(self.totals)
        self.max_pages = max(self.pages)
        self.cu_q = [0]
        for q in q_lens:
            self.cu_q.append(self.cu_q[-1] + q)
        i32 = dict(dtype=torch.int32, device=device)
        i64 = dict(dtype=torch.int64, device=device)
        cu_pages = [0]
        for p in self.pages:
            cu_pages.append(cu_pages[-1] + p)
        self.qo_indptr = torch.tensor(self.cu_q, **i32)
        self.kv_indptr = torch.tensor(cu_pages, **i32)
        self.kv_indices = torch.arange(page_base, page_base + self.n_pages, **i32)
        self.last_page_len = torch.tensor([(t - 1) % PAGE + 1 for t in self.totals], **i32)
        self.seq_lens = torch.tensor(self.totals, **i32)
        table = torch.zeros(n, self.max_pages, dtype=torch.int32)
        for i, (p0, p) in enumerate(zip(self.page0, self.pages)):
            table[i, :p] = torch.arange(p0, p0 + p, dtype=torch.int32)
        self.block_tables = table.to(device)
        # Per new token: owning sequence, position, page id, slot in page.
        q_t = torch.tensor(q_lens, dtype=torch.int64)
        cuq_t = torch.tensor(self.cu_q[:-1], dtype=torch.int64)
        bidx = torch.repeat_interleave(torch.arange(n, dtype=torch.int64), q_t)
        pos = (torch.tensor(kv_lens, dtype=torch.int64)[bidx]
               + torch.arange(self.total_q, dtype=torch.int64) - cuq_t[bidx])
        page = torch.tensor(self.page0, dtype=torch.int64)[bidx] + pos // PAGE
        self.batch_idx = bidx.to(torch.int32).to(device)
        self.positions = pos.to(torch.int32).to(device)
        self.tok_page = page.to(device)
        self.tok_slot = (pos % PAGE).to(**i64)
        # int32 cumulative lengths for trtllm-gen / cudnn
        self.cu_kv = torch.tensor([0] + list(_cumsum(self.totals)), **i32)
        self._subs: dict[tuple[int, int], "Batch"] = {}

    def sub(self, lo: int, hi: int) -> "Batch":
        if (lo, hi) not in self._subs:
            self._subs[(lo, hi)] = Batch(self.q_lens[lo:hi], self.kv_lens[lo:hi], self.device,
                                         page_base=self.page0[lo])
        return self._subs[(lo, hi)]


def _cumsum(xs):
    acc = 0
    for x in xs:
        acc += x
        yield acc


class PagedKV:
    """``kv`` is ``[L, pages, 2, KVH, 16, D]`` bf16 (HND); ``batch`` holds the indices."""

    def __init__(self, w: Workload, inputs: dict, device: str) -> None:
        m = w.model
        L, KVH, D = m.layers, m.kv_heads, m.head_dim
        self.batch = Batch([s.q_len for s in w.seqs], [s.kv_len for s in w.seqs], device)
        b = self.batch
        self.kv = torch.zeros(L, b.n_pages, 2, KVH, PAGE, D, dtype=torch.bfloat16, device=device)
        cu_cache = [int(x) for x in inputs["cu_cache"]]
        for j, name in enumerate(("kcache", "vcache")):
            src = inputs[name]
            for i, n_pre in enumerate(b.kv_lens):
                if n_pre == 0:
                    continue
                row0, p0 = cu_cache[i], b.page0[i]
                full = n_pre // PAGE
                if full:
                    self.kv[:, p0:p0 + full, j] = src[:, row0:row0 + full * PAGE].view(
                        L, full, PAGE, KVH, D).permute(0, 1, 3, 2, 4)
                rem = n_pre - full * PAGE
                if rem:
                    self.kv[:, p0 + full, j, :, :rem] = src[
                        :, row0 + full * PAGE:row0 + n_pre].permute(0, 2, 1, 3)

    def gather_new(self, which: int) -> torch.Tensor:
        """K (which=0) or V (1) of every new token: ``[total_q, L, KVH, D]``."""
        b = self.batch
        return self.kv[:, :, which][:, b.tok_page, :, b.tok_slot]

    def fill_new_slots(self, layer: int, gen: torch.Generator) -> None:
        """Random K/V in the new-token slots of one layer (autotune validation only)."""
        b, kv = self.batch, self.kv[layer]
        vals = (torch.randn(b.total_q, 2, kv.shape[2], kv.shape[4], generator=gen,
                            device=kv.device) * 0.1).to(torch.bfloat16)
        kv[b.tok_page, :, :, b.tok_slot] = vals


# --------------------------------------------------------------------------- shapes / candidates
def gemm_shapes(w: Workload) -> dict[str, tuple[int, int, int]]:
    """(M, N, K) per GEMM group; the LM head runs on the last token of each sequence."""
    m, t = w.model, w.total_q
    return {"qkv": (t, m.q_dim + 2 * m.kv_dim, m.hidden), "o": (t, m.hidden, m.q_dim),
            "gu": (t, 2 * m.intermediate, m.hidden), "down": (t, m.hidden, m.intermediate),
            "lm_head": (w.n_seq, m.vocab, m.hidden)}


def split_point(w: Workload) -> int | None:
    """Number of leading q_len>1 sequences when the batch is [prefill seqs..., decode seqs...]."""
    qs = [s.q_len for s in w.seqs]
    npre = 0
    while npre < len(qs) and qs[npre] > 1:
        npre += 1
    if 0 < npre < len(qs) and all(q == 1 for q in qs[npre:]):
        return npre
    return None


def cfg_name(cfg: dict) -> str:
    k = cfg["kind"]
    if k == "decode-wrapper":
        return f"decode/{cfg['backend']}/tc={cfg['tc']}"
    if k == "decode-direct":
        return "decode/trtllm-direct"
    if k == "prefill-wrapper":
        return f"prefill/{cfg['backend']}"
    if k == "prefill-trtllm":
        return "prefill/trtllm-direct"
    if k == "prefill-cudnn":
        return "prefill/cudnn-direct"
    if k == "split":
        return f"split[{cfg_name(cfg['prefill'])}+{cfg_name(cfg['decode'])}]"
    raise ValueError(k)


def _decode_candidates() -> list[dict]:
    c = [{"kind": "decode-wrapper", "backend": be, "tc": tc}
         for be in _DECODE_BACKENDS for tc in (False, True)]
    c.append({"kind": "decode-direct"})
    return c


def _prefill_candidates() -> list[dict]:
    c = [{"kind": "prefill-wrapper", "backend": be} for be in _PREFILL_BACKENDS]
    c += [{"kind": "prefill-trtllm"}, {"kind": "prefill-cudnn"}]
    return c


def attention_candidates(w: Workload) -> list[dict]:
    """Attention configurations to autotune for ``w`` (names via ``cfg_name``)."""
    if w.is_pure_decode:
        return _decode_candidates()
    cands = _prefill_candidates()
    if split_point(w) is not None:
        for pre in (c for c in cands if c["kind"] == "prefill-wrapper"):
            for dec in ({"kind": "decode-wrapper", "backend": "fa2", "tc": False},
                        {"kind": "decode-direct"}):
                cands.append({"kind": "split", "prefill": pre, "decode": dec})
    return cands


def _static_skip(cfg: dict) -> str | None:
    if cfg["kind"] == "decode-wrapper" and cfg["backend"] in ("trtllm-gen", "cute-dsl") \
            and not cfg["tc"]:
        return "skipped: backend forces tensor cores (same as tc=True)"
    return None


def _ws_bytes(w: Workload) -> int:
    """Float workspace for attention: 128 MB, more for big token counts (split-KV partials)."""
    need = w.total_q * w.model.q_heads * w.model.head_dim * 4 * 4
    return int(min(max(need, 128 << 20), 2 << 30))


def _gemm_pdl_capable(backend: str, m: int) -> bool:
    return backend in _PDL_GEMM_BACKENDS or (backend == "cute-dsl" and m <= 32)


def check_seq_ids(n: int, k: int = _N_CHECK_SEQS) -> list[int]:
    if n <= k:
        return list(range(n))
    ids = {0, n - 1}
    for i in range(k):
        ids.add(round(i * (n - 1) / (k - 1)))
    return sorted(ids)


def pdl_summary(ops: dict[str, list], pdl: bool) -> dict:
    """``ops``: name -> [count, pdl_supported]. Counts are API calls, not device kernels."""
    per_op = {op: {"count": n, "pdl_supported": bool(sup), "pdl_requested": bool(sup and pdl)}
              for op, (n, sup) in ops.items()}
    total = sum(v["count"] for v in per_op.values())
    with_pdl = sum(v["count"] for v in per_op.values() if v["pdl_requested"])
    return {"launches_with_pdl": with_pdl, "launches_total": total, "per_op": per_op,
            "note": "counts API calls per step, not device kernels"}


# --------------------------------------------------------------------------- reference attention
def ref_attention(kv_l: torch.Tensor, q: torch.Tensor, b: Batch, seq_ids: list[int],
                  scale: float, chunk: int = 512) -> dict[int, torch.Tensor]:
    """fp32 causal GQA attention for the chosen sequences over one layer's paged cache.

    ``kv_l`` [pages, 2, KVH, 16, D]; ``q`` [total_q, QH, D]. New token j of sequence i sits at
    position kv_len_i + j and sees positions 0..kv_len_i + j. Returns {seq: [q_len, QH, D]}.
    """
    out: dict[int, torch.Tensor] = {}
    for i in seq_ids:
        p0, npg, total, qn = b.page0[i], b.pages[i], b.totals[i], b.q_lens[i]
        kv0 = total - qn
        blk = kv_l[p0:p0 + npg]
        k = blk[:, 0].permute(0, 2, 1, 3).reshape(-1, blk.shape[2], blk.shape[4])[:total].float()
        v = blk[:, 1].permute(0, 2, 1, 3).reshape(-1, blk.shape[2], blk.shape[4])[:total].float()
        g = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
        qi = q[b.cu_q[i]:b.cu_q[i] + qn].float()
        parts = []
        for c0 in range(0, qn, chunk):
            qc = qi[c0:c0 + chunk]
            s = torch.einsum("qhd,thd->hqt", qc, k) * scale
            limit = kv0 + c0 + torch.arange(qc.shape[0], device=q.device)  # last visible pos
            mask = torch.arange(total, device=q.device)[None, :] > limit[:, None]
            s = s.masked_fill(mask[None], float("-inf"))
            parts.append(torch.einsum("hqt,thd->qhd", s.softmax(-1), v))
        out[i] = torch.cat(parts, 0)
    return out


# --------------------------------------------------------------------------- attention builders
def _counter_buffer(batch: int, qh: int, device: str) -> torch.Tensor:
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    n = -(-max(batch * qh, sm) // 8) * 8
    return torch.zeros(n * 4, dtype=torch.uint8, device=device)


def _make_attn(cfg: dict, w: Workload, b: Batch, device: str, ws_bytes: int | None = None
               ) -> Callable:
    """Build ``run(kv_layer, q[T,QH,D], out[T,QH,D], pdl)`` for ``cfg`` over batch ``b``."""
    import flashinfer
    m = w.model
    scale = 1.0 / math.sqrt(m.head_dim)
    bf16 = torch.bfloat16
    kind = cfg["kind"]
    if kind == "split":
        lo = sum(1 for q in b.q_lens if q > 1)
        pre = _make_attn(cfg["prefill"], w, b.sub(0, lo), device, ws_bytes)
        dec = _make_attn(cfg["decode"], w, b.sub(lo, b.n_seq), device, ws_bytes)
        tp = b.cu_q[lo]

        def run_split(kv_l, q, out, pdl):
            pre(kv_l, q[:tp], out[:tp], pdl)
            dec(kv_l, q[tp:], out[tp:], pdl)
        run_split.parts = (pre, dec)
        return run_split

    ws = torch.zeros(ws_bytes or (128 << 20), dtype=torch.uint8, device=device)
    if kind == "decode-direct":
        cnt = _counter_buffer(b.n_seq, m.q_heads, device)

        def run_dd(kv_l, q, out, pdl):
            flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                query=q, kv_cache=kv_l, workspace_buffer=ws, block_tables=b.block_tables,
                seq_lens=b.seq_lens, max_seq_len=b.max_kv, bmm1_scale=scale, bmm2_scale=1.0,
                out=out, kv_layout="HND", enable_pdl=pdl, multi_ctas_kv_counter_buffer=cnt)
        run_dd.keep = (ws, cnt)
        return run_dd
    if kind == "decode-wrapper":
        wr = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws, "HND", use_cuda_graph=True, use_tensor_cores=cfg["tc"],
            paged_kv_indptr_buffer=b.kv_indptr, paged_kv_indices_buffer=b.kv_indices,
            paged_kv_last_page_len_buffer=b.last_page_len, backend=cfg["backend"])
        wr.plan(b.kv_indptr, b.kv_indices, b.last_page_len, m.q_heads, m.kv_heads, m.head_dim,
                PAGE, pos_encoding_mode="NONE", q_data_type=bf16, kv_data_type=bf16,
                sm_scale=scale)

        # The cute-dsl decode kernel accumulates into `out` with atomics when it splits KV
        # (reduction == "atomic": kv_splits in {2,4,8}, e.g. batch 8 x 8 KV heads); `out` must be
        # zero on entry (FlashInfer docs), so zero it inside the step (a real memset, timed).
        cd = getattr(wr, "_cute_dsl_wrapper", None)
        zero_out = cd is not None and getattr(cd, "_reduction", None) == "atomic"

        def run_dw(kv_l, q, out, pdl):
            if zero_out:
                out.zero_()
            wr.run(q, kv_l, out=out, enable_pdl=pdl)
        run_dw.keep = (wr, ws)
        run_dw.zero_out = zero_out
        run_dw.resolved_backend = getattr(wr, "_backend", cfg["backend"])
        return run_dw
    if kind == "prefill-wrapper":
        wr = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend=cfg["backend"])
        wr.plan(b.qo_indptr, b.kv_indptr, b.kv_indices, b.last_page_len, m.q_heads, m.kv_heads,
                m.head_dim, PAGE, causal=True, pos_encoding_mode="NONE", q_data_type=bf16,
                kv_data_type=bf16, sm_scale=scale)

        def run_pw(kv_l, q, out, pdl):
            wr.run(q, kv_l, out=out, enable_pdl=pdl)
        run_pw.keep = (wr, ws)
        run_pw.resolved_backend = getattr(wr, "_backend", cfg["backend"])
        return run_pw
    if kind == "prefill-trtllm":
        cnt = _counter_buffer(b.n_seq, m.q_heads, device)

        def run_pt(kv_l, q, out, pdl):
            flashinfer.prefill.trtllm_batch_context_with_kv_cache(
                query=q, kv_cache=kv_l, workspace_buffer=ws, block_tables=b.block_tables,
                seq_lens=b.seq_lens, max_q_len=b.max_q, max_kv_len=b.max_kv,
                bmm1_scale=scale, bmm2_scale=1.0, batch_size=b.n_seq,
                cum_seq_lens_q=b.qo_indptr, cum_seq_lens_kv=b.cu_kv, out=out,
                kv_layout="HND", enable_pdl=pdl, causal=True,
                multi_ctas_kv_counter_buffer=cnt)
        run_pt.keep = (ws, cnt)
        return run_pt
    if kind == "prefill-cudnn":
        from flashinfer.cudnn import cudnn_batch_prefill_with_kv_cache
        n = b.n_seq
        lens_q = b.seq_lens.new_tensor(b.q_lens).view(n, 1, 1, 1)
        lens_kv = b.seq_lens.view(n, 1, 1, 1)
        off = (b.qo_indptr.to(torch.int64) * (m.q_heads * m.head_dim)).to(torch.int32)

        def run_pc(kv_l, q, out, pdl):  # pdl: not exposed by the cudnn path
            cudnn_batch_prefill_with_kv_cache(
                q, kv_l[:, 0], kv_l[:, 1], scale, ws, max_token_per_sequence=b.max_q,
                max_sequence_kv=b.max_kv, actual_seq_lens_q=lens_q, actual_seq_lens_kv=lens_kv,
                block_tables=b.block_tables, causal=True, return_lse=False,
                batch_offsets_q=off, batch_offsets_o=off, out=out,
                is_cuda_graph_compatible=True)
        run_pc.keep = (ws, lens_q, lens_kv, off)
        return run_pc
    raise ValueError(f"unknown attention kind {kind!r}")


def _attn_key(w: Workload) -> tuple:
    return (w.id, tuple((s.kv_len, s.q_len) for s in w.seqs))


def _autotune_attention(w: Workload, kvs: PagedKV, device: str) -> tuple[dict, dict]:
    key = _attn_key(w)
    if key in _ATTN_CACHE:
        return _ATTN_CACHE[key]
    m, b = w.model, kvs.batch
    gen = torch.Generator(device=device).manual_seed(11)
    kvs.fill_new_slots(0, gen)  # random new-token K/V in layer 0 so validation is meaningful
    q = torch.randn(b.total_q, m.q_heads, m.head_dim, generator=gen, device=device).to(
        torch.bfloat16)
    scale = 1.0 / math.sqrt(m.head_dim)
    ids = check_seq_ids(b.n_seq)
    ref = torch.cat(list(ref_attention(kvs.kv[0], q, b, ids, scale).values()), 0)
    rows = torch.cat([torch.arange(b.cu_q[i], b.cu_q[i] + b.q_lens[i]) for i in ids]).to(device)
    ws_bytes = _ws_bytes(w)
    record: dict[str, Any] = {}
    best: tuple[float, dict, str] | None = None
    for cfg in attention_candidates(w):
        name = cfg_name(cfg)
        skip = _static_skip(cfg)
        if skip:
            record[name] = skip
            continue
        run = None
        try:
            run = _make_attn(cfg, w, b, device, ws_bytes)
            out = torch.zeros(b.total_q, m.q_heads, m.head_dim, dtype=torch.bfloat16,
                              device=device)
            # Validate exactly as the step will run it: poisoned (NaN) output buffer, eager
            # warmup, then CUDA-graph capture and replays (stale output, poisoned, repeated).
            out.fill_(float("nan"))
            run(kvs.kv[0], q, out, True)
            torch.cuda.synchronize()
            rels = [_rel_l2(out[rows], ref)]
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                run(kvs.kv[0], q, out, True)
            for poison in (False, True, False):
                if poison:
                    out.fill_(float("nan"))
                g.replay()
                torch.cuda.synchronize()
                rels.append(_rel_l2(out[rows], ref))
            del g
            rel = max(rels) if all(r == r for r in rels) else float("nan")
            if not rel <= _REL_L2_TOL:
                record[name] = f"invalid: rel_l2={rels} (eager, replay, replay-after-poison, replay)"
                continue

            def many():
                for l in range(m.layers):
                    run(kvs.kv[l], q, out, True)
            ms = _time_graph(many, m.layers) * m.layers
            record[name] = {"ms_per_28_layers": ms, "rel_l2": rel,
                            "zeroes_out": bool(getattr(run, "zero_out", False)),
                            "resolved": getattr(run, "resolved_backend", None)}
            if best is None or ms < best[0]:
                best = (ms, cfg, name)
        except Exception as e:  # noqa: BLE001 - record and move on
            record[name] = _err(e)
        finally:
            del run
            torch.cuda.empty_cache()
    if best is None:
        raise RuntimeError(f"no valid attention backend for {w.id}: {record}")
    chosen = dict(best[1], name=best[2])
    _ATTN_CACHE[key] = (chosen, record)
    return chosen, record


def _autotune_gemm(tag: str, wts: list[torch.Tensor], m_rows: int, device: str
                   ) -> tuple[str, dict]:
    """Fastest of torch.mm and the ``mm_bf16`` backends for weights ``wts`` (each ``[k, n]``
    column-major view). PDL on for the FlashInfer backends."""
    import flashinfer
    k, n = wts[0].shape
    key = (m_rows, n, k)
    if key in _GEMM_CACHE:
        return _GEMM_CACHE[key]
    g = torch.Generator(device=device).manual_seed(5)
    a = torch.randn(m_rows, k, generator=g, device=device).to(torch.bfloat16)
    out = torch.empty(m_rows, n, dtype=torch.bfloat16, device=device)
    ref = torch.mm(a, wts[0])
    record: dict[str, Any] = {}
    best: tuple[float, str] | None = None
    for be in ("cublas",) + _FI_GEMM_BACKENDS:
        try:
            if be == "cublas":
                def one(wt):
                    torch.mm(a, wt, out=out)
            else:
                def one(wt, be=be):
                    flashinfer.mm_bf16(a, wt, pdl=True, out=out, backend=be)
            one(wts[0])
            torch.cuda.synchronize()
            rel = _rel_l2(out, ref)
            if not rel <= _REL_L2_TOL:
                record[be] = f"invalid: rel_l2={rel:.3e}"
                continue

            def many():
                for wt in wts:
                    one(wt)
            ms = _time_graph(many, len(wts))
            record[be] = {"ms_per_call": ms, "rel_l2": rel,
                          "pdl_capable": be != "cublas" and _gemm_pdl_capable(be, m_rows)}
            if best is None or ms < best[0]:
                best = (ms, be)
        except Exception as e:  # noqa: BLE001
            record[be] = _err(e)
    assert best is not None, f"cuBLAS GEMM failed for {tag} {key}: {record}"
    record["winner"] = best[1]
    _GEMM_CACHE[key] = (best[1], record)
    return _GEMM_CACHE[key]


# --------------------------------------------------------------------------- Step
class FiGraphStep:
    arm = "fi-graph"
    precision = "bf16"

    def __init__(self, arm: "FiGraphArm", w: Workload, inputs: dict, variant: str,
                 device: str) -> None:
        import flashinfer
        self._fi = flashinfer
        self.variant, self.w = variant, w
        graph, pdl, gemm_mode = _FACTORS[variant]
        self.graph, self.pdl, self.gemm_mode = graph, pdl, gemm_mode
        self.config: dict[str, Any] = {}
        m = w.model
        self.m, self.eps = m, m.eps
        T, n = w.total_q, w.n_seq
        bf16 = torch.bfloat16
        dev = dict(device=device)
        self.inputs = inputs  # read-only references; keeps weights alive

        fused = arm._fused_weights(w, inputs)
        L = m.layers
        self.layers = [dict(
            ln1=inputs["ln1"][l], qn=inputs["qn"][l], kn=inputs["kn"][l], ln2=inputs["ln2"][l],
            qkv=fused["w_qkv"][l].t(), o=inputs["wo"][l].t(), gu=fused["w_gu"][l].t(),
            down=inputs["wd"][l].t()) for l in range(L)]
        self.fnorm, self.embed = inputs["fnorm"], inputs["embed"]
        self.lm_head = self.embed.t()

        # RoPE cos|sin cache (fp32, [max_pos, head_dim]; first half cos, second half sin).
        D = m.head_dim
        max_pos = max(max(x.kv_len + x.q_len for x in w.seqs), 4096)
        inv = m.rope_theta ** (-torch.arange(0, D, 2, dtype=torch.float64, **dev) / D)
        ang = torch.outer(torch.arange(max_pos, dtype=torch.float64, **dev), inv)
        self.cos_sin = torch.cat((ang.cos(), ang.sin()), -1).float().contiguous()

        self.kvs = PagedKV(w, inputs, device)
        self.b = self.kvs.batch
        self._check_batch_indices(device)

        # Static activation buffers (token-major).
        self.tok = inputs["tokens"].to(device=device, dtype=torch.int64).clone()
        self.next = torch.zeros(n, dtype=torch.int64, **dev)
        self.res = torch.empty(T, m.hidden, dtype=bf16, **dev)
        self.h = torch.empty(T, m.hidden, dtype=bf16, **dev)
        self.qkv = torch.empty(T, m.q_dim + 2 * m.kv_dim, dtype=bf16, **dev)
        self.q_n = torch.empty(T, m.q_dim, dtype=bf16, **dev)
        self.k_n = torch.empty(T, m.kv_dim, dtype=bf16, **dev)
        self.attn = torch.empty(T, m.q_heads, D, dtype=bf16, **dev)
        self.gu = torch.empty(T, 2 * m.intermediate, dtype=bf16, **dev)
        self.act = torch.empty(T, m.intermediate, dtype=bf16, **dev)
        self.logits_bf16 = torch.empty(n, m.vocab, dtype=bf16, **dev)
        self.need_gather = T != n
        if self.need_gather:
            self.last_idx = torch.tensor([c - 1 for c in self.b.cu_q[1:]], dtype=torch.int64,
                                         **dev)
            self.h_last = torch.empty(n, m.hidden, dtype=bf16, **dev)
            self.res_last = torch.empty(n, m.hidden, dtype=bf16, **dev)
        # Views (strided slices of the fused QKV output; the kernels take row strides).
        self.q_in = self.qkv[:, :m.q_dim].view(T, m.q_heads, D)
        self.k_in = self.qkv[:, m.q_dim:m.q_dim + m.kv_dim].view(T, m.kv_heads, D)
        self.v_in = self.qkv[:, m.q_dim + m.kv_dim:].view(T, m.kv_heads, D)
        self.q_out3 = self.q_n.view(T, m.q_heads, D)
        self.k_out3 = self.k_n.view(T, m.kv_heads, D)
        self.attn2 = self.attn.view(T, m.q_dim)

        # Attention backend: autotuned once per workload (PDL on), fixed for every variant.
        self.attn_cfg, attn_rec = _autotune_attention(w, self.kvs, device)
        self._attn_run = _make_attn(self.attn_cfg, w, self.b, device, _ws_bytes(w))

        # GEMM backends: "best" = autotuned winner per group (may be cuBLAS), else cuBLAS.
        shapes = gemm_shapes(w)
        self._gemm_be: dict[str, str] = {tag: "cublas" for tag in shapes}
        gemm_rec: dict[str, Any] = {}
        if gemm_mode == "best":
            for tag, wts in (("qkv", [x["qkv"] for x in self.layers]),
                             ("o", [x["o"] for x in self.layers]),
                             ("gu", [x["gu"] for x in self.layers]),
                             ("down", [x["down"] for x in self.layers]),
                             ("lm_head", [self.lm_head])):
                be, rec = _autotune_gemm(tag, wts, shapes[tag][0], device)
                self._gemm_be[tag], gemm_rec[tag] = be, rec
        self._shapes = shapes

        self._ops: dict[str, list] = {}
        self._recording = True
        self._graph_obj = None
        self._warm_and_capture()
        self._recording = False
        self._fill_config(attn_rec, gemm_rec)

    def _check_batch_indices(self, device: str) -> None:
        """Cross-check flashinfer.get_batch_indices_positions against the torch construction."""
        b = self.b
        src = "flashinfer"
        try:
            bi, pos = self._fi.get_batch_indices_positions(b.qo_indptr, b.seq_lens, b.total_q)
            if not (torch.equal(bi.to(torch.int32), b.batch_idx)
                    and torch.equal(pos.to(torch.int32), b.positions)):
                src = "torch (flashinfer.get_batch_indices_positions MISMATCH)"
        except Exception as e:  # noqa: BLE001
            src = f"torch (flashinfer raised {_err(e)})"
        self.batch_indices_source = src

    # -- op helpers --------------------------------------------------------
    def _note(self, op: str, supported: bool) -> None:
        if self._recording:
            e = self._ops.setdefault(op, [0, supported])
            e[0] += 1

    def _mm(self, tag: str, a: torch.Tensor, wt: torch.Tensor, out: torch.Tensor) -> None:
        be = self._gemm_be[tag]
        self._note(f"gemm_{tag}", be != "cublas" and _gemm_pdl_capable(be, self._shapes[tag][0]))
        if be == "cublas":
            torch.mm(a, wt, out=out)
        else:
            self._fi.mm_bf16(a, wt, pdl=self.pdl, out=out, backend=be)

    def _step(self) -> None:
        """Enqueue one forward step. No sync, no allocation (beyond the caching allocator in
        eager mode)."""
        fi, pdl, kvs, b, m = self._fi, self.pdl, self.kvs, self.b, self.m
        res, h, eps = self.res, self.h, self.eps
        self._note("embedding", False)
        torch.index_select(self.embed, 0, self.tok, out=res)
        for l, lw in enumerate(self.layers):
            if l == 0:
                self._note("rmsnorm", True)
                fi.rmsnorm(res, lw["ln1"], eps, out=h, enable_pdl=pdl)
            else:
                self._note("fused_add_rmsnorm", True)
                fi.fused_add_rmsnorm(h, res, lw["ln1"], eps, enable_pdl=pdl)
            self._mm("qkv", h, lw["qkv"], self.qkv)
            self._note("qk_rmsnorm", True)
            fi.rmsnorm(self.q_in, lw["qn"], eps, out=self.q_out3, enable_pdl=pdl)
            fi.rmsnorm(self.k_in, lw["kn"], eps, out=self.k_out3, enable_pdl=pdl)
            self._note("rope", False)
            fi.apply_rope_with_cos_sin_cache_inplace(
                b.positions, self.q_n, self.k_n, m.head_dim, self.cos_sin, True)
            self._note("append_paged_kv", False)
            fi.append_paged_kv_cache(
                self.k_out3, self.v_in, b.batch_idx, b.positions, kvs.kv[l],
                b.kv_indices, b.kv_indptr, b.last_page_len, kv_layout="HND")
            self._note("attention", True)
            self._attn_run(kvs.kv[l], self.q_out3, self.attn, pdl)
            self._mm("o", self.attn2, lw["o"], h)
            self._note("fused_add_rmsnorm", True)
            fi.fused_add_rmsnorm(h, res, lw["ln2"], eps, enable_pdl=pdl)
            self._mm("gu", h, lw["gu"], self.gu)
            self._note("silu_and_mul", True)
            fi.silu_and_mul(self.gu, out=self.act, enable_pdl=pdl)
            self._mm("down", self.act, lw["down"], h)
        if self.need_gather:
            self._note("gather_last_rows", False)
            torch.index_select(h, 0, self.last_idx, out=self.h_last)
            torch.index_select(res, 0, self.last_idx, out=self.res_last)
            hf, rf = self.h_last, self.res_last
        else:
            hf, rf = h, res
        self._note("fused_add_rmsnorm", True)
        fi.fused_add_rmsnorm(hf, rf, self.fnorm, eps, enable_pdl=pdl)
        self._mm("lm_head", hf, self.lm_head, self.logits_bf16)
        self._note("argmax", False)
        torch.argmax(self.logits_bf16, -1, out=self.next)

    def _warm_and_capture(self) -> None:
        cur = torch.cuda.current_stream()
        s = torch.cuda.Stream()
        s.wait_stream(cur)
        with torch.cuda.stream(s):  # warmup also triggers every JIT build
            self._step()  # first step records the PDL op accounting
            self._recording = False
            self._step()
            self._step()
        cur.wait_stream(s)
        torch.cuda.synchronize()
        if not self.graph:
            return
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            self._step()
        self._graph_obj = g
        torch.cuda.synchronize()

    def _fill_config(self, attn_rec: dict, gemm_rec: dict) -> None:
        fi = self._fi
        try:
            norm_impl = "cuda-jit" if fi.norm._use_cuda_norm(torch.device("cuda")) else "cute-dsl"
        except Exception:  # noqa: BLE001
            norm_impl = "unknown"
        self.config.update({
            "graph": self.graph, "pdl_enabled": self.pdl, "gemm_mode": self.gemm_mode,
            "gemm": dict(self._gemm_be), "gemm_autotune": gemm_rec or None,
            "attn_backend": self.attn_cfg["name"], "attn_autotune": attn_rec,
            "pdl": pdl_summary(self._ops, self.pdl),
            "paged_kv": {"page": PAGE, "layout": "HND", "pages": self.b.n_pages},
            "launches_per_step": 1, "proxy": False,
            "logits": "bf16 from the LM-head GEMM, argmax timed on bf16, fp32 cast untimed",
            "batch_indices_source": self.batch_indices_source,
            "env_FLASHINFER_USE_CUDA_NORM": os.environ.get("FLASHINFER_USE_CUDA_NORM"),
            "flashinfer_version": getattr(fi, "__version__", "unknown"),
            "norm_impl": norm_impl, "torch_version": torch.__version__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        })

    # -- Step protocol -------------------------------------------------------
    def launch(self) -> None:
        if self._graph_obj is not None:
            self._graph_obj.replay()
        else:
            self._step()

    def outputs(self) -> dict[str, Any]:
        return {"logits": self.logits_bf16.float(), "next_token": self.next.clone(),
                "k_new": self.kvs.gather_new(0).contiguous(),
                "v_new": self.kvs.gather_new(1).contiguous()}

    def close(self) -> None:
        self._graph_obj = None
        for name in ("layers", "kvs", "b", "_attn_run", "inputs", "res", "h", "qkv", "q_n", "k_n",
                     "attn", "gu", "act", "logits_bf16", "cos_sin", "q_in", "k_in", "v_in",
                     "q_out3", "k_out3", "attn2"):
            setattr(self, name, None)


# --------------------------------------------------------------------------- Arm
class FiGraphArm:
    name = "fi-graph"

    def __init__(self, **options: Any) -> None:
        self.options = options
        self._weights: dict[tuple, dict] = {}

    def variants(self) -> tuple[str, ...]:
        return VARIANTS

    def supports(self, w: Workload, variant: str) -> str | None:
        if variant not in _FACTORS:
            return f"unknown variant {variant!r}"
        if variant.startswith("eager") and w.regime != "R0":
            return "eager variant is only measured at R0"
        try:
            if not torch.cuda.is_available():
                return "CUDA device required"
            major, _ = torch.cuda.get_device_capability()
        except Exception as e:  # noqa: BLE001
            return f"torch/CUDA unavailable: {_err(e)}"
        if _FACTORS[variant][1] and major < 9:
            return f"PDL needs compute capability >= 9 (got {major}.x)"
        return None

    def _fused_weights(self, w: Workload, inputs: dict) -> dict:
        """Fused QKV / gate-up weights, shared by the variants of one workload; only the latest
        workload's copy is kept (the entry pins the source tensors so data_ptr keys stay valid)."""
        key = (w.id, inputs["wq"].data_ptr(), inputs["wg"].data_ptr())
        if key not in self._weights:
            self._weights.clear()
            self._weights[key] = {
                "w_qkv": torch.cat((inputs["wq"], inputs["wk"], inputs["wv"]), 1).contiguous(),
                "w_gu": torch.cat((inputs["wg"], inputs["wu"]), 1).contiguous(),
                "_keep": (inputs["wq"], inputs["wg"])}
        return self._weights[key]

    def prepare(self, w: Workload, inputs: dict, variant: str, device: str) -> FiGraphStep:
        reason = self.supports(w, variant)
        if reason:
            raise RuntimeError(f"unsupported: {reason}")
        return FiGraphStep(self, w, inputs, variant, device)
