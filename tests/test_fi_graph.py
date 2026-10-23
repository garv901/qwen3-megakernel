"""CPU tests for the fi-graph arm: paged-KV layout, index math, reference attention, candidates."""

from __future__ import annotations

import math

import torch

from regimebench.arms import fi_graph as fg
from regimebench.arms.base import load_arm
from regimebench.workload import Seq, Workload, regime_grid, select

L, KVH, D = 28, 8, 128


def _wl(seqs, regime="R6", wid="t"):
    return Workload(wid, regime, tuple(Seq(*s) for s in seqs))


def _fixture(w, seed=0):
    g = torch.Generator().manual_seed(seed)
    cu = [0]
    for s in w.seqs:
        cu.append(cu[-1] + s.kv_len + s.q_len)
    k = torch.zeros(L, cu[-1], KVH, D, dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    for i, s in enumerate(w.seqs):
        for c in (k, v):
            c[:, cu[i]:cu[i] + s.kv_len] = (torch.randn(L, s.kv_len, KVH, D, generator=g)
                                            * 0.1).to(torch.bfloat16)
    return {"kcache": k, "vcache": v, "cu_cache": torch.tensor(cu)}


W = _wl([(40, 3), (5, 1), (0, 20), (16, 1), (33, 2)])


def test_batch_indices():
    b = fg.Batch([s.q_len for s in W.seqs], [s.kv_len for s in W.seqs], "cpu")
    exp_b, exp_p = [], []
    for i, s in enumerate(W.seqs):
        for j in range(s.q_len):
            exp_b.append(i)
            exp_p.append(s.kv_len + j)
    assert b.batch_idx.tolist() == exp_b and b.positions.tolist() == exp_p
    assert b.qo_indptr.tolist() == W.cu_q
    assert b.seq_lens.tolist() == [s.kv_len + s.q_len for s in W.seqs]
    assert b.kv_indptr.tolist()[-1] == b.n_pages == b.kv_indices.numel()
    for i, t in enumerate(b.totals):
        assert b.last_page_len[i].item() == (t - 1) % 16 + 1
        assert (b.pages[i] - 1) * 16 + b.last_page_len[i].item() == t
        row = b.block_tables[i, :b.pages[i]].tolist()
        assert row == list(range(b.page0[i], b.page0[i] + b.pages[i]))
    # token -> (page, slot) is the (position // 16, position % 16) of its own sequence
    for t, (i, p) in enumerate(zip(exp_b, exp_p)):
        assert b.tok_page[t].item() == b.page0[i] + p // 16 and b.tok_slot[t].item() == p % 16
    # cumulative kv lens for trtllm-gen / cudnn
    assert b.cu_kv.tolist()[-1] == sum(b.totals)


def test_sub_batch_rebases_pages():
    b = fg.Batch([4, 1, 1], [10, 20, 30], "cpu")
    s = b.sub(1, 3)
    assert s.page_base == b.page0[1] and s.kv_indices[0].item() == b.page0[1]
    assert s.kv_indptr.tolist() == [0, s.pages[0], s.pages[0] + s.pages[1]]
    assert s.qo_indptr.tolist() == [0, 1, 2]
    assert s.positions.tolist() == [20, 30]


def test_paged_kv_roundtrip_and_gather():
    inputs = _fixture(W)
    kvs = fg.PagedKV(W, inputs, "cpu")
    b = kvs.batch
    assert tuple(kvs.kv.shape) == (L, b.n_pages, 2, KVH, 16, D)
    cu = inputs["cu_cache"].tolist()
    for i, s in enumerate(W.seqs):
        for which, name in enumerate(("kcache", "vcache")):
            blk = kvs.kv[:, b.page0[i]:b.page0[i] + b.pages[i], which]  # [L,np,KVH,16,D]
            flat = blk.permute(0, 1, 3, 2, 4).reshape(L, -1, KVH, D)
            assert torch.equal(flat[:, :s.kv_len], inputs[name][:, cu[i]:cu[i] + s.kv_len])
            assert not flat[:, s.kv_len:].any()  # new slots and page padding start zero
    gen = torch.Generator().manual_seed(3)
    kvs.fill_new_slots(0, gen)
    vals = (torch.randn(b.total_q, 2, KVH, D, generator=torch.Generator().manual_seed(3))
            * 0.1).to(torch.bfloat16)
    for which in (0, 1):
        got = kvs.gather_new(which)
        assert tuple(got.shape) == (W.total_q, L, KVH, D)
        assert torch.equal(got[:, 0], vals[:, which])
        assert not got[:, 1:].any()


def _naive_attention(inputs, w, q, seq, scale):
    """Per-token loop on the fixture layout (independent of paging)."""
    cu = inputs["cu_cache"].tolist()
    s = w.seqs[seq]
    k = inputs["kcache"][0, cu[seq]:cu[seq + 1]].float()
    v = inputs["vcache"][0, cu[seq]:cu[seq + 1]].float()
    off = w.cu_q[seq]
    outs = []
    for j in range(s.q_len):
        n = s.kv_len + j + 1
        qq = q[off + j].float()  # [QH, D]
        o = torch.zeros_like(qq)
        for h in range(qq.shape[0]):
            kh = h // (qq.shape[0] // KVH)
            p = (k[:n, kh] @ qq[h] * scale).softmax(0)
            o[h] = p @ v[:n, kh]
        outs.append(o)
    return torch.stack(outs)


def test_ref_attention_matches_naive_loop():
    w = _wl([(40, 3), (5, 1), (0, 20), (16, 4)])
    inputs = _fixture(w, seed=1)
    kvs = fg.PagedKV(w, inputs, "cpu")
    b = kvs.batch
    # New-token K/V are also needed in the fixture view for the naive loop.
    g = torch.Generator().manual_seed(9)
    cu = inputs["cu_cache"].tolist()
    for i, s in enumerate(w.seqs):
        for c, name in ((0, "kcache"), (1, "vcache")):
            new = (torch.randn(L, s.q_len, KVH, D, generator=g) * 0.1).to(torch.bfloat16)
            inputs[name][:, cu[i] + s.kv_len:cu[i + 1]] = new
            for t in range(s.q_len):
                pos = s.kv_len + t
                kvs.kv[:, b.page0[i] + pos // 16, c, :, pos % 16] = new[:, t]
    q = torch.randn(w.total_q, 16, D, generator=g).to(torch.bfloat16)
    scale = 1 / math.sqrt(D)
    got = fg.ref_attention(kvs.kv[0], q, b, list(range(w.n_seq)), scale, chunk=3)
    for i in range(w.n_seq):
        exp = _naive_attention(inputs, w, q, i, scale)
        assert (got[i] - exp).abs().max().item() < 2e-5, i


def test_check_seq_ids():
    assert fg.check_seq_ids(3) == [0, 1, 2]
    ids = fg.check_seq_ids(256)
    assert ids[0] == 0 and ids[-1] == 255 and len(ids) <= 8 and ids == sorted(set(ids))


def _grid(i):
    return next(w for w in regime_grid() if w.id == i)


def test_split_point():
    assert fg.split_point(_grid("R7-p512-d8")) == 1
    assert fg.split_point(_grid("R0-b1-kv128")) is None
    assert fg.split_point(_grid("R5-b4-T512")) is None
    assert fg.split_point(_wl([(10, 1), (10, 4)])) is None  # decode before prefill


def test_attention_candidates_per_regime():
    names = lambda w: [fg.cfg_name(c) for c in fg.attention_candidates(w)]
    for w in regime_grid():
        c = names(w)
        assert len(c) == len(set(c)), w.id
        if w.is_pure_decode:
            assert "decode/fa2/tc=False" in c and "decode/trtllm-direct" in c
            assert "decode/cute-dsl/tc=True" in c and "decode/auto/tc=False" in c
            assert not any(n.startswith("prefill") or n.startswith("split") for n in c)
        else:
            for n in ("prefill/fa2", "prefill/fa3", "prefill/trtllm-gen",
                      "prefill/trtllm-direct", "prefill/cudnn-direct"):
                assert n in c, (w.id, n)
            assert not any(n.startswith("decode") for n in c)
            has_split = any(n.startswith("split") for n in c)
            assert has_split == (w.regime == "R7"), w.id
    r7 = names(_grid("R7-p512-d32"))
    assert "split[prefill/fa2+decode/fa2/tc=False]" in r7
    assert "split[prefill/trtllm-gen+decode/trtllm-direct]" in r7
    assert len(r7) == 5 + 6


def test_static_skip_matches_upstream_rule():
    assert fg._static_skip({"kind": "decode-wrapper", "backend": "trtllm-gen", "tc": False})
    assert fg._static_skip({"kind": "decode-wrapper", "backend": "fa2", "tc": False}) is None
    assert fg._static_skip({"kind": "decode-direct"}) is None


def test_gemm_shapes():
    w = _grid("R7-p512-d8")
    s = fg.gemm_shapes(w)
    assert s["qkv"] == (520, 4096, 1024) and s["o"] == (520, 1024, 2048)
    assert s["gu"] == (520, 6144, 1024) and s["down"] == (520, 1024, 3072)
    assert s["lm_head"] == (9, 151936, 1024)
    assert fg.gemm_shapes(_grid("R5-b16-T2048"))["qkv"][0] == 16 * 2048


def test_workspace_scales_with_tokens():
    assert fg._ws_bytes(_grid("R0-b1-kv128")) == 128 << 20
    big = fg._ws_bytes(_grid("R5-b16-T2048"))
    assert (128 << 20) < big <= 2 << 30
    assert big == min(16 * 2048 * 16 * 128 * 16, 2 << 30)


def test_gemm_pdl_capability():
    assert fg._gemm_pdl_capable("tgv", 4096) and fg._gemm_pdl_capable("tinygemm", 8)
    assert fg._gemm_pdl_capable("cute-dsl", 32) and not fg._gemm_pdl_capable("cute-dsl", 33)
    assert not fg._gemm_pdl_capable("cudnn", 1) and not fg._gemm_pdl_capable("cutlass", 1)


def test_pdl_summary():
    s = fg.pdl_summary({"a": [3, True], "b": [2, False]}, True)
    assert (s["launches_with_pdl"], s["launches_total"]) == (3, 5)
    s = fg.pdl_summary({"a": [3, True], "b": [2, False]}, False)
    assert (s["launches_with_pdl"], s["launches_total"]) == (0, 5)
    assert s["per_op"]["a"]["pdl_supported"] and not s["per_op"]["a"]["pdl_requested"]


def test_arm_variants_and_supports():
    arm = load_arm("fi-graph")
    assert arm.name == "fi-graph"
    assert arm.variants()[0] == "graph-pdl-best"
    assert set(arm.variants()) == {"graph-pdl-best", "graph-pdl-cublas", "graph-nopdl-best",
                                   "graph-nopdl-cublas", "eager-pdl-best"}
    r0, r1 = _grid("R0-b1-kv128"), _grid("R1-b8-kv128")
    assert "unknown" in arm.supports(r0, "nope")
    assert "R0" in arm.supports(r1, "eager-pdl-best")
    reason = arm.supports(r0, "eager-pdl-best")
    assert (reason is None) == torch.cuda.is_available()
    if not torch.cuda.is_available():
        assert "CUDA" in reason
        assert "CUDA" in arm.supports(r0, "graph-pdl-best")


def test_every_grid_point_has_paged_capacity():
    for w in select(["R0", "R7", "R8"]):
        b = fg.Batch([s.q_len for s in w.seqs], [s.kv_len for s in w.seqs], "cpu")
        assert b.n_pages * 16 >= w.cache_tokens
        assert b.n_pages * 16 < w.cache_tokens + 16 * w.n_seq
