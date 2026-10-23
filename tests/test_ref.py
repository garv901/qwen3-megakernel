import pytest
import torch

from regimebench.fixture import make_inputs
from regimebench.geometry import ModelGeometry
from regimebench.ref import reference
from regimebench.workload import Seq, Workload

TINY = ModelGeometry("tiny", hidden=32, layers=2, q_heads=4, kv_heads=2, head_dim=8,
                     intermediate=48, vocab=100)


def tiny(seqs, wid="t"):
    return Workload(wid, "RX", tuple(Seq(*s) for s in seqs), model=TINY)


def test_shapes_and_dtypes():
    w = tiny([(5, 1), (0, 4), (7, 3)])
    inp = make_inputs(w, 1, "cpu")
    out = reference.forward(w, inp, torch.bfloat16)
    assert out["logits"].shape == (3, 100) and out["logits"].dtype == torch.float32
    assert out["next_token"].shape == (3,) and out["next_token"].dtype == torch.int64
    assert out["k_new"].shape == (8, 2, 2, 8) and out["k_new"].dtype == torch.bfloat16
    out32 = reference.forward(w, inp, torch.float32)
    assert out32["k_new"].dtype == torch.float32


def test_does_not_mutate_inputs():
    w = tiny([(5, 2)])
    inp = make_inputs(w, 2, "cpu")
    before = {k: v.clone() for k, v in inp.items()}
    reference.forward(w, inp, torch.float32)
    assert all(torch.equal(before[k], inp[k]) for k in before)


def test_multi_token_equals_sequential_decode():
    """q_len=3 over kv=6 == three decode steps where earlier new K/V are in the cache."""
    kv, q = 6, 3
    w = tiny([(kv, q)])
    inp = make_inputs(w, 3, "cpu")
    out = reference.forward(w, inp, torch.float32)
    for j in range(q):
        wj = tiny([(kv + j, 1)])
        ij = make_inputs(wj, 3, "cpu")  # same weights (same seed)
        ij["tokens"] = inp["tokens"][j:j + 1]
        for name, new in (("kcache", out["k_new"]), ("vcache", out["v_new"])):
            ij[name] = torch.zeros_like(ij[name], dtype=torch.float32)
            ij[name][:, :kv] = inp[name][:, :kv]
            for jj in range(j):
                ij[name][:, kv + jj] = new[jj]
        oj = reference.forward(wj, ij, torch.float32)
        assert torch.allclose(oj["k_new"][0], out["k_new"][j], atol=1e-5)
        if j == q - 1:
            assert torch.allclose(oj["logits"][0], out["logits"][0], atol=1e-4)


def test_ragged_matches_individual_sequences():
    seqs = [(4, 1), (9, 2), (0, 3)]
    w = tiny(seqs)
    inp = make_inputs(w, 4, "cpu")
    out = reference.forward(w, inp, torch.float32)
    cu_q = w.cu_q
    for i, s in enumerate(seqs):
        wi = tiny([s])
        ii = make_inputs(wi, 4, "cpu")
        base = int(inp["cu_cache"][i])
        for name in ("kcache", "vcache"):
            ii[name] = torch.zeros_like(ii[name])
            ii[name][:, :s[0]] = inp[name][:, base:base + s[0]]
        ii["tokens"] = inp["tokens"][cu_q[i]:cu_q[i + 1]]
        oi = reference.forward(wi, ii, torch.float32)
        assert torch.allclose(oi["logits"][0], out["logits"][i], atol=1e-4)
        assert torch.allclose(oi["k_new"], out["k_new"][cu_q[i]:cu_q[i + 1]], atol=1e-5)


def test_chunked_attention_matches_unchunked(monkeypatch):
    w = tiny([(3, 9)])
    inp = make_inputs(w, 5, "cpu")
    a = reference.forward(w, inp, torch.float32)
    monkeypatch.setattr(reference, "_SCORE_ELEMS", 4 * 8)  # forces 1-3 query chunks
    b = reference.forward(w, inp, torch.float32)
    assert torch.allclose(a["logits"], b["logits"], atol=1e-5)
    assert torch.allclose(a["v_new"], b["v_new"], atol=1e-5)


def test_causal_future_tokens_do_not_matter():
    """Changing a later new token must not change an earlier token's K/V."""
    w = tiny([(2, 4)])
    inp = make_inputs(w, 6, "cpu")
    a = reference.forward(w, inp, torch.float32)
    inp2 = dict(inp)
    inp2["tokens"] = inp["tokens"].clone()
    inp2["tokens"][3] = (inp2["tokens"][3] + 1) % 100
    b = reference.forward(w, inp2, torch.float32)
    assert torch.allclose(a["k_new"][:3], b["k_new"][:3], atol=1e-6)
    assert not torch.allclose(a["k_new"][3], b["k_new"][3])


def test_bf16_close_to_fp32():
    w = tiny([(8, 1), (3, 2)])
    inp = make_inputs(w, 7, "cpu")
    o16 = reference.forward(w, inp, torch.bfloat16)
    o32 = reference.forward(w, inp, torch.float32)
    rel = (o16["logits"] - o32["logits"]).norm() / o32["logits"].norm()
    assert rel < 0.2
