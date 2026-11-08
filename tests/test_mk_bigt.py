"""MK_BIGT (step 1 of the large-T engine): CPU checks of the build, the host-side sizes and the arm's routing. GPU correctness runs separately."""

import pytest

from regimebench.arms.mk import MkArm
from regimebench.kernels.mk import build as B
from regimebench.workload import Seq, Workload


def _bigt(maxt=8192, **k):
    return tuple(sorted({"MK_BIGT": 1, "MK_MAXT": maxt, **k}.items()))


@pytest.mark.parametrize("maxt", (256, 1024, 8192))
def test_bigt_compiles_without_spills(maxt):
    _, info, _ = B.compile_build(B.BIGT_NPAD, knobs=_bigt(maxt))
    assert info["spill_stores"] == 0 and info["spill_loads"] == 0 and info["stack"] == 0
    assert info["regs"] <= 128


@pytest.mark.parametrize("ks", ({"MK_BIG_QKS": 4}, {"MK_BIG_OKS": 4}, {"MK_BIG_DKS": 8}, {"MK_BIG_DKS": 2}))
def test_bigt_split_k_knobs_compile(ks):
    B.compile_build(B.BIGT_NPAD, knobs=_bigt(2048, **ks))


def test_bigt_rejected_combinations():
    with pytest.raises(ValueError):
        B.compile_build(64, knobs=_bigt())
    with pytest.raises(ValueError):
        B.compile_build(128, knobs=_bigt(), attn="splitkv")
    with pytest.raises(RuntimeError):   # MK_MAXT must be a multiple of 128 (mk_common.cuh static_assert)
        B.compile_build(128, knobs=_bigt(1000))
    with pytest.raises(RuntimeError):   # MK_FIN must stay -1 (= 2 at NPAD 128)
        B.compile_build(128, knobs=_bigt(1024, MK_FIN=0))


def test_bigt_sizes():
    assert B.tk_size() == B.TK_SIZE == 192
    assert B.tk_size(8192) == 32 * 64 and B.tk_size(1000) == 32 * 8
    assert B.WS_ORDER == ("xb", "xb2", "wgu", "act", "ao")   # ao = map 12
    sp = dict(zip(B.WS_ORDER, B.workspace_map_specs(128, B.WS_ORDER, maxt=4096)))
    assert sp["xb"] == ([1024, 4096], [2048], [64, 128]) and sp["act"][0] == [3072, 4096] and sp["xb2"][0] == [1024, 4096]
    assert sp["ao"] == ([2048, 4096], [4096], [64, 128])
    assert B.workspace_map_specs() == [([1024, 256], [2048], [64, 128])]   # bucket default unchanged


def _w(seqs, wid="t"):
    return Workload(wid, "R7", tuple(seqs))


def test_bigt_routing_and_supports():
    arm = MkArm()
    small = _w([Seq(128, 1)] * 200)
    mixed = _w([Seq(4096, 2048)] + [Seq(1024, 1)] * 32)          # 2080 tokens, 33 LM rows
    assert not arm.use_bigt(small) and arm.use_bigt(mixed)
    assert arm.supports(mixed, "auto") is None
    assert "8192" in arm.supports(_w([Seq(0, 8193)]), "auto")
    assert "n_seq 300" in arm.supports(_w([Seq(16, 2)] * 300), "auto")
    assert "placeholder" in MkArm(attn="splitkv").supports(mixed, "auto")
    assert "bigt" in MkArm(bigt=0).supports(mixed, "auto")
    assert MkArm(bigt=1).use_bigt(small) and MkArm(bigt=1).supports(small, "auto") is None
    assert MkArm(maxt=2048).supports(mixed, "auto") is not None
    with pytest.raises(ValueError):
        MkArm(maxt=1000)


def test_bigt_lm_compaction_descriptors():
    """lm_inv is built from the single chunk of build_descriptors(.., maxt): one slot per sequence, at its last token."""
    seqs = [(4096, 2048), (1024, 1), (0, 300)]
    d = B.build_descriptors(seqs, 4096)
    assert len(d.chunks) == 1
    c = d.chunks[0]
    assert c.ntok == 2349 and c.lm_tok == [2047, 2048, 2348] and c.lm_seq == [0, 1, 2]
