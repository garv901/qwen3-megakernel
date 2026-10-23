"""CPU-only tests for hv2-repeat / hv2g (NVRTC compile checks, descriptors, chunking, RoPE table, arm gating)."""

import math

import pytest
import torch

from regimebench.arms.hv2_repeat import MAX_LAUNCHES, Hv2RepeatArm, hv2_sha256
from regimebench.arms.hv2g import Hv2gArm
from regimebench.kernels.hv2g import build as B
from regimebench.workload import Seq, Workload, regime_grid, select


@pytest.fixture(scope="module")
def builds():
    pytest.importorskip("cuda.bindings.nvrtc")
    return {mt1: B.compile_build(mt1, 148) for mt1 in (True, False)}


def test_mt1_compiles_without_spills(builds):
    cubin, info, log = builds[True]
    assert len(cubin) > 1000
    assert info["regs"] <= 128
    assert info["spill_stores"] == 0 and info["spill_loads"] == 0, info


def test_mtn_compiles(builds):
    cubin, info, log = builds[False]
    assert len(cubin) > 1000
    assert info["regs"] <= 128, info


def test_mt1_dyn_smem_fits_one_block_per_sm(builds):
    for mt1 in (True, False):
        assert builds[mt1][1]["static_smem"] + B.dyn_smem(mt1) <= 227 * 1024


def test_hv2_frozen_sha():
    assert hv2_sha256().startswith("4253a0db")


def test_rope_table_matches_formula():
    t = B.rope_table(300)
    assert t.shape == (300, 128) and t.dtype == torch.float32
    p, i = 257, 5
    inv = 1e6 ** (-(2 * i) / 128)
    assert math.isclose(t[p, i].item(), math.cos(p * inv), rel_tol=1e-6, abs_tol=1e-7)
    assert math.isclose(t[p, 64 + i].item(), math.sin(p * inv), rel_tol=1e-6, abs_tol=1e-7)
    assert torch.allclose(t[0, :64], torch.ones(64)) and torch.allclose(t[0, 64:], torch.zeros(64))


def test_descriptors_decode_batch():
    d = B.build_descriptors([(128, 1)] * 3)
    assert d.t_base == [0, 129, 258] and d.t_vis == [129] * 3
    assert d.lm_tok == [0, 1, 2] and d.lm_seq == [0, 1, 2]
    assert len(d.chunks) == 1 and d.cache_rows == 387 and d.max_pos == 129


def test_descriptors_prefill_causal_visibility():
    d = B.build_descriptors([(10, 4), (0, 2)])
    assert d.t_vis == [11, 12, 13, 14, 1, 2]
    assert d.t_base == [0] * 4 + [14] * 2
    assert d.lm_tok == [3, 5] and d.lm_seq == [0, 1]


def test_descriptors_chunking_large_step():
    d = B.build_descriptors([(0, 8192)], B.MAXTOK)
    assert len(d.chunks) == 32 and all(c.ntok == B.MAXTOK for c in d.chunks)
    assert [c.t0 for c in d.chunks] == list(range(0, 8192, B.MAXTOK))
    assert sum(len(c.lm_tok) for c in d.chunks) == 1
    last = d.chunks[-1]
    assert last.lm_tok == [B.MAXTOK - 1] and last.lm_seq == [0] and last.lm_off == 0
    assert d.t_vis[-1] == 8192


def test_descriptors_chunk_local_lists_ragged():
    seqs = [(100, 1)] * 600
    d = B.build_descriptors(seqs, 256)
    assert [c.ntok for c in d.chunks] == [256, 256, 88]
    assert [c.lm_off for c in d.chunks] == [0, 256, 512]
    assert d.chunks[1].lm_tok == list(range(256)) and d.chunks[1].lm_seq == list(range(256, 512))
    assert d.lm_seq == list(range(600))


def test_descriptors_every_grid_workload():
    for w in regime_grid():
        d = B.build_descriptors([(s.kv_len, s.q_len) for s in w.seqs])
        assert d.cache_rows == w.cache_tokens
        assert len(d.t_vis) == w.total_q
        assert all(c.ntok <= B.MAXTOK for c in d.chunks)
        assert sum(len(c.lm_tok) for c in d.chunks) == w.n_seq == len(d.lm_seq)
        assert sorted(d.lm_seq) == list(range(w.n_seq))
        assert max(d.t_vis) == w.max_kv + w.max_q or d.max_pos <= w.max_kv + w.max_q


def test_arm_gating():
    arm = Hv2gArm()
    assert arm.variants()[0] == "auto"
    w = select(["R0-b1-kv128"])[0]
    assert arm.supports(w, "auto") is None and arm.supports(w, "mtn") is None
    assert arm.supports(w, "auto-direct") is None
    assert arm.supports(w, "bogus") is not None
    rep = Hv2RepeatArm()
    assert rep.supports(w, "graph") is None
    big = Workload("x", "R4", (Seq(0, MAX_LAUNCHES + 1),))
    assert rep.supports(big, "graph") is not None
    assert rep.supports(Workload("y", "R4", (Seq(0, MAX_LAUNCHES),)), "direct") is None


def test_use_mt1_selection():
    from regimebench.arms.hv2g import use_mt1
    assert use_mt1(select(["R0-b1-kv128"])[0], "auto")
    assert not use_mt1(select(["R0-b1-kv128"])[0], "mtn")
    assert not use_mt1(select(["R1-b8-kv128"])[0], "auto")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (run scripts/gpu_check_hv2g.py)")
def test_gpu_smoke_placeholder():
    pytest.skip("GPU validation lives in scripts/gpu_check_hv2g.py")
