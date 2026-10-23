import pytest

from regimebench.sol import (SPEC_BW, SPEC_PEAK, byte_breakdown, flop_breakdown, fp32_ffma_peak,
                             sol, sol_both)
from regimebench.workload import Seq, Workload, regime_grid


def by_id(i):
    return next(w for w in regime_grid() if w.id == i)


def test_r0_matches_upstream_floor():
    s = sol(by_id("R0-b1-kv128"))
    assert s["bytes"] == 1_207_502_352
    assert s["t_bw_us"] == pytest.approx(150.94, abs=0.01)
    assert s["bound"] == "memory"
    assert byte_breakdown(by_id("R0-b1-kv128"))["embed_lm_head_once"] == 311_164_928


def test_grid_bounds_and_monotonic():
    assert sol(by_id("R4-b1-T16"))["bound"] == "memory"
    assert sol(by_id("R5-b16-T2048"))["bound"] == "compute"
    for w in regime_grid():
        s = sol(w)
        assert s["t_sol_us"] == max(s["t_bw_us"], s["t_fl_us"])
        assert s["bytes"] == sum(s["bytes_breakdown"].values())
        assert s["flops"] == sum(s["flops_breakdown"].values())
    assert sol(by_id("R2-b1-kv8192"))["bytes"] > sol(by_id("R2-b1-kv1024"))["bytes"]


def test_flops_formula():
    w = Workload("x", "RX", (Seq(10, 3),))
    f = flop_breakdown(w)
    attended = 11 + 12 + 13
    assert f["attention"] == 4 * 28 * 16 * 128 * attended
    assert f["lm_head"] == 2 * 151936 * 1024
    per_layer = 2048 * 1024 * 2 + 1024 * 1024 * 2 + 3 * 3072 * 1024
    assert f["linear"] == 2 * per_layer * 28 * 3


def test_ridge_crossing():
    s = sol(by_id("R4-b1-T16"))
    assert 300 < s["ridge_tokens"] < 500


def test_measured_variant_and_fp32_peak():
    w = by_id("R0-b1-kv128")
    both = sol_both(w, 4e12, 1e15)
    assert both["spec"]["bw"] == SPEC_BW and both["spec"]["peak"] == SPEC_PEAK
    assert both["measured"]["t_bw_us"] == pytest.approx(2 * both["spec"]["t_bw_us"])
    assert sol_both(w)["measured"] is None
    assert fp32_ffma_peak(2e9) == 148 * 128 * 2 * 2e9
