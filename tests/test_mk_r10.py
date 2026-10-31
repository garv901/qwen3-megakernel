"""R10 workloads (decode, 33-256 requests, long KV): preset, memory caps, real trace steps, mk plan / build choices (CPU only)."""
from regimebench.arms.mk import MkArm
from regimebench.kernels.mk import build as B
from regimebench.workload import (OWNER_POINTS, R10_CHECK_CAP, R10_KV_CAP, Workload, r10_points, r10_trace_steps,
                                  regime_grid, select)


def test_preset_resolves_to_r10_grid():
    pts = select(["mk_r10"])
    assert [w.id for w in pts] == [w.id for w in r10_points()] == list(OWNER_POINTS["mk_r10"])
    assert len(pts) == 19 and len({w.id for w in pts}) == 19
    assert {w.id for w in pts} <= {w.id for w in regime_grid()}
    assert all(w.regime == "R10" for w in pts)


def test_points_are_r10_decode_shapes_under_memory_caps():
    for w in r10_points():
        assert w.is_pure_decode and 33 <= w.n_seq <= 256 and w.total_q <= B.BUCKETS[-1], w.id
        assert w.kv_cache_bytes <= (R10_CHECK_CAP if w.check else R10_KV_CAP), w.id
    assert sum(w.check for w in r10_points()) == 3


def test_trace_steps_are_heterogeneous_and_roundtrip():
    steps = r10_trace_steps()
    assert len(steps) >= 2
    by_id = {w.id: w for w in r10_points()}
    for st in steps:
        w = by_id[st["id"]]
        assert [s.kv_len for s in w.seqs] == st["kv"] and "steps.jsonl line" in st["source"]
        assert max(st["kv"]) >= 4 * min(st["kv"])   # per-sequence kv differs (not a uniform point)
        assert Workload.from_dict(w.to_dict()) == w


def test_mk_supports_and_plans_every_r10_point():
    """Every R10 point: mk supports it, auto picks splitkv (q_len 1, kv >= 512), the splitkv plan matches mk's layout, skv_tune resolves."""
    from regimebench.kernels.common.splitkv import plan as P
    arm = MkArm()
    for w in r10_points():
        assert arm.supports(w, "auto") is None, w.id
        assert arm.resolve_attn(w) == "splitkv", w.id
        seqs = [(s.kv_len, s.q_len) for s in w.seqs]
        pl = P.plan(seqs, n_cta=B.GRID)
        d = B.build_descriptors(seqs, B.MAXTOK)
        assert list(pl.row_base) == list(d.t_base) and pl.total_q == w.total_q and pl.cache_rows == d.cache_rows == w.cache_tokens, w.id
        npad = B.bucket_for(w.total_q)
        assert npad in (64, 128, 256)
        assert isinstance(B.skv_tune(npad, 1, seqs, tc=1), tuple)
