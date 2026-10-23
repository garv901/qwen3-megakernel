"""CPU tests for the vLLM worker: planning, sizing, stats, and the step-construction logic
against a fake scheduler that mimics vLLM's token-budget semantics."""

import json
import random
from collections import Counter

import pytest

from regimebench.arms import vllm_worker as vw
from regimebench.workload import Seq, Workload, regime_grid, select


class FakeReq:
    def __init__(self, plen, max_tokens):
        self.num_prompt_tokens, self.max_tokens = plen, max_tokens
        self.num_computed_tokens, self.outputs = 0, 0


class FakeHandle:
    """Scheduler: requests in admission order, token budget, optional per-request threshold."""

    def __init__(self, budget=16384):
        self.reqs: dict[str, FakeReq] = {}
        self.budget0, self.budget = budget, budget
        self.thr = 0
        self.last, self.n = {}, 0

    def add(self, prompt_ids, max_tokens):
        self.n += 1
        rid = f"r{self.n}"
        self.reqs[rid] = FakeReq(len(prompt_ids), max_tokens)
        return rid

    def computed(self, rid):
        r = self.reqs.get(rid)
        return r.num_computed_tokens if r else 0

    def step_untimed(self):
        left = self.budget
        thr = self.thr if len(self.reqs) > 1 else 0
        sched = {}
        for rid, r in self.reqs.items():
            need = r.num_prompt_tokens + r.outputs - r.num_computed_tokens
            if thr and need > thr:
                need = thr
            n = min(need, left)
            if n > 0:
                sched[rid] = n
                left -= n
        for rid, n in sched.items():
            r = self.reqs[rid]
            r.num_computed_tokens += n
            if r.num_computed_tokens >= r.num_prompt_tokens + r.outputs:
                r.outputs += 1
        for rid in [k for k, r in self.reqs.items() if r.outputs >= r.max_tokens]:
            del self.reqs[rid]
        self.last = sched

    def step(self):
        self.step_untimed()
        return 0.001

    def last_scheduled(self):
        return self.last

    def last_gpu_ms(self):
        return 0.5 * len(self.last)

    def set_budget(self, n):
        self.budget = self.budget0 if n is None else min(n, self.budget0)

    def set_threshold(self, n):
        self.thr = n

    def abort(self, rids):
        for r in rids:
            self.reqs.pop(r, None)

    def has_unfinished(self):
        return bool(self.reqs)


def by_id(i):
    return select([i])[0]


@pytest.mark.parametrize("wid,kind", [
    ("R0-b1-kv128", "decode"), ("R1-b256-kv128", "decode"), ("R8-b32-ragged", "decode"),
    ("R4-b1-T1024", "prefill"), ("R5-b16-T2048", "prefill"),
    ("R6-b1-kv1024-q4", "chunk"), ("R7-p512-d8", "chunk")])
def test_plan_kinds(wid, kind):
    assert vw.plan_workload(by_id(wid)).kind == kind


def test_every_grid_point_plans_and_sizes():
    for w in regime_grid():
        p = vw.plan_workload(w)
        sz = vw.engine_sizing(w)
        assert sz["max_num_batched_tokens"] >= w.total_q
        assert sz["max_num_seqs"] >= w.n_seq
        assert sz["max_model_len"] >= w.max_kv + w.max_q + 64
        # KV reservation covers the workload's cache with room to spare
        assert sz["kv_cache_memory_bytes"] > w.kv_cache_bytes
        assert sum(p.expected.values()) >= w.n_seq


@pytest.mark.parametrize("wid", ["R0-b1-kv128", "R1-b8-kv128", "R4-b1-T256", "R5-b4-T512",
                                 "R6-b1-kv1024-q4", "R6-b1-kv1024-q16", "R7-p512-d8",
                                 "R7-p512-d128", "R8-b32-ragged", "R2-b1-kv8192"])
def test_fake_step_shape_exact(wid):
    w = by_id(wid)
    plan = vw.plan_workload(w)
    h = FakeHandle()
    res = vw.measure(h, plan, seed=1, rounds=3, warmup_rounds=1, k_steps=8)
    assert res.exact and all(res.exact), (wid, res.shapes[:2])
    assert len(res.gpu_ms) == len(res.times_ms)
    assert not h.reqs                      # everything aborted / drained
    n_per_round = 8 if plan.kind == "decode" else 1
    assert len(res.times_ms) == 3 * n_per_round
    if wid.startswith("R7"):
        assert res.shapes[0]["hist"]["512"] == 1
        assert res.shapes[0]["total"] == 512 + w.n_seq - 1
        # decoders drift by the steps taken while their prefill finished
        assert 1024 <= res.kv_before[0] <= 1024 + 64
    if wid.startswith("R6"):
        q = w.seqs[0].q_len
        assert res.shapes[0]["total"] == q
        assert res.kv_before[0] == 1024
    if wid == "R0-b1-kv128":
        assert res.kv_before[0] == 128 and res.kv_before[-1] == 128 + 7


def test_chunk_multi_seq_uses_threshold():
    w = Workload("t", "R6", (Seq(256, 4), Seq(256, 4)), {})
    plan = vw.plan_workload(w)
    res = vw.measure(FakeHandle(), plan, 1, 2, 1, 8)
    assert all(res.exact) and res.shapes[0]["total"] == 8


def test_unsupported_shapes():
    with pytest.raises(vw.Unsupported):
        vw.plan_workload(Workload("t", "R6", (Seq(128, 4), Seq(256, 4)), {}))
    with pytest.raises(vw.Unsupported):
        vw.plan_workload(Workload("t", "R7", (Seq(0, 8), Seq(128, 1)), {}))


def test_inexact_detected():
    plan = vw.plan_workload(by_id("R4-b1-T1024"))
    h = FakeHandle(budget=256)             # budget too small: prompt split over steps
    res = vw.run_round(h, plan, random.Random(0), 8)
    assert not all(res.exact)


def test_stats_and_sizing_helpers():
    s = vw.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["n"] == 5 and s["median"] == 3.0 and s["min"] == 1.0
    assert set(s) >= {"median", "mean", "std", "p05", "p95", "mad", "ci95_median"}
    assert vw.summarize([]) == {"n": 0}
    assert vw.KV_BYTES_PER_BLOCK == 1835008
    assert vw.cudagraph_capture_max(by_id("R7-p512-d8")) == 2048
    assert vw.cudagraph_capture_max(by_id("R1-b256-kv128")) is None
    assert vw.cudagraph_capture_max(by_id("R5-b16-T2048")) is None
    shape = vw.step_shape({"a": 1, "b": 1, "c": 512})
    assert shape == {"total": 514, "n_req": 3, "hist": {"1": 2, "512": 1}}


def test_engine_kwargs_contract():
    w = by_id("R7-p512-d8")
    kw = vw.build_engine_kwargs(w, "m", 8, 7301, None)
    assert kw["load_format"] == "dummy" and kw["dtype"] == "bfloat16"
    assert kw["enable_prefix_caching"] is False and kw["async_scheduling"] is False
    assert kw["compilation_config"] == {"max_cudagraph_capture_size": 2048}
    assert "cudagraph_mode" not in kw["compilation_config"]     # default mode kept


def test_cli_parse(tmp_path):
    a = vw.parse_args(["--workload-json", "w.json", "--variant", "nopdl", "--out", "o.jsonl",
                       "--run-id", "x"])
    assert a.variant == "nopdl" and a.seed == 7301 and a.k_steps == 8


def test_skip_record_without_vllm(tmp_path):
    """An unsupported workload writes a skip record before importing vllm."""
    w = Workload("t", "R6", (Seq(128, 4), Seq(256, 4)), {})
    wj, out = tmp_path / "w.json", tmp_path / "o.jsonl"
    wj.write_text(json.dumps(w.to_dict()))
    rc = vw.main(["--workload-json", str(wj), "--out", str(out), "--run-id", "t"])
    rec = [json.loads(line) for line in out.read_text().splitlines()]
    assert rc == 0 and rec[0]["record_type"] == "skip" and rec[0]["arm"] == "vllm"
    assert rec[0]["schema"] == "regimebench/v1" and rec[0]["workload"]["id"] == "t"
