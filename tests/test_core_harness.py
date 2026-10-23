import json

import pytest
import torch

from regimebench import run
from regimebench.harness import accuracy as acc
from regimebench.harness.records import SCHEMA, RecordWriter, read_records


def test_schema_and_writer(tmp_path):
    assert SCHEMA == "regimebench/v1"
    w = RecordWriter(tmp_path / "r.jsonl", "id1")
    w.write("skip", arm="a", variant="v", reason="why")
    w.close()
    (rec,) = read_records(tmp_path / "r.jsonl")
    assert rec["schema"] == SCHEMA and rec["record_type"] == "skip" and rec["run_id"] == "id1"
    with pytest.raises(FileExistsError):
        RecordWriter(tmp_path / "r.jsonl", "id1").write("skip")


def outputs(scale=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(2, 50, generator=g)
    return {"logits": logits + scale * torch.randn(2, 50), "next_token": logits.argmax(-1),
            "k_new": torch.randn(3, 2, 2, 4, generator=g), "v_new": torch.randn(3, 2, 2, 4, generator=g)}


def test_oracle_band_grading():
    ref32 = outputs()
    ref16 = {k: (v + 0.01 * torch.randn_like(v) if v.is_floating_point() else v) for k, v in ref32.items()}
    band = acc.oracle_band(ref16, ref32)
    assert all(0 < b < 0.05 for b in band.values())
    close = {k: (v + 0.015 * torch.randn_like(v) if v.is_floating_point() else v) for k, v in ref32.items()}
    far = {k: (v + 0.5 * torch.randn_like(v) if v.is_floating_point() else v) for k, v in ref32.items()}
    assert acc.grade_vs_oracle(acc.compare(close, ref32), band)["verdict"] == "pass"
    assert acc.grade_vs_oracle(acc.compare(far, ref32), band)["verdict"] == "fail"
    m = acc.compare(ref16, ref32)
    assert m["next_token"]["match_rate"] == 1.0


def test_parse_variants_and_launches():
    assert run.parse_variants("fi-graph=a,b;vllm=pdl") == {"fi-graph": ["a", "b"], "vllm": ["pdl"]}
    assert run.parse_variants(None) == {}
    with pytest.raises(ValueError):
        run.parse_variants("bad")
    assert run.launches_per_sample(0.38) == 50
    assert run.launches_per_sample(2.0) == 10
    assert run.launches_per_sample(100.0) == 3
    assert run.launches_per_sample(0) == 50


def test_smoke_selection():
    ws = run.select_workloads(None, True)
    regimes = [w.regime for w in ws]
    assert regimes == [f"R{i}" for i in range(9)]
    assert ws[0].id == "R0-b1-kv128"
    assert [w.id for w in run.select_workloads("R1,R2-b1-kv8192", False)][-1] == "R2-b1-kv8192"
    with pytest.raises(SystemExit):
        run.select_workloads("nope", False)


def test_dry_run_cli(capsys):
    from regimebench.__main__ import main
    assert main(["run", "--tag", "t", "--smoke", "--dry-run", "--workloads", "R0,R1-b8-kv128"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["rounds"] == 3 and plan["seeds"] == [7301]
    assert [w["id"] for w in plan["workloads"]] == ["R0-b1-kv128", "R1-b8-kv128"]
    assert plan["workloads"][0]["check"] is True


class FakeStep:
    arm, variant, precision = "fake", "v", "bf16"

    def __init__(self):
        self.config = {}


def test_missing_arm_becomes_error_record(tmp_path):
    """load_arm failures are recorded, never raised (arms may not exist yet)."""
    from types import SimpleNamespace
    args = SimpleNamespace(arms=["hv2g"], seeds=[1], rounds=1, smoke=True)
    w = RecordWriter(tmp_path / "r.jsonl", "id")
    r = run.Runner(args, w, [], {})
    import regimebench.arms.base as base
    base.ARMS["hv2g"] = "regimebench.arms.does_not_exist:X"
    try:
        assert r.get_arm("hv2g") is None
    finally:
        base.ARMS["hv2g"] = "regimebench.arms.hv2g:Hv2gArm"
    w.close()
    recs = read_records(tmp_path / "r.jsonl")
    assert recs[0]["record_type"] == "error" and recs[0]["stage"] == "load_arm"
