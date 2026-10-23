import csv

from regimebench.report import Data, main_report, regime_table
from regimebench.sol import sol_both
from regimebench.workload import regime_grid


def rec(w, arm, variant, us, config=None):
    s = sol_both(w, 7e12, 2e15)
    return {"schema": "regimebench/v1", "run_id": "r", "record_type": "timing", "slurm_job_id": "9",
            "arm": arm, "variant": variant, "workload": w.to_dict(), "config": config or {},
            "steady": {"median": us / 1e3, "ci95_median": [us / 1e3 * 0.99, us / 1e3 * 1.01]},
            "isolated": {"median": us / 1e3 * 1.1}, "sol": s, "tokens_per_s": w.total_q / (us * 1e-6),
            "n_launch_per_sample": 20}


def fake_records():
    grid = {w.id: w for w in regime_grid()}
    out = [{"record_type": "env", "run_id": "r", "env": {"gpu": {"name": "B200"}}}]
    for wid, h, b in (("R0-b1-kv128", 390, 700), ("R1-b8-kv128", 420, 800), ("R1-b16-kv128", 500, 900)):
        w = grid[wid]
        out += [rec(w, "hv2g", "auto", h), rec(w, "hv2-repeat", "graph", h * w.total_q),
                rec(w, "fi-graph", "graph-pdl-best", b, {"attn_backend": "fa2", "pdl": {"launches_with_pdl": 3, "launches_total": 9}}),
                rec(w, "fi-graph", "graph-nopdl-best", b * 1.1), rec(w, "vllm", "pdl", b * 2)]
    w = grid["R0-b1-kv128"]
    out.append({"record_type": "skip", "arm": "fi-graph", "variant": "eager-pdl-best",
                "workload": grid["R1-b8-kv128"].to_dict(), "reason": "R0 only"})
    out.append({"record_type": "accuracy", "arm": "hv2g", "variant": "auto", "workload": w.to_dict(),
                "seed": 1, "worst_ratio": 1.2, "ratios": {"logits": 1.2, "k_new": 1.0, "v_new": 0.9},
                "verdict": "pass", "next_token_match_rate": 1.0})
    return out


def test_best_and_table():
    d = Data(fake_records())
    assert d.best("R0-b1-kv128", "fi-graph") == ("graph-pdl-best", 700)
    t = regime_table(d, "R1")
    assert "R1-b8-kv128" in t and "graph-pdl-best" in t and "0.5" in t


def test_main_report_writes_files(tmp_path):
    import json
    p = tmp_path / "records.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in fake_records()))
    out = tmp_path / "out"
    assert main_report([str(p)], str(out)) == 0
    text = (out / "REPORT.md").read_text()
    assert "## R0" in text and "R0 only" in text and "pass" in text
    rows = list(csv.DictReader(open(out / "summary.csv")))
    assert {r["arm"] for r in rows} >= {"hv2g", "fi-graph", "vllm", "hv2-repeat"}
    assert (out / "R1.png").exists()


def test_summary_and_ffma():
    from regimebench.report import summary_rows, summary_table
    recs = fake_records()
    recs.append({"record_type": "probe", "run_id": "r", "clock": {"sm_mhz": 1900}})
    d = Data(recs)
    rows = {r["regime"]: r for r in summary_rows(d)}
    assert rows["R1"]["points"] == 2 and rows["R1"]["wins"] == 2
    assert rows["R1"]["gmean"] == __import__("pytest").approx((420 / 800 * 500 / 900) ** 0.5)
    assert rows["R1"]["crossover"] == "none"
    assert d.clock_mhz("R0-b1-kv128") == 1900
    assert d.ffma_sol_us("R4-b1-T16") is None
    assert d.ffma_sol_us("R0-b1-kv128") >= d.sol("R0-b1-kv128")["t_bw_us"]
    assert "R1" in summary_table(d)
