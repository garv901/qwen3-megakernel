"""GPU test of A1 v1 (splitkv): runs probe/mk_attn/a1bench.py --check (ao vs torch fp32 reference, rel-L2 <= 1e-2, and the combine_head SW128 bytes
bit-exact vs plan.pack_b_slot) on b1 kv {128, 1K, 32K}, b32 kv1K, b256 kv128. Skipped without a CUDA device."""
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cuda.bindings.driver")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU (sm_100a)")

BENCH = Path(__file__).resolve().parents[4] / "probe" / "mk_attn" / "a1bench.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("a1bench", BENCH)
    m = importlib.util.module_from_spec(spec)
    sys.modules["a1bench"] = m
    spec.loader.exec_module(m)
    return m.Bench(None)


@pytest.mark.parametrize("point", ["b1kv128", "b1kv1024", "b1kv32768", "b32kv1024", "b256kv128"])
@pytest.mark.parametrize("engine", ["auto", "ldg", "tma"])
def test_check(bench, point, engine):
    if engine == "ldg" and point in ("b1kv32768", "b256kv128"):
        pytest.skip("LDG engine at a huge share is correct but slow; covered by auto")
    r = bench.run(point, engine, 1 | 2 | 4, True, 1)
    assert r["pass"], r
    assert max(r["rel_l2"].values()) <= 1e-2
    assert r.get("sw128_bad_bytes") == 0
