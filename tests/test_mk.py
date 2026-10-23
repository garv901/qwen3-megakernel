"""CPU-only tests for the mk arm (NVRTC compile of every token bucket, bucket choice, supports(), source hash, knob builds, SASS identity)."""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from regimebench.arms.mk import MkArm
from regimebench.kernels.mk import build as B
from regimebench.workload import Seq, Workload, regime_grid


@pytest.fixture(scope="module")
def builds():
    pytest.importorskip("cuda.bindings.nvrtc")
    return {n: B.compile_build(n) for n in B.BUCKETS}


@pytest.mark.parametrize("npad", B.BUCKETS)
def test_bucket_compiles_without_spills(builds, npad):
    cubin, info, log = builds[npad]
    assert len(cubin) > 1000
    assert info["regs"] <= 128, info
    assert info["spill_stores"] == 0 and info["spill_loads"] == 0 and info["stack"] == 0, info
    assert info["static_smem"] + B.dyn_smem(npad) <= 227 * 1024


def test_bucket_choice():
    assert [B.bucket_for(n) for n in (1, 16, 17, 32, 33, 64, 65, 128, 129, 256)] == [16, 16, 32, 32, 64, 64, 128, 128, 256, 256]
    with pytest.raises(ValueError):
        B.bucket_for(257)


def test_supports_reasons():
    arm = MkArm()
    ok = Workload("t", "R1", tuple(Seq(128, 1) for _ in range(256)))
    big = Workload("t2", "R1", tuple(Seq(128, 1) for _ in range(257)))
    assert arm.supports(ok, "auto") is None and arm.supports(ok, "auto-direct") is None
    assert "257" in arm.supports(big, "auto")
    assert "unknown variant" in arm.supports(ok, "nope")
    assert arm.variants() == ("auto", "auto-direct")
    pts = [w for w in regime_grid() if w.id in ("R0-b1-kv128", "R1-b256-kv128", "R6-b1-kv1024-q4")]
    assert len(pts) == 3 and all(arm.supports(w, "auto") is None for w in pts)


def test_source_hash_tracks_attention_header(tmp_path):
    base = B.source_sha256()
    assert base == B.source_sha256("placeholder")
    hdr = tmp_path / "attn.cuh"
    hdr.write_text(B.ATTN_PLACEHOLDER.read_text())
    assert B.source_sha256(attn_path=hdr) == base
    hdr.write_text(hdr.read_text() + "\n// changed\n")
    assert B.source_sha256(attn_path=hdr) != base


def test_splitkv_flag_errors_clearly_when_missing():
    if B.ATTN_SPLITKV.exists():
        pytest.skip("splitkv present")
    with pytest.raises(FileNotFoundError, match="splitkv"):
        B.source("splitkv")


def test_knob_builds_compile():
    pytest.importorskip("cuda.bindings.nvrtc")
    for knobs in ((("MK_STAMPS", 1),), (("MK_NULL", 0x7F),), (("MK_KU", 64),), (("MK_RING", 3),), (("MK_RING", 4),)):
        _, info, _ = B.compile_build(64, knobs=knobs)
        assert info["regs"] <= 128 and info["spill_stores"] == 0, (knobs, info)
    with pytest.raises(KeyError):
        B.knob_defines({"MK_BOGUS": 1})


def _sass(cubin: bytes) -> str:
    exe = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "k.cubin"
        p.write_bytes(cubin)
        out = subprocess.run([exe, "-sass", str(p)], capture_output=True, text=True, check=True).stdout
    return "\n".join(re.sub(r"/\*[0-9a-f]+\*/", "", ln).strip() for ln in out.splitlines() if not ln.strip().startswith("code for"))


def test_knob_off_sass_identical_to_stripped_source():
    pytest.importorskip("cuda.bindings.nvrtc")
    if not (shutil.which("cuobjdump") or Path("/usr/local/cuda/bin/cuobjdump").exists()):
        pytest.skip("no cuobjdump")
    a = _sass(B.compile_build(32)[0])
    b = _sass(B.compile_build(32, stripped=True)[0])
    assert a == b and len(a) > 10000


@pytest.mark.parametrize("npad,lmr", [(16, 2), (16, 4), (16, 5), (32, 2), (32, 4)])
def test_lmp_builds_compile(npad, lmr):
    """Pipelined LM head: compiles without spill; MK_EXPECT_DYN == DYN_SMEM is a static_assert, so a compile checks B.dyn_smem."""
    pytest.importorskip("cuda.bindings.nvrtc")
    _, info, _ = B.compile_build(npad, knobs=(("MK_LMP", 1), ("MK_LMR", lmr)))
    assert info["regs"] <= 128 and info["spill_stores"] == 0 and info["spill_loads"] == 0 and info["stack"] == 0, info
    assert info["static_smem"] + B.dyn_smem(npad, lmp=1, lmr=lmr) <= 227 * 1024


def test_lmp_too_many_slots_rejected():
    pytest.importorskip("cuda.bindings.nvrtc")
    with pytest.raises(Exception, match="exceed 227 KB|static_assert|error"):
        B.compile_build(16, knobs=(("MK_LMP", 1), ("MK_LMR", 6)))


def test_lmp_off_sass_identical():
    """MK_LMP=1, MK_LMR=0 is the default build (LMR 4 at NPAD 16, 3 at NPAD 32); MK_LMP=1 at NPAD >= 64 keeps the v0 LM path; MK_LMP=0 at NPAD 16 differs."""
    pytest.importorskip("cuda.bindings.nvrtc")
    if not (shutil.which("cuobjdump") or Path("/usr/local/cuda/bin/cuobjdump").exists()):
        pytest.skip("no cuobjdump")
    assert _sass(B.compile_build(32, knobs=(("MK_LMP", 1), ("MK_LMR", 3)))[0]) == _sass(B.compile_build(32)[0])
    assert _sass(B.compile_build(16, knobs=(("MK_LMP", 1), ("MK_LMR", 4)))[0]) == _sass(B.compile_build(16)[0])
    assert _sass(B.compile_build(64, knobs=(("MK_LMP", 0),))[0]) == _sass(B.compile_build(64)[0])
    assert _sass(B.compile_build(16, knobs=(("MK_LMP", 0),))[0]) != _sass(B.compile_build(16)[0])
    assert B.dyn_smem(64, lmp=0) == B.dyn_smem(64)


def test_stamps_build_differs_from_default():
    pytest.importorskip("cuda.bindings.nvrtc")
    assert _sass(B.compile_build(32, knobs=(("MK_STAMPS", 1),))[0]) != _sass(B.compile_build(32)[0])


def test_map_specs():
    class T:
        def __init__(self, *s): self.shape = s
    w = {"wq": T(28, 2048, 1024), "wk": T(28, 1024, 1024), "wv": T(28, 1024, 1024), "wo": T(28, 1024, 2048),
         "wg": T(28, 3072, 1024), "wu": T(28, 3072, 1024), "wd": T(28, 1024, 3072), "embed": T(151936, 1024)}
    sp = B.map_specs(w)
    assert sp[0] == ([1024, 28 * 2048], [2048], [64, 128])
    assert sp[3] == ([2048, 28 * 1024], [4096], [64, 128])
    assert sp[6] == ([3072, 28 * 1024], [6144], [64, 128])
    assert sp[7] == ([1024, 151936], [2048], [64, 128])
