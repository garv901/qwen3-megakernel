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


@pytest.mark.parametrize("npad,lmr", [(16, 2), (16, 4), (16, 5), (32, 2), (32, 4), (64, 2)])
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
    """MK_LMP=1, MK_LMR=0 is the default build (LMR 4 at NPAD 16, 3 at NPAD 32); MK_LMP=1 at NPAD >= 128 keeps the v0 LM path; MK_LMP=0 at NPAD 16 differs."""
    pytest.importorskip("cuda.bindings.nvrtc")
    if not (shutil.which("cuobjdump") or Path("/usr/local/cuda/bin/cuobjdump").exists()):
        pytest.skip("no cuobjdump")
    assert _sass(B.compile_build(32, knobs=(("MK_LMP", 1), ("MK_LMR", 3)))[0]) == _sass(B.compile_build(32)[0])
    assert _sass(B.compile_build(16, knobs=(("MK_LMP", 1), ("MK_LMR", 4)))[0]) == _sass(B.compile_build(16)[0])
    assert _sass(B.compile_build(128, knobs=(("MK_LMP", 0),))[0]) == _sass(B.compile_build(128)[0])
    assert _sass(B.compile_build(16, knobs=(("MK_LMP", 0),))[0]) != _sass(B.compile_build(16)[0])
    assert B.dyn_smem(128, lmp=0) == B.dyn_smem(128)


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


# ---------------------------------------------------------------- MK_TS: tile-stream QKV (SPEC-P5a)
HEAD_P5A = "6f1e1cc"   # the last commit before the tile-stream engine (SPEC-P5a); MK_TS=0 must be SASS-identical to it


@pytest.mark.parametrize("npad", (128, 256))
@pytest.mark.parametrize("knobs", ((), (("MK_NULL", 1),), (("MK_NULL", 0x7F),), (("MK_STAMPS", 1),)))
def test_ts_builds_compile(npad, knobs):
    """MK_TS=1 at NPAD >= 128: no spill, <= 128 regs; MK_EXPECT_DYN == DYN_SMEM (static_assert) checks B.dyn_smem(ts=1) and the 227 KB limit."""
    pytest.importorskip("cuda.bindings.nvrtc")
    _, info, _ = B.compile_build(npad, knobs=(("MK_TS", 1), *knobs))
    assert info["regs"] <= 128 and info["spill_stores"] == 0 and info["spill_loads"] == 0 and info["stack"] == 0, info
    assert info["static_smem"] + B.dyn_smem(npad, ts=1) <= 227 * 1024


def test_ts_dyn_smem_and_ring():
    assert B.ts_ring(1) == 3 and B.ts_ring(0) == 1
    assert B.dyn_smem(128, ts=1) == B.dyn_smem(256, ts=1) == 1024 + 3 * (32768 + 128 * 256) + 384 == 198016
    assert B.dyn_smem(128, ts=1) + 11280 <= 227 * 1024
    for npad in (16, 32, 64):   # TS does not apply below NPAD 128
        assert B.dyn_smem(npad, ts=1) == B.dyn_smem(npad)
    assert B.dyn_smem(256, ku=64, ts=1) == B.dyn_smem(256, ku=64)   # MK_KU 64 keeps the v0 QKV
    assert B.dyn_smem(128) == 132352 and B.dyn_smem(256) == 197888   # MK_TS=0 layouts unchanged


def test_ts_off_sass_identical():
    """MK_TS=1 at NPAD <= 64 (and with MK_KU=64) is the MK_TS=0 build; default MK_TS=256 is on at NPAD 256 only."""
    pytest.importorskip("cuda.bindings.nvrtc")
    if not (shutil.which("cuobjdump") or Path("/usr/local/cuda/bin/cuobjdump").exists()):
        pytest.skip("no cuobjdump")
    for npad in (16, 32, 64):
        assert _sass(B.compile_build(npad, knobs=(("MK_TS", 1),))[0]) == _sass(B.compile_build(npad)[0])
    assert _sass(B.compile_build(128, knobs=(("MK_TS", 1), ("MK_KU", 64)))[0]) == _sass(B.compile_build(128, knobs=(("MK_KU", 64),))[0])
    assert _sass(B.compile_build(128, knobs=(("MK_TS", 0),))[0]) == _sass(B.compile_build(128)[0])
    assert _sass(B.compile_build(256, knobs=(("MK_TS", 1),))[0]) == _sass(B.compile_build(256)[0])   # default MK_TS = 256
    assert _sass(B.compile_build(256, knobs=(("MK_TS", 0),))[0]) != _sass(B.compile_build(256)[0])
    assert _sass(B.compile_build(128, knobs=(("MK_TS", 128),))[0]) != _sass(B.compile_build(128)[0])


@pytest.mark.parametrize("npad", (32, 256))
def test_ts0_sass_identical_to_head(npad, tmp_path):
    """MK_TS=0 == the sources of HEAD_P5A (built from `git show`), `cuobjdump -sass` with addresses removed. Skipped when the commit is not in the repo."""
    pytest.importorskip("cuda.bindings.nvrtc")
    if not (shutil.which("cuobjdump") or Path("/usr/local/cuda/bin/cuobjdump").exists()):
        pytest.skip("no cuobjdump")
    root = Path(__file__).resolve().parents[1]
    arc = subprocess.run(["git", "-C", str(root), "archive", HEAD_P5A, "regimebench"], capture_output=True)
    if arc.returncode != 0:
        pytest.skip(f"commit {HEAD_P5A} not available")
    subprocess.run(["tar", "-x", "-C", str(tmp_path)], input=arc.stdout, check=True)
    code = ("import sys,re,subprocess,tempfile,pathlib;from regimebench.kernels.mk import build as B;"
            "c=B.compile_build(%d)[0];p=pathlib.Path(tempfile.mkdtemp())/'k.cubin';p.write_bytes(c);"
            "o=subprocess.run(['/usr/local/cuda/bin/cuobjdump','-sass',str(p)],capture_output=True,text=True,check=True).stdout;"
            "print('\\n'.join(re.sub(r'/\\*[0-9a-f]+\\*/','',l).strip() for l in o.splitlines() if not l.strip().startswith('code for')))" % npad)
    env = {**__import__("os").environ, "PYTHONPATH": str(tmp_path), "CUDA_CACHE_DISABLE": "1"}
    head = subprocess.run([__import__("sys").executable, "-c", code], capture_output=True, text=True, env=env, cwd=tmp_path, check=True).stdout
    assert head.strip() == _sass(B.compile_build(npad, knobs=(("MK_TS", 0),))[0]).strip() and len(head) > 10000


def test_ts_workspace_map_specs():
    """Map 8 = xb: bf16 [MAXTOK][HID] -> dims {HID, MAXTOK}, stride HID * 2 B, box {64 k, 128 tokens} (CPU: dims only; encoding needs a GPU)."""
    assert B.WORKSPACE_MAPS == ("xb",) and len(B.WEIGHT_MAPS) == 8
    assert B.workspace_map_specs() == [([1024, 256], [2048], [64, 128])]
    assert B.workspace_map_specs()[0][2] == B.map_specs({n: type("T", (), {"shape": (28, 128, 1024)})() for n in B.WEIGHT_MAPS if n != "embed"} | {"embed": type("T", (), {"shape": (151936, 1024)})()})[0][2]
    pytest.importorskip("cuda.bindings.driver")
    pytest.skip("cuTensorMapEncodeTiled needs a GPU (cuInit)")


def test_ts_knob_default_and_arm():
    assert B.DEFAULT_KNOBS["MK_TS"] == 256 and B.ts_min(1) == 128 and B.ts_min(256) == 256
    assert dict(B.knob_defines({"MK_TS": 1}) and [d.split("=") for d in B.knob_defines({"MK_TS": 1})])["MK_TS"] == "1"
    assert MkArm(ts=1).knobs["MK_TS"] == 1 and MkArm().knobs["MK_TS"] == 256


# ---------------------------------------------------------------- MK_TSB: GU and D on the tile stream, idle-CTA finalizers (SPEC-P5b)
HEAD_P5B = "b7a78e2"   # the last commit before P5b; MK_TSB=0 must be SASS-identical to it


@pytest.mark.parametrize("npad,ts", ((128, 128), (256, 128), (256, 256)))
@pytest.mark.parametrize("knobs", ((), (("MK_NULL", 1 << 4),), (("MK_NULL", 1 << 5),), (("MK_NULL", 0x7F),), (("MK_STAMPS", 1),)))
def test_tsb_builds_compile(npad, ts, knobs):
    """MK_TSB=1: no spill, <= 128 regs; MK_EXPECT_DYN == DYN_SMEM (static_assert) checks B.dyn_smem(tsb=1) and the 227 KB limit."""
    pytest.importorskip("cuda.bindings.nvrtc")
    _, info, _ = B.compile_build(npad, knobs=(("MK_TS", ts), ("MK_TSB", 1), *knobs))
    assert info["regs"] <= 128 and info["spill_stores"] == 0 and info["spill_loads"] == 0 and info["stack"] == 0, info
    assert info["static_smem"] + B.dyn_smem(npad, ts=ts, tsb=1) <= 227 * 1024


def test_tsb_dyn_smem_and_rings():
    """Ring slots 32 KB + TT * 256 B; the GU exchange region (12 KB) sits after the largest ring and below OFF_BARS."""
    assert B.ts_tt(256) == {"qkv": 128, "gu": 128, "d": 128} and B.ts_tt(128) == {"qkv": 128, "gu": 64, "d": 128}
    assert B.ts_ks(256) == {"qkv": 2, "gu": 1, "d": 8} and B.ts_ks(128) == {"qkv": 4, "gu": 1, "d": 12}
    assert B.ts_ring(1, 128, 1) == 3 and B.ts_ring(1, 64, 1) == 4
    assert B.dyn_smem(128, ts=1, tsb=1) == B.dyn_smem(256, ts=1, tsb=1) == 1024 + 3 * 65536 + 12288 + 384 == 210304
    assert B.dyn_smem(128, ts=1, tsb=1) + 11280 <= 227 * 1024
    assert B.dyn_smem(128, ts=1) == B.dyn_smem(128, ts=1, tsb=0) == 198016   # P5a layouts unchanged
    for npad in (16, 32, 64):   # TS does not apply below NPAD 128
        assert B.dyn_smem(npad, ts=1, tsb=1) == B.dyn_smem(npad)
    assert B.dyn_smem(128, ts=256, tsb=1) == B.dyn_smem(128)   # MK_TS = 256: NPAD 128 is below TS_MIN
    assert B.tsb_on(256, {"MK_TSB": 1}) and not B.tsb_on(128, {"MK_TSB": 1}) and B.tsb_on(128, {"MK_TSB": 1, "MK_TS": 128})
    assert not B.tsb_on(256, {"MK_TSB": 1, "MK_TS": 0}) and not B.tsb_on(256, {})


def test_tsb_off_sass_identical():
    """MK_TSB=0 (explicit) is the default build; MK_TSB=1 where TS is off (NPAD <= 64, NPAD 128 with the default MK_TS = 256) is MK_TSB=0."""
    pytest.importorskip("cuda.bindings.nvrtc")
    if not (shutil.which("cuobjdump") or Path("/usr/local/cuda/bin/cuobjdump").exists()):
        pytest.skip("no cuobjdump")
    assert _sass(B.compile_build(256, knobs=(("MK_TSB", 0),))[0]) == _sass(B.compile_build(256)[0])
    for npad in (32, 128):
        assert _sass(B.compile_build(npad, knobs=(("MK_TSB", 1),))[0]) == _sass(B.compile_build(npad)[0])
    assert _sass(B.compile_build(128, knobs=(("MK_TS", 128), ("MK_TSB", 1)))[0]) != _sass(B.compile_build(128, knobs=(("MK_TS", 128),))[0])
    assert _sass(B.compile_build(256, knobs=(("MK_TSB", 1),))[0]) != _sass(B.compile_build(256)[0])


@pytest.mark.parametrize("npad,ts", ((32, 256), (128, 128), (256, 256)))
def test_tsb0_sass_identical_to_head(npad, ts, tmp_path):
    """MK_TSB=0 == the sources of HEAD_P5B (built from `git archive`), `cuobjdump -sass` with addresses removed. Skipped when the commit is not in the repo."""
    pytest.importorskip("cuda.bindings.nvrtc")
    if not (shutil.which("cuobjdump") or Path("/usr/local/cuda/bin/cuobjdump").exists()):
        pytest.skip("no cuobjdump")
    root = Path(__file__).resolve().parents[1]
    arc = subprocess.run(["git", "-C", str(root), "archive", HEAD_P5B, "regimebench"], capture_output=True)
    if arc.returncode != 0:
        pytest.skip(f"commit {HEAD_P5B} not available")
    subprocess.run(["tar", "-x", "-C", str(tmp_path)], input=arc.stdout, check=True)
    code = ("import sys,re,subprocess,tempfile,pathlib;from regimebench.kernels.mk import build as B;"
            "c=B.compile_build(%d,knobs=(('MK_TS',%d),))[0];p=pathlib.Path(tempfile.mkdtemp())/'k.cubin';p.write_bytes(c);"
            "o=subprocess.run(['/usr/local/cuda/bin/cuobjdump','-sass',str(p)],capture_output=True,text=True,check=True).stdout;"
            "print('\\n'.join(re.sub(r'/\\*[0-9a-f]+\\*/','',l).strip() for l in o.splitlines() if not l.strip().startswith('code for')))" % (npad, ts))
    env = {**__import__("os").environ, "PYTHONPATH": str(tmp_path), "CUDA_CACHE_DISABLE": "1"}
    head = subprocess.run([__import__("sys").executable, "-c", code], capture_output=True, text=True, env=env, cwd=tmp_path, check=True).stdout
    assert head.strip() == _sass(B.compile_build(npad, knobs=(("MK_TS", ts), ("MK_TSB", 0)))[0]).strip() and len(head) > 10000


def test_interleave_gu_formula():
    """wgu rows [128 p, 128 p + 64) = wg rows [64 p, +64), rows [128 p + 64, 128 p + 128) = wu rows [64 p, +64), per layer, on a small random tensor (INTER = 3072)."""
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    nl, inter, h = 3, 3072, 16
    wg = torch.randn(nl, inter, h, generator=g).to(torch.bfloat16)
    wu = torch.randn(nl, inter, h, generator=g).to(torch.bfloat16)
    wgu = B.interleave_gu(wg, wu)
    assert wgu.shape == (nl * 6144, h) and wgu.dtype == torch.bfloat16
    for l in range(nl):
        for p in (0, 1, 17, 46, 47):
            base = l * 6144 + 128 * p
            assert torch.equal(wgu[base:base + 64], wg[l, 64 * p:64 * p + 64])
            assert torch.equal(wgu[base + 64:base + 128], wu[l, 64 * p:64 * p + 64])
    full = torch.empty(nl * 6144, h, dtype=torch.bfloat16)   # the whole formula, row by row
    for l in range(nl):
        for p in range(48):
            full[l * 6144 + 128 * p:l * 6144 + 128 * p + 64] = wg[l, 64 * p:64 * p + 64]
            full[l * 6144 + 128 * p + 64:l * 6144 + 128 * p + 128] = wu[l, 64 * p:64 * p + 64]
    assert torch.equal(wgu, full)
    # lane -> channel rule of the GU epilogue: TMEM lane L of row block p is gate channel 64 p + L (L < 64) or up channel 64 p + L - 64
    ch = [(p, L) for p in (0, 5) for L in (0, 31, 32, 63, 64, 95, 96, 127)]
    for p, L in ch:
        src = wg if L < 64 else wu
        assert torch.equal(wgu[128 * p + L], src[0, 64 * p + (L % 64)])


def test_tsb_map_specs():
    """Maps 9, 10, 11: xb2 {HID, MAXTOK} box {64, TT_GU}; wgu {HID, NL * 6144} box {64, 128}; act {INTER, MAXTOK} box {64, TT_D}; maps 8..11 are encoded per bucket."""
    assert B.TSB_MAPS == ("xb2", "wgu", "act") and B.WS_ORDER == ("xb", "xb2", "wgu", "act")
    assert B.workspace_map_specs(256, B.WS_ORDER) == [([1024, 256], [2048], [64, 128]), ([1024, 256], [2048], [64, 128]),
                                                      ([1024, 28 * 6144], [2048], [64, 128]), ([3072, 256], [6144], [64, 128])]
    assert B.workspace_map_specs(128, B.WS_ORDER)[1] == ([1024, 256], [2048], [64, 64])
    assert B.workspace_map_specs() == [([1024, 256], [2048], [64, 128])]
    assert MkArm(tsb=1).knobs["MK_TSB"] == 1 and MkArm().knobs["MK_TSB"] == 0 and B.DEFAULT_KNOBS["MK_TSB"] == 0 and B.TK_SIZE == 64
