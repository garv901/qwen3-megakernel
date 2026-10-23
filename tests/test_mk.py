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


def _cb(npad, **k):
    """compile_build on the P5a knob base (MK_TS 256, MK_TSB 0; the defaults before job 9622) plus k."""
    return B.compile_build(npad, knobs=tuple({"MK_TS": 256, "MK_TSB": 0, **k}.items()))[0]


def test_ts_off_sass_identical():
    """MK_TS=1 at NPAD <= 64 (and with MK_KU=64) is the MK_TS=0 build; MK_TS=256 is on at NPAD 256 only (MK_TSB 0 base)."""
    pytest.importorskip("cuda.bindings.nvrtc")
    if not (shutil.which("cuobjdump") or Path("/usr/local/cuda/bin/cuobjdump").exists()):
        pytest.skip("no cuobjdump")
    for npad in (16, 32, 64):
        assert _sass(_cb(npad, MK_TS=1)) == _sass(_cb(npad))
    assert _sass(_cb(128, MK_TS=1, MK_KU=64)) == _sass(_cb(128, MK_KU=64))
    assert _sass(_cb(128, MK_TS=0)) == _sass(_cb(128))
    assert _sass(_cb(256, MK_TS=1)) == _sass(_cb(256))
    assert _sass(_cb(256, MK_TS=0)) != _sass(_cb(256))
    assert _sass(_cb(128, MK_TS=128)) != _sass(_cb(128))



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
    assert B.DEFAULT_KNOBS["MK_TS"] == 128 and B.ts_min(1) == 128 and B.ts_min(256) == 256
    assert dict(B.knob_defines({"MK_TS": 1}) and [d.split("=") for d in B.knob_defines({"MK_TS": 1})])["MK_TS"] == "1"
    assert MkArm(ts=1).knobs["MK_TS"] == 1 and MkArm().knobs["MK_TS"] == 128


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
    assert B.tsb_on(256, {"MK_TSB": 1, "MK_TS": 256}) and not B.tsb_on(128, {"MK_TSB": 1, "MK_TS": 256}) and B.tsb_on(128, {"MK_TSB": 1, "MK_TS": 128})
    assert not B.tsb_on(256, {"MK_TSB": 1, "MK_TS": 0}) and not B.tsb_on(256, {"MK_TSB": 0}) and B.tsb_on(128, {}) and not B.tsb_on(64, {})


def test_tsb_off_sass_identical():
    """MK_TSB=1 where TS is off (NPAD <= 64, NPAD 128 with MK_TS = 256) is MK_TSB=0; the default build is MK_TS 128 + MK_TSB 1 (job 9622)."""
    pytest.importorskip("cuda.bindings.nvrtc")
    if not (shutil.which("cuobjdump") or Path("/usr/local/cuda/bin/cuobjdump").exists()):
        pytest.skip("no cuobjdump")
    assert _sass(_cb(256, MK_TS=128, MK_TSB=1)) == _sass(B.compile_build(256)[0])
    for npad in (32, 128):
        assert _sass(_cb(npad, MK_TSB=1)) == _sass(_cb(npad))
    assert _sass(_cb(128, MK_TS=128, MK_TSB=1)) != _sass(_cb(128, MK_TS=128))
    assert _sass(_cb(256, MK_TSB=1)) != _sass(_cb(256))



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
    assert MkArm(tsb=1).knobs["MK_TSB"] == 1 and MkArm().knobs["MK_TSB"] == 1 and B.DEFAULT_KNOBS["MK_TSB"] == 1 and B.TK_SIZE == 64


# ---------------------------------------------------------------- attn = splitkv (A1 v1, SPEC-P2)
HEAD_P2 = "57f8fdc"   # the last commit before SPEC-P2 (P5b); attn = placeholder must be SASS-identical to it
def _skv_consts() -> dict:
    """kStageRows, kStages of the imported header (defaults) -> kBarOff, kBarBytes, kSmemBytes (the formulas of splitkv.cuh)."""
    txt = B.ATTN_SPLITKV.read_text()
    rows = int(re.search(r"#define SKV_STAGE_ROWS (\d+)", txt).group(1))
    stages = int(re.search(r"#define SKV_STAGES (\d+)", txt).group(1))
    bar_off = stages * 2 * rows * 256 + 16 * 2 * 132 * 4
    bar_bytes = max(64, 16 * stages)
    return {"rows": rows, "stages": stages, "bar_off": bar_off, "bar_bytes": bar_bytes, "smem": bar_off + bar_bytes}


def test_skv_header_constants_match_build_py():
    c = _skv_consts()
    assert (c["bar_off"], c["bar_bytes"], c["smem"]) == (B.SKV_BAR_OFF, B.SKV_BAR_BYTES, 148032)
    assert B.SKV_STAGE_ROWS == c["rows"] and B.SKV_NUM_MAPS == 8


def test_skv_dyn_smem_layout():
    """OFF_BARS >= kBarOff and = 512 mod 1024 (scratch base OFF_BARS - kBarOff 1024 B aligned); mk's barriers behind A1's 64 B tail; placeholder layouts unchanged."""
    for npad in B.BUCKETS:
        for ts, tsb in ((0, 0), (256, 0), (256, 1), (128, 0), (128, 1)):
            ob = B.skv_off_bars(npad, 128, 2, "splitkv", 1, 0, ts, tsb)
            assert ob >= B.SKV_BAR_OFF and ob % 1024 == 512 and (ob - B.SKV_BAR_OFF) % 1024 == 0, (npad, ts, tsb, ob)
            lay = B.dyn_smem(npad, ts=ts, tsb=tsb, attn="splitkv")
            assert lay == 1024 + ob + 64 + (384 if (npad <= 64 or (ts and npad >= B.ts_min(ts))) else 256)
    assert B.dyn_smem(256, ts=256, attn="splitkv") == 198592 and B.dyn_smem(256, ts=256, tsb=1, attn="splitkv") == 210880
    assert B.dyn_smem(16, attn="splitkv") == 165824 and B.dyn_smem(128, attn="splitkv") == 149312
    assert B.dyn_smem(128) == 132352 and B.dyn_smem(256, ts=256) == 198016   # placeholder unchanged
    with pytest.raises(ValueError):
        B.dyn_smem(64, ku=64, attn="splitkv")


@pytest.mark.parametrize("npad", B.BUCKETS)
@pytest.mark.parametrize("ts,tsb", ((0, 0), (128, 1)))   # TS on with TSB 0 is rejected under splitkv (hang, job 9662)
def test_skv_builds_compile(npad, ts, tsb):
    """attn = splitkv at every bucket x MK_TSB: compiles (MK_EXPECT_DYN == DYN_SMEM static_assert checks B.dyn_smem(attn="splitkv"); the 227 KB static_assert),
    <= 128 regs, static + dyn smem fits. KNOWN DEVIATION (SPEC-P2 report): the 128-register kernel spills (A1's attend / combine_head peak at 128 registers);
    the bound below guards against growth, it is not the 0-spill goal."""
    pytest.importorskip("cuda.bindings.nvrtc")
    _, info, _ = B.compile_build(npad, knobs=(("MK_TS", ts), ("MK_TSB", tsb)), attn="splitkv")
    assert info["regs"] <= 128, info
    assert info["spill_stores"] <= 1024 and info["spill_loads"] <= 1024, info
    assert info["static_smem"] + B.dyn_smem(npad, ts=ts, tsb=tsb, attn="splitkv") <= 227 * 1024


@pytest.mark.parametrize("npad,knobs", [(16, (("MK_RING", 3),)), (32, (("MK_LMR", 4),)), (64, (("MK_LMP", 0),)), (128, (("MK_TS", 128),)), (128, (("MK_TS", 128), ("MK_TSB", 1))),
                                       (256, (("MK_NULL", 1 << 2),)), (256, (("MK_NULL", 0x7F),)), (64, (("MK_STAMPS", 1),)), (32, (("MK_NULL", 1 << 3),))])
def test_skv_knob_builds_compile(npad, knobs):
    """dyn_smem host == kernel (MK_EXPECT_DYN static_assert) for splitkv at other knob sets, MK_NULL bit 2 (attend + combine skipped) and bit 3 (O phase null)."""
    pytest.importorskip("cuda.bindings.nvrtc")
    _, info, _ = B.compile_build(npad, knobs=knobs, attn="splitkv")
    assert info["regs"] <= 128
    k = {**B.DEFAULT_KNOBS, **dict(knobs)}
    assert info["static_smem"] + B.dyn_smem(npad, 128, k["MK_RING"], lmp=k["MK_LMP"], lmr=k["MK_LMR"], ts=k["MK_TS"], tsb=k["MK_TSB"], attn="splitkv") <= 227 * 1024


def test_skv_ku64_rejected():
    pytest.importorskip("cuda.bindings.nvrtc")
    with pytest.raises(Exception, match="MK_KU|static_assert|error"):
        B.compile_build(64, knobs=(("MK_KU", 64),), attn="splitkv")


def test_skv_source_defines():
    """splitkv builds the header + mk text with MK_SKV; the placeholder build text does not mention splitkv symbols outside #ifdef MK_SKV."""
    assert "namespace skv" in B.source("splitkv") and B.source_sha256("splitkv") != B.source_sha256("placeholder")
    assert "MK_SKV" in B.source("placeholder")   # guarded by #ifdef: the placeholder compile does not define it


def test_skv_plan_matches_mk_layout_on_mk_points():
    """plan.plan row_base == mk's t_base, total_q == ntok, cache_rows == the arm's rows for every q_len == 1 point of --workloads mk; R6 (q > 1) is the placeholder under auto."""
    from regimebench.kernels.common.splitkv import plan as P
    from regimebench.workload import select
    pts = select(["mk"])
    assert len(pts) >= 20
    arm = MkArm(attn="auto")
    used = {}
    for w in pts:
        used[w.id] = arm.resolve_attn(w)
        if all(s.q_len == 1 for s in w.seqs):
            pl = P.plan([(s.kv_len, s.q_len) for s in w.seqs], n_cta=B.GRID)
            d = B.build_descriptors([(s.kv_len, s.q_len) for s in w.seqs], B.MAXTOK)
            assert list(pl.row_base) == list(d.t_base) and pl.total_q == w.total_q == d.chunks[0].ntok and pl.cache_rows == d.cache_rows == w.cache_tokens, w.id
            assert pl.s_max <= 32 and pl.stride >= 1 and pl.engine in (0, 1)
            assert (pl.items[:, :, 7] <= pl.stride).all()
            assert all(len(w2) == 1 for w2 in [pl.cta_positions[:1]])
        else:
            assert used[w.id] == "placeholder", w.id
    exp = {w.id: ("splitkv" if all(s.q_len == 1 for s in w.seqs) and max(s.kv_len for s in w.seqs) >= B.SKV_AUTO_MIN_KV else "placeholder") for w in pts}
    assert used == exp and sum(v == "splitkv" for v in used.values()) == 13   # R2-kv512..32768, R3-*


def test_skv_attn_option_and_supports():
    assert MkArm().attn == "auto"
    with pytest.raises(ValueError):
        MkArm(attn="nope")
    w1 = Workload("t", "R1", tuple(Seq(1024, 1) for _ in range(8)))
    wq = Workload("t2", "R6", (Seq(1024, 4),))
    assert MkArm(attn="auto").resolve_attn(w1) == "splitkv" and MkArm(attn="auto").resolve_attn(wq) == "placeholder"
    assert MkArm(attn="auto").resolve_attn(Workload("t3", "R1", tuple(Seq(128, 1) for _ in range(8)))) == "placeholder"   # kv < SKV_AUTO_MIN_KV
    assert MkArm(attn="auto", ku=64).resolve_attn(w1) == "placeholder"
    assert MkArm(attn="splitkv").supports(w1, "auto") is None
    assert "q_len == 1" in MkArm(attn="splitkv").supports(wq, "auto")
    assert "MK_KU" in MkArm(attn="splitkv", ku=64).supports(w1, "auto")
    assert MkArm(attn="placeholder").supports(wq, "auto") is None and B.auto_attn([(1024, 1), (5, 1)]) == "splitkv" and B.auto_attn([(128, 1), (5, 1)]) == "placeholder" and B.auto_attn([(128, 1), (5, 2)]) == "placeholder"


def test_skv_map_specs():
    """16 maps: K then V, box rows 128, 64, .., 1; dims {128, 8, 28 * rows}, strides {256, 2048} B (a1bench.py tensor_maps)."""
    sp = B.skv_map_specs(1000)
    assert len(sp) == 16
    assert [s[2][2] for s in sp[:8]] == [128, 64, 32, 16, 8, 4, 2, 1] and [s[2][2] for s in sp[8:]] == [128, 64, 32, 16, 8, 4, 2, 1]
    assert all(s[0] == [128, 8, 28 * 1000] and s[1] == [256, 2048] and s[2][:2] == [128, 1] for s in sp)


def test_skv_placeholder_sass_identical_to_head(tmp_path):
    """attn = placeholder (the default) == the sources of HEAD_P2 (the P5b commit) at every bucket, MK_TS 0 / 256 x MK_TSB 0 / 1 (the full 20-config sweep ran in the P2 report;
    here one config per bucket class). Skipped when the commit is not in the repo."""
    pytest.importorskip("cuda.bindings.nvrtc")
    if not (shutil.which("cuobjdump") or Path("/usr/local/cuda/bin/cuobjdump").exists()):
        pytest.skip("no cuobjdump")
    root = Path(__file__).resolve().parents[1]
    arc = subprocess.run(["git", "-C", str(root), "archive", HEAD_P2, "regimebench"], capture_output=True)
    if arc.returncode != 0:
        pytest.skip(f"commit {HEAD_P2} not available")
    subprocess.run(["tar", "-x", "-C", str(tmp_path)], input=arc.stdout, check=True)
    cfgs = ((16, 256, 0), (64, 256, 0), (256, 256, 0))   # TSB 1 changed after P2 on purpose (GU r2 setup)
    code = ("import sys,re,subprocess,tempfile,pathlib;from regimebench.kernels.mk import build as B\n"
            "for npad,ts,tsb in %r:\n"
            " c=B.compile_build(npad,knobs=(('MK_TS',ts),('MK_TSB',tsb)))[0];p=pathlib.Path(tempfile.mkdtemp())/'k.cubin';p.write_bytes(c)\n"
            " o=subprocess.run(['/usr/local/cuda/bin/cuobjdump','-sass',str(p)],capture_output=True,text=True,check=True).stdout\n"
            " print('@@',npad,ts,tsb);print('\\n'.join(re.sub(r'/\\*[0-9a-f]+\\*/','',l).strip() for l in o.splitlines() if not l.strip().startswith('code for')))\n" % (cfgs,))
    env = {**__import__("os").environ, "PYTHONPATH": str(tmp_path), "CUDA_CACHE_DISABLE": "1"}
    head = subprocess.run([__import__("sys").executable, "-c", code], capture_output=True, text=True, env=env, cwd=tmp_path, check=True).stdout
    mine = "".join(f"@@ {n} {t} {b}\n" + _sass(B.compile_build(n, knobs=(("MK_TS", t), ("MK_TSB", b)))[0]) + "\n" for n, t, b in cfgs)
    assert head.strip() == mine.strip() and len(head) > 30000
