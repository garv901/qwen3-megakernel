"""Opus arm: the VibeSys/Opus 5.5 run-B Qwen3-0.6B cooperative decode kernel.

The candidate ``submission.py`` (NVRTC source string, ``cuLaunchCooperativeKernel``)
is read-only. It is copied into ``$MEGABENCH_SOTA_CACHE/opus/<sha16>/`` and the
copy is imported, so nothing is written next to the original. ``prepare``
rebuilds the candidate's launch (``build()``) with preallocated outputs, because
the candidate's ``run()`` allocates its four outputs on every call.

Variants (``direct`` is the headline):

* ``direct``: one ``cuLaunchCooperativeKernel`` per step, all buffers preallocated.
* ``graph``: the same launch captured in a CUDA graph.
* ``submission``: the candidate's own ``run()`` (allocates outputs per call).

Chained mode aliases ``token`` and ``next_token``: the kernel reads ``token[0]``
before its first grid barrier and the last block writes ``next_token`` after the
final barrier, so every step reads the previous step's argmax on the device.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from ..shapes import BASE_CASE_ID, Shape, megabench_case

DEFAULT_SUBMISSION = Path(
    "/raid/keisuke/vibe-megakernel/megabench/experiments/2026-10-06/"
    "20-41-04-vibesys-claude-opus55-p0-10rounds-slurm/dense/submission.py")
DEFAULT_CACHE = "/raid/garv901/.cache/megabench_sota"
VARIANTS = ("direct", "graph", "submission")
OUTPUT_KEYS = ("logits", "next_token", "k_write", "v_write")
GEOMETRY_KEYS = ("batch", "context", "layers", "hidden", "q_heads", "kv_heads",
                 "head_dim", "intermediate", "vocab")
THREADS = 512


def cache_root() -> Path:
    return Path(os.environ.get("MEGABENCH_SOTA_CACHE", DEFAULT_CACHE))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def copy_submission(submission: Path) -> tuple[Path, str]:
    data = Path(submission).read_bytes()
    sha = _sha(data)
    dest = cache_root() / "opus" / sha[:16]
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "submission.py"
    if not path.exists() or path.read_bytes() != data:
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    return dest, sha


def _import_copy(dest: Path):
    name = f"megabench_opus_{dest.name}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, dest / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Kernel:
    """The compiled ``qwen3_step`` function plus the buffers ``build()`` owns."""

    def __init__(self, module, device):
        import torch
        from cuda.bindings import driver
        self.driver = driver
        self.check = module._check
        self.input_order = tuple(module._INPUT_ORDER)
        torch.cuda.init()
        torch.zeros(1, device=device)  # make torch's primary context current
        self.grid = torch.cuda.get_device_properties(device).multi_processor_count
        cubin = module._compile(device, self.grid)
        self.module = self.check(driver.cuModuleLoadData(cubin))
        self.func = self.check(driver.cuModuleGetFunction(self.module, b"qwen3_step"))
        fn = getattr(module, "_dyn_smem", None)
        self.dyn_smem = int(fn()) if callable(fn) else 0
        if self.dyn_smem > 0:
            self.check(driver.cuFuncSetAttribute(
                self.func,
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                self.dyn_smem))
        per_sm = self.check(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            self.func, THREADS, self.dyn_smem))
        if per_sm < 1:
            raise RuntimeError("qwen3_step cannot be resident")
        self.regs = int(self.check(driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, self.func)))
        self.static_smem = int(self.check(driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, self.func)))
        # Same tail buffers and RoPE row (position 128, theta 1e6) as build().
        sf = getattr(module, "_scratch_floats", None)  # R11_SKD_NX > 1 needs more scratch than build()'s 8192 floats of the older kernels
        self.scratch = torch.zeros(int(sf()) if callable(sf) else 1024 + 4096 + 3072, dtype=torch.float32, device=device)
        self.sync = torch.zeros(128, dtype=torch.int32, device=device)  # 128 words: R11_GRPQ counters at [16 + 8 g]
        inv = 1e6 ** (-torch.arange(0, 128, 2, dtype=torch.float64) / 128)
        ang = 128.0 * inv
        self.rope = torch.cat([torch.cos(ang), torch.sin(ang)]).float().to(device)
        self.pval = torch.zeros(self.grid, dtype=torch.float32, device=device)
        self.pidx = torch.zeros(self.grid, dtype=torch.int32, device=device)
        self.tail = [self.scratch.data_ptr(), self.sync.data_ptr(), self.pval.data_ptr(),
                     self.pidx.data_ptr(), self.rope.data_ptr()]
        torch.cuda.synchronize(device)

    def args(self, pointers: list[int]):
        """A (values, pointer-array) pair in cuLaunchCooperativeKernel layout."""
        ptrs = list(pointers) + self.tail
        vals = (ctypes.c_void_p * len(ptrs))(*ptrs)
        addrs = (ctypes.c_void_p * len(ptrs))(
            *[ctypes.addressof(vals) + i * ctypes.sizeof(ctypes.c_void_p)
              for i in range(len(ptrs))])
        return vals, addrs

    def launch(self, arg_pair, stream: int) -> None:
        self.check(self.driver.cuLaunchCooperativeKernel(
            self.func, self.grid, 1, 1, THREADS, 1, 1, self.dyn_smem, stream,
            ctypes.addressof(arg_pair[1])))


class OpusStep:
    precision = "fp32"

    def __init__(self, name: str, variant: str, kernel: _Kernel | None, inputs: dict,
                 config: dict[str, Any], runner=None):
        import torch
        self.torch = torch
        self.arm = name
        self.variant = variant
        self.config = config
        self._kernel = kernel
        self._inputs = inputs  # their device pointers are baked into the arg arrays
        self._runner = runner
        self._chained = False
        self._graphs: dict[bool, Any] = {}
        self._outs: dict | None = None
        dev = inputs["token"].device
        self._token_fixture = inputs["token"].detach().clone()
        if variant == "submission":
            self._outs = runner(inputs)
            return
        self.token_buf = self._token_fixture.clone()
        self.next_buf = torch.zeros((), device=dev, dtype=torch.int64)
        self.chain_buf = self._token_fixture.clone()
        self.logits = torch.zeros(151936, device=dev, dtype=torch.float32)
        self.kw = torch.zeros(28, 8, 128, device=dev, dtype=torch.bfloat16)
        self.vw = torch.zeros(28, 8, 128, device=dev, dtype=torch.bfloat16)
        self._arrays = {False: kernel.args(self._pointers(self.token_buf, self.next_buf)),
                        True: kernel.args(self._pointers(self.chain_buf, self.chain_buf))}
        if variant == "graph":
            self._capture()

    def _pointers(self, token, nxt) -> list[int]:
        ptrs = [(token if n == "token" else self._inputs[n]).data_ptr()
                for n in self._kernel.input_order]
        return ptrs + [self.logits.data_ptr(), nxt.data_ptr(),
                       self.kw.data_ptr(), self.vw.data_ptr()]

    def _capture(self) -> None:
        torch = self.torch
        try:
            self._kernel.launch(self._arrays[False], torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            for chained in (False, True):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._kernel.launch(self._arrays[chained],
                                        torch.cuda.current_stream().cuda_stream)
                self._graphs[chained] = g
        except Exception as e:  # capture failures surface as several exception types
            self._graphs.clear()
            raise RuntimeError(f"cooperative graph capture unavailable: {e}") from e

    # -- Step protocol
    def launch(self) -> None:
        if self.variant == "graph":
            self._graphs[self._chained].replay()
        elif self.variant == "submission":
            # Documented exception to the no-allocation rule (the MegaBench call contract).
            self._outs = self._runner(self._inputs)
        else:
            self._kernel.launch(self._arrays[self._chained],
                                self.torch.cuda.current_stream().cuda_stream)

    def outputs(self) -> dict[str, Any]:
        if self.variant == "submission":
            return self._outs
        return {"logits": self.logits,
                "next_token": self.chain_buf if self._chained else self.next_buf,
                "k_write": self.kw, "v_write": self.vw}

    def set_chained(self, on: bool) -> None:
        if self.variant == "submission":
            return  # unsupported: config["chained_supported"] is False
        on = bool(on)
        if on:
            self.chain_buf.copy_(self._token_fixture)
        self._chained = on

    def close(self) -> None:
        self._graphs.clear()
        for n in ("token_buf", "next_buf", "chain_buf", "logits", "kw", "vw"):
            if hasattr(self, n):
                delattr(self, n)
        self._arrays = {}
        self._outs = None
        self._runner = None
        self._inputs = {}


class OpusArm:
    precision = "fp32"

    def __init__(self, name: str = "opus", submission: Path | str = DEFAULT_SUBMISSION,
                 case_id: str = BASE_CASE_ID, **_: Any):
        self.name = name
        self.submission = Path(submission)
        self.case_id = case_id
        self._module = None
        self._sha = ""
        self._kernels: dict[str, _Kernel] = {}
        self._runner = None

    def variants(self) -> tuple[str, ...]:
        return VARIANTS

    def supports(self, shape: Shape, variant: str) -> str | None:
        if variant not in VARIANTS:
            return f"unknown variant {variant!r}"
        from ...cases import select_cases
        want = select_cases("all", [self.case_id])[0].params
        have = megabench_case(shape).params
        if any(want.get(k) != have.get(k) for k in GEOMETRY_KEYS):
            return "compiled for b1-s128 only (#define CTX 128)"
        return None

    def _load(self):
        if self._module is None:
            dest, self._sha = copy_submission(self.submission)
            self._module = _import_copy(dest)
        return self._module

    def _case_dict(self) -> dict:
        from ...cases import select_cases
        case = select_cases("all", [self.case_id])[0]
        return {"id": case.id, "params": dict(case.params)}

    def prepare(self, shape: Shape, inputs: dict, variant: str, device: str):
        import torch
        why = self.supports(shape, variant)
        if why:
            raise ValueError(why)
        module = self._load()
        if self._runner is None:
            self._runner = module.build(self._case_dict())
        kernel = None
        if variant != "submission":
            if device not in self._kernels:
                self._kernels[device] = _Kernel(module, torch.device(device))
            kernel = self._kernels[device]
        cfg = {"source": str(self.submission),
               "source_sha256": {"submission.py": self._sha},
               "variant": variant,
               "launch_mechanism": {
                   "direct": "cuLaunchCooperativeKernel via cuda.bindings, outputs preallocated",
                   "graph": "cuLaunchCooperativeKernel captured in a CUDA graph",
                   "submission": "candidate run() (allocates outputs per call)"}[variant],
               "graph": variant == "graph",
               "chained_supported": variant != "submission"}
        if kernel is not None:
            cfg.update(grid=kernel.grid, block=THREADS, regs=kernel.regs,
                       static_smem=kernel.static_smem, dyn_smem=kernel.dyn_smem)
        step = OpusStep(self.name, variant, kernel, inputs, cfg, self._runner)
        if variant != "submission":
            ref = self._runner(inputs)
            torch.cuda.synchronize()
            step.set_chained(False)
            step.launch()
            torch.cuda.synchronize()
            got = step.outputs()
            same = all(torch.equal(got[k], ref[k]) for k in OUTPUT_KEYS)
            cfg["bitexact_vs_submission"] = bool(same)
            if not same:
                cfg["bitexact_max_abs_diff"] = {
                    k: float((got[k].double() - ref[k].double()).abs().max())
                    for k in OUTPUT_KEYS}
        return step
