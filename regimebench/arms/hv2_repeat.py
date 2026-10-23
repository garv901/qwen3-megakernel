"""hv2-repeat: the unchanged hv2 binary as a timing PROXY for a step of N tokens.

hv2 (kernels/hv2/hv2_submission.py, frozen) decodes ONE token at batch 1 with a 128-position cache. One Step here is
N = total_q back-to-back launches of exactly that kernel, i.e. "what the step would cost if the megakernel simply ran
the b1 / kv128 decode once per token". It computes the wrong thing for every other workload on purpose (every launch
is the same b1/kv128 decode), so ``config["proxy"] = True`` and the harness skips its accuracy gate; ``outputs()``
returns zeros of the contract shapes.

Variants: ``graph`` (headline: the N launches captured in ONE CUDA graph, cooperative launches captured as in
_upstream/opus.py) and ``direct`` (N cuLaunchCooperativeKernel calls per ``launch``). Outputs are preallocated (the
frozen run() allocates per call).
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from ..workload import Workload
from .base import expected_outputs

HV2_PATH = Path(__file__).resolve().parent.parent / "kernels" / "hv2" / "hv2_submission.py"
MAX_LAUNCHES = 8192
VARIANTS = ("graph", "direct")
THREADS = 512
HV2_CASE = {"params": {"batch": 1, "context": 128, "layers": 28, "hidden": 1024, "q_heads": 16, "kv_heads": 8,
                       "head_dim": 128, "intermediate": 3072, "vocab": 151936}}
WEIGHT_KEYS = ("ln1", "wq", "wk", "wv", "qn", "kn", "wo", "ln2", "wg", "wu", "wd", "fnorm", "embed")


def load_hv2():
    """Import the frozen hv2 module by path (no edits, nothing written next to it)."""
    name = "regimebench_hv2_submission"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HV2_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def hv2_sha256() -> str:
    return hashlib.sha256(HV2_PATH.read_bytes()).hexdigest()


class Hv2Kernel:
    """The compiled hv2 ``qwen3_step`` plus the tail buffers build() would own (same layout, preallocated)."""

    def __init__(self, module, device):
        import torch
        from cuda.bindings import driver
        self.torch, self.driver, self.module_py = torch, driver, module
        self.check = module._check
        if module._skd_on() or module._grpq_on():
            raise RuntimeError("hv2-repeat expects the default hv2 build (no R11_SKD / R11_GRPQ)")
        self.input_order = tuple(module._INPUT_ORDER)
        torch.cuda.init()
        torch.zeros(1, device=device)  # make torch's primary context current
        self.grid = torch.cuda.get_device_properties(device).multi_processor_count
        cubin = module._compile(device, self.grid)
        self.module = self.check(driver.cuModuleLoadData(cubin))
        self.func = self.check(driver.cuModuleGetFunction(self.module, b"qwen3_step"))
        self.dyn_smem = int(module._dyn_smem())
        if self.dyn_smem:
            self.check(driver.cuFuncSetAttribute(
                self.func, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self.dyn_smem))
        if self.check(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(self.func, THREADS, self.dyn_smem)) < 1:
            raise RuntimeError("qwen3_step cannot be resident")
        self.regs = int(self.check(driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, self.func)))
        self.scratch = torch.zeros(int(module._scratch_floats()), dtype=torch.float32, device=device)
        self.sync = torch.zeros(int(module._SYNC_INTS), dtype=torch.int32, device=device)
        inv = 1e6 ** (-torch.arange(0, 128, 2, dtype=torch.float64) / 128)
        ang = 128.0 * inv   # hv2 bakes RoPE for position 128
        self.rope = torch.cat([torch.cos(ang), torch.sin(ang)]).float().to(device)
        self.pval = torch.zeros(self.grid, dtype=torch.float32, device=device)
        self.pidx = torch.zeros(self.grid, dtype=torch.int32, device=device)
        torch.cuda.synchronize(device)

    def pack(self, pointers: list[int]):
        import ctypes
        ptrs = list(pointers) + [self.scratch.data_ptr(), self.sync.data_ptr(), self.pval.data_ptr(),
                                 self.pidx.data_ptr(), self.rope.data_ptr()]
        vals = (ctypes.c_void_p * len(ptrs))(*ptrs)
        addrs = (ctypes.c_void_p * len(ptrs))(
            *[ctypes.addressof(vals) + i * ctypes.sizeof(ctypes.c_void_p) for i in range(len(ptrs))])
        return vals, addrs

    def launch(self, packed, stream: int) -> None:
        import ctypes
        self.check(self.driver.cuLaunchCooperativeKernel(
            self.func, self.grid, 1, 1, THREADS, 1, 1, self.dyn_smem, stream, ctypes.addressof(packed[1])))


class Hv2RepeatStep:
    arm = "hv2-repeat"
    precision = "bf16"

    def __init__(self, w: Workload, variant: str, kernel: Hv2Kernel, tensors: dict, config: dict[str, Any], device):
        import torch
        self.torch, self.w, self.variant, self.config = torch, w, variant, config
        self._kernel, self._t, self._device = kernel, tensors, device
        self.n = w.total_q
        self.logits = torch.zeros(151936, dtype=torch.float32, device=device)
        self.next_token = torch.zeros((), dtype=torch.int64, device=device)
        self.k_write = torch.zeros(28, 8, 128, dtype=torch.bfloat16, device=device)
        self.v_write = torch.zeros(28, 8, 128, dtype=torch.bfloat16, device=device)
        ptrs = [tensors[name].data_ptr() for name in kernel.input_order]
        ptrs += [self.logits.data_ptr(), self.next_token.data_ptr(), self.k_write.data_ptr(), self.v_write.data_ptr()]
        self._packed = kernel.pack(ptrs)
        self._graph = None
        if variant == "graph":
            self._capture()

    def _stream(self) -> int:
        return self.torch.cuda.current_stream().cuda_stream

    def _capture(self) -> None:
        torch = self.torch
        try:
            self._kernel.launch(self._packed, self._stream())   # warm-up outside the capture
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                s = self._stream()
                for _ in range(self.n):
                    self._kernel.launch(self._packed, s)
            self._graph = g
        except Exception as e:  # capture failures surface as several exception types
            self._graph = None
            raise RuntimeError(f"cooperative graph capture of {self.n} launches failed: {e}") from e

    def launch(self) -> None:
        if self._graph is not None:
            self._graph.replay()
        else:
            s = self._stream()
            for _ in range(self.n):
                self._kernel.launch(self._packed, s)

    def raw_outputs(self) -> dict:
        """What the kernel wrote (hv2 output names), for the N=1 check against hv2 run()."""
        return {"logits": self.logits, "next_token": self.next_token,
                "k_write": self.k_write, "v_write": self.v_write}

    def outputs(self) -> dict[str, Any]:
        torch = self.torch
        return {name: torch.zeros(shape, dtype=getattr(torch, dt), device=self._device)
                for name, (shape, dt) in expected_outputs(self.w).items()}

    def close(self) -> None:
        self._graph = None
        for n in ("logits", "next_token", "k_write", "v_write"):
            if hasattr(self, n):
                delattr(self, n)
        self._packed = None
        self._t = {}


class Hv2RepeatArm:
    name = "hv2-repeat"

    def __init__(self, **_: Any):
        self._kernels: dict[str, Hv2Kernel] = {}
        self._module = None
        self._cache: dict[str, tuple] = {}   # per device: (kcache, vcache) random [28,128,8,128]

    def variants(self) -> tuple[str, ...]:
        return VARIANTS

    def supports(self, w: Workload, variant: str) -> str | None:
        if variant not in VARIANTS:
            return f"unknown variant {variant!r}"
        if w.total_q > MAX_LAUNCHES:
            return f"proxy needs {w.total_q} launches per step (> {MAX_LAUNCHES})"
        return None

    def _tensors(self, inputs: dict, device: str) -> dict:
        import torch
        if device not in self._cache:
            gen = torch.Generator(device=device).manual_seed(128)
            mk = lambda: (torch.randn((28, 128, 8, 128), generator=gen, device=device) * 0.1).to(torch.bfloat16)
            self._cache[device] = (mk(), mk())
        kc, vc = self._cache[device]
        t = {k: inputs[k] for k in WEIGHT_KEYS}   # fixture weights, read only
        t["token"] = inputs["tokens"][:1].clone()
        t["kcache"], t["vcache"] = kc, vc
        return t

    def prepare(self, w: Workload, inputs: dict, variant: str, device: str) -> Hv2RepeatStep:
        import torch
        why = self.supports(w, variant)
        if why:
            raise ValueError(why)
        if self._module is None:
            self._module = load_hv2()
        if device not in self._kernels:
            self._kernels[device] = Hv2Kernel(self._module, torch.device(device))
        kernel = self._kernels[device]
        cfg = {"proxy": True, "launches_per_step": w.total_q, "graph": variant == "graph",
               "source": str(HV2_PATH.relative_to(HV2_PATH.parents[3])), "source_sha256": hv2_sha256(),
               "grid": kernel.grid, "block": THREADS, "regs": kernel.regs, "dyn_smem": kernel.dyn_smem,
               "note": "N copies of the b1/kv128 hv2 decode; not the workload's math"}
        return Hv2RepeatStep(w, variant, kernel, self._tensors(inputs, device), cfg, device)
