"""NVRTC compile + cooperative launch helpers shared by the mk arm (copied from kernels/hv2g/coop.py, default arch sm_100a) (cuda.bindings driver API).

Compiling needs no GPU (``nvrtc_compile`` takes an explicit arch); loading and launching do.
"""

from __future__ import annotations

import ctypes
from typing import Any, Sequence

DEFAULT_ARCH = "sm_100a"


def check(res):
    err = res[0]
    if int(err) != 0:
        raise RuntimeError(f"CUDA error: {err}")
    return res[1:] if len(res) > 2 else (res[1] if len(res) == 2 else None)


def nvrtc_compile(src: str, defines: Sequence[str] = (), arch: str = DEFAULT_ARCH,
                  ptxas_verbose: bool = False, name: str = "kernel.cu") -> tuple[bytes, str]:
    """Compile ``src`` for ``arch`` -> (cubin, nvrtc log). ``defines`` like ["NB=148", "MAXTOK=256"]."""
    from cuda.bindings import nvrtc

    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
    if int(err) != 0:
        raise RuntimeError(f"nvrtcCreateProgram: {err}")
    opts = [f"--gpu-architecture={arch}".encode(), b"-std=c++17", b"--fmad=true", b"-default-device"]
    opts += [f"-D{d}".encode() for d in defines]
    if ptxas_verbose:
        opts.append(b"--ptxas-options=-v")
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    _, n = nvrtc.nvrtcGetProgramLogSize(prog)
    log = b" " * n
    nvrtc.nvrtcGetProgramLog(prog, log)
    text = log.decode(errors="replace").strip("\x00 \n")
    if int(err) != 0:
        raise RuntimeError("NVRTC compile failed:\n" + text)
    _, n = nvrtc.nvrtcGetCUBINSize(prog)
    cubin = b" " * n
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    nvrtc.nvrtcDestroyProgram(prog)
    return cubin, text


def parse_ptxas(log: str) -> dict:
    """Pull registers / spills / smem out of a ``--ptxas-options=-v`` log (first entry function)."""
    import re
    out: dict[str, Any] = {}
    m = re.search(r"Used (\d+) registers.*?(\d+) bytes smem", log)
    if m:
        out["regs"], out["static_smem"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log)
    if m:
        out["stack"], out["spill_stores"], out["spill_loads"] = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return out


class Args:
    """Kernel argument block for cuLaunchKernel: ints (ctypes.c_int) and pointers (ctypes.c_void_p)."""

    def __init__(self, values: Sequence[Any]):
        self.values = []
        for v in values:
            if isinstance(v, bool):
                self.values.append(ctypes.c_int(int(v)))
            elif isinstance(v, int) and not isinstance(v, ctypes._SimpleCData):
                self.values.append(ctypes.c_void_p(v))   # plain int = device pointer
            else:
                self.values.append(v)                    # ctypes object (c_int, c_float, ...)
        self.ptrs = (ctypes.c_void_p * len(self.values))(
            *[ctypes.addressof(v) for v in self.values])


def i32(v: int) -> ctypes.c_int:
    return ctypes.c_int(int(v))


class CoopKernel:
    """A loaded kernel plus its launch geometry. One 512-thread block per SM, cooperative launch."""

    def __init__(self, cubin: bytes, func_name: str, dyn_smem: int, threads: int = 512):
        from cuda.bindings import driver
        self.driver = driver
        self.dyn_smem = int(dyn_smem)
        self.threads = threads
        self.module = check(driver.cuModuleLoadData(cubin))
        self.func = check(driver.cuModuleGetFunction(self.module, func_name.encode()))
        if self.dyn_smem:
            check(driver.cuFuncSetAttribute(
                self.func, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                self.dyn_smem))
        per_sm = check(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(self.func, threads, self.dyn_smem))
        if per_sm < 1:
            raise RuntimeError(f"{func_name} cannot be resident (dyn smem {self.dyn_smem})")
        self.regs = int(check(driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, self.func)))
        self.static_smem = int(check(driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, self.func)))
        self.local_bytes = int(check(driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, self.func)))

    def launch(self, grid: int, args: Args, stream: int) -> None:
        check(self.driver.cuLaunchCooperativeKernel(
            self.func, grid, 1, 1, self.threads, 1, 1, self.dyn_smem, stream,
            ctypes.addressof(args.ptrs)))
