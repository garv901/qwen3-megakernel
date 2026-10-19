"""Compile, load and launch the Qwen3-0.6B decode megakernel.

The kernel source is `csrc/qwen3_step.cu`. NVRTC compiles it when a `Megakernel`
is created. One cooperative launch runs the whole decode step.

The kernel is specialized to one case: batch 1, decode at position 128
(KV prefix of 128 tokens, 129 attended positions), 28 layers, hidden 1024,
16 query heads, 8 KV heads, head dim 128, intermediate 3072, vocab 151936.
The kernel does not append to the KV cache. The caller writes `k_write` and
`v_write` into slot 128 if it wants a longer cache.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch

SOURCE_PATH = Path(__file__).resolve().parent.parent / "csrc" / "qwen3_step.cu"

# Kernel arguments, in order. All are device pointers.
INPUT_ORDER = ("token", "ln1", "wq", "wk", "wv", "qn", "kn", "wo", "ln2",
               "wg", "wu", "wd", "fnorm", "embed", "kcache", "vcache")

# Shape and dtype of every input.
INPUT_SPECS = {
    "token": ((), torch.int64),
    "ln1": ((28, 1024), torch.bfloat16),
    "wq": ((28, 2048, 1024), torch.bfloat16),
    "wk": ((28, 1024, 1024), torch.bfloat16),
    "wv": ((28, 1024, 1024), torch.bfloat16),
    "qn": ((28, 128), torch.bfloat16),
    "kn": ((28, 128), torch.bfloat16),
    "wo": ((28, 1024, 2048), torch.bfloat16),
    "ln2": ((28, 1024), torch.bfloat16),
    "wg": ((28, 3072, 1024), torch.bfloat16),
    "wu": ((28, 3072, 1024), torch.bfloat16),
    "wd": ((28, 1024, 3072), torch.bfloat16),
    "fnorm": ((1024,), torch.bfloat16),
    "embed": ((151936, 1024), torch.bfloat16),
    "kcache": ((28, 128, 8, 128), torch.bfloat16),
    "vcache": ((28, 128, 8, 128), torch.bfloat16),
}
OUTPUT_SPECS = {
    "logits": ((151936,), torch.float32),
    "next_token": ((), torch.int64),
    "k_write": ((28, 8, 128), torch.bfloat16),
    "v_write": ((28, 8, 128), torch.bfloat16),
}

THREADS = 512
# Dynamic shared memory of the kernel in bytes: STAGE_BYTES (5632) + 86016 for the
# gate / up rows + 43008 for the down rows. tests/test_cpu.py checks this number.
DYN_SMEM_BYTES = 5632 + 86016 + 43008
# Scratch floats: x (1024) + qkv (4096) + hid (3072).
SCRATCH_FLOATS = 1024 + 4096 + 3072
# Sync words: [0] barrier counter, [2] argmax ticket. The rest is spare.
SYNC_INTS = 128
# The kernel has no problem-size arguments. These must match the constants in the source.
EXPECTED_PARAMS = {"batch": 1, "context": 128, "layers": 28, "hidden": 1024, "q_heads": 16,
                   "kv_heads": 8, "head_dim": 128, "intermediate": 3072, "vocab": 151936}


def source_text() -> str:
    return SOURCE_PATH.read_text()


def compile_options(arch: str, n_blocks: int) -> list[str]:
    """NVRTC options. `arch` is for example "sm_100"."""
    return [f"--gpu-architecture={arch}", "-std=c++17", "--fmad=true", "-default-device", f"-DNB={n_blocks}"]


def compile_cubin(src: str, arch: str, n_blocks: int) -> bytes:
    """Compile `src` with NVRTC and return the cubin. This needs no GPU."""
    from cuda.bindings import nvrtc

    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"qwen3_step.cu", 0, [], [])
    if int(err) != 0:
        raise RuntimeError(f"nvrtcCreateProgram: {err}")
    opts = [o.encode() for o in compile_options(arch, n_blocks)]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if int(err) != 0:
        _, n = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * n
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError("NVRTC compile failed:\n" + log.decode(errors="replace"))
    _, n = nvrtc.nvrtcGetCUBINSize(prog)
    cubin = b" " * n
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    nvrtc.nvrtcDestroyProgram(prog)
    return bytes(cubin)


def _check(res):
    err = res[0]
    if int(err) != 0:
        raise RuntimeError(f"CUDA error: {err}")
    return res[1:] if len(res) > 2 else (res[1] if len(res) == 2 else None)


def rope_table(device) -> torch.Tensor:
    """RoPE cos then sin at position 128 (theta 1e6, head dim 128): fp32 [128]. The math runs in fp64."""
    inv = 1e6 ** (-torch.arange(0, 128, 2, dtype=torch.float64) / 128)
    ang = 128.0 * inv
    return torch.cat([torch.cos(ang), torch.sin(ang)]).float().to(device)


class Megakernel:
    """The decode megakernel on one GPU.

    `step(inputs)` runs one decode step and returns a dict with `logits`, `next_token`,
    `k_write` and `v_write`. `launch(stream)` is the low-level path: it enqueues one
    launch on the bound inputs and does not synchronize. Use `bind(inputs)` first.

    Attributes `cubin`, `module` and `func` give access to the loaded code, for tools
    that hook the launch.
    """

    def __init__(self, device=None):
        from cuda.bindings import driver

        self._driver = driver
        self.device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        torch.cuda.init()
        torch.cuda.set_device(self.device)
        torch.zeros(1, device=self.device)  # make the primary context of torch current
        major, minor = torch.cuda.get_device_capability(self.device)
        self.arch = f"sm_{major}{minor}"
        self.grid = torch.cuda.get_device_properties(self.device).multi_processor_count  # one CTA per SM
        self.cubin = compile_cubin(source_text(), self.arch, self.grid)
        self.module = _check(driver.cuModuleLoadData(self.cubin))
        self.func = _check(driver.cuModuleGetFunction(self.module, b"qwen3_step"))
        self.dyn_smem = DYN_SMEM_BYTES
        _check(driver.cuFuncSetAttribute(
            self.func, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self.dyn_smem))
        per_sm = _check(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(self.func, THREADS, self.dyn_smem))
        if per_sm < 1:
            raise RuntimeError("the kernel cannot be resident")

        dev = self.device
        self.scratch = torch.zeros(SCRATCH_FLOATS, dtype=torch.float32, device=dev)
        self.sync = torch.zeros(SYNC_INTS, dtype=torch.int32, device=dev)
        self.rope = rope_table(dev)
        self.pval = torch.zeros(self.grid, dtype=torch.float32, device=dev)
        self.pidx = torch.zeros(self.grid, dtype=torch.int32, device=dev)
        torch.cuda.synchronize(dev)

        n_args = len(INPUT_ORDER) + 4 + 5
        self._arg_vals = (ctypes.c_void_p * n_args)()
        self._arg_ptrs = (ctypes.c_void_p * n_args)(
            *[ctypes.addressof(self._arg_vals) + i * ctypes.sizeof(ctypes.c_void_p) for i in range(n_args)])
        tail = [self.scratch.data_ptr(), self.sync.data_ptr(), self.pval.data_ptr(), self.pidx.data_ptr(), self.rope.data_ptr()]
        for i, p in enumerate(tail):
            self._arg_vals[len(INPUT_ORDER) + 4 + i] = p
        self._i_out = len(INPUT_ORDER)       # index of the logits pointer; next_token, k_write, v_write follow
        self._next = None                    # Host V2: outputs of the next step() call
        self._bound = None                   # inputs kept alive by bind()
        self._out = None                     # outputs of the bound path
        self._tok = None                     # two token buffers of the chained path
        self._n_launch = 0
        self._chained = False

    # ------------------------------------------------------------------ outputs

    def _alloc(self):
        dev = self.device
        return tuple(torch.empty(shape, dtype=dt, device=dev) for shape, dt in OUTPUT_SPECS.values())

    @staticmethod
    def check_inputs(inputs: dict) -> None:
        for name in INPUT_ORDER:
            t = inputs[name]
            shape, dt = INPUT_SPECS[name]
            if tuple(t.shape) != shape or t.dtype != dt or not t.is_contiguous() or not t.is_cuda:
                raise ValueError(f"input {name}: expected a contiguous CUDA tensor {dt} {shape}, got {t.dtype} {tuple(t.shape)}")

    # ------------------------------------------------------------------ step (Host V2)

    def step(self, inputs: dict) -> dict:
        """Run one decode step. Host V2: the outputs of the next call are allocated right
        after the launch, while the kernel runs. Each call returns fresh tensors that no later call writes."""
        logits, next_token, k_write, v_write = self._next if self._next is not None else self._alloc()
        av = self._arg_vals
        for i, name in enumerate(INPUT_ORDER):
            av[i] = inputs[name].data_ptr()
        o = self._i_out
        av[o], av[o + 1], av[o + 2], av[o + 3] = logits.data_ptr(), next_token.data_ptr(), k_write.data_ptr(), v_write.data_ptr()
        self._launch(torch.cuda.current_stream(self.device).cuda_stream)
        self._next = self._alloc()
        return {"logits": logits, "next_token": next_token, "k_write": k_write, "v_write": v_write}

    # ------------------------------------------------------------------ bound path

    def bind(self, inputs: dict, chained: bool = False) -> dict:
        """Fix the inputs for `launch`. Returns the output tensors that `launch` writes.

        With `chained=True` the kernel reads its token from one device buffer and writes
        the next token to a second buffer. The two swap on every launch. No host sync and
        no extra kernel is needed. The final token is in `chain_token`."""
        self.check_inputs(inputs)
        self._bound = {k: inputs[k] for k in INPUT_ORDER}
        self._out = self._alloc()
        av = self._arg_vals
        for i, name in enumerate(INPUT_ORDER):
            av[i] = inputs[name].data_ptr()
        o = self._i_out
        av[o], av[o + 1], av[o + 2], av[o + 3] = [t.data_ptr() for t in self._out]
        self._chained = chained
        self._n_launch = 0
        if chained:
            self._tok = (inputs["token"].clone(), torch.zeros((), dtype=torch.int64, device=self.device))
        return dict(zip(OUTPUT_SPECS, self._out))

    @property
    def chain_token(self) -> torch.Tensor:
        """The token that the next chained launch reads (the argmax of the last one)."""
        return self._tok[self._n_launch & 1]

    def launch(self, stream=None) -> None:
        """Enqueue one launch on the bound inputs. No host sync. `stream` is a CUDA stream handle (int)."""
        if self._bound is None:
            raise RuntimeError("call bind(inputs) before launch()")
        if self._chained:
            av, o = self._arg_vals, self._i_out
            av[0] = self._tok[self._n_launch & 1].data_ptr()
            av[o + 1] = self._tok[(self._n_launch + 1) & 1].data_ptr()
        self._launch(torch.cuda.current_stream(self.device).cuda_stream if stream is None else stream)
        self._n_launch += 1

    def _launch(self, stream) -> None:
        _check(self._driver.cuLaunchCooperativeKernel(
            self.func, self.grid, 1, 1, THREADS, 1, 1, self.dyn_smem, stream, ctypes.addressof(self._arg_ptrs)))
