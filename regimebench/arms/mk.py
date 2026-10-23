"""mk: the bespoke tcgen05 decode megakernel (kernels/mk/, docs/briefs/). v0 = hv2g MTN skeleton + xover tcgen05 GEMM phases + the hv2g
attention as a placeholder behind the A1 interface.

Variants: ``auto`` (smallest token bucket NPAD in {16, 32, 64, 128, 256} >= total_q, CUDA graph) and ``auto-direct`` (same, no graph).
One cooperative launch per step (total_q <= 256). Options (``load_arm("mk", **opts)``): ``stamps`` (MK_STAMPS), ``null`` (MK_NULL bitmask),
``ku``, ``ring``, ``ts`` (MK_TS: tile-stream QKV with B = bf16 ``xb`` by TMA at NPAD >= ts (1 = 128); default 256, 0 = off), ``attn`` ("placeholder" | "splitkv"). The step clones the fixture KV into a packed cache with CTX zero padding rows
(as hv2g), encodes the 8 weight tensor maps once, and ``outputs()`` gathers k_new / v_new from the cache.
"""

from __future__ import annotations

from typing import Any

from ..kernels.mk import build as B
from ..kernels.mk.coop import Args, CoopKernel, i32
from ..workload import Workload

BASE_VARIANTS = ("auto",)
VARIANTS = ("auto", "auto-direct")
# kernel weight arguments, in kernel order (the weight tensors that are read through pointers; GEMM weights go through the tensor maps)
VEC_WEIGHTS = ("ln1", "qn", "kn", "ln2", "fnorm", "embed")


def _parse(variant: str) -> tuple[str, bool]:
    direct = variant.endswith("-direct")
    return (variant[:-7] if direct else variant), direct


class _Loaded:
    def __init__(self, npad: int, knobs: tuple, attn: str, device):
        import torch
        torch.cuda.init()
        torch.zeros(1, device=device)
        self.nb = torch.cuda.get_device_properties(device).multi_processor_count
        d = {**B.DEFAULT_KNOBS, **dict(knobs)}
        cubin, info, _ = B.compile_build(npad, self.nb, knobs, attn=attn)
        self.ptxas = info
        self.knobs = d
        self.dyn = B.dyn_smem(npad, d["MK_KU"], d["MK_RING"], lmp=d["MK_LMP"], lmr=d["MK_LMR"], ts=d["MK_TS"])
        self.kernel = CoopKernel(cubin, B.KERNEL, self.dyn)
        self.regs, self.static_smem = self.kernel.regs, self.kernel.static_smem
        self.local_bytes, self.dyn_smem = self.kernel.local_bytes, self.kernel.dyn_smem


class MkStep:
    arm = "mk"
    precision = "bf16"

    def __init__(self, arm: "MkArm", w: Workload, inputs: dict, variant: str, device: str):
        import torch
        self.torch, self.w, self.variant, self.device = torch, w, variant, device
        _, direct = _parse(variant)
        self.npad = B.bucket_for(w.total_q)
        g = w.model
        dev = torch.device(device)
        ld = arm._kernel(self.npad, dev)
        self.kern = ld
        self.nb = ld.nb
        self.desc = d = B.build_descriptors([(s.kv_len, s.q_len) for s in w.seqs], B.MAXTOK)
        assert len(d.chunks) == 1, "mk handles one launch of <= 256 tokens"
        c = d.chunks[0]
        rows = d.cache_rows
        assert rows == w.cache_tokens
        self.rows = rows
        row_elems = g.kv_heads * g.head_dim
        self.kc = torch.zeros((g.layers * rows + B.CTX) * row_elems, dtype=torch.bfloat16, device=dev)   # CTX zero rows at the end
        self.vc = torch.zeros_like(self.kc)
        self.kview = self.kc[: g.layers * rows * row_elems].view(g.layers, rows, g.kv_heads, g.head_dim)
        self.vview = self.vc[: g.layers * rows * row_elems].view(g.layers, rows, g.kv_heads, g.head_dim)
        self.kview.copy_(inputs["kcache"])
        self.vview.copy_(inputs["vcache"])
        self.logits = torch.zeros((w.n_seq, g.vocab), dtype=torch.float32, device=dev)
        self.next_token = torch.zeros(w.n_seq, dtype=torch.int64, device=dev)
        M = B.MAXTOK_MK
        f32 = dict(dtype=torch.float32, device=dev)
        self.x = torch.zeros(M * B.H, **f32)
        self.qkv = torch.zeros(M * 4096, **f32)
        self.qb = torch.zeros(M * B.QH * B.HD, dtype=torch.bfloat16, device=dev)
        self.ao = torch.zeros(M * B.QH * B.HD, dtype=torch.bfloat16, device=dev)
        self.gu = torch.zeros(M * 6144, **f32)
        self.rr = torch.zeros(2 * M, **f32)
        self.xb = torch.zeros(M * B.H, dtype=torch.bfloat16, device=dev)   # MK_TS: bf16(x * ln1) [MAXTOK][HID], rows >= ntok stay zero (B tiles of the tile-stream QKV)
        self.tk = torch.zeros(8, dtype=torch.int32, device=dev)             # MK_TS: D-phase row-block tickets (the last user resets them)
        self.sync = torch.zeros(B.SYNC_INTS, dtype=torch.int32, device=dev)
        self.pkey = torch.zeros(self.nb * M, dtype=torch.int32, device=dev)
        self.pidx = torch.zeros(self.nb * M, dtype=torch.int32, device=dev)
        self.rope = B.rope_table(d.max_pos).to(dev)
        self.tokens = inputs["tokens"]
        self.t_base = torch.tensor(d.t_base, dtype=torch.int32, device=dev)
        self.t_vis = torch.tensor(d.t_vis, dtype=torch.int32, device=dev)
        self.lm_tok = torch.tensor(c.lm_tok or [0], dtype=torch.int32, device=dev)
        self.lm_seq = torch.tensor(c.lm_seq or [0], dtype=torch.int32, device=dev)
        self._weights = {k: inputs[k] for k in (*B.WEIGHT_MAPS, *VEC_WEIGHTS)}   # keep the fixture tensors referenced
        ts = bool(ld.knobs["MK_TS"])
        blob = B.encode_maps(self._weights, {k: self._weights[k].data_ptr() for k in B.WEIGHT_MAPS},
                             workspace={"xb": self.xb.data_ptr()} if ts else None)   # map 8 = xb (128 B more) only for MK_TS builds
        self.maps = torch.frombuffer(bytearray(blob), dtype=torch.uint8).to(dev)
        assert self.maps.data_ptr() % 128 == 0
        wp = [self._weights[k].data_ptr() for k in VEC_WEIGHTS]
        self.args = Args([self.tokens.data_ptr(), *wp, self.maps.data_ptr(), self.kc.data_ptr(), self.vc.data_ptr(),
                          self.logits.data_ptr(), self.next_token.data_ptr(), self.x.data_ptr(), self.qkv.data_ptr(),
                          self.qb.data_ptr(), self.ao.data_ptr(), self.gu.data_ptr(), self.rr.data_ptr(), self.sync.data_ptr(),
                          self.pkey.data_ptr(), self.pidx.data_ptr(), self.rope.data_ptr(), self.t_base.data_ptr(),
                          self.t_vis.data_ptr(), self.lm_tok.data_ptr(), self.lm_seq.data_ptr(),
                          i32(c.ntok), i32(len(c.lm_tok)), i32(self.rows), self.xb.data_ptr(), self.tk.data_ptr()])   # xb, tk: last kernel arguments
        self.config: dict[str, Any] = {
            "kernel": B.KERNEL, "npad": self.npad, "graph": not direct, "proxy": False, "launches_per_step": 1,
            "grid": self.nb, "block": 512, "dyn_smem": ld.dyn_smem, "regs": ld.regs, "static_smem": ld.static_smem,
            "local_bytes": ld.local_bytes, "ptxas": ld.ptxas, "attn": arm.attn, "knobs": {**B.DEFAULT_KNOBS, **arm.knobs},
            "source_sha256": B.source_sha256(arm.attn),
        }
        self._graph = None
        if not direct:
            self._capture()

    def _stream(self) -> int:
        return self.torch.cuda.current_stream().cuda_stream

    def _launch_all(self, stream: int) -> None:
        self.kern.kernel.launch(self.nb, self.args, stream)

    def _capture(self) -> None:
        torch = self.torch
        try:
            self._launch_all(self._stream())   # warm-up outside the capture
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._launch_all(self._stream())
            self._graph = g
        except Exception as e:
            self._graph = None
            raise RuntimeError(f"cooperative graph capture failed: {e}") from e

    def launch(self) -> None:
        if self._graph is not None:
            self._graph.replay()
        else:
            self._launch_all(self._stream())

    def stamps(self):
        """MK_STAMPS builds: uint64 [nb, n_stamp] %globaltimer values of the last launch (even 2b = arrive of barrier b, 2b + 1 = wait exit,
        then the end of the LM phase). Unverified on a GPU."""
        import numpy as np
        drv = self.kern.kernel.driver
        err, dptr, size = drv.cuModuleGetGlobal(self.kern.kernel.module, b"mk_stamps")
        if int(err) != 0:
            raise RuntimeError("mk_stamps not found (build with stamps=1)")
        self.torch.cuda.synchronize()
        a = np.empty(size // 8, dtype=np.uint64)
        (err,) = drv.cuMemcpyDtoH(a.ctypes.data, dptr, size)
        if int(err) != 0:
            raise RuntimeError(f"cuMemcpyDtoH: {err}")
        return a.reshape(self.nb, -1)

    def outputs(self) -> dict[str, Any]:
        torch = self.torch
        d = self.desc
        idx = torch.tensor([b + v - 1 for b, v in zip(d.t_base, d.t_vis)], dtype=torch.int64, device=self.kview.device)
        k_new = self.kview[:, idx].permute(1, 0, 2, 3).contiguous()
        v_new = self.vview[:, idx].permute(1, 0, 2, 3).contiguous()
        return {"logits": self.logits.clone(), "next_token": self.next_token.clone(), "k_new": k_new, "v_new": v_new}

    def close(self) -> None:
        self._graph = None
        self.args = None
        for n in ("kc", "vc", "kview", "vview", "logits", "next_token", "x", "qkv", "qb", "ao", "gu", "rr", "xb", "tk", "sync", "pkey", "pidx",
                  "rope", "t_base", "t_vis", "lm_tok", "lm_seq", "maps"):
            if hasattr(self, n):
                delattr(self, n)
        self._weights = {}


class MkArm:
    name = "mk"

    def __init__(self, stamps: int = 0, null: int = 0, ku: int = 128, ring: int = 2, ts: int = 256, attn: str = "placeholder", **_: Any):
        self.attn = attn
        self.knobs = {"MK_STAMPS": int(stamps), "MK_NULL": int(null), "MK_KU": int(ku), "MK_RING": int(ring), "MK_TS": int(ts)}
        self._loaded: dict[tuple, _Loaded] = {}

    def variants(self) -> tuple[str, ...]:
        return VARIANTS

    def supports(self, w: Workload, variant: str) -> str | None:
        base, _ = _parse(variant)
        if base not in BASE_VARIANTS:
            return f"unknown variant {variant!r}"
        if w.model.name != "qwen3-0.6b":
            return "mk is specialised to Qwen3-0.6B"
        if w.total_q < 1:
            return "empty step"
        if w.total_q > B.BUCKETS[-1]:
            return f"mk handles 1..{B.BUCKETS[-1]} tokens per step (total_q {w.total_q})"
        return None

    def _kernel(self, npad: int, device) -> _Loaded:
        key = (npad, str(device))
        if key not in self._loaded:
            self._loaded[key] = _Loaded(npad, tuple(sorted(self.knobs.items())), self.attn, device)
        return self._loaded[key]

    def prepare(self, w: Workload, inputs: dict, variant: str, device: str) -> MkStep:
        why = self.supports(w, variant)
        if why:
            raise ValueError(why)
        return MkStep(self, w, inputs, variant, device)
