"""hv2g: the hv2 megakernel generalized to a runtime context length and many tokens / sequences (kernels/hv2g/).

Variants: ``auto`` (headline: the MT1 build when total_q == 1, else the general MTN build), ``mtn`` (always MTN);
an optional ``-direct`` suffix (``auto-direct``) launches without a CUDA graph. One launch = one cooperative kernel; MTN
steps of more than MAXTOK = 256 tokens run as consecutive launches captured in one graph (``launches_per_step``).
The step clones the fixture KV into a packed cache with CTX padding rows at the end (the kernel reads, and masks, up to
CTX - 1 rows past the last visible one) and writes the new K/V into it; ``outputs()`` gathers k_new / v_new from there.
"""

from __future__ import annotations

from typing import Any

from ..kernels.hv2g import build as B
from ..kernels.hv2g.coop import Args, CoopKernel, i32
from ..workload import Workload
from .base import expected_outputs

BASE_VARIANTS = ("auto", "mtn")
VARIANTS = BASE_VARIANTS
WEIGHT_ORDER = ("ln1", "wq", "wk", "wv", "qn", "kn", "wo", "ln2", "wg", "wu", "wd", "fnorm", "embed")


def _parse(variant: str) -> tuple[str, bool]:
    direct = variant.endswith("-direct")
    base = variant[:-7] if direct else variant
    return base, direct


def use_mt1(w: Workload, base: str) -> bool:
    return base == "auto" and w.total_q == 1


class Hv2gStep:
    arm = "hv2g"
    precision = "bf16"

    def __init__(self, arm: "Hv2gArm", w: Workload, inputs: dict, variant: str, device: str):
        import torch
        self.torch, self.w, self.variant, self.device = torch, w, variant, device
        base, direct = _parse(variant)
        self.mt1 = use_mt1(w, base)
        g = w.model
        dev = torch.device(device)
        kern = arm._kernel(self.mt1, dev)
        self.kern = kern
        self.desc = d = B.build_descriptors([(s.kv_len, s.q_len) for s in w.seqs], B.MAXTOK)
        rows = d.cache_rows
        assert rows == w.cache_tokens
        self.rows = rows
        row_elems = g.kv_heads * g.head_dim
        # packed cache with CTX padding rows at the end (zeros)
        self.kc = torch.zeros((g.layers * rows + B.CTX) * row_elems, dtype=torch.bfloat16, device=dev)
        self.vc = torch.zeros_like(self.kc)
        self.kview = self.kc[: g.layers * rows * row_elems].view(g.layers, rows, g.kv_heads, g.head_dim)
        self.vview = self.vc[: g.layers * rows * row_elems].view(g.layers, rows, g.kv_heads, g.head_dim)
        self.kview.copy_(inputs["kcache"])
        self.vview.copy_(inputs["vcache"])
        self.logits = torch.zeros((w.n_seq, g.vocab), dtype=torch.float32, device=dev)
        self.next_token = torch.zeros(w.n_seq, dtype=torch.int64, device=dev)
        self.scratch = torch.zeros(B.scratch_floats(self.mt1), dtype=torch.float32, device=dev)
        self.sync = torch.zeros(B.SYNC_INTS, dtype=torch.int32, device=dev)
        self.nb = kern.nb
        npv = self.nb * (1 if self.mt1 else B.MAXTOK)
        self.pval = torch.zeros(npv, dtype=torch.float32, device=dev)
        self.pidx = torch.zeros(npv, dtype=torch.int32, device=dev)
        self.rope = B.rope_table(d.max_pos).to(dev)
        self.tokens = inputs["tokens"]
        self.t_base = torch.tensor(d.t_base, dtype=torch.int32, device=dev)
        self.t_vis = torch.tensor(d.t_vis, dtype=torch.int32, device=dev)
        self.lm_tok = torch.tensor(d.lm_tok or [0], dtype=torch.int32, device=dev)
        self.lm_seq = torch.tensor(d.lm_seq or [0], dtype=torch.int32, device=dev)
        self._weights = [inputs[k] for k in WEIGHT_ORDER]   # keep the fixture tensors referenced
        self.launches = self._build_launches()
        self.config: dict[str, Any] = {
            "kernel": B.kernel_name(self.mt1), "mt1": self.mt1, "graph": not direct, "proxy": False,
            "launches_per_step": len(self.launches), "grid": self.nb, "block": 512, "dyn_smem": kern.dyn_smem,
            "regs": kern.regs, "static_smem": kern.static_smem, "local_bytes": kern.local_bytes,
            "ptxas": kern.ptxas, "maxtok": B.MAXTOK, "source_sha256": B.source_sha256(),
        }
        self._graph = None
        if not direct:
            self._capture()

    def _build_launches(self) -> list[Args]:
        w, d = self.w, self.desc
        wp = [t.data_ptr() for t in self._weights]
        common_tail = [self.kc.data_ptr(), self.vc.data_ptr(), self.logits.data_ptr(), self.next_token.data_ptr(),
                       self.scratch.data_ptr(), self.sync.data_ptr(), self.pval.data_ptr(), self.pidx.data_ptr()]
        out: list[Args] = []
        tok = self.tokens.data_ptr()
        if self.mt1:
            kv = w.seqs[0].kv_len
            out.append(Args([tok, *wp, *common_tail, self.rope.data_ptr() + kv * 128 * 4,
                             i32(kv), i32(self.rows), i32(0)]))
            return out
        for c in d.chunks:
            out.append(Args([tok + c.t0 * 8, *wp, *common_tail, self.rope.data_ptr(),
                             self.t_base.data_ptr() + c.t0 * 4, self.t_vis.data_ptr() + c.t0 * 4,
                             self.lm_tok.data_ptr() + c.lm_off * 4, self.lm_seq.data_ptr() + c.lm_off * 4,
                             i32(c.ntok), i32(len(c.lm_tok)), i32(self.rows)]))
        return out

    def _stream(self) -> int:
        return self.torch.cuda.current_stream().cuda_stream

    def _launch_all(self, stream: int) -> None:
        for a in self.launches:
            self.kern.kernel.launch(self.nb, a, stream)

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
            raise RuntimeError(f"cooperative graph capture of {len(self.launches)} launches failed: {e}") from e

    def launch(self) -> None:
        if self._graph is not None:
            self._graph.replay()
        else:
            self._launch_all(self._stream())

    def outputs(self) -> dict[str, Any]:
        torch = self.torch
        d = self.desc
        idx = torch.tensor([b + v - 1 for b, v in zip(d.t_base, d.t_vis)], dtype=torch.int64, device=self.kview.device)
        k_new = self.kview[:, idx].permute(1, 0, 2, 3).contiguous()
        v_new = self.vview[:, idx].permute(1, 0, 2, 3).contiguous()
        return {"logits": self.logits.clone(), "next_token": self.next_token.clone(), "k_new": k_new, "v_new": v_new}

    def close(self) -> None:
        self._graph = None
        self.launches = []
        for n in ("kc", "vc", "kview", "vview", "logits", "next_token", "scratch", "sync", "pval", "pidx", "rope",
                  "t_base", "t_vis", "lm_tok", "lm_seq"):
            if hasattr(self, n):
                delattr(self, n)
        self._weights = []


class _Loaded:
    def __init__(self, mt1: bool, device):
        import torch
        torch.cuda.init()
        torch.zeros(1, device=device)
        self.nb = torch.cuda.get_device_properties(device).multi_processor_count
        cubin, info, _ = B.compile_build(mt1, self.nb)
        self.ptxas = info
        self.kernel = CoopKernel(cubin, B.kernel_name(mt1), B.dyn_smem(mt1))
        self.regs, self.static_smem = self.kernel.regs, self.kernel.static_smem
        self.local_bytes, self.dyn_smem = self.kernel.local_bytes, self.kernel.dyn_smem


class Hv2gArm:
    name = "hv2g"

    def __init__(self, **_: Any):
        self._loaded: dict[tuple, _Loaded] = {}

    def variants(self) -> tuple[str, ...]:
        return VARIANTS

    def supports(self, w: Workload, variant: str) -> str | None:
        base, _ = _parse(variant)
        if base not in BASE_VARIANTS:
            return f"unknown variant {variant!r}"
        if w.total_q < 1:
            return "empty step"
        if w.model.name != "qwen3-0.6b":
            return "hv2g is specialised to Qwen3-0.6B"
        return None

    def _kernel(self, mt1: bool, device) -> _Loaded:
        key = (mt1, str(device))
        if key not in self._loaded:
            self._loaded[key] = _Loaded(mt1, device)
        return self._loaded[key]

    def prepare(self, w: Workload, inputs: dict, variant: str, device: str) -> Hv2gStep:
        why = self.supports(w, variant)
        if why:
            raise ValueError(why)
        return Hv2gStep(self, w, inputs, variant, device)
