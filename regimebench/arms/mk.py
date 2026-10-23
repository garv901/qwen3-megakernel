"""mk: the bespoke tcgen05 decode megakernel (kernels/mk/, docs/briefs/). v0 = hv2g MTN skeleton + xover tcgen05 GEMM phases + the hv2g
attention as a placeholder behind the A1 interface.

Variants: ``auto`` (smallest token bucket NPAD in {16, 32, 64, 128, 256} >= total_q, CUDA graph) and ``auto-direct`` (same, no graph).
One cooperative launch per step (total_q <= 256). Options (``load_arm("mk", **opts)``): ``stamps`` (MK_STAMPS), ``null`` (MK_NULL bitmask),
``ku``, ``ring``, ``ts`` (MK_TS: tile-stream QKV with B = bf16 ``xb`` by TMA at NPAD >= ts (1 = 128); default 256, 0 = off), ``tsb`` (MK_TSB, default 0: P5b, GU and D on
the tile stream with act / xb2 buffers, the interleaved wg|wu weights and the idle-CTA finalizers; needs MK_TS on at the bucket), ``lmts`` (MK_LMTS, default 0: SPEC-P3c, the LM head on the tile stream
at TSB buckets when n_lm == ntok and lm_tok[s] == s; else the v0 LM path), ``attn`` ("placeholder" | "splitkv" | "auto"; default "placeholder"; SPEC-P2: splitkv = A1 split-KV attention v2 (splitkv-v2)
+ combine_head in the O phase when plan.s_max > 1 (else ao is staged, no combine); q_len > 1 (<= 16) builds with -DSKV_QMAX=16; auto = splitkv when every seq has q_len == 1 and max kv >= 512, else the placeholder, chosen per workload at prepare()). The step clones the fixture KV into a packed cache with CTX zero padding rows
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
ATTN_OPTIONS = ("placeholder", "splitkv", "auto")
VEC_WEIGHTS = ("ln1", "qn", "kn", "ln2", "fnorm", "embed")


def _parse(variant: str) -> tuple[str, bool]:
    direct = variant.endswith("-direct")
    return (variant[:-7] if direct else variant), direct


class _Loaded:
    def __init__(self, npad: int, knobs: tuple, attn: str, device, qmax: int = 1):
        self.attn = attn
        self.qmax = qmax
        import torch
        torch.cuda.init()
        torch.zeros(1, device=device)
        self.nb = torch.cuda.get_device_properties(device).multi_processor_count
        d = {**B.DEFAULT_KNOBS, **dict(knobs)}
        cubin, info, _ = B.compile_build(npad, self.nb, knobs, attn=attn, skv_qmax=qmax)
        self.ptxas = info
        self.knobs = d
        self.dyn = B.dyn_smem(npad, d["MK_KU"], d["MK_RING"], lmp=d["MK_LMP"], lmr=d["MK_LMR"], ts=d["MK_TS"], tsb=d["MK_TSB"], attn=attn)
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
        self.attn = arm.resolve_attn(w)   # "placeholder" | "splitkv" (auto resolved per workload)
        self.qmax = B.skv_qmax_of([(s.kv_len, s.q_len) for s in w.seqs]) if self.attn == "splitkv" else 1   # R6 build (-DSKV_QMAX=16) iff some q_len > 1
        ld = arm._kernel(self.npad, dev, self.attn, self.qmax)
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
        self.tk = torch.zeros(B.TK_SIZE, dtype=torch.int32, device=dev)     # MK_TS: D-phase row-block tickets (the last user resets them); MK_TSB: O tickets [0, 16), D tickets [16, 32)
        self.tsb = B.tsb_on(self.npad, ld.knobs)
        if self.tsb:   # P5b: bf16(x * ln2) [MAXTOK][HID], sum x^2 per token, act = bf16(silu * up) [MAXTOK][INTER] (rows >= ntok stay zero), wgu (built once per weight set)
            self.xb2 = torch.zeros(M * B.H, dtype=torch.bfloat16, device=dev)
            self.ss2 = torch.zeros(2 * M, **f32)   # [0, M) = ss2 (sum x^2 per token after O), [M, 2M) = ssf (after the last D, MK_LMTS; SPEC-P3c)
            self.act = torch.zeros(M * B.INTER, dtype=torch.bfloat16, device=dev)
            self.wgu = arm._wgu(inputs["wg"], inputs["wu"])
        self.sync = torch.zeros(B.SYNC_INTS, dtype=torch.int32, device=dev)
        skv_args: tuple = ()
        if self.attn == "splitkv":   # A1: plan once per prepare (every replay attends over the same visible lengths: the cache does not grow between replays)
            from ..kernels.common.splitkv import plan as skv_plan
            assert ld.knobs["MK_KU"] == 128, "attn=splitkv needs MK_KU 128"
            pl = skv_plan.plan([(s.kv_len, s.q_len) for s in w.seqs], n_cta=self.nb)
            assert list(pl.row_base) == list(d.t_base), "A1 row_base != mk t_base (fixture packs seq s at the sum of the previous kv_len + 1)"
            assert pl.total_q == c.ntok == w.total_q and pl.cache_rows == rows
            self.plan = pl
            t = pl.to_torch(dev)
            self.skv_items, self.skv_pair = t["items"], t["pair_info"]
            self.skv_part = torch.empty(max(pl.part_floats, 4), dtype=torch.float32, device=dev)
            self.skv_maps = None
            if pl.engine == 1:   # TMA engine: 2 * kNumMaps KV maps (16 default build, 14 R6; K, V x SW128 box {64, 1, rows}) in a separate 64 B aligned device buffer
                blob = B.encode_skv_maps(self.kc.data_ptr(), self.vc.data_ptr(), rows, g.layers, self.qmax)
                self.skv_maps = torch.frombuffer(bytearray(blob), dtype=torch.uint8).to(dev)
                assert self.skv_maps.data_ptr() % 64 == 0
            skv_args = (self.skv_items.data_ptr(), i32(pl.stride), self.skv_pair.data_ptr(), self.skv_part.data_ptr(),
                        self.skv_maps.data_ptr() if self.skv_maps is not None else 0, i32(pl.engine), i32(int(pl.s_max > 1)))
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
        ws = {"xb": self.xb.data_ptr()} if ts else None   # map 8 = xb (128 B more) only for MK_TS builds
        if self.tsb:   # maps 9, 10, 11 = xb2, wgu, act
            ws.update({"xb2": self.xb2.data_ptr(), "wgu": self.wgu.data_ptr(), "act": self.act.data_ptr()})
        blob = B.encode_maps(self._weights, {k: self._weights[k].data_ptr() for k in B.WEIGHT_MAPS}, workspace=ws, npad=self.npad)
        self.maps = torch.frombuffer(bytearray(blob), dtype=torch.uint8).to(dev)
        assert self.maps.data_ptr() % 128 == 0
        wp = [self._weights[k].data_ptr() for k in VEC_WEIGHTS]
        self.args = Args([self.tokens.data_ptr(), *wp, self.maps.data_ptr(), self.kc.data_ptr(), self.vc.data_ptr(),
                          self.logits.data_ptr(), self.next_token.data_ptr(), self.x.data_ptr(), self.qkv.data_ptr(),
                          self.qb.data_ptr(), self.ao.data_ptr(), self.gu.data_ptr(), self.rr.data_ptr(), self.sync.data_ptr(),
                          self.pkey.data_ptr(), self.pidx.data_ptr(), self.rope.data_ptr(), self.t_base.data_ptr(),
                          self.t_vis.data_ptr(), self.lm_tok.data_ptr(), self.lm_seq.data_ptr(),
                          i32(c.ntok), i32(len(c.lm_tok)), i32(self.rows), self.xb.data_ptr(), self.tk.data_ptr(),
                          *((self.xb2.data_ptr(), self.ss2.data_ptr(), self.act.data_ptr()) if self.tsb else (0, 0, 0)),
                          *skv_args])   # xb, tk, xb2, ss2, act, then (attn=splitkv) items, items_stride, pair_info, part, skv_maps, engine, skv_comb: last kernel arguments
        self.config: dict[str, Any] = {
            "kernel": B.KERNEL, "npad": self.npad, "graph": not direct, "proxy": False, "launches_per_step": 1,
            "grid": self.nb, "block": 512, "dyn_smem": ld.dyn_smem, "regs": ld.regs, "static_smem": ld.static_smem,
            "local_bytes": ld.local_bytes, "ptxas": ld.ptxas, "attn": self.attn, "attn_option": arm.attn, "skv_engine": getattr(getattr(self, "plan", None), "engine_name", None), "knobs": {**B.DEFAULT_KNOBS, **arm.knobs},
            "source_sha256": B.source_sha256(self.attn),
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
        for n in ("kc", "vc", "kview", "vview", "logits", "next_token", "x", "qkv", "qb", "ao", "gu", "rr", "xb", "tk", "xb2", "ss2", "act", "wgu", "sync", "pkey", "pidx",
                  "rope", "t_base", "t_vis", "lm_tok", "lm_seq", "maps", "skv_items", "skv_pair", "skv_part", "skv_maps", "plan"):
            if hasattr(self, n):
                delattr(self, n)
        self._weights = {}


class MkArm:
    name = "mk"

    def __init__(self, stamps: int = 0, null: int = 0, ku: int = 128, ring: int = 2, ts: int = 128, tsb: int = 1, lmts: int = 0, attn: str = "auto", **_: Any):
        if attn not in ATTN_OPTIONS:
            raise ValueError(f"unknown attn {attn!r} (one of {ATTN_OPTIONS})")
        self.attn = attn
        self.knobs = {"MK_STAMPS": int(stamps), "MK_NULL": int(null), "MK_KU": int(ku), "MK_RING": int(ring), "MK_TS": int(ts), "MK_TSB": int(tsb), "MK_LMTS": int(lmts)}
        self._loaded: dict[tuple, _Loaded] = {}
        self._wgu_cache: dict[tuple, Any] = {}

    def _wgu(self, wg, wu):
        """wgu = B.interleave_gu(wg, wu), built once per (wg, wu) pair of the arm (MK_TSB; 352 MB at 28 layers)."""
        key = (wg.data_ptr(), wu.data_ptr())
        if key not in self._wgu_cache:
            self._wgu_cache = {key: B.interleave_gu(wg, wu)}   # one entry: a new weight set replaces the old
        return self._wgu_cache[key]

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
        if self.attn == "splitkv":
            if any(s.q_len > 16 for s in w.seqs):
                return "attn=splitkv supports q_len <= 16 (R6 build, SKV_QMAX 16)"
            if self.knobs["MK_KU"] != 128:
                return "attn=splitkv needs MK_KU 128"
        return None

    def resolve_attn(self, w: Workload) -> str:
        """The attention build of a workload: the option, with "auto" = splitkv iff every sequence has q_len == 1 (A1 v1 limit), else the placeholder.
        splitkv also needs MK_KU 128 (the O phase stages one q-head per unit); auto falls back to the placeholder at MK_KU 64."""
        if self.attn == "auto":
            return B.auto_attn([(s.kv_len, s.q_len) for s in w.seqs]) if self.knobs["MK_KU"] == 128 else "placeholder"
        return self.attn

    def _kernel(self, npad: int, device, attn: str | None = None, qmax: int = 1) -> _Loaded:
        attn = attn or self.attn
        key = (npad, str(device), attn, qmax)
        if key not in self._loaded:
            self._loaded[key] = _Loaded(npad, tuple(sorted(self.knobs.items())), attn, device, qmax)
        return self._loaded[key]

    def prepare(self, w: Workload, inputs: dict, variant: str, device: str) -> MkStep:
        why = self.supports(w, variant)
        if why:
            raise ValueError(why)
        return MkStep(self, w, inputs, variant, device)
