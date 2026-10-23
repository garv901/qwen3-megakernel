"""vLLM 0.31 out-of-process reference: wall time of one engine step per Workload.

One process per (workload, variant). The model is Qwen3-0.6B with dummy
(random) weights, bf16, CUDA graphs on (default compilation config), prefix
caching off, engine core in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) so that
the PDL monkeypatch applies and the scheduler is reachable. The timed unit is a
direct ``LLMEngine.step()`` (scheduler + execute + sample + output processing,
i.e. it INCLUDES host time); async scheduling is off so every step is
synchronous and its shape is exactly what the scheduler output says.

How the step shape of each workload is built (all CPU-testable, see
``plan_workload`` / ``run_round``):

* pure decode (q_len 1 everywhere): n_seq requests with kv_len-token prompts are
  prefilled, then K <= 16 decode steps are timed. kv grows by K per round.
* prefill (all kv_len 0): n_seq requests with q_len-token prompts, one step
  schedules all of them (max_num_batched_tokens >= total_q); fresh requests
  every round.
* chunked (some q_len > 1 with kv_len > 0: R6 and R7): decoders (q_len 1) are
  prefilled first; the chunk request has a prompt of kv_len + q_len tokens. The
  scheduler token budget (``scheduler.max_num_scheduled_tokens``, a plain
  attribute) is set to (#decoders + kv_len) for one untimed step, so the chunk
  request advances exactly kv_len tokens, then restored; the timed step then has
  the q_len-token last chunk plus one token per decoder. With several chunk
  requests the per-request ``long_prefill_token_threshold`` is used instead.

The realised step composition is read back from the scheduler output
(``num_scheduled_tokens``) and recorded; a step whose composition differs from
the workload is reported (skip, or ``step_shape_exact`` False for mixed steps).

Records are plain JSON lines (schema ``regimebench/v1``); this module imports
only ``regimebench.workload`` from the repo. Environment variables are set
before vllm is imported, so vllm is imported lazily.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..workload import Workload

SCHEMA = "regimebench/v1"
DEFAULT_MODEL = ("/raid/hf/hub/models--Qwen--Qwen3-0.6B/snapshots/"
                 "c1899de289a04d12100db370d81485cdf75e47ca")
FALLBACK_MODEL = "Qwen/Qwen3-0.6B"
VARIANTS = ("pdl", "nopdl")
VOCAB_HI = 151643                 # random prompt ids stay below the first special token
BLOCK_SIZE = 16
KV_BYTES_PER_BLOCK = 2 * 28 * 8 * 128 * 2 * BLOCK_SIZE   # 1,835,008 for Qwen3-0.6B
MAX_K = 16
MAX_NBT_FLOOR = 16384
EXTENDED_CAPTURE_MAX = 2048


class Unsupported(Exception):
    """The workload's step cannot be constructed exactly in vLLM."""


# ---------------------------------------------------------------------------
# Pure helpers (no vllm / torch import)


@dataclass
class Plan:
    kind: str                       # "decode" | "prefill" | "chunk"
    decoders: list[int]             # kv_len of q_len==1 sequences
    chunkers: list[tuple[int, int]]  # (kv_len, q_len) of q_len>1 sequences
    prefillers: list[int] = field(default_factory=list)   # q_len of kv_len==0 sequences
    expected: Counter = field(default_factory=Counter)    # tokens scheduled -> count of requests

    @property
    def mixed(self) -> bool:
        return bool(self.decoders) and bool(self.chunkers)


def plan_workload(w: Workload) -> Plan:
    """Classify a workload and the step composition vLLM must produce."""
    seqs = w.seqs
    if all(s.q_len == 1 for s in seqs):
        if any(s.kv_len < 1 for s in seqs):
            raise Unsupported("decode sequence with kv_len 0 has no prompt to prefill")
        return Plan("decode", [s.kv_len for s in seqs], [],
                    expected=Counter({1: len(seqs)}))
    if all(s.kv_len == 0 for s in seqs):
        return Plan("prefill", [], [], [s.q_len for s in seqs],
                    expected=Counter(s.q_len for s in seqs))
    dec = [s.kv_len for s in seqs if s.q_len == 1]
    chunk = [(s.kv_len, s.q_len) for s in seqs if s.q_len > 1]
    if any(kv < 1 for kv in dec):
        raise Unsupported("decode sequence with kv_len 0 in a mixed step")
    if any(kv < 1 for kv, _ in chunk):
        raise Unsupported("mixed step with a kv_len-0 prefill sequence is not constructed")
    if len({kv for kv, _ in chunk}) > 1:
        raise Unsupported("several chunk sequences with different kv_len cannot be pre-staged "
                          "exactly (one global token budget)")
    exp = Counter({1: len(dec)}) if dec else Counter()
    exp.update(q for _, q in chunk)
    return Plan("chunk", dec, chunk, expected=exp)


def kv_cache_bytes_needed(w: Workload, k_steps: int = MAX_K) -> int:
    """KV bytes to reserve in vLLM: every sequence's kv + q + decode growth + slack, in blocks."""
    blocks = 0
    for s in w.seqs:
        blocks += math.ceil((s.kv_len + s.q_len + k_steps + 64) / BLOCK_SIZE) + 1
    return int(blocks * KV_BYTES_PER_BLOCK * 1.05) + (64 << 20)


def engine_sizing(w: Workload, k_steps: int = MAX_K) -> dict:
    plan = plan_workload(w)
    stage1 = len(plan.decoders) + sum(kv for kv, _ in plan.chunkers)
    return {
        "max_model_len": w.max_kv + w.max_q + 64 + k_steps if plan.kind != "prefill"
        else max(w.max_q + 64, 512),
        "max_num_seqs": w.n_seq,
        "max_num_batched_tokens": max(MAX_NBT_FLOOR, w.total_q, stage1),
        "kv_cache_memory_bytes": kv_cache_bytes_needed(w, k_steps),
    }


def cudagraph_capture_max(w: Workload) -> int | None:
    """Extend the default capture range (<= 512 tokens) to cover a small mixed/prefill step."""
    return EXTENDED_CAPTURE_MAX if 512 < w.total_q <= EXTENDED_CAPTURE_MAX else None


def summarize(samples: list[float], seed: int = 0) -> dict:
    """Same keys as regimebench.harness.timing.summarize (median, CI of the median, ...)."""
    x = sorted(float(v) for v in samples)
    if not x:
        return {"n": 0}

    def pct(v, p):
        if len(v) == 1:
            return v[0]
        pos = (len(v) - 1) * p / 100.0
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (pos - lo)

    med = statistics.median(x)
    rng = random.Random(seed)
    boots = sorted(statistics.median(rng.choices(x, k=len(x))) for _ in range(2000))
    return {"n": len(x), "median": med, "mean": statistics.fmean(x),
            "std": statistics.stdev(x) if len(x) > 1 else 0.0,
            "p05": pct(x, 5), "p95": pct(x, 95), "min": x[0], "max": x[-1],
            "mad": statistics.median(abs(v - med) for v in x),
            "ci95_median": [pct(boots, 2.5), pct(boots, 97.5)]}


def step_shape(tokens_by_req: dict) -> dict:
    """{"total": ..., "n_req": ..., "hist": {"tokens": count}} of one scheduler output."""
    c = Counter(tokens_by_req.values())
    return {"total": int(sum(tokens_by_req.values())), "n_req": len(tokens_by_req),
            "hist": {str(k): v for k, v in sorted(c.items())}}


def shape_matches(plan: Plan, tokens_by_req: dict) -> bool:
    return Counter(tokens_by_req.values()) == plan.expected


# ---------------------------------------------------------------------------
# Round driver (talks to an "engine handle"; real one below, fake one in tests)
#
# Handle interface:
#   add(prompt_ids, max_tokens) -> rid          computed(rid) -> int
#   step() -> seconds (synchronised wall time)  last_scheduled() -> {rid: n_tokens}
#   set_budget(n | None)  set_threshold(n)      abort(rids)   has_unfinished() -> bool
#   step_untimed() -> None (same as step, result ignored)


@dataclass
class RoundResult:
    times_ms: list[float] = field(default_factory=list)
    shapes: list[dict] = field(default_factory=list)
    kv_before: list[float] = field(default_factory=list)   # mean cached tokens of scheduled reqs
    gpu_ms: list[float] = field(default_factory=list)      # event-span GPU ms per timed step
    exact: list[bool] = field(default_factory=list)


def _prompt(rng: random.Random, n: int) -> list[int]:
    return [rng.randrange(VOCAB_HI) for _ in range(n)]


def _timed_step(h, plan: Plan, rids: list, res: RoundResult) -> None:
    before = {r: h.computed(r) for r in rids}
    dt = h.step()
    sched = h.last_scheduled()
    res.times_ms.append(dt * 1e3)
    g = h.last_gpu_ms() if hasattr(h, "last_gpu_ms") else None
    if g is not None:
        res.gpu_ms.append(g)
    res.shapes.append(step_shape(sched))
    kvs = [before[r] for r in sched if r in before]
    res.kv_before.append(sum(kvs) / len(kvs) if kvs else float("nan"))
    res.exact.append(shape_matches(plan, sched))


def _prefill_until(h, rids: list, lens: list, max_steps: int = 100000) -> None:
    for _ in range(max_steps):
        if all(h.computed(r) >= n for r, n in zip(rids, lens)):
            return
        h.step_untimed()
    raise RuntimeError("prefill did not complete")


def run_round(h, plan: Plan, rng: random.Random, k_steps: int) -> RoundResult:
    res = RoundResult()
    rids: list = []
    try:
        dec_max_tokens = k_steps + 64
        if plan.kind == "decode":
            rids = [h.add(_prompt(rng, kv), dec_max_tokens) for kv in plan.decoders]
            _prefill_until(h, rids, plan.decoders)
            for _ in range(k_steps):
                _timed_step(h, plan, rids, res)
        elif plan.kind == "prefill":
            rids = [h.add(_prompt(rng, q), 1) for q in plan.prefillers]
            _timed_step(h, plan, rids, res)
        else:
            dec = [h.add(_prompt(rng, kv), dec_max_tokens) for kv in plan.decoders]
            rids = list(dec)
            _prefill_until(h, dec, plan.decoders)
            chunk = [h.add(_prompt(rng, kv + q), 8) for kv, q in plan.chunkers]
            rids += chunk
            pre = sum(kv for kv, _ in plan.chunkers)
            h.set_budget(len(dec) + pre)
            if len(chunk) > 1:
                h.set_threshold(plan.chunkers[0][0])
            try:
                h.step_untimed()                      # chunk requests advance exactly kv_len
            finally:
                h.set_budget(None)
                h.set_threshold(0)
            _timed_step(h, plan, rids, res)           # last chunk of q_len + one token per decoder
    finally:
        h.abort(rids)
        for _ in range(8):
            if not h.has_unfinished():
                break
            h.step_untimed()
    return res


def measure(h, plan: Plan, seed: int, rounds: int, warmup_rounds: int, k_steps: int) -> RoundResult:
    rng = random.Random(seed)
    for _ in range(warmup_rounds):
        run_round(h, plan, rng, k_steps)
    total = RoundResult()
    for _ in range(rounds):
        r = run_round(h, plan, rng, k_steps)
        total.times_ms += r.times_ms
        total.shapes += r.shapes
        total.kv_before += r.kv_before
        total.gpu_ms += r.gpu_ms
        total.exact += r.exact
    return total


# ---------------------------------------------------------------------------
# PDL control (vLLM 0.31 / FlashInfer 0.7: see _upstream/ENV.md section 4)


class PdlPatch:
    """Counts (and optionally disables) PDL queries made by FlashInfer and vLLM.

    FlashInfer decides PDL per call with ``device_support_pdl(dev)`` when the caller
    passes ``enable_pdl=None`` (vLLM never passes it for attention). Each FlashInfer
    module binds that function by ``from .utils import device_support_pdl`` so every
    module attribute is replaced. vLLM asks ``current_platform.is_arch_support_pdl()``
    (FlashInfer bf16 mm, Triton launch_pdl). Must be installed before CUDA graph
    capture (graphs bake in the launch attribute).
    """

    def __init__(self, disable: bool):
        self.disable = disable
        self.queries = 0
        self.flashinfer_modules: list[str] = []
        self.vllm_sites: list[str] = []
        self._orig = None
        self._wrapper = None

    def install(self) -> None:
        import flashinfer
        import flashinfer.utils as fu
        self._orig = fu.device_support_pdl
        orig, me = self._orig, self

        def device_support_pdl(device):
            me.queries += 1
            return False if me.disable else orig(device)

        self._wrapper = device_support_pdl
        fu.device_support_pdl = device_support_pdl
        self.refresh()
        from vllm.platforms import current_platform
        cls = type(current_platform)
        real = cls.is_arch_support_pdl

        def is_arch_support_pdl(klass):
            me.queries += 1
            return False if me.disable else real()

        cls.is_arch_support_pdl = classmethod(is_arch_support_pdl)
        self.vllm_sites = [f"{cls.__name__}.is_arch_support_pdl"]

    def refresh(self) -> None:
        found = set(self.flashinfer_modules)
        for name, mod in list(sys.modules.items()):
            if name.split(".")[0] != "flashinfer" or mod is None:
                continue
            if getattr(mod, "device_support_pdl", None) not in (None, self._wrapper):
                mod.device_support_pdl = self._wrapper
                found.add(name)
        self.flashinfer_modules = sorted(found)

    def describe(self) -> dict:
        return {"disabled": self.disable,
                "flashinfer_modules_patched": self.flashinfer_modules,
                "vllm_sites_patched": self.vllm_sites,
                "pdl_queries_so_far": self.queries,
                "not_covered": ["vLLM/PyTorch C++ and Triton kernels that launch with PDL "
                                "independently of these probes"]}


# ---------------------------------------------------------------------------
# Real engine handle


class EngineHandle:
    """Adapter over a vLLM ``LLMEngine`` running its EngineCore in-process."""

    def __init__(self, llm_engine):
        import torch
        self._torch = torch
        self.eng = llm_engine
        core = llm_engine.engine_core.engine_core     # InprocClient.engine_core -> EngineCore
        self.sched = core.scheduler
        self._budget0 = self.sched.max_num_scheduled_tokens
        self._thr0 = self.sched.scheduler_config.long_prefill_token_threshold
        self._last: dict = {}
        orig = self.sched.schedule

        def schedule(*a, **k):
            out = orig(*a, **k)
            self._last = dict(out.num_scheduled_tokens)
            return out

        self.sched.schedule = schedule
        self._n = 0
        self._events: list = []
        self.wrapped = self._wrap_runner(core)

    def _wrap_runner(self, core) -> list[str]:
        """Record CUDA events around the model runner's execute_model and sample_tokens.

        Wraps the instance methods of ``model_executor.driver_worker.worker.model_runner``
        (v1 ``vllm.v1.worker.gpu_model_runner.GPUModelRunner`` or the v2 runner, whichever the
        worker built). Each call enqueues a start and an end event on the current stream; the
        span includes any GPU idle time while the host is still launching inside the call.
        """
        torch = self._torch
        runner = core.model_executor.driver_worker.worker.model_runner
        names = []
        for name in ("execute_model", "sample_tokens"):
            orig = getattr(runner, name, None)
            if orig is None:
                continue

            def make(orig):
                def wrapped(*a, **k):
                    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    st.record()
                    try:
                        return orig(*a, **k)
                    finally:
                        en.record()
                        self._events.append((st, en))
                return wrapped

            setattr(runner, name, make(orig))
            names.append(f"{type(runner).__module__}.{type(runner).__name__}.{name}")
        return names

    def last_gpu_ms(self):
        """Sum of event spans of the calls in the last step() (call after step's sync)."""
        return sum(st.elapsed_time(en) for st, en in self._events)

    def add(self, prompt_ids: list[int], max_tokens: int) -> str:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        self._n += 1
        sp = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True,
                            detokenize=False)
        return self.eng.add_request(f"rb{self._n}", TokensPrompt(prompt_token_ids=prompt_ids), sp)

    def computed(self, rid: str) -> int:
        r = self.sched.requests.get(rid)
        return r.num_computed_tokens if r is not None else 0

    def step(self) -> float:
        torch = self._torch
        torch.cuda.synchronize()
        self._events.clear()
        t0 = time.perf_counter()
        self.eng.step()
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    def step_untimed(self) -> None:
        self.eng.step()
        self._events.clear()

    def last_scheduled(self) -> dict:
        return self._last

    def set_budget(self, n) -> None:
        self.sched.max_num_scheduled_tokens = self._budget0 if n is None else min(n, self._budget0)

    def set_threshold(self, n: int) -> None:
        self.sched.scheduler_config.long_prefill_token_threshold = n or self._thr0

    def abort(self, rids: list) -> None:
        if rids:
            self.eng.abort_request(list(rids))

    def has_unfinished(self) -> bool:
        return self.eng.has_unfinished_requests()


# ---------------------------------------------------------------------------
# Records


class JsonlWriter:
    def __init__(self, path: Path, run_id: str):
        self.path, self.run_id = Path(path), run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a")

    def write(self, record_type: str, **fields) -> dict:
        rec = {"schema": SCHEMA, "run_id": self.run_id, "record_type": record_type,
               "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
        rec.update(fields)
        self._f.write(json.dumps(rec, default=str) + "\n")
        self._f.flush()
        return rec

    def close(self) -> None:
        self._f.close()


def _version(dist: str) -> str | None:
    from importlib import metadata
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def build_engine_kwargs(w: Workload, model: str, k_steps: int, seed: int,
                        cudagraph_max: int | None) -> dict:
    sz = engine_sizing(w, k_steps)
    comp: dict = {}
    cg = cudagraph_max if cudagraph_max is not None else cudagraph_capture_max(w)
    if cg:
        comp["max_cudagraph_capture_size"] = cg
    kw = dict(
        model=model, dtype="bfloat16", load_format="dummy", seed=seed,
        block_size=BLOCK_SIZE, enable_prefix_caching=False, async_scheduling=False,
        max_model_len=sz["max_model_len"], max_num_seqs=sz["max_num_seqs"],
        max_num_batched_tokens=sz["max_num_batched_tokens"],
        kv_cache_memory_bytes=sz["kv_cache_memory_bytes"],
        attention_config={"backend": "FLASHINFER"},
        disable_log_stats=True,
    )
    if comp:
        kw["compilation_config"] = comp
    return kw


def run(args: argparse.Namespace, writer: JsonlWriter) -> int:
    w = Workload.from_dict(json.loads(Path(args.workload_json).read_text()))
    wd = w.to_dict()
    base = dict(arm="vllm", variant=args.variant, workload=wd)
    if args.k_steps > MAX_K:
        raise ValueError(f"--k-steps must be <= {MAX_K}")
    try:
        plan = plan_workload(w)
    except Unsupported as e:
        writer.write("skip", **base, reason=str(e))
        return 0

    import torch
    import vllm
    from vllm import LLM

    pdl = PdlPatch(disable=args.variant == "nopdl")
    try:
        pdl.install()
    except Exception as e:
        if args.variant == "nopdl":
            writer.write("skip", **base, reason=f"cannot disable PDL: {e!r}")
            return 0
        pdl = None
    if args.variant == "nopdl" and not pdl.flashinfer_modules:
        writer.write("skip", **base, reason="no FlashInfer module exposes device_support_pdl")
        return 0

    kwargs = build_engine_kwargs(w, args.model, args.k_steps, args.seed, args.cudagraph_max)
    llm = None
    try:
        llm = LLM(**kwargs)
        if pdl:
            pdl.refresh()
        vc = llm.llm_engine.vllm_config
        cc = vc.compilation_config
        engine_cfg = {
            "attention_backend": str(vc.attention_config.backend),
            "cudagraph_mode": str(cc.cudagraph_mode),
            "cudagraph_capture_sizes": list(cc.cudagraph_capture_sizes or []),
            "max_cudagraph_capture_size": cc.max_cudagraph_capture_size,
            "compilation_mode": str(cc.mode),
            "async_scheduling": vc.scheduler_config.async_scheduling,
            "enable_chunked_prefill": vc.scheduler_config.enable_chunked_prefill,
            "long_prefill_token_threshold": vc.scheduler_config.long_prefill_token_threshold,
            "max_num_batched_tokens": vc.scheduler_config.max_num_batched_tokens,
        }
        h = EngineHandle(llm.llm_engine)
        res = measure(h, plan, args.seed, args.rounds, args.warmup_rounds, args.k_steps)
        if pdl:
            pdl.refresh()

        shapes = res.shapes
        exact = all(res.exact)
        reason = None
        if not exact:
            bad = next(s for s, ok in zip(shapes, res.exact) if not ok)
            reason = (f"timed step composition differs from the workload: got {bad}, "
                      f"expected {dict(plan.expected)}")
        if not exact and not plan.mixed:
            writer.write("skip", **base, reason=reason,
                         config={"engine_kwargs": kwargs, "engine_config": engine_cfg})
            return 0

        target_kv = (sum(s.kv_len for s in w.seqs) / w.n_seq)
        mean_kv = statistics.fmean(res.kv_before)
        cap = engine_cfg["max_cudagraph_capture_size"] or 0
        total_tok = shapes[0]["total"]
        cfg = {
            "engine_kwargs": kwargs, "engine_config": engine_cfg, "vllm_version": vllm.__version__,
            "flashinfer_version": _version("flashinfer-python"), "torch_version": torch.__version__,
            "multiprocessing": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
            "load_format": "dummy", "step_kind": plan.kind,
            "step_construction": _construction_text(plan, args.k_steps),
            "step_includes_host_time": True,
            "step_shapes": _dedupe(shapes), "step_shape_exact": exact, "step_shape_note": reason,
            "total_scheduled_tokens": total_tok,
            "cudagraph_expected_by_size": (total_tok <= cap and "NONE" not in
                                           engine_cfg["cudagraph_mode"]),
            "mean_kv_cached_before_step": mean_kv, "workload_mean_kv": target_kv,
            "k_steps": args.k_steps if plan.kind == "decode" else 1,
            "rounds": args.rounds, "warmup_rounds": args.warmup_rounds, "seed": args.seed,
            "pdl_patch": pdl.describe() if pdl else None, "proxy": False,
        }
        cfg["gpu_time_wrapped_methods"] = h.wrapped
        cfg["gpu_time_note"] = ("gpu = sum over execute_model/sample_tokens calls of the CUDA event "
                                "span (start/end recorded on the current stream around each call); "
                                "includes GPU idle gaps inside those calls, excludes scheduler and "
                                "output processing")
        st = summarize(res.times_ms)
        gpu = summarize(res.gpu_ms) if res.gpu_ms else None
        writer.write("timing", **base, config=cfg, steady=st, isolated=st,
                     gpu=gpu, steady_gpu=gpu, samples_gpu_ms=res.gpu_ms,
                     n_launch_per_sample=1, samples_ms=res.times_ms, unit="ms_per_step",
                     tokens_per_s=w.total_q / (st["median"] * 1e-3))
        return 0
    finally:
        _shutdown(llm)


def _dedupe(shapes: list[dict]) -> list[dict]:
    seen, out = set(), []
    for s in shapes:
        key = json.dumps(s, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out[:8]


def _construction_text(plan: Plan, k_steps: int) -> str:
    if plan.kind == "decode":
        return (f"prefill {len(plan.decoders)} prompts of kv_len tokens, then time {k_steps} decode "
                "LLMEngine.step() calls per round (kv grows by the steps already taken)")
    if plan.kind == "prefill":
        return "one LLMEngine.step() that prefills all prompts (kv 0); fresh requests per round"
    return ("decoders prefilled; chunk request prompt = kv_len + q_len; scheduler budget set to "
            "(#decoders + kv_len) for one untimed step so the chunk advances exactly kv_len; timed "
            "step = last q_len chunk + one token per decoder")


def _shutdown(llm) -> None:
    if llm is None:
        return
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    try:
        import gc
        del llm
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--workload-json", type=Path, required=True)
    p.add_argument("--variant", choices=VARIANTS, default="pdl")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--seed", type=int, default=7301)
    p.add_argument("--rounds", type=int, default=10, help="timed rounds (fresh requests each)")
    p.add_argument("--warmup-rounds", type=int, default=1)
    p.add_argument("--k-steps", type=int, default=8, help=f"decode steps timed per round (<= {MAX_K})")
    p.add_argument("--model", default=DEFAULT_MODEL if Path(DEFAULT_MODEL).exists() else FALLBACK_MODEL)
    p.add_argument("--cudagraph-max", type=int, default=None,
                   help="override max_cudagraph_capture_size (default: vLLM default, widened to "
                        f"{EXTENDED_CAPTURE_MAX} when 512 < total_q <= {EXTENDED_CAPTURE_MAX})")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("HF_HOME", "/raid/hf")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    writer = JsonlWriter(args.out, args.run_id)
    try:
        return run(args, writer)
    except Exception:
        tail = traceback.format_exc()
        try:
            wd = Workload.from_dict(json.loads(Path(args.workload_json).read_text())).to_dict()
        except Exception:
            wd = None
        writer.write("error", arm="vllm", variant=args.variant, workload=wd, stage="run",
                     traceback=tail)
        print(tail, file=sys.stderr)
        return 1
    finally:
        writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
