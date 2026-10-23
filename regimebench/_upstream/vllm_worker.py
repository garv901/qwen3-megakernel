"""vLLM end-to-end decode worker: ms per decode step from a wall-time-vs-N slope.

One process per (batch, context, variant). Wall time of ``llm.generate`` with
``max_tokens = 1 + N`` is fit against N, so prefill, scheduling, and the first
token cancel in the slope: it is the end-to-end cost of one extra decode step
for the whole batch. Variants:

* ``mp1``: default vLLM (engine core in a separate process).
* ``mp0``: in-process engine core (VLLM_ENABLE_V1_MULTIPROCESSING=0).
* ``mp0-nopdl``: mp0 with Programmatic Dependent Launch disabled for every
  FlashInfer call and vLLM's own PDL probe (see ``disable_pdl``).

Run ``python -m megabench.sota.vllm_worker --help`` for options. Environment
variables are set before vllm is imported, so vllm is imported inside main.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
import time
import traceback
from pathlib import Path

DEFAULT_MODEL = ("/raid/hf/hub/models--Qwen--Qwen3-0.6B/snapshots/"
                 "c1899de289a04d12100db370d81485cdf75e47ca")
VARIANTS = ("mp0", "mp1", "mp0-nopdl")
VOCAB_HI = 151643


# ---------------------------------------------------------------------------
# Pure helpers


def lstsq(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Slope, intercept, r^2 of an ordinary least-squares line."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else float("nan")
    icpt = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (icpt + slope * x)) ** 2 for x, y in zip(xs, ys))
    return slope, icpt, (1 - ss_res / ss_tot) if ss_tot else float("nan")


def median(v: list[float]) -> float:
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--context", type=int, required=True)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--n-list", default="0,16,32,64",
                   help="comma-separated extra decode steps N (max_tokens = 1 + N)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--gpu-mem", type=float, default=0.3)
    args = p.parse_args(argv)
    args.n_list = [int(x) for x in args.n_list.split(",") if x != ""]
    if len(set(args.n_list)) < 2:
        p.error("--n-list needs at least two distinct values")
    return args


# ---------------------------------------------------------------------------
# PDL control


class PdlPatch:
    """Counts (and optionally disables) PDL queries made by FlashInfer and vLLM.

    FlashInfer decides PDL per call with ``enable_pdl = device_support_pdl(dev)``
    when the caller passes ``enable_pdl=None`` (vLLM 0.31 never passes it for
    decode/prefill/norm/sampling). Each FlashInfer module binds that function
    by ``from .utils import device_support_pdl``, so every module attribute is
    replaced, not only ``flashinfer.utils``. vLLM itself asks
    ``current_platform.is_arch_support_pdl()`` (FlashInfer bf16 mm, DeepGEMM).
    """

    def __init__(self, disable: bool):
        self.disable = disable
        self.queries = 0
        self.flashinfer_modules: list[str] = []
        self.vllm_sites: list[str] = []
        self._orig = None

    def install(self) -> None:
        import flashinfer
        import flashinfer.utils as fu
        self._orig = fu.device_support_pdl
        orig, me = self._orig, self

        def device_support_pdl(device):
            me.queries += 1
            return False if me.disable else orig(device)

        self._wrapper = device_support_pdl
        # Patch utils first: later `from .utils import ...` then binds the wrapper.
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
        """Re-scan modules (some FlashInfer modules import lazily)."""
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
                                "independently of these probes (LoRA only: unused here)"]}


# ---------------------------------------------------------------------------
# Main


def run(args: argparse.Namespace, writer) -> int:
    import torch

    from .shapes import Shape
    from .records import env_info
    from .timing import summarize

    shape = Shape(args.batch, args.context)
    base = dict(arm="vllm", variant=args.variant, shape=shape.to_dict())
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    pdl = None
    if args.variant in ("mp0", "mp0-nopdl"):
        pdl = PdlPatch(disable=args.variant == "mp0-nopdl")
        try:
            pdl.install()
        except Exception as e:
            if args.variant == "mp0-nopdl":
                writer.write("skip", **base, reason=f"cannot disable PDL: {e!r}")
                return 0
            pdl = None
    if args.variant == "mp0-nopdl" and not pdl.flashinfer_modules:
        writer.write("skip", **base, reason="no FlashInfer module exposes device_support_pdl")
        return 0

    max_n = max(args.n_list)
    sizes = sorted({s for s in (1, 2, 4, 8, 16, 32) if s <= args.batch} | {args.batch})
    kwargs = dict(
        model=args.model, dtype="bfloat16",
        max_model_len=max(512, args.context + max_n + 16),
        max_num_seqs=args.batch, enable_prefix_caching=False, block_size=16,
        gpu_memory_utilization=args.gpu_mem, seed=0, disable_log_stats=False,
        attention_backend="FLASHINFER", async_scheduling=True,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                            "cudagraph_capture_sizes": sizes},
    )
    llm = None
    try:
        llm = LLM(**kwargs)
        if pdl:
            pdl.refresh()
        engine_cfg = {}
        try:
            vc = llm.llm_engine.vllm_config
            engine_cfg = {
                "attention_backend_config": str(vc.attention_config.backend),
                "cudagraph_mode": str(vc.compilation_config.cudagraph_mode),
                "cudagraph_capture_sizes": list(vc.compilation_config.cudagraph_capture_sizes or []),
                "async_scheduling": vc.scheduler_config.async_scheduling,
            }
        except Exception as e:  # attribute layout differs: record, do not fail
            engine_cfg = {"readback_error": repr(e)}
        env = env_info()
        writer.write("env", **base, env=env, vllm_version=vllm.__version__,
                     flashinfer_version=_version("flashinfer-python"),
                     torch_version=torch.__version__, engine_kwargs=kwargs,
                     engine_config=engine_cfg,
                     multiprocessing=os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"],
                     pdl_patch=pdl.describe() if pdl else None)

        rng = random.Random(0)
        prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(VOCAB_HI)
                                                  for _ in range(args.context)])
                   for _ in range(args.batch)]

        def sp(n: int) -> SamplingParams:
            return SamplingParams(temperature=0, max_tokens=1 + n, ignore_eos=True,
                                  detokenize=False)

        for _ in range(3):
            llm.generate(prompts, sp(max_n), use_tqdm=False)
        torch.cuda.synchronize()

        by_n: dict[int, list[float]] = {n: [] for n in args.n_list}
        trial_slopes: list[float] = []
        itl: list[float] = []
        for t in range(args.trials):
            order = list(args.n_list)
            random.Random(t).shuffle(order)
            pts = []
            for n in order:
                t0 = time.perf_counter()
                outs = llm.generate(prompts, sp(n), use_tqdm=False)
                dt = time.perf_counter() - t0
                by_n[n].append(dt)
                pts.append((n, dt))
                if n == max_n:
                    itl.extend(_engine_itl_ms(outs))
            trial_slopes.append(lstsq([p[0] for p in pts], [p[1] for p in pts])[0] * 1e3)
        xs = [n for n in args.n_list for _ in by_n[n]]
        ys = [v for n in args.n_list for v in by_n[n]]
        slope, icpt, r2 = lstsq(xs, ys)
        stats = summarize(trial_slopes)
        per_n = {str(n): median(v) for n, v in by_n.items()}
        mean_context = args.context + (sum(args.n_list) / len(args.n_list)) / 2
        itl_stats = summarize(itl) if itl else None
        if pdl:
            pdl.refresh()
        cfg = {"engine_kwargs": kwargs, "engine_config": engine_cfg,
               "n_list": args.n_list, "trials": args.trials,
               "multiprocessing": os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"],
               "pdl_patch": pdl.describe() if pdl else None,
               "intercept_s": icpt}
        writer.write("timing", **base, mode="vllm_slope", unit="ms_per_decode_step",
                     samples_ms=trial_slopes, stats=stats,
                     slope_ms=stats["median"], slope_all_ms=slope * 1e3,
                     slope_ms_ci95=stats["ci95_median"], r2=r2,
                     per_n_median_s=per_n, mean_context=mean_context,
                     itl_ms=(itl_stats["median"] if itl_stats else "unavailable"),
                     config=cfg)
        if itl:
            writer.write("timing", **base, mode="vllm_engine_itl", unit="ms",
                         samples_ms=itl, stats=itl_stats, mean_context=args.context + max_n / 2,
                         config=cfg)
        else:
            writer.write("timing", **base, mode="vllm_engine_itl", unavailable=True,
                         reason="RequestOutput.metrics is None", config=cfg)
        return 0
    finally:
        _shutdown(llm)


def _engine_itl_ms(outs) -> list[float]:
    """Per-request (last_token_ts - first_token_ts) / (generated - 1), in ms."""
    vals = []
    for o in outs:
        m = getattr(o, "metrics", None)
        if m is None or m.num_generation_tokens < 2:
            continue
        vals.append((m.last_token_ts - m.first_token_ts) / (m.num_generation_tokens - 1) * 1e3)
    return vals


def _version(dist: str) -> str | None:
    from importlib import metadata
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _shutdown(llm) -> None:
    if llm is None:
        return
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    try:
        del llm
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1" if args.variant == "mp1" else "0"
    from .records import RecordWriter
    writer = RecordWriter(args.out, args.run_id)
    try:
        return run(args, writer)
    except Exception:
        tail = "".join(traceback.format_exc().splitlines(keepends=True)[-25:])
        from .shapes import Shape
        writer.write("error", arm="vllm", variant=args.variant,
                     shape=Shape(args.batch, args.context).to_dict(), traceback_tail=tail)
        print(tail, file=sys.stderr)
        return 1
    finally:
        writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
