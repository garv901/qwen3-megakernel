"""Run the megakernel-vs-SOTA decode benchmark: ``python -m megabench.sota run``."""

from __future__ import annotations

import argparse
import gc
import math
import subprocess
import sys
import traceback
from pathlib import Path

DEFAULT_ARMS = "megakernel,sota-graph,torch-compile,vllm"
HEADLINE_VARIANT = "direct"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="megabench.sota run", description=__doc__)
    p.add_argument("--arms", default=DEFAULT_ARMS)
    p.add_argument("--variants", default="",
                   help="filter, e.g. 'sota-graph:graph-pdl,graph-nopdl;megakernel:direct'")
    p.add_argument("--batch", default="1", help="comma list")
    p.add_argument("--context", default="128", help="comma list")
    p.add_argument("--seeds", default="7301,8917,42811")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--steady-steps", type=int, default=50)
    p.add_argument("--isolated-samples", type=int, default=100)
    p.add_argument("--single-seed-arms", default="torch-compile",
                   help="arms checked on the timing seed only (max-autotune recompiles "
                        "for ~4 min per variant and seed)")
    p.add_argument("--flush-l2", action="store_true")
    p.add_argument("--no-profile", action="store_true")
    p.add_argument("--tag", default="sota")
    p.add_argument("--megakernel", action="append", default=[], metavar="NAME=PATH[:CASE_ID]")
    p.add_argument("--opus", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--vllm-variants", default="mp0,mp1,mp0-nopdl")
    p.add_argument("--vllm-trials", type=int, default=10)
    p.add_argument("--vllm-n", default="0,16,32,64")
    p.add_argument("--vllm-timeout", type=int, default=1500)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out-dir", type=Path, default=None)
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.rounds, args.isolated_samples = 3, 10
        args.vllm_trials, args.vllm_n = 2, "0,16"
    args.seed_list = [int(s) for s in args.seeds.split(",") if s]
    if args.smoke:
        args.seed_list = args.seed_list[:1]
    args.arm_list = [a for a in args.arms.split(",") if a]
    args.single_seed = {a for a in args.single_seed_arms.split(",") if a}
    args.variant_filter = parse_variant_filter(args.variants)
    args.megakernels = [parse_megakernel(s) for s in args.megakernel]
    args.opus_arms = [parse_opus(s) for s in args.opus]
    return args


def parse_variant_filter(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for part in filter(None, (x.strip() for x in text.split(";"))):
        arm, _, vs = part.partition(":")
        out[arm.strip()] = [v.strip() for v in vs.split(",") if v.strip()]
    return out


def parse_megakernel(spec: str) -> tuple[str, str, str | None]:
    name, sep, rest = spec.partition("=")
    if not sep or not name or not rest:
        raise argparse.ArgumentTypeError(f"--megakernel expects NAME=PATH[:CASE_ID], got {spec!r}")
    path, _, case = rest.partition(":")
    return name, path, case or None


def parse_opus(spec: str) -> tuple[str, str]:
    name, sep, path = spec.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"--opus expects NAME=PATH, got {spec!r}")
    return name, path


def parse_shapes_arg(args):
    from .shapes import parse_shapes
    return parse_shapes([int(x) for x in args.batch.split(",")],
                        [int(x) for x in args.context.split(",")])


def build_arms(args) -> tuple[list[tuple[str, object]], list[tuple[str, str]]]:
    """In-process arms as ``[(record_name, Arm)]`` plus ``[(name, load error)]``."""
    import importlib
    from .backend import ARMS
    arms, failed = [], []
    wanted = [(a, a, {}) for a in args.arm_list if a in ARMS]
    for name, path, case in args.megakernels:
        opts = {"name": name, "submission": path, "case_id": case}
        wanted.append((name, "megakernel", opts))
    for name, path in args.opus_arms:
        wanted.append((name, "opus", {"name": name, "submission": path}))
    for key, base, opts in wanted:
        try:
            # backend.load_arm(name, **options) cannot forward a "name" option, so build here.
            module_name, _, class_name = ARMS[base].partition(":")
            cls = getattr(importlib.import_module(module_name), class_name)
            arms.append((key, cls(**opts)))
        except Exception as exc:
            failed.append((key, f"{type(exc).__name__}: {exc}"))
    return arms, failed


def selected_variants(key: str, arm, args) -> list[str]:
    allv = list(arm.variants())
    flt = args.variant_filter.get(key)
    if flt is None:
        flt = args.variant_filter.get(getattr(arm, "name", key))
    return allv if flt is None else [v for v in allv if v in flt]


def vllm_commands(args, shape, run_dir: Path, run_id: str) -> list[tuple[str, list[str], Path]]:
    cmds = []
    for v in [x for x in args.vllm_variants.split(",") if x]:
        out = run_dir / f"vllm-{v}-{shape.id}.jsonl"
        cmds.append((v, [sys.executable, "-m", "megabench.sota.vllm_worker",
                         "--batch", str(shape.batch), "--context", str(shape.context),
                         "--variant", v, "--trials", str(args.vllm_trials),
                         "--n-list", args.vllm_n, "--out", str(out),
                         "--run-id", run_id], out))
    return cmds


def plan_matrix(args) -> list[dict]:
    """Planned (arm, variant, shape, status) rows without touching CUDA."""
    shapes = parse_shapes_arg(args)
    arms, failed = build_arms(args)
    rows = []
    for shape in shapes:
        for key, arm in arms:
            for v in selected_variants(key, arm, args):
                try:
                    reason = arm.supports(shape, v)
                except Exception as exc:
                    reason = f"supports() failed: {exc!r}"
                rows.append({"arm": key, "variant": v, "shape": shape.id,
                             "status": "skip" if reason else "run", "reason": reason})
        for key, err in failed:
            rows.append({"arm": key, "variant": "*", "shape": shape.id,
                         "status": "load-error", "reason": err})
        if "vllm" in args.arm_list:
            for v in [x for x in args.vllm_variants.split(",") if x]:
                rows.append({"arm": "vllm", "variant": v, "shape": shape.id,
                             "status": "run", "reason": None})
    return rows


def dry_run(args) -> int:
    rows = plan_matrix(args)
    modes = ["steady", "isolated"] + (["isolated+flushed"] if args.flush_l2 else [])
    print(f"seeds={args.seed_list} timing_seed={args.seed_list[0]} rounds={args.rounds} "
          f"steady_steps={args.steady_steps} isolated_samples={args.isolated_samples}")
    print(f"modes: {', '.join(modes)}; profile={'off' if args.no_profile else 'on'}")
    for r in rows:
        extra = f"  ({r['reason']})" if r["reason"] else ""
        print(f"{r['shape']:>10}  {r['arm']:<14} {r['variant']:<22} {r['status']}{extra}")
    return 0


def _tb() -> str:
    return traceback.format_exc()[-3000:]


def _own_ref(step) -> str:
    return "fp32" if getattr(step, "precision", "bf16") == "fp32" else "bf16"


def run(args) -> int:
    if args.dry_run:
        return dry_run(args)
    import torch
    from . import accuracy, records, report, timing
    from .shapes import bytes_floor, expected_outputs, make_shape_inputs, reference_outputs

    run_dir = args.out_dir or records.new_run_dir(args.tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    rec_path = run_dir / "records.jsonl"
    w = records.RecordWriter(rec_path, run_id)
    device = "cuda"
    shapes = parse_shapes_arg(args)
    clock = timing.ClockLogger(0)

    w.write("env", env=records.env_info(),
            args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()})
    bw = timing.hbm_copy_bw()
    w.write("hbm_bw", **bw)
    print(f"records: {rec_path}\nmeasured HBM copy bandwidth: {bw['median_gbps']:.0f} GB/s",
          flush=True)

    arms, failed = build_arms(args)
    for key, err in failed:
        w.write("error", arm=key, variant=None, shape=None, stage="load_arm", error=err)
    headline = next((k for k, _ in arms if k == "megakernel"), None) or \
        (args.megakernels[0][0] if args.megakernels else None)

    for shape in shapes:
        floor = bytes_floor(shape)
        w.write("shape", shape=shape.to_dict(), sol=floor, expected=expected_outputs(shape))
        kept: list[dict] = []
        first_inputs = None
        for si, seed in enumerate(args.seed_list):
            try:
                inputs = make_shape_inputs(shape, seed, device)
                snap = accuracy.snapshot_inputs(inputs)
                ref_bf16 = reference_outputs(shape, inputs)
            except NotImplementedError as exc:
                w.write("error", arm=None, variant=None, shape=shape.id, seed=seed,
                        stage="inputs", error=str(exc))
                break
            try:
                ref_fp32, fp32_err = accuracy.fp32_reference(shape, inputs, device), None
            except Exception as exc:
                ref_fp32, fp32_err = None, f"{type(exc).__name__}: {exc}"[:500]
            for key, arm in arms:
                if si > 0 and getattr(arm, "name", key) in args.single_seed:
                    continue
                for v in selected_variants(key, arm, args):
                    reason = arm.supports(shape, v)
                    if reason:
                        if si == 0:
                            w.write("skip", arm=key, variant=v, shape=shape.id, reason=reason)
                        continue
                    try:
                        step = arm.prepare(shape, inputs, v, device)
                    except Exception:
                        w.write("error", arm=key, variant=v, shape=shape.id, seed=seed,
                                stage="prepare", error=_tb())
                        continue
                    ok = _check_step(w, accuracy, torch, key, v, step, shape, seed, inputs,
                                     snap, ref_bf16, ref_fp32, fp32_err, device)
                    if ok and si == 0:
                        kept.append({"arm": key, "variant": v, "step": step})
                    else:
                        step.close()
            if si == 0:
                first_inputs = inputs  # keep alive while steps reference it
            del ref_bf16, ref_fp32
        if kept:
            _time_steps(w, timing, torch, kept, shape, floor, bw, headline, args, clock, run_dir)
        for item in kept:
            item["step"].close()
        kept.clear()
        first_inputs = None
        gc.collect()
        torch.cuda.empty_cache()

    if "vllm" in args.arm_list:
        for shape in shapes:
            _run_vllm(w, args, shape, run_dir, run_id)
    w.close()
    recs = records.read_records(rec_path)
    text = report.render(recs)
    print(text)
    return 0


def _check_step(w, accuracy, torch, key, v, step, shape, seed, inputs, snap,
                ref_bf16, ref_fp32, fp32_err, device) -> bool:
    try:
        step.set_chained(False)
        step.launch()
        torch.cuda.synchronize()
        got = step.outputs()
        m_bf16 = accuracy.compare(got, ref_bf16)
        m_fp32 = accuracy.compare(got, ref_fp32) if ref_fp32 is not None else None
        own = _own_ref(step)
        m_own = m_fp32 if own == "fp32" else m_bf16
        band = accuracy.oracle_band(ref_bf16, ref_fp32) if ref_fp32 is not None else None
        if band is not None:
            grade = accuracy.grade_vs_oracle(m_fp32, step.precision, band)
        else:
            grade = accuracy.grade(m_own) if m_own is not None else "unknown"
        w.write("correctness", arm=key, variant=v, shape=shape.id, seed=seed,
                precision=step.precision, own_ref=own, config=step.config,
                grade=grade, oracle_band=band,
                grading="oracle_band" if band is not None else "own_reference",
                vs_bf16=m_bf16, vs_fp32=m_fp32, fp32_reference_error=fp32_err,
                megabench=accuracy.megabench_verdict(ref_bf16, got, shape, device),
                mutated_inputs=accuracy.mutated_inputs(inputs, snap))
        return True
    except Exception:
        w.write("error", arm=key, variant=v, shape=shape.id, seed=seed,
                stage="correctness", error=_tb())
        return False


def _time_steps(w, timing, torch, kept, shape, floor, bw, headline, args, clock, run_dir):
    name = lambda it: f"{it['arm']}:{it['variant']}"  # noqa: E731
    steps = {name(it): it["step"] for it in kept}
    seed = args.seed_list[0]
    flush = timing.l2_flush_buffer(0) if args.flush_l2 else None
    per_block = math.ceil(args.isolated_samples / args.rounds)
    modes = [("steady", "warm", True, lambda s: timing.steady(s, args.steady_steps), 1)]
    modes.append(("isolated", "warm", False, lambda s: timing.isolated(s), per_block))
    if flush is not None:
        modes.append(("isolated", "flushed", False, lambda s: timing.isolated(s, flush),
                      per_block))
    results: dict = {}
    for mode, l2, chained, measure, pb in modes:
        for s in steps.values():
            s.set_chained(chained)
        try:
            # Warm each mode separately: switching chained/unchained can re-record
            # graphs (torch.compile cudagraph trees took >1 s on the first call).
            timing.warmup(list(steps.values()))
            results[(mode, l2)] = timing.interleaved(steps, measure, args.rounds, pb, seed, clock)
        except Exception:
            w.write("error", arm=None, variant=None, shape=shape.id, stage=f"timing-{mode}-{l2}",
                    error=_tb())
    head = f"{headline}:{HEADLINE_VARIANT}"
    for (mode, l2), res in results.items():
        for it in kept:
            n = name(it)
            x = res.samples[n]
            stats = timing.summarize(x, seed)
            med = stats["median"]
            ratio = (timing.ratio_ci(x, res.samples[head], seed) if head in res.samples
                     else {"ratio": None, "ci95": None})
            gbps = floor["bytes"] / (med * 1e-3) / 1e9
            csum = timing.ClockLogger.summarize([c for c in res.clocks if c["arm"] == n])
            w.write("timing", arm=it["arm"], variant=it["variant"], shape=shape.id,
                    mode=mode, l2=l2, samples_ms=x, round_idx=res.round_idx[n], stats=stats,
                    n_steps=args.steady_steps if mode == "steady" else 1,
                    ratio_vs_headline=ratio, headline=head,
                    sol={"bytes": floor["bytes"], "floor_ms": floor["floor_ms"],
                         "efficiency": floor["floor_ms"] / med,
                         "gbps": gbps},
                    pct_measured_bw=100 * gbps / bw["median_gbps"],
                    clock=csum, throttled_rounds=csum.get("throttled_rounds", []),
                    config=it["step"].config)
    if not args.no_profile:
        for it in kept:
            try:
                stats = timing.kernel_timeline(it["step"], 10, run_dir / "traces")
                w.write("profile", arm=it["arm"], variant=it["variant"], shape=shape.id,
                        stats=stats, config=it["step"].config)
            except Exception:
                w.write("error", arm=it["arm"], variant=it["variant"], shape=shape.id,
                        stage="profile", error=_tb())


def _run_vllm(w, args, shape, run_dir, run_id) -> None:
    from .records import read_records
    for v, cmd, out in vllm_commands(args, shape, run_dir, run_id):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.vllm_timeout)
            failed, err = r.returncode != 0, (r.stderr or "")[-3000:]
        except subprocess.TimeoutExpired as exc:
            raw = exc.stderr or ""
            failed = True
            err = "timeout: " + (raw.decode(errors="replace") if isinstance(raw, bytes) else raw)[-2900:]
        if failed:
            w.write("error", arm="vllm", variant=v, shape=shape.id, stage="vllm_worker", error=err)
        if out.exists():
            for rec in read_records(out):
                w.append(rec)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))
