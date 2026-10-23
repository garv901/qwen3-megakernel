"""Run driver: ``python -m regimebench run --arms ... --workloads ... --tag NAME``.

Per workload: fixture (timing seed = first seed) -> probes (once per job) -> for
each arm/variant ``supports`` (skip record) -> ``prepare`` (error record on
failure) -> accuracy (``w.check`` or R0; every seed; proxy arms skipped) ->
warmup -> interleaved timing over all prepared steps. vLLM runs out of process
(``arms/vllm_worker.py``) after the in-process steps are freed. Nothing aborts
the whole run: every failure becomes an ``error`` record with a traceback.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from .sol import sol, sol_both
from .workload import Workload, regime_grid, select

DEFAULT_SEEDS = (7301, 8917, 42811)
DEFAULT_ARMS = ("fi-graph", "hv2g", "hv2-repeat", "vllm")
VLLM_VARIANTS = ("pdl", "nopdl")
TARGET_SAMPLE_MS = 20.0
MIN_LAUNCHES, MAX_LAUNCHES = 3, 50
ISOLATED_PER_ROUND = 5
VLLM_TIMEOUT_S = 1800


# --------------------------------------------------------------------------
# Pure helpers (CPU-testable)


def parse_variants(spec: str | None) -> dict[str, list[str]]:
    """``"fi-graph=graph-pdl-best,graph-pdl-cublas;vllm=pdl"`` -> {arm: [variants]}."""
    out: dict[str, list[str]] = {}
    for part in (spec or "").split(";"):
        part = part.strip()
        if not part:
            continue
        arm, _, vs = part.partition("=")
        if not vs:
            raise ValueError(f"bad --variants entry {part!r}; use arm=v1,v2")
        out[arm.strip()] = [v.strip() for v in vs.split(",") if v.strip()]
    return out


def launches_per_sample(estimate_ms: float) -> int:
    """n back-to-back launches so one sample is ~TARGET_SAMPLE_MS of work (3..50)."""
    if not estimate_ms or estimate_ms <= 0:
        return MAX_LAUNCHES
    return max(MIN_LAUNCHES, min(MAX_LAUNCHES, round(TARGET_SAMPLE_MS / estimate_ms)))


def smoke_workloads(pool: list[Workload]) -> list[Workload]:
    """R0 plus the first point of every regime present in ``pool``."""
    seen, out = set(), []
    for w in pool:
        if w.regime not in seen:
            seen.add(w.regime)
            out.append(w)
    return out


def select_workloads(ids: str | None, smoke: bool) -> list[Workload]:
    names = [x.strip() for x in ids.split(",") if x.strip()] if ids else None
    pool = select(names)
    if names:
        known = {w.id for w in regime_grid()} | {w.regime for w in regime_grid()}
        unknown = [n for n in names if n not in known]
        if unknown:
            raise SystemExit(f"unknown workloads/regimes {unknown}; "
                             f"{len(known)} known names, see regimebench.workload.regime_grid()")
    return smoke_workloads(pool) if smoke else pool


def wants_accuracy(w: Workload) -> bool:
    return bool(w.check or w.regime == "R0")


def tokens_per_s(total_q: int, steady_median_ms: float | None) -> float | None:
    if not steady_median_ms or steady_median_ms <= 0:
        return None
    return total_q / (steady_median_ms * 1e-3)


def plan(args, workloads: list[Workload], variants: dict[str, list[str]]) -> dict:
    return {"tag": args.tag, "arms": args.arms, "variants_override": variants,
            "seeds": args.seeds, "rounds": args.rounds, "smoke": args.smoke,
            "workloads": [{"id": w.id, "regime": w.regime, "total_q": w.total_q,
                           "total_kv": w.total_kv, "check": wants_accuracy(w),
                           "sol_us": round(sol(w)["t_sol_us"], 2),
                           "bound": sol(w)["bound"]} for w in workloads]}


# --------------------------------------------------------------------------
# GPU run


class Runner:
    def __init__(self, args, writer, workloads, variants):
        import torch
        self.torch = torch
        self.args, self.writer = args, writer
        self.workloads, self.variants_override = workloads, variants
        self.arm_names = [a for a in args.arms if a != "vllm"]
        self.do_vllm = "vllm" in args.arms
        self.seeds = args.seeds
        self.rounds = args.rounds
        self.device = "cuda"
        self.arms: dict = {}
        self.probes: dict | None = None
        self.clock = None
        self.run_dir: Path = writer.path.parent

    # -- records ---------------------------------------------------------
    def error(self, w, arm, variant, stage, exc=None):
        tb = traceback.format_exc() if exc is not None else ""
        self.writer.write("error", arm=arm, variant=variant,
                          workload=w.to_dict() if w else None, stage=stage,
                          error=repr(exc) if exc is not None else None, traceback=tb)
        print(f"[error] {arm}:{variant} {w.id if w else '-'} {stage}: {exc!r}", flush=True)

    def sol_dict(self, w):
        bw = (self.probes or {}).get("hbm", {}).get("bw_bytes_s")
        peak = (self.probes or {}).get("gemm", {}).get("peak_flops")
        return sol_both(w, bw, peak)

    # -- setup -----------------------------------------------------------
    def ensure_probes(self):
        if self.probes is not None:
            return
        try:
            from . import probes
            self.probes = probes.run_probes()
            self.writer.write("probe", **self.probes)
        except Exception as exc:
            self.probes = {}
            self.error(None, "probes", "-", "probes", exc)

    def get_arm(self, name):
        if name not in self.arms:
            try:
                from .arms.base import load_arm
                self.arms[name] = load_arm(name)
            except Exception as exc:  # ImportError while arms are still being written
                self.arms[name] = None
                self.error(None, name, "-", "load_arm", exc)
        return self.arms[name]

    def variants_for(self, name, arm):
        if name in self.variants_override:
            return self.variants_override[name]
        return list(arm.variants())

    # -- one workload ----------------------------------------------------
    def free(self):
        gc.collect()
        self.torch.cuda.empty_cache()

    def run_workload(self, w: Workload):
        from .fixture import make_inputs
        torch = self.torch
        print(f"=== {w.id} (total_q {w.total_q}, kv {w.total_kv}) ===", flush=True)
        inputs = make_inputs(w, self.seeds[0], self.device)
        self.ensure_probes()
        steps: dict = {}
        if self.arm_names:
            for name in self.arm_names:
                arm = self.get_arm(name)
                if arm is None:
                    continue
                try:
                    vs = self.variants_for(name, arm)
                except Exception as exc:
                    self.error(w, name, "-", "variants", exc)
                    continue
                for v in vs:
                    self.prepare_one(w, arm, name, v, inputs, steps)
        if wants_accuracy(w) and steps:
            self.accuracy(w, inputs, steps)
        if not w.check:
            for k in ("kcache", "vcache"):
                inputs.pop(k, None)
        if steps:
            self.time_steps(w, steps)
        for st in steps.values():
            try:
                st.close()
            except Exception as exc:
                self.error(w, st.arm, st.variant, "close", exc)
        steps.clear()
        del inputs
        self.free()
        if self.do_vllm:
            self.run_vllm(w)

    def prepare_one(self, w, arm, name, v, inputs, steps):
        try:
            reason = arm.supports(w, v)
        except Exception as exc:
            self.error(w, name, v, "supports", exc)
            return
        if reason is not None:
            self.writer.write("skip", arm=name, variant=v, workload=w.to_dict(), reason=reason)
            print(f"[skip] {name}:{v}: {reason}", flush=True)
            return
        try:
            t0 = time.perf_counter()
            steps[f"{name}:{v}"] = arm.prepare(w, inputs, v, self.device)
            print(f"[prepared] {name}:{v} in {time.perf_counter() - t0:.1f}s", flush=True)
        except Exception as exc:
            self.error(w, name, v, "prepare", exc)
            self.free()

    # -- accuracy --------------------------------------------------------
    def references(self, w, inputs) -> tuple[dict, dict, dict]:
        from .harness.accuracy import oracle_band
        from .ref.reference import forward
        torch = self.torch
        to_cpu = lambda d: {k: v.detach().cpu() for k, v in d.items()}  # noqa: E731
        ref32 = to_cpu(forward(w, inputs, torch.float32))
        ref16 = to_cpu(forward(w, inputs, torch.bfloat16))
        return ref16, ref32, oracle_band(ref16, ref32)

    def grade_step(self, w, st, seed, ref16, ref32, band):
        from .harness import accuracy as acc
        torch = self.torch
        st.launch()
        torch.cuda.synchronize()
        out = st.outputs()
        m32, m16 = acc.compare(out, ref32), acc.compare(out, ref16)
        g = acc.grade_vs_oracle(m32, band)
        self.writer.write("accuracy", arm=st.arm, variant=st.variant, workload=w.to_dict(),
                          config=st.config, seed=seed, precision=getattr(st, "precision", "bf16"),
                          vs_fp32=m32, vs_bf16_ref=m16, band=band,
                          next_token_match_rate=m32["next_token"]["match_rate"], **g)
        print(f"[accuracy] {st.arm}:{st.variant} seed {seed}: {g['verdict']} "
              f"(worst {g['worst_ratio']:.2f}x band)", flush=True)

    def accuracy(self, w, inputs, steps):
        from .fixture import make_inputs
        torch = self.torch
        proxies = {k for k, st in steps.items() if st.config.get("proxy")}
        for k in proxies:
            st = steps[k]
            self.writer.write("skip", arm=st.arm, variant=st.variant, workload=w.to_dict(),
                              reason="accuracy skipped: proxy arm", stage="accuracy")
        live = [k for k in steps if k not in proxies]
        if not live:
            return
        for seed in self.seeds:
            try:
                if seed == self.seeds[0]:
                    cur = inputs
                else:
                    cur = make_inputs(w, seed, self.device)
                ref16, ref32, band = self.references(w, cur)
                self.writer.write("accuracy", arm="reference", variant="bf16", workload=w.to_dict(),
                                  seed=seed, band=band, verdict="reference",
                                  note="BF16 reference rel-L2 to FP32 oracle (the band)")
            except Exception as exc:
                self.error(w, "reference", "-", "reference", exc)
                continue
            for k in live:
                try:
                    if seed == self.seeds[0]:
                        self.grade_step(w, steps[k], seed, ref16, ref32, band)
                    else:
                        arm_name, _, v = k.partition(":")
                        tmp = self.arms[arm_name].prepare(w, cur, v, self.device)
                        try:
                            self.grade_step(w, tmp, seed, ref16, ref32, band)
                        finally:
                            tmp.close()
                except Exception as exc:
                    self.error(w, k.split(":")[0], k.split(":", 1)[1], "accuracy", exc)
            if cur is not inputs:
                del cur
            del ref16, ref32
            self.free()

    # -- timing ----------------------------------------------------------
    def time_steps(self, w, steps):
        from .harness import timing
        torch = self.torch
        if self.clock is None:
            self.clock = timing.ClockLogger(0)
        warm_s = 0.2 if self.args.smoke else 1.0
        n_launch: dict[str, int] = {}
        for k in list(steps):
            st = steps[k]
            try:
                timing.warmup([st], seconds=warm_s)
                est = min(timing.isolated(st) for _ in range(3))
                n_launch[k] = launches_per_sample(est)
            except Exception as exc:
                self.error(w, st.arm, st.variant, "warmup", exc)
                try:
                    st.close()
                except Exception:
                    pass
                del steps[k]
        if not steps:
            return
        seed = self.seeds[0]
        try:
            res_s = timing.interleaved(steps, lambda st: timing.steady(st, n_launch[f"{st.arm}:{st.variant}"]),
                                       self.rounds, 1, seed, self.clock)
            res_i = timing.interleaved(steps, timing.isolated, self.rounds,
                                       ISOLATED_PER_ROUND, seed + 1000, None)
        except Exception as exc:
            for st in steps.values():
                self.error(w, st.arm, st.variant, "timing", exc)
            return
        sol_d = self.sol_dict(w)
        for k, st in steps.items():
            steady = timing.summarize(res_s.samples[k], seed)
            iso = timing.summarize(res_i.samples[k], seed)
            clocks = [c for c in res_s.clocks if c["arm"] == k]
            self.writer.write(
                "timing", arm=st.arm, variant=st.variant, workload=w.to_dict(), config=st.config,
                steady=steady, isolated=iso, n_launch_per_sample=n_launch[k], rounds=self.rounds,
                seed=seed, clock=timing.ClockLogger.summarize(clocks), sol=sol_d,
                tokens_per_s=tokens_per_s(w.total_q, steady.get("median")),
                unit="ms_per_step")
            print(f"[timing] {k}: steady {steady['median'] * 1e3:.1f} us, "
                  f"isolated {iso['median'] * 1e3:.1f} us "
                  f"(SOL {sol_d['spec']['t_sol_us']:.1f} us)", flush=True)

    # -- vLLM ------------------------------------------------------------
    def run_vllm(self, w):
        self.ensure_probes()
        sol_d = self.sol_dict(w)
        variants = self.variants_override.get("vllm", list(VLLM_VARIANTS))
        for v in variants:
            work = Path(tempfile.mkdtemp(prefix="vllm-", dir=self.run_dir))
            wj, out = work / "workload.json", work / "out.jsonl"
            wj.write_text(json.dumps(w.to_dict()))
            cmd = [sys.executable, "-m", "regimebench.arms.vllm_worker", "--workload-json", str(wj),
                   "--variant", v, "--out", str(out), "--run-id", self.writer.run_id,
                   "--seed", str(self.seeds[0]), "--rounds", str(self.rounds)]
            print(f"[vllm] {w.id} {v}", flush=True)
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=VLLM_TIMEOUT_S)
                (work / "stdout.log").write_text(r.stdout)
                (work / "stderr.log").write_text(r.stderr)
                merged = self.merge_worker(out, w, sol_d)
                if r.returncode != 0 or not merged:
                    self.writer.write("error", arm="vllm", variant=v, workload=w.to_dict(),
                                      stage="vllm_worker", error=f"returncode {r.returncode}, "
                                      f"{merged} records", traceback=r.stderr[-4000:])
            except Exception as exc:
                self.error(w, "vllm", v, "vllm_worker", exc)

    def merge_worker(self, path: Path, w, sol_d) -> int:
        from .harness.records import read_records
        if not path.exists():
            return 0
        n = 0
        for rec in read_records(path):
            rec.setdefault("slurm_job_id", os.environ.get("SLURM_JOB_ID"))
            rec.setdefault("workload", w.to_dict())
            if rec.get("record_type") == "timing":
                rec.setdefault("sol", sol_d)
                med = (rec.get("steady") or {}).get("median")
                rec.setdefault("tokens_per_s", tokens_per_s(w.total_q, med))
            self.writer.append(rec)
            n += 1
        return n

    def run(self):
        for w in self.workloads:
            try:
                self.run_workload(w)
            except Exception as exc:
                self.error(w, "-", "-", "workload", exc)
                self.free()


def main_run(args) -> int:
    variants = parse_variants(args.variants)
    workloads = select_workloads(args.workloads, args.smoke)
    if args.smoke:
        args.rounds, args.seeds = 3, args.seeds[:1]
    if args.dry_run:
        print(json.dumps(plan(args, workloads, variants), indent=1))
        return 0
    from .harness.records import RecordWriter, env_info, new_run_dir
    run_dir = new_run_dir(args.tag)
    run_id = run_dir.name
    writer = RecordWriter(run_dir / "records.jsonl", run_id)
    writer.write("env", env=env_info(), run_config=plan(args, workloads, variants))
    print(f"run dir {run_dir}", flush=True)
    try:
        Runner(args, writer, workloads, variants).run()
    finally:
        writer.close()
    return 0


def add_run_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--arms", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   default=list(DEFAULT_ARMS))
    p.add_argument("--workloads", default=None, help="comma list of workload ids / regimes; default all")
    p.add_argument("--tag", required=True)
    p.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")],
                   default=list(DEFAULT_SEEDS))
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--variants", default=None, help='"arm=v1,v2;arm2=v3"')
