#!/usr/bin/env python3
"""Command line for the Qwen3-0.6B decode megakernel.

  run.py check [--weights hf|synthetic] [--seed N]
  run.py bench [--weights hf|synthetic] [--iters N]
  run.py demo --prompt "..."

Set HF_HOME to the Hugging Face cache. Set HF_HUB_OFFLINE=1 to forbid downloads.
"""
from __future__ import annotations

import argparse
import sys

import torch

from qwen3mk import Megakernel, bench as bench_mod, prefill as prefill_mod, reference, weights

DEFAULT_PROMPT = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. " * 8 + "The capital of Spain is"
GATE_LOGITS_REL_L2 = 1e-4
GATE_KV_ULPS = 2  # bf16 k/v: kernel and reference both round fp32 to bf16, so 1 ulp differences are normal
GATE_KV_REL_L2 = 1e-2
SYNC_LOGITS_REL_L2 = 1e-5  # the O-projection uses float atomics, so repeated launches differ by rounding


def get_inputs(kind: str, seed: int, prompt: str = DEFAULT_PROMPT, device="cuda") -> tuple[dict, dict]:
    """Kernel inputs and an info dict. 'hf' loads real weights and builds the KV cache from `prompt`."""
    if kind == "synthetic":
        return weights.synthetic(seed, device), {"weights": "synthetic", "seed": seed}
    w = weights.load_hf(device=device)
    tok = prefill_mod.load_tokenizer()
    inputs, info = prefill_mod.prefix_state(w, tok, prompt, device)
    info["weights"] = "hf"
    return inputs, info


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float((a - b).norm() / b.norm())


def bf16_ulps(a: torch.Tensor, ref: torch.Tensor) -> float:
    """Largest |a - ref| in bf16 ulps (8-bit significand: ulp(x) = 2^(floor(log2 |x|) - 7)).

    The unit is the ulp of max(|ref|, rms(ref)). Near zero, the ulp of the tensor RMS is the floor:
    a value close to 0 comes from a difference of large terms, so its error scales with the tensor, not with the value.
    """
    r = ref.float().abs().clamp_min(float(ref.float().pow(2).mean().sqrt()))
    ulp = torch.exp2(torch.floor(torch.log2(r)) - 7)
    return float(((a.float() - ref.float()).abs() / ulp).max())


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max())


def cmd_check(args) -> int:
    inputs, info = get_inputs(args.weights, args.seed)
    print(f"inputs: {info}")
    mk = Megakernel()
    snap = {k: v.clone() for k, v in inputs.items()}
    out = mk.step(inputs)
    torch.cuda.synchronize()
    ref = reference.decode_step(reference.to_fp32(inputs), inputs["token"], inputs["kcache"], inputs["vcache"])
    ok = True

    def gate(name: str, value, limit, good: bool) -> None:
        nonlocal ok
        ok &= good
        print(f"  {name:<28} {value}  (limit {limit})  {'ok' if good else 'FAIL'}")

    print("kernel vs fp32 reference")
    tok_match = int(out["next_token"]) == int(ref["next_token"])
    print(f"  next_token kernel={int(out['next_token'])} reference={int(ref['next_token'])}")
    gate("next_token equal", tok_match, "must be True", tok_match)
    print(f"  logits argmax match         {int(out['logits'].argmax()) == int(ref['logits'].argmax())}")
    e = rel_l2(out["logits"], ref["logits"])
    print(f"  logits max-abs              {max_abs(out['logits'], ref['logits']):.3e}")
    gate("logits rel-L2", f"{e:.3e}", GATE_LOGITS_REL_L2, e <= GATE_LOGITS_REL_L2)
    for n in ("k_write", "v_write"):
        r = rel_l2(out[n], ref[n])
        gate(f"{n} rel-L2", f"{r:.3e}", GATE_KV_REL_L2, r <= GATE_KV_REL_L2)
        print(f"  {n} max-abs              {max_abs(out[n], ref[n]):.3e}")
        u = bf16_ulps(out[n], ref[n])
        gate(f"{n} max bf16 ulps", f"{u:.0f}", GATE_KV_ULPS, u <= GATE_KV_ULPS)

    changed = [k for k in inputs if not torch.equal(inputs[k], snap[k])]
    gate("inputs not mutated", changed or "none changed", "none", not changed)
    del snap

    first = {k: v.clone() for k, v in out.items()}
    bound = mk.bind(inputs)
    for _ in range(200):
        mk.launch()
    torch.cuda.synchronize()
    last = (bound["logits"], bound["next_token"], bound["k_write"], bound["v_write"])
    e = rel_l2(last[0], first["logits"])
    gate("200 launches: next_token", int(last[1]) == int(first["next_token"]), "equal", int(last[1]) == int(first["next_token"]))
    gate("200 launches: logits rel-L2", f"{e:.3e}", SYNC_LOGITS_REL_L2, e <= SYNC_LOGITS_REL_L2)
    for n, t in (("k_write", last[2]), ("v_write", last[3])):
        u = bf16_ulps(t, first[n])
        gate(f"200 launches: {n} max bf16 ulps", f"{u:.0f}", GATE_KV_ULPS, u <= GATE_KV_ULPS)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_bench(args) -> int:
    inputs, info = get_inputs(args.weights, args.seed)
    mk = Megakernel()
    g = bench_mod.gpu_info(mk.device)
    print(f"gpu: {g['name']}  SMs: {g['sm_count']}  SM clock: {g['sm_clock_mhz']} MHz  inputs: {info}")
    iso = bench_mod.isolated(mk, inputs, iters=args.iters)
    print(f"isolated: median {iso['event_us_median']:.1f} us (events), {iso['wall_us_median']:.1f} us (wall); "
          f"event min {iso['event_us_min']:.1f} max {iso['event_us_max']:.1f}; iters {iso['iters']}")
    st = bench_mod.steady(mk, inputs)
    print(f"steady:   median {st['us_per_step_median']:.1f} us/step; min {st['us_per_step_min']:.1f} max {st['us_per_step_max']:.1f}; "
          f"{st['reps']} chains of {st['steps']} steps")
    return 0


def cmd_demo(args) -> int:
    tok = prefill_mod.load_tokenizer()
    try:
        prefill_mod.build_prefix(tok, args.prompt)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    w = weights.load_hf(device="cuda")
    inputs, info = prefill_mod.prefix_state(w, tok, args.prompt, "cuda")
    print(f"prompt tokens: {info['prompt_tokens']}  cut: {info['cut']}  input id ({info['input_id_source']}): "
          f"{info['input_id']} {tok.decode([info['input_id']])!r}")
    mk = Megakernel()
    out = mk.step(inputs)
    ref = reference.decode_step(reference.to_fp32(inputs), inputs["token"], inputs["kcache"], inputs["vcache"])
    got, want = int(out["next_token"]), int(ref["next_token"])
    print(f"megakernel next token: {got} {tok.decode([got])!r}")
    print(f"reference  next token: {want} {tok.decode([want])!r}")
    print(f"match: {got == want}")
    return 0 if got == want else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "bench"):
        p = sub.add_parser(name)
        p.add_argument("--weights", choices=["hf", "synthetic"], default="synthetic")
        p.add_argument("--seed", type=int, default=0)
        if name == "bench":
            p.add_argument("--iters", type=int, default=200)
    p = sub.add_parser("demo")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = ap.parse_args(argv)
    return {"check": cmd_check, "bench": cmd_bench, "demo": cmd_demo}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
