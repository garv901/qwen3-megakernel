"""R1 numerical feasibility: fixed-reference softmax accumulation (m_ref instead of the running max), torch on GPU, RANDOM q / k / v.
Part 1 (gap sweep): for g = m_ref - true_max in log2 units, each row computes p = ex2(s - m_ref) (fp32, with the ex2.approx.ftz flush: p < 2^-126 -> 0), S pieces (o_s = sum p v, l_s = sum p, fp32), the pieces added into one fp32
  accumulator in a random order (atomic-order emulation), out = o / l. Error against an fp64 exact softmax; also with p rounded to bf16 before the PV product (the tensor-core path; l stays fp32), and against the usual
  local-max merge (g = 0 per piece) as the baseline. nondet = max |out(order 1) - out(order 2)| / scale: the run-to-run spread of fp32 atomic order.
Part 2 (bound): q = rmsnorm(x) * gq, k = rmsnorm(x) * gk (Qwen3-like q_norm / k_norm, unit-RMS before the weight), m_ref = |q| * max_k |k| * scale * log2e; gap = m_ref - true_max for random data and several weight scales.
NOTE: random data only. The bound looseness here says nothing about real Qwen3 activations (learned norm weights, aligned q / k, attention sinks); it must be measured on real q / k before any port.
Usage: python mref_num.py [--out file.json]
"""
import argparse, json
import torch

dev = "cuda"
D = 128
LOG2E = 1.4426950408889634
TINY = 2.0 ** -126


def flush(x):
    return torch.where(x < TINY, torch.zeros_like(x), x)


def gen(nq, kv, seed, gq=1.0, gk=1.0):
    g = torch.Generator(device=dev); g.manual_seed(seed)
    def rms(x, gmax):
        x = x / x.pow(2).mean(-1, keepdim=True).sqrt()
        if gmax != 1.0:
            x = x * (0.5 + (gmax - 0.5) * torch.rand(D, device=dev, generator=g))      # learned weight, uniform in [0.5, gmax]
        return x
    q = rms(torch.randn(nq, D, device=dev, generator=g), gq)
    k = rms(torch.randn(kv, D, device=dev, generator=g), gk)
    v = torch.randn(kv, D, device=dev, generator=g)
    return q, k, v


def scores(q, k):
    return (q @ k.T) * (D ** -0.5) * LOG2E                                         # log2 domain


def exact(s, v):
    p = torch.exp2((s - s.max(1, keepdim=True).values).double())
    return (p @ v.double()) / p.sum(1, keepdim=True)


def fixed_ref(s, v, mref, S, perm, bf16p=False):
    kv = s.shape[1]
    ch = kv // S
    o = torch.zeros(s.shape[0], D, device=dev); l = torch.zeros(s.shape[0], 1, device=dev)
    for i in perm:
        sl = slice(i * ch, (i + 1) * ch)
        p = flush(torch.exp2(s[:, sl] - mref))
        pm = p.to(torch.bfloat16).float() if bf16p else p
        o = o + pm @ v[sl]; l = l + p.sum(1, keepdim=True)                         # fp32 adds in the given piece order (the atomics)
    return o / l, l


def local_max(s, v, S):
    kv = s.shape[1]; ch = kv // S
    ms, ls, os_ = [], [], []
    for i in range(S):
        sl = slice(i * ch, (i + 1) * ch)
        m = s[:, sl].max(1, keepdim=True).values
        p = flush(torch.exp2(s[:, sl] - m))
        ms.append(m); ls.append(p.sum(1, keepdim=True)); os_.append(p @ v[sl])
    M = torch.stack(ms).max(0).values
    w = [flush(torch.exp2(m - M)) for m in ms]
    return sum(wi * oi for wi, oi in zip(w, os_)) / sum(wi * li for wi, li in zip(w, ls))


def err(out, ref):
    sc = ref.abs().amax(1, keepdim=True)
    return float(((out.double() - ref).abs() / sc).amax())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); ap.add_argument("--nq", type=int, default=16); ap.add_argument("--S", type=int, default=32)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    res = {"gap_sweep": [], "bound": []}
    print("== part 1: error vs gap g = m_ref - true_max (log2), S =", a.S, "pieces, fp32 accumulation in random order; err = max_rows max_d |out - exact| / max_d |exact|")
    print(f"{'kv':>7} {'g':>5} {'err fp32 p':>11} {'err bf16 p':>11} {'nondet':>9} {'log2 l':>7} {'NaN rows':>8}")
    for kv in (4096, 32768, 262144):
        q, k, v = gen(a.nq, kv, 1)
        s = scores(q, k)
        ref = exact(s, v)
        tm = s.max(1, keepdim=True).values
        ln = local_max(s, v, a.S)
        base = err(ln, ref)
        r0 = {"kv": kv, "g": "local-max", "err": base}; res["gap_sweep"].append(r0)
        print(f"{kv:7d} {'local':>5} {base:11.2e}")
        gen_ = torch.Generator(); gen_.manual_seed(7)
        for g in (0, 8, 16, 32, 64, 100, 104, 108, 112, 116, 120, 124, 126, 130):
            mref = tm + g
            p1 = torch.randperm(a.S, generator=gen_).tolist(); p2 = torch.randperm(a.S, generator=gen_).tolist()
            o1, l1 = fixed_ref(s, v, mref, a.S, p1); o2, _ = fixed_ref(s, v, mref, a.S, p2)
            ob, _ = fixed_ref(s, v, mref, a.S, p1, bf16p=True)
            nan = int((~torch.isfinite(o1).all(1)).sum())
            e1 = err(torch.nan_to_num(o1, nan=1e9), ref); eb = err(torch.nan_to_num(ob, nan=1e9), ref)
            nd = float(((o1 - o2).abs() / ref.abs().amax(1, keepdim=True).float()).nan_to_num(nan=1e9).amax())
            lg = float(torch.log2(l1[l1 > 0]).min()) if (l1 > 0).any() else float("-inf")
            res["gap_sweep"].append({"kv": kv, "g": g, "err_fp32p": e1, "err_bf16p": eb, "nondet": nd, "log2_l_min": lg, "nan_rows": nan})
            print(f"{kv:7d} {g:5d} {e1:11.2e} {eb:11.2e} {nd:9.1e} {lg:7.1f} {nan:8d}")
    print("\n== part 2: m_ref = |q| max|k| scale log2e (norm bound) vs true max, random data; gap = m_ref - true_max (log2 units)")
    print(f"{'kv':>7} {'gq':>4} {'gk':>4} {'bound':>7} {'true max':>9} {'gap min':>8} {'gap med':>8} {'gap max':>8}  (bound = max over rows of the bound)")
    for kv in (4096, 262144):
        for gq, gk in ((1, 1), (2, 2), (3, 3), (4, 4), (6, 6), (8, 8)):
            q, k, v = gen(64, kv, 2, gq, gk)
            s = scores(q, k)
            tm = s.max(1).values
            bd = q.norm(dim=1) * k.norm(dim=1).max() * (D ** -0.5) * LOG2E
            gap = bd - tm
            res["bound"].append({"kv": kv, "gq": gq, "gk": gk, "bound_max": float(bd.max()), "true_max_med": float(tm.median()), "gap_min": float(gap.min()), "gap_med": float(gap.median()), "gap_max": float(gap.max())})
            print(f"{kv:7d} {gq:4d} {gk:4d} {float(bd.max()):7.1f} {float(tm.median()):9.1f} {float(gap.min()):8.1f} {float(gap.median()):8.1f} {float(gap.max()):8.1f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
