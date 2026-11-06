"""R1 (fixed-reference split-K accumulation) on REAL Qwen3-0.6B decode activations.
  python mref_real.py run     --out DIR [--quick]   capture + stats + numerics -> DIR/records.npz, DIR/meta.json
  python mref_real.py analyze --out DIR             tables -> DIR/summary.json, DIR/report.txt
Scores s are in mk's domain (log2): s = (q . k) / sqrt(128) * log2(e), q post q_norm, post RoPE, k post k_norm, post RoPE (cached K).
Every (seq, step, layer, q-head) record is one decode-step attention row over all kv cached keys (GQA: q-head h -> kv-head h//2).
Window (per record), g = m_ref - true_max (log2):
  g_lo : g must be > g_lo = max(log2 sum p, log2 max_d sum p|v_d|) - 128   (no fp32 overflow of l or o partial sums, p_exact = 2^(s-max), terms of mass 1 at max)
  g_hi : g must be <= g_hi = 126 - d*, where the ex2.ftz-flushed terms (p < 2^-126, i.e. max - s > 126 - g) hold <= 1e-6 of the total mass l
"""
import argparse, glob, json, math, os, sys, time
import numpy as np

D = 128; NH = 16; NKV = 8; NL = 28
LOG2E = 1.4426950408889634
QS = D ** -0.5 * LOG2E
TOL = 1e-6
TINY = 2.0 ** -126
CANDS = ["oracle", "a_norm", "e1_norm_seqmax", "e2_norm_static", "b_static0", "b_static8", "c_prev0", "c_prev8", "d_sink0", "d_sink8"]
CFGS = [f"S{S}_{m}" for S in (32, 128) for m in ("fp32", "bf16")]
LOCAL = [f"local_S{S}_{m}" for S in (32, 128) for m in ("fp32", "bf16")]
STEPS = 8  # decode steps; record index 0 is the prefill last token (step -1), 1..8 = decode steps 0..7


# ---------------------------------------------------------------- data
def load_sharegpt():
    p = glob.glob("/raid/hf/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/*/ShareGPT_V3_unfiltered_cleaned_split.json")[0]
    d = json.load(open(p))
    d = [c["conversations"] for c in d if len(c["conversations"]) >= 4]
    return d


def code_files():
    import transformers, torch
    fs = []
    for root in (os.path.dirname(transformers.__file__), os.path.dirname(torch.__file__)):
        for dp, dn, fn in os.walk(root):
            dn.sort()
            for f in sorted(fn):
                if f.endswith(".py") and "test" not in dp:
                    fs.append(os.path.join(dp, f))
    return fs


def build_prompts(tok, lengths, quick):
    """-> list of dict(name, kind, split, L, ids). ids has L-STEPS tokens. cal/test use disjoint documents."""
    sg = load_sharegpt()
    rng = np.random.RandomState(0)
    order = rng.permutation(len(sg)).tolist()
    cf = code_files(); crng = np.random.RandomState(1); cf = [cf[i] for i in crng.permutation(len(cf))]
    out = []
    sg_pos = {"cal": 0, "test": len(sg) // 2}
    cf_pos = {"cal": 0, "test": len(cf) // 2}
    splits = ("cal", "test")
    for L in lengths:
        P = L - STEPS
        for split in splits:
            for kind in ("prose", "chat", "code"):
                if kind == "prose":
                    txt = ""; ids = []
                    while len(ids) < P:
                        c = sg[order[sg_pos[split]]]; sg_pos[split] += 1
                        txt += "\n\n".join(t["value"] for t in c) + "\n\n"
                        if len(txt) > 3 * P: ids = tok(txt)["input_ids"]
                    ids = ids[:P]
                elif kind == "chat":
                    msgs = []; ids = []
                    while len(ids) < P + 8:
                        c = sg[order[sg_pos[split]]]; sg_pos[split] += 1
                        for t in c:
                            msgs.append({"role": "user" if t["from"] == "human" else "assistant", "content": t["value"]})
                        if msgs[-1]["role"] == "user": msgs.append({"role": "assistant", "content": "ok"})
                        if sum(len(m["content"]) for m in msgs) > 2 * P:
                            ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, return_dict=True)["input_ids"]
                    ids = ids[:P]  # truncated mid-dialogue; first token is <|im_start|>
                else:
                    txt = ""; ids = []
                    while len(ids) < P:
                        txt += open(cf[cf_pos[split]], errors="ignore").read() + "\n"; cf_pos[split] += 1
                        if len(txt) > 3 * P: ids = tok(txt)["input_ids"]
                    ids = ids[:P]
                out.append(dict(name=f"{kind}-{split}-{L // 1024}K", kind=kind, split=split, L=L, ids=ids))
    return out


# ---------------------------------------------------------------- stats / numerics
def flush(x):
    import torch
    return torch.where(x < TINY, torch.zeros_like(x), x)


def stats(q, K, V):
    """q [16,D] (bf16 post-RoPE), K,V [8,n,D] bf16. returns dict of torch tensors [16] (+ s, mx for numerics)."""
    import torch
    n = K.shape[1]
    qs = q.float() * QS
    Kf = K.float()
    s = torch.einsum("kgd,knd->kgn", qs.view(NKV, 2, D), Kf).reshape(NH, n)
    sd, si = torch.sort(s, dim=1, descending=True)
    mx = sd[:, 0]
    r = {"mx": mx, "mn": sd[:, -1], "sec": sd[:, 1], "p999": sd[:, int(0.001 * n)], "argmax": si[:, 0].float(), "s0": s[:, 0]}
    d = (mx[:, None] - sd).double()
    p = torch.exp2(-d)
    L = p.sum(1)
    r["lL"] = torch.log2(L).float()
    r["sink_mass"] = (torch.exp2(s[:, 0].double() - mx.double()) / L).float()
    tc = torch.flip(torch.cumsum(torch.flip(p, [1]), 1), [1]) / L[:, None]   # mass of the suffix starting at j
    jst = (tc > TOL).sum(1).clamp(min=1, max=n - 1)                          # first j whose suffix mass <= tol
    dstar = d.gather(1, (jst - 1)[:, None])[:, 0]
    r["g_hi"] = (126.0 - dstar).float()
    pe = torch.exp2(s - mx[:, None])                                         # fp32, <= 1
    Vabs = V.float().abs()
    ov = torch.einsum("kgn,knd->kgd", pe.view(NKV, 2, n), Vabs).reshape(NH, D)  # sum_k p |v_d|
    r["lO"] = torch.log2(ov.amax(1).clamp(min=1e-30))
    r["lOn"] = r["lO"] - r["lL"]
    r["vmax"] = Vabs.amax((1, 2)).repeat_interleave(2)
    r["g_lo"] = torch.maximum(r["lL"], r["lO"]) - 128.0
    r["qn"] = qs.norm(dim=1)
    kn = Kf.norm(dim=2)                                                      # [8,n]
    knk = kn.amax(1)
    r["knmax"] = knk.repeat_interleave(2)
    r["knmax_layer"] = knk.amax().expand(NH)
    return r, s, qs, knk


def pad_to(x, S, dim, val):
    n = x.shape[dim]; ch = -(-n // S); npad = ch * S
    if npad == n: return x
    import torch
    sh = list(x.shape); sh[dim] = npad - n
    return torch.cat([x, torch.full(sh, val, dtype=x.dtype, device=x.device)], dim)


def numerics(s, V, mx, mrefs, gen):
    """s [16,n] fp32, V [8,n,D] bf16. returns err [ncand, ncfg], flags [ncand, ncfg] (bit0 nonfinite, bit1 l==0 or <tiny, bit2 piece overflow), err_local [4]."""
    import torch
    ref = torch.einsum("kgn,knd->kgd", torch.exp2(s.double() - mx.double()[:, None]).view(NKV, 2, -1), V.double()).reshape(NH, D)
    ref = ref / torch.exp2(s.double() - mx.double()[:, None]).sum(1)[:, None]
    rs = ref.abs().amax(1)
    err = np.zeros((len(CANDS), len(CFGS)), np.float64); flg = np.zeros((len(CANDS), len(CFGS)), np.int32); lerr = np.zeros(len(LOCAL))
    Vf = V.float()
    for si_, S in enumerate((32, 128)):
        sp = pad_to(s, S, 1, float("-inf")); Vp = pad_to(Vf, S, 1, 0.0)
        ch = sp.shape[1] // S
        perm = torch.randperm(S, generator=gen).to(s.device)
        Vr = Vp.view(NKV, S, ch, D)
        def pieces(p, bf):
            pb = p.to(torch.bfloat16).float() if bf else p
            o_s = torch.einsum("kgsc,kscd->kgsd", pb.view(NKV, 2, S, ch), Vr).reshape(NH, S, D)
            return o_s
        # local-max baseline
        sv = sp.view(NH, S, ch)
        ms = sv.amax(2, keepdim=True); ms0 = torch.where(torch.isfinite(ms), ms, torch.zeros_like(ms))
        pl = flush(torch.exp2(sv - ms0)); l_s = pl.sum(2)
        M = ms0[:, :, 0].amax(1, keepdim=True); w = flush(torch.exp2(ms0[:, :, 0] - M))
        for mi, bf in enumerate((False, True)):
            o_s = pieces(pl.view(NH, -1), bf)
            out = (w[:, :, None] * o_s).sum(1) / (w * l_s).sum(1)[:, None]
            lerr[si_ * 2 + mi] = float(((out.double() - ref).abs().amax(1) / rs).amax())
        for ci, name in enumerate(CANDS):
            mref = mrefs[name]
            with torch.no_grad():
                p = flush(torch.exp2(sp - mref[:, None]))                    # fp32, flush emulates ex2.approx.ftz
                l_s = p.view(NH, S, ch).sum(2)
                for mi, bf in enumerate((False, True)):
                    o_s = pieces(p, bf)
                    o = o_s[:, perm].cumsum(1)[:, -1]; l = l_s[:, perm].cumsum(1)[:, -1]
                    f = 0
                    if (not torch.isfinite(o).all()) or (not torch.isfinite(l).all()): f |= 1
                    if (l < TINY).any(): f |= 2
                    if (not torch.isfinite(o_s).all()) or (not torch.isfinite(l_s).all()): f |= 4
                    out = o / l[:, None]
                    e = ((out.double() - ref).abs().amax(1) / rs)
                    e = torch.where(torch.isfinite(e), e, torch.full_like(e, float("inf")))
                    ei = si_ * 2 + mi
                    err[ci, ei] = float(e.max()); flg[ci, ei] = f
    return err, flg, lerr


# ---------------------------------------------------------------- run
FIELDS = ["mx", "mn", "sec", "p999", "argmax", "s0", "lL", "lO", "lOn", "sink_mass", "g_hi", "g_lo", "vmax", "qn", "knmax", "knmax_layer"]


def run(a):
    import torch
    import transformers.models.qwen3.modeling_qwen3 as mq
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    lengths = [1024, 4096] if a.quick else [1024, 4096, 16384, 32768]
    prompts = build_prompts(tok, lengths, a.quick)
    print("prompts:", [(p["name"], len(p["ids"])) for p in prompts], flush=True)
    qs = [None] * NL; ctr = [0]
    orig = mq.apply_rotary_pos_emb
    def patched(q, k, cos, sin, *args, **kw):
        qo, ko = orig(q, k, cos, sin, *args, **kw)
        qs[ctr[0] % NL] = qo[0, :, -1, :].detach(); ctr[0] += 1
        return qo, ko
    mq.apply_rotary_pos_emb = patched
    order = sorted(range(len(prompts)), key=lambda i: (prompts[i]["split"] != "cal", i))
    recs = {}  # idx -> dict
    knorm_tab = np.zeros((NL, NKV)); bmax = np.full((NL, NH), -1e30)
    gen = torch.Generator(); gen.manual_seed(5)
    t0 = time.time()
    for pi in order:
        P = prompts[pi]
        ids = torch.tensor([P["ids"]], device=dev)
        R = {f: np.zeros((STEPS + 1, NL, NH), np.float32) for f in FIELDS}
        R["err"] = np.full((STEPS + 1, NL, len(CANDS), len(CFGS)), np.nan, np.float32)
        R["flg"] = np.zeros((STEPS + 1, NL, len(CANDS), len(CFGS)), np.int32)
        R["lerr"] = np.full((STEPS + 1, NL, len(LOCAL)), np.nan, np.float32)
        R["gap_c"] = np.full((STEPS + 1, NL, NH), np.nan, np.float32)
        R["mref"] = np.full((STEPS + 1, NL, len(CANDS), NH), np.nan, np.float32)
        R["knk"] = np.zeros((STEPS + 1, NL, NKV), np.float32)
        prev_mx = None
        with torch.no_grad():
            ctr[0] = 0
            out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            for step in range(-1, STEPS):
                if step >= 0:
                    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                    ctr[0] = 0
                    out = model(input_ids=nxt, past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache = out.past_key_values
                ri = step + 1
                cur_mx = torch.zeros(NL, NH, device=dev)
                for l in range(NL):
                    K = cache.layers[l].keys[0]; V = cache.layers[l].values[0]
                    r, s, qsc, knk = stats(qs[l], K, V)
                    for f in FIELDS: R[f][ri, l] = r[f].cpu().numpy()
                    R["knk"][ri, l] = knk.cpu().numpy()
                    cur_mx[l] = r["mx"]
                    if P["split"] == "cal":
                        knorm_tab[l] = np.maximum(knorm_tab[l], knk.cpu().numpy())
                        bmax[l] = np.maximum(bmax[l], r["mx"].cpu().numpy())
                    else:
                        mx = r["mx"]
                        qn = r["qn"]
                        mrefs = {"oracle": mx, "a_norm": qn * r["knmax"], "e1_norm_seqmax": qn * r["knmax_layer"],
                                 "e2_norm_static": qn * torch.tensor(knorm_tab[l], device=dev, dtype=torch.float32).repeat_interleave(2),
                                 "b_static0": torch.tensor(bmax[l], device=dev, dtype=torch.float32),
                                 "b_static8": torch.tensor(bmax[l] + 8, device=dev, dtype=torch.float32),
                                 "c_prev0": prev_mx[l] if prev_mx is not None else mx, "c_prev8": (prev_mx[l] + 8) if prev_mx is not None else mx,
                                 "d_sink0": r["s0"], "d_sink8": r["s0"] + 8}
                        for ci, c in enumerate(CANDS): R["mref"][ri, l, ci] = mrefs[c].cpu().numpy()
                        e, fl, le = numerics(s, V, mx, mrefs, gen)
                        R["err"][ri, l] = e; R["flg"][ri, l] = fl; R["lerr"][ri, l] = le
                prev_mx = cur_mx
        recs[pi] = R
        print(f"{P['name']:16s} done  t={time.time() - t0:.0f}s  maxmem={torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
        del cache, out; torch.cuda.empty_cache()
    os.makedirs(a.out, exist_ok=True)
    idx = sorted(recs)
    arrs = {f: np.stack([recs[i][f] for i in idx]) for f in FIELDS + ["err", "flg", "lerr", "mref", "knk"]}
    arrs["b_table"] = bmax; arrs["k_table"] = knorm_tab
    arrs["seq_L"] = np.array([prompts[i]["L"] for i in idx]); arrs["seq_split"] = np.array([prompts[i]["split"] for i in idx]); arrs["seq_name"] = np.array([prompts[i]["name"] for i in idx])
    arrs["seq_first_tok"] = np.array([prompts[i]["ids"][0] for i in idx])
    np.savez_compressed(os.path.join(a.out, "records.npz"), **arrs)
    json.dump({"cands": CANDS, "cfgs": CFGS, "local": LOCAL, "seqs": [{k: prompts[i][k] for k in ("name", "kind", "split", "L")} | {"ntok": len(prompts[i]["ids"]), "first_tok": tok.decode(prompts[i]["ids"][:1])} for i in idx]},
              open(os.path.join(a.out, "meta.json"), "w"), indent=1)


# ---------------------------------------------------------------- analyze
def pct(x):
    x = x[np.isfinite(x)]
    return [float(np.min(x)), float(np.percentile(x, 0.1)), float(np.percentile(x, 50)), float(np.percentile(x, 99.9)), float(np.max(x))]


def analyze(a):
    Z = np.load(os.path.join(a.out, "records.npz"))
    meta = json.load(open(os.path.join(a.out, "meta.json")))
    test = Z["seq_split"] == "test"
    cal = ~test
    g = {}
    def sel(f, m): return Z[f][m]          # [seq, 9, 28, 16]
    mx = Z["mx"]
    qn = Z["qn"]
    # candidate gaps per (seq, ri, layer, head); NaN where undefined
    kt = np.repeat(Z["k_table"], 2, axis=1)                                    # [28,16]
    gaps = {"a_norm": qn * Z["knmax"] - mx, "e1_norm_seqmax": qn * Z["knmax_layer"] - mx,
            "e2_norm_static": qn * kt[None, None] - mx,
            "b_static0": Z["b_table"][None, None] - mx, "b_static8": Z["b_table"][None, None] + 8 - mx,
            "c_prev0": np.full_like(mx, np.nan), "d_sink0": Z["s0"] - mx, "d_sink8": Z["s0"] + 8 - mx}
    gaps["c_prev0"][:, 1:] = mx[:, :-1] - mx[:, 1:]
    gaps["c_prev8"] = gaps["c_prev0"] + 8
    gl, gh = Z["g_lo"], Z["g_hi"]
    rep = []; S = {"meta": meta}
    def P(*x): rep.append(" ".join(str(i) for i in x))
    P("# mref_real: R1 fixed-reference accumulation on real Qwen3-0.6B (test split unless stated)")
    P(f"sequences: {len(meta['seqs'])} ({int(cal.sum())} cal, {int(test.sum())} test), 9 records each (prefill last token + 8 decode steps), 28 layers x 16 heads")
    for sq, nm in zip(meta["seqs"], Z["seq_name"]): P(f"  {sq['name']:16s} ntok={sq['ntok']} first token {sq['first_tok']!r}")
    # 1 spreads
    P("\n## 1. score spreads (log2 units), test split; per layer: median / max over (seq, step, head)")
    P("layer  maxscore(med,max)  max-min(med,max)  max-2nd(med,max)  max-p99.9(med,max)  sink=argmax frac  sink mass(med)  log2 sum p(med,max)  log2 sum p|v|(max)  max|v|(max)")
    S["spread"] = []
    for l in range(NL):
        m = Z["mx"][test][:, :, l]; rngs = (Z["mx"] - Z["mn"])[test][:, :, l]; d2 = (Z["mx"] - Z["sec"])[test][:, :, l]; d3 = (Z["mx"] - Z["p999"])[test][:, :, l]
        am = (Z["argmax"][test][:, :, l] == 0).mean(); sm = np.median(Z["sink_mass"][test][:, :, l])
        row = dict(layer=l, mx_med=float(np.median(m)), mx_max=float(m.max()), rng_med=float(np.median(rngs)), rng_max=float(rngs.max()), d2_med=float(np.median(d2)), d2_max=float(d2.max()),
                   d3_med=float(np.median(d3)), d3_max=float(d3.max()), sink_frac=float(am), sink_mass_med=float(sm), lL_med=float(np.median(Z["lL"][test][:, :, l])), lL_max=float(Z["lL"][test][:, :, l].max()),
                   lO_max=float(Z["lO"][test][:, :, l].max()), vmax_max=float(Z["vmax"][test][:, :, l].max()))
        S["spread"].append(row)
        P(f"{l:3d}  {row['mx_med']:7.1f} {row['mx_max']:7.1f}  {row['rng_med']:7.1f} {row['rng_max']:7.1f}  {row['d2_med']:6.1f} {row['d2_max']:6.1f}  {row['d3_med']:6.1f} {row['d3_max']:6.1f}   {am:6.3f}  {sm:8.3f}   {row['lL_med']:5.1f} {row['lL_max']:5.1f}   {row['lO_max']:6.1f}   {row['vmax_max']:7.1f}")
    # 2 window
    P("\n## 2. safe window per record: g must satisfy g_lo < g <= g_hi (g = m_ref - true_max); test split")
    wl = gl[test]; wh = gh[test]
    P("layer: g_lo (max over records)   g_hi (min over records, tol 1e-6)")
    S["window"] = []
    for l in range(NL):
        S["window"].append(dict(layer=l, g_lo_max=float(wl[:, :, l].max()), g_hi_min=float(wh[:, :, l].min()), g_hi_p50=float(np.median(wh[:, :, l]))))
        P(f"{l:3d}: {wl[:, :, l].max():7.1f}   {wh[:, :, l].min():7.1f}  (median g_hi {np.median(wh[:, :, l]):6.1f})")
    P(f"overall: g_lo max {wl.max():.1f}, g_hi min {wh.min():.1f}")
    # 3 gaps
    S["gaps"] = {}
    for c in ["a_norm", "e1_norm_seqmax", "e2_norm_static", "b_static0", "c_prev0", "d_sink0"]:
        P(f"\n## gap g = m_ref - true_max for {c} (test split): per layer min / p0.1 / p50 / p99.9 / max ; margin_lo = min(g - g_lo), margin_hi = min(g_hi - g)")
        S["gaps"][c] = []
        gt = gaps[c][test]
        for l in range(NL):
            x = gt[:, :, l]; ok = np.isfinite(x)
            q = pct(x)
            mlo = float(np.min((x - wl[:, :, l])[ok])); mhi = float(np.min((wh[:, :, l] - x)[ok]))
            S["gaps"][c].append(dict(layer=l, pct=q, margin_lo=mlo, margin_hi=mhi))
            P(f"{l:3d}: " + " ".join(f"{v:8.1f}" for v in q) + f"   margin_lo {mlo:8.1f}  margin_hi {mhi:8.1f}")
    # 4 margins / fraction inside, all candidates, margins sweep
    P("\n## 3. two-sided margin per candidate (test split): m = min over records of min(g - g_lo, g_hi - g); fraction of records outside the window, vs extra margin delta added to m_ref")
    S["margin"] = {}
    for c in ["a_norm", "e1_norm_seqmax", "e2_norm_static", "b_static0", "c_prev0", "d_sink0"]:
        x = gaps[c][test]; ok = np.isfinite(x)
        row = {}
        for dl in (-8, 0, 4, 8, 16, 32):
            out = ((x + dl <= wl) | (x + dl > wh)) & ok
            row[str(dl)] = float(out.sum() / ok.sum())
        m = float(np.min(np.where(ok, np.minimum(x - wl, wh - x), np.inf)))
        S["margin"][c] = dict(min_margin=m, outside_frac_by_delta=row, best_delta=None)
        # best delta maximizing the worst-case margin
        best = max(np.arange(-64, 128, 1.0), key=lambda dl: float(np.min(np.where(ok, np.minimum(x + dl - wl, wh - x - dl), np.inf))))
        S["margin"][c]["best_delta"] = float(best)
        S["margin"][c]["margin_at_best_delta"] = float(np.min(np.where(ok, np.minimum(x + best - wl, wh - x - best), np.inf)))
        P(f"{c:16s} min margin {m:8.1f}  outside-window frac by delta {row}  best delta {best:+.0f} -> margin {S['margin'][c]['margin_at_best_delta']:.1f}")
    # b generalisation: also in-sample on cal
    xb = gaps["b_static0"][cal]
    if xb.size:
        P(f"b_static0 in-sample (cal): g min {np.nanmin(xb):.1f} (negative = true max exceeded the constant), g max {np.nanmax(xb):.1f}; test: g min {np.nanmin(gaps['b_static0'][test]):.1f}")
        by_L = {}
        for L in sorted(set(Z["seq_L"].tolist())):
            mm = test & (Z["seq_L"] == L); by_L[L] = [float(np.nanmin(gaps['b_static0'][mm])), float(np.nanmax(gaps['b_static0'][mm]))]
            P(f"  b_static0 test kv {L}: g range {by_L[L][0]:.1f} .. {by_L[L][1]:.1f}")
        S["b_by_L"] = by_L
    # worst heads for a_norm
    ga = gaps["a_norm"][test]
    med = np.median(ga, axis=(0, 1)); mxg = ga.max(axis=(0, 1)); mng = ga.min(axis=(0, 1))
    flat = np.dstack(np.unravel_index(np.argsort(-mxg, axis=None), mxg.shape))[0][:12]
    P("\n## worst (layer, head) for (a): largest max gap (and median / min)")
    S["worst_a"] = []
    for l, h in flat:
        P(f"  L{l:2d} H{h:2d}: max {mxg[l, h]:7.1f} median {med[l, h]:7.1f} min {mng[l, h]:7.1f}  (g_hi min {wh[:, :, l, h].min():.1f})")
        S["worst_a"].append(dict(layer=int(l), head=int(h), gmax=float(mxg[l, h]), gmed=float(med[l, h]), gmin=float(mng[l, h])))
    P(f"\nper-layer (a): median over heads of median gap: " + " ".join(f"{v:.0f}" for v in np.median(med, axis=1)))
    # 5 errors
    P("\n## 4. numerics (test split), max over (seq, step, head) of max_d |out-ref| / max_d |ref|, per layer; inf = nonfinite / l underflow")
    err = Z["err"][test]; flg = Z["flg"][test]; lerr = Z["lerr"][test]
    S["err"] = {}
    P("local-max split-K (fp32 / bf16 p) per config max over everything: " + ", ".join(f"{LOCAL[i]} {np.nanmax(lerr[..., i]):.2e}" for i in range(len(LOCAL))))
    S["err"]["local"] = {LOCAL[i]: float(np.nanmax(lerr[..., i])) for i in range(len(LOCAL))}
    P("(median over (seq, step, layer) of the per-row max error:)  local: " + ", ".join(f"{LOCAL[i]} {np.nanmedian(lerr[..., i]):.2e}" for i in range(len(LOCAL))))
    P("  " + "  ".join(f"{c}: " + "/".join(f"{np.nanmedian(err[:, (1 if c.startswith('c_') else 0):, :, CANDS.index(c), j]):.1e}" for j in range(len(CFGS))) for c in CANDS if c not in ("a_norm", "e1_norm_seqmax", "e2_norm_static")) + "  (cfg order " + ",".join(CFGS) + ")")
    hdr = "cand".ljust(16) + " " + " ".join(c.rjust(10) for c in CFGS) + "   flag records (nonfinite/l<tiny/piece-overflow)"
    P(hdr)
    for ci, c in enumerate(CANDS):
        e = err[:, 1:, :, ci, :]                                   # exclude prefill-last record for c? keep all but c needs step>=0
        e = err[:, :, :, ci, :] if c != "c_prev0" and c != "c_prev8" else err[:, 1:, :, ci, :]
        f = flg[:, :, :, ci, :] if c not in ("c_prev0", "c_prev8") else flg[:, 1:, :, ci, :]
        S["err"][c] = {CFGS[j]: float(np.nanmax(e[..., j])) for j in range(len(CFGS))}
        S["err"][c]["flags"] = [int(((f & b) > 0).sum()) for b in (1, 2, 4)]
        P(c.ljust(16) + " " + " ".join(f"{np.nanmax(e[..., j]):10.2e}" for j in range(len(CFGS))) + "   " + str(S["err"][c]["flags"]))
    P("\nper-layer max error, fp32-p S128, selected candidates (local-max S128 fp32 for comparison):")
    j = CFGS.index("S128_fp32"); jb = CFGS.index("S128_bf16")
    P("layer " + " ".join(c[:10].rjust(11) for c in ["local", "oracle", "a_norm", "e1_norm_seqmax", "e2_norm_static", "b_static0", "c_prev0", "d_sink0"]))
    S["err_layer"] = []
    for l in range(NL):
        vals = [np.nanmax(lerr[:, :, l, LOCAL.index("local_S128_fp32")])] + [np.nanmax((err[:, 1:, l] if c.startswith("c_") else err[:, :, l])[..., CANDS.index(c), j]) for c in ["oracle", "a_norm", "e1_norm_seqmax", "e2_norm_static", "b_static0", "c_prev0", "d_sink0"]]
        S["err_layer"].append(vals)
        P(f"{l:3d}   " + " ".join(f"{v:11.2e}" for v in vals))
    # sink / BOS
    P("\n## attention sink: fraction of records whose argmax is position 0, by sequence kind / first token (test)")
    for sq, am, nm in zip(meta["seqs"], Z["argmax"], Z["seq_name"]):
        if sq["split"] != "test": continue
        P(f"  {sq['name']:16s} first token {sq['first_tok']!r:16s} argmax==0 frac {np.mean(am == 0):.3f}  sink mass median {np.median(Z['sink_mass'][list(Z['seq_name']).index(nm)]):.3f}")
    open(os.path.join(a.out, "report.txt"), "w").write("\n".join(rep) + "\n")
    json.dump(S, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    print("\n".join(rep))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "analyze"]); ap.add_argument("--out", required=True); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    {"run": run, "analyze": analyze}[a.cmd](a)
