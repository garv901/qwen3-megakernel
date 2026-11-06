"""kvq: work-assignment arms (static / static_il / queue / queue_pf / hybrid) on mk's real KV access pattern (TMA ring over [L, rows, 8, 128] bf16, 2 KB row stride). See kvq.cu.
Usage (repo root): $PY probe/mk_core/kvq_pf/kvq.py --points b8kv32768 --arms static [--swiz 0|1] [--chunks 128,256,..] [--orders pos,pair] [--check] [--units-rec] [--compile-only]
Calibration: a1ub `ld` build, b8kv32768: A_med 156.864 us, imb_med 4.352 us (results/2026-10-27-attn-ub/ub_direct-ub1_s2.jsonl).
Metrics (medians over launches x layers 1..NL-1): A = block-0 stamp after the post-attend barrier - stamp after the pre-attend barrier (a1ub A); W per CTA = attend end - that stamp0;
imb = max W - median W; spread = max - min of CTA attend ends; tail = A - median W. S_max = most pieces of one (seq, kvh) pair.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from regimebench.kernels.common.splitkv import plan as skv_plan            # noqa: E402
from regimebench.kernels.hv2g.coop import CoopKernel, Args, check as cu_check, i32, nvrtc_compile, parse_ptxas   # noqa: E402

NCTA = 148
NL = 4
DYN_SMEM = 165248
KVH = 8
GRAN = 16
DATA = "randn"
FLUSH = "read"           # L2 flush before every launch: "read" = clean 512 MB read, "write" = dirty fill (a1ub), "none". A dirty flush leaves dirty lines the evict_first KV loads never evict: +12 us at b8kv32768 (jobs 11663 vs 11666)
FILL_MB = 31.5          # a1ub F phase: cold weight read before each layer (0 = off)
MERGE_MIN = 128          # a pair-end remainder shorter than this joins the previous piece
POINTS = {"b8kv4096": [4096] * 8, "b8kv16384": [16384] * 8, "b8kv32768": [32768] * 8, "b8kv30720": [30720] * 8, "b1kv245760": [245760], "b1kv32768": [32768],
          "b64kv1024": [1024] * 64, "b128kv512": [512] * 128, "ragged8": [1024, 2048, 4096, 8192, 16384, 24576, 32768, 65536]}
rs_ = np.random.RandomState(0)
POINTS["ragged64"] = [int(x) for x in np.exp(rs_.uniform(math.log(128), math.log(32768), 64))]
ARMS = ("static", "static_il", "queue", "queue_pf", "hybrid", "fs")
PS, PBF, EOI, TL = 132, 0, 3, 0          # set from --define (KVQ_PS / KVQ_PBF / KVQ_EOI / KVQ_TL)
CMODE, CK = "head", 1                  # combine: "head" (mk) or "wide" with CK slices per row
TLN = 512


# ------------------------------------------------------------------------------------------------ unit construction (host)
def cut_pair(pr, vis, C):
    """pieces (pr, p0, p1) of one pair, C positions each; a remainder < MERGE_MIN joins the previous piece."""
    out, p = [], 0
    while p < vis:
        out.append([pr, p, min(vis, p + C)])
        p += C
    if len(out) > 1 and out[-1][2] - out[-1][1] < MERGE_MIN:
        r = out.pop()
        out[-1][2] = r[2]
    return [tuple(x) for x in out]


def csize(v, C, K):
    """piece size of a pair of v positions: C, or with a cap K on the pieces per pair max(C, roundup128(ceil(v / K))) (so S <= K)."""
    return max(C, -(-(-(-v // K)) // 128) * 128) if K else C


def pieces(vis_of_pair, C, order, K=0):
    per = [cut_pair(pr, v, csize(v, C, K)) for pr, v in enumerate(vis_of_pair)]
    if order == "pair":
        return [x for pl in per for x in pl]
    out = []
    for j in range(max(len(pl) for pl in per)):
        out += [pl[j] for pl in per if j < len(pl)]
    return out


def slice_flat(plist, n):
    """cut the flat position axis of plist into n slices of equal length (+- GRAN); a slice spanning pieces yields one unit per piece."""
    T = sum(p1 - p0 for _, p0, p1 in plist)
    bnd = [(i * T // n) // GRAN * GRAN for i in range(n)] + [T]
    out = [[] for _ in range(n)]
    i, cum = 0, 0
    for pr, p0, p1 in plist:
        a = cum
        cum += p1 - p0
        while i < n and bnd[i + 1] <= a:
            i += 1
        j = i
        while j < n and bnd[j] < cum:
            lo, hi = max(bnd[j], a), min(bnd[j + 1], cum)
            if hi > lo:
                out[j].append((pr, p0 + lo - a, p0 + hi - a))
            j += 1
    return out


def round_up(x, m):
    return max(m, -(-x // m) * m)


class Work:
    """units (flat list of (row_base, kvh, p0, p1)), per-CTA static ranges, queue range."""

    def __init__(self, units, sbeg, scnt, qbeg, qcnt, pf, s_max, pairs_of_unit):
        self.units, self.sbeg, self.scnt, self.qbeg, self.qcnt, self.pf, self.s_max = units, sbeg, scnt, qbeg, qcnt, pf, s_max
        self.pairs = pairs_of_unit


def build_work(arm, pt, order, C, F=0.9, ctail=256, plan_obj=None, K=0):
    kvs = POINTS[pt]
    vis = [k + 1 for k in kvs]
    rb = [int(x) for x in np.concatenate([[0], np.cumsum(vis)[:-1]])]
    pair_vis = [vis[s] for s in range(len(vis)) for _ in range(KVH)]            # pair = seq * 8 + kvh
    def rec(pr, p0, p1):
        s, h = divmod(pr, KVH)
        return (rb[s], h, p0, p1)
    if arm == "static":
        pl = plan_obj
        flat = pl.flat_items
        units = [(int(r[0]), int(r[1]) & 15, int(r[2]), int(r[3])) for r in flat]
        pairs = [((int(r[1]) >> 12) * KVH + (int(r[1]) & 15)) for r in flat]
        scnt = [int(x) for x in pl.n_items_per_cta]
        sbeg = [int(x) for x in pl.item_off[:-1]]
        return Work(units, sbeg, scnt, 0, 0, 0, pl.s_max, pairs)
    if arm == "fs":                                                            # D2: every pair cut into exactly S = C equal pieces, static round-robin assignment (combine-only study)
        S = C
        per = [[] for _ in range(NCTA)]
        units, pairs = [], []
        u = 0
        for pr, v in enumerate(pair_vis):
            for j in range(S):
                p0, p1 = j * v // S, (j + 1) * v // S
                if p1 > p0:
                    per[u % NCTA].append((pr, p0, p1)); u += 1
        sbeg, scnt = [], []
        for lst in per:
            sbeg.append(len(units)); scnt.append(len(lst))
            for pr, p0, p1 in lst:
                units.append(rec(pr, p0, p1)); pairs.append(pr)
        return Work(units, sbeg, scnt, 0, 0, 0, S, pairs)
    pl_c = pieces(pair_vis, C, order, K)
    def finish(per_cta, tail, pf):
        units, pairs, sbeg, scnt = [], [], [], []
        for lst in per_cta:
            sbeg.append(len(units)); scnt.append(len(lst))
            for pr, p0, p1 in lst:
                units.append(rec(pr, p0, p1)); pairs.append(pr)
        qbeg = len(units)
        for pr, p0, p1 in tail:
            units.append(rec(pr, p0, p1)); pairs.append(pr)
        cnt = {}
        for pr in pairs:
            cnt[pr] = cnt.get(pr, 0) + 1
        return Work(units, sbeg, scnt, qbeg, len(tail), pf, max(cnt.values()), pairs)
    def rr(slices):
        per = [[] for _ in range(NCTA)]
        for u, s in enumerate(slices):
            per[u % NCTA] += s
        return per
    if arm == "static_il":
        n = round_up(len(pl_c), NCTA)
        return finish(rr(slice_flat(pl_c, n)), [], 0)
    if arm in ("queue", "queue_pf"):
        return finish([[] for _ in range(NCTA)], pl_c, int(arm == "queue_pf"))
    if arm == "hybrid":
        T = sum(p1 - p0 for _, p0, p1 in pl_c)
        tpre = int(F * T) // GRAN * GRAN
        pre, rest, cum = [], [], 0
        for pr, p0, p1 in pl_c:
            ln = p1 - p0
            if cum + ln <= tpre:
                pre.append((pr, p0, p1))
            elif cum >= tpre:
                rest.append((pr, p0, p1))
            else:
                c = p0 + (tpre - cum)
                pre.append((pr, p0, c)); rest.append((pr, c, p1))
            cum += ln
        Ch = max([C] + [csize(v, C, K) for v in pair_vis]) if K else C       # S cap: the slice length is at least the largest capped piece (the slices cut the pieces further)
        n = round_up(-(-sum(p1 - p0 for _, p0, p1 in pre) // Ch), NCTA)
        tail = []
        for pr, p0, p1 in rest:
            for q in cut_pair(0, p1 - p0, csize(pair_vis[pr], ctail, K)):
                tail.append((pr, p0 + q[1], p0 + q[2]))
        return finish(rr(slice_flat(pre, n)), tail, 1)
    raise ValueError(arm)


def coverage_ok(work, pt):
    """every pair tiled exactly once over [0, vis): returns (ok, sum positions)."""
    vis = [k + 1 for k in POINTS[pt]]
    rb = [int(x) for x in np.concatenate([[0], np.cumsum(vis)[:-1]])]
    by = {}
    for (r, h, p0, p1), pr in zip(work.units, work.pairs):
        if p1 <= p0:
            return False, 0
        by.setdefault(pr, []).append((p0, p1))
        if r != rb[pr // KVH] or h != pr % KVH:
            return False, 0
    tot = 0
    for pr in range(len(vis) * KVH):
        iv = sorted(by.get(pr, []))
        p = 0
        for a, b in iv:
            if a != p:
                return False, 0
            p = b
        if p != vis[pr // KVH]:
            return False, 0
        tot += p
    return True, tot


def comb_info(w, pt):
    """mk's partial bookkeeping from the unit list: per pair S = its units, part_base = cumulative S (pair = seq * 8 + kvh), slot of a unit = part_base + rank by p0.
    -> uinfo int32 [n_units, 4] {slot, S, tok, kvh}, pair_info int32 [ntok * 8, 4] {part_base, S, tok0, vis_len}, n_slots."""
    nt = len(POINTS[pt])
    by = {}
    for u, ((r, h, p0, p1), pr) in enumerate(zip(w.units, w.pairs)):
        by.setdefault(pr, []).append((p0, u))
    uinfo = np.zeros((len(w.units), 4), np.int32)
    pinfo = np.zeros((nt * KVH, 4), np.int32)
    base = 0
    for pr in range(nt * KVH):
        lst = sorted(by[pr])
        tok, kvh = divmod(pr, KVH)
        pinfo[pr] = (base, len(lst), tok, POINTS[pt][tok] + 1)
        for k, (_, u) in enumerate(lst):
            uinfo[u] = (base + k, len(lst), tok, kvh)
        base += len(lst)
    return uinfo, pinfo, base


def npad_of(n):
    return next(x for x in (16, 32, 64, 128, 256) if n <= x)


def host_merge(m, l, o):
    """numpy merge of S partials (m, l [S], o [S, 128]) -> o / l, as merge_th."""
    M = m.max()
    w = np.exp2(m - M)
    return (w[:, None] * o).sum(0) / (w * l).sum()


def bf16_round(x):
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    u = ((u + 0x7fff + ((u >> 16) & 1)) >> 16) << 16
    return u.astype(np.uint32).view(np.float32)


# ------------------------------------------------------------------------------------------------ GPU side
class Bench:
    def __init__(self, swiz: int, defines=()):
        import torch
        from cuda.bindings import driver
        torch.cuda.init(); torch.zeros(1, device="cuda")
        self.torch, self.d = torch, driver
        self.swiz = swiz
        global PS, PBF, EOI, TL
        dd = dict(x.split('=') for x in defines)
        EOI, PBF, TL = int(dd.get('KVQ_EOI', 3)), int(dd.get('KVQ_PBF', 0)), int(dd.get('KVQ_TL', 0))
        PS = int(dd.get('KVQ_PS', 68 if PBF else 132))
        if PBF and 'KVQ_PS' not in dd:
            defines = list(defines) + ['KVQ_PS=68']
        self.defs = dd
        self.acc = int(dd.get('KVQ_ACC', 0))
        src = (Path(__file__).with_name("kvq.cu")).read_text()
        self.cubin, self.log = nvrtc_compile(src, [f"KVQ_SWIZ={swiz}"] + list(defines), arch="sm_100a", ptxas_verbose=True)
        self.ptxas = parse_ptxas(self.log)
        self.kernel = CoopKernel(self.cubin, "kvq", DYN_SMEM)
        self.nb = torch.cuda.get_device_properties(0).multi_processor_count
        assert self.nb == NCTA, self.nb
        self.flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
        self.pool = torch.ones(2 << 30, dtype=torch.uint8, device="cuda")
        self.pt = None

    def point(self, pt):
        """KV [NL * rows, 8, 128] bf16 (random bits), 2 * 8 TMA maps, host XOR reference per layer."""
        torch, d = self.torch, self.d
        if self.pt and self.pt[0] == pt:
            return self.pt
        self.pt = None
        torch.cuda.empty_cache()
        vis = [k + 1 for k in POINTS[pt]]
        rows = int(sum(vis))
        g = torch.Generator(device="cuda"); g.manual_seed(1234)
        if DATA == "bits":                                                      # uniformly random bits
            kc = torch.randint(-32768, 32768, (NL * rows, 8, 128), dtype=torch.int16, device="cuda", generator=g)
            vc = torch.randint(-32768, 32768, (NL * rows, 8, 128), dtype=torch.int16, device="cuda", generator=g)
        else:                                                                   # randn * 0.1 bf16, as a1ub
            kc = torch.empty(NL * rows, 8, 128, dtype=torch.bfloat16, device="cuda").normal_(0, 0.1, generator=g)
            vc = torch.empty_like(kc).normal_(0, 0.1, generator=g)
        blob = b""
        for t in (kc, vc):
            for rs in range(8):
                rr = 128 >> rs
                res = d.cuTensorMapEncodeTiled(
                    d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, int(t.data_ptr()),
                    [d.cuuint64_t(v) for v in (128, 8, NL * rows)], [d.cuuint64_t(v) for v in (256, 2048)],
                    [d.cuuint32_t(v) for v in (64 if self.swiz else 128, 1, rr)], [d.cuuint32_t(v) for v in (1, 1, 1)],
                    d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
                    d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B if self.swiz else d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
                    d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE, d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
                if int(res[0]) != 0:
                    raise RuntimeError(f"cuTensorMapEncodeTiled: {res[0]}")
                blob += struct.pack("<16Q", *[int(v) for v in res[1].opaque])
        maps = torch.frombuffer(bytearray(blob), dtype=torch.uint8).cuda()
        assert maps.data_ptr() % 64 == 0

        def xor_all(x):
            x = x.reshape(-1).view(torch.int32)
            while x.numel() > 1:
                if x.numel() & 1:
                    x = torch.cat([x, x.new_zeros(1)])
                h = x.numel() // 2
                x = x[:h] ^ x[h:]
            return int(x[0].item()) & 0xffffffff
        ref = [xor_all(kc[l * rows:(l + 1) * rows]) ^ xor_all(vc[l * rows:(l + 1) * rows]) for l in range(NL)]
        self.pt = (pt, rows, kc, vc, maps, ref)
        return self.pt

    def run(self, pt, w, launches, check=False, rec=False, tick=False):
        torch = self.torch
        _, rows, kc, vc, maps, ref = self.point(pt)
        nu = len(w.units)
        units = torch.tensor([list(u) for u in w.units], dtype=torch.int32, device="cuda").reshape(-1)
        sbeg = torch.tensor(w.sbeg, dtype=torch.int32, device="cuda"); scnt = torch.tensor(w.scnt, dtype=torch.int32, device="cuda")
        ctr = torch.zeros(NL, dtype=torch.int32, device="cuda"); bar = torch.zeros(4, dtype=torch.int32, device="cuda")
        stamps = torch.zeros(NL * 2, dtype=torch.int64, device="cuda")
        cstart = torch.zeros(NL * NCTA, dtype=torch.int64, device="cuda"); cend = torch.zeros(NL * NCTA, dtype=torch.int64, device="cuda")
        cstat = torch.zeros(NL * NCTA * 2, dtype=torch.int32, device="cuda"); chk = torch.zeros(NL, dtype=torch.int32, device="cuda")
        visited = torch.zeros(NL * nu if check else 1, dtype=torch.int32, device="cuda")
        urec = torch.zeros(NL * nu * 4 if rec else 4, dtype=torch.int64, device="cuda")
        sink = torch.zeros(NCTA * 512, dtype=torch.int32, device="cuda")
        uinfo_h, pinfo_h, nslot = comb_info(w, pt)
        ntok = len(POINTS[pt]); tpad = npad_of(ntok)
        comb = int(int(uinfo_h[:, 1].max()) > 1 and not tick) * (2 if CMODE == "wide" else 1)                   # mk: skv_comb = s_max > 1; the ticket variant has no combine phase
        uinfo = torch.from_numpy(uinfo_h.reshape(-1).copy()).cuda(); pinfo = torch.from_numpy(pinfo_h.reshape(-1).copy()).cuda()
        part = torch.full((nslot * 2 * PS,), float("nan"), dtype=torch.float32, device="cuda")
        ao = torch.full((ntok * 2048,), 0x7FC0, dtype=torch.int16, device="cuda")
        cout = torch.full((ntok * 16 * 128,), float("nan"), dtype=torch.float32, device="cuda")
        cend2 = torch.zeros(NL * NCTA, dtype=torch.int64, device="cuda")
        tl = torch.zeros(NCTA * 16 * TLN * 8 + NCTA * TLN * 8 if TL else 8, dtype=torch.int32, device="cuda")
        wscr = torch.zeros(ntok * 16 * max(CK, 1) * 136, dtype=torch.float32, device="cuda"); rowctr = torch.zeros(ntok * 16, dtype=torch.int32, device="cuda")
        accb = torch.zeros(2 * ntok * 16 * PS if self.acc else 4, dtype=torch.float32, device="cuda")
        tickets = torch.zeros(ntok * 8, dtype=torch.int32, device="cuda"); ulist = torch.zeros(NCTA * 512, dtype=torch.int32, device="cuda")

        fill_vec = int(FILL_MB * 1e6) // NCTA // 16
        fill_adv = (fill_vec * NCTA * 16 + 4095) & ~4095
        pool_mod = self.pool.numel() - fill_adv - 4096

        def launch(it):
            args = Args([maps.data_ptr(), units.data_ptr(), sbeg.data_ptr(), scnt.data_ptr(), i32(w.qbeg), i32(w.qcnt), ctr.data_ptr(), i32(w.pf), i32(NL), i32(rows), i32(nu),
                         bar.data_ptr(), stamps.data_ptr(), cstart.data_ptr(), cend.data_ptr(), cstat.data_ptr(), chk.data_ptr(), visited.data_ptr(), urec.data_ptr(),
                         i32(int(check)), i32(int(rec)), sink.data_ptr(), part.data_ptr(),
                         self.pool.data_ptr(), i32(fill_vec), ctypes.c_size_t(fill_adv), ctypes.c_size_t(pool_mod), ctypes.c_size_t(it * NL * fill_adv),
                         uinfo.data_ptr(), pinfo.data_ptr(), ao.data_ptr(), cout.data_ptr(), cend2.data_ptr(), i32(comb), i32(int(tick)), i32(ntok), i32(tpad), tickets.data_ptr(), ulist.data_ptr(),
                         tl.data_ptr(), i32(CK), wscr.data_ptr(), rowctr.data_ptr(), accb.data_ptr()])
            if FLUSH == "write":                                                # dirty 512 MB (a1ub's flush)
                self.flush.fill_(it & 255)
            elif FLUSH == "read":                                               # clean 512 MB
                self.flush.max()
            bar.zero_(); ctr.zero_(); chk.zero_(); tickets.zero_(); rowctr.zero_(); accb.zero_()
            torch.cuda.synchronize()
            self.kernel.launch(NCTA, args, torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
        if check:
            launch(0)
            vv = visited.cpu().view(NL, nu)
            got = [int(x) & 0xffffffff for x in chk.cpu().tolist()]
            out = {"visited_ok": bool((vv == 1).all()), "xor_ok": got == ref, "ctr_final": ctr.cpu().tolist()}
            out.update(self.check_comb(w, uinfo_h, pinfo_h, nslot, part, ao, cout, tickets, comb, tick, ntok))
            return out
        res = {"A": [], "C": [], "AC": [], "imb": [], "spread": [], "tail": [], "W": [], "Wmax": [], "skew": []}
        per_cta = []
        for it in range(launches + 1):
            launch(it)
            if it == 0:
                continue
            c2 = cend2.cpu().view(NL, NCTA); st = stamps.cpu().view(NL, 2); cs = cstart.cpu().view(NL, NCTA); ce = cend.cpu().view(NL, NCTA)
            for l in range(1, NL):
                A = float(st[l, 1] - st[l, 0]) / 1e3
                Cc = float(c2[l].max() - st[l, 1]) / 1e3
                res["C"].append(Cc); res["AC"].append(A + Cc)
                W = (ce[l].double() - float(st[l, 0])) / 1e3
                res["A"].append(A); res["W"].append(float(W.median())); res["Wmax"].append(float(W.max())); res["imb"].append(float(W.max() - W.median()))
                res["spread"].append(float(ce[l].max() - ce[l].min()) / 1e3); res["tail"].append(A - float(W.median()))
                res["skew"].append(float(cs[l].max() - st[l, 0]) / 1e3)
                per_cta.append(W.tolist())
        cs_ = cstat.cpu().view(NL, NCTA, 2).numpy().astype(np.int64)[1:]
        out = {k + "_med": statistics.median(v) for k, v in res.items()}
        out.update(comb=comb, tick=int(tick), s_max_dev=int(uinfo_h[:, 1].max()))
        srt = sorted(res["A"])
        out.update(A_p10=srt[int(0.1 * (len(srt) - 1) + .5)], A_p90=srt[int(0.9 * (len(srt) - 1) + .5)], A_min=srt[0], A_max=srt[-1], n_samples=len(srt),
                   cta_bytes_min=int(cs_[..., 0].min() * 512), cta_bytes_max=int(cs_[..., 0].max() * 512), cta_units_min=int(cs_[..., 1].min()), cta_units_max=int(cs_[..., 1].max()),
                   cta_work=[round(float(x), 2) for x in np.median(np.array(per_cta), axis=0)])
        self.last_tl = tl.cpu().numpy() if TL else None
        self.last_cstat = cstat.cpu().numpy()
        self.last_ts = (cstart.cpu().numpy(), cend.cpu().numpy(), stamps.cpu().numpy())
        if rec:
            out.update(urec_stats(urec.cpu().view(NL, nu, 4).numpy()))
        return out


    def check_comb(self, w, uinfo, pinfo, nslot, part, ao, cout, tickets, comb, tick, ntok):
        """the last layer's partials (m, l = the unit's synthetic values, o finite) and the device merge (cout fp32 / ao bf16 when there is no combine phase or a ticket merge) vs a numpy merge of the same partials."""
        pt_ = part.cpu().view(nslot, 2, PS).numpy()
        if PBF:                                                              # o bf16 in the first 256 B, m / l at float 64, 65
            raw = part.cpu().view(self.torch.int16).view(nslot, 2, 2 * PS)[:, :, :128].numpy().astype(np.int32)
            o_ = (raw.astype(np.uint32) << 16).view(np.float32)
            pt_ = np.concatenate([o_, pt_[:, :, 64:66]], axis=2)
        ok_p, ok_m = bool(np.isfinite(pt_[:, :, :130]).all()), True
        for u in range(len(uinfo)):
            slot = int(uinfo[u, 0])
            for hq in range(2):
                m = 0.1 * ((u * 37 + hq * 11) % 19) - 1.0; l = 1.0 + 0.25 * ((u + hq) % 7)
                if self.acc:                                                    # R1: the slot holds the piece against the fixed reference: m = M_REF, l scaled by 2^(m - M_REF)
                    l *= 2.0 ** (m - 1.0); m = 1.0
                ok_m &= abs(pt_[slot, hq, 128] - m) < 1e-5 and abs(pt_[slot, hq, 129] - l) < 1e-5
        ref = np.zeros((ntok, 16, 128), np.float32)
        for pr in range(ntok * KVH):
            b, S, tok, _ = (int(x) for x in pinfo[pr])
            for hq in range(2):
                sl = pt_[b:b + S, hq]
                ref[tok, 2 * (pr % KVH) + hq] = host_merge(sl[:, 128].astype(np.float64), sl[:, 129].astype(np.float64), sl[:, :128].astype(np.float64))
        if comb:
            got, tol = cout.cpu().view(ntok, 16, 128).numpy(), None
        else:
            got, tol = ao.cpu().view(self.torch.bfloat16).float().view(ntok, 16, 128).numpy(), 1
        sone = np.repeat(pinfo[:, 1] == 1, 1).reshape(ntok, KVH)             # S == 1 pairs: bf16 ao
        lowp = np.repeat(sone, 2, axis=1)[:, :, None] | (tol is not None) | bool(PBF)
        err = np.abs(got - ref)
        lim = np.where(lowp, 1e-2 * np.abs(ref) + 1e-3, 5e-4 * np.abs(ref) + 1e-4)
        ok_c = bool(np.isfinite(got).all() and (err <= lim).all())
        out = {"part_ok": ok_p, "ml_ok": bool(ok_m), "merge_ok": ok_c, "merge_max_rel": float((err / (np.abs(ref) + 1e-6)).max())}
        if tick:
            exp = np.array([S * NL if S > 1 else 0 for S in pinfo[:, 1]], np.int32)
            out["tickets_ok"] = bool((tickets.cpu().numpy() == exp).all())
        return out


def urec_stats(u):
    gaps, ivl = [], []
    for l in range(1, NL):
        d = u[l]
        for c in np.unique(d[:, 0]):
            m = d[d[:, 0] == c]
            m = m[np.argsort(m[:, 2])]
            gaps += list((m[1:, 2] - m[:-1, 3]).astype(float) / 1e3)
            ivl += [float(x) / 1e3 for x in (m[:, 3] - m[:, 2])]
    return {"unit_gap_med_us": statistics.median(gaps) if gaps else None, "unit_gap_p90_us": sorted(gaps)[int(.9 * (len(gaps) - 1))] if gaps else None,
            "unit_span_med_us": statistics.median(ivl) if ivl else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", default="b8kv32768"); ap.add_argument("--arms", default="static")
    ap.add_argument("--chunks", default="128,256,512,1024,2048", help="C in positions (one position = 512 B)")
    ap.add_argument("--orders", default="pos,pair"); ap.add_argument("--launches", type=int, default=10)
    ap.add_argument("--swiz", type=int, default=0); ap.add_argument("--check", action="store_true", help="correctness before timing; a FAIL skips that config")
    ap.add_argument("--check-only", action="store_true"); ap.add_argument("--units-rec", action="store_true"); ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--hybrid-f", type=float, default=0.9); ap.add_argument("--ctail", type=int, default=256)
    ap.add_argument("--caps", default="0", help="S cap K per pair (0 = none), comma list; static ignores it")
    ap.add_argument("--tick", default="0", help="0 / 1 comma list: 1 = SKV_TICKET merge in attend (needs S <= 32), no combine phase")
    ap.add_argument("--comb-mode", default="head", help="comma list: head (mk combine_head) | wide (combine_wide, --wide-k slices per row)"); ap.add_argument("--wide-k", default="1")
    ap.add_argument("--tl-out", default=None, help="KVQ_TL=1: save the last launch's clock64 timeline (npz) per config to this prefix")
    ap.add_argument("--plan-opt", action="append", default=[], metavar="K=V")
    ap.add_argument("--job", default="direct"); ap.add_argument("--out", default=None)
    ap.add_argument("--define", action="append", default=[]); ap.add_argument("--fill-mb", type=float, default=31.5); ap.add_argument("--flush", default="read", choices=["write", "read", "none"]); ap.add_argument("--data", default="randn", choices=["randn", "bits"]); ap.add_argument("--nl", type=int, default=4)
    a = ap.parse_args()
    global NL, DATA, FILL_MB, FLUSH, CMODE, CK
    NL, DATA, FILL_MB, FLUSH = a.nl, a.data, a.fill_mb, a.flush
    b = Bench(a.swiz, a.define)
    print(b.log)
    print("ptxas:", b.ptxas, f"regs {b.kernel.regs} local {b.kernel.local_bytes} dyn_smem {DYN_SMEM} swiz {a.swiz}")
    if a.compile_only:
        return
    out = open(a.out, "a") if a.out else None
    caps = [int(x) for x in a.caps.split(",")]; ticks = [int(x) for x in a.tick.split(",")]
    pts = a.points.split(","); arms = a.arms.split(","); chunks = [int(x) for x in a.chunks.split(",")]; orders = a.orders.split(",")
    popts = {}
    for it in a.plan_opt:
        k, _, v = it.partition("=")
        popts[k] = v
    nfail = 0
    cfgs = []
    for pt in pts:
        for arm in arms:
            for tk in ticks:
                if arm == "static":
                    cfgs.append((pt, arm, 0, "-", 0, tk))
                else:
                    cfgs += [(pt, arm, C, o, K, tk) for K in caps for C in chunks for o in orders]
    cms = [(m, int(k)) for m in a.comb_mode.split(",") for k in (a.wide_k.split(",") if m == "wide" else ["1"])]
    cfgs = [c + cm for c in cfgs for cm in cms]
    plans = {}
    for pt, arm, C, order, K, tk, cmode, ck in cfgs:
        CMODE, CK = cmode, ck
        if pt not in plans:
            plans[pt] = skv_plan.plan([(k, 1) for k in POINTS[pt]], n_cta=NCTA)
        w = build_work(arm, pt, order, C, a.hybrid_f, a.ctail, plans[pt], K)
        if tk and max(comb_info(w, pt)[0][:, 1]) > 32:
            continue                                                        # ticket_merge handles S <= 32
        ok, tot = coverage_ok(w, pt)
        row = {"point": pt, "arm": arm, "C": C, "order": order, "swiz": a.swiz, "job": a.job, "n_units": len(w.units), "s_max": w.s_max, "n_ctas_static": sum(1 for x in w.scnt if x),
               "qcnt": w.qcnt, "cap": K, "ticket": tk, "eoi": EOI, "ps": PS, "pbf": PBF, "mk": int(b.defs.get("KVQ_MK", 20)), "cmode": cmode, "ck": ck, "defs": a.define, "acc": b.acc, "rpf": int(b.defs.get("KVQ_RPF", 0)), "rd": int(b.defs.get("KVQ_RD", 2)), "rld": int(b.defs.get("KVQ_RLD", 0)), "host_cover_ok": ok, "positions": tot}
        if arm == "static":
            row["plan"] = plans[pt].summary() if hasattr(plans[pt], "summary") else {}
            row["no_split"] = plans[pt].no_split; row["one_item"] = plans[pt].one_item
        status = "PASS"
        if a.check or a.check_only:
            r = b.run(pt, w, 0, check=True, tick=bool(tk)) if ok else {"visited_ok": False, "xor_ok": False}
            row.update(check=r)
            full = EOI >= 3 or not any(r.get(k_) is not None for k_ in ("part_ok",))
            pm = (r.get("part_ok") and r.get("ml_ok") and r.get("merge_ok")) if EOI >= 3 else True      # EOI < 3 stores no partials: only coverage / visited / XOR are checkable
            status = "PASS" if ok and r["visited_ok"] and r["xor_ok"] and pm and r.get("tickets_ok", True) else "FAIL"
            print(f"CHECK {pt} {arm} C={C} {order} K={K} tk={tk}: {status} (cover {ok}, visited {r['visited_ok']}, xor {r['xor_ok']}, part {r.get('part_ok')}, ml {r.get('ml_ok')}, merge {r.get('merge_ok')} "
                  f"relmax {r.get('merge_max_rel', 0):.1e}, tickets {r.get('tickets_ok')}, units {len(w.units)})", flush=True)
            row["check_status"] = status
            if status == "FAIL":
                nfail += 1
        if status == "PASS" and not a.check_only:
            row.update(b.run(pt, w, a.launches, rec=a.units_rec, tick=bool(tk)))
            if a.tl_out and TL:
                np.savez_compressed(f"{a.tl_out}_{pt}_{arm}_C{C}_K{K}_eoi{EOI}.npz", tl=b.last_tl, cstat=b.last_cstat, cstart=b.last_ts[0], cend=b.last_ts[1], stamps=b.last_ts[2], nl=NL, nct=NCTA, tln=TLN)
            print(f"{pt:12s} {arm:9s} C={C:5d} {order:4s} K={K:3d} tk={tk} A {row['A_med']:8.2f} Cc {row['C_med']:6.2f} A+C {row['AC_med']:8.2f} imb {row['imb_med']:6.2f} tail {row['tail_med']:6.2f} spread {row['spread_med']:6.2f} "
                  f"B/CTA {row['cta_bytes_min']/1e6:.2f}-{row['cta_bytes_max']/1e6:.2f}MB u/CTA {row['cta_units_min']}-{row['cta_units_max']} units {len(w.units)} S_max {w.s_max}/{row['s_max_dev']}", flush=True)
            if arm == "static" and pt == "b8kv32768":
                print(f"CALIBRATION vs a1ub ld: A {row['A_med']:.2f} (156.864, {100*(row['A_med']/156.864-1):+.1f}%)  imb {row['imb_med']:.2f} (4.352, {100*(row['imb_med']/4.352-1):+.1f}%)")
        if out:
            out.write(json.dumps(row) + "\n"); out.flush()
    print(f"done; check FAIL count {nfail}")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
