"""Host planner for A1 (split-KV decode attention, interface v1). Pure Python + numpy; torch only in the helpers that take tensors.

Model: Qwen3-0.6B decode, q_len == 1 per sequence (q_len 2..16: see "q_len > 1" below), 16 q-heads, 8 kv-heads (GQA pair per item), head dim 128.
Cache layout [L, cache_rows, 8, 128]; sequence s owns rows row_base[s] .. row_base[s] + kv_len[s] + q_len[s] - 1 (the last q_len rows are the new tokens,
written by the caller before attention), so for q_len == 1 the visible positions are [0, kv_len + 1).

Work item = (seq, kvh, p0, p1, split). Items are cut so every CTA streams ~equal bytes: either one item per CTA (`one_item`, when there are
at most n_cta (seq, kvh) pairs and the cut is balanced to 1.10) or a greedy cut of the flat (seq, kvh) position axis.

Device encoding (matches splitkv.cuh):
  items = int32 [n_cta][stride][8], CTA b's records at items + b * stride * 8, run in order; record =
      {row_base, y = kvh | split << 4 | seq << 12, p0, p1, part_slot (= part_base + split), S, tok0, n_items of this CTA}
      kvh < 16, split < 256, seq < 2^19; n_items is the same in every valid record of the CTA; unused records are zeros
  item_off[b] .. item_off[b+1] = CTA b's items in the compact `flat_items` order (not read by attend)
  pair_info[(seq * 8 + kvh) * 4 + {0..3}] = {part_base, S, tok0, vis_len}      (partial slot of (pair, split) = part_base + split; combine only)
  q_len > 1 (R6, SKV_QMAX > 1 builds, interface v1 only): seqs = (kv_len, q_len), 1 <= q_len <= 16, sum q_len <= 256, seq < 4096. Sequence s owns rows
      row_base .. row_base + kv_len + q_len - 1 (the q_len new rows are written by the caller before attention); token t (global token row tok0 + t) sees
      positions p < vis_t = kv_len + t + 1; items cut [0, kv_len + q_len). y gets (q_len - 1) << 24. Partial slots are token-major:
      slot(pair, split, t) = part_base + t * S + split (the record's part_slot stays part_base + split, attend adds t * S). pair_info has one row per
      (token row, kvh) at index (tok0 + t) * 8 + kvh: {part_base + t * S, S, tok0, vis_t}; combine reads only words 0 and 1. engine is TMA if any q_len > 1.
  iface=2 (SKV_IFACE 2) changes two words of the record: word 5 = S | n_items << 8 (S < 256) and word 7 = vis (= kv_len + 1 of the sequence); the piece
      with p1 == vis owns the new row (position vis - 1). Everything else, pair_info and the partial layout are the same as iface 1.
  no_split (plan(no_split=...)): every (seq, kvh) pair is ONE item (S = 1, attend writes ao directly, no combine), pairs packed whole onto CTAs by LPT
      (longest first, to the least loaded CTA). "auto" (default): used iff n_pairs >= n_cta and the LPT max / mean CTA load <= LPT_MAX_IMB (1.20), else the cut above.
      Plan.no_split / summary()["no_split"] tell which mode was used. SKV_TICKET builds (splitkv.cuh) use `Plan.n_pairs` (tickets[pair], pair = seq * 8 + kvh, zeroed once per plan).
  partial layout (floats): part[((part_base + split) * 2 + hq) * 132 + {0..127 o, 128 m, 129 l, 130..131 pad}]
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

QH, KVH, D = 16, 8, 128
PSTRIDE = 132                      # floats per partial (o[128], m, l, pad): 528 B, 16 B aligned
POS_BYTES = 2 * D * 2              # K + V bytes of one position of one kv-head = 512 B
Y_KVH_BITS, Y_SPLIT_SHIFT, Y_SEQ_SHIFT, Y_QLEN_SHIFT = 4, 4, 12, 24


@dataclass
class Plan:
    n_cta: int
    gran: int
    seqs: list
    row_base: list                 # per seq
    cache_rows: int                # sum(kv_len + q_len)
    items: np.ndarray              # int32 [n_cta, stride, 8] item records (see module docstring)
    item_off: np.ndarray           # int32 [n_cta + 1]
    pair_info: np.ndarray          # int32 [total_q * 8, 4], row (token row) * 8 + kvh
    part_floats: int
    engine: int                    # 0 LDG, 1 TMA
    total_q: int
    cta_positions: list = field(default_factory=list)   # positions per CTA (bytes = * POS_BYTES)
    one_item: bool = False                               # True: one item per CTA (S_i-proportional cut)
    balance: dict = field(default_factory=dict)          # {"one_item": max/mean CTA positions or None (not applicable), "greedy": ..., "chosen": ...}
    iface: int = 1                                       # record format: 1 = n_items in word 7; 2 = S | n_items << 8 in word 5, vis in word 7
    tok0s: list = field(default_factory=list)            # first token row of each sequence
    no_split: bool = False                               # True: every pair is one item (S = 1), pairs packed whole by LPT

    @property
    def stride(self) -> int:
        return int(self.items.shape[1])

    @property
    def flat_items(self) -> np.ndarray:
        """Valid records only, CTA order: int32 [n_items, 8]."""
        n = self.n_items_per_cta
        return np.concatenate([self.items[b, :int(n[b])] for b in range(self.n_cta)]) if self.n_cta else self.items.reshape(0, 8)

    @property
    def n_items_per_cta(self) -> np.ndarray:
        """int32 [n_cta]: items of each CTA as stored in the records (word 7 in iface 1, word 5 >> 8 in iface 2)."""
        return (self.items[:, 0, 5] >> 8) if self.iface == 2 else self.items[:, 0, 7]

    @property
    def n_pairs(self) -> int:
        """SKV_TICKET: size of the u32 ticket array (tickets[pair], pair = seq * 8 + kvh); zero it once per plan."""
        return KVH * len(self.seqs)

    def cta_ticket_items(self, b: int) -> list:
        """SKV_TICKET host model of the device's end-of-CTA re-read: [(record index, pair = seq * 8 + kvh, S)] of CTA b's items with S > 1, in record order."""
        out = []
        for k in range(int(self.n_items_per_cta[b])):
            r = self.items[b, k]
            y, S = int(r[1]), int(r[5]) & 255 if self.iface == 2 else int(r[5])
            seq = (y >> Y_SEQ_SHIFT) & (0xFFF if any(q != 1 for _, q in self.seqs) else 0x7FFFF)
            if S > 1:
                out.append((k, seq * KVH + (y & 15), S))
        return out

    @property
    def s_max(self) -> int:
        return int(self.pair_info[:, 1].max()) if len(self.pair_info) else 0

    @property
    def max_items(self) -> int:
        return int(np.diff(self.item_off).max()) if self.n_cta else 0

    @property
    def cta_bytes(self) -> list:
        return [p * POS_BYTES for p in self.cta_positions]

    @property
    def part_bytes(self) -> int:
        return self.part_floats * 4

    @property
    def engine_name(self) -> str:
        return "tma" if self.engine else "ldg"

    def tok0(self, seq: int) -> int:
        return int(self.tok0s[seq]) if self.tok0s else int(self.pair_info[seq * KVH, 2])

    def to_torch(self, device="cuda"):
        """-> dict of int32 device tensors: items (flat int32 [n_cta * stride * 8]), item_off, pair_info (flat)."""
        import torch
        return {"items": torch.from_numpy(self.items.reshape(-1).copy()).to(device),
                "item_off": torch.from_numpy(self.item_off.copy()).to(device),
                "pair_info": torch.from_numpy(self.pair_info.reshape(-1).copy()).to(device)}

    def summary(self) -> dict:
        return {"engine": self.engine_name, "s_max": self.s_max, "items_per_cta_max": self.max_items, "part_bytes": self.part_bytes,
                "cta_bytes_max": max(self.cta_bytes), "cta_bytes_min": min(self.cta_bytes), "n_items": int(self.item_off[-1]),
                "one_item": self.one_item, "balance_one_item": self.balance.get("one_item"), "balance_greedy": self.balance.get("greedy"),
                "no_split": self.no_split, "balance_lpt": self.balance.get("lpt")}


def _cut(vis: list, n_cta: int, gran: int, min_target: int):
    """Greedy cut of the flat pair axis (pairs in (seq, kvh) order, lengths vis[seq]) into n_cta consecutive ranges.
    -> per-CTA list of (pair, p0, p1). Target per CTA = ceil(remaining / remaining CTAs), at least min_target; a piece shorter than gran/2 is
    never created (a short tail of a pair is absorbed by the CTA that cuts it, a short head piece is extended) except for whole short pairs."""
    pair_len = [vis[s] for s in range(len(vis)) for _ in range(KVH)]
    total = sum(pair_len)
    half = max(gran // 2, 1)
    pi, po, done = 0, 0, 0
    out = []
    for b in range(n_cta):
        left = max(-(-(total - done) // (n_cta - b)), min_target) if done < total else 0
        pieces = []
        while left > 0 and pi < len(pair_len):
            avail = pair_len[pi] - po
            take = min(avail, left)
            if avail <= half:
                take = avail                                  # short pair / short remainder: one piece
            elif take < avail:                                # this CTA cuts the pair: both sides of the cut must be >= half
                if take < half:
                    take = half if half - take <= take else 0   # head piece: extend or drop, whichever moves the cut less
                if take and avail - take < half:
                    take = avail if (avail - take) <= half - (avail - take) else avail - half   # tail piece: absorb or shrink
                if take == 0:
                    break
            pieces.append((pi, po, po + take))
            po += take
            done += take
            left -= take
            if po == pair_len[pi]:
                pi, po = pi + 1, 0
        out.append(pieces)
    assert pi == len(pair_len) and po == 0, "cut did not consume all positions"
    return out


def _one_item_cut(vis: list, n_cta: int, smax: int):
    """One item per CTA: pair i (length len_i) gets S_i = round(n_cta * len_i / total) >= 1 pieces (sum exactly n_cta, largest remainder), cut into
    near-equal pieces (the first len_i % S_i pieces are one position longer); CTAs in pair order. -> per-CTA [(pair, p0, p1)], or None if not applicable
    (more pairs than CTAs, a pair would need more than smax pieces or more pieces than positions)."""
    pair_len = [vis[s] for s in range(len(vis)) for _ in range(KVH)]
    n = len(pair_len)
    if n == 0 or n > n_cta:
        return None
    total = sum(pair_len)
    quota = [n_cta * ln / total for ln in pair_len]
    S = [max(1, int(q)) for q in quota]
    diff = n_cta - sum(S)
    if diff > 0:                                   # give the missing pieces to the largest remainders
        for i in sorted(range(n), key=lambda i: quota[i] - S[i], reverse=True)[:diff]:
            S[i] += 1
        # diff can exceed n only if every quota was < 1, impossible when n <= n_cta unless some pair is huge: loop for safety
        while sum(S) < n_cta:
            i = max(range(n), key=lambda i: quota[i] - S[i])
            S[i] += 1
    elif diff < 0:                                 # the >= 1 floor over-allocated: take pieces back from the smallest remainders (S_i > 1 only)
        while sum(S) > n_cta:
            i = min((i for i in range(n) if S[i] > 1), key=lambda i: quota[i] - S[i])
            S[i] -= 1
    if max(S) > smax or any(S[i] > pair_len[i] for i in range(n)):
        return None
    out = []
    for i in range(n):
        base, rem = divmod(pair_len[i], S[i])
        p0 = 0
        for j in range(S[i]):
            ln = base + (1 if j < rem else 0)
            out.append([(i, p0, p0 + ln)])
            p0 += ln
    assert len(out) == n_cta
    return out


def _lpt_cut(vis: list, n_cta: int):
    """no_split: every pair whole (one piece [0, vis)), longest pair first onto the least loaded CTA (ties: lowest CTA index). -> per-CTA [(pair, 0, len)]"""
    pair_len = [vis[s] for s in range(len(vis)) for _ in range(KVH)]
    heap = [(0, b) for b in range(n_cta)]
    out = [[] for _ in range(n_cta)]
    for i in sorted(range(len(pair_len)), key=lambda i: -pair_len[i]):
        load, b = heapq.heappop(heap)
        out[b].append((i, 0, pair_len[i]))
        heapq.heappush(heap, (load + pair_len[i], b))
    return out


def _lptcut_groups(vis: list, n_cta: int, groups: int, gran: int, eps: float, smax: int, exact_counts: int = 0):
    """R10 P19 (groups >= 1): balance positions per CONSUMER GROUP, not per CTA. The SKV_HALVES consumers run item k of a CTA on group k % G with
    1/G of the ring, so a CTA whose positions sit in one group streams at about one group's rate (job 14897: conv-L3-n54 2.85 TB/s, plan group
    max / ideal 3.86). Bins = n_cta * G (bin = b * G + h). LPT with cut: pairs longest first onto the least loaded bin; a pair that would push the
    bin over cap = T * (1 + eps) (T = ceil(total / bins)) fills it to T and the rest goes on (S > 1 only for the cut pairs). Then each CTA's
    G bins are made equal in item count to within 1 (exact_counts 1: equal) (split the longest piece of the shortest list; an empty bin takes half of the CTA's longest piece) and
    interleaved (CTA item k = bin k % G, position k // G). -> per-CTA [(pair, p0, p1)] or None if counts cannot be equalised or S > smax."""
    pair_len = [vis[s] for s in range(len(vis)) for _ in range(KVH)]
    total = sum(pair_len)
    nb = n_cta * groups
    if total == 0 or nb == 0:
        return None
    half = max(gran // 2, 1)
    T = -(-total // nb)
    cap = int(T * (1.0 + eps)) + 1
    heap = [(0, b) for b in range(nb)]
    load_of = [0] * nb
    bins = [[] for _ in range(nb)]
    for i in sorted(range(len(pair_len)), key=lambda i: -pair_len[i]):
        p0, rem = 0, pair_len[i]
        while rem > 0:
            if heap:
                load, b = heapq.heappop(heap)
                if load + rem <= cap:
                    take = rem
                else:
                    take = T - load
                    if take < half:                    # bin full: retire it
                        continue
                    if rem - take < half:              # no tail piece shorter than gran / 2
                        take = rem - half
                    if take < half:
                        take = rem
            else:                                      # every bin retired (rounding): the least loaded bin takes the rest
                b = min(range(nb), key=lambda x: load_of[x])
                load, take = load_of[b], rem
            bins[b].append((i, p0, p0 + take))
            p0, rem = p0 + take, rem - take
            load_of[b] = load + take
            if not (load_of[b] >= T and rem > 0):
                heapq.heappush(heap, (load_of[b], b))
    out = []
    for c in range(n_cta):
        L = [bins[c * groups + h] for h in range(groups)]
        for _ in range(4096):
            cnt = [len(x) for x in L]
            if max(cnt) - min(cnt) <= exact_counts ^ 1:      # counts within 1: group h of a CTA with ni items runs ceil((ni - h) / G) of them
                break
            h = cnt.index(min(cnt))
            if cnt[h] == 0:                            # empty bin: half of the CTA's longest piece
                hs, k = max(((g, k) for g in range(groups) for k in range(len(L[g]))), key=lambda t: L[t[0]][t[1]][2] - L[t[0]][t[1]][1])
            else:
                hs, k = h, max(range(cnt[h]), key=lambda k: L[h][k][2] - L[h][k][1])
            pr, a, z = L[hs][k]
            if z - a < 2 * half:
                return None
            m = a + (z - a) // 2
            L[hs][k] = (pr, a, m)
            L[h].append((pr, m, z))
        else:
            return None
        L.sort(key=len, reverse=True)                  # bins of a CTA are interchangeable: longer lists on the lower group indices
        ni = sum(len(x) for x in L)
        out.append([L[k % groups][k // groups] for k in range(ni)])
    cnt = {}
    for pieces in out:
        for pr, _, _ in pieces:
            cnt[pr] = cnt.get(pr, 0) + 1
    if cnt and max(cnt.values()) > min(smax, 255):
        return None
    return out


def _imbalance(cut) -> float:
    pos = [sum(p1 - p0 for _, p0, p1 in pieces) for pieces in cut]
    mean = sum(pos) / max(len(pos), 1)
    return max(pos) / mean if mean else 1.0


# R10 P19 fluid model of the KV stream (fit to job 14914, 19 points x 7 arms, rms log error 0.061): every busy consumer group streams at
# min(SIM_RG4 * 4 / G, SIM_HBM / busy groups) bytes/s (the per-group cap scales with its ring share), + SIM_FIX us.
SIM_HBM, SIM_RG4, SIM_FIX = 6.6e12, 25e9, 5.0
GROUP_AUTO_MARGIN, GROUP_AUTO_SPLIT_US = 0.02, 3.0     # group plan iff sim * (1 + margin) + split_us * (split pairs / pairs) < sim of the base plan


def _group_loads(cut, groups: int) -> list:
    """bytes per (CTA, group) of a per-CTA cut whose item k runs on group k % groups"""
    out = []
    for pieces in cut:
        gl = [0] * groups
        for k, (_, p0, p1) in enumerate(pieces):
            gl[k % groups] += (p1 - p0) * POS_BYTES
        out += gl
    return out


def sim_us(cut, groups: int, hbm: float = None, rg4: float = None, fix: float = None) -> float:
    """predicted attend time (us) of a cut on a build with `groups` consumer groups (fluid model above)"""
    hbm, rg4, fix = hbm or SIM_HBM, rg4 or SIM_RG4, SIM_FIX if fix is None else fix
    rem = sorted(x for x in _group_loads(cut, max(groups, 1)) if x > 0)
    cap = rg4 * 4 / max(groups, 1)
    t, done, n = 0.0, 0.0, len(rem)
    for i, x in enumerate(rem):
        t += (x - done) / min(cap, hbm / (n - i))
        done = x
    return t * 1e6 + fix


# P6 job 9692 [M]: no_split (A only) beats split (A + C) at b32kv1K (29.15 vs 32.35 us), b64kv128 (17.06 vs 24.0), b256kv128 (60.2 vs 74.5);
# their LPT imbalance is 1.156, so the auto limit is 1.20 (not 1.10). b8kv1K (64 pairs < n_cta) keeps the cut (14.0 vs 13.5).
LPT_MAX_IMB = 1.20

def plan(seqs, n_cta: int = 148, gran: int = 16, ldg_max_bytes: int = 160 * 1024, smax: int = 32, one_item: bool = True, iface: int = 1, no_split="auto",
         groups: int = 0, group_eps: float = 0.02, group_exact: int = 0, group_mode: str = "force") -> Plan:
    """seqs: list of (kv_len, q_len). q_len must be 1 unless iface == 1 and (1 <= q_len <= 16, sum q_len <= 256, nseq < 4096): R6, see the module docstring. Raises if a pair would need more than smax splits that cannot be avoided.
    one_item: when n_pairs <= n_cta, cut every pair into S_i ~ n_cta * len_i / total pieces, one item per CTA; used only if max/mean CTA positions <= 1.10,
    otherwise (and for one_item=False) the greedy cut. ldg_max_bytes: LDG engine iff the largest CTA share is <= this many KV bytes (160 KB: with the mma TMA consumer, job 9661, TMA wins at 226 KB/CTA, ties at 113 KB).
    iface: 1 or 2, only the item record differs (see the module docstring); the cut is the same.
    no_split: "auto" | True | False (ints 0 / 1 accepted). True: every pair is one item (S = 1), LPT-packed whole onto the CTAs (raises if a CTA's item count does not fit the record
    word). "auto": True iff n_pairs >= n_cta and LPT max / mean CTA load <= LPT_MAX_IMB (1.20), else the cut above.
    groups: 0 (default) = the modes above, unchanged. G >= 1 (q_len 1 only): balance per consumer group (_lptcut_groups; G = SKV_GROUPS of an SKV_HALVES build, 1 for
    the single consumers), max bin load <= T * (1 + group_eps); group_mode "auto": only if sim_us() says it beats the base plan by the margin; falls back to the modes above if the group counts cannot be equalised. Plan.no_split is True iff every pair stayed whole."""
    assert iface in (1, 2), iface
    assert isinstance(no_split, str) and no_split == "auto" or not isinstance(no_split, str), f"no_split must be 'auto', True or False, got {no_split!r}"
    if not isinstance(no_split, str):
        no_split = bool(no_split)
    seqs = [(int(k), int(q)) for k, q in seqs]
    multi = any(q != 1 for _, q in seqs)
    for s, (k, q) in enumerate(seqs):
        assert 1 <= q <= (16 if multi else 1), f"splitkv supports q_len 1..16 (seq {s} has q_len {q}); q_len > 1 needs q_len <= 16"
        assert k >= 0
    if multi:
        assert iface == 1, "q_len > 1 (R6) is interface v1 only"
        assert len(seqs) < 4096 and sum(q for _, q in seqs) <= 256, "q_len > 1: seq < 4096 (y bits 24..27 hold q_len - 1) and sum q_len <= 256"
    assert len(seqs) < (1 << 19)
    vis = [k + q for k, q in seqs]                           # rows of each sequence (items cut [0, kv_len + q_len))
    row_base = [int(x) for x in np.concatenate([[0], np.cumsum(vis)[:-1]])] if seqs else []
    tok0s = [int(x) for x in np.concatenate([[0], np.cumsum([q for _, q in seqs])[:-1]])] if seqs else []
    balance = {"one_item": None, "greedy": None}
    cut = None
    oi = _one_item_cut(vis, n_cta, smax) if one_item else None
    if oi is not None:
        balance["one_item"] = _imbalance(oi)
    n_pairs = len(vis) * KVH
    lcut = _lpt_cut(vis, n_cta) if (no_split is True or (no_split == "auto" and n_pairs >= n_cta and n_cta > 0)) and n_pairs else None
    balance["lpt"] = _imbalance(lcut) if lcut is not None else None
    used_lpt = lcut is not None and (no_split is True or balance["lpt"] <= LPT_MAX_IMB)
    if used_lpt:
        nmax = max(len(p) for p in lcut)
        assert nmax < ((1 << 23) if iface == 2 else (1 << 31)), f"no_split: a CTA would hold {nmax} items, more than the record's n_items field"
        cut, used_one, gcut = lcut, False, None
        balance["greedy"] = None
        balance["chosen"] = balance["lpt"]
    else:
        min_target = 1
        while True:
            gcut = _cut(vis, n_cta, gran, min_target)
            cnt = {}
            for pieces in gcut:
                for pr, _, _ in pieces:
                    cnt[pr] = cnt.get(pr, 0) + 1
            if not cnt or max(cnt.values()) <= smax:
                break
            min_target = int(min_target * 1.25) + 1                # raise the per-CTA target until S <= smax
            assert min_target <= 1 << 22
        balance["greedy"] = _imbalance(gcut)
        used_one = oi is not None and balance["one_item"] <= 1.10
        cut = oi if used_one else gcut
        balance["chosen"] = balance["one_item"] if used_one else balance["greedy"]
    grp = _lptcut_groups(vis, n_cta, int(groups), gran, group_eps, smax, int(group_exact)) if groups and not multi and n_pairs and n_cta else None
    if grp is not None and group_mode == "auto":             # keep the base cut unless the model says the group plan is faster by the margin
        g = int(groups)
        cntg = {}
        for pc in grp:
            for pr, _, _ in pc:
                cntg[pr] = cntg.get(pr, 0) + 1
        nsplit = sum(1 for v in cntg.values() if v > 1)
        s_base, s_grp = sim_us(cut, g), sim_us(grp, g)
        balance["sim_base"], balance["sim_groups"] = s_base, s_grp
        if not (s_grp * (1 + GROUP_AUTO_MARGIN) + GROUP_AUTO_SPLIT_US * nsplit / max(n_pairs, 1) < s_base):
            grp = None
    if grp is not None:                                      # groups >= 1 (R10 P19): group-balanced LPT with cut replaces the modes above
        g = int(groups)
        balance["groups"] = _imbalance([[p for k, p in enumerate(pc) if k % g == h] for pc in grp for h in range(g)])
        balance["chosen"] = balance["groups"]
        cut, used_one = grp, False
        used_lpt = len({pr for pc in grp for pr, _, _ in pc}) == sum(len(pc) for pc in grp)   # every pair whole: S == 1 (no combine)
    cnt = {}
    for pieces in cut:
        for pr, _, _ in pieces:
            cnt[pr] = cnt.get(pr, 0) + 1
    nseq = len(seqs)
    total_q = sum(q for _, q in seqs)
    pair_info = np.zeros((total_q * KVH, 4), np.int32)
    base = 0
    for pr in range(nseq * KVH):
        s, kvh = divmod(pr, KVH)
        kv_len, ql = seqs[s]
        for t in range(ql):                                   # token-major slots: part_base + t * S + split
            pair_info[(tok0s[s] + t) * KVH + kvh] = (base + t * cnt[pr], cnt[pr], tok0s[s], kv_len + t + 1)
        base += cnt[pr] * ql
    pbase = {}
    for pr in range(nseq * KVH):
        s, kvh = divmod(pr, KVH)
        pbase[pr] = int(pair_info[tok0s[s] * KVH + kvh, 0])
    stride = max([len(p) for p in cut] + [1])
    recs = np.zeros((n_cta, stride, 8), np.int32)
    seen = {}
    item_off, cta_pos, n_items = [0], [], 0
    for b, pieces in enumerate(cut):
        npos = 0
        for k, (pr, p0, p1) in enumerate(pieces):
            sp = seen.get(pr, 0)
            seen[pr] = sp + 1
            s, kvh = divmod(pr, KVH)
            y = kvh | (sp << Y_SPLIT_SHIFT) | (s << Y_SEQ_SHIFT) | ((seqs[s][1] - 1) << Y_QLEN_SHIFT)
            if iface == 2:
                assert cnt[pr] < 256
                recs[b, k] = (row_base[s], y, p0, p1, pbase[pr] + sp, cnt[pr] | (len(pieces) << 8), s, vis[s])
            else:
                recs[b, k] = (row_base[s], y, p0, p1, pbase[pr] + sp, cnt[pr], tok0s[s], len(pieces))
            npos += p1 - p0
        n_items += len(pieces)
        item_off.append(n_items)
        cta_pos.append(npos)
    engine = int(max(cta_pos, default=0) * POS_BYTES > ldg_max_bytes) or int(multi)   # q_len > 1: no LDG path
    return Plan(n_cta, gran, seqs, row_base, int(sum(vis)), recs, np.array(item_off, np.int32), pair_info,
                base * 2 * PSTRIDE, engine, total_q, cta_pos, used_one, balance, iface, tok0s, used_lpt)


def decode_item(y: int):
    return y & ((1 << Y_KVH_BITS) - 1), (y >> Y_SPLIT_SHIFT) & 0xFF, y >> Y_SEQ_SHIFT   # kvh, split, seq (q_len > 1 plans: use decode_item_q)


def decode_item_q(y: int):
    """(kvh, split, seq, q_len) of a record's y word of a plan with q_len > 1 (seq in bits 12..23, q_len - 1 in bits 24..27)."""
    return y & 15, (y >> Y_SPLIT_SHIFT) & 0xFF, (y >> Y_SEQ_SHIFT) & 0xFFF, ((y >> Y_QLEN_SHIFT) & 15) + 1


def torch_reference(q, kc, vc, layer, seqs, row_base=None):
    """Plain causal softmax attention in fp32. q [total_q, 16, 128] bf16, kc/vc [L, rows, 8, 128] bf16, seqs [(kv_len, q_len)] -> ao fp32 [total_q, 2048].
    Token t of a sequence sees positions p < kv_len + t + 1 (q_len == 1: kv_len + 1)."""
    import torch
    ext = [k + ql for k, ql in seqs]
    if row_base is None:
        row_base = [0]
        for v in ext[:-1]:
            row_base.append(row_base[-1] + v)
    out = torch.zeros(sum(ql for _, ql in seqs), QH * D, dtype=torch.float32, device=q.device)
    tok = 0
    for s, (kv, ql) in enumerate(seqs):
        k = kc[layer, row_base[s]:row_base[s] + ext[s]].float()      # [kv + ql, 8, 128]
        val = vc[layer, row_base[s]:row_base[s] + ext[s]].float()
        qs = q[tok:tok + ql].float().view(ql, KVH, 2, D)             # [t, kvh, hq, d]
        sc = torch.einsum("thqd,vhd->thqv", qs, k) * (D ** -0.5)
        vis = (torch.arange(ql, device=q.device) + kv + 1).view(ql, 1, 1, 1)
        sc = sc.masked_fill(torch.arange(ext[s], device=q.device).view(1, 1, 1, -1) >= vis, float("-inf"))
        pr = torch.softmax(sc, dim=-1)
        out[tok:tok + ql] = torch.einsum("thqv,vhd->thqd", pr, val).reshape(ql, QH * D)
        tok += ql
    return out


def pack_b_slot(ao_head_bf16, npad: int) -> np.ndarray:
    """Host packer of one head's bf16 [ntok, 128] into MK-core's SW128 K-major B slot: uint8 [2 * npad * 128].
    Element (t, k) lives in the 16 B chunk at (k // 64) * npad * 128 + t * 128 + ((((k % 64) // 8) ^ (t & 7)) << 4), position k % 8 inside; rows >= ntok are 0."""
    a = ao_head_bf16
    if hasattr(a, "detach"):
        a = a.detach().cpu().contiguous().view(__import__("torch").int16).numpy()
    a = np.ascontiguousarray(a).view(np.uint16)
    ntok = a.shape[0]
    assert a.shape == (ntok, D) and ntok <= npad
    out = np.zeros(2 * npad * 64, np.uint16)                          # uint16 view of the slot
    for t in range(ntok):
        for c in range(16):                                           # 16 chunks of 8 elements
            off = (c >> 3) * npad * 128 + t * 128 + (((c & 7) ^ (t & 7)) << 4)
            out[off // 2: off // 2 + 8] = a[t, c * 8:(c + 1) * 8]
    return out.view(np.uint8)
