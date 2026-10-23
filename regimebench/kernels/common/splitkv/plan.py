"""Host planner for A1 (split-KV decode attention, interface v1). Pure Python + numpy; torch only in the helpers that take tensors.

Model: Qwen3-0.6B decode, q_len == 1 per sequence, 16 q-heads, 8 kv-heads (GQA pair per item), head dim 128.
Cache layout [L, cache_rows, 8, 128]; sequence s owns rows row_base[s] .. row_base[s] + kv_len[s] + 1 (the last row is the new token,
written by the caller before attention), so visible positions are [0, kv_len + 1).

Work item = (seq, kvh, p0, p1, split). Items are cut so every CTA streams ~equal bytes: either one item per CTA (`one_item`, when there are
at most n_cta (seq, kvh) pairs and the cut is balanced to 1.10) or a greedy cut of the flat (seq, kvh) position axis.

Device encoding (matches splitkv.cuh):
  items = int32 [n_cta][stride][8], CTA b's records at items + b * stride * 8, run in order; record =
      {row_base, y = kvh | split << 4 | seq << 12, p0, p1, part_slot (= part_base + split), S, tok0, n_items of this CTA}
      kvh < 16, split < 256, seq < 2^19; n_items is the same in every valid record of the CTA; unused records are zeros
  item_off[b] .. item_off[b+1] = CTA b's items in the compact `flat_items` order (not read by attend)
  pair_info[(seq * 8 + kvh) * 4 + {0..3}] = {part_base, S, tok0, vis_len}      (partial slot of (pair, split) = part_base + split; combine only)
  partial layout (floats): part[((part_base + split) * 2 + hq) * 132 + {0..127 o, 128 m, 129 l, 130..131 pad}]
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

QH, KVH, D = 16, 8, 128
PSTRIDE = 132                      # floats per partial (o[128], m, l, pad): 528 B, 16 B aligned
POS_BYTES = 2 * D * 2              # K + V bytes of one position of one kv-head = 512 B
Y_KVH_BITS, Y_SPLIT_SHIFT, Y_SEQ_SHIFT = 4, 4, 12


@dataclass
class Plan:
    n_cta: int
    gran: int
    seqs: list
    row_base: list                 # per seq
    cache_rows: int                # sum(kv_len + 1)
    items: np.ndarray              # int32 [n_cta, stride, 8] item records (see module docstring)
    item_off: np.ndarray           # int32 [n_cta + 1]
    pair_info: np.ndarray          # int32 [nseq * 8, 4]
    part_floats: int
    engine: int                    # 0 LDG, 1 TMA
    total_q: int
    cta_positions: list = field(default_factory=list)   # positions per CTA (bytes = * POS_BYTES)
    one_item: bool = False                               # True: one item per CTA (S_i-proportional cut)
    balance: dict = field(default_factory=dict)          # {"one_item": max/mean CTA positions or None (not applicable), "greedy": ..., "chosen": ...}

    @property
    def stride(self) -> int:
        return int(self.items.shape[1])

    @property
    def flat_items(self) -> np.ndarray:
        """Valid records only, CTA order: int32 [n_items, 8]."""
        n = self.items[:, 0, 7]
        return np.concatenate([self.items[b, :int(n[b])] for b in range(self.n_cta)]) if self.n_cta else self.items.reshape(0, 8)

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
        return int(self.pair_info[seq * KVH, 2])

    def to_torch(self, device="cuda"):
        """-> dict of int32 device tensors: items (flat int32 [n_cta * stride * 8]), item_off, pair_info (flat)."""
        import torch
        return {"items": torch.from_numpy(self.items.reshape(-1).copy()).to(device),
                "item_off": torch.from_numpy(self.item_off.copy()).to(device),
                "pair_info": torch.from_numpy(self.pair_info.reshape(-1).copy()).to(device)}

    def summary(self) -> dict:
        return {"engine": self.engine_name, "s_max": self.s_max, "items_per_cta_max": self.max_items, "part_bytes": self.part_bytes,
                "cta_bytes_max": max(self.cta_bytes), "cta_bytes_min": min(self.cta_bytes), "n_items": int(self.item_off[-1]),
                "one_item": self.one_item, "balance_one_item": self.balance.get("one_item"), "balance_greedy": self.balance.get("greedy")}


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


def _imbalance(cut) -> float:
    pos = [sum(p1 - p0 for _, p0, p1 in pieces) for pieces in cut]
    mean = sum(pos) / max(len(pos), 1)
    return max(pos) / mean if mean else 1.0


def plan(seqs, n_cta: int = 148, gran: int = 16, ldg_max_bytes: int = 512 * 1024, smax: int = 32, one_item: bool = True) -> Plan:
    """seqs: list of (kv_len, q_len). v1: q_len must be 1. Raises if a pair would need more than smax splits that cannot be avoided.
    one_item: when n_pairs <= n_cta, cut every pair into S_i ~ n_cta * len_i / total pieces, one item per CTA; used only if max/mean CTA positions <= 1.10,
    otherwise (and for one_item=False) the greedy cut. ldg_max_bytes: LDG engine iff the largest CTA share is <= this many KV bytes."""
    seqs = [(int(k), int(q)) for k, q in seqs]
    for s, (k, q) in enumerate(seqs):
        assert q == 1, f"splitkv v1 supports q_len == 1 only (seq {s} has q_len {q})"
        assert k >= 0
    assert len(seqs) < (1 << 19)
    vis = [k + 1 for k, _ in seqs]
    row_base = [int(x) for x in np.concatenate([[0], np.cumsum(vis)[:-1]])] if seqs else []
    balance = {"one_item": None, "greedy": None}
    cut = None
    oi = _one_item_cut(vis, n_cta, smax) if one_item else None
    if oi is not None:
        balance["one_item"] = _imbalance(oi)
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
    cnt = {}
    for pieces in cut:
        for pr, _, _ in pieces:
            cnt[pr] = cnt.get(pr, 0) + 1
    nseq = len(seqs)
    pair_info = np.zeros((nseq * KVH, 4), np.int32)
    base = 0
    for pr in range(nseq * KVH):
        pair_info[pr] = (base, cnt[pr], pr // KVH, vis[pr // KVH])
        base += cnt[pr]
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
            recs[b, k] = (row_base[s], kvh | (sp << Y_SPLIT_SHIFT) | (s << Y_SEQ_SHIFT), p0, p1, int(pair_info[pr, 0]) + sp, cnt[pr], s, len(pieces))
            npos += p1 - p0
        n_items += len(pieces)
        item_off.append(n_items)
        cta_pos.append(npos)
    engine = int(max(cta_pos, default=0) * POS_BYTES > ldg_max_bytes)
    return Plan(n_cta, gran, seqs, row_base, int(sum(vis)), recs, np.array(item_off, np.int32), pair_info,
                base * 2 * PSTRIDE, engine, nseq, cta_pos, used_one, balance)


def decode_item(y: int):
    return y & ((1 << Y_KVH_BITS) - 1), (y >> Y_SPLIT_SHIFT) & 0xFF, y >> Y_SEQ_SHIFT   # kvh, split, seq


def torch_reference(q, kc, vc, layer, seqs, row_base=None):
    """Plain softmax attention in fp32. q [total_q, 16, 128] bf16, kc/vc [L, rows, 8, 128] bf16, seqs [(kv_len, 1)] -> ao fp32 [total_q, 2048]."""
    import torch
    vis = [k + 1 for k, _ in seqs]
    if row_base is None:
        row_base = [0]
        for v in vis[:-1]:
            row_base.append(row_base[-1] + v)
    out = torch.zeros(len(seqs), QH * D, dtype=torch.float32, device=q.device)
    for s, v in enumerate(vis):
        k = kc[layer, row_base[s]:row_base[s] + v].float()           # [v, 8, 128]
        val = vc[layer, row_base[s]:row_base[s] + v].float()
        qs = q[s].float().view(KVH, 2, D)                            # [kvh, hq, d]
        sc = torch.einsum("hqd,vhd->hqv", qs, k) * (D ** -0.5)
        pr = torch.softmax(sc, dim=-1)
        out[s] = torch.einsum("hqv,vhd->hqd", pr, val).reshape(QH * D)
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
