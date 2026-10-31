"""numba port of plan.plan(..., gran=16, smax=32, one_item=True, iface=1, no_split="auto", align=align, paged=True,
groups=G, group_mode="auto" | "force") fused with the per-token arrays of a serving step.

Serving mode needs the plan on the host every step; the Python planner (plan.py) costs 0.2-1.9 ms, this one a few us. ``fill_step`` writes straight into an int32 staging row:
    row[off_req + t] = request of token t        row[off_vis + t] = visible length of token t (kv + t_in_seq + 1)
    row[off_lmt + r] = packed index of the last token of request r      row[off_lms + r] = r
    with want_plan: row[off_items ...] = items int32 [nb][stride][8] then pair_info int32 [total_q * 8][4] (right behind the items)
and returns (words, stride, s_max, part_floats, status): words = used int32 words of the row, status 0 = ok (1 = cut failed, 2 = capacity).
tests/test_e2e_mk_cpu.py and tests/test_splitkv_fast.py check it equal to plan.plan on random shapes.
R10 P22: groups = G >= 1 (q_len 1 only; ignored when any q > 1) adds plan.py's group-balanced plan (_lptcut_groups with group_exact = 0, group_eps = 0.02, _group_loads, sim_us); group_mode 1 = "auto"
(keep it only if the fluid model beats the base plan by the margin), 0 = "force". groups = 0 is the old behaviour exactly. The record format is plan.py's module docstring.
"""
from __future__ import annotations

import numpy as np
from numba import njit

KVH = 8
PSTRIDE = 132
LPT_MAX_IMB = 1.20
SMAX = 32
GRAN = 16
# R10 P19 group plan constants (plan.py: SIM_*, GROUP_*, POS_BYTES)
SIM_HBM, SIM_RG4, SIM_FIX = 6.6e12, 25e9, 5.0
GROUP_AUTO_MARGIN = 0.02
COMB_FIX_US, COMB_PER_PART_US = 2.5, 0.0017      # plan.comb_us


@njit(cache=True)
def _comb_us(o_pr, k, npairs):
    """plan.comb_us of a cut (pieces j < k with pair o_pr[j])"""
    cg = np.zeros(npairs, np.int64)
    for j in range(k):
        cg[o_pr[j]] += 1
    parts = 0
    for i in range(npairs):
        if cg[i] > 1:
            parts += cg[i]
    return COMB_FIX_US + COMB_PER_PART_US * parts if parts > 0 else 0.0
GROUP_EPS = 0.02
POS_BYTES = 512
HSHIFT = 16


@njit(cache=True)
def _greedy(pair_len, n_cta, gran, min_target, o_cta, o_pr, o_p0, o_p1):
    npairs = pair_len.shape[0]
    total = 0
    for i in range(npairs):
        total += pair_len[i]
    half = max(gran // 2, 1)
    pi = 0
    po = 0
    done = 0
    k = 0
    cap = o_cta.shape[0]
    for b in range(n_cta):
        left = 0
        if done < total:
            left = max(-(-(total - done) // (n_cta - b)), min_target)
        while left > 0 and pi < npairs:
            avail = pair_len[pi] - po
            take = min(avail, left)
            if avail <= half:
                take = avail
            elif take < avail:
                if take < half:
                    take = half if half - take <= take else 0
                if take > 0 and avail - take < half:
                    take = avail if (avail - take) <= half - (avail - take) else avail - half
                if take == 0:
                    break
            if k >= cap:
                return -2
            o_cta[k] = b
            o_pr[k] = pi
            o_p0[k] = po
            o_p1[k] = po + take
            k += 1
            po += take
            done += take
            left -= take
            if po == pair_len[pi]:
                pi += 1
                po = 0
    if pi != npairs or po != 0:
        return -1
    return k


@njit(cache=True)
def _one_item(pair_len, n_cta, o_cta, o_pr, o_p0, o_p1):
    """-> number of pieces (== n_cta) or 0 when not applicable."""
    n = pair_len.shape[0]
    if n == 0 or n > n_cta:
        return 0
    total = 0
    for i in range(n):
        total += pair_len[i]
    quota = np.empty(n)
    S = np.empty(n, np.int64)
    ssum = 0
    for i in range(n):
        quota[i] = n_cta * pair_len[i] / total
        S[i] = max(1, int(quota[i]))
        ssum += S[i]
    diff = n_cta - ssum
    if diff > 0:
        key = np.empty(n)
        for i in range(n):
            key[i] = -(quota[i] - S[i])
        order = np.argsort(key, kind="mergesort")
        for j in range(min(diff, n)):
            S[order[j]] += 1
        ssum = 0
        for i in range(n):
            ssum += S[i]
        while ssum < n_cta:
            bi = 0
            bv = quota[0] - S[0]
            for i in range(1, n):
                v = quota[i] - S[i]
                if v > bv:
                    bv = v
                    bi = i
            S[bi] += 1
            ssum += 1
    elif diff < 0:
        while ssum > n_cta:
            bi = -1
            bv = 0.0
            for i in range(n):
                if S[i] > 1:
                    v = quota[i] - S[i]
                    if bi < 0 or v < bv:
                        bv = v
                        bi = i
            if bi < 0:
                return 0
            S[bi] -= 1
            ssum -= 1
    for i in range(n):
        if S[i] > SMAX or S[i] > pair_len[i]:
            return 0
    k = 0
    for i in range(n):
        base = pair_len[i] // S[i]
        rem = pair_len[i] % S[i]
        p0 = 0
        for j in range(S[i]):
            ln = base + (1 if j < rem else 0)
            o_cta[k] = k
            o_pr[k] = i
            o_p0[k] = p0
            o_p1[k] = p0 + ln
            k += 1
            p0 += ln
    return k


@njit(cache=True)
def _lpt(pair_len, n_cta, o_cta, o_pr, o_p0, o_p1):
    n = pair_len.shape[0]
    heap = np.empty(n_cta, np.int64)          # key = load * 1024 + cta (min-heap; ties -> lowest cta)
    for b in range(n_cta):
        heap[b] = b
    order = _desc_order(pair_len)
    for j in range(n):
        i = order[j]
        top = heap[0]
        b = top & 1023
        load = top >> 10
        o_cta[j] = b
        o_pr[j] = i
        o_p0[j] = 0
        o_p1[j] = pair_len[i]
        key = ((load + pair_len[i]) << 10) | b
        # replace root and sift down
        pos = 0
        while True:
            c = 2 * pos + 1
            if c >= n_cta:
                break
            if c + 1 < n_cta and heap[c + 1] < heap[c]:
                c += 1
            if heap[c] < key:
                heap[pos] = heap[c]
                pos = c
            else:
                break
        heap[pos] = key
    return n


@njit(cache=True)
def _imbalance(o_cta, o_p0, o_p1, k, n_cta):
    pos = np.zeros(n_cta, np.int64)
    tot = 0
    for j in range(k):
        d = o_p1[j] - o_p0[j]
        pos[o_cta[j]] += d
        tot += d
    mean = tot / n_cta
    mx = 0
    for b in range(n_cta):
        if pos[b] > mx:
            mx = pos[b]
    return mx / mean if mean > 0 else 1.0


@njit(cache=True)
def _desc_order(vals):
    """indices of vals, longest first, ties by index (== argsort(-vals, kind="mergesort")); sorts unique packed keys (np.sort is ~3x faster than the numba mergesort argsort)"""
    n = vals.shape[0]
    key = np.empty(n, np.int64)
    for i in range(n):
        key[i] = ((1 << 40) - vals[i]) << 22 | i
    key.sort()
    for i in range(n):
        key[i] &= (1 << 22) - 1
    return key


@njit(cache=True)
def _heap_push(heap, n, key):
    pos = n
    while pos > 0:
        par = (pos - 1) >> 1
        if heap[par] > key:
            heap[pos] = heap[par]
            pos = par
        else:
            break
    heap[pos] = key


@njit(cache=True)
def _heap_replace_top(heap, n, key):
    """replace the min of heap[:n] by key (sift down)"""
    pos = 0
    while True:
        c = 2 * pos + 1
        if c >= n:
            break
        if c + 1 < n and heap[c + 1] < heap[c]:
            c += 1
        if heap[c] < key:
            heap[pos] = heap[c]
            pos = c
        else:
            break
    heap[pos] = key


@njit(cache=True)
def _heap_pop(heap, n):
    """pop the min of heap[:n] -> (key, n - 1)"""
    top = heap[0]
    n -= 1
    if n > 0:
        key = heap[n]
        pos = 0
        while True:
            c = 2 * pos + 1
            if c >= n:
                break
            if c + 1 < n and heap[c + 1] < heap[c]:
                c += 1
            if heap[c] < key:
                heap[pos] = heap[c]
                pos = c
            else:
                break
        heap[pos] = key
    return top, n


@njit(cache=True)
def _lptcut_groups(pair_len, n_cta, G, gran, eps, smax):
    """numba port of plan._lptcut_groups (group_exact = 0). -> (k, o_cta, o_pr, o_p0, o_p1), k = pieces in (CTA, slot) order, k < 0 = None (-1) or capacity (-2)."""
    npairs = pair_len.shape[0]
    nb = n_cta * G
    total = 0
    for i in range(npairs):
        total += pair_len[i]
    e = np.empty(0, np.int64)
    if total == 0 or nb == 0 or nb >= (1 << HSHIFT):
        return -1, e, e, e, e
    half = max(gran // 2, 1)
    T = -(-total // nb)
    cap = int(T * (1.0 + eps)) + 1
    heap = np.empty(nb, np.int64)
    for b in range(nb):
        heap[b] = b                           # load 0, bin b: already a valid min-heap
    hn = nb
    load_of = np.zeros(nb, np.int64)
    PC = npairs + 4 * nb + 4096
    pp_pr = np.empty(PC, np.int64)
    pp_p0 = np.empty(PC, np.int64)
    pp_p1 = np.empty(PC, np.int64)
    pp_nx = np.empty(PC, np.int64)
    head = np.full(nb, -1, np.int64)
    tail = np.full(nb, -1, np.int64)
    bcnt = np.zeros(nb, np.int64)
    npc = 0
    order = _desc_order(pair_len)
    mask = (1 << HSHIFT) - 1
    for oi in range(npairs):
        i = order[oi]
        p0 = 0
        rem = pair_len[i]
        while rem > 0:
            popped = hn > 0                           # the root is only peeked: a bin that goes back is a replace-top (one sift)
            if popped:
                key = heap[0]
                b = key & mask
                load = key >> HSHIFT
                if load + rem <= cap:
                    take = rem
                else:
                    take = T - load
                    if take < half:
                        key, hn = _heap_pop(heap, hn)
                        continue
                    if rem - take < half:
                        take = rem - half
                    if take < half:
                        take = rem
            else:
                b = 0
                for x in range(1, nb):
                    if load_of[x] < load_of[b]:
                        b = x
                load = load_of[b]
                take = rem
            if npc >= PC:
                return -2, e, e, e, e
            pp_pr[npc] = i
            pp_p0[npc] = p0
            pp_p1[npc] = p0 + take
            pp_nx[npc] = -1
            if tail[b] < 0:
                head[b] = npc
            else:
                pp_nx[tail[b]] = npc
            tail[b] = npc
            bcnt[b] += 1
            npc += 1
            p0 += take
            rem -= take
            load_of[b] = load + take
            if not (load_of[b] >= T and rem > 0):
                if popped:
                    _heap_replace_top(heap, hn, (load_of[b] << HSHIFT) | b)
                else:
                    _heap_push(heap, hn, (load_of[b] << HSHIFT) | b)
                    hn += 1
            elif popped:
                key, hn = _heap_pop(heap, hn)
    # per CTA: equalise the item counts of its G bins, sort, interleave
    maxc = 0
    capout = 0
    for c in range(n_cta):
        m = 0
        for h in range(G):
            if bcnt[c * G + h] > m:
                m = bcnt[c * G + h]
        capout += G * m
        if m > maxc:
            maxc = m
    Lc = maxc + 2
    Lpr = np.empty((G, Lc), np.int64)
    Lp0 = np.empty((G, Lc), np.int64)
    Lp1 = np.empty((G, Lc), np.int64)
    cnt = np.zeros(G, np.int64)
    srt = np.empty(G, np.int64)
    o_cta = np.empty(capout, np.int64)
    o_pr = np.empty(capout, np.int64)
    o_p0 = np.empty(capout, np.int64)
    o_p1 = np.empty(capout, np.int64)
    k = 0
    for c in range(n_cta):
        for h in range(G):
            j = head[c * G + h]
            n = 0
            while j >= 0:
                Lpr[h, n] = pp_pr[j]
                Lp0[h, n] = pp_p0[j]
                Lp1[h, n] = pp_p1[j]
                n += 1
                j = pp_nx[j]
            cnt[h] = n
        ok = False
        for it in range(4096):
            mn = cnt[0]
            mx = cnt[0]
            hmin = 0
            for h in range(1, G):
                if cnt[h] < mn:
                    mn = cnt[h]
                    hmin = h
                if cnt[h] > mx:
                    mx = cnt[h]
            if mx - mn <= 1:
                ok = True
                break
            h = hmin
            hs = 0
            kk = 0
            if cnt[h] == 0:
                best = -1
                for g in range(G):
                    for q in range(cnt[g]):
                        ln = Lp1[g, q] - Lp0[g, q]
                        if ln > best:
                            best = ln
                            hs = g
                            kk = q
            else:
                hs = h
                best = -1
                for q in range(cnt[h]):
                    ln = Lp1[h, q] - Lp0[h, q]
                    if ln > best:
                        best = ln
                        kk = q
            a = Lp0[hs, kk]
            z = Lp1[hs, kk]
            if z - a < 2 * half or cnt[h] >= Lc:
                return -1, e, e, e, e
            m = a + (z - a) // 2
            pr = Lpr[hs, kk]
            Lp1[hs, kk] = m
            Lpr[h, cnt[h]] = pr
            Lp0[h, cnt[h]] = m
            Lp1[h, cnt[h]] = z
            cnt[h] += 1
        if not ok:
            return -1, e, e, e, e
        # stable sort of the G lists by count, descending
        for h in range(G):
            srt[h] = h
        for x in range(1, G):
            v = srt[x]
            y = x - 1
            while y >= 0 and cnt[srt[y]] < cnt[v]:
                srt[y + 1] = srt[y]
                y -= 1
            srt[y + 1] = v
        ni = 0
        for h in range(G):
            ni += cnt[h]
        for kq in range(ni):
            g = srt[kq % G]
            q = kq // G
            o_cta[k] = c
            o_pr[k] = Lpr[g, q]
            o_p0[k] = Lp0[g, q]
            o_p1[k] = Lp1[g, q]
            k += 1
    pc = np.zeros(npairs, np.int64)
    mxs = 0
    for j in range(k):
        pc[o_pr[j]] += 1
        if pc[o_pr[j]] > mxs:
            mxs = pc[o_pr[j]]
    if mxs > min(smax, 255):
        return -1, e, e, e, e
    return k, o_cta, o_pr, o_p0, o_p1


@njit(cache=True)
def _sim_us(o_cta, o_p0, o_p1, k, n_cta, G):
    """plan.sim_us of a cut whose item j of a CTA (array order) runs on group j % G"""
    loads = np.zeros(n_cta * G, np.int64)
    slot = np.zeros(n_cta, np.int64)
    for j in range(k):
        b = o_cta[j]
        loads[b * G + slot[b] % G] += (o_p1[j] - o_p0[j]) * POS_BYTES
        slot[b] += 1
    rem = np.sort(loads)
    cap = SIM_RG4 * 4 / G
    t = 0.0
    done = 0.0
    n = 0
    for x in range(rem.shape[0]):
        if rem[x] > 0:
            n += 1
    i = 0
    for x in range(rem.shape[0]):
        if rem[x] > 0:
            v = float(rem[x])
            t += (v - done) / min(cap, SIM_HBM / (n - i))
            done = v
            i += 1
    return t * 1e6 + SIM_FIX


@njit(cache=True)
def _scale(o_p0, o_p1, o_pr, k, align, vis):
    if align > 1:
        for j in range(k):
            o_p0[j] = o_p0[j] * align
            o_p1[j] = min(o_p1[j] * align, vis[o_pr[j] // KVH])


@njit(cache=True)
def fill_step(kv, q, nb, align, row, off_req, off_vis, off_lmt, off_lms, off_items, want_plan, groups=0, group_mode=1):
    n = kv.shape[0]
    ntok = 0
    for r in range(n):
        ntok += q[r]
    t = 0
    for r in range(n):
        for j in range(q[r]):
            row[off_req + t] = r
            row[off_vis + t] = kv[r] + j + 1
            t += 1
        row[off_lmt + r] = t - 1
        row[off_lms + r] = r
    if not want_plan:
        return off_items, 1, 0, 0, 0
    vis = np.empty(n, np.int64)
    vis_c = np.empty(n, np.int64)
    tok0 = np.empty(n, np.int64)
    t0 = 0
    for r in range(n):
        vis[r] = kv[r] + q[r]
        vis_c[r] = -(-vis[r] // align) if align > 1 else vis[r]
        tok0[r] = t0
        t0 += q[r]
    npairs = n * KVH
    gran_c = max(GRAN // align, 1) if align > 1 else GRAN
    pair_c = np.empty(npairs, np.int64)
    for p in range(npairs):
        pair_c[p] = vis_c[p // KVH]
    cap = npairs + nb + 16
    cap2 = max(cap, nb)
    a_cta = np.empty(cap2, np.int64)
    a_pr = np.empty(cap2, np.int64)
    a_p0 = np.empty(cap2, np.int64)
    a_p1 = np.empty(cap2, np.int64)
    ko = _one_item(pair_c, nb, a_cta, a_pr, a_p0, a_p1)
    bal_one = 0.0
    if ko > 0:
        _scale(a_p0, a_p1, a_pr, ko, align, vis)
        bal_one = _imbalance(a_cta, a_p0, a_p1, ko, nb)
    b_cta = np.empty(cap2, np.int64)
    b_pr = np.empty(cap2, np.int64)
    b_p0 = np.empty(cap2, np.int64)
    b_p1 = np.empty(cap2, np.int64)
    used_lpt = False
    kl = 0
    if npairs >= nb and npairs > 0:
        kl = _lpt(pair_c, nb, b_cta, b_pr, b_p0, b_p1)
        _scale(b_p0, b_p1, b_pr, kl, align, vis)
        used_lpt = _imbalance(b_cta, b_p0, b_p1, kl, nb) <= LPT_MAX_IMB
    if used_lpt:
        c_cta, c_pr, c_p0, c_p1, k = b_cta, b_pr, b_p0, b_p1, kl
    else:
        min_target = 1
        g_cta = np.empty(cap2, np.int64)
        g_pr = np.empty(cap2, np.int64)
        g_p0 = np.empty(cap2, np.int64)
        g_p1 = np.empty(cap2, np.int64)
        cnt = np.zeros(npairs, np.int64)
        while True:
            kg = _greedy(pair_c, nb, gran_c, min_target, g_cta, g_pr, g_p0, g_p1)
            if kg < 0:
                return 0, 0, 0, 0, 1 if kg == -1 else 2
            _scale(g_p0, g_p1, g_pr, kg, align, vis)
            cnt[:] = 0
            mx = 0
            for j in range(kg):
                cnt[g_pr[j]] += 1
                if cnt[g_pr[j]] > mx:
                    mx = cnt[g_pr[j]]
            if kg == 0 or mx <= SMAX:
                break
            min_target = int(min_target * 1.25) + 1
        bal_g = _imbalance(g_cta, g_p0, g_p1, kg, nb)
        if ko > 0 and bal_one <= 1.10:
            c_cta, c_pr, c_p0, c_p1, k = a_cta, a_pr, a_p0, a_p1, ko
        else:
            c_cta, c_pr, c_p0, c_p1, k = g_cta, g_pr, g_p0, g_p1, kg
    if groups > 0 and npairs > 0 and nb > 0:
        multi = False
        for r in range(n):
            if q[r] != 1:
                multi = True
        if not multi and group_mode == 1:                    # P22 cheap exit (as plan.py): s_grp >= total bytes / HBM + fix
            tot = 0
            for r in range(n):
                tot += vis[r]
            lb = tot * KVH * POS_BYTES / SIM_HBM * 1e6 + SIM_FIX
            if _sim_us(c_cta, c_p0, c_p1, k, nb, groups) + _comb_us(c_pr, k, npairs) <= lb * (1 + GROUP_AUTO_MARGIN):
                multi = True                                 # skip the group plan
        if not multi:
            kq, d_cta, d_pr, d_p0, d_p1 = _lptcut_groups(pair_c, nb, groups, gran_c, GROUP_EPS, SMAX)
            if kq == -2:
                return 0, 0, 0, 0, 2
            if kq >= 0:
                if align > 1:
                    _scale(d_p0, d_p1, d_pr, kq, align, vis)
                keep = True
                if group_mode == 1:
                    s_base = _sim_us(c_cta, c_p0, c_p1, k, nb, groups) + _comb_us(c_pr, k, npairs)
                    s_grp = _sim_us(d_cta, d_p0, d_p1, kq, nb, groups)
                    keep = s_grp * (1 + GROUP_AUTO_MARGIN) + _comb_us(d_pr, kq, npairs) < s_base
                if keep:
                    c_cta, c_pr, c_p0, c_p1, k = d_cta, d_pr, d_p0, d_p1, kq
    # group pieces by CTA (stable)
    order = np.empty(k, np.int64)
    for j in range(k):
        order[j] = (c_cta[j] << 32) | j
    order.sort()
    for j in range(k):
        order[j] &= 0xFFFFFFFF
    cnt = np.zeros(npairs, np.int64)
    ncta = np.zeros(nb, np.int64)
    for j in range(k):
        cnt[c_pr[j]] += 1
        ncta[c_cta[j]] += 1
    stride = 1
    for b in range(nb):
        if ncta[b] > stride:
            stride = ncta[b]
    smx = 0
    for p in range(npairs):
        if cnt[p] > smx:
            smx = cnt[p]
    base = 0
    pbase = np.empty(npairs, np.int64)
    for p in range(npairs):
        pbase[p] = base
        base += cnt[p] * q[p // KVH]
    part_floats = base * 2 * PSTRIDE
    nitems = nb * stride * 8
    pair_off = off_items + nitems
    words = pair_off + ntok * KVH * 4
    if words > row.shape[0]:
        return 0, stride, smx, part_floats, 2
    for w in range(off_items, words):
        row[w] = 0
    seen = np.zeros(npairs, np.int64)
    slot = np.zeros(nb, np.int64)
    for jj in range(k):
        j = order[jj]
        b = c_cta[j]
        p = c_pr[j]
        s = p // KVH
        kvh = p % KVH
        sp = seen[p]
        seen[p] += 1
        y = kvh | (sp << 4) | (s << 12) | ((q[s] - 1) << 24)
        o = off_items + (b * stride + slot[b]) * 8
        slot[b] += 1
        row[o] = s
        row[o + 1] = y
        row[o + 2] = c_p0[j]
        row[o + 3] = c_p1[j]
        row[o + 4] = pbase[p] + sp
        row[o + 5] = cnt[p]
        row[o + 6] = tok0[s]
        row[o + 7] = ncta[b]
    for s in range(n):
        for tt in range(q[s]):
            for kvh in range(KVH):
                p = s * KVH + kvh
                o = pair_off + ((tok0[s] + tt) * KVH + kvh) * 4
                row[o] = pbase[p] + tt * cnt[p]
                row[o + 1] = cnt[p]
                row[o + 2] = tok0[s]
                row[o + 3] = kv[s] + tt + 1
    return words, stride, smx, part_floats, 0
