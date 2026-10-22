"""3×3×3 for Method 1: cubie coordinates, Korf's pattern databases, IDA* (Section 4.2-4.4).

* **Cubies.** Corner slots and edge slots are read off the engine's clusters; every move is
  turned into per-slot tables (source slot, orientation change), so the engine and the
  search can never disagree on conventions. A colour state maps to (cp, co, ep, eo).
* **Pattern databases** (Korf 1997): corners 8!·3⁷ = 88,179,840 entries; two 6-edge tables
  of 12!/6!·2⁶ = 42,577,920 entries each; one byte per entry, built by breadth-first search
  from solved in Numba and cached under ``checkpoints/m1/`` (git-ignored, rebuilt on demand).
* **IDA\\*** with h = max of the three lookups (admissible), canonical move order (no face
  twice in a row, one order for opposite faces). HTM uses the 18 face turns, QTM the 12
  quarter turns with its own databases.
"""
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import numba as nb
import numpy as np

from ...engine.orbits import clusters
from ...engine.puzzle import Move, cube

CACHE = Path(__file__).resolve().parents[3] / "checkpoints" / "m1"
N_CP, N_CO = 40320, 2187
N_EDGE6 = 665280 * 64


# ------------------------------------------------------------------ cubie model
class Cubies:
    """Slot tables for cube(3) under a move set (face turns only)."""

    def __init__(self, metric: str = "htm"):
        P = cube(3)
        self.P = P
        turns = (1, 2, 3) if metric == "htm" else (1, 3)
        self.moves = [P.move_index[Move(a, layer, t)] for a in range(3) for layer in (0, 2) for t in turns]
        self.metric = metric
        cl = {c.kind: c for c in clusters(P)}
        self.corner = self._slots(cl["corner"])
        self.edge = self._slots(cl["midge"])
        self.c_src, self.c_d = self._tables(self.corner, 3)
        self.e_src, self.e_d = self._tables(self.edge, 2)
        # canonical order: forbid same axis+layer, and for one axis only ascending layer
        M = len(self.moves)
        allowed = np.ones((M + 1, M), dtype=np.bool_)
        for i, a in enumerate(self.moves):
            ma = P.moves[a]
            for j, b in enumerate(self.moves):
                mb = P.moves[b]
                if ma.axis == mb.axis and mb.layer <= ma.layer:
                    allowed[i, j] = False
        self.allowed = allowed
        self.inverse = np.array([self.moves.index(int(P.inverse_move[m])) for m in self.moves])

    def _slots(self, c):
        """Per slot: global sticker ids, reference (U/D, else F/B) first, then right-handed."""
        P = self.P
        by: dict[tuple, list[int]] = {}
        for s in c.stickers:
            by.setdefault(tuple(P.sticker_pos[s]), []).append(int(s))
        slots = []
        for pos in sorted(by):
            st = by[pos]
            def rank(s):
                n = P.sticker_normal[s]
                return 0 if n[1] != 0 else (1 if n[2] != 0 else 2)
            st.sort(key=rank)
            if len(st) == 3:
                n = [P.sticker_normal[s] for s in st]
                if np.dot(np.cross(n[0], n[1]), n[2]) < 0:
                    st = [st[0], st[2], st[1]]
            slots.append(st)
        return slots

    def _tables(self, slots, k):
        P = self.P
        where = {s: (j, p) for j, st in enumerate(slots) for p, s in enumerate(st)}
        src = np.zeros((len(self.moves), len(slots)), dtype=np.int64)
        d = np.zeros_like(src)
        for mi, m in enumerate(self.moves):
            perm = P.move_perms[m]
            for j, st in enumerate(slots):
                # new[st[p]] = old[perm[st[p]]]: the sticker at old (i, q) lands at (j, p)
                i, q = where[int(perm[st[0]])]
                src[mi, j] = i
                d[mi, j] = (0 - q) % k  # old orientation q' at i becomes p=0 when q' == q
        return src, d

    def from_colours(self, state: np.ndarray):
        """Colour state (oriented: fixed centers at reference) -> (cp, co, ep, eo)."""
        def read(slots, k):
            home = {}
            for j, st in enumerate(slots):
                home[frozenset(int(self.P.solved[s]) for s in st)] = j
            ref = [int(self.P.solved[st[0]]) for st in slots]  # each piece's reference colour
            perm, ori = np.zeros(len(slots), dtype=np.int64), np.zeros(len(slots), dtype=np.int64)
            for j, st in enumerate(slots):
                cols = [int(state[s]) for s in st]
                piece = home[frozenset(cols)]
                perm[j] = piece
                ori[j] = cols.index(ref[piece]) % k
            return perm, ori
        cp, co = read(self.corner, 3)
        ep, eo = read(self.edge, 2)
        return cp, co, ep, eo


# ------------------------------------------------------------------ coordinates (numba)
@nb.njit(cache=True)
def rank_perm(p):
    n = len(p)
    r = 0
    for i in range(n):
        c = 0
        for j in range(i + 1, n):
            if p[j] < p[i]:
                c += 1
        r = r * (n - i) + c
    return r


@nb.njit(cache=True)
def unrank_perm(r, n, out):
    digits = np.zeros(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        digits[i] = r % (n - i)
        r //= (n - i)
    used = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        c = digits[i]
        for v in range(n):
            if not used[v]:
                if c == 0:
                    out[i] = v
                    used[v] = True
                    break
                c -= 1


@nb.njit(cache=True)
def ori_index(o, k):
    r = 0
    for i in range(len(o) - 1):
        r = r * k + o[i]
    return r


@nb.njit(cache=True)
def edge6_index(slot, ori):
    """Positions (12 slots) and flips of 6 tracked pieces -> 0 .. 42,577,919."""
    used = 0
    r = 0
    for i in range(6):
        s = slot[i]
        below = 0
        for t in range(s):
            if (used >> t) & 1:
                below += 1
        r = r * (12 - i) + (s - below)
        used |= 1 << s
    o = 0
    for i in range(6):
        o = o * 2 + ori[i]
    return r * 64 + o


@nb.njit(cache=True)
def edge6_unindex(idx, slot, ori):
    o = idx % 64
    r = idx // 64
    for i in range(5, -1, -1):
        ori[i] = o & 1
        o >>= 1
    digits = np.zeros(6, dtype=np.int64)
    for i in range(5, -1, -1):
        digits[i] = r % (12 - i)
        r //= (12 - i)
    used = 0
    for i in range(6):
        c = digits[i]
        for v in range(12):
            if not (used >> v) & 1:
                if c == 0:
                    slot[i] = v
                    used |= 1 << v
                    break
                c -= 1


# ------------------------------------------------------------------ move tables
def build_tables(cb: Cubies):
    """Corner perm / orientation coordinate tables, and per-piece edge (slot, flip) tables."""
    M = len(cb.moves)
    src, d = cb.c_src, cb.c_d
    # destination of each slot: piece at slot i goes to dst[m, i]
    c_dst = np.argsort(src, axis=1)
    cpm = np.zeros((M, N_CP), dtype=np.int32)
    com = np.zeros((M, N_CO), dtype=np.int32)
    _fill_corner_tables(src, d, cpm, com)
    e_dst = np.argsort(cb.e_src, axis=1)
    # edge code = slot * 2 + flip; new slot = dst[slot], new flip = flip + d[new slot]
    etab = np.zeros((M, 24), dtype=np.int64)
    for m in range(M):
        for s in range(12):
            for f in range(2):
                ns = e_dst[m, s]
                etab[m, s * 2 + f] = ns * 2 + (f + cb.e_d[m, ns]) % 2
    return cpm, com, etab, c_dst


@nb.njit(cache=True)
def _fill_corner_tables(src, d, cpm, com):
    M = src.shape[0]
    p = np.zeros(8, dtype=np.int64)
    q = np.zeros(8, dtype=np.int64)
    for r in range(40320):
        unrank_perm(r, 8, p)
        for m in range(M):
            for j in range(8):
                q[j] = p[src[m, j]]
            cpm[m, r] = rank_perm(q)
    o = np.zeros(8, dtype=np.int64)
    o2 = np.zeros(8, dtype=np.int64)
    for r in range(2187):
        x = r
        s = 0
        for i in range(6, -1, -1):
            o[i] = x % 3
            x //= 3
            s += o[i]
        o[7] = (3 - s % 3) % 3
        for m in range(M):
            for j in range(8):
                o2[j] = (o[src[m, j]] + d[m, j]) % 3
            com[m, r] = ori_index(o2, 3)


# ------------------------------------------------------------------ pattern databases
@nb.njit(cache=True)
def _bfs_corners(cpm, com, dist):
    dist[:] = 255
    dist[0] = 0  # solved: identity perm (rank 0), zero twist (index 0)
    depth = 0
    count = 1
    M = cpm.shape[0]
    while True:
        new = 0
        for idx in range(dist.shape[0]):
            if dist[idx] != depth:
                continue
            p, o = idx // 2187, idx % 2187
            for m in range(M):
                j = cpm[m, p] * 2187 + com[m, o]
                if dist[j] == 255:
                    dist[j] = depth + 1
                    new += 1
        if new == 0:
            break
        count += new
        depth += 1
    return count


@nb.njit(cache=True)
def _bfs_edges(etab, pieces_home, dist):
    dist[:] = 255
    slot = np.zeros(6, dtype=np.int64)
    ori = np.zeros(6, dtype=np.int64)
    for i in range(6):
        slot[i] = pieces_home[i]
        ori[i] = 0
    dist[edge6_index(slot, ori)] = 0
    depth = 0
    count = 1
    M = etab.shape[0]
    s2 = np.zeros(6, dtype=np.int64)
    o2 = np.zeros(6, dtype=np.int64)
    while True:
        new = 0
        for idx in range(dist.shape[0]):
            if dist[idx] != depth:
                continue
            edge6_unindex(idx, slot, ori)
            for m in range(M):
                for i in range(6):
                    c = etab[m, slot[i] * 2 + ori[i]]
                    s2[i] = c >> 1
                    o2[i] = c & 1
                j = edge6_index(s2, o2)
                if dist[j] == 255:
                    dist[j] = depth + 1
                    new += 1
        if new == 0:
            break
        count += new
        depth += 1
    return count


@lru_cache(maxsize=2)
def databases(metric: str = "htm", verbose: bool = False):
    """(Cubies, tables, corner PDB, edge PDB A, edge PDB B), built once and cached on disk."""
    cb = Cubies(metric)
    tables = build_tables(cb)
    cpm, com, etab, _ = tables
    CACHE.mkdir(parents=True, exist_ok=True)
    out = []
    for name, size in (("corners", N_CP * N_CO), ("edgesA", N_EDGE6), ("edgesB", N_EDGE6)):
        path = CACHE / f"pdb_{metric}_{name}.npy"
        if path.exists():
            out.append(np.load(path, mmap_mode=None))
            continue
        t = time.time()
        dist = np.empty(size, dtype=np.uint8)
        if name == "corners":
            n = _bfs_corners(cpm, com, dist)
        else:
            home = np.arange(6) if name == "edgesA" else np.arange(6, 12)
            n = _bfs_edges(etab, home.astype(np.int64), dist)
        if verbose:
            print(f"built {path.name}: {n} states, max depth {int(dist.max())}, {time.time() - t:.0f}s")
        if n != size:
            raise RuntimeError(f"{name} database reached {n} of {size} states")
        np.save(path, dist)
        out.append(dist)
    return cb, tables, out[0], out[1], out[2]


# ------------------------------------------------------------------ IDA*
@nb.njit(cache=True)
def _edge6_codes(codes, off):
    """edge6_index of pieces off..off+5 straight from their (slot*2 + flip) codes."""
    used = 0
    r = 0
    o = 0
    for i in range(6):
        c = codes[off + i]
        s = c >> 1
        below = 0
        for t in range(s):
            if (used >> t) & 1:
                below += 1
        r = r * (12 - i) + (s - below)
        used |= 1 << s
        o = o * 2 + (c & 1)
    return r * 64 + o


@nb.njit(cache=True)
def _h(cp, co, codes, pc, pa, pb):
    h = pc[cp * 2187 + co]
    ha = pa[_edge6_codes(codes, 0)]
    hb = pb[_edge6_codes(codes, 6)]
    if ha > h:
        h = ha
    if hb > h:
        h = hb
    return h


@nb.njit(cache=True)
def _search(cp0, co0, codes0, last0, bound, cpm, com, etab, allowed, pc, pa, pb, path, nodes, limit, stop,
            shared):
    """One IDA* iteration from a root whose last move was ``last0``, iterative depth-first.
    Returns -1 (found, length in nodes[1]), -2 (node budget, counted across threads in
    ``shared[0]``), -3 (another thread found one) or the smallest f that exceeded ``bound``."""
    M = cpm.shape[0]
    D = bound + 1
    cps = np.zeros(D + 1, dtype=np.int64)
    cos = np.zeros(D + 1, dtype=np.int64)
    st = np.zeros((D + 1, 12), dtype=np.int64)
    nxt = np.zeros(D + 1, dtype=np.int64)
    last = np.zeros(D + 1, dtype=np.int64)
    cps[0], cos[0] = cp0, co0
    for i in range(12):
        st[0, i] = codes0[i]
    if _h(cp0, co0, codes0, pc, pa, pb) == 0:
        nodes[1] = 0
        return -1
    last[0] = last0
    best = 1 << 30
    g = 0
    while g >= 0:
        m = nxt[g]
        if m >= M:
            g -= 1
            continue
        nxt[g] = m + 1
        if not allowed[last[g], m]:
            continue
        cp = cpm[m, cps[g]]
        co = com[m, cos[g]]
        for i in range(12):
            st[g + 1, i] = etab[m, st[g, i]]
        h = _h(cp, co, st[g + 1], pc, pa, pb)
        nodes[0] += 1
        if nodes[0] & 4095 == 0:
            shared[0] += 4096  # racy across threads; only an approximate budget is needed
            if shared[0] > limit:
                return -2
            if stop[0]:
                return -3
        f = g + 1 + h
        if f > bound:
            if f < best:
                best = f
            continue
        path[g] = m
        if h == 0:
            nodes[1] = g + 1
            return -1
        g += 1
        cps[g], cos[g] = cp, co
        nxt[g] = 0
        last[g] = m
    return best


@nb.njit(cache=True, parallel=True)
def _search_par(cp0, co0, codes0, bound, cpm, com, etab, allowed, pc, pa, pb, paths, nodes, limit, stop, shared):
    """One iteration split over the root's moves (one subtree per thread)."""
    M = cpm.shape[0]
    res = np.full(M, 1 << 30, dtype=np.int64)
    for m in nb.prange(M):
        cp = cpm[m, cp0]
        co = com[m, co0]
        codes = np.empty(12, dtype=np.int64)
        for i in range(12):
            codes[i] = etab[m, codes0[i]]
        h = _h(cp, co, codes, pc, pa, pb)
        if 1 + h > bound:
            res[m] = 1 + h
            continue
        if h == 0:
            nodes[m, 1] = 0
            res[m] = -1
            stop[0] = 1
            continue
        r = _search(cp, co, codes, m, bound - 1, cpm, com, etab, allowed, pc, pa, pb, paths[m], nodes[m],
                    limit, stop, shared)
        if r == -1:
            stop[0] = 1
        res[m] = r + 1 if r >= 0 else r  # subtree f is one move short of the root's
    return res


def ida_star(state: np.ndarray, metric: str = "htm", node_limit: int = 10**12, time_limit: float | None = None):
    """Optimal face-turn solution of an oriented 3×3×3 colour state.
    Returns (engine move ids or None if a limit was hit, stats)."""
    cb, (cpm, com, etab, _), pc, pa, pb = databases(metric)
    cp, co, ep, eo = cb.from_colours(np.asarray(state))
    cpi = rank_perm(cp)
    coi = int(ori_index(co, 3))
    # piece-major edge codes: for piece k, (slot, flip)
    codes = np.zeros(12, dtype=np.int64)
    for j in range(12):
        codes[ep[j]] = j * 2 + eo[j]
    M = len(cb.moves)
    paths = np.zeros((M, 64), dtype=np.int64)
    nodes = np.zeros((M, 2), dtype=np.int64)
    if _h(cpi, coi, codes, pc, pa, pb) == 0:
        return [], {"nodes": 0, "bound": 0, "time_s": 0.0}
    bound = int(_h(cpi, coi, codes, pc, pa, pb))
    t0 = time.time()
    rate = 8e6  # nodes / s on 4 cores; refined from each long iteration
    total = 0
    while True:
        left = None if time_limit is None else time_limit - (time.time() - t0)
        if left is not None and left <= 0:
            return None, {"nodes": total, "bound": bound, "time_s": round(time.time() - t0, 3)}
        lim = node_limit - total if left is None else min(node_limit - total, int(rate * left) + 100000)
        nodes[:] = 0
        stop = np.zeros(1, dtype=np.int64)
        shared = np.zeros(1, dtype=np.int64)
        t1 = time.time()
        res = _search_par(cpi, coi, codes, bound, cpm, com, etab, cb.allowed, pc, pa, pb, paths, nodes,
                          lim, stop, shared)
        n = int(nodes[:, 0].sum())
        total += n
        dt = time.time() - t1
        if n > 10**6 and dt > 0.5:  # skip short iterations (and the first call's compile time)
            rate = n / dt
        stats = {"nodes": total, "bound": bound, "time_s": round(time.time() - t0, 3)}
        hit = np.nonzero(res == -1)[0]
        if len(hit):
            m = int(hit[0])
            L = int(nodes[m, 1])
            return [cb.moves[m]] + [cb.moves[int(x)] for x in paths[m, :L]], stats
        if (res == -2).any():
            left = None if time_limit is None else time_limit - (time.time() - t0)
            if total >= node_limit or left is None or left < 0.1 * time_limit:
                return None, stats
            rate *= 2  # budget estimate was too low: redo this bound with a larger one
            continue
        bound = int(res.min())
