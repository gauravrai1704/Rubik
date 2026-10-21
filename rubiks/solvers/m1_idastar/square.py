"""Rubik's Square for Method 1 (technical basis, Sections 4.1-4.3).

* :func:`gods_number` — breadth-first search of the whole state space from solved (small N):
  the exact distance distribution, hence God's number. Distances are to the reference
  solved state; states that differ by a whole-Square rotation are counted separately.
* :func:`ida_star` — IDA* with h = max over clusters of that cluster's exact distance
  (each cluster's table from :func:`rubiks.engine.projection.bfs_projection`), admissible
  because no solution fixes a cluster in fewer moves than it needs alone. Pruning: never the
  same move twice, one order for commuting moves (Section 3.2 canonical form). The inner
  loop is Numba over the sticker array.
"""
from __future__ import annotations

import time
from functools import lru_cache

import numba as nb
import numpy as np

from ...engine.moves import canonical_successor_mask
from ...engine.orbits import clusters
from ...engine.projection import bfs_projection
from ...engine.puzzle import square


@lru_cache(maxsize=32)
def cluster_tables(N: int):
    """Concatenated per-cluster lookup data for the numba heuristic."""
    P = square(N)
    st_idx, st_off, codes, dists, tab_off = [], [0], [], [], [0]
    for c in clusters(P):
        t = bfs_projection(c.gen_perms, c.solved, base=6, cap=5_000_000)
        if t is None:
            raise RuntimeError(f"cluster {c.index} table too large")
        st_idx += [int(s) for s in c.stickers]
        st_off.append(len(st_idx))
        codes.append(t.codes)
        dists.append(t.dist.astype(np.int8))
        tab_off.append(tab_off[-1] + len(t.codes))
    return (np.array(st_idx, dtype=np.int64), np.array(st_off, dtype=np.int64), np.concatenate(codes),
            np.concatenate(dists), np.array(tab_off, dtype=np.int64))


@nb.njit(cache=True)
def _h(state, st_idx, st_off, codes, dists, tab_off):
    best = 0
    for c in range(len(st_off) - 1):
        code = 0
        for k in range(st_off[c], st_off[c + 1]):
            code = code * 6 + state[st_idx[k]]
        lo, hi = tab_off[c], tab_off[c + 1]
        i = np.searchsorted(codes[lo:hi], code)
        d = dists[lo + i]
        if d > best:
            best = d
    return best


@nb.njit(cache=True)
def _dfs(state, g, bound, last, perms, allowed, st_idx, st_off, codes, dists, tab_off, path, nodes, limit):
    h = _h(state, st_idx, st_off, codes, dists, tab_off)
    f = g + h
    if f > bound:
        return f
    if h == 0:
        nodes[1] = g
        return -1
    nodes[0] += 1
    if nodes[0] > limit:
        return -2
    best = 1 << 30
    child = np.empty_like(state)
    for m in range(perms.shape[0]):
        if not allowed[last, m]:
            continue
        p = perms[m]
        for i in range(len(state)):
            child[i] = state[p[i]]
        path[g] = m
        t = _dfs(child, g + 1, bound, m, perms, allowed, st_idx, st_off, codes, dists, tab_off, path, nodes, limit)
        if t == -1 or t == -2:
            return t
        if t < best:
            best = t
    return best


def ida_star(N: int, state: np.ndarray, time_limit: float = 60.0, node_limit: int = 10**12):
    """Optimal solution (engine move ids to the reference solved state), or None on a limit."""
    P = square(N)
    tabs = cluster_tables(N)
    perms = P.move_perms.astype(np.int64)
    allowed = canonical_successor_mask(P)
    s = np.asarray(state).astype(np.int64)
    path = np.zeros(256, dtype=np.int64)
    nodes = np.zeros(2, dtype=np.int64)
    bound = int(_h(s, *tabs))
    t0 = time.time()
    rate = 2e5  # nodes / s, refined after each iteration
    while True:
        left = time_limit - (time.time() - t0)
        if left <= 0:
            return None, {"nodes": int(nodes[0]), "bound": bound, "time_s": time.time() - t0}
        before, t1 = int(nodes[0]), time.time()
        lim = min(node_limit, before + int(rate * left) + 1000)
        t = _dfs(s, 0, bound, len(P.moves), perms, allowed, *tabs, path, nodes, lim)
        dt = time.time() - t1
        if dt > 0.05:
            rate = max(rate, (int(nodes[0]) - before) / dt)
        if t == -1:
            return [int(m) for m in path[: int(nodes[1])]], {"nodes": int(nodes[0]), "bound": bound,
                                                             "time_s": time.time() - t0}
        if t == -2:
            if int(nodes[0]) >= node_limit or time.time() - t0 >= time_limit:
                return None, {"nodes": int(nodes[0]), "bound": bound, "time_s": time.time() - t0}
            continue  # rate estimate was low: keep searching this bound
        bound = t


def gods_number(N: int, cap: int = 50_000_000) -> dict:
    """Full BFS from solved. Returns {"states", "distribution", "gods_number"} or raises past cap."""
    P = square(N)
    perms = P.move_perms
    start = P.solved.astype(np.int8)
    seen = {start.tobytes()}
    frontier = start[None]
    dist = [1]
    while len(frontier):
        kids = frontier[:, perms].reshape(-1, P.num_stickers)
        new = []
        for row in np.unique(kids, axis=0):
            b = row.tobytes()
            if b not in seen:
                seen.add(b)
                new.append(row)
        if len(seen) > cap:
            raise RuntimeError(f"square{N} has more than {cap} states")
        if not new:
            break
        dist.append(len(new))
        frontier = np.array(new)
    return {"N": N, "states": len(seen), "distribution": dist, "gods_number": len(dist) - 1}
