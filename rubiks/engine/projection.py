"""Projected puzzles: look only at a subset of stickers (one cluster, or a few).

The moves still act on the subset, everything else is ignored. The projected puzzle is
small enough for exact breadth-first search in many cases (every Rubik's Square
cluster, 3×3×3 fixed centers, ...). The resulting distance table is an exact lower bound
on the moves needed for that subset, and its reachable set is an exact solvability test
for it. Used by the checker (Section 3.3) and by Method 5B's per-cluster heuristics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_ENCODABLE_BITS = 62


def can_encode(K: int, base: int) -> bool:
    return K * np.log2(base) <= MAX_ENCODABLE_BITS


def encode(states: np.ndarray, base: int) -> np.ndarray:
    """(n, K) small ints -> (n,) int64 codes (base-``base`` number)."""
    states = np.asarray(states, dtype=np.int64)
    code = np.zeros(states.shape[0], dtype=np.int64)
    for k in range(states.shape[1]):
        code = code * base + states[:, k]
    return code


@dataclass
class ProjectionTable:
    """Exact distances for every reachable state of a projected puzzle."""

    codes: np.ndarray  # sorted int64 codes of reachable states
    dist: np.ndarray  # distance to solved (in moves of the generator set), aligned with codes
    base: int
    K: int

    @property
    def size(self) -> int:
        return len(self.codes)

    @property
    def max_dist(self) -> int:
        return int(self.dist.max())

    def lookup(self, states: np.ndarray) -> np.ndarray:
        """Distance for each row; -1 if the row is not reachable."""
        c = encode(states, self.base)
        i = np.searchsorted(self.codes, c)
        i = np.clip(i, 0, len(self.codes) - 1)
        hit = self.codes[i] == c
        return np.where(hit, self.dist[i], -1)

    def contains(self, states: np.ndarray) -> np.ndarray:
        return self.lookup(states) >= 0


def bfs_projection(
    gen_perms: np.ndarray, solved: np.ndarray, base: int = 6, cap: int = 2_000_000, chunk: int = 20_000
) -> ProjectionTable | None:
    """Breadth-first search of a projected puzzle from its solved pattern.

    ``gen_perms`` (G, K): new = state[gen_perms[g]]. Returns None if the reachable set is
    larger than ``cap`` or cannot be encoded in 62 bits. The frontier is expanded in chunks
    so memory stays at O(chunk · G · K)."""
    K = len(solved)
    if not can_encode(K, base):
        return None
    start = np.asarray(solved, dtype=np.int64)[None, :]
    seen_codes = [encode(start, base)]
    seen_dist = [np.zeros(1, dtype=np.int16)]
    seen_sorted = np.sort(seen_codes[0])
    frontier = start.astype(np.int8)
    depth = 0
    total = 1
    while len(frontier):
        depth += 1
        level_codes, level_states = [], []
        for a in range(0, len(frontier), chunk):
            kids = frontier[a : a + chunk][:, gen_perms].reshape(-1, K)  # (c*G, K)
            codes = encode(kids, base)
            codes, first = np.unique(codes, return_index=True)
            new = ~np.isin(codes, seen_sorted)
            level_codes.append(codes[new])
            level_states.append(kids[first[new]])
        codes = np.concatenate(level_codes)
        kids = np.concatenate(level_states)
        codes, first = np.unique(codes, return_index=True)
        kids = kids[first]
        if len(codes) == 0:
            break
        total += len(codes)
        if total > cap:
            return None
        seen_codes.append(codes)
        seen_dist.append(np.full(len(codes), depth, dtype=np.int16))
        seen_sorted = np.union1d(seen_sorted, codes)
        frontier = kids
    codes = np.concatenate(seen_codes)
    dist = np.concatenate(seen_dist)
    order = np.argsort(codes)
    return ProjectionTable(codes=codes[order], dist=dist[order], base=base, K=K)


_TABLE_CACHE: dict[tuple, ProjectionTable | None] = {}


def cached_table(signature_key: tuple, gen_perms: np.ndarray, solved: np.ndarray, base: int = 6,
                 cap: int = 2_000_000) -> ProjectionTable | None:
    """Memoised :func:`bfs_projection`, keyed by an isomorphism signature."""
    key = (signature_key, base, cap)
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = bfs_projection(gen_perms, solved, base, cap)
    return _TABLE_CACHE[key]
