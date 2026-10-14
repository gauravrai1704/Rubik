"""Orbits (Demaine's "clusters"): sets of sticker positions that moves only permute among
themselves. Computed automatically from the move permutations, then classified by
geometry and given an N-invariant sticker order, so that the same *type* of orbit looks
identical at every N. Methods 3, 5A and 5B are built on this.

Each orbit gets:
* ``kind``: e.g. ``corner``, ``midge``, ``wing``, ``center_x``, ``center_plus``,
  ``center_oblique``, ``center_fixed`` (cube) or ``sq_corner``, ``sq_edge``,
  ``sq_edge_cross``, ``sq_interior``, ``sq_cross``, ``sq_center`` (square).
* ``stickers``: sticker indices in a canonical order that does not depend on N.
* ``depth``: geometric descriptor in [0, 1] (how far in from the outside the orbit sits).
* ``generators``: the moves that act on it, and the permutation each induces on the
  orbit's stickers. Orbits with the same ``signature`` are the same puzzle, whatever N.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .puzzle import PuzzleSpec

CUBE_KINDS = ("corner", "midge", "wing", "center_fixed", "center_plus", "center_x", "center_oblique")
SQUARE_KINDS = ("sq_corner", "sq_edge", "sq_edge_cross", "sq_interior", "sq_cross", "sq_center")
ALL_KINDS = CUBE_KINDS + SQUARE_KINDS


@dataclass
class Orbit:
    index: int
    kind: str
    stickers: np.ndarray  # canonical order
    depth: tuple[float, float]  # sorted |in-face coords| / (N-1)
    gen_moves: np.ndarray  # indices into puzzle.moves that act on this orbit
    gen_perms: np.ndarray  # (G, K) local permutations, new_local = local[gen_perms[g]]
    solved: np.ndarray  # (K,) solved labels on this orbit

    @property
    def size(self) -> int:
        return len(self.stickers)

    @property
    def signature(self) -> str:
        """Hash of (generator set, solved pattern): equal signature = isomorphic puzzle."""
        rows = sorted(tuple(int(x) for x in p) for p in self.gen_perms)
        h = hashlib.sha1()
        h.update(repr((self.kind, rows, tuple(int(x) for x in self.solved))).encode())
        return h.hexdigest()[:16]


def _classify(puzzle: PuzzleSpec, s: int) -> str:
    d = np.array(puzzle.dims)
    c = puzzle.sticker_pos[s]
    big = [a for a in range(3) if d[a] > 1]
    extreme = [a for a in big if abs(c[a]) == d[a] - 1]
    inner = [a for a in big if a not in extreme]
    if puzzle.kind == "square":
        if len(extreme) == 2:
            return "sq_corner"
        if len(extreme) == 1:
            return "sq_edge_cross" if c[inner[0]] == 0 else "sq_edge"
        zeros = sum(1 for a in inner if c[a] == 0)
        return {0: "sq_interior", 1: "sq_cross", 2: "sq_center"}[zeros]
    if len(extreme) == 3:
        return "corner"
    if len(extreme) == 2:
        return "midge" if c[inner[0]] == 0 else "wing"
    u, v = (abs(c[a]) for a in inner)
    if u == 0 and v == 0:
        return "center_fixed"
    if u == 0 or v == 0:
        return "center_plus"
    if u == v:
        return "center_x"
    return "center_oblique"


def _sticker_key(puzzle: PuzzleSpec, s: int) -> tuple:
    """N-invariant ordering key: face, then signs and relative magnitudes of the in-face
    coordinates, then for edge/corner stickers the signs of the cubie's other coordinates."""
    d = np.array(puzzle.dims)
    f = int(puzzle.sticker_face[s])
    ax = int(np.nonzero(puzzle.sticker_normal[s])[0][0])
    c = puzzle.sticker_pos[s]
    other = [a for a in range(3) if a != ax]
    u, v = int(c[other[0]]), int(c[other[1]])
    ext_u = abs(u) == d[other[0]] - 1 and d[other[0]] > 1
    ext_v = abs(v) == d[other[1]] - 1 and d[other[1]] > 1
    return (f, int(np.sign(u)), int(np.sign(v)), ext_u, ext_v, abs(u) > abs(v), abs(u) == abs(v))


def _build(puzzle: PuzzleSpec, members: list[int], kind: str, order: list[int] | None = None) -> Orbit:
    """Make an Orbit from sticker indices. ``order`` (already canonical) skips sorting."""
    d = np.array(puzzle.dims)
    if order is None:
        keys = [_sticker_key(puzzle, s) for s in members]
        idx = sorted(range(len(members)), key=lambda k: (keys[k], tuple(puzzle.sticker_pos[members[k]])))
        stickers = np.array([members[k] for k in idx])
    else:
        stickers = np.array(order)
    local = {int(s): i for i, s in enumerate(stickers)}
    gmoves, gperms = [], []
    perms = puzzle.move_perms
    for mi in range(len(perms)):
        sub = perms[mi][stickers]
        if np.any(sub != stickers):
            gmoves.append(mi)
            gperms.append([local[int(x)] for x in sub])
    c = np.abs(puzzle.sticker_pos[stickers[0]])
    ax = int(np.nonzero(puzzle.sticker_normal[stickers[0]])[0][0])
    inface = sorted(float(c[a]) / max(d[a] - 1, 1) for a in range(3) if a != ax and d[a] > 1)
    while len(inface) < 2:
        inface.insert(0, 0.0)
    return Orbit(
        index=0,
        kind=kind,
        stickers=stickers,
        depth=(inface[0], inface[1]),
        gen_moves=np.array(gmoves, dtype=np.int64),
        gen_perms=np.array(gperms, dtype=np.int64).reshape(len(gmoves), len(stickers)),
        solved=puzzle.solved[stickers].copy(),
    )


def _sort_and_index(puzzle: PuzzleSpec, items: list[Orbit]) -> tuple:
    kind_rank = {k: i for i, k in enumerate(ALL_KINDS)}
    items.sort(
        key=lambda o: (kind_rank[o.kind], -o.depth[1], -o.depth[0], _sticker_key(puzzle, int(o.stickers[0])))
    )
    for i, o in enumerate(items):
        o.index = i
    return tuple(items)


@lru_cache(maxsize=64)
def _orbits_cached(dims: tuple, labeling: str) -> tuple:
    puzzle = PuzzleSpec(dims, labeling)
    S = puzzle.num_stickers
    parent = np.arange(S)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in puzzle.move_perms:
        for i in range(S):
            a, b = find(i), find(int(p[i]))
            if a != b:
                parent[a] = b
    groups: dict[int, list[int]] = {}
    for i in range(S):
        groups.setdefault(int(find(i)), []).append(i)
    items = [_build(puzzle, m, _classify(puzzle, m[0])) for m in groups.values()]
    return _sort_and_index(puzzle, items)


@lru_cache(maxsize=64)
def _clusters_cached(dims: tuple, labeling: str) -> tuple:
    """Merge sticker orbits that share cubies into cubie clusters (Demaine's clusters):
    e.g. the two sticker orbits of one wing layer, or the top / bottom / side sticker
    orbits of one Rubik's Square cluster."""
    puzzle = PuzzleSpec(dims, labeling)
    orbs = _orbits_cached(dims, labeling)
    cubie_of = {}
    for o in orbs:
        for s in o.stickers:
            cubie_of.setdefault(tuple(puzzle.sticker_pos[s]), set()).add(o.index)
    parent = list(range(len(orbs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ids in cubie_of.values():
        ids = sorted(ids)
        for j in ids[1:]:
            a, b = find(ids[0]), find(j)
            if a != b:
                parent[b] = a
    groups: dict[int, list[Orbit]] = {}
    for o in orbs:
        groups.setdefault(find(o.index), []).append(o)
    items = []
    for subs in groups.values():
        subs = sorted(subs, key=lambda o: _sticker_key(puzzle, int(o.stickers[0])))
        order = [int(s) for o in subs for s in o.stickers]
        items.append(_build(puzzle, order, subs[0].kind, order=order))
    return _sort_and_index(puzzle, items)


def orbits(puzzle: PuzzleSpec) -> list[Orbit]:
    """All orbits of a puzzle, sorted by kind then from the outside inward."""
    return list(_orbits_cached(puzzle.dims, puzzle.labeling))


def clusters(puzzle: PuzzleSpec) -> list[Orbit]:
    """Cubie clusters: unions of sticker orbits that move the same cubies. This is the unit
    Methods 3, 5A and 5B reason about (24 wings, 4 Square cubies, ...)."""
    return list(_clusters_cached(puzzle.dims, puzzle.labeling))


def orbit_summary(puzzle: PuzzleSpec, level: str = "orbit") -> dict[str, int]:
    out: dict[str, int] = {}
    for o in (orbits(puzzle) if level == "orbit" else clusters(puzzle)):
        out[o.kind] = out.get(o.kind, 0) + 1
    return out
