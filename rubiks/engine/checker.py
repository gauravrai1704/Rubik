"""Solvability checker (technical basis, Section 3.3).

``is_reachable(puzzle, state)`` returns (ok, reason). It combines:

1. Colour counts: globally and per cubie cluster, they must match the solved state.
2. Exact projection tests: every cluster whose projected puzzle is small enough for BFS
   (all Rubik's Square clusters, fixed centers, ...) must be in its reachable set.
3. Cube piece rules: valid corner / middle-edge / wing pieces, each present once; corner
   twist sum ≡ 0 mod 3; for odd N (orientation fixed by the centers) middle-edge flip sum
   even and corner parity equal to middle-edge parity; wings in their own handedness.
4. Square coupling: each cross cluster is checked jointly with the boundary clusters it
   shares moves with (Demaine 2011, Lemma 3).

The checks are necessary conditions derived from the group structure; the property tests
in ``tests/test_checker.py`` confirm they accept every scrambled state and reject
standard unreachable ones. Treat a rejection as authoritative and an acceptance as
"passes every known constraint".
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .orbits import clusters
from .projection import cached_table, can_encode
from .puzzle import PuzzleSpec

U, R, F, D, L, B = range(6)


def _parity(perm: list[int]) -> int:
    seen = [False] * len(perm)
    p = 0
    for i in range(len(perm)):
        if not seen[i]:
            j, n = i, 0
            while not seen[j]:
                seen[j] = True
                j = perm[j]
                n += 1
            p ^= (n - 1) & 1
    return p


@lru_cache(maxsize=32)
def _cubie_groups(dims: tuple) -> dict[str, list[list[int]]]:
    """Sticker indices of each corner, middle-edge and wing cubie, keyed by kind."""
    puzzle = PuzzleSpec(dims)
    by_pos: dict[tuple, list[int]] = {}
    for s in range(puzzle.num_stickers):
        by_pos.setdefault(tuple(puzzle.sticker_pos[s]), []).append(s)
    out: dict[str, list[list[int]]] = {"corner": [], "midge": [], "wing": []}
    kind_of = {}
    for c in clusters(puzzle):
        for s in c.stickers:
            kind_of[int(s)] = c.kind
    for pos, ss in sorted(by_pos.items()):
        k = kind_of[ss[0]]
        if k in out:
            out[k].append(ss)
    return out


def _corner_order(puzzle: PuzzleSpec, ss: list[int]) -> list[int]:
    """Right-handed cyclic order of a corner's stickers, starting from its y-face sticker."""
    n = puzzle.sticker_normal
    y = [s for s in ss if n[s][1] != 0][0]
    a, b = [s for s in ss if s != y]
    if np.linalg.det(np.stack([n[y], n[a], n[b]])) > 0:
        return [y, a, b]
    return [y, b, a]


def _wing_order(puzzle: PuzzleSpec, ss: list[int]) -> list[int]:
    n = puzzle.sticker_normal
    d = np.array(puzzle.dims)
    c = puzzle.sticker_pos[ss[0]]
    inner = [a for a in range(3) if abs(c[a]) != d[a] - 1][0]
    e = np.zeros(3)
    e[inner] = np.sign(c[inner])
    a, b = ss
    return [a, b] if np.dot(np.cross(n[a], n[b]), e) > 0 else [b, a]


def _midge_order(puzzle: PuzzleSpec, ss: list[int]) -> list[int]:
    """Priority sticker first: the y-face one, else the z-face one."""
    n = puzzle.sticker_normal
    pri = [s for s in ss if n[s][1] != 0] or [s for s in ss if n[s][2] != 0]
    return [pri[0]] + [s for s in ss if s != pri[0]]


def _normalize_orientation(puzzle: PuzzleSpec, colors: np.ndarray) -> np.ndarray | None:
    """Odd N: rotate the whole cube so every fixed center shows its home colour."""
    fixed = [c for c in clusters(puzzle) if c.kind == "center_fixed"]
    if not fixed:
        return colors
    st = fixed[0].stickers
    for rot in puzzle.rotations:
        r = colors[rot]
        if np.array_equal(r[st], puzzle.sticker_face[st]):
            return r
    return None


def _cube_piece_rules(puzzle: PuzzleSpec, colors: np.ndarray) -> str | None:
    groups = _cubie_groups(puzzle.dims)
    solved = puzzle.sticker_face
    odd = puzzle.N % 2 == 1
    if odd:
        colors = _normalize_orientation(puzzle, colors)
        if colors is None:
            return "fixed centers are not six distinct colours in a valid arrangement"

    # corners
    corner_parity = None
    if groups["corner"]:
        orders = [_corner_order(puzzle, ss) for ss in groups["corner"]]
        home = {frozenset(int(solved[s]) for s in o): i for i, o in enumerate(orders)}
        perm, twist = [], 0
        for o in orders:
            cs = [int(colors[s]) for s in o]
            key = frozenset(cs)
            if key not in home or len(set(cs)) != 3:
                return f"invalid corner colours {cs}"
            # the piece must also keep its colours in the right cyclic order
            ref = [int(solved[s]) for s in orders[home[key]]]
            if cs not in ([ref[k:] + ref[:k]][0] for k in range(3)):
                return f"corner {cs} is mirrored (stickers swapped)"
            perm.append(home[key])
            twist += [k for k, c in enumerate(cs) if c in (U, D)][0]
        if sorted(perm) != list(range(len(orders))):
            return "a corner piece appears twice"
        if twist % 3:
            return "corner twist sum is not 0 mod 3"
        corner_parity = _parity(perm)

    # middle edges (odd N)
    if groups["midge"]:
        orders = [_midge_order(puzzle, ss) for ss in groups["midge"]]
        home = {frozenset(int(solved[s]) for s in o): i for i, o in enumerate(orders)}
        perm, flip = [], 0
        for o in orders:
            cs = [int(colors[s]) for s in o]
            key = frozenset(cs)
            if key not in home or len(set(cs)) != 2:
                return f"invalid middle edge {cs}"
            perm.append(home[key])
            pri = [c for c in cs if c in (U, D)] or [c for c in cs if c in (F, B)]
            flip += 0 if cs[0] == pri[0] else 1
        if sorted(perm) != list(range(len(orders))):
            return "a middle edge appears twice"
        if flip % 2:
            return "middle-edge flip sum is odd"
        if corner_parity is not None and _parity(perm) != corner_parity:
            return "corner parity differs from middle-edge parity"

    # wings: each layer's 24 wings, ordered by handedness, must each appear once
    if groups["wing"]:
        by_layer: dict[int, list[list[int]]] = {}
        for ss in groups["wing"]:
            c = puzzle.sticker_pos[ss[0]]
            d = np.array(puzzle.dims)
            inner = [a for a in range(3) if abs(c[a]) != d[a] - 1][0]
            by_layer.setdefault(int(abs(c[inner])), []).append(ss)
        for layer, wings in by_layer.items():
            orders = [_wing_order(puzzle, ss) for ss in wings]
            want = sorted(tuple(int(solved[s]) for s in o) for o in orders)
            have = sorted(tuple(int(colors[s]) for s in o) for o in orders)
            if want != have:
                return f"wing layer {layer}: pieces missing, duplicated or flipped"
    return None


def _square_coupling(puzzle: PuzzleSpec, colors: np.ndarray) -> str | None:
    cl = clusters(puzzle)
    boundary = [c for c in cl if c.kind in ("sq_edge", "sq_edge_cross", "sq_corner")]
    for cr in (c for c in cl if c.kind == "sq_cross"):
        moves = set(cr.gen_moves.tolist())
        nb = [c for c in boundary if moves & set(c.gen_moves.tolist())]
        stickers = np.concatenate([cr.stickers] + [c.stickers for c in nb])
        if not can_encode(len(stickers), 6):
            continue
        idx = {int(s): i for i, s in enumerate(stickers)}
        allm = sorted(set().union(*[set(c.gen_moves.tolist()) for c in [cr] + nb]))
        gp = np.array([[idx[int(x)] for x in puzzle.move_perms[m][stickers]] for m in allm])
        key = ("sqjoint", puzzle.dims, int(cr.index))
        table = cached_table(key, gp, puzzle.sticker_face[stickers])
        if table is not None and not table.contains(colors[stickers][None, :])[0]:
            return f"cross cluster {cr.index} inconsistent with its boundary clusters (parity)"
    return None


def is_reachable(puzzle: PuzzleSpec, state: np.ndarray) -> tuple[bool, str]:
    state = np.asarray(state)
    if state.shape != (puzzle.num_stickers,):
        return False, f"expected {puzzle.num_stickers} stickers, got shape {state.shape}"
    if puzzle.labeling == "sticker":
        if sorted(state.tolist()) != list(range(puzzle.num_stickers)):
            return False, "sticker labels are not a permutation"
    colors = puzzle.colors_of(state).astype(np.int64)
    solved = puzzle.sticker_face.astype(np.int64)
    if colors.min() < 0 or colors.max() > 5:
        return False, "colour labels must be 0..5"
    if not np.array_equal(np.bincount(colors, minlength=6), np.bincount(solved, minlength=6)):
        return False, "global colour counts are wrong"
    for c in clusters(puzzle):
        a = np.bincount(colors[c.stickers], minlength=6)
        b = np.bincount(solved[c.stickers], minlength=6)
        if not np.array_equal(a, b):
            return False, f"{c.kind} cluster {c.index}: colour counts are wrong"
    for c in clusters(puzzle):
        if c.kind in ("corner", "midge", "wing"):
            continue  # handled by piece rules (projections too large for BFS)
        if not can_encode(c.size, 6):
            continue
        table = cached_table((c.signature,), c.gen_perms, c.solved.astype(np.int64), cap=300_000)
        if table is not None and not table.contains(colors[c.stickers][None, :])[0]:
            return False, f"{c.kind} cluster {c.index}: configuration not reachable"
    if puzzle.kind == "cube":
        r = _cube_piece_rules(puzzle, colors)
        if r:
            return False, r
    elif puzzle.kind == "square":
        r = _square_coupling(puzzle, colors)
        if r:
            return False, r
    return True, "ok"
