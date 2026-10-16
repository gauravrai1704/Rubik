"""Macro actions: short move sequences offered to beam search as single steps.

A summed per-cluster heuristic has *hills*: a commutator that fixes one cluster first
breaks protected clusters and only then restores them, so plain beam search drops it.
Offering the commutator as one action turns the hill into a single downhill step.
This is the meta-action idea of Q* search (Agostinelli et al.) with Demaine's cluster
sequences as the action set. Every macro still expands to ordinary moves in the output.

* Square: for row r and column c, the commutator ``c r c r`` (Demaine 2011, Lemma 2 / 4:
  changes only the cluster at their crossing) and the conjugates ``r c r`` and ``c r c``
  (Lemma 3 / Fig. 2 edge sequences).
* Cube: commutators ``X Y X' Y'`` of an inner-slice quarter turn X with an outer face
  quarter turn Y on another axis. They never move corners (X does not touch them, Y is
  undone); when X is not the middle slice they also keep middle edges. So they suit the
  ``edges_first`` stage order: every later stage gets cheap actions that keep earlier
  work. ``inner_only=False`` also allows outer-layer X.
"""
from __future__ import annotations

from ..engine.puzzle import Move, PuzzleSpec


def square_macros(puzzle: PuzzleSpec, conjugates: bool = True) -> list[list[int]]:
    idx = puzzle.move_index
    N = puzzle.N
    out = []
    for r in range(N):
        for c in range(N):
            R, C = idx[Move(1, r, 2)], idx[Move(0, c, 2)]
            out.append([C, R, C, R])
            if conjugates:
                out.append([R, C, R])
                out.append([C, R, C])
    return out


def cube_macros(puzzle: PuzzleSpec, inner_only: bool = True) -> list[list[int]]:
    idx = puzzle.move_index
    N = puzzle.N
    faces = [(ax, l) for ax in range(3) for l in (0, N - 1)]
    out = []
    for ax in range(3):
        layers = range(1, N - 1) if inner_only else range(N)
        for l in layers:
            for t in (1, 3):
                X, Xi = idx[Move(ax, l, t)], idx[Move(ax, l, 4 - t)]
                for fa, fl in faces:
                    if fa == ax:
                        continue
                    for ft in (1, 3):
                        Y, Yi = idx[Move(fa, fl, ft)], idx[Move(fa, fl, 4 - ft)]
                        out.append([X, Y, Xi, Yi])
    return out


def cube_center_macros(puzzle: PuzzleSpec) -> list[list[int]]:
    """Pure center commutators [A, B C B'] = A B C B' A' B C' B' (e.g. r U l' U' r' U l U').

    A: inner-slice quarter turn (axis a, layer k); B: outer face quarter turn on another axis;
    C: inner-slice quarter turn on axis a at layer k or N-1-k. Each candidate is checked
    and kept only if it moves center stickers alone (corners, edges and wings untouched)."""
    import numpy as np

    from ..engine.orbits import clusters

    N = puzzle.N
    if N < 4:
        return []
    idx = puzzle.move_index
    S = puzzle.num_stickers
    center = np.zeros(S, dtype=bool)
    for c in clusters(puzzle):
        if c.kind.startswith("center_") and c.kind != "center_fixed":
            center[c.stickers] = True
    faces = [(ax, l) for ax in range(3) for l in (0, N - 1)]
    out, seen = [], set()
    for a in range(3):
        for k in range(1, N - 1):
            if N % 2 == 1 and k == N // 2:
                continue  # middle slice moves middle edges
            for ta in (1, 3):
                A, Ai = idx[Move(a, k, ta)], idx[Move(a, k, 4 - ta)]
                for fa, fl in faces:
                    if fa == a:
                        continue
                    for tb in (1, 3):
                        Bm, Bi = idx[Move(fa, fl, tb)], idx[Move(fa, fl, 4 - tb)]
                        for j in sorted({k, N - 1 - k}):
                            for tc in (1, 3):
                                C, Ci = idx[Move(a, j, tc)], idx[Move(a, j, 4 - tc)]
                                seq = [A, Bm, C, Bi, Ai, Bm, Ci, Bi]
                                q = np.arange(S)
                                for m in seq:
                                    q = q[puzzle.move_perms[m]]
                                moved = q != np.arange(S)
                                key = q.tobytes()
                                if moved.any() and not moved[~center].any() and key not in seen:
                                    seen.add(key)
                                    out.append(seq)
    return out


def default_macros(puzzle: PuzzleSpec) -> list[list[int]]:
    if puzzle.kind == "square":
        return square_macros(puzzle)
    if puzzle.kind == "cube":
        return cube_macros(puzzle) + cube_center_macros(puzzle)
    return []
