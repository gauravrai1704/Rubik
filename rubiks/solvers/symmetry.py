"""Solving a rotated copy of a state and mapping the answer back.

For a whole-cube rotation R (a sticker permutation from ``PuzzleSpec.rotations``) the state
``s' = sigma(s[R])``, where ``sigma`` relabels colours so that ``sigma(solved[R]) = solved``, is the same
puzzle seen from another side, with the same exact goal. A solution of ``s'`` maps back to a solution of
``s`` move by move (a rotation sends every slice turn to a slice turn), with the same length. A
deterministic solver gives a different answer on each of the 24 copies, so the copies are a free source
of diverse runs (a "symmetry ensemble"), even for a program whose random seed is fixed.

Mirror images give 24 more: reflecting the cube (x -> -x) maps the state to a legal state of the mirrored
puzzle, and every slice quarter turn to the mirrored slice's opposite quarter turn, so a solution maps
back with the same length too. ``symmetries(P)`` lists all 48 (rotations first, identity at index 0).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..engine import PuzzleSpec
from ..engine.puzzle import Move


@lru_cache(maxsize=16)
def symmetries(P: PuzzleSpec) -> np.ndarray:
    """Sticker permutations of the 24 rotations and, for cubes, their 24 mirror images."""
    S = P.num_stickers
    ident = np.arange(S)
    rots = [r for r in P.rotations if not np.array_equal(r, ident)]
    out = [ident] + rots
    if P.kind == "cube":
        look = {(tuple(int(v) for v in p), tuple(int(v) for v in n)): i
                for i, (p, n) in enumerate(zip(P.sticker_pos, P.sticker_normal))}
        flip = np.array([1 if a else -1 for a in (False, True, True)])
        M = np.array([look[(tuple(int(v) for v in p * flip), tuple(int(v) for v in n * flip))]
                      for p, n in zip(P.sticker_pos, P.sticker_normal)])
        out += [M[r] for r in out]
    return np.array(out)


def rotated(P: PuzzleSpec, state: np.ndarray, r: int) -> np.ndarray:
    """The state seen through symmetry ``r`` (an index into :func:`symmetries`)."""
    R = symmetries(P)[r]
    src, dst = P.solved[R], P.solved
    sigma = np.empty(max(int(P.solved.max()), int(src.max())) + 1, dtype=np.int64)
    sigma[src] = dst
    return sigma[np.asarray(state)[R]].astype(np.asarray(state).dtype)


@lru_cache(maxsize=64)
def _move_map(P: PuzzleSpec, r: int) -> tuple[int, ...]:
    """Engine move j on the rotated copy == engine move out[j] on the original."""
    R = symmetries(P)[r]
    Rinv = np.argsort(R)
    look = {p.tobytes(): i for i, p in enumerate(P.move_perms.astype(np.int64))}
    out = []
    for q in P.move_perms.astype(np.int64):
        # s'[q] = sigma(s[R][q]); we need p with s[p] seen through R: R[q] = p[R]  =>  p = R[q][Rinv]
        p = R[q][Rinv]
        out.append(look[p.tobytes()])
    return tuple(out)


def map_back(P: PuzzleSpec, moves: list[Move], r: int) -> list[Move]:
    mm = _move_map(P, r)
    idx = P.move_index
    return [P.moves[mm[idx[m]]] for m in moves]
