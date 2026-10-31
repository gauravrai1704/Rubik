"""Solving a rotated copy of a state and mapping the answer back.

For a whole-cube rotation R (a sticker permutation from ``PuzzleSpec.rotations``) the state
``s' = sigma(s[R])``, where ``sigma`` relabels colours so that ``sigma(solved[R]) = solved``, is the same
puzzle seen from another side, with the same exact goal. A solution of ``s'`` maps back to a solution of
``s`` move by move (a rotation sends every slice turn to a slice turn), with the same length. A
deterministic solver gives a different answer on each of the 24 copies, so the copies are a free source
of diverse runs (a "symmetry ensemble"), even for a program whose random seed is fixed.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..engine import PuzzleSpec
from ..engine.puzzle import Move


def rotated(P: PuzzleSpec, state: np.ndarray, r: int) -> np.ndarray:
    R = P.rotations[r]
    src, dst = P.solved[R], P.solved
    sigma = np.empty(max(int(P.solved.max()), int(src.max())) + 1, dtype=np.int64)
    sigma[src] = dst
    return sigma[np.asarray(state)[R]].astype(np.asarray(state).dtype)


@lru_cache(maxsize=64)
def _move_map(P: PuzzleSpec, r: int) -> tuple[int, ...]:
    """Engine move j on the rotated copy == engine move out[j] on the original."""
    R = P.rotations[r]
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
