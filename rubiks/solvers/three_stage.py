"""The 3×3×3 stage shared by Methods 2 and 3 (technical basis, Section 5.1 step 4).

Corners, middle edges (odd N) and fixed centers of an N×N×N cube behave as a 3×3×3
under outer-face turns, which also keep every center cluster on its face. This stage reads
them into Kociemba's 54-facelet string, solves with the two-phase algorithm (``kociemba``
package) and maps the answer back to outer-layer engine moves.

Even N has no middle edges: the edges are passed as solved, so the corners must be an even
permutation; an odd one (PLL parity) is fixed first with one outer quarter turn. Wings are
not this stage's business: Methods 2 and 3 solve them afterwards with pure 3-cycles.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..engine.moves import parse_notation
from ..engine.puzzle import Move, PuzzleSpec
from .cluster_tools import parity, piece_permutation

# Kociemba facelet order: U R F D L B, each 3x3 row-major in the standard net.
# face -> (outward normal, "right" in the net, "down" in the net), axes x=R, y=U, z=F
_NET = {
    "U": ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
    "R": ((1, 0, 0), (0, 0, -1), (0, -1, 0)),
    "F": ((0, 0, 1), (1, 0, 0), (0, -1, 0)),
    "D": ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
    "L": ((-1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "B": ((0, 0, -1), (-1, 0, 0), (0, -1, 0)),
}


@lru_cache(maxsize=32)
def facelet_stickers(dims: tuple) -> np.ndarray:
    """(54,) big-cube sticker for each Kociemba facelet; -1 for an edge facelet on even N."""
    P = PuzzleSpec(dims)
    N = P.N
    look = {(tuple(p), tuple(n)): i for i, (p, n) in enumerate(zip(P.sticker_pos, P.sticker_normal))}
    out = []
    for f in "URFDLB":
        n, r, d = (np.array(v) for v in _NET[f])
        for row in range(3):
            for col in range(3):
                # 3x3 offsets -1, 0, +1 scale to the big cube's extreme / middle coordinates
                off = (col - 1) * r + (row - 1) * d
                pos = n * (N - 1) + off * (N - 1)
                if N % 2 == 0 and (off != 0).sum() == 1 and (row, col) != (1, 1):
                    out.append(-1)  # edge facelet, no middle edge
                    continue
                if N % 2 == 0 and (row, col) == (1, 1):
                    out.append(-2)  # center: colour of the face
                    continue
                out.append(look[(tuple(int(x) for x in pos), tuple(int(x) for x in n))])
    return np.array(out)


def facelets(puzzle: PuzzleSpec, state: np.ndarray) -> str:
    st = facelet_stickers(puzzle.dims)
    letters = "URFDLB"
    out = []
    for i, s in enumerate(st):
        if s < 0:  # even N: edges and centers taken as solved
            out.append(letters[i // 9])
        else:
            out.append(letters[int(state[s])])
    return "".join(out)


def corner_parity_odd(puzzle: PuzzleSpec, state: np.ndarray) -> bool:
    from ..engine.orbits import clusters

    corner = next(c for c in clusters(puzzle) if c.kind == "corner")
    return bool(parity(piece_permutation(puzzle, corner, state)))


def solve_3x3_stage(puzzle: PuzzleSpec, state: np.ndarray) -> list[Move]:
    """Outer-layer moves that solve corners, middle edges and fixed centers. ``state`` must
    already be oriented (:func:`cluster_tools.orient`)."""
    import kociemba

    pre: list[Move] = []
    if puzzle.N % 2 == 0 and corner_parity_odd(puzzle, state):
        pre = [Move(1, puzzle.N - 1, 1)]  # PLL parity: one outer quarter turn
        state = puzzle.apply(state, pre)
    fl = facelets(puzzle, state)
    if fl == "".join(c * 9 for c in "URFDLB"):
        return pre
    text = kociemba.solve(fl)
    return pre + parse_notation(text, puzzle)
