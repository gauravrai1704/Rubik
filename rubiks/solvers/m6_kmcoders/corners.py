"""Optimal corner solving in QTM for any N (KMCoders' "special parts" step for even N).

The corners of an N×N×N cube under outer-face quarter turns are exactly the 2×2×2 cube, and Method 1's
QTM corner pattern database (8!·3^7 entries) is the exact distance of every corner state under those 12
moves. Descending it greedily gives an optimal (shortest QTM) corner solution to the exact solved state.
"""
from __future__ import annotations

import numpy as np

from ...engine.puzzle import Move, PuzzleSpec, cube


def corner_solution_qtm(P: PuzzleSpec, state: np.ndarray) -> list[Move]:
    from ..m1_idastar.cube3 import databases, ori_index, rank_perm

    cb, (cpm, com, _, _), pc, _, _ = databases("qtm")
    P3 = cube(3)
    N = P.N
    look = {(tuple(int(v) for v in p), tuple(int(v) for v in n)): s
            for s, (p, n) in enumerate(zip(P.sticker_pos, P.sticker_normal))}
    # read the corner pieces of the N-cube through cube(3)'s corner slots (positions scaled by (N-1)/2)
    home = {frozenset(int(P3.solved[s]) for s in st): j for j, st in enumerate(cb.corner)}
    ref = [int(P3.solved[st[0]]) for st in cb.corner]
    cp, co = np.zeros(8, dtype=np.int64), np.zeros(8, dtype=np.int64)
    for j, st in enumerate(cb.corner):
        cols = []
        for s in st:
            p = tuple(int(v) * (N - 1) // 2 for v in P3.sticker_pos[s])
            n = tuple(int(v) for v in P3.sticker_normal[s])
            cols.append(int(np.asarray(state)[look[(p, n)]]))
        piece = home[frozenset(cols)]
        cp[j], co[j] = piece, cols.index(ref[piece]) % 3
    c, o = int(rank_perm(cp)), int(ori_index(co, 3))
    out = []
    d = int(pc[c * 2187 + o])
    while d > 0:
        for k in range(len(cb.moves)):
            c2, o2 = int(cpm[k, c]), int(com[k, o])
            if int(pc[c2 * 2187 + o2]) == d - 1:
                m3 = P3.moves[cb.moves[k]]
                out.append(Move(m3.axis, 0 if m3.layer == 0 else N - 1, m3.turn))
                c, o, d = c2, o2, d - 1
                break
        else:
            raise RuntimeError("corner database descent failed")
    return out
