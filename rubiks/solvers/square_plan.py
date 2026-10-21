"""Rubik's Square plan shared by Methods 2 and 3 (Demaine et al. 2011, Lemmas 2-3).

Cells are (x, y) with x the column (axis-0 slab) and y the row (axis-1 slab).

1. **Frame**: corner, center and edge-cross clusters (slabs 0, mid, N-1), solved exactly
   over the projected 3×3 Square.
2. **Edge groups** (Lemma 3), one per index k = 1 .. ⌊N/2⌋-1 and side: the edge cluster at
   (border, k) plus the cross cluster at (mid, k) for odd N. Moves: the two single flips of
   slabs k, N-1-k (they also disturb interior clusters, solved later) and the pure
   commutators with the border slabs. Solved exactly over the projected group.
3. **Interior clusters** (Lemma 2): each solved with the four pure commutators
   [C, R] = C R C R of its two columns and two rows (and their inverses), which change only
   that cluster. Every one of the 6 reachable configurations is at most 3 commutators away.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..engine.orbits import Orbit, clusters
from ..engine.puzzle import Move, PuzzleSpec
from .cluster_tools import ExactStage, commutator, sequence_perm


def cell_cluster(puzzle: PuzzleSpec) -> dict[tuple[int, int], Orbit]:
    out = {}
    for c in clusters(puzzle):
        for s in c.stickers:
            x, y = (int(v) for v in puzzle.sticker_layer[s, :2])
            out[(x, y)] = c
    return out


def is_pure(puzzle: PuzzleSpec, seq: list[int], cluster: Orbit) -> bool:
    q = sequence_perm(puzzle, seq)
    moved = np.nonzero(q != np.arange(len(q)))[0]
    return bool(len(moved)) and set(moved.tolist()) <= set(cluster.stickers.tolist())


@dataclass
class InteriorCluster:
    cluster: Orbit
    cols: tuple[int, int]  # (c, N-1-c)
    rows: tuple[int, int]  # (r, N-1-r)
    ops: list[tuple[int, int, bool]]  # (col side, row side, inverse) per action
    stage: ExactStage

    def op_sequence(self, puzzle: PuzzleSpec, op: tuple[int, int, bool]) -> list[int]:
        idx, inv = puzzle.move_index, puzzle.inverse_move
        C, R = idx[Move(0, self.cols[op[0]], 2)], idx[Move(1, self.rows[op[1]], 2)]
        return commutator([R], [C], inv) if op[2] else commutator([C], [R], inv)


@dataclass
class SquarePlan:
    frame: ExactStage
    groups: list[tuple[str, ExactStage]]
    interiors: list[InteriorCluster]


@lru_cache(maxsize=32)
def square_plan(N: int) -> SquarePlan:
    from ..engine.puzzle import square

    P = square(N)
    idx, inv = P.move_index, P.inverse_move
    cc = cell_cluster(P)
    mid = N // 2 if N % 2 else None
    border = (0, N - 1)
    frame_cl = [c for c in clusters(P) if c.kind in ("sq_corner", "sq_center", "sq_edge_cross")]
    frame_layers = [0, N - 1] + ([mid] if mid is not None else [])
    frame_moves = [[idx[Move(a, layer, 2)]] for a in (0, 1) for layer in frame_layers]
    frame = ExactStage(P, np.concatenate([c.stickers for c in frame_cl]), frame_moves)

    groups = []
    for k in range(1, N):
        if k >= N - 1 - k:
            break
        for side in (0, 1):  # side 0: edge cells (border, k) moved by rows k; side 1: (k, border) by columns
            line_axis = 1 - side  # the axis whose slabs k, N-1-k cross the edge cells
            members = [cc[(0, k)] if side == 0 else cc[(k, 0)]]
            if mid is not None:
                members.append(cc[(mid, k)] if side == 0 else cc[(k, mid)])
            singles = [[idx[Move(line_axis, layer, 2)]] for layer in (k, N - 1 - k)]
            pure = []
            for layer in (k, N - 1 - k):
                for b in border:
                    L, B = idx[Move(line_axis, layer, 2)], idx[Move(1 - line_axis, b, 2)]
                    for seq in (commutator([L], [B], inv), commutator([B], [L], inv)):
                        assert is_pure(P, seq, members[0]), "Lemma 2 commutator is not pure"
                        pure.append(seq)
            stickers = np.concatenate([c.stickers for c in members])
            groups.append((f"edges {k}{'ab'[side]}", ExactStage(P, stickers, singles + pure)))

    interiors = []
    seen = set()
    for (x, y), c in sorted(cc.items()):
        if c.kind != "sq_interior" or c.index in seen or x > N - 1 - x or y > N - 1 - y:
            continue
        seen.add(c.index)
        cols, rows = (x, N - 1 - x), (y, N - 1 - y)
        ops = [(a, b, f) for a in (0, 1) for b in (0, 1) for f in (False, True)]
        tmp = InteriorCluster(c, cols, rows, ops, None)  # type: ignore[arg-type]
        actions = [tmp.op_sequence(P, op) for op in ops]
        for a in actions:
            assert is_pure(P, a, c)
        tmp.stage = ExactStage(P, c.stickers, actions)
        interiors.append(tmp)
    return SquarePlan(frame, groups, interiors)
