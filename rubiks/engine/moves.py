"""Move sequences: canonical form, length metrics, and text notation.

Moves on the same axis commute, so any run of same-axis moves can be merged per layer
and sorted by layer. That canonical form is used to count length fairly, to prune
search (see :func:`canonical_successor_mask`) and to generate non-redundant scrambles.

Metrics
-------
* HTM (half-turn metric): every single-slice turn counts 1 (90°, 180° or 270°).
* QTM (quarter-turn metric): 90° counts 1, 180° counts 2. Square flips count 1.
* STM (block-turn metric): a run of *adjacent* layers on one axis turned by the same
  amount counts 1 (e.g. a wide move ``3Rw``).
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np

from .puzzle import Move, PuzzleSpec

# axis -> (face on the + side, face on the - side)
_AXIS_FACES = {0: ("R", "L"), 1: ("U", "D"), 2: ("F", "B")}
_FACE_AXIS = {"R": (0, +1), "L": (0, -1), "U": (1, +1), "D": (1, -1), "F": (2, +1), "B": (2, -1)}


def canonicalize(moves: Sequence[Move]) -> list[Move]:
    """Merge commuting same-axis runs, drop identity turns, sort by layer. Repeats until stable."""
    cur = list(moves)
    while True:
        out: list[Move] = []
        i = 0
        while i < len(cur):
            ax = cur[i].axis
            acc: dict[int, int] = {}
            while i < len(cur) and cur[i].axis == ax:
                acc[cur[i].layer] = (acc.get(cur[i].layer, 0) + cur[i].turn) % 4
                i += 1
            out.extend(Move(ax, l, t) for l, t in sorted(acc.items()) if t)
        if out == cur:
            return out
        cur = out


def metrics(moves: Sequence[Move], puzzle: PuzzleSpec | None = None) -> dict[str, int]:
    c = canonicalize(moves)
    htm = len(c)
    if puzzle is not None and puzzle.kind == "square":
        qtm = htm
    else:
        qtm = sum(2 if m.turn == 2 else 1 for m in c)
    stm = 0
    i = 0
    while i < len(c):
        j = i
        while (
            j + 1 < len(c)
            and c[j + 1].axis == c[i].axis
            and c[j + 1].turn == c[i].turn
            and c[j + 1].layer == c[j].layer + 1
        ):
            j += 1
        stm += 1
        i = j + 1
    return {"htm": htm, "qtm": qtm, "stm": stm}


def canonical_successor_mask(puzzle: PuzzleSpec) -> np.ndarray:
    """(M+1, M) bool: allowed[last, next]. Row M is 'no previous move'.

    Disallows a next move on the same axis with a layer <= the last move's layer, which
    removes every redundant ordering of commuting moves and every immediate repeat."""
    mv = puzzle.moves
    M = len(mv)
    allowed = np.ones((M + 1, M), dtype=bool)
    ax = np.array([m.axis for m in mv])
    ly = np.array([m.layer for m in mv])
    for i in range(M):
        allowed[i] = ~((ax == ax[i]) & (ly <= ly[i]))
    return allowed


# ----------------------------------------------------------------- notation
def to_notation(moves: Iterable[Move], puzzle: PuzzleSpec) -> str:
    """Single-slice notation: ``R``, ``3R'``, ``2U2``. Layer counted from the named face.

    ``kR`` turns only the k-th layer (not a wide move); clockwise as seen from that face."""
    out = []
    for m in moves:
        d = puzzle.dims[m.axis]
        plus, minus = _AXIS_FACES[m.axis]
        if m.layer >= d / 2:
            face, depth = plus, d - m.layer
            cw = 3  # clockwise seen from + face = -90° about +axis
        else:
            face, depth = minus, m.layer + 1
            cw = 1
        if m.turn == 2:
            suf = "2"
        elif m.turn == cw:
            suf = ""
        else:
            suf = "'"
        out.append(f"{depth if depth > 1 else ''}{face}{suf}")
    return " ".join(out)


_TOKEN = re.compile(r"^(\d*)([URFDLB])(2|'|)$")


def parse_notation(text: str, puzzle: PuzzleSpec) -> list[Move]:
    moves = []
    for tok in text.split():
        mt = _TOKEN.match(tok)
        if not mt:
            raise ValueError(f"cannot parse move {tok!r}")
        depth = int(mt.group(1) or 1)
        ax, sg = _FACE_AXIS[mt.group(2)]
        d = puzzle.dims[ax]
        if not 1 <= depth <= d:
            raise ValueError(f"layer {depth} out of range for {tok!r}")
        layer = d - depth if sg > 0 else depth - 1
        cw = 3 if sg > 0 else 1
        suf = mt.group(3)
        turn = 2 if suf == "2" else (cw if suf == "" else 4 - cw)
        m = Move(ax, layer, turn)
        if m not in puzzle.move_index:
            raise ValueError(f"move {tok!r} not allowed on {puzzle}")
        moves.append(m)
    return moves


def moves_to_ids(moves: Iterable[Move], puzzle: PuzzleSpec) -> list[int]:
    return [puzzle.move_index[m] for m in moves]


def ids_to_moves(ids: Iterable[int], puzzle: PuzzleSpec) -> list[Move]:
    return [puzzle.moves[int(i)] for i in ids]
