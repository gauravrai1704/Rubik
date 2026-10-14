"""Puzzle geometry: one engine for the N×N×N Rubik's Cube and the N×N×1 Rubik's Square.

Every puzzle is a cuboid of dims (d0, d1, d2). A *sticker* is one visible face of one
surface cubie. A state is an array with one label per sticker (its colour by default).
A *move* rotates one slab of cubies (all cubies with coordinate == layer along an axis)
by a multiple of 90° about that axis. Moves are generated from 3D geometry, never typed
by hand, so the same code builds every N.

Conventions
-----------
* Axis 0 = x (R is +x, L is -x), axis 1 = y (U is +y, D is -y), axis 2 = z (F is +z, B is -z).
* Faces are enumerated in the order U, R, F, D, L, B (Kociemba order). Colour c = face index.
* ``Move(axis, layer, turn)``: ``turn`` in {1, 2, 3} = number of +90° (right-handed,
  counter-clockwise when looking from +axis toward the origin) quarter turns.
* Applying a move: ``new_state = state[perm]`` where ``perm[dest] = src``.
* A slab whose cross-section is not square (e.g. a row of the N×N×1 square) only allows
  180° turns. A dimension of length 1 has no moves at all (the 1-thick axis of the square).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

import numpy as np

FACE_NAMES = ("U", "R", "F", "D", "L", "B")
# face index -> (axis, sign)
FACE_AXIS_SIGN = {0: (1, +1), 1: (0, +1), 2: (2, +1), 3: (1, -1), 4: (0, -1), 5: (2, -1)}
AXIS_SIGN_FACE = {v: k for k, v in FACE_AXIS_SIGN.items()}


@dataclass(frozen=True, order=True)
class Move:
    axis: int
    layer: int
    turn: int  # 1, 2 or 3 quarter turns about +axis

    def inverse(self) -> "Move":
        return Move(self.axis, self.layer, (4 - self.turn) % 4)

    def __repr__(self) -> str:  # compact, e.g. M(0,3,1)
        return f"M({self.axis},{self.layer},{self.turn})"


def _rot90(axis: int, v: np.ndarray, k: int) -> np.ndarray:
    """Rotate integer vectors v (..., 3) by k * 90° about +axis (right-handed)."""
    v = v.copy()
    a, b = [(1, 2), (2, 0), (0, 1)][axis]  # rotation plane (a -> b)
    for _ in range(k % 4):
        va, vb = v[..., a].copy(), v[..., b].copy()
        v[..., a], v[..., b] = -vb, va
    return v


@dataclass(frozen=True)
class PuzzleSpec:
    """A cuboid puzzle. Use :func:`cube` or :func:`square` to build one."""

    dims: tuple[int, int, int]
    labeling: str = "color"  # "color": label = face colour; "sticker": label = sticker id

    # ------------------------------------------------------------------ basics
    @property
    def kind(self) -> str:
        d = self.dims
        if d[0] == d[1] == d[2]:
            return "cube"
        if d[0] == d[1] and d[2] == 1:
            return "square"
        return "cuboid"

    @property
    def N(self) -> int:
        return self.dims[0]

    def __repr__(self) -> str:
        return f"{self.kind}{self.N}" if self.kind != "cuboid" else f"cuboid{self.dims}"

    # ------------------------------------------------------------- geometry
    @cached_property
    def _geometry(self):
        d = np.array(self.dims)
        pos, nrm, face = [], [], []
        for f in range(6):
            ax, sg = FACE_AXIS_SIGN[f]
            other = [a for a in range(3) if a != ax]
            # cubies on this face: coordinate along ax is the extreme layer
            layer = d[ax] - 1 if sg > 0 else 0
            grid = []
            for i in range(d[other[0]]):
                for j in range(d[other[1]]):
                    idx = [0, 0, 0]
                    idx[ax], idx[other[0]], idx[other[1]] = layer, i, j
                    grid.append(idx)
            for idx in grid:
                c = 2 * np.array(idx) - (d - 1)  # doubled centred coordinates
                n = np.zeros(3, dtype=int)
                n[ax] = sg
                pos.append(c)
                nrm.append(n)
                face.append(f)
        pos, nrm, face = np.array(pos), np.array(nrm), np.array(face)
        lookup = {(tuple(p), tuple(n)): i for i, (p, n) in enumerate(zip(pos, nrm))}
        return pos, nrm, face, lookup

    @property
    def sticker_pos(self) -> np.ndarray:
        """(S, 3) doubled centred cubie coordinates of each sticker."""
        return self._geometry[0]

    @property
    def sticker_normal(self) -> np.ndarray:
        return self._geometry[1]

    @property
    def sticker_face(self) -> np.ndarray:
        return self._geometry[2]

    @property
    def num_stickers(self) -> int:
        return len(self.sticker_face)

    @cached_property
    def face_slices(self) -> list[slice]:
        f = self.sticker_face
        out = []
        for k in range(6):
            idx = np.nonzero(f == k)[0]
            out.append(slice(int(idx[0]), int(idx[-1]) + 1))
        return out

    @cached_property
    def sticker_layer(self) -> np.ndarray:
        """(S, 3) integer layer index of each sticker's cubie along each axis."""
        d = np.array(self.dims)
        return (self.sticker_pos + (d - 1)) // 2

    # ---------------------------------------------------------------- moves
    def allowed_turns(self, axis: int) -> tuple[int, ...]:
        d = self.dims
        if d[axis] == 1:
            return ()
        others = [d[a] for a in range(3) if a != axis]
        return (1, 2, 3) if others[0] == others[1] else (2,)

    @cached_property
    def moves(self) -> list[Move]:
        out = []
        for ax in range(3):
            for layer in range(self.dims[ax]):
                for t in self.allowed_turns(ax):
                    out.append(Move(ax, layer, t))
        return out

    @cached_property
    def move_index(self) -> dict[Move, int]:
        return {m: i for i, m in enumerate(self.moves)}

    def _perm_for(self, ax: int, layer: int, turn: int) -> np.ndarray:
        pos, nrm, _, lookup = self._geometry
        S = len(pos)
        perm = np.arange(S)
        sel = np.nonzero(self.sticker_layer[:, ax] == layer)[0]
        new_pos = _rot90(ax, pos[sel], turn)
        new_nrm = _rot90(ax, nrm[sel], turn)
        for src, p, n in zip(sel, new_pos, new_nrm):
            dest = lookup[(tuple(p), tuple(n))]
            perm[dest] = src
        return perm

    @cached_property
    def move_perms(self) -> np.ndarray:
        """(M, S) int array; ``state[move_perms[m]]`` applies move m."""
        return np.stack([self._perm_for(m.axis, m.layer, m.turn) for m in self.moves])

    @cached_property
    def inverse_move(self) -> np.ndarray:
        return np.array([self.move_index[m.inverse()] for m in self.moves])

    @cached_property
    def rotations(self) -> np.ndarray:
        """Whole-puzzle rotations reachable with moves, as (R, S) sticker permutations.

        24 for a cube, 4 for a square (180° about x, y, z and identity)."""
        S = self.num_stickers
        gens = []
        for ax in range(3):
            ts = self.allowed_turns(ax)
            if not ts:
                continue
            t = min(ts)
            p = np.arange(S)
            for layer in range(self.dims[ax]):
                p = p[self._perm_for(ax, layer, t)]
            gens.append(p)
        if self.kind == "square":  # 180° about z = rows ∘ cols, both already generated
            pass
        seen = {tuple(np.arange(S))}
        frontier = [np.arange(S)]
        while frontier:
            nxt = []
            for p in frontier:
                for g in gens:
                    q = p[g]
                    key = tuple(q)
                    if key not in seen:
                        seen.add(key)
                        nxt.append(q)
            frontier = nxt
        return np.array(sorted(seen))

    # ---------------------------------------------------------------- states
    @cached_property
    def solved(self) -> np.ndarray:
        if self.labeling == "sticker":
            return np.arange(self.num_stickers, dtype=np.int16)
        return self.sticker_face.astype(np.uint8)

    @property
    def num_colors(self) -> int:
        return 6 if self.labeling == "color" else self.num_stickers

    def colors_of(self, state: np.ndarray) -> np.ndarray:
        """Colour view of a state (identity for colour labeling)."""
        if self.labeling == "sticker":
            return self.sticker_face[state].astype(np.uint8)
        return state

    def apply(self, state: np.ndarray, moves: Iterable[Move | int]) -> np.ndarray:
        s = np.asarray(state)
        for m in moves:
            mi = m if isinstance(m, (int, np.integer)) else self.move_index[m]
            s = s[self.move_perms[mi]]
        return s

    def is_solved(self, state: np.ndarray) -> bool:
        """Every face shows a single colour (any of the reachable orientations)."""
        c = self.colors_of(np.asarray(state))
        for sl in self.face_slices:
            f = c[..., sl]
            if not np.all(f == f[..., :1]):
                return False
        return True

    def is_identity(self, state: np.ndarray) -> bool:
        """Strictly equal to the reference solved state (one fixed orientation)."""
        return bool(np.array_equal(np.asarray(state), self.solved))


def cube(N: int, labeling: str = "color") -> PuzzleSpec:
    if not 2 <= N <= 64:
        raise ValueError("cube size must be 2..64")
    return PuzzleSpec((N, N, N), labeling)


def square(N: int, labeling: str = "color") -> PuzzleSpec:
    if not 2 <= N <= 64:
        raise ValueError("square size must be 2..64")
    return PuzzleSpec((N, N, 1), labeling)


def make(puzzle: str, N: int, labeling: str = "color") -> PuzzleSpec:
    if puzzle == "cube":
        return cube(N, labeling)
    if puzzle == "square":
        return square(N, labeling)
    raise ValueError(f"unknown puzzle {puzzle!r}; use 'cube' or 'square'")
