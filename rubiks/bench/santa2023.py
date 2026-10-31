"""Kaggle Santa 2023 cube puzzles on our engine (external benchmark).

The competition's cube puzzles (``cube_N/N/N``, N = 2..10, 19, 33) come with a fixed solution
state and a move set of quarter turns of every slice (``f0 .. f{N-1}``, ``r..``, ``d..`` and their
inverses ``-f0`` ...), each costing 1: exactly our QTM. KMCoders' published final submission is the
reference ("leaderboard") baseline.

The data is downloaded on first use from the KMCoders repository (which mirrors the Kaggle files) into
``data/santa2023/`` and is not committed: redistribution is governed by Kaggle's competition terms.

Mapping to the engine is *derived*, not assumed: each Santa sticker's slab coordinates come from which
moves touch it, the face it lies on from which move rotates that face in place, and the axis alignment
is the one (of 48) under which every Santa move equals one engine move. Replaying KMCoders' solutions in
the engine (tests) confirms it.
"""
from __future__ import annotations

import ast
import csv
import itertools
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..engine.puzzle import Move, PuzzleSpec, cube
from .datasets import DATA_ROOT, Instance

ROOT = DATA_ROOT / "santa2023"
SOURCE = "https://raw.githubusercontent.com/wata-orz/santa2023_permutation_puzzle/main"
FILES = {"puzzles.csv": "tools/puzzles.csv", "puzzle_info.csv": "tools/puzzle_info.csv",
         "submission.csv": "submission.csv"}
csv.field_size_limit(10**9)


def fetch(root: Path = ROOT) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, src in FILES.items():
        p = root / name
        if not p.exists():
            urllib.request.urlretrieve(f"{SOURCE}/{src}", p)
    return root


@lru_cache(maxsize=1)
def _tables():
    root = fetch()
    puzzles = {int(r["id"]): r for r in csv.DictReader(open(root / "puzzles.csv"))}
    info = {r["puzzle_type"]: ast.literal_eval(r["allowed_moves"]) for r in csv.DictReader(open(root / "puzzle_info.csv"))}
    subs = {int(r["id"]): r["moves"] for r in csv.DictReader(open(root / "submission.csv"))}
    return puzzles, info, subs


@dataclass
class SantaMap:
    """Santa sticker i <-> engine sticker ``to_engine[i]``; Santa move name -> engine Move."""

    N: int
    to_engine: np.ndarray
    moves: dict[str, Move]

    def state_to_engine(self, santa_labels: list[int], P: PuzzleSpec, dtype=np.uint8) -> np.ndarray:
        out = np.empty(P.num_stickers, dtype=dtype)
        out[self.to_engine] = santa_labels
        return out

    def moves_to_engine(self, text: str) -> list[Move]:
        return [self.moves[t] for t in text.split(".") if t]

    def moves_to_santa(self, moves: list[Move]) -> str:
        inv = {m: k for k, m in self.moves.items()}
        out = []
        for m in moves:
            if m.turn == 2:  # Santa has quarter turns only: a half turn costs two
                q = Move(m.axis, m.layer, 1)
                out += [inv[q], inv[q]]
            else:
                out.append(inv[m])
        return ".".join(out)


@lru_cache(maxsize=16)
def santa_map(N: int) -> SantaMap:
    _, info, _ = _tables()
    gens = {k: np.array(v) for k, v in info[f"cube_{N}/{N}/{N}"].items() if not k.startswith("-")}
    P = cube(N)
    S = 6 * N * N
    axes = "frd"
    # slab coordinates: which layer of each Santa axis moves the sticker (face stickers are moved by
    # their face layer of the normal axis, and by one layer of each other axis)
    touch = {a: np.full(S, -1) for a in axes}
    for name, p in gens.items():
        a, layer = name[0], int(name[1:])
        moved = np.nonzero(p != np.arange(S))[0]
        touch[a][moved] = np.where(touch[a][moved] < 0, layer, touch[a][moved])
    # face of each Santa sticker block (stickers k*N*N .. (k+1)*N*N - 1): the move that keeps it in place
    face_axis = {}
    for k in range(6):
        block = set(range(k * N * N, (k + 1) * N * N))
        for name, p in gens.items():
            if name[1:] in ("0", str(N - 1)) and N > 1:
                if set(p[list(block)].tolist()) == block and any(p[i] != i for i in block):
                    face_axis[k] = (name[0], int(name[1:]))
                    break
    # engine lookup: (normal axis, side, layer coords) -> sticker
    look = {}
    for s in range(P.num_stickers):
        ax = int(np.nonzero(P.sticker_normal[s])[0][0])
        look[(ax, int(P.sticker_normal[s][ax]), tuple(int(x) for x in P.sticker_layer[s]))] = s
    for perm in itertools.permutations(range(3)):  # Santa axis f/r/d -> engine axis
        for flips in itertools.product((False, True), repeat=3):
            to_engine = np.empty(S, dtype=np.int64)
            ok = True
            for i in range(S):
                k = i // (N * N)
                sa, slayer = face_axis[k]
                lay = [0, 0, 0]
                for j, a in enumerate(axes):
                    v = slayer if a == sa else touch[a][i]
                    if v < 0:
                        ok = False
                        break
                    lay[perm[j]] = (N - 1 - v) if flips[j] else v
                if not ok:
                    break
                e_ax = perm[axes.index(sa)]
                side = 1 if lay[e_ax] == N - 1 else -1
                s = look.get((e_ax, side, tuple(lay)))
                if s is None:
                    ok = False
                    break
                to_engine[i] = s
            if not ok or len(set(to_engine.tolist())) != S:
                continue
            moves = {}
            inv_map = np.empty(S, dtype=np.int64)
            inv_map[to_engine] = np.arange(S)
            for name, p in info[f"cube_{N}/{N}/{N}"].items():
                # Santa: new[i] = old[p[i]]; engine: new[e] = old[q[e]]
                q = to_engine[np.asarray(p)[inv_map]]
                hit = [m for mi, m in enumerate(P.moves) if np.array_equal(P.move_perms[mi], q)]
                if len(hit) != 1:
                    ok = False
                    break
                moves[name] = hit[0]
            if ok:
                moves.update({"-" + k: v.inverse() for k, v in list(moves.items())})
                return SantaMap(N, to_engine, moves)
    raise RuntimeError(f"no consistent Santa -> engine mapping for N={N}")


def instances(N: int, colours_only: bool = True, no_wildcards: bool = True) -> list[Instance]:
    """Santa 2023 cube_N instances as engine Instances (colour states, our face colours).

    ``colours_only``: skip the puzzles whose solution state gives every sticker its own label.
    ``no_wildcards``: skip puzzles that allow some stickers to stay wrong. Puzzles whose goal is not
    one colour per face (the competition also has checkerboard goals) are always skipped."""
    puzzles, _, _ = _tables()
    P = cube(N)
    m = santa_map(N)
    out = []
    for pid, r in sorted(puzzles.items()):
        if r["puzzle_type"] != f"cube_{N}/{N}/{N}":
            continue
        sol = r["solution_state"].split(";")
        if colours_only and len(set(sol)) > 6:
            continue
        if no_wildcards and int(r["num_wildcards"]) > 0:
            continue
        nn = N * N
        if any(len(set(sol[k * nn:(k + 1) * nn])) != 1 for k in range(6)):
            continue  # another goal pattern (e.g. checkerboard faces): not our task
        # Santa letter -> engine colour: the colour the engine's solved state has where the letter is
        letter = {}
        target = m.state_to_engine(list(range(len(sol))), P, np.int64)  # engine sticker -> santa index
        for e, si in enumerate(target):
            letter.setdefault(sol[si], int(P.solved[e]))
        init = [letter[x] for x in r["initial_state"].split(";")]
        st = m.state_to_engine(init, P)
        out.append(Instance(f"santa-cube{N}-{pid:04d}", "cube", N, st, None, "santa", 0))
    return out


def kmcoders_solution(pid: int, N: int) -> list[Move]:
    _, _, subs = _tables()
    return santa_map(N).moves_to_engine(subs[pid])
