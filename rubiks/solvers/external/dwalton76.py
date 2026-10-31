"""dwalton76's NxNxN reduction solver (github.com/dwalton76/rubiks-cube-NxNxN-solver) as an external
baseline, run as a subprocess and re-verified by our harness.

Setup (once): clone the repository, build it (``make init``; it needs dwalton76's ``kociemba`` binary on
``PATH``) and point ``DWALTON76_DIR`` at the checkout. Its lookup tables are downloaded from S3 on first
use (about 1.8 GB for the 4x4x4). The state is passed in Kociemba facelet order (URFDLB, each face
row-major in the standard net); its answer uses wide turns (``Rw``, ``3Rw2``), which we expand into
single-slice turns, so lengths are in our metrics (Santa / QTM counts every slice quarter turn).
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from ...engine import PuzzleSpec
from ...engine.puzzle import Move
from ..base import Budget, Solver, SolveResult, register

# face -> (outward normal, "right" in the net, "down" in the net); axes x = R, y = U, z = F
_NET = {
    "U": ((0, 1, 0), (1, 0, 0), (0, 0, 1)), "R": ((1, 0, 0), (0, 0, -1), (0, -1, 0)),
    "F": ((0, 0, 1), (1, 0, 0), (0, -1, 0)), "D": ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
    "L": ((-1, 0, 0), (0, 0, 1), (0, -1, 0)), "B": ((0, 0, -1), (-1, 0, 0), (0, -1, 0)),
}
_FACE = {"R": (0, 1), "L": (0, -1), "U": (1, 1), "D": (1, -1), "F": (2, 1), "B": (2, -1)}
_TOKEN = re.compile(r"^(\d*)([URFDLB])(w?)('?)(2?)('?)$")


def facelet_order(P: PuzzleSpec) -> np.ndarray:
    """Engine sticker for each facelet of the URFDLB string."""
    N = P.N
    look = {(tuple(int(v) for v in p), tuple(int(v) for v in n)): s
            for s, (p, n) in enumerate(zip(P.sticker_pos, P.sticker_normal))}
    out = []
    for f in "URFDLB":
        n, r, d = (np.array(v) for v in _NET[f])
        for row in range(N):
            for col in range(N):
                pos = n * (N - 1) + (2 * col - (N - 1)) * r + (2 * row - (N - 1)) * d
                out.append(look[(tuple(int(x) for x in pos), tuple(int(x) for x in n))])
    return np.array(out)


def to_moves(text: str, P: PuzzleSpec) -> list[Move]:
    """dwalton76 notation -> engine moves (a k-layer wide turn becomes k slice turns)."""
    N = P.N
    out = []
    for tok in text.split():
        m = _TOKEN.match(tok)
        if not m:
            raise ValueError(f"cannot parse {tok!r}")
        k, face, wide, p1, two, p2 = m.groups()
        depth = int(k) if k else (2 if wide else 1)
        layers = range(depth) if (wide or not k) else [depth - 1]
        ax, sg = _FACE[face]
        cw = 3 if sg > 0 else 1
        turn = 2 if two else (4 - cw if (p1 or p2) else cw)
        for d in layers:
            out.append(Move(ax, N - 1 - d if sg > 0 else d, turn))
    return out


@register("dwalton76")
class DWalton76(Solver):
    def __init__(self, puzzle: PuzzleSpec, budget: Budget | None = None, root: str | None = None, **kwargs):
        super().__init__(puzzle, budget)
        if puzzle.kind != "cube":
            raise ValueError("dwalton76 solves cubes")
        self.root = Path(root or os.environ.get("DWALTON76_DIR", ""))
        if not (self.root / "rubiks-cube-solver.py").exists():
            raise FileNotFoundError("set DWALTON76_DIR to a built rubiks-cube-NxNxN-solver checkout")
        self.order = facelet_order(puzzle)

    def solve(self, state: np.ndarray) -> SolveResult:
        t0 = time.time()
        s = np.asarray(state)
        # letters by the colour of each face's center sticker (dwalton76 reads colours from centers)
        text = "".join("URFDLB"[int(c)] for c in s[self.order])
        env = dict(os.environ, PATH=f"{self.root / 'bin'}:{os.environ.get('PATH', '')}")
        py = self.root / "venv" / "bin" / "python3"
        r = subprocess.run([str(py), "rubiks-cube-solver.py", "--state", text], cwd=self.root, env=env,
                           capture_output=True, text=True, timeout=self.budget.wall_time_s)
        line = [x for x in (r.stdout + r.stderr).splitlines() if x.startswith("Solution:")]
        if not line:
            raise RuntimeError("dwalton76 gave no solution: " + (r.stdout + r.stderr)[-300:])
        sol = line[-1].split(":", 1)[1].strip()
        return SolveResult(moves=to_moves(sol, self.puzzle), wall_time_s=time.time() - t0,
                           info={"dwalton76": sol})
