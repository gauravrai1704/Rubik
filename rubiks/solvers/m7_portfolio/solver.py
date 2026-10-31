"""Method 7: a symmetry portfolio over a deterministic cube solver.

The same scramble seen from the 24 sides of the cube is 24 different inputs to a solver whose search
is driven by fixed seeds, tie-breaks and move orders; its answers differ in length, and each maps back to
a solution of the original state of the same length (:mod:`rubiks.solvers.symmetry`). Method 7 runs the
inner solver on ``copies`` rotated copies (identity first) and keeps the shortest answer in QTM.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ...engine import PuzzleSpec, metrics
from ..base import Budget, Solver, SolveResult, get_solver, register
from ..symmetry import map_back, rotated


@register("m7_portfolio")
class SymmetryPortfolio(Solver):
    def __init__(self, puzzle: PuzzleSpec, budget: Budget | None = None, inner: str = "kmcoders",
                 copies: int = 8, workers: int = 1, inner_kwargs: dict | None = None, **kwargs):
        super().__init__(puzzle, budget)
        if puzzle.kind != "cube":
            raise ValueError("Method 7 rotates cubes")
        self.copies = min(copies, len(puzzle.rotations))
        self.workers = workers
        self.inner = get_solver(inner, puzzle, budget, **(inner_kwargs or {}))
        self.inner_name = inner
        # identity first, then the other rotations in a fixed order
        ident = int(np.nonzero((puzzle.rotations == np.arange(puzzle.num_stickers)).all(1))[0][0])
        self.order = [ident] + [r for r in range(len(puzzle.rotations)) if r != ident]

    def _one(self, state, r):
        P = self.puzzle
        res = self.inner.solve(rotated(P, state, r))
        moves = map_back(P, res.moves, r)
        ok = np.array_equal(P.apply(np.asarray(state), moves), P.solved)
        return r, moves, ok

    def solve(self, state: np.ndarray) -> SolveResult:
        t0 = time.time()
        P = self.puzzle
        rs = self.order[:self.copies]
        with ThreadPoolExecutor(self.workers) as ex:
            runs = list(ex.map(lambda r: self._one(state, r), rs))
        lens = {r: metrics(m, P)["qtm"] if ok else None for r, m, ok in runs}
        good = [(lens[r], i, m) for i, (r, m, ok) in enumerate(runs) if ok]
        if not good:
            return SolveResult(moves=[], wall_time_s=time.time() - t0, info={"copies": lens})
        _, i, best = min(good, key=lambda x: (x[0], x[1]))
        return SolveResult(moves=best, wall_time_s=time.time() - t0,
                           info={"inner": self.inner_name, "copies": [lens[r] for r in rs], "best_copy": i})
