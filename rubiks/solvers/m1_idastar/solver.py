"""Method 1: IDA* with pattern databases — owner: Shree (implemented by Gaurav).

Spec: technical basis, Section 4; checklist in ./README.md.

Finds provably shortest solutions where feasible: the ground truth every other method is
measured against, plus the measured curve of where exact search stops working.

* 3×3×3 (``cube3.py``): Korf's three pattern databases, IDA* on cubie coordinates in
  Numba; ``metric="htm"`` (18 face turns, default) or ``"qtm"`` (12 quarter turns).
* Rubik's Square (``square.py``): IDA* with h = max over clusters of each cluster's exact
  distance; full BFS for the exact God's number at the smallest N.

Solutions are optimal for the colour goal: the 3×3×3's fixed centers pick the orientation,
and on the Square IDA* runs to each of the 4 rotations of the goal and keeps the shortest.
Other puzzles are out of Method 1's scope (the solver raises ``ValueError``).

Contract (do not change): subclass ``Solver``, keep the registered name ``m1_idastar``, return
engine ``Move`` objects in ``SolveResult.moves``. The harness re-applies them and marks
``solved`` itself.
"""
from __future__ import annotations

import time

import numpy as np

from ...engine import PuzzleSpec, canonicalize
from ..base import Budget, Solver, SolveResult, register
from ..cluster_tools import orient


@register("m1_idastar")
class IDAStarSolver(Solver):
    def __init__(self, puzzle: PuzzleSpec, budget: Budget | None = None, metric: str = "htm",
                 node_limit: int = 10**13, **kwargs):
        super().__init__(puzzle, budget)
        if not ((puzzle.kind == "cube" and puzzle.N == 3) or puzzle.kind == "square"):
            raise ValueError("Method 1 covers the 3×3×3 cube and Rubik's Squares")
        if metric not in ("htm", "qtm"):
            raise ValueError("metric must be 'htm' or 'qtm'")
        self.metric = metric
        self.node_limit = node_limit
        if puzzle.kind == "cube":
            from .cube3 import databases

            databases(metric)  # build or load once per process
        else:
            from .square import cluster_tables

            cluster_tables(puzzle.N)

    def solve(self, state: np.ndarray) -> SolveResult:
        t0 = time.time()
        P = self.puzzle
        s, _ = orient(P, state)
        if P.kind == "cube":
            from .cube3 import ida_star

            moves, stats = ida_star(s, self.metric, self.node_limit, self.budget.wall_time_s)
            moves = None if moves is None else [P.moves[m] for m in moves]
        else:
            from .square import ida_star_any_orientation

            ids, stats = ida_star_any_orientation(P.N, state, self.budget.wall_time_s, self.node_limit)
            moves = None if ids is None else [P.moves[m] for m in ids]
        info = {"optimal": moves is not None, "metric": self.metric if P.kind == "cube" else "htm", **stats}
        return SolveResult(moves=canonicalize(moves or []), wall_time_s=time.time() - t0,
                           nodes_expanded=int(stats.get("nodes", 0)), info=info)
