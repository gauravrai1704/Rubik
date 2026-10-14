"""Method 2: Reduction + Kociemba two-phase — owner: Sunaina.

Spec: technical basis, Section 5; checklist in ./README.md.

The standard way to solve any N: reduce to a 3×3×3 stage by stage, then use a fast 3×3×3
solver. Works for every N with O(N²) moves.

Contract (do not change): subclass ``Solver``, keep the registered name ``m2_reduction``, return
engine ``Move`` objects in ``SolveResult.moves``. The harness re-applies them and marks
``solved`` itself. Run ``pytest tests/test_m2_reduction.py`` and
``python -m rubiks.bench run --solver m2_reduction --puzzle square --N 3-5 --dataset d20``.
"""
from __future__ import annotations

import time

import numpy as np

from ...engine import PuzzleSpec, canonicalize
from ..base import Budget, Solver, SolveResult, register


@register("m2_reduction")
class ReductionSolver(Solver):
    def __init__(self, puzzle: PuzzleSpec, budget: Budget | None = None, **kwargs):
        super().__init__(puzzle, budget)
        # TODO(Sunaina): precompute tables / load models here (once per puzzle size).

    def solve(self, state: np.ndarray) -> SolveResult:
        t0 = time.time()
        moves = []  # TODO(Sunaina): implement — see README.md in this folder.
        raise NotImplementedError("Method 2: Reduction + Kociemba two-phase is not implemented yet (Sunaina)")
        return SolveResult(moves=canonicalize(moves), wall_time_s=time.time() - t0)
