"""Method 1: IDA* with pattern databases — owner: Shree.

Spec: technical basis, Section 4; checklist in ./README.md.

Finds provably shortest solutions where feasible: the ground truth every other method is
measured against, plus the measured curve of where exact search stops working.

Contract (do not change): subclass ``Solver``, keep the registered name ``m1_idastar``, return
engine ``Move`` objects in ``SolveResult.moves``. The harness re-applies them and marks
``solved`` itself. Run ``pytest tests/test_m1_idastar.py`` and
``python -m rubiks.bench run --solver m1_idastar --puzzle square --N 3-5 --dataset d20``.
"""
from __future__ import annotations

import time

import numpy as np

from ...engine import PuzzleSpec, canonicalize
from ..base import Budget, Solver, SolveResult, register


@register("m1_idastar")
class IDAStarSolver(Solver):
    def __init__(self, puzzle: PuzzleSpec, budget: Budget | None = None, **kwargs):
        super().__init__(puzzle, budget)
        # TODO(Shree): precompute tables / load models here (once per puzzle size).

    def solve(self, state: np.ndarray) -> SolveResult:
        t0 = time.time()
        moves = []  # TODO(Shree): implement — see README.md in this folder.
        raise NotImplementedError("Method 1: IDA* with pattern databases is not implemented yet (Shree)")
        return SolveResult(moves=canonicalize(moves), wall_time_s=time.time() - t0)
