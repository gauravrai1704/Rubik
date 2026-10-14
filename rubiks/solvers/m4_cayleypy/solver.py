"""Method 4: CayleyPy reproduction — owner: Shaurya.

Spec: technical basis, Section 7; checklist in ./README.md.

Reproduces the current ML state of the art (Chervov et al., NeurIPS 2025) inside our
engine: the per-N baseline that 5A must match while using one model for all N.

Contract (do not change): subclass ``Solver``, keep the registered name ``m4_cayleypy``, return
engine ``Move`` objects in ``SolveResult.moves``. The harness re-applies them and marks
``solved`` itself. Run ``pytest tests/test_m4_cayleypy.py`` and
``python -m rubiks.bench run --solver m4_cayleypy --puzzle square --N 3-5 --dataset d20``.
"""
from __future__ import annotations

import time

import numpy as np

from ...engine import PuzzleSpec, canonicalize
from ..base import Budget, Solver, SolveResult, register


@register("m4_cayleypy")
class CayleyPySolver(Solver):
    def __init__(self, puzzle: PuzzleSpec, budget: Budget | None = None, **kwargs):
        super().__init__(puzzle, budget)
        # TODO(Shaurya): precompute tables / load models here (once per puzzle size).

    def solve(self, state: np.ndarray) -> SolveResult:
        t0 = time.time()
        moves = []  # TODO(Shaurya): implement — see README.md in this folder.
        raise NotImplementedError("Method 4: CayleyPy reproduction is not implemented yet (Shaurya)")
        return SolveResult(moves=canonicalize(moves), wall_time_s=time.time() - t0)
