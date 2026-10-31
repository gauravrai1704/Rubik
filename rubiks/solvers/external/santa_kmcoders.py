"""KMCoders' published Kaggle Santa 2023 solutions as a solver (reference baseline).

Not a method we run: it looks the given Santa instance up and returns KMCoders' final submission for
it, converted to engine moves, so the harness re-verifies it and measures it exactly like our methods.
Works only on the Santa 2023 cube instances (``bench make --puzzle cube --N <N> --depth santa``).
"""
from __future__ import annotations

import time

import numpy as np

from ...bench import santa2023
from ...engine import PuzzleSpec
from ..base import Budget, Solver, SolveResult, register


@register("santa_kmcoders")
class KMCodersPublished(Solver):
    def __init__(self, puzzle: PuzzleSpec, budget: Budget | None = None, **kwargs):
        super().__init__(puzzle, budget)
        if puzzle.kind != "cube":
            raise ValueError("Santa 2023 KMCoders solutions exist for cubes only")
        self._by_state = {}
        for ins in santa2023.instances(puzzle.N):
            pid = int(ins.id.split("-")[-1])
            self._by_state[np.asarray(ins.state, dtype=np.uint8).tobytes()] = pid

    def solve(self, state: np.ndarray) -> SolveResult:
        t0 = time.time()
        pid = self._by_state.get(np.asarray(state, dtype=np.uint8).tobytes())
        if pid is None:
            raise KeyError("not a Santa 2023 instance: KMCoders' solutions exist only for those")
        moves = santa2023.kmcoders_solution(pid, self.puzzle.N)  # kept as published (no canonicalisation)
        return SolveResult(moves=moves, wall_time_s=time.time() - t0, info={"santa_id": pid, "source": "KMCoders"})
