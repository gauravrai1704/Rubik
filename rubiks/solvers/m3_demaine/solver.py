"""Method 3: Demaine's parallel cluster algorithm — owner: Kashvi (implemented by Gaurav).

Spec: technical basis, Section 6; checklist in ./README.md.

The asymptotically optimal O(N²/log N) algorithm of Demaine et al. (2011): one slice move
touches many clusters, so clusters needing the same fix share one move sequence.

* **Square** (Theorem 1): frame and edge groups in O(N) (Lemma 3, shared with Method 2),
  then every interior cluster at once: round r applies each cluster's r-th Lemma 2
  commutator, batched into blocks X × Y with one C_X R_Y C_X R_Y each (Lemmas 4-6).
* **Cube** (Theorem 3 with the Section 6.4 fix): wing parity with single slice moves
  (Lemma 7), centers and then wings by greedy rounds of *batched* 3-cycles — each round
  picks one parametric commutator type and applies it with bulk slice sets to every
  cluster it helps — and the 3×3×3 part (corners, middle edges) by Kociemba's algorithm.

Methods 2 and 3 share every non-batched stage, so their difference is the parallelism.

Contract (do not change): subclass ``Solver``, keep the registered name ``m3_demaine``, return
engine ``Move`` objects in ``SolveResult.moves``. The harness re-applies them and marks
``solved`` itself.
"""
from __future__ import annotations

import time

import numpy as np

from ...engine import PuzzleSpec, canonicalize
from ...engine.orbits import clusters
from ..base import Budget, Solver, SolveResult, register
from ..cluster_tools import CycleLibrary, prepare
from ..m2_reduction.solver import CENTER_KINDS, run_stages, wing_parity
from ..square_plan import square_plan
from ..three_stage import solve_3x3_stage
from .batching import TypeBank, batched_solve, center_templates, square_interiors, wing_templates


class BatchedKinds:
    """Type bank + per-cluster fallback libraries for one cluster family of one cube size."""

    def __init__(self, puzzle: PuzzleSpec, kinds, templates, setup_depth: int = 1):
        self.puzzle = puzzle
        self.clusters = [c for c in clusters(puzzle) if c.kind in kinds]
        self.bank = TypeBank(puzzle, self.clusters, templates, setup_depth) if self.clusters else None
        self.fallback = {c.index: CycleLibrary(puzzle, c) for c in self.clusters}
        self.stats: dict = {}

    def __call__(self, state: np.ndarray):
        if not self.clusters:
            return []
        ids, self.stats = batched_solve(self.puzzle, self.bank, state, self.fallback)
        return [self.puzzle.moves[i] for i in ids]


@register("m3_demaine")
class DemaineSolver(Solver):
    def __init__(self, puzzle: PuzzleSpec, budget: Budget | None = None, setup_depth: int = 1, exact_orientation: bool = False, **kwargs):
        super().__init__(puzzle, budget)
        self.exact_orientation = exact_orientation  # Kaggle Santa: the exact solved state, not a rotation
        P = puzzle
        if P.kind == "square":
            plan = square_plan(P.N)
            ids = lambda f: (lambda s: [P.moves[i] for i in f(s)])  # noqa: E731
            self.stages = [("frame", ids(plan.frame.solve))] + [(n, ids(g.solve)) for n, g in plan.groups]
            self._sq_stats: dict = {}

            def interiors(s):
                mv, self._sq_stats = square_interiors(P, plan, s)
                return [P.moves[i] for i in mv]

            if plan.interiors:
                self.stages.append(("interiors (batched)", interiors))
            self.batched = []
        elif P.kind == "cube":
            self.centers = BatchedKinds(P, CENTER_KINDS, center_templates(P), setup_depth) if P.N > 3 else None
            self.wings = BatchedKinds(P, ("wing",), wing_templates(P), setup_depth) if P.N > 3 else None
            self.stages = [("three", lambda s: solve_3x3_stage(P, s))]
            if P.N > 3:
                self.stages = [("wing_parity", lambda s: wing_parity(P, s)),
                               ("centers (batched)", self.centers),
                               ("three", lambda s: solve_3x3_stage(P, s)),
                               ("wings (batched)", self.wings)]
            self.batched = [b for b in (self.centers, self.wings) if b is not None]
        else:
            raise ValueError("Method 3 supports cubes and squares")

    def solve(self, state: np.ndarray) -> SolveResult:
        t0 = time.time()
        s, pre = prepare(self.puzzle, state, self.exact_orientation)
        moves, info = run_stages(self.puzzle, s, self.stages)
        moves = pre + moves
        if self.puzzle.kind == "square":
            info["batching"] = dict(self._sq_stats)
        else:
            info["batching"] = {name: dict(b.stats) for name, b in
                                zip(("centers", "wings"), self.batched)}
        return SolveResult(moves=canonicalize(moves), wall_time_s=time.time() - t0, info=info)
