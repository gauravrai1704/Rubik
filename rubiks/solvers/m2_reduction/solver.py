"""Method 2: Reduction + Kociemba two-phase — owner: Sunaina (implemented by Gaurav).

Spec: technical basis, Section 5; checklist in ./README.md.

The standard way to solve any N: one cluster at a time, each with sequences that leave
every other cluster alone. Every stage is a separate function ``(puzzle, state) -> moves``
so Method 5B can swap stages, and moves per stage are logged in ``SolveResult.info``.

Cube stages (colours relabelled first so the goal is the reference orientation):

1. ``wing_parity``: an odd wing orbit gets one inner-slice quarter turn (Demaine Lemma 7).
   Done first: it disturbs centers, and nothing later changes wing parity.
2. ``centers``: each center orbit with pure 3-cycles ``s [A, B C B'] s'``.
3. ``three``: corners, middle edges and fixed centers by Kociemba's two-phase algorithm on
   outer layers, which keep centers in place; PLL parity on even N is one outer turn.
4. ``wings``: each wing orbit with pure 3-cycles ``s [X, Y Z Y'] s'`` (this is the edge
   pairing: wings go home, so OLL parity never reaches the 3×3×3 stage).

Square stages: frame (exact), edge groups (Lemma 3, exact), interior clusters one at a time
(Lemma 2) — the one-cluster-at-a-time algorithm Method 3 parallelises.

Contract (do not change): subclass ``Solver``, keep the registered name ``m2_reduction``, return
engine ``Move`` objects in ``SolveResult.moves``. The harness re-applies them and marks
``solved`` itself.
"""
from __future__ import annotations

import time

import numpy as np

from ...engine import PuzzleSpec, canonicalize
from ...engine.orbits import clusters
from ...engine.puzzle import Move
from ..base import Budget, Solver, SolveResult, register
from ..cluster_tools import CycleLibrary, greedy_solve, orient, parity, piece_permutation
from ..square_plan import square_plan
from ..three_stage import solve_3x3_stage

CENTER_KINDS = ("center_plus", "center_x", "center_oblique")


def wing_parity(puzzle: PuzzleSpec, state: np.ndarray) -> list[Move]:
    """One inner-slice quarter turn for every wing orbit whose permutation is odd."""
    out = []
    for c in clusters(puzzle):
        if c.kind != "wing":
            continue
        if parity(piece_permutation(puzzle, c, state)):
            k = min(int(puzzle.sticker_layer[s, a]) for s in c.stickers for a in range(3)
                    if 0 < puzzle.sticker_layer[s, a] < puzzle.N - 1)
            out.append(Move(0, k, 1))
    return out


class ClusterSolver:
    """Per-cluster 3-cycle libraries for one cube size, built once and reused."""

    def __init__(self, puzzle: PuzzleSpec):
        self.puzzle = puzzle
        self._libs: dict[int, CycleLibrary] = {}

    def library(self, c) -> CycleLibrary:
        if c.index not in self._libs:
            self._libs[c.index] = CycleLibrary(self.puzzle, c)
        return self._libs[c.index]

    def solve_kinds(self, state: np.ndarray, kinds) -> list[Move]:
        P = self.puzzle
        out: list[Move] = []
        for c in clusters(P):
            if c.kind not in kinds:
                continue
            x = state[c.stickers]
            if np.array_equal(x, c.solved):
                continue
            ids, _ = greedy_solve(self.library(c), x)
            mv = [P.moves[i] for i in ids]
            state = P.apply(state, mv)
            out += mv
        return out


def run_stages(puzzle: PuzzleSpec, state: np.ndarray, stages) -> tuple[list[Move], dict]:
    """Apply ``stages`` = [(name, fn(state) -> moves)] in order; log moves per stage."""
    moves, info = [], {"stages": []}
    for name, fn in stages:
        t = time.time()
        mv = fn(state)
        state = puzzle.apply(state, mv)
        moves += mv
        info["stages"].append({"stage": name, "moves": len(canonicalize(mv)), "time_s": round(time.time() - t, 3)})
    return moves, info


def square_stages(puzzle: PuzzleSpec):
    plan = square_plan(puzzle.N)
    ids = lambda f: (lambda s: [puzzle.moves[i] for i in f(s)])  # noqa: E731
    stages = [("frame", ids(plan.frame.solve))]
    stages += [(name, ids(g.solve)) for name, g in plan.groups]

    def interiors(state):
        out = []
        for ic in plan.interiors:
            mv = ids(ic.stage.solve)(state)
            state = puzzle.apply(state, mv)
            out += mv
        return out

    if plan.interiors:
        stages.append(("interiors", interiors))
    return stages


def cube_stages(puzzle: PuzzleSpec, cs: ClusterSolver):
    stages = [("wing_parity", lambda s: wing_parity(puzzle, s)),
              ("centers", lambda s: cs.solve_kinds(s, CENTER_KINDS)),
              ("three", lambda s: solve_3x3_stage(puzzle, s)),
              ("wings", lambda s: cs.solve_kinds(s, ("wing",)))]
    if puzzle.N == 3:
        stages = [stages[2]]
    return stages


@register("m2_reduction")
class ReductionSolver(Solver):
    def __init__(self, puzzle: PuzzleSpec, budget: Budget | None = None, **kwargs):
        super().__init__(puzzle, budget)
        if puzzle.kind == "cube":
            self.cs = ClusterSolver(puzzle)
            self.stages = cube_stages(puzzle, self.cs)
        elif puzzle.kind == "square":
            self.stages = square_stages(puzzle)
        else:
            raise ValueError("Method 2 supports cubes and squares")

    def solve(self, state: np.ndarray) -> SolveResult:
        t0 = time.time()
        s, _ = orient(self.puzzle, state)
        moves, info = run_stages(self.puzzle, s, self.stages)
        return SolveResult(moves=canonicalize(moves), wall_time_s=time.time() - t0, info=info)
