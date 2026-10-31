"""Method 5C: net-gain scheduling of bulk commutators — owner: Gaurav.

Method 3 (Demaine's parallel algorithm, with 3-cycle batching) with a new scheduler: each
round applies the bulk commutator whose block of layers X × Y has the best *net* slots
fixed per move, cells that do not help included when the block still pays (see
``scheduler.py``), with setup sharing and one step of lookahead. Every other stage (wing
parity, the 3×3×3 stage, the Square plan) is Method 3's, so 5C vs 3 isolates the schedule.

Contract: subclass ``Solver``, registered as ``m5c_schedule``, return engine ``Move`` objects.
"""
from __future__ import annotations

import time

import numpy as np

from ...engine import PuzzleSpec, canonicalize
from ...engine.orbits import clusters
from ..base import Budget, Solver, SolveResult, register
from ..cluster_tools import CycleLibrary, prepare
from ..m2_reduction.solver import CENTER_KINDS, run_stages, wing_parity
from ..m3_demaine.batching import TypeBank, center_templates, wing_templates
from ..m3_demaine.solver import DemaineSolver
from ..three_stage import solve_3x3_stage
from .scheduler import NetGainScheduler


class ScheduledKinds:
    def __init__(self, puzzle: PuzzleSpec, kinds, templates, setup_depth: int = 1, **sched):
        self.puzzle = puzzle
        cl = [c for c in clusters(puzzle) if c.kind in kinds]
        self.active = bool(cl)
        self.stats: dict = {}
        if cl:
            bank = TypeBank(puzzle, cl, templates, setup_depth)
            fallback = {c.index: CycleLibrary(puzzle, c) for c in cl}
            self.sched = NetGainScheduler(puzzle, bank, fallback, **sched)

    def __call__(self, state: np.ndarray):
        if not self.active:
            return []
        ids, self.stats = self.sched.solve(state)
        return [self.puzzle.moves[i] for i in ids]


@register("m5c_schedule")
class ScheduledSolver(Solver):
    def __init__(self, puzzle: PuzzleSpec, budget: Budget | None = None, setup_depth: int = 1,
                 lookahead: int | None = None, beam: int = 4, top_types: int = 24, share_setups: bool = True,
                 exact_orientation: bool = False, **kwargs):
        super().__init__(puzzle, budget)
        self.exact_orientation = exact_orientation  # Kaggle Santa: the exact solved state, not a rotation
        P = puzzle
        self._m3 = None
        if P.kind == "cube" and P.N == 3:
            self._m3 = DemaineSolver(P, budget, exact_orientation=exact_orientation)  # nothing to schedule: Method 3 as is
            return
        if P.kind == "square":
            from ..square_plan import square_plan
            from .scheduler import square_interiors_netgain

            plan = square_plan(P.N)
            ids = lambda f: (lambda s: [P.moves[i] for i in f(s)])  # noqa: E731
            self.stages = [("frame", ids(plan.frame.solve))] + [(n, ids(g.solve)) for n, g in plan.groups]
            self._sq_stats: dict = {}

            def interiors(s):
                # Both schedules take milliseconds here: keep the shorter (never worse than Method 3)
                from ..m3_demaine.batching import square_interiors

                a, sa = square_interiors_netgain(P, plan, s)
                b, sb = square_interiors(P, plan, s)
                la, lb = (len(canonicalize([P.moves[i] for i in x])) for x in (a, b))
                self._sq_stats = {**sa, "chosen": "net-gain"} if la < lb else {**sb, "chosen": "method 3"}
                return [P.moves[i] for i in (a if la < lb else b)]

            if plan.interiors:
                self.stages.append(("interiors (scheduled)", interiors))
            return
        if lookahead is None:  # one step of lookahead costs ~5x time: worth it up to N = 15
            lookahead = 1 if P.N <= 15 else 0
        sched = dict(lookahead=lookahead, beam=beam, top_types=top_types, share_setups=share_setups)
        self.centers = ScheduledKinds(P, CENTER_KINDS, center_templates(P), setup_depth, **sched)
        self.wings = ScheduledKinds(P, ("wing",), wing_templates(P), setup_depth, **sched)
        self.stages = [("wing_parity", lambda s: wing_parity(P, s)),
                       ("centers (scheduled)", self.centers),
                       ("three", lambda s: solve_3x3_stage(P, s)),
                       ("wings (scheduled)", self.wings)]

    def solve(self, state: np.ndarray) -> SolveResult:
        if self._m3 is not None:
            r = self._m3.solve(state)
            r.info["delegated"] = "m3_demaine"
            return r
        t0 = time.time()
        s, pre = prepare(self.puzzle, state, self.exact_orientation)
        moves, info = run_stages(self.puzzle, s, self.stages)
        moves = pre + moves
        if self.puzzle.kind == "square":
            info["schedule"] = dict(self._sq_stats)
        else:
            info["schedule"] = {"centers": dict(self.centers.stats), "wings": dict(self.wings.stats)}
        return SolveResult(moves=canonicalize(moves), wall_time_s=time.time() - t0, info=info)
