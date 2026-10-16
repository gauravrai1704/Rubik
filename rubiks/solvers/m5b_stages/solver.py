"""Method 5B solver: learned search inside reduction stages (technical basis, Section 9).

For each stage in :func:`plan_stages`, run the shared beam search with the stage's
summed heuristic until the stage goal holds. A stage that fails is retried with a 4×
wider beam, up to ``max_width``. Per-stage lengths and widths are reported in ``info``.
"""
from __future__ import annotations

import time

import numpy as np

from ...engine.moves import canonicalize
from ...search.beam import BeamConfig, BeamSearch, subset_goal
from ...search.macros import default_macros
from ..base import Budget, Solver, SolveResult, register
from .cluster_models import ModelBank
from .stages import StageHeuristic, plan_stages


@register("m5b_stages")
class StagedSolver(Solver):
    def __init__(self, puzzle, budget: Budget | None = None, width: int = 256, max_width: int = 8192,
                 stage_steps: int | None = None, keep_weight: float = 1.0, tier: str = "device",
                 packed: bool = True, chunk: int = 2048, bank: ModelBank | None = None,
                 macros: bool | None = None, future_weight: float = 0.0, heuristic_factory=None,
                 exact_weight: float = 0.5, exact_cap: int = 100_000, center_macros_only: bool = True,
                 cheap_width_budget: int = 1_200_000, **plan_kwargs):
        super().__init__(puzzle, budget)
        self.width, self.max_width = width, max_width
        self.stage_steps = stage_steps or (40 if puzzle.kind == "square" else 60)
        self.keep_weight = keep_weight
        self.device = self.budget.device
        self.tier, self.packed, self.chunk = tier, packed, chunk
        # corners: the exact 3-class tables alone beat max(MLP, tables) (docs/STATUS.md)
        self.bank = bank or ModelBank(device=self.device, allow_missing=heuristic_factory is not None,
                                      tables_only=("corner",))
        self.stages = plan_stages(puzzle, **plan_kwargs)
        # macro actions (commutators) on by default: they remove the heuristic's hills
        use_macros = True if macros is None else macros
        self.actions = default_macros(puzzle) if use_macros else []
        self.future_weight = future_weight
        self.exact_weight = exact_weight
        self.exact_cap = exact_cap
        # a stage with few actions may widen past max_width while width x actions <= this budget
        self.cheap_width_budget = cheap_width_budget
        # center stages after every edge/corner is solved: search only pure-center commutators,
        # which keep all earlier work by construction (verified in search.macros)
        self.center_macros_only = center_macros_only and puzzle.kind == "cube"
        self._center_macros = None
        # heuristic_factory(puzzle, [(cluster, weight), ...]) -> callable; default: per-cluster bank.
        # Method 5A plugs its cluster-token model in here (technical basis, Section 8.5).
        self.heuristic_factory = heuristic_factory
        self._heuristics: dict[int, StageHeuristic] = {}

    def _stage_setup(self, i: int):
        """Actions that touch this stage's targets, and the heuristic over clusters they move.

        Moves that never touch a target cannot make progress on it; leaving them out cuts
        the branching factor from O(N²) macros to a handful. Protected clusters that no
        allowed action touches stay constant, so they are left out of the heuristic too."""
        if i in self._heuristics:
            return self._heuristics[i]
        P = self.puzzle
        targets = self.stages[i].goal_stickers()
        tset = set(int(x) for x in targets)

        def touches(perm: np.ndarray, stickers) -> bool:
            st = np.asarray(sorted(stickers))
            return bool(np.any(perm[st] != st))

        singles = [m for m in range(len(P.moves)) if touches(P.move_perms[m], tset)]
        pool = self.actions
        if self.center_macros_only and self._is_late_center_stage(i):
            if self._center_macros is None:
                from ...search.macros import cube_center_macros

                self._center_macros = cube_center_macros(P)
            touching = [a for a in self._center_macros if self._touches_seq(a, tset)]
            if touching:
                singles, pool = [], touching
        ident = np.arange(P.num_stickers)
        macros, moved = [], set()
        for a in pool:
            q = ident
            for m in a:
                q = q[P.move_perms[m]]
            if touches(q, tset):
                macros.append(a)
                moved |= set(np.nonzero(q != ident)[0].tolist())
        for m in singles:
            moved |= set(np.nonzero(P.move_perms[m] != ident)[0].tolist())
        rel = lambda c: bool(moved & set(int(x) for x in c.stickers))  # noqa: E731
        weighted = [(c, self.keep_weight) for s in self.stages[:i] for c in s.targets if rel(c)]
        weighted += [(c, 1.0) for c in self.stages[i].targets]
        if self.future_weight > 0:
            weighted += [(c, self.future_weight) for s in self.stages[i + 1 :] for c in s.targets if rel(c)]
        if self.heuristic_factory is not None:
            h = self.heuristic_factory(P, weighted)
        else:
            h = StageHeuristic(P, weighted, self.bank, self.device,
                               exact_stickers=self._goal(i), exact_weight=self.exact_weight)
        self._heuristics[i] = (h, singles, macros)
        return self._heuristics[i]

    def _exact_table(self, i: int, singles: list[int]):
        """BFS distance table over this stage's goal stickers, using only the moves that touch
        them, if that projected puzzle has at most ``exact_cap`` states (e.g. the Square frame:
        always a 3×3 Square, at most 768 states). None otherwise."""
        key = ("exact", i)
        if key in self._heuristics:
            return self._heuristics[key]
        table = None
        if self.exact_cap and len(singles) <= 16:
            P = self.puzzle
            g = self._goal(i)
            loc = {int(s): k for k, s in enumerate(g)}
            gp = [np.array([loc[int(x)] for x in P.move_perms[m][g]]) for m in singles]
            start = P.solved[g]
            seen = {start.tobytes(): 0}
            frontier, d = [start], 0
            while frontier and len(seen) <= self.exact_cap:
                d += 1
                nxt = []
                for st in frontier:
                    for p in gp:
                        k = st[p]
                        b = k.tobytes()
                        if b not in seen:
                            seen[b] = d
                            nxt.append(k)
                frontier = nxt
            if len(seen) <= self.exact_cap:
                table = (g, gp, seen)
        self._heuristics[key] = table
        return table

    def _solve_exact(self, cur: np.ndarray, table, singles: list[int]):
        """Optimal moves for the stage by greedy descent on the exact distance table."""
        g, gp, dist = table
        x = cur[g]
        d = dist.get(x.tobytes())
        if d is None:
            return None
        moves = []
        while d > 0:
            for m, p in zip(singles, gp):
                y = x[p]
                dy = dist.get(y.tobytes())
                if dy is not None and dy < d:
                    moves.append(self.puzzle.moves[m])
                    x, d = y, dy
                    break
            else:
                return None
        return moves

    def _touches_seq(self, seq, tset) -> bool:
        q = np.arange(self.puzzle.num_stickers)
        for m in seq:
            q = q[self.puzzle.move_perms[m]]
        st = np.asarray(sorted(tset))
        return bool(np.any(q[st] != st))

    def _is_late_center_stage(self, i: int) -> bool:
        """True if stage i targets only centers and every non-center cluster is already solved."""
        if not all(c.kind.startswith("center_") and c.kind != "center_fixed" for c in self.stages[i].targets):
            return False
        done = {c.index for s in self.stages[:i] for c in s.targets}
        from ...engine.orbits import clusters

        return all(c.index in done for c in clusters(self.puzzle) if not c.kind.startswith("center_")
                   or c.kind == "center_fixed")

    def _goal(self, i: int) -> np.ndarray:
        return np.unique(np.concatenate([s.goal_stickers() for s in self.stages[: i + 1]]))

    def solve(self, state: np.ndarray) -> SolveResult:
        P = self.puzzle
        t0 = time.time()
        deadline = t0 + self.budget.wall_time_s
        cur = np.asarray(state).copy()
        all_moves, info = [], {"stages": []}
        expanded, peak_gpu, peak_host = 0, 0.0, 0.0
        for i, stage in enumerate(self.stages):
            stickers = self._goal(i)
            if np.array_equal(cur[stickers], P.solved[stickers]):
                info["stages"].append({"stage": stage.name, "moves": 0, "width": 0})
                continue
            goal = subset_goal(P, stickers, self.device)
            h, singles, macros = self._stage_setup(i)
            table = self._exact_table(i, singles)
            if table is not None:
                mv = self._solve_exact(cur, table, singles)
                if mv is not None:
                    cur = P.apply(cur, mv)
                    all_moves += mv
                    info["stages"].append({"stage": stage.name, "moves": len(mv), "width": "exact"})
                    continue
            width = self.width
            res = None
            n_actions = len(singles) + len(macros)
            stage_max = max(self.max_width, min(1 << 16, self.cheap_width_budget // max(1, n_actions)))
            while width <= stage_max:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                cfg = BeamConfig(width=width, max_steps=self.stage_steps, device=self.device, tier=self.tier,
                                 packed=self.packed, chunk=self.chunk, time_limit_s=remaining)
                res = BeamSearch(P, h, cfg, move_subset=singles, actions=macros).solve(cur, goal)
                expanded += res.expanded
                peak_gpu, peak_host = max(peak_gpu, res.peak_gpu_mb), max(peak_host, res.peak_host_mb)
                if res.found:
                    break
                if width * 4 > stage_max:
                    if width < stage_max:
                        width = stage_max
                        continue
                    break
                width *= 4
            if res is None or not res.found:
                info["failed_stage"] = stage.name
                info["stages"].append({"stage": stage.name, "moves": None, "width": width})
                break
            cur = P.apply(cur, res.moves)
            all_moves += res.moves
            info["stages"].append({"stage": stage.name, "moves": len(res.moves), "width": width})
        moves = canonicalize(all_moves)
        return SolveResult(moves=moves, wall_time_s=time.time() - t0, nodes_expanded=expanded,
                           peak_gpu_mb=peak_gpu, peak_host_mb=peak_host, info=info)
