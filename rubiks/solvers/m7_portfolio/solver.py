"""Method 7: a symmetry portfolio over deterministic solvers (cube and Rubik's Square).

The same scramble seen from the 24 sides of the cube is 24 different inputs to a solver whose search
is driven by fixed seeds, tie-breaks and move orders; its answers differ in length, and each maps back to
a solution of the original state of the same length (:mod:`rubiks.solvers.symmetry`). Method 7 runs the
inner solver on ``copies`` rotated copies (identity first) and keeps the shortest answer in QTM. With ``inners``
(a list of (solver name, kwargs)) every copy is solved by each inner solver and the shortest answer overall is kept.
On the Square (4 rotations x 2 mirror images = 8 copies) the goal is any solved orientation; on the cube it is
the exact solved state (Santa 2023).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ...engine import PuzzleSpec, metrics
from ...engine.puzzle import Move
from ..base import Budget, Solver, SolveResult, get_solver, register
from ..symmetry import map_back, rotated, symmetries


@register("m7_portfolio")
class SymmetryPortfolio(Solver):
    def __init__(self, puzzle: PuzzleSpec, budget: Budget | None = None, inner: str = "kmcoders",
                 copies: int = 8, workers: int = 1, inner_kwargs: dict | None = None,
                 inners: list | None = None, **kwargs):
        super().__init__(puzzle, budget)
        self.copies = min(copies, len(symmetries(puzzle)))
        self.workers = workers
        specs = inners if inners else [(inner, inner_kwargs or {})]
        self.inners = [(name, get_solver(name, puzzle, budget, **kw)) for name, kw in specs]
        self.inner = self.inners[0][1]
        self.inner_name = "+".join(n for n, _ in specs)
        # identity, its mirror image, then the other rotations each followed by its mirror image
        n_rot = len(puzzle.rotations)
        # optional on-disk record of finished copies (env M7_COPY_CACHE, a directory): a run killed part-way through
        # a puzzle resumes from the copies it already finished
        self.cache_dir = os.environ.get("M7_COPY_CACHE")
        self.spec_key = json.dumps([[n, kw] for n, kw in specs], sort_keys=True, default=str)
        self._cache_lock = threading.Lock()
        self.order = [0, n_rot] + [x for r in range(1, n_rot) for x in (r, n_rot + r)]

    def _solved(self, s) -> bool:
        P = self.puzzle
        return bool(P.is_solved(s)) if P.kind == "square" else bool(np.array_equal(s, P.solved))

    def _cache_file(self, state):
        h = hashlib.sha1(np.asarray(state).tobytes() + self.spec_key.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{self.puzzle.kind}{self.puzzle.N}_{h}.jsonl")

    def _one(self, state, r):
        if self.cache_dir:
            f = self._cache_file(state)
            if os.path.exists(f):
                for line in open(f):
                    c = json.loads(line)
                    if c["r"] == r:
                        return r, [Move(*m) for m in c["moves"]], c["ok"], c["info"]
        out = self._one_run(state, r)
        if self.cache_dir and out[2]:  # a failed copy (e.g. killed for lack of memory) is retried on a rerun
            os.makedirs(self.cache_dir, exist_ok=True)
            row = {"r": r, "moves": [[m.axis, m.layer, m.turn] for m in out[1]], "ok": out[2], "info": out[3]}
            with self._cache_lock, open(self._cache_file(state), "a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
        return out

    def _one_run(self, state, r):
        P = self.puzzle
        best, errors = None, []
        for name, inner in self.inners:
            try:
                res = inner.solve(rotated(P, state, r))
            except Exception as e:  # an inner solver failing on one copy does not sink the portfolio
                res = SolveResult(moves=[], wall_time_s=0.0, info={"error": repr(e)})
            moves = map_back(P, res.moves, r)
            ok = self._solved(P.apply(np.asarray(state), moves))
            info = dict(res.info or {}, inner=name)
            if not ok:
                errors.append({"inner": name, **info})
            if ok and (best is None or metrics(moves, P)["qtm"] < metrics(best[1], P)["qtm"]):
                best = (r, moves, True, info)
        return best or (r, [], False, {"errors": errors})

    def solve(self, state: np.ndarray) -> SolveResult:
        t0 = time.time()
        P = self.puzzle
        rs = self.order[:self.copies]
        with ThreadPoolExecutor(self.workers) as ex:
            runs = list(ex.map(lambda r: self._one(state, r), rs))
        lens = {r: metrics(m, P)["qtm"] if ok else None for r, m, ok, _ in runs}
        good = [(lens[r], i, m) for i, (r, m, ok, _) in enumerate(runs) if ok]
        if not good:
            return SolveResult(moves=[], wall_time_s=time.time() - t0, info={"copies": lens})
        _, i, best = min(good, key=lambda x: (x[0], x[1]))
        return SolveResult(moves=best, wall_time_s=time.time() - t0,
                           info={"inner": self.inner_name, "copies": [lens[r] for r in rs], "best_copy": i,
                                 "copy_info": [inf for *_, inf in runs]})
