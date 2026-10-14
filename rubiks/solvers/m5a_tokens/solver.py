"""Method 5A solver (technical basis, Section 8.5).

Two modes:

* ``mode="full"`` — beam search over the whole puzzle, scoring children with the model's
  value head (or its move head with ``use_q=True``: one forward pass per parent).
* ``mode="staged"`` (default for N > 7) — Method 5B's stage plan, with each stage's
  heuristic taken from the model's *per-token costs* over that stage's clusters. This is
  the "5A and 5B merge into one system" path: one model, every N, every stage.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ...engine.moves import canonicalize
from ...search.beam import BeamConfig, BeamSearch
from ..base import Budget, Solver, SolveResult, register
from .model import Tokenizer, load

DEFAULT_CKPT = Path(__file__).resolve().parents[3] / "checkpoints" / "m5a"


class ValueHeuristic:
    def __init__(self, model, tok: Tokenizer, use_q: bool = False, batch: int = 4096):
        self.model, self.tok, self.batch = model, tok, batch
        if use_q:
            self.score_children = self._score_children

    @torch.no_grad()
    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        out = [self.model(states[i : i + self.batch], self.tok)[0] for i in range(0, len(states), self.batch)]
        return torch.cat(out)

    @torch.no_grad()
    def _score_children(self, parents: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.model.move_values(parents[i : i + self.batch], self.tok)
                          for i in range(0, len(parents), self.batch)])


class TokenStageHeuristic:
    """Weighted sum of the model's per-token costs over a stage's clusters."""

    def __init__(self, model, tok: Tokenizer, weighted, batch: int = 4096):
        self.model, self.tok, self.batch = model, tok, batch
        w = torch.zeros(tok.num_tokens, device=tok.kind_id.device)
        for c, wt in weighted:
            w[tok.cluster_token[c.index]] = wt
        self.w = w

    @torch.no_grad()
    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        out = []
        for i in range(0, len(states), self.batch):
            _, costs = self.model(states[i : i + self.batch], self.tok)
            out.append((costs * self.w).sum(1))
        return torch.cat(out)


def find_checkpoint(kind: str) -> Path:
    p = DEFAULT_CKPT / f"{kind}.pt"
    if not p.exists():
        raise FileNotFoundError(
            f"no Method 5A checkpoint at {p}. Train one, e.g.\n"
            f"  python -m rubiks.solvers.m5a_tokens.train --puzzle {kind} --train-N 4-8 --out {p}")
    return p


@register("m5a_tokens")
class ClusterTokenSolver(Solver):
    def __init__(self, puzzle, budget: Budget | None = None, checkpoint: str | None = None, mode: str | None = None,
                 width: int = 1024, max_steps: int | None = None, use_q: bool = False, **staged_kwargs):
        super().__init__(puzzle, budget)
        self.device = self.budget.device
        path = Path(checkpoint) if checkpoint else find_checkpoint(puzzle.kind)
        self.model, self.meta = load(path, self.device)
        self.tok = Tokenizer(puzzle, self.device)
        self.model.check_supported(self.tok)
        self.mode = mode or ("full" if puzzle.N <= 7 else "staged")
        self.width = width
        self.max_steps = max_steps or int(6 * self.tok.scale) + 20
        self.use_q = use_q
        self._staged = None
        if self.mode == "staged":
            from ..m5b_stages.solver import StagedSolver

            factory = lambda P, weighted: TokenStageHeuristic(self.model, self.tok, weighted)  # noqa: E731
            self._staged = StagedSolver(puzzle, self.budget, width=width, heuristic_factory=factory, **staged_kwargs)

    def solve(self, state: np.ndarray) -> SolveResult:
        if self._staged is not None:
            r = self._staged.solve(state)
            r.info["mode"] = "staged"
            return r
        t0 = time.time()
        h = ValueHeuristic(self.model, self.tok, self.use_q)
        cfg = BeamConfig(width=self.width, max_steps=self.max_steps, device=self.device,
                         time_limit_s=self.budget.wall_time_s)
        res = BeamSearch(self.puzzle, h, cfg).solve(state)
        return SolveResult(moves=canonicalize(res.moves), wall_time_s=time.time() - t0, nodes_expanded=res.expanded,
                           peak_gpu_mb=res.peak_gpu_mb, peak_host_mb=res.peak_host_mb,
                           info={"mode": "full", "found": res.found, "steps": res.steps})
