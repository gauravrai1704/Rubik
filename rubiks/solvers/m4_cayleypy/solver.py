"""Method 4: CayleyPy reproduction — owner: Shaurya.

Spec: technical basis, Section 7; checklist in ./README.md.

Reproduces the current ML state of the art (Chervov et al., NeurIPS 2025) inside our
engine: the per-N baseline that 5A must match while using one model for all N.

One ResMLP per puzzle size predicts the random-walk distance to solved; beam search
expands every move of the paper's move set (QTM on the cube), scores the children with
the network, keeps the W best, drops duplicates by hash and stops at the solved state or
after 200 steps (Section 7.1). Several checkpoints form an ensemble: each agent searches
and the shortest solution wins.

Contract (do not change): subclass ``Solver``, keep the registered name ``m4_cayleypy``, return
engine ``Move`` objects in ``SolveResult.moves``. The harness re-applies them and marks
``solved`` itself. Run ``pytest tests/test_m4_cayleypy.py`` and
``python -m rubiks.bench run --solver m4_cayleypy --puzzle square --N 3-5 --dataset d20``.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ...engine import PuzzleSpec, canonicalize
from ...search.beam import BeamConfig, BeamSearch
from ..base import Budget, Solver, SolveResult, register
from .model import ResMLP, load, one_hot, successor_mask

DEFAULT_CKPT = Path(__file__).resolve().parents[3] / "checkpoints" / "m4"


class ResMLPHeuristic:
    """(B, S) uint8 states -> (B,) predicted distance. FP16 inference on GPU, as in the paper."""

    def __init__(self, model: ResMLP, device: str = "cpu"):
        self.dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
        self.model = model.to(device=device, dtype=self.dtype).eval()

    @torch.no_grad()
    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        return self.model(one_hot(states, dtype=self.dtype)).float()


def find_checkpoints(puzzle: PuzzleSpec) -> list[Path]:
    """``checkpoints/m4/<kind><N>.large.pt`` (GPU-trained) if present, else ``<kind><N>.pt``;
    extra ensemble agents are ``<kind><N>.<tag>.large.pt`` / ``<kind><N>.<tag>.pt``."""
    stem = f"{puzzle.kind}{puzzle.N}"
    for suffix in (".large.pt", ".pt"):
        main = DEFAULT_CKPT / f"{stem}{suffix}"
        if main.exists():
            extra = sorted(p for p in DEFAULT_CKPT.glob(f"{stem}.*{suffix}") if p != main and
                           p.name.count(".") == suffix.count(".") + 1)
            return [main] + extra
    raise FileNotFoundError(
        f"no Method 4 checkpoint for {puzzle} in {DEFAULT_CKPT}. Train one, e.g.\n"
        f"  python -m rubiks.solvers.m4_cayleypy.train --puzzle {puzzle.kind} --N {puzzle.N} "
        f"--out {DEFAULT_CKPT / (stem + '.pt')}")


@register("m4_cayleypy")
class CayleyPySolver(Solver):
    def __init__(self, puzzle: PuzzleSpec, budget: Budget | None = None, checkpoint: str | list[str] | None = None,
                 width: int = 2**12, max_width: int | None = None, max_steps: int = 200, tier: str = "device",
                 agents: int | None = None, **kwargs):
        """``width``: beam width W. ``max_width``: if no solution at W, retry at 2W, 4W, ...
        up to this (default: W only, as in the paper). ``checkpoint``: one path or a list
        (ensemble); default every agent found under ``checkpoints/m4/``, at most ``agents``."""
        super().__init__(puzzle, budget)
        self.device = self.budget.device
        if checkpoint is None:
            paths = find_checkpoints(puzzle)
        else:
            paths = [Path(c) for c in ([checkpoint] if isinstance(checkpoint, (str, Path)) else checkpoint)]
        paths = paths[:agents] if agents else paths
        self.agents = []
        for p in paths:
            model, meta = load(p, self.device)
            if meta.get("puzzle") != puzzle.kind or meta.get("N") != puzzle.N:
                raise ValueError(f"{p} was trained on {meta.get('puzzle')}{meta.get('N')}, not {puzzle}")
            self.agents.append((p.name, ResMLPHeuristic(model, self.device), list(meta["move_ids"])))
        self.width = width
        self.max_width = max(width, max_width or width)
        self.max_steps = max_steps
        self.tier = tier

    def _search(self, h: ResMLPHeuristic, subset: list[int], state: np.ndarray, width: int, time_left: float):
        cfg = BeamConfig(width=width, max_steps=self.max_steps, device=self.device, tier=self.tier,
                         time_limit_s=time_left)
        bs = BeamSearch(self.puzzle, h, cfg, move_subset=subset)
        # QTM: allow R R (= R2), forbid only R R' and non-ascending commuting orders
        bs.allowed = torch.as_tensor(successor_mask(self.puzzle, subset), device=bs.dev)
        return bs.solve(state)

    def solve(self, state: np.ndarray) -> SolveResult:
        t0 = time.time()
        deadline = t0 + self.budget.wall_time_s
        best, runs = None, []
        expanded, peak_gpu, peak_host = 0, 0.0, 0.0
        for name, h, subset in self.agents:
            w = self.width
            while True:
                left = deadline - time.time()
                if left <= 0:
                    break
                res = self._search(h, subset, state, w, left)
                expanded += res.expanded
                peak_gpu, peak_host = max(peak_gpu, res.peak_gpu_mb), max(peak_host, res.peak_host_mb)
                moves = canonicalize(res.moves)
                runs.append({"agent": name, "width": w, "found": res.found, "steps": res.steps,
                             "len": len(moves) if res.found else None, "time_s": round(res.wall_time_s, 2)})
                if res.found:
                    if best is None or len(moves) < len(best):
                        best = moves
                    break
                if w >= self.max_width:
                    break
                w *= 2
        return SolveResult(moves=best or [], wall_time_s=time.time() - t0, nodes_expanded=expanded,
                           peak_gpu_mb=peak_gpu, peak_host_mb=peak_host,
                           info={"width": self.width, "agents": len(self.agents), "runs": runs})
