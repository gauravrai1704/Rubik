"""Stage plans and the summed per-cluster stage heuristic (technical basis, Section 9.1–9.2).

A stage targets some clusters. Its exact goal: those clusters, and every cluster solved
by an earlier stage, show their solved stickers. Its heuristic:

    h(state) = Σ_{targets} h_c(state) + keep_weight · Σ_{earlier clusters} h_c(state)

The second sum penalises breaking work already done. It is not admissible; beam search
does not need it to be.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...engine.orbits import Orbit, clusters
from ...engine.puzzle import PuzzleSpec


@dataclass
class Stage:
    name: str
    targets: list[Orbit]


def _chunks(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), max(1, size))]


def plan_stages(puzzle: PuzzleSpec, centers_per_stage: int = 1, wings_per_stage: int = 1,
                edges_per_stage: int = 2, interior_per_stage: int = 3, order: str = "edges_first") -> list[Stage]:
    """Default stage order.

    Cube, ``order="edges_first"`` (default, Demaine's order): fixed centers → corners →
    middle edges → wing clusters → center clusters. Inner-slice moves never disturb
    corners or middle edges, so later stages have cheap moves that keep earlier work.
    Cube, ``order="reduction"``: fixed centers → center clusters → middle edges → wings →
    corners (the classical reduction order of Method 2).
    Square (Demaine order, Lemma 3 then Lemma 2): frame (corner, center, edge-cross
    clusters) → edge clusters → cross clusters → interior clusters, outside in."""
    cl = clusters(puzzle)
    by = lambda *kinds: [c for c in cl if c.kind in kinds]  # noqa: E731
    stages: list[Stage] = []
    if puzzle.kind == "cube":
        centers = [Stage(f"centers {i + 1}", g)
                   for i, g in enumerate(_chunks(by("center_plus", "center_x", "center_oblique"), centers_per_stage))]
        edges = ([Stage("middle edges", by("midge"))] if by("midge") else []) + [
            Stage(f"wings {i + 1}", g) for i, g in enumerate(_chunks(by("wing"), wings_per_stage))]
        if by("center_fixed"):
            stages.append(Stage("fixed centers", by("center_fixed")))
        if order == "edges_first":
            stages += [Stage("corners", by("corner"))] + edges + centers
        elif order == "reduction":
            stages += centers + edges + [Stage("corners", by("corner"))]
        else:
            raise ValueError(f"unknown order {order!r}")
    elif puzzle.kind == "square":
        stages.append(Stage("frame", by("sq_corner", "sq_center", "sq_edge_cross")))
        for i, g in enumerate(_chunks(by("sq_edge"), edges_per_stage)):
            stages.append(Stage(f"edges {i + 1}", g))
        for i, g in enumerate(_chunks(by("sq_cross"), interior_per_stage)):
            stages.append(Stage(f"cross {i + 1}", g))
        for i, g in enumerate(_chunks(by("sq_interior"), interior_per_stage)):
            stages.append(Stage(f"interior {i + 1}", g))
    else:
        raise ValueError("Method 5B supports cubes and squares")
    return [s for s in stages if s.targets]


class StageHeuristic:
    """Weighted sum of per-cluster heuristics, batched by cluster signature."""

    def __init__(self, puzzle: PuzzleSpec, weighted: list[tuple[Orbit, float]], bank, device: str = "cpu"):
        self.device = torch.device(device)
        groups: dict[str, list[tuple[Orbit, float]]] = {}
        for c, w in weighted:
            groups.setdefault(c.signature, []).append((c, w))
        self.groups = []
        for sig, items in groups.items():
            model = bank.get(items[0][0])
            idx = torch.as_tensor(np.stack([c.stickers for c, _ in items]), dtype=torch.long, device=self.device)
            w = torch.as_tensor([w for _, w in items], dtype=torch.float32, device=self.device)
            self.groups.append((model, idx, w))

    @torch.no_grad()
    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        B = states.shape[0]
        total = torch.zeros(B, device=states.device)
        for model, idx, w in self.groups:
            n, K = idx.shape
            x = states[:, idx].reshape(B * n, K)
            h = model(x).float().reshape(B, n)
            total += (h * w).sum(1)
        return total
