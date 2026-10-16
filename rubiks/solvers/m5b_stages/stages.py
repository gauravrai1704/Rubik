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
    """``targets``: clusters whose heuristics guide this stage. ``stickers``: the exact goal
    subset (default: every sticker of the targets). A big cluster can be split over several
    stages, each adding a few more of its pieces to the goal."""

    name: str
    targets: list[Orbit]
    stickers: np.ndarray | None = None

    def goal_stickers(self) -> np.ndarray:
        if self.stickers is not None:
            return self.stickers
        return np.concatenate([c.stickers for c in self.targets])


def _split_cluster(puzzle: PuzzleSpec, c: Orbit, pieces: int, name: str) -> list[Stage]:
    """Split one cluster into stages of ``pieces`` cubies each (canonical order)."""
    by_cubie: dict[tuple, list[int]] = {}
    for s in c.stickers:
        by_cubie.setdefault(tuple(puzzle.sticker_pos[s]), []).append(int(s))
    cubies = list(by_cubie.values())
    if pieces <= 0 or len(cubies) <= pieces:
        return [Stage(name, [c])]
    out = []
    for i in range(0, len(cubies), pieces):
        st = np.array([s for cb in cubies[i : i + pieces] for s in cb])
        out.append(Stage(f"{name}.{i // pieces + 1}", [c], st))
    return out


def _chunks(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), max(1, size))]


def plan_stages(puzzle: PuzzleSpec, centers_per_stage: int = 1, wings_per_stage: int = 1,
                edges_per_stage: int = 2, interior_per_stage: int = 3, order: str = "edges_first",
                pieces_per_stage: int = 6) -> list[Stage]:
    """Default stage order.

    Cube, ``order="edges_first"`` (default, Demaine's order): fixed centers → corners →
    middle edges → wing clusters → center clusters. Inner-slice moves never disturb
    corners or middle edges, so later stages have cheap moves that keep earlier work.
    Cube, ``order="reduction"``: fixed centers → center clusters → middle edges → wings →
    corners (the classical reduction order of Method 2).
    Square (Demaine order, Lemma 3 then Lemma 2): frame (corner, center, edge-cross
    clusters) → edge clusters → cross clusters → interior clusters, outside in.

    ``pieces_per_stage`` (cube): split each wing and center cluster into stages of that many
    pieces (0 = whole clusters). Smaller stages = shorter searches with sharper signal."""
    cl = clusters(puzzle)
    by = lambda *kinds: [c for c in cl if c.kind in kinds]  # noqa: E731
    stages: list[Stage] = []
    if puzzle.kind == "cube":
        if pieces_per_stage > 0:
            centers = [st for i, c in enumerate(by("center_plus", "center_x", "center_oblique"))
                       for st in _split_cluster(puzzle, c, pieces_per_stage, f"centers {i + 1}")]
            wings = [st for i, c in enumerate(by("wing"))
                     for st in _split_cluster(puzzle, c, pieces_per_stage, f"wings {i + 1}")]
        else:
            centers = [Stage(f"centers {i + 1}", g) for i, g in
                       enumerate(_chunks(by("center_plus", "center_x", "center_oblique"), centers_per_stage))]
            wings = [Stage(f"wings {i + 1}", g) for i, g in enumerate(_chunks(by("wing"), wings_per_stage))]
        edges = ([Stage("middle edges", by("midge"))] if by("midge") else []) + wings
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

    def __init__(self, puzzle: PuzzleSpec, weighted: list[tuple[Orbit, float]], bank, device: str = "cpu",
                 exact_stickers: np.ndarray | None = None, exact_weight: float = 0.0):
        """``exact_stickers`` / ``exact_weight``: add weight × (number of those stickers not
        yet at their solved label), an exact local signal for split stages."""
        self.device = torch.device(device)
        merged: dict[int, tuple[Orbit, float]] = {}
        for c, w in weighted:  # a cluster split over several stages appears once
            merged[c.index] = (c, max(w, merged.get(c.index, (c, 0.0))[1]))
        groups: dict[str, list[tuple[Orbit, float]]] = {}
        for c, w in merged.values():
            groups.setdefault(c.signature, []).append((c, w))
        self.exact = None
        if exact_stickers is not None and exact_weight > 0:
            idx = torch.as_tensor(np.asarray(exact_stickers), dtype=torch.long, device=self.device)
            tgt = torch.as_tensor(puzzle.solved[np.asarray(exact_stickers)], device=self.device)
            self.exact = (idx, tgt, float(exact_weight))
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
        if self.exact is not None:
            idx, tgt, w = self.exact
            total += w * (states[:, idx] != tgt.to(states.dtype)).sum(1).float()
        return total
