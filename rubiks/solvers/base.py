"""The interface every method implements (technical basis, Section 3.5).

A solver gets a :class:`PuzzleSpec` and a :class:`Budget`, and returns a
:class:`SolveResult` for one state. It never decides whether it succeeded: the harness
re-applies the moves with the engine and sets ``solved`` itself.

Register a solver so the harness can find it by name::

    @register("m1_idastar")
    class IDAStarSolver(Solver):
        ...
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..engine.puzzle import Move, PuzzleSpec


@dataclass
class Budget:
    wall_time_s: float = 300.0
    max_nodes: int | None = None
    max_gpu_mb: float | None = None
    max_host_mb: float | None = None
    device: str = "cpu"

    def deadline(self) -> float:
        return time.time() + self.wall_time_s


@dataclass
class SolveResult:
    moves: list[Move]
    solved: bool = False  # set by the harness after re-applying moves, not by the solver
    wall_time_s: float = 0.0
    nodes_expanded: int = 0
    peak_gpu_mb: float = 0.0
    peak_host_mb: float = 0.0
    info: dict = field(default_factory=dict)  # method-specific extras (per-stage lengths, ...)


class Solver(ABC):
    name: str = "base"

    def __init__(self, puzzle: PuzzleSpec, budget: Budget | None = None, **kwargs):
        self.puzzle = puzzle
        self.budget = budget or Budget()

    @abstractmethod
    def solve(self, state: np.ndarray) -> SolveResult:
        ...


REGISTRY: dict[str, Callable[..., Solver]] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        REGISTRY[name] = cls
        return cls

    return deco


def get_solver(name: str, puzzle: PuzzleSpec, budget: Budget | None = None, **kwargs) -> Solver:
    from . import load_all  # noqa: F401  (imports every method so it registers)

    load_all()
    if name not in REGISTRY:
        raise KeyError(f"unknown solver {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](puzzle, budget, **kwargs)
