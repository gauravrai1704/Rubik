"""Benchmark datasets (technical basis, Section 3.6 and 10.1).

One JSON-lines file per (puzzle, N, dataset). Each row::

    {"id": "cube-5-d30-0007", "puzzle": "cube", "N": 5, "state": [...], "scramble": "R 2U' ...",
     "depth": 30, "seed": 0, "optimal_len": null}

``depth`` = scramble length, or ``"random"`` for an approximately uniform random state.
Files live in ``data/<puzzle>/N<N>/<name>.jsonl``; generate them, never edit by hand.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..engine import is_reachable, make, random_state, scramble, to_notation

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


@dataclass
class Instance:
    id: str
    puzzle: str
    N: int
    state: np.ndarray
    scramble: str | None
    depth: int | str
    seed: int
    optimal_len: int | None = None

    def to_json(self) -> dict:
        return {
            "id": self.id, "puzzle": self.puzzle, "N": self.N, "state": self.state.tolist(),
            "scramble": self.scramble, "depth": self.depth, "seed": self.seed, "optimal_len": self.optimal_len,
        }

    @staticmethod
    def from_json(d: dict) -> "Instance":
        return Instance(d["id"], d["puzzle"], int(d["N"]), np.array(d["state"], dtype=np.uint8),
                        d.get("scramble"), d.get("depth"), int(d.get("seed", 0)), d.get("optimal_len"))


def default_count(N: int) -> int:
    """100 instances for N <= 7, 20 above (Section 10.1)."""
    return 100 if N <= 7 else 20


def generate(puzzle: str, N: int, count: int, depth: int | str, seed: int = 0) -> list[Instance]:
    P = make(puzzle, N)
    rng = np.random.default_rng([seed, N, 0 if depth == "random" else int(depth), 0 if puzzle == "cube" else 1])
    out = []
    for i in range(count):
        if depth == "random":
            st, sc = random_state(P, rng), None
        else:
            mv = scramble(P, int(depth), rng)
            st, sc = P.apply(P.solved, mv), to_notation(mv, P)
        ok, why = is_reachable(P, st)
        assert ok, why
        out.append(Instance(f"{puzzle}-{N}-d{depth}-{i:04d}", puzzle, N, st.astype(np.uint8), sc, depth, seed))
    return out


def path_for(puzzle: str, N: int, name: str, root: Path | None = None) -> Path:
    return (root or DATA_ROOT) / puzzle / f"N{N}" / f"{name}.jsonl"


def save(instances: list[Instance], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ins in instances:
            f.write(json.dumps(ins.to_json()) + "\n")
    return path


def load(path: Path) -> list[Instance]:
    with open(path) as f:
        return [Instance.from_json(json.loads(line)) for line in f if line.strip()]


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
