"""Write Method 1's optimal lengths into benchmark datasets (``optimal_len``), and print God's
numbers for the smallest Squares from full BFS (technical basis, Section 4.5).

    python -m rubiks.solvers.m1_idastar.label --puzzle square --N 3-5 --dataset random --time 120
    python -m rubiks.solvers.m1_idastar.label --gods 3
"""
from __future__ import annotations

import argparse
import json

from ...bench import datasets
from ...bench.__main__ import parse_range
from ...engine import make, metrics
from ..base import Budget
from .solver import IDAStarSolver


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzle", choices=["cube", "square"], default="square")
    ap.add_argument("--N", default="3")
    ap.add_argument("--dataset", default="random")
    ap.add_argument("--time", type=float, default=120.0, help="IDA* budget per instance (s)")
    ap.add_argument("--metric", choices=["htm", "qtm"], default="htm")
    ap.add_argument("--gods", default=None, help="full BFS God's number for these Square sizes, e.g. 3")
    a = ap.parse_args(argv)
    if a.gods:
        from .square import gods_number

        for N in parse_range(a.gods):
            print(json.dumps(gods_number(N)))
        return
    for N in parse_range(a.N):
        P = make(a.puzzle, N)
        path = datasets.path_for(a.puzzle, N, a.dataset)
        rows = datasets.load(path)
        solver = IDAStarSolver(P, Budget(wall_time_s=a.time), metric=a.metric)
        done = 0
        for ins in rows:
            r = solver.solve(ins.state)
            if r.info["optimal"] and P.is_solved(P.apply(ins.state, r.moves)):
                ins.optimal_len = metrics(r.moves, P)[a.metric]
                done += 1
            print(f"{ins.id}: {ins.optimal_len}  ({r.wall_time_s:.1f}s)")
        datasets.save(rows, path)
        print(f"{path}: optimal length for {done}/{len(rows)}")


if __name__ == "__main__":
    main()
