"""Baseline B3 (dwalton76's NxNxN solver) on every Santa 2023 cube puzzle of the given sizes (resumable).

    DWALTON76_DIR=... python experiments/b3_santa.py 8 9 10

Harness-style rows are appended to docs/results/santa/dwalton76.jsonl (the file of the earlier N = 4..7 rows);
puzzles already there are skipped.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from rubiks.bench import santa2023
from rubiks.engine import cube, metrics
from rubiks.solvers import Budget, get_solver, load_all

OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "santa" / "dwalton76.jsonl"


def main(sizes: list[int]) -> None:
    if "DWALTON76_DIR" not in os.environ and "KMCODERS_DIR" in os.environ:  # our layout: <dir>/kmc/repo, <dir>/ext/dwalton
        os.environ["DWALTON76_DIR"] = str(Path(os.environ["KMCODERS_DIR"]).parents[1] / "ext" / "dwalton")
    load_all()
    done = {json.loads(l)["id"] for l in OUT.read_text().splitlines() if l.strip()} if OUT.exists() else set()
    for N in sizes:
        P = cube(N)
        S = get_solver("dwalton76", P, Budget(wall_time_s=6 * 3600))
        for ins in santa2023.instances(N):
            if ins.id in done:
                continue
            t0 = time.time()
            err, moves = None, []
            try:
                r = S.solve(ins.state)
                moves, info = r.moves, r.info
            except Exception as e:  # recorded as unsolved
                err, info = repr(e)[-300:], {}
            ok = bool(moves) and bool(np.array_equal(P.apply(ins.state, moves), P.solved))
            m = metrics(moves, P) if ok else {"htm": None, "qtm": None, "stm": None}
            row = {"id": ins.id, "solver": "dwalton76", "puzzle": "cube", "N": N, "depth": "santa", "solved": ok, **m,
                   "wall_time_s": round(time.time() - t0, 1), "error": err, "info": info}
            with OUT.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(ins.id, row["qtm"], row["wall_time_s"], err or "", flush=True)


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]])
