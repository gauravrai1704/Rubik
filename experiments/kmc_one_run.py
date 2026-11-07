"""One run of KMCoders' program (baseline B2) on every Santa 2023 cube puzzle of the given sizes (resumable).

    KMCODERS_DIR=... NXOPT=... python experiments/kmc_one_run.py 10 19

Same configuration as the earlier B2 rows (their default program; for odd N Kociemba's central 3x3x3, the program's
own fallback). Harness-style rows are appended to docs/results/santa/kmcoders_program_N<N>.jsonl; puzzles already there
are skipped.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from rubiks.bench import santa2023
from rubiks.engine import cube, metrics
from rubiks.solvers import Budget, get_solver, load_all

OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "santa"


def main(sizes: list[int]) -> None:
    load_all()
    for N in sizes:
        P = cube(N)
        out = OUT / f"kmcoders_program_N{N}.jsonl"
        done = {json.loads(l)["id"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
        S = get_solver("kmcoders", P, Budget(wall_time_s=12 * 3600))
        for ins in santa2023.instances(N):
            if ins.id in done:
                continue
            t0 = time.time()
            r = S.solve(ins.state)
            ok = bool(np.array_equal(P.apply(ins.state, r.moves), P.solved))
            m = metrics(r.moves, P) if ok else {"htm": None, "qtm": None, "stm": None}
            row = {"id": ins.id, "solver": "kmcoders", "puzzle": "cube", "N": N, "depth": "santa", "solved": ok, **m,
                   "wall_time_s": round(time.time() - t0, 1), "info": r.info}
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(ins.id, row["qtm"], row["wall_time_s"], flush=True)


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]])
