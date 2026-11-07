"""Method 7 on the paper's random Rubik's Squares (resumable).

    python experiments/m7_square.py 3-20

Symmetry portfolio over the 8 rotated / mirrored copies of each state; every copy is solved by each inner solver
and the shortest answer is kept. Inner solvers: Method 4 (per-size network, W = 4096) and 5C for N <= 8 (Method 4
has models only up to N = 8), 5C and Method 3 above. Instances: the first 10 random states per size for N <= 12 and
the first 5 for N >= 13, i.e. the same instances as the paper's Square table. Rows are harness-style and appended to
docs/results/m7_square.jsonl.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from rubiks.bench import datasets
from rubiks.engine import metrics, square
from rubiks.solvers import Budget, get_solver, load_all

OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "m7_square.jsonl"


def sizes(arg: str) -> list[int]:
    a, _, b = arg.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main(Ns: list[int]) -> None:
    load_all()
    done = {json.loads(l)["id"] for l in OUT.read_text().splitlines() if l.strip()} if OUT.exists() else set()
    for N in Ns:
        P = square(N)
        inners = ([("m4_cayleypy", {"width": 4096}), ("m5c_schedule", {})] if N <= 8
                  else [("m5c_schedule", {}), ("m3_demaine", {})])
        S = get_solver("m7_portfolio", P, Budget(wall_time_s=1800), copies=8, workers=2, inners=inners)
        for ins in datasets.load(datasets.path_for("square", N, "random"))[: 10 if N <= 12 else 5]:
            if ins.id in done:
                continue
            t0 = time.time()
            r = S.solve(ins.state)
            ok = bool(P.is_solved(P.apply(ins.state, r.moves)))
            m = metrics(r.moves, P) if ok else {"htm": None, "qtm": None, "stm": None}
            row = {"id": ins.id, "solver": "m7_portfolio", "puzzle": "square", "N": N, "depth": "random", "solved": ok,
                   **m, "wall_time_s": round(time.time() - t0, 3), "optimal_len": ins.optimal_len,
                   "info": {"inners": [n for n, _ in inners], "copies": r.info.get("copies"),
                            "best_inner": (r.info.get("copy_info") or [{}])[r.info.get("best_copy", 0)].get("inner")}}
            with OUT.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(ins.id, row["htm"], row["wall_time_s"], flush=True)


if __name__ == "__main__":
    main(sizes(sys.argv[1]) if len(sys.argv) > 1 else list(range(3, 21)))
