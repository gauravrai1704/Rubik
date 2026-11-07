"""Equal-compute control for Method 7: random seeds instead of symmetric copies (resumable).

    KMCODERS_DIR=... python experiments/m7_seed_control.py

On the same ten Santa 4x4x4 puzzles as Method 7's first run, KMCoders' program is run on the unrotated puzzle with
48 seeds of the beam variant (1M stage-2 iterations) and 8 seeds of the default program: exactly the number and kind
of runs of Method 7's N = 4 configuration (48 + 8 symmetric copies), so the two portfolios cost the same. Rows go to
docs/results/santa/m7_seed_control_N4.jsonl.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from rubiks.bench import santa2023
from rubiks.engine import cube, metrics
from rubiks.solvers import Budget, get_solver, load_all

OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "santa" / "m7_seed_control_N4.jsonl"
PARTS = [("beam48", 48, {"binary": "solve_cube_beam", "sa_iters": 1_000_000}), ("default8", 8, {"binary": "solve_cube"})]


def main() -> None:
    load_all()
    P = cube(4)
    done = {json.loads(l)["id"] for l in OUT.read_text().splitlines() if l.strip()} if OUT.exists() else set()
    for ins in santa2023.instances(4)[:10]:
        if ins.id in done:
            continue
        t0 = time.time()
        parts, best = {}, None
        for tag, k, kw in PARTS:
            def run(seed):
                S = get_solver("kmcoders", P, Budget(wall_time_s=3600), seed=seed, **kw)
                r = S.solve(ins.state)
                ok = bool(np.array_equal(P.apply(ins.state, r.moves), P.solved))
                return (metrics(r.moves, P)["qtm"] if ok else None), r.info.get("cpu_s")

            with ThreadPoolExecutor(3) as ex:
                res = list(ex.map(run, range(1, k + 1)))
            q = [x for x, _ in res]
            parts[tag] = {"copies": q, "cpu_s": [c for _, c in res]}
            good = [x for x in q if x is not None]
            if good and (best is None or min(good) < best):
                best = min(good)
        row = {"id": ins.id, "N": 4, "solver": "kmcoders_seed_portfolio", "solved": best is not None, "qtm": best,
               "parts": parts, "wall_time_s": round(time.time() - t0, 1)}
        with OUT.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(ins.id, best, row["wall_time_s"], flush=True)


if __name__ == "__main__":
    main()
