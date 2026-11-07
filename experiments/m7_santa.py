"""Method 7 on every Santa 2023 standard-goal cube puzzle of the given sizes (resumable).

    KMCODERS_DIR=... NXOPT=... NXOPT_DIR=... python experiments/m7_santa.py 5 6 7 4

One fixed configuration per size (the paper's):
  * N = 4: best of 48 symmetric copies of KMCoders' beam variant with 1M stage-2 iterations, and of 8 copies of
    their default program (greedy, 20M iterations);
  * N = 5..7: best of 24 copies of the beam variant with 1M stage-2 iterations; N = 8..10: 12 copies, N = 19: 6 (cost);
    for odd N the central 3x3x3 is solved optimally in QTM by nxopt.
Rows are appended to docs/results/santa/m7_full_N<N>.jsonl as soon as a puzzle finishes; puzzles already in that
file, or in the earlier subset files of the same configuration, are skipped.
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "santa"
# earlier runs of exactly these configurations (first puzzles of N = 4, 5)
EARLIER = {4: [("m7_beam_sa1M_48copies_N4.jsonl", "beam48"), ("m7_default_8copies_N4.jsonl", "default8")],
           5: [("m7_beam_sa1M_24copies_nxopt_N5.jsonl", "beam24")]}


def configs(N: int):
    odd = N % 2 == 1
    beam = {"binary": "solve_cube_beam", "sa_iters": 1_000_000, "stage1": "nxopt" if odd else "kociemba"}
    if N == 4:
        return [("beam48", 48, beam), ("default8", 8, {"binary": "solve_cube"})]
    if N >= 11:  # one copy takes hours at N = 19
        return [("beam6", 6, beam)]
    if N >= 8:  # one copy takes 14-35 min at N = 8-10 on 4 cores
        return [("beam12", 12, beam)]
    return [("beam24", 24, beam)]


def done_ids(N: int) -> set[str]:
    ids = set()
    p = OUT / f"m7_full_N{N}.jsonl"
    if p.exists():
        ids |= {json.loads(l)["id"] for l in p.read_text().splitlines() if l.strip()}
    files = EARLIER.get(N, [])
    if files:
        sets = [{json.loads(l)["id"] for l in (OUT / f).read_text().splitlines() if l.strip()} for f, _ in files]
        ids |= set.intersection(*sets)
    return ids


def main(sizes: list[int]) -> None:
    load_all()
    for N in sizes:
        P = cube(N)
        out = OUT / f"m7_full_N{N}.jsonl"
        skip = done_ids(N)
        solvers = [(tag, get_solver("m7_portfolio", P, Budget(wall_time_s=4 * 3600), copies=k, workers=3,
                                    inner_kwargs=kw)) for tag, k, kw in configs(N)]
        for ins in santa2023.instances(N):
            if ins.id in skip:
                continue
            t0 = time.time()
            parts, best = {}, None
            for tag, S in solvers:
                r = S.solve(ins.state)
                ok = bool(np.array_equal(P.apply(ins.state, r.moves), P.solved))
                q = metrics(r.moves, P)["qtm"] if ok else None
                parts[tag] = {"qtm": q, "copies": r.info.get("copies"),
                              "cpu_s": [i.get("cpu_s") for i in r.info.get("copy_info", [])],
                              "stage1": [i.get("stage1") for i in r.info.get("copy_info", [])]}
                if ok and (best is None or q < best[0]):
                    best = (q, r.moves)
            row = {"id": ins.id, "N": N, "solver": "m7_portfolio", "solved": best is not None,
                   "qtm": best[0] if best else None, "htm": metrics(best[1], P)["htm"] if best else None,
                   "parts": parts, "wall_time_s": round(time.time() - t0, 1),
                   "cpu_s": round(sum(x or 0 for p in parts.values() for x in p["cpu_s"]), 1)}
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(ins.id, row["qtm"], row["wall_time_s"], flush=True)


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [5, 6, 7, 4])
