"""Exact optimal lengths for the 3x3 and 4x4 Rubik's Square by full BFS (paper, Section 6.2).

Goal = every face one colour, so BFS starts from all whole-puzzle rotations of the solved
state at once (multi-source). Writes results/paper/optimal_square.json:
{"N": {"states": ..., "gods_number": ..., "histogram": {...}, "instances": {id: optimal_len}}}
"""
import json
from pathlib import Path

import numpy as np

from rubiks.bench import datasets
from rubiks.engine import square

out = {}
for N in (3, 4):
    P = square(N)
    starts = {P.solved[r].tobytes(): P.solved[r] for r in P.rotations}
    seen = {k: 0 for k in starts}
    frontier, d = list(starts.values()), 0
    while frontier:
        d += 1
        nxt = []
        for s in frontier:
            for p in P.move_perms:
                k = s[p]
                b = k.tobytes()
                if b not in seen:
                    seen[b] = d
                    nxt.append(k)
        frontier = nxt
    hist = {}
    for v in seen.values():
        hist[v] = hist.get(v, 0) + 1
    inst = {i.id: seen[np.asarray(i.state, dtype=P.solved.dtype).tobytes()]
            for i in datasets.load(datasets.path_for("square", N, "random"))}
    out[N] = {"states": len(seen), "rotations": len(starts), "gods_number": max(seen.values()),
              "histogram": dict(sorted(hist.items())), "mean_random": float(np.mean(list(inst.values()))),
              "instances": inst}
    print(N, out[N]["states"], "states; God's number", out[N]["gods_number"], "; histogram", out[N]["histogram"])
Path("results/paper").mkdir(parents=True, exist_ok=True)
Path("results/paper/optimal_square.json").write_text(json.dumps(out, indent=1))
