"""Method 5A (GPU-trained "large" Square model, run here on CPU) on the paper's Square instances that the Colab
evaluation did not cover (it ran the first 5 states of N = 4-10, 12, 14). Appends harness-style rows to
docs/results/m5a_square_gpu_large.jsonl, skipping ids already there."""
import json
import time
from pathlib import Path

from rubiks.bench import datasets
from rubiks.engine import metrics, square
from rubiks.solvers import Budget, get_solver, load_all

OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "m5a_square_gpu_large.jsonl"
load_all()
done = {json.loads(l)["id"] for l in OUT.read_text().splitlines() if l.strip()}
for N in range(4, 13):
    P = square(N)
    S = get_solver("m5a_tokens", P, Budget(wall_time_s=900))
    for ins in datasets.load(datasets.path_for("square", N, "random"))[:10]:
        if ins.id in done:
            continue
        t0 = time.time()
        try:
            r = S.solve(ins.state)
            ok = bool(P.is_solved(P.apply(ins.state, r.moves)))
            m, info, err = (metrics(r.moves, P) if ok else {"htm": None, "qtm": None, "stm": None}), r.info, None
        except Exception as e:  # a failure counts as unsolved
            ok, m, info, err = False, {"htm": None, "qtm": None, "stm": None}, {}, repr(e)
        row = {"id": ins.id, "solver": "m5a_tokens", "puzzle": "square", "N": N, "depth": "random", "solved": ok, **m,
               "wall_time_s": round(time.time() - t0, 3), "optimal_len": ins.optimal_len, "error": err,
               "info": {k: v for k, v in (info or {}).items() if not isinstance(v, (list, dict))},
               "hardware": "CPU (this machine)", "checkpoint": "checkpoints/m5a/square.large.pt"}
        with OUT.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(ins.id, row["htm"], row["wall_time_s"], flush=True)
