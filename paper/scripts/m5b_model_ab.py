"""Method 5B on the same cube instances (N = 3-6, 2 each) with only the cluster networks swapped:
same CPU, same code, same budget (1800 s).

    python paper/scripts/m5b_model_ab.py old paper/results/ablations/cube5b_cpu_models.jsonl
    python paper/scripts/m5b_model_ab.py checkpoints/m5b_gpu paper/results/ablations/cube5b_gpu_models.jsonl
"""
import json, sys, time
from pathlib import Path

from rubiks.bench import datasets
from rubiks.engine import cube, metrics
from rubiks.solvers import Budget
from rubiks.solvers.m5b_stages.cluster_models import CKPT_DIR, ModelBank
from rubiks.solvers.m5b_stages.solver import StagedSolver

root = CKPT_DIR if sys.argv[1] == "old" else Path(sys.argv[1])
out = open(sys.argv[2], "w")
for N in (3, 4, 5, 6):
    P = cube(N)
    S = StagedSolver(P, Budget(wall_time_s=1800), width=512, max_width=8192,
                     bank=ModelBank(root=root, tables_only=("corner",)))
    for ins in datasets.load(datasets.path_for("cube", N, "random"))[:2]:
        t = time.time()
        r = S.solve(ins.state)
        ok = P.is_solved(P.apply(ins.state, r.moves))
        row = {"id": ins.id, "N": N, "solved": ok, "htm": metrics(r.moves, P)["htm"] if ok else None,
               "time_s": round(time.time() - t, 1), "failed_stage": r.info.get("failed_stage")}
        out.write(json.dumps(row) + "\n"); out.flush()
        print(row, flush=True)
