"""Ablations for Method 5B (paper, Section 6.4). Writes harness rows to results/paper/ablations/.

    python paper/scripts/run_ablations.py square   # Square N = 8, 12
    python paper/scripts/run_ablations.py cube     # Cube N = 4, 5
"""
import sys
from pathlib import Path

from rubiks.bench import datasets
from rubiks.bench.harness import run
from rubiks.solvers import Budget
from rubiks.solvers.m5b_stages.cluster_models import ModelBank

OUT = Path("results/paper/ablations")
which = sys.argv[1]
if which == "square":
    sizes, n, budget = [8, 12], 5, Budget(wall_time_s=600)
    base = {"width": 128, "max_width": 2048}
    variants = {
        "full": {},
        "no_macros": {"macros": False},
        "no_exact_stages": {"exact_cap": 0},
    }
else:
    sizes, n, budget = [4, 5], 2, Budget(wall_time_s=300)
    base = {"width": 512, "max_width": 8192}
    variants = {
        "full": {},
        "no_center_macro_restriction": {"center_macros_only": False},
        "corner_mlp_plus_tables": {"bank": ModelBank(tables_only=())},
        "no_coarse_tables": {"bank": ModelBank(coarse=False)},
    }
for name, kw in variants.items():
    ins = []
    for N in sizes:
        ins += datasets.load(datasets.path_for(which, N, "random"))[:n]
    run("m5b_stages", ins, budget, OUT / f"{which}_{name}.jsonl", {**base, **kw})
