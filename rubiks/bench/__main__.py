"""Command line: ``python -m rubiks.bench <make|run|summary> ...``

Examples::

    # 100 scrambles of depth 30 for the 5x5x5, saved under data/cube/N5/d30.jsonl
    python -m rubiks.bench make --puzzle cube --N 5 --depth 30

    # make datasets for every N in a range, random states
    python -m rubiks.bench make --puzzle square --N 3-20 --depth random

    # run a solver over N = 3..8 and write results/<solver>/<puzzle>_d30.jsonl
    python -m rubiks.bench run --solver m5b_stages --puzzle square --N 3-8 --dataset d30

    # summary table of one or more results files
    python -m rubiks.bench summary results/m5b_stages/square_d30.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..solvers import Budget
from . import datasets
from .harness import RESULTS_ROOT, print_summary, run, summarize


def parse_range(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m rubiks.bench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    mk = sub.add_parser("make", help="generate benchmark datasets")
    mk.add_argument("--puzzle", choices=["cube", "square"], required=True)
    mk.add_argument("--N", required=True, help="e.g. 5 or 3-20 or 3,5,10")
    mk.add_argument("--depth", default="random",
                    help="scramble length, 'random', or 'santa' (Kaggle Santa 2023 cube instances; downloaded)")
    mk.add_argument("--count", type=int, default=None, help="default: 100 for N<=7, else 20")
    mk.add_argument("--seed", type=int, default=0)

    rn = sub.add_parser("run", help="run a solver on datasets")
    rn.add_argument("--solver", required=True)
    rn.add_argument("--puzzle", choices=["cube", "square"], required=True)
    rn.add_argument("--N", required=True)
    rn.add_argument("--dataset", default="random", help="dataset name, e.g. d30 or random")
    rn.add_argument("--limit", type=int, default=None, help="only the first k instances per N")
    rn.add_argument("--time", type=float, default=300.0, help="wall-time budget per instance (s)")
    rn.add_argument("--device", default="cpu")
    rn.add_argument("--kw", default="{}", help='solver kwargs as JSON, e.g. \'{"width": 4096}\'')
    rn.add_argument("--out", default=None)

    sm = sub.add_parser("summary", help="summarise results files")
    sm.add_argument("files", nargs="+")

    a = ap.parse_args(argv)
    if a.cmd == "make" and a.depth == "santa":
        from . import santa2023

        for N in parse_range(a.N):
            ins = santa2023.instances(N)
            p = datasets.save(ins, datasets.path_for(a.puzzle, N, "santa"))
            print(f"wrote {len(ins)} Santa 2023 instances to {p}")
    elif a.cmd == "make":
        depth = a.depth if a.depth == "random" else int(a.depth)
        name = "random" if depth == "random" else f"d{depth}"
        for N in parse_range(a.N):
            ins = datasets.generate(a.puzzle, N, a.count or datasets.default_count(N), depth, a.seed)
            p = datasets.save(ins, datasets.path_for(a.puzzle, N, name))
            print(f"wrote {len(ins)} instances to {p}")
    elif a.cmd == "run":
        instances = []
        for N in parse_range(a.N):
            p = datasets.path_for(a.puzzle, N, a.dataset)
            if not p.exists():
                raise SystemExit(f"{p} not found: run `python -m rubiks.bench make --puzzle {a.puzzle} --N {N} "
                                 f"--depth {a.dataset.lstrip('d')}` first")
            instances += datasets.load(p)[: a.limit]
        out = Path(a.out) if a.out else RESULTS_ROOT / a.solver / f"{a.puzzle}_{a.dataset}.jsonl"
        rows = run(a.solver, instances, Budget(wall_time_s=a.time, device=a.device), out, json.loads(a.kw))
        print_summary(summarize(rows))
        print(f"results: {out}")
    else:
        rows = []
        for f in a.files:
            rows += list(datasets.iter_jsonl(Path(f)))
        print_summary(summarize(rows))


if __name__ == "__main__":
    main()
