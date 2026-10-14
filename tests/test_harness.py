"""Datasets and harness: generation, saving, running, verification, summaries."""
import json

import numpy as np

from rubiks.bench import datasets
from rubiks.bench.harness import run, summarize
from rubiks.solvers import Budget


def test_generate_save_load(tmp_path):
    ins = datasets.generate("square", 4, 5, 10, seed=1)
    p = datasets.save(ins, tmp_path / "x.jsonl")
    back = datasets.load(p)
    assert [b.id for b in back] == [i.id for i in ins]
    assert np.array_equal(back[0].state, ins[0].state)
    assert ins[0].scramble is not None


def test_run_m5b_and_summarize(tmp_path):
    ins = datasets.generate("square", 3, 3, 8) + datasets.generate("square", 5, 3, 8)
    rows = run("m5b_stages", ins, Budget(wall_time_s=60), tmp_path / "r.jsonl", {"width": 128}, verbose=False)
    assert all(r["solved"] for r in rows)
    lines = (tmp_path / "r.jsonl").read_text().splitlines()
    assert len(lines) == 6 and "commit" in json.loads(lines[0])
    s = summarize(rows)
    assert [x["N"] for x in s] == [3, 5] and all(x["success"] == 1.0 for x in s)
