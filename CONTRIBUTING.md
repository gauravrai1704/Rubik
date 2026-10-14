# Contributing

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CUDA build on the GPU machine
pip install -e ".[dev]"
pytest -q -m "not slow"
```

## Your method
1. `git checkout -b m1-idastar` (your method's folder name, with dashes).
2. Read your folder's `README.md` and the matching section of `docs/technical_basis.pdf`.
3. Implement `solver.py`. Keep the class registered under its name (`@register("m1_idastar")`)
   and return engine `Move` objects. Never set `solved` yourself; the harness checks.
4. In `tests/test_methods_1_4.py`, remove the `xfail` for your method once it works, and add
   tests of your own in `tests/test_<method>.py`.
5. Benchmark with the harness only:
   ```bash
   python -m rubiks.bench make --puzzle square --N 3-8 --depth 20
   python -m rubiks.bench run --solver m1_idastar --puzzle square --N 3-8 --dataset d20
   ```
6. Open a PR. CI (lint + fast tests) must be green and one teammate approves.

## Ground rules
* **Engine is shared.** Don't change `rubiks/engine/` in your method's PR; open a separate PR
  for it, approved by Gaurav. If you need something, ask in the weekly sync.
* **Move IDs:** `Move(axis, layer, turn)`; axis 0/1/2 = x/y/z (R/U/F on the + side), `turn` =
  quarter turns about +axis (1, 2, 3). `to_notation` / `parse_notation` convert to `3R'` style.
* **States** are `uint8` arrays of colours 0–5 (faces U R F D L B), one per sticker.
* **Budgets:** respect `self.budget.wall_time_s`; a run that times out is a failure, not a hang.
* **Determinism:** take seeds as arguments; record them in `SolveResult.info`.
* **Results files** come only from `python -m rubiks.bench run`; commit hash and hardware are
  recorded automatically. Don't edit them by hand.
* **Big files** (datasets, checkpoints, results) stay out of git; share via the team drive or
  GitHub release assets.

## Style
`ruff check rubiks tests` (CI runs the error subset). Line length 120. Type hints on public
functions. Docstrings say what a function is for and cite the technical-basis section.
