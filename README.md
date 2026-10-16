# NxN Rubik's — six solvers, one engine

Solvers for the **N×N×1 Rubik's Square** and the **N×N×N Rubik's Cube** for every
N from 3 to 20, benchmarked on one shared engine. Methods 1–4 are established techniques
(the course requirement); Methods 5A and 5B are new learned solvers aimed at a paper in the
style of CayleyPy (NeurIPS 2025). The full spec is the technical basis document
([docs/technical_basis.pdf](docs/technical_basis.pdf)); section numbers below refer to it.

| # | Method | Owner | Package | Status |
|---|---|---|---|---|
| 1 | IDA* + pattern databases | Shree | `rubiks/solvers/m1_idastar` | scaffold + spec |
| 2 | Reduction + Kociemba two-phase | Sunaina | `rubiks/solvers/m2_reduction` | scaffold + spec |
| 3 | Demaine parallel cluster algorithm | Kashvi | `rubiks/solvers/m3_demaine` | scaffold + spec |
| 4 | CayleyPy reproduction | Shaurya | `rubiks/solvers/m4_cayleypy` | scaffold + spec |
| 5A | Cluster-token transformer, one model for all N | Sharanya | `rubiks/solvers/m5a_tokens` | implemented; small CPU Square model trained, beats 5B at N ≤ 10 |
| 5B | Learned search inside reduction stages | Gaurav | `rubiks/solvers/m5b_stages` | implemented; 100% solved on Square N = 3–20 and Cube N = 3–20 (solutions still long) |
| — | Engine, checker, beam search, harness | Gaurav | `rubiks/engine`, `rubiks/search`, `rubiks/bench` | implemented, tested |

Each teammate's folder has a `README.md` with their spec, the engine functions to use,
and a done-when checklist. Measured results so far: [docs/STATUS.md](docs/STATUS.md).

**Publish to GitHub** (once, from your machine, with `gh auth login` done):
`bash scripts/publish_to_github.sh <your-user>/nxn-rubiks` — creates the repo, pushes `main`,
turns on branch protection. Then invite teammates in Settings → Collaborators.

## Quick start

```bash
git clone <repo-url> && cd nxn-rubiks
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build
pip install -e ".[dev]"
pytest -q -m "not slow"          # ~1 minute; must pass before every PR
```

Solve something:

```python
import numpy as np
from rubiks.engine import square, random_state, to_notation
from rubiks.solvers import get_solver, Budget

P = square(10)
state = random_state(P, np.random.default_rng(0))
r = get_solver("m5b_stages", P, Budget(wall_time_s=120)).solve(state)
print(len(r.moves), P.is_solved(P.apply(state, r.moves)), to_notation(r.moves[:10], P))
```

Benchmark (Section 3.6, 10):

```bash
python -m rubiks.bench make --puzzle square --N 3-20 --depth random     # data/square/N*/random.jsonl
python -m rubiks.bench run  --solver m5b_stages --puzzle square --N 3-12 --dataset random
python -m rubiks.bench summary results/m5b_stages/square_random.jsonl
```

## Layout

```
rubiks/
  engine/      puzzle geometry and moves, canonical form + metrics, orbits & clusters,
               projections (exact BFS tables), solvability checker, scrambles and walks
  search/      shared beam search (3-bit packing, GPU / host / disk tiers, macro actions,
               Q*-style child scoring), macro-action sets
  bench/       datasets (JSONL), harness (re-verifies every solution), CLI
  solvers/     base interface + registry, one package per method
tests/         engine, checker, beam, 5A, 5B, harness, methods 1–4 (xfail until done)
experiments/   one script per experiment E1–E8 (Section 10.3)
checkpoints/   trained weights (small ones committed; see checkpoints/README.md)
scripts/       publish_to_github.sh
docs/          technical basis, architecture notes
```

The technical basis put `engine/`, `bench/`, `search/` at the top level; they live under one
`rubiks/` package instead so imports cannot clash with other installed packages.

## The engine in five lines

```python
from rubiks.engine import cube, Move, clusters, is_reachable, metrics
P = cube(5)                                   # also square(N); dims (N,N,N) or (N,N,1)
s = P.apply(P.solved, [Move(0, 4, 3)])        # Move(axis, layer, quarter-turns about +axis) = R
P.is_solved(s), is_reachable(P, s)            # (False, (True, 'ok'))
[c.kind for c in clusters(P)]                 # corner, midge, wing, center_fixed, center_plus, center_x
```

* States are `uint8` arrays, one colour (0–5, faces U R F D L B) per sticker; a move is
  `state[perm]`. Moves are generated from 3D geometry, so every N comes from the same code.
* `metrics(moves, P)` gives HTM, QTM and STM after canonicalisation (Section 3.2).
* `clusters(P)` are Demaine's cubie clusters with an N-invariant sticker order. Clusters of
  the same type have the same `signature` at every N — the basis of 5A and 5B.

## Rules (Section 11.3)

* One branch per method; merge to `main` only with green CI and one review.
* `rubiks/engine/` changes go through a PR that Gaurav approves: every result depends on it.
* Results files come only from the harness. Never hand-edit them.
* A failed instance counts as failed. Never drop failures from an average.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
