# NxN Rubik's — nine solvers, one engine

Solvers for the **N×N×1 Rubik's Square** and the **N×N×N Rubik's Cube** for every
N from 3 to 20, benchmarked on one shared engine. Methods 1–4 are established techniques
(the course requirement); Methods 5A and 5B are new learned solvers, and 5C a new schedule for
Demaine's parallel algorithm that is the best of our original methods at large N, aimed at a paper in the
style of CayleyPy (NeurIPS 2025). Methods 6 and 7 take on the strongest public baseline, KMCoders' winning
Kaggle Santa 2023 solution (see *External baselines* below). The full spec is the technical basis document
([docs/technical_basis.pdf](docs/technical_basis.pdf)); section numbers below refer to it.

| # | Method | Owner | Package | Status |
|---|---|---|---|---|
| 1 | IDA* + pattern databases | Shree | `rubiks/solvers/m1_idastar` | implemented; optimal on Squares N ≤ 5 and 3×3×3 scrambles (Korf PDBs, parallel IDA*); Square God's numbers 6 / 10 / 12 for N = 3 / 4 / 5 |
| 2 | Reduction + Kociemba two-phase | Sunaina | `rubiks/solvers/m2_reduction` | implemented; 100% solved on Square and Cube N = 3–20 |
| 3 | Demaine parallel cluster algorithm | Kashvi | `rubiks/solvers/m3_demaine` | implemented; 100% solved on Square and Cube N = 3–20 |
| 4 | CayleyPy reproduction | Shaurya | `rubiks/solvers/m4_cayleypy` | implemented; per-N Square models N = 3–8 (shortest of the learned methods; the Colab T4 "large" model is the default at N = 8, 7% shorter); 3×3×3 needs GPU training |
| 5A | Cluster-token transformer, one model for all N | Sharanya | `rubiks/solvers/m5a_tokens` | implemented; small CPU Square model trained, beats 5B at N ≤ 10 |
| 5B | Learned search inside reduction stages | Gaurav | `rubiks/solvers/m5b_stages` | implemented; 100% solved on Square N = 3–20 and Cube N = 3–20 (solutions still long) |
| 5C | **New:** net-gain scheduling of bulk commutators | Gaurav | `rubiks/solvers/m5c_schedule` | implemented; best method on the Cube for N ≥ 5 (20×20×20: 4,925 HTM, −26% vs Method 3) and on the Square for N ≥ 9 |
| 6 | KMCoders-style insertion method, our implementation | Gaurav | `rubiks/solvers/m6_kmcoders` | implemented; best of our from-scratch methods on Santa 2023 cubes for N ≥ 6 (1.5–1.7× KMCoders' published lengths) |
| 7 | **New:** symmetry portfolio around KMCoders' own solver | Gaurav | `rubiks/solvers/m7_portfolio` | in progress; best of 8 rotated/mirrored copies is 14% shorter than one run of their program on Santa 4×4×4 (67.9 vs 77.1 QTM), still above their published best-of-many (63.5) |
| — | Engine, checker, beam search, harness | Gaurav | `rubiks/engine`, `rubiks/search`, `rubiks/bench` | implemented, tested |

Each method's folder has a `README.md` with its spec, what was built, and a done-when checklist. Measured results so far: [docs/STATUS.md](docs/STATUS.md).

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

## External baselines (Kaggle Santa 2023, dwalton76)

| Solver name | What it is | Setup |
|---|---|---|
| `santa_kmcoders` | KMCoders' published final Santa 2023 submission, replayed and re-verified | none (data downloaded on first use into `data/santa2023/`, not committed: Kaggle terms) |
| `kmcoders` | KMCoders' own Rust solver ([wata-orz/santa2023_permutation_puzzle](https://github.com/wata-orz/santa2023_permutation_puzzle)) run once on any cube state | clone it, apply `rubiks/solvers/external/kmcoders_env.patch` (seed and stage-2 length from env vars; defaults unchanged), `cargo build --release` in `solution/`, set `KMCODERS_DIR` |
| `dwalton76` | [dwalton76's NxNxN solver](https://github.com/dwalton76/rubiks-cube-NxNxN-solver), wide turns expanded to slice turns | build it, set `DWALTON76_DIR` (lookup tables ~8 GB for N ≤ 7) |

```bash
python -m rubiks.bench make --puzzle cube --N 4-10 --depth santa      # the 187 standard-goal Santa cubes
python -m rubiks.bench run  --solver kmcoders --puzzle cube --N 4 --dataset santa
```

Santa 2023 cubes, mean QTM (Santa's move count; lower is better):

| N | KMCoders published | KMCoders' program, one run | dwalton76 | Method 6 | 5C | Method 3 | Method 2 |
|---|---|---|---|---|---|---|---|
| 4 | **66.7** | 77.0 | 88.6 | 109.4 | 217.5 | 229.3 | 266.9 |
| 5 | **97.1** | running | 160.2 | 159.3 | 282.5 | 296.3 | 361.1 |
| 6 | **161** | running | 324 | 276 | 503 | 534.7 | 696.3 |
| 7 | **208.8** | running | 539 | 356.2 | 593.8 | 680.8 | 880.2 |
| 10 | **461.5** | running | — | 743 | 1314.5 | 1597 | 2107.5 |
| 19 | **1620** | — | — | 2484.7 | 4710.7 | 6300.7 | 8188.7 |

The published column is the best of many competition runs; no method here beats it yet. Method 7 is the
attempt (`docs/STATUS.md` has the running numbers).

## Training on a GPU (Google Colab)

[`notebooks/train_gpu_colab.ipynb`](notebooks/train_gpu_colab.ipynb) trains every network that needs a GPU
(Method 4's paper-size cube models and per-N Square models, Method 5A for Squares and cubes, Method 5B's
cluster MLPs) on a free Colab T4, with evaluation cells. Everything is saved to Google Drive and every
training resumes after a disconnect (`--save-every`, `--stop-after` on the Method 4 and 5A trainers).
[Open in Colab](https://colab.research.google.com/github/gauravrai1704/Rubik/blob/full_proj/notebooks/train_gpu_colab.ipynb)
(for a private repository, upload the notebook instead: *File → Upload notebook*).

## Layout

```
rubiks/
  engine/      puzzle geometry and moves, canonical form + metrics, orbits & clusters,
               projections (exact BFS tables), solvability checker, scrambles and walks
  search/      shared beam search (3-bit packing, GPU / host / disk tiers, macro actions,
               Q*-style child scoring), macro-action sets
  bench/       datasets (JSONL), harness (re-verifies every solution), CLI
  solvers/     base interface + registry, one package per method
tests/         engine, checker, beam, every method, harness, Santa 2023 adapter, external baselines
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
