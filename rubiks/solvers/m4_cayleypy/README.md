# Method 4: CayleyPy reproduction

**Owner:** Shaurya · **Spec:** technical basis, Section 7 · **Registered name:** `m4_cayleypy`

Reproduces the current ML state of the art (Chervov et al., NeurIPS 2025) inside our engine: the per-N baseline that 5A must match while using one model for all N.

## Scope
* Cube N = 3, 4, 5; Square N = 3..8 (one model per N).

## How (implemented)
| File | What |
|---|---|
| `model.py` | `ResMLP` (Fig. 1b: Linear N1 + BN + ReLU, Linear N2 + BN + ReLU, Nr residual blocks, one output), the paper's Table II sizes as `PRESETS` (`paper3` / `paper4` / `paper5` = the 4M 10-layer models), the move sets, non-backtracking walk generator, save / load |
| `train.py` | One model per (puzzle, N): fresh walks every epoch, target k, MSE, Adam 1e-3, FP32. Writes `<out>.json` with training time, peak GPU memory, examples seen and a fit log |
| `solver.py` | `m4_cayleypy`: shared `BeamSearch` (hash dedup, 200 steps, any memory tier), FP16 scoring on GPU, optional W-doubling retries, ensembles (shortest wins) |

* **Move set (`--moves paper`).** 3×3×3: the 12 face quarter turns (DeepCubeA's QTM). 4×4×4 / 5×5×5:
  quarter turns of every slice (Santa 2023's UQTM). Square: every flip. `R R` is legal in the search
  (it is how QTM spells `R2`); only `R R'` and non-ascending orders of commuting moves are pruned.
  `--moves all` trains / searches with every engine move instead.
* **Walks.** Non-backtracking in the paper's sense: never the inverse of the previous move.
  `kmax` = 26 / 45 / 65 for cubes 3 / 4 / 5; Squares use 5A's horizon (`m5a_tokens.train.kmax_for`).
* **Checkpoints.** `checkpoints/m4/<puzzle><N>.pt` (committed, CPU-sized) or `<puzzle><N>.large.pt`
  (GPU, git-ignored, preferred when present). Extra ensemble agents: `<puzzle><N>.<tag>[.large].pt`.

```bash
# paper settings (GPU): 8192 epochs x 1M = 8B examples
python -m rubiks.solvers.m4_cayleypy.train --puzzle cube --N 3 --preset paper3 --epochs 8192 \
    --device cuda --out checkpoints/m4/cube3.large.pt
# everything in Section 7.3 + the length-vs-W curve
DEVICE=cuda bash experiments/m4_cayleypy.sh
# solve at a given width; an ensemble is a list of checkpoints
python -m rubiks.bench run --solver m4_cayleypy --puzzle cube --N 3 --dataset random --device cuda \
    --kw '{"width": 262144}'
```

## Engine pieces you can use
| Need | Use |
|---|---|
| Build a puzzle | `rubiks.engine.cube(N)`, `rubiks.engine.square(N)` |
| Moves | `puzzle.moves` (list of `Move(axis, layer, turn)`), `puzzle.move_perms`, `puzzle.apply(state, moves)` |
| Solved check | `puzzle.is_solved(state)` (every face one colour) |
| Clusters (Demaine's) | `rubiks.engine.clusters(puzzle)` → `Orbit` with `.kind`, `.stickers`, `.gen_moves`, `.gen_perms` |
| Solvability | `rubiks.engine.is_reachable(puzzle, state)` |
| Notation | `rubiks.engine.to_notation(moves, puzzle)`, `parse_notation(text, puzzle)` |
| Canonical form / metrics | `rubiks.engine.canonicalize(moves)`, `metrics(moves, puzzle)` → HTM / QTM / STM |
| Search pruning | `rubiks.engine.canonical_successor_mask(puzzle)` |
| Beam search (learned methods) | `rubiks.search.beam.BeamSearch`, `BeamConfig` |

## Done when
- [ ] 3×3×3 mean within ~0.5 moves of the paper's 21.14 QTM at W = 2¹⁸.
- [ ] Length-vs-W curves for 3×3×3 and 4×4×4.
- [ ] Per-N Square models for N = 3..8 on our benchmark.
- [ ] Training time and peak GPU memory recorded for every run.

## Workflow
1. Benchmark: `python -m rubiks.bench make --puzzle square --N 3-8 --depth 20`, then
   `python -m rubiks.bench run --solver m4_cayleypy --puzzle square --N 3-8 --dataset d20`.
2. Open a PR; CI must pass and one reviewer approves.
