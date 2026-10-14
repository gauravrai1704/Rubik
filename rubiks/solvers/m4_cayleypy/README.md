# Method 4: CayleyPy reproduction

**Owner:** Shaurya · **Spec:** technical basis, Section 7 · **Registered name:** `m4_cayleypy`

Reproduces the current ML state of the art (Chervov et al., NeurIPS 2025) inside our engine: the per-N baseline that 5A must match while using one model for all N.

## Scope
* Cube N = 3, 4, 5; Square N = 3..8 (one model per N).

## How
* Training data: `rubiks.engine.random_walks(perms, solved, batch, kmax, ...)` already generates non-backtracking walks on GPU; record (state, k). Max walk length 26 / 45 / 65 for N = 3 / 4 / 5.
* Network: ResMLP — one-hot 6·6N² input, two Linear+BatchNorm+ReLU layers, residual blocks, one output predicting k. 10 layers, ~4M parameters.
* Training: MSE, Adam lr 1e-3, 1M fresh examples per epoch, FP32 training, FP16 inference.
* Search: use the shared `rubiks.search.beam.BeamSearch(puzzle, heuristic, BeamConfig(width=W, max_steps=200))`; your heuristic is a callable `(B, S) uint8 tensor -> (B,) float`. Report length vs W (2¹⁰ … the widest your GPU allows; use `tier='host'` for wider beams).
* Reference code: github.com/k1242/cayleypy-cube — wrap it in our `Solver` interface so the harness re-checks every solution.
* Optional: ensembles of independently trained agents, shortest solution wins.

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
1. Branch: `git checkout -b m4-cayleypy`.
2. Implement `solver.py` (add modules in this folder as needed).
3. Remove the `xfail` mark for `m4_cayleypy` in `tests/test_methods_1_4.py` and make it pass.
4. Benchmark: `python -m rubiks.bench make --puzzle square --N 3-8 --depth 20`, then
   `python -m rubiks.bench run --solver m4_cayleypy --puzzle square --N 3-8 --dataset d20`.
5. Open a PR; CI must pass and one reviewer approves.
