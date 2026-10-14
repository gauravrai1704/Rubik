# Method 1: IDA* with pattern databases

**Owner:** Shree · **Spec:** technical basis, Section 4 · **Registered name:** `m1_idastar`

Finds provably shortest solutions where feasible: the ground truth every other method is measured against, plus the measured curve of where exact search stops working.

## Scope
* 3×3×3: optimal solutions in HTM and QTM for the DeepCubeA test set (or a subset).
* Rubik's Square: full BFS of the whole state space for the smallest N (exact God's number), then IDA* for as large an N as the budget allows.
* Report the largest N, per puzzle, at which IDA* still solves random states within the budget.

## How
* IDA*: depth-first searches with bound f = g + h, raising the bound each round; with admissible h the first solution is optimal.
* 3×3×3 heuristic: max of Korf's three pattern databases — corners 8!·3⁷ = 88,179,840 entries; two 6-edge tables of 12!/6!·2⁶ = 42,577,920 entries each; 4 bits per entry; built by BFS backwards from solved.
* Square heuristic: max over clusters of each cluster's exact distance. `rubiks.engine.projection.bfs_projection` already builds these tables (see `rubiks/solvers/m5b_stages/cluster_models.py::table_for`).
* Pruning: `rubiks.engine.canonical_successor_mask(puzzle)` gives the allowed next moves (no same face twice, one order for commuting moves).
* Write the inner loop in C++ or Numba, on a compact piece-level state (corner perm/twist, edge perm/flip), not the sticker array.
* Cross-check a sample of optimal lengths against Rokicki's cube20src.

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
- [ ] Pattern databases built, entry counts matching the table above.
- [ ] Optimal solutions for ≥ 100 3×3×3 instances, mean length in HTM and QTM.
- [ ] Exact God's number for the smallest Squares from full BFS.
- [ ] Plot of IDA* success rate against N for the Square.
- [ ] Write optimal lengths back into the datasets' `optimal_len` field (`rubiks.bench.datasets`).

## Workflow
1. Branch: `git checkout -b m1-idastar`.
2. Implement `solver.py` (add modules in this folder as needed).
3. Remove the `xfail` mark for `m1_idastar` in `tests/test_methods_1_4.py` and make it pass.
4. Benchmark: `python -m rubiks.bench make --puzzle square --N 3-8 --depth 20`, then
   `python -m rubiks.bench run --solver m1_idastar --puzzle square --N 3-8 --dataset d20`.
5. Open a PR; CI must pass and one reviewer approves.
