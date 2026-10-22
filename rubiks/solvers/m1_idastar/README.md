# Method 1: IDA* with pattern databases

**Owner:** Shree (implemented by Gaurav) · **Spec:** technical basis, Section 4 · **Registered name:** `m1_idastar`

Finds provably shortest solutions where feasible: the ground truth every other method is measured against, plus the measured curve of where exact search stops working.

## Scope
* 3×3×3: optimal solutions in HTM and QTM for the DeepCubeA test set (or a subset).
* Rubik's Square: full BFS of the whole state space for the smallest N (exact God's number), then IDA* for as large an N as the budget allows.
* Report the largest N, per puzzle, at which IDA* still solves random states within the budget.

## How (implemented)
| File | What |
|---|---|
| `cube3.py` | Cubie model of the 3×3×3 read off the engine's clusters (move tables are derived from the engine's sticker permutations, and a test checks they agree), Korf's three pattern databases built by BFS in Numba, parallel IDA* (one root move per thread) |
| `square.py` | Full-state BFS (`gods_number`), IDA* with h = max over clusters of exact cluster distances (Numba) |
| `solver.py` | `m1_idastar`: 3×3×3 (`metric="htm"` 18 face turns, or `"qtm"` 12 quarter turns) and Squares; other puzzles raise `ValueError` |
| `label.py` | Writes optimal lengths into datasets (`optimal_len`) and prints God's numbers |

* Pattern databases: one byte per entry (not 4 bits: 173 MB per metric fits in RAM and keeps
  lookups one load). Built on first use in ~1 min per metric, cached in `checkpoints/m1/` (git-ignored).
* Optimal for the colour goal: on the 3×3×3 the fixed centers fix the orientation; on the Square,
  IDA* runs to each of the 4 rotations of the goal (most promising first, later runs capped
  below the best length found) and keeps the shortest.

```bash
python -m rubiks.solvers.m1_idastar.label --gods 3                      # exact God's number, Square 3
python -m rubiks.solvers.m1_idastar.label --puzzle square --N 3-5 --time 120
python -m rubiks.bench run --solver m1_idastar --puzzle cube --N 3 --dataset random --time 1800
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
- [x] Pattern databases built, entry counts matching the table above.
- [ ] Optimal solutions for ≥ 100 3×3×3 instances, mean length in HTM and QTM. *Done for 91/100
      scrambles of depth 12 (13.2 HTM, 17.6 QTM); random states are out of reach on 4 CPU cores — see docs/STATUS.md.*
- [x] Exact God's number for the smallest Squares from full BFS.
- [x] Plot of IDA* success rate against N for the Square.
- [x] Write optimal lengths back into the datasets' `optimal_len` field (`rubiks.bench.datasets`).

## Workflow
1. Benchmark: `python -m rubiks.bench make --puzzle square --N 3-8 --depth 20`, then
   `python -m rubiks.bench run --solver m1_idastar --puzzle square --N 3-8 --dataset d20`.
2. Open a PR; CI must pass and one reviewer approves.
