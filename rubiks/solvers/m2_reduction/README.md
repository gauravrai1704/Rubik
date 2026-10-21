# Method 2: Reduction + Kociemba two-phase

**Owner:** Sunaina (implemented by Gaurav) · **Spec:** technical basis, Section 5 · **Registered name:** `m2_reduction`

The standard way to solve any N: reduce to a 3×3×3 stage by stage, then use a fast 3×3×3 solver. Works for every N with O(N²) moves.

## Scope
* Cube N = 3..20 and Square N = 3..20, 100% of instances.

## How (implemented)
Cube stages (each a function `state -> moves`; moves per stage in `SolveResult.info["stages"]`):

| Stage | What |
|---|---|
| orient | Relabel colours so the target is the reference orientation (odd N: the fixed centers decide; even N: the rotation matching the most centers) |
| `wing_parity` | One inner-slice quarter turn per odd wing orbit (Demaine Lemma 7); first, because it disturbs centers |
| `centers` | Each center orbit with pure 3-cycles `s [A, B C B'] s'` (`cluster_tools.CycleLibrary`: every 3-cycle of the orbit, found by conjugating automatically found base commutators) |
| `three` | Corners, middle edges, fixed centers: Kociemba two-phase (`kociemba` package) on outer layers, which keep centers in place; even-N PLL parity = one outer quarter turn first |
| `wings` | Each wing orbit with pure 3-cycles `s [X, Y Z Y'] s'`: wings go home, so OLL parity never reaches the 3×3×3 stage |

Square: frame (exact BFS over the projected 3×3 Square), edge groups (Lemma 3: edge cluster +
cross cluster, single flips + border commutators, exact BFS), then each interior cluster with its
four Lemma 2 commutators (exact, ≤ 3 commutators). Shared with Method 3 in `solvers/square_plan.py`.

Not done: the dwalton76 external baseline (no network access to install it here).

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
- [ ] Solves 100% of benchmark instances for N = 3..20 on both puzzles.
- [ ] Every parity case reached at least once in testing, for odd and even N.
- [ ] Moves per stage reported for each N.

## Workflow
1. Benchmark: `python -m rubiks.bench make --puzzle square --N 3-8 --depth 20`, then
   `python -m rubiks.bench run --solver m2_reduction --puzzle square --N 3-8 --dataset d20`.
2. Open a PR; CI must pass and one reviewer approves.
