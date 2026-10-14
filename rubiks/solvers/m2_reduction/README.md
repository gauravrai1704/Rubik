# Method 2: Reduction + Kociemba two-phase

**Owner:** Sunaina · **Spec:** technical basis, Section 5 · **Registered name:** `m2_reduction`

The standard way to solve any N: reduce to a 3×3×3 stage by stage, then use a fast 3×3×3 solver. Works for every N with O(N²) moves.

## Scope
* Cube N = 3..20 and Square N = 3..20, 100% of instances.

## How
* Centers: solve every center cluster (odd N: around the fixed centers; even N: pick face colours first). Use commutators [A, B] = A B A' B' that 3-cycle center pieces of one cluster; parameterise them by layer so they work for any N.
* Edge pairing: group each wing cluster's wings with the middle edge (odd N) into 12 composite edges with slice moves undone afterwards (slice-flip-slice).
* Parity: fix odd wing-orbit parity with the standard inner-slice sequences; for even N these are OLL and PLL parity.
* 3×3×3 stage: Kociemba two-phase — `pip install kociemba`, or min2phase (cs0x7f) / RubiksCube-TwophaseSolver (hkociemba). Map its face turns to outer-layer `Move`s with `rubiks.engine.parse_notation`.
* Square version: solve clusters one at a time with Demaine (2011) Lemma 2 sequences (≤ 6 moves per interior cluster) after the O(N) frame procedure of Lemma 3. Method 3 parallelises exactly this, so keep them comparable.
* Keep each stage a separate function `state -> moves` so Method 5B can swap stages, and log moves per stage in `SolveResult.info`.
* Run the dwalton76 NxNxN solver on the same instances as an external baseline.

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
1. Branch: `git checkout -b m2-reduction`.
2. Implement `solver.py` (add modules in this folder as needed).
3. Remove the `xfail` mark for `m2_reduction` in `tests/test_methods_1_4.py` and make it pass.
4. Benchmark: `python -m rubiks.bench make --puzzle square --N 3-8 --depth 20`, then
   `python -m rubiks.bench run --solver m2_reduction --puzzle square --N 3-8 --dataset d20`.
5. Open a PR; CI must pass and one reviewer approves.
