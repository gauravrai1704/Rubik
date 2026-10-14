# Method 3: Demaine's parallel cluster algorithm

**Owner:** Kashvi · **Spec:** technical basis, Section 6 · **Registered name:** `m3_demaine`

The asymptotically optimal O(N²/log N) algorithm of Demaine et al. (2011): one slice move touches many clusters, so clusters needing the same fix share one move sequence. Build the Square first (6 interior configurations), then the Cube.

## Scope
* Square N = 3..20, then Cube N = 3..20.

## How
* Clusters and their canonical sticker order come from `rubiks.engine.clusters(puzzle)` — `Orbit.stickers`, `Orbit.gen_moves`, `Orbit.gen_perms`.
* Square batching (Lemma 4): if every cluster in columns X × rows Y has the same configuration, flip X, then Y, then X, then Y: each cluster in X × Y gets its own 4-move fix, others see every move twice.
* Square algorithm (Lemmas 5–6, Theorem 1): frame clusters in O(N) (Lemma 3), then for each of the 6 interior configurations group columns into sets of size ½·log₂|Y| and apply Lemma 4 per subset pattern.
* Cube (Lemmas 7–12, Theorem 3): fix edge-cluster parity with single slice moves, solve edge clusters, then for each ordered pair of index groups apply the bulk construction of Lemma 10 with Schütze's 10-move commutator S (Table 1 lists its 24 variants; together they generate A₂₄).
* Practical fix (Section 6.4): cube clusters have ~3·10¹⁵ configurations, so batch at the level of 3-cycles: write each cluster's solution as basic 3-cycles, and each round apply one 3-cycle type to every cluster that needs it next (greedy: the most-needed type). KMCoders' Santa 2023 solution does this.
* `rubiks.search.macros.square_macros` already builds the single-cluster commutators c r c r — reuse for testing your sequences.

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
- [ ] Square solves all instances for N = 3..20.
- [ ] Cube solves all instances for N = 3..20.
- [ ] Plot of moves / (N²/ln N) against N for both puzzles.
- [ ] Direct comparison with Method 2 on the same instances.

## Workflow
1. Branch: `git checkout -b m3-demaine`.
2. Implement `solver.py` (add modules in this folder as needed).
3. Remove the `xfail` mark for `m3_demaine` in `tests/test_methods_1_4.py` and make it pass.
4. Benchmark: `python -m rubiks.bench make --puzzle square --N 3-8 --depth 20`, then
   `python -m rubiks.bench run --solver m3_demaine --puzzle square --N 3-8 --dataset d20`.
5. Open a PR; CI must pass and one reviewer approves.
