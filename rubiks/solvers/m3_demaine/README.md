# Method 3: Demaine's parallel cluster algorithm

**Owner:** Kashvi (implemented by Gaurav) · **Spec:** technical basis, Section 6 · **Registered name:** `m3_demaine`

The asymptotically optimal O(N²/log N) algorithm of Demaine et al. (2011): one slice move touches many clusters, so clusters needing the same fix share one move sequence. Build the Square first (6 interior configurations), then the Cube.

## Scope
* Square N = 3..20, then Cube N = 3..20.

## How (implemented)
* **Square (Theorem 1).** Frame and edge groups exactly as Method 2 (Lemma 3, O(N)). Interior
  clusters: each cluster's optimal word of Lemma 2 commutators is read off its exact table; round r
  applies every cluster's r-th commutator, grouped into blocks X × Y (`batching.blocks`: Demaine's
  column groups of size ℓ, with ℓ chosen per round to minimise the real cost) and one
  `C_X R_Y C_X R_Y` per block (Lemma 4).
* **Cube (Theorem 3 + the Section 6.4 fix).** Wing parity (Lemma 7) and the 3×3×3 part as Method 2.
  Centers, then wings, by greedy rounds of batched 3-cycles: a *type* is an outer-face setup plus a
  commutator with layer parameters (`[A_i, B C_j B']`, `[X_i, Y Z Y']`); with layer sets I, J it acts
  on every cluster (i, j) ∈ I × J at once. Each round picks the type with the most slots fixed per
  move over all clusters (or a single-cluster 3-cycle when that is better), and checks the result
  sticker by sticker; a block that would disturb anything else is split.

So Methods 2 and 3 differ only in the batching, which is the comparison Section 6.5 asks for.

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
1. Benchmark: `python -m rubiks.bench make --puzzle square --N 3-8 --depth 20`, then
   `python -m rubiks.bench run --solver m3_demaine --puzzle square --N 3-8 --dataset d20`.
2. Open a PR; CI must pass and one reviewer approves.
