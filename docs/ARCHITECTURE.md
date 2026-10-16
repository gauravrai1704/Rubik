# Architecture notes

How the code maps to the technical basis, and the design decisions made while building it.

## Engine (Section 3)
* **One geometry for both puzzles.** A cuboid (d0, d1, d2); stickers carry a doubled, centred
  cubie coordinate and an outward normal. A move rotates every sticker of one slab with an
  integer rotation matrix; the result is looked up to give a permutation. A slab whose cross
  section is not square (a row of the Square) only gets 180° turns; a length-1 axis gets none.
* **Orbits vs clusters.** `orbits()` are sticker orbits (union-find over all move permutations).
  `clusters()` merge sticker orbits that share cubies, giving Demaine's clusters: 24 wings
  (48 stickers), 4 Square cubies (top, bottom, side stickers), and so on. Methods work on clusters.
* **N-invariant order.** Inside a cluster, stickers are sorted by face, the signs of their
  in-face coordinates and their relative magnitudes. So cluster types have one `signature`
  (generator set + solved pattern) at every N; `tests/test_engine.py` checks this.
* **Checker** combines colour counts, exact projection membership for small clusters, cube
  piece rules (corners, middle edges, wings; orientation normalised by fixed centers for odd
  N) and the Square cross/boundary coupling. These are necessary conditions; property tests
  confirm scrambles are accepted and classic unreachable states rejected.
* **Random states** are long canonical random walks (≥ 25·M moves), not an exact uniform
  sampler; for these Cayley graphs that is close to uniform. Revisit if E2 needs exact uniformity.

## Beam search (Section 9.3)
* The beam lives in a `BeamStore` (`tier='device' | 'host' | 'disk'`), 3-bit packed. Per step
  only (score, hash, parent, action) is kept for children; the next beam is rebuilt from
  (parent, action). The GPU holds one chunk of parents and its children; chunk size is capped
  by `max_chunk_mb`.
* Duplicates removed by a 64-bit random-linear hash, within a step and across the last
  `visited_window` steps.
* **Macro actions** (`rubiks/search/macros.py`): commutators / conjugates offered as single
  steps. A summed per-cluster heuristic has hills (fix one cluster = break then restore
  others); a macro turns the hill into one downhill step. Square: `c r c r`, `r c r`, `c r c`
  for every row/column pair (Demaine Lemmas 2–4). Cube: `X Y X' Y'` (inner slice X, face Y;
  never moves corners) and pure-center commutators `[A, B C B']` such as `r U l' U' r' U l U'`,
  each generated and then *verified* to move only center stickers.
* **Q*-style scoring:** a heuristic with `score_children(parents) -> (c, A)` is used to score
  all children of a parent in one pass (5A's move head).

## Method 5B (Section 9)
* Per-cluster heuristics keyed by signature. Exact BFS tables when the projected puzzle has at
  most 1.5M states — every Square cluster and the 3×3×3 fixed centers. Otherwise a small ResMLP
  trained on random walks of the projected puzzle (`train.py`), reused for every N, combined by
  `max` with **coarse exact tables**: the projection with one opposite-colour pair mapped to 1
  and the rest to 0 (corners 2,187 states, middle edges 126,720, every center type 735,471).
  These are admissible abstraction heuristics; they are cached in `checkpoints/m5b/tables/`.
* Split stages: wing and center clusters are solved `pieces_per_stage` pieces (default 6) at a
  time; every stage adds `exact_weight` × (goal stickers not yet solved) to its heuristic.
* Stage goal is exact: targets plus every earlier cluster at their solved stickers.
* Each stage only searches actions that touch its targets, and its heuristic only includes
  protected clusters those actions can move. This took the 20×20 Square from > 300 s to ~50 s.
* Default orders: Square = frame → edges → crosses → interiors (Demaine). Cube = `edges_first`
  (fixed centers → corners → middle edges → wings → centers), or `order="reduction"`.
* Measured status is in `docs/STATUS.md`.

## Method 5A (Section 8)
* `Tokenizer`: one token per cluster; content projected by a linear map per cluster type
  (`kind:size`), plus a kind embedding and depth / N features.
* `ClusterTokenModel`: transformer encoder; additive value head (per-token costs, summed, plus
  a global correction scaled by N²/ln N); optional move head (move tokens cross-attend to
  cluster tokens) distilled from the value head in phase 2.
* `mode="full"`: beam search on the whole puzzle with the value (or move) head.
  `mode="staged"`: Method 5B's stages with per-token costs as the stage heuristic — the merge
  described in Section 8.5. Default: full for N ≤ 7, staged above.
