# Checkpoints

Trained weights are not committed (see `.gitignore`). Share them as GitHub release
assets or on the team drive, and put them here:

* `m5b/<kind>_<signature>.pt` — Method 5B per-cluster MLPs (one per cluster type; reused for every N).
  Train: `python -m rubiks.solvers.m5b_stages.train --puzzle cube --N 3-7 --device cuda`
* `m5a/square.pt`, `m5a/cube.pt` — Method 5A cluster-token models.
  Train: see `python -m rubiks.solvers.m5a_tokens.train --help`.

Every Rubik's Square cluster uses an exact table built on the fly, so Method 5B on the
Square needs no checkpoint at all.
