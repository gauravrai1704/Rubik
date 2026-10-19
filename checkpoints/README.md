# Checkpoints

Committed (small, CPU-trained, enough to run everything out of the box):

* `m5b/<kind>_<signature>.pt` — Method 5B per-cluster MLPs, one per cube cluster type,
  reused for every N. Retrain larger on GPU:
  `python -m rubiks.solvers.m5b_stages.train --puzzle cube --N 3-7 --steps 20000 --hidden 512 --device cuda --force`
* `m5b/tables/*.npz` — coarse exact tables (built automatically on first use if missing).
* `m4/square<N>.pt` — Method 4 per-N ResMLPs for Squares 3–8 (CPU, "small" preset). The 3×3×3
  paper3 model is `m4/cube3.large.pt` (16 MB, not committed; `m4/cube3.large.json` records its training).
* `m5a/square.pt` — Method 5A cluster-token model, 0.2M params, Squares 4–8, 2,500 CPU steps.

Large GPU-trained models: name them `*.large.pt` (git-ignored) and share them as GitHub
release assets or on the team drive. Every Rubik's Square cluster uses an exact table, so
Method 5B on the Square needs no checkpoint at all.
