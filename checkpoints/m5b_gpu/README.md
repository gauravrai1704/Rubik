# Method 5B cluster MLPs trained on a Colab T4 (optional, not the default)

Trained by Shaurya with `notebooks/train_gpu_colab.ipynb` (§6): hidden width 512, 20,000 steps per
cluster type, on cubes N = 3-7 (`train.log`). The default models in `checkpoints/m5b/` are the CPU ones
(width 256, 1,500 steps). The coarse tables are byte-identical to `checkpoints/m5b/tables/`.

Use them with `ModelBank(root=Path("checkpoints/m5b_gpu"), tables_only=("corner",))`, passed to
`StagedSolver(..., bank=...)`.

**Why they are not the default.** They predict walk depth better (rank correlation with depth on fresh
walks: center_x 0.80 -> 0.84, wing 0.75 -> 0.79, corner 0.57 -> 0.65; less overestimation near the goal),
but as beam-search heuristics they give longer solutions and run slower. Same CPU, same code, same
instances, only the models swapped (`docs/results/m5b_ab_*.jsonl`):

| N | CPU models (HTM) | these models (HTM) | time, CPU models | time, these models |
|---|---|---|---|---|
| 3 | 60, 44 | 60, 52 | 4-18 s | 9-23 s |
| 4 | 232, 260 | 253, 297 | 15-17 s | 45-54 s |
| 5 | 454, 451 | 431, 435 | 36-39 s | 288-1,443 s |
| 6 | 668, 781 | 762, 881 | 61-71 s | 169-195 s |

The Colab run itself (`docs/results/m5b_cube_gpu_colab.jsonl`, T4) was longer on all 14 cube states up
to N = 10 and failed 2 (wing stages exhausted the beam width). A better walk-length predictor is not a
better beam heuristic here, as the paper's corner ablation also found.
