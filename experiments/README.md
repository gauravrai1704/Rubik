# Experiments (technical basis, Section 10.3)

Each script regenerates its datasets (seeded), runs the harness and prints a summary. Run
from the repo root on the GPU machine with `DEVICE=cuda`. Results land in `results/`.

| Script | Question |
|---|---|
| `e1_optimality.sh` | How close to optimal is each method? (needs Method 1) |
| `e2_lengths.sh` | Shortest solutions at each size, N = 3–20, both puzzles |
| `e3_scaling.sh` | Does length grow as N²/ln N, and with what constant? (`norm_htm` column) |
| `e4_beam_width.sh` | Length vs beam width W |
| `e5_generalization.sh` | Does 5A generalize to sizes it never saw? (train ≤ 7, test 8–20) |
| `e6_tiers.sh` | Tiered memory: how much wider a beam, at what speed cost |
| `e7_gods_number.sh` | Upper-bound evidence for God's number, N = 4–20 |
| `e8_canonical_walks.sh` | Does removing commuting moves from training walks help? |
| `go_no_go_5a.sh` | Section 8.6: train 5A on Squares 4–8, test 10/14/20 against Method 2 |

Methods 1–4 drop into these scripts by name once they are implemented.
