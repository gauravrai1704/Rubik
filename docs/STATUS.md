# Status (2026-10-16)

All numbers come from the harness or the evaluation scripts on a 2-core CPU, no GPU.
Random states = long random canonical walks (Section 3.4). HTM after canonicalisation.

## Method 5B: Rubik's Square, N = 3–20 — complete
`results/m5b_stages/square_random.jsonl` (width 128, widening to 2048; 900 s budget):

| N | n | solved | HTM mean ± 95% CI | HTM / (N²/ln N) | mean time (s) |
|---|---|---|---|---|---|
| 3 | 5 | 100% | 5.6 ± 1.9 | 0.68 | 0.0 |
| 4 | 5 | 100% | 12.4 ± 2.3 | 1.07 | 0.0 |
| 5 | 5 | 100% | 22.4 ± 4.0 | 1.44 | 1.2 |
| 6 | 5 | 100% | 27.6 ± 5.2 | 1.37 | 0.8 |
| 7 | 5 | 100% | 40.4 ± 5.7 | 1.60 | 1.3 |
| 8 | 5 | 100% | 59.2 ± 7.8 | 1.92 | 3.7 |
| 9 | 5 | 100% | 80.8 ± 9.7 | 2.19 | 5.3 |
| 10 | 5 | 100% | 100.8 ± 6.6 | 2.32 | 6.3 |
| 11 | 5 | 100% | 123.2 ± 13.6 | 2.44 | 9.3 |
| 12 | 5 | 100% | 144.4 ± 8.0 | 2.49 | 15.3 |
| 13 | 2 | 100% | 166.0 ± 15.7 | 2.52 | 24.8 |
| 14 | 2 | 100% | 230.0 ± 23.5 | 3.10 | 48.9 |
| 15 | 2 | 100% | 251.0 ± 21.6 | 3.02 | 53.3 |
| 16 | 2 | 100% | 290.0 ± 35.3 | 3.14 | 59.6 |
| 17 | 2 | 100% | 275.0 ± 2.0 | 2.70 | 65.1 |
| 18 | 2 | 100% | 356.0 ± 19.6 | 3.18 | 94.3 |
| 19 | 2 | 100% | 406.0 ± 31.4 | 3.31 | 120.1 |
| 20 | 2 | 100% | 472.0 ± 109.8 | 3.53 | 92.4 |

The normalised length still rises with N, so these solutions grow faster than N²/ln N: 5B
solves interior clusters a few at a time and does not yet batch them the way Demaine's
Lemma 4 does. That is the gap Method 3 (and a learned batching policy) should close.

## Method 5B: Cube
Cluster MLPs trained on CPU (1,500 steps each, hidden 256) + coarse exact tables.

| Cube | tried | solved | HTM mean | time per solve |
|---|---|---|---|---|
| 3×3×3 | 5 | 5 | 56.4 | ~40 s |
| 4×4×4 | 4 | 4 | 289 | ~200 s (CPU shared with training) |
| 5×5×5 | see below | | | |

Lengths are far from optimal (3×3×3 optimum averages ~18 HTM). The edges-first order with
commutator macros is reliable but verbose. Expected gains: longer GPU training of the
cluster MLPs, wider beams, and min2phase for the 3×3×3 stage (shared with Method 2).

## Method 5A: first signal
Model: d_model 64, 2 layers, 0.2M parameters, 2,500 steps on CPU, trained on Squares 4–8 only.
Distance-estimate correlation with true walk depth at the end of training:
N=6: 0.67, N=8: 0.64, **N=10: 0.66, N=14: 0.65, N=20: 0.53** (last three never seen).

Same instances as 5B (3 random states per N):

| N | 5A mode | 5A solved | 5A HTM | 5B HTM |
|---|---|---|---|---|
| 4 | full | 3/3 | 6.7 | 13.3 |
| 5 | full | 3/3 | 10.3 | 19.3 |
| 6 | full | 3/3 | 16.0 | 27.3 |
| 5 | staged | 3/3 | 16.0 | 19.3 |
| 8 | staged | 3/3 | 57.3 | 62.0 |
| 10 (unseen) | staged | 3/3 | 77.3 | 102.0 |
| 14 (unseen) | staged | 0/2 | timed out (300 s, CPU) | 230 |

Small samples, but in the direction the paper needs. Next: the go / no-go run on GPU
(`experiments/go_no_go_5a.sh`, 20k steps, d_model 192) and the cube model.

## Methods 1–4
Scaffolds, specs and xfail tests in place; owners implement.

## Known limitations
* Random states come from long random walks, not an exact uniform sampler.
* The checker enforces necessary conditions (property-tested); wing and center rules are
  per-layer multiset checks rather than full parity proofs.
* All timings are CPU-only and include contention with concurrent jobs.
