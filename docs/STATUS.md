# Status (2026-10-16)

Summary: Method 5B solves 100% of random states tested for the Square (N = 3–20) and the
Cube (N = 3–10; 15 and 20 in progress). Method 5A beats 5B on solution length at the Square
sizes tested, including unseen N = 10. Solutions are still long; see each section.

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

## Method 5B: Cube, N = 3–10 — complete (15 and 20 running)
`results/m5b_stages/cube_random.jsonl` (2 random states per N, width 512, widening per stage):

| N | n | solved | HTM mean ± 95% CI | QTM mean | HTM / (N²/ln N) | mean time (s) |
|---|---|---|---|---|---|---|
| 3 | 2 | 100% | 52.0 ± 15.7 | 54.5 | 6.35 | 19 |
| 4 | 2 | 100% | 246.0 ± 27.4 | 249.5 | 21.31 | 26 |
| 5 | 2 | 100% | 452.5 ± 2.9 | 459.0 | 29.13 | 62 |
| 6 | 2 | 100% | 724.5 ± 110.7 | 729.5 | 36.06 | 109 |
| 7 | 2 | 100% | 1045.0 ± 33.3 | 1049.5 | 41.50 | 161 |
| 8 | 2 | 100% | 1510.5 ± 120.5 | 1515.0 | 49.08 | 269 |
| 10 | 2 | 100% | 2260.5 ± 22.5 | 2270.5 | 52.05 | 530 |

What made the cube work (each step measured on the failing cases):
1. Edges-first stage order (corners → middle edges → wings → centers), split into 6-piece stages.
2. Coarse exact tables (opposite-colour pair; 3-class for corners, 153,090 states) combined with
   the cluster MLPs. Corners use the tables alone: 4–15× faster than max(MLP, tables).
3. Late center stages search only pure-center commutators [A, B C B'] (middle slice included),
   verified to move only centers: 5×5×5 centers went from budget-limited to < 5 s per stage.
4. Per-stage width ceiling scaled by action count (width × actions ≤ 1.2M).

**Solution length is the weak point.** These are 5–50× longer than good NxN solvers (a
reduction solver does a 4×4×4 in ~60 moves; the 3×3×3 optimum averages ~18). The stage
structure trades length for reliability: every 6-piece stage pays for commutators
(4–8 moves each) and keeps earlier work intact. The paper needs this to fall a lot.
Ranked by expected effect:
* min2phase for the final 3×3×3-equivalent part (corners + edges) instead of beam search.
* Larger pieces per stage once the cluster MLPs are trained properly on GPU
  (fewer, wider stages = less protection overhead), and wider beams via the host tier.
* Method 5A's per-token costs as the stage heuristic (5A beat 5B on the Square, below).
* Batching like Demaine Lemma 10: one commutator fixing many clusters at once.

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
