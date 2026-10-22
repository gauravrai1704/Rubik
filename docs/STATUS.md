# Status (2026-10-22)

Summary: **all six methods are implemented**. On the same random instances:
* **Square:** Method 1 gives exact optima up to N = 5 and Method 4's per-N models match them.
  For N = 9–20 (beyond the per-N models), Method 3 (Demaine, batched) is shortest at every N,
  Method 2 next, both ahead of 5B. Exact God's numbers from full BFS: 6 (N = 3), 10 (N = 4), 12 (N = 5).
* **Cube:** Methods 2 and 3 solve 100% for N = 3–20; Method 3 is 20% shorter than Method 2 at
  N = 20 and 33% shorter than 5B. Method 1 solves 3×3×3 scrambles optimally (91/100 at
  depth 12 within 120 s each); random 3×3×3 states are beyond IDA* with Korf's databases on 4 CPU cores.
* HTM / (N²/ln N) still rises with N for every method, so none reaches Demaine's rate yet at N ≤ 20.

All numbers come from the harness or the evaluation scripts on CPU, no GPU (Methods 1–3: 4 cores).
Random states = long random canonical walks (Section 3.4). HTM after canonicalisation.
Instances are generated deterministically, so rows with the same N and id are the same states
across methods; the tables below average every method over the same instances.

## All methods, same instances

Rubik's Square, HTM (Square moves are all 180°, so QTM = HTM). n = instances per N.
Method 4 at W = 4096; 5A as in its section; "—" = not run at that N.

| N | n | M1 (optimal) | M2 | M3 | M4 | 5A | 5B |
|---|---|---|---|---|---|---|---|
| 3 | 5 | 4.2 | 4.8 | 4.8 | 4.2 | — | 5.6 |
| 4 | 5 | 6.4 | 10.8 | 10.8 | 6.4 | 6.4 | 12.4 |
| 5 | 5 | 8.8 | 19.2 | 19.2 | 8.8 | 9.8 | 22.4 |
| 6 | 5 | 0/5 in 120 s | 29.6 | 27.6 | 14.0 | 14.8 | 27.6 |
| 7 | 5 | — | 37.6 | 36.4 | 15.0 | 22.7 (3/5 solved) | 40.4 |
| 8 | 5 | — | 60.4 | 56.4 | 20.4 | 55.2 | 59.2 |
| 9 | 5 | — | 66.0 | 63.2 | — | — | 80.8 |
| 10 | 5 | — | 96.4 | 86.8 | — | — | 100.8 |
| 11 | 5 | — | 108.6 | 95.0 | — | — | 123.2 |
| 12 | 5 | — | 138.4 | 121.2 | — | — | 144.4 |
| 13 | 2 | — | 154.5 | 136.5 | — | — | 166.0 |
| 14 | 2 | — | 179.0 | 153.0 | — | — | 230.0 |
| 15 | 2 | — | 185.0 | 165.0 | — | — | 251.0 |
| 16 | 2 | — | 242.0 | 203.0 | — | — | 290.0 |
| 17 | 2 | — | 248.5 | 205.5 | — | — | 275.0 |
| 18 | 2 | — | 290.0 | 244.0 | — | — | 356.0 |
| 19 | 2 | — | 312.0 | 258.0 | — | — | 406.0 |
| 20 | 2 | — | 380.0 | 303.0 | — | — | 472.0 |

Methods 2 and 3 were also run on 5 states for every N (`docs/results/m*_square_random.jsonl`):
at N = 20, 379.6 vs 304.0 HTM.

Rubik's Cube, HTM (QTM in brackets), every state solved:

| N | n | M2 | M3 | 5B |
|---|---|---|---|---|
| 3 | 2 | 21.0 (31.0) | 21.0 (31.0) | 52.0 (54.5) |
| 4 | 2 | 219.0 (259.5) | 192.5 (226.5) | 246.0 (249.5) |
| 5 | 2 | 319.5 (359.0) | 258.0 (284.0) | 452.5 (459.0) |
| 6 | 2 | 574.5 (654.5) | 464.5 (505.5) | 724.5 (729.5) |
| 7 | 2 | 811.0 (911.5) | 633.0 (689.5) | 1045.0 (1049.5) |
| 8 | 2 | 1131.0 (1255.0) | 897.5 (960.0) | 1510.5 (1515.0) |
| 9 | 2 | 1498.0 (1648.5) | 1141.5 (1217.5) | — |
| 10 | 2 | 1900.5 (2113.5) | 1509.0 (1603.5) | 2260.5 (2270.5) |
| 12 | 1 | 2843 (3123) | 2290 (2409) | — |
| 15 | 1 | 4565 (4985) | 3578 (3767) | 5280 (5292) |
| 20 | 1 | 8259 (8976) | 6612 (6898) | 9904 (9916) |

![HTM / (N²/ln N) for Methods 2, 3 and 5B](figures/methods_2_3_scaling.png)

## Method 1: IDA* + pattern databases — implemented
* **Pattern databases** (Korf): corners 88,179,840 entries (max depth 11 HTM / 14 QTM), two 6-edge
  tables of 42,577,920 entries (max depth 10 HTM / 11 QTM), built by BFS in ~1 min per metric on 4 cores.
* **3×3×3, IDA\*** (parallel over the root's moves, ~12M nodes/s on 4 cores):
  * 100 scrambles of depth 12 (`docs/results/m1_cube3_d12.jsonl`, 120 s each): 91 solved optimally,
    mean 13.2 HTM (17.6 QTM), optimal lengths 8–16; the other 9 need more than 120 s.
  * Face-turn scrambles of depth 14 / 15 / 16: optimal in 3 s / 8 s / 905 s (10.2·10⁹ nodes).
    Each extra move costs ~13×, so a random state (~18 HTM) needs ~10¹²–10¹³ nodes: about a day per
    state here.
    Measured: 2 random states, 30 min each (`docs/results/m1_cube3_random.jsonl`): neither solved;
    each search covered ~2.7·10¹⁰ nodes and proved the optimum is longer than 16 HTM.
* **Square:** exact God's number from full BFS (colour goal, i.e. any of the 4 solved orientations):

  | N | states | God's number | distance distribution |
  |---|---|---|---|
  | 3 | 768 | 6 | 4, 24, 68, 160, 288, 156, 68 |
  | 4 | 82,944 | 10 | 4, 32, 152, 640, 2,464, 7,152, 16,248, 24,720, 20,700, 8,928, 1,904 |
  | 5 | 10,616,832 | 12 | 4, 40, 276, 1,648, 8,876, 42,752, 181,000, 641,772, 1,738,920, 3,105,096, 3,211,932, 1,446,544, 237,972 |

  (To the reference orientation only, the 3×3 Square's God's number is 8.)
* **Where exact search stops (Square):** IDA* with the max-cluster heuristic solves 100% of random
  states at N = 3, 4, 5 (N = 5 in 21 s on average) and 0% at N = 6 within 120 s.
  Optimal lengths are written into `data/square/N{3,4,5}/random.jsonl` (`optimal_len`).

  ![IDA* success rate by N](figures/m1_square_ida_success.png)

## Methods 2 and 3: reduction vs Demaine's parallel algorithm — implemented
Both solve **100% of random states for N = 3–20 on both puzzles** (Square: 5 per N; cube: 2 per N
for N ≤ 10, 1 for 12, 15, 20). They share every non-batched stage, so the gap is the parallelism.

Moves per cube stage (mean HTM):

| N | wing parity | centers M2 | centers M3 | 3×3×3 (Kociemba) | wings M2 | wings M3 |
|---|---|---|---|---|---|---|
| 4 | 0 | 82 | 74 | 20 | 118 | 99 |
| 5 | 1 | 182 | 143 | 22 | 116 | 93 |
| 6 | 0.5 | 324 | 257 | 18 | 233 | 190 |
| 7 | 0.5 | 548 | 424 | 21 | 242 | 188 |
| 8 | 0.5 | 777 | 610 | 19 | 336 | 270 |
| 9 | 1.5 | 1109 | 844 | 21 | 368 | 277 |
| 10 | 1.5 | 1431 | 1141 | 20 | 449 | 347 |
| 12 | 3 | 2202 | 1799 | 19 | 621 | 469 |
| 15 | 4 | 3830 | 3035 | 21 | 710 | 518 |
| 20 | 4 | 7194 | 5885 | 19 | 1042 | 704 |

* Centers dominate. Method 3's batching works best early in the center stage (one round of the
  N = 20 cube applied one commutator type to 18.6 orbits on average) and fades as the orbits
  diverge, when single-orbit 3-cycles win the per-move score. Batching more of the late rounds is
  the obvious next step (a learned schedule is the Section 6.4 stretch goal).
* Every parity case is reached in the tests: odd wing orbits on odd and even N (Lemma 7 slice turn),
  and PLL parity on even N (one outer quarter turn before Kociemba).
* Speed: Method 2 solves a 20×20×20 in 92 s, Method 3 in 32 s (plus 92 s to build its type bank once per N).

## Method 5B: Rubik's Square, N = 3–20 — complete
`docs/results/m5b_square_random.jsonl` (width 128, widening to 2048; 900 s budget):

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

## Method 5B: Cube, N = 3–20 — complete
`docs/results/m5b_cube_random*.jsonl` (2 random states per N for N ≤ 10, 1 for 15 and 20; width 512, widening per stage):

| N | n | solved | HTM mean ± 95% CI | QTM mean | HTM / (N²/ln N) | mean time (s) |
|---|---|---|---|---|---|---|
| 3 | 2 | 100% | 52.0 ± 15.7 | 54.5 | 6.35 | 19 |
| 4 | 2 | 100% | 246.0 ± 27.4 | 249.5 | 21.31 | 26 |
| 5 | 2 | 100% | 452.5 ± 2.9 | 459.0 | 29.13 | 62 |
| 6 | 2 | 100% | 724.5 ± 110.7 | 729.5 | 36.06 | 109 |
| 7 | 2 | 100% | 1045.0 ± 33.3 | 1049.5 | 41.50 | 161 |
| 8 | 2 | 100% | 1510.5 ± 120.5 | 1515.0 | 49.08 | 269 |
| 10 | 2 | 100% | 2260.5 ± 22.5 | 2270.5 | 52.05 | 530 |
| 15 | 1 | 100% | 5280 | 5292 | 63.2 | 1,875 |
| 20 | 1 | 100% | 9904 | 9916 | 74.2 | 4,760 |

What made the cube work (each step measured on the failing cases):
1. Edges-first stage order (corners → middle edges → wings → centers), split into 6-piece stages.
2. Coarse exact tables (opposite-colour pair; 3-class for corners, 153,090 states) combined with
   the cluster MLPs. Corners use the tables alone: 4–15× faster than max(MLP, tables).
3. Late center stages search only pure-center commutators [A, B C B'] (middle slice included),
   verified to move only centers: 5×5×5 centers went from budget-limited to < 5 s per stage.
4. Per-stage width ceiling scaled by action count (width × actions ≤ 1.2M).

**Solution length is the weak point.** These are several times longer than good NxN solvers:
the 3×3×3 optimum averages ~18 HTM; for scale, the KMCoders Kaggle Santa 2023 write-up
reports a 33×33×33 in 4,719 moves (their move metric may differ from our HTM), about half our
20×20×20 count. HTM / (N²/ln N) still rises with N, so
5B does not yet reach Demaine's growth rate on the cube. The stage
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

## Method 4 (CayleyPy reproduction): implemented; Square done, cube needs GPU training
Per-N ResMLP + beam search, as in Chervov et al. (Section 7). Committed CPU models
`checkpoints/m4/square{3..8}.pt` ("small" preset, 0.25–0.45M parameters, 4M examples for N ≤ 5 and
10M for N = 6–8, 1–4 min each on CPU; time and fit are recorded in each `.json`).

Squares, 5 random states per N, the same instances for every method (HTM; Square moves are all flips, so QTM = HTM):

| N | M4 W=16 | M4 W=64 | M4 W=256 | M4 W=1024 | M4 W=4096 | 5A | 5B |
|---|---|---|---|---|---|---|---|
| 3 | 4.2 | 4.2 | 4.2 | 4.2 | 4.2 | — | 5.6 |
| 4 | 6.8 | 6.4 | 6.4 | 6.4 | 6.4 | 6.4 | 12.4 |
| 5 | 9.8 | 9.8 | 9.4 | 9.0 | 8.8 | 9.8 | 22.0 |
| 6 | 16.0 (4/5) | 14.8 | 14.0 | 14.0 | 14.0 | 14.8 | 30.4 |
| 7 | 17.6 | 16.0 | 15.8 | 15.4 | 15.0 | 22.7 (3/5) | 40.8 |
| 8 | 27.3 (3/5) | 44.7 (3/5) | 24.5 (4/5) | 22.0 | 20.4 | 55.2 | 63.2 |

All 100% solved unless marked. W = 4096 takes ≤ 10 s per state on CPU. Length falls with log W, as the
paper reports, and the per-N model sets a hard bar for 5A: 5A roughly matches it on N ≤ 6 but is 2.7× longer at N = 8.

**3×3×3: not reproduced yet.** The paper3 model (4.0M parameters, 10 layers, QTM moves, kmax 26) trained
for 32M examples on CPU (77 min). That is 250× less data than the paper's 8B, and the fit is weak
(correlation with walk depth 0.90). With it, beam search solves short scrambles (depth 6/10/14 QTM: all
solved, at or below scramble length; depth 18: solved in 24–28) but **0 of 10 random states** at W = 256,
1024 or 4096, and 0 of 4 at W = 16384 (900 s each). The weights are `cube3.large.pt` (16 MB, git-ignored).
The 21.14 QTM check needs the GPU run: `DEVICE=cuda bash experiments/m4_cayleypy.sh`
(8192 epochs × 1M, then W up to 2¹⁸; `--kw '{"tier": "host"}'` for wider).

## Known limitations
* Random states come from long random walks, not an exact uniform sampler.
* The checker enforces necessary conditions (property-tested); wing and center rules are
  per-layer multiset checks rather than full parity proofs.
* All timings are CPU-only and include contention with concurrent jobs.
* Method 1's pattern databases are the spec's 6-edge tables; 7-edge tables or inverse-state
  lookups would cut 3×3×3 search several-fold but not to random states on this CPU.
* Not run: the dwalton76 NxNxN solver (external baseline, needs network access to install).
