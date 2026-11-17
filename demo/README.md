# Demo: short solutions for large Rubik's Cubes and Squares

A local web page that scrambles, solves and replays Rubik's Squares (N×N×1) and Cubes (N×N×N) with our engine,
plus a replay of real SymPort results on Santa 2023 puzzles. Every solution shown is re-applied by the engine
before it is marked verified.

## Run it

```bash
pip install -e .                 # once, from the repo root (same environment as the paper's code)
python demo/server.py            # wait for "demo running at http://localhost:8000", then open it
# (--host 0.0.0.0 to reach it from another device on the network, e.g. a projector laptop)
```

- Fallback without a browser: `python demo/cli_demo.py` (same three parts, in text, under 20 s).
- Backup video: `demo/demo_recording.webm` (a full run-through, recorded automatically).
- The Santa data (`demo/data/santa.json`) is not in git: it holds Kaggle's puzzle states and KMCoders' published
  moves, which we do not redistribute. Copy the file you were given into `demo/data/`. It is precomputed from our result files. To rebuild it, run
  `python demo/precompute_santa.py`. This needs `data/santa2023/` (downloaded on first use) and
  `docs/results/santa/m7_copy_cache/`.
- No GPU, no KMCoders binary and no nxopt are needed for the demo.

## What runs live, and how fast (4-core laptop-class CPU)

| Puzzle | Sizes offered | Solvers | Typical time |
|---|---|---|---|
| Square | 3–20 | Reduction, Demaine, NetGain | < 0.1 s at any size |
| Square | 3–8 | CayleyPy (learned, W = 1024), SymPort (8 copies × CayleyPy + NetGain) | 1–4 s at N = 6; up to ~25 s at N = 8 |
| Square | 3–20 | SymPort above N = 8 (8 copies × NetGain + Demaine) | < 2 s |
| Square | 3–4 | IDA*-PDB (optimal) | < 1 s |
| Square | 4–8 | OneNet (unticked by default) | 20–40 s |
| Cube | 3–8 | Reduction, Demaine, NetGain | 5–7 s at N = 6; ~18 s at N = 8 |

SymPort on the cube is not run live: it runs KMCoders' program on up to 48 copies, taking minutes (4×4×4) to hours
(9×9×9) per puzzle. Tab 2 replays its real results instead.

## Run-through (about 7 minutes)

Rehearse with these exact settings: the seeds are fixed, so the numbers below will repeat.

**0. Open (15 s).** "One engine for both puzzles and any size; eight solvers. We'll scramble, solve live, then
show our main method on the competition puzzles."

**1. The puzzles (1 min). Tab 1, Square, N = 6.**
- Show the solved Square. A move flips one row or column over (top ↔ bottom); side stickers sit around the
  top view.
- Tick **show clusters**: "A piece only ever moves inside its own cluster: corners, edges, interior.
  Solving the puzzle means solving every cluster." Untick it.
- Switch to **Cube**, N = 6, tick **show clusters**: corners, wings, × and oblique centers. Untick it and switch
  back to Square.

**2. Solve a Square live (1.5 min). Square, N = 6, seed 1.**
- Click **Scramble**, then **Solve with selected** (about 4 s).
- Expected: Reduction 34, Demaine 36, NetGain 36, CayleyPy 14, SymPort 14. All verified ✓.
- Say: "The learned per-size network finds the short solution at small N; SymPort runs the strongest solvers on
  all 8 symmetric copies and keeps the shortest."
- Click **replay** on SymPort, then **▶ play**. The stickers moved by each step are outlined.
- Optional: N = 4, Scramble, Solve. IDA*-PDB gives the proven optimum, and CayleyPy, OneNet and SymPort match it.
- Optional: N = 14. Only the cluster methods and SymPort run there; SymPort is shortest: 158 vs 166 for
  NetGain and Demaine, 204 for Reduction.

**3. Solve a cube live (1.5 min). Cube, N = 6, seed 1.**
- **Scramble**, **Solve** (about 7 s). Expected: Reduction 629, Demaine 476, NetGain 449 moves (HTM).
- Say: "Same pipeline; the difference is the schedule. Reduction fixes one cluster at a time. Demaine batches
  only when every cluster needs the move. NetGain batches whenever the block pays. On 20 random cubes per size it
  is significantly shorter from N = 5, and 17% shorter at N = 10."
- Replay NetGain for a few seconds (speed slider up).

**4. SymPort on Santa 2023 (2 min). Tab 2.**
- Default puzzle: 4×4×4, puzzle 162.
- Point at the dots: "48 views of one scramble, each solved by KMCoders' own program and mapped back. A symmetry
  maps every slice turn to a slice turn of the same cost, so the lengths carry over exactly."
- Read the three numbers: **SymPort 55, published 73, one run 85.**
- Click **Replay SymPort**. It ends at the exact goal ✓.
- Click **Verify (exact goal)**: both move lists are checked by the engine.
- Pick **9×9×9 puzzle 271** (SymPort 361 vs published 341): "Honest limit: at 8 and 9 the copies vary too
  little relative to the solution length, so we stop there."
- Point at Table 2: best for N = 4–7, second at 8 and 9.

**5. Close (15 s).** "Everything is open: engine, benchmarks, solvers and every result file."

## If asked

- *Why not run SymPort on the cube live?* One copy is about 20 s at 4×4×4 and 47 CPU-min at 9×9×9; 48 copies
  would not fit a demo. The results shown are the paper's runs, replayed and re-verified live.
- *How do you know the solutions are correct?* The engine re-applies every move list; the solvers never decide
  success. Tab 2's **Verify** does this on the spot.
- *Is the scramble random?* Each scramble is the end of a long random walk from solved (seeded), as in the paper's
  benchmark.

## Files

| File | What |
|---|---|
| `server.py` | Local web server (standard library); solvers run in worker processes |
| `static/index.html` | The page (no external libraries; works offline) |
| `precompute_santa.py` | Builds `data/santa.json` from saved SymPort copies and KMCoders' published moves |
| `cli_demo.py` | Terminal fallback |
| `record_demo.js` | Re-records the backup video with Playwright (`node demo/record_demo.js` while the server runs) |
