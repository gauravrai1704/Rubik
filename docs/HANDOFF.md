# Hand-off: where the project stands (2026-11-01)

Paused at the user's request to save usage. Everything below is committed on `main` (author: Gaurav Rai).

## Headline results
* **Method 7** (`rubiks/solvers/m7_portfolio`): symmetry portfolio around KMCoders' own Santa 2023 solver (48 rotations +
  mirror images; their beam variant `solve_cube_beam`; stage-2 annealing 1M instead of 20M; odd N: optimal QTM central
  3x3x3 from Rokicki's `nxopt`). Against KMCoders' published best-of-many solutions (mean QTM):
  * 4x4x4, 10/43 puzzles: **62.7 vs 63.5** (3 wins / 4 ties / 3 losses), best of 48 cheap + 8 default copies.
  * 5x5x5, 3/19 puzzles: **95.3 vs 100.7** (3 wins), 24 copies with nxopt.
  * Results: `docs/results/santa/m7_*.jsonl`. Paper Table "Method 7 against every baseline" (`tab:m7all`).
* **5C** best of our original methods on random cubes N >= 5 (significant from N = 5, 20 states/size).
* **Counting lower bound** (`rubiks/analysis/counting_bound.py`): KMCoders published = 1.81x bound at N = 4, 2.54x at N = 33.
* Method 4 GPU "large" Square models: better only at N = 8 (18.8 vs 20.3); default there.

## Paper
`paper/main.tex` + `paper/sections/*.tex`; every number from `python paper/scripts/make_tables.py` (Santa / Method 7 part
in `paper/scripts/santa_part.py`). Build: `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`.
Before submission check the two citations written from memory: `rokicki2014qtm` (QTM God's number 26) and
`ellenberg2011diameter`; arXiv 2112.08602 was not cited (could not verify).

## Next steps (in order)
1. **Finish the Method 7 campaign** (fills the Method 7 column for every puzzle of N = 4..7; ~24 h on 4 cores):
   ```bash
   S=<scratchpad>   # external tools below must be rebuilt in a fresh container
   export KMCODERS_DIR=$S/kmc/repo NXOPT=$S/ext/cube20src/nxopt13q NXOPT_DIR=$S/ext/nxopt_run
   nohup python experiments/m7_santa.py 5 6 7 4 > m7_campaign.log 2>&1 &
   ```
   Resumable: skips puzzles already in `docs/results/santa/m7_full_N*.jsonl` and the earlier subset files. Then
   `python paper/scripts/make_tables.py`, rebuild the paper, commit (`docs/results` and `paper/results` need `git add -f`;
   copy `docs/results/santa/*.jsonl` to `paper/results/santa/`).
2. Equal-compute control: their program with k random seeds (`kmcoders` solver kwarg `seed`) vs k symmetric copies.
3. One run of KMCoders' program at N = 10 (`rubiks.bench run --solver kmcoders --dataset santa --N 10`), killed by a
   container restart.

## External tools (not in the repo; rebuild in a new container)
* KMCoders: `git clone https://github.com/wata-orz/santa2023_permutation_puzzle`, apply
  `rubiks/solvers/external/kmcoders_env.patch`, `cargo build --release --bin solve_cube --bin solve_cube_beam` in
  `solution/`; set `KMCODERS_DIR` to the checkout. (No licence file upstream: do not commit their code.)
* nxopt: `git clone https://github.com/rokicki/cube20src`; `ctangle nxopt.w; ctangle cubepos.w`;
  `g++ -O3 -march=native -std=c++11 -DPS=16 -DQUARTER -DTHREADS -DEO1 -DEP3 -o nxopt13q nxopt.cpp cubepos.cpp -lpthread`;
  first run builds its 1.42 GB table in `NXOPT_DIR` (~1 min).
* dwalton76 (only for re-running B3): see `rubiks/solvers/external/dwalton76.py`.

## Conventions the user set
* Commit directly to `main` as `Gaurav Rai <gauravrai2114@gmail.com>`, no co-author or session trailers.
* Shaurya's MIT LICENSE commit may stay in history.
* Report results honestly; Method 7 "beats" means on average / in total, not on every puzzle.
