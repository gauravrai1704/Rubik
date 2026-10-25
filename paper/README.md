# Paper

`main.tex` (acmart, `sigconf,nonacm`) with `results_body.tex`, `references.bib`.
Every table body, figure and in-text number is generated from `paper/results/` by
`scripts/make_tables.py` into `generated/` and `figures/`; nothing is typed by hand.

## Rebuild
```bash
pip install -e . matplotlib
python paper/scripts/make_tables.py
cd paper && latexmk -pdf main.tex
```
Needs TeX Live with `acmart`, `libertine`, `newtx` (Ubuntu: `texlive-publishers texlive-fonts-extra texlive-plain-generic`).

## Re-run the experiments (2-core CPU: a few hours)
```bash
python -m rubiks.bench make --puzzle square --N 3-12 --depth random --count 10
python -m rubiks.bench make --puzzle square --N 13-20 --depth random --count 5
python -m rubiks.bench make --puzzle cube --N 3-8,10 --depth random --count 2
python -m rubiks.bench make --puzzle cube --N 15,20 --depth random --count 1
bash paper/scripts/streamA.sh   # 5B Square sweep + ablations
bash paper/scripts/streamB.sh   # exact Square optima, Method 4 width sweep, Method 5A
# (the paper's run of the last step, 5A at N = 12, 14, was stopped after one N = 12 state timed out)
# cube sweep (N = 3..20): python -m rubiks.bench run --solver m5b_stages --puzzle cube --N 3-8,10,15,20 \
#   --dataset random --time 9000 --kw '{"width": 512, "max_width": 8192}'
```
Scripts write to `results/paper/`; copy the files into `paper/results/` before `make_tables.py`.

## Provenance of the result files
| File | Produced by | Commit recorded |
|---|---|---|
| `m5b_square.jsonl`, `m4_square_W*.jsonl`, `m5a_square*.jsonl`, `ablations/*`, `optimal_square.json` | scripts above, Oct 2026 | `618e560` (main) |
| `m5b_cube.jsonl`, `m5b_cube_15_20.jsonl` | cube sweep, Oct 16 2026 | `a40c343` |

The cube rows record `a40c343` because they ran on the uncommitted working tree that was
committed next as `ce5b06e`. The engine, search, 5A and 5B code and checkpoints are identical
between `ce5b06e` and `618e560`, and the regenerated cube instances match these rows.
