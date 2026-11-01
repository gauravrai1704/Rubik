# Method 4 (CayleyPy) Square models trained on a Colab T4 ("large" preset)

Trained with `notebooks/train_gpu_colab.ipynb, `--preset large --amp`, 256 epochs of ~1M examples
(N = 3..8; 1.27M parameters at N = 8, about 7 minutes per model on a T4). The weights
(`square<N>.large.pt`, 3.8-5.1 MB each) are git-ignored like every `*.large.pt`; they are in the team
Drive folder <https://drive.google.com/drive/folders/11DDgOTqIb2V-Q9HNP6gKW-9QBpIXjoLM> (`m4/`).
This folder keeps their metadata (`*.json`) and training logs (`m4_square*.log`).

Same-hardware comparison with the committed CPU-trained models (`checkpoints/m4/square<N>.pt`), 20 random
states per N, beam width 4096, CPU (`docs/results/m4_square_cpu_vs_gpu_large.jsonl`):

| N | CPU model, mean HTM | GPU large, mean HTM | states shorter / longer with GPU |
|---|---|---|---|
| 3 | 3.75 | 3.75 | 0 / 0 |
| 4 | 6.70 | 6.70 | 0 / 0 |
| 5 | 9.10 | 9.10 | 0 / 0 |
| 6 | 12.60 | 12.60 | 0 / 0 |
| 7 | 15.40 | 15.60 | 0 / 4 |
| 8 | 20.30 | **18.80** | 12 / 0 |

Every state solved by both. Optimal lengths (Method 1) are 3.75 / 6.70 / 8.95 at N = 3 / 4 / 5, so
both models are optimal or within 0.15 there. The larger model only pays off at N = 8 (-7%), so it is
the default there: `checkpoints/m4/square8.large.pt` is committed (5 MB, like `cube3.large.pt`) and
`find_checkpoints` prefers it; for N <= 7 the CPU models stay the default.
`docs/results/m4_square_W4096_colab_t4.jsonl` is the notebook's own evaluation on the T4 (10 states per N).
