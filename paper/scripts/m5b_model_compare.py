"""Method 5B cluster networks: CPU-trained (checkpoints/m5b) vs GPU-trained (checkpoints/m5b_gpu).

For every cluster type: size, training steps and time, final training loss, and predictor quality on
fresh random walks of the projected cluster (seeded, 400 walks to the type's kmax): Spearman rank
correlation of the network's estimate with the walk depth, and its mean excess over the exact coarse-table
lower bound on states at depth <= 5. Writes paper/results/ablations/m5b_models.json.

    python paper/scripts/m5b_model_compare.py
"""
import json
from pathlib import Path

import numpy as np
import torch

from rubiks.engine import clusters, cube
from rubiks.engine.random_states import random_walks
from rubiks.solvers.m5b_stages.cluster_models import (CKPT_DIR, DEFAULT_KMAX, CoarseTableModel, _local_axis_layer,
                                                       coarse_tables, load_mlp)

ROOTS = {"cpu": CKPT_DIR, "gpu": CKPT_DIR.parent / "m5b_gpu"}


def rank(v):
    return torch.argsort(torch.argsort(v)).float()


out = {}
seen = set()
for N in (3, 4, 5, 6):
    P = cube(N)
    for c in clusters(P):
        if c.signature in seen or c.kind == "center_fixed":
            continue
        seen.add(c.signature)
        kmax = DEFAULT_KMAX.get(c.kind, 30)
        perms = torch.as_tensor(c.gen_perms, dtype=torch.long)
        solved = torch.as_tensor(c.solved.astype(np.int64))
        gen = torch.Generator().manual_seed(123)
        X, Y = [], []
        for st, k in random_walks(perms, solved, 400, kmax, gen, _local_axis_layer(P, c)):
            X.append(st.clone())
            Y.append(torch.full((400,), float(k)))
        X, Y = torch.cat(X), torch.cat(Y)
        tabs = [CoarseTableModel(tb, cm, "cpu") for tb, cm in coarse_tables(c)]
        with torch.no_grad():
            lb = torch.stack([t(X).float() for t in tabs]).max(0).values if tabs else torch.zeros(len(X))
        row = {"kind": c.kind, "K": c.size}
        for tag, root in ROOTS.items():
            meta = json.loads((root / f"{c.kind}_{c.signature}.json").read_text())
            m = load_mlp(c.signature, c.kind, "cpu", root)
            with torch.no_grad():
                h = m(X).float()
            near = Y <= 5
            row[tag] = {"hidden": meta["hidden"], "blocks": meta["blocks"], "steps": meta["steps"],
                        "params": sum(p.numel() for p in m.parameters()),
                        "train_time_s": meta["log"][-1]["time_s"], "final_loss": meta["log"][-1]["loss"],
                        "spearman": float(torch.corrcoef(torch.stack([rank(h), rank(Y)]))[0, 1]),
                        "excess_near_goal": float((h[near] - lb[near]).mean())}
        out[c.kind] = row
        print(c.kind, {t: (round(row[t]["spearman"], 3), round(row[t]["excess_near_goal"], 2)) for t in ROOTS})
Path("paper/results/ablations/m5b_models.json").write_text(json.dumps(out, indent=1))
