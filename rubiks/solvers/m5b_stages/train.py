"""Train Method 5B's per-cluster heuristics.

One model per cluster *signature*: a model trained on a 4×4×4 x-center cluster is reused
for every x-center cluster at every N. Clusters with an exact table (all Square clusters,
fixed centers) need no training.

    python -m rubiks.solvers.m5b_stages.train --puzzle cube --N 3-7 --steps 3000 --device cuda
"""
from __future__ import annotations

import argparse

from ...bench.__main__ import parse_range
from ...engine import clusters, make
from .cluster_models import CKPT_DIR, ModelBank, save_mlp, table_for, train_cluster_mlp


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzle", default="cube", choices=["cube", "square"])
    ap.add_argument("--N", default="3-7")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--walks", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true", help="retrain even if a checkpoint exists")
    a = ap.parse_args(argv)
    bank = ModelBank(device="cpu", allow_missing=True)
    done = set()
    for N in parse_range(a.N):
        P = make(a.puzzle, N)
        for c in clusters(P):
            if c.signature in done:
                continue
            done.add(c.signature)
            if table_for(c) is not None:
                print(f"{P} {c.kind:15s} K={c.size:3d}: exact table, no training needed")
                continue
            if not a.force and not bank.needs_training(c):
                print(f"{P} {c.kind:15s} K={c.size:3d}: exact table or checkpoint exists, skipping")
                continue
            print(f"{P} {c.kind:15s} K={c.size:3d}: training ({a.steps} steps)")
            model, meta = train_cluster_mlp(P, c, steps=a.steps, batch=a.batch, walks=a.walks, kmax=a.kmax,
                                            device=a.device, hidden=a.hidden, blocks=a.blocks)
            print("   saved", save_mlp(model.cpu(), meta, CKPT_DIR))


if __name__ == "__main__":
    main()
