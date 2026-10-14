"""Train the Method 5A cluster-token model on several sizes at once (Section 8.4).

Training data: fresh non-backtracking random walks from solved, as in CayleyPy, on every
N in ``--train-N``. Target: number of random steps k. Loss: MSE of (prediction - k) /
(N²/ln N), so every N contributes on the same scale. Augmentation needs no extra code:
walks from a single solved state already cover all orientations.

    # go / no-go test (Section 8.6): Squares, train 4-8, check 10, 14, 20
    python -m rubiks.solvers.m5a_tokens.train --puzzle square --train-N 4-8 --val-N 10,14,20 \
        --steps 20000 --out checkpoints/m5a/square.pt --device cuda

    # cube, all cluster types (oblique centers need N >= 6)
    python -m rubiks.solvers.m5a_tokens.train --puzzle cube --train-N 3-7 --val-N 8,10 \
        --steps 50000 --move-head --out checkpoints/m5a/cube.pt --device cuda
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from ...bench.__main__ import parse_range
from ...engine import make
from ...engine.random_states import random_walks
from .model import ClusterTokenModel, ModelConfig, Tokenizer, distance_scale, save, type_sizes_for


def kmax_for(puzzle) -> int:
    """Random-walk horizon: CayleyPy's values for 3-5 (26, 45, 65), else ~4.2·N²/ln N."""
    N = puzzle.N
    if puzzle.kind == "cube":
        return {3: 26, 4: 45, 5: 65}.get(N, round(4.2 * distance_scale(N)))
    return round(2.5 * distance_scale(N)) + 4


class WalkSampler:
    def __init__(self, puzzle, device, canonical: bool = False):
        self.p = puzzle
        self.dev = torch.device(device)
        self.perms = torch.as_tensor(puzzle.move_perms, dtype=torch.long, device=self.dev)
        self.solved = torch.as_tensor(puzzle.solved, device=self.dev)
        self.al = torch.as_tensor([[m.axis, m.layer] for m in puzzle.moves], device=self.dev)
        self.kmax = kmax_for(puzzle)
        self.canonical = canonical

    def sample(self, walks: int, batch: int, gen) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for st, k in random_walks(self.perms, self.solved, walks, self.kmax, gen, self.al, self.canonical):
            xs.append(st)
            ys.append(torch.full((walks,), float(k), device=self.dev))
        X = torch.cat(xs + [self.solved[None].expand(max(1, walks // 8), -1)])
        Y = torch.cat(ys + [torch.zeros(max(1, walks // 8), device=self.dev)])
        idx = torch.randint(0, len(X), (batch,), device=self.dev, generator=gen)
        return X[idx], Y[idx]


@torch.no_grad()
def evaluate(model, tok, sampler, gen, n: int = 1024) -> dict:
    model.eval()
    X, Y = sampler.sample(max(32, n // sampler.kmax), n, gen)
    v, _ = model(X, tok)
    model.train()
    err = (v - Y) / tok.scale
    corr = float(torch.corrcoef(torch.stack([v, Y]))[0, 1]) if len(v) > 2 else float("nan")
    return {"N": tok.puzzle.N, "rmse_norm": float(err.pow(2).mean().sqrt()), "corr": corr}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzle", choices=["cube", "square"], default="square")
    ap.add_argument("--train-N", default="4-8")
    ap.add_argument("--val-N", default="10")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--walks", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--move-head", action="store_true", help="also train the Q*-style move head (phase 2)")
    ap.add_argument("--move-steps", type=int, default=None, help="phase-2 steps (default steps/4)")
    ap.add_argument("--canonical-walks", action="store_true", help="ablation E8")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default="checkpoints/m5a/model.pt")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    train_P = [make(a.puzzle, N) for N in parse_range(a.train_N)]
    val_P = [make(a.puzzle, N) for N in parse_range(a.val_N)] if a.val_N else []
    cfg = ModelConfig(d_model=a.d_model, layers=a.layers, heads=a.heads, move_head=a.move_head)
    model = ClusterTokenModel(cfg, type_sizes_for(train_P)).to(dev)
    toks = {p.N: Tokenizer(p, a.device) for p in train_P + val_P}
    samplers = {p.N: WalkSampler(p, a.device, a.canonical_walks) for p in train_P + val_P}
    for p in val_P:
        model.check_supported(toks[p.N])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    log = []
    t0 = time.time()
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M; train N={[p.N for p in train_P]}")
    for it in range(1, a.steps + 1):
        p = train_P[int(torch.randint(0, len(train_P), (1,), generator=torch.Generator().manual_seed(a.seed + it)))]
        X, Y = samplers[p.N].sample(a.walks, a.batch, gen)
        tok = toks[p.N]
        v, _ = model(X, tok)
        loss = nn.functional.mse_loss(v / tok.scale, Y / tok.scale)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % a.eval_every == 0 or it == a.steps:
            ev = [evaluate(model, toks[q.N], samplers[q.N], gen) for q in train_P + val_P]
            entry = {"step": it, "loss": loss.item(), "time_s": round(time.time() - t0, 1), "eval": ev}
            log.append(entry)
            print(f"step {it:6d} loss {loss.item():.4f} "
                  + " ".join(f"N{e['N']}:rmse={e['rmse_norm']:.3f},r={e['corr']:.2f}" for e in ev))

    if a.move_head:  # phase 2: distil Q(s, a) = 1 + V(child) from the frozen value head
        for name, prm in model.named_parameters():
            prm.requires_grad = any(name.startswith(k) for k in ("axis_emb", "turn_emb", "layer_mlp", "move_attn", "move_out"))
        steps2 = a.move_steps or max(1, a.steps // 4)
        opt2 = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr)
        for it in range(1, steps2 + 1):
            p = train_P[it % len(train_P)]
            tok, smp = toks[p.N], samplers[p.N]
            X, _ = smp.sample(a.walks, max(16, a.batch // 8), gen)
            with torch.no_grad():
                h = model.encode(X, tok)
                kids = X[:, smp.perms]  # (b, M, S)
                b, M, S = kids.shape
                target, _ = model(kids.reshape(b * M, S), tok)
                target = target.reshape(b, M)
            q = model.move_values(X, tok, h)
            loss = nn.functional.mse_loss(q / tok.scale, target / tok.scale)
            opt2.zero_grad()
            loss.backward()
            opt2.step()
            if it % a.eval_every == 0 or it == steps2:
                print(f"move-head step {it} loss {loss.item():.4f}")
        for prm in model.parameters():
            prm.requires_grad = True

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"puzzle": a.puzzle, "train_N": [p.N for p in train_P], "val_N": [p.N for p in val_P],
            "steps": a.steps, "log": log, "args": vars(a)}
    save(model.cpu(), out, meta)
    (out.with_suffix(".json")).write_text(json.dumps(meta, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
