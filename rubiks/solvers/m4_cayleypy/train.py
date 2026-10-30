"""Train one Method 4 ResMLP for one puzzle size (technical basis, Section 7.1-7.3).

As in the paper: every epoch draws a fresh set of non-backtracking random walks from
solved (``--epoch-size`` examples, 1M in the paper), the target is the step count k, MSE
loss, Adam at a fixed 1e-3, FP32 training. Training time and peak GPU memory are written
to the checkpoint's ``.json`` (Section 7.4).

    # paper settings, 3x3x3 (8B examples = 8192 epochs of 1M; on one GPU, hours)
    python -m rubiks.solvers.m4_cayleypy.train --puzzle cube --N 3 --preset paper3 \
        --epochs 8192 --device cuda --out checkpoints/m4/cube3.large.pt

    # per-N Square baselines for 5A, CPU-sized (what is committed under checkpoints/m4/)
    for N in 3 4 5 6 7 8; do
      python -m rubiks.solvers.m4_cayleypy.train --puzzle square --N $N --preset small \
          --epochs 64 --epoch-size 100000 --out checkpoints/m4/square$N.pt
    done
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from ...engine import make
from .model import PRESETS, ResMLP, ResMLPConfig, WalkGenerator, kmax_for, move_set, num_params, one_hot, save


@torch.no_grad()
def evaluate(model: ResMLP, gen_walks: WalkGenerator, gen: torch.Generator, n_walks: int = 256) -> dict:
    """Fit on fresh walks: RMSE in steps and correlation of prediction with k."""
    model.eval()
    X, Y = gen_walks.walks(n_walks, gen)
    v = torch.cat([model(one_hot(X[i : i + 8192])) for i in range(0, len(X), 8192)])
    model.train()
    return {"rmse": float((v - Y).pow(2).mean().sqrt()), "corr": float(torch.corrcoef(torch.stack([v, Y]))[0, 1])}


def train(puzzle: str, N: int, preset: str = "small", n1: int | None = None, n2: int | None = None,
          nr: int | None = None, epochs: int = 64, epoch_size: int = 1_000_000, batch: int = 1024,
          lr: float = 1e-3, kmax: int | None = None, moves: str = "paper", device: str = "cpu", seed: int = 0,
          out: str | Path | None = None, log_every: int = 1, verbose: bool = True, save_every: int = 0,
          state: str | Path | None = None, amp: bool = False, stop_after: int = 0) -> tuple[ResMLP, dict]:
    """``save_every`` > 0 writes a resumable state every that many epochs to ``state`` (default
    <out>.state); an existing state is resumed. ``amp``: FP16 autocast for training (the paper
    trains in FP32; on a T4 this is ~2-3x faster). ``stop_after`` > 0: train at most that many
    epochs in this call, save the state and return early (meta has ``stopped_at``)."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    gen = torch.Generator(device=dev).manual_seed(seed)
    P = make(puzzle, N)
    subset = move_set(P, moves)
    kmax = kmax or kmax_for(P)
    walker = WalkGenerator(P, subset, kmax, dev)
    p1, p2, pr = PRESETS[preset]
    cfg = ResMLPConfig(input_size=6 * P.num_stickers, n1=n1 or p1, n2=n2 or p2, nr=pr if nr is None else nr)
    model = ResMLP(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    use_amp = amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n_walks = max(1, epoch_size // kmax)
    state_path = Path(state) if state else (Path(out).with_suffix(".state") if out else None)
    first, log, seen, prior_time = 1, [], 0, 0.0
    if state_path is not None and state_path.exists():  # resume (e.g. after a Colab disconnect)
        st = torch.load(state_path, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        gen.set_state(st["gen"].cpu())
        scaler.load_state_dict(st["scaler"])
        first, log, seen, prior_time = st["epoch"] + 1, st["log"], st["examples"], st["time_s"]
        if verbose:
            print(f"resumed from {state_path} at epoch {st['epoch']}")
    if verbose:
        print(f"{P}: {num_params(model) / 1e6:.2f}M params, {len(subset)} moves, kmax {kmax}, "
              f"{n_walks * kmax} examples/epoch, {epochs} epochs")
    t0 = time.time() - prior_time
    for ep in range(first, epochs + 1):
        X, Y = walker.walks(n_walks, gen)
        order = torch.randperm(len(X), device=dev, generator=gen)
        total = 0.0
        for i in range(0, len(X), batch):
            idx = order[i : i + batch]
            if len(idx) < 2:  # BatchNorm needs two rows
                continue
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = nn.functional.mse_loss(model(one_hot(X[idx])).float(), Y[idx])
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(idx)
        seen += len(X)
        if ep % log_every == 0 or ep == epochs:
            ev = evaluate(model, walker, gen)
            entry = {"epoch": ep, "examples": seen, "train_mse": total / len(X), **ev,
                     "time_s": round(time.time() - t0, 1)}
            log.append(entry)
            if verbose:
                print(f"epoch {ep:5d} mse {entry['train_mse']:.3f} rmse {ev['rmse']:.3f} r {ev['corr']:.3f} "
                      f"{entry['time_s']:.0f}s", flush=True)
        if state_path is not None and save_every and (ep % save_every == 0 or ep == epochs):
            state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = state_path.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(),
                        "scaler": scaler.state_dict(), "epoch": ep, "log": log, "examples": seen,
                        "time_s": time.time() - t0}, tmp)
            tmp.replace(state_path)  # atomic: a disconnect mid-write never corrupts the state
        if stop_after and ep - first + 1 >= stop_after and ep < epochs:
            if verbose:
                print(f"stopping after {stop_after} epochs of this session (epoch {ep}); rerun to resume")
            return model.eval(), {"stopped_at": ep, "log": log}
    meta = {
        "puzzle": puzzle, "N": N, "moves": moves, "move_ids": subset, "kmax": kmax, "cfg": cfg.__dict__,
        "params": num_params(model), "epochs": epochs, "epoch_size": n_walks * kmax, "examples": seen,
        "batch": batch, "lr": lr, "seed": seed, "device": str(dev), "amp": use_amp,
        "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        "train_time_s": round(time.time() - t0, 1),
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated(dev) / 2**20, 1) if dev.type == "cuda" else 0.0,
        "log": log,
    }
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        save(model.cpu(), out, meta)
        out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        model.to(dev)
        if verbose:
            print(f"saved {out}")
    return model.eval(), meta


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzle", choices=["cube", "square"], default="square")
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="small")
    ap.add_argument("--n1", type=int, default=None)
    ap.add_argument("--n2", type=int, default=None)
    ap.add_argument("--nr", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=64)
    ap.add_argument("--epoch-size", type=int, default=1_000_000, help="fresh examples per epoch (paper: 1M)")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kmax", type=int, default=None, help="walk length (paper: 26 / 45 / 65 for cubes 3-5)")
    ap.add_argument("--moves", choices=["paper", "all"], default="paper")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-every", type=int, default=0,
                    help="write a resumable training state every this many epochs (0: never)")
    ap.add_argument("--state", default=None,
                    help="training-state file (default <out>.state); training resumes from it if it exists")
    ap.add_argument("--amp", action="store_true", help="FP16 autocast training on GPU (paper: FP32)")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="train at most this many epochs in this session, save the state and exit (0: no limit)")
    a = ap.parse_args(argv)
    train(a.puzzle, a.N, a.preset, a.n1, a.n2, a.nr, a.epochs, a.epoch_size, a.batch, a.lr, a.kmax, a.moves,
          a.device, a.seed, a.out, a.log_every, save_every=a.save_every, state=a.state, amp=a.amp,
          stop_after=a.stop_after)


if __name__ == "__main__":
    main()
