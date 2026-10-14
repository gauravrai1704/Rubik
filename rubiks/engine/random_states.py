"""Scrambles and random states (technical basis, Section 3.4).

* :func:`scramble` — a random canonical move sequence of a given length (benchmarks at
  fixed scramble depth).
* :func:`random_state` — an approximately uniform random reachable state, by a long
  random canonical walk (default length 25·M, M = number of moves, never below 400).
  Random walks on these Cayley graphs mix quickly; tests check per-cluster statistics.
* :func:`random_walks` — vectorised (torch) non-backtracking walks from solved, the
  training-data generator for Methods 4, 5A and 5B.
"""
from __future__ import annotations

import numpy as np

from .moves import canonical_successor_mask
from .puzzle import Move, PuzzleSpec


def scramble(puzzle: PuzzleSpec, length: int, rng: np.random.Generator) -> list[Move]:
    """Random sequence that is already in canonical form (no redundant commuting orders)."""
    allowed = canonical_successor_mask(puzzle)
    last = len(puzzle.moves)
    out = []
    for _ in range(length):
        choices = np.nonzero(allowed[last])[0]
        mi = int(rng.choice(choices))
        out.append(puzzle.moves[mi])
        last = mi
    return out


def random_state(puzzle: PuzzleSpec, rng: np.random.Generator, length: int | None = None) -> np.ndarray:
    if length is None:
        length = max(400, 25 * len(puzzle.moves))
    return puzzle.apply(puzzle.solved, scramble(puzzle, length, rng))


def random_walks(
    perms,  # torch.LongTensor (M, K): local move permutations
    solved,  # torch tensor (K,)
    batch: int,
    length: int,
    generator=None,
    axis_layer=None,  # optional (M, 2) tensor of (axis, layer) for non-backtracking
    canonical: bool = False,
):
    """Yield (states (batch, K), step) for step = 1..length, all walks in parallel.

    Non-backtracking: the next move never acts on the same (axis, layer) as the last one,
    so a walk never undoes itself immediately. ``canonical=True`` additionally forbids
    lower layers on the same axis (removes commuting duplicates; ablation E8)."""
    import torch

    M, K = perms.shape
    dev = perms.device
    states = solved.to(dev).unsqueeze(0).expand(batch, K).clone()
    if axis_layer is None:
        last_bad = None
    else:
        ax, ly = axis_layer[:, 0].to(dev), axis_layer[:, 1].to(dev)
        same = ax[:, None] == ax[None, :]
        bad = same & ((ly[:, None] >= ly[None, :]) if canonical else (ly[:, None] == ly[None, :]))
        last_bad = bad  # (M_last, M_next)
    last = None
    for step in range(1, length + 1):
        if last is None or last_bad is None:
            mv = torch.randint(0, M, (batch,), device=dev, generator=generator)
        else:
            # uniform among allowed moves: argmax of random keys over the allowed set
            keys = torch.rand((batch, M), device=dev, generator=generator)
            mv = keys.masked_fill(last_bad[last], -1.0).argmax(1)
        states = torch.gather(states, 1, perms[mv])
        last = mv
        yield states, step
