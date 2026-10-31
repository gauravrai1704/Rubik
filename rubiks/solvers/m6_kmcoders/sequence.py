"""A move sequence that supports insertion at any time (the core of the KMCoders method).

Moves are quarter turns (QTM primitives; a half turn is stored as two). For a sequence a_0 .. a_{T-1},
``suf[t]`` is the permutation of the suffix a_t .. a_{T-1} (engine convention: applying it is
``state[suf[t]]``) and ``inv[t]`` its inverse. Inserting a sequence B at time t changes the final state
F into F[C] with

    C = inv[t] ∘ B ∘ suf[t]        (as index arrays: C = inv[t][B[suf[t]]])

so a move sequence that acts on a few stickers at time t acts on the stickers ``inv[t][support]`` of the
final state. Scoring a candidate at every time t is therefore a gather over ``inv``, with no replay.
"""
from __future__ import annotations

import numpy as np

from ...engine.puzzle import Move, PuzzleSpec


def quarter_turns(puzzle: PuzzleSpec, moves) -> list[int]:
    """Engine moves -> quarter-turn move ids (a half turn becomes two quarter turns)."""
    idx = puzzle.move_index
    out = []
    for m in moves:
        m = puzzle.moves[m] if isinstance(m, (int, np.integer)) else m
        if m.turn == 2:
            q = idx[Move(m.axis, m.layer, 1)]
            out += [q, q]
        else:
            out.append(idx[m])
    return out


class InsertionSequence:
    def __init__(self, puzzle: PuzzleSpec, start: np.ndarray, moves: list[int] | None = None,
                 dtype=np.int32):
        self.P = puzzle
        self.start = np.asarray(start).copy()
        S = puzzle.num_stickers
        self.perms = puzzle.move_perms.astype(dtype)
        self.invperms = np.argsort(self.perms, axis=1).astype(dtype)
        self.inverse_move = puzzle.inverse_move
        self.moves: list[int] = []
        self.suf = np.arange(S, dtype=dtype)[None].copy()
        self.inv = self.suf.copy()
        if moves:
            self.insert(0, list(moves))

    # ---------------------------------------------------------------- state
    @property
    def T(self) -> int:
        return len(self.moves)

    def final(self) -> np.ndarray:
        return self.start[self.prefix(self.T)]

    def prefix(self, t: int) -> np.ndarray:
        """Permutation of a_0 .. a_{t-1} (recomputed; used rarely)."""
        q = np.arange(self.P.num_stickers)
        for m in self.moves[:t]:
            q = q[self.perms[m]]
        return q

    # ---------------------------------------------------------------- insertion
    def insert(self, t: int, seq: list[int]) -> None:
        """Insert quarter-turn moves ``seq`` before move t (t = T appends)."""
        if not seq:
            return
        L = len(seq)
        self.moves[t:t] = list(seq)
        old_suf, old_inv = self.suf, self.inv
        T = len(self.moves)
        S = old_suf.shape[1]
        suf = np.empty((T + 1, S), dtype=old_suf.dtype)
        inv = np.empty_like(suf)
        suf[t + L:] = old_suf[t:]
        inv[t + L:] = old_inv[t:]
        for tau in range(t + L - 1, -1, -1):  # Suf_tau = P_{a_tau}[Suf_{tau+1}]
            m = self.moves[tau]
            suf[tau] = self.perms[m][suf[tau + 1]]
            inv[tau] = inv[tau + 1][self.invperms[m]]
        self.suf, self.inv = suf, inv

    def cancel_cost(self, t: int, seq_first: np.ndarray, seq_last: np.ndarray, length: np.ndarray) -> np.ndarray:
        """QTM cost of inserting sequences (first move, last move, length; arrays) at time t, net of
        the immediate cancellations a_{t-1} . first = id and last . a_t = id."""
        cost = length.astype(np.int64).copy()
        if t > 0:
            cost -= 2 * (self.inverse_move[seq_first] == self.moves[t - 1])
        if t < self.T:
            cost -= 2 * (self.inverse_move[seq_last] == self.moves[t])
        return cost
