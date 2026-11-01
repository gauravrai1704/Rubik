"""Counting lower bound on optimal solution length ("how far from optimal is a solution?").

A state at distance d is the endpoint of at least one canonical move word of length d, so if c(d) counts
canonical words of length d, the radius L of the Cayley graph satisfies sum_{d<=L} c(d) >= |G|, and typical
optimal distances are close to log|G| / log b with b the growth rate of c(d) (e.g. on the 3x3x3 this is
within a few moves of the typical optimal distance). Over-counting words keeps the bound valid.

Cube, Santa / QTM metric (every slice quarter turn costs 1, half turns 2). A canonical word is a sequence of
blocks; a block turns a non-empty set of layers of one axis (each layer by +1, -1 or 2), consecutive blocks
use different axes. Per block the cost generating function is f(x) = (1 + x)^(2N) - 1 and the words have
G(x) = 1 + 3f / (1 - 2f), so b = 1 / x* with 2 f(x*) = 1, i.e. x* = (3/2)^(1/(2N)) - 1.

Square (every move a 180-degree flip of a row or a column): rows commute with rows and columns with columns,
blocks alternate between a non-empty set of rows and a non-empty set of columns, f(x) = (1 + x)^N - 1 and
b = 1 / (2^(1/N) - 1).

|G|, colour states (pieces of the same colour are indistinguishable), cluster by cluster: corners 8! 3^7,
middle edges 12! 2^11, each wing orbit 24!, each non-fixed center orbit 24! / (4!)^6. Parity couplings
between clusters and whole-cube orientation are ignored: they change log2|G| by at most about one bit per
cluster, which moves the bound by well under 1% (log2 b is 4-7 bits per move for N = 4-33).
"""
from __future__ import annotations

import collections
import math

from ..engine import cube
from ..engine.orbits import clusters

_lf = math.lgamma


def log_states_cube(N: int) -> float:
    """Natural log of the number of colour states of the NxNxN cube."""
    kinds = collections.Counter(c.kind for c in clusters(cube(N)))
    L = 0.0
    for k, n in kinds.items():
        if k == "corner":
            L += n * (_lf(9) + 7 * math.log(3))
        elif k == "midge":
            L += n * (_lf(13) + 11 * math.log(2))
        elif k == "wing":
            L += n * _lf(25)
        elif k == "center_fixed":
            continue
        else:  # any other center orbit: 24 pieces, 4 of each colour
            L += n * (_lf(25) - 6 * _lf(5))
    return L


def growth_cube_qtm(N: int) -> float:
    return 1.0 / (1.5 ** (1.0 / (2 * N)) - 1.0)


def growth_square(N: int) -> float:
    return 1.0 / (2.0 ** (1.0 / N) - 1.0)


def bound_cube_qtm(N: int) -> float:
    """log|G| / log b: the estimate of the typical optimal QTM distance (a lower bound up to O(1))."""
    return log_states_cube(N) / math.log(growth_cube_qtm(N))


def bound_square(N: int, states: int) -> float:
    """Same for the Rubik's Square, given its number of states (e.g. from Method 1's BFS)."""
    return math.log(states) / math.log(growth_square(N))


def exact_word_bound(log_states: float, f_coeffs: list[int], families: int, max_len: int = 100000) -> int:
    """Exact version of the counting bound: the least L with sum_{d<=L} c(d) >= |G|, for words made of
    blocks with cost-coefficients ``f_coeffs`` (f_coeffs[k] = blocks of cost k) and ``families`` mutually
    non-commuting families (consecutive blocks from different families). Counts are kept as logs."""
    K = len(f_coeffs)
    # c_end[d]: words of cost d ending in a block of a fixed family; total = 1 + families * c_end
    lc = [-math.inf] * (max_len + 1)
    total = 0.0  # log of sum_{d<=L} c(d); the empty word
    for d in range(1, max_len + 1):
        acc = -math.inf
        for k in range(1, min(K - 1, d) + 1):
            if f_coeffs[k] == 0:
                continue
            prev = 0.0 if d == k else (math.log(families - 1) + lc[d - k] if lc[d - k] > -math.inf else -math.inf)
            if prev == -math.inf:
                continue
            term = math.log(f_coeffs[k]) + prev
            acc = term if acc == -math.inf else max(acc, term) + math.log1p(math.exp(-abs(acc - term)))
        lc[d] = acc
        if acc > -math.inf:
            t = math.log(families) + acc
            total = max(total, t) + math.log1p(math.exp(-abs(total - t)))
        if total >= log_states:
            return d
    raise ValueError("max_len too small")


def cube_block_coeffs(N: int) -> list[int]:
    """Coefficients of (1 + x)^(2N) - 1 (per-axis block of slice turns, QTM costs)."""
    return [0] + [math.comb(2 * N, k) for k in range(1, 2 * N + 1)]


def square_block_coeffs(N: int) -> list[int]:
    return [0] + [math.comb(N, k) for k in range(1, N + 1)]
