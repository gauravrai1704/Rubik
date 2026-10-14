"""Solvability checker property tests (Section 3.3)."""
import pytest

from rubiks.engine import cube, is_reachable, random_state, square
from rubiks.engine.checker import _cubie_groups, _corner_order, _midge_order


@pytest.mark.parametrize("P", [cube(3), cube(4), cube(5), cube(6), cube(7), square(3), square(4), square(5),
                               square(6), square(7)], ids=repr)
def test_accepts_scrambles(P, rng):
    for _ in range(25):
        ok, why = is_reachable(P, random_state(P, rng, length=150))
        assert ok, why


@pytest.mark.parametrize("N", [3, 4, 5, 6])
def test_rejects_single_corner_twist(N):
    P = cube(N)
    s = P.solved.copy()
    corner = _corner_order(P, _cubie_groups(P.dims)["corner"][0])
    a, b, c = corner
    s[a], s[b], s[c] = P.solved[c], P.solved[a], P.solved[b]
    assert not is_reachable(P, s)[0]


@pytest.mark.parametrize("N", [3, 5, 7])
def test_rejects_single_midge_flip(N):
    P = cube(N)
    s = P.solved.copy()
    a, b = _midge_order(P, _cubie_groups(P.dims)["midge"][0])
    s[a], s[b] = P.solved[b], P.solved[a]
    assert not is_reachable(P, s)[0]


@pytest.mark.parametrize("N", [3, 5])
def test_rejects_two_corner_swap_odd(N):
    """Odd N: corner parity must equal middle-edge parity."""
    P = cube(N)
    g = _cubie_groups(P.dims)["corner"]
    s = P.solved.copy()
    # swap two corners that share the U face, keeping each one's orientation reference
    o1, o2 = _corner_order(P, g[0]), _corner_order(P, g[1])
    for x, y in zip(o1, o2):
        s[x], s[y] = P.solved[y], P.solved[x]
    assert not is_reachable(P, s)[0]


def test_rejects_wrong_counts():
    P = square(5)
    s = P.solved.copy()
    s[0] = (s[0] + 1) % 6
    assert not is_reachable(P, s)[0]
