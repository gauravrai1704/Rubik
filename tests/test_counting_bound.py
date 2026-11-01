"""Counting lower bound: closed form vs exact word counts, and below every known optimum."""
import math

from rubiks.analysis.counting_bound import (bound_cube_qtm, bound_square, cube_block_coeffs, exact_word_bound,
                                            log_states_cube, square_block_coeffs)


def test_cube_state_count_3x3x3():
    # 8! 3^7 12! 2^11 = 2 * 43,252,003,274,489,856,000 (no parity coupling; the true count is half)
    assert abs(log_states_cube(3) - math.log(2 * 43252003274489856000)) < 1e-6


def test_closed_form_matches_exact_word_count():
    for N in (3, 4, 10):
        exact = exact_word_bound(log_states_cube(N), cube_block_coeffs(N), 3)
        assert abs(exact - bound_cube_qtm(N)) <= 1.5


def test_square_bound_below_gods_numbers():
    # exact God's numbers from Method 1's BFS for the colour goal (any of the 4 solved colourings): 6, 10, 12;
    # with 4 goal states, words from the goals must cover states / 4
    for N, states, gods in ((3, 768, 6), (4, 82944, 10), (5, 10616832, 12)):
        b = exact_word_bound(math.log(states / 4), square_block_coeffs(N), 2)
        assert b <= gods and abs(b - bound_square(N, states / 4)) <= 1


def test_3x3x3_qtm_bound_below_known_qtm_distance():
    assert bound_cube_qtm(3) < 26  # QTM God's number of the 3x3x3 (with slice turns the true value is lower)
