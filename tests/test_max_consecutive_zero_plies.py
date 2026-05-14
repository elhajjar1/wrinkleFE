"""Regression tests for ``_max_consecutive_zero_plies`` (issue #14).

The function previously floored its return value to 1, fabricating an
adjacent 0-degree block (and a curved-beam OOP knockdown) for layups
containing no 0-degree plies.  It must now return the true count: 0
when no 0-degree plies are present.
"""

from __future__ import annotations

from wrinklefe.analysis import _max_consecutive_zero_plies


def test_no_zero_plies_returns_zero():
    # [45/-45/90]_s has no 0-degree plies.
    angles = [45.0, -45.0, 90.0, 90.0, -45.0, 45.0]
    assert _max_consecutive_zero_plies(angles) == 0


def test_empty_layup_returns_zero():
    assert _max_consecutive_zero_plies([]) == 0


def test_single_isolated_zero_ply_returns_one():
    angles = [45.0, 0.0, -45.0, 90.0]
    assert _max_consecutive_zero_plies(angles) == 1


def test_three_consecutive_zero_plies_returns_three():
    angles = [45.0, 0.0, 0.0, 0.0, -45.0, 90.0]
    assert _max_consecutive_zero_plies(angles) == 3


def test_two_separated_blocks_returns_max_block():
    # Two pairs of 0-deg plies separated by 45-deg interleaves.
    angles = [0.0, 0.0, 45.0, 0.0, 0.0, 0.0, 45.0]
    assert _max_consecutive_zero_plies(angles) == 3


def test_near_zero_within_tolerance_counts_as_zero():
    # Default tol=5.0 deg; 3 deg is within tolerance.
    angles = [3.0, -2.0, 1.0]
    assert _max_consecutive_zero_plies(angles) == 3


def test_outside_tolerance_not_counted():
    # 10 deg is outside the default 5-deg tolerance.
    angles = [10.0, -10.0, 10.0]
    assert _max_consecutive_zero_plies(angles) == 0
