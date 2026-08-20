"""Tests for the controller's pure geometry: angle math, braking, corner turns.

Covers the corner-turn logic extracted from the Controller (regressions for the
max-rate rotation / settle-before-release fixes).
"""

from __future__ import annotations

import math

import pytest

from control.controller import (
    BANG_TURN_DEADBAND_RAD,
    CORNER_TURN_MAX_RPS,
    CORNER_TURN_RADIUS_M,
    _angle_diff,
    _brake_speed_limit,
    corner_ahead,
    near_sharp_corner,
    turn_at_corner,
)


# Path: east along y=8 then a sharp 90-degree north turn at (12,8).
PATH = [(10.0, 8.0), (12.0, 8.0), (12.0, 12.0), (12.0, 16.0)]


def test_angle_diff_wraps_to_pi():
    assert _angle_diff(math.pi / 2, 0.0) == pytest.approx(math.pi / 2)
    assert _angle_diff(0.0, math.pi / 2) == pytest.approx(-math.pi / 2)
    # 3pi/2 - 0 wraps to -pi/2.
    assert _angle_diff(3 * math.pi / 2, 0.0) == pytest.approx(-math.pi / 2)
    # pi vs -pi are the same angle.
    assert _angle_diff(math.pi, -math.pi) == pytest.approx(0.0, abs=1e-9)


def test_brake_speed_limit_basic():
    # Zero remaining distance -> must already be stopped (reaction-limited to 0).
    assert _brake_speed_limit(0.0, 1.0, 0.7) == pytest.approx(0.0, abs=1e-9)
    # Negative distance is clamped.
    assert _brake_speed_limit(-5.0, 1.0, 0.7) == pytest.approx(0.0, abs=1e-9)
    # More room -> higher allowed speed.
    a = _brake_speed_limit(1.0, 1.0, 0.7)
    b = _brake_speed_limit(4.0, 1.0, 0.7)
    assert b > a > 0.0


def test_brake_speed_limit_formula():
    # v·T + v²/2a = d  ->  v = sqrt((aT)² + 2ad) - aT
    d, decel, react = 3.0, 1.5, 0.8
    v = _brake_speed_limit(d, decel, react)
    assert v == pytest.approx(
        math.sqrt((decel * react) ** 2 + 2 * decel * d) - decel * react
    )


def test_corner_ahead_finds_corner():
    assert corner_ahead(PATH, 0) == ((12.0, 8.0), math.pi / 2)
    # After rolling over onto the outgoing segment, the same corner is found at
    # the start of the current segment.
    assert corner_ahead(PATH, 1) == ((12.0, 8.0), math.pi / 2)


def test_corner_ahead_none_when_no_corner():
    straight = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    assert corner_ahead(straight, 0) is None


def test_near_sharp_corner():
    assert near_sharp_corner(PATH, 0, 11.8, 8.0) is True
    assert near_sharp_corner(PATH, 0, 11.0, 8.0) is False  # 1 m away


def test_turn_at_corner_capped_rate():
    """Regression: rotation is capped at CORNER_TURN_MAX_RPS, not the robot's
    full angular limit (which overshoots the heading once pose/drive lag)."""
    v, w = turn_at_corner(PATH, 0, 12.0, 8.0, 0.0, est_omega=0.0)
    assert v == 0.0
    assert w == pytest.approx(CORNER_TURN_MAX_RPS)  # 1.2, not 3.0


def test_turn_at_corner_releases_when_aligned_and_settled():
    result = turn_at_corner(PATH, 0, 12.0, 8.0, math.pi / 2, est_omega=0.0)
    assert result is None  # resume pure pursuit


def test_turn_at_corner_holds_while_spinning():
    """Nearly aligned but still rotating must NOT release (settle condition)."""
    result = turn_at_corner(PATH, 0, 12.0, 8.0, math.pi / 2 - 0.2, est_omega=1.0)
    assert result is not None
    assert result[0] == 0.0  # still holding v = 0
    assert abs(result[1]) < CORNER_TURN_MAX_RPS


def test_turn_at_corner_not_parked():
    assert turn_at_corner(PATH, 0, 11.0, 8.0, 0.0, est_omega=0.0) is None


def test_turn_at_corner_bang_deadband_release():
    """Within the deadband and settled -> release even if not exactly aligned."""
    result = turn_at_corner(
        PATH, 0, 12.0, 8.0, math.pi / 2 - BANG_TURN_DEADBAND_RAD / 2, est_omega=0.0
    )
    assert result is None


def test_turn_at_corner_not_inside_radius():
    # 0.6 m away from the corner waypoint, beyond CORNER_TURN_RADIUS_M.
    assert turn_at_corner(PATH, 0, 12.0, 8.6, 0.0, est_omega=0.0) is None
