"""Tests for differential-drive kinematics and the square footprint."""

from __future__ import annotations

import math

import pytest

from simulation.kinematics import (
    BOT_RADIUS_M,
    BOT_SIZE_M,
    WHEEL_RADIUS_M,
    WHEEL_TRACK_M,
    square_footprint_radius,
    unicycle_to_wheel,
    wheel_to_unicycle,
)


def test_unicycle_wheel_round_trip():
    for linear, angular in [(0.0, 0.0), (1.0, 0.0), (0.0, 2.0), (0.8, -0.5), (3.0, 1.2)]:
        l, r = unicycle_to_wheel(linear, angular)
        lin2, ang2 = wheel_to_unicycle(l, r)
        assert lin2 == pytest.approx(linear, abs=1e-9)
        assert ang2 == pytest.approx(angular, abs=1e-9)


def test_pure_translation_same_wheel_speed():
    l, r = unicycle_to_wheel(1.0, 0.0)
    assert l == pytest.approx(r)
    assert l == pytest.approx(1.0 / WHEEL_RADIUS_M)


def test_pure_rotation_opposite_wheels():
    l, r = unicycle_to_wheel(0.0, 2.0)
    # v = ± angular * track / 2, then / wheel radius
    assert l == pytest.approx(-2.0 * WHEEL_TRACK_M / 2.0 / WHEEL_RADIUS_M)
    assert r == pytest.approx(2.0 * WHEEL_TRACK_M / 2.0 / WHEEL_RADIUS_M)


def test_track_matches_footprint_and_wheels():
    # Wheels sit flush with the footprint edges, inset by the wheel radius.
    assert WHEEL_TRACK_M == pytest.approx(BOT_SIZE_M - 2.0 * WHEEL_RADIUS_M)


def test_square_footprint_radius_bounds():
    # Smallest toward face centres (half side), largest toward the corners.
    assert square_footprint_radius(0.0) == pytest.approx(BOT_RADIUS_M)
    assert square_footprint_radius(math.pi / 2) == pytest.approx(BOT_RADIUS_M)
    assert square_footprint_radius(math.pi / 4) == pytest.approx(
        BOT_RADIUS_M * math.sqrt(2.0)
    )
    # Periodic in pi/2.
    assert square_footprint_radius(math.pi / 4 + math.pi / 2) == pytest.approx(
        square_footprint_radius(math.pi / 4)
    )


def test_square_footprint_radius_never_exceeds_circumradius():
    for i in range(361):
        r = square_footprint_radius(math.radians(i))
        assert BOT_RADIUS_M <= r <= BOT_RADIUS_M * math.sqrt(2.0) + 1e-9
