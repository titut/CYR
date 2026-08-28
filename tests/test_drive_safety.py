"""Tests for the drive node's velocity loop and safety zones (T-016)."""

from __future__ import annotations

import math

import pytest

from control.drive import (
    _ESTOP_CLEARANCE_M,
    _SLOW_DOWN_CLEARANCE_M,
    _STOP_CLEARANCE_M,
    Drive,
)
from core.sim_drivers import _velocity_step
from simulation.kinematics import square_footprint_radius

_safety_limit = Drive._safety_limited_linear
_safety_ang_limit = Drive._safety_limited_angular
_directional = Drive._directional_clearance
_heading_cap = Drive._heading_speed_cap
_estop_action = Drive._estop_threshold_action


# ---------------------------------------------------------------------------
# Velocity loop
# ---------------------------------------------------------------------------


def test_velocity_step_accel_capped():
    # Large step ramps at max_accel * dt.
    out = _velocity_step(0.0, 100.0, 0.02, max_accel_rps2=35.0, max_decel_rps2=70.0)
    assert out == pytest.approx(35.0 * 0.02)


def test_velocity_step_decel_capped():
    # Braking uses the (higher) decel cap.
    out = _velocity_step(100.0, 0.0, 0.02, max_accel_rps2=35.0, max_decel_rps2=70.0)
    assert out == pytest.approx(100.0 - 70.0 * 0.02)


def test_velocity_step_small_change_tracks():
    # Small step follows the first-order response (alpha = dt / time_constant).
    out = _velocity_step(1.0, 1.1, 0.02, time_constant_s=0.05)
    assert out == pytest.approx(1.0 + 0.4 * 0.1)


# ---------------------------------------------------------------------------
# Safety zones (T-016)
# ---------------------------------------------------------------------------


def test_safety_zone_full_speed_far_away():
    assert _safety_limit(3.0, _SLOW_DOWN_CLEARANCE_M + 1.0) == 3.0


def test_safety_zone_scales_in_slow_down():
    # Halfway through the slow-down zone -> roughly half the max speed.
    mid = (_SLOW_DOWN_CLEARANCE_M + _STOP_CLEARANCE_M) / 2.0
    limit = 3.0 * (mid - _STOP_CLEARANCE_M) / (_SLOW_DOWN_CLEARANCE_M - _STOP_CLEARANCE_M)
    assert _safety_limit(3.0, mid) == pytest.approx(limit)


def test_safety_zone_stop_blocks_forward():
    assert _safety_limit(3.0, _STOP_CLEARANCE_M) == 0.0
    assert _safety_limit(3.0, _ESTOP_CLEARANCE_M) == 0.0


def test_safety_zone_stop_allows_slow_reverse():
    # Inside the stop zone the robot may still back away slowly.
    assert _safety_limit(-1.0, _ESTOP_CLEARANCE_M) == pytest.approx(-0.5)


def test_safety_zone_slow_reverse_at_boundary():
    # Below the slow-down zone, a small reverse command is limited by the zone.
    got = _safety_limit(-0.1, 0.6)
    assert got <= 0.0
    assert got > -0.5


def test_safety_zone_pass_through_when_command_small():
    # A command well below the zone limit passes through unchanged.
    assert _safety_limit(0.2, 0.6) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Angular safety cap (T-016)
# ---------------------------------------------------------------------------


def test_safety_angular_full_rate_far_away():
    assert _safety_ang_limit(3.0, _SLOW_DOWN_CLEARANCE_M + 1.0) == 3.0


def test_safety_angular_scales_in_slow_down():
    # Halfway through the slow-down zone -> roughly half the commanded rate.
    mid = (_SLOW_DOWN_CLEARANCE_M + _STOP_CLEARANCE_M) / 2.0
    scale = (mid - _STOP_CLEARANCE_M) / (_SLOW_DOWN_CLEARANCE_M - _STOP_CLEARANCE_M)
    assert _safety_ang_limit(3.0, mid) == pytest.approx(3.0 * scale)


def test_safety_angular_zero_in_stop_zone():
    # Rotation is halted inside the stop zone (turning could swing into a wall).
    assert _safety_ang_limit(3.0, _STOP_CLEARANCE_M) == 0.0
    assert _safety_ang_limit(-3.0, _ESTOP_CLEARANCE_M) == 0.0


# ---------------------------------------------------------------------------
# Heading-direction clearance + speed cap (T-016)
# ---------------------------------------------------------------------------


def _ray(angle_rad: float, distance_m: float) -> tuple:
    return (angle_rad, distance_m)


def test_directional_clearance_uses_forward_cone_only():
    # A close obstacle dead ahead dominates; a closer one on the side is
    # outside the travel cone and must be ignored.
    rays = [_ray(0.0, 1.0), _ray(math.pi / 2, 0.8)]
    got = _directional(rays, forward=True)
    expected = 1.0 - square_footprint_radius(0.0, bot_radius_m=0.375)
    assert got == pytest.approx(expected)


def test_directional_clearance_reverse_cone():
    rays = [_ray(math.pi, 1.2), _ray(0.0, 0.5)]
    got = _directional(rays, forward=False)
    assert got == pytest.approx(1.2 - 0.375)
    # The close obstacle dead ahead is not relevant when reversing.
    assert got > 0.5


def test_directional_clearance_empty_cone_is_infinite():
    # Rays only on the sides: nothing in the travel cone -> no cap.
    rays = [_ray(math.pi / 2, 0.2), _ray(-math.pi / 2, 0.2)]
    assert _directional(rays, forward=True) == float("inf")
    assert _directional(rays, forward=False) == float("inf")


def test_heading_speed_cap_zero_at_stop_boundary():
    # At the stop boundary (and inside it) the cap is zero.
    assert _heading_cap(_STOP_CLEARANCE_M, 4.0) == 0.0
    assert _heading_cap(0.0, 4.0) == 0.0
    assert _heading_cap(-0.1, 4.0) == 0.0


def test_heading_speed_cap_braking_distance():
    # sqrt(2 * a * (d - stop)): 2 m/s of braking room at 4 m/s².
    d = _STOP_CLEARANCE_M + 0.5
    assert _heading_cap(d, 4.0) == pytest.approx(math.sqrt(2.0 * 4.0 * 0.5))


def test_heading_speed_cap_does_not_exceed_max_speed():
    assert _heading_cap(10.0, 4.0, max_speed_mps=2.0) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# E-stop threshold command triage
# ---------------------------------------------------------------------------


def test_estop_threshold_action_reverse_allowed():
    assert _estop_action(-0.5, 0.0) == "reverse"


def test_estop_threshold_action_forward_latches():
    assert _estop_action(0.5, 0.0) == "latch"


def test_estop_threshold_action_rotation_latches():
    # Turning can swing the body into the hazard.
    assert _estop_action(0.0, 1.0) == "latch"


def test_estop_threshold_action_zero_holds_without_latching():
    # A zero command is safe: after a reset an idle robot must not instantly
    # re-latch (this caused the reset/re-latch storm in the field logs).
    assert _estop_action(0.0, 0.0) == "hold"
