"""Tests for the drive node's velocity loop and safety zones (T-016)."""

from __future__ import annotations

import pytest

from control.drive import (
    _ESTOP_CLEARANCE_M,
    _SLOW_DOWN_CLEARANCE_M,
    _STOP_CLEARANCE_M,
    Drive,
)
from core.sim_drivers import _velocity_step

_safety_limit = Drive._safety_limited_linear


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
