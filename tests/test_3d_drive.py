"""Headless tests for T3D-06 (base drive + navigation integration).

Tests the deterministic drive-command logic (wheel_speed vs teleop preference,
clamping, slip corruption) rather than the raw base displacement, which the
4-fixed-wheel base's poor traction makes unreliable.  Run in PyBullet DIRECT
mode (no display), zenoh disabled.
"""

from __future__ import annotations

import time

import pybullet as p
import pytest

from core.robot_config import load_robot_config
from simulation3d.simulator import Simulator3D, WHEEL_TIMEOUT_S


@pytest.fixture()
def sim():
    s = Simulator3D(gui=False, map_path=None, enable_zenoh=False)
    # Disable slip corruption so the *commanded* wheel speeds are exact.
    s._wheel_scale = 1.0
    s._track_scale = 1.0
    s._physics.cross_coupling_rad_per_m = 0.0
    s._physics.slip_noise = 0.0
    yield s
    p.disconnect()


def _limit():
    cfg = load_robot_config()
    return 1.2 * cfg.chassis.linear_speed_mps / cfg.chassis.wheel_radius_m


def test_clamp_limits_wheel_speed(sim):
    limit = _limit()
    assert sim._clamp_wheel(999.0) == pytest.approx(limit)
    assert sim._clamp_wheel(-999.0) == pytest.approx(-limit)
    assert sim._clamp_wheel(10.0) == pytest.approx(10.0)


def test_slip_is_identity_without_corruption(sim):
    # _corrupt_unicycle with no corruption returns the input unchanged.
    sim._wheel_scale = 1.0
    sim._track_scale = 1.0
    sim._physics.cross_coupling_rad_per_m = 0.0
    sim._physics.slip_noise = 0.0
    l, a = sim._corrupt_unicycle(1.0, 0.5)
    assert l == pytest.approx(1.0)
    assert a == pytest.approx(0.5)


def test_teleop_w_commands_forward(sim):
    sim._keys["w"] = True
    el, er = sim._compute_drive_command()
    assert el > 0 and er > 0  # both wheels spin forward


def test_teleop_a_commands_left_turn(sim):
    sim._keys["a"] = True
    el, er = sim._compute_drive_command()
    # For this base (rear-drive + front casters), a left turn is left wheel
    # forward / right wheel back (verified empirically).
    assert el > 0 > er


def test_wheel_speed_is_preferred_over_teleop(sim):
    sim._wheel_speed = (5.0, 5.0)
    sim._wheel_speed_time = time.monotonic()
    sim._keys["w"] = True  # teleop would push forward, but drive wins
    el, er = sim._compute_drive_command()
    assert (el, er) == pytest.approx((5.0, 5.0))


def test_wheel_speed_is_clamped(sim):
    sim._wheel_speed = (999.0, -999.0)
    sim._wheel_speed_time = time.monotonic()
    el, er = sim._compute_drive_command()
    assert el == pytest.approx(_limit())
    assert er == pytest.approx(-_limit())


def test_stale_wheel_speed_falls_back_to_teleop(sim):
    sim._wheel_speed = (0.0, 0.0)
    sim._wheel_speed_time = time.monotonic() - (WHEEL_TIMEOUT_S + 0.5)
    sim._keys["w"] = True
    el, er = sim._compute_drive_command()
    assert el > 0 and er > 0  # stale drive ignored -> teleop W drives forward
