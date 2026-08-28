"""Tests for the 1D heading EKF (gyro + odometry + absolute yaw fusion)."""

from __future__ import annotations

import math

import pytest

from pose_estimation.heading_filter import HeadingFilter


def test_heading_integrates_gyro():
    hf = HeadingFilter(initial_heading=0.0)
    # The first step's returned increment is 0 by design (no previous heading);
    # it still advances the internal state.
    hf.step(dt=0.1, gyro_rate=1.0, odom_rate=1.0)
    dtheta = hf.step(dt=0.1, gyro_rate=1.0, odom_rate=1.0)
    assert dtheta == pytest.approx(0.1, abs=0.02)
    # A few more steps keeps tracking the gyro.
    for _ in range(4):
        hf.step(dt=0.1, gyro_rate=1.0, odom_rate=1.0)
    assert hf.heading == pytest.approx(0.6, abs=0.15)


def test_absolute_yaw_pulls_heading():
    hf = HeadingFilter(initial_heading=2.0)
    for _ in range(50):
        hf.step(dt=0.1, gyro_rate=0.0, odom_rate=0.0, yaw=0.0)
    assert hf.heading == pytest.approx(0.0, abs=0.2)


def test_yaw_wrap_handled():
    """Heading near +pi should be pulled to -pi by a yaw measurement without
    spinning the wrong way around the circle (+pi == -pi physically)."""
    hf = HeadingFilter(initial_heading=math.pi - 0.1)
    for _ in range(50):
        hf.step(dt=0.1, gyro_rate=0.0, odom_rate=0.0, yaw=-math.pi)
    # +pi and -pi are the same angle; assert circular distance to -pi is ~0.
    d = abs((hf.heading + math.pi + math.pi) % (2 * math.pi) - math.pi)
    assert d == pytest.approx(0.0, abs=0.2)


def test_track_scale_estimated():
    """A persistent odometry-rate error should be absorbed into track_scale."""
    hf = HeadingFilter()
    # Gyro says 1.0 rad/s but odometry only reports 0.8 -> track_scale < 1.
    for _ in range(200):
        hf.step(dt=0.1, gyro_rate=1.0, odom_rate=0.8, yaw=0.0)
    assert hf.track_scale < 1.0


def test_track_scale_clamped():
    hf = HeadingFilter()
    for _ in range(200):
        hf.step(dt=0.1, gyro_rate=1.0, odom_rate=0.05, yaw=0.0)
    # Track scale is bounded to its physical range (near unity), so a wildly
    # inconsistent odometry rate can't run it away (and corrupt the heading).
    assert 0.9 <= hf.track_scale <= 1.1


def test_odom_slip_is_ignored():
    """A grossly inconsistent wheel rate (slip) must not corrupt the heading:
    the gyro + absolute yaw stay authoritative."""
    hf = HeadingFilter(initial_heading=0.0)
    for _ in range(100):
        # Gyro says 1.0 rad/s (truth), but the wheels report ~2x that (slip).
        hf.step(dt=0.1, gyro_rate=1.0, odom_rate=2.5, yaw=None)
    # The heading is integrated from the gyro, not the slipped odometry rate.
    assert hf.heading == pytest.approx(10.0, abs=2.0)
