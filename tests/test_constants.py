"""Tests for the shared constants module and canonical map_format (T-027).

These lock in the "single source of truth" invariant: ``constants.py``
re-exports the *default* robot parameters from ``robot_config.py``, and
``simulation/kinematics.py`` derives its geometry from those constants rather
than carrying its own copies.  Runtime values come from ``robot.yaml`` via
``robot_config`` (see tests/test_robot_config.py).
"""

from __future__ import annotations

import math

import pytest

import core.constants as constants
import core.robot_config as robot_config
import simulation.kinematics as kinematics


def test_constants_re_export_default_robot_config():
    default = robot_config.default_robot_config()
    assert constants.BOT_SIZE_M == default.chassis.footprint_m
    assert constants.WHEEL_RADIUS_M == default.chassis.wheel_radius_m
    assert constants.BOT_LINEAR_SPEED_MPS == default.chassis.linear_speed_mps
    assert constants.BOT_ANGULAR_SPEED_RPS == default.chassis.angular_speed_rps
    assert constants.LIDAR_MAX_RANGE_M == default.sensors.lidar.range_m
    assert constants.LIDAR_RAY_COUNT == default.sensors.lidar.ray_count
    assert constants.CAMERA_FOV_RAD == default.sensors.camera.fov_rad


def test_kinematics_reuses_canonical_constants():
    # T-027: kinematics derives its geometry from constants.py, not a copy.
    assert kinematics.BOT_SIZE_M is constants.BOT_SIZE_M
    assert kinematics.WHEEL_RADIUS_M is constants.WHEEL_RADIUS_M
    assert kinematics.BOT_LINEAR_SPEED_MPS is constants.BOT_LINEAR_SPEED_MPS
    assert kinematics.BOT_ANGULAR_SPEED_RPS is constants.BOT_ANGULAR_SPEED_RPS


def test_kinematics_derived_values():
    assert kinematics.BOT_RADIUS_M == pytest.approx(constants.BOT_SIZE_M / 2.0)
    # Wheel track now comes from the config (corner wheels), not derived.
    assert kinematics.WHEEL_TRACK_M is constants.WHEEL_TRACK_M
    assert kinematics.WHEEL_TRACK_M == pytest.approx(0.69)


def test_kinematics_functions_accept_custom_geometry():
    # T-019: nodes pass their loaded config geometry; defaults = nominal robot.
    from simulation.kinematics import unicycle_to_wheel, wheel_to_unicycle

    l, r = unicycle_to_wheel(1.0, 0.0, wheel_radius_m=0.1, wheel_track_m=0.4)
    assert l == pytest.approx(1.0 / 0.1)
    assert l == pytest.approx(r)
    lin, ang = wheel_to_unicycle(l, r, wheel_radius_m=0.1, wheel_track_m=0.4)
    assert lin == pytest.approx(1.0)


def test_canonical_values_preserved():
    # Guard against accidental changes to the shared geometry/noise model.
    assert constants.BOT_SIZE_M == pytest.approx(0.75)
    assert constants.WHEEL_RADIUS_M == pytest.approx(0.12)
    assert constants.LIDAR_MAX_RANGE_M == pytest.approx(10.0)
    assert constants.LIDAR_RAY_COUNT == 360
    assert constants.CAMERA_FOV_RAD == pytest.approx(math.radians(90))
    assert constants.CAMERA_MAX_RANGE_M == pytest.approx(5.0)
    assert constants.CAMERA_RANGE_NOISE_M == pytest.approx(0.02)
    assert constants.CAMERA_BEARING_NOISE_RAD == pytest.approx(math.radians(1.0))
    assert constants.CAMERA_YAW_NOISE_RAD == pytest.approx(math.radians(2.0))


def test_map_format_is_canonical_under_core():
    # T-027: the map schema lives only in core.map_format (the editor's private
    # copy was removed in the directory reorganisation); loading a map works.
    from core.map_format import MapData, new_empty_map

    m = new_empty_map()
    assert isinstance(m, MapData)
    assert m.metadata.size_m == (8.0, 6.0)
