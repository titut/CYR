"""Tests for the robot configuration layer (T-019)."""

from __future__ import annotations

import math

import pytest

import core.robot_config as robot_config
from core.robot_config import (
    ConfigError,
    default_robot_config,
    load_robot_config,
    reset_robot_config,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_robot_config()
    yield
    reset_robot_config()


def _write_config(tmp_path, text: str):
    path = tmp_path / "robot.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_default_config_matches_current_values():
    cfg = default_robot_config()
    assert cfg.chassis.footprint_m == pytest.approx(0.75)
    assert cfg.chassis.wheel_radius_m == pytest.approx(0.12)
    assert cfg.chassis.wheel_track_m == pytest.approx(0.69)
    assert cfg.sensors.lidar.range_m == pytest.approx(10.0)
    assert cfg.sensors.lidar.ray_count == 360
    assert cfg.sensors.camera.fov_rad == pytest.approx(math.radians(90))
    assert cfg.drive.motor_time_constant_s == pytest.approx(0.05)
    assert cfg.safety.estop_clearance_m == pytest.approx(0.05)


def test_load_config_overrides_values(tmp_path):
    path = _write_config(
        tmp_path,
        """
name: big_robot
chassis:
  footprint_m: 1.2
  wheel_radius_m: 0.15
sensors:
  lidar:
    ray_count: 720
""",
    )
    cfg = load_robot_config(path)
    assert cfg.name == "big_robot"
    assert cfg.chassis.footprint_m == pytest.approx(1.2)
    assert cfg.chassis.wheel_radius_m == pytest.approx(0.15)
    # Untouched values keep the defaults.
    assert cfg.chassis.wheel_track_m == pytest.approx(0.69)
    assert cfg.sensors.lidar.range_m == pytest.approx(10.0)
    assert cfg.sensors.lidar.ray_count == 720


def test_load_config_derived_properties(tmp_path):
    path = _write_config(
        tmp_path,
        """
chassis:
  footprint_m: 1.0
""",
    )
    cfg = load_robot_config(path)
    assert cfg.chassis.radius_m == pytest.approx(0.5)
    assert cfg.chassis.collision_radius_m == pytest.approx(0.5 * math.sqrt(2.0))


def test_default_yaml_exists_and_matches_defaults():
    # The checked-in robot.yaml is the nominal robot the stack was tuned against.
    cfg = load_robot_config()
    default = default_robot_config()
    assert cfg.chassis.footprint_m == pytest.approx(default.chassis.footprint_m)
    assert cfg.sensors.lidar.range_m == pytest.approx(default.sensors.lidar.range_m)
    assert cfg.sensors.lidar.ray_count == default.sensors.lidar.ray_count
    assert cfg.drive.max_wheel_accel_rps2 == pytest.approx(default.drive.max_wheel_accel_rps2)


def test_unknown_key_raises(tmp_path):
    path = _write_config(tmp_path, "chassis:\n  bogus_field: 1.0\n")
    with pytest.raises(ConfigError):
        load_robot_config(path)


def test_wrong_type_raises(tmp_path):
    path = _write_config(tmp_path, "chassis:\n  footprint_m: not_a_number\n")
    with pytest.raises(ConfigError):
        load_robot_config(path)


def test_invalid_int_type_raises(tmp_path):
    path = _write_config(tmp_path, "sensors:\n  lidar:\n    ray_count: 3.5\n")
    with pytest.raises(ConfigError):
        load_robot_config(path)


def test_nonsensical_value_raises(tmp_path):
    path = _write_config(tmp_path, "chassis:\n  wheel_radius_m: -1.0\n")
    with pytest.raises(ConfigError):
        load_robot_config(path)


def test_missing_file_raises():
    with pytest.raises(ConfigError):
        load_robot_config("/nonexistent/robot.yaml")


def test_null_value_keeps_default(tmp_path):
    path = _write_config(tmp_path, "chassis:\n  wheel_track_m: null\n")
    cfg = load_robot_config(path)
    assert cfg.chassis.wheel_track_m == pytest.approx(0.69)


def test_get_robot_config_caches():
    a = robot_config.get_robot_config()
    b = robot_config.get_robot_config()
    assert a is b


def test_reset_robot_config_reloads():
    robot_config.get_robot_config()
    reset_robot_config()
    # After reset the cache is cleared, so the next load is fresh.
    assert robot_config.get_robot_config() is not None


def test_kinematics_functions_use_loaded_geometry(tmp_path):
    # The drive/sim nodes pass their loaded geometry into the pure helpers.
    from simulation.kinematics import square_footprint_radius, unicycle_to_wheel

    path = _write_config(tmp_path, "chassis:\n  footprint_m: 1.0\n  wheel_radius_m: 0.1\n")
    cfg = load_robot_config(path)
    l, r = unicycle_to_wheel(
        1.0, 0.0,
        wheel_radius_m=cfg.chassis.wheel_radius_m,
        wheel_track_m=cfg.chassis.wheel_track_m,
    )
    assert l == pytest.approx(1.0 / 0.1)
    assert square_footprint_radius(0.0, bot_radius_m=cfg.chassis.radius_m) == pytest.approx(0.5)
