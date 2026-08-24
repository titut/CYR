"""Tests for the hardware abstraction layer / driver model (T-019)."""

from __future__ import annotations

import pytest

from core.hal import (
    load_camera_driver,
    load_drive_driver,
    load_imu_driver,
    load_lidar_driver,
)
from core.map_format import Apriltag, Wall
from core.robot_config import ConfigError, load_robot_config
from core.sim_drivers import (
    LoggingDriveDriver,
    SimCameraDriver,
    SimDriveDriver,
    SimImuDriver,
    SimLidarDriver,
)


def _write_config(tmp_path, text: str):
    path = tmp_path / "robot.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _default_cfg():
    return load_robot_config()


def test_load_sim_drive_driver():
    assert isinstance(load_drive_driver(_default_cfg()), SimDriveDriver)


def test_load_logging_drive_driver_via_config(tmp_path):
    path = _write_config(tmp_path, "hardware:\n  drive: {driver: logging}\n")
    cfg = load_robot_config(path)
    assert isinstance(load_drive_driver(cfg), LoggingDriveDriver)


def test_unknown_driver_raises(tmp_path):
    path = _write_config(tmp_path, "hardware:\n  drive: {driver: bogus}\n")
    cfg = load_robot_config(path)
    with pytest.raises(ConfigError):
        load_drive_driver(cfg)


def test_sensor_drivers_load():
    cfg = _default_cfg()
    assert isinstance(load_lidar_driver(cfg), SimLidarDriver)
    assert isinstance(load_imu_driver(cfg), SimImuDriver)
    assert isinstance(load_camera_driver(cfg), SimCameraDriver)


def test_sim_drive_driver_reaches_steady_state():
    cfg = _default_cfg()
    driver = SimDriveDriver(cfg)
    driver.set_command(1.0, 0.0)  # 1 m/s straight
    target_rps = 1.0 / cfg.chassis.wheel_radius_m
    for _ in range(2000):  # ~20 s at 0.01 s steps
        left, right = driver.step(0.01)
    assert left == pytest.approx(target_rps, rel=0.05)
    assert right == pytest.approx(target_rps, rel=0.05)
    assert left == pytest.approx(right, abs=0.1)


def test_sim_drive_driver_tracks_geometry_from_config():
    # A config with a different wheel radius must change the wheel speeds.
    from core.robot_config import default_robot_config

    c1 = default_robot_config()
    c2 = default_robot_config()
    c2.chassis.wheel_radius_m = 0.1  # smaller wheel -> faster spin for 1 m/s
    d1 = SimDriveDriver(c1)
    d2 = SimDriveDriver(c2)
    for d in (d1, d2):
        d.set_command(1.0, 0.0)
        for _ in range(2000):
            d.step(0.01)
    left1, _ = d1.step(0.0)
    left2, _ = d2.step(0.0)
    assert left2 > left1 + 0.5


def test_logging_drive_driver_reports_zero(capsys):
    cfg = _default_cfg()
    driver = LoggingDriveDriver(cfg)
    driver.set_command(1.0, 0.5)
    out = capsys.readouterr().out
    assert "set velocity" in out
    assert driver.step(0.01) == (0.0, 0.0)


def _wall_grid():
    return [
        Wall(0, 0, 10, 0),
        Wall(10, 0, 10, 10),
        Wall(0, 10, 10, 10),
        Wall(0, 0, 0, 10),
        Wall(8, 0, 8, 10),
    ]


def test_sim_lidar_driver_scan_hits_wall_and_max_range():
    driver = SimLidarDriver(_default_cfg())
    hits = driver.scan((5.0, 5.0), 0.0, _wall_grid())
    # At least one ray hit something (the east wall 3 m away at x=8).
    distances = [h.distance for h in hits]
    assert distances
    assert min(distances) <= 3.5
    assert max(distances) <= 10.0 + 1e-6


def test_sim_imu_driver_reads_around_truth():
    driver = SimImuDriver(_default_cfg())
    for _ in range(50):
        data = driver.read(theta_rad=0.5, angular_velocity_rps=1.0)
        assert data["yaw_rad"] == pytest.approx(0.5, abs=0.2)
        assert data["angular_velocity_rps"] == pytest.approx(1.0, abs=0.2)


def test_sim_camera_driver_detects_nearby_tag():
    cfg = _default_cfg()
    driver = SimCameraDriver(cfg)
    tags = [Apriltag(id=0, x=9.0, y=5.0, yaw_rad=0.0, size_m=0.16)]
    dets = driver.detect(bot_x=5.0, bot_y=5.0, bot_theta_rad=0.0, tags=tags)
    assert len(dets) == 1
    assert dets[0]["id"] == 0
    assert dets[0]["range_m"] == pytest.approx(4.0, abs=0.2)


def test_sim_camera_driver_ignores_out_of_range_tag():
    driver = SimCameraDriver(_default_cfg())
    tags = [Apriltag(id=1, x=50.0, y=50.0)]
    assert driver.detect(bot_x=5.0, bot_y=5.0, bot_theta_rad=0.0, tags=tags) == []
