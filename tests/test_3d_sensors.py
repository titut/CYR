"""Headless tests for T3D-05 (sensor bridge: LIDAR / IMU / AprilTag camera).

Run in PyBullet DIRECT mode (no display), so it is CI-safe.
"""

from __future__ import annotations

import math

import pybullet as p
import pybullet_data
import pytest

from core.map_format import MapData
from core.messages import SchemaError, decode
from core.robot_config import load_robot_config
from simulation3d.sensors import detect_apriltags, read_imu, scan_lidar
from simulation3d.world import _spawn_wall_box


@pytest.fixture()
def direct():
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(numSubSteps=4, fixedTimeStep=1.0 / 240.0)
    yield cid
    p.disconnect()


@pytest.fixture()
def base_arm(direct):
    cfg = load_robot_config()
    ch = cfg.chassis
    base = p.loadURDF("urdf/base.urdf", basePosition=[0, 0, ch.wheel_radius_m + ch.height_m / 2.0])
    arm = p.loadURDF("urdf/mycobot_280_pi_3d.urdf")
    for _ in range(240):
        p.stepSimulation()
    # The base slides/yaws while settling (a known T3D-06 limitation), so park
    # it at a known pose facing +x before the sensor tests.
    p.resetBasePositionAndOrientation(
        base, [0.0, 0.0, ch.wheel_radius_m + ch.height_m / 2.0], [0.0, 0.0, 0.0, 1.0]
    )
    for _ in range(30):
        p.stepSimulation()
    return cfg, base, arm


def _min_distance(rays):
    return min(r["distance_m"] for r in rays)


def test_lidar_open_space_is_max_range(base_arm):
    cfg, base, arm = base_arm
    rays = scan_lidar(base, arm, cfg, noise_sigma_m=0.0)  # no noise
    assert len(rays) == cfg.sensors.lidar.ray_count == 360
    # open space (no walls) -> every ray at max range
    assert _min_distance(rays) == pytest.approx(cfg.sensors.lidar.range_m, abs=0.01)


def test_lidar_hits_wall(base_arm):
    cfg, base, arm = base_arm
    _spawn_wall_box(2.0, -5.0, 2.0, 5.0)  # a wall at x=2 m
    p.resetBasePositionAndOrientation(base, [0.0, 0.0, 0.195], [0.0, 0.0, 0.0, 1.0])
    for _ in range(30):
        p.stepSimulation()
    rays = scan_lidar(base, arm, cfg, noise_sigma_m=0.0)
    # some ray must be short (~2 m) — the wall is the nearest thing in its arc
    assert _min_distance(rays) < 2.5


def test_lidar_hits_object(base_arm):
    cfg, base, arm = base_arm
    # a tall box in front so the horizontal LIDAR (z=0.2) hits it
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.5], rgbaColor=[1, 0, 0, 1])
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.5])
    p.createMultiBody(0, col, visual, basePosition=[1.0, 0.0, 0.5])
    p.resetBasePositionAndOrientation(base, [0.0, 0.0, 0.195], [0.0, 0.0, 0.0, 1.0])
    for _ in range(30):
        p.stepSimulation()
    rays = scan_lidar(base, arm, cfg, noise_sigma_m=0.0)
    assert _min_distance(rays) < 1.5


def test_imu_reads_yaw_and_accel(base_arm):
    cfg, base, arm = base_arm
    imu = read_imu(base, cfg, gyro_bias_rps=0.0, prev_lin_vel=None, dt=0.02)
    # accelerometer at rest reads ~+9.8 upward (proper accel = dv/dt - g)
    assert imu["linear_acceleration_mps2"]["z"] == pytest.approx(9.81, abs=0.1)
    assert isinstance(imu["yaw_rad"], float)
    assert isinstance(imu["angular_velocity_rps"], float)


def test_apriltag_camera_detects_map_tags():
    cfg = load_robot_config()
    map_data = MapData.from_json("home.json")
    # stand in the first room centre facing +x (toward the tags) and check we
    # get detections with the expected schema fields
    bot_pose = (8.0, 8.5, 0.0)
    dets = detect_apriltags(map_data, bot_pose, cfg)
    for d in dets:
        for field in ("id", "range_m", "bearing_rad", "tag_yaw_rad", "tag_size_m"):
            assert field in d


def test_lidar_message_schema(base_arm):
    cfg, base, arm = base_arm
    from core.messages import encode

    rays = scan_lidar(base, arm, cfg)
    msg = encode("sensor/lidar", {"t": 1.0, "rays": rays})
    data = decode("sensor/lidar", msg)  # raises SchemaError if invalid
    assert len(data["rays"]) == 360
