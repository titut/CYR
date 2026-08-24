"""Default robot parameters, re-exported from ``robot_config.py`` (T-019).

The single source of truth for robot-specific values is the ``robot.yaml``
config loaded by :mod:`robot_config`.  This module re-exports the *default*
(nominal) values as plain constants so existing code that imports e.g.
``BOT_SIZE_M`` keeps working unchanged, and so ``simulation/kinematics.py`` can
build its geometry helpers from the nominal robot.

Runtime configuration: nodes load ``robot_config.get_robot_config()`` and read
their values from there (the YAML overrides these defaults).
"""

from __future__ import annotations

from .robot_config import default_robot_config

_DEFAULT = default_robot_config()
_chassis = _DEFAULT.chassis
_lidar = _DEFAULT.sensors.lidar
_camera = _DEFAULT.sensors.camera

BOT_SIZE_M = _chassis.footprint_m  # robot footprint width (m), square side
WHEEL_RADIUS_M = _chassis.wheel_radius_m
WHEEL_TRACK_M = _chassis.wheel_track_m  # drive-wheel separation (m)
BOT_LINEAR_SPEED_MPS = _chassis.linear_speed_mps
BOT_ANGULAR_SPEED_RPS = _chassis.angular_speed_rps

LIDAR_MAX_RANGE_M = _lidar.range_m
LIDAR_RAY_COUNT = _lidar.ray_count

CAMERA_FOV_RAD = _camera.fov_rad
CAMERA_MAX_RANGE_M = _camera.max_range_m
CAMERA_RANGE_NOISE_M = _camera.range_noise_m
CAMERA_BEARING_NOISE_RAD = _camera.bearing_noise_rad
CAMERA_YAW_NOISE_RAD = _camera.yaw_noise_rad
