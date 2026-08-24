"""Simulated (and logging) driver implementations (T-019, driver model).

These implement the interfaces in :mod:`hal` using the 2D simulated world.  The
robot config (``robot.yaml`` ``hardware:`` section) selects which driver backs
each device; adding real hardware means implementing the same interface and
registering it in the relevant ``*_DRIVERS`` dict here.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

from .hal import CameraDriver, DriveDriver, ImuDriver, LidarDriver
from .map_format import Apriltag
from .robot_config import default_robot_config
from simulation.kinematics import unicycle_to_wheel
from simulation.raycast import RayHit, cast_rays

_DEF = default_robot_config()


# ---------------------------------------------------------------------------
# Velocity-loop plant step (shared by SimDriveDriver; kept module-level so the
# safety tests can exercise the pure math).
# ---------------------------------------------------------------------------


def _velocity_step(
    current: float,
    target: float,
    dt: float,
    time_constant_s: float = _DEF.drive.motor_time_constant_s,
    max_accel_rps2: float = _DEF.drive.max_wheel_accel_rps2,
    max_decel_rps2: float = _DEF.drive.max_wheel_decel_rps2,
) -> float:
    """Advance ``current`` toward ``target`` with first-order dynamics, capped
    by per-step acceleration/deceleration limits.

    Models a velocity-controlled driver (velocity-loop PID in firmware,
    approximated as a first-order response) with a finite torque limit: large
    step changes ramp linearly at ``max_accel_rps2`` (and brake at
    ``max_decel_rps2``, which is higher so stops are crisp), while small
    changes decay with ``time_constant_s``.
    """
    alpha = min(1.0, dt / time_constant_s)
    step = alpha * (target - current)
    if step > max_accel_rps2 * dt:
        step = max_accel_rps2 * dt
    elif step < -max_decel_rps2 * dt:
        step = -max_decel_rps2 * dt
    return current + step


# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------


class SimDriveDriver(DriveDriver):
    """The simulated motor controller + wheel encoders (the old drive.py plant).

    Holds the actual (simulated) wheel speeds and advances them with a
    first-order velocity loop; reports them with encoder noise, exactly as the
    previous inline model did.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        d = cfg.drive
        ch = cfg.chassis
        self._time_constant_s = d.motor_time_constant_s
        self._max_accel_rps2 = d.max_wheel_accel_rps2
        self._max_decel_rps2 = d.max_wheel_decel_rps2
        self._encoder_noise_rps = d.encoder_noise_rps
        self._wheel_radius_m = ch.wheel_radius_m
        self._wheel_track_m = ch.wheel_track_m
        self._target_linear = 0.0
        self._target_angular = 0.0
        self._left_rps = 0.0
        self._right_rps = 0.0

    def set_command(self, linear_mps: float, angular_rps: float) -> None:
        self._target_linear = linear_mps
        self._target_angular = angular_rps

    def step(self, dt: float) -> Tuple[float, float]:
        target_left, target_right = unicycle_to_wheel(
            self._target_linear,
            self._target_angular,
            wheel_radius_m=self._wheel_radius_m,
            wheel_track_m=self._wheel_track_m,
        )
        self._left_rps = _velocity_step(
            self._left_rps,
            target_left,
            dt,
            self._time_constant_s,
            self._max_accel_rps2,
            self._max_decel_rps2,
        )
        self._right_rps = _velocity_step(
            self._right_rps,
            target_right,
            dt,
            self._time_constant_s,
            self._max_accel_rps2,
            self._max_decel_rps2,
        )
        return (
            self._left_rps + random.gauss(0.0, self._encoder_noise_rps),
            self._right_rps + random.gauss(0.0, self._encoder_noise_rps),
        )


class LoggingDriveDriver(DriveDriver):
    """A no-hardware stub proving the driver pattern: it logs the command and
    reports the wheels at rest.  Selecting it via ``robot.yaml``
    (``hardware.drive.driver: logging``) runs the whole stack with no physics —
    a new robot is literally a driver + a config line."""

    def __init__(self, cfg):
        super().__init__(cfg)

    def set_command(self, linear_mps: float, angular_rps: float) -> None:
        print(
            f"[drive:logging] set velocity {linear_mps:.3f} m/s, "
            f"{angular_rps:.3f} rad/s"
        )

    def step(self, dt: float) -> Tuple[float, float]:
        return (0.0, 0.0)


# ---------------------------------------------------------------------------
# LIDAR
# ---------------------------------------------------------------------------


class SimLidarDriver(LidarDriver):
    """Ray-casts the 2D world (walls + obstacles) into a scan."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._range_m = cfg.sensors.lidar.range_m
        self._ray_count = cfg.sensors.lidar.ray_count

    def scan(self, origin, forward_direction, walls, obstacles=None) -> List[RayHit]:
        return cast_rays(
            origin=origin,
            forward_direction=forward_direction,
            walls=walls,
            num_rays=self._ray_count,
            max_range=self._range_m,
            fov_rad=2.0 * math.pi,
            obstacles=obstacles,
        )


# ---------------------------------------------------------------------------
# IMU
# ---------------------------------------------------------------------------


class SimImuDriver(ImuDriver):
    """Adds the per-run gyro bias + Gaussian noise model to true angular state."""

    def __init__(self, cfg):
        super().__init__(cfg)
        imu = cfg.sensors.imu
        self._bias_rps = random.uniform(-imu.gyro_bias_rps, imu.gyro_bias_rps)
        self._gyro_noise_rps = imu.gyro_noise_rps
        self._yaw_noise_rad = imu.yaw_noise_rad

    def read(self, theta_rad: float, angular_velocity_rps: float) -> dict:
        noisy_yaw = math.atan2(
            math.sin(theta_rad + random.gauss(0.0, self._yaw_noise_rad)),
            math.cos(theta_rad + random.gauss(0.0, self._yaw_noise_rad)),
        )
        return {
            "yaw_rad": noisy_yaw,
            "angular_velocity_rps": (
                angular_velocity_rps + self._bias_rps + random.gauss(0.0, self._gyro_noise_rps)
            ),
        }


# ---------------------------------------------------------------------------
# AprilTag camera
# ---------------------------------------------------------------------------


class SimCameraDriver(CameraDriver):
    """Detects tags within the camera's FOV cone, with measurement noise."""

    def __init__(self, cfg):
        super().__init__(cfg)
        cam = cfg.sensors.camera
        self._max_range_m = cam.max_range_m
        self._fov_rad = cam.fov_rad
        self._range_noise_m = cam.range_noise_m
        self._bearing_noise_rad = cam.bearing_noise_rad
        self._yaw_noise_rad = cam.yaw_noise_rad

    def detect(
        self, bot_x: float, bot_y: float, bot_theta_rad: float, tags: List[Apriltag]
    ) -> List[dict]:
        detections = []
        half_fov = self._fov_rad / 2.0
        for tag in tags:
            dx = tag.x - bot_x
            dy = tag.y - bot_y
            dist_m = math.hypot(dx, dy)
            if dist_m > self._max_range_m:
                continue
            angle_to_tag = math.atan2(dy, dx)
            bearing = math.atan2(
                math.sin(angle_to_tag - bot_theta_rad),
                math.cos(angle_to_tag - bot_theta_rad),
            )
            if abs(bearing) > half_fov:
                continue
            noisy_range = dist_m + random.gauss(0.0, self._range_noise_m)
            noisy_bearing = bearing + random.gauss(0.0, self._bearing_noise_rad)
            tag_yaw_in_camera = math.atan2(
                math.sin(tag.yaw_rad - bot_theta_rad),
                math.cos(tag.yaw_rad - bot_theta_rad),
            )
            noisy_tag_yaw = tag_yaw_in_camera + random.gauss(0.0, self._yaw_noise_rad)
            detections.append(
                {
                    "id": tag.id,
                    "range_m": max(0.0, noisy_range),
                    "bearing_rad": noisy_bearing,
                    "tag_yaw_rad": noisy_tag_yaw,
                    "tag_size_m": tag.size_m,
                }
            )
        return detections


# ---------------------------------------------------------------------------
# Driver registries (hardware.<device>.driver -> implementation)
# ---------------------------------------------------------------------------

DRIVE_DRIVERS = {"sim": SimDriveDriver, "logging": LoggingDriveDriver}
LIDAR_DRIVERS = {"sim": SimLidarDriver}
IMU_DRIVERS = {"sim": SimImuDriver}
CAMERA_DRIVERS = {"sim": SimCameraDriver}
