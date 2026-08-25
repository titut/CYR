"""3D sensor models: LIDAR, IMU and AprilTag camera (T3D-05).

These generate the same zenoh messages the 2D sim did, but from the PyBullet
world: LIDAR via ``p.rayTest``, IMU from the base's rigid-body state, and the
AprilTag camera by porting the 2D FOV-cone logic against the map tags.

Noise/bias values come from the robot config (``sensors.*``), which is the same
single source the 2D sim used.  All randomness goes through the module ``random``
so the sim's ``seed_all(seed)`` keeps runs deterministic.
"""

from __future__ import annotations

import math
import random
from typing import Callable, List, Optional, Sequence, Tuple

import pybullet as p

from core.map_format import MapData
from core.robot_config import RobotConfig
from simulation3d.world import WALL_THICKNESS_M

GRAVITY_Z = -9.81


def _wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _ray_clear(
    robot_bodies: set, origin: Sequence[float], dir2d: Tuple[float, float], range_m: float
) -> float:
    """Cast a horizontal LIDAR ray, ignoring hits on the robot's own body.

    Returns the distance to the first *non-robot* hit (walls, objects), or
    ``range_m`` if none.  Rays that first hit the base/arm are continued just
    past that hit and re-cast, so the robot's own body does not blind it.
    """
    remaining = range_m
    cur = [origin[0], origin[1], origin[2]]
    while remaining > 1e-4:
        to = [cur[0] + dir2d[0] * remaining, cur[1] + dir2d[1] * remaining, cur[2]]
        hit = p.rayTest(cur, to)[0]
        # rayTest result: (objectUniqueId, linkIndex, hitFraction, hitPos, hitNormal)
        if hit[0] < 0:
            return range_m
        if hit[0] not in robot_bodies:
            return (range_m - remaining) + hit[2] * remaining
        hp = hit[3]
        consumed = math.hypot(cur[0] - hp[0], cur[1] - hp[1])
        remaining -= consumed
        cur = [hp[0] + dir2d[0] * 0.002, hp[1] + dir2d[1] * 0.002, hp[2]]
    return range_m


def scan_lidar(
    base_uid: int,
    arm_uid: int,
    cfg: RobotConfig,
    noise_sigma_m: float = 0.02,
) -> List[dict]:
    """A full 2π LIDAR scan from the base at the configured mount height.

    Returns a list of ``{"angle_rad", "distance_m"}`` (same schema as the 2D
    sim), one per ray, with the base's own body filtered out.
    """
    pos, orn = p.getBasePositionAndOrientation(base_uid)
    yaw = p.getEulerFromQuaternion(orn)[2]
    lidar = cfg.sensors.lidar
    origin = [pos[0], pos[1], lidar.mount_xyz_m[2]]
    robot = {base_uid, arm_uid}
    rays: List[dict] = []
    for i in range(lidar.ray_count):
        angle = 2.0 * math.pi * i / lidar.ray_count
        world = yaw + angle
        dx, dy = math.cos(world), math.sin(world)
        d = _ray_clear(robot, origin, (dx, dy), lidar.range_m)
        d = min(d, lidar.range_m) + random.gauss(0.0, noise_sigma_m)
        rays.append({"angle_rad": angle, "distance_m": max(0.0, d)})
    return rays


def read_imu(
    base_uid: int,
    cfg: RobotConfig,
    gyro_bias_rps: float,
    prev_lin_vel: Sequence[float] | None,
    dt: float,
) -> dict:
    """IMU reading from the base's rigid-body state (same schema as the 2D sim)."""
    pos, orn = p.getBasePositionAndOrientation(base_uid)
    yaw = p.getEulerFromQuaternion(orn)[2]
    ang_vel, lin_vel = p.getBaseVelocity(base_uid)
    imu = cfg.sensors.imu

    noisy_yaw = _wrap_angle(yaw + random.gauss(0.0, imu.yaw_noise_rad))
    noisy_rate = ang_vel[2] + gyro_bias_rps + random.gauss(0.0, imu.gyro_noise_rps)

    # Proper acceleration = dv/dt - g (accelerometer at rest reads +g upward).
    if dt > 0 and prev_lin_vel is not None:
        accel = [(lin_vel[i] - prev_lin_vel[i]) / dt for i in range(3)]
    else:
        accel = [0.0, 0.0, 0.0]
    accel[2] -= GRAVITY_Z
    accel = [a + random.gauss(0.0, 0.01) for a in accel]

    return {
        "yaw_rad": noisy_yaw,
        "angular_velocity_rps": noisy_rate,
        "linear_acceleration_mps2": {
            "x": accel[0],
            "y": accel[1],
            "z": accel[2],
        },
    }


def detect_apriltags(
    map_data: MapData,
    bot_pose: Tuple[float, float, float],
    cfg: RobotConfig,
    is_occluded: Optional[Callable[[float, float], bool]] = None,
) -> List[dict]:
    """AprilTag camera: 3D-aware FOV cone + line-of-sight detection.

    ``bot_pose`` is the base's (x, y, yaw) ground truth; the camera sits at the
    configured mount offset rotated with the base.

    A tag is "printed" on the wall face its ``yaw_rad`` points into, so it is
    only visible from that side (facing gate).  ``is_occluded(x, y)`` is an
    optional line-of-sight callback: when provided, a tag whose path from the
    camera is blocked by a wall/object is not reported, so tags on the far side
    of a wall no longer inject false pose anchors.

    Returns the same schema as the 2D sim (``{id, range_m, bearing_rad,
    tag_yaw_rad, tag_size_m}``).
    """
    cam = cfg.sensors.camera
    half_fov = cam.fov_rad / 2.0
    bx, by, btheta = bot_pose
    mx, my, _ = cam.mount_xyz_m
    # Camera world xy: base pose + mount offset rotated by the base yaw.
    cam_x = bx + mx * math.cos(btheta) - my * math.sin(btheta)
    cam_y = by + mx * math.sin(btheta) + my * math.cos(btheta)

    detections: List[dict] = []
    for tag in map_data.apriltags:
        # The tag faces along its yaw_rad; a camera on the opposite side of the
        # wall is behind the tag and cannot see it.
        facing_x = math.cos(tag.yaw_rad)
        facing_y = math.sin(tag.yaw_rad)
        if (cam_x - tag.x) * facing_x + (cam_y - tag.y) * facing_y <= 0:
            continue

        dx, dy = tag.x - cam_x, tag.y - cam_y
        dist = math.hypot(dx, dy)
        if dist > cam.max_range_m:
            continue
        angle_to_tag = math.atan2(dy, dx)
        bearing = _wrap_angle(angle_to_tag - btheta)
        if abs(bearing) > half_fov:
            continue

        # Line of sight.  Probe the tag's printed face itself (exactly on the
        # wall surface) so the tag's own wall doesn't occlude it, but any
        # wall/object in between still does.  The probe is horizontal (camera
        # height), so low floor objects don't block a wall-mounted tag.
        if is_occluded is not None:
            sx = tag.x + (WALL_THICKNESS_M / 2.0) * facing_x
            sy = tag.y + (WALL_THICKNESS_M / 2.0) * facing_y
            if is_occluded(sx, sy):
                continue

        noisy_range = dist + random.gauss(0.0, cam.range_noise_m)
        noisy_bearing = bearing + random.gauss(0.0, cam.bearing_noise_rad)
        tag_yaw_in_camera = _wrap_angle(tag.yaw_rad - btheta)
        noisy_tag_yaw = tag_yaw_in_camera + random.gauss(0.0, cam.yaw_noise_rad)
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
