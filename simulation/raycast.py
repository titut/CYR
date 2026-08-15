"""2D ray casting utilities for LIDAR simulation.

The functions here operate on map-format Wall objects and are used by both the
simulator and the particle filter so that both agree on what a given pose sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from map_format import Obstacle, Wall


@dataclass(frozen=True)
class RayHit:
    """Result of casting a single ray."""

    angle: float  # radians, relative to the ray origin's forward direction
    distance: float  # meters, Euclidean distance to hit point
    point: Tuple[float, float]  # hit point in map coordinates


def _ray_segment_intersection(
    origin: Tuple[float, float],
    direction: float,
    wall: Wall,
) -> Optional[float]:
    """Return parametric distance t along the ray where it hits the wall segment.

    Returns None if the ray does not intersect the segment.
    """
    ox, oy = origin
    dx = math.cos(direction)
    dy = math.sin(direction)

    wx = wall.x2 - wall.x1
    wy = wall.y2 - wall.y1

    denom = dx * wy - dy * wx
    if abs(denom) < 1e-9:
        # Ray is parallel to the wall segment.
        return None

    t_numer = (wall.x1 - ox) * wy - (wall.y1 - oy) * wx
    u_numer = (wall.x1 - ox) * dy - (wall.y1 - oy) * dx

    t = t_numer / denom
    u = u_numer / denom

    if t >= 0 and 0.0 <= u <= 1.0:
        return t
    return None


def _ray_circle_intersection(
    origin: Tuple[float, float],
    direction: float,
    obstacle: Obstacle,
) -> Optional[float]:
    """Return the parametric distance t to the nearest circle intersection.

    Returns None if the ray misses the circle entirely.
    """
    ox, oy = origin
    dx = math.cos(direction)
    dy = math.sin(direction)

    # Vector from the ray origin to the circle center: v = origin - center.
    vx = ox - obstacle.x
    vy = oy - obstacle.y

    # Solve |v + t*d|^2 = r^2  (d is a unit vector):
    #   t^2 + 2*(v·d)*t + |v|^2 - r^2 = 0
    b = 2.0 * (vx * dx + vy * dy)
    c = vx * vx + vy * vy - obstacle.radius_m * obstacle.radius_m

    discriminant = b * b - 4.0 * c
    if discriminant < 0.0:
        return None

    sqrt_d = math.sqrt(discriminant)
    t0 = (-b - sqrt_d) / 2.0
    t1 = (-b + sqrt_d) / 2.0

    # Return the nearest non-negative entry point.
    if t0 >= 0.0:
        return t0
    if t1 >= 0.0:
        return t1
    # Both behind the ray origin.
    return None


def cast_ray(
    origin: Tuple[float, float],
    direction: float,
    walls: Sequence[Wall],
    max_range: float,
    obstacles: Optional[Sequence[Obstacle]] = None,
) -> Optional[Tuple[float, Tuple[float, float]]]:
    """Cast a single ray and return (distance, hit_point) or None if no hit.

    ``origin``, ``max_range`` and the returned ``distance`` are in meters; the
    ``walls``/``obstacles`` must be expressed in meters as well.
    """
    closest: Optional[float] = None
    for wall in walls:
        t = _ray_segment_intersection(origin, direction, wall)
        if t is not None and (closest is None or t < closest):
            closest = t

    if obstacles:
        for obstacle in obstacles:
            t = _ray_circle_intersection(origin, direction, obstacle)
            if t is not None and (closest is None or t < closest):
                closest = t

    if closest is None or closest > max_range:
        return None

    ox, oy = origin
    hit_x = ox + closest * math.cos(direction)
    hit_y = oy + closest * math.sin(direction)
    return (closest, (hit_x, hit_y))


def cast_rays(
    origin: Tuple[float, float],
    forward_direction: float,
    walls: Sequence[Wall],
    num_rays: int = 360,
    max_range: float = 200.0,
    fov_rad: float = 2.0 * math.pi,
    obstacles: Optional[Sequence[Obstacle]] = None,
) -> List[RayHit]:
    """Cast a fan of rays around `origin`.

    Args:
        origin: Ray origin in world coordinates (meters).
        forward_direction: Orientation of the sensor in radians.
        walls: Wall segments to intersect against (meters).
        num_rays: Number of evenly spaced rays.
        max_range: Maximum range in meters.
        fov_rad: Total field of view in radians. Use 2*pi for 360 degree LIDAR.
        obstacles: Optional circular obstacles to intersect against (meters).

    Returns:
        A list of RayHit objects, one per ray. Rays that hit nothing are omitted.
    """
    if num_rays <= 1:
        step = 0.0
        start = forward_direction
    else:
        step = fov_rad / (num_rays - 1)
        start = forward_direction - fov_rad / 2.0

    hits: List[RayHit] = []
    for i in range(num_rays):
        angle = start + i * step
        result = cast_ray(origin, angle, walls, max_range, obstacles)
        if result is not None:
            distance, point = result
            hits.append(
                RayHit(
                    angle=angle - forward_direction,
                    distance=distance,
                    point=point,
                )
            )
    return hits
