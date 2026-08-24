"""Differential-drive kinematics and shared robot constants.

Converts between the unicycle model (linear m/s + angular rad/s), which the
controller works in, and the differential wheel speeds (left/right rad/s) of a
two-wheel robot, which the drive node works in.

Geometry assumptions:
    - Robot footprint is a 0.75 m square (``BOT_SIZE_M``), matching the simulator.
    - Wheels sit on the robot's left/right edge, flush with the footprint so no
      part of the wheel protrudes.  Wheel radius is ``WHEEL_RADIUS_M``.
    - The wheel track (distance between wheel centres) is therefore
      ``BOT_SIZE_M - 2 * WHEEL_RADIUS_M``.

The differential-drive equations:

    linear  = (v_left + v_right) / 2
    angular = (v_right - v_left) / track

where ``v_left``/``v_right`` are wheel linear velocities (m/s).  Wheel angular
speed (rad/s) is ``linear_wheel_velocity / wheel_radius``.
"""

from __future__ import annotations

import math

# Canonical constants live in core.constants.py (single source of truth, T-027).
# Re-exported here so callers can keep importing from this module unchanged.
from core.constants import (
    BOT_ANGULAR_SPEED_RPS,
    BOT_LINEAR_SPEED_MPS,
    BOT_SIZE_M,
    WHEEL_RADIUS_M,
    WHEEL_TRACK_M,
)

BOT_RADIUS_M = BOT_SIZE_M / 2.0


def unicycle_to_wheel(
    linear_mps: float,
    angular_rps: float,
    wheel_radius_m: float = WHEEL_RADIUS_M,
    wheel_track_m: float = WHEEL_TRACK_M,
) -> tuple[float, float]:
    """Convert unicycle velocity to differential wheel angular speeds.

    Args:
        linear_mps: forward linear velocity (m/s).
        angular_rps: angular velocity (rad/s), positive = counter-clockwise.
        wheel_radius_m: wheel radius (m); defaults to the nominal robot.
        wheel_track_m: distance between wheel centres (m).

    Returns:
        (left_rps, right_rps) wheel angular speeds in rad/s, positive = forward.
    """
    v_left = linear_mps - angular_rps * wheel_track_m / 2.0
    v_right = linear_mps + angular_rps * wheel_track_m / 2.0
    return (v_left / wheel_radius_m, v_right / wheel_radius_m)


def wheel_to_unicycle(
    left_rps: float,
    right_rps: float,
    wheel_radius_m: float = WHEEL_RADIUS_M,
    wheel_track_m: float = WHEEL_TRACK_M,
) -> tuple[float, float]:
    """Convert differential wheel angular speeds to unicycle velocity.

    Args:
        left_rps: left wheel angular speed (rad/s), positive = forward.
        right_rps: right wheel angular speed (rad/s), positive = forward.
        wheel_radius_m: wheel radius (m); defaults to the nominal robot.
        wheel_track_m: distance between wheel centres (m).

    Returns:
        (linear_mps, angular_rps) unicycle velocity.
    """
    v_left = left_rps * wheel_radius_m
    v_right = right_rps * wheel_radius_m
    return ((v_left + v_right) / 2.0, (v_right - v_left) / wheel_track_m)


def square_footprint_radius(
    angle_rad: float, bot_radius_m: float = BOT_RADIUS_M
) -> float:
    """Distance from centre to the edge of the square footprint in a direction.

    The footprint is a square of side ``2 * bot_radius_m`` aligned with the
    body frame (+x forward).  A ray at body-frame angle ``angle_rad`` reaches
    the boundary after this many metres.  It is smallest toward the face
    centres (``bot_radius_m``) and largest toward the corners
    (``bot_radius_m * sqrt(2)``), so the effective radius depends on angle.

    For a ray ``t * (cos φ, sin φ)`` the boundary is hit when
    ``t * max(|cos φ|, |sin φ|) == bot_radius_m``.
    """
    m = max(abs(math.cos(angle_rad)), abs(math.sin(angle_rad)))
    return bot_radius_m / m
