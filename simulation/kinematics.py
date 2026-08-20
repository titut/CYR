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

BOT_SIZE_M = 0.75  # robot footprint width (m), matches simulator BOT_SIZE_M
BOT_RADIUS_M = BOT_SIZE_M / 2.0
WHEEL_RADIUS_M = 0.12
# Wheels are flush with the footprint, so their centres are inset by the wheel
# radius from each edge.
WHEEL_TRACK_M = BOT_SIZE_M - 2.0 * WHEEL_RADIUS_M

# Maximum commanded speeds (used by the controller for teleop and path
# following).  Kept in sync with the previous simulator constants.
BOT_LINEAR_SPEED_MPS = 3.0
BOT_ANGULAR_SPEED_RPS = 3.0


def unicycle_to_wheel(linear_mps: float, angular_rps: float) -> tuple[float, float]:
    """Convert unicycle velocity to differential wheel angular speeds.

    Args:
        linear_mps: forward linear velocity (m/s).
        angular_rps: angular velocity (rad/s), positive = counter-clockwise.

    Returns:
        (left_rps, right_rps) wheel angular speeds in rad/s, positive = forward.
    """
    v_left = linear_mps - angular_rps * WHEEL_TRACK_M / 2.0
    v_right = linear_mps + angular_rps * WHEEL_TRACK_M / 2.0
    return (v_left / WHEEL_RADIUS_M, v_right / WHEEL_RADIUS_M)


def wheel_to_unicycle(left_rps: float, right_rps: float) -> tuple[float, float]:
    """Convert differential wheel angular speeds to unicycle velocity.

    Args:
        left_rps: left wheel angular speed (rad/s), positive = forward.
        right_rps: right wheel angular speed (rad/s), positive = forward.

    Returns:
        (linear_mps, angular_rps) unicycle velocity.
    """
    v_left = left_rps * WHEEL_RADIUS_M
    v_right = right_rps * WHEEL_RADIUS_M
    return ((v_left + v_right) / 2.0, (v_right - v_left) / WHEEL_TRACK_M)


def square_footprint_radius(angle_rad: float) -> float:
    """Distance from centre to the edge of the square footprint in a direction.

    The footprint is a square of side ``BOT_SIZE_M`` aligned with the body frame
    (+x forward).  A ray at body-frame angle ``angle_rad`` reaches the boundary
    after this many metres.  It is smallest toward the face centres
    (``BOT_RADIUS_M``) and largest toward the corners
    (``BOT_RADIUS_M * sqrt(2)``), so the effective radius depends on angle.

    For a ray ``t * (cos φ, sin φ)`` the boundary is hit when
    ``t * max(|cos φ|, |sin φ|) == BOT_RADIUS_M``.
    """
    m = max(abs(math.cos(angle_rad)), abs(math.sin(angle_rad)))
    return BOT_RADIUS_M / m
