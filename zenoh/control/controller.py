"""Zenoh controller node.

The motion controller: decides what velocity the robot should move at.  It
consumes the planned path (nav/path), the estimated pose (estimate/pose), and
raw teleop keys (sensor/wasd), and publishes unicycle velocity commands on
cmd/velocity.

Path following uses *regulated pure pursuit*: a lookahead point is chosen ahead
of the robot on the path, the commanded curvature is 2·e/L² (e = lateral offset
of the lookahead point), and the linear speed is limited by centripetal
acceleration and by distance to the goal.  The resulting velocity command is
ramped through an accel/jerk-limited velocity profile, so it is smooth rather
than bang-bang.

Teleop takes priority: while any WASD key is held the robot is driven by the
keys and any in-progress path following is cancelled.

Topics:
    Subscribes:  nav/path       — planned waypoints [[x, y], ...] (meters)
                 estimate/pose  — {"x_m", "y_m", "theta_rad"}
                 sensor/wasd    — {"w", "a", "s", "d"} booleans
    Publishes:   cmd/velocity   — {"linear_mps": float, "angular_rps": float}

Usage:
    python zenoh/control/controller.py
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import List, Tuple

import zenoh

# Allow running this file directly or as a module.
_SCRIPT_DIR = Path(__file__).resolve().parent
_ZENOH_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _ZENOH_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_ZENOH_DIR) not in sys.path:
    sys.path.insert(0, str(_ZENOH_DIR))

from simulation.kinematics import (
    BOT_ANGULAR_SPEED_RPS,
    BOT_LINEAR_SPEED_MPS,
)

_LOOP_HZ = 50

# --- Regulated pure pursuit ---

# Lookahead distance (m): proportional to speed plus a floor, capped above.
# The floor is kept small so the robot tracks tight corners closely (a large
# minimum would make the pure-pursuit arc bulge wide and clip walls).
LOOKAHEAD_MIN_M = 0.15
LOOKAHEAD_GAIN_S = 0.35
LOOKAHEAD_MAX_M = 2.0

# Distance (m) ahead over which to scan the path for sharp corners, so the robot
# starts slowing down *before* a tight turn enters the (short) pure-pursuit
# lookahead.  Must be at least the braking distance for the top speed.
CURVATURE_HORIZON_M = 4.0

# A turn sharper than this angle (rad) is treated as a sharp corner: the pure
# pursuit carrot is stopped at the corner waypoint so the robot drives to the
# corner and turns there, rather than looking past it and cutting the corner.
SHARP_TURN_RAD = math.radians(30.0)

# The robot brakes so it would come to rest this far *past* a sharp corner.
# Braking to zero exactly *at* the corner deadlocks (it can never quite reach
# the corner to turn), so a small overshoot is reserved so the robot crosses the
# corner at low speed, turns, and only then stops.
CORNER_OVERSHOOT_M = 0.05

# Goal behaviour.
GOAL_RADIUS_M = 0.3  # declare arrival once the goal is this close

# Deceleration used by the braking-distance speed limit (m/s²).  The robot's
# speed is capped so it can always brake to a stop within the remaining distance
# to the goal: v <= sqrt(2 * a * d_goal).  Kept below MAX_LIN_ACCEL_MPS2 to
# leave margin for the jerk-limited velocity ramp (which can't apply full
# deceleration instantly).
GOAL_BRAKE_DECEL_MPS2 = 1.0

# Reaction time (s) added to the braking limit, covering the velocity-ramp lag,
# the drive's velocity loop, and pose-estimate staleness.  During this time the
# robot keeps moving at its current speed before deceleration takes effect, so
# the speed limit must reserve that distance too.
BRAKE_REACTION_S = 0.5

# Centripetal acceleration limit (m/s²): caps speed on curved paths so the robot
# doesn't skid or tip.
MAX_CENTRIPETAL_ACCEL_MPS2 = 0.8

# --- Velocity profile (time-parameterized, accel/jerk limited) ---
MAX_LIN_ACCEL_MPS2 = 1.2
MAX_LIN_JERK_MPS3 = 4.0
MAX_ANG_ACCEL_RPS2 = 4.0
MAX_ANG_JERK_RPS3 = 12.0


def _velocity_ramp(
    v: float,
    a: float,
    target: float,
    max_accel: float,
    max_jerk: float,
    dt: float,
) -> Tuple[float, float]:
    """Advance (velocity, acceleration) one step toward ``target``.

    The desired acceleration is proportional to the velocity error (a first-order
    velocity loop) and clamped to ``max_accel``; its rate of change is clamped to
    ``max_jerk``.  The result is a smooth S-curve velocity profile bounded by both
    accel and jerk limits.
    """
    a_des = (target - v) / 0.25  # ~0.25 s velocity-loop time constant
    a_des = max(-max_accel, min(max_accel, a_des))
    da = a_des - a
    da = max(-max_jerk * dt, min(max_jerk * dt, da))
    a = a + da
    v_new = v + a * dt
    # Snap to the target once essentially there, to avoid limit-cycle creep.
    if abs(target - v_new) < 0.005 and abs(a) < 0.5:
        v_new = target
        a = 0.0
    return v_new, a


def _angle_diff(a: float, b: float) -> float:
    """Signed angular difference a - b, wrapped to [-pi, pi]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _brake_speed_limit(d_eff: float, decel: float, reaction: float) -> float:
    """Max speed that can brake to zero within ``d_eff`` meters, given a
    ``reaction``-second lag before deceleration takes effect.

        v·T + v²/(2a) <= d_eff   →   v <= sqrt((aT)² + 2a·d_eff) - aT
    """
    d = max(0.0, d_eff)
    return math.sqrt((decel * reaction) ** 2 + 2.0 * decel * d) - decel * reaction


class Controller:
    def __init__(self):
        self._session = zenoh.open(zenoh.Config())

        self._pub_cmd = self._session.declare_publisher(
            "cmd/velocity",
            congestion_control=zenoh.CongestionControl.DROP,
            reliability=zenoh.Reliability.BEST_EFFORT,
        )

        # All shared state is guarded by a single lock.
        self._lock = threading.Lock()

        # Latest estimated pose.
        self._est_x = 0.0
        self._est_y = 0.0
        self._est_theta = 0.0

        # Current path and progress along it.
        self._path: List[Tuple[float, float]] = []

        # Velocity-profile state (accel/jerk-limited ramp), linear and angular.
        self._lin_v = 0.0
        self._lin_a = 0.0
        self._ang_v = 0.0
        self._ang_a = 0.0

        # Latest teleop key state.
        self._keys = {"w": False, "a": False, "s": False, "d": False}

        self._sub_pose = self._session.declare_subscriber(
            "estimate/pose", self._on_pose
        )
        self._sub_path = self._session.declare_subscriber("nav/path", self._on_path)
        self._sub_wasd = self._session.declare_subscriber(
            "sensor/wasd", self._on_wasd
        )

    # -------------------------------------------------------------------
    # Zenoh callbacks
    # -------------------------------------------------------------------

    def _on_pose(self, sample):
        try:
            data = json.loads(sample.payload.to_string())
            x = float(data["x_m"])
            y = float(data["y_m"])
            theta = float(data["theta_rad"])
        except (json.JSONDecodeError, KeyError, Exception):
            return
        with self._lock:
            self._est_x, self._est_y, self._est_theta = x, y, theta

    def _on_path(self, sample):
        try:
            waypoints = json.loads(sample.payload.to_string())
            path = [tuple(p) for p in waypoints]
        except (json.JSONDecodeError, Exception):
            return
        with self._lock:
            self._path = path

    def _on_wasd(self, sample):
        try:
            data = json.loads(sample.payload.to_string())
        except (json.JSONDecodeError, Exception):
            return
        with self._lock:
            self._keys = {
                "w": bool(data.get("w", False)),
                "a": bool(data.get("a", False)),
                "s": bool(data.get("s", False)),
                "d": bool(data.get("d", False)),
            }

    # -------------------------------------------------------------------
    # Control
    # -------------------------------------------------------------------

    @staticmethod
    def _wasd_to_velocity(keys: dict) -> Tuple[float, float]:
        linear = 0.0
        if keys.get("w"):
            linear += BOT_LINEAR_SPEED_MPS
        if keys.get("s"):
            linear -= BOT_LINEAR_SPEED_MPS
        angular = 0.0
        if keys.get("a"):
            angular -= BOT_ANGULAR_SPEED_RPS
        if keys.get("d"):
            angular += BOT_ANGULAR_SPEED_RPS
        return linear, angular

    def _closest_point_on_path(
        self, path: List[Tuple[float, float]], x: float, y: float
    ) -> Tuple[int, float]:
        """Return (seg_idx, t) of the polyline point closest to (x, y)."""
        best: Tuple[float, int, float] | None = None
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            wx, wy = x2 - x1, y2 - y1
            length_sq = wx * wx + wy * wy
            t = (
                0.0
                if length_sq == 0
                else max(0.0, min(1.0, ((x - x1) * wx + (y - y1) * wy) / length_sq))
            )
            px = x1 + t * wx
            py = y1 + t * wy
            d2 = (px - x) ** 2 + (py - y) ** 2
            if best is None or d2 < best[0]:
                best = (d2, i, t)
        if best is None:
            return 0, 0.0
        return best[1], best[2]

    def _lookahead_point(
        self,
        path: List[Tuple[float, float]],
        seg_idx: int,
        t: float,
        lookahead: float,
    ) -> Tuple[float, float]:
        """Return the point on the path ``lookahead`` meters (arc length) ahead
        of the projection (seg_idx, t), stopping at any sharp corner instead of
        looking past it (which would make pure pursuit cut the corner)."""
        if len(path) < 2:
            return path[-1] if path else (0.0, 0.0)

        remaining = lookahead
        i = seg_idx
        cur_t = t
        while i < len(path) - 1:
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            heading = math.atan2(y2 - y1, x2 - x1)

            # If there's a sharp turn at waypoint i+1, stop the carrot there.
            if i + 2 < len(path):
                nx1, ny1 = path[i + 1]
                nx2, ny2 = path[i + 2]
                next_heading = math.atan2(ny2 - ny1, nx2 - nx1)
                if abs(_angle_diff(next_heading, heading)) > SHARP_TURN_RAD:
                    return (x2, y2)

            seg_len = math.hypot(x2 - x1, y2 - y1)
            ahead = (1.0 - cur_t) * seg_len
            if remaining <= ahead:
                s = cur_t + (remaining / seg_len if seg_len > 0 else 0.0)
                s = min(1.0, s)
                return (x1 + s * (x2 - x1), y1 + s * (y2 - y1))
            remaining -= ahead
            i += 1
            cur_t = 0.0
        return path[-1]

    def _first_corner_ahead(
        self,
        path: List[Tuple[float, float]],
        seg_idx: int,
        t: float,
        horizon: float,
    ) -> Tuple[Optional[float], float]:
        """Return (arc-length distance, turn angle) of the first sharp turn
        within ``horizon`` meters of the projection, or (None, 0.0)."""
        if len(path) < 3:
            return None, 0.0

        dist = 0.0
        i = seg_idx
        cur_t = t
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        prev_heading = math.atan2(y2 - y1, x2 - x1)

        while i < len(path) - 2 and dist < horizon:
            nx1, ny1 = path[i + 1]
            nx2, ny2 = path[i + 2]
            heading = math.atan2(ny2 - ny1, nx2 - nx1)
            delta = abs(_angle_diff(heading, prev_heading))
            seg_len = math.hypot(x2 - x1, y2 - y1)
            dist_to_corner = dist + (1.0 - cur_t) * seg_len
            if delta > SHARP_TURN_RAD:
                return dist_to_corner, delta
            prev_heading = heading
            dist = dist_to_corner
            cur_t = 0.0
            i += 1
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
        return None, 0.0

    def _follow_path(
        self, x: float, y: float, theta: float, current_speed: float
    ) -> Tuple[float, float]:
        """Regulated pure pursuit: track the path via a lookahead point.

        Returns the desired (linear, angular) velocity.  Curvature is 2·e/L²
        (e = signed lateral offset of the lookahead point in the robot frame),
        and the linear speed is limited by (a) the centripetal acceleration on
        the current arc, (b) sharp corners detected ahead of the robot, and
        (c) the remaining braking distance to the goal.  If the required turn
        rate exceeds the robot's limit, the linear speed is scaled down rather
        than cutting the corner.
        """
        path = self._path
        if not path:
            return 0.0, 0.0

        goal_dist = math.hypot(path[-1][0] - x, path[-1][1] - y)
        if goal_dist < GOAL_RADIUS_M:
            self._path = []
            return 0.0, 0.0

        lookahead = min(
            LOOKAHEAD_MAX_M, LOOKAHEAD_MIN_M + LOOKAHEAD_GAIN_S * current_speed
        )

        # Projection of the robot onto the path.
        seg_idx, t = self._closest_point_on_path(path, x, y)
        # Roll over to the next segment when the projection is at a waypoint, so
        # a robot that has just passed a corner looks ahead on the *new* segment
        # rather than being stuck behind the corner waypoint.
        if t > 0.999 and seg_idx < len(path) - 2:
            seg_idx += 1
            t = 0.0
        lx, ly = self._lookahead_point(path, seg_idx, t, lookahead)

        dx = lx - x
        dy = ly - y
        # Signed lateral offset of the lookahead point in the robot frame.
        e_lat = -dx * math.sin(theta) + dy * math.cos(theta)
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return 0.0, 0.0

        curvature = 2.0 * e_lat / (dist * dist)

        # (a) Desired speed: capped by centripetal acceleration on the arc.
        v_des = BOT_LINEAR_SPEED_MPS
        if abs(curvature) > 1e-6:
            v_des = min(v_des, math.sqrt(MAX_CENTRIPETAL_ACCEL_MPS2 / abs(curvature)))

        # (b) Slow down for a sharp corner detected ahead: brake to (near) zero
        # by the time the robot reaches the corner, so it turns in place there
        # rather than cutting it.
        corner_dist, _ = self._first_corner_ahead(
            path, seg_idx, t, CURVATURE_HORIZON_M
        )
        if corner_dist is not None:
            v_des = min(
                v_des,
                _brake_speed_limit(
                    corner_dist + CORNER_OVERSHOOT_M,
                    GOAL_BRAKE_DECEL_MPS2,
                    BRAKE_REACTION_S,
                ),
            )

        # (c) Braking-distance limit with reaction-time compensation, so the
        # robot is (essentially) stopped by the time it reaches the goal radius.
        v_des = min(
            v_des,
            _brake_speed_limit(
                goal_dist - GOAL_RADIUS_M, GOAL_BRAKE_DECEL_MPS2, BRAKE_REACTION_S
            ),
        )

        w_des = v_des * curvature
        # If the required turn rate exceeds the limit, scale the speed down so
        # the robot still follows the arc instead of cutting the corner.
        if abs(w_des) > BOT_ANGULAR_SPEED_RPS:
            v_des = BOT_ANGULAR_SPEED_RPS / abs(curvature)
            w_des = math.copysign(BOT_ANGULAR_SPEED_RPS, w_des)

        return v_des, w_des

    def _compute_command(self) -> Tuple[float, float]:
        with self._lock:
            keys = dict(self._keys)
            x, y, theta = self._est_x, self._est_y, self._est_theta

            if any(keys.values()):
                # Teleop priority: cancel path following and drive from keys.
                self._path = []
                target_lin, target_ang = self._wasd_to_velocity(keys)
            elif not self._path:
                target_lin, target_ang = 0.0, 0.0
            else:
                target_lin, target_ang = self._follow_path(x, y, theta, self._lin_v)

        # Ramp the commanded velocity toward the target with accel/jerk limits,
        # producing a smooth velocity profile.
        dt = 1.0 / _LOOP_HZ
        self._lin_v, self._lin_a = _velocity_ramp(
            self._lin_v, self._lin_a, target_lin,
            MAX_LIN_ACCEL_MPS2, MAX_LIN_JERK_MPS3, dt,
        )
        self._ang_v, self._ang_a = _velocity_ramp(
            self._ang_v, self._ang_a, target_ang,
            MAX_ANG_ACCEL_RPS2, MAX_ANG_JERK_RPS3, dt,
        )
        return self._lin_v, self._ang_v

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------

    def run(self):
        print("[controller] Running. Press Ctrl+C to stop.")
        dt = 1.0 / _LOOP_HZ
        try:
            while True:
                time.sleep(dt)
                linear, angular = self._compute_command()
                self._pub_cmd.put(
                    json.dumps({"linear_mps": linear, "angular_rps": angular})
                )
        except KeyboardInterrupt:
            print("[controller] Stopping…")
        finally:
            self._session.close()


def main():
    Controller().run()


if __name__ == "__main__":
    main()
