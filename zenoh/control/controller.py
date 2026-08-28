"""Zenoh controller node.

The motion controller: decides what velocity the robot should move at.  It
consumes the planned path (nav/path), the estimated pose (estimate/pose), and
raw teleop keys (sensor/wasd), and publishes unicycle velocity commands on
cmd/velocity.

Path following uses *regulated pure pursuit*: a lookahead point is chosen ahead
of the robot on the path, the commanded curvature is 2·e/L² (e = lateral offset
of the lookahead point), and the linear speed is limited by centripetal
acceleration and by distance to the goal.  At sharp corners the robot brakes
and then rotates *in place* onto the next segment (pure pursuit's angular rate
w = v·curvature would vanish at zero speed).  Bang-bang recovery takes over
when the robot strays too far from the path.

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

import logging
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

from core.clock import sleep_until
from core.messages import SchemaError, decode, decode_path, encode
from core.robot_config import get_robot_config

_LOOP_HZ = 50

# --- Regulated pure pursuit ---

# Lookahead distance (m): proportional to speed plus a floor, capped above.
# Longer lookahead = lower curvature gain = less lateral overshoot / weaving.
# Sharp corners are handled separately (SHARP_TURN_RAD / corner braking), so a
# long lookahead doesn't make the robot cut corners.
LOOKAHEAD_MIN_M = 0.2
LOOKAHEAD_GAIN_S = 0.2
LOOKAHEAD_MAX_M = 3.0

# Distance (m) ahead over which to scan the path for sharp corners, so the robot
# starts slowing down *before* a tight turn enters the pure-pursuit lookahead.
# Must be at least the braking distance for the top speed.
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
# to the goal.  The drive node applies the actual deceleration (it owns the
# low-level velocity loop).
GOAL_BRAKE_DECEL_MPS2 = 1.0

# Reaction time (s) added to the braking limit, covering the drive's velocity
# loop and pose-estimate staleness.  During this time the robot keeps moving at
# its current speed before deceleration takes effect.
BRAKE_REACTION_S = 0.7

# Centripetal acceleration limit (m/s²): caps speed on curved paths so the robot
# doesn't skid or tip.  Lower = slower through turns = less overshoot.
MAX_CENTRIPETAL_ACCEL_MPS2 = 0.6

# --- Bang-bang recovery ---
# When the robot is far off the path, pure pursuit is too gentle (it corrects
# gradually and can weave).  Past this threshold we switch to a decisive
# bang-bang controller: turn in place toward the path, then creep back onto it.

# Lateral deviation from the path (m) that triggers bang-bang recovery.
OFF_PATH_BANG_BANG_M = 0.1
# Heading error (rad) below which bang-bang drives instead of rotating in place.
BANG_TURN_DEADBAND_RAD = 0.3
# Recovery is about getting back ON the path, not making progress, so the
# forward speed is kept low.  Driving at top speed here is what sent the robot
# barrelling into a wall instead of braking for a corner.
BANG_RECOVERY_SPEED_MPS = 0.5
# Max heading error (rad) at which bang-bang still drives forward while
# correcting the heading.  An offset is usually caused by a small unaccounted
# yaw, so we counter that yaw in motion — turning up to ~45° toward the next
# target while creeping forward — rather than pivoting 90° back onto the path.
# Only when pointed more than this far off does the robot rotate in place first.
BANG_MAX_TURN_RAD = math.radians(45.0)
# Proportional turn gain: angular velocity = TURN_GAIN * heading error, capped at
# BOT_ANGULAR_SPEED_RPS.  Used for recovery turns so rotation is smooth
# (proportional), not full-on/off.
TURN_GAIN = 4.0

# --- In-place corner turning ---
# Pure pursuit's angular rate is w = v·curvature, so it vanishes as the
# corner-braking drops v to ~0 right where the robot needs to rotate.  Once the
# robot is parked near a sharp-corner waypoint and moving slowly, it instead
# rotates in place onto the outgoing segment (proportional heading control),
# then resumes pure pursuit.

# Parked within this distance (m) of the corner waypoint before turning in place.
CORNER_TURN_RADIUS_M = 0.4
# Only rotate in place when moving this slowly (m/s); otherwise keep driving
# through the corner-braking approach.
CORNER_TURN_MAX_SPEED_MPS = 0.5
# Max in-place rotation rate at a corner (rad/s).  Rotating at the robot's full
# angular limit (~3 rad/s) overshoots the target heading badly once the pose
# estimate and drive response lag, so the corner turn is deliberately slower.
CORNER_TURN_MAX_RPS = 1.2
# Estimated angular rate (rad/s) below which the corner rotation counts as
# settled.  The turn is only released once the heading is inside the deadband
# AND the rotation has actually stopped, so the robot does not drive off while
# still swinging.
CORNER_SETTLE_OMEGA_RPS = 0.5
# Forward speed (m/s) right after leaving a corner turn, so the robot creeps
# onto the outgoing segment instead of lurching forward while it settles.
CORNER_RESUME_SPEED_MPS = 0.5


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


def corner_ahead(
    path: List[Tuple[float, float]], seg_idx: int
) -> Optional[Tuple[Tuple[float, float], float]]:
    """Return (waypoint, outgoing heading) of the first sharp corner at or
    ahead of the projection, or None if there is none.

    Checks the corner at the *start* of the current segment (the robot has
    just rolled over onto the outgoing segment) first, then any sharp corner
    at the end of the current segment or further ahead.
    """
    if len(path) < 3:
        return None

    # Corner at the start of the current segment.
    if seg_idx >= 1 and seg_idx + 1 < len(path):
        x1, y1 = path[seg_idx - 1]
        x2, y2 = path[seg_idx]
        incoming = math.atan2(y2 - y1, x2 - x1)
        nx1, ny1 = path[seg_idx]
        nx2, ny2 = path[seg_idx + 1]
        outgoing = math.atan2(ny2 - ny1, nx2 - nx1)
        if abs(_angle_diff(outgoing, incoming)) > SHARP_TURN_RAD:
            return path[seg_idx], outgoing

    # Corner at the end of the current segment, or further ahead.
    i = seg_idx
    while i < len(path) - 2:
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        incoming = math.atan2(y2 - y1, x2 - x1)
        nx1, ny1 = path[i + 1]
        nx2, ny2 = path[i + 2]
        outgoing = math.atan2(ny2 - ny1, nx2 - nx1)
        if abs(_angle_diff(outgoing, incoming)) > SHARP_TURN_RAD:
            return path[i + 1], outgoing
        i += 1
    return None


def near_sharp_corner(
    path: List[Tuple[float, float]], seg_idx: int, x: float, y: float
) -> bool:
    """True if the robot is within ``CORNER_TURN_RADIUS_M`` of the next
    sharp-corner waypoint (used to keep the speed low right after a turn)."""
    corner = corner_ahead(path, seg_idx)
    if corner is None:
        return False
    waypoint, _ = corner
    return math.hypot(waypoint[0] - x, waypoint[1] - y) < CORNER_TURN_RADIUS_M


def turn_at_corner(
    path: List[Tuple[float, float]],
    seg_idx: int,
    x: float,
    y: float,
    theta: float,
    est_omega: float,
) -> Optional[Tuple[float, float]]:
    """If the robot is parked at a sharp corner, return (0.0, w) to rotate
    in place onto the outgoing segment; otherwise return None so pure
    pursuit proceeds.

    The rotation rate is capped at ``CORNER_TURN_MAX_RPS`` (rotating at the
    robot's full angular limit overshoots once the pose estimate / drive
    response lag).  It is only released when the heading is inside the
    deadband *and* the estimated angular rate has settled, so the robot does
    not drive off while still swinging.
    """
    corner = corner_ahead(path, seg_idx)
    if corner is None:
        return None
    waypoint, out_heading = corner

    if math.hypot(waypoint[0] - x, waypoint[1] - y) >= CORNER_TURN_RADIUS_M:
        return None  # not parked at the corner yet

    err = _angle_diff(out_heading, theta)
    if abs(err) <= BANG_TURN_DEADBAND_RAD and abs(est_omega) <= CORNER_SETTLE_OMEGA_RPS:
        return None  # aimed and settled -> resume pure pursuit

    w = max(
        -CORNER_TURN_MAX_RPS,
        min(CORNER_TURN_MAX_RPS, TURN_GAIN * err),
    )
    return (0.0, w)


class Controller:
    def __init__(self):
        self._session = zenoh.open(zenoh.Config())

        # Robot description (T-019): max speeds from robot.yaml.
        cfg = get_robot_config()
        self._max_linear_mps = cfg.chassis.linear_speed_mps
        self._max_angular_rps = cfg.chassis.angular_speed_rps

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

        # Monotonic time until which the robot must hold position (a
        # localization jump was detected in the pose stream).
        self._halt_until: float = 0.0

        # Current path and progress along it.
        self._path: List[Tuple[float, float]] = []

        # Estimated robot speed (m/s) and angular rate (rad/s), derived from
        # consecutive pose updates and low-pass filtered.  Speed adapts the
        # pure-pursuit lookahead; the angular rate is used to detect that an
        # in-place corner turn has settled.  The controller does NOT ramp
        # velocity — that is the drive's job.
        self._est_speed = 0.0
        self._est_omega = 0.0
        self._last_pose_x = 0.0
        self._last_pose_y = 0.0
        self._last_pose_theta = 0.0
        self._last_pose_time: Optional[float] = None

        # Latest teleop key state.
        self._keys = {"w": False, "a": False, "s": False, "d": False}

        self._sub_pose = self._session.declare_subscriber(
            "estimate/pose", self._on_pose
        )
        self._sub_halt = self._session.declare_subscriber(
            "estimate/halt", self._on_halt
        )
        self._sub_path = self._session.declare_subscriber("nav/path", self._on_path)
        self._sub_wasd = self._session.declare_subscriber("sensor/wasd", self._on_wasd)

    # -------------------------------------------------------------------
    # Zenoh callbacks
    # -------------------------------------------------------------------

    def _on_pose(self, sample):
        try:
            data = decode("estimate/pose", sample)
            x = float(data["x_m"])
            y = float(data["y_m"])
            theta = float(data["theta_rad"])
        except SchemaError as exc:
            logging.warning("estimate/pose dropped: %s", exc)
            return
        with self._lock:
            self._est_x, self._est_y, self._est_theta = x, y, theta

            # Estimate speed / angular rate from consecutive pose updates.
            now = time.monotonic()
            if self._last_pose_time is not None:
                dt = now - self._last_pose_time
                if 0.0 < dt < 0.5:
                    measured = (
                        math.hypot(x - self._last_pose_x, y - self._last_pose_y) / dt
                    )
                    self._est_speed = 0.85 * self._est_speed + 0.15 * measured
                    measured_omega = _angle_diff(theta, self._last_pose_theta) / dt
                    self._est_omega = 0.85 * self._est_omega + 0.15 * measured_omega
            self._last_pose_x, self._last_pose_y = x, y
            self._last_pose_theta = theta
            self._last_pose_time = now

    def _on_halt(self, sample):
        """Localization-jump halt command from the pose estimator: hold position
        for the requested duration so the corrected pose can settle."""
        try:
            data = decode("estimate/halt", sample)
            hold_s = float(data["hold_s"])
        except SchemaError as exc:
            logging.warning("estimate/halt dropped: %s", exc)
            return
        with self._lock:
            self._halt_until = time.monotonic() + hold_s
            logging.warning("estimate/halt received — holding for %.2f s.", hold_s)

    def _on_path(self, sample):
        try:
            waypoints = decode_path("nav/path", sample)
            path = [tuple(p) for p in waypoints]
        except SchemaError as exc:
            logging.warning("nav/path dropped: %s", exc)
            return
        with self._lock:
            self._path = path

    def _on_wasd(self, sample):
        try:
            data = decode("sensor/wasd", sample)
        except SchemaError as exc:
            logging.warning("sensor/wasd dropped: %s", exc)
            return
        with self._lock:
            self._keys = {
                "w": bool(data.get("w", False)),
                "a": bool(data.get("a", False)),
                "s": bool(data.get("s", False)),
                "d": bool(data.get("d", False)),
            }

    def _closest_path_distance(self, path, x: float, y: float) -> float:
        """Perpendicular distance from (x, y) to the path polyline."""
        best_d2 = float("inf")
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
            if d2 < best_d2:
                best_d2 = d2
        if len(path) == 1:
            best_d2 = (path[0][0] - x) ** 2 + (path[0][1] - y) ** 2
        return math.sqrt(max(0.0, best_d2))

    # -------------------------------------------------------------------
    # Control
    # -------------------------------------------------------------------

    def _wasd_to_velocity(self, keys: dict) -> Tuple[float, float]:
        linear = 0.0
        if keys.get("w"):
            linear += self._max_linear_mps
        if keys.get("s"):
            linear -= self._max_linear_mps
        angular = 0.0
        if keys.get("a"):
            angular -= self._max_angular_rps
        if keys.get("d"):
            angular += self._max_angular_rps
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

        # At a sharp corner, w = v·curvature would be ~0 (v was braked to ~0),
        # so rotate in place onto the next segment instead of creeping around it.
        if current_speed < CORNER_TURN_MAX_SPEED_MPS:
            corner_turn = turn_at_corner(
                path, seg_idx, x, y, theta, self._est_omega
            )
            if corner_turn is not None:
                return corner_turn

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
        v_des = self._max_linear_mps
        if abs(curvature) > 1e-6:
            v_des = min(v_des, math.sqrt(MAX_CENTRIPETAL_ACCEL_MPS2 / abs(curvature)))

        # (b) Slow down for a sharp corner detected ahead: brake to (near) zero
        # by the time the robot reaches the corner, so it turns in place there
        # rather than cutting it.
        corner_dist, _ = self._first_corner_ahead(path, seg_idx, t, CURVATURE_HORIZON_M)
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

        # (d) Right after an in-place corner turn, creep forward instead of
        # lurching off at the corner-brake speed while the heading settles.
        if near_sharp_corner(path, seg_idx, x, y):
            v_des = min(v_des, CORNER_RESUME_SPEED_MPS)

        w_des = v_des * curvature
        # If the required turn rate exceeds the limit, scale the speed down so
        # the robot still follows the arc instead of cutting the corner.
        if abs(w_des) > self._max_angular_rps:
            v_des = self._max_angular_rps / abs(curvature)
            w_des = math.copysign(self._max_angular_rps, w_des)

        return v_des, w_des

    def _compute_command(self) -> Tuple[float, float]:
        """Decide the desired velocity and publish it directly.

        No velocity ramp here: the controller emits the *target* velocity, and
        the low-level drive node applies the accel/decel limits (that is a
        firmware concern, not a navigation one).  This removes a whole layer of
        lag that caused weaving and sluggish stops.
        """
        with self._lock:
            keys = dict(self._keys)
            x, y, theta = self._est_x, self._est_y, self._est_theta
            halt = time.monotonic() < self._halt_until

            if any(keys.values()):
                # Teleop priority: cancel path following and drive from keys.
                self._path = []
                target_lin, target_ang = self._wasd_to_velocity(keys)
            elif halt:
                # A localization jump was just absorbed: hold position and keep
                # the path so motion resumes once the pose has settled.
                target_lin, target_ang = 0.0, 0.0
            elif not self._path:
                target_lin, target_ang = 0.0, 0.0
            else:
                # Bang-bang recovery takes priority over pure pursuit: recover
                # decisively when far off the path, otherwise follow smoothly.
                if self._closest_path_distance(self._path, x, y) > OFF_PATH_BANG_BANG_M:
                    target_lin, target_ang = self._recover_bang_bang(x, y, theta)
                else:
                    target_lin, target_ang = self._follow_path(
                        x, y, theta, self._est_speed
                    )

        return target_lin, target_ang

    # -------------------------------------------------------------------
    # Bang-bang recovery
    # -------------------------------------------------------------------

    def _recover_bang_bang(
        self, x: float, y: float, theta: float
    ) -> Tuple[float, float]:
        """Decisively get back on the path, using proportional heading control.

        Used when the robot has drifted far off the path (where pure pursuit
        corrects too gently).  An offset is caused by an unaccounted yaw error,
        so recovery counters that yaw toward the *next target* (a lookahead
        point on the path) while still moving forward — the turn is proportional
        to the heading error and capped around 45°, not a 90° pivot back onto
        the perpendicular.  Only when pointed more than that far off does it
        rotate in place first.
        """
        path = self._path
        seg_idx, t = self._closest_point_on_path(path, x, y)
        lx, ly = self._lookahead_point(path, seg_idx, t, LOOKAHEAD_MIN_M)
        target_heading = math.atan2(ly - y, lx - x)
        err = _angle_diff(target_heading, theta)
        angular = max(
            -self._max_angular_rps,
            min(self._max_angular_rps, TURN_GAIN * err),
        )
        if abs(err) > BANG_MAX_TURN_RAD:
            return 0.0, angular  # way off heading: correct the yaw in place
        return BANG_RECOVERY_SPEED_MPS, angular  # counter the yaw while driving on

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------

    def run(self):
        print("[controller] Running. Press Ctrl+C to stop.")
        period = 1.0 / _LOOP_HZ
        next_tick = time.monotonic()
        try:
            while True:
                linear, angular = self._compute_command()
                self._pub_cmd.put(
                    encode("cmd/velocity", {"linear_mps": linear, "angular_rps": angular})
                )
                # Deadline-driven pacing: no cumulative drift from sleep jitter.
                next_tick += period
                sleep_until(next_tick)
        except KeyboardInterrupt:
            print("[controller] Stopping…")
        finally:
            self._session.close()


def main():
    Controller().run()


if __name__ == "__main__":
    main()
