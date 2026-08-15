"""Zenoh controller node.

The motion controller: decides what velocity the robot should move at.  It
consumes the planned path (nav/path), the estimated pose (estimate/pose), and
raw teleop keys (sensor/wasd), and publishes unicycle velocity commands on
cmd/velocity.

Teleop takes priority: while any WASD key is held the robot is driven by the
keys and any in-progress path following is cancelled, matching the previous
simulator behaviour.

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
    BOT_RADIUS_M,
)

_LOOP_HZ = 50


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
        self._path_index = 0

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
            self._path_index = 0

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

    def _follow_path(self, x: float, y: float, theta: float) -> Tuple[float, float]:
        """Compute the velocity command to follow the next waypoint."""
        wp = self._path[self._path_index]
        to_wp = math.hypot(wp[0] - x, wp[1] - y)

        target_heading = math.atan2(wp[1] - y, wp[0] - x)
        angle_err = math.atan2(
            math.sin(target_heading - theta),
            math.cos(target_heading - theta),
        )

        if abs(angle_err) > 0.05:
            angular = max(
                -BOT_ANGULAR_SPEED_RPS,
                min(BOT_ANGULAR_SPEED_RPS, 4.0 * angle_err),
            )
            linear = 0.0
        else:
            angular = 4.0 * angle_err
            speed = BOT_LINEAR_SPEED_MPS
            if to_wp < BOT_RADIUS_M * 4:
                speed *= max(0.2, to_wp / (BOT_RADIUS_M * 4))
            linear = speed

        if to_wp < BOT_RADIUS_M * 1.5:
            self._path_index += 1
            if self._path_index >= len(self._path):
                self._path = []
                self._path_index = 0
                return 0.0, 0.0

        return linear, angular

    def _compute_command(self) -> Tuple[float, float]:
        with self._lock:
            keys = dict(self._keys)
            x, y, theta = self._est_x, self._est_y, self._est_theta

            # Teleop priority: cancel path following and drive from keys.
            if any(keys.values()):
                self._path = []
                self._path_index = 0
                return self._wasd_to_velocity(keys)

            if not self._path:
                return 0.0, 0.0

            return self._follow_path(x, y, theta)

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
