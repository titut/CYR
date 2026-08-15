"""Zenoh drive node.

Simulates the robot's motor controller and wheel encoders.  Subscribes to
unicycle velocity commands (cmd/velocity), converts them to differential wheel
speeds, applies first-order motor dynamics and encoder noise, and publishes the
measured wheel speeds (sensor/wheel_speed).

This node models the *hardware* only: the motor lag and encoder error.  Wheel
slip is a wheel-ground interaction and is modelled in the simulator's physics,
not here.

It also implements an emergency stop: it compares each LIDAR ray against the
square footprint at that angle and halts the wheels when the clearance between
the body edge and a detected surface drops below ``_ESTOP_CLEARANCE_M``.

Topics:
    Subscribes:  cmd/velocity       — {"linear_mps": float, "angular_rps": float}
                 sensor/lidar       — {"t": float, "rays": [{"angle_rad", "distance_m"}]}
    Publishes:   sensor/wheel_speed — {"left_rps": float, "right_rps": float, "t": float}

Usage:
    python zenoh/control/drive.py
"""

from __future__ import annotations

import json
import random
import sys
import threading
import time
from pathlib import Path

import zenoh

# Allow running this file directly or as a module.
_SCRIPT_DIR = Path(__file__).resolve().parent
_ZENOH_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _ZENOH_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_ZENOH_DIR) not in sys.path:
    sys.path.insert(0, str(_ZENOH_DIR))

from simulation.kinematics import square_footprint_radius, unicycle_to_wheel

# ---------------------------------------------------------------------------
# Motor + encoder simulation
# ---------------------------------------------------------------------------

_MOTOR_TIME_CONSTANT_S = 0.1  # first-order lag toward commanded wheel speed
_ENCODER_NOISE_RPS = 0.05  # 1σ Gaussian noise on measured wheel speed
_LOOP_HZ = 50
_CMD_TIMEOUT_S = 0.5  # coast to zero if no command arrives for this long

# LIDAR ranges are measured from the robot's centre.  The body is a square, so
# its radial extent varies with ray angle; the e-stop triggers when the minimum
# clearance between the body edge and a detected surface drops below this.
_ESTOP_CLEARANCE_M = 0.05


class Drive:
    def __init__(self):
        self._session = zenoh.open(zenoh.Config())

        self._pub_wheel = self._session.declare_publisher(
            "sensor/wheel_speed",
            congestion_control=zenoh.CongestionControl.DROP,
            reliability=zenoh.Reliability.BEST_EFFORT,
        )

        # Latest velocity command, guarded by _lock.
        self._lock = threading.Lock()
        self._cmd_linear = 0.0
        self._cmd_angular = 0.0
        self._last_cmd_time = time.time()

        # Latest minimum body-to-surface clearance (m), for the e-stop.  Defaults
        # to infinity (no reading) so the e-stop only triggers on real data.
        self._min_clearance_m = float("inf")
        self._estop_active = False

        # Simulated actual wheel speeds (rad/s), starting at rest.
        self._left_rps = 0.0
        self._right_rps = 0.0

        self._sub_cmd = self._session.declare_subscriber(
            "cmd/velocity", self._on_cmd
        )
        self._sub_lidar = self._session.declare_subscriber(
            "sensor/lidar", self._on_lidar
        )

    def _on_cmd(self, sample):
        """Store the latest velocity command."""
        try:
            data = json.loads(sample.payload.to_string())
            linear = float(data["linear_mps"])
            angular = float(data["angular_rps"])
        except (json.JSONDecodeError, KeyError, Exception):
            return
        with self._lock:
            self._cmd_linear = linear
            self._cmd_angular = angular
            self._last_cmd_time = time.time()

    def _on_lidar(self, sample):
        """Store the minimum body-to-surface clearance of the latest scan."""
        try:
            scan = json.loads(sample.payload.to_string())
            min_clearance = min(
                (
                    float(entry["distance_m"])
                    - square_footprint_radius(float(entry["angle_rad"]))
                    for entry in scan.get("rays", [])
                ),
                default=float("inf"),
            )
        except (json.JSONDecodeError, KeyError, Exception):
            return
        with self._lock:
            self._min_clearance_m = min_clearance

    def _step(self, dt: float):
        """Advance the motor simulation one step and publish the encoders."""
        with self._lock:
            linear = self._cmd_linear
            angular = self._cmd_angular
            if time.time() - self._last_cmd_time > _CMD_TIMEOUT_S:
                linear = 0.0
                angular = 0.0
            min_clearance_m = self._min_clearance_m

        # Emergency stop: body edge too close to a surface halts the wheels now.
        if min_clearance_m < _ESTOP_CLEARANCE_M:
            if not self._estop_active:
                self._estop_active = True
                print(
                    f"[drive] E-STOP: clearance {min_clearance_m:.3f} m "
                    f"(< {_ESTOP_CLEARANCE_M} m), halting."
                )
            self._left_rps = 0.0
            self._right_rps = 0.0
            self._pub_wheel.put(
                json.dumps({"left_rps": 0.0, "right_rps": 0.0, "t": time.time()})
            )
            return

        if self._estop_active:
            self._estop_active = False
            print("[drive] E-STOP cleared.")

        target_left, target_right = unicycle_to_wheel(linear, angular)

        # First-order motor dynamics toward the commanded wheel speeds.
        alpha = min(1.0, dt / _MOTOR_TIME_CONSTANT_S)
        self._left_rps += alpha * (target_left - self._left_rps)
        self._right_rps += alpha * (target_right - self._right_rps)

        # Encoder measurement noise.
        left = self._left_rps + random.gauss(0.0, _ENCODER_NOISE_RPS)
        right = self._right_rps + random.gauss(0.0, _ENCODER_NOISE_RPS)

        self._pub_wheel.put(json.dumps({"left_rps": left, "right_rps": right, "t": time.time()}))

    def run(self):
        print("[drive] Running. Press Ctrl+C to stop.")
        dt = 1.0 / _LOOP_HZ
        try:
            while True:
                time.sleep(dt)
                self._step(dt)
        except KeyboardInterrupt:
            print("[drive] Stopping…")
        finally:
            self._session.close()


def main():
    Drive().run()


if __name__ == "__main__":
    main()
