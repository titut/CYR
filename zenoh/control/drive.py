"""Zenoh drive node.

Simulates the robot's motor controller and wheel encoders.  Subscribes to
unicycle velocity commands (cmd/velocity), converts them to differential wheel
speeds, applies a torque-limited first-order velocity-loop model (with encoder
noise), and publishes the measured wheel speeds (sensor/wheel_speed).

This node models the *hardware* only: the velocity-loop response, its
acceleration (torque) limit, and encoder error.  Velocity-loop PID itself lives
in the motor driver firmware (see T-019 / the HAL); this node approximates that
closed loop as a first-order system with an acceleration cap.  Wheel slip is a
wheel-ground interaction and is modelled in the simulator's physics, not here.

It also implements a layered safety response (T-016) keyed on the minimum
body-to-surface clearance (from LIDAR, each ray minus the square footprint at
that angle):

    slow_down zone  (< 1.0 m)   linear speed is scaled down with distance
    stop zone       (< 0.15 m)  forward motion is halted (slow reverse allowed)
    latched e-stop  (< 0.05 m)  wheels held at zero until an explicit
                                safety/reset — never auto-cleared

On hardware, the stop tier is the safety function and the latched e-stop is its
safe state; both belong on a safety-rated channel (IEC 61508 / ISO 3691-4)
separate from the best-effort navigation path, and the latch must be re-armed
manually.  This node models that behaviour; the physical safety-rated channel
is a HAL concern (T-019).

Topics:
    Subscribes:  cmd/velocity       — {"linear_mps": float, "angular_rps": float}
                 sensor/lidar       — {"t": float, "rays": [{"angle_rad", "distance_m"}]}
                 safety/reset       — any payload: clears a latched e-stop
    Publishes:   sensor/wheel_speed — {"left_rps": float, "right_rps": float, "t": float}
                 safety/status      — {"state": str, "min_clearance_m": float, "t": float}

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

from simulation.kinematics import (
    BOT_LINEAR_SPEED_MPS,
    square_footprint_radius,
    unicycle_to_wheel,
)
from clock import sleep_until

# ---------------------------------------------------------------------------
# Motor + encoder simulation
# ---------------------------------------------------------------------------

_MOTOR_TIME_CONSTANT_S = 0.05  # velocity-loop time constant toward the target
_MAX_WHEEL_ACCEL_RPS2 = 35.0  # torque-limited max wheel acceleration (rad/s²)
_MAX_WHEEL_DECEL_RPS2 = 70.0  # harder wheel deceleration, for crisp stops
_ENCODER_NOISE_RPS = 0.05  # 1σ Gaussian noise on measured wheel speed
_LOOP_HZ = 50
_CMD_TIMEOUT_S = 0.5  # coast to zero if no command arrives for this long

# ---------------------------------------------------------------------------
# Safety zones (T-016)
# ---------------------------------------------------------------------------
# LIDAR ranges are measured from the robot's centre.  The body is a square, so
# its radial extent varies with ray angle; clearances are body-edge to surface.
# Three tiers keyed on the minimum body clearance:
#   slow_down (< _SLOW_DOWN_CLEARANCE_M)  linear speed scaled down with distance
#   stop      (< _STOP_CLEARANCE_M)       forward motion halted (slow reverse ok)
#   e-stop    (< _ESTOP_CLEARANCE_M)      latched: wheels at zero until reset
_SLOW_DOWN_CLEARANCE_M = 1.0
_STOP_CLEARANCE_M = 0.15
_ESTOP_CLEARANCE_M = 0.05
# Slow reverse (m/s) allowed inside the stop zone so the robot can back away.
_REVERSE_ESCAPE_MPS = 0.5


def _velocity_step(
    current: float,
    target: float,
    dt: float,
    time_constant_s: float = _MOTOR_TIME_CONSTANT_S,
    max_accel_rps2: float = _MAX_WHEEL_ACCEL_RPS2,
    max_decel_rps2: float = _MAX_WHEEL_DECEL_RPS2,
) -> float:
    """Advance ``current`` toward ``target`` with first-order dynamics, capped
    by per-step acceleration/deceleration limits.

    This models a velocity-controlled driver (velocity-loop PID in firmware,
    approximated as a first-order response) with a finite torque limit: large
    step changes ramp linearly at ``max_accel_rps2`` (and brake at
    ``max_decel_rps2``, which is higher so stops are crisp), while small changes
    decay with ``time_constant_s``.
    """
    alpha = min(1.0, dt / time_constant_s)
    step = alpha * (target - current)
    if step > max_accel_rps2 * dt:
        step = max_accel_rps2 * dt
    elif step < -max_decel_rps2 * dt:
        step = -max_decel_rps2 * dt
    return current + step


class Drive:
    def __init__(self):
        self._session = zenoh.open(zenoh.Config())

        self._pub_wheel = self._session.declare_publisher(
            "sensor/wheel_speed",
            congestion_control=zenoh.CongestionControl.DROP,
            reliability=zenoh.Reliability.BEST_EFFORT,
        )
        self._pub_status = self._session.declare_publisher(
            "safety/status",
            congestion_control=zenoh.CongestionControl.DROP,
            reliability=zenoh.Reliability.BEST_EFFORT,
        )

        # Latest velocity command, guarded by _lock.
        self._lock = threading.Lock()
        self._cmd_linear = 0.0
        self._cmd_angular = 0.0
        self._last_cmd_time = time.monotonic()

        # Latest minimum body-to-surface clearance (m).  Defaults to infinity
        # (no reading) so the safety zones only act on real data.
        self._min_clearance_m = float("inf")
        # Latched e-stop: once set, wheels stay at zero until safety/reset.
        self._estop_latched = False
        # Last published safety state, so safety/status is only sent on change.
        self._last_status_state = ""

        # Simulated actual wheel speeds (rad/s), starting at rest.
        self._left_rps = 0.0
        self._right_rps = 0.0

        self._sub_cmd = self._session.declare_subscriber(
            "cmd/velocity", self._on_cmd
        )
        self._sub_lidar = self._session.declare_subscriber(
            "sensor/lidar", self._on_lidar
        )
        self._sub_reset = self._session.declare_subscriber(
            "safety/reset", self._on_reset
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
            self._last_cmd_time = time.monotonic()

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

    def _on_reset(self, sample):
        """Clear a latched e-stop.  The command is zeroed so the robot does not
        lurch back onto the command that caused the stop."""
        with self._lock:
            if self._estop_latched:
                self._estop_latched = False
                print("[drive] E-STOP cleared by safety/reset.")
            self._cmd_linear = 0.0
            self._cmd_angular = 0.0

    @staticmethod
    def _safety_limited_linear(linear: float, clearance: float) -> float:
        """Cap the commanded linear speed by proximity to a surface.

        Above the slow-down zone the command passes through.  Inside it the
        speed is scaled linearly down to zero at the stop boundary, and inside
        the stop zone forward motion is halted while a slow reverse is allowed
        so the robot can back away.
        """
        if clearance >= _SLOW_DOWN_CLEARANCE_M:
            return linear
        if clearance > _STOP_CLEARANCE_M:
            limit = (
                BOT_LINEAR_SPEED_MPS
                * (clearance - _STOP_CLEARANCE_M)
                / (_SLOW_DOWN_CLEARANCE_M - _STOP_CLEARANCE_M)
            )
            return max(-limit, min(limit, linear))
        # stop zone
        if linear < 0:
            return max(linear, -_REVERSE_ESCAPE_MPS)
        return 0.0

    def _publish_status(self, state: str):
        """Publish safety/status, but only when the state changes."""
        if state != self._last_status_state:
            self._last_status_state = state
            self._pub_status.put(
                json.dumps(
                    {
                        "state": state,
                        "min_clearance_m": round(self._min_clearance_m, 3),
                        "t": time.time(),
                    }
                )
            )

    def _step(self, dt: float):
        """Advance the motor simulation one step and publish the encoders."""
        with self._lock:
            linear = self._cmd_linear
            angular = self._cmd_angular
            if time.monotonic() - self._last_cmd_time > _CMD_TIMEOUT_S:
                linear = 0.0
                angular = 0.0
            min_clearance_m = self._min_clearance_m

        # Latched e-stop: wheels stay at zero and commands are ignored until an
        # explicit safety/reset.  This is the safety function's safe state.
        if self._estop_latched:
            self._left_rps = 0.0
            self._right_rps = 0.0
            self._pub_wheel.put(
                json.dumps({"left_rps": 0.0, "right_rps": 0.0, "t": time.time()})
            )
            self._publish_status("estop_latched")
            return

        # Body edge inside the e-stop threshold: latch it.
        if min_clearance_m < _ESTOP_CLEARANCE_M:
            with self._lock:
                self._estop_latched = True
            print(
                f"[drive] E-STOP LATCHED: clearance {min_clearance_m:.3f} m "
                f"(< {_ESTOP_CLEARANCE_M} m). Send safety/reset to clear."
            )
            self._left_rps = 0.0
            self._right_rps = 0.0
            self._pub_wheel.put(
                json.dumps({"left_rps": 0.0, "right_rps": 0.0, "t": time.time()})
            )
            self._publish_status("estop_latched")
            return

        # Slow-down / stop zones: scale the linear command by proximity.
        linear = self._safety_limited_linear(linear, min_clearance_m)

        target_left, target_right = unicycle_to_wheel(linear, angular)

        # Velocity loop: first-order response toward the target, capped by the
        # torque/acceleration limit so step changes ramp instead of jumping.
        self._left_rps = _velocity_step(self._left_rps, target_left, dt)
        self._right_rps = _velocity_step(self._right_rps, target_right, dt)

        # Encoder measurement noise.
        left = self._left_rps + random.gauss(0.0, _ENCODER_NOISE_RPS)
        right = self._right_rps + random.gauss(0.0, _ENCODER_NOISE_RPS)

        self._pub_wheel.put(json.dumps({"left_rps": left, "right_rps": right, "t": time.time()}))

        if min_clearance_m < _STOP_CLEARANCE_M:
            state = "stop"
        elif min_clearance_m < _SLOW_DOWN_CLEARANCE_M:
            state = "slow_down"
        else:
            state = "nominal"
        self._publish_status(state)

    def run(self):
        print("[drive] Running. Press Ctrl+C to stop.")
        period = 1.0 / _LOOP_HZ
        next_tick = time.monotonic()
        last = next_tick
        try:
            while True:
                now = time.monotonic()
                dt = now - last
                last = now
                self._step(dt)
                # Deadline-driven pacing (no cumulative drift from jitter).
                next_tick += period
                sleep_until(next_tick)
        except KeyboardInterrupt:
            print("[drive] Stopping…")
        finally:
            self._session.close()


def main():
    Drive().run()


if __name__ == "__main__":
    main()
