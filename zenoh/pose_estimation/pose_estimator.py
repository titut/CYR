"""Zenoh-based pose estimator.

Subscribes to sensor/lidar from the simulator and sensor/wheel_speed from the
drive node, runs a particle filter, and publishes the estimated pose on
estimate/pose.

Uses RingChannel(1) for the LIDAR subscriber so the particle filter
always processes only the most recent scan — stale backlogs are dropped.

Usage:
    python zenoh/pose_estimation/pose_estimator.py [path/to/map.json]

Defaults to test_map.json if no map is provided.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import zenoh
import zenoh.handlers

# Allow running this file directly or as a module.
_SCRIPT_DIR = Path(__file__).resolve().parent
_ZENOH_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _ZENOH_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_ZENOH_DIR) not in sys.path:
    sys.path.insert(0, str(_ZENOH_DIR))

from map_format import MapData, Apriltag, new_empty_map
from pose_estimation.particle_filter import ParticleFilter
from simulation.kinematics import wheel_to_unicycle
from simulation.raycast import RayHit

# ---------------------------------------------------------------------------
# Configuration (mirrors simulation/simulator.py)
# ---------------------------------------------------------------------------

LIDAR_MAX_RANGE_M = 10.0

# After an AprilTag anchor, keep random particle injection localized to the
# current estimate (rather than global across the map) for this many seconds.
ANCHOR_LOCAL_INJECT_S = 30.0

# How much odometry history to keep (seconds), so a delayed AprilTag detection
# can be projected forward to "now" from its capture time.
ODOM_HISTORY_S = 2.0


class PoseEstimator:
    def __init__(self, map_path: Path):
        # Load the same map the simulator uses (geometry is in meters).
        self.map_data = self._load_map(map_path)

        # Particle filter (same parameters as the old simulator).
        self.pf = ParticleFilter(
            self.map_data,
            num_particles=200,
            num_beams=36,
            max_range_m=LIDAR_MAX_RANGE_M,
        )
        self._pf_lock = threading.Lock()
        self._initialized = False
        self.estimated_x: float = 0.0
        self.estimated_y: float = 0.0
        self.estimated_theta: float = 0.0

        # Timestamped odometry deltas (t_epoch, delta_forward, delta_theta)
        # waiting to be consumed.  Protected by _pf_lock.
        self._odom_buffer: List[Tuple[float, float, float]] = []

        # Short rolling history of odometry deltas, kept separately from
        # _odom_buffer (which LIDAR drains) so a delayed AprilTag anchor can sum
        # "odometry since capture" and project itself forward to now.  Protected
        # by _pf_lock.
        self._odom_history: List[Tuple[float, float, float]] = []

        # Wheel speed tracking for delta integration.
        self._last_wheel_time: Optional[float] = None

        # Zenoh session and publishers/subscribers.
        self._session = zenoh.open(zenoh.Config())

        # Publisher for the estimated pose.
        self._pub_pose = self._session.declare_publisher(
            "estimate/pose",
            congestion_control=zenoh.CongestionControl.DROP,
            reliability=zenoh.Reliability.BEST_EFFORT,
        )

        # -------------------------------------------------------------------
        # LIDAR subscriber: RingChannel(1) keeps ONLY the latest sample.
        # New arrivals overwrite old ones so the particle filter never
        # processes stale backlogs.  recv() in the main loop always
        # returns the most recent scan.
        # -------------------------------------------------------------------
        self._sub_lidar = self._session.declare_subscriber(
            "sensor/lidar",
            zenoh.handlers.RingChannel(1),
        )

        # -------------------------------------------------------------------
        # Wheel speed subscriber: uses a callback because delta accumulation
        # is trivial (a few float additions) and never blocks.
        # -------------------------------------------------------------------
        self._sub_wheel = self._session.declare_subscriber(
            "sensor/wheel_speed", self._on_wheel_speed
        )

        # -------------------------------------------------------------------
        # Pre-build an Apriltag lookup table (tag id → world position).
        # -------------------------------------------------------------------
        self._tag_lookup: dict[int, Apriltag] = {
            t.id: t for t in self.map_data.apriltags
        }

        # -------------------------------------------------------------------
        # Latest AprilTag detection, set by callback, consumed by main loop.
        # Protected by _tag_det_lock.
        # -------------------------------------------------------------------
        self._tag_det_lock = threading.Lock()
        self._latest_tag_dets: Optional[Tuple[Optional[float], List[dict]]] = None
        self._last_anchor_time: float = 0.0  # rate-limit anchoring to 2 Hz

        # Subscribe with a callback that just stores the latest detection.
        self._sub_apriltag = self._session.declare_subscriber(
            "detection/apriltag", self._on_apriltag
        )

    # -------------------------------------------------------------------
    # Map loading
    # -------------------------------------------------------------------

    @staticmethod
    def _load_map(path: Path) -> MapData:
        if not path.exists():
            print(f"Map file not found: {path}. Creating an empty map.")
            return new_empty_map()
        return MapData.from_json(path)

    # -------------------------------------------------------------------
    # Wheel speed callback (fast, non-blocking delta accumulation)
    # -------------------------------------------------------------------

    def _on_wheel_speed(self, sample):
        """Accumulate timestamped odometry deltas from wheel speed messages.

        Runs on a Zenoh I/O thread.  Appends (t, delta_forward, delta_theta)
        to the odometry buffer (lock-protected) using the message's source
        timestamp, so the filter can integrate odometry up to each LIDAR scan's
        capture time rather than by arrival time.  Never touches the particle
        filter directly.
        """
        try:
            payload = sample.payload.to_string()
            wheel_data = json.loads(payload)
            left_rps = float(wheel_data["left_rps"])
            right_rps = float(wheel_data["right_rps"])
            t = float(wheel_data.get("t", time.time()))
            linear_mps, angular_rps = wheel_to_unicycle(left_rps, right_rps)
        except (json.JSONDecodeError, Exception) as exc:
            print(f"[pose_estimator] Failed to parse wheel speed message: {exc}")
            return

        if self._last_wheel_time is not None:
            dt = t - self._last_wheel_time
            if 0.0 < dt < 1.0:
                delta_forward = linear_mps * dt
                delta_theta = angular_rps * dt
                with self._pf_lock:
                    self._odom_buffer.append((t, delta_forward, delta_theta))
                    self._odom_history.append((t, delta_forward, delta_theta))
                    # Bound both buffers: drop entries older than the retention
                    # window.  _odom_buffer is normally drained by LIDAR, but if
                    # LIDAR stalls (or clocks skew) it would otherwise grow
                    # without bound.
                    cutoff = t - ODOM_HISTORY_S
                    while self._odom_history and self._odom_history[0][0] < cutoff:
                        self._odom_history.pop(0)
                    while self._odom_buffer and self._odom_buffer[0][0] < cutoff:
                        self._odom_buffer.pop(0)
        else:
            # First wheel speed message: initialize the particle filter
            # near the origin (room 0 center).  Better start than uniform
            # across the whole map.
            self._initialize_near_origin()

        self._last_wheel_time = t

    # -------------------------------------------------------------------
    # AprilTag callback (fast, just stores latest detection for main loop)
    # -------------------------------------------------------------------

    def _on_apriltag(self, sample):
        """Store the latest AprilTag detection JSON for processing by the
        main loop.  Only the most recent detection matters for anchoring.

        Runs on a Zenoh I/O thread — only touches _latest_tag_dets under
        its own lock.
        """
        try:
            data = json.loads(sample.payload.to_string())
            dets = data.get("detections", [])
            tag_t = data.get("t")
        except (json.JSONDecodeError, Exception):
            return
        with self._tag_det_lock:
            self._latest_tag_dets = (tag_t, dets)

    def _process_apriltags(self):
        """Consume the latest AprilTag detection (if any) and anchor the
        particle filter to the absolute pose measurement it provides.

        Called from the main loop after each LIDAR filter step.
        """
        with self._tag_det_lock:
            latest = self._latest_tag_dets
            self._latest_tag_dets = None

        if not latest or not self._initialized:
            return

        tag_t, dets = latest

        for det in dets:
            tag_id = det["id"]
            tag = self._tag_lookup.get(tag_id)
            if tag is None:
                continue

            # Derive the robot's absolute pose from the tag detection.
            #
            # The detection gives the tag's position in the robot frame:
            #   x_rel, y_rel  — Cartesian coordinates (m), +x forward
            #   yaw_rel       — tag yaw in robot frame
            #
            # To get the robot's world pose from this:
            #
            #   robot_world_xy = tag_world_xy - R(robot_theta) * (x_rel, y_rel)
            #   robot_world_θ  = tag_world_yaw - yaw_rel
            #
            x_rel_m = det["x_rel"]
            y_rel_m = det["y_rel"]
            yaw_rel = det["yaw_rel"]

            # The tag detection is an *absolute* measurement: derive the robot's
            # world pose directly from the tag's known world pose and the
            # relative transform, without relying on the (possibly wrong)
            # current estimate.
            anchor_theta = tag.yaw_rad - yaw_rel
            cos_a = math.cos(anchor_theta)
            sin_a = math.sin(anchor_theta)

            world_dx = x_rel_m * cos_a - y_rel_m * sin_a
            world_dy = x_rel_m * sin_a + y_rel_m * cos_a

            anchor_x = tag.x - world_dx
            anchor_y = tag.y - world_dy

            # Jump the filter to the absolute pose (reset all particles).
            # Rate-limit to 2 Hz so we don't re-snap every scan while the tag
            # stays in view.
            now = time.time()
            if now - self._last_anchor_time >= 0.5:
                with self._pf_lock:
                    self.pf.anchor_fraction(
                        anchor_x,
                        anchor_y,
                        anchor_theta,
                        std_xy_m=0.25,
                        std_theta_rad=0.05,
                        fraction=1.0,
                    )

                    # The detection is delayed: its pose is valid at the capture
                    # time, not now.  Project the freshly-anchored particles
                    # forward using the odometry collected since capture, so the
                    # estimate lands at "now" instead of "now - delay".
                    if tag_t is not None:
                        fwd = 0.0
                        th = 0.0
                        for t, df, dth in self._odom_history:
                            if t > tag_t:
                                fwd += df
                                th += dth
                        if fwd != 0.0 or th != 0.0:
                            self.pf.predict(fwd, th)
                        # Those deltas are now baked into the estimate; drop any
                        # still-pending copies so LIDAR doesn't re-apply them.
                        self._odom_buffer.clear()
                self._last_anchor_time = now

            print(
                f"[pose_estimator] AprilTag {tag_id} anchor: "
                f"x={anchor_x:.2f}m, y={anchor_y:.2f}m, "
                f"θ={math.degrees(anchor_theta):.1f}°"
            )
            # Only use the first tag for anchoring.
            break

    def _initialize_near_origin(self):
        """Initialize particles near the map origin (center of first room)."""
        rooms = self.map_data.rooms
        if rooms:
            xs = [p[0] for p in rooms[0].polygon]
            ys = [p[1] for p in rooms[0].polygon]
            cx = (min(xs) + max(xs)) / 2.0
            cy = (min(ys) + max(ys)) / 2.0
        else:
            cx = self.map_data.metadata.size_m[0] / 2.0
            cy = self.map_data.metadata.size_m[1] / 2.0

        with self._pf_lock:
            self.pf.initialize_near(cx, cy, 0.0, std_xy_m=1.5, std_theta_rad=0.5)
            self.estimated_x, self.estimated_y, self.estimated_theta = (
                self.pf.estimate()
            )
            self._initialized = True
            self._odom_buffer.clear()

        print(f"[pose_estimator] Initialized particles near " f"({cx:.2f}, {cy:.2f}) m")

    # -------------------------------------------------------------------
    # LIDAR processing (heavy — called from main loop with RingChannel)
    # -------------------------------------------------------------------

    def _process_lidar(self, sample):
        """Called from the main loop when a fresh LIDAR scan arrives.

        This is the main filter step: consume the odometry deltas up to the
        scan's capture time, predict, then update with the LIDAR scan,
        estimate, resample.
        The entire sequence is protected by a lock so wheel-speed callbacks
        only accumulate deltas without touching the filter during update.
        """
        if not self._initialized:
            # We haven't received any wheel speed yet — nothing to predict.
            return

        try:
            payload = sample.payload.to_string()
            lidar_data = json.loads(payload)
        except (json.JSONDecodeError, Exception) as exc:
            print(f"[pose_estimator] Failed to parse LIDAR message: {exc}")
            return

        scan_t = lidar_data.get("t")
        if scan_t is not None:
            scan_t = float(scan_t)
        rays = lidar_data.get("rays", [])

        # Convert {angle_rad, distance_m} back to RayHit objects.
        ray_hits: List[RayHit] = []
        for entry in rays:
            angle_rad = entry["angle_rad"]
            distance_m = entry["distance_m"]

            # The particle filter only uses hit.angle (relative to forward)
            # and hit.distance.  The absolute hit point is not used by the
            # filter, so we reconstruct it from the current estimate.
            hit_x = self.estimated_x + distance_m * math.cos(
                self.estimated_theta + angle_rad
            )
            hit_y = self.estimated_y + distance_m * math.sin(
                self.estimated_theta + angle_rad
            )

            ray_hits.append(
                RayHit(
                    angle=angle_rad,
                    distance=distance_m,
                    point=(hit_x, hit_y),
                )
            )

        # ---- Atomic predict → update → estimate → resample ----
        with self._pf_lock:
            # Apply odometry deltas up to the scan's capture time (or all of
            # them if the scan carries no timestamp), so the scan is fused
            # against the pose at the moment it was taken, not when it arrived.
            total_forward = 0.0
            total_theta = 0.0
            if scan_t is None:
                while self._odom_buffer:
                    _, df, dth = self._odom_buffer.pop(0)
                    total_forward += df
                    total_theta += dth
            else:
                while self._odom_buffer and self._odom_buffer[0][0] <= scan_t:
                    _, df, dth = self._odom_buffer.pop(0)
                    total_forward += df
                    total_theta += dth

            if total_forward != 0.0 or total_theta != 0.0:
                self.pf.predict(total_forward, total_theta)

            # Weight particles by how well they explain the LIDAR scan.
            self.pf.update(ray_hits)

            # Weighted mean pose.
            self.estimated_x, self.estimated_y, self.estimated_theta = (
                self.pf.estimate()
            )

            # Systematic resampling + random injection.
            # After a recent AprilTag anchor, inject the random fraction
            # locally around the estimate; otherwise scatter it globally.
            if time.time() - self._last_anchor_time < ANCHOR_LOCAL_INJECT_S:
                self.pf.resample(
                    inject_mode="local",
                    anchor=(self.estimated_x, self.estimated_y),
                )
            else:
                self.pf.resample()

        # Publish the estimated pose (outside the lock).
        self._publish_pose()

    # -------------------------------------------------------------------
    # Publishing
    # -------------------------------------------------------------------

    def _publish_pose(self):
        """Publish the estimated pose as JSON on estimate/pose."""
        msg = json.dumps(
            {
                "x_m": self.estimated_x,
                "y_m": self.estimated_y,
                "theta_rad": self.estimated_theta,
                "t": time.time(),
            }
        )
        self._pub_pose.put(msg)
        print(
            f"[pose_estimator] Pose: x={self.estimated_x:.2f}m, "
            f"y={self.estimated_y:.2f}m, "
            f"θ={math.degrees(self.estimated_theta):.1f}°"
        )

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------

    def run(self):
        """Block forever, processing the latest LIDAR and AprilTag data.

        RingChannel(1) on the LIDAR subscriber ensures recv() always
        returns the newest scan.  After each filter update, any pending
        AprilTag detection is consumed to anchor the particle filter.
        """
        print("[pose_estimator] Running. Press Ctrl+C to stop.")
        try:
            while True:
                # Block until a LIDAR sample is available.
                # RingChannel(1) ensures we always get the newest one.
                sample = self._sub_lidar.recv()
                self._process_lidar(sample)

                # After the filter step, consume any tag detections.
                self._process_apriltags()
        except KeyboardInterrupt:
            print("[pose_estimator] Stopping…")
        finally:
            self._session.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    map_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_map.json")
    estimator = PoseEstimator(map_path)
    estimator.run()


if __name__ == "__main__":
    main()
