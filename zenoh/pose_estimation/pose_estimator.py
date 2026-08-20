"""Zenoh-based pose estimator.

Subscribes to sensor/lidar, sensor/wheel_speed and sensor/imu, runs a particle
filter, and publishes the estimated pose on estimate/pose.

Heading is fused from the IMU gyro + wheel-odometry angular rate + absolute yaw
by a small EKF (heading_filter.HeadingFilter), which also estimates the gyro
bias and track scale.  The particle filter estimates the pose and the wheel
radius (forward) scale.

Delayed AprilTag detections are fused retroactively: the particle cloud is
rewound to a buffered snapshot at the detection's capture time, the measurement
is fused there, and the intervening motion + LIDAR scans are replayed forward
to now (correct time-travel, not a raw-sum projection).

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

import numpy as np
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
from pose_estimation.heading_filter import HeadingFilter
from pose_estimation.particle_filter import ParticleFilter
from simulation.kinematics import wheel_to_unicycle
from simulation.raycast import RayHit

# ---------------------------------------------------------------------------
# Configuration (mirrors simulation/simulator.py)
# ---------------------------------------------------------------------------

LIDAR_MAX_RANGE_M = 10.0

# AprilTag camera measurement noise (mirrors simulation/simulator.py).  Used to
# derive the anchor measurement's covariance instead of hardcoding it.
CAMERA_RANGE_NOISE_M = 0.02
CAMERA_BEARING_NOISE_RAD = math.radians(1.0)
CAMERA_YAW_NOISE_RAD = math.radians(2.0)

# After an AprilTag anchor, keep random particle injection localized to the
# current estimate (rather than global across the map) for this many seconds.
ANCHOR_LOCAL_INJECT_S = 30.0

# How much measurement/motion/state history to retain (seconds), used for the
# retroactive fusion of delayed AprilTag detections.
HISTORY_S = 2.0

# Resample the particle filter only when the effective sample size drops below
# this fraction of the total particle count.  Keeps a healthy, well-distributed
# filter from being needlessly roughened every scan.
ESS_RESAMPLE_FRACTION = 0.5

# Warn that localization is lost when the positional uncertainty (sqrt of the
# trace of the xy covariance) exceeds this many meters.
LOST_XY_STD_M = 1.0


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

        # 3x3 pose covariance [[x,y,theta] x [x,y,theta]], updated each filter
        # step and published with the estimate.  Starts as a large value
        # ("unknown") until the first scan.
        self._cov: List[List[float]] = [
            [1e6, 0.0, 0.0],
            [0.0, 1e6, 0.0],
            [0.0, 0.0, 1e6],
        ]
        self._last_lost_warn: float = 0.0  # rate-limit "lost" warnings

        # Timestamped forward deltas (t_epoch, delta_forward) from wheel
        # odometry.  Bounded to HISTORY_S.  Protected by _pf_lock.
        self._odom_history: List[Tuple[float, float]] = []

        # Timestamped fused heading deltas (t_epoch, delta_theta) from the
        # heading filter.  Bounded to HISTORY_S.  Protected by _pf_lock.
        self._heading_history: List[Tuple[float, float]] = []

        # Recent LIDAR scans (scan_t, ray_hits) for retroactive replay.
        # Bounded to HISTORY_S.  Protected by _pf_lock.
        self._lidar_history: List[Tuple[float, List[RayHit]]] = []

        # Particle-cloud snapshots (t, particles, weights) after each scan, for
        # rewinding during retroactive fusion.  Bounded to HISTORY_S.  Protected
        # by _pf_lock.
        self._snapshots: List[Tuple[float, np.ndarray, np.ndarray]] = []

        # Wheel speed / gyro tracking for delta integration.
        self._last_wheel_time: Optional[float] = None
        self._last_imu_time: Optional[float] = None
        # Latest odometry angular rate, fed to the heading filter.
        self._last_odom_angular_rps: float = 0.0
        # Capture time of the last processed LIDAR scan (motion windowing).
        self._last_scan_t: Optional[float] = None

        # Heading filter: fuses gyro + odometry angular rate + absolute yaw.
        self._heading_filter = HeadingFilter()

        # Online odometry forward-scale (wheel-radius) estimate, published with
        # the pose for observability.
        self._odom_scale: float = 1.0
        # Heading-filter track scale and gyro bias, published for observability.
        self._track_scale: float = 1.0
        self._gyro_bias: float = 0.0

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

        # IMU subscriber: the gyro angular rate is fused as the heading source
        # (wheel-odometry heading drifts from track-calibration error).
        self._sub_imu = self._session.declare_subscriber("sensor/imu", self._on_imu)

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
        """Accumulate timestamped forward deltas from wheel speed messages.

        Runs on a Zenoh I/O thread.  Appends (t, delta_forward) to the odometry
        history (lock-protected) and stores the latest odometry angular rate for
        the heading filter.  Never touches the particle filter directly.
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
                with self._pf_lock:
                    self._odom_history.append((t, delta_forward))
                    self._last_odom_angular_rps = angular_rps
                    self._prune_histories(t - HISTORY_S)
        else:
            # First wheel speed message: initialize the particle filter
            # near the origin (room 0 center).  Better start than uniform
            # across the whole map.
            self._initialize_near_origin()

        self._last_wheel_time = t

    def _on_imu(self, sample):
        """Fuse gyro + odometry + absolute yaw into heading via the heading
        filter, and accumulate the resulting timestamped heading deltas.

        Runs on a Zenoh I/O thread.  Steps the heading filter and appends
        (t, delta_theta) to the heading history (lock-protected), so heading is
        available up to each LIDAR scan's capture time.
        """
        try:
            data = json.loads(sample.payload.to_string())
            gyro_rps = float(data["angular_velocity_rps"])
            yaw = data.get("yaw_rad")
            yaw = float(yaw) if yaw is not None else None
            t = float(data.get("t", time.time()))
        except (json.JSONDecodeError, Exception):
            return

        if self._last_imu_time is not None:
            dt = t - self._last_imu_time
            if 0.0 < dt < 1.0:
                with self._pf_lock:
                    dtheta = self._heading_filter.step(
                        dt, gyro_rps, self._last_odom_angular_rps, yaw,
                        wheel_scale=self._odom_scale,
                    )
                    self._heading_history.append((t, dtheta))
                    self._prune_histories(t - HISTORY_S)

        self._last_imu_time = t

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

            # Fuse the absolute pose measurement retroactively (a Bayesian
            # weighting update that preserves the current belief).  Rate-limit
            # to 2 Hz so the same tag isn't re-fused every scan while it stays
            # in view.
            now = time.monotonic()
            if now - self._last_anchor_time >= 0.5:
                # Measurement covariance from the camera noise model and the
                # detection geometry (distance to the tag).
                range_m = math.hypot(x_rel_m, y_rel_m)
                std_xy = math.sqrt(
                    CAMERA_RANGE_NOISE_M ** 2
                    + (range_m * CAMERA_BEARING_NOISE_RAD) ** 2
                    + (range_m * CAMERA_YAW_NOISE_RAD) ** 2
                )
                std_theta = CAMERA_YAW_NOISE_RAD

                self._retroactively_fuse(
                    anchor_x, anchor_y, anchor_theta, std_xy, std_theta, tag_t
                )
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
            self._cov = self.pf.covariance().tolist()
            self._odom_scale = self.pf.odom_scale_estimate()
            self._initialized = True
            self._last_scan_t = None

        print(f"[pose_estimator] Initialized particles near " f"({cx:.2f}, {cy:.2f}) m")

    # -------------------------------------------------------------------
    # Filter helpers (predict → update → estimate → resample)
    # -------------------------------------------------------------------

    def _prune_histories(self, cutoff: float):
        """Drop history/snapshot entries older than ``cutoff``.

        Called under _pf_lock.
        """
        while self._odom_history and self._odom_history[0][0] < cutoff:
            self._odom_history.pop(0)
        while self._heading_history and self._heading_history[0][0] < cutoff:
            self._heading_history.pop(0)
        while self._lidar_history and self._lidar_history[0][0] < cutoff:
            self._lidar_history.pop(0)
        while self._snapshots and self._snapshots[0][0] < cutoff:
            self._snapshots.pop(0)

    def _motion_since(self, from_t: float, to_t: float) -> Tuple[float, float]:
        """Sum forward and fused-heading deltas with from_t < t <= to_t."""
        fwd = sum(df for (t, df) in self._odom_history if from_t < t <= to_t)
        th = sum(dt for (t, dt) in self._heading_history if from_t < t <= to_t)
        return fwd, th

    def _update_estimate(self):
        self.estimated_x, self.estimated_y, self.estimated_theta = self.pf.estimate()
        self._cov = self.pf.covariance().tolist()
        self._odom_scale = self.pf.odom_scale_estimate()
        self._track_scale = self._heading_filter.track_scale
        self._gyro_bias = self._heading_filter.gyro_bias

    def _maybe_resample(self):
        """Adaptive resampling, only when the weights have degenerated."""
        ess = self.pf.effective_sample_size()
        if ess < ESS_RESAMPLE_FRACTION * self.pf.num_particles:
            if time.monotonic() - self._last_anchor_time < ANCHOR_LOCAL_INJECT_S:
                self.pf.resample(
                    inject_mode="local",
                    anchor=(self.estimated_x, self.estimated_y),
                )
            else:
                self.pf.resample()

    def _apply_filter_step(self, from_t: float, to_t: float, ray_hits: List[RayHit]):
        """Predict motion over (from_t, to_t], then update + resample.

        Called under _pf_lock.
        """
        fwd, th = self._motion_since(from_t, to_t)
        if fwd != 0.0 or th != 0.0:
            self.pf.predict(fwd, th)
        self.pf.update(ray_hits)
        self._maybe_resample()

    def _snapshot_at(self, t: float) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        """Return the latest snapshot at time <= t, or None."""
        for s in reversed(self._snapshots):
            if s[0] <= t:
                return s
        return None

    def _retroactively_fuse(
        self,
        x: float,
        y: float,
        theta: float,
        std_xy: float,
        std_theta: float,
        tag_t: Optional[float],
    ):
        """Fuse a delayed absolute-pose measurement via snapshot rewind + replay.

        Rewinds the particle cloud to the buffered snapshot at (or just before)
        the measurement's capture time, fuses the measurement there, then
        replays the intervening motion and LIDAR scans forward to now.
        Replaying the actual predict steps (rather than a single raw-sum)
        preserves per-step odometry noise, so covariance grows correctly over
        the delay, and the mechanism generalizes to any measurement that arrives
        out of order within the history window.

        Falls back to back-propagation weighting on the current cloud if no
        snapshot is old enough.
        """
        with self._pf_lock:
            if tag_t is None:
                self.pf.fuse_absolute_pose(x, y, theta, std_xy, std_theta)
                return

            snap = self._snapshot_at(tag_t)
            if snap is None:
                # No snapshot old enough (the measurement predates the history
                # window): fall back to back-propagation on the current cloud.
                # NOTE: this raw-sums the intervening odometry and ignores its
                # covariance growth, so it is only an approximation.  It is
                # bounded by HISTORY_S and should be rare in practice; log it
                # so it is visible when it happens.
                print(
                    "[pose_estimator] WARNING: delayed measurement older than "
                    "history window; using approximate back-propagation."
                )
                df = sum(d for (t, d) in self._odom_history if t > tag_t)
                dth = sum(d for (t, d) in self._heading_history if t > tag_t)
                self.pf.fuse_absolute_pose(
                    x, y, theta, std_xy, std_theta,
                    delta_forward_m=df, delta_theta=dth,
                )
                return

            snap_t, snap_particles, snap_weights = snap
            self.pf.particles = snap_particles.copy()
            self.pf.weights = snap_weights.copy()

            # Replay from the snapshot up to the capture time.
            prev = snap_t
            for (t, hits) in self._lidar_history:
                if t <= snap_t:
                    continue
                if t > tag_t:
                    break
                self._apply_filter_step(prev, t, hits)
                prev = t

            # Fuse the measurement at its capture time.
            self.pf.fuse_absolute_pose(x, y, theta, std_xy, std_theta)

            # Replay from the capture time to now, rebuilding snapshots.
            self._snapshots = [s for s in self._snapshots if s[0] < tag_t]
            prev = tag_t
            for (t, hits) in self._lidar_history:
                if t <= tag_t:
                    continue
                self._apply_filter_step(prev, t, hits)
                prev = t
                self._snapshots.append(
                    (t, self.pf.particles.copy(), self.pf.weights.copy())
                )

            # If no scans followed the capture time, still propagate motion to
            # the latest scan so the cloud ends at "now".
            if prev < (self._last_scan_t or tag_t):
                fwd, th = self._motion_since(prev, self._last_scan_t)
                if fwd != 0.0 or th != 0.0:
                    self.pf.predict(fwd, th)
                self._snapshots.append(
                    (self._last_scan_t, self.pf.particles.copy(), self.pf.weights.copy())
                )

            self._update_estimate()

    # -------------------------------------------------------------------
    # LIDAR processing (heavy — called from main loop with RingChannel)
    # -------------------------------------------------------------------

    def _process_lidar(self, sample):
        """Parse and process a LIDAR scan (predict motion + update + resample)."""
        if not self._initialized:
            return

        try:
            payload = sample.payload.to_string()
            lidar_data = json.loads(payload)
        except (json.JSONDecodeError, Exception) as exc:
            print(f"[pose_estimator] Failed to parse LIDAR message: {exc}")
            return

        scan_t = lidar_data.get("t")
        if scan_t is None:
            scan_t = time.time()
        scan_t = float(scan_t)

        ray_hits = self._parse_rays(lidar_data.get("rays", []))

        with self._pf_lock:
            from_t = self._last_scan_t if self._last_scan_t is not None else float("-inf")
            self._apply_filter_step(from_t, scan_t, ray_hits)
            self._update_estimate()
            self._last_scan_t = scan_t

            # Record the scan and a particle-cloud snapshot for retroactive
            # fusion of delayed measurements.
            self._lidar_history.append((scan_t, ray_hits))
            self._snapshots.append(
                (scan_t, self.pf.particles.copy(), self.pf.weights.copy())
            )
            self._prune_histories(scan_t - HISTORY_S)

    def _parse_rays(self, rays: List[dict]) -> List[RayHit]:
        """Convert {angle_rad, distance_m} entries to RayHit objects.

        Only the angle and distance are used by the particle filter; the
        absolute hit point is not, so it is left unset.
        """
        ray_hits: List[RayHit] = []
        for entry in rays:
            ray_hits.append(
                RayHit(
                    angle=float(entry["angle_rad"]),
                    distance=float(entry["distance_m"]),
                    point=(0.0, 0.0),
                )
            )
        return ray_hits

    # -------------------------------------------------------------------
    # Publishing
    # -------------------------------------------------------------------

    def _publish_pose(self):
        """Publish the estimated pose (and its covariance) as JSON."""
        msg = json.dumps(
            {
                "x_m": self.estimated_x,
                "y_m": self.estimated_y,
                "theta_rad": self.estimated_theta,
                "cov": self._cov,
                "odom_scale": self._odom_scale,
                "track_scale": self._track_scale,
                "gyro_bias": self._gyro_bias,
                "t": time.time(),
            }
        )
        self._pub_pose.put(msg)

        # Warn (rate-limited) when localization has likely diverged.
        std_xy = math.sqrt(self._cov[0][0] + self._cov[1][1])
        if std_xy > LOST_XY_STD_M and time.monotonic() - self._last_lost_warn > 5.0:
            self._last_lost_warn = time.monotonic()
            print(
                f"[pose_estimator] WARNING: localization uncertain "
                f"(std_xy={std_xy:.2f} m)"
            )

        print(
            f"[pose_estimator] Pose: x={self.estimated_x:.2f}m, "
            f"y={self.estimated_y:.2f}m, "
            f"θ={math.degrees(self.estimated_theta):.1f}°, "
            f"ws={self._odom_scale:.3f}, ts={self._track_scale:.3f}"
        )

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------

    def run(self):
        """Block forever, processing the latest LIDAR and AprilTag data.

        RingChannel(1) on the LIDAR subscriber ensures recv() always
        returns the newest scan.  After each filter update, any pending
        AprilTag detection is consumed (and fused retroactively), then the
        pose is published.
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

                # Publish the (possibly retroactively-updated) estimate.
                self._publish_pose()
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
