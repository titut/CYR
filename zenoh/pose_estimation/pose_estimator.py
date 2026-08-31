"""Zenoh-based pose estimator.

Subscribes to sensor/lidar, sensor/wheel_speed, sensor/imu and
detection/apriltag, runs a particle filter, and publishes the estimated pose
on estimate/pose.

All measurements are fused Bayesian-style into one particle filter:

- LIDAR scans weight particles through the beam model (ParticleFilter.update).
- AprilTag detections are absolute pose measurements: each fresh detection
  re-weights the cloud by the measurement's Gaussian likelihood
  (ParticleFilter.fuse_absolute_pose) with a covariance derived from the
  camera noise model.  The filter does the smoothing (repeated detections
  converge the cloud onto the tag mean), and a measurement that contradicts
  every particle triggers the recovery re-seed inside fuse_absolute_pose.
- The heading filter (gyro + wheel-odometry rate + absolute yaw) supplies
  fused heading deltas for the motion model and estimates gyro bias and track
  scale; the particle filter estimates the pose and the wheel-radius scale.

LIDAR scan matching (Gauss-Newton ICP with a coordinate-descent fallback)
refines the estimate when the scan is structured enough to localize against
and no tag has fused recently, so ICP cannot fight an active tag correction.

Uses RingChannel(1) for the LIDAR subscriber so the filter always processes
only the most recent scan.  AprilTag detections are consumed on a dedicated
thread so they fuse at the camera rate, not the LIDAR rate.

Usage:
    python zenoh/pose_estimation/pose_estimator.py [path/to/map.json]

Defaults to test_map.json if no map is provided.
"""

from __future__ import annotations

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

from core.map_format import MapData, Apriltag, new_empty_map
from core.messages import SchemaError, decode, encode
from pose_estimation.heading_filter import HeadingFilter
from pose_estimation.particle_filter import ParticleFilter
from pose_estimation.scan_match import refine_discrete, refine_gauss_newton, scan_match_error
from core.robot_config import get_robot_config
from simulation.kinematics import wheel_to_unicycle
from simulation.raycast import RayHit

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# AprilTag camera measurement noise and LIDAR range come from the robot config
# (robot.yaml), the single source shared with the simulator's sensor models.
# They are used to derive the tag measurement's covariance instead of
# hardcoding it.

# How much measurement/motion history to retain (seconds) for the motion/heading
# windows used to predict the particle filter between LIDAR scans.
HISTORY_S = 2.0

# Resample the particle filter only when the effective sample size drops below
# this fraction of the total particle count.  Keeps a healthy, well-distributed
# filter from being needlessly roughened every scan.
ESS_RESAMPLE_FRACTION = 0.5

# Warn that localization is lost when the positional uncertainty (sqrt of the
# trace of the xy covariance) exceeds this many meters.
LOST_XY_STD_M = 1.0

# A LIDAR scan is "informative" — carries enough structure to localize against
# — when at least this fraction of rays return a real surface well inside the
# sensor's max range.  In open space most rays return max range (nothing in
# range), so the scan carries almost no positional information; its likelihood
# is near-flat and the update is self-neutralizing (the filter degrades
# gracefully to motion-model dead reckoning with honestly growing covariance).
# The score gates the tag-association gate and scan matching, which both need
# actual structure to judge against.
SCAN_INFO_RANGE_FRAC = 0.9    # a ray is informative if it hits inside 90% of range

# Tag-association gating: a detection whose derived anchor contradicts the
# LIDAR (an independent sensor) is a mis-associated tag — e.g. the id read
# wrong at long range.  Field logs: a far tag (~5 m) dragged the estimate 3 m
# off over 2.6 s.  A *correct* anchor always explains the observed scan better
# than a wrong belief, and a mis-associated one does not, so the anchor is only
# accepted when it fits the scan at least as well as the current estimate (or
# when the scan is too uninformative to judge).
TAG_GATE_ENABLED = True
TAG_GATE_FACTOR = 1.5        # reject if anchor error > factor * current estimate error
TAG_GATE_MARGIN_M = 0.05     # absolute slack above the noise floor
TAG_GATE_SCAN_MAX_AGE_S = 1.0  # skip the gate if the last scan is older than this
TAG_RANGE_MARGIN_M = 0.25    # reject detections beyond camera max range + margin

# Localization-jump safety: when an AprilTag fix differs from the current
# estimate in *position* by more than this threshold, the belief was badly
# wrong.  The estimator publishes a halt command on ``estimate/halt``; the
# controller subscribes and holds position for ``RELOCALIZE_HOLD_S`` seconds so
# the corrected pose settles before the robot keeps moving.  (Only the
# positional discrepancy is considered — the heading is IMU-driven and is not
# a reliable "lost" signal.)
RELOCALIZE_JUMP_M = 0.5                 # positional discrepancy (m)
RELOCALIZE_HOLD_S = 2.0                 # hold duration sent to the controller (s)

# Scan-matching refinement (the "gradient descent" element).  When the scan is
# structured enough, snap the estimate to the pose that best explains the
# LIDAR scan — Gauss-Newton ICP, with coordinate descent as a robust fallback —
# and re-anchor the cloud + odometry fallback to it.
SCAN_MATCH_ENABLED = True
SCAN_MATCH_MIN_INFO = 0.4               # run only when scan informativeness >= this
SCAN_MATCH_RAY_STEP = 4                 # subsample 360 rays -> 90 for cheap matching
SCAN_MATCH_MAX_JUMP_M = 5.0             # reject a refinement that moves the estimate this far
SCAN_MATCH_ACCEPT_RATIO = 0.5           # accept only if the error is at least halved
SCAN_MATCH_SNAP_STD_M = 0.05            # cloud re-anchor spread after a match
SCAN_MATCH_SNAP_STD_RAD = 0.02

# Scan matching is a hard snap, so it must not fight an actively-fusing
# AprilTag: while tags are correcting the belief, the ICP snap is skipped and
# the tag measurement owns the update.  This only gates the *snap*; plain
# LIDAR weight updates continue regardless.
SCAN_MATCH_TAG_HOLD_S = 1.0


class PoseEstimator:
    def __init__(self, map_path: Path):
        # Load the same map the simulator uses (geometry is in meters).
        self.map_data = self._load_map(map_path)

        # Robot description (T-019): LIDAR range + camera noise from robot.yaml,
        # matching what the simulator's sensor models use.
        cfg = get_robot_config()
        self._cfg = cfg
        self._lidar_range_m = cfg.sensors.lidar.range_m
        self._cam_noise = cfg.sensors.camera

        # Particle filter (same parameters as the old simulator).
        self.pf = ParticleFilter(
            self.map_data,
            num_particles=200,
            num_beams=36,
            max_range_m=self._lidar_range_m,
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

        # Informativeness of the latest processed LIDAR scan (0..1).
        self._scan_info_score: float = 1.0

        # monotonic() timestamp of the last fused AprilTag detection, used to
        # hold scan-matching off while tags are actively correcting the belief.
        self._last_tag_fuse_mono: float = float("-inf")

        # Zenoh session and publishers/subscribers.
        self._session = zenoh.open(zenoh.Config())

        # Publisher for the estimated pose.
        self._pub_pose = self._session.declare_publisher(
            "estimate/pose",
            congestion_control=zenoh.CongestionControl.DROP,
            reliability=zenoh.Reliability.BEST_EFFORT,
        )

        # Publisher for the localization-jump halt command.  The controller
        # subscribes to ``estimate/halt`` and holds position when it arrives.
        self._pub_halt = self._session.declare_publisher(
            "estimate/halt",
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

        # Latest LIDAR scan subsampled for matching (guarded by _pf_lock), used
        # by the tag-association gate to sanity-check tag anchors.
        self._latest_match_angles: Optional[np.ndarray] = None
        self._latest_match_observed: Optional[np.ndarray] = None
        self._latest_match_time: float = float("-inf")

        # AprilTag subscriber: a FIFO channel drained by a dedicated thread so
        # detections fuse at the camera rate (the LIDAR loop must not be the
        # bottleneck for tag measurements).
        self._sub_apriltag = self._session.declare_subscriber(
            "detection/apriltag", zenoh.handlers.DefaultHandler()
        )
        self._tag_thread = threading.Thread(
            target=self._tag_loop, name="apriltag-fusion", daemon=True
        )
        self._tag_thread.start()

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
            wheel_data = decode("sensor/wheel_speed", sample)
            left_rps = float(wheel_data["left_rps"])
            right_rps = float(wheel_data["right_rps"])
            t = float(wheel_data["t"])
            linear_mps, angular_rps = wheel_to_unicycle(left_rps, right_rps)
        except SchemaError as exc:
            print(f"[pose_estimator] sensor/wheel_speed dropped: {exc}")
            return

        if self._last_wheel_time is not None:
            dt = t - self._last_wheel_time
            if 0.0 < dt < 1.0:
                with self._pf_lock:
                    self._odom_history.append((t, linear_mps * dt))
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
            data = decode("sensor/imu", sample)
            gyro_rps = float(data["angular_velocity_rps"])
            yaw = data.get("yaw_rad")
            yaw = float(yaw) if yaw is not None else None
            t = float(data["t"])
        except SchemaError as exc:
            print(f"[pose_estimator] sensor/imu dropped: {exc}")
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
    # AprilTag fusion (dedicated thread, camera rate)
    # -------------------------------------------------------------------

    def _tag_loop(self):
        """Drain the AprilTag channel and fuse each detection.

        Blocking recv() on the FIFO channel means every detection is
        processed — none are collapsed or dropped the way the old
        latest-sample handoff did.  Fusion is a cheap weight update (200
        particles), so this thread adds negligible load.
        """
        while True:
            try:
                sample = self._sub_apriltag.recv()
            except Exception:
                continue  # session closed on shutdown
            self._fuse_apriltag(sample)

    def _fuse_apriltag(self, sample):
        """Fuse one detection/apriltag message into the particle filter.

        Each detection is an absolute pose measurement: derive the robot's
        world pose from the tag's known world pose and the relative transform,
        gate it against the LIDAR (mis-association check), then re-weight the
        cloud by its Gaussian likelihood.
        """
        try:
            data = decode("detection/apriltag", sample)
            dets = data.get("detections", [])
            tag_t = data.get("t")
        except SchemaError as exc:
            print(f"[pose_estimator] detection/apriltag dropped: {exc}")
            return

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

            # Range sanity: a detection beyond the camera's max range (plus a
            # small noise margin) is not a trustworthy association.
            det_range_m = math.hypot(x_rel_m, y_rel_m)
            if det_range_m > self._cam_noise.max_range_m + TAG_RANGE_MARGIN_M:
                print(
                    f"[pose_estimator] Tag {tag_id} rejected: range "
                    f"{det_range_m:.2f} m beyond camera max range."
                )
                continue

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

            # Measurement covariance from the camera noise model and the
            # detection geometry (distance to the tag).
            std_xy = math.sqrt(
                self._cam_noise.range_noise_m ** 2
                + (det_range_m * self._cam_noise.bearing_noise_rad) ** 2
                + (det_range_m * self._cam_noise.yaw_noise_rad) ** 2
            )
            std_theta = self._cam_noise.yaw_noise_rad

            with self._pf_lock:
                if not self._initialized:
                    # First absolute fix before anything else arrived: seed the
                    # cloud around the tag reading (wider than the measurement
                    # std to absorb the transform error).
                    self.pf.initialize_near(
                        anchor_x, anchor_y, anchor_theta,
                        std_xy_m=0.5, std_theta_rad=0.25,
                    )
                    self._update_estimate()
                    self._initialized = True
                    print(
                        f"[pose_estimator] Initialized from AprilTag {tag_id}: "
                        f"x={anchor_x:.2f}m, y={anchor_y:.2f}m, "
                        f"θ={math.degrees(anchor_theta):.1f}°"
                    )
                    return

                # Association gate: a correct anchor must explain the latest
                # LIDAR scan at least as well as the current belief.  A
                # mis-associated tag puts the robot in the wrong part of the
                # map, where the observed walls do not fit — reject it before
                # it can drag the estimate off the truth.
                if TAG_GATE_ENABLED and not self._tag_anchor_fits_scan(
                    anchor_x, anchor_y, anchor_theta
                ):
                    print(
                        f"[pose_estimator] Tag {tag_id} anchor rejected: "
                        "derived pose contradicts the latest LIDAR scan "
                        "(likely mis-association)."
                    )
                    continue

                # Localization-jump safety: if this tag reading differs from
                # what the robot currently believes in *position* by more than
                # the threshold, the belief was badly wrong.  Command the
                # controller to halt (via the ``estimate/halt`` topic) so the
                # robot stops and lets the corrected pose settle before it
                # keeps moving.  The heading is IMU-driven and is not used for
                # this.
                jump_m = math.hypot(
                    anchor_x - self.estimated_x, anchor_y - self.estimated_y
                )
                if jump_m > RELOCALIZE_JUMP_M:
                    self._pub_halt.put(
                        encode(
                            "estimate/halt",
                            {"t": time.time(), "hold_s": RELOCALIZE_HOLD_S},
                        )
                    )
                    print(
                        f"[pose_estimator] Localization jump {jump_m:.2f} m — "
                        "sending halt command."
                    )

                # Back-propagate the measurement to its capture time: the
                # motion between when the camera frame was taken and now is
                # what fuse_absolute_pose needs to time-align the measurement.
                if tag_t is not None:
                    now_t = time.time()
                    delta_forward, delta_theta = self._motion_since(
                        float(tag_t), now_t
                    )
                else:
                    delta_forward, delta_theta = 0.0, 0.0

                fused = self.pf.fuse_absolute_pose(
                    anchor_x, anchor_y, anchor_theta,
                    std_xy, std_theta,
                    delta_forward_m=delta_forward,
                    delta_theta=delta_theta,
                )
                if not fused:
                    print(
                        f"[pose_estimator] AprilTag {tag_id} contradicted the "
                        "whole cloud — recovery re-seed applied."
                    )
                self._update_estimate()
                self._last_tag_fuse_mono = time.monotonic()

                print(
                    f"[pose_estimator] AprilTag {tag_id} fused: "
                    f"x={anchor_x:.2f}m, y={anchor_y:.2f}m, "
                    f"θ={math.degrees(anchor_theta):.1f}°"
                )
                # Only use the first tag per message.
                break

    def _tag_anchor_fits_scan(
        self, x: float, y: float, theta: float
    ) -> bool:
        """Whether a tag-derived pose explains the latest LIDAR scan at least
        as well as the current belief.

        Returns True (gate skipped) when the last scan is missing, stale, or
        too uninformative to judge.

        Called under _pf_lock.
        """
        angles = self._latest_match_angles
        observed = self._latest_match_observed
        if angles is None or observed is None:
            return True
        if time.monotonic() - self._latest_match_time > TAG_GATE_SCAN_MAX_AGE_S:
            return True
        if self._scan_info_score < SCAN_MATCH_MIN_INFO:
            return True
        walls = self.map_data.walls
        max_range = self._lidar_range_m
        err_anchor = scan_match_error(
            (x, y, theta), walls, angles, observed, max_range
        )
        err_est = scan_match_error(
            (self.estimated_x, self.estimated_y, self.estimated_theta),
            walls, angles, observed, max_range,
        )
        return err_anchor <= (
            TAG_GATE_FACTOR * err_est + TAG_GATE_MARGIN_M
        )

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
            self._update_estimate()
            self._initialized = True
            self._last_scan_t = None

        print(f"[pose_estimator] Initialized particles near " f"({cx:.2f}, {cy:.2f}) m")

    # -------------------------------------------------------------------
    # Filter helpers (predict → update → estimate → resample)
    # -------------------------------------------------------------------

    def _prune_histories(self, cutoff: float):
        """Drop history entries older than ``cutoff``.

        Called under _pf_lock.
        """
        while self._odom_history and self._odom_history[0][0] < cutoff:
            self._odom_history.pop(0)
        while self._heading_history and self._heading_history[0][0] < cutoff:
            self._heading_history.pop(0)

    def _motion_since(self, from_t: float, to_t: float) -> Tuple[float, float]:
        """Sum forward and fused-heading deltas with from_t < t <= to_t."""
        fwd = sum(df for (t, df) in self._odom_history if from_t < t <= to_t)
        th = sum(dt for (t, dt) in self._heading_history if from_t < t <= to_t)
        return fwd, th

    def _update_estimate(self):
        """Refresh the published pose/covariance/scales from the filter.

        Called under _pf_lock.
        """
        self.estimated_x, self.estimated_y, self.estimated_theta = self.pf.estimate()
        self._cov = self.pf.covariance().tolist()
        self._odom_scale = self.pf.odom_scale_estimate()
        self._track_scale = self._heading_filter.track_scale
        self._gyro_bias = self._heading_filter.gyro_bias

    def _maybe_resample(self):
        """Adaptive resampling, only when the weights have degenerated.

        Resampling is always *local* (random injection stays in a small disc
        around the current estimate).  Global random injection scatters
        particles across the whole (symmetric) map and is what made the estimate
        teleport between similar-looking rooms mid-run.  Global re-localization
        is the AprilTag's job — an absolute measurement fused Bayesian-style —
        not the LIDAR's.  When the scan is too uninformative to localize (open
        space), the likelihood is near-flat, so a low ESS just means the
        measurement cannot discriminate; the weights stay near uniform there and
        this does not fire.

        Called under _pf_lock.
        """
        ess = self.pf.effective_sample_size()
        if ess < ESS_RESAMPLE_FRACTION * self.pf.num_particles:
            self.pf.resample(
                inject_mode="local",
                anchor=(self.estimated_x, self.estimated_y),
            )

    def _apply_filter_step(self, from_t: float, to_t: float, ray_hits: List[RayHit]):
        """Predict motion over (from_t, to_t], then update + resample.

        Called under _pf_lock.
        """
        fwd, th = self._motion_since(from_t, to_t)
        if fwd != 0.0 or th != 0.0:
            self.pf.predict(fwd, th)
        self.pf.update(ray_hits)
        self._maybe_resample()

    def _refine_from_scan(self, angles: np.ndarray, observed: np.ndarray):
        """Snap the estimate to the pose that best explains the LIDAR scan.

        Runs the Gauss-Newton point-to-line ICP from the current estimate,
        falling back to coordinate descent if it diverges.  If the refined pose
        improves the match AND does not jump the estimate far, the particle
        cloud and the estimate are re-anchored to it.  Called under _pf_lock.
        """
        walls = self.map_data.walls
        max_range = self._lidar_range_m

        est = (self.estimated_x, self.estimated_y, self.estimated_theta)
        e0 = scan_match_error(est, walls, angles, observed, max_range)

        refined = refine_gauss_newton(est, walls, angles, observed, max_range)

        def _acceptable(candidate):
            jump = math.hypot(candidate[0] - est[0], candidate[1] - est[1])
            e = scan_match_error(candidate, walls, angles, observed, max_range)
            # Accept only if the match meaningfully improves the fit (error at
            # least halved) and did not move the estimate implausibly far.  A
            # fixed 1 m cap would reject the very corrections this exists to
            # make (the particle mean can start a couple of metres off).
            return e < e0 * SCAN_MATCH_ACCEPT_RATIO and jump <= SCAN_MATCH_MAX_JUMP_M

        if not _acceptable(refined):
            refined = refine_discrete(est, walls, angles, observed, max_range)
            if not _acceptable(refined):
                return  # neither solver produced a trusted fix

        self.pf.snap_absolute(
            refined[0], refined[1], refined[2],
            SCAN_MATCH_SNAP_STD_M, SCAN_MATCH_SNAP_STD_RAD,
        )
        self.estimated_x, self.estimated_y, self.estimated_theta = refined

    def _process_lidar(self, sample):
        """Parse and process a LIDAR scan (predict motion + update + resample)."""
        if not self._initialized:
            return

        try:
            lidar_data = decode("sensor/lidar", sample)
        except SchemaError as exc:
            print(f"[pose_estimator] sensor/lidar dropped: {exc}")
            return

        scan_t = lidar_data.get("t")
        if scan_t is None:
            scan_t = time.time()
        scan_t = float(scan_t)

        ray_hits = self._parse_rays(lidar_data.get("rays", []))

        # Informativeness: fraction of rays that hit a real surface well inside
        # the sensor's max range.  In open space most rays return "no return"
        # near max range, so the scan carries almost no structure to localize
        # against.
        info_range = SCAN_INFO_RANGE_FRAC * self._lidar_range_m
        informative = sum(1 for h in ray_hits if h.distance < info_range)
        scan_info_score = informative / len(ray_hits) if ray_hits else 1.0

        with self._pf_lock:
            self._scan_info_score = scan_info_score
            from_t = self._last_scan_t if self._last_scan_t is not None else float("-inf")
            self._apply_filter_step(from_t, scan_t, ray_hits)
            self._update_estimate()

            # Scan-matching refinement ("gradient descent"): snap the estimate
            # to the pose that best explains the scan, unless the scan is too
            # open to match against or an AprilTag fused very recently (the
            # tag measurement owns the update while it is active).
            tag_recent = (
                time.monotonic() - self._last_tag_fuse_mono < SCAN_MATCH_TAG_HOLD_S
            )
            angles = np.asarray(
                [h.angle for h in ray_hits][::SCAN_MATCH_RAY_STEP], dtype=np.float64
            )
            observed = np.asarray(
                [h.distance for h in ray_hits][::SCAN_MATCH_RAY_STEP], dtype=np.float64
            )
            if (
                SCAN_MATCH_ENABLED
                and not tag_recent
                and scan_info_score >= SCAN_MATCH_MIN_INFO
            ):
                self._refine_from_scan(angles, observed)

            # Cache the scan (subsampled to the match resolution) for the
            # AprilTag association gate in _fuse_apriltag.
            self._latest_match_angles = angles
            self._latest_match_observed = observed
            self._latest_match_time = time.monotonic()

            self._last_scan_t = scan_t

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
        """Publish the estimated pose (and its covariance) as JSON.

        The published pose is always the particle-filter mean.  With Bayesian
        fusion there is no separate "tag mode" or "odometry fallback": when the
        LIDAR is uninformative (open space) the update likelihood is flat, the
        weights stay near uniform, and the filter degrades gracefully to
        motion-model dead reckoning with honestly growing covariance — which
        downstream consumers read from ``cov``.
        """
        with self._pf_lock:
            x, y, theta = self.estimated_x, self.estimated_y, self.estimated_theta
            cov = self._cov
            scan_info = self._scan_info_score

        std_xy = math.sqrt(cov[0][0] + cov[1][1])
        if std_xy > LOST_XY_STD_M and time.monotonic() - self._last_lost_warn > 5.0:
            self._last_lost_warn = time.monotonic()
            print(
                f"[pose_estimator] WARNING: localization uncertain "
                f"(std_xy={std_xy:.2f} m)"
            )

        msg = encode(
            "estimate/pose",
            {
                "x_m": x,
                "y_m": y,
                "theta_rad": theta,
                "cov": cov,
                "odom_scale": self._odom_scale,
                "track_scale": self._track_scale,
                "gyro_bias": self._gyro_bias,
                "scan_info": round(scan_info, 4),
                "t": time.time(),
            },
        )
        self._pub_pose.put(msg)

        print(
            f"[pose_estimator] Pose: x={x:.2f}m, y={y:.2f}m, "
            f"θ={math.degrees(theta):.1f}° std_xy={std_xy:.2f}m "
            f"scan_info={scan_info:.2f} "
            f"ws={self._odom_scale:.3f}, ts={self._track_scale:.3f}"
        )

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------

    def run(self):
        """Block forever, processing the latest LIDAR scan and publishing.

        RingChannel(1) on the LIDAR subscriber ensures recv() always
        returns the newest scan.  AprilTag detections are fused on their
        own thread at the camera rate.
        """
        print("[pose_estimator] Running. Press Ctrl+C to stop.")
        try:
            while True:
                # Block until a LIDAR sample is available.
                # RingChannel(1) ensures we always get the newest one.
                sample = self._sub_lidar.recv()
                self._process_lidar(sample)

                # Publish the current estimate.
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
