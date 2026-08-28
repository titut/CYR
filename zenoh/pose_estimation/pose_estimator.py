"""Zenoh-based pose estimator.

Subscribes to sensor/lidar, sensor/wheel_speed and sensor/imu, runs a particle
filter, and publishes the estimated pose on estimate/pose.

Heading is fused from the IMU gyro + wheel-odometry angular rate + absolute yaw
by a small EKF (heading_filter.HeadingFilter), which also estimates the gyro
bias and track scale.  The particle filter estimates the pose and the wheel
radius (forward) scale.

An AprilTag fix is authoritative when available: each fresh detection re-seeds
the pose (and the particle cloud / odometry fallback) directly onto the smoothed
tag reading at the camera rate, bypassing the filter.  When no tag is visible,
localization falls back to LIDAR particle filtering, and to dead-reckoned
odometry when the scan is too unstructured to localize.

Uses RingChannel(1) for the LIDAR subscriber so the particle filter
always processes only the most recent scan — stale backlogs are dropped.

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
# They are used to derive the anchor measurement's covariance instead of
# hardcoding it.

# A fresh AprilTag is authoritative: for this long after an anchor, suppress the
# adaptive resample entirely.  Resampling resets weights to uniform and re-scatters
# particles around the current estimate, which washes the anchor out and lets a
# weak (open-space) LIDAR scan drag the belief back off the tag.  While the hold
# is active the cloud stays where the tag put it; the next anchor (or a
# structured scan) corrects it again.
ANCHOR_RESAMPLE_HOLD_S = 5.0

# AprilTag readings are accurate but noisy per-detection (camera range/bearing/
# yaw noise, ~0.2 m at typical tag range).  Snapping the cloud to every single
# reading makes the estimate follow that noise (it jumps around while parked).
# Instead the anchor stream is low-pass filtered with this EMA alpha before the
# snap, so the estimate converges to the *mean* of the readings (which sits on
# the tag) with the per-detection jitter smoothed out.  Tuned for the anchor
# rate: at 30 Hz, alpha 0.2 gives ~0.06 m steady-state jitter and sub-200 ms
# response.
ANCHOR_SMOOTH_ALPHA = 0.2

# Reset the anchor EMA when the same tag has not been seen for this long (or a
# different tag appears): the EMA only smooths consecutive readings of one tag,
# so a long gap means the robot has moved and the held value is stale.
ANCHOR_EMA_RESET_S = 1.0

# How often a fresh AprilTag reading may re-anchor the pose.  Set to the camera
# detection rate (30 Hz): a tag fix is authoritative, so each reading updates
# the pose.  The EMA above keeps the pose smooth across these samples.
ANCHOR_RATE_LIMIT_S = 1.0 / 30.0

# While an AprilTag fix is this fresh, the published pose IS the (smoothed) tag
# reading — particle filter, odometry fallback and LIDAR are bypassed entirely.
# This is the "if a tag is available, it's the goto" rule: no filtering
# tug-of-war, no dead-reckoning drift, the estimate sits on the bot until the
# tag is lost.
ANCHOR_FRESH_S = 2.0
ANCHOR_FRESH_STD_M = 0.05      # published xy std while in "tag" mode
ANCHOR_FRESH_STD_RAD = 0.02    # published heading std while in "tag" mode

# Localization-jump safety: when an AprilTag fix differs from the current
# estimate in *position* by more than this threshold, the belief was badly
# wrong.  The estimator publishes a halt command on ``estimate/halt``; the
# controller subscribes and holds position for ``RELOCALIZE_HOLD_S`` seconds so
# the corrected pose settles before the robot keeps moving.  (Only the
# positional discrepancy is considered — the heading is IMU-driven and is not
# a reliable "lost" signal.)
RELOCALIZE_JUMP_M = 0.5                 # positional discrepancy (m)
RELOCALIZE_HOLD_S = 2.0                 # hold duration sent to the controller (s)

# Tag-association gating: a detection whose derived anchor contradicts the
# LIDAR (an independent sensor) is a mis-associated tag — e.g. the id read
# wrong at long range.  Field logs: a far tag (~5 m) dragged the estimate 3 m
# off over 2.6 s while the anchor age reset kept confidence at 1.0, so the
# navigator's recovery never triggered.  A *correct* anchor always explains
# the observed scan better than a wrong belief, and a mis-associated one does
# not, so the anchor is only accepted when it fits the scan at least as well
# as the current estimate (or when the scan is too uninformative to judge).
TAG_GATE_ENABLED = True
TAG_GATE_FACTOR = 1.5        # reject if anchor error > factor * current estimate error
TAG_GATE_MARGIN_M = 0.05     # absolute slack above the noise floor
TAG_GATE_SCAN_MAX_AGE_S = 1.0  # skip the gate if the last scan is older than this
TAG_RANGE_MARGIN_M = 0.25    # reject detections beyond camera max range + margin

# Scan-matching refinement (the "gradient descent" element).  When the scan is
# structured enough and no AprilTag is fresh, snap the estimate to the pose that
# best explains the LIDAR scan — Gauss-Newton ICP, with coordinate descent as a
# robust fallback — and re-anchor the cloud + odometry fallback to it.
SCAN_MATCH_ENABLED = True
SCAN_MATCH_MIN_INFO = 0.4               # run only when scan informativeness >= this
SCAN_MATCH_RAY_STEP = 4                 # subsample 360 rays -> 90 for cheap matching
SCAN_MATCH_MAX_JUMP_M = 5.0             # reject a refinement that moves the estimate this far
SCAN_MATCH_ACCEPT_RATIO = 0.5           # accept only if the error is at least halved
SCAN_MATCH_SNAP_STD_M = 0.05            # cloud re-anchor spread after a match
SCAN_MATCH_SNAP_STD_RAD = 0.02


def blend_pose(
    prev: Optional[Tuple[float, float, float]],
    new: Tuple[float, float, float],
    alpha: float,
) -> Tuple[float, float, float]:
    """Exponential moving average of an absolute pose stream (circular θ)."""
    if prev is None:
        return new
    px, py, pt = prev
    nx, ny, nt = new
    dtheta = math.atan2(math.sin(nt - pt), math.cos(nt - pt))
    bx = alpha * nx + (1.0 - alpha) * px
    by = alpha * ny + (1.0 - alpha) * py
    bt = pt + alpha * dtheta
    return (bx, by, bt)

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
# range), so the scan carries almost no positional information; trusting the
# particle mean there is unreliable (it can jump when a resample scatters the
# cloud), so the pose falls back to odometry dead-reckoning until a structured
# scan or an AprilTag anchors it again.
SCAN_INFO_RANGE_FRAC = 0.9    # a ray is informative if it hits inside 90% of range
SCAN_INFO_THRESHOLD = 0.4    # need >=40% informative rays to trust the scan

# When falling back to odometry, report this positional standard deviation so
# downstream consumers treat the estimate as uncertain (the odometry pose
# drifts slowly, but it is a fallback, not a confident fix).
ODOM_FALLBACK_STD_M = 1.0

# Confidence model (0..1) published on estimate/pose.  The robot is only
# considered "unconfident" when BOTH of these hold at once: it has gone a while
# without an AprilTag anchor AND its LIDAR scan is blind (open space with no
# structure to localize against).  A fresh anchor or a structured scan each keep
# confidence near 1 on their own, so a robot that just anchored in open space —
# or that has been driving walls with a stale anchor — does not waste time
# re-localizing.  When both degrade, confidence falls toward 0 and the navigator
# can send the robot to re-anchor at the nearest tag.
ANCHOR_AGE_DECAY_S = 20.0  # anchor "freshness" time constant (s)


def compute_confidence(anchor_age_s: float, scan_info_score: float) -> float:
    """Confidence in [0, 1] for the current pose estimate.

    ``anchor_age_s`` is the time since the last AprilTag anchor; 
    ``scan_info_score`` is the fraction of LIDAR rays that hit real structure
    (0 = blind, 1 = well-structured).  Confidence falls toward 0 only when the
    anchor is stale AND the LIDAR is blind.
    """
    stale = 1.0 - math.exp(-max(0.0, anchor_age_s) / ANCHOR_AGE_DECAY_S)
    blind = 1.0 - min(1.0, max(0.0, scan_info_score))
    return max(0.0, min(1.0, 1.0 - stale * blind))


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

        # Recent LIDAR scans (scan_t, ray_hits) for the motion/heading window.
        # Bounded to HISTORY_S.  Protected by _pf_lock.
        self._lidar_history: List[Tuple[float, List[RayHit]]] = []

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

        # Odometry dead-reckoning fallback pose (x, y, theta).  Integrated from
        # the wheel forward deltas and fused heading deltas that the particle
        # filter's motion model already consumes, and re-anchored to the filter
        # estimate whenever a scan is informative enough to trust.  Published in
        # place of the particle mean when the latest scan carries too little
        # structure to localize (open space), so a flat likelihood can never
        # jump the estimate.  Protected by _pf_lock.
        self._odom_x: float = 0.0
        self._odom_y: float = 0.0
        self._odom_theta: float = 0.0
        # Informativeness of the latest processed LIDAR scan (0..1).
        self._scan_info_score: float = 1.0

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

        # -------------------------------------------------------------------
        # Latest AprilTag detection, set by callback, consumed by main loop.
        # Protected by _tag_det_lock.
        # -------------------------------------------------------------------
        self._tag_det_lock = threading.Lock()
        self._latest_tag_dets: Optional[Tuple[Optional[float], List[dict]]] = None
        self._last_anchor_time: float = 0.0  # rate-limit anchoring to 2 Hz
        # Low-pass filtered anchor pose (x, y, θ), smoothed across detections
        # so a parked robot's estimate does not follow per-detection noise.
        self._anchor_smooth: Optional[Tuple[float, float, float]] = None
        # The tag id the current EMA was built from; a different tag resets it.
        self._anchor_tag_id: Optional[int] = None

        # Latest LIDAR scan subsampled for matching (guarded by _pf_lock), used
        # by the tag-association gate to sanity-check tag anchors.
        self._latest_match_angles: Optional[np.ndarray] = None
        self._latest_match_observed: Optional[np.ndarray] = None
        self._latest_match_time: float = float("-inf")

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
                delta_forward = linear_mps * dt
                with self._pf_lock:
                    self._odom_history.append((t, delta_forward))
                    self._last_odom_angular_rps = angular_rps
                    # Dead-reckon the odometry fallback pose.  The heading used
                    # here is the fused heading (gyro+odom+yaw), so the fallback
                    # inherits the heading filter's accuracy, not raw wheel rates.
                    # The forward motion is scaled by the wheel-radius estimate so
                    # the fallback drifts the way the filter *believes* it moves.
                    fwd = delta_forward * self._odom_scale
                    self._odom_x += fwd * math.cos(self._odom_theta)
                    self._odom_y += fwd * math.sin(self._odom_theta)
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
                    # Same fused heading increment feeds the odometry fallback.
                    self._odom_theta += dtheta
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
            data = decode("detection/apriltag", sample)
            dets = data.get("detections", [])
            tag_t = data.get("t")
        except SchemaError as exc:
            print(f"[pose_estimator] detection/apriltag dropped: {exc}")
            return
        with self._tag_det_lock:
            self._latest_tag_dets = (tag_t, dets)

    def _tag_anchor_fits_scan(
        self, x: float, y: float, theta: float
    ) -> bool:
        """Whether a tag-derived pose explains the latest LIDAR scan at least
        as well as the current belief.

        Returns True (gate skipped) when the last scan is missing, stale, or
        too uninformative to judge — the navigator's confidence machinery
        covers those cases instead.
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

            # Association gate: a correct anchor must explain the latest LIDAR
            # scan at least as well as the current belief.  A mis-associated
            # tag puts the robot in the wrong part of the map, where the
            # observed walls do not fit — reject it before it can drag the
            # estimate (and the anchor-age-based confidence) off the truth.
            with self._pf_lock:
                if TAG_GATE_ENABLED and not self._tag_anchor_fits_scan(
                    anchor_x, anchor_y, anchor_theta
                ):
                    print(
                        f"[pose_estimator] Tag {tag_id} anchor rejected: "
                        "derived pose contradicts the latest LIDAR scan "
                        "(likely mis-association)."
                    )
                    continue

                # The tag fix is authoritative.  Rate-limit to the camera
                # detection rate so the pose tracks the live tag reading while
                # it stays in view.
                now = time.monotonic()
                if now - self._last_anchor_time >= ANCHOR_RATE_LIMIT_S:
                    # Localization-jump safety: if this tag reading differs from
                    # what the robot currently believes in *position* by more than
                    # the threshold, the belief was badly wrong.  Command the
                    # controller to halt (via the ``estimate/halt`` topic) so the
                    # robot stops and lets the corrected pose settle before it keeps
                    # moving.  The heading is IMU-driven and is not used for this.
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

                    # Measurement covariance from the camera noise model and the
                    # detection geometry (distance to the tag).
                    range_m = math.hypot(x_rel_m, y_rel_m)
                    std_xy = math.sqrt(
                        self._cam_noise.range_noise_m ** 2
                        + (range_m * self._cam_noise.bearing_noise_rad) ** 2
                        + (range_m * self._cam_noise.yaw_noise_rad) ** 2
                    )
                    std_theta = self._cam_noise.yaw_noise_rad

                    # Smooth the anchor stream before snapping, so the estimate
                    # converges to the mean of the readings (on the tag) instead of
                    # following each noisy detection.  The EMA only smooths *within*
                    # a tag: when a different tag is seen, or this tag reappears
                    # after a gap, reset the EMA so the estimate snaps to the new
                    # absolute reading instead of blending from a stale room (which
                    # is what made the estimate "snap" into the meeting room while
                    # the robot was actually in the office).
                    if (
                        self._anchor_tag_id is not None
                        and self._anchor_tag_id != tag_id
                    ) or (now - self._last_anchor_time) > ANCHOR_EMA_RESET_S:
                        self._anchor_smooth = None
                    self._anchor_tag_id = tag_id

                    self._anchor_smooth = blend_pose(
                        self._anchor_smooth,
                        (anchor_x, anchor_y, anchor_theta),
                        ANCHOR_SMOOTH_ALPHA,
                    )
                    sx, sy, _ = self._anchor_smooth
                    self._last_anchor_time = now

                    # A tag fix is authoritative for position: snap the cloud and
                    # the position to the (smoothed) tag reading.  The heading is
                    # left to the heading filter (gyro + magnetometer yaw, both
                    # continuous and accurate) — re-anchoring the heading to the
                    # EMA-smoothed tag heading would lag the true heading during a
                    # fast rotation (the EMA smooths the spin and the tag is only
                    # intermittently visible), which is what caused the heading to
                    # lag right after leaving tag view.
                    fused_theta = self._heading_filter.heading
                    self.pf.snap_absolute(sx, sy, fused_theta, std_xy, std_theta)
                    self.estimated_x, self.estimated_y, self.estimated_theta = (
                        sx, sy, fused_theta,
                    )
                    self._odom_x, self._odom_y, self._odom_theta = (
                        sx, sy, fused_theta,
                    )

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
            self._odom_x, self._odom_y, self._odom_theta = (
                self.estimated_x,
                self.estimated_y,
                self.estimated_theta,
            )
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
        """Adaptive resampling, only when the weights have degenerated.

        Resampling is always *local* (random injection stays in a small disc
        around the current estimate).  Global random injection scatters
        particles across the whole (symmetric) map and is what made the estimate
        teleport between similar-looking rooms mid-run.  Global re-localization
        is the AprilTag's job — an absolute, cm-accurate anchor — not the
        LIDAR's.  When the scan is too uninformative to localize (open space),
        the likelihood is near-flat, so a low ESS just means the measurement
        cannot discriminate; scattering particles across the map there would
        average random poses and jump the estimate, so staying local is correct
        in every case.
        """
        ess = self.pf.effective_sample_size()
        if ess < ESS_RESAMPLE_FRACTION * self.pf.num_particles:
            # A fresh AprilTag already fixed the belief and set uniform weights;
            # resampling here would scatter particles around the current
            # estimate and let a weak LIDAR scan drag the cloud off the tag.
            if time.monotonic() - self._last_anchor_time < ANCHOR_RESAMPLE_HOLD_S:
                return
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
        cloud, the estimate and the odometry fallback are re-anchored to it
        (mirrors the AprilTag snap, but LIDAR-driven).  Called under _pf_lock.
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
        self._odom_x, self._odom_y, self._odom_theta = refined

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

            # Scan-matching refinement ("gradient descent"): snap the estimate to
            # the pose that best explains the scan, unless a fresh AprilTag is
            # authoritative (tag mode is already cm-accurate) or the scan is too
            # open to match against.
            anchor_age = time.monotonic() - self._last_anchor_time
            angles = np.asarray(
                [h.angle for h in ray_hits][::SCAN_MATCH_RAY_STEP], dtype=np.float64
            )
            observed = np.asarray(
                [h.distance for h in ray_hits][::SCAN_MATCH_RAY_STEP], dtype=np.float64
            )
            if (
                SCAN_MATCH_ENABLED
                and scan_info_score >= SCAN_MATCH_MIN_INFO
                and anchor_age >= ANCHOR_FRESH_S
            ):
                self._refine_from_scan(angles, observed)

            # Cache the scan (subsampled to the match resolution) for the
            # AprilTag association gate in _process_apriltags.
            self._latest_match_angles = angles
            self._latest_match_observed = observed
            self._latest_match_time = time.monotonic()

            # When the scan is informative enough to trust the particle
            # estimate, re-anchor the odometry fallback pose to it so the
            # fallback does not accumulate unbounded drift during the stretches
            # where we *do* localize.
            if scan_info_score >= SCAN_INFO_THRESHOLD:
                self._odom_x, self._odom_y, self._odom_theta = (
                    self.estimated_x,
                    self.estimated_y,
                    self.estimated_theta,
                )

            self._last_scan_t = scan_t

            # Record the scan for the motion/heading history.
            self._lidar_history.append((scan_t, ray_hits))
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
        """Publish the estimated pose (and its covariance) as JSON.

        When the latest LIDAR scan carries too little structure to localize
        against (open space) — or the particle filter's covariance has grown
        past the "lost" threshold — the particle mean is unreliable: a flat
        likelihood cannot correct it, and a resample can jump it arbitrarily.
        In that case publish the odometry dead-reckoned pose instead.  It
        drifts slowly (the drift over a few seconds of open space is small),
        and it is re-anchored to the filter the moment an informative scan or
        AprilTag arrives.  Open space also means there is little to collide
        with, so a slightly stale pose is low-risk.
        """
        with self._pf_lock:
            est_x, est_y, est_theta = self.estimated_x, self.estimated_y, self.estimated_theta
            odom_x, odom_y, odom_theta = self._odom_x, self._odom_y, self._odom_theta
            cov = self._cov
            scan_info = self._scan_info_score
            anchor_smooth = self._anchor_smooth
            anchor_age = time.monotonic() - self._last_anchor_time

        # A fresh AprilTag fix is authoritative: publish the smoothed tag pose
        # directly, bypassing the particle filter / odometry / LIDAR entirely.
        # No filtering tug-of-war — the estimate sits on the bot.
        if anchor_smooth is not None and anchor_age < ANCHOR_FRESH_S:
            # Position comes from the (smoothed) tag reading; heading comes from
            # the gyro-integrated heading filter, which the tag anchors on every
            # detection.  Using the raw tag heading here would freeze the
            # estimate when the tag leaves view while the robot is turning, so
            # the controller would never see its heading change and would spin
            # forever chasing an error that never converges.
            x, y = anchor_smooth[0], anchor_smooth[1]
            theta = odom_theta
            cov = [
                [ANCHOR_FRESH_STD_M**2, 0.0, 0.0],
                [0.0, ANCHOR_FRESH_STD_M**2, 0.0],
                [0.0, 0.0, ANCHOR_FRESH_STD_RAD**2],
            ]
            use_odom = False
        else:
            std_xy = math.sqrt(cov[0][0] + cov[1][1])
            use_odom = scan_info < SCAN_INFO_THRESHOLD or std_xy > LOST_XY_STD_M

            if use_odom:
                x, y, theta = odom_x, odom_y, odom_theta
                # Report the odometry fallback as uncertain so downstream consumers
                # don't treat it as a confident fix.  (The particle covariance may
                # still be small even when its mean is unreliable.)
                cov = [
                    [ODOM_FALLBACK_STD_M**2, 0.0, 0.0],
                    [0.0, ODOM_FALLBACK_STD_M**2, 0.0],
                    [0.0, 0.0, max(cov[2][2], (math.pi / 6) ** 2)],
                ]
            else:
                x, y, theta = est_x, est_y, est_theta

        # Confidence for the recovery behavior: low only when the anchor is
        # stale AND the LIDAR scan is blind (see compute_confidence).
        confidence = compute_confidence(anchor_age, scan_info)
        mode = "tag" if anchor_smooth is not None and anchor_age < ANCHOR_FRESH_S else (
            "odom" if use_odom else "pf"
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
                "confidence": round(confidence, 4),
                "anchor_age_s": round(anchor_age, 2),
                "scan_info": round(scan_info, 4),
                "mode": mode,
                "t": time.time(),
            },
        )
        self._pub_pose.put(msg)

        # Warn (rate-limited) when localization has likely diverged.
        std_xy = math.sqrt(cov[0][0] + cov[1][1])
        if std_xy > LOST_XY_STD_M and time.monotonic() - self._last_lost_warn > 5.0:
            self._last_lost_warn = time.monotonic()
            print(
                f"[pose_estimator] WARNING: localization uncertain "
                f"(std_xy={std_xy:.2f} m)"
            )

        print(
            f"[pose_estimator] Pose: x={x:.2f}m, y={y:.2f}m, "
            f"θ={math.degrees(theta):.1f}° [{mode}] "
            f"conf={confidence:.2f} anchor_age={anchor_age:.0f}s "
            f"scan_info={scan_info:.2f} "
            f"ws={self._odom_scale:.3f}, ts={self._track_scale:.3f}"
        )

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------

    def run(self):
        """Block forever, processing the latest LIDAR and AprilTag data.

        RingChannel(1) on the LIDAR subscriber ensures recv() always
        returns the newest scan.  After each filter update, any pending
        AprilTag detection is consumed (authoritative snap), then the
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
