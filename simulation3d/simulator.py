"""3D PyBullet simulator core loop (T3D-04).

Builds the base + arm + world, runs a deterministic fixed-step physics loop
(gravity, 4 substeps per 60 Hz frame, seeded RNG), and bridges to zenoh:

    Publishes:   sim/truth/pose   — base ground-truth (x, y, theta) at 50 Hz
                 sensor/wasd      — teleop key state
                 object/registry  — current object poses every 1 s
    Subscribes:  (none yet; wheel-speed + arm topics come in T3D-06/08)

Headless (DIRECT) by default; ``--gui`` opens the PyBullet viewer with arrow-key
teleop (drive), 1/2/3 camera modes, ESC to quit.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

# Allow running this file directly (python3 simulation3d/simulator.py): the
# project root must be on sys.path for `core.*` / `simulation.*` imports.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pybullet as p
import pybullet_data
import zenoh

from core.clock import seed_all, sleep_until
from core.map_format import MapData
from core.messages import SchemaError, decode, decode_path, encode
from core.robot_config import get_robot_config
from simulation.kinematics import unicycle_to_wheel, wheel_to_unicycle
from simulation3d.sensors import detect_apriltags, read_imu, scan_lidar
from simulation3d.world import (
    build_ground,
    build_walls,
    load_world_registry,
    object_registry_message,
    spawn_objects,
    update_object_poses,
)

PHYSICS_STEP_S = 1.0 / 240.0
SUBSTEPS = 4
LOOP_HZ = 60
SENSOR_HZ = 50
REGISTRY_HZ = 1
# If no fresh sensor/wheel_speed arrives within this window (drive.py not
# running), fall back to direct WASD teleop.
WHEEL_TIMEOUT_S = 0.2

# Teleop acceleration ramp: standalone teleop jumps to full speed instantly,
# which pitches the base forward (arm up).  These cap how fast it accelerates.
_TELEOP_DT = 1.0 / LOOP_HZ
# Teleop ramps the velocity up gently so the base doesn't lunge to full speed
# (which pitches the high-COM arm base).  With the casters rolling freely the
# base stays level, so this ramp is just for comfort, not stability.
_TELEOP_ACCEL_MPS2 = 1.2  # linear accel cap (m/s²)
_TELEOP_ANG_ACCEL_RPS2 = 2.0  # angular accel cap (rad/s²)

# Wheel joint names in base.urdf (T3D-01).
_DRIVE_L = "wheel_drive_L_joint"
_DRIVE_R = "wheel_drive_R_joint"


def _ramp(current: float, target: float, max_step: float) -> float:
    """Move ``current`` toward ``target`` by at most ``max_step``."""
    if current < target:
        return min(target, current + max_step)
    return max(target, current - max_step)


def _mouse_ray(x: float, y: float) -> tuple[list, list]:
    """World-space ray under a GUI pixel click (official PyBullet getRayFromTo
    formulation).  ``x``/``y`` are PyBullet mouse coords (top-left origin)."""
    cam = p.getDebugVisualizerCamera()
    width, height = cam[0], cam[1]
    cam_forward, horizon, vertical = cam[5], cam[6], cam[7]
    dist, cam_target = cam[10], cam[11]
    cam_pos = [
        cam_target[0] - dist * cam_forward[0],
        cam_target[1] - dist * cam_forward[1],
        cam_target[2] - dist * cam_forward[2],
    ]
    ray_forward = [cam_target[i] - cam_pos[i] for i in range(3)]
    inv_len = 10000.0 / math.sqrt(sum(c * c for c in ray_forward))
    ray_forward = [c * inv_len for c in ray_forward]
    d_hor = [c / width for c in horizon]
    d_ver = [c / height for c in vertical]
    ray_to = [
        cam_pos[i] + ray_forward[i] - 0.5 * horizon[i] + 0.5 * vertical[i]
        + x * d_hor[i] - y * d_ver[i]
        for i in range(3)
    ]
    return cam_pos, ray_to


def _ray_ground_intersection(ray_from, ray_to):
    """Where the ray crosses the ground plane z=0 -> (x, y), or None when the
    ray does not point downward (no ground in view)."""
    dx = ray_to[0] - ray_from[0]
    dy = ray_to[1] - ray_from[1]
    dz = ray_to[2] - ray_from[2]
    if dz >= 0:
        return None
    t = -ray_from[2] / dz
    return (ray_from[0] + t * dx, ray_from[1] + t * dy)


def _arrow(x: float, y: float, theta: float, length: float, color):
    """Debug-line arrow at (x, y) heading theta (rad, CCW from +X)."""
    items = []
    dx, dy = length * math.cos(theta), length * math.sin(theta)
    base = (x, y, 0.05)
    tip = (x + dx, y + dy, 0.05)
    items.append(p.addUserDebugLine(base, tip, color))
    back = (x + dx * 0.6, y + dy * 0.6, 0.05)
    px, py = -dy, dx
    norm = math.hypot(px, py) or 1.0
    s = 0.06 * length / (length or 1.0)
    px, py = px / norm * s, py / norm * s
    items.append(p.addUserDebugLine(tip, [back[0] + px, back[1] + py, 0.05], color))
    items.append(p.addUserDebugLine(tip, [back[0] - px, back[1] - py, 0.05], color))
    return items


class Simulator3D:
    def __init__(
        self,
        gui: bool = False,
        map_path: str | None = None,
        world_path: str | None = None,
        seed: int = 0,
        enable_zenoh: bool = True,
    ):
        self._cfg = get_robot_config()

        self._gui = gui
        connect = p.GUI if gui else p.DIRECT
        self._cid = p.connect(connect)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(
            numSubSteps=SUBSTEPS,
            fixedTimeStep=PHYSICS_STEP_S,
            deterministicOverlappingPairs=1,
        )
        p.setRealTimeSimulation(False)
        # PyBullet does not expose an RNG seed; determinism comes from the fixed
        # timestep + no realtime above.  `seed` seeds our own RNGs (used by the
        # sensor noise models in T3D-05) for reproducible runs.
        seed_all(seed)

        # ---- world (T3D-03) ----
        self._map = (
            MapData.from_json(map_path) if map_path and Path(map_path).exists() else None
        )
        self._ground = build_ground()
        self._wall_ids = build_walls(self._map)
        self._objects = load_world_registry(world_path)
        spawn_objects(self._objects)

        # ---- base (T3D-01), spawned inside the map ----
        ch = self._cfg.chassis
        base_z = ch.wheel_radius_m + ch.height_m / 2.0
        spawn_x, spawn_y = self._spawn_xy()
        self._base = p.loadURDF(
            "urdf/base.urdf", basePosition=[spawn_x, spawn_y, base_z + 0.02]
        )
        self._setup_wheel_friction()
        self._wheel_joints = {}
        for j in range(p.getNumJoints(self._base)):
            name = p.getJointInfo(self._base, j)[1].decode()  # joint name
            if name in (_DRIVE_L, _DRIVE_R):
                self._wheel_joints[name] = j

        # ---- arm (T3D-02), mounted rigidly on the base's arm_mount ----
        arm_urdf = self._cfg.arm.urdf
        arm_mount = [
            j
            for j in range(p.getNumJoints(self._base))
            if p.getJointInfo(self._base, j)[12].decode() == "arm_mount"
        ][0]
        # Load the arm already at the mount's world pose so the constraint has
        # no initial snap (which would shove the base).
        mount_pos, mount_orn = p.getLinkState(self._base, arm_mount)[:2]
        self._arm = p.loadURDF(
            arm_urdf, basePosition=mount_pos, baseOrientation=mount_orn
        )
        self._arm_constraint = p.createConstraint(
            self._base, arm_mount, self._arm, -1, p.JOINT_FIXED,
            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0, 1],
        )
        # The arm is rigidly mounted on the base, so its links must not collide
        # with the base (the primitive boxes would otherwise fight the mount).
        for la in range(-1, p.getNumJoints(self._arm)):
            for lb in range(-1, p.getNumJoints(self._base)):
                p.setCollisionFilterPair(self._base, self._arm, lb, la, enableCollision=0)

        # ---- teleop state ----
        self._keys = {"w": False, "a": False, "s": False, "d": False}
        self._camera_mode = 0
        self._teleop_lin = 0.0
        self._teleop_ang = 0.0
        # GUI rendering is expensive: resetting the camera and re-adding every
        # debug overlay line each frame is what pushes the loop below 60 Hz
        # (and the whole sim into slow motion).  Throttle both.
        self._last_gui_t = 0.0

        # ---- drive state (T3D-06) ----
        phy = self._cfg.physics
        self._physics = phy
        self._wheel_scale = random.uniform(phy.wheel_scale_min, phy.wheel_scale_max)
        self._track_scale = random.uniform(phy.track_scale_min, phy.track_scale_max)
        self._wheel_speed: tuple[float, float] | None = None
        self._wheel_speed_time: float = 0.0
        # Last messages from the nav stack (subscribed for display).
        self._est_pose: tuple[float, float, float] | None = None
        self._nav_path: list = []
        self._detected: list = []
        # Click-to-goal (GUI) + debug overlay of the nav stack's guess.
        self._nav_goal: tuple[float, float] | None = None
        self._overlay = True
        self._debug_items: list[int] = []

        # ---- sensor state (T3D-05) ----
        imu = self._cfg.sensors.imu
        self._gyro_bias_rps = random.uniform(-imu.gyro_bias_rps, imu.gyro_bias_rps)
        self._prev_lin_vel: tuple | None = None
        self._last_sensor_t = 0.0
        self._last_sensor_mono = time.monotonic()

        # ---- zenoh ----
        self._session = None
        if enable_zenoh:
            self._session = zenoh.open(zenoh.Config())
            self._pub_truth = self._session.declare_publisher(
                "sim/truth/pose",
                congestion_control=zenoh.CongestionControl.DROP,
                reliability=zenoh.Reliability.BEST_EFFORT,
            )
            self._pub_wasd = self._session.declare_publisher(
                "sensor/wasd",
                congestion_control=zenoh.CongestionControl.DROP,
                reliability=zenoh.Reliability.BEST_EFFORT,
            )
            self._pub_registry = self._session.declare_publisher(
                "object/registry",
                congestion_control=zenoh.CongestionControl.DROP,
                reliability=zenoh.Reliability.BEST_EFFORT,
            )
            self._pub_lidar = self._session.declare_publisher(
                "sensor/lidar",
                congestion_control=zenoh.CongestionControl.DROP,
                reliability=zenoh.Reliability.BEST_EFFORT,
            )
            self._pub_imu = self._session.declare_publisher(
                "sensor/imu",
                congestion_control=zenoh.CongestionControl.DROP,
                reliability=zenoh.Reliability.BEST_EFFORT,
            )
            self._pub_camera = self._session.declare_publisher(
                "sensor/camera/apriltag",
                congestion_control=zenoh.CongestionControl.DROP,
                reliability=zenoh.Reliability.BEST_EFFORT,
            )
            self._pub_goal = self._session.declare_publisher(
                "nav/goal",
                congestion_control=zenoh.CongestionControl.DROP,
                reliability=zenoh.Reliability.BEST_EFFORT,
            )

            # Drive + nav-stack subscriptions (T3D-06).
            self._sub_wheel = self._session.declare_subscriber(
                "sensor/wheel_speed", self._on_wheel_speed
            )
            self._sub_pose = self._session.declare_subscriber(
                "estimate/pose", self._on_estimate_pose
            )
            self._sub_path = self._session.declare_subscriber(
                "nav/path", self._on_nav_path
            )
            self._sub_detection = self._session.declare_subscriber(
                "detection/obstacles", self._on_detection
            )
            self._sub_goal = self._session.declare_subscriber(
                "nav/goal", self._on_nav_goal
            )

        # Let the base settle onto the wheels (and the arm onto the mount).
        for _ in range(240 * 2):
            p.stepSimulation()

        # Point the GUI camera at the base spawn (the map's first room), not the
        # world origin — otherwise the map is off-screen.
        if gui:
            bx, by, _ = self.truth_pose()
            p.resetDebugVisualizerCamera(
                cameraDistance=3.0, cameraYaw=45, cameraPitch=-30,
                cameraTargetPosition=[bx, by, 0.2],
            )
            # On machines without GPU-accelerated GL each stepSimulation render
            # costs ~17 ms — the whole 60 Hz budget — so rendering every frame
            # starves the physics.  Disable per-step rendering; the main loop
            # re-enables it for a frame ~30 Hz to refresh the view.
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    # ------------------------------------------------------------------
    # Wheel friction (driving is refined in T3D-06)
    # ------------------------------------------------------------------

    def _setup_wheel_friction(self):
        # PyBullet's default lateral friction (0.5) lets the drive wheels slip
        # when rotating the heavy base+arm, so the base skids sideways instead
        # of turning.  physics.wheel_friction_mu (~2.0) grips well; much higher
        # becomes numerically unstable.
        #
        # The high mu is applied ONLY to the driven wheels.  The free casters
        # get low friction: with the arm making the base front-heavy, the front
        # casters are heavily loaded, and the same 2.0 grip makes them scrub
        # instead of swivelling/rolling — they dig in and pitch the driven rear
        # wheels off the ground (rear-up "stoppie"), killing traction.
        mu = self._cfg.physics.wheel_friction_mu
        for j in range(p.getNumJoints(self._base)):
            if "free" in p.getJointInfo(self._base, j)[1].decode():
                p.changeDynamics(
                    self._base, j,
                    lateralFriction=0.2, spinningFriction=0.01, rollingFriction=0.005,
                )
            else:
                p.changeDynamics(
                    self._base, j,
                    lateralFriction=mu, spinningFriction=0.1, rollingFriction=0.02,
                )

    # ------------------------------------------------------------------
    # Accessors (used by tests and by the loop)
    # ------------------------------------------------------------------

    def _spawn_xy(self) -> tuple[float, float]:
        """Where to spawn the base: the first room's centre (as the 2D sim
        does), else the map centre, else the origin."""
        if self._map is not None:
            if self._map.rooms:
                xs = [p_[0] for p_ in self._map.rooms[0].polygon]
                ys = [p_[1] for p_ in self._map.rooms[0].polygon]
                return (sum(xs) / len(xs), sum(ys) / len(ys))
            size = self._map.metadata.size_m
            return (size[0] / 2.0, size[1] / 2.0)
        return (0.0, 0.0)

    def truth_pose(self) -> tuple[float, float, float]:
        """Base ground truth (x, y, theta) in world metres/radians."""
        pos, orn = p.getBasePositionAndOrientation(self._base)
        yaw = p.getEulerFromQuaternion(orn)[2]
        return (pos[0], pos[1], yaw)

    # ------------------------------------------------------------------
    # Drive (T3D-06)
    # ------------------------------------------------------------------

    def _clamp_wheel(self, rps: float) -> float:
        """Clamp a commanded wheel speed to ±1.2 × max linear speed (spec)."""
        ch = self._cfg.chassis
        max_rps = 1.2 * ch.linear_speed_mps / ch.wheel_radius_m
        return max(-max_rps, min(max_rps, rps))

    def _corrupt_unicycle(self, linear_mps: float, angular_rps: float) -> tuple[float, float]:
        """Corrupt a unicycle velocity with the 2D sim's slip model.

        Wheel scale (radius error) scales linear, track scale scales angular,
        cross-coupling turns linear motion into yaw, and per-step slip noise
        nudges both.  Same values/order as ``simulation/simulator.py``.
        """
        phy = self._physics
        actual_linear = linear_mps * self._wheel_scale
        actual_angular = angular_rps * self._wheel_scale / self._track_scale
        actual_angular += actual_linear * phy.cross_coupling_rad_per_m
        actual_linear += random.gauss(0.0, phy.slip_noise * abs(actual_linear))
        actual_angular += random.gauss(0.0, phy.slip_noise * abs(actual_angular))
        return actual_linear, actual_angular

    def _compute_drive_command(self) -> tuple[float, float]:
        """The (slip-corrupted) wheel speeds to command this frame.

        Prefers a fresh ``sensor/wheel_speed`` from the drive node (clamped);
        otherwise falls back to direct WASD teleop.  The unicycle command passes
        through the slip corruption, then a yaw-compensation term counteracts
        the fixed-wheel base's natural curve so W/S actually go forward/back.
        Pure logic — no physics — so it is testable.
        """
        ch = self._cfg.chassis
        if (
            self._wheel_speed is not None
            and time.monotonic() - self._wheel_speed_time < WHEEL_TIMEOUT_S
        ):
            left, right = self._wheel_speed
            left, right = self._clamp_wheel(left), self._clamp_wheel(right)
            linear, angular = wheel_to_unicycle(
                left, right,
                wheel_radius_m=ch.wheel_radius_m,
                wheel_track_m=ch.wheel_track_m,
            )
        else:
            target_lin = (self._keys["w"] - self._keys["s"]) * ch.linear_speed_mps
            # Left arrow = turn left (CCW, +yaw), right arrow = turn right.  With
            # the corrected wheel-axis and the _L/_R wheels on the
            # physically-correct sides, the kinematics convention (positive
            # angular = left) holds here.
            target_ang = (self._keys["a"] - self._keys["d"]) * ch.angular_speed_rps
            # Ramp the teleop velocity so the base doesn't lunge to full speed
            # instantly (which makes it pitch forward with the arm up).
            self._teleop_lin = _ramp(self._teleop_lin, target_lin, _TELEOP_ACCEL_MPS2 * _TELEOP_DT)
            self._teleop_ang = _ramp(self._teleop_ang, target_ang, _TELEOP_ANG_ACCEL_RPS2 * _TELEOP_DT)
            linear, angular = self._teleop_lin, self._teleop_ang

        eff_linear, eff_angular = self._corrupt_unicycle(linear, angular)
        eff_angular += self._physics.yaw_compensation_rad_per_m * eff_linear
        return unicycle_to_wheel(
            eff_linear, eff_angular,
            wheel_radius_m=ch.wheel_radius_m,
            wheel_track_m=ch.wheel_track_m,
        )

    def _apply_drive(self):
        """Command the drive wheels this frame from the drive command."""
        eff_left, eff_right = self._compute_drive_command()
        for name, vel in ((_DRIVE_L, eff_left), (_DRIVE_R, eff_right)):
            p.setJointMotorControl2(
                self._base, self._wheel_joints[name], p.VELOCITY_CONTROL,
                targetVelocity=vel, force=500,
            )

    # ------------------------------------------------------------------
    # Zenoh subscribers
    # ------------------------------------------------------------------

    def _on_wheel_speed(self, sample):
        try:
            data = decode("sensor/wheel_speed", sample)
            self._wheel_speed = (float(data["left_rps"]), float(data["right_rps"]))
            self._wheel_speed_time = time.monotonic()
        except SchemaError as exc:
            print(f"[sim3d] sensor/wheel_speed dropped: {exc}")

    def _on_estimate_pose(self, sample):
        try:
            data = decode("estimate/pose", sample)
            self._est_pose = (
                float(data["x_m"]), float(data["y_m"]), float(data["theta_rad"])
            )
        except SchemaError:
            pass

    def _on_nav_path(self, sample):
        try:
            self._nav_path = decode_path("nav/path", sample)
        except SchemaError:
            pass

    def _on_detection(self, sample):
        try:
            data = decode("detection/obstacles", sample)
            self._detected = data["points"]
        except SchemaError:
            pass

    def _on_nav_goal(self, sample):
        try:
            data = decode("nav/goal", sample)
            self._nav_goal = (float(data["x_m"]), float(data["y_m"]))
        except SchemaError:
            pass

    def _set_goal_from_click(self, x: float, y: float):
        """Turn a GUI left-click on the ground into a nav/goal publication."""
        goal = _ray_ground_intersection(*_mouse_ray(x, y))
        if goal is None:
            return
        self._nav_goal = goal
        if self._session is not None:
            self._pub_goal.put(encode("nav/goal", {"x_m": goal[0], "y_m": goal[1]}))
        print(f"[sim3d] nav/goal -> ({goal[0]:.2f}, {goal[1]:.2f})")

    def _draw_overlay(self):
        """Debug overlay of the nav stack's state: truth (green arrow),
        pose-estimator guess (red arrow), planned path (blue) and the goal
        (yellow X).  Toggle with the G key.

        On machines without GPU-accelerated PyBullet GL (Mesa software
        rendering) each debug line costs ~30 ms to add/remove, so re-drawing
        them every frame is what drops the GUI loop below 60 Hz.  Redraw is
        throttled to ~4 Hz and the nav path is downsampled.
        """
        if not self._gui:
            return
        now = time.monotonic()
        if now - self._last_gui_t < 0.25:
            return
        self._last_gui_t = now
        for uid in self._debug_items:
            p.removeUserDebugItem(uid)
        self._debug_items = []
        if not self._overlay:
            return
        if self._nav_goal is not None:
            gx, gy = self._nav_goal
            s = 0.12
            c = [0.95, 0.75, 0.05]
            self._debug_items.append(
                p.addUserDebugLine([gx - s, gy - s, 0.06], [gx + s, gy + s, 0.06], c)
            )
            self._debug_items.append(
                p.addUserDebugLine([gx - s, gy + s, 0.06], [gx + s, gy - s, 0.06], c)
            )
        if len(self._nav_path) >= 2:
            pts = self._nav_path
            if len(pts) > 16:
                step = (len(pts) - 1) / 15
                idx = [int(round(i * step)) for i in range(16)]
                idx.append(len(pts) - 1)
                pts = [pts[i] for i in sorted(set(idx))]
            tri = [(px, py, 0.04) for px, py in pts]
            for a, b in zip(tri[:-1], tri[1:]):
                self._debug_items.append(p.addUserDebugLine(a, b, [0.25, 0.45, 1.0]))
        x, y, th = self.truth_pose()
        self._debug_items.extend(_arrow(x, y, th, 0.35, [0.1, 0.8, 0.1]))
        if self._est_pose is not None:
            ex, ey, eth = self._est_pose
            self._debug_items.extend(_arrow(ex, ey, eth, 0.35, [0.9, 0.1, 0.1]))

    def _read_keyboard_gui(self) -> bool:
        """Update teleop keys from the PyBullet GUI.  Returns False on ESC.
        Drive with the arrow keys (WASD is reserved by the PyBullet viewer);
        left-click on the ground sets a nav/goal; G toggles the overlay."""
        for key, state in p.getKeyboardEvents().items():
            down = (state & p.KEY_IS_DOWN) != 0
            if key == p.B3G_UP_ARROW:
                self._keys["w"] = down
            elif key == p.B3G_LEFT_ARROW:
                self._keys["a"] = down
            elif key == p.B3G_DOWN_ARROW:
                self._keys["s"] = down
            elif key == p.B3G_RIGHT_ARROW:
                self._keys["d"] = down
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                self._camera_mode = key - ord("1")
            elif key == ord("g"):
                if down:
                    self._overlay = not self._overlay
            elif key == p.B3G_ESCAPE:
                return False
        # Mouse events are (event_type, x, y, button, state); a left-click press
        # is event_type=2, button=0 with the trigger+down flags set.
        for evt, x, y, btn, state in p.getMouseEvents():
            if (
                evt == 2
                and btn == 0
                and (state & p.KEY_WAS_TRIGGERED)
                and (state & p.KEY_IS_DOWN)
            ):
                self._set_goal_from_click(x, y)
        return True

    def _update_camera(self):
        """GUI camera: 0 = follow base (preserve orbit), 1 = top-down,
        2 = arm end-effector, 3 = free (user orbits).  Throttled to ~10 Hz;
        resetDebugVisualizerCamera every frame is expensive."""
        if time.monotonic() - self._last_gui_t < 0.1:
            return
        if self._camera_mode == 0:
            # Follow the base: keep the user's current distance/yaw/pitch but
            # re-centre on the base so it doesn't drive out of frame (which also
            # avoided the "see-through" clipping as the camera never ends up
            # inside a body).
            cam = p.getDebugVisualizerCamera()
            # getDebugVisualizerCamera: [8]=yaw, [9]=pitch, [10]=distance,
            # [11]=target.  Keep the user's current orbit but re-centre on the
            # base so it doesn't drive out of frame (which also avoided the
            # "see-through" clipping as the camera never ends up inside a body).
            yaw, pitch, distance = cam[8], cam[9], cam[10]
            pos = p.getBasePositionAndOrientation(self._base)[0]
            p.resetDebugVisualizerCamera(
                cameraDistance=distance, cameraYaw=yaw, cameraPitch=pitch,
                cameraTargetPosition=[pos[0], pos[1], 0.2],
            )
        elif self._camera_mode == 1:
            pos = p.getBasePositionAndOrientation(self._base)[0]
            p.resetDebugVisualizerCamera(cameraDistance=8.0, cameraYaw=90, cameraPitch=-89,
                                         cameraTargetPosition=[pos[0], pos[1], 0.0])
        elif self._camera_mode == 2:
            # follow the arm end-effector (gripper_link is the last joint)
            last = p.getNumJoints(self._arm) - 1
            ee = p.getLinkState(self._arm, last)[0]
            p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0, cameraPitch=-20,
                                         cameraTargetPosition=[ee[0], ee[1], ee[2]])
        # mode 3 = free: leave the camera alone (mouse orbit).

    # ------------------------------------------------------------------
    # Physics + publishing
    # ------------------------------------------------------------------

    def step(self):
        """Advance one frame (4 physics substeps) and apply the drive command."""
        self._apply_drive()
        for _ in range(SUBSTEPS):
            p.stepSimulation()

    def _publish(self):
        """Publish sim/truth/pose, sensor/wasd, (1 Hz) object/registry and the
        50 Hz sensor bridge (LIDAR / IMU / AprilTag)."""
        if self._session is None:
            return
        now = time.time()
        x, y, theta = self.truth_pose()
        self._pub_truth.put(
            encode("sim/truth/pose", {"t": now, "x_m": x, "y_m": y, "theta_rad": theta})
        )
        # The sim's own keyboard only exists in GUI mode; headless, the teleop
        # keys come from the viewer (zenoh/sim_viewer.py), which publishes
        # sensor/wasd here so there is a single publisher.
        if self._gui:
            self._pub_wasd.put(encode("sensor/wasd", self._keys))

        if now - getattr(self, "_last_registry_t", 0.0) >= 1.0 / REGISTRY_HZ:
            self._last_registry_t = now
            update_object_poses(self._objects)
            self._pub_registry.put(object_registry_message(self._objects, t=now))

        if now - self._last_sensor_t >= 1.0 / SENSOR_HZ:
            self._last_sensor_t = now
            self._publish_sensors(now, (x, y, theta))

    def _publish_sensors(self, t: float, bot_pose):
        """Publish sensor/lidar, sensor/imu and sensor/camera/apriltag (T3D-05)."""
        mono = time.monotonic()
        dt = mono - self._last_sensor_mono
        self._last_sensor_mono = mono

        self._pub_lidar.put(
            encode(
                "sensor/lidar",
                {"t": t, "rays": scan_lidar(self._base, self._arm, self._cfg)},
            )
        )
        imu = read_imu(self._base, self._cfg, self._gyro_bias_rps, self._prev_lin_vel, dt)
        self._prev_lin_vel = tuple(p.getBaseVelocity(self._base)[1])
        imu_msg = encode(
            "sensor/imu",
            {"t": t, "pitch_rad": 0.0, "roll_rad": 0.0, **imu},
        )
        self._pub_imu.put(imu_msg)

        if self._map is not None:
            # Camera at the configured mount, rotated with the base.
            pos, orn = p.getBasePositionAndOrientation(self._base)
            yaw = p.getEulerFromQuaternion(orn)[2]
            mx, my, mz = self._cfg.sensors.camera.mount_xyz_m
            cam_pos = (
                pos[0] + mx * math.cos(yaw) - my * math.sin(yaw),
                pos[1] + mx * math.sin(yaw) + my * math.cos(yaw),
                pos[2] + mz,
            )
            robot = {self._base, self._arm}

            def is_occluded(sx: float, sy: float) -> bool:
                """True if a wall/object blocks the camera's line of sight to
                the tag's printed face (robot self-hits ignored).  The probe
                point lies on the tag's own wall face, so a hit essentially at
                the endpoint is the tag's wall and does not count."""
                to = (sx, sy, cam_pos[2])
                for hit in p.rayTest(cam_pos, to):
                    if hit[0] < 0 or hit[0] in robot:
                        continue
                    return hit[2] < 0.98
                return False

            dets = detect_apriltags(
                self._map, bot_pose, self._cfg, is_occluded=is_occluded
            )
            if dets:
                self._pub_camera.put(
                    encode("sensor/camera/apriltag", {"t": t, "detections": dets})
                )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        running = True
        next_tick = time.monotonic()
        period = 1.0 / LOOP_HZ
        frame = 0
        try:
            while running:
                if self._gui:
                    running = self._read_keyboard_gui()
                    self._update_camera()
                    self._draw_overlay()
                    # Physics steps every frame; the ~17 ms GL render only runs
                    # every other frame (~30 Hz refresh) so it can't starve the
                    # physics loop.
                    if frame % 2 == 0:
                        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
                        self.step()
                        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
                    else:
                        self.step()
                else:
                    self.step()
                self._publish()
                frame += 1
                next_tick += period
                sleep_until(next_tick)
        finally:
            if self._session is not None:
                self._session.close()
            p.disconnect()


def main():
    parser = argparse.ArgumentParser(description="3D mobile-manipulation simulator")
    parser.add_argument("--gui", action="store_true", help="open the PyBullet viewer")
    parser.add_argument("--map", default=None, help="map JSON (default arena if omitted)")
    parser.add_argument("--objects", default=None, help="world registry JSON")
    parser.add_argument("--seed", type=int, default=0, help="deterministic seed")
    args = parser.parse_args()

    sim = Simulator3D(
        gui=args.gui,
        map_path=args.map,
        world_path=args.objects,
        seed=args.seed,
    )
    sim.run()


if __name__ == "__main__":
    main()
