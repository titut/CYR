"""2D simulator for semantic navigation maps.

Run from the project root with:
    python -m simulation.simulator [path/to/map.json]

Or:
    python simulation/simulator.py [path/to/map.json]

Defaults to `test_map.json` if no map is provided.

The simulator publishes sensor data and user navigation goals over Zenoh.
Navigation (planning, LLM queries, path-following) is handled by external nodes.

Controls:
    W / S - Move forward / backward in the direction the bot is facing.
    A / D - Turn left / right.
    1 / 2 / 3 - Switch tabs: Map | LIDAR | Guess.
    R     - Send safety/reset to clear a latched drive e-stop.
    + / - - Zoom in / out (Map and Guess tabs).
    ESC   - Quit.

Zenoh topics:
    Published:
        sensor/lidar         — {"t": float, "rays": [{angle_rad, distance_m}]}
        sensor/imu           — {"t": float, yaw_rad, pitch_rad, roll_rad, ...}
        sensor/camera/apriltag — {"t": float, "detections": [{id, range_m, ...}]}
        sensor/wasd          — {"w","a","s","d"} booleans, raw teleop key state
        nav/goal             — {x_m, y_m} user-clicked map target (meters)
        nav/command          — string, LLM text command from chat bar
        safety/reset         — any payload: clears a latched drive e-stop (R key)
    Subscribed:
        estimate/pose        — external pose estimate {x_m, y_m, theta_rad, t}
        nav/path             — planned waypoints [[x, y], ...] (meters)
        sensor/wheel_speed   — measured wheel speeds {left_rps, right_rps, t}
        detection/obstacles  — LIDAR-detected obstacle points (Guess tab)
"""

from __future__ import annotations

import logging
import math
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pygame
import zenoh

# Allow running this file directly or as a module.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.hal import load_camera_driver, load_imu_driver, load_lidar_driver
from core.map_format import MapData, Obstacle, Wall, new_empty_map
from core.messages import SchemaError, decode, decode_path, encode, encode_text
from core.robot_config import get_robot_config
from simulation.kinematics import wheel_to_unicycle
from simulation.raycast import RayHit

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
TAB_BAR_HEIGHT = 44
CHAT_BAR_HEIGHT = 36
FPS = 60

COLORS = {
    "bg": (245, 245, 245),
    "grid": (230, 230, 230),
    "wall": (30, 30, 30),
    "room_fill": (227, 242, 253),
    "room_outline": (33, 150, 243),
    "bot_true": (33, 150, 243),
    "bot_guess": (255, 87, 34),
    "particle": (255, 152, 0),
    "lidar_ray": (0, 200, 83, 120),
    "lidar_point": (0, 150, 60),
    "tab_active": (33, 150, 243),
    "tab_inactive": (220, 220, 220),
    "tab_text": (255, 255, 255),
    "ui_text": (40, 40, 40),
    "nav_target": (255, 50, 50),
    "nav_path": (100, 100, 255),
    "apriltag": (0, 180, 0),
    "apriltag_detected": (255, 215, 0),
    "obstacle": (255, 100, 0),
    "obstacle_outline": (180, 60, 0),
    "detected_obstacle": (230, 0, 100),
}

# Wheel slip simulation.  Slip is a wheel-ground interaction, so it is applied
# here in the physics step (not in the drive node, which only models the motor
# and encoder).  It corrupts the ground-truth motion relative to what the wheel
# encoders report, so odometry (integrated encoder speeds) drifts from truth.
# Wheel-ground errors: independent wheel-radius and track calibration errors,
# plus a small cross-coupling (differential drives couple linear motion into
# rotation) and per-step slip noise.  These corrupt ground truth relative to
# what the wheel encoders report, so odometry drifts from truth.  Values come
# from the robot config (robot_config.PhysicsConfig).

# IMU realism: a small per-run gyro bias plus per-sample Gaussian noise on the
# reported angular rate, and Gaussian noise on the absolute yaw (a
# magnetometer-like reference).  The gyro bias drifts heading slowly; the yaw
# has noise but no drift.  Values come from the robot config (ImuConfig).


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def polygon_center(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not polygon:
        return (0.0, 0.0)
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def rotate_point(
    point: Tuple[float, float],
    angle: float,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[float, float]:
    """Rotate a point around an origin by angle (radians)."""
    x, y = point[0] - origin[0], point[1] - origin[1]
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        origin[0] + x * cos_a - y * sin_a,
        origin[1] + x * sin_a + y * cos_a,
    )


def point_to_segment_distance(x: float, y: float, wall: Wall) -> float:
    """Distance from point (x, y) to the wall segment."""
    wx = wall.x2 - wall.x1
    wy = wall.y2 - wall.y1
    length_sq = wx * wx + wy * wy
    if length_sq == 0:
        return math.hypot(x - wall.x1, y - wall.y1)
    t = max(0.0, min(1.0, ((x - wall.x1) * wx + (y - wall.y1) * wy) / length_sq))
    proj_x = wall.x1 + t * wx
    proj_y = wall.y1 + t * wy
    return math.hypot(x - proj_x, y - proj_y)


def bot_collides(
    bot_pos: Tuple[float, float],
    radius: float,
    walls: List[Wall],
    obstacles: Optional[List[Obstacle]] = None,
) -> bool:
    """Check whether a circular bot collides with any wall or obstacle.

    All coordinates and radii are in meters.
    """
    for wall in walls:
        if point_to_segment_distance(bot_pos[0], bot_pos[1], wall) < radius:
            return True
    if obstacles:
        for obstacle in obstacles:
            d = math.hypot(bot_pos[0] - obstacle.x, bot_pos[1] - obstacle.y)
            if d < radius + obstacle.radius_m:
                return True
    return False


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class Simulator:
    def __init__(self, map_path: Path):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Semantic Navigation Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Helvetica", 16)
        self.big_font = pygame.font.SysFont("Helvetica", 20, bold=True)

        raw_map = self._load_map(map_path)
        # Remember the pixels-per-meter ratio so we can render the meter world
        # to screen.  The map is stored in meters; scale_m_per_px is only the
        # editor's canvas calibration.
        self.px_per_m = 1.0 / raw_map.metadata.scale_m_per_px
        self.map_data = raw_map

        # Robot description (T-019): every physical/sensor parameter comes from
        # the robot config (robot.yaml), and the sensor models are behind the
        # driver interfaces (hal) selected by the config's hardware section.
        self._cfg = get_robot_config()
        self._camera = self._cfg.sensors.camera  # also used for the GUI overlays
        self._physics = self._cfg.physics
        self._lidar_driver = load_lidar_driver(self._cfg)
        self._imu_driver = load_imu_driver(self._cfg)
        self._camera_driver = load_camera_driver(self._cfg)

        self.bot_radius_m = self._cfg.chassis.radius_m
        # Physics collision radius: the square's circumradius, so the collision
        # check conservatively bounds the body in any orientation.  (A half-side
        # 0.375 m square's corners reach 0.53 m when rotated 45°.)
        self._collision_radius_m = self._cfg.chassis.collision_radius_m

        # Bot state in meters and radians.
        start_center = (
            polygon_center(self.map_data.rooms[0].polygon)
            if self.map_data.rooms
            else (20.0, 15.0)
        )
        self.bot_x, self.bot_y = start_center
        self.bot_theta = 0.0
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        # External pose estimate (received from estimate/pose).
        self._est_lock = threading.Lock()
        self.guess_x, self.guess_y = self.bot_x, self.bot_y
        self.guess_theta = self.bot_theta

        # Camera for Map and Guess tabs.
        self.camera_x, self.camera_y = self.bot_x, self.bot_y
        self.zoom = 1.0

        # Tab state.
        self.active_tab = 0
        self.tabs = ["Map", "LIDAR", "Guess"]

        # Cached LIDAR scan.
        self.lidar_hits: List[RayHit] = []

        # Wheel slip simulation: independent wheel-radius and track calibration
        # errors per run, so odometry drifts from truth in both distance and
        # heading.
        self._wheel_scale = random.uniform(
            self._physics.wheel_scale_min, self._physics.wheel_scale_max
        )
        self._track_scale = random.uniform(
            self._physics.track_scale_min, self._physics.track_scale_max
        )

        # Measured wheel speeds from the drive node (rad/s).
        self._wheel_lock = threading.Lock()
        self._left_rps = 0.0
        self._right_rps = 0.0

        # Navigation state (path received from navigator over Zenoh, for display).
        self.nav_target: Optional[Tuple[float, float]] = None
        self.nav_path: List[Tuple[float, float]] = []

        # Obstacle points detected by the navigator (world frame, for display).
        self._detected_obstacles: List[Tuple[float, float]] = []
        self._detection_lock = threading.Lock()

        # Chat bar state.
        self._chat_active = False
        self._chat_text = ""
        self._chat_status = ""

        # Zenoh session, publishers, subscribers.
        self._zenoh_session = zenoh.open(zenoh.Config())

        # Publishers: sensor data.
        self._pub_lidar = self._zenoh_session.declare_publisher(
            "sensor/lidar",
            congestion_control=zenoh.CongestionControl.DROP,
            reliability=zenoh.Reliability.BEST_EFFORT,
        )
        self._pub_wasd = self._zenoh_session.declare_publisher(
            "sensor/wasd",
            congestion_control=zenoh.CongestionControl.DROP,
            reliability=zenoh.Reliability.BEST_EFFORT,
        )
        self._pub_imu = self._zenoh_session.declare_publisher(
            "sensor/imu",
            congestion_control=zenoh.CongestionControl.DROP,
            reliability=zenoh.Reliability.BEST_EFFORT,
        )

        # Publishers: navigation requests.
        self._pub_goal = self._zenoh_session.declare_publisher(
            "nav/goal",
            congestion_control=zenoh.CongestionControl.DROP,
            reliability=zenoh.Reliability.BEST_EFFORT,
        )
        self._pub_command = self._zenoh_session.declare_publisher(
            "nav/command",
            congestion_control=zenoh.CongestionControl.DROP,
            reliability=zenoh.Reliability.BEST_EFFORT,
        )

        # Publisher: raw camera-visible AprilTag data.
        self._pub_camera_apriltag = self._zenoh_session.declare_publisher(
            "sensor/camera/apriltag",
            congestion_control=zenoh.CongestionControl.DROP,
            reliability=zenoh.Reliability.BEST_EFFORT,
        )

        # Publisher: drive e-stop reset (pressed R, see T-016).
        self._pub_safety_reset = self._zenoh_session.declare_publisher(
            "safety/reset",
            congestion_control=zenoh.CongestionControl.DROP,
            reliability=zenoh.Reliability.BEST_EFFORT,
        )

        # Subscribers.
        self._sub_pose = self._zenoh_session.declare_subscriber(
            "estimate/pose", self._on_estimate_pose
        )
        self._sub_path = self._zenoh_session.declare_subscriber(
            "nav/path", self._on_nav_path
        )
        self._sub_wheel = self._zenoh_session.declare_subscriber(
            "sensor/wheel_speed", self._on_wheel_speed
        )
        self._sub_detection = self._zenoh_session.declare_subscriber(
            "detection/obstacles", self._on_detection
        )

    @staticmethod
    def _load_map(path: Path) -> MapData:
        if not path.exists():
            print(f"Map file not found: {path}. Creating an empty map.")
            return new_empty_map()
        return MapData.from_json(path)

    # -----------------------------------------------------------------------
    # Zenoh callbacks
    # -----------------------------------------------------------------------

    def _on_estimate_pose(self, sample):
        """Receive estimated pose from pose_estimator."""
        try:
            data = decode("estimate/pose", sample)
            with self._est_lock:
                self.guess_x = float(data["x_m"])
                self.guess_y = float(data["y_m"])
                self.guess_theta = float(data["theta_rad"])
        except SchemaError as exc:
            logging.warning("estimate/pose dropped: %s", exc)

    def _on_nav_path(self, sample):
        """Receive planned path from navigator (for display only)."""
        try:
            waypoints = decode_path("nav/path", sample)
            if not waypoints:
                logging.warning("Navigator returned empty path.")
                self.nav_path = []
                return
            self.nav_path = [tuple(p) for p in waypoints]
            logging.info("Received path with %d waypoints.", len(self.nav_path))
        except SchemaError as exc:
            logging.warning("nav/path dropped: %s", exc)

    def _on_detection(self, sample):
        """Receive obstacle points detected by the navigator (for display)."""
        try:
            data = decode("detection/obstacles", sample)
            points = [tuple(p) for p in data["points"]]
            with self._detection_lock:
                self._detected_obstacles = points
        except SchemaError as exc:
            logging.warning("detection/obstacles dropped: %s", exc)

    def _on_wheel_speed(self, sample):
        """Receive measured wheel speeds from the drive node."""
        try:
            data = decode("sensor/wheel_speed", sample)
            with self._wheel_lock:
                self._left_rps = float(data["left_rps"])
                self._right_rps = float(data["right_rps"])
        except SchemaError as exc:
            logging.warning("sensor/wheel_speed dropped: %s", exc)

    # -----------------------------------------------------------------------
    # Publishing
    # -----------------------------------------------------------------------

    def _publish_zenoh(self):
        """Publish sensor data over Zenoh."""
        now = time.time()

        # LIDAR.
        lidar_msg = encode(
            "sensor/lidar",
            {
                "t": now,
                "rays": [
                    {"angle_rad": hit.angle, "distance_m": hit.distance}
                    for hit in self.lidar_hits
                ],
            },
        )
        self._pub_lidar.put(lidar_msg)

        # IMU (through the configured IMU driver, which adds bias + noise).
        imu = self._imu_driver.read(self.bot_theta, self.angular_velocity)
        imu_msg = encode(
            "sensor/imu",
            {
                "t": now,
                "yaw_rad": imu["yaw_rad"],
                "pitch_rad": 0.0,
                "roll_rad": 0.0,
                "angular_velocity_rps": imu["angular_velocity_rps"],
                "linear_acceleration_mps2": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
        )
        self._pub_imu.put(imu_msg)

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self):
        running = True
        try:
            while running:
                dt = self.clock.tick(FPS) / 1000.0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    else:
                        self._handle_event(event)
                self._update(dt)
                self._render()
        finally:
            self._zenoh_session.close()
        pygame.quit()

    # -----------------------------------------------------------------------
    # Events
    # -----------------------------------------------------------------------

    def _handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            if self._chat_active:
                self._handle_chat_key(event)
            elif event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
            elif event.key == pygame.K_RETURN:
                self._chat_active = True
                self._chat_text = ""
                self._chat_status = ""
            elif event.key == pygame.K_1:
                self.active_tab = 0
            elif event.key == pygame.K_2:
                self.active_tab = 1
            elif event.key == pygame.K_3:
                self.active_tab = 2
            elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                self.zoom = min(5.0, self.zoom * 1.1)
            elif event.key == pygame.K_MINUS:
                self.zoom = max(0.2, self.zoom / 1.1)
            elif event.key == pygame.K_r:
                # Clear a latched drive e-stop (T-016).
                self._pub_safety_reset.put(encode_text("safety/reset", "reset"))
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and event.pos[1] <= TAB_BAR_HEIGHT:
                tab_width = WINDOW_WIDTH // len(self.tabs)
                self.active_tab = min(event.pos[0] // tab_width, len(self.tabs) - 1)
            elif event.button == 1 and self.active_tab == 0:
                content_rect = pygame.Rect(
                    0,
                    TAB_BAR_HEIGHT,
                    WINDOW_WIDTH,
                    WINDOW_HEIGHT - TAB_BAR_HEIGHT - CHAT_BAR_HEIGHT,
                )
                world = self._screen_to_world(
                    event.pos,
                    content_rect,
                    (self.camera_x, self.camera_y),
                    self.zoom,
                )
                # Publish nav/goal so the navigator plans a path.
                self._pub_goal.put(
                    encode("nav/goal", {"x_m": world[0], "y_m": world[1]})
                )
                self.nav_target = world
            elif event.button == 3 and self.active_tab == 0:
                self.nav_target = None
                self.nav_path = []

    # -----------------------------------------------------------------------
    # Update
    # -----------------------------------------------------------------------

    def _update(self, dt: float):
        keys = pygame.key.get_pressed()

        # Publish raw teleop key state for the controller node.
        wasd = {
            "w": bool(keys[pygame.K_w]),
            "a": bool(keys[pygame.K_a]),
            "s": bool(keys[pygame.K_s]),
            "d": bool(keys[pygame.K_d]),
        }
        self._pub_wasd.put(encode("sensor/wasd", wasd))

        # Drive the bot from the measured wheel speeds published by the drive
        # node.  The drive node reports encoder values (motor + encoder error);
        # wheel-ground slip is applied here, in the physics step.
        with self._wheel_lock:
            left_rps, right_rps = self._left_rps, self._right_rps
        linear, angular = wheel_to_unicycle(
            left_rps,
            right_rps,
            wheel_radius_m=self._cfg.chassis.wheel_radius_m,
            wheel_track_m=self._cfg.chassis.wheel_track_m,
        )

        actual_linear = linear * self._wheel_scale
        actual_angular = angular * self._wheel_scale / self._track_scale
        actual_angular += actual_linear * self._physics.cross_coupling_rad_per_m
        actual_linear += random.gauss(0.0, self._physics.slip_noise * abs(actual_linear))
        actual_angular += random.gauss(0.0, self._physics.slip_noise * abs(actual_angular))

        self.linear_velocity = actual_linear
        self.angular_velocity = actual_angular

        # Advance moving obstacles (LIDAR, collision and rendering all read
        # obstacle.x/y every frame, so moving them here is all that's needed).
        self._update_obstacles(dt)

        new_theta = self.bot_theta + actual_angular * dt
        new_x = self.bot_x + actual_linear * math.cos(self.bot_theta) * dt
        new_y = self.bot_y + actual_linear * math.sin(self.bot_theta) * dt

        obstacles = self.map_data.obstacles
        if not bot_collides(
            (new_x, self.bot_y), self._collision_radius_m, self.map_data.walls, obstacles
        ):
            self.bot_x = new_x
        if not bot_collides(
            (self.bot_x, new_y), self._collision_radius_m, self.map_data.walls, obstacles
        ):
            self.bot_y = new_y
        self.bot_theta = new_theta

        self.camera_x += (self.bot_x - self.camera_x) * 0.1
        self.camera_y += (self.bot_y - self.camera_y) * 0.1

        self.lidar_hits = self._lidar_driver.scan(
            origin=(self.bot_x, self.bot_y),
            forward_direction=self.bot_theta,
            walls=self.map_data.walls,
            obstacles=self.map_data.obstacles,
        )
        self._publish_zenoh()
        self._detect_apriltags()

    def _update_obstacles(self, dt: float):
        """Advance moving obstacles and bounce them off walls.

        Obstacles with zero velocity are left untouched (static).  A moving
        obstacle's velocity is reflected about a wall's normal when it would
        collide, and its centre is pushed back out of the wall so it never
        tunnels through.
        """
        for o in self.map_data.obstacles:
            if o.vx_mps == 0.0 and o.vy_mps == 0.0:
                continue
            o.x += o.vx_mps * dt
            o.y += o.vy_mps * dt
            for wall in self.map_data.walls:
                self._bounce_obstacle(o, wall)

    @staticmethod
    def _bounce_obstacle(o: Obstacle, wall: Wall):
        """Reflect ``o``'s velocity off ``wall`` if the two intersect."""
        d = point_to_segment_distance(o.x, o.y, wall)
        if d >= o.radius_m:
            return

        wx = wall.x2 - wall.x1
        wy = wall.y2 - wall.y1
        length_sq = wx * wx + wy * wy
        if length_sq == 0:
            cx, cy = wall.x1, wall.y1
        else:
            t = max(
                0.0,
                min(1.0, ((o.x - wall.x1) * wx + (o.y - wall.y1) * wy) / length_sq),
            )
            cx = wall.x1 + t * wx
            cy = wall.y1 + t * wy

        nx = o.x - cx
        ny = o.y - cy
        n_len = math.hypot(nx, ny)
        if n_len < 1e-9:
            # Centre is exactly on the segment; use the segment normal.
            nx = -wy
            ny = wx
            n_len = math.hypot(nx, ny)
        nx /= n_len
        ny /= n_len

        # Reflect the velocity component along the contact normal.
        dot = o.vx_mps * nx + o.vy_mps * ny
        if dot < 0.0:
            o.vx_mps -= 2.0 * dot * nx
            o.vy_mps -= 2.0 * dot * ny

        # Push the centre out of the wall so it doesn't tunnel.
        o.x = cx + nx * o.radius_m
        o.y = cy + ny * o.radius_m

    # -----------------------------------------------------------------------
    # AprilTag detection
    # -----------------------------------------------------------------------

    def _detect_apriltags(self):
        """Simulate a forward-facing camera detecting AprilTags.

        The camera model (FOV cone, range, noise) lives in the configured
        camera driver; detections are published with noisy range/bearing so the
        pose estimator can anchor its filter.
        """
        detections = self._camera_driver.detect(
            self.bot_x, self.bot_y, self.bot_theta, self.map_data.apriltags
        )
        if detections:
            self._pub_camera_apriltag.put(
                encode(
                    "sensor/camera/apriltag",
                    {"t": time.time(), "detections": detections},
                )
            )

    # -----------------------------------------------------------------------
    # Chat bar
    # -----------------------------------------------------------------------

    def _handle_chat_key(self, event: pygame.event.Event):
        if event.key == pygame.K_ESCAPE:
            self._chat_active = False
            self._chat_text = ""
            self._chat_status = ""
        elif event.key == pygame.K_RETURN:
            self._submit_chat()
        elif event.key == pygame.K_BACKSPACE:
            self._chat_text = self._chat_text[:-1]
        else:
            if event.unicode and event.unicode.isprintable():
                self._chat_text += event.unicode

    def _submit_chat(self):
        """Publish the LLM command over Zenoh so the navigator can handle it."""
        text = self._chat_text.strip()
        if not text:
            self._chat_active = False
            return

        self._chat_status = "thinking"
        self._chat_text = ""
        self._chat_active = False

        logging.info("Publishing nav/command: %r", text)
        self._pub_command.put(encode_text("nav/command", text))

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def _render(self):
        self.screen.fill(COLORS["bg"])
        content_height = WINDOW_HEIGHT - TAB_BAR_HEIGHT - CHAT_BAR_HEIGHT
        content_rect = pygame.Rect(0, TAB_BAR_HEIGHT, WINDOW_WIDTH, content_height)
        pygame.draw.rect(self.screen, (255, 255, 255), content_rect)

        if self.active_tab == 0:
            self._render_map_tab(content_rect)
        elif self.active_tab == 1:
            self._render_lidar_tab(content_rect)
        elif self.active_tab == 2:
            self._render_guess_tab(content_rect)

        self._render_tab_bar()
        self._render_chat_bar()
        self._render_overlay()
        pygame.display.flip()

    def _render_tab_bar(self):
        tab_width = WINDOW_WIDTH // len(self.tabs)
        for i, label in enumerate(self.tabs):
            rect = pygame.Rect(i * tab_width, 0, tab_width, TAB_BAR_HEIGHT)
            color = (
                COLORS["tab_active"] if i == self.active_tab else COLORS["tab_inactive"]
            )
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (180, 180, 180), rect, 1)
            surf = self.big_font.render(label, True, COLORS["tab_text"])
            self.screen.blit(surf, surf.get_rect(center=rect.center))

    def _render_overlay(self):
        lines = [
            f"Tab: {self.tabs[self.active_tab]}  (1/2/3 to switch)",
            f"Pose: x={self.bot_x:.1f}, y={self.bot_y:.1f}, θ={math.degrees(self.bot_theta):.1f}°",
            f"LIDAR rays: {len(self.lidar_hits)}",
            "WASD to drive | R reset e-stop | +/- zoom | ESC quit",
        ]
        y = TAB_BAR_HEIGHT + 10
        for line in lines:
            surf = self.font.render(line, True, COLORS["ui_text"])
            self.screen.blit(surf, (10, y))
            y += 22

    def _render_chat_bar(self):
        chat_y = WINDOW_HEIGHT - CHAT_BAR_HEIGHT
        bar_rect = pygame.Rect(0, chat_y, WINDOW_WIDTH, CHAT_BAR_HEIGHT)

        if self._chat_status == "thinking":
            bg, prompt = (255, 240, 200), "Thinking…"
        elif self._chat_status == "error":
            bg, prompt = (255, 220, 220), "Error — try again"
        elif self._chat_active:
            bg, prompt = (230, 240, 255), ""
        else:
            bg, prompt = (
                220,
                220,
                220,
            ), "Press ENTER to type a command (e.g. 'go to shrine')"

        pygame.draw.rect(self.screen, bg, bar_rect)
        pygame.draw.rect(self.screen, (180, 180, 180), bar_rect, 1)

        display = self._chat_text + "|" if self._chat_active else prompt
        text_surf = self.font.render(display, True, (40, 40, 40))
        self.screen.blit(text_surf, (10, chat_y + CHAT_BAR_HEIGHT // 2 - 8))

    # -----------------------------------------------------------------------
    # Coordinate transforms
    # -----------------------------------------------------------------------

    def _world_to_screen(self, point, content_rect, camera, zoom):
        cx, cy = camera
        return (
            content_rect.centerx + (point[0] - cx) * self.px_per_m * zoom,
            content_rect.centery + (point[1] - cy) * self.px_per_m * zoom,
        )

    def _screen_to_world(self, screen_pos, content_rect, camera, zoom):
        sx, sy = screen_pos
        cx, cy = camera
        return (
            (sx - content_rect.centerx) / (self.px_per_m * zoom) + cx,
            (sy - content_rect.centery) / (self.px_per_m * zoom) + cy,
        )

    def _world_to_screen_rotated(
        self, point, content_rect, origin, forward_theta, zoom
    ):
        dx, dy = point[0] - origin[0], point[1] - origin[1]
        rotation = -(forward_theta + math.pi / 2.0)
        cos_a, sin_a = math.cos(rotation), math.sin(rotation)
        return (
            content_rect.centerx
            + (dx * cos_a - dy * sin_a) * self.px_per_m * zoom,
            content_rect.centery
            + (dx * sin_a + dy * cos_a) * self.px_per_m * zoom,
        )

    # -----------------------------------------------------------------------
    # Map tab
    # -----------------------------------------------------------------------

    def _render_map_tab(self, content_rect):
        cam = (self.camera_x, self.camera_y)
        self._draw_grid(content_rect, cam, self.zoom)
        self._draw_rooms(content_rect, cam, self.zoom)
        self._draw_walls(content_rect, cam, self.zoom)
        self._draw_obstacles(content_rect, cam, self.zoom)
        self._draw_apriltags(content_rect, cam, self.zoom)
        self._draw_nav_target(content_rect, cam, self.zoom)
        self._draw_nav_path(content_rect, cam, self.zoom)
        self._draw_apriltag_camera_fov(content_rect, cam, self.zoom)
        self._draw_lidar_rays(content_rect, cam, self.zoom, self.bot_theta)
        self._draw_bot(
            content_rect,
            cam,
            self.zoom,
            (self.bot_x, self.bot_y, self.bot_theta),
            COLORS["bot_true"],
        )

    def _draw_obstacles(self, content_rect, camera, zoom):
        """Draw all obstacles as filled circles on the map."""
        for obstacle in self.map_data.obstacles:
            sc = self._world_to_screen(
                (obstacle.x, obstacle.y), content_rect, camera, zoom
            )
            radius = obstacle.radius_m * self.px_per_m * zoom
            pygame.draw.circle(self.screen, COLORS["obstacle"], sc, max(1, int(radius)))
            pygame.draw.circle(
                self.screen, COLORS["obstacle_outline"], sc, max(1, int(radius)), 2
            )

    def _draw_detected_obstacles(self, content_rect, camera, zoom):
        """Draw the navigator's LIDAR-detected obstacle points."""
        with self._detection_lock:
            points = list(self._detected_obstacles)
        radius = max(2, int(0.1 * self.px_per_m * zoom))
        for px, py in points:
            sc = self._world_to_screen((px, py), content_rect, camera, zoom)
            pygame.draw.circle(self.screen, COLORS["detected_obstacle"], sc, radius)
            pygame.draw.circle(
                self.screen, (90, 0, 40), sc, radius, 1
            )

    def _draw_apriltags(self, content_rect, camera, zoom):
        """Draw all AprilTags as coloured squares on the map.

        Tags within the camera's detection cone are highlighted yellow;
        out-of-range tags are drawn in a muted green.
        """
        max_range_m = self._camera.max_range_m
        half_fov = self._camera.fov_rad / 2.0

        for tag in self.map_data.apriltags:
            sc = self._world_to_screen((tag.x, tag.y), content_rect, camera, zoom)
            half = max(3.0, (tag.size_m * self.px_per_m) * zoom / 2.0)

            # Check if this tag is currently detected.
            dx = tag.x - self.bot_x
            dy = tag.y - self.bot_y
            dist = math.hypot(dx, dy)
            within_range = dist <= max_range_m
            angle_to_tag = math.atan2(dy, dx)
            bearing = math.atan2(
                math.sin(angle_to_tag - self.bot_theta),
                math.cos(angle_to_tag - self.bot_theta),
            )
            within_fov = abs(bearing) <= half_fov
            detected = within_range and within_fov

            color = COLORS["apriltag_detected"] if detected else COLORS["apriltag"]
            rect = pygame.Rect(sc[0] - half, sc[1] - half, half * 2, half * 2)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

            # Direction line.
            tip = (
                sc[0] + half * 1.6 * math.cos(tag.yaw_rad),
                sc[1] + half * 1.6 * math.sin(tag.yaw_rad),
            )
            pygame.draw.line(self.screen, (0, 0, 0), sc, tip, 2)

            # Label.
            label = self.font.render(f"T{tag.id}", True, (0, 0, 0))
            self.screen.blit(
                label, (int(sc[0] - label.get_width() / 2), int(sc[1] - half - 16))
            )

    def _draw_apriltag_camera_fov(self, content_rect, camera, zoom):
        """Draw the camera's field of view cone on the map."""
        bot_screen = self._world_to_screen(
            (self.bot_x, self.bot_y), content_rect, camera, zoom
        )
        max_range_m = self._camera.max_range_m
        half_fov = self._camera.fov_rad / 2.0

        left_angle = self.bot_theta - half_fov
        right_angle = self.bot_theta + half_fov

        left_pt = (
            self.bot_x + max_range_m * math.cos(left_angle),
            self.bot_y + max_range_m * math.sin(left_angle),
        )
        right_pt = (
            self.bot_x + max_range_m * math.cos(right_angle),
            self.bot_y + max_range_m * math.sin(right_angle),
        )

        left_s = self._world_to_screen(left_pt, content_rect, camera, zoom)
        right_s = self._world_to_screen(right_pt, content_rect, camera, zoom)

        fov_color = (100, 100, 100, 200)
        # Use a small surface with alpha for the cone fill.
        try:
            surf = pygame.Surface(
                (content_rect.width, content_rect.height), pygame.SRCALPHA
            )
            pygame.draw.polygon(
                surf,
                (*fov_color[:3], 40),
                [bot_screen, left_s, right_s],
            )
            self.screen.blit(surf, (0, 0))
        except (pygame.error, ValueError):
            pass

        pygame.draw.line(self.screen, fov_color[:3], bot_screen, left_s, 1)
        pygame.draw.line(self.screen, fov_color[:3], bot_screen, right_s, 1)

    def _draw_nav_target(self, content_rect, camera, zoom):
        if self.nav_target is None:
            return
        cx, cy = self._world_to_screen(self.nav_target, content_rect, camera, zoom)
        s = 10
        pygame.draw.line(
            self.screen, COLORS["nav_target"], (cx - s, cy - s), (cx + s, cy + s), 2
        )
        pygame.draw.line(
            self.screen, COLORS["nav_target"], (cx + s, cy - s), (cx - s, cy + s), 2
        )

    def _draw_nav_path(self, content_rect, camera, zoom):
        if len(self.nav_path) < 2:
            return
        pts = [
            self._world_to_screen(p, content_rect, camera, zoom) for p in self.nav_path
        ]
        for i in range(len(pts) - 1):
            pygame.draw.line(self.screen, COLORS["nav_path"], pts[i], pts[i + 1], 2)

    # -----------------------------------------------------------------------
    # LIDAR tab
    # -----------------------------------------------------------------------

    def _render_lidar_tab(self, content_rect):
        self._draw_centered_grid(content_rect)
        cx, cy = content_rect.center
        for hit in self.lidar_hits:
            sa = hit.angle - math.pi / 2.0
            r_px = hit.distance * self.px_per_m
            sx = cx + r_px * math.cos(sa)
            sy = cy + r_px * math.sin(sa)
            pygame.draw.circle(
                self.screen, COLORS["lidar_point"], (int(sx), int(sy)), 2
            )
            pygame.draw.line(
                self.screen, COLORS["lidar_ray"], (cx, cy), (int(sx), int(sy)), 1
            )
        self._draw_bot_at_center(content_rect, COLORS["bot_true"])
        for r_m in (2, 4, 6, 8, 10):
            r_px = int(r_m * self.px_per_m)
            pygame.draw.circle(self.screen, (200, 200, 200), (cx, cy), r_px, 1)
            label = self.font.render(f"{r_m}m", True, (150, 150, 150))
            self.screen.blit(label, (cx + 5, cy - r_px - 16))

    # -----------------------------------------------------------------------
    # Guess tab
    # -----------------------------------------------------------------------

    def _render_guess_tab(self, content_rect):
        cam = (self.camera_x, self.camera_y)
        self._draw_grid(content_rect, cam, self.zoom)
        self._draw_rooms(content_rect, cam, self.zoom)
        self._draw_walls(content_rect, cam, self.zoom)
        self._draw_detected_obstacles(content_rect, cam, self.zoom)
        self._draw_apriltags(content_rect, cam, self.zoom)
        self._draw_nav_target(content_rect, cam, self.zoom)
        self._draw_nav_path(content_rect, cam, self.zoom)
        self._draw_bot(
            content_rect,
            cam,
            self.zoom,
            (self.guess_x, self.guess_y, self.guess_theta),
            COLORS["bot_guess"],
        )

    # -----------------------------------------------------------------------
    # Drawing primitives
    # -----------------------------------------------------------------------

    def _draw_grid(self, content_rect, camera, zoom):
        spacing = 2.5  # meters between grid lines
        cam_x, cam_y = camera
        w2 = content_rect.width / (2 * zoom * self.px_per_m)
        h2 = content_rect.height / (2 * zoom * self.px_per_m)

        x_start = math.floor((cam_x - w2) / spacing)
        x_end = math.ceil((cam_x + w2) / spacing)
        for i in range(x_start, x_end + 1):
            sx2, _ = self._world_to_screen((i * spacing, 0), content_rect, camera, zoom)
            pygame.draw.line(
                self.screen,
                COLORS["grid"],
                (sx2, content_rect.top),
                (sx2, content_rect.bottom),
            )

        y_start = math.floor((cam_y - h2) / spacing)
        y_end = math.ceil((cam_y + h2) / spacing)
        for j in range(y_start, y_end + 1):
            _, sy2 = self._world_to_screen((0, j * spacing), content_rect, camera, zoom)
            pygame.draw.line(
                self.screen,
                COLORS["grid"],
                (content_rect.left, sy2),
                (content_rect.right, sy2),
            )

    def _draw_centered_grid(self, content_rect):
        spacing = 50
        for x in range(content_rect.left, content_rect.right + 1, spacing):
            pygame.draw.line(
                self.screen,
                COLORS["grid"],
                (x, content_rect.top),
                (x, content_rect.bottom),
            )
        for y in range(content_rect.top, content_rect.bottom + 1, spacing):
            pygame.draw.line(
                self.screen,
                COLORS["grid"],
                (content_rect.left, y),
                (content_rect.right, y),
            )
        cx, cy = content_rect.center
        pygame.draw.line(
            self.screen,
            (180, 180, 180),
            (cx, content_rect.top),
            (cx, content_rect.bottom),
        )
        pygame.draw.line(
            self.screen,
            (180, 180, 180),
            (content_rect.left, cy),
            (content_rect.right, cy),
        )

    def _draw_rooms(self, content_rect, camera, zoom):
        for room in self.map_data.rooms:
            screen_poly = [
                self._world_to_screen(p, content_rect, camera, zoom)
                for p in room.polygon
            ]
            if len(screen_poly) >= 3:
                pygame.draw.polygon(self.screen, COLORS["room_fill"], screen_poly)
                pygame.draw.polygon(self.screen, COLORS["room_outline"], screen_poly, 2)
            cx, cy = self._world_to_screen(room.center, content_rect, camera, zoom)
            label = self.font.render(room.name, True, COLORS["room_outline"])
            self.screen.blit(
                label,
                (int(cx - label.get_width() / 2), int(cy - label.get_height() / 2)),
            )

    def _draw_walls(self, content_rect, camera, zoom):
        for wall in self.map_data.walls:
            p1 = self._world_to_screen((wall.x1, wall.y1), content_rect, camera, zoom)
            p2 = self._world_to_screen((wall.x2, wall.y2), content_rect, camera, zoom)
            pygame.draw.line(self.screen, COLORS["wall"], p1, p2, max(1, int(3 * zoom)))

    def _draw_lidar_rays(self, content_rect, camera, zoom, forward_theta):
        bot = self._world_to_screen(
            (self.bot_x, self.bot_y), content_rect, camera, zoom
        )
        for hit in self.lidar_hits:
            hs = self._world_to_screen(hit.point, content_rect, camera, zoom)
            pygame.draw.line(self.screen, COLORS["lidar_ray"], bot, hs, 1)
            pygame.draw.circle(
                self.screen, COLORS["lidar_point"], (int(hs[0]), int(hs[1])), 2
            )

    def _draw_bot(self, content_rect, camera, zoom, pose, color):
        x, y, theta = pose
        center = self._world_to_screen((x, y), content_rect, camera, zoom)
        radius = self.bot_radius_m * self.px_per_m * zoom
        half = radius
        corners = [(half, -half), (half, half), (-half, half), (-half, -half)]
        rotated = [rotate_point(c, theta) for c in corners]
        screen_c = [(center[0] + c[0], center[1] + c[1]) for c in rotated]
        pygame.draw.polygon(self.screen, color, screen_c)
        pygame.draw.polygon(self.screen, (0, 0, 0), screen_c, 2)
        nose = (
            center[0] + radius * 1.4 * math.cos(theta),
            center[1] + radius * 1.4 * math.sin(theta),
        )
        pygame.draw.line(self.screen, (0, 0, 0), center, nose, 3)

    def _draw_bot_at_center(self, content_rect, color):
        cx, cy = content_rect.center
        radius = self.bot_radius_m * self.px_per_m
        half = radius
        theta_s = -math.pi / 2.0
        corners = [(half, -half), (half, half), (-half, half), (-half, -half)]
        rotated = [rotate_point(c, theta_s) for c in corners]
        screen_c = [(cx + c[0], cy + c[1]) for c in rotated]
        pygame.draw.polygon(self.screen, color, screen_c)
        pygame.draw.polygon(self.screen, (0, 0, 0), screen_c, 2)
        nose = (cx, cy - radius * 1.4)
        pygame.draw.line(self.screen, (0, 0, 0), (cx, cy), nose, 3)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    map_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_map.json")
    Simulator(map_path).run()


if __name__ == "__main__":
    main()
