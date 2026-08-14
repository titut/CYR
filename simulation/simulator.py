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
    + / - - Zoom in / out (Map and Guess tabs).
    ESC   - Quit.

Zenoh topics:
    Published:
        sensor/lidar       — LIDAR scan data (list of {angle_rad, distance_m})
        sensor/wheel_speed — {linear_mps, angular_rps}
        sensor/imu         — {yaw_rad, pitch_rad, roll_rad, ...}
        nav/goal           — {x_px, y_px} user-clicked map target
        nav/command        — string, LLM text command from chat bar
    Subscribed:
        estimate/pose      — external pose estimate {x_px, y_px, theta_rad}
        nav/path           — planned waypoints [[x, y], ...]
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import pygame
import zenoh

# Allow running this file directly or as a module.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from map_format import MapData, Obstacle, Wall, new_empty_map
from simulation.raycast import RayHit, cast_rays

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
TAB_BAR_HEIGHT = 44
CHAT_BAR_HEIGHT = 36
FPS = 60

BOT_SIZE_M = 1.0
BOT_LINEAR_SPEED_MPS = 1.5
BOT_ANGULAR_SPEED_RPS = 2.0
LIDAR_MAX_RANGE_M = 10.0
LIDAR_RAY_COUNT = 360

# AprilTag camera simulation.
CAMERA_FOV_RAD = math.radians(90)  # horizontal field of view
CAMERA_MAX_RANGE_M = 5.0  # max detection distance
CAMERA_RANGE_NOISE_M = 0.02  # 1σ Gaussian noise on range measurement
CAMERA_BEARING_NOISE_RAD = math.radians(1.0)  # 1σ Gaussian noise on bearing

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
}

# Wheel slip simulation for autonomous navigation.
_SLIP_BIAS_MIN = 0.80
_SLIP_BIAS_MAX = 1.20
_SLIP_NOISE = 0.10
_CROSS_COUPLING_RAD_PER_M = 0.1


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


def point_to_segment_distance(px: float, py: float, wall: Wall) -> float:
    """Distance from point (px, py) to the wall segment."""
    wx = wall.x2 - wall.x1
    wy = wall.y2 - wall.y1
    length_sq = wx * wx + wy * wy
    if length_sq == 0:
        return math.hypot(px - wall.x1, py - wall.y1)
    t = max(0.0, min(1.0, ((px - wall.x1) * wx + (py - wall.y1) * wy) / length_sq))
    proj_x = wall.x1 + t * wx
    proj_y = wall.y1 + t * wy
    return math.hypot(px - proj_x, py - proj_y)


def bot_collides(
    bot_pos: Tuple[float, float],
    radius_px: float,
    walls: List[Wall],
    obstacles: Optional[List[Obstacle]] = None,
) -> bool:
    """Check whether a circular bot collides with any wall or obstacle."""
    for wall in walls:
        if point_to_segment_distance(bot_pos[0], bot_pos[1], wall) < radius_px:
            return True
    if obstacles:
        for obstacle in obstacles:
            d = math.hypot(bot_pos[0] - obstacle.x, bot_pos[1] - obstacle.y)
            if d < radius_px + obstacle.radius_px:
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

        self.map_data = self._load_map(map_path)
        self.scale_m_per_px = self.map_data.metadata.scale_m_per_px
        self.bot_radius_px = (BOT_SIZE_M / 2.0) / self.scale_m_per_px

        # Bot state in map pixels and radians.
        start_center = (
            polygon_center(self.map_data.rooms[0].polygon)
            if self.map_data.rooms
            else (400.0, 300.0)
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

        # Wheel slip simulation: random calibration bias per run.
        self._slip_bias = random.uniform(_SLIP_BIAS_MIN, _SLIP_BIAS_MAX)

        # Navigation state (path received from navigator over Zenoh).
        self.nav_target: Optional[Tuple[float, float]] = None
        self.nav_path: List[Tuple[float, float]] = []
        self.nav_path_index: int = 0

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
        self._pub_wheel_speed = self._zenoh_session.declare_publisher(
            "sensor/wheel_speed",
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

        # Subscribers.
        self._sub_pose = self._zenoh_session.declare_subscriber(
            "estimate/pose", self._on_estimate_pose
        )
        self._sub_path = self._zenoh_session.declare_subscriber(
            "nav/path", self._on_nav_path
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
            data = json.loads(sample.payload.to_string())
            with self._est_lock:
                self.guess_x = float(data["x_px"])
                self.guess_y = float(data["y_px"])
                self.guess_theta = float(data["theta_rad"])
        except (json.JSONDecodeError, KeyError, Exception) as exc:
            logging.warning("Failed to parse estimate/pose: %s", exc)

    def _on_nav_path(self, sample):
        """Receive planned path from navigator."""
        try:
            waypoints = json.loads(sample.payload.to_string())
            if not waypoints:
                logging.warning("Navigator returned empty path.")
                self.nav_path = []
                self.nav_path_index = 0
                return
            self.nav_path = [tuple(p) for p in waypoints]
            self.nav_path_index = 0
            logging.info("Received path with %d waypoints.", len(self.nav_path))
        except (json.JSONDecodeError, Exception) as exc:
            logging.warning("Failed to parse nav/path: %s", exc)

    # -----------------------------------------------------------------------
    # Publishing
    # -----------------------------------------------------------------------

    def _publish_zenoh(self):
        """Publish sensor data over Zenoh."""
        # Wheel speed: clean encoder-reported values.
        wheel_msg = json.dumps(
            {
                "linear_mps": self.linear_velocity * self.scale_m_per_px,
                "angular_rps": self.angular_velocity,
            }
        )
        self._pub_wheel_speed.put(wheel_msg)

        # LIDAR.
        lidar_msg = json.dumps(
            [
                {
                    "angle_rad": hit.angle,
                    "distance_m": hit.distance * self.scale_m_per_px,
                }
                for hit in self.lidar_hits
            ]
        )
        self._pub_lidar.put(lidar_msg)

        # IMU.
        imu_msg = json.dumps(
            {
                "yaw_rad": self.bot_theta,
                "pitch_rad": 0.0,
                "roll_rad": 0.0,
                "angular_velocity_rps": self.angular_velocity,
                "linear_acceleration_mps2": {"x": 0.0, "y": 0.0, "z": 0.0},
            }
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
                self._pub_goal.put(json.dumps({"x_px": world[0], "y_px": world[1]}))
                self.nav_target = world
            elif event.button == 3 and self.active_tab == 0:
                self.nav_target = None
                self.nav_path = []
                self.nav_path_index = 0

    # -----------------------------------------------------------------------
    # Update
    # -----------------------------------------------------------------------

    def _update(self, dt: float):
        keys = pygame.key.get_pressed()

        # ---- Autonomous navigation (path received from navigator) ----
        if self.nav_path and not (
            keys[pygame.K_w] or keys[pygame.K_s] or keys[pygame.K_a] or keys[pygame.K_d]
        ):
            self._update_navigation(dt)
        else:
            # ---- Manual mode (WASD) ----
            if any(keys[k] for k in (pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d)):
                self.nav_path = []
                self.nav_path_index = 0

            if keys[pygame.K_w]:
                self.linear_velocity = BOT_LINEAR_SPEED_MPS / self.scale_m_per_px
            elif keys[pygame.K_s]:
                self.linear_velocity = -BOT_LINEAR_SPEED_MPS / self.scale_m_per_px
            else:
                self.linear_velocity = 0.0

            if keys[pygame.K_a]:
                self.angular_velocity = -BOT_ANGULAR_SPEED_RPS
            elif keys[pygame.K_d]:
                self.angular_velocity = BOT_ANGULAR_SPEED_RPS
            else:
                self.angular_velocity = 0.0

        # ---- Apply wheel slip to actual movement (autonomous only) ----
        actual_linear = self.linear_velocity
        actual_angular = self.angular_velocity

        if self.nav_path and (actual_linear != 0.0 or actual_angular != 0.0):
            actual_linear *= self._slip_bias
            actual_angular *= self._slip_bias
            actual_angular += (
                actual_linear * self.scale_m_per_px * _CROSS_COUPLING_RAD_PER_M
            )
            actual_linear += random.gauss(0.0, _SLIP_NOISE * abs(actual_linear))
            actual_angular += random.gauss(0.0, _SLIP_NOISE * abs(actual_angular))

        new_theta = self.bot_theta + actual_angular * dt
        new_x = self.bot_x + actual_linear * math.cos(self.bot_theta) * dt
        new_y = self.bot_y + actual_linear * math.sin(self.bot_theta) * dt

        obstacles = self.map_data.obstacles
        if not bot_collides(
            (new_x, self.bot_y), self.bot_radius_px, self.map_data.walls, obstacles
        ):
            self.bot_x = new_x
        if not bot_collides(
            (self.bot_x, new_y), self.bot_radius_px, self.map_data.walls, obstacles
        ):
            self.bot_y = new_y
        self.bot_theta = new_theta

        self.camera_x += (self.bot_x - self.camera_x) * 0.1
        self.camera_y += (self.bot_y - self.camera_y) * 0.1

        self.lidar_hits = cast_rays(
            origin=(self.bot_x, self.bot_y),
            forward_direction=self.bot_theta,
            walls=self.map_data.walls,
            num_rays=LIDAR_RAY_COUNT,
            max_range=LIDAR_MAX_RANGE_M / self.scale_m_per_px,
            fov_rad=2.0 * math.pi,
            obstacles=self.map_data.obstacles,
        )
        self._publish_zenoh()
        self._detect_apriltags()

    # -----------------------------------------------------------------------
    # AprilTag detection
    # -----------------------------------------------------------------------

    def _detect_apriltags(self):
        """Simulate a forward-facing camera detecting AprilTags.

        For each tag in the map, check whether it falls within the camera's
        horizontal FOV and max range cone.  Detections are published with
        noisy range/bearing so the pose estimator can anchor its filter.
        """
        detections = []
        max_range_px = CAMERA_MAX_RANGE_M / self.scale_m_per_px
        half_fov = CAMERA_FOV_RAD / 2.0

        for tag in self.map_data.apriltags:
            # Vector from bot to tag in world coordinates.
            dx = tag.x - self.bot_x
            dy = tag.y - self.bot_y
            dist_px = math.hypot(dx, dy)
            if dist_px > max_range_px:
                continue

            # Bearing of the tag relative to the bot's forward direction.
            angle_to_tag = math.atan2(dy, dx)
            bearing = math.atan2(
                math.sin(angle_to_tag - self.bot_theta),
                math.cos(angle_to_tag - self.bot_theta),
            )

            if abs(bearing) > half_fov:
                continue

            # Add Gaussian noise to simulate real camera measurement.
            noisy_range_m = dist_px * self.scale_m_per_px + random.gauss(
                0.0, CAMERA_RANGE_NOISE_M
            )
            noisy_bearing = bearing + random.gauss(0.0, CAMERA_BEARING_NOISE_RAD)

            # The tag's yaw relative to the camera (used to derive robot pose).
            # Observed yaw of the tag from the camera's perspective.
            tag_yaw_in_camera = math.atan2(
                math.sin(tag.yaw_rad - self.bot_theta),
                math.cos(tag.yaw_rad - self.bot_theta),
            )

            detections.append(
                {
                    "id": tag.id,
                    "range_m": max(0.0, noisy_range_m),
                    "bearing_rad": noisy_bearing,
                    "tag_yaw_rad": tag_yaw_in_camera,
                    "tag_size_m": tag.size_m,
                }
            )

        if detections:
            self._pub_camera_apriltag.put(json.dumps(detections))

    # -----------------------------------------------------------------------
    # Navigation (path-following only; planning is external)
    # -----------------------------------------------------------------------

    def _update_navigation(self, dt: float):
        """Follow the path received from the navigator."""
        if not self.nav_path or self.nav_path_index >= len(self.nav_path):
            self.linear_velocity = 0.0
            self.angular_velocity = 0.0
            self.nav_path = []
            return

        gx, gy = self.guess_x, self.guess_y
        wp = self.nav_path[self.nav_path_index]
        to_wp = math.hypot(wp[0] - gx, wp[1] - gy)

        target_heading = math.atan2(wp[1] - gy, wp[0] - gx)
        angle_err = math.atan2(
            math.sin(target_heading - self.bot_theta),
            math.cos(target_heading - self.bot_theta),
        )

        if abs(angle_err) > 0.05:
            self.angular_velocity = max(
                -BOT_ANGULAR_SPEED_RPS,
                min(BOT_ANGULAR_SPEED_RPS, 4.0 * angle_err),
            )
            self.linear_velocity = 0.0
        else:
            self.angular_velocity = 4.0 * angle_err
            speed = BOT_LINEAR_SPEED_MPS / self.scale_m_per_px
            if to_wp < self.bot_radius_px * 4:
                speed *= max(0.2, to_wp / (self.bot_radius_px * 4))
            self.linear_velocity = speed

        if to_wp < self.bot_radius_px * 1.5:
            self.nav_path_index += 1
            if self.nav_path_index >= len(self.nav_path):
                self.linear_velocity = 0.0
                self.angular_velocity = 0.0
                self.nav_path = []
                self.nav_target = None

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
        self._pub_command.put(text)

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
            "WASD to drive | +/- zoom | ESC quit",
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
            content_rect.centerx + (point[0] - cx) * zoom,
            content_rect.centery + (point[1] - cy) * zoom,
        )

    def _screen_to_world(self, screen_pos, content_rect, camera, zoom):
        sx, sy = screen_pos
        cx, cy = camera
        return (
            (sx - content_rect.centerx) / zoom + cx,
            (sy - content_rect.centery) / zoom + cy,
        )

    def _world_to_screen_rotated(
        self, point, content_rect, origin, forward_theta, zoom
    ):
        dx, dy = point[0] - origin[0], point[1] - origin[1]
        rotation = -(forward_theta + math.pi / 2.0)
        cos_a, sin_a = math.cos(rotation), math.sin(rotation)
        return (
            content_rect.centerx + (dx * cos_a - dy * sin_a) * zoom,
            content_rect.centery + (dx * sin_a + dy * cos_a) * zoom,
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
            radius = obstacle.radius_px * zoom
            pygame.draw.circle(self.screen, COLORS["obstacle"], sc, max(1, int(radius)))
            pygame.draw.circle(
                self.screen, COLORS["obstacle_outline"], sc, max(1, int(radius)), 2
            )

    def _draw_apriltags(self, content_rect, camera, zoom):
        """Draw all AprilTags as coloured squares on the map.

        Tags within the camera's detection cone are highlighted yellow;
        out-of-range tags are drawn in a muted green.
        """
        max_range_px = CAMERA_MAX_RANGE_M / self.scale_m_per_px
        half_fov = CAMERA_FOV_RAD / 2.0

        for tag in self.map_data.apriltags:
            sc = self._world_to_screen((tag.x, tag.y), content_rect, camera, zoom)
            half = max(3.0, (tag.size_m / self.scale_m_per_px) * zoom / 2.0)

            # Check if this tag is currently detected.
            dx = tag.x - self.bot_x
            dy = tag.y - self.bot_y
            dist = math.hypot(dx, dy)
            within_range = dist <= max_range_px
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
        max_range_px = CAMERA_MAX_RANGE_M / self.scale_m_per_px
        half_fov = CAMERA_FOV_RAD / 2.0

        left_angle = self.bot_theta - half_fov
        right_angle = self.bot_theta + half_fov

        left_pt = (
            self.bot_x + max_range_px * math.cos(left_angle),
            self.bot_y + max_range_px * math.sin(left_angle),
        )
        right_pt = (
            self.bot_x + max_range_px * math.cos(right_angle),
            self.bot_y + max_range_px * math.sin(right_angle),
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
            sx = cx + hit.distance * math.cos(sa)
            sy = cy + hit.distance * math.sin(sa)
            pygame.draw.circle(
                self.screen, COLORS["lidar_point"], (int(sx), int(sy)), 2
            )
            pygame.draw.line(
                self.screen, COLORS["lidar_ray"], (cx, cy), (int(sx), int(sy)), 1
            )
        self._draw_bot_at_center(content_rect, COLORS["bot_true"])
        for r_m in (2, 4, 6, 8, 10):
            r_px = int(r_m / self.scale_m_per_px)
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
        self._draw_obstacles(content_rect, cam, self.zoom)
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
        spacing = 50
        cam_x, cam_y = camera
        w2 = content_rect.width / (2 * zoom)
        h2 = content_rect.height / (2 * zoom)
        sx, ex = (
            int((cam_x - w2) / spacing) * spacing,
            int((cam_x + w2) / spacing) * spacing,
        )
        sy, ey = (
            int((cam_y - h2) / spacing) * spacing,
            int((cam_y + h2) / spacing) * spacing,
        )
        for x in range(sx, ex + spacing, spacing):
            sx2, _ = self._world_to_screen((x, 0), content_rect, camera, zoom)
            pygame.draw.line(
                self.screen,
                COLORS["grid"],
                (sx2, content_rect.top),
                (sx2, content_rect.bottom),
            )
        for y in range(sy, ey + spacing, spacing):
            _, sy2 = self._world_to_screen((0, y), content_rect, camera, zoom)
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
        radius = self.bot_radius_px * zoom
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
        radius = self.bot_radius_px
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
