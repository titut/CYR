"""Live top-down viewer for the 3D simulation (T3D).

A passive zenoh client that renders the same views as the 2D simulator's GUI,
but from the 3D sim's published data instead of computing its own physics:

    Map   — the map, ground-truth pose (blue), the pose estimator's guess
            (orange), LIDAR rays in world frame, planned path and goal.
    LIDAR — the raw LIDAR scan, robot-centred, with range rings.
    Guess — the world as the estimator believes it: its guess pose, the LIDAR
            scan drawn from that pose, and the navigator's detected obstacles.

Run it alongside ``simulation3d/simulator.py --map home.json --gui`` (and the
pose-estimator/controller/drive stack if you want live estimates):

    python zenoh/sim_viewer.py home.json

Topics:
    Subscribes:  sensor/lidar        — {"t", "rays": [{angle_rad, distance_m}]}
                 estimate/pose       — {"x_m", "y_m", "theta_rad", "cov", ...}
                 sim/truth/pose      — {"x_m", "y_m", "theta_rad"}
                 nav/path            — [[x, y], ...]
                 nav/goal            — {"x_m", "y_m"}
                 detection/obstacles — {"points": [[x, y], ...]}

Controls:
    1 / 2 / 3 - Switch tabs: Map | LIDAR | Guess.
    Arrows    - Teleop (publishes sensor/wasd; needs the controller to relay).
    Left-click on a map view - set a nav/goal for the navigator.
    R         - Send safety/reset to clear a latched drive e-stop.
    + / -     - Zoom in / out.
    ESC       - Quit.

The bottom banner shows the drive's safety state (nominal / slow_down / stop /
estop_latched) from safety/status; R clears a latched e-stop.

AprilTags the camera currently sees are highlighted (gold with a ring) on the
Map and Guess tabs, and the camera's field-of-view cone is drawn from the robot
heading, fed by sensor/camera/apriltag.
"""

from __future__ import annotations

import logging
import math
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

from core.map_format import MapData, new_empty_map
from core.messages import SchemaError, decode, decode_path, encode, encode_text
from core.robot_config import load_robot_config

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
TAB_BAR_HEIGHT = 44
FPS = 60

COLORS = {
    "bg": (245, 245, 245),
    "grid": (230, 230, 230),
    "wall": (30, 30, 30),
    "room_fill": (227, 242, 253),
    "room_outline": (33, 150, 243),
    "bot_true": (33, 150, 243),
    "bot_guess": (255, 87, 34),
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
    "detected_obstacle": (230, 0, 100),
}


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


class SimViewer:
    def __init__(self, map_path: Path):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("3D Sim Viewer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Helvetica", 16)
        self.big_font = pygame.font.SysFont("Helvetica", 20, bold=True)

        self.map_data = self._load_map(map_path)
        self.px_per_m = 1.0 / self.map_data.metadata.scale_m_per_px

        # All shared state below is written by zenoh callbacks (IO threads) and
        # read by the render loop (main thread); guard with a single lock.
        self._lock = threading.Lock()

        # Ground truth from sim/truth/pose; default to map origin so the
        # camera has somewhere to point before the first message arrives.
        self.truth = (0.0, 0.0, 0.0)
        # Pose estimator guess from estimate/pose (None until first estimate).
        self.guess: Optional[Tuple[float, float, float]] = None
        self.cov: Optional[List[List[float]]] = None

        self.lidar_rays: List[Tuple[float, float]] = []  # (angle_rad, distance_m)
        self.nav_path: List[Tuple[float, float]] = []
        self.nav_goal: Optional[Tuple[float, float]] = None
        self.detected: List[Tuple[float, float]] = []
        # Drive safety state from safety/status (nominal/slow_down/stop/
        # estop_latched) and the body clearance it was computed from.
        self.safety_state: str = "unknown"
        self.safety_clearance: Optional[float] = None
        # AprilTags currently seen by the camera: tag id -> last-seen
        # monotonic time (from sensor/camera/apriltag).
        self.detected_tags: dict = {}
        self._cfg = load_robot_config()
        self._cam = self._cfg.sensors.camera

        self.active_tab = 0
        self.tabs = ["Map", "LIDAR", "Guess"]
        self.zoom = 1.0

        # Zenoh.
        self._session = zenoh.open(zenoh.Config())
        self._session.declare_subscriber("sensor/lidar", self._on_lidar)
        self._session.declare_subscriber("estimate/pose", self._on_pose)
        self._session.declare_subscriber("sim/truth/pose", self._on_truth)
        self._session.declare_subscriber("nav/path", self._on_path)
        self._session.declare_subscriber("nav/goal", self._on_goal)
        self._session.declare_subscriber("detection/obstacles", self._on_detection)
        self._session.declare_subscriber("safety/status", self._on_safety)
        self._session.declare_subscriber(
            "sensor/camera/apriltag", self._on_camera_apriltag
        )

        # Remote control: in headless runs the sim has no keyboard/mouse, so the
        # viewer relays them over zenoh — arrow keys publish sensor/wasd (the
        # controller turns them into cmd/velocity), clicking the map publishes
        # nav/goal for the navigator, and R sends safety/reset to clear a
        # latched drive e-stop.
        self._pub_wasd = self._session.declare_publisher(
            "sensor/wasd",
            congestion_control=zenoh.CongestionControl.DROP,
            reliability=zenoh.Reliability.BEST_EFFORT,
        )
        self._pub_goal = self._session.declare_publisher(
            "nav/goal",
            congestion_control=zenoh.CongestionControl.DROP,
            reliability=zenoh.Reliability.BEST_EFFORT,
        )
        self._pub_reset = self._session.declare_publisher(
            "safety/reset",
            congestion_control=zenoh.CongestionControl.DROP,
            reliability=zenoh.Reliability.BEST_EFFORT,
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

    def _on_lidar(self, sample):
        try:
            data = decode("sensor/lidar", sample)
            rays = [
                (float(r["angle_rad"]), float(r["distance_m"]))
                for r in data.get("rays", [])
            ]
        except SchemaError as exc:
            logging.warning("sensor/lidar dropped: %s", exc)
            return
        with self._lock:
            self.lidar_rays = rays

    def _on_pose(self, sample):
        try:
            data = decode("estimate/pose", sample)
        except SchemaError as exc:
            logging.warning("estimate/pose dropped: %s", exc)
            return
        with self._lock:
            self.guess = (
                float(data["x_m"]),
                float(data["y_m"]),
                float(data["theta_rad"]),
            )
            self.cov = data.get("cov")

    def _on_truth(self, sample):
        try:
            data = decode("sim/truth/pose", sample)
        except SchemaError as exc:
            logging.warning("sim/truth/pose dropped: %s", exc)
            return
        with self._lock:
            self.truth = (
                float(data["x_m"]),
                float(data["y_m"]),
                float(data["theta_rad"]),
            )

    def _on_path(self, sample):
        try:
            waypoints = decode_path("nav/path", sample)
            path = [tuple(p) for p in waypoints]
        except SchemaError as exc:
            logging.warning("nav/path dropped: %s", exc)
            return
        with self._lock:
            self.nav_path = path

    def _on_goal(self, sample):
        try:
            data = decode("nav/goal", sample)
            goal = (float(data["x_m"]), float(data["y_m"]))
        except SchemaError as exc:
            logging.warning("nav/goal dropped: %s", exc)
            return
        with self._lock:
            self.nav_goal = goal

    def _on_detection(self, sample):
        try:
            data = decode("detection/obstacles", sample)
            points = [tuple(p) for p in data["points"]]
        except SchemaError as exc:
            logging.warning("detection/obstacles dropped: %s", exc)
            return
        with self._lock:
            self.detected = points

    def _on_safety(self, sample):
        try:
            data = decode("safety/status", sample)
        except SchemaError as exc:
            logging.warning("safety/status dropped: %s", exc)
            return
        with self._lock:
            self.safety_state = str(data.get("state", "unknown"))
            self.safety_clearance = data.get("min_clearance_m")

    def _on_camera_apriltag(self, sample):
        """Remember which AprilTags the camera currently sees."""
        try:
            data = decode("sensor/camera/apriltag", sample)
            dets = data.get("detections", [])
        except SchemaError as exc:
            logging.warning("sensor/camera/apriltag dropped: %s", exc)
            return
        now = time.monotonic()
        with self._lock:
            for d in dets:
                self.detected_tags[int(d["id"])] = now

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self):
        running = True
        try:
            while running:
                self.clock.tick(FPS)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        self._handle_key(event)
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        self._handle_click(event)
                self._publish_teleop()
                self._render()
        finally:
            self._session.close()
        pygame.quit()

    def _publish_teleop(self):
        """Relay arrow keys as sensor/wasd (headless remote control)."""
        keys = pygame.key.get_pressed()
        wasd = {
            "w": bool(keys[pygame.K_UP]),
            "a": bool(keys[pygame.K_LEFT]),
            "s": bool(keys[pygame.K_DOWN]),
            "d": bool(keys[pygame.K_RIGHT]),
        }
        self._pub_wasd.put(encode("sensor/wasd", wasd))

    def _handle_key(self, event):
        if event.key == pygame.K_ESCAPE:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        elif event.key == pygame.K_1:
            self.active_tab = 0
        elif event.key == pygame.K_2:
            self.active_tab = 1
        elif event.key == pygame.K_3:
            self.active_tab = 2
        elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
            self.zoom = min(8.0, self.zoom * 1.1)
        elif event.key == pygame.K_MINUS:
            self.zoom = max(0.1, self.zoom / 1.1)
        elif event.key == pygame.K_r:
            # Clear a latched drive e-stop (T-016).
            self._pub_reset.put(encode_text("safety/reset", "reset"))

    def _handle_click(self, event):
        if event.button == 1 and event.pos[1] <= TAB_BAR_HEIGHT:
            tab_width = WINDOW_WIDTH // len(self.tabs)
            self.active_tab = min(event.pos[0] // tab_width, len(self.tabs) - 1)
            return
        # Left-click on a map view sets a nav/goal (headless remote control).
        if event.button == 1 and self.active_tab in (0, 2):
            content_rect = pygame.Rect(0, TAB_BAR_HEIGHT, WINDOW_WIDTH,
                                       WINDOW_HEIGHT - TAB_BAR_HEIGHT)
            with self._lock:
                if self.active_tab == 0:
                    cam = (self.truth[0], self.truth[1])
                elif self.guess is not None:
                    cam = (self.guess[0], self.guess[1])
                else:
                    cam = (self.truth[0], self.truth[1])
            world = self._screen_to_world(event.pos, content_rect, cam, self.zoom)
            msg = encode("nav/goal", {"x_m": world[0], "y_m": world[1]})
            self._pub_goal.put(msg)
            with self._lock:
                self.nav_goal = world

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def _render(self):
        self.screen.fill(COLORS["bg"])
        content_rect = pygame.Rect(0, TAB_BAR_HEIGHT, WINDOW_WIDTH,
                                   WINDOW_HEIGHT - TAB_BAR_HEIGHT)

        with self._lock:
            truth = self.truth
            guess = self.guess
            cov = self.cov
            rays = list(self.lidar_rays)
            path = list(self.nav_path)
            goal = self.nav_goal
            detected = list(self.detected)
            safety_state = self.safety_state
            safety_clearance = self.safety_clearance
            detected_tags = dict(self.detected_tags)

        if self.active_tab == 0:
            self._render_map(content_rect, truth, guess, rays, path, goal, detected,
                             detected_tags)
        elif self.active_tab == 1:
            self._render_lidar(content_rect, rays)
        else:
            self._render_guess(content_rect, guess, rays, detected, detected_tags)

        self._render_tab_bar()
        self._render_overlay(truth, guess, cov, len(rays), safety_state, safety_clearance)
        pygame.display.flip()

    def _render_tab_bar(self):
        tab_width = WINDOW_WIDTH // len(self.tabs)
        for i, label in enumerate(self.tabs):
            rect = pygame.Rect(i * tab_width, 0, tab_width, TAB_BAR_HEIGHT)
            color = COLORS["tab_active"] if i == self.active_tab else COLORS["tab_inactive"]
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (180, 180, 180), rect, 1)
            surf = self.big_font.render(label, True, COLORS["tab_text"])
            self.screen.blit(surf, surf.get_rect(center=rect.center))

    def _render_overlay(self, truth, guess, cov, n_rays, safety_state, safety_clearance):
        lines = [
            f"Tab: {self.tabs[self.active_tab]}  (1/2/3 to switch)",
            f"truth: x={truth[0]:6.2f}  y={truth[1]:6.2f}  θ={math.degrees(truth[2]):7.1f}°",
        ]
        if guess is not None:
            err = math.hypot(guess[0] - truth[0], guess[1] - truth[1])
            lines.append(
                f"guess: x={guess[0]:6.2f}  y={guess[1]:6.2f}  θ={math.degrees(guess[2]):7.1f}°  "
                f"|err|={err:5.2f} m"
            )
            if cov is not None:
                std = math.sqrt(cov[0][0] + cov[1][1])
                lines.append(f"  cov std_xy={std:5.2f} m")
        else:
            lines.append("guess: (no estimate/pose yet)")
        lines.append(f"LIDAR rays: {n_rays}")
        y = TAB_BAR_HEIGHT + 10
        for line in lines:
            surf = self.font.render(line, True, COLORS["ui_text"])
            self.screen.blit(surf, (10, y))
            y += 22

        # Drive safety banner (T-016).  Color it by severity so a latched
        # e-stop is unmissable; R sends safety/reset to clear it.
        if safety_state == "estop_latched":
            bg = (255, 60, 60)
            text = "E-STOP LATCHED — press R, then reverse (down arrow) to back away"
            fg = (255, 255, 255)
        elif safety_state == "stop":
            bg = (255, 150, 50)
            text = "STOP ZONE — forward halted"
            fg = (0, 0, 0)
        elif safety_state == "slow_down":
            bg = (255, 230, 120)
            text = "SLOW DOWN"
            fg = (0, 0, 0)
        elif safety_state == "nominal":
            bg = (120, 220, 120)
            text = "NOMINAL"
            fg = (0, 0, 0)
        else:
            bg = (200, 200, 200)
            text = f"SAFETY: {safety_state}"
            fg = (0, 0, 0)
        if safety_clearance is not None:
            text += f"  (clearance {safety_clearance:.2f} m)"
        banner = pygame.Rect(0, WINDOW_HEIGHT - 36, WINDOW_WIDTH, 36)
        pygame.draw.rect(self.screen, bg, banner)
        surf = self.font.render(text, True, fg)
        self.screen.blit(surf, surf.get_rect(center=banner.center))

    # -----------------------------------------------------------------------
    # Coordinate transforms
    # -----------------------------------------------------------------------

    def _world_to_screen(self, point, rect, camera, zoom):
        cx, cy = camera
        return (
            rect.centerx + (point[0] - cx) * self.px_per_m * zoom,
            rect.centery + (point[1] - cy) * self.px_per_m * zoom,
        )

    def _screen_to_world(self, screen_pos, rect, camera, zoom):
        cx, cy = camera
        sx, sy = screen_pos
        return (
            (sx - rect.centerx) / (self.px_per_m * zoom) + cx,
            (sy - rect.centery) / (self.px_per_m * zoom) + cy,
        )

    def _lidar_world_point(self, pose, ray, distance):
        """World-frame (x, y) of a LIDAR ray from ``pose`` heading."""
        angle, d = ray
        world_angle = pose[2] + angle
        return (
            pose[0] + d * math.cos(world_angle),
            pose[1] + d * math.sin(world_angle),
        )

    # -----------------------------------------------------------------------
    # Map tab
    # -----------------------------------------------------------------------

    def _render_map(self, rect, truth, guess, rays, path, goal, detected,
                    detected_tags):
        cam = (truth[0], truth[1])
        self._draw_grid(rect, cam, self.zoom)
        self._draw_rooms(rect, cam, self.zoom)
        self._draw_walls(rect, cam, self.zoom)
        self._draw_camera_fov(rect, cam, self.zoom, truth)
        self._draw_apriltags(rect, cam, self.zoom, detected_tags)
        self._draw_nav_path(rect, cam, self.zoom, path)
        self._draw_nav_goal(rect, cam, self.zoom, goal)
        self._draw_detected(rect, cam, self.zoom, detected)
        self._draw_lidar_world(rect, cam, self.zoom, truth, rays)
        self._draw_bot(rect, cam, self.zoom, truth, COLORS["bot_true"])
        if guess is not None:
            self._draw_bot(rect, cam, self.zoom, guess, COLORS["bot_guess"])

    # -----------------------------------------------------------------------
    # LIDAR tab
    # -----------------------------------------------------------------------

    def _render_lidar(self, rect, rays):
        self._draw_centered_grid(rect)
        cx, cy = rect.center
        for angle, d in rays:
            sa = angle - math.pi / 2.0
            r_px = d * self.px_per_m
            sx = cx + r_px * math.cos(sa)
            sy = cy + r_px * math.sin(sa)
            pygame.draw.circle(self.screen, COLORS["lidar_point"], (int(sx), int(sy)), 2)
            pygame.draw.line(self.screen, COLORS["lidar_ray"], (cx, cy), (int(sx), int(sy)), 1)
        # Robot marker (facing up).
        half = max(3, int(0.375 * self.px_per_m))
        sq = pygame.Rect(cx - half, cy - half, 2 * half, 2 * half)
        pygame.draw.rect(self.screen, COLORS["bot_true"], sq)
        pygame.draw.rect(self.screen, (0, 0, 0), sq, 2)
        pygame.draw.line(self.screen, (0, 0, 0), (cx, cy), (cx, cy - 1.4 * half), 3)
        # Range rings.
        for r_m in (2, 4, 6, 8, 10):
            r_px = int(r_m * self.px_per_m)
            pygame.draw.circle(self.screen, (200, 200, 200), (cx, cy), r_px, 1)
            label = self.font.render(f"{r_m}m", True, (150, 150, 150))
            self.screen.blit(label, (cx + 5, cy - r_px - 16))

    # -----------------------------------------------------------------------
    # Guess tab
    # -----------------------------------------------------------------------

    def _render_guess(self, rect, guess, rays, detected, detected_tags):
        if guess is None:
            self._draw_centered_grid(rect)
            msg = self.font.render("No estimate/pose yet", True, COLORS["ui_text"])
            self.screen.blit(msg, msg.get_rect(center=rect.center))
            return
        cam = (guess[0], guess[1])
        self._draw_grid(rect, cam, self.zoom)
        self._draw_rooms(rect, cam, self.zoom)
        self._draw_walls(rect, cam, self.zoom)
        self._draw_camera_fov(rect, cam, self.zoom, guess)
        self._draw_apriltags(rect, cam, self.zoom, detected_tags)
        self._draw_detected(rect, cam, self.zoom, detected)
        # LIDAR drawn from the *guess* pose: the world the estimator believes it
        # is in, given the actual scan.
        self._draw_lidar_world(rect, cam, self.zoom, guess, rays)
        self._draw_bot(rect, cam, self.zoom, guess, COLORS["bot_guess"])

    # -----------------------------------------------------------------------
    # Drawing primitives
    # -----------------------------------------------------------------------

    def _draw_grid(self, rect, camera, zoom):
        spacing = 2.5
        cam_x, cam_y = camera
        w2 = rect.width / (2 * zoom * self.px_per_m)
        h2 = rect.height / (2 * zoom * self.px_per_m)
        for i in range(math.floor((cam_x - w2) / spacing), math.ceil((cam_x + w2) / spacing) + 1):
            sx, _ = self._world_to_screen((i * spacing, 0), rect, camera, zoom)
            pygame.draw.line(self.screen, COLORS["grid"], (sx, rect.top), (sx, rect.bottom))
        for j in range(math.floor((cam_y - h2) / spacing), math.ceil((cam_y + h2) / spacing) + 1):
            _, sy = self._world_to_screen((0, j * spacing), rect, camera, zoom)
            pygame.draw.line(self.screen, COLORS["grid"], (rect.left, sy), (rect.right, sy))

    def _draw_centered_grid(self, rect):
        for x in range(rect.left, rect.right + 1, 50):
            pygame.draw.line(self.screen, COLORS["grid"], (x, rect.top), (x, rect.bottom))
        for y in range(rect.top, rect.bottom + 1, 50):
            pygame.draw.line(self.screen, COLORS["grid"], (rect.left, y), (rect.right, y))

    def _draw_rooms(self, rect, camera, zoom):
        for room in self.map_data.rooms:
            poly = [self._world_to_screen(p, rect, camera, zoom) for p in room.polygon]
            if len(poly) >= 3:
                pygame.draw.polygon(self.screen, COLORS["room_fill"], poly)
                pygame.draw.polygon(self.screen, COLORS["room_outline"], poly, 2)
            cx, cy = self._world_to_screen(room.center, rect, camera, zoom)
            label = self.font.render(room.name, True, COLORS["room_outline"])
            self.screen.blit(label, (int(cx - label.get_width() / 2), int(cy - label.get_height() / 2)))

    def _draw_walls(self, rect, camera, zoom):
        for wall in self.map_data.walls:
            p1 = self._world_to_screen((wall.x1, wall.y1), rect, camera, zoom)
            p2 = self._world_to_screen((wall.x2, wall.y2), rect, camera, zoom)
            pygame.draw.line(self.screen, COLORS["wall"], p1, p2, max(1, int(3 * zoom)))

    def _draw_apriltags(self, rect, camera, zoom, detected_tags):
        now = time.monotonic()
        for tag in self.map_data.apriltags:
            sc = self._world_to_screen((tag.x, tag.y), rect, camera, zoom)
            half = max(3.0, tag.size_m * self.px_per_m * zoom / 2.0)
            # Highlight tags the camera currently sees (within the last 1.5 s;
            # the sim only publishes on detection, so this fades them out).
            seen = now - detected_tags.get(tag.id, 0.0) < 1.5
            color = COLORS["apriltag_detected"] if seen else COLORS["apriltag"]
            sq = pygame.Rect(sc[0] - half, sc[1] - half, half * 2, half * 2)
            pygame.draw.rect(self.screen, color, sq)
            pygame.draw.rect(self.screen, (0, 0, 0), sq, 1)
            if seen:
                radius = int(half * 2.2)
                pygame.draw.circle(
                    self.screen, (255, 170, 0), (int(sc[0]), int(sc[1])), radius, 2
                )
            tip = (sc[0] + half * 1.6 * math.cos(tag.yaw_rad),
                   sc[1] + half * 1.6 * math.sin(tag.yaw_rad))
            pygame.draw.line(self.screen, (0, 0, 0), sc, tip, 2)
            label = self.font.render(f"T{tag.id}", True, (0, 0, 0))
            self.screen.blit(label, (int(sc[0] - label.get_width() / 2), int(sc[1] - half - 16)))

    def _draw_camera_fov(self, rect, camera, zoom, pose):
        """Draw the camera's field-of-view cone from ``pose``'s heading."""
        bot = self._world_to_screen((pose[0], pose[1]), rect, camera, zoom)
        max_range = self._cam.max_range_m
        half_fov = self._cam.fov_rad / 2.0
        theta = pose[2]
        lp = (pose[0] + max_range * math.cos(theta - half_fov),
              pose[1] + max_range * math.sin(theta - half_fov))
        rp = (pose[0] + max_range * math.cos(theta + half_fov),
              pose[1] + max_range * math.sin(theta + half_fov))
        ls = self._world_to_screen(lp, rect, camera, zoom)
        rs = self._world_to_screen(rp, rect, camera, zoom)
        try:
            surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            pygame.draw.polygon(
                surf, (*COLORS["apriltag_detected"][:3], 40), [bot, ls, rs]
            )
            self.screen.blit(surf, (0, 0))
        except (pygame.error, ValueError):
            pass
        pygame.draw.line(self.screen, (180, 140, 0), bot, ls, 1)
        pygame.draw.line(self.screen, (180, 140, 0), bot, rs, 1)

    def _draw_nav_goal(self, rect, camera, zoom, goal):
        if goal is None:
            return
        cx, cy = self._world_to_screen(goal, rect, camera, zoom)
        s = 10
        pygame.draw.line(self.screen, COLORS["nav_target"], (cx - s, cy - s), (cx + s, cy + s), 2)
        pygame.draw.line(self.screen, COLORS["nav_target"], (cx + s, cy - s), (cx - s, cy + s), 2)

    def _draw_nav_path(self, rect, camera, zoom, path):
        if len(path) < 2:
            return
        pts = [self._world_to_screen(p, rect, camera, zoom) for p in path]
        for i in range(len(pts) - 1):
            pygame.draw.line(self.screen, COLORS["nav_path"], pts[i], pts[i + 1], 2)

    def _draw_detected(self, rect, camera, zoom, detected):
        radius = max(2, int(0.1 * self.px_per_m * zoom))
        for px, py in detected:
            sc = self._world_to_screen((px, py), rect, camera, zoom)
            pygame.draw.circle(self.screen, COLORS["detected_obstacle"], sc, radius)
            pygame.draw.circle(self.screen, (90, 0, 40), sc, radius, 1)

    def _draw_lidar_world(self, rect, camera, zoom, pose, rays):
        origin = self._world_to_screen((pose[0], pose[1]), rect, camera, zoom)
        for angle, d in rays:
            p = self._lidar_world_point(pose, (angle, d), d)
            hp = self._world_to_screen(p, rect, camera, zoom)
            pygame.draw.line(self.screen, COLORS["lidar_ray"], origin, hp, 1)
            pygame.draw.circle(self.screen, COLORS["lidar_point"], (int(hp[0]), int(hp[1])), 2)

    def _draw_bot(self, rect, camera, zoom, pose, color):
        x, y, theta = pose
        center = self._world_to_screen((x, y), rect, camera, zoom)
        radius = 0.375 * self.px_per_m * zoom
        corners = [(radius, -radius), (radius, radius), (-radius, radius), (-radius, -radius)]
        rotated = [rotate_point(c, theta) for c in corners]
        screen_c = [(center[0] + c[0], center[1] + c[1]) for c in rotated]
        pygame.draw.polygon(self.screen, color, screen_c)
        pygame.draw.polygon(self.screen, (0, 0, 0), screen_c, 2)
        nose = (center[0] + radius * 1.4 * math.cos(theta),
                center[1] + radius * 1.4 * math.sin(theta))
        pygame.draw.line(self.screen, (0, 0, 0), center, nose, 3)


def main():
    map_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("home.json")
    SimViewer(map_path).run()


if __name__ == "__main__":
    main()
