"""Zenoh-based navigation node.

Subscribes to:
    estimate/pose  — estimated pose from pose_estimator (meters)
    nav/goal       — user-clicked target in meters
    nav/command    — natural-language navigation command

Publishes:
    nav/path           — planned waypoints as JSON list of [x, y] pairs (meters)
    detection/obstacles — LIDAR-detected obstacle points as [[x, y], ...] (meters)

Usage:
    python zenoh/navigation/navigator.py [path/to/map.json]

Defaults to test_map.json if no map is provided.
"""

from __future__ import annotations

import json
import logging
import math
import os
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

from core.constants import LIDAR_MAX_RANGE_M
from core.map_format import MapData, new_empty_map
from core.messages import SchemaError, decode, decode_text, encode
from core.robot_config import get_robot_config
from simulation.occupancy_grid import OccupancyGrid, _bresenham_line
from navigation.astar import plan_path
from navigation.footprint import make_footprint
from navigation.llm_nav import query_location_async
from core.clock import sleep_until

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_REPLAN_THRESHOLD_M = 1.0  # Replan if bot strays >1 m from path.

# Probabilistic obstacle detection (T-009): obstacles come from the raw LIDAR
# point cloud via a log-odds occupancy grid with free-space clearing, instead
# of comparing each beam to a pose-rendered "expected wall scan" (which turns
# pose error into ghost obstacles).  Each hit cell gains occupancy log-odds;
# every cell between the robot and the hit is cleared.  Log-odds are clamped
# and slowly decay, so a pose-error ghost needs several consistent hits to
# appear and fades once the robot stops seeing it.
_LO_OCCUPIED_INC = 0.5    # log-odds added to a hit cell per scan
_LO_FREE_DEC = 0.25       # log-odds subtracted from a cleared cell per scan
_LO_MAX = 3.0             # log-odds clamp; a few consistent hits saturate
_LO_OCCUPIED = 0.7        # cell counts as occupied above this
_LO_DECAY = 0.05          # per-update pull toward 0 so stale cells fade

_CHECK_INTERVAL_S = 0.25  # how often to scan and (maybe) replan


def update_occupancy_log_odds(
    dyn_log_odds: np.ndarray,
    walls_grid: OccupancyGrid,
    est_x: float,
    est_y: float,
    est_theta: float,
    rays,
    max_range_m: float = LIDAR_MAX_RANGE_M,
) -> np.ndarray:
    """Update a log-odds occupancy array from one LIDAR scan (T-009).

    Each beam is registered to the map at the estimated pose: the hit cell
    gains occupancy log-odds, and every cell between the robot and the hit is
    cleared (free space).  Log-odds are clamped and slowly decay toward zero,
    so transient pose-error ghosts need several consistent hits to appear and
    fade once the robot looks away.  Known wall cells are left to the static
    grid rather than re-marked, so walls do not thicken into the free space
    when the pose estimate is slightly off.

    ``dyn_log_odds`` is mutated in place.  Returns the boolean mask of cells
    currently counted as occupied.
    """
    grid = walls_grid
    lo = dyn_log_odds
    gx0, gy0 = grid.world_to_grid(est_x, est_y)

    for entry in rays:
        angle = est_theta + float(entry["angle_rad"])
        dist = float(entry["distance_m"])
        hit_dist = min(dist, max_range_m)
        hx = est_x + hit_dist * math.cos(angle)
        hy = est_y + hit_dist * math.sin(angle)
        gx1, gy1 = grid.world_to_grid(hx, hy)

        # Free space between the robot and the hit (skipping known walls).
        # The hit cell itself is excluded here — it is handled below.
        line = _bresenham_line(gx0, gy0, gx1, gy1)
        for gx, gy in line[:-1]:
            if 0 <= gx < grid.cols and 0 <= gy < grid.rows:
                if not grid.grid[gy, gx]:
                    lo[gy, gx] -= _LO_FREE_DEC

        # The hit cell itself is occupied — unless it is a known wall.
        if dist < max_range_m:
            if 0 <= gx1 < grid.cols and 0 <= gy1 < grid.rows:
                if not grid.grid[gy1, gx1]:
                    lo[gy1, gx1] += _LO_OCCUPIED_INC

    # Decay stale observations toward "unknown" and clamp.
    lo[lo > 0] -= _LO_DECAY
    lo[lo < 0] += _LO_DECAY
    np.clip(lo, -_LO_MAX, _LO_MAX, out=lo)

    return lo > _LO_OCCUPIED


def closest_segment_index(path, x: float, y: float) -> int:
    """Index of the path segment whose point is closest to (x, y)."""
    best_idx = 0
    best_d2 = float("inf")
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        wx, wy = x2 - x1, y2 - y1
        length_sq = wx * wx + wy * wy
        t = (
            0.0
            if length_sq == 0
            else max(0.0, min(1.0, ((x - x1) * wx + (y - y1) * wy) / length_sq))
        )
        px = x1 + t * wx
        py = y1 + t * wy
        d2 = (px - x) ** 2 + (py - y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_idx = i
    return best_idx


class Navigator:
    def __init__(self, map_path: Path):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [navigator] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

        self.map_data = self._load_map(map_path)

        # Robot description (T-019): footprint and LIDAR specs from robot.yaml.
        cfg = get_robot_config()
        self._cfg = cfg
        self.bot_radius = cfg.chassis.radius_m
        self._lidar_range_m = cfg.sensors.lidar.range_m

        # Occupancy grids + footprint for planning.
        self._grid_resolution = 0.25  # meters per grid cell
        # Static walls (never re-marked as obstacles) and the combined grid
        # (walls + currently-detected dynamic obstacles) passed to A*.
        self._walls_grid = OccupancyGrid.from_walls(
            self.map_data.walls, self._grid_resolution
        )
        self.occ_grid = self._walls_grid
        # Log-odds of a dynamic obstacle per cell; 0 = no information.
        self._dyn_log_odds = np.zeros(
            (self._walls_grid.rows, self._walls_grid.cols), dtype=float
        )
        # World-meter points of cells currently counted as occupied (cached by
        # _update_occupancy, used for path-blocking checks and display).
        self._occupied_points: List[Tuple[float, float]] = []
        self._bot_footprint = make_footprint(self.bot_radius, self._grid_resolution)

        # Latest estimate from pose_estimator.
        self._est_x: float = 0.0
        self._est_y: float = 0.0
        self._est_theta: float = 0.0

        # LIDAR scan state for obstacle detection.
        self._lidar_lock = threading.Lock()
        self._latest_lidar: Optional[dict] = None

        # Current path waypoints (world meters), for blocking checks.
        self._current_path: List[Tuple[float, float]] = []

        # Current navigation target.
        self._nav_target: Optional[Tuple[float, float]] = None

        # LLM API key.
        self._api_key = os.environ.get("DEEPSEEK_API_KEY", "")

        # Zenoh session.
        self._session = zenoh.open(zenoh.Config())

        # Publisher: planned path.
        self._pub_path = self._session.declare_publisher(
            "nav/path",
            congestion_control=zenoh.CongestionControl.DROP,
            reliability=zenoh.Reliability.BEST_EFFORT,
        )

        # Publisher: LIDAR-detected obstacle points (world frame), for display.
        self._pub_detection = self._session.declare_publisher(
            "detection/obstacles",
            congestion_control=zenoh.CongestionControl.DROP,
            reliability=zenoh.Reliability.BEST_EFFORT,
        )

        # Subscriber: estimated pose.
        self._sub_pose = self._session.declare_subscriber(
            "estimate/pose", self._on_pose
        )

        # Subscriber: user-clicked goal.
        self._sub_goal = self._session.declare_subscriber("nav/goal", self._on_goal)

        # Subscriber: LLM text command.
        self._sub_command = self._session.declare_subscriber(
            "nav/command", self._on_command
        )

        # Subscriber: LIDAR scan (RingChannel(1) keeps only the latest).
        self._sub_lidar = self._session.declare_subscriber(
            "sensor/lidar", self._on_lidar
        )

    @staticmethod
    def _load_map(path: Path) -> MapData:
        if not path.exists():
            print(f"[navigator] Map not found: {path}. Using empty map.")
            return new_empty_map()
        return MapData.from_json(path)

    # -------------------------------------------------------------------
    # Zenoh callbacks
    # -------------------------------------------------------------------

    def _on_pose(self, sample):
        """Update the latest estimated pose."""
        try:
            data = decode("estimate/pose", sample)
            self._est_x = float(data["x_m"])
            self._est_y = float(data["y_m"])
            self._est_theta = float(data["theta_rad"])
        except SchemaError as exc:
            logging.warning("estimate/pose dropped: %s", exc)

    def _on_lidar(self, sample):
        """Store the latest LIDAR scan for obstacle detection."""
        try:
            scan = decode("sensor/lidar", sample)
        except SchemaError as exc:
            logging.warning("sensor/lidar dropped: %s", exc)
            return
        with self._lidar_lock:
            self._latest_lidar = scan

    def _on_goal(self, sample):
        """A user clicked a point on the map. Plan a path to it."""
        try:
            data = decode("nav/goal", sample)
            target = (float(data["x_m"]), float(data["y_m"]))
        except SchemaError as exc:
            logging.warning("nav/goal dropped: %s", exc)
            return

        self._nav_target = target
        self._current_path = []

        logging.info("Goal received: (%.2f, %.2f) m", target[0], target[1])
        # Plan immediately, but include any currently-visible obstacle particles
        # so the initial path already routes around known obstacles.
        self._plan_fresh(force=True)

    def _on_command(self, sample):
        """A user typed an LLM command. Resolve to coordinates, then plan."""
        try:
            text = decode_text("nav/command", sample)
        except SchemaError as exc:
            logging.warning("nav/command dropped: %s", exc)
            return
        if not self._api_key:
            logging.warning("No API key or empty command.")
            return

        logging.info("LLM command: %r", text)

        def _on_result(target_m: Optional[Tuple[float, float]]):
            if target_m is not None:
                self._nav_target = target_m
                self._current_path = []

                logging.info(
                    "LLM resolved to: (%.2f, %.2f) m",
                    target_m[0],
                    target_m[1],
                )
                # Plan immediately, including currently-visible obstacle
                # particles (see _on_goal).
                self._plan_fresh(force=True)
            else:
                logging.error("LLM query failed.")

        query_location_async(self.map_data, text, self._api_key, _on_result)

    # -------------------------------------------------------------------
    # Path planning
    # -------------------------------------------------------------------

    def _plan_and_publish(self, force: bool = False) -> bool:
        """Run A* from the latest estimated pose to the current target.

        The grid already includes any currently-detected obstacle cells, so A*
        naturally routes around them — no incremental "repair" needed.

        Args:
            force: If True, publish the result even if planning fails.
                   If False and a current path exists, keep the current path
                   on planning failure to avoid spamming empty paths.

        Returns:
            True if a non-empty path was found and published.
        """
        if self._nav_target is None:
            return False

        path = plan_path(
            self.occ_grid,
            (self._est_x, self._est_y),
            self._nav_target,
            self._bot_footprint,
        )

        if not path:
            logging.warning(
                "A* failed: no path from (%.1f, %.1f) to (%.1f, %.1f)",
                self._est_x,
                self._est_y,
                self._nav_target[0],
                self._nav_target[1],
            )
            # Only publish an empty path when we have no existing path (e.g.
            # initial planning failure). Otherwise keep the current path so the
            # robot does not stop every time a reactive replan fails.
            if not self._current_path or force:
                self._pub_path.put(json.dumps([]))
            return False

        # Track the path for obstacle-blocking checks.
        self._current_path = [(float(p[0]), float(p[1])) for p in path]

        # Convert to list of [x, y] for JSON.
        msg = json.dumps([[round(p[0], 1), round(p[1], 1)] for p in path])
        self._pub_path.put(msg)
        logging.info("Published path with %d waypoints.", len(path))
        return True

    def _plan_fresh(self, force: bool = False) -> bool:
        """Update occupancy from the latest scan, then plan.

        Used for brand-new goals so the initial path already routes around any
        obstacle the LIDAR can currently see, rather than planning on a clean
        walls-only grid and relying on a later reactive replan.
        """
        self._update_occupancy()
        return self._plan_and_publish(force=force)

    # -------------------------------------------------------------------
    # Obstacle detection + dynamic replanning
    # -------------------------------------------------------------------

    def _update_occupancy(self) -> None:
        """Update the dynamic occupancy grid from the latest LIDAR scan.

        See ``update_occupancy_log_odds`` for the per-scan update.  This method
        pulls the latest scan, applies the update, rebuilds the combined
        planning grid (walls OR dynamic-occupied), and publishes the occupied
        cells for display.
        """
        with self._lidar_lock:
            scan = self._latest_lidar
            self._latest_lidar = None
        if not scan:
            return

        occ = update_occupancy_log_odds(
            self._dyn_log_odds,
            self._walls_grid,
            self._est_x,
            self._est_y,
            self._est_theta,
            scan.get("rays", []),
            max_range_m=self._lidar_range_m,
        )

        # Rebuild the combined planning grid: walls OR dynamic-occupied.
        self.occ_grid = OccupancyGrid.from_walls(
            self.map_data.walls, self._grid_resolution
        )
        self.occ_grid.grid |= occ

        # Cache occupied points for path-blocking checks + display.
        points: List[Tuple[float, float]] = []
        for gy, gx in zip(*np.where(occ)):
            points.append(self._walls_grid.grid_to_world(gx, gy))
        self._occupied_points = points
        self._publish_detection(points)

    def _publish_detection(self, particles: List[Tuple[float, float]]) -> None:
        """Publish the detected obstacle points for the simulator to display."""
        msg = encode(
            "detection/obstacles",
            {
                "t": time.time(),
                "points": [[round(p[0], 3), round(p[1], 3)] for p in particles],
            },
        )
        self._pub_detection.put(msg)

    @staticmethod
    def _distance_point_to_segment(
        x: float, y: float, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        """Distance from (x, y) to the line segment (x1,y1)-(x2,y2)."""
        wx = x2 - x1
        wy = y2 - y1
        length_sq = wx * wx + wy * wy
        if length_sq == 0:
            return math.hypot(x - x1, y - y1)
        t = max(0.0, min(1.0, ((x - x1) * wx + (y - y1) * wy) / length_sq))
        proj_x = x1 + t * wx
        proj_y = y1 + t * wy
        return math.hypot(x - proj_x, y - proj_y)

    def _closest_segment_index(self, path, x: float, y: float) -> int:
        """Index of the path segment whose point is closest to (x, y)."""
        return closest_segment_index(path, x, y)

    def _first_blocked_segment_index(self) -> Optional[int]:
        """Return the index of the first remaining path segment blocked by a
        currently-detected obstacle cell, or None if the remaining path is clear.

        The scan starts at the segment the robot is currently projected onto,
        so the segment it is driving along is always checked.  Starting from a
        *waypoint* index instead can skip the robot's own segment — the old
        "already past this waypoint" heuristic jumped straight to the next
        waypoint, so a path that ran close to an obstacle right out of the
        start was never re-planned.
        """
        if len(self._current_path) < 2:
            return None

        start_idx = self._closest_segment_index(
            self._current_path, self._est_x, self._est_y
        )
        clearance = self.bot_radius * 1.5
        for i in range(start_idx, len(self._current_path) - 1):
            x1, y1 = self._current_path[i]
            x2, y2 = self._current_path[i + 1]
            for cx, cy in self._occupied_points:
                if self._distance_point_to_segment(cx, cy, x1, y1, x2, y2) < clearance:
                    return i
        return None

    def _check_and_replan(self):
        """Every cycle: update occupancy, then (re)plan if needed.

        Runs every _CHECK_INTERVAL_S seconds.  The dynamic occupancy grid is
        rebuilt from the latest LIDAR scan; then, if there is no current path
        or the remaining path is blocked, the path is re-planned from the
        current pose.  A* always routes around the currently-detected
        obstacles, so no incremental repair is needed.
        """
        if self._nav_target is None:
            return

        self._update_occupancy()

        blocked = self._first_blocked_segment_index()
        if blocked is not None:
            # Path is blocked: stop the robot, then re-plan from the current
            # pose (A* routes around whatever is marked on the grid).
            self._pub_path.put(json.dumps([]))
            if not self._plan_and_publish():
                # No route found: forget the stale blocked path so the next
                # cycle just tries to plan again instead of re-checking it.
                self._current_path = []
        elif not self._current_path:
            self._plan_and_publish()

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------

    def run(self):
        print("[navigator] Running. Press Ctrl+C to stop.")
        next_tick = time.monotonic()
        try:
            while True:
                self._check_and_replan()
                # Deadline-driven pacing (no cumulative drift from jitter).
                next_tick += _CHECK_INTERVAL_S
                sleep_until(next_tick)
        except KeyboardInterrupt:
            print("[navigator] Stopping…")
        finally:
            self._session.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    map_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_map.json")
    Navigator(map_path).run()


if __name__ == "__main__":
    main()
