"""Zenoh-based navigation node.

Subscribes to:
    estimate/pose  — estimated pose from pose_estimator (meters)
    nav/goal       — user-clicked target in meters
    nav/command    — natural-language navigation command

Publishes:
    nav/path       — planned waypoints as JSON list of [x, y] pairs (meters)

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

from map_format import MapData, new_empty_map
from simulation.occupancy_grid import OccupancyGrid
from simulation.raycast import cast_ray
from navigation.astar import plan_path
from navigation.footprint import make_footprint
from navigation.llm_nav import query_location_async

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BOT_SIZE_M = 1.0  # Must match simulator.
_REPLAN_THRESHOLD_M = 1.0  # Replan if bot strays >1 m from path.

# Obstacle detection via LIDAR residual (actual vs. expected wall-only scan).
LIDAR_MAX_RANGE_M = 10.0  # Must match simulator.
_OBSTACLE_DETECTION_THRESHOLD_M = 0.3  # residual larger than this = obstacle
# Pose-estimation error makes the residual non-zero even where there is no
# obstacle, producing transient "ghost" hit points.  A cell is only marked
# occupied after being hit this many times within the age window, so single-
# scan ghosts are filtered out while real (persistent) obstacles are kept.
_OBSTACLE_CONFIRM_HITS = 2
_OBSTACLE_HIT_MAX_AGE_S = 1.0
_CHECK_INTERVAL_S = 0.25  # how often to scan and (maybe) replan
_REPLAN_BACKOFF_S = 5.0  # wait before retrying after a failed replan


class Navigator:
    def __init__(self, map_path: Path):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [navigator] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

        self.map_data = self._load_map(map_path)
        self.bot_radius = BOT_SIZE_M / 2.0

        # Occupancy grid + footprint for planning.
        self._grid_resolution = 0.25  # meters per grid cell
        self.occ_grid = OccupancyGrid.from_walls(
            self.map_data.walls, self._grid_resolution
        )
        self._bot_footprint = make_footprint(
            self.bot_radius, self._grid_resolution
        )

        # Latest estimate from pose_estimator.
        self._est_x: float = 0.0
        self._est_y: float = 0.0
        self._est_theta: float = 0.0

        # LIDAR scan state for obstacle detection.
        self._lidar_lock = threading.Lock()
        self._latest_lidar: Optional[dict] = None

        # Per-cell obstacle hit counts: (gx, gy) -> (count, last_seen).  A cell
        # is only marked occupied after being hit _OBSTACLE_CONFIRM_HITS times,
        # which filters out single-scan pose-error ghosts.
        self._obstacle_hits: dict = {}

        # Current path waypoints (world meters), for blocking checks.
        self._current_path: List[Tuple[float, float]] = []

        # Current navigation target.
        self._nav_target: Optional[Tuple[float, float]] = None

        # Replanning state. Remember a blockage we failed to route around so we
        # don't retry the same unreachable plan every scan, and throttle
        # replanning after failures with a short backoff.
        self._failed_blocked_idx: Optional[int] = None
        self._replan_backoff_until: float = 0.0

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
            data = json.loads(sample.payload.to_string())
            self._est_x = float(data["x_m"])
            self._est_y = float(data["y_m"])
            self._est_theta = float(data["theta_rad"])
        except (json.JSONDecodeError, KeyError, Exception):
            pass

    def _on_lidar(self, sample):
        """Store the latest LIDAR scan for obstacle detection."""
        try:
            scan = json.loads(sample.payload.to_string())
        except (json.JSONDecodeError, Exception):
            return
        with self._lidar_lock:
            self._latest_lidar = scan

    def _on_goal(self, sample):
        """A user clicked a point on the map. Plan a path to it."""
        try:
            data = json.loads(sample.payload.to_string())
            target = (float(data["x_m"]), float(data["y_m"]))
        except (json.JSONDecodeError, KeyError, Exception) as exc:
            logging.warning("Failed to parse nav/goal: %s", exc)
            return

        self._nav_target = target
        self._current_path = []
        self._failed_blocked_idx = None
        self._replan_backoff_until = 0.0
        self._obstacle_hits = {}

        logging.info("Goal received: (%.2f, %.2f) m", target[0], target[1])
        # Plan immediately, but include any currently-visible obstacle particles
        # so the initial path already routes around known obstacles.
        self._plan_fresh(force=True)

    def _on_command(self, sample):
        """A user typed an LLM command. Resolve to coordinates, then plan."""
        text = sample.payload.to_string().strip()
        if not text or not self._api_key:
            logging.warning("No API key or empty command.")
            return

        logging.info("LLM command: %r", text)

        def _on_result(target_m: Optional[Tuple[float, float]]):
            if target_m is not None:
                self._nav_target = target_m
                self._current_path = []
                self._failed_blocked_idx = None
                self._replan_backoff_until = 0.0
                self._obstacle_hits = {}

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
        """Run RRT* from the latest estimated pose to the current target.

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
                "RRT* failed: no path from (%.1f, %.1f) to (%.1f, %.1f)",
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
        """Detect current obstacle particles, rebuild the grid, then plan.

        Used for brand-new goals so the initial path already routes around any
        obstacle the LIDAR can currently see, rather than planning on a clean
        walls-only grid and relying on a later reactive replan.
        """
        particles = self._detect_obstacle_particles()
        self._populate_grid(particles)
        return self._plan_and_publish(force=force)

    # -------------------------------------------------------------------
    # Obstacle detection + dynamic replanning
    # -------------------------------------------------------------------

    def _detect_obstacle_particles(self) -> List[Tuple[float, float]]:
        """Compare actual LIDAR scan to the expected wall-only scan.

        Any beam whose actual range is significantly shorter than the wall it
        should be seeing indicates an obstacle blocking the view.  The hit
        points (obstacle particles) are returned as world-meter coordinates.
        """
        with self._lidar_lock:
            scan = self._latest_lidar
            self._latest_lidar = None

        if not scan or self._nav_target is None:
            return []

        max_range_m = LIDAR_MAX_RANGE_M
        particles: List[Tuple[float, float]] = []

        for entry in scan.get("rays", []):
            angle_rad = entry["angle_rad"]
            actual_m = entry["distance_m"]

            # Expected distance to the nearest wall at this angle.
            result = cast_ray(
                (self._est_x, self._est_y),
                self._est_theta + angle_rad,
                self.map_data.walls,
                max_range_m,
            )
            expected_m = max_range_m if result is None else result[0]

            if expected_m - actual_m > _OBSTACLE_DETECTION_THRESHOLD_M:
                hit_x = self._est_x + actual_m * math.cos(
                    self._est_theta + angle_rad
                )
                hit_y = self._est_y + actual_m * math.sin(
                    self._est_theta + angle_rad
                )
                particles.append((hit_x, hit_y))

        return particles

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

    def _current_path_index(self) -> int:
        """Return the index of the waypoint the robot is currently driving toward.

        The closest waypoint is used, but if the robot has already moved past it
        along the path direction, the next waypoint is returned instead.
        """
        if len(self._current_path) < 2:
            return 0

        best_idx = 0
        best_d2 = float("inf")
        for i, (wx, wy) in enumerate(self._current_path):
            d2 = (wx - self._est_x) ** 2 + (wy - self._est_y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i

        if best_idx + 1 < len(self._current_path):
            wx, wy = self._current_path[best_idx]
            nx, ny = self._current_path[best_idx + 1]
            dx_to_robot = self._est_x - wx
            dy_to_robot = self._est_y - wy
            dx_path = nx - wx
            dy_path = ny - wy
            if dx_to_robot * dx_path + dy_to_robot * dy_path > 0:
                best_idx += 1
        return best_idx

    def _first_blocked_segment_index(
        self, obstacles: List[Tuple[float, float]]
    ) -> Optional[int]:
        """Return the index of the first remaining path segment blocked by
        any obstacle particle, or None if the remaining path is clear.
        """
        if len(self._current_path) < 2:
            return None

        start_idx = self._current_path_index()
        clearance = self.bot_radius
        for i in range(start_idx, len(self._current_path) - 1):
            x1, y1 = self._current_path[i]
            x2, y2 = self._current_path[i + 1]
            for cx, cy in obstacles:
                if (
                    self._distance_point_to_segment(cx, cy, x1, y1, x2, y2)
                    < clearance
                ):
                    return i
        return None

    def _populate_grid(self, particles: List[Tuple[float, float]]) -> None:
        """Rebuild the occupancy grid from walls and mark confirmed obstacles.

        Each LIDAR hit point lands in a single grid cell.  A cell is only
        marked occupied after being hit _OBSTACLE_CONFIRM_HITS times within a
        short window, so transient pose-error ghosts (which move between scans)
        are filtered out while real obstacles (hit consistently) are kept.  The
        planner's robot footprint already supplies the clearance, so no extra
        inflation is applied.
        """
        self.occ_grid = OccupancyGrid.from_walls(
            self.map_data.walls, self._grid_resolution
        )
        now = time.time()

        # Forget hits that have not been re-observed recently.
        for key in list(self._obstacle_hits.keys()):
            if now - self._obstacle_hits[key][1] > _OBSTACLE_HIT_MAX_AGE_S:
                del self._obstacle_hits[key]

        # Record this scan's hits.
        seen: set = set()
        for cx, cy in particles:
            gx, gy = self.occ_grid.world_to_grid(cx, cy)
            if 0 <= gx < self.occ_grid.cols and 0 <= gy < self.occ_grid.rows:
                seen.add((gx, gy))
        for key in seen:
            count, _ = self._obstacle_hits.get(key, (0, 0.0))
            self._obstacle_hits[key] = (count + 1, now)

        # Mark confirmed cells.
        for (gx, gy), (count, _) in self._obstacle_hits.items():
            if count >= _OBSTACLE_CONFIRM_HITS:
                self.occ_grid.grid[gy, gx] = True

    def _repair_path(self, blocked_idx: int) -> bool:
        """Repair the current path by keeping the portion from the robot's
        current position to just before the blocked segment, planning a new
        suffix from that safe point to the goal, and publishing the result.

        Returns True if a repaired (or fallback) path was published.
        """
        # Stop the robot while a new path is being computed.
        self._pub_path.put(json.dumps([]))
        logging.info("Published empty path while replanning.")

        # Determine the first waypoint on the current path that is ahead of the
        # robot. We start the repaired path at the robot's estimated pose.
        start_idx = self._current_path_index()

        # The waypoint just before the blocked segment. Never go earlier than
        # the robot's current progress, so we do not ask the robot to reverse.
        safe_idx = max(start_idx, blocked_idx - 1)
        safe_point = self._current_path[safe_idx]

        logging.info(
            "Repairing path from waypoint %d (%.1f, %.1f) onward.",
            safe_idx,
            safe_point[0],
            safe_point[1],
        )

        suffix = plan_path(
            self.occ_grid,
            safe_point,
            self._nav_target,
            self._bot_footprint,
        )

        if not suffix:
            logging.warning(
                "Path repair failed from (%.1f, %.1f); falling back to full replan.",
                safe_point[0],
                safe_point[1],
            )
            return self._plan_and_publish()

        # Build the repaired path: the still-valid waypoints from the current
        # path up to and including safe_point, then the new suffix (excluding
        # its first point, which is safe_point).  We start from the first
        # still-valid waypoint rather than the raw estimated pose, which can
        # drift outside the map and would otherwise be published as a waypoint.
        prefix = list(self._current_path[start_idx : safe_idx + 1])
        suffix_without_start = list(suffix[1:])
        repaired = prefix + suffix_without_start

        self._current_path = [(float(p[0]), float(p[1])) for p in repaired]
        msg = json.dumps([[round(p[0], 1), round(p[1], 1)] for p in repaired])
        self._pub_path.put(msg)
        logging.info(
            "Published repaired path with %d waypoints (prefix=%d, suffix=%d).",
            len(repaired),
            len(prefix),
            len(suffix_without_start),
        )
        return True

    def _check_and_replan(self):
        """Detect obstacle particles, mark them on the grid, and replan if needed.

        Each scan rebuilds the grid from the static walls plus the currently
        visible obstacle particles, so a plan made here already accounts for
        obstacles.  If a particle blocks the remaining path, the path is
        repaired around it.  A failed repair is remembered so the same blockage
        is not retried every scan; planning resumes only once the obstacle
        situation changes.
        """
        if self._nav_target is None:
            return

        now = time.time()

        # Detect obstacle particles and rebuild the grid from walls + particles
        # before any planning, so paths account for known obstacles up front.
        particles = self._detect_obstacle_particles()
        self._populate_grid(particles)

        # If we have a target but no current path (for example an initial plan
        # failed, or the previous path was cleared), plan now on the freshly
        # populated grid — but only after the backoff has elapsed, so a
        # persistently unreachable goal is not hammered every scan.
        if not self._current_path:
            if now < self._replan_backoff_until:
                return
            logging.info(
                "No current path to target (%.1f, %.1f); attempting to plan.",
                self._nav_target[0],
                self._nav_target[1],
            )
            if not self._plan_and_publish():
                self._replan_backoff_until = now + _REPLAN_BACKOFF_S
            return

        # Only replan if a particle actually blocks the remaining path; there is
        # no need to issue a new plan for particles behind the robot or far from
        # the current route.
        blocked_idx = self._first_blocked_segment_index(particles)

        if blocked_idx is None:
            # Path is clear. If a previous repair failed and we have been
            # waiting for the blockage to clear, resume now.
            if self._failed_blocked_idx is not None:
                self._failed_blocked_idx = None
                logging.info("Obstacle cleared; replanning from current pose.")
                self._plan_and_publish()
            return

        # Same blockage we already failed to route around: wait for the world to
        # change instead of re-running RRT every scan.
        if blocked_idx == self._failed_blocked_idx:
            return

        logging.info(
            "Obstacle particle blocks remaining path segment %d; replanning.",
            blocked_idx,
        )
        if self._repair_path(blocked_idx):
            self._failed_blocked_idx = None
        else:
            self._failed_blocked_idx = blocked_idx
            self._replan_backoff_until = now + _REPLAN_BACKOFF_S

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------

    def run(self):
        print("[navigator] Running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(_CHECK_INTERVAL_S)
                self._check_and_replan()
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
