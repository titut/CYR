"""Zenoh-based navigation node.

Subscribes to:
    estimate/pose  — estimated pose from pose_estimator
    nav/goal       — user-clicked target in map pixels
    nav/command    — natural-language navigation command

Publishes:
    nav/path       — planned waypoints as JSON list of [x, y] pairs

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
from navigation.rrt import plan_path
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
_OBSTACLE_CLUSTER_RADIUS_M = 0.5  # points within this cluster together
# Detected hit points sit on the obstacle's *near* surface, so we mark a circle
# big enough to cover the obstacle body plus the robot's radius plus a margin.
_OBSTACLE_MARK_MARGIN_PX = 10.0  # extra clearance beyond bot radius
_CHECK_INTERVAL_S = 0.25  # how often to scan and (maybe) replan

# Dynamic obstacle confirmation / decay.
# A point is only marked on the grid after it has been observed this many
# times; this filters out single-scan pose-error ghosts.
_OBSTACLE_CONFIRM_SCANS = 3
# Confirmed obstacles are forgotten if not re-observed for this long.
_OBSTACLE_DECAY_AGE_S = 5.0
# Candidate obstacles (not yet confirmed) decay faster.
_OBSTACLE_CANDIDATE_DECAY_AGE_S = 1.5
# Candidates within this distance of an existing track are merged.
_OBSTACLE_MERGE_RADIUS_M = 0.5


class Navigator:
    def __init__(self, map_path: Path):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [navigator] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

        self.map_data = self._load_map(map_path)
        self.scale_m_per_px = self.map_data.metadata.scale_m_per_px
        self.bot_radius_px = (BOT_SIZE_M / 2.0) / self.scale_m_per_px

        # Occupancy grid + footprint for A*.
        self._grid_resolution_px = max(1.0, 0.25 / self.scale_m_per_px)
        self.occ_grid = OccupancyGrid.from_walls(
            self.map_data.walls, self._grid_resolution_px
        )
        self._bot_footprint = make_footprint(
            self.bot_radius_px, self._grid_resolution_px
        )

        # Latest estimate from pose_estimator.
        self._est_x: float = 0.0
        self._est_y: float = 0.0
        self._est_theta: float = 0.0

        # LIDAR scan state for obstacle detection.
        self._lidar_lock = threading.Lock()
        self._latest_lidar: Optional[List[dict]] = None

        # Tracked dynamic obstacles. Each entry maps an arbitrary id to a dict:
        #   center: (float, float)  - estimated world position
        #   confidence: int           - consecutive observation count
        #   last_seen: float            - time of last observation
        # An obstacle is only marked on the grid once confidence reaches
        # _OBSTACLE_CONFIRM_SCANS.
        self._dynamic_obstacles: dict = {}
        self._confirmed_obstacle_keys: set = set()

        # Current path waypoints (world px), for blocking checks.
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
            self._est_x = float(data["x_px"])
            self._est_y = float(data["y_px"])
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
            target = (float(data["x_px"]), float(data["y_px"]))
        except (json.JSONDecodeError, KeyError, Exception) as exc:
            logging.warning("Failed to parse nav/goal: %s", exc)
            return

        self._nav_target = target
        self._current_path = []

        # New goal means a fresh world model: unknown dynamic obstacles from a
        # previous navigation should not poison this plan.
        self._reset_dynamic_obstacles_and_grid()

        logging.info("Goal received: (%.1f, %.1f) px", target[0], target[1])
        # Force-publish the new plan (even empty) because the old path, if any,
        # leads to a different goal.
        self._plan_and_publish(force=True)

    def _on_command(self, sample):
        """A user typed an LLM command. Resolve to coordinates, then plan."""
        text = sample.payload.to_string().strip()
        if not text or not self._api_key:
            logging.warning("No API key or empty command.")
            return

        logging.info("LLM command: %r", text)

        def _on_result(target_px: Optional[Tuple[float, float]]):
            if target_px is not None:
                self._nav_target = target_px
                self._current_path = []

                # See _on_goal: brand-new goal starts with a clean grid.
                self._reset_dynamic_obstacles_and_grid()

                logging.info(
                    "LLM resolved to: (%.1f, %.1f) px",
                    target_px[0],
                    target_px[1],
                )
                # Force-publish because the target changed.
                self._plan_and_publish(force=True)
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

    # -------------------------------------------------------------------
    # Obstacle detection + dynamic replanning
    # -------------------------------------------------------------------

    def _detect_obstacles(self) -> List[Tuple[float, float]]:
        """Compare actual LIDAR scan to the expected wall-only scan.

        Any beam whose actual range is significantly shorter than the wall
        it should be seeing indicates an obstacle blocking the view.  Hit
        points are clustered into obstacle centers (world px).
        """
        with self._lidar_lock:
            scan = self._latest_lidar
            self._latest_lidar = None

        if not scan or self._nav_target is None:
            return []

        max_range_px = LIDAR_MAX_RANGE_M / self.scale_m_per_px
        blocked_hits: List[Tuple[float, float]] = []

        for entry in scan:
            angle_rad = entry["angle_rad"]
            actual_m = entry["distance_m"]

            # Expected distance to the nearest wall at this angle.
            result = cast_ray(
                (self._est_x, self._est_y),
                self._est_theta + angle_rad,
                self.map_data.walls,
                max_range_px,
            )
            expected_px = max_range_px if result is None else result[0]
            expected_m = expected_px * self.scale_m_per_px

            if expected_m - actual_m > _OBSTACLE_DETECTION_THRESHOLD_M:
                hit_dist_px = actual_m / self.scale_m_per_px
                hit_x = self._est_x + hit_dist_px * math.cos(
                    self._est_theta + angle_rad
                )
                hit_y = self._est_y + hit_dist_px * math.sin(
                    self._est_theta + angle_rad
                )
                blocked_hits.append((hit_x, hit_y))

        return self._cluster_obstacle_points(blocked_hits)

    def _cluster_obstacle_points(
        self, points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Greedy-cluster blocked hit points into obstacle centers."""
        cluster_radius_px = _OBSTACLE_CLUSTER_RADIUS_M / self.scale_m_per_px
        clusters: List[List] = []  # each is [cx, cy, count]

        for px, py in points:
            placed = False
            for cluster in clusters:
                cx, cy, count = cluster
                if math.hypot(px - cx, py - cy) < cluster_radius_px:
                    new_count = count + 1
                    cluster[0] = cx + (px - cx) / new_count
                    cluster[1] = cy + (py - cy) / new_count
                    cluster[2] = new_count
                    placed = True
                    break
            if not placed:
                clusters.append([px, py, 1])

        return [(cx, cy) for cx, cy, _ in clusters]

    @staticmethod
    def _distance_point_to_segment(
        px: float, py: float, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        """Distance from (px, py) to the line segment (x1,y1)-(x2,y2)."""
        wx = x2 - x1
        wy = y2 - y1
        length_sq = wx * wx + wy * wy
        if length_sq == 0:
            return math.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, ((px - x1) * wx + (py - y1) * wy) / length_sq))
        proj_x = x1 + t * wx
        proj_y = y1 + t * wy
        return math.hypot(px - proj_x, py - proj_y)

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
            dy_to_robot = self._est_x - wy
            dx_path = nx - wx
            dy_path = ny - wy
            if dx_to_robot * dx_path + dy_to_robot * dy_path > 0:
                best_idx += 1
        return best_idx

    def _obstacle_blocks_remaining_path(self, cx: float, cy: float) -> bool:
        """True if the obstacle circle (with clearance) intersects the path.

        Only the portion of the path still ahead of the robot is considered.
        """
        if len(self._current_path) < 2:
            return False

        start_idx = self._current_path_index()
        clearance_px = self.bot_radius_px + _OBSTACLE_MARK_MARGIN_PX
        for i in range(start_idx, len(self._current_path) - 1):
            x1, y1 = self._current_path[i]
            x2, y2 = self._current_path[i + 1]
            if self._distance_point_to_segment(cx, cy, x1, y1, x2, y2) < clearance_px:
                return True
        return False

    def _first_blocked_segment_index(self) -> Optional[int]:
        """Return the index of the first remaining path segment blocked by
        any confirmed obstacle, or None if the remaining path is clear.
        """
        if len(self._current_path) < 2:
            return None

        start_idx = self._current_path_index()
        clearance_px = self.bot_radius_px + _OBSTACLE_MARK_MARGIN_PX
        for i in range(start_idx, len(self._current_path) - 1):
            x1, y1 = self._current_path[i]
            x2, y2 = self._current_path[i + 1]
            for key in self._confirmed_obstacle_keys:
                cx, cy = self._dynamic_obstacles[key]["center"]
                if self._distance_point_to_segment(cx, cy, x1, y1, x2, y2) < clearance_px:
                    return i
        return None

    def _find_nearest_tracked_obstacle(
        self, cx: float, cy: float, radius_px: float
    ) -> Optional[str]:
        """Return the id of a tracked obstacle within radius_px of (cx, cy)."""
        best_key: Optional[str] = None
        best_d2 = radius_px * radius_px
        for key, obs in self._dynamic_obstacles.items():
            ox, oy = obs["center"]
            d2 = (ox - cx) ** 2 + (oy - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_key = key
        return best_key

    def _rebuild_occupancy_grid(self) -> None:
        """Rebuild the occupancy grid from walls plus confirmed obstacles."""
        self.occ_grid = OccupancyGrid.from_walls(
            self.map_data.walls, self._grid_resolution_px
        )
        clearance_px = self.bot_radius_px + _OBSTACLE_MARK_MARGIN_PX
        for key in self._confirmed_obstacle_keys:
            cx, cy = self._dynamic_obstacles[key]["center"]
            self.occ_grid.mark_circle(cx, cy, clearance_px)

    def _reset_dynamic_obstacles_and_grid(self) -> None:
        """Clear all dynamic obstacle state and rebuild the occupancy grid
        from walls only.

        Called when a brand-new navigation goal is received so every new goal
        starts from the same clean world model as the first navigation.
        Dynamic obstacles will be re-detected and re-added if they block the
        new path.
        """
        self._dynamic_obstacles = {}
        self._confirmed_obstacle_keys = set()
        self.occ_grid = OccupancyGrid.from_walls(
            self.map_data.walls, self._grid_resolution_px
        )

    def _repair_path(self) -> bool:
        """Repair the current path by keeping the portion from the robot's
        current position to just before the blocked segment, planning a new
        suffix from that safe point to the goal, and publishing the result.

        Returns True if a repaired (or fallback) path was published.
        """
        blocked_idx = self._first_blocked_segment_index()
        if blocked_idx is None:
            return False

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

        # Build the repaired path: current estimated pose, then the still-valid
        # waypoints from the current path up to and including safe_point, then
        # the new suffix (excluding its first point, which is safe_point).
        current_pose = (self._est_x, self._est_y)
        prefix = [current_pose] + list(self._current_path[start_idx : safe_idx + 1])
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
        """Detect obstacles, confirm them over time, decay stale ones, and replan.

        Obstacles are only marked on the grid after being observed multiple
        times; this suppresses single-scan ghosts caused by pose errors.
        Confirmed obstacles that are no longer observed decay and are removed,
        so old false positives do not permanently block the map.
        """
        if self._nav_target is None:
            return

        # If we have a target but no current path (for example an initial plan
        # failed, or the previous path was cleared), retry planning now instead
        # of waiting for an obstacle-blocked path event. This ensures a failed
        # attempt does not permanently stop navigation.
        if not self._current_path:
            logging.info(
                "No current path to target (%.1f, %.1f); attempting to plan.",
                self._nav_target[0],
                self._nav_target[1],
            )
            self._plan_and_publish()

        centers = self._detect_obstacles()
        now = time.time()
        merge_radius_px = _OBSTACLE_MERGE_RADIUS_M / self.scale_m_per_px
        updated_keys: set = set()

        # Update existing tracks or create new ones for each candidate.
        for cx, cy in centers:
            key = self._find_nearest_tracked_obstacle(cx, cy, merge_radius_px)
            if key is not None:
                obs = self._dynamic_obstacles[key]
                n = obs["confidence"]
                obs["center"] = (
                    (obs["center"][0] * n + cx) / (n + 1),
                    (obs["center"][1] * n + cy) / (n + 1),
                )
                obs["confidence"] = n + 1
                obs["last_seen"] = now
                updated_keys.add(key)
            else:
                key = f"{round(cx)}:{round(cy)}:{now:.4f}"
                self._dynamic_obstacles[key] = {
                    "center": (cx, cy),
                    "confidence": 1,
                    "last_seen": now,
                }
                updated_keys.add(key)

        # Decay obstacles that were not updated this scan.
        for key in list(self._dynamic_obstacles.keys()):
            if key in updated_keys:
                continue
            obs = self._dynamic_obstacles[key]
            age = now - obs["last_seen"]
            max_age = (
                _OBSTACLE_DECAY_AGE_S
                if obs["confidence"] >= _OBSTACLE_CONFIRM_SCANS
                else _OBSTACLE_CANDIDATE_DECAY_AGE_S
            )
            if age > max_age:
                del self._dynamic_obstacles[key]

        # Recompute confirmed set.
        new_confirmed = {
            key
            for key, obs in self._dynamic_obstacles.items()
            if obs["confidence"] >= _OBSTACLE_CONFIRM_SCANS
        }
        confirmed_changed = new_confirmed != self._confirmed_obstacle_keys
        self._confirmed_obstacle_keys = new_confirmed

        if confirmed_changed:
            self._rebuild_occupancy_grid()
            logging.info(
                "Confirmed obstacle set changed (%d confirmed).",
                len(self._confirmed_obstacle_keys),
            )

        # Only replan if the remaining path is actually blocked. Rebuilding the
        # grid on every confirmed-set change keeps the world model up to date,
        # but there is no need to issue a new plan for obstacles behind the
        # robot or far from the current route.
        path_blocked = False
        for key in self._confirmed_obstacle_keys:
            cx, cy = self._dynamic_obstacles[key]["center"]
            if self._obstacle_blocks_remaining_path(cx, cy):
                logging.info(
                    "Confirmed obstacle at (%.1f, %.1f) blocks remaining path; replanning.",
                    cx,
                    cy,
                )
                path_blocked = True
                break

        if path_blocked:
            self._repair_path()

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
