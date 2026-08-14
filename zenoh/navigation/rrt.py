"""RRT* path planner for 2D occupancy grids.

Adapted from Atsushi Sakai's RRT* implementation.
Uses OccupancyGrid for collision checking so it works with our wall geometry.

Exposes a plan_path() function matching the A* interface so the navigator
can swap planners transparently.
"""

from __future__ import annotations

import logging
import math
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Allow running this file directly or as a module.
_SCRIPT_DIR = Path(__file__).resolve().parent
_ZENOH_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _ZENOH_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_ZENOH_DIR) not in sys.path:
    sys.path.insert(0, str(_ZENOH_DIR))

from simulation.occupancy_grid import OccupancyGrid

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_EXPAND_DIS = 15.0  # px — how far each tree edge extends
_DEFAULT_PATH_RESOLUTION = 2.5  # px — resolution for edge collision sampling
_DEFAULT_GOAL_SAMPLE_RATE = 25  # % — chance to sample the goal directly
_DEFAULT_MAX_ITER = 3000  # max RRT iterations
_DEFAULT_CONNECT_CIRCLE_DIST = 45.0  # px — rewiring ball radius factor


def _segment_collision_free(
    grid: OccupancyGrid,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    half_size_px: float,
    step: float = _DEFAULT_PATH_RESOLUTION,
) -> bool:
    """True if the straight line (x1,y1)→(x2,y2) has no occupied cells."""
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        return _point_collision_free(grid, x1, y1, half_size_px)

    steps = max(1, int(dist / step))
    for i in range(steps + 1):
        t = i / steps
        x = x1 + dx * t
        y = y1 + dy * t
        gx, gy = grid.world_to_grid(x, y)
        # Check a small square around the point for the robot's size.
        margin_cells = max(1, int(half_size_px / grid.resolution))
        for dx_c in range(-margin_cells, margin_cells + 1):
            for dy_c in range(-margin_cells, margin_cells + 1):
                if grid.is_occupied(gx + dx_c, gy + dy_c):
                    return False
    return True


def _point_collision_free(
    grid: OccupancyGrid,
    x: float,
    y: float,
    half_size_px: float,
) -> bool:
    """True if the robot footprint centered at (x, y) is collision-free."""
    gx, gy = grid.world_to_grid(x, y)
    margin_cells = max(1, int(half_size_px / grid.resolution))
    for dx_c in range(-margin_cells, margin_cells + 1):
        for dy_c in range(-margin_cells, margin_cells + 1):
            if grid.is_occupied(gx + dx_c, gy + dy_c):
                return False
    return True


def _find_free_nearby(
    grid: OccupancyGrid,
    x: float,
    y: float,
    half_size_px: float,
    search_radius_px: float = 25.0,
    angle_steps: int = 16,
) -> Optional[Tuple[float, float]]:
    """Return a collision-free point near (x, y), or None if none exists.

    If (x, y) is already free, returns it. Otherwise searches the occupancy
    grid cells within search_radius_px of the requested point and returns the
    closest cell centre whose robot footprint is collision-free.
    """
    if _point_collision_free(grid, x, y, half_size_px):
        return (x, y)

    center_gx, center_gy = grid.world_to_grid(x, y)
    radius_cells = int(math.ceil(search_radius_px / grid.resolution))

    best_dist = float("inf")
    best_point: Optional[Tuple[float, float]] = None

    for dgx in range(-radius_cells, radius_cells + 1):
        for dgy in range(-radius_cells, radius_cells + 1):
            gx = center_gx + dgx
            gy = center_gy + dgy
            if gx < 0 or gx >= grid.cols or gy < 0 or gy >= grid.rows:
                continue

            wx, wy = grid.grid_to_world(gx, gy)
            if _point_collision_free(grid, wx, wy, half_size_px):
                dist = math.hypot(wx - x, wy - y)
                if dist < best_dist:
                    best_dist = dist
                    best_point = (wx, wy)

    return best_point


class _Node:
    """RRT tree node."""

    __slots__ = ("x", "y", "parent", "cost")

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.parent: Optional["_Node"] = None
        self.cost: float = 0.0


class RRTStar:
    """RRT* path planner.

    Parameters correspond to the sample code, adapted to map-pixel units.
    """

    def __init__(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        grid: OccupancyGrid,
        half_size_px: float,
        rand_area: Tuple[float, float, float, float],  # min_x, min_y, max_x, max_y
        expand_dis: float = _DEFAULT_EXPAND_DIS,
        path_resolution: float = _DEFAULT_PATH_RESOLUTION,
        goal_sample_rate: int = _DEFAULT_GOAL_SAMPLE_RATE,
        max_iter: int = _DEFAULT_MAX_ITER,
        connect_circle_dist: float = _DEFAULT_CONNECT_CIRCLE_DIST,
    ):
        self.start = _Node(start[0], start[1])
        self.goal = _Node(goal[0], goal[1])
        self.grid = grid
        self.half_size_px = half_size_px
        self.min_rand, self.max_rand = rand_area[0], rand_area[2]  # x-range
        self.min_y, self.max_y = rand_area[1], rand_area[3]  # y-range
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.connect_circle_dist = connect_circle_dist
        self.node_list: List[_Node] = []

    # -------------------------------------------------------------------
    # Main planning loop
    # -------------------------------------------------------------------

    def plan(self) -> Optional[List[Tuple[float, float]]]:
        """Run RRT* and return an ordered list of (x, y) waypoints, or None."""
        self.node_list = [self.start]

        for _ in range(self.max_iter):
            rnd = self._get_random_node()
            nearest_idx = self._get_nearest_node_index(self.node_list, rnd)
            nearest = self.node_list[nearest_idx]
            new_node = self._steer(nearest, rnd, self.expand_dis)

            if new_node is None:
                continue

            new_node.cost = nearest.cost + math.hypot(
                new_node.x - nearest.x, new_node.y - nearest.y
            )

            if not self._check_collision(new_node):
                continue

            near_inds = self._find_near_nodes(new_node)
            new_node = self._choose_parent(new_node, near_inds) or new_node
            if new_node is None:
                continue

            self._rewire(new_node, near_inds)
            self.node_list.append(new_node)

        # Best path to goal.
        best_idx = self._search_best_goal_node()
        if best_idx is not None:
            return self._generate_final_course(best_idx)

        return None

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _get_random_node(self) -> _Node:
        if random.randint(0, 100) < self.goal_sample_rate:
            return _Node(self.goal.x, self.goal.y)
        rx = random.uniform(self.min_rand, self.max_rand)
        ry = random.uniform(self.min_y, self.max_y)
        return _Node(rx, ry)

    @staticmethod
    def _get_nearest_node_index(node_list: List[_Node], rnd: _Node) -> int:
        best_idx = 0
        best_d2 = float("inf")
        for i, node in enumerate(node_list):
            d2 = (node.x - rnd.x) ** 2 + (node.y - rnd.y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i
        return best_idx

    def _steer(
        self, from_node: _Node, to_node: _Node, extend_length: float
    ) -> Optional[_Node]:
        """Create a new node at most extend_length from from_node toward to_node."""
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        if d == 0:
            return None

        scale = min(1.0, extend_length / d)
        new_node = _Node(
            from_node.x + dx * scale,
            from_node.y + dy * scale,
        )
        new_node.parent = from_node
        return new_node

    def _check_collision(self, node: _Node) -> bool:
        """Check if the edge from node.parent to node is collision-free."""
        if node.parent is None:
            return True
        return _segment_collision_free(
            self.grid,
            node.parent.x,
            node.parent.y,
            node.x,
            node.y,
            self.half_size_px,
            step=self.path_resolution,
        )

    def _find_near_nodes(self, new_node: _Node) -> List[int]:
        """Return indices of nodes within the rewiring ball."""
        n = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt(math.log(n) / max(1, n))
        r = min(r, self.expand_dis)

        near_inds: List[int] = []
        for i, node in enumerate(self.node_list):
            d2 = (node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2
            if d2 <= r * r:
                near_inds.append(i)
        return near_inds

    def _choose_parent(self, new_node: _Node, near_inds: List[int]) -> Optional[_Node]:
        """Reassign new_node's parent to the cheapest collision-free near node."""
        if not near_inds:
            return None

        best_cost = float("inf")
        best_node: Optional[_Node] = None

        for i in near_inds:
            near_node = self.node_list[i]
            candidate = self._steer(near_node, new_node, self.expand_dis)
            if candidate is None:
                continue
            candidate.cost = near_node.cost + math.hypot(
                candidate.x - near_node.x, candidate.y - near_node.y
            )
            if self._check_collision(candidate) and candidate.cost < best_cost:
                best_cost = candidate.cost
                best_node = candidate

        if best_node is None:
            # Fall back to the original connection (nearest node).
            return new_node if self._check_collision(new_node) else None

        return best_node

    def _rewire(self, new_node: _Node, near_inds: List[int]) -> None:
        """Rewire near nodes through new_node if it yields a cheaper path.

        This keeps each near node at its original position and only updates its
        parent pointer, which guarantees every existing edge stays collision-free.
        """
        for i in near_inds:
            near_node = self.node_list[i]
            if near_node is new_node:
                continue
            if near_node.parent is new_node:
                continue

            new_cost = new_node.cost + math.hypot(
                near_node.x - new_node.x, near_node.y - new_node.y
            )
            if new_cost >= near_node.cost:
                continue

            if not _segment_collision_free(
                self.grid,
                new_node.x,
                new_node.y,
                near_node.x,
                near_node.y,
                self.half_size_px,
                step=self.path_resolution,
            ):
                continue

            near_node.parent = new_node
            near_node.cost = new_cost
            self._propagate_cost_to_leaves(near_node)

    def _propagate_cost_to_leaves(self, parent_node: _Node) -> None:
        """Iteratively update costs of all descendants after a rewire."""
        stack = [parent_node]
        while stack:
            current = stack.pop()
            for node in self.node_list:
                if node.parent is current:
                    node.cost = current.cost + math.hypot(
                        node.x - current.x, node.y - current.y
                    )
                    stack.append(node)

    def _search_best_goal_node(self) -> Optional[int]:
        """Find the lowest-cost node that can directly connect to the goal."""
        best_cost = float("inf")
        best_idx: Optional[int] = None

        for i, node in enumerate(self.node_list):
            d = math.hypot(self.goal.x - node.x, self.goal.y - node.y)
            if d > self.expand_dis:
                continue
            if not _segment_collision_free(
                self.grid,
                node.x,
                node.y,
                self.goal.x,
                self.goal.y,
                self.half_size_px,
                step=self.path_resolution,
            ):
                continue
            cost = node.cost + d
            if cost < best_cost:
                best_cost = cost
                best_idx = i

        return best_idx

    def _generate_final_course(self, goal_ind: int) -> Optional[List[Tuple[float, float]]]:
        """Backtrack from goal-connected node to start, producing waypoints.

        Returns None if any backtracked edge is no longer collision-free.
        """
        path: List[Tuple[float, float]] = [(self.goal.x, self.goal.y)]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            if not _segment_collision_free(
                self.grid,
                node.parent.x,
                node.parent.y,
                node.x,
                node.y,
                self.half_size_px,
                step=self.path_resolution,
            ):
                return None
            path.append((node.x, node.y))
            node = node.parent
        path.append((self.start.x, self.start.y))
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# Public function — matches A* plan_path signature
# ---------------------------------------------------------------------------


def plan_path(
    grid: OccupancyGrid,
    start_world: Tuple[float, float],
    goal_world: Tuple[float, float],
    footprint: List[Tuple[int, int]],
    max_iter: int = _DEFAULT_MAX_ITER,
    retry_attempts: int = 3,
) -> List[Tuple[float, float]]:
    """RRT* path planner with the same interface as astar.plan_path.

    Args:
        grid: Occupancy grid for collision checking.
        start_world: Start position in world coordinates (px).
        goal_world: Goal position in world coordinates (px).
        footprint: List of (dx, dy) grid-cell offsets (not used directly;
                   half-size is derived from it for collision).
        max_iter: Maximum RRT iterations per attempt.
        retry_attempts: How many independent RRT* runs to attempt with different
            random seeds before giving up.

    Returns:
        Ordered list of (x, y) waypoints, or empty list if no path found.
    """
    # Derive robot half-size from the footprint.
    # The footprint's max cell offset already includes the robot radius plus a
    # safety margin, so converting it back to world units gives the full
    # clearance the planner should respect.
    if footprint:
        max_offset = max(max(abs(dx), abs(dy)) for dx, dy in footprint)
        half_size_px = max_offset * grid.resolution
    else:
        half_size_px = 0.0

    # If the start or goal is slightly inside an inflated obstacle (common with
    # pose estimation error), nudge it to the nearest free point instead of
    # immediately failing.
    free_start = _find_free_nearby(grid, start_world[0], start_world[1], half_size_px)
    if free_start is None:
        logging.warning(
            "[rrt] Start %.1f,%.1f is in collision and no free point nearby.",
            start_world[0],
            start_world[1],
        )
        return []
    free_goal = _find_free_nearby(grid, goal_world[0], goal_world[1], half_size_px)
    if free_goal is None:
        logging.warning(
            "[rrt] Goal %.1f,%.1f is in collision and no free point nearby.",
            goal_world[0],
            goal_world[1],
        )
        return []

    # Bounding box of the grid in world coordinates.
    ox, oy = grid.origin_world
    min_x = ox
    min_y = oy
    max_x = ox + grid.cols * grid.resolution
    max_y = oy + grid.rows * grid.resolution

    for attempt in range(retry_attempts):
        # Deterministic but different seed for each retry so failures are
        # reproducible while still exploring the space.
        random.seed(42 + attempt)
        rrt = RRTStar(
            start=free_start,
            goal=free_goal,
            grid=grid,
            half_size_px=half_size_px,
            rand_area=(min_x, min_y, max_x, max_y),
            max_iter=max_iter,
        )
        path = rrt.plan()
        if path is not None:
            return path

    logging.warning(
        "[rrt] No path found after %d attempts from %.1f,%.1f to %.1f,%.1f",
        retry_attempts,
        free_start[0],
        free_start[1],
        free_goal[0],
        free_goal[1],
    )
    return []
