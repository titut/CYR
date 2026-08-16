"""A* pathfinder on an OccupancyGrid with robot-footprint collision checking.

Uses 8-connected grid expansion (with corner-cut prevention) and an octile
heuristic, then smooths the resulting path with greedy line-of-sight
shortcutting so it has no stair-step zigzags.
"""

from __future__ import annotations

import heapq
import math
import itertools
import sys
from pathlib import Path
from typing import List, Set, Tuple

# Allow running this file directly or as a module.
_SCRIPT_DIR = Path(__file__).resolve().parent
_ZENOH_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _ZENOH_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_ZENOH_DIR) not in sys.path:
    sys.path.insert(0, str(_ZENOH_DIR))

import numpy as np

from simulation.occupancy_grid import OccupancyGrid  # stays in simulation
from navigation.footprint import make_footprint

_DIAG_COST = math.sqrt(2.0)

# 8-connected neighbour offsets: (dx, dy, cost).  Cardinal moves cost 1, diagonal
# moves cost √2.
_NEIGHBOURS = [
    (-1, 0, 1.0),  # W
    (1, 0, 1.0),  # E
    (0, -1, 1.0),  # N
    (0, 1, 1.0),  # S
    (-1, -1, _DIAG_COST),  # NW
    (1, -1, _DIAG_COST),  # NE
    (-1, 1, _DIAG_COST),  # SW
    (1, 1, _DIAG_COST),  # SE
]


def _heuristic(ax: int, ay: int, bx: int, by: int) -> float:
    """Octile distance: admissible and consistent for 8-connected grids."""
    dx = abs(ax - bx)
    dy = abs(ay - by)
    return max(dx, dy) + (_DIAG_COST - 1.0) * min(dx, dy)


def _footprint_clear(
    grid: OccupancyGrid,
    gx: int,
    gy: int,
    footprint: List[Tuple[int, int]],
) -> bool:
    """True if every cell in the robot's footprint is free and in-bounds."""
    for dx, dy in footprint:
        if grid.is_occupied(gx + dx, gy + dy):
            return False
    return True


def _find_free_nearby(
    grid: OccupancyGrid,
    x: float,
    y: float,
    footprint: List[Tuple[int, int]],
    search_radius: float = 1.25,
) -> Optional[Tuple[float, float]]:
    """Return a collision-free point near (x, y), or None if none exists.

    If (x, y) is already free, returns it unchanged.  Otherwise searches the
    cells within ``search_radius`` (meters) for the closest cell whose robot
    footprint is free.  This keeps pose-estimation error from making A* reject
    a slightly-in-collision start or goal outright.
    """
    gx, gy = grid.world_to_grid(x, y)
    if _footprint_clear(grid, gx, gy, footprint):
        return (x, y)

    radius_cells = int(math.ceil(search_radius / grid.resolution))
    best_d = float("inf")
    best: Optional[Tuple[float, float]] = None
    for dgx in range(-radius_cells, radius_cells + 1):
        for dgy in range(-radius_cells, radius_cells + 1):
            nx, ny = gx + dgx, gy + dgy
            if not (0 <= nx < grid.cols and 0 <= ny < grid.rows):
                continue
            if _footprint_clear(grid, nx, ny, footprint):
                wx, wy = grid.grid_to_world(nx, ny)
                d = math.hypot(wx - x, wy - y)
                if d < best_d:
                    best_d = d
                    best = (wx, wy)
    return best


def _line_collision_free(
    grid: OccupancyGrid,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    footprint: List[Tuple[int, int]],
) -> bool:
    """True if the straight world line (x1,y1)→(x2,y2) is collision-free.

    Samples the segment at half the grid resolution and checks the full robot
    footprint at each sample, so the swept corridor is respected.
    """
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        gx, gy = grid.world_to_grid(x1, y1)
        return _footprint_clear(grid, gx, gy, footprint)

    step = grid.resolution / 2.0
    steps = max(1, int(math.ceil(dist / step)))
    for i in range(steps + 1):
        t = i / steps
        x = x1 + dx * t
        y = y1 + dy * t
        gx, gy = grid.world_to_grid(x, y)
        if not _footprint_clear(grid, gx, gy, footprint):
            return False
    return True


def _smooth_path(
    grid: OccupancyGrid,
    path_world: List[Tuple[float, float]],
    footprint: List[Tuple[int, int]],
) -> List[Tuple[float, float]]:
    """Greedy line-of-sight shortcutting to remove stair-step zigzags.

    From each waypoint, jump to the farthest later waypoint that is reachable
    by a collision-free straight line.  Produces a shorter, smoother path with
    no 90°-staircase artifacts.
    """
    if len(path_world) < 3:
        return path_world

    smoothed: List[Tuple[float, float]] = [path_world[0]]
    i = 0
    while i < len(path_world) - 1:
        j = len(path_world) - 1
        while j > i + 1:
            if _line_collision_free(
                grid,
                path_world[i][0],
                path_world[i][1],
                path_world[j][0],
                path_world[j][1],
                footprint,
            ):
                break
            j -= 1
        smoothed.append(path_world[j])
        i = j
    return smoothed


def plan_path(
    grid: OccupancyGrid,
    start_world: Tuple[float, float],
    goal_world: Tuple[float, float],
    footprint: List[Tuple[int, int]],
) -> List[Tuple[float, float]]:
    """A* on the occupancy grid with full-footprint collision checking.

    Each grid cell is only considered traversable if every cell in the
    robot's footprint (relative offsets) is free.  This ensures the A*
    path naturally stays centred in corridors and passages.

    Args:
        grid: The occupancy grid (walls = occupied cells).
        start_world: Start position in world coordinates (meters).
        goal_world: Goal position in world coordinates (meters).
        footprint: List of (dx, dy) grid-cell offsets for the robot body.

    Returns:
        Ordered list of (x, y) waypoints from start to goal, or an empty list
        if no path exists.
    """
    # Nudge slightly-in-collision endpoints (common with pose estimation error)
    # to the nearest free point instead of failing immediately.
    free_start = _find_free_nearby(grid, start_world[0], start_world[1], footprint)
    if free_start is None:
        return []
    free_goal = _find_free_nearby(grid, goal_world[0], goal_world[1], footprint)
    if free_goal is None:
        return []

    sx, sy = grid.world_to_grid(*free_start)
    gx, gy = grid.world_to_grid(*free_goal)

    open_set: List[Tuple[float, int, int]] = []
    heapq.heappush(open_set, (_heuristic(sx, sy, gx, gy), sx, sy))

    came_from: dict = {}
    g_score = np.full((grid.rows, grid.cols), np.inf)
    g_score[sy, sx] = 0.0

    while open_set:
        _, cx, cy = heapq.heappop(open_set)
        if cx == gx and cy == gy:
            # Reconstruct path.
            path_grid = [(cx, cy)]
            while (cx, cy) in came_from:
                cx, cy = came_from[(cx, cy)]
                path_grid.append((cx, cy))
            path_grid.reverse()
            path_world = [grid.grid_to_world(gx, gy) for gx, gy in path_grid]
            return _smooth_path(grid, path_world, footprint)

        for dx, dy, cost in _NEIGHBOURS:
            nx, ny = cx + dx, cy + dy

            # Full-footprint check: the entire robot body must fit.
            if not _footprint_clear(grid, nx, ny, footprint):
                continue

            # Don't cut corners on diagonal moves: the two cells the robot
            # sweeps through must also be clear.
            if dx != 0 and dy != 0:
                if not _footprint_clear(grid, cx + dx, cy, footprint):
                    continue
                if not _footprint_clear(grid, cx, cy + dy, footprint):
                    continue

            tentative_g = g_score[cy, cx] + cost
            if tentative_g < g_score[ny, nx]:
                came_from[(nx, ny)] = (cx, cy)
                g_score[ny, nx] = tentative_g
                f = tentative_g + _heuristic(nx, ny, gx, gy)
                heapq.heappush(open_set, (f, nx, ny))

    return []
