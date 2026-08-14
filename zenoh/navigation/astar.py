"""A* pathfinder on an OccupancyGrid with robot-footprint collision checking."""

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

# 4-connected neighbour offsets: (dx, dy, cost)
_NEIGHBOURS = [
    (-1, 0, 1.0),  # W
    (1, 0, 1.0),  # E
    (0, -1, 1.0),  # N
    (0, 1, 1.0),  # S
]


def _heuristic(ax: int, ay: int, bx: int, by: int) -> float:
    """Manhattan distance for 4-connected grids."""
    return float(abs(ax - bx) + abs(ay - by))


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
        start_world: Start position in world coordinates (pixels).
        goal_world: Goal position in world coordinates (pixels).
        footprint: List of (dx, dy) grid-cell offsets for the robot body.

    Returns:
        Ordered list of (x, y) waypoints from start to goal, or an empty list
        if no path exists.
    """
    sx, sy = grid.world_to_grid(*start_world)
    gx, gy = grid.world_to_grid(*goal_world)

    if not _footprint_clear(grid, sx, sy, footprint):
        return []
    if not _footprint_clear(grid, gx, gy, footprint):
        return []

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
            return [grid.grid_to_world(px, py) for px, py in path_grid]

        for dx, dy, cost in _NEIGHBOURS:
            nx, ny = cx + dx, cy + dy

            # Full-footprint check: the entire robot body must fit.
            if not _footprint_clear(grid, nx, ny, footprint):
                continue

            tentative_g = g_score[cy, cx] + cost
            if tentative_g < g_score[ny, nx]:
                came_from[(nx, ny)] = (cx, cy)
                g_score[ny, nx] = tentative_g
                f = tentative_g + _heuristic(nx, ny, gx, gy)
                heapq.heappush(open_set, (f, nx, ny))

    return []
