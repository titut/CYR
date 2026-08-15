"""Robot footprint utility shared by path planners (A*, RRT*).

Builds a list of grid-cell offsets representing the robot's body
for collision checking against an OccupancyGrid.
"""

from __future__ import annotations

import math
from typing import List, Tuple


def make_footprint(half_size: float, resolution: float) -> List[Tuple[int, int]]:
    """Build a list of (dx, dy) grid-cell offsets for a square footprint.

    The robot is a square of side 2 × half_size (meters) centred at the origin.
    A 10 % safety margin is added so the path does not scrape walls.  A grid
    cell at offset (dx, dy) is included if its centre lies within the
    (safety-margined) robot half-size, which keeps the footprint symmetric
    about the robot centre.
    ``resolution`` is the grid cell size in meters.

    This is pre-computed once for reuse at every path-planner collision check.
    """
    margin = half_size * 1.35
    offsets: List[Tuple[int, int]] = []
    max_cell = max(0, int(math.ceil(margin / resolution - 0.5)))
    for dx in range(-max_cell, max_cell + 1):
        for dy in range(-max_cell, max_cell + 1):
            if abs(dx) * resolution <= margin and abs(dy) * resolution <= margin:
                offsets.append((dx, dy))
    return offsets
