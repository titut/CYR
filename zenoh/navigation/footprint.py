"""Robot footprint utility shared by path planners (A*, RRT*).

Builds a list of grid-cell offsets representing the robot's body
for collision checking against an OccupancyGrid.
"""

from __future__ import annotations

import math
from typing import List, Tuple


def make_footprint(half_size_px: float, resolution_px: float) -> List[Tuple[int, int]]:
    """Build a list of (dx, dy) grid-cell offsets for a square footprint.

    The robot is a square of side 2 × half_size_px centred at the origin.
    A 20 % safety margin is added so the path does not scrape walls.
    A grid cell at offset (dx, dy) is included if its world-coordinate
    rectangle overlaps the (safety-margined) robot bounding box.

    This is pre-computed once for reuse at every path-planner collision check.
    """
    margin = half_size_px * 1.20
    offsets: List[Tuple[int, int]] = []
    max_cell = int(math.ceil(margin / resolution_px))
    for dx in range(-max_cell, max_cell + 1):
        for dy in range(-max_cell, max_cell + 1):
            cell_x0 = dx * resolution_px
            cell_x1 = (dx + 1) * resolution_px
            cell_y0 = dy * resolution_px
            cell_y1 = (dy + 1) * resolution_px
            if (
                cell_x0 < margin
                and cell_x1 > -margin
                and cell_y0 < margin
                and cell_y1 > -margin
            ):
                offsets.append((dx, dy))
    return offsets
