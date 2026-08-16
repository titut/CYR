"""Robot footprint utility shared by path planners (A*, RRT*).

Builds a list of grid-cell offsets representing the robot's body
for collision checking against an OccupancyGrid.
"""

from __future__ import annotations

import math
from typing import List, Tuple


def make_footprint(half_size: float, resolution: float) -> List[Tuple[int, int]]:
    """Build a list of (dx, dy) grid-cell offsets for a square footprint.

    The robot is a square of side ``2 * half_size`` (meters), but it rotates in
    place, so its maximum axis-aligned extent at any heading is the square's
    circumradius ``half_size * sqrt(2)`` (the corner-to-centre distance).  The
    footprint is inflated to that radius so a path that is clear for the
    footprint is clear for the robot at *any* heading.

    Using the half-side (``half_size``) instead is a bug: on 45° diagonal
    segments the robot's corners reach ``half_size * sqrt(2)`` from its centre,
    so an axis-aligned half-side footprint under-reserves clearance and the
    corners clip walls.

    ``resolution`` is the grid cell size in meters.  The footprint is
    pre-computed once for reuse at every path-planner collision check.
    """
    radius = half_size * math.sqrt(2.0)
    offsets: List[Tuple[int, int]] = []
    max_cell = int(math.ceil(radius / resolution))
    for dx in range(-max_cell, max_cell + 1):
        for dy in range(-max_cell, max_cell + 1):
            offsets.append((dx, dy))
    return offsets
