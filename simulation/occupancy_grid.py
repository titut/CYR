"""Fast occupancy grid with DDA ray casting for particle-filter updates.

This is a performance layer: the simulator still uses exact segment-based ray
casting for visualization, while the particle filter uses the grid for speed.

The grid auto-detects the bounding box of wall geometry so that it works
correctly even after the map editor rebases coordinates relative to an origin.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from map_format import Wall


def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """Return integer grid cells along a line."""
    cells: List[Tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return cells


class OccupancyGrid:
    """A 2D boolean occupancy grid.

    Cells are ``True`` if occupied and ``False`` if free. Out-of-bounds is
    treated as occupied so that rays leaving the map stop there.

    The grid uses a world-to-grid offset so that the grid itself starts at
    index (0, 0) regardless of whether world coordinates start at a negative
    or positive minimum.  ``origin_world`` is the world-coordinate position
    of grid cell (0, 0).
    """

    def __init__(
        self,
        min_x: float,
        min_y: float,
        width_px: float,
        height_px: float,
        resolution_px: float,
    ):
        self.resolution = resolution_px
        self.cols = int(math.ceil(width_px / resolution_px))
        self.rows = int(math.ceil(height_px / resolution_px))
        self.grid = np.zeros((self.rows, self.cols), dtype=bool)

        # World position of grid cell (0, 0).
        # world_to_grid(origin_world.x, origin_world.y) → (0, 0)
        self.origin_world = (min_x, min_y)

    @classmethod
    def from_walls(
        cls,
        walls: Sequence[Wall],
        resolution_px: float = 5.0,
        margin_px: float = 20.0,
    ) -> "OccupancyGrid":
        """Build an occupancy grid by rasterizing wall segments.

        The grid bounds are computed automatically from the wall geometry so
        that coordinates may be negative (after origin rebasing) without issue.
        A small margin is added to prevent paths from hugging the grid edge.
        """
        if not walls:
            # Trivial empty grid.
            grid = cls(0.0, 0.0, 100.0, 100.0, resolution_px)
            return grid

        # Compute the bounding box of all wall vertices.
        min_x = min(min(w.x1, w.x2) for w in walls)
        max_x = max(max(w.x1, w.x2) for w in walls)
        min_y = min(min(w.y1, w.y2) for w in walls)
        max_y = max(max(w.y1, w.y2) for w in walls)

        width = (max_x - min_x) + 2.0 * margin_px
        height = (max_y - min_y) + 2.0 * margin_px

        grid = cls(min_x - margin_px, min_y - margin_px, width, height, resolution_px)

        for wall in walls:
            x0, y0 = grid.world_to_grid(wall.x1, wall.y1)
            x1, y1 = grid.world_to_grid(wall.x2, wall.y2)
            for gx, gy in _bresenham_line(x0, y0, x1, y1):
                if 0 <= gx < grid.cols and 0 <= gy < grid.rows:
                    grid.grid[gy, gx] = True

        return grid

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        ox, oy = self.origin_world
        return (int((x - ox) / self.resolution), int((y - oy) / self.resolution))

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Return the centre of a grid cell in world coordinates."""
        ox, oy = self.origin_world
        return (
            ox + (gx + 0.5) * self.resolution,
            oy + (gy + 0.5) * self.resolution,
        )

    def is_occupied(self, gx: int, gy: int) -> bool:
        if 0 <= gx < self.cols and 0 <= gy < self.rows:
            return self.grid[gy, gx]
        return True

    def mark_circle(self, cx: float, cy: float, radius: float) -> None:
        """Mark a circular region as occupied (e.g., a detected obstacle).

        This is used by the navigator to add dynamically-detected obstacles
        to the grid so subsequent plans route around them.
        """
        gcx, gcy = self.world_to_grid(cx, cy)
        r_cells = max(1, int(math.ceil(radius / self.resolution)))
        for gy in range(gcy - r_cells, gcy + r_cells + 1):
            for gx in range(gcx - r_cells, gcx + r_cells + 1):
                if 0 <= gx < self.cols and 0 <= gy < self.rows:
                    dx = gx - gcx
                    dy = gy - gcy
                    if dx * dx + dy * dy <= r_cells * r_cells:
                        self.grid[gy, gx] = True

    def inflate(self, radius_cells: int) -> "OccupancyGrid":
        """Return a new grid where each occupied cell is dilated by *radius_cells*.

        This grows obstacles so that an A* path leaves enough clearance for a
        robot of the corresponding radius.
        """
        ox, oy = self.origin_world
        dilated = OccupancyGrid(
            ox,
            oy,
            self.cols * self.resolution,
            self.rows * self.resolution,
            self.resolution,
        )
        dilated.grid = np.array(self.grid)  # copy
        if radius_cells <= 0:
            return dilated

        # Find all occupied cells.
        occ_y, occ_x = np.where(dilated.grid)
        # For each occupied cell, fill a square of side (2*radius_cells + 1).
        for cx, cy in zip(occ_x, occ_y):
            dx = radius_cells
            x0 = max(0, cx - dx)
            x1 = min(dilated.cols - 1, cx + dx)
            y0 = max(0, cy - dx)
            y1 = min(dilated.rows - 1, cy + dx)
            dilated.grid[y0 : y1 + 1, x0 : x1 + 1] = True

        return dilated

    def cast_ray(
        self,
        origin: Tuple[float, float],
        direction: float,
        max_range: float,
    ) -> Optional[float]:
        """Return distance to the nearest occupied cell, or None if no hit.

        Uses a 2D DDA traversal. Distances are in the same units as the grid
        resolution (usually map pixels).
        """
        ox, oy = origin

        # Shift origin relative to grid offset.
        ox_rel = ox - self.origin_world[0]
        oy_rel = oy - self.origin_world[1]

        dx = math.cos(direction)
        dy = math.sin(direction)

        step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
        step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)

        if step_x == 0:
            t_delta_x = float("inf")
            t_max_x = float("inf")
        else:
            t_delta_x = self.resolution / abs(dx)
            gx = int(ox_rel / self.resolution)
            if dx > 0:
                t_max_x = ((gx + 1) * self.resolution - ox_rel) / dx
            else:
                t_max_x = (gx * self.resolution - ox_rel) / dx

        if step_y == 0:
            t_delta_y = float("inf")
            t_max_y = float("inf")
        else:
            t_delta_y = self.resolution / abs(dy)
            gy = int(oy_rel / self.resolution)
            if dy > 0:
                t_max_y = ((gy + 1) * self.resolution - oy_rel) / dy
            else:
                t_max_y = (gy * self.resolution - oy_rel) / dy

        gx = int(ox_rel / self.resolution)
        gy = int(oy_rel / self.resolution)

        # Traverse. The first cell checked is the one adjacent to the origin.
        distance = 0.0
        while distance < max_range:
            if t_max_x < t_max_y:
                distance = t_max_x
                t_max_x += t_delta_x
                gx += step_x
            else:
                distance = t_max_y
                t_max_y += t_delta_y
                gy += step_y

            if self.is_occupied(gx, gy):
                return distance

        return None
