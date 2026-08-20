"""Tests for the RRT* path planner (seeded for determinism)."""

from __future__ import annotations

import math

import pytest

from map_format import Wall
from navigation.footprint import make_footprint
from navigation.rrt import plan_path
from simulation.occupancy_grid import OccupancyGrid


def _box_grid() -> OccupancyGrid:
    return OccupancyGrid.from_walls(
        [
            Wall(0, 0, 10, 0),
            Wall(10, 0, 10, 10),
            Wall(0, 10, 10, 10),
            Wall(0, 0, 0, 10),
        ],
        0.25,
    )


FP = make_footprint(0.375, 0.25)


def test_rrt_finds_path_in_open_space():
    grid = _box_grid()
    path = plan_path(grid, (1.0, 1.0), (9.0, 9.0), FP)
    assert path, "RRT* should find a path in open space with the seeded RNG"
    # Starts at the start and ends at the goal.
    assert math.hypot(path[0][0] - 1.0, path[0][1] - 1.0) <= 1.0
    assert math.hypot(path[-1][0] - 9.0, path[-1][1] - 9.0) <= 1.0


def test_rrt_deterministic_with_seed():
    """Same seed -> same path (uses the global RNG seeded by conftest)."""
    grid = _box_grid()
    p1 = plan_path(grid, (1.0, 1.0), (9.0, 9.0), FP)
    p2 = plan_path(grid, (1.0, 1.0), (9.0, 9.0), FP)
    assert p1 == p2
