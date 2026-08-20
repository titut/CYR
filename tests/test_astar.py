"""Tests for the A* path planner."""

from __future__ import annotations

import math

import pytest

from map_format import Wall
from navigation.astar import plan_path
from navigation.footprint import make_footprint
from simulation.occupancy_grid import OccupancyGrid


def _box_grid(extra_walls=()) -> OccupancyGrid:
    walls = [
        Wall(0, 0, 10, 0),
        Wall(10, 0, 10, 10),
        Wall(0, 10, 10, 10),
        Wall(0, 0, 0, 10),
    ]
    walls.extend(extra_walls)
    return OccupancyGrid.from_walls(walls, 0.25)


def _assert_path_clear(grid, path, footprint):
    for wx, wy in path:
        gx, gy = grid.world_to_grid(wx, wy)
        for dx, dy in footprint:
            assert not grid.is_occupied(gx + dx, gy + dy), (
                f"path waypoint ({wx:.2f},{wy:.2f}) footprint hits an obstacle"
            )


FP = make_footprint(0.375, 0.25)


def test_path_found_in_open_space():
    grid = _box_grid()
    path = plan_path(grid, (1.0, 1.0), (9.0, 9.0), FP)
    assert path
    _assert_path_clear(grid, path, FP)
    # Ends at the goal (within a cell).
    assert math.hypot(path[-1][0] - 9.0, path[-1][1] - 9.0) <= grid.resolution


def test_path_avoids_internal_wall():
    wall = Wall(5.0, 2.0, 5.0, 8.0)
    grid = _box_grid([wall])
    path = plan_path(grid, (1.0, 1.0), (9.0, 9.0), FP)
    assert path
    _assert_path_clear(grid, path, FP)


def test_no_path_when_fully_blocked():
    # A wall spanning the full height splits the box in two.
    grid = _box_grid([Wall(0.0, 5.0, 10.0, 5.0)])
    path = plan_path(grid, (1.0, 1.0), (9.0, 9.0), FP)
    assert path == []


def test_narrow_gap_blocked_by_footprint():
    # Two rooms joined only by a 0.2 m gap at x~4.5 in a y=5 wall: too narrow
    # for the 0.6 m-radius footprint, so no path crosses between the rooms.
    grid = _box_grid([Wall(0.0, 5.0, 4.4, 5.0), Wall(4.6, 5.0, 10.0, 5.0)])
    path = plan_path(grid, (1.0, 2.0), (9.0, 8.0), FP)
    assert path == []


def test_wide_gap_allows_crossing():
    # A 2 m gap in the same wall is easily wide enough for the footprint.
    grid = _box_grid([Wall(0.0, 5.0, 4.0, 5.0), Wall(6.0, 5.0, 10.0, 5.0)])
    path = plan_path(grid, (1.0, 2.0), (9.0, 8.0), FP)
    assert path
    _assert_path_clear(grid, path, FP)


def test_start_nudged_out_of_collision():
    # Start right up against the west wall: A* nudges it to free space.
    grid = _box_grid()
    path = plan_path(grid, (0.05, 5.0), (9.0, 5.0), FP)
    assert path
    _assert_path_clear(grid, path, FP)


def test_goal_reachable_after_nudge():
    grid = _box_grid()
    path = plan_path(grid, (1.0, 1.0), (9.95, 9.0), FP)
    assert path
    _assert_path_clear(grid, path, FP)
