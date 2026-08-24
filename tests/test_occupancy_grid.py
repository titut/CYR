"""Tests for the boolean occupancy grid."""

from __future__ import annotations

import math

import pytest

from core.map_format import Wall
from simulation.occupancy_grid import OccupancyGrid, _bresenham_line


def _box_walls(size: float = 10.0) -> list[Wall]:
    return [
        Wall(0, 0, size, 0),
        Wall(size, 0, size, size),
        Wall(0, size, size, size),
        Wall(0, 0, 0, size),
    ]


def test_empty_walls_grid():
    g = OccupancyGrid.from_walls([], 0.25)
    assert g.cols == 400 and g.rows == 400
    assert g.origin_world == (0.0, 0.0)


def test_wall_rasterized():
    g = OccupancyGrid.from_walls(_box_walls(), 0.25)
    # A point on the south wall (y=0) must be occupied.
    gx, gy = g.world_to_grid(5.0, 0.0)
    assert g.is_occupied(gx, gy)
    # A point in the middle of the box must be free.
    gx, gy = g.world_to_grid(5.0, 5.0)
    assert not g.is_occupied(gx, gy)


def test_world_grid_round_trip():
    g = OccupancyGrid.from_walls(_box_walls(), 0.25)
    for x, y in [(5.0, 5.0), (1.2, 9.8), (7.7, 2.1)]:
        gx, gy = g.world_to_grid(x, y)
        wx, wy = g.grid_to_world(gx, gy)
        assert math.hypot(wx - x, wy - y) <= g.resolution


def test_out_of_bounds_is_occupied():
    g = OccupancyGrid.from_walls(_box_walls(), 0.25)
    assert g.is_occupied(-100, -100)
    assert g.is_occupied(9999, 9999)
    assert g.is_occupied(-1, 5)


def test_mark_circle():
    g = OccupancyGrid.from_walls(_box_walls(), 0.25)
    g.mark_circle(5.0, 5.0, 0.5)
    gx, gy = g.world_to_grid(5.0, 5.0)
    assert g.is_occupied(gx, gy)
    # Far away must remain free.
    gx, gy = g.world_to_grid(1.0, 1.0)
    assert not g.is_occupied(gx, gy)


def test_inflate():
    g = OccupancyGrid.from_walls(_box_walls(), 0.25)
    gx, gy = g.world_to_grid(5.0, 5.0)
    g.grid[gy, gx] = True
    inflated = g.inflate(2)
    # A cell 2 away is now occupied; 5 away is not.
    gx2, gy2 = g.world_to_grid(5.0 + 2 * 0.25, 5.0)
    assert inflated.is_occupied(gx2, gy2)
    gx5, gy5 = g.world_to_grid(5.0 + 5 * 0.25, 5.0)
    assert not inflated.is_occupied(gx5, gy5)


def test_cast_ray_hits_wall():
    g = OccupancyGrid.from_walls(_box_walls(), 0.25)
    # East from the centre of a 10 m box -> ~5 m to the east wall.
    d = g.cast_ray((5.0, 5.0), 0.0, 10.0)
    assert d is not None
    assert d == pytest.approx(5.0, abs=0.5)
    # North -> ~5 m too.
    d = g.cast_ray((5.0, 5.0), math.pi / 2, 10.0)
    assert d == pytest.approx(5.0, abs=0.5)


def test_cast_ray_no_hit_returns_none():
    g = OccupancyGrid.from_walls(_box_walls(), 0.25)
    # A short ray inside the box hits nothing.
    assert g.cast_ray((5.0, 5.0), 0.0, 0.5) is None


def test_bresenham_includes_endpoints():
    line = _bresenham_line(0, 0, 3, 0)
    assert line[0] == (0, 0)
    assert line[-1] == (3, 0)
    assert len(line) == 4
