"""Tests for the navigator's probabilistic occupancy update (T-009)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.map_format import Wall
from navigation.navigator import closest_segment_index, update_occupancy_log_odds
from simulation.occupancy_grid import OccupancyGrid


def _box_grid() -> OccupancyGrid:
    return OccupancyGrid.from_walls(
        [
            Wall(0, 0, 10, 0),
            Wall(10, 0, 10, 10),
            Wall(0, 10, 10, 10),
            Wall(0, 0, 0, 10),
            Wall(8, 0, 8, 10),  # an extra wall east of the robot
        ],
        0.25,
    )


def _scan(blocked_angles=None, blocked_dist=2.0):
    """A full 360-ray scan from the origin facing east; optionally a cone of
    beams is blocked by an obstacle at ``blocked_dist``."""
    rays = []
    for i in range(360):
        a = math.radians(i)
        d = 10.0
        if -math.pi / 2 < a < math.pi / 2:
            d = (8.0 - 5.0) / math.cos(a) if abs(math.cos(a)) > 1e-9 else 10.0
        if blocked_angles is not None and abs(a) < blocked_angles:
            d = blocked_dist
        rays.append({"angle_rad": a, "distance_m": d})
    return rays


def _make_state():
    grid = _box_grid()
    lo = np.zeros((grid.rows, grid.cols), dtype=float)
    return grid, lo


def test_no_obstacle_means_no_occupied_cells():
    grid, lo = _make_state()
    occ = update_occupancy_log_odds(lo, grid, 5.0, 5.0, 0.0, _scan())
    assert int(occ.sum()) == 0  # wall hits are not re-marked as obstacles


def test_obstacle_detected_after_a_few_scans():
    grid, lo = _make_state()
    # A 1 m-radius obstacle east of the robot blocks a cone of beams at 2 m.
    cone = math.atan2(1.0, 2.0)
    for _ in range(3):
        occ = update_occupancy_log_odds(lo, grid, 5.0, 5.0, 0.0, _scan(blocked_angles=cone))
    assert int(occ.sum()) >= 1
    # The occupied cells sit where the obstacle is (~x=7, y=5).
    gx, gy = grid.world_to_grid(6.9, 5.0)
    assert lo[gy, gx] > 0.7


def test_free_space_is_cleared():
    grid, lo = _make_state()
    update_occupancy_log_odds(lo, grid, 5.0, 5.0, 0.0, _scan(blocked_angles=math.atan2(1.0, 2.0)))
    # A cell between the robot and the wall/obstacle is strongly free.
    gx, gy = grid.world_to_grid(6.0, 5.0)
    assert lo[gy, gx] < 0.0


def test_known_wall_not_marked():
    grid, lo = _make_state()
    update_occupancy_log_odds(lo, grid, 5.0, 5.0, 0.0, _scan())
    # The east wall at x=8 is already in the static grid; not re-marked.
    gx, gy = grid.world_to_grid(8.0, 5.0)
    assert not (lo[gy, gx] > 0.7)


def test_single_scan_ghost_below_threshold():
    grid, lo = _make_state()
    occ = update_occupancy_log_odds(lo, grid, 5.0, 5.0, 0.0, _scan(blocked_angles=0.02))
    assert int(occ.sum()) == 0  # a single inconsistent beam does not stick


def test_occupancy_decays_without_reobservation():
    grid, lo = _make_state()
    cone = math.atan2(1.0, 2.0)
    for _ in range(3):
        update_occupancy_log_odds(lo, grid, 5.0, 5.0, 0.0, _scan(blocked_angles=cone))
    gx, gy = grid.world_to_grid(6.9, 5.0)
    before = lo[gy, gx]
    assert before > 0.7
    # Once the obstacle is out of view (no rays at all), log-odds decay to 0.
    for _ in range(200):
        update_occupancy_log_odds(lo, grid, 5.0, 5.0, 0.0, [])
    assert lo[gy, gx] < 0.7


def test_closest_segment_index():
    path = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    assert closest_segment_index(path, 5.0, 1.0) == 0
    assert closest_segment_index(path, 11.0, 5.0) == 1
    assert closest_segment_index(path, 9.9, 9.9) == 1


def test_closest_segment_index_regression():
    """Regression: the robot parked at the start must land on segment 0 (the
    old waypoint heuristic jumped to segment 1 and skipped the dangerous
    first segment entirely)."""
    path = [(18.6, 8.6), (11.4, 13.9), (6.1, 15.9)]
    for x, y in [(18.6, 8.7), (18.0, 9.0), (17.61, 9.28), (16.0, 10.5), (14.0, 12.3)]:
        assert closest_segment_index(path, x, y) == 0
