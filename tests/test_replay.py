"""Deterministic replay + recorded-failure regression tests (T-026, T-020).

The recorded-session tests skip cleanly when the log files are not present
(e.g. a fresh clone), so CI does not depend on them.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest

from clock import SimClock, seed_all, sleep_until
from map_format import MapData
from navigation.navigator import closest_segment_index, update_occupancy_log_odds
from simulation.occupancy_grid import OccupancyGrid

_FAILURE_LOG = Path("zenoh/logs/recording_20260818_180323.jsonl")
_GOAL2_PATH = [(18.6, 8.6), (11.4, 13.9), (6.1, 15.9)]


# ---------------------------------------------------------------------------
# Clock (minimal T-020)
# ---------------------------------------------------------------------------


def test_simclock_is_deterministic():
    a, b = SimClock(start=1.0, step=0.25), SimClock(start=1.0, step=0.25)
    for _ in range(10):
        assert a.tick() == b.tick()
        assert a.monotonic() == b.monotonic()
    assert a.monotonic() == pytest.approx(1.0 + 10 * 0.25)


def test_simclock_advance():
    c = SimClock()
    assert c.advance(1.5) == pytest.approx(1.5)
    assert c.advance(0.5) == pytest.approx(2.0)


def test_seed_all_makes_rng_reproducible():
    seed_all(7)
    r1 = np.random.rand(5)
    seed_all(7)
    r2 = np.random.rand(5)
    assert (r1 == r2).all()


def test_sleep_until_past_deadline_returns_immediately():
    t0 = time.monotonic()
    sleep_until(t0 - 5.0)
    assert time.monotonic() - t0 < 0.05


def test_sleep_until_waits_for_future_deadline():
    t0 = time.monotonic()
    sleep_until(t0 + 0.05)
    assert time.monotonic() - t0 >= 0.05


# ---------------------------------------------------------------------------
# Deterministic occupancy trace
# ---------------------------------------------------------------------------


def _box_grid() -> OccupancyGrid:
    from map_format import Wall

    return OccupancyGrid.from_walls(
        [
            Wall(0, 0, 10, 0),
            Wall(10, 0, 10, 10),
            Wall(0, 10, 10, 10),
            Wall(0, 0, 0, 10),
            Wall(8, 0, 8, 10),
        ],
        0.25,
    )


def _scan_trace(seed):
    """A small deterministic scan stream (cone obstacle at varying positions)."""
    rng = np.random.default_rng(seed)
    grid = _box_grid()
    lo = np.zeros((grid.rows, grid.cols), dtype=float)
    occ_sums = []
    for _ in range(8):
        cone = 0.3 + 0.05 * float(rng.normal())
        rays = []
        for i in range(360):
            a = math.radians(i)
            d = 10.0
            if -math.pi / 2 < a < math.pi / 2:
                d = (8.0 - 5.0) / math.cos(a) if abs(math.cos(a)) > 1e-9 else 10.0
            if abs(a) < cone:
                d = 2.0
            rays.append({"angle_rad": a, "distance_m": d})
        occ = update_occupancy_log_odds(lo, grid, 5.0, 5.0, 0.0, rays)
        occ_sums.append(int(occ.sum()))
    return occ_sums


def test_occupancy_trace_is_reproducible():
    t1 = _scan_trace(seed=42)
    t2 = _scan_trace(seed=42)
    assert t1 == t2


def test_occupancy_trace_differs_with_seed():
    t1 = _scan_trace(seed=42)
    t2 = _scan_trace(seed=43)
    assert t1 != t2


# ---------------------------------------------------------------------------
# Recorded-failure regression (T-009): the path into obstacle (16,9)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def failure_session():
    if not _FAILURE_LOG.exists():
        pytest.skip(f"recorded session {_FAILURE_LOG} not present")

    map_data = MapData.from_json("home.json")
    grid = OccupancyGrid.from_walls(map_data.walls, 0.25)
    lo = np.zeros((grid.rows, grid.cols), dtype=float)

    pose = (0.0, 0.0, 0.0)
    t0 = None
    scans = 0
    with open(_FAILURE_LOG) as f:
        for line in f:
            rec = json.loads(line)
            t = rec["t_mono"]
            if t0 is None:
                t0 = t
            if rec["topic"] == "estimate/pose":
                d = json.loads(rec["payload"])
                pose = (d["x_m"], d["y_m"], d["theta_rad"])
            elif rec["topic"] == "sensor/lidar":
                if t - t0 < 21.1:
                    continue
                scan = json.loads(rec["payload"])
                update_occupancy_log_odds(
                    lo, grid, pose[0], pose[1], pose[2], scan.get("rays", [])
                )
                scans += 1
    return grid, lo, scans


def test_recorded_obstacle_detected(failure_session):
    grid, lo, scans = failure_session
    assert scans > 0
    occ = lo > 0.7
    # Some occupied cell must land within the obstacle (16,9) (r=1) footprint.
    gys, gxs = np.where(occ)
    hits = 0
    for gy, gx in zip(gys, gxs):
        wx, wy = grid.grid_to_world(gx, gy)
        if math.hypot(wx - 16.0, wy - 9.0) < 1.6:
            hits += 1
    assert hits > 0, "obstacle (16,9) was not detected in the replay"


def test_recorded_path_flagged_blocked(failure_session):
    grid, lo, _ = failure_session
    occ = lo > 0.7
    points = [grid.grid_to_world(gx, gy) for gy, gx in zip(*np.where(occ))]

    clearance = 0.375 * 1.5
    # Segment 0 of the goal-2 path is the dangerous one hugging obstacle (16,9).
    x1, y1 = _GOAL2_PATH[0]
    x2, y2 = _GOAL2_PATH[1]
    blocked = any(
        _seg_dist(cx, cy, x1, y1, x2, y2) < clearance for cx, cy in points
    )
    assert blocked, "the path hugging (16,9) should have been flagged blocked"


def _seg_dist(x, y, x1, y1, x2, y2):
    wx, wy = x2 - x1, y2 - y1
    ls = wx * wx + wy * wy
    t = 0.0 if ls == 0 else max(0.0, min(1.0, ((x - x1) * wx + (y - y1) * wy) / ls))
    return math.hypot(x - (x1 + t * wx), y - (y1 + t * wy))
