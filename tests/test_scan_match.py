"""Tests for the scan-matching refinement (discrete coordinate descent + ICP)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.map_format import Wall
from pose_estimation.scan_match import (
    refine_discrete,
    refine_gauss_newton,
    scan_match_error,
)
from simulation.raycast import cast_ray


def _box_walls():
    # A 10x6 room (non-square so the scan uniquely identifies the pose).
    return [
        Wall(0.0, 0.0, 10.0, 0.0),
        Wall(10.0, 0.0, 10.0, 6.0),
        Wall(10.0, 6.0, 0.0, 6.0),
        Wall(0.0, 6.0, 0.0, 0.0),
    ]


def _synthetic_scan(pose, walls, n=360, max_range=10.0):
    """A 360-ray scan as the sim produces it: relative angles + distances, with
    "no return" rays clamped to max_range."""
    x, y, th = pose
    angles = np.linspace(-math.pi, math.pi, n, endpoint=False)
    observed = np.empty(n, dtype=np.float64)
    for i, a in enumerate(angles):
        res = cast_ray((x, y), th + a, walls, max_range)
        observed[i] = res[0] if res is not None else max_range
    return angles, observed


def _heading_diff(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def test_cost_is_zero_at_true_pose():
    walls = _box_walls()
    true = (4.0, 3.0, 0.3)
    angles, observed = _synthetic_scan(true, walls)
    assert scan_match_error(true, walls, angles, observed, 10.0) == pytest.approx(0.0, abs=1e-6)


def test_refine_discrete_recovers_perturbed_pose():
    walls = _box_walls()
    true = (4.0, 3.0, 0.3)
    angles, observed = _synthetic_scan(true, walls)
    start = (4.3, 2.7, 0.15)
    out = refine_discrete(start, walls, angles, observed, 10.0)
    assert math.hypot(out[0] - true[0], out[1] - true[1]) < 0.05
    assert _heading_diff(out[2], true[2]) < 0.05


def test_refine_gauss_newton_recovers_perturbed_pose():
    walls = _box_walls()
    true = (4.0, 3.0, 0.3)
    angles, observed = _synthetic_scan(true, walls)
    start = (4.3, 2.7, 0.15)
    out = refine_gauss_newton(start, walls, angles, observed, 10.0)
    assert math.hypot(out[0] - true[0], out[1] - true[1]) < 0.05
    assert _heading_diff(out[2], true[2]) < 0.05


def test_refiners_agree():
    walls = _box_walls()
    true = (5.0, 2.0, -0.6)
    angles, observed = _synthetic_scan(true, walls)
    start = (4.6, 2.4, -0.9)
    d = refine_discrete(start, walls, angles, observed, 10.0)
    g = refine_gauss_newton(start, walls, angles, observed, 10.0)
    assert math.hypot(d[0] - g[0], d[1] - g[1]) < 0.05
    assert _heading_diff(d[2], g[2]) < 0.05


def test_no_op_at_optimum():
    walls = _box_walls()
    true = (4.0, 3.0, 0.3)
    angles, observed = _synthetic_scan(true, walls)
    for refine in (refine_discrete, refine_gauss_newton):
        out = refine(true, walls, angles, observed, 10.0)
        assert math.hypot(out[0] - true[0], out[1] - true[1]) < 0.01
