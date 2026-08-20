"""Tests for the robot footprint (regression: the circumradius circle fix)."""

from __future__ import annotations

import math

import pytest

from navigation.footprint import make_footprint

# The footprint is a circle whose radius is this multiple of the half-size.
# >= sqrt(2) guarantees clearance for the rotating square's corners; the extra
# margin is a clearance tuning knob.
_FOOTPRINT_RADIUS_FACTOR = 1.6


def test_footprint_is_a_circle_not_a_box():
    """Regression: the old code filled the whole bounding box, whose corners
    reach max_cell*sqrt(2) — over-inflating passages.  Every cell must sit
    within the intended circle radius."""
    half_size = 0.375
    resolution = 0.25
    radius = half_size * _FOOTPRINT_RADIUS_FACTOR

    fp = make_footprint(half_size, resolution)
    assert fp
    for dx, dy in fp:
        assert math.hypot(dx, dy) * resolution <= radius + 1e-9


def test_footprint_covers_the_disc_axes():
    """The axis-aligned cells out to the radius are included (no gaps)."""
    half_size = 0.375
    resolution = 0.25
    radius = half_size * _FOOTPRINT_RADIUS_FACTOR
    fp = set(make_footprint(half_size, resolution))

    for dx, dy in [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2)]:
        assert math.hypot(dx, dy) * resolution <= radius
        assert (dx, dy) in fp, f"missing offset {(dx, dy)}"


def test_footprint_size_regression():
    """Exact cell count for the 0.75 m bot at 0.25 m resolution."""
    fp = make_footprint(0.375, 0.25)
    # Circle of radius 0.6 m at 0.25 m cells = 21 cells (was 49 for the box).
    assert len(fp) == 21
    assert max(max(abs(dx), abs(dy)) for dx, dy in fp) == 2


def test_footprint_scales_with_resolution():
    half_size = 0.375
    radius = half_size * _FOOTPRINT_RADIUS_FACTOR
    fp = make_footprint(half_size, 0.1)
    assert all(math.hypot(dx, dy) * 0.1 <= radius + 1e-9 for dx, dy in fp)


def test_footprint_grows_with_size():
    small = make_footprint(0.375, 0.25)
    big = make_footprint(1.0, 0.25)
    assert len(big) > len(small)
