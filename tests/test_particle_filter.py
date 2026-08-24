"""Tests for the particle filter: likelihood, convergence, covariance, anchors."""

from __future__ import annotations

import math

import pytest

from core.map_format import MapData, Room, Wall
from pose_estimation.particle_filter import ParticleFilter
from simulation.raycast import cast_rays


def _box_map() -> MapData:
    # Rectangular (non-square) room so the LIDAR scan uniquely identifies the
    # heading: a square is 90°-rotationally symmetric, which makes the heading
    # weakly observable.
    walls = [
        Wall(0, 0, 10, 0),
        Wall(10, 0, 10, 6),
        Wall(0, 6, 10, 6),
        Wall(0, 0, 0, 6),
    ]
    room = Room(
        id="room",
        name="room",
        polygon=[(0.1, 0.1), (9.9, 0.1), (9.9, 5.9), (0.1, 5.9)],
        center=(5.0, 3.0),
    )
    return MapData(walls=walls, rooms=[room])


def _observe(pf: ParticleFilter, x: float, y: float, theta: float):
    return cast_rays(
        (x, y),
        theta,
        pf.map_data.walls,
        num_rays=pf.num_beams,
        max_range=pf.max_range,
    )


TRUE = (3.0, 2.0, 0.6)


def test_weights_never_zero():
    """T-001 regression: the mixture model keeps every beam likelihood > 0."""
    pf = ParticleFilter(_box_map(), num_particles=300, num_beams=36)
    pf.initialize_near(*TRUE, std_xy_m=0.5, std_theta_rad=0.2)
    hits = _observe(pf, *TRUE)
    pf.update(hits)
    assert pf.weights.min() > 0.0
    assert math.isclose(pf.weights.sum(), 1.0, abs_tol=1e-6)


def test_converges_to_true_pose():
    pf = ParticleFilter(_box_map(), num_particles=500, num_beams=36)
    # Deliberately start with a poor prior.
    pf.initialize_near(1.0, 1.0, 1.5, std_xy_m=1.0, std_theta_rad=0.6)

    for _ in range(25):
        hits = _observe(pf, *TRUE)
        pf.update(hits)
        pf.resample()

    ex, ey, et = pf.estimate()
    assert math.hypot(ex - TRUE[0], ey - TRUE[1]) < 0.4
    # circular heading error
    dth = abs((et - TRUE[2] + math.pi) % (2 * math.pi) - math.pi)
    assert dth < 0.3


def test_covariance_and_ess_sane():
    """T-002 regression: covariance and effective sample size are reported."""
    pf = ParticleFilter(_box_map(), num_particles=400, num_beams=36)
    pf.initialize_near(*TRUE, std_xy_m=0.8, std_theta_rad=0.4)
    for _ in range(10):
        pf.update(_observe(pf, *TRUE))
        pf.resample()

    cov = pf.covariance()
    assert cov.shape == (3, 3)
    assert all(cov[i, i] >= 0 for i in range(3))
    ess = pf.effective_sample_size()
    assert 0.0 < ess <= pf.num_particles


def test_predict_moves_particles():
    pf = ParticleFilter(_box_map(), num_particles=200, num_beams=36)
    pf.initialize_near(*TRUE, std_xy_m=0.1, std_theta_rad=0.05)
    before = pf.estimate()
    pf.predict(1.0, 0.0)
    after = pf.estimate()
    # Moving 1 m forward (heading ~0.3 rad) shifts x by ~cos(0.3), y by ~sin(0.3).
    assert after[0] - before[0] == pytest.approx(math.cos(TRUE[2]), abs=0.2)
    assert after[1] - before[1] == pytest.approx(math.sin(TRUE[2]), abs=0.2)


def test_anchor_fraction_preserves_rest():
    pf = ParticleFilter(_box_map(), num_particles=200, num_beams=36)
    pf.initialize_near(*TRUE, std_xy_m=0.5, std_theta_rad=0.2)
    pf.anchor_fraction(1.0, 1.0, 0.0, fraction=0.3)
    # Only ~30% were replaced; the rest still cluster near the old mean.
    ex, ey, _ = pf.estimate()
    assert math.hypot(ex - TRUE[0], ey - TRUE[1]) < 2.0


def test_odom_scale_estimate_bounded():
    pf = ParticleFilter(_box_map(), num_particles=100, num_beams=36)
    pf.initialize_near(*TRUE, std_xy_m=0.3, std_theta_rad=0.1)
    assert pf.odom_scale_min <= pf.odom_scale_estimate() <= pf.odom_scale_max
