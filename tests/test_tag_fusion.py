"""Tests for Bayesian AprilTag fusion (ParticleFilter.fuse_absolute_pose).

The tag measurement re-weights the cloud by its Gaussian likelihood instead of
hard-snapping it; a measurement that contradicts every particle triggers the
recovery re-seed inside fuse_absolute_pose.
"""

from __future__ import annotations

import math

from core.map_format import MapData, Room, Wall
from pose_estimation.particle_filter import ParticleFilter


def _box_map() -> MapData:
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


def test_fuse_reweights_cloud_toward_measurement():
    """A tag consistent with the belief sharpens the cloud toward itself."""
    pf = ParticleFilter(_box_map(), num_particles=400, num_beams=36)
    pf.initialize_near(8.0, 5.0, 0.5, std_xy_m=1.0, std_theta_rad=0.3)
    pre = pf.estimate()

    for _ in range(5):
        assert pf.fuse_absolute_pose(
            8.0, 5.0, 0.5, std_xy_m=0.1, std_theta_rad=0.05
        )

    ex, ey, et = pf.estimate()
    assert math.hypot(ex - 8.0, ey - 5.0) < math.hypot(
        pre[0] - 8.0, pre[1] - 5.0
    )
    assert math.hypot(ex - 8.0, ey - 5.0) < 0.3
    dth = abs((et - 0.5 + math.pi) % (2 * math.pi) - math.pi)
    assert dth < 0.2


def test_fuse_contradicting_measurement_triggers_recovery_reseed():
    """A tag 5+ std away from the whole cloud re-seeds around the measurement."""
    pf = ParticleFilter(_box_map(), num_particles=400, num_beams=36)
    pf.initialize_near(1.0, 1.0, 0.0, std_xy_m=0.2, std_theta_rad=0.1)

    assert not pf.fuse_absolute_pose(
        8.0, 5.0, 0.5, std_xy_m=0.05, std_theta_rad=0.02
    )

    # The re-seed anchors half the cloud at the measurement and keeps half as
    # diversity, so the first mean lands between the two clusters; the next
    # fusion re-weights toward the re-seeded half and converges.
    ex, ey, _ = pf.estimate()
    assert math.hypot(ex - 8.0, ey - 5.0) < 5.0

    assert pf.fuse_absolute_pose(
        8.0, 5.0, 0.5, std_xy_m=0.05, std_theta_rad=0.02
    )
    ex, ey, et = pf.estimate()
    assert math.hypot(ex - 8.0, ey - 5.0) < 0.5


def test_fuse_time_aligned_back_propagation():
    """Motion between capture time and now is undone before the weight update:
    fusing a measurement taken before a forward move lands the cloud behind the
    current position, at the measurement pose."""
    pf = ParticleFilter(_box_map(), num_particles=300, num_beams=36)
    pf.initialize_near(5.0, 3.0, 0.0, std_xy_m=0.1, std_theta_rad=0.05)

    # The robot moved 1 m forward after the measurement was taken; the
    # measurement reflects the pose *before* the move.
    assert pf.fuse_absolute_pose(
        4.0, 3.0, 0.0, std_xy_m=0.05, std_theta_rad=0.02,
        delta_forward_m=1.0, delta_theta=0.0,
    )

    ex, ey, _ = pf.estimate()
    # Particles were pulled to (4, 3) then re-projected 1 m forward → ~ (5, 3).
    assert math.hypot(ex - 5.0, ey - 3.0) < 0.3
    assert abs(ey - 3.0) < 0.3
