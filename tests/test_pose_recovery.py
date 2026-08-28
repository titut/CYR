"""Tests for the pose-recovery confidence model (T-0XX).

The robot considers its localization "unconfident" only when BOTH conditions
hold at once: a stale AprilTag anchor AND a blind (open-space) LIDAR scan.
"""

from __future__ import annotations

from pose_estimation.pose_estimator import blend_pose, compute_confidence
from navigation.navigator import sort_tags_by_distance


def test_fresh_anchor_in_open_space_is_confident():
    # Just re-anchored; even a blind scan does not drop confidence.
    assert compute_confidence(0.5, 0.05) > 0.9


def test_structured_scan_with_stale_anchor_is_confident():
    # LIDAR is doing the localizing, so a stale anchor alone is not a problem.
    assert compute_confidence(120.0, 0.95) > 0.8


def test_stale_anchor_and_blind_scan_is_unconfident():
    # The recovery trigger condition: no anchor for a while AND no LIDAR.
    assert compute_confidence(60.0, 0.05) < 0.3


def test_confidence_is_bounded_to_unit_interval():
    for age, info in [(-5.0, 1.5), (1e6, 0.0), (10.0, 0.5)]:
        c = compute_confidence(age, info)
        assert 0.0 <= c <= 1.0


def test_confidence_monotonic_in_anchor_age():
    # All else equal, older anchors lower confidence.
    blind = 0.0
    assert compute_confidence(5.0, blind) > compute_confidence(200.0, blind)


def test_sort_tags_by_distance():
    tags = [(10.0, 0.0), (0.0, 0.0), (5.0, 5.0)]
    ordered = sort_tags_by_distance(tags, 0.0, 1.0)
    assert ordered == [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]


def test_blend_pose_first_reading_sets_the_blend():
    # No previous blend: the first reading is taken as-is.
    assert blend_pose(None, (1.0, 2.0, 0.5), 0.35) == (1.0, 2.0, 0.5)


def test_blend_pose_smooths_jittery_detections():
    # A parked robot gets noisy tag readings that oscillate around the truth;
    # the EMA pulls them toward the mean instead of following each one.
    truth = (10.0, 5.0, 0.0)
    blend = None
    import math
    noisy = [
        (10.2, 5.1, 0.05),
        (9.8, 4.9, -0.04),
        (10.1, 5.0, 0.03),
        (9.9, 4.8, -0.02),
        (10.0, 5.1, 0.0),
    ]
    for reading in noisy:
        blend = blend_pose(blend, reading, 0.35)
    bx, by, bt = blend
    assert math.hypot(bx - truth[0], by - truth[1]) < 0.15
    assert abs(bt) < 0.05


def test_blend_pose_heading_does_not_wrap():
    # A heading that crosses +/-pi stays continuous under the EMA.
    b = blend_pose((1.0, 0.0, 3.0), (1.0, 0.0, -3.0), 0.5)
    # 3.0 rad and -3.0 rad differ by ~0.28 rad around the wrap; the blend must
    # stay near the wrap, not average to 0.
    assert 3.0 <= b[2] <= 3.2 or -3.2 <= b[2] <= -3.0
