"""Headless tests for T3D-04 (physics loop + sim core).

Run in PyBullet DIRECT mode (no display), zenoh disabled, so it is CI-safe.
"""

from __future__ import annotations

import pybullet as p
import pytest

from core.robot_config import load_robot_config
from simulation3d.simulator import _ray_ground_intersection, Simulator3D


@pytest.fixture()
def sim():
    s = Simulator3D(gui=False, seed=0, enable_zenoh=False)
    yield s
    p.disconnect()


def test_spawns_and_rests_stable(sim):
    ch = sim._cfg.chassis
    base_z = ch.wheel_radius_m + ch.height_m / 2.0
    pos, orn = p.getBasePositionAndOrientation(sim._base)
    roll, pitch, _ = p.getEulerFromQuaternion(orn)
    # base rests on its wheels, level, at the expected height
    assert pos[2] == pytest.approx(base_z, abs=0.03)
    assert abs(roll) < 0.05 and abs(pitch) < 0.05
    # arm mounted on top
    arm_z = p.getBasePositionAndOrientation(sim._arm)[0][2]
    assert arm_z > base_z
    # world + objects spawned
    assert sim._ground >= 0
    assert len(sim._objects) == 5
    assert all(o.body_id >= 0 for o in sim._objects)


def test_truth_pose_shape(sim):
    x, y, theta = sim.truth_pose()
    assert isinstance(x, float) and isinstance(y, float) and isinstance(theta, float)


def test_loop_is_stable_without_drift(sim):
    # Step a few seconds; the stationary base should not drift.
    p0 = sim.truth_pose()
    for _ in range(240 * 3):
        sim.step()
    p1 = sim.truth_pose()
    assert abs(p1[0] - p0[0]) < 0.01
    assert abs(p1[1] - p0[1]) < 0.01


def test_deterministic_given_seed():
    def run():
        s = Simulator3D(gui=False, seed=42, enable_zenoh=False)
        trace = []
        for i in range(240 * 2):
            s.step()
            if i % 120 == 0:
                trace.append(tuple(round(v, 6) for v in s.truth_pose()))
        p.disconnect()
        return trace

    assert run() == run()


def test_teleop_commands_wheels(sim):
    # Pressing W should command the drive wheels (base starts to move).
    sim._keys["w"] = True
    before = sim.truth_pose()
    for _ in range(240):
        sim.step()
    after = sim.truth_pose()
    # The base must not remain perfectly stationary once teleop is applied.
    assert (abs(after[0] - before[0]) + abs(after[2] - before[2])) > 1e-4


def test_ray_ground_intersection_basic():
    # A vertical ray from above the origin lands at (0, 0).
    assert _ray_ground_intersection([0, 0, 5], [0, 0, 0]) == pytest.approx((0.0, 0.0))
    # A slanted ray lands where the ground plane actually is (z=0 along the line).
    assert _ray_ground_intersection([2, 1, 4], [4, 3, 0]) == pytest.approx((4.0, 3.0))
    # A ray pointing up (camera looking at the sky) has no ground intersection.
    assert _ray_ground_intersection([0, 0, 1], [0, 0, 2]) is None


def test_ray_ground_intersection_consistent_with_ray():
    # The intersection must lie on the same line as ray_from/ray_to.
    ray_from = [0.5, -0.5, 3.0]
    ray_to = [1.0, 0.0, -1.0]
    gx, gy = _ray_ground_intersection(ray_from, ray_to)
    # Parametrize the line and check the ground point is on it at z = 0.
    t = 3.0 / (3.0 - (-1.0))  # z goes 3 -> -1, reach 0 at this fraction
    assert gx == pytest.approx(0.5 + t * 0.5)
    assert gy == pytest.approx(-0.5 + t * 0.5)
