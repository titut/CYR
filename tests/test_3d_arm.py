"""Headless tests for T3D-02 (arm URDF: primitives + inertials + gripper).

Run in PyBullet DIRECT mode (no display), so it is CI-safe.
"""

from __future__ import annotations

import pybullet as p
import pytest

from core.robot_config import load_robot_config
from simulation3d.urdf_assets import generate_arm_urdf


@pytest.fixture()
def arm_urdf(tmp_path):
    cfg = load_robot_config()
    path = tmp_path / "arm.urdf"
    path.write_text(generate_arm_urdf(cfg), encoding="utf-8")
    return str(path)


def test_arm_urdf_generates_primitives_with_inertials(tmp_path):
    xml = generate_arm_urdf(load_robot_config())
    # Every arm link has an inertial + a primitive box (no mesh references).
    for link in ("g_base", "joint1", "joint2", "joint3", "joint4", "joint5",
                 "joint6", "joint6_flange"):
        assert f'<link name="{link}">' in xml
        assert xml.count('<inertial>') >= 8
        assert "<mesh " not in xml  # primitive-only
    # Gripper joints exist with the specified range.
    assert 'name="gripper_joint_left"' in xml
    assert 'name="gripper_joint_right"' in xml
    assert 'lower="0.0" upper="0.04"' in xml


def test_arm_loads_with_no_mass_zero(arm_urdf):
    p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81)
        uid = p.loadURDF(arm_urdf)
        # base link + all joints
        masses = [p.getDynamicsInfo(uid, j)[0] for j in range(-1, p.getNumJoints(uid))]
        assert all(m > 0 for m in masses)
    finally:
        p.disconnect()


def test_arm_holds_pose_when_anchored(arm_urdf):
    p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(numSubSteps=4, fixedTimeStep=1.0 / 240.0)
        uid = p.loadURDF(arm_urdf)
        # Anchor g_base (as if mounted on the robot) and lock all joints at 0.
        p.createConstraint(uid, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
        for j in range(p.getNumJoints(uid)):
            p.setJointMotorControl2(uid, j, p.POSITION_CONTROL, targetPosition=0.0, force=1000)
        for _ in range(240):
            p.stepSimulation()

        def base_pos():
            return p.getBasePositionAndOrientation(uid)[0]

        def link_pos(j):
            return p.getLinkState(uid, j)[0]

        base0, link0 = base_pos(), {j: link_pos(j) for j in range(p.getNumJoints(uid))}
        for _ in range(240 * 3):  # 3 s
            p.stepSimulation()
        drift = max(
            sum((a - b) ** 2 for a, b in zip(base0, base_pos())) ** 0.5,
            *(
                sum((a - b) ** 2 for a, b in zip(link0[j], link_pos(j))) ** 0.5
                for j in range(p.getNumJoints(uid))
            ),
        )
        assert drift < 0.001, f"arm links drifted {drift * 1000:.3f} mm"
    finally:
        p.disconnect()


def test_gripper_joint_range(arm_urdf):
    p.connect(p.DIRECT)
    try:
        uid = p.loadURDF(arm_urdf)
        for side in ("left", "right"):
            jid = [
                j
                for j in range(p.getNumJoints(uid))
                if p.getJointInfo(uid, j)[1].decode() == f"gripper_joint_{side}"
            ][0]
            lower, upper = p.getJointInfo(uid, jid)[8:10]
            assert lower == 0.0
            assert upper == pytest.approx(0.04)
    finally:
        p.disconnect()
