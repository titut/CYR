"""Headless tests for T3D-01 (base URDF generation + stability).

Run in PyBullet DIRECT mode (no display), so it is CI-safe.
"""

from __future__ import annotations

import math

import pybullet as p
import pybullet_data
import pytest

from core.robot_config import load_robot_config
from simulation3d.urdf_assets import generate_base_urdf


@pytest.fixture()
def base_urdf(tmp_path):
    cfg = load_robot_config()
    path = tmp_path / "base.urdf"
    path.write_text(generate_base_urdf(cfg), encoding="utf-8")
    return str(path), cfg


def test_base_urdf_generates_from_config(tmp_path):
    # The URDF must reflect the chassis geometry in robot.yaml.
    cfg = load_robot_config()
    xml = generate_base_urdf(cfg)
    assert f'{cfg.name}_base' in xml
    assert f'size="{cfg.chassis.footprint_m} {cfg.chassis.footprint_m} {cfg.chassis.height_m}"' in xml
    # 2 driven wheels + 2 front casters (each = fork + wheel).
    assert xml.count('type="continuous"') == 6  # 4 wheel joints + 2 fork joints
    assert xml.count('<link name="wheel_drive_') == 2
    assert xml.count('<link name="wheel_free_') == 4  # 2 forks + 2 wheels
    assert xml.count('<link name="wheel_free_L_fork">') == 1
    assert xml.count('<link name="wheel_free_R_fork">') == 1


def test_base_urdf_rests_stable(base_urdf):
    path, cfg = base_urdf
    ch = cfg.chassis
    base_z = ch.wheel_radius_m + ch.height_m / 2.0

    p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(numSubSteps=4, fixedTimeStep=1.0 / 240.0)
        uid = p.loadURDF(path, basePosition=[0, 0, base_z + 0.02])

        # Let it settle under gravity.
        for _ in range(240 * 5):
            p.stepSimulation()

        pos, orn = p.getBasePositionAndOrientation(uid)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        contacts = len(p.getContactPoints(uid))

        assert math.isfinite(pos[2])
        assert abs(pos[2] - base_z) < 0.03, f"sank/floated (z={pos[2]:.3f})"
        assert abs(roll) < 0.05 and abs(pitch) < 0.05, "base tipped"
        assert contacts >= 4, "expected the 4 wheels in contact"
    finally:
        p.disconnect()
