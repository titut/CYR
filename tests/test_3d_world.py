"""Headless tests for T3D-03 (environment, objects, map import).

Run in PyBullet DIRECT mode (no display), so it is CI-safe.
"""

from __future__ import annotations

import math
from pathlib import Path

import pybullet as p
import pybullet_data
import pytest

from core.map_format import MapData
from core.messages import SchemaError, decode
from core.robot_config import load_robot_config
from simulation3d.urdf_assets import generate_object_urdf
from simulation3d.world import (
    WorldObject,
    build_ground,
    build_walls,
    load_world_registry,
    object_registry_message,
    spawn_object,
    spawn_objects,
    update_object_poses,
)

_HOME = Path("home.json")


@pytest.fixture()
def direct():
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(numSubSteps=4, fixedTimeStep=1.0 / 240.0)
    yield cid
    p.disconnect()


def _load_urdf_text(tmp_path, xml: str) -> str:
    path = tmp_path / "obj.urdf"
    path.write_text(xml, encoding="utf-8")
    return str(path)


def test_object_urdf_generates_all_shapes(tmp_path, direct):
    for shape in ("cube", "cylinder", "ball"):
        xml = generate_object_urdf(shape, 0.06, 0.1)
        assert f"<robot name=\"{shape}_0.06\">" in xml
        path = _load_urdf_text(tmp_path, xml)
        body = p.loadURDF(path)
        assert p.getDynamicsInfo(body, -1)[0] == pytest.approx(0.1)
    # invalid shape rejected
    with pytest.raises(ValueError):
        generate_object_urdf("tetrahedron", 0.1, 0.1)


def test_default_arena_has_four_walls(direct):
    walls = build_walls()
    assert len(walls) == 4
    # walls are static boxes at height ~1 m (half of 2.0)
    for wid in walls:
        pos, _ = p.getBasePositionAndOrientation(wid)
        assert pos[2] == pytest.approx(1.0)


def test_map_walls_imported(direct):
    if not _HOME.exists():
        pytest.skip("home.json not present")
    map_data = MapData.from_json(str(_HOME))
    walls = build_walls(map_data)
    assert len(walls) > 0
    # wall count should match the map's wall segments
    assert len(walls) == len(map_data.walls)


def test_registry_objects_spawn_and_rest(direct):
    build_ground()
    objects = load_world_registry()
    assert len(objects) == 5
    assert {o.shape for o in objects} == {"cube", "cylinder", "ball"}
    spawn_objects(objects)
    # let them drop and settle onto the floor
    for _ in range(240 * 6):
        p.stepSimulation()
    for obj in objects:
        pos, _ = p.getBasePositionAndOrientation(obj.body_id)
        lin, _ = p.getBaseVelocity(obj.body_id)
        # no tunnelling through the floor, and settled vertically on it
        # (balls/cylinders may still roll laterally on a flat floor).
        assert math.isfinite(pos[2])
        assert pos[2] >= -0.01, f"{obj.label} tunnelled (z={pos[2]:.3f})"
        assert abs(lin[2]) < 0.1


def test_registry_message_encodes_and_validates(direct):
    build_ground()
    objects = spawn_objects(load_world_registry())
    update_object_poses(objects)
    msg = object_registry_message(objects)
    data = decode("object/registry", msg)  # raises SchemaError if invalid
    assert len(data["objects"]) == 5
    # live pose came from PyBullet ground truth
    assert data["objects"][0]["pose"]["z"] >= 0.0


def test_single_object_spawn(tmp_path, direct):
    obj = WorldObject(
        id=9, label="test_cube", shape="cube", size_m=0.06, mass_kg=0.1,
        x=1.0, y=1.0, z=0.05,
    )
    spawn_object(obj)
    assert obj.body_id >= 0
