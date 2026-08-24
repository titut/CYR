"""3D world: ground plane, walls, graspable objects and the object registry (T3D-03).

Builds the static environment and spawns objects from a *world registry*
(``world.json``).  The registry is the v1 object source for the LLM / arm
controller (no vision): it carries each object's id, label, shape, size, mass,
colour and spawn pose, and it is published on ``object/registry`` so downstream
nodes always know what exists and where it currently is (pose read from PyBullet
ground truth).
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pybullet as p
import pybullet_data

from core.map_format import MapData
from core.messages import encode
from simulation3d.urdf_assets import generate_object_urdf

WALL_HEIGHT_M = 2.0
WALL_THICKNESS_M = 0.1
ARENA_SIZE_M = 10.0

DEFAULT_WORLD_PATH = str(Path(__file__).resolve().parent.parent / "world.json")


@dataclass
class WorldObject:
    """A graspable object from the world registry."""

    id: int
    label: str
    shape: str  # cube | cylinder | ball
    size_m: float
    mass_kg: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    color: Tuple[float, float, float] = (0.7, 0.7, 0.7)
    body_id: int = -1  # set when spawned


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def load_world_registry(path: Optional[str] = None) -> List[WorldObject]:
    """Parse a world registry JSON file (default ``world.json``)."""
    resolved = path or DEFAULT_WORLD_PATH
    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)

    objects: List[WorldObject] = []
    for o in data.get("objects", []):
        pose = o.get("pose", {})
        rpy = pose.get("rpy", [0.0, 0.0, 0.0])
        color = o.get("color", [0.7, 0.7, 0.7])
        objects.append(
            WorldObject(
                id=int(o["id"]),
                label=str(o["label"]),
                shape=str(o["shape"]),
                size_m=float(o["size_m"]),
                mass_kg=float(o["mass_kg"]),
                x=float(pose.get("x", 0.0)),
                y=float(pose.get("y", 0.0)),
                z=float(pose.get("z", 0.0)),
                rpy=(float(rpy[0]), float(rpy[1]), float(rpy[2])),
                color=(float(color[0]), float(color[1]), float(color[2])),
            )
        )
    return objects


# ---------------------------------------------------------------------------
# Environment building
# ---------------------------------------------------------------------------


def build_ground() -> int:
    """Infinite ground plane at z=0."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    return p.loadURDF("plane.urdf")


def _spawn_wall_box(
    x1: float, y1: float, x2: float, y2: float,
    height: float = WALL_HEIGHT_M,
    thickness: float = WALL_THICKNESS_M,
) -> Optional[int]:
    """A static box wall between (x1,y1)-(x2,y2), extruded up to ``height``."""
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1e-6:
        return None
    angle = math.atan2(y2 - y1, x2 - x1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = [length / 2.0, thickness / 2.0, height / 2.0]
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.45, 0.45, 0.5, 1.0])
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    return p.createMultiBody(
        0, col, visual,
        basePosition=[cx, cy, height / 2.0],
        baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, angle]),
    )


def build_walls(map_data: Optional[MapData] = None) -> List[int]:
    """Create wall boxes.

    If ``map_data`` is given, its walls (meters) are extruded to
    ``WALL_HEIGHT_M``.  Otherwise a default ``ARENA_SIZE_M`` square arena of 4
    walls is built.
    """
    ids: List[int] = []
    if map_data is not None and map_data.walls:
        for w in map_data.walls:
            bid = _spawn_wall_box(w.x1, w.y1, w.x2, w.y2)
            if bid is not None:
                ids.append(bid)
    else:
        s = ARENA_SIZE_M / 2.0
        perimeter = [
            (-s, -s, s, -s),
            (s, -s, s, s),
            (s, s, -s, s),
            (-s, s, -s, -s),
        ]
        for x1, y1, x2, y2 in perimeter:
            bid = _spawn_wall_box(x1, y1, x2, y2)
            if bid is not None:
                ids.append(bid)
    return ids


# ---------------------------------------------------------------------------
# Object spawning + registry message
# ---------------------------------------------------------------------------


def spawn_object(obj: WorldObject) -> int:
    """Load an object's primitive URDF at its registry pose.  Sets ``body_id``.

    ``obj.z`` is the object centre height; a ``z`` at/below the floor (the
    common "sits on the floor" case) is lifted so the object drops a few cm and
    settles onto the floor rather than spawning half-buried.
    """
    urdf = generate_object_urdf(obj.shape, obj.size_m, obj.mass_kg, color=obj.color)
    rest_centre_z = obj.size_m / 2.0  # all shapes rest centred at size_m/2 on a floor
    spawn_z = max(obj.z, rest_centre_z) + 0.02
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        f.write(urdf)
        path = f.name
    try:
        body = p.loadURDF(
            path,
            basePosition=[obj.x, obj.y, spawn_z],
            baseOrientation=p.getQuaternionFromEuler(list(obj.rpy)),
        )
    finally:
        import os

        os.unlink(path)
    obj.body_id = body
    return body


def spawn_objects(objects: List[WorldObject]) -> List[WorldObject]:
    """Spawn every object in the registry and return it (body_id set)."""
    for obj in objects:
        spawn_object(obj)
    return objects


def update_object_poses(objects: List[WorldObject]) -> None:
    """Refresh each object's x/y/z/rpy from PyBullet ground truth."""
    for obj in objects:
        if obj.body_id < 0:
            continue
        pos, orn = p.getBasePositionAndOrientation(obj.body_id)
        obj.x, obj.y, obj.z = pos
        obj.rpy = tuple(p.getEulerFromQuaternion(orn))


def object_registry_message(objects: List[WorldObject], t: Optional[float] = None) -> str:
    """Build the ``object/registry`` message (validated by the schema)."""
    return encode(
        "object/registry",
        {
            "t": t if t is not None else time.time(),
            "objects": [
                {
                    "id": o.id,
                    "label": o.label,
                    "shape": o.shape,
                    "size_m": o.size_m,
                    "mass_kg": o.mass_kg,
                    "pose": {
                        "x": o.x,
                        "y": o.y,
                        "z": o.z,
                        "rpy": list(o.rpy),
                    },
                    "color": list(o.color),
                }
                for o in objects
            ],
        },
    )
