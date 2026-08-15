"""Map file format for the semantic navigation library.

This module defines the JSON schema and provides serialization helpers for the
map data produced by the map editor and consumed by the simulator.

All geometry in this format is expressed in **meters**.  ``scale_m_per_px`` is
retained only as the map editor's canvas calibration: the number of meters per
canvas pixel, used by the editor to convert between its pixel canvas and the
stored meter coordinates.

Example JSON:
    {
        "metadata": {
            "scale_m_per_px": 0.01,
            "origin_m": [0.0, 0.0],
            "size_m": [8.0, 6.0]
        },
        "walls": [
            {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 0.0}
        ],
        "rooms": [
            {
                "id": "kitchen",
                "name": "Kitchen",
                "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 0.8], [0.0, 0.8]],
                "center": [0.5, 0.4]
            }
        ]
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class Metadata:
    """Metric calibration and canvas bounds (meters)."""

    scale_m_per_px: float = 0.01  # meters per canvas pixel (editor calibration)
    origin_m: Tuple[float, float] = (0.0, 0.0)
    size_m: Tuple[float, float] = (8.0, 6.0)


@dataclass
class Wall:
    """A single wall segment in meters."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Room:
    """A named free-space polygon, usually a room or zone (meters)."""

    id: str
    name: str
    polygon: List[Tuple[float, float]]
    center: Tuple[float, float]


@dataclass
class Apriltag:
    """A fiducial marker at a known world position for absolute localization.

    The tag sits in the world at (x, y) meter coordinates and faces a specific
    direction.  The robot's simulated camera detects the tag when it falls
    within its field of view, providing an absolute pose measurement that
    anchors the particle filter.
    """

    id: int  # numeric tag ID (e.g. 0, 1, 2, …)
    x: float  # world x in meters
    y: float  # world y in meters
    yaw_rad: float = 0.0  # facing direction in radians (0 = east / +x)
    size_m: float = 0.16  # physical side length in meters


@dataclass
class Obstacle:
    """A circular physical obstacle in the world.

    Obstacles are *unknown* to the planner: the navigation node does not add
    them to its occupancy grid up front.  Instead the robot detects them with
    its LIDAR (actual scan vs. expected wall-only scan) and routes around them.
    The simulator treats them as solid for both collision and ray casting.
    """

    id: int  # numeric obstacle ID
    x: float  # world x in meters
    y: float  # world y in meters
    radius_m: float = 0.2  # radius in meters


@dataclass
class MapData:
    """Top-level container for a semantic navigation map."""

    metadata: Metadata = field(default_factory=Metadata)
    walls: List[Wall] = field(default_factory=list)
    rooms: List[Room] = field(default_factory=list)
    apriltags: List[Apriltag] = field(default_factory=list)
    obstacles: List[Obstacle] = field(default_factory=list)

    @staticmethod
    def _point_from_json(point: List[float]) -> Tuple[float, ...]:
        return tuple(point)

    @staticmethod
    def _point_to_json(point: Tuple[float, ...]) -> List[float]:
        return list(point)

    def to_dict(self) -> dict:
        """Convert the map to a plain dictionary matching the JSON schema."""
        return {
            "metadata": {
                "scale_m_per_px": self.metadata.scale_m_per_px,
                "origin_m": list(self.metadata.origin_m),
                "size_m": list(self.metadata.size_m),
            },
            "walls": [
                {"x1": w.x1, "y1": w.y1, "x2": w.x2, "y2": w.y2} for w in self.walls
            ],
            "rooms": [
                {
                    "id": r.id,
                    "name": r.name,
                    "polygon": [list(p) for p in r.polygon],
                    "center": list(r.center),
                }
                for r in self.rooms
            ],
            "apriltags": [
                {
                    "id": t.id,
                    "x": t.x,
                    "y": t.y,
                    "yaw_rad": t.yaw_rad,
                    "size_m": t.size_m,
                }
                for t in self.apriltags
            ],
            "obstacles": [
                {
                    "id": o.id,
                    "x": o.x,
                    "y": o.y,
                    "radius_m": o.radius_m,
                }
                for o in self.obstacles
            ],
        }

    def to_json(self, path: Path | str, indent: int = 2) -> None:
        """Serialize the map to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "MapData":
        """Deserialize a map from a plain dictionary."""
        metadata = data.get("metadata", {})

        def _point(p) -> Tuple[float, float]:
            return (float(p[0]), float(p[1]))

        return cls(
            metadata=Metadata(
                scale_m_per_px=float(metadata.get("scale_m_per_px", 0.01)),
                origin_m=_point(metadata.get("origin_m", [0.0, 0.0])),
                size_m=_point(metadata.get("size_m", [8.0, 6.0])),
            ),
            walls=[
                Wall(
                    x1=float(w["x1"]),
                    y1=float(w["y1"]),
                    x2=float(w["x2"]),
                    y2=float(w["y2"]),
                )
                for w in data.get("walls", [])
            ],
            rooms=[
                Room(
                    id=str(r["id"]),
                    name=str(r["name"]),
                    polygon=[_point(p) for p in r["polygon"]],
                    center=_point(r["center"]),
                )
                for r in data.get("rooms", [])
            ],
            apriltags=[
                Apriltag(
                    id=int(t["id"]),
                    x=float(t["x"]),
                    y=float(t["y"]),
                    yaw_rad=float(t.get("yaw_rad", 0.0)),
                    size_m=float(t.get("size_m", 0.16)),
                )
                for t in data.get("apriltags", [])
            ],
            obstacles=[
                Obstacle(
                    id=int(o["id"]),
                    x=float(o["x"]),
                    y=float(o["y"]),
                    radius_m=float(o.get("radius_m", 0.2)),
                )
                for o in data.get("obstacles", [])
            ],
        )

    @classmethod
    def from_json(cls, path: Path | str) -> "MapData":
        """Deserialize a map from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def new_empty_map(width_m: float = 8.0, height_m: float = 6.0) -> MapData:
    """Create a new empty map with default metadata."""
    return MapData(
        metadata=Metadata(
            scale_m_per_px=0.01,
            origin_m=(0.0, 0.0),
            size_m=(width_m, height_m),
        )
    )
