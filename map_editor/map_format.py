"""Map file format for the semantic navigation library.

This module defines the JSON schema and provides serialization helpers for the
map data produced by the map editor and consumed by the simulator.

Example JSON:
    {
        "metadata": {
            "scale_m_per_px": 0.05,
            "origin_px": [0, 0],
            "size_px": [800, 600]
        },
        "walls": [
            {"x1": 0, "y1": 0, "x2": 100, "y2": 0}
        ],
        "rooms": [
            {
                "id": "kitchen",
                "name": "Kitchen",
                "polygon": [[0, 0], [100, 0], [100, 80], [0, 80]],
                "center": [50, 40]
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
    """Metric calibration and canvas bounds."""

    scale_m_per_px: float = 0.05
    origin_px: Tuple[int, int] = (0, 0)
    size_px: Tuple[int, int] = (800, 600)


@dataclass
class Wall:
    """A single wall segment in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Room:
    """A named free-space polygon, usually a room or zone."""

    id: str
    name: str
    polygon: List[Tuple[float, float]]
    center: Tuple[float, float]


@dataclass
class Apriltag:
    """A fiducial marker at a known world position for absolute localization.

    The tag sits in the world at (x, y) map-pixel coordinates and faces a
    specific direction.  The robot's simulated camera detects the tag when it
    falls within its field of view, providing an absolute pose measurement
    that anchors the particle filter.
    """

    id: int  # numeric tag ID (e.g. 0, 1, 2, …)
    x: float  # world x in map pixels
    y: float  # world y in map pixels
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
    x: float  # world x in map pixels
    y: float  # world y in map pixels
    radius_px: float = 20.0  # radius in map pixels


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
                "origin_px": list(self.metadata.origin_px),
                "size_px": list(self.metadata.size_px),
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
                    "radius_px": o.radius_px,
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
        return cls(
            metadata=Metadata(
                scale_m_per_px=float(metadata.get("scale_m_per_px", 0.05)),
                origin_px=tuple(metadata.get("origin_px", [0, 0])),
                size_px=tuple(metadata.get("size_px", [800, 600])),
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
                    polygon=[tuple(p) for p in r["polygon"]],
                    center=tuple(r["center"]),
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
                    radius_px=float(o.get("radius_px", 20.0)),
                )
                for o in data.get("obstacles", [])
            ],
        )

    @classmethod
    def from_json(cls, path: Path | str) -> "MapData":
        """Deserialize a map from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def new_empty_map(width_px: int = 800, height_px: int = 600) -> MapData:
    """Create a new empty map with default metadata."""
    return MapData(
        metadata=Metadata(
            scale_m_per_px=0.05,
            origin_px=(0, 0),
            size_px=(width_px, height_px),
        )
    )
