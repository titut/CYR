"""Tests for map_format serialization and defaults."""

from __future__ import annotations

import json

import pytest

from map_format import MapData, Obstacle, Wall, new_empty_map


def _sample_map() -> MapData:
    return MapData.from_dict(
        {
            "metadata": {"scale_m_per_px": 0.02, "origin_m": [1.0, 2.0], "size_m": [10.0, 8.0]},
            "walls": [{"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 0.0}],
            "rooms": [
                {
                    "id": "r1",
                    "name": "room1",
                    "polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
                    "center": [2.5, 2.5],
                }
            ],
            "apriltags": [{"id": 3, "x": 1.0, "y": 1.0, "yaw_rad": 0.5, "size_m": 0.2}],
            "obstacles": [
                {"id": 0, "x": 3.0, "y": 3.0, "radius_m": 1.0, "vx_mps": 0.3, "vy_mps": -0.2}
            ],
        }
    )


def test_dict_round_trip():
    m = _sample_map()
    m2 = MapData.from_dict(m.to_dict())

    assert m2.metadata.scale_m_per_px == pytest.approx(0.02)
    assert m2.metadata.origin_m == (1.0, 2.0)
    assert m2.metadata.size_m == (10.0, 8.0)
    assert len(m2.walls) == 1
    assert m2.walls[0].x2 == pytest.approx(10.0)
    assert m2.rooms[0].name == "room1"
    assert m2.rooms[0].polygon[0] == (0.0, 0.0)
    assert m2.apriltags[0].id == 3
    assert m2.apriltags[0].size_m == pytest.approx(0.2)
    assert m2.obstacles[0].vx_mps == pytest.approx(0.3)
    assert m2.obstacles[0].vy_mps == pytest.approx(-0.2)


def test_json_round_trip(tmp_path):
    m = _sample_map()
    p = tmp_path / "map.json"
    m.to_json(p)

    m2 = MapData.from_json(p)
    assert m2.to_dict() == m.to_dict()
    # moving-obstacle velocity survives the JSON round trip
    assert m2.obstacles[0].vx_mps == pytest.approx(0.3)


def test_from_dict_defaults():
    m = MapData.from_dict({})
    assert m.metadata.size_m == (8.0, 6.0)
    assert m.metadata.origin_m == (0.0, 0.0)
    assert m.walls == []
    assert m.rooms == []
    assert m.apriltags == []
    assert m.obstacles == []


def test_obstacle_defaults():
    o = Obstacle(id=1, x=2.0, y=3.0)
    assert o.radius_m == pytest.approx(0.2)
    assert o.vx_mps == 0.0 and o.vy_mps == 0.0  # static by default


def test_moving_obstacle_parsed_from_json(tmp_path):
    raw = {
        "obstacles": [
            {"id": 0, "x": 1.0, "y": 1.0, "radius_m": 1.0, "vx_mps": 0.5, "vy_mps": 0.1}
        ]
    }
    p = tmp_path / "m.json"
    p.write_text(json.dumps(raw))
    m = MapData.from_json(p)
    assert m.obstacles[0].vx_mps == pytest.approx(0.5)


def test_new_empty_map():
    m = new_empty_map()
    assert m.metadata.size_m == (8.0, 6.0)
    assert not m.walls and not m.obstacles


def test_wall_dataclass():
    w = Wall(0.0, 0.0, 5.0, 5.0)
    assert (w.x1, w.y1, w.x2, w.y2) == (0.0, 0.0, 5.0, 5.0)
