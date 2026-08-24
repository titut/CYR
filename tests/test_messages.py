"""Tests for the typed, versioned message schema (T-021)."""

from __future__ import annotations

import json

import pytest

from core.messages import (
    SchemaError,
    VERSION,
    decode,
    decode_path,
    decode_text,
    encode,
    encode_text,
)


def _round_trip(topic, fields):
    raw = encode(topic, fields)
    assert json.loads(raw)["v"] == VERSION
    data = decode(topic, raw)
    for key, value in fields.items():
        assert data[key] == value
    return data


def test_encode_stamps_version_and_round_trips():
    data = _round_trip(
        "estimate/pose",
        {"x_m": 1.0, "y_m": 2.0, "theta_rad": 0.5, "t": 123.0},
    )
    assert data["x_m"] == 1.0


def test_lidar_round_trip():
    data = _round_trip(
        "sensor/lidar",
        {"t": 1.0, "rays": [{"angle_rad": 0.0, "distance_m": 3.2}]},
    )
    assert data["rays"][0]["distance_m"] == 3.2


def test_wasd_round_trip():
    data = _round_trip("sensor/wasd", {"w": True, "a": False, "s": False, "d": True})
    assert data["w"] is True


def test_decode_rejects_missing_version():
    with pytest.raises(SchemaError):
        decode("nav/goal", json.dumps({"x_m": 1.0, "y_m": 2.0}))


def test_decode_rejects_wrong_version():
    with pytest.raises(SchemaError):
        decode("nav/goal", json.dumps({"v": 999, "x_m": 1.0, "y_m": 2.0}))


def test_decode_rejects_invalid_json():
    with pytest.raises(SchemaError):
        decode("nav/goal", "not json {")


def test_decode_rejects_non_object():
    with pytest.raises(SchemaError):
        decode("nav/goal", json.dumps([1, 2, 3]))


def test_decode_rejects_missing_required_field():
    with pytest.raises(SchemaError):
        decode("nav/goal", json.dumps({"v": VERSION, "x_m": 1.0}))


def test_decode_rejects_wrong_field_type():
    with pytest.raises(SchemaError):
        decode("sensor/wasd", json.dumps({"v": VERSION, "w": 1, "a": 0, "s": 0, "d": 0}))


def test_encode_validates_before_sending():
    with pytest.raises(SchemaError):
        encode("sensor/wasd", {"w": "yes", "a": False, "s": False, "d": False})


def test_lidar_ray_missing_distance_rejected():
    with pytest.raises(SchemaError):
        encode("sensor/lidar", {"t": 0.0, "rays": [{"angle_rad": 0.0}]})


def test_imu_requires_accel_object():
    with pytest.raises(SchemaError):
        encode(
            "sensor/imu",
            {
                "t": 0.0,
                "yaw_rad": 0.0,
                "pitch_rad": 0.0,
                "roll_rad": 0.0,
                "angular_velocity_rps": 0.0,
                "linear_acceleration_mps2": "bad",
            },
        )


def test_apriltag_detection_id_must_be_int():
    with pytest.raises(SchemaError):
        encode(
            "detection/apriltag",
            {
                "detections": [
                    {"id": 1.5, "x_rel": 0.1, "y_rel": 0.2, "yaw_rel": 0.0, "size_m": 0.16}
                ]
            },
        )


def test_decode_path_accepts_points_and_empty():
    assert decode_path("nav/path", json.dumps([[1.0, 2.0], [3.0, 4.0]])) == [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
    assert decode_path("nav/path", "[]") == []


def test_decode_path_rejects_bad_point():
    with pytest.raises(SchemaError):
        decode_path("nav/path", json.dumps([[1.0, 2.0], [3.0]]))


def test_decode_path_rejects_non_array():
    with pytest.raises(SchemaError):
        decode_path("nav/path", json.dumps({"x": 1}))


def test_text_round_trip():
    assert encode_text("nav/command", "go to shrine") == "go to shrine"
    assert decode_text("nav/command", "  go to shrine  ") == "go to shrine"


def test_decode_text_rejects_empty():
    with pytest.raises(SchemaError):
        decode_text("safety/reset", "   ")


def test_sim_truth_pose_round_trip():
    data = _round_trip("sim/truth/pose", {"t": 1.0, "x_m": 2.0, "y_m": 3.0, "theta_rad": 0.5})
    assert data["x_m"] == 2.0


def test_object_registry_round_trip():
    msg = encode(
        "object/registry",
        {
            "t": 1.0,
            "objects": [
                {
                    "id": 0,
                    "label": "cube",
                    "shape": "cube",
                    "size_m": 0.06,
                    "mass_kg": 0.1,
                    "pose": {"x": 1.0, "y": 2.0, "z": 0.03, "rpy": [0.0, 0.0, 0.0]},
                    "color": [0.8, 0.2, 0.2],
                }
            ],
        },
    )
    data = decode("object/registry", msg)
    assert data["objects"][0]["label"] == "cube"


def test_object_registry_rejects_bad_shape():
    with pytest.raises(SchemaError):
        encode(
            "object/registry",
            {"t": 1.0, "objects": [
                {"id": 0, "label": "x", "shape": "tetra", "size_m": 0.1,
                 "mass_kg": 0.1, "pose": {"x": 0, "y": 0, "z": 0, "rpy": [0, 0, 0]}}
            ]},
        )
