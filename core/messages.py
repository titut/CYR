"""Typed, versioned Zenoh message schema (T-021).

Every JSON message published on a zenoh topic carries a version field (``v``).
This module is the **schema registry**: producers serialize with ``encode()``
(which stamps the version and validates the payload before sending), and
consumers parse with ``decode()`` (which checks the version and validates every
field's type).  Anything malformed raises :class:`SchemaError`; callers log it,
so bad messages are *never silently dropped* — the old pattern of
``except (json.JSONDecodeError, KeyError, Exception): pass`` is gone.

Topics whose payloads are not a JSON object have dedicated helpers:

- ``nav/command``, ``safety/reset``  -> ``encode_text`` / ``decode_text``
- ``nav/path``                      -> ``decode_path`` (JSON array of [x, y])

The schema registry itself is the ``_VALIDATORS`` dict: topic -> validator.
Adding a topic here gives every producer/consumer the same type checks.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

# Current wire-format version.  Bump when the schema changes incompatibly;
# ``decode`` rejects any other value so mismatched producers/consumers fail
# loudly instead of corrupting state.
VERSION = 1


class SchemaError(Exception):
    """Raised when a message cannot be parsed, is the wrong version, or fails
    field validation."""


# ---------------------------------------------------------------------------
# Field helpers (raise SchemaError with a topic-qualified message)
# ---------------------------------------------------------------------------


def _num(data: dict, topic: str, key: str, required: bool = True):
    val = data.get(key)
    if val is None:
        if required:
            raise SchemaError(f"{topic}: missing required field {key!r}")
        return None
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise SchemaError(
            f"{topic}: field {key!r} must be a number, got {type(val).__name__}"
        )
    return float(val)


def _bool(data: dict, topic: str, key: str):
    val = data.get(key)
    if not isinstance(val, bool):
        raise SchemaError(
            f"{topic}: field {key!r} must be a boolean, got {type(val).__name__}"
        )


def _int(data: dict, topic: str, key: str):
    val = data.get(key)
    if isinstance(val, bool) or not isinstance(val, int):
        raise SchemaError(
            f"{topic}: field {key!r} must be an integer, got {type(val).__name__}"
        )


def _text(data: dict, topic: str, key: str):
    val = data.get(key)
    if not isinstance(val, str):
        raise SchemaError(
            f"{topic}: field {key!r} must be a string, got {type(val).__name__}"
        )


def _obj_list(data: dict, topic: str, key: str) -> List[dict]:
    val = data.get(key)
    if not isinstance(val, list):
        raise SchemaError(f"{topic}: field {key!r} must be a list")
    for i, item in enumerate(val):
        if not isinstance(item, dict):
            raise SchemaError(f"{topic}: {key}[{i}] must be an object")
    return val


def _pair_list(data: dict, topic: str, key: str):
    val = data.get(key)
    if not isinstance(val, list):
        raise SchemaError(f"{topic}: field {key!r} must be a list")
    for i, item in enumerate(val):
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(isinstance(c, bool) or not isinstance(c, (int, float)) for c in item)
        ):
            raise SchemaError(f"{topic}: {key}[{i}] must be a [x, y] number pair")


# ---------------------------------------------------------------------------
# Per-topic validators
# ---------------------------------------------------------------------------


def _validate_lidar(data: dict):
    _num(data, "sensor/lidar", "t")
    rays = _obj_list(data, "sensor/lidar", "rays")
    for i, r in enumerate(rays):
        _num(r, f"sensor/lidar rays[{i}]", "angle_rad")
        _num(r, f"sensor/lidar rays[{i}]", "distance_m")


def _validate_imu(data: dict):
    _num(data, "sensor/imu", "t")
    _num(data, "sensor/imu", "yaw_rad")
    _num(data, "sensor/imu", "pitch_rad")
    _num(data, "sensor/imu", "roll_rad")
    _num(data, "sensor/imu", "angular_velocity_rps")
    accel = data.get("linear_acceleration_mps2")
    if not isinstance(accel, dict):
        raise SchemaError("sensor/imu: 'linear_acceleration_mps2' must be an object")
    for axis in ("x", "y", "z"):
        _num(accel, "sensor/imu linear_acceleration_mps2", axis)


def _validate_wasd(data: dict):
    for key in ("w", "a", "s", "d"):
        _bool(data, "sensor/wasd", key)


def _validate_wheel_speed(data: dict):
    _num(data, "sensor/wheel_speed", "left_rps")
    _num(data, "sensor/wheel_speed", "right_rps")
    _num(data, "sensor/wheel_speed", "t")


def _validate_camera_apriltag(data: dict):
    _num(data, "sensor/camera/apriltag", "t")
    dets = _obj_list(data, "sensor/camera/apriltag", "detections")
    for i, d in enumerate(dets):
        tag = f"sensor/camera/apriltag detections[{i}]"
        _int(d, tag, "id")
        _num(d, tag, "range_m")
        _num(d, tag, "bearing_rad")
        _num(d, tag, "tag_yaw_rad")
        _num(d, tag, "tag_size_m")


def _validate_detection_apriltag(data: dict):
    _num(data, "detection/apriltag", "t", required=False)
    dets = _obj_list(data, "detection/apriltag", "detections")
    for i, d in enumerate(dets):
        tag = f"detection/apriltag detections[{i}]"
        _int(d, tag, "id")
        _num(d, tag, "x_rel")
        _num(d, tag, "y_rel")
        _num(d, tag, "yaw_rel")
        _num(d, tag, "size_m")


def _validate_cmd_velocity(data: dict):
    _num(data, "cmd/velocity", "linear_mps")
    _num(data, "cmd/velocity", "angular_rps")


def _validate_nav_goal(data: dict):
    _num(data, "nav/goal", "x_m")
    _num(data, "nav/goal", "y_m")


def _validate_estimate_pose(data: dict):
    _num(data, "estimate/pose", "x_m")
    _num(data, "estimate/pose", "y_m")
    _num(data, "estimate/pose", "theta_rad")
    _num(data, "estimate/pose", "t", required=False)
    _num(data, "estimate/pose", "odom_scale", required=False)
    _num(data, "estimate/pose", "track_scale", required=False)
    _num(data, "estimate/pose", "gyro_bias", required=False)
    _num(data, "estimate/pose", "confidence", required=False)
    _num(data, "estimate/pose", "anchor_age_s", required=False)
    _num(data, "estimate/pose", "scan_info", required=False)


def _validate_estimate_halt(data: dict):
    _num(data, "estimate/halt", "t")
    _num(data, "estimate/halt", "hold_s")


def _validate_detection_obstacles(data: dict):
    _num(data, "detection/obstacles", "t")
    _pair_list(data, "detection/obstacles", "points")


def _validate_safety_status(data: dict):
    _text(data, "safety/status", "state")
    _num(data, "safety/status", "min_clearance_m")
    _num(data, "safety/status", "t")


def _validate_sim_truth_pose(data: dict):
    _num(data, "sim/truth/pose", "t")
    _num(data, "sim/truth/pose", "x_m")
    _num(data, "sim/truth/pose", "y_m")
    _num(data, "sim/truth/pose", "theta_rad")


def _validate_object_pose(data: dict, topic: str):
    pose = data.get("pose")
    if not isinstance(pose, dict):
        raise SchemaError(f"{topic}: 'pose' must be an object")
    for k in ("x", "y", "z"):
        _num(pose, topic, k)
    rpy = pose.get("rpy")
    if not isinstance(rpy, list) or len(rpy) != 3 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) for v in rpy
    ):
        raise SchemaError(f"{topic}: 'pose.rpy' must be a 3-number list")


def _validate_object_registry(data: dict):
    _num(data, "object/registry", "t")
    objs = _obj_list(data, "object/registry", "objects")
    for i, o in enumerate(objs):
        tag = f"object/registry objects[{i}]"
        _int(o, tag, "id")
        _text(o, tag, "label")
        _text(o, tag, "shape")
        if o["shape"] not in ("cube", "cylinder", "ball"):
            raise SchemaError(f"{tag}: shape must be cube|cylinder|ball")
        _num(o, tag, "size_m")
        _num(o, tag, "mass_kg")
        _validate_object_pose(o, tag)
        color = o.get("color")
        if color is not None and (
            not isinstance(color, list)
            or len(color) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in color)
        ):
            raise SchemaError(f"{tag}: 'color' must be a 3-number list")


_VALIDATORS: Dict[str, Callable[[dict], None]] = {
    "sensor/lidar": _validate_lidar,
    "sensor/imu": _validate_imu,
    "sensor/wasd": _validate_wasd,
    "sensor/wheel_speed": _validate_wheel_speed,
    "sensor/camera/apriltag": _validate_camera_apriltag,
    "detection/apriltag": _validate_detection_apriltag,
    "cmd/velocity": _validate_cmd_velocity,
    "nav/goal": _validate_nav_goal,
    "estimate/pose": _validate_estimate_pose,
    "estimate/halt": _validate_estimate_halt,
    "detection/obstacles": _validate_detection_obstacles,
    "safety/status": _validate_safety_status,
    "sim/truth/pose": _validate_sim_truth_pose,
    "object/registry": _validate_object_registry,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _payload_text(payload) -> str:
    """Return the payload as text from either a raw string or a zenoh Sample
    (whose payload is wrapped in ``sample.payload``)."""
    if isinstance(payload, str):
        return payload
    inner = getattr(payload, "payload", None)
    if inner is not None:
        payload = inner
    to_string = getattr(payload, "to_string", None)
    if to_string is None:
        raise SchemaError("cannot read payload (expected str or zenoh Sample)")
    return to_string()


def encode(topic: str, fields: Dict[str, Any]) -> str:
    """Serialize a dict message for a registered topic (stamps the version and
    validates before sending)."""
    data = {"v": VERSION}
    data.update(fields)
    validator = _VALIDATORS.get(topic)
    if validator is not None:
        validator(data)
    return json.dumps(data)


def decode(topic: str, payload) -> dict:
    """Parse and validate an incoming message.  Raises :class:`SchemaError`."""
    try:
        data = json.loads(_payload_text(payload))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{topic}: invalid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise SchemaError(f"{topic}: expected a JSON object, got a {type(data).__name__}")
    version = data.get("v")
    if version != VERSION:
        raise SchemaError(
            f"{topic}: unsupported version {version!r} (expected {VERSION})"
        )
    validator = _VALIDATORS.get(topic)
    if validator is not None:
        validator(data)
    return data


def encode_text(topic: str, text: str) -> str:
    """Serialize a plain-text message (nav/command, safety/reset)."""
    if not isinstance(text, str) or not text.strip():
        raise SchemaError(f"{topic}: payload must be a non-empty string")
    return text


def decode_text(topic: str, payload) -> str:
    """Parse a plain-text message; raises on empty payload."""
    text = _payload_text(payload).strip()
    if not text:
        raise SchemaError(f"{topic}: empty text payload")
    return text


def decode_path(topic: str, payload) -> List[List[float]]:
    """Parse a JSON-array path message ([[x, y], ...]); raises on malformed input."""
    try:
        data = json.loads(_payload_text(payload))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{topic}: invalid JSON ({exc})") from exc
    if not isinstance(data, list):
        raise SchemaError(f"{topic}: expected a JSON array, got a {type(data).__name__}")
    for i, p in enumerate(data):
        if (
            not isinstance(p, (list, tuple))
            or len(p) != 2
            or any(isinstance(c, bool) or not isinstance(c, (int, float)) for c in p)
        ):
            raise SchemaError(f"{topic}: point {i} must be a [x, y] number pair")
    return [[float(p[0]), float(p[1])] for p in data]
