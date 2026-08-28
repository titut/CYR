"""Robot configuration (T-019, configuration layer).

Every robot-specific parameter lives in one YAML file (default ``robot.yaml``
at the project root, overridable with the ``ROBOT_CONFIG`` environment
variable).  The brains and drivers read their values from a :class:`RobotConfig`
loaded at startup instead of hardcoded module constants — a new robot is a new
YAML file, not an edit to the brains.

The schema is defined here as nested dataclasses.  ``default_robot_config()``
holds the nominal/default values (the 0.75 m base the stack was tuned against);
``load_robot_config()`` starts from those defaults and overlays the YAML file,
so a config file only needs to list what differs.  Validation rejects unknown
keys, wrong types and nonsensical values loudly.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Tuple

import yaml

# Repo-root default config file (this module lives in core/, one level down).
DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "robot.yaml")


class ConfigError(Exception):
    """Raised when a robot config file is missing, invalid or out of range."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class ChassisConfig:
    """Differential-drive body geometry and speed limits."""

    footprint_m: float = 0.75  # square footprint side length (m)
    wheel_radius_m: float = 0.12  # wheel radius (m)
    wheel_track_m: float = 0.69  # drive-wheel separation (m); wheels sit at the corners
    linear_speed_mps: float = 3.0  # max commanded linear speed (m/s)
    angular_speed_rps: float = 3.0  # max commanded angular speed (rad/s)
    # --- 3D body geometry (T3D-01) ---
    height_m: float = 0.15  # chassis box height (m)
    mass_kg: float = 8.0  # chassis mass (kg)
    # Vertical offset of the chassis centre of mass below its geometric centre
    # (m).  A negative value lowers the CG to resist tipping, but PyBullet sinks
    # the base when it is set below ~0, so keep it 0 and rely on the teleop
    # acceleration ramp instead.
    cg_offset_m: float = 0.0
    wheel_width_m: float = 0.10  # wheel cylinder width (m); wider = better grip
    # Front/back free-wheel separation along x (m).  With wheels at the corners
    # this equals the drive track; the front pair is free, the rear pair driven.
    wheelbase_m: float = 0.69

    @property
    def radius_m(self) -> float:
        """Half of the footprint (in-circle radius)."""
        return self.footprint_m / 2.0

    @property
    def collision_radius_m(self) -> float:
        """Circumradius of the square footprint (conservative for any yaw)."""
        return self.radius_m * math.sqrt(2.0)


@dataclass
class DriveConfig:
    """Motor controller + encoder model (the drive node's plant model)."""

    motor_time_constant_s: float = 0.05  # velocity-loop time constant (s)
    max_wheel_accel_rps2: float = 35.0  # torque-limited accel (rad/s²)
    max_wheel_decel_rps2: float = 70.0  # harder decel for crisp stops (rad/s²)
    encoder_noise_rps: float = 0.05  # 1σ encoder noise (rad/s)
    command_timeout_s: float = 0.5  # coast to zero after this long without a cmd
    loop_hz: int = 50  # control loop rate


@dataclass
class SafetyConfig:
    """Body-clearance safety zones (drive node)."""

    slow_down_clearance_m: float = 1.0
    stop_clearance_m: float = 0.15
    estop_clearance_m: float = 0.05
    reverse_escape_mps: float = 0.5  # slow reverse allowed inside the stop zone
    # Heading-direction speed cap: the max safe speed toward an obstacle in
    # the travel direction is sqrt(2 * decel * clearance), where decel is the
    # plant's max wheel decel times this safety factor (covers detection
    # latency, scan staleness, and plant lag).
    heading_decel_safety_factor: float = 0.5
    # Rays within this angle of the travel direction count as "ahead".
    heading_cone_half_angle_rad: float = math.pi / 3


@dataclass
class LidarConfig:
    range_m: float = 10.0
    ray_count: int = 360
    range_noise_m: float = 0.02  # 1σ range noise (m)
    mount_xyz_m: Tuple[float, float, float] = (0.0, 0.0, 0.2)


@dataclass
class ImuConfig:
    gyro_bias_rps: float = 0.005  # ± per-run gyro bias
    gyro_noise_rps: float = 0.01  # 1σ gyro rate noise
    yaw_noise_rad: float = 0.05  # 1σ absolute yaw noise
    mount_xyz_m: Tuple[float, float, float] = (0.0, 0.0, 0.05)


@dataclass
class CameraConfig:
    fov_rad: float = math.radians(90)  # horizontal field of view
    max_range_m: float = 5.0
    range_noise_m: float = 0.02  # 1σ range noise
    bearing_noise_rad: float = math.radians(1.0)  # 1σ bearing noise
    yaw_noise_rad: float = math.radians(2.0)  # 1σ tag yaw noise
    mount_xyz_m: Tuple[float, float, float] = (0.0, 0.0, 0.35)


@dataclass
class SensorsConfig:
    lidar: LidarConfig = field(default_factory=LidarConfig)
    imu: ImuConfig = field(default_factory=ImuConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)


@dataclass
class ArmConfig:
    """Manipulator mounted on the chassis (used by the 3D sim)."""

    urdf: str = "urdf/mycobot_280_pi_3d.urdf"  # primitive-only arm (T3D-02)
    mount_xyz_m: Tuple[float, float, float] = (0.0, 0.0, 0.27)
    mount_rpy_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class PhysicsConfig:
    """Sim-only wheel-ground model (wheel slip, track calibration error)."""

    wheel_scale_min: float = 0.80
    wheel_scale_max: float = 1.20
    track_scale_min: float = 0.85
    track_scale_max: float = 1.15
    slip_noise: float = 0.02
    cross_coupling_rad_per_m: float = 0.1
    # The 4 fixed-wheel base naturally curves at this many rad of yaw per metre
    # of forward travel (a traction artefact).  A positive value adds an opposing
    # angular velocity when driving straight, so W/S actually go forward/back.
    yaw_compensation_rad_per_m: float = 0.0
    # Wheel-ground lateral friction coefficient (PyBullet changeDynamics
    # lateralFriction on every wheel/caster link).  PyBullet's default (0.5)
    # lets the drive wheels slip when rotating the heavy base+arm, so the base
    # skids sideways instead of turning.  1.5-2.0 turns reliably; much higher
    # becomes numerically unstable.
    wheel_friction_mu: float = 2.0


@dataclass
class DeviceConfig:
    """Which driver implementation backs a device (T-019 driver model).

    The name must be registered in the matching ``*_DRIVERS`` registry (see
    ``hal.py`` / ``sim_drivers.py``); an unknown name fails loudly at load.
    """

    driver: str = "sim"


@dataclass
class HardwareConfig:
    """Driver selection for each device the robot exposes."""

    drive: DeviceConfig = field(default_factory=DeviceConfig)
    lidar: DeviceConfig = field(default_factory=DeviceConfig)
    imu: DeviceConfig = field(default_factory=DeviceConfig)
    camera: DeviceConfig = field(default_factory=DeviceConfig)


@dataclass
class RobotConfig:
    """Top-level robot description."""

    name: str = "default"
    chassis: ChassisConfig = field(default_factory=ChassisConfig)
    drive: DriveConfig = field(default_factory=DriveConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    sensors: SensorsConfig = field(default_factory=SensorsConfig)
    arm: ArmConfig = field(default_factory=ArmConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


def default_robot_config() -> RobotConfig:
    """A fresh config holding the nominal/default robot parameters."""
    return RobotConfig()


# ---------------------------------------------------------------------------
# Loading + merging
# ---------------------------------------------------------------------------


def _coerce_like(current, value, key: str):
    """Coerce a YAML value to the type of the existing field value."""
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"field {key!r} must be a boolean, got {type(value).__name__}")
        return value
    if isinstance(current, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"field {key!r} must be an integer, got {type(value).__name__}")
        return value
    if isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"field {key!r} must be a number, got {type(value).__name__}")
        return float(value)
    if isinstance(current, str):
        if not isinstance(value, str):
            raise ConfigError(f"field {key!r} must be a string, got {type(value).__name__}")
        return value
    if isinstance(current, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"field {key!r} must be a list, got {type(value).__name__}")
        return tuple(value)
    raise ConfigError(f"unsupported config field type for {key!r}")


def _apply(obj, data: dict):
    """Recursively overlay plain-dict ``data`` onto a dataclass instance."""
    for key, value in data.items():
        fld = next((f for f in fields(obj) if f.name == key), None)
        if fld is None:
            raise ConfigError(
                f"unknown key {key!r} in {type(obj).__name__} (expected one of "
                f"{', '.join(f.name for f in fields(obj))})"
            )
        if value is None:
            continue  # explicit null leaves the default
        current = getattr(obj, key)
        if isinstance(value, dict) and is_dataclass(current):
            _apply(current, value)
        else:
            setattr(obj, key, _coerce_like(current, value, key))


def _validate(cfg: RobotConfig):
    """Reject physically nonsensical values with a clear message."""
    ch = cfg.chassis
    if ch.footprint_m <= 0:
        raise ConfigError("chassis.footprint_m must be positive")
    if ch.wheel_radius_m <= 0:
        raise ConfigError("chassis.wheel_radius_m must be positive")
    if ch.wheel_track_m <= 0:
        raise ConfigError("chassis.wheel_track_m must be positive")
    if ch.linear_speed_mps < 0 or ch.angular_speed_rps < 0:
        raise ConfigError("chassis speeds must be non-negative")
    if ch.height_m <= 0 or ch.mass_kg <= 0:
        raise ConfigError("chassis height and mass must be positive")
    if ch.wheelbase_m <= 0:
        raise ConfigError("chassis.wheelbase_m must be positive")

    lidar = cfg.sensors.lidar
    if lidar.range_m <= 0:
        raise ConfigError("sensors.lidar.range_m must be positive")
    if lidar.ray_count < 1:
        raise ConfigError("sensors.lidar.ray_count must be at least 1")
    if cfg.sensors.camera.max_range_m <= 0 or cfg.sensors.camera.fov_rad <= 0:
        raise ConfigError("sensors.camera range and fov must be positive")

    if cfg.drive.max_wheel_accel_rps2 < 0 or cfg.drive.max_wheel_decel_rps2 < 0:
        raise ConfigError("drive accel/decel limits must be non-negative")
    for name in (
        "slow_down_clearance_m",
        "stop_clearance_m",
        "estop_clearance_m",
        "heading_decel_safety_factor",
        "heading_cone_half_angle_rad",
    ):
        if getattr(cfg.safety, name) <= 0:
            raise ConfigError(f"safety.{name} must be positive")

    for device in ("drive", "lidar", "imu", "camera"):
        if not getattr(getattr(cfg.hardware, device), "driver").strip():
            raise ConfigError(f"hardware.{device}.driver must be a non-empty string")


def load_robot_config(path: str | None = None) -> RobotConfig:
    """Load a robot config, overlaying defaults with the YAML file.

    ``path`` defaults to the ``ROBOT_CONFIG`` environment variable, then to
    ``robot.yaml`` at the project root.  A missing file raises
    :class:`ConfigError` (a robot must be described explicitly).
    """
    resolved = path or os.environ.get("ROBOT_CONFIG") or DEFAULT_CONFIG_PATH
    if not os.path.exists(resolved):
        raise ConfigError(f"robot config not found: {resolved}")

    with open(resolved, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cfg = default_robot_config()
    _apply(cfg, data)
    _validate(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Process-wide singleton (nodes load once at startup)
# ---------------------------------------------------------------------------

_CACHE: RobotConfig | None = None


def get_robot_config() -> RobotConfig:
    """Load and cache the robot config for this process."""
    global _CACHE
    if _CACHE is None:
        _CACHE = load_robot_config()
    return _CACHE


def reset_robot_config() -> None:
    """Clear the cached config (tests reload between cases)."""
    global _CACHE
    _CACHE = None
