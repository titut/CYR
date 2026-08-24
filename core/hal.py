"""Hardware Abstraction Layer — driver interfaces (T-019, driver model).

Each device the robot has (drive, LIDAR, IMU, camera) is behind a small
interface class.  A *driver* is an implementation of one of these interfaces;
the zenoh topics are the transport.  The brains (navigator, controller, pose
estimator, LLM) never touch a driver directly — they only read/write the topics.

Topic contract each interface maps to:

    DriveDriver    cmd/velocity        -> set_command(linear, angular)
                   sensor/wheel_speed  <- step(dt) returns measured wheel speeds
    LidarDriver    sensor/lidar        <- scan(...) returns ray hits
    ImuDriver      sensor/imu          <- read(...) returns noisy angular data
    CameraDriver   sensor/camera/apriltag <- detect(...) returns detections

Adding a hardware backend = implementing the interface + registering it in the
matching ``*_DRIVERS`` registry (see ``sim_drivers.py``) + selecting it in the
``hardware:`` section of ``robot.yaml``.  The brains never change.

Factories resolve the configured driver name (``cfg.hardware.<device>.driver``)
to a class and instantiate it with the :class:`RobotConfig`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from .robot_config import ConfigError, RobotConfig


class DriveDriver(ABC):
    """Wheel actuator: turns a unicycle command into measured wheel speeds.

    The node owns the safety logic (latched e-stop, slow-down/stop zones) and
    passes the *safety-limited* command to :meth:`set_command`; :meth:`step`
    advances the plant by ``dt`` seconds and returns the measured left/right
    wheel speeds (rad/s) as the node would read from encoders.
    """

    def __init__(self, cfg: RobotConfig):
        self._cfg = cfg

    @abstractmethod
    def set_command(self, linear_mps: float, angular_rps: float) -> None:
        """Command a unicycle velocity (safety-limited by the node)."""

    @abstractmethod
    def step(self, dt: float) -> Tuple[float, float]:
        """Advance one control step; return measured (left_rps, right_rps)."""


class LidarDriver(ABC):
    """2D LIDAR: produce one scan from the world."""

    def __init__(self, cfg: RobotConfig):
        self._cfg = cfg

    @abstractmethod
    def scan(self, origin, forward_direction, walls, obstacles=None) -> List:
        """Return a list of ray hits (see simulation.raycast.RayHit)."""


class ImuDriver(ABC):
    """IMU: report noisy angular state."""

    def __init__(self, cfg: RobotConfig):
        self._cfg = cfg

    @abstractmethod
    def read(self, theta_rad: float, angular_velocity_rps: float) -> dict:
        """Return {"yaw_rad": ..., "angular_velocity_rps": ...} with noise."""


class CameraDriver(ABC):
    """AprilTag camera: detect visible tags from the robot pose."""

    def __init__(self, cfg: RobotConfig):
        self._cfg = cfg

    @abstractmethod
    def detect(self, bot_x: float, bot_y: float, bot_theta_rad: float, tags) -> List[dict]:
        """Return a list of tag detections (see simulator's schema)."""


# ---------------------------------------------------------------------------
# Driver factories (resolve the robot.yaml "hardware" section)
# ---------------------------------------------------------------------------


def _load_driver(cfg: RobotConfig, device: str, registry: dict):
    """Instantiate the configured driver for a device, failing loudly on an
    unknown name."""
    name = getattr(getattr(cfg.hardware, device), "driver")
    try:
        driver_cls = registry[name]
    except KeyError:
        raise ConfigError(
            f"hardware.{device}.driver unknown: {name!r} "
            f"(available: {', '.join(sorted(registry))})"
        ) from None
    return driver_cls(cfg)


def load_drive_driver(cfg: RobotConfig) -> DriveDriver:
    """Load the configured drive driver (lazy import avoids a cycle)."""
    from .sim_drivers import DRIVE_DRIVERS

    return _load_driver(cfg, "drive", DRIVE_DRIVERS)


def load_lidar_driver(cfg: RobotConfig) -> LidarDriver:
    from .sim_drivers import LIDAR_DRIVERS

    return _load_driver(cfg, "lidar", LIDAR_DRIVERS)


def load_imu_driver(cfg: RobotConfig) -> ImuDriver:
    from .sim_drivers import IMU_DRIVERS

    return _load_driver(cfg, "imu", IMU_DRIVERS)


def load_camera_driver(cfg: RobotConfig) -> CameraDriver:
    from .sim_drivers import CAMERA_DRIVERS

    return _load_driver(cfg, "camera", CAMERA_DRIVERS)
