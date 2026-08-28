"""1D heading filter: fuses the IMU gyro, wheel-odometry angular rate, and an
absolute yaw reference into a single heading estimate.

The gyro has low short-term noise but a slowly-drifting bias; the odometry
angular rate has a track-calibration *scale* error but no bias; the absolute
yaw (magnetometer-like) has noise but no drift.  None of these is good on its
own, so we fuse them.

An EKF over [heading, gyro_bias, track_scale] does the fusion:

  - predict  : integrate the bias-corrected gyro rate
  - odometry : omega_odom ~= (track_scale / wheel_scale) * (gyro - bias)
  - yaw      : yaw_rad ~= heading

The odometry angular rate shares the wheel-radius error (``wheel_scale``, fed
in from the particle filter) with the forward speed, and carries an additional
track-scale error.  Factoring ``wheel_scale`` out makes the estimated
``track_scale`` the true track scale.  The odometry measurement ties the gyro
bias to the track scale; the absolute yaw makes the heading (and hence the
bias, and hence the track scale) observable.  The filter returns the fused
heading *increment* per step, which the particle filter applies as its heading
motion.
"""

from __future__ import annotations

import numpy as np


class HeadingFilter:
    def __init__(
        self,
        initial_heading: float = 0.0,
        gyro_bias_init_std: float = 0.05,
        track_scale_init_std: float = 0.1,
        odom_rate_noise_rps: float = 0.05,
        yaw_noise_rad: float = 0.1,
        bias_process_noise_rps: float = 0.001,
        track_process_noise: float = 0.001,
        gyro_bias_max_rps: float = 0.02,
        track_scale_min: float = 0.9,
        track_scale_max: float = 1.1,
        odom_slip_threshold_rps: float = 0.5,
    ):
        # State: [heading (rad), gyro_bias (rad/s), track_scale (dimensionless)].
        self.x = np.array([initial_heading, 0.0, 1.0], dtype=np.float64)
        self.P = np.diag(
            [
                0.5 ** 2,
                gyro_bias_init_std ** 2,
                track_scale_init_std ** 2,
            ]
        )
        self.odom_rate_noise = odom_rate_noise_rps
        self.yaw_noise = yaw_noise_rad
        self.bias_process_noise = bias_process_noise_rps
        self.track_process_noise = track_process_noise
        # Physical bounds: these calibration parameters are near-constant and
        # near unity in reality (the sim's gyro bias is ±0.005 rad/s, its track
        # scale within ~±5%).  They are only loosely observable in open space,
        # so without tight bounds they random-walk and corrupt the heading (the
        # gyro bias alone was observed swinging to ±0.38 rad/s, causing 20-70°
        # of heading drift).  Clamp them to their physical range.
        self.gyro_bias_max_rps = gyro_bias_max_rps
        self.track_scale_min = track_scale_min
        self.track_scale_max = track_scale_max
        # Wheel-slip gate: when the wheel-derived angular rate disagrees with
        # the gyro by more than this, the wheels are slipping (the sim's base
        # under-rotates during in-place turns, so the wheel rate can be ~2x the
        # true rate).  In that case the odometry rate is not fused — the gyro
        # and absolute yaw are accurate and sufficient on their own.
        self.odom_slip_threshold_rps = odom_slip_threshold_rps
        self._prev_heading = None

    def step(
        self,
        dt: float,
        gyro_rate: float,
        odom_rate: float,
        yaw: float | None = None,
        wheel_scale: float = 1.0,
    ) -> float:
        """Fuse one time step and return the fused heading increment (rad).

        ``gyro_rate`` is the IMU angular velocity (rad/s), ``odom_rate`` is the
        odometry angular velocity (rad/s) derived from the wheel speeds,
        ``yaw`` is an optional absolute heading measurement (rad), and
        ``wheel_scale`` is the wheel-radius scale factor (estimated by the
        particle filter) that the odometry angular rate shares with the forward
        speed.

        The odometry model is::

            odom_rate = (track_scale / wheel_scale) * (gyro_rate - bias)

        A wheel-radius error scales the odometry angular rate the same way it
        scales the forward speed, while a track error is an *additional* scale
        factor.  Factoring out ``wheel_scale`` here makes the estimated
        ``track_scale`` the true track scale, not a track/wheel blend.
        """
        # ---- Predict: integrate the bias-corrected gyro ----
        self.x[0] += (gyro_rate - self.x[1]) * dt
        F = np.array(
            [
                [1.0, -dt, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        Q = np.diag([0.0, self.bias_process_noise ** 2 * dt, self.track_process_noise ** 2 * dt])
        self.P = F @ self.P @ F.T + Q

        # ---- Update 1: odometry angular rate ----
        # h = (track_scale / wheel_scale) * (gyro - bias);  observed = odom_rate
        # Skip when the wheels are slipping (odom rate grossly inconsistent with
        # the gyro): fusing a ~2x-wrong rate would corrupt the heading.
        ws = max(wheel_scale, 1e-3)
        ratio = self.x[2] / ws
        h = ratio * (gyro_rate - self.x[1])
        H = np.array([0.0, -ratio, (gyro_rate - self.x[1]) / ws])
        R = self.odom_rate_noise ** 2
        S = float(H @ self.P @ H.T + R)
        K = self.P @ H.T / S
        y = odom_rate - h
        if abs(y) <= self.odom_slip_threshold_rps:
            self.x += K * y
            self.P = (np.eye(3) - np.outer(K, H)) @ self.P

        # ---- Update 2: absolute yaw (if available) ----
        if yaw is not None:
            H2 = np.array([1.0, 0.0, 0.0])
            R2 = self.yaw_noise ** 2
            S2 = float(H2 @ self.P @ H2.T + R2)
            K2 = self.P @ H2.T / S2
            # Circular difference so a wrap in yaw is handled correctly.
            y2 = float(np.arctan2(np.sin(yaw - self.x[0]), np.cos(yaw - self.x[0])))
            self.x[0] += K2[0] * y2
            self.P = (np.eye(3) - np.outer(K2, H2)) @ self.P

        # Clamp the gyro bias and track scale to their physical ranges so they
        # can't diverge when loosely observed (open space).
        self.x[1] = float(
            np.clip(self.x[1], -self.gyro_bias_max_rps, self.gyro_bias_max_rps)
        )
        self.x[2] = float(
            np.clip(self.x[2], self.track_scale_min, self.track_scale_max)
        )

        # Fused heading increment for this step.
        prev = self.x[0] if self._prev_heading is None else self._prev_heading
        dtheta = float(self.x[0] - prev)
        self._prev_heading = float(self.x[0])
        return dtheta

    @property
    def heading(self) -> float:
        return float(self.x[0])

    @property
    def gyro_bias(self) -> float:
        return float(self.x[1])

    @property
    def track_scale(self) -> float:
        return float(self.x[2])
