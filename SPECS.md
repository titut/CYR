# Specifications

> Source: `ROADMAP.md`.
> These specs define acceptance criteria and implementation constraints for
> prioritized backlog tickets.

## T-001 — Likelihood-field measurement model + mixture components

- **Priority:** P0
- **Effort:** L
- **Depends on:** —
- **Epic:** Localization

### Objective
Replace the classic beam model in `particle_filter.py::update` with a
likelihood-field (or mixture) measurement model so a single unmodeled obstacle
cannot zero a correct particle's weight and collapse the filter.

### Current behavior
`particle_filter.py::update` scores beams with `exp(-diff²/2σ²)` plus a
`0.001·max` floor. It has:
- no free-space model,
- no obstacle component,
- no max-range component,
- no uniform/random component.

A chair or person in the beam path produces a distance no correct particle can
explain, zeroing its weight and driving the filter to a wrong pose.

### Requirements
1. Measurement likelihood is a likelihood-field model, or equivalently a
   mixture of components:
   - Gaussian **hit** component (expected-range match),
   - **obstacle** component (short-range return from unmapped object),
   - **max-range** component (no return within sensor range),
   - **uniform/random** component (outlier robustness).
2. Weights are never allowed to reach exactly zero; the uniform component
   guarantees a nonzero floor.
3. Component weights are configurable and normalize to 1.
4. The likelihood field must be precomputable from the static map for O(1)
   per-beam lookup (no per-particle raycasting against walls).

### Acceptance criteria
- With a moving obstacle in the room, the filter stays converged on the true
  pose.
- No particle weight reaches exactly zero.
- Regression: filter still converges on a clean, obstacle-free map at least as
  fast as the current beam model.

---

## T-002 — Pose covariance + ESS-based adaptive resampling

- **Priority:** P0
- **Effort:** S
- **Depends on:** T-001
- **Epic:** Localization

### Objective
Publish pose covariance with every estimate and gate resampling on the
effective sample size so the system can detect divergence, gate measurements,
and report "I'm lost."

### Current behavior
- `estimate/pose` publishes no covariance.
- Resampling runs every scan (`pose_estimator.py`) with no effective-sample-size
  check.

### Requirements
1. Compute the pose covariance (x, y, θ) from the weighted particle set and
   publish it with every `estimate/pose` message.
2. Compute the effective sample size (ESS); skip resampling when ESS is above a
   configurable threshold.
3. When ESS falls below threshold, resample proportionally to weights and reset
   weights to uniform.
4. Expose the covariance so downstream nodes can gate measurements / detect
   divergence.

### Acceptance criteria
- Every published estimate carries a valid covariance matrix.
- Resampling is skipped when ESS > threshold (verified by instrumentation/logs).
- Divergence (ESS collapse) is observable from the published estimate.

---

## T-003 — Replace particle-wipe anchors with real sensor fusion

- **Priority:** P0
- **Effort:** XL
- **Depends on:** T-002
- **Epic:** Localization

### Objective
Fuse AprilTag detections as measurements rather than hard-resetting the belief
state to a single delayed, noisy measurement.

### Current behavior
AprilTag anchoring is `anchor_fraction(..., fraction=1.0)` — a hard reset that
discards the entire belief state and slams to a single delayed, noisy
measurement with hardcoded `std_xy=0.25, std_theta=0.05`.

### Requirements
1. Tag detections are fused as measurements (Kalman update or factor graph,
   e.g. gtsam), preserving the prior belief instead of discarding it.
2. The tag's *actual* covariance (from the detector) is used, not hardcoded
   values.
3. A correct tag fusion is a soft update: the posterior is a weighted blend of
   prior and measurement, not a reset.
4. `anchor_fraction(..., fraction=1.0)` particle-wipe path is removed or
   replaced with the fusion update.

### Acceptance criteria
- A tag detection no longer collapses the particle distribution to the single
  detection pose.
- Posterior preserves information from the prior; repeated detections converge
  gradually rather than teleporting.
- Covariance from the detector is propagated into the fused estimate.

---

## T-004 — IMU yaw in motion model + odometry calibration

- **Priority:** P1
- **Effort:** M
- **Depends on:** T-003
- **Epic:** Localization

### Objective
Fuse heading from IMU (gyro) + odometry in the motion model, and estimate or
calibrate wheel radius / track instead of assuming them perfect.

### Current behavior
- `sensor/imu` yaw is published but never consumed.
- The motion model is pure wheel odometry, which drifts unboundedly from
  wheelbase/track error.
- `simulator.py` injects slip bias but there is no online calibration.

### Requirements
1. Consume `sensor/imu` yaw and fuse it with odometry to produce the heading
   used by the motion model.
2. Estimate or calibrate wheel radius and track (online or offline) rather than
   treating them as known constants.
3. The motion model accounts for odometry noise growth (non-constant covariance)
   when propagating particles.
4. Slip bias injected by the simulator must not cause unbounded heading drift.

### Acceptance criteria
- Heading is derived from IMU gyro + odometry fusion, not odometry alone.
- Wheel radius / track are calibrated, not assumed perfect.
- Heading drift is bounded under simulated slip bias.

---

## T-005 — Proper delayed / out-of-order measurement handling

- **Priority:** P1
- **Effort:** M
- **Depends on:** T-003
- **Epic:** Localization

### Objective
Handle delayed and out-of-order measurements correctly via retroactive fusion
rather than a raw odometry-sum hack.

### Current behavior
Delayed tags are handled by summing a 2 s odometry ring buffer and one
`predict()` in `pose_estimator.py`. This ignores odometry covariance growth and
does not generalize to out-of-order measurements.

### Requirements
1. Measurements carry timestamps.
2. Delayed measurements are retroactively fused via buffered states and/or
   preintegrated odometry factors (correct time-travel), not a raw-sum hack.
3. Odometry covariance growth over the delay window is accounted for.
4. The mechanism generalizes to out-of-order measurements (not just a fixed
   2 s tag delay).

### Acceptance criteria
- A delayed measurement is fused at its original timestamp with correct
  covariance, not approximated by a single `predict()`.
- Out-of-order measurements (older than the latest processed) are handled
  without corrupting the current estimate.
- No fixed-time raw-sum workaround remains in `pose_estimator.py`.

---

## T-010 — Kinematic-aware global planner + path smoothing

- **Priority:** P1
- **Effort:** M
- **Depends on:** —
- **Epic:** Navigation

### Objective
Produce globally consistent, kinematically feasible paths instead of
stair-stepped grid paths or raw randomized-tree segments that ignore the
robot's turning limits.

### Current behavior
- Global search is 4-connected, producing stair-stepped paths that the robot
  cannot follow directly.
- Randomized planning returns raw tree segments with no smoothing.
- Neither approach respects nonholonomic (curvature / minimum turning radius)
  constraints.

### Requirements
1. Global search uses an 8-connected grid, a lattice, or Hybrid-A*, so diagonal
   and kinematically meaningful transitions are available (no 4-connected
   stair-stepping).
2. Search respects the robot's nonholonomic constraints (minimum turning radius
   / maximum curvature).
3. The raw search output is post-processed — shortcut and/or spline smoothing —
   before it is handed to execution.
4. The planner emits a continuous, collision-free path, not raw tree segments.

### Acceptance criteria
- Output paths contain no stair-step artifacts.
- Paths respect a defined maximum curvature / minimum turning radius.
- Smoothed paths remain collision-free and are no longer than the raw path.

---

## T-011 — Local planner + trajectory generation

- **Priority:** P0
- **Effort:** L
- **Depends on:** T-010
- **Epic:** Navigation

### Objective
Track the global path with a smooth, time-parameterized trajectory rather than
bang-bang waypoint chasing.

### Current behavior
- Path following is turn-in-place-then-drive-straight (bang-bang).
- There is no trajectory, no velocity profile, no acceleration/jerk limits, and
  no lookahead.
- The follower tracks discrete waypoints, not the underlying path, and clips
  obstacles.

### Requirements
1. A local planner (pure-pursuit, MPC, or DWA) follows the global path.
2. The planner produces a time-parameterized trajectory with a smooth velocity
   profile (not bang-bang).
3. Acceleration and jerk limits are respected.
4. Lookahead is used so the planner tracks the path itself, not discrete
   waypoints.
5. Obstacles are avoided while tracking (no clipping).

### Acceptance criteria
- No turn-in-place-then-drive-straight behavior remains.
- Velocity commands are continuous and stay within acceleration/jerk limits.
- The trajectory tracks the path with bounded cross-track error.
- The robot does not collide with obstacles while following.

---

## T-015 — Closed-loop velocity control

- **Priority:** P0
- **Effort:** M
- **Depends on:** —
- **Epic:** Control & Safety

### Objective
Drive wheel speeds to match the commanded velocity in closed loop, with
feedforward and explicit limits, instead of issuing open-loop commands.

### Current behavior
- The drive layer is a first-order lag with a fixed assumed time step.
- There is no closed-loop velocity control, no feedforward, no acceleration or
  torque limits, and no deadband.
- Velocity commands are issued open-loop.

### Requirements
1. Cascaded velocity control (outer velocity loop, inner wheel/motor loop) with
   feedforward.
2. Acceleration and torque limits are enforced (no instantaneous step changes).
3. A deadband prevents integrator windup and oscillation around the setpoint.
4. Control uses measured time, not a fixed assumed time step.
5. Measured wheel speeds track the commanded velocity.

### Acceptance criteria
- Wheel speeds converge to and track the commanded velocity.
- Commands respect acceleration/torque limits (no infinite-rate changes).
- A deadband is present; no chattering at rest.
- Timing is derived from measured elapsed time, not a fixed assumption.
