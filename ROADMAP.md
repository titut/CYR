# Robotics OS — Engineering Roadmap

> **North star:** a universal Robotics OS — the same software stack dropped onto any
> robot hardware and it works. Sensors and actuators are abstracted behind a stable
> Hardware Abstraction Layer; the navigation, perception, and control brains are
> hardware-agnostic and deterministic. The current codebase is a well-structured 2D
> localization demo on a perfect pre-surveyed map. This document turns the gap
> between that demo and the north star into a prioritized ticket backlog.

## Priority legend

| Label | Meaning |
|-------|---------|
| **P0** | Critical — safety/viability blocker. Fails the moment reality touches it. Fix first. |
| **P1** | High — core capability required to be competitive. |
| **P2** | Medium — performance, quality, scale. |
| **P3** | Low — polish / hygiene. |

## Effort legend

| Label | Rough size |
|-------|-----------|
| S | hours |
| M | days |
| L | week(s) |
| XL | month(s) |

---

## Summary

| ID | Title | Priority | Epic | Effort | Depends on |
|----|-------|----------|-------|--------|------------|
| T-001 | Likelihood-field measurement model + mixture components | P0 | Localization | L | — |
| T-002 | Pose covariance + ESS-based adaptive resampling | P0 | Localization | S | T-001 |
| T-003 | Replace particle-wipe anchors with real sensor fusion | P0 | Localization | XL | T-002 |
| T-004 | IMU yaw in motion model + odometry calibration | P1 | Localization | M | T-003 |
| T-005 | Proper delayed / out-of-order measurement handling | P1 | Localization | M | T-003 |
| T-006 | Real camera model + AprilTag detector | P1 | Perception | L | — |
| T-007 | Object detection / free-space segmentation | P2 | Perception | XL | T-006 |
| T-008 | 3D / 2.5D support | P3 | Perception | XL | T-017 |
| T-009 | Pose-independent obstacle detection (probabilistic costmap) | P0 | Navigation | L | T-002 |
| T-010 | Kinematic-aware global planner + path smoothing | P1 | Navigation | M | — |
| T-011 | Local planner + trajectory generation | P0 | Navigation | L | T-010 |
| T-012 | Costmap layering (static/inflation/dynamic/semantic) | P1 | Navigation | M | T-009 |
| T-013 | Dynamic obstacle detection, tracking, prediction | P1 | Navigation | XL | T-009 |
| T-014 | Recovery behaviors | P1 | Navigation | M | T-011 |
| T-015 | Drive node: acceleration limits + velocity-loop model | P0 | Control & Safety | S | — |
| T-016 | Safety architecture (soft-stop tiers, latched e-stop) | P0 | Control & Safety | L | T-015 |
| T-017 | SLAM (scan-to-submap, loop closure, pose graph) | P1 | SLAM & Maps | XL | T-001, T-002 |
| T-018 | Multi-session / metric-semantic maps | P2 | SLAM & Maps | L | T-017 |
| T-019 | Hardware Abstraction Layer + config-driven parameters | P1 | Universal OS | L | — |
| T-020 | Deterministic clock + sim-time/replay | P1 | Universal OS | M | T-019 |
| T-021 | Typed, versioned message schema | P1 | Universal OS | M | — |
| T-022 | Cross-platform build/deploy (ARM/Jetson) | P2 | Universal OS | M | T-019 |
| T-023 | Orchestration, supervisor, watchdog, health monitor | P1 | Reliability | M | — |
| T-024 | Structured logging, telemetry, latency, replay | P1 | Reliability | M | T-020 |
| T-025 | Safety/security process (hazard analysis, secrets) | P2 | Reliability | M | T-016 |
| T-026 | Test suite + CI | P1 | Quality | M | T-020 |
| T-027 | Dedupe constants + `map_format` single source of truth | P2 | Quality | S | — |
| T-028 | Performance: vectorize PF, native serialization | P2 | Quality | M | T-001 |
| T-029 | Ground + validate LLM nav targets | P1 | LLM Navigation | S | — |

---

## Epic: Localization & State Estimation

### T-001 — Likelihood-field measurement model + mixture components
- **Priority:** P0
- **Effort:** L
- **Depends on:** —
- **Status:** Done

**Problem:** `particle_filter.py::update` uses the classic *beam model*:
`exp(-diff²/2σ²)` plus a `0.001·max` floor. It has no free-space model, no
obstacle component, no max-range component, and no uniform/random component.
One unmodeled object (a chair, a person) produces a beam distance the correct
particle can't explain and zeroes its weight, collapsing the filter to a wrong
pose.

**Done when:** measurement likelihood is a likelihood-field (or equivalently a
mixture: Gaussian hit + obstacle + max-range + uniform), weights never go to
exactly zero, and the filter stays converged with a moving obstacle in the room.

### T-002 — Pose covariance + ESS-based adaptive resampling
- **Priority:** P0
- **Effort:** S
- **Depends on:** T-001
- **Status:** Done

**Problem:** `estimate/pose` publishes no covariance. Resampling runs every scan
(`pose_estimator.py`) with no effective-sample-size check, so the system can't
detect divergence, gate measurements, or tell the user "I'm lost."

**Done when:** covariance is computed and published with every estimate, and
resampling is skipped when ESS is above threshold.

### T-003 — Replace particle-wipe anchors with real sensor fusion
- **Priority:** P0
- **Effort:** XL
- **Depends on:** T-002
- **Status:** Done

**Problem:** AprilTag anchoring is `anchor_fraction(..., fraction=1.0)` — a hard
reset that discards the whole belief state and slams to a single delayed,
noisy measurement with hardcoded `std_xy=0.25, std_theta=0.05`.

**Done when:** tag detections are fused as measurements (Kalman update or factor
graph, e.g. gtsam) with the tag's *actual* covariance, preserving the prior.

### T-004 — IMU yaw in motion model + odometry calibration
- **Priority:** P1
- **Effort:** M
- **Depends on:** T-003
- **Status:** Done

**Problem:** `sensor/imu` yaw is published but never consumed; the motion model
is pure wheel odometry, which drifts unboundedly from wheelbase/track error.
`simulator.py` injects slip bias but there is no online calibration.

**Done when:** heading is fused from IMU (gyro) + odometry, and wheel radius /
track are estimated (or calibrated) rather than assumed perfect.

### T-005 — Proper delayed / out-of-order measurement handling
- **Priority:** P1
- **Effort:** M
- **Depends on:** T-003
- **Status:** Done

**Problem:** delayed tags are handled by summing a 2 s odometry ring buffer and
one `predict()` (`pose_estimator.py`). This ignores odometry covariance growth
and doesn't generalize to out-of-order measurements.

**Done when:** measurements carry timestamps and are retroactively fused via
buffered states / preintegrated odometry factors (correct time-travel), not a
raw-sum hack.

---

## Epic: Perception

### T-006 — Real camera model + AprilTag detector
- **Priority:** P1
- **Effort:** L
- **Depends on:** —
- **Status:** Backlog

**Problem:** `detector.py` is a polar→cartesian lookup with a `time.sleep`
latency. No intrinsics/distortion, no PnP, no real detector, no 6-DOF pose with
covariance, no outlier rejection. `yaw_rel` is simulator truth.

**Done when:** an actual detector (e.g. apriltag3) + calibrated camera model
produces SE(3) detections with covariance, and the blocking `sleep` is removed
(detection runs in its own thread/queue).

### T-007 — Object detection / free-space segmentation
- **Priority:** P2
- **Effort:** XL
- **Depends on:** T-006
- **Status:** Backlog

**Problem:** no notion of what's in front of the robot beyond walls and AprilTags.

**Done when:** objects are detected and segmented (2.5D) to feed a semantic
costmap.

### T-008 — 3D / 2.5D support
- **Priority:** P3
- **Effort:** XL
- **Depends on:** T-017
- **Status:** Backlog

**Problem:** everything is planar; doors, stairs, ramps, overhangs don't exist.

**Done when:** elevation and overhangs are represented and avoided in planning.

---

## Epic: Navigation & Planning

### T-009 — Pose-independent obstacle detection (probabilistic costmap)
- **Priority:** P0
- **Effort:** L
- **Depends on:** T-002
- **Status:** Backlog

**Problem:** `navigator.py::_detect_obstacle_particles` renders an "expected
wall-only scan" from the *estimated* pose and flags any beam 0.3 m shorter as an
obstacle. Pose error (~0.3 m) produces ghost obstacles; the 2-hit confirmation
is a symptom patch, not a fix.

**Done when:** obstacles come from the raw point cloud registered to the map or a
probabilistic occupancy update, not from comparing against a pose-rendered
reference scan.

### T-010 — Kinematic-aware global planner + path smoothing
- **Priority:** P1
- **Effort:** M
- **Depends on:** —
- **Status:** Done

**Problem:** A* is 4-connected (stair-stepped paths), RRT* returns raw tree
segments with no smoothing, and neither respects nonholonomic constraints.

**Done when:** 8-connected/lattice/Hybrid-A* planner produces curvature-feasible
paths that are smoothed (shortcut/spline) before execution.

### T-011 — Local planner + trajectory generation
- **Priority:** P0
- **Effort:** L
- **Depends on:** T-010
- **Status:** Done

**Problem:** `controller.py::_follow_path` is turn-in-place-then-drive-straight
bang-bang: no trajectory, no velocity profile, no accel/jerk limits, no
lookahead. It tracks waypoints, not the path, and clips obstacles.

**Done when:** a local planner (pure-pursuit/MPC/DWA) tracks the path with a
smooth time-parameterized velocity profile respecting accel/jerk limits.

### T-012 — Costmap layering
- **Priority:** P1
- **Effort:** M
- **Depends on:** T-009
- **Status:** Backlog

**Problem:** the "costmap" is a binary hit-count dict rebuilt from scratch every
0.25 s; no static layer + inflation + dynamic + semantic layers.

**Done when:** a layered costmap with inflation gradients feeds the planners.

### T-013 — Dynamic obstacle detection, tracking, prediction
- **Priority:** P1
- **Effort:** XL
- **Depends on:** T-009
- **Status:** Backlog

**Problem:** moving objects are treated as static clutter; no tracks, no
velocity, no time-to-collision.

**Done when:** moving obstacles are tracked and predicted (or at minimum trigger
timely avoidance) rather than being treated as static cells.

### T-014 — Recovery behaviors
- **Priority:** P1
- **Effort:** M
- **Depends on:** T-011
- **Status:** Backlog

**Problem:** the robot cannot reverse out of a dead-end or rotate to clear a
blocked local plan.

**Done when:** recovery behaviors (reverse, rotate-in-place, clear-costmap) run
before declaring failure.

---

## Epic: Motion Control & Safety

### T-015 — Drive node: acceleration limits + velocity-loop model
- **Priority:** P0
- **Effort:** S
- **Depends on:** —
- **Status:** Done

**Problem:** `drive.py` models the motor driver as a first-order velocity loop
with no explicit torque/acceleration limit, so a step change in `cmd/velocity`
produces a physically-impossible near-instantaneous velocity change.  Velocity
PID itself is deliberately *out of scope* for the OS — it lives in the motor
driver firmware / HAL (see T-019); this ticket only models a velocity-controlled
driver accurately.

**Done when:** `drive.py` enforces an explicit per-wheel acceleration limit
(torque-limited slew) on top of the velocity-loop time constant, while keeping
the unicycle→wheel conversion, command timeout, and e-stop.

### T-016 — Safety architecture
- **Priority:** P0
- **Effort:** L
- **Depends on:** T-015
- **Status:** Backlog

**Problem:** e-stop sets wheel speed to zero instantly (infinite deceleration),
doesn't latch or zero the command, and rides on best-effort LIDAR from a perfect
simulator. No slow-down tier, no safety-rated channel.

**Done when:** slow-down zone → stop zone → latched e-stop requiring reset, on a
safety-rated channel (documented IEC 61508 / ISO 3691-4 path for hardware).

---

## Epic: SLAM & Maps

### T-017 — SLAM (scan-to-submap, loop closure, pose graph)
- **Priority:** P1
- **Effort:** XL
- **Depends on:** T-001, T-002
- **Status:** Backlog

**Problem:** map is assumed known and static. No loop closure, no pose graph, no
submap, no relocalization.

**Done when:** the system builds/maintains its own map (Cartographer-style
scan-to-submap) and relocalizes after kidnapping.

### T-018 — Multi-session / metric-semantic maps
- **Priority:** P2
- **Effort:** L
- **Depends on:** T-017
- **Status:** Backlog

**Problem:** single map, no persistence, no semantic layer beyond room names.

**Done when:** maps persist across sessions with metric-semantic labels.

---

## Epic: Universal OS (Hardware Abstraction)

### T-019 — Hardware Abstraction Layer + config-driven parameters
- **Priority:** P1
- **Effort:** L
- **Depends on:** —
- **Status:** Backlog

**Problem:** the code is hardwired to one sim (constants like `BOT_SIZE_M`,
`LIDAR_MAX_RANGE_M` copy-pasted with "must match" comments). This is the core
enabler for the north star — same software on any hardware.

**Done when:** standardized sensor/actuator interfaces (driver model) with a
configuration layer; zero hardcoded robot constants; a new robot is added by
writing a driver + config, not by editing the brains.

### T-020 — Deterministic clock + sim-time/replay
- **Priority:** P1
- **Effort:** M
- **Depends on:** T-019
- **Status:** Backlog

**Problem:** `time.time()` for durations (can jump), `time.sleep(dt)` loops
(jitter, not deadline-driven), no clock abstraction, no sim-time, no replay.

**Done when:** monotonic clocks for durations, deadline-driven loops, a clock
abstraction (wall vs sim time) enabling deterministic replay.

### T-021 — Typed, versioned message schema
- **Priority:** P1
- **Effort:** M
- **Depends on:** —
- **Status:** Backlog

**Problem:** ad-hoc JSON with no schema/versioning; callbacks do
`except (json.JSONDecodeError, KeyError, Exception)` swallowing every error
silently.

**Done when:** typed, versioned messages (or a schema registry) with explicit
error handling; malformed messages are logged, not silently dropped.

### T-022 — Cross-platform build/deploy
- **Priority:** P2
- **Effort:** M
- **Depends on:** T-019
- **Status:** Backlog

**Problem:** no packaging or deployment story for embedded targets (Jetson/ARM).

**Done when:** reproducible builds/containers for target platforms.

---

## Epic: Reliability & Observability

### T-023 — Orchestration, supervisor, watchdog, health monitor
- **Priority:** P1
- **Effort:** M
- **Depends on:** —
- **Status:** Backlog

**Problem:** `start.sh` launches raw background processes; if any node dies the
rest silently degrade with no restart or alert.

**Done when:** a supervisor/launch system manages node lifecycle with
watchdog/health checks and auto-restart.

### T-024 — Structured logging, telemetry, latency, replay
- **Priority:** P1
- **Effort:** M
- **Depends on:** T-020
- **Status:** Backlog

**Problem:** `print()`-based logging, no telemetry, no end-to-end latency
measurement, no rosbag-equivalent recording.

**Done when:** structured logs + metrics + latency instrumentation + record/replay
of all sensor/command streams.

### T-025 — Safety/security process
- **Priority:** P2
- **Effort:** M
- **Depends on:** T-016
- **Status:** Backlog

**Problem:** no hazard analysis (STPA/FMEA), no threat model, no safety case.

**Done when:** hazard analysis and threat model documented; secrets handled
uniformly; safety requirements traceable to code.

---

## Epic: Quality & Tooling

### T-026 — Test suite + CI
- **Priority:** P1
- **Effort:** M
- **Depends on:** T-020
- **Status:** Backlog

**Problem:** ~6000 lines, zero tests, no CI. Regression risk on every change.

**Done when:** unit + integration tests (PF convergence, planner correctness,
deterministic replay) running in CI on every commit.

### T-027 — Dedupe constants + `map_format` single source of truth
- **Priority:** P2
- **Effort:** S
- **Depends on:** —
- **Status:** Backlog

**Problem:** `map_format.py` and `map_editor/map_format.py` are identical
234-line copies; `LIDAR_MAX_RANGE_M` / `BOT_SIZE_M` duplicated across 3-4 files.

**Done when:** one canonical `map_format` and one constants module; no "must
match" comments.

### T-028 — Performance: vectorize PF, native serialization
- **Priority:** P2
- **Effort:** M
- **Depends on:** T-001
- **Status:** Backlog

**Problem:** PF raycast is O(particles × beams × walls) in Python loops;
`json.dumps/loads` on every 50 Hz message.

**Done when:** likelihood field or vectorized raycast (or compiled backend), and
native/binary serialization on hot paths.

---

## Epic: LLM Navigation

### T-029 — Ground + validate LLM nav targets
- **Priority:** P1
- **Effort:** S
- **Depends on:** —
- **Status:** Backlog

**Problem:** single prompt, no validation that the returned (x,y) is free-space
or reachable, no fallback; a hallucinated coordinate drives the robot into a
wall. Also `query_location` defaults to `deepseek-v4-pro` while
`query_location_async` defaults to `deepseek-chat` (inconsistent).

**Done when:** LLM output is validated against the map (free-space + reachability)
with a fallback, and the model name is a single shared config.

---

## Suggested first sprint (the critical path)

1. **T-001 + T-002** — measurement model + covariance/ESS. Highest leverage, stays
   inside the existing node design.
2. **T-015** — drive acceleration limits + velocity-loop model (prerequisite for any real motion).
3. **T-003** — real sensor fusion, eliminate particle wipes.
4. **T-011 + T-010** — local planner + trajectory, then a smoothed global planner.
5. **T-019 + T-020** — HAL + determinism. This is the foundation of the
   "universal OS" north star and unlocks everything downstream (T-021–T-024, T-026).
6. **T-009** — pose-independent obstacle detection.

After this, T-017 (SLAM) is the moat that separates a demo from a product.
