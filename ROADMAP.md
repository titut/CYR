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
| T-006 | Real camera model + AprilTag detector | P1 | Perception | L | — |
| T-007 | Object detection / free-space segmentation | P2 | Perception | XL | T-006 |
| T-008 | 3D / 2.5D support | P3 | Perception | XL | T-017 |
| T-009 | Pose-independent obstacle detection (probabilistic costmap) | P0 | Navigation | L | — |
| T-012 | Costmap layering (static/inflation/dynamic/semantic) | P1 | Navigation | M | T-009 |
| T-013 | Dynamic obstacle detection, tracking, prediction | P1 | Navigation | XL | T-009 |
| T-014 | Recovery behaviors | P1 | Navigation | M | — |
| T-016 | Safety architecture (soft-stop tiers, latched e-stop) | P0 | Control & Safety | L | — |
| T-017 | SLAM (scan-to-submap, loop closure, pose graph) | P1 | SLAM & Maps | XL | — |
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
| T-028 | Performance: vectorize PF, native serialization | P2 | Quality | M | — |

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
- **Depends on:** —
- **Status:** Done

**Problem:** `navigator.py::_detect_obstacle_particles` renders an "expected
wall-only scan" from the *estimated* pose and flags any beam 0.3 m shorter as an
obstacle. Pose error (~0.3 m) produces ghost obstacles; the 2-hit confirmation
is a symptom patch, not a fix.

**Done when:** obstacles come from the raw point cloud registered to the map or a
probabilistic occupancy update, not from comparing against a pose-rendered
reference scan.

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
- **Depends on:** —
- **Status:** Backlog

**Problem:** the robot cannot reverse out of a dead-end or rotate to clear a
blocked local plan.

**Done when:** recovery behaviors (reverse, rotate-in-place, clear-costmap) run
before declaring failure.

---

## Epic: Motion Control & Safety

### T-016 — Safety architecture
- **Priority:** P0
- **Effort:** L
- **Depends on:** —
- **Status:** Done

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
- **Depends on:** —
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
- **Status:** Done

**Problem:** `time.time()` for durations (can jump), `time.sleep(dt)` loops
(jitter, not deadline-driven), no clock abstraction, no sim-time, no replay.

**Done when:** monotonic clocks for durations, deadline-driven loops, a clock
abstraction (wall vs sim time) enabling deterministic replay.

**Status note:** `clock.py` provides `Clock`/`WallClock`/`SimClock`,
`sleep_until` and `seed_all`.  All node main loops are deadline-driven
(`sleep_until` pacing — no cumulative drift), durations use
`time.monotonic()`, and `zenoh/replay.py` replays a recorded JSONL session back
onto Zenoh paced by its timestamps (`--speed`, `--topics`, `--dry-run`), which
the deterministic replay tests build on.

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
- **Status:** Done

**Problem:** ~6000 lines, zero tests, no CI. Regression risk on every change.

**Done when:** unit + integration tests (PF convergence, planner correctness,
deterministic replay) running in CI on every commit.

**Status note:** 83 unit + integration tests (map format, kinematics, occupancy
grid, footprint incl. the circumradius regression, A*/RRT*, particle filter,
heading filter, controller corner turns, drive safety zones, navigator
occupancy, and deterministic replay incl. a recorded-failure regression) pass
via `make test`. CI-on-every-commit is wired but deferred to GitHub Actions —
the layout is CI-ready (`requirements-dev.txt` + `pytest.ini` + `tests/`).

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
- **Depends on:** —
- **Status:** Backlog

**Problem:** PF raycast is O(particles × beams × walls) in Python loops;
`json.dumps/loads` on every 50 Hz message.

**Done when:** likelihood field or vectorized raycast (or compiled backend), and
native/binary serialization on hot paths.

---

## Suggested next work

The critical path (localization measurement model, drive model, local/global
planners, probabilistic obstacle detection, safety architecture, test suite,
clock abstraction + replay) is done.  The highest-leverage remaining work,
roughly in order:

1. **T-019** — HAL + config-driven parameters. The foundation of the
   "universal OS" north star; unlocks T-021–T-024 and T-022.
2. **T-012 + T-013** — layered costmap, then dynamic-obstacle tracking.
3. **T-027** — constant dedup (`map_format` single source of truth, no "must
   match" comments) — a cheap hygiene win.

After this, T-017 (SLAM) is the moat that separates a demo from a product.

---

## Testing

Run the suite locally with:

    make test          # or: python -m pytest
    make coverage      # pytest with coverage report

The recorded-session replay tests in `tests/test_replay.py` skip cleanly when
`zenoh/logs/*.jsonl` are absent (e.g. a fresh clone), so they never block a
clean run.  A GitHub Actions workflow can be added by installing
`requirements.txt` + `requirements-dev.txt` and running `pytest`.
