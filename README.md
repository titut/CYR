# ZenNav — a Robotics OS on Zenoh

A full robot autonomy stack — perception, localization, planning, control, and safety — built as **8 decoupled pub/sub nodes on [Eclipse Zenoh](https://zenoh.io), with no ROS 2**. The same software stack is intended to drop onto any robot hardware: sensors and actuators sit behind a YAML-configured Hardware Abstraction Layer, and the "brains" never touch a driver directly.

> This project started as a question: *what does ROS 2 actually give me, and what does it cost?* This repo is the answer — everything the framework was doing, rebuilt by hand on a thinner transport.

**Status:** working 2D/3D simulated stack (differential-drive robot, LIDAR + IMU + camera). Hardware drivers are pluggable via the HAL; the sim drivers are the reference implementation.

---

## Why not ROS 2?

I learned robotics on ROS 2 ([CYPIU](https://github.com/titut/CYPIU) is a ROS 2 package), and wanted to own the parts the framework hides:

| Concern | ROS 2 way | ZenNav way |
|---|---|---|
| Transport | DDS, rmw layer, graph daemon | Zenoh pub/sub, zero-config peer discovery |
| Message schema | `.msg` IDL, codegen at build time | Typed, **versioned** JSON registry validated on *both* publish and subscribe — malformed messages raise loudly instead of being silently dropped |
| Real-time QoS | QoS profiles | Explicit `CongestionControl.DROP` + `Reliability.BEST_EFFORT` on every real-time stream; `RingChannel(1)` on the LIDAR subscriber so the estimator only ever sees the newest scan |
| Launch | `launch` XML/Python, node lifecycle | One shell script |
| Bagging | rosbag2 | A logger node (wildcard `**` subscription → JSONL) + a replay node that republishes paced by recorded timestamps |

The payoff is a stack where the transport semantics are visible, testable, and mine — at the cost of building (and testing) them.

## Architecture

```
                 ┌────────────────────┐
                 │  3D Simulator      │  LIDAR raycast, IMU, camera
                 │  (world + physics) │  AprilTags, differential drive
                 └─────────┬──────────┘
                           │ sensor/*  (Zenoh)
     ┌─────────────┬───────┴────────┬──────────────────┐
     ▼             ▼                ▼                  ▼
┌─────────┐  ┌───────────┐  ┌──────────────┐   ┌──────────────┐
│ Apriltag│  │   Pose    │  │  Navigator   │   │  Controller  │
│ Detector│  │ Estimator │  │  (A*/RRT*)   │   │ (pure pursuit│
│         │  │ MCL + ICP │  │  + LLM nav   │   │ + teleop)    │
└─────────┘  └─────┬─────┘  └──────┬───────┘   └──────┬───────┘
                   │ estimate/pose │ nav/path         │ cmd/velocity
                   └───────┬───────┴──────────────────┘
                           ▼
                   ┌───────────────┐
                   │  Drive Node   │  motor model, encoder noise,
                   │  (safety)     │  3-tier safety zones
                   └───────────────┘

   + logger (records ** → JSONL), sim_viewer, topic_view, map_editor
```

Every box is an independent process that only knows topic names — swap any of them out (or run them across machines; Zenoh peers auto-discover) and the rest doesn't notice.

## What's inside

**Perception**
- Simulated LIDAR (raycast against polygon world), IMU with noise model, camera-based AprilTag detections at ~30 Hz
- AprilTag detector node publishing structured relative transforms

**Localization**
- Monte Carlo Localization: particle filter fusing LIDAR (beam model), AprilTags (absolute-pose Gaussian likelihood with camera noise covariance), and a heading filter that estimates gyro bias and wheel-radius scale
- Gauss-Newton ICP scan matching with a coordinate-descent fallback, gated to run only when no tag has fused recently (so ICP can't fight an active correction)
- Automatic re-seeding of the particle cloud when a measurement contradicts every hypothesis

**Planning & control**
- Footprint-aware A* (8-connected, corner-cut prevention, octile heuristic) with greedy line-of-sight smoothing; RRT* behind the same planner interface as a drop-in swap
- Regulated pure pursuit (lookahead curvature, centripetal + goal-distance speed limits), rotate-in-place at sharp corners, bang-bang recovery when off-path, teleop always takes priority

**Safety**
- Three-tier layered response modeled on IEC 61508 / ISO 3691-4 concepts: distance-scaled slow-down (< 1.0 m), stop zone (< 0.15 m), and a **latched e-stop** (< 0.05 m) that never auto-clears — requires explicit reset, allows only slow reverse so the robot can back out without re-latching

**Hardware Abstraction Layer**
- `DriveDriver`, `LidarDriver`, `ImuDriver`, `CameraDriver` interfaces; brains read/write topics only
- New hardware = implement the interface + register it + select it in `robot.yaml`. Nothing upstream changes.

**Developer tooling**
- **Recorder/replayer**: wildcard subscription logs every topic to JSONL; replay republishes with `--speed` pacing and topic filters for deterministic debugging — a hand-rolled rosbag
- `topic_view`, `sim_viewer`, and a visual `map_editor`
- **~2,700 lines of tests** across 24 files (`make test` / `make coverage`)

## Quick start

```bash
pip install -r requirements.txt

# full stack: simulator + all 8 nodes
./start.sh

# record & replay a session
python3 zenoh/logger.py &
python3 zenoh/replay.py zenoh/logs/recording_<ts>.jsonl --speed 2.0
```

Natural-language navigation (any OpenAI-compatible endpoint — DeepSeek, Ollama, vLLM):

> `nav/command`: "go to the kitchen" → LLM resolves the room → `nav/goal` → A* → drive

## Project structure

```
core/           message schema registry, HAL interfaces, robot config, clock
zenoh/          the nodes: control/, pose_estimation/, navigation/, apriltag_detection/
simulation/     2D kinematics, raycast, occupancy grid
simulation3d/   3D world, sensors, URDF assets
map_editor/     visual map editor
tests/          24 test modules
```

## See also

- **[CYPIU](https://github.com/titut/CYPIU)** — the ROS 2 robotic arm that motivated this rewrite

## Roadmap

`docs/ROADMAP.md` holds the full prioritized backlog (P0–P3) — SLAM with loop closure, dynamic obstacle tracking, costmap layering, and real-camera perception are the next milestones.
