# Mobile Manipulation Simulation — 3D / PyBullet Spec

> **Goal:** replace the 2D pygame simulator with a 3D PyBullet simulator that runs
> the same zenoh node stack (drive, controller, pose estimator, navigator, LLM) and
> adds a mycobot 280 arm with a two-finger gripper. The robot must be able to drive
> to an object, pick it up with the arm, drive to a destination, and drop it.
>
> This document is the build plan. Nothing here is implemented yet. It follows the
> ROADMAP.md ticket style: every task has a problem, a precise spec, and a "done
> when" acceptance bar.

---

## 1. Overview & architecture

Single PyBullet world (gravity, friction, contacts) simulates:

- **Mobile base** — 0.75 m square (matches `BOT_SIZE_M`) with two differential
  wheels + caster, driven by the existing drive/controller/navigator stack over
  zenoh.
- **Arm** — the mycobot 280 arm rebuilt as primitive-only (`mycobot_280_pi_3d.urdf`,
  T3D-02) with inertials and a two-finger gripper, mounted rigidly on top of the base.
- **Objects** — cube/cylinder/ball bodies that the arm can grasp.
- **Environment** — ground plane + optional walls from the existing map format.

```
┌───────────────────────────────────────────────────────────────┐
│ simulation3d/ (PyBullet world, authoritative physics)         │
│   base body + wheels        arm (mycobot + gripper)  objects  │
│   └─ raycast LIDAR, IMU, camera  └─ FK/IK, grasp constraints │
└───────────────────────────┬───────────────────────────────────┘
                     zenoh (JSON, existing ad-hoc schema)
     ┌─────────┬───────────┬────────────┬───────────────┬─────────┐
 drive.py   controller   pose_est   navigator.py   llm_nav.py   arm_controller.py
 (existing)  (existing)   (existing)  (existing)     (existing)  (NEW)
```

The old `simulation/simulator.py` is kept only as a legacy 2D debugging view until
parity is verified (Task 11), then retired.

### Key reuse (do not re-implement)

| Asset | Where | Reuse |
|-------|-------|-------|
| Wheel/unicycle model | `simulation/kinematics.py` | `wheel_to_unicycle` / `unicycle_to_wheel` unchanged |
| Base dimensions | `robot.yaml` / `core/robot_config.py` (T-019) | `chassis.*` — footprint, wheel radius/track |
| Clock / deadline loop | `core/clock.py` | `Clock.sleep_until` for the 60 Hz loop (no drift) |
| Map format | `core/map_format.py` | `MapData`, `Wall`, `Obstacle` for the 3D arena |
| LIDAR/IMU/AprilTag realism constants | `robot.yaml` (T-019) | `sensors.*` — same noise/bias + schemas |
| Zenoh publisher/subscriber pattern | `simulation/simulator.py` | same topic names + JSON schemas |

---

## 2. Dependencies & project layout

### Dependency
- Add `pybullet>=3.2.6` to `requirements.txt`.

### New files

```
simulation3d/
  __init__.py
  urdf_assets.py     # emits base.urdf + extended arm URDF text
  world.py           # ground, walls, objects, spawn bookkeeping
  simulator.py       # main loop: physics step, teleop, zenoh bridge
  sensors.py         # LIDAR raycast, IMU, camera/AprilTag from world state
  grasp.py           # gripper position control + grasp constraint lifecycle
  arm_kinematics.py  # FK (PyBullet getLinkState) + IK wrapper
zenoh/control/
  arm_controller.py  # NEW node: FK/IK, pick/place state machine, zenoh arm/*
urdf/
  mycobot_280_pi_3d.urdf  (primitive-only arm: inertials + gripper, T3D-02)
  base.urdf             (new)
  objects/{cube,cylinder,ball}.urdf   (new)
sim3d.sh              # launcher (replaces `python3 simulation/simulator.py`)
tests/
  test_3d_base.py, test_3d_arm.py, test_3d_grasp.py, test_3d_lidar.py
  test_3d_arm_controller.py
```

Robot-specific values (base size, wheel geometry, LIDAR specs, arm mount
offset, joint limits) come from `robot.yaml` via `core/robot_config.py` (T-019) —
the 3D sim is another config-driven backend, so it must not introduce its own
`config.py` constants module. The arm section of `robot.yaml` gets a `joints`
map (name → {lower, upper, effort}) that T3D-08's IK reads.

---

## 3. Zenoh topic contract

Keep every existing topic byte-compatible with the old sim so drive/controller/
pose_estimator/navigator/llm_nav run unchanged.

| Topic | Direction | Schema (JSON) | Rate |
|-------|-----------|---------------|------|
| `sensor/lidar` | pub (sim) | `{"t", "rays": [{angle_rad, distance_m}]}` — 360 rays, 10 m max | 50 Hz |
| `sensor/imu` | pub (sim) | `{"t", yaw_rad, pitch_rad, roll_rad, angular_velocity_rps, linear_acceleration_mps2:{x,y,z}}` | 50 Hz |
| `sensor/wasd` | pub (sim) | `{"w","a","s","d": bool}` | 60 Hz |
| `sensor/camera/apriltag` | pub (sim) | `{"t", "detections":[{id, range_m, bearing_rad, tag_yaw_rad, tag_size_m}]}` | on change |
| `sensor/wheel_speed` | sub (sim) | `{"left_rps","right_rps","t"}` | 50 Hz |
| `nav/goal`, `nav/command`, `safety/reset`, `nav/path`, `estimate/pose`, `detection/obstacles` | unchanged | unchanged | — |
| `sim/truth/pose` | pub (sim) | `{"t","x_m","y_m","theta_rad"}` — ground truth for tests/verification | 50 Hz |
| `arm/joint_states` | pub (controller) | `{"t","names":[...],"position":[...]}` | 50 Hz |
| `arm/ee_pose` | pub (controller) | `{"t","frame":"g_base","position":[x,y,z],"orientation":{"quat":[w,x,y,z]},"euler_rpy":[r,p,y]}` | 50 Hz |
| `arm/command` | sub (controller) | see §7.3 | — |
| `arm/state` | pub (controller) | `{"t","id","state":"busy"|"ok"|"failed","detail"}` — ack per command `id` | per op |
| `object/registry` | pub (sim) | `{"t","objects":[{id,label,shape,size_m,mass_kg,pose:{x,y,z,rpy}}]}` | on spawn + every 1 s |

---

## 4. Task list (summary)

| ID | Title | Priority | Effort | Depends on |
|----|-------|----------|--------|------------|
| T3D-01 | Base URDF | P1 | S | — |
| T3D-02 | Arm URDF: inertials + gripper + primitive collisions | P1 | S | — |
| T3D-03 | Environment, objects, map import | P1 | M | T3D-01 |
| T3D-04 | Physics loop + sim core (Direct/GUI, teleop) | P1 | M | T3D-01, T3D-02, T3D-03 |
| T3D-05 | Sensor bridge: LIDAR / IMU / camera | P1 | M | T3D-04 |
| T3D-06 | Base drive + navigation integration | P1 | M | T3D-04, T3D-05 |
| T3D-07 | Arm FK + joint-state publishing | P1 | S | T3D-02 |
| T3D-08 | IK + arm controller node | P1 | L | T3D-07 |
| T3D-09 | Grasp mechanics (pick/lift/carry/drop) | P1 | L | T3D-08 |
| T3D-10 | LLM semantic pick-and-place | P2 | M | T3D-09 |
| T3D-11 | Launch/orchestration + sim retirement | P2 | S | T3D-04…T3D-09 |
| T3D-12 | Test suite + CI (headless) | P1 | M | T3D-01…T3D-09 |

---

## 5. T3D-01 — Base URDF

**Problem:** there is no URDF for the mobile base; the 2D sim only has a drawn
square + kinematic wheel model.

**Spec (`urdf/base.urdf`, generated by `simulation3d/urdf_assets.py`):**

- Coordinate convention: x forward, y left, z up (matches body frame in
  `kinematics.py`). All lengths in metres, mass in kg.
- **chassis link:** box `0.75 × 0.75 × 0.15` (`BOT_SIZE_M × BOT_SIZE_M ×
  BASE_HEIGHT_M`), `mass=12.0`, COM at geometric centre. Inertia
  `I=m/12·(w²+h²)`, `I=m/12·(l²+h²)`, `I=m/12·(l²+w²)` on the principal axes.
  Friction: `lateralFriction=1.0`, `spinningFriction=0.05`.
- **Wheels (4, at the corners):** all four cylinders `radius=0.12`, `width=0.10`,
  `mass=0.4`, at `z=0.12` (bottom at ground). The **rear pair** (`x=-corner_x`)
  is **driven** (`wheel_drive_L`/`wheel_drive_R`, fixed wheels rolling forward);
  the **front pair** (`x=+corner_x`) are **swivelling casters**
  (`wheel_free_L`/`wheel_free_R` — each a fork that rotates about the vertical,
  carrying a free-rolling wheel) so the base can turn without the wheels
  scrubbing. Corner offsets from config: `corner_y = WHEEL_TRACK_M/2`,
  `corner_x = WHEELBASE_M/2`.
  - **Wheel axis note:** PyBullet reads the joint `<axis>` in the *child* frame.
    The wheel cylinder axis lies along the child Z, so the joint axis must be
    `0 0 -1` (not `0 1 0`) for the wheel to spin about its horizontal axle and
    roll forward. (The original `0 1 0` made the wheels spin about the vertical —
    the "wrong axis" the drive-direction bug came from.)
- **arm_mount link:** small box `0.10 × 0.10 × 0.01`, `mass=0.1`, fixed to the
  top of the chassis at `xyz=0 0 +0.075` (chassis frame is at its centre). The
  arm's `g_base` frame is constrained identity to `arm_mount` (see T3D-02).
- Chassis resting height: wheel bottom at `z=0`; chassis centre `z=0.195`;
  `arm_mount` top / arm `g_base` origin at **`z=0.27`**.
- Wheel geometry (radius/width, drive track, front/back wheelbase, chassis
  height/mass) all come from `robot.yaml` (T-019) — a different base is a config
  edit + regenerate. The base is spawned at the map's first-room centre so it
  starts inside the map (T3D-04).

**Done when:** URDF parses (`check_urdf` clean), loads in PyBullet DIRECT mode
with zero warnings, robot rests stable on the ground (no sinking, no tipping) for
10 s at fixed timestep with gravity.

> **Note:** the front wheels are swivelling casters (not fixed) so the
> differential base can drive straight and turn without scrubbing; the rear
> wheels are the driven pair. Raw PyBullet cylinder traction is imperfect
> (the base drives with some speed wobble), but T3D-01 guarantees a stable,
> driveable body.

---

## 6. T3D-02 — Arm URDF: inertials + gripper + primitive collisions

**Problem:** the original `mycobot_280_pi.urdf` had (a) **no `<inertial>` on any
link** — PyBullet assigned every link a default mass, (b) **no gripper** — it
ended at `joint6_flange`, (c) full 3D COLLADA meshes that **PyBullet cannot
load** (the Rhinoceros `.dae` files fail mesh extraction), and (d) inconsistent
mesh units (some mm, some m).

**Decision (per the build session): primitive-only.** The arm URDF is rebuilt
from scratch with a representative box per link (visual + collision) and box
inertials, so it loads and renders reliably in PyBullet without any mesh files.

**Spec:**

1. **`generate_arm_urdf(cfg)` in `simulation3d/urdf_assets.py`** rebuilds the
   arm as pure primitives: a box `<visual>` + `<collision>` per link (the
   per-link sizes/poses were baked once from the original COLLADA AABBs,
   mm→m, into the `_ARM_PRIMITIVES` table), plus an `<inertial>` on every
   link with the T3D-02 masses (`g_base: 0.8` … `joint6_flange: 0.1` kg) and
   box inertia `m/12·(w²+d²)` at each box's centre.
2. **Fixed `g_base_to_joint1`**: rebuilt as a clean `fixed` joint (no invalid
   `<axis>`/`<limit>`).
3. **Gripper (simple two-finger):** `gripper_link` (box 0.06×0.06×0.02,
   mass 0.08) fixed to `joint6_flange` at `xyz=0 0 0.02`; `left_finger` /
   `right_finger` (boxes 0.05×0.012×0.12, mass 0.03) at `y=±0.012`, prismatic
   joints along gripper X (`gripper_joint_left`/`gripper_joint_right`,
   `lower=0.0` open, `upper=0.04` closed, `effort=10`), with fingertip pads
   fixed at each finger tip. Exact grasp geometry is T3D-09.
4. **Output:** `urdf/mycobot_280_pi_3d.urdf`; `robot.yaml` → `arm.urdf` points
   at it. The original `mycobot_280_pi.urdf` and the `.dae` meshes were
   deleted (no longer referenced).

**Done when:** `check_urdf` parses with no errors; PyBullet loads with no
"mass 0" warnings; with `g_base` anchored and all joints position-locked, every
link holds its pose (max drift < 1 mm over 5 s); gripper joints open/close over
full range. (All verified.)

---

## 7. T3D-03 — Environment, objects, map import

**Problem:** the sim needs a physical world — floor, optional walls, and graspable
objects with known labels (the LLM needs a registry).

**Spec:**

1. **Ground:** infinite plane at `z=0`, `lateralFriction=1.0`.
2. **Arena:** default arena `10 m × 10 m` with 4 walls (box colliders 0.1 m
   thick, 2.0 m high) when no map is given. Optional: import `MapData` walls
   (`core/map_format.py`) and extrude each wall segment to `2.0 m` height at `z=0`,
   so the existing map files (`home.json`) become a 3D world. Object spawn poses
   come from a **world registry** (not vision in v1):
   - `world.json` (default) or `--objects` CLI: `{"objects":[{id, label, shape:
     "cube"|"cylinder"|"ball", size_m, mass_kg, pose:{x,y,z,rpy}, color}]}`.
3. **Object URDFs** (`urdf/objects/`), primitives only:
   - `cube.urdf`: box, `size_m` per side, `mass=mass_kg`, uniform 0.05 friction.
   - `cylinder.urdf`: `radius=size_m/2`, height `=size_m`, axis Z.
   - `ball.urdf`: sphere `radius=size_m/2`.
   - Parametrize via a single generator function (`urdf_assets.py`) so size/mass
     come from the registry, not hand-written files.
4. **Registry topic:** publish `object/registry` (§3) on spawn and every 1 s so
   `arm_controller` and `llm_nav` always know object IDs, labels and current
   poses (pose streamed from PyBullet ground truth).

**Done when:** with `home.json` loaded, walls appear at their map coords; a
registry with 5 mixed objects spawns, comes to rest (no tunnelling through the
floor), and `object/registry` is published with correct poses.

> **Status (done):** `simulation3d/world.py` builds the ground plane, walls
> (from `MapData` via `core/map_format.py`, extruded to `WALL_HEIGHT_M`, or a
> default 10×10 arena), and spawns registry objects. `urdf_assets.py` gained
> `generate_object_urdf` (cube/cylinder/ball). `world.json` is the default
> registry (5 mixed objects). `object/registry` and `sim/truth/pose` were added
> to the `core/messages.py` schema registry. Covered by `tests/test_3d_world.py`
> and the message round-trip tests. Continuous 1 s registry publishing runs in
> the sim loop (T3D-04).

---

## 8. T3D-04 — Physics loop + sim core

**Problem:** the pygame sim drives the robot with a hand-rolled 2D model; we now
need a deterministic, fixed-step physics loop around PyBullet with optional GUI.

**Spec:**

1. **Modes:** `--gui` → `p.connect(p.GUI)` (orbit camera, mouse + WASD keys);
   default/CI → `p.connect(p.DIRECT)` (headless). `--map` and `--objects` args.
2. **Step discipline:** `PHYSICS_STEP_S = 1/240`, 4 substeps per 60 Hz frame;
   `p.setPhysicsEngineParameter(numSubSteps=4, fixedTimeStep=1/240)`.
   `p.setGravity(0, 0, -9.81)`, `p.setRealTimeSimulation(False)`,
   `p.setSeed(...)` (seed from `--seed`, default 0 → deterministic). Loop paced
   with    `clock.sleep_until` (no cumulative drift, matches the existing clock module in `core/clock.py`).
3. **Spawning:** base at `pose` (default `[0,0,0.12,0,0,0]`), arm
   `p.loadURDF(mycobot…)` then
   `p.createConstraint(base, arm_mount, arm, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0], [0,0,0,1])`
   so `g_base` rides rigidly on the base. Objects from registry.
4. **Teleop:** same W/A/S/D keyboard handling + `sensor/wasd` publishing as the
   old sim. ESC quits. In GUI mode: `1/2/3` toggles camera follows-base / top-
   down / arm-ee follow.
5. **Ground-truth topic:** publish `sim/truth/pose` (base x/y/theta) at 50 Hz.
6. **Zenoh lifecycle:** open session in `__init__`, `close()` in `finally`
   (mirrors old sim).

**Done when:** 30 s headless run at full rate with zero zenoh/step exceptions;
deterministic given a seed (two runs, same seed → identical `sim/truth/pose`
series); base + arm + objects spawn and stay stable; GUI mode renders.

> **Status (done):** `simulation3d/simulator.py` builds the base (T3D-01) + arm
> (T3D-02, mounted on `arm_mount`, arm↔base collision disabled, arm loaded at
> the mount pose to avoid a snap) + world (T3D-03), and runs a fixed-step
> (1/240, 4 substeps) seeded loop with `sleep_until` pacing. Publishes
> `sim/truth/pose` (50 Hz), `sensor/wasd` and `object/registry` (1 Hz); `--gui`
> adds WASD teleop + 1/2/3 camera modes, ESC quits. Deterministic for a given
> seed (verified); base rests stable at its wheel height. Covered by
> `tests/test_3d_sim.py`.
>
> **Resolved:** the base's driving issues were (a) a **wrong wheel spin axis** —
> PyBullet reads the joint axis in the child frame, so the wheels spun about the
> vertical instead of their horizontal axle (fixed with axis `0 0 -1`), and (b)
> the fixed-wheel over-constraint (front wheels are now swivelling casters,
> T3D-01). W/S now drive forward/back straight and A/D turn correctly; the
> initial "robot doesn't move" bug was a GUI-keyboard bug (`p.connect(p.GUI)`
> returns 0 ≠ `p.GUI`, so keys were never read) — fixed by tracking the GUI mode
> explicitly. To stop the base pitching forward when driving with the arm up,
> the base mass was raised (to 8 kg) and the teleop drive applies a gentle
> acceleration ramp (no instant lunge). The base no longer flips; it may wobble
> ~5–15° of pitch at speed (a raw-PyBullet traction/arm-CG artefact), which
> could be further reduced by lowering the arm's mount height or top speed.

---

## 9. T3D-05 — Sensor bridge: LIDAR / IMU / camera

**Problem:** nav stack (pose estimator, navigator) consumes LIDAR/IMU/AprilTags
with specific schemas; these must come from the 3D world now.

**Spec:**

1. **LIDAR** (`sensor/lidar`): 360 rays, `max_range=10.0 m`, full 2π FOV, sampled
   at 50 Hz from base centre at `z=0.2`. For each angle: ray origin at base pose,
   ray along the XY plane rotated by base yaw; `p.rayTest` a 10 m segment; first
   hit distance; hit anything (walls **or objects**). Add range noise
   `σ=0.02 m` (clamp ≥ 0). Schema identical to old sim (`angle_rad`,
   `distance_m`).
2. **IMU** (`sensor/imu`): yaw from base quaternion; angular rate from base
   angular velocity + gyro bias `±0.005` + `σ=0.01` noise; yaw noise `σ=0.05`
   (reuse `simulator.py` constants verbatim); linear acceleration from
   base linear velocity finite-difference minus gravity + `σ=0.01`.
3. **AprilTag camera** (`sensor/camera/apriltag`): port the 2D FOV-cone logic
   from `simulator.py::_detect_apriltags` unchanged (tags from `MapData`),
   using the base pose truth.
4. **Wheel speeds:** keep the existing loop — sim publishes `sensor/wasd`,
   drive.py computes and publishes `sensor/wheel_speed`, sim subscribes and
   applies (§10). No double-drive-model.

**Done when:** a navigator/pose-estimator node run against the 3D sim converges
to ground truth with the same accuracy ballpark as the 2D sim (RMSE < 0.15 m on
a straight drive); LIDAR returns `max_range` in open space and short hits at
walls and objects.

> **Status (done):** `simulation3d/sensors.py` implements the 3D sensor models —
> LIDAR via `p.rayTest` (360 rays, 2π, configurable mount height, range noise,
> robot's own body filtered out), IMU (yaw from base quaternion, gyro bias +
> noise, proper acceleration = dv/dt − g), and the AprilTag camera. The AprilTag
> camera is 3D-aware: the camera sits at the configured mount, each tag is
> "printed" on the wall face its `yaw_rad` points into and is only visible from
> that side (facing gate), and an optional line-of-sight check (`p.rayTest`,
> robot self-hits ignored) drops tags blocked by a wall/object — so tags on the
> far side of a wall no longer inject false pose anchors. The simulator
> publishes `sensor/lidar`, `sensor/imu` (both 50 Hz) and
> `sensor/camera/apriltag` (on detection) over zenoh, with the config-driven
> noise/bias values. `home.json` tag yaws were fixed so every tag faces into the
> adjacent room. Covered by `tests/test_3d_sensors.py`. Note: a horizontal LIDAR
> at `z=0.2` sees walls (2 m tall) but passes over small floor objects — those
> are tracked via `object/registry` instead.

---

## 10. T3D-06 — Base drive + navigation integration

**Problem:** the physics loop must consume commanded wheel speeds and move the
base like the old sim (including the slip realism the stack was tuned against).

**Spec:**

1. Subscribe `sensor/wheel_speed` (`{left_rps, right_rps, t}`). Clamp each to
   `±max(1.2 · BOT_LINEAR_SPEED_MPS)` equivalent (`BOT_LINEAR_SPEED_MPS=3.0`).
   Apply via `p.setJointMotorControl2(..., p.VELOCITY_CONTROL, targetVelocity=
   left_rps)` (and right) with `force=500`.
2. **Slip realism parity:** apply the same per-run wheel-scale `±20%`,
   track-scale `±15%`, cross-coupling `0.1 rad/m`, and per-step slip `σ=0.02`
   from `simulator.py` before converting to unicycle — i.e. compute the
   *effective* linear/angular velocity from the commanded wheel speeds with the
   same corruption, and drive the wheel joints from those. This keeps
   pose-estimator tuning identical.
3. `estimate/pose` is subscribed but only used for an optional guess overlay
   (default off). `nav/path`, `detection/obstacles` subscribed for display.
4. Publish `sim/truth/pose` (base ground truth) at 50 Hz.

**Done when:** with controller + navigator running unmodified, the robot drives a
planned path around walls to a `nav/goal` and stops within 0.2 m; the same
behavior on the old 2D sim for the same map/goal. Wheel-slip corruption makes
odometry drift vs truth the same way it did in 2D.

> **Status (done, with a caveat):** the simulator now subscribes to
> `sensor/wheel_speed`, clamps it to `±1.2 × max linear speed` equivalent,
> applies the same wheel-scale/track-scale/cross-coupling/slip-noise corruption
> as the 2D sim (via `_compute_drive_command`/`_apply_slip`), and drives the
> drive-wheel joints from the effective speeds (`force=500`). It falls back to
> direct WASD teleop when no fresh `sensor/wheel_speed` is arriving, and
> subscribes `estimate/pose`, `nav/path`, `detection/obstacles` for display.
> Covered by `tests/test_3d_drive.py`.
>
> **Status update (after the wheel-axis fix + casters):** the base now drives
> forward/back straight and turns correctly (W/S/A/D). The "curves instead of
> straight" behaviour was largely a **wrong wheel spin axis** (the joints spun
> about the vertical, not the horizontal axle — fixed by using child-frame axis
> `0 0 -1`), compounded by the fixed-wheel over-constraint; the front wheels are
> now swivelling casters (T3D-01). Some traction/speed wobble remains in raw
> PyBullet, but the base is usable. The 2D sim remains the navigation-accuracy
> reference.
>
> **Goal-setting + estimator overlay (GUI):** left-clicking the ground in the
> `--gui` viewer publishes `nav/goal` (raycast through the PyBullet camera to
> the z=0 plane), so the unchanged navigator/controller/drive stack can drive
> the base to the click point. A debug overlay (toggle with **G**) draws the
> ground-truth pose (green arrow), the pose estimator's guess (red arrow, from
> `estimate/pose`), the planned `nav/path` (blue) and the current goal (yellow
> X). The pose estimator's running value can also be watched live with
> `python3 zenoh/topic_view.py estimate/pose` and compared against
> `sim/truth/pose`.
>
> **End-to-end nav verified (headless):** with the unmodified stack
> (drive/controller/pose_est/navigator) running against the 3D sim, publishing
> a `nav/goal` (6, 6.5) from spawn (8, 8.5) makes the robot turn to face it and
> drive to the goal, stopping within ~0.6 m (controller `GOAL_RADIUS_M` 0.3 +
> ~0.15 m estimator offset + path-end offset). This surfaced and fixed two real
> bugs:
>   1. **Swapped drive wheels** — `wheel_drive_L`/`wheel_drive_R` were attached
>      to the physically-opposite sides (`_L` at −y = the *right* wheel facing
>      +x), so every angular command was inverted and the robot drove *away*
>      from the goal. The URDF now places `_L` at +y / `_R` at −y, and the
>      teleop sign that compensated for the inversion was removed.
>   2. **Wheel traction** — PyBullet's default lateral friction (0.5) let the
>      drive wheels slip when rotating the heavy base+arm, so the base skidded
>      sideways instead of turning (and hit walls). `physics.wheel_friction_mu`
>      (default 2.0) is applied to all base links via `changeDynamics`.
> The pose estimator converges against the 3D LIDAR/IMU/AprilTags (est≈truth
> within ~0.1–0.2 m). Remaining gap to the 0.2 m "done when" bar is the
> estimator offset + controller arrival radius; both are shared with the 2D
> stack.

---

## 11. T3D-07 — Arm FK + joint-state publishing

**Problem:** nothing computes the arm's end-effector pose; the arm controller and
LLM need it.

**Spec:**

1. **FK** (`simulation3d/arm_kinematics.py`): pure function `fk(joint_positions)
   → SE(3) of `gripper_link` in the **`g_base`** frame. Implementation:
   `p.resetJointState(arm, joints, q)`, `p.getLinkState(arm, eeLink,
   computeForwardKinematics=True)`, transform from world to `g_base` frame using
   `g_base` link state. Deterministic (no stepping).
2. Controllable joint set: `joint2_to_joint1 … joint6output_to_joint6` +
   `gripper_joint_left/right`. Expose the joint index→name mapping from the arm
   section of `robot.yaml` (T-019).
3. **`arm_controller.py`** publishes `arm/joint_states` (all 8 joints, current
   PyBullet state) and `arm/ee_pose` at 50 Hz once connected to the sim.

**Done when:** `fk` matches `check_urdf`/urdfdom at ≥ 5 hand-computed joint
configs to `< 1e-3 m / 1e-3 rad`; `arm/ee_pose` streams at 50 Hz and equals
`fk(joint_states)`.

---

## 12. T3D-08 — IK + arm controller node

**Problem:** reaching a target SE(3) (pre-grasp, drop) requires inverse
kinematics and smooth joint motion.

**Spec:**

1. **IK** (`arm_kinematics.py`): `ik(target_pos, target_quat) → joint_positions`
   via `p.calculateInverseKinematics(arm, eeLink, targetPosition, targetOrientation,
   lowerLimits, upperLimits, jointRanges, restPoses, maxNumIterations=50,
   residualThreshold=1e-4)`. Rest pose = `[0, -0.8, 0, -1.6, 0, 0]` (joints
   2–5) to keep out of singularities. Lower/upper from the URDF joint limits.
2. **Trajectory control:** when moving, re-solve IK every control tick (50 Hz),
   feed solutions to `p.setJointMotorControl2(..., p.POSITION_CONTROL, maxVelocity=
   0.5 rad/s)` for all 6 arm joints; gripper joints are position-controlled to
   target gap. Move-until-`‖fk(q)-target‖< 0.01 m & 0.02 rad` then report done.
3. **Node** (`zenoh/control/arm_controller.py`): connects to sim zenoh topics.
   Owns the pick/place state machine (§13). No physics of its own — commands the
   sim via joint states? **No** — the controller runs *in the sim process* (it
   needs PyBullet handles). So `arm_controller` is a **module inside the sim
   process** that speaks zenoh: it subscribes `arm/command`, executes, and
   acks on `arm/state`. (Standalone-process option exists for later hardware:
   it would talk to real servos instead.)
4. **Command schema (`arm/command`):**
   ```
   {"t": float, "id": str,
    "op": "move_joint"|"move_to"|"pick"|"place"|"open"|"close"|"home"|"cancel",
    "joints": {name: rad},            # move_joint / open / close targets
    "pose": {x,y,z,r,p,y},            # move_to: SE(3) in g_base frame
    "object_id": str,                 # pick / place target
    "speed": float}                   # optional, default 1.0 (scale of maxVel)
   ```
5. **Ack (`arm/state`):** per command `id`, publish `busy` on accept, `ok` /
   `failed` with `detail` on completion. `cancel` aborts the current op at the
   next trajectory waypoint.

**Done when:** `move_to` drives the ee to 5 random feasible target poses within
`0.01 m / 0.02 rad` in < 3 s each; `move_joint`, `open`, `close`, `home`, `cancel`
all behave per spec; every op acks `ok`/`failed` exactly once with the matching
`id`.

---

## 13. T3D-09 — Grasp mechanics (pick / lift / carry / drop)

**Problem:** the end state — actually pick up, move, and drop an object.

**Spec:**

1. **Grasp attachment** (`simulation3d/grasp.py`): on successful grasp, create
   `p.createConstraint(parent=arm, parentLink=eeLink, child=objectBody,
   jointFramePoseParent=[0,0,0,1,0,0,0], jointFramePoseChild=[0,0,0,1,0,0,0],
   jointType=JOINT_FIXED)`. Store the constraint id; on drop,
   `p.removeConstraint(id)` then open gripper.
2. **Pick state machine** (in `arm_controller`, states: `IDLE, HOME, PREGRASP,
   GRASP, LIFT, CARRY, DROP`):
   - `pick {object_id}` → look up pose from `object/registry` (world frame) →
     transform to `g_base` frame (uses `sim/truth/pose` + mount offset) →
     **PREGRASP**: `move_to` pose = object + `[0,0,+0.12]` above, ee pointing
     down; then **GRASP**: descend at 0.04 m/s while closing fingers at
     `0.01 m/s` until (a) contact detected via `p.getContactPoints(finger, object)`
     force > 1.0 N, or (b) finger gap ≤ 0.004 m, or (c) timeout 3 s. Hold 0.5 s,
     verify grip: object ≤ 0.02 m from ee. Create constraint → **LIFT**: raise
     `+0.15 m` over 2 s → **CARRY**: hold constraint, report `ok` (base may now
     drive).
   - Failure (no contact, object fell, timeout): no constraint, open gripper,
     return to `IDLE`, ack `failed {detail}`.
   - `place {object_id}` → `move_to` pose = drop target `[x,y,z]` (world → arm
     frame) → descend to `z`, open gripper, `removeConstraint`, retreat `+0.05 m`,
     ack `ok`. If object is not carried → ack `failed "not carrying"`.
3. **CARRY invariant:** while carrying, assert object-ee distance < 0.03 m each
   control tick; if violated (dropped en route) → publish
   `arm/state {failed "dropped"}`, clear constraint.
4. **Gripper control:** `open`/`close` set finger joint targets `0.0`/`0.04`
   with `POSITION_CONTROL`; `open` also removes any active constraint.

**Done when:** scripted sequence on a fixed object — `pick` → base drives 2 m →
`place` at a target — leaves the object within `0.03 m` of target, no drops
en route, all acks `ok`; grasping a too-tall object or out-of-reach target
reports `failed` without leaving a stray constraint or a stuck arm.

---

## 14. T3D-10 — LLM semantic pick-and-place

**Problem:** `llm_nav.py` only understands navigation ("go to shrine"); it must
orchestrate "pick up the cube, put it at the shrine".

**Spec:**

1. Extend the LLM command grammar (in `llm_nav.py`) with a small, closed set:
   `pick up <object_label>`, `place <object_label> at <destination>`, and
   compound `pick up X and put it at Y` (map to pick → place sequence).
   Everything else still routes to navigation as today.
2. **Orchestration:** `pick X` → resolve X from `object/registry` → plan nav to a
   pre-grasp standoff point (0.45 m in front of the object, arm facing it) →
   publish `arm/command pick {object_id}` → wait `arm/state ok` → proceed to
   destination (room centre from the map) → `arm/command place`. Timeouts on each
   `arm/state` wait (default 10 s) → abort with an error string to the user.
3. Validation: reject unknown labels / destinations with a clear message, before
   any motion.

**Done when:** `nav/command = "pick up the cube and put it at the shrine"` results
in: robot drives to cube, arm picks, robot drives to shrine, arm drops, `arm/state`
`ok` throughout; unknown object → immediate refusal; timeout path aborts cleanly
(no stuck arm, no dropped constraint).

---

## 15. T3D-11 — Launch / orchestration + sim retirement

**Problem:** `start.sh` launches the 2D sim's nodes and expects `simulator.py`.

**Spec:**

1. `sim3d.sh`: `python3 -m simulation3d.simulator --gui --map home.json
   --objects world.json` with the same Ctrl+C process-group teardown as
   `start.sh`.
2. `start.sh`: add `zenoh/control/arm_controller` no longer needed as separate
   process (it lives in-sim, §12.3) — instead add `python3 -m simulation3d.simulator`
   and keep all existing nodes. Document that the sim is now the sensor/actuator
   HAL.
3. Legacy: `simulation/simulator.py` stays for one release; ROADMAP entry to
   delete it once T3D-12 parity tests pass. Its robot-specific values already
   live in `robot.yaml`/`core/robot_config.py` (T-019); only sim-ui constants
   (window size, colors) are removed with it.

**Done when:** `./sim3d.sh` boots the whole stack (sim + 5 nodes + logger) and a
full pick-and-place runs end-to-end from one launcher; `./start.sh` still works
but points at the 3D sim.

---

## 16. T3D-12 — Test suite + CI (headless)

**Problem:** new physics code must be verifiable headlessly and deterministically
(PyBullet DIRECT + fixed timestep + seed).

**Spec:**

1. `tests/test_3d_base.py`: command wheel velocities for 1 s → assert base
   displacement ≈ `wheel_radius · ω · t` (±5% with slip disabled via seed/flag);
   reverse; rotate in place.
2. `tests/test_3d_arm.py`: `fk` vs urdfdom/`check_urdf` at 5 configs
   (`< 1e-3 m`); `ik(fk(q₀)) ≈ q₀`; `fk(ik(T)) ≈ T` for 5 feasible targets.
3. `tests/test_3d_grasp.py`: spawn object under ee → `pick` → assert constraint
   exists + object moves with gripper; `place` → constraint removed, object at
   target; out-of-reach target → `failed`, no stray constraint.
4. `tests/test_3d_lidar.py`: open arena → all `distance_m ≈ 10.0`; wall 2 m
   away → hits ≈ 2 m (within noise σ); object between → shorter hit.
5. `tests/test_3d_arm_controller.py`: zenoh-less unit path (call the controller
   state machine directly with a fake command channel) — `pick`/`place`/`cancel`
   transitions and acks. (Zenoh-router integration tests marked `@pytest.mark.
   skipif` when no router is up, matching `test_replay.py`.)
6. Determinism: same `--seed` → identical `sim/truth/pose` and `arm/joint_states`
   logs. Run via `make test` / `make coverage` (no changes to Makefile needed).

**Done when:** `make test` runs the full suite (existing 83 + new 3D tests) green
on a headless box with no display.

---

## 17. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| CollADA meshes not loadable in PyBullet + inconsistent units (mm/m) | Resolved in T3D-02: arm rebuilt primitive-only; mesh files deleted |
| Missing inertials → PyBullet mass-0 behavior | Resolved in T3D-02: every link has an `<inertial>` |
| IK singularities near arm extremes | Rest-pose hints + joint-limit clamping + `residualThreshold`; pre-grasp standoff keeps target in reachable box |
| Grasp slipping (friction tuning) | High fingertip friction (1.5) + fixed constraint on success (deterministic, not fickle contact-only grasps) |
| Zenoh tests flaky without router | Direct unit tests for logic; router tests skip cleanly (existing `test_replay` pattern) |

## 18. Build order

1. T3D-01 base URDF → 2. T3D-02 arm URDF (inertials/gripper/collision) → 3. T3D-03
   world/objects → 4. T3D-04 sim core (spawn everything, verify visually) → 5. T3D-05
   sensors → 6. T3D-06 base drive/nav integration (full-stack nav works in 3D) →
   7. T3D-07 FK → 8. T3D-08 IK + controller → 9. T3D-09 grasp → 10. T3D-10 LLM →
   11. T3D-11 launch/retirement → 12. T3D-12 tests (written alongside 1–9, green at
   the end).

**Definition of done for the whole feature:** with `sim3d.sh` running and one
LLM command, the robot picks an object up off the floor, drives across the arena,
and sets it down at a target — deterministically, headlessly testable, with the
full existing nav stack running unmodified.
