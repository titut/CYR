# Semantic Navigation Library: Architecture & Implementation Plan

A focused plan for building a 2D simulation environment with LIDAR-based particle-filter localization and a simple desktop map editor.

---

## 1. Map Drawer (Simple Desktop GUI, Not React)

The editor produces the single source-of-truth map file used by the simulator and any future planner.

**Recommended tool:** Python with Tkinter or CustomTkinter.

- Tkinter ships with Python, so there is no install friction.
- CustomTkinter is a drop-in modernization if the default look becomes a problem.

**Core tools in the editor:**

- **Wall tool:** click-drag or click-click to draw line segments.
- **Room tool:** close a polygon (auto-snap the last point to the first), then assign a name.
- **Scale tool:** draw a reference line and type its real-world length (e.g., "this wall is 5 m"). Everything else derives from this single scale value.
- **Export:** write the map to a JSON file.

The editor does not need live simulation, path planning, or rendering polish. Its job is to produce a clean, well-formed map file.

**Note on connectivity:** Room-to-room passage points are not stored as explicit "door" objects. The simulator or planner will derive connectivity from wall geometry or room adjacency later.

---

## 2. Map File Format

The map is stored as JSON. Walls are kept as vectors so the simulator can rasterize them at whatever grid resolution it needs. Rooms are stored explicitly so downstream modules do not have to rediscover them from walls.

### Proposed schema

```json
{
  "metadata": {
    "scale_m_per_px": 0.05,
    "origin_px": [0, 0],
    "size_px": [800, 600]
  },
  "walls": [
    {"x1": 0, "y1": 0, "x2": 100, "y2": 0}
  ],
  "rooms": [
    {
      "id": "kitchen",
      "name": "Kitchen",
      "polygon": [[0, 0], [100, 0], [100, 80], [0, 80]],
      "center": [50, 40]
    }
  ]
}
```

### Design decisions

- **Walls as line segments, not a baked grid.** This avoids duplicating geometry and lets the simulator choose its own resolution.
- **Rooms as explicit polygons.** The editor validates closed spaces and writes them out. The simulator and planner read them directly.
- **One scale value in metadata.** All metric conversions (bot size, LIDAR range, planner costs) use `scale_m_per_px`.
- **No explicit door layer.** Connectivity is derived from geometry or room adjacency when needed.

An occupancy grid can still be generated from this file at simulator load time, but it is treated as a cache, not the master representation.

---

## 3. 2D Simulation Environment

The simulator loads the map file and runs the bot, sensors, and optional particle-filter visualization.

**Recommended tool:** Python with Pygame.

Pygame is a good fit because it handles the render loop, input, and simple 2D geometry without adding a physics engine or web stack.

### Simulator responsibilities

1. Load `map.json`.
2. Rasterize walls into an internal occupancy grid for fast collision and ray casting.
3. Render walls, rooms, the bot, and LIDAR rays.
4. Run a simple bot kinematics model from keyboard or scripted input.
5. Feed LIDAR scans to the particle filter and visualize the resulting pose estimate.

### Bot model

- State: `(x, y, theta)` in pixels or meters.
- Input: linear/angular velocity or left/right wheel velocities.
- The bot has a radius and respects wall collision.

### LIDAR model

- Cast 360 rays around the bot at a configurable angular resolution.
- For each ray, find the nearest wall intersection.
- Return a scan as `(angle, range)` pairs or a flat array of ranges.
- Add a max range (e.g., 10 m) for rays that hit nothing.
- Optionally add small Gaussian range noise for realism.

The ray-cast function is shared between the simulator and the particle filter, so both agree on what a given pose should see.

---

## 4. Particle Filter Localization

Use Monte Carlo Localization (MCL) because the map is known. The particle filter maintains a set of candidate poses and weights them by how well each candidate explains the current LIDAR scan.

### Particle state

Each particle is `(x, y, theta)` with an associated weight.

### Main loop

1. **Motion update:** propagate each particle by the latest odometry delta plus motion noise.
2. **Sensor update:** for each particle, simulate the expected LIDAR scan by ray casting against the map. Compute a likelihood score against the real scan.
3. **Resample:** draw a new particle set with replacement, weighted by likelihood.
4. **Estimate:** the current pose estimate is the weighted mean or the highest-weight particle.

### Sensor likelihood model

A standard beam model mixes three cases:

- A Gaussian hit near the expected range.
- A small probability for the max-range case.
- A small random-measurement probability to handle dynamic obstacles and sensor noise.

This keeps the implementation simple while behaving realistically enough for simulation.

### Practical notes

- Start with ~1000 particles spread uniformly over free space.
- Once converged, the population can drop to a few hundred.
- Odometry in the simulator can be ground-truth motion plus controlled noise. On a real bot it would come from wheel encoders or IMU.

---

## 5. How the Pieces Connect

```
┌─────────────┐      map.json       ┌──────────────────┐
│ Map Editor  │ ───────────────────► │ 2D Simulator     │
│ (Tkinter)   │                      │ (Pygame)         │
└─────────────┘                      │                  │
                                     │  - Bot kinematics│
                                     │  - LIDAR raycast │
                                     │  - Particle filter
                                     │  - Visualization │
                                     └──────────────────┘
```

The editor and simulator are deliberately decoupled: save in the editor, then load in the simulator. This keeps both tools simple and avoids a live IPC or server layer in the first pass.

A future semantic navigation layer can read the same `map.json`, resolve room names via the `rooms` list, and plan paths. Room connectivity can be derived from wall gaps or room adjacency when that layer is built.

---

## 6. Recommended Tech Stack

| Component         | Choice                          | Reason |
|-------------------|----------------------------------|--------|
| Map editor        | Python + Tkinter / CustomTkinter | Simple desktop GUI, no web stack |
| Simulator         | Python + Pygame                  | Easy 2D render loop and input |
| Math/geometry     | NumPy (+ Shapely if needed)      | Vector math and polygon ops |
| Particle filter   | Plain NumPy                      | Keeps the core algorithm readable |
| Map file format   | JSON                             | Human-readable, easy to debug |

No ROS, no physics engine, no React/Electron, no SLAM framework are needed at this stage.

---

## 7. Implementation Roadmap

Build in this order so the riskiest part—localization—is proven early.

1. **Map editor** that draws walls, closes polygons, labels rooms, sets scale, and exports JSON.
2. **Simulator shell** that loads the JSON, renders walls, and drives a circle with arrow keys.
3. **LIDAR ray casting** against the wall list, visualized as rays from the bot.
4. **Particle filter** using the same ray-cast function, visualized as particles converging on the true pose.
5. **Semantic layer** (room resolver and `go_to("kitchen")`) only after localization is reliable. Connectivity can be derived from wall geometry or room adjacency at that stage.

This order keeps each step demonstrable and avoids building navigation code on top of an unproven pose estimate.

---

## 8. Open Decisions

- Angular resolution of the LIDAR: 360 rays (1°) to start, adjustable later.
- Particle count: 1000 for global localization, fewer after convergence.
- Odometry noise model: start simple (Gaussian on `dx`, `dy`, `dtheta`) and tune by hand.
- Whether to store a cached occupancy grid inside the JSON. Recommendation: do not; generate it at load time.
- How to derive room connectivity for semantic navigation: wall-gap detection, room-adjacency search, or manual annotation added later if needed.
