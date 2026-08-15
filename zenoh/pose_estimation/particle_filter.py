"""Monte Carlo Localization using a particle filter.

The particle filter maintains a set of weighted pose hypotheses. It uses the
same ray casting function as the simulator so that expected LIDAR scans agree
with the sensor model.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Allow running this file directly or as a module.
_SCRIPT_DIR = Path(__file__).resolve().parent
_ZENOH_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _ZENOH_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_ZENOH_DIR) not in sys.path:
    sys.path.insert(0, str(_ZENOH_DIR))

import numpy as np

from map_format import MapData
from simulation.raycast import RayHit


def _point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    """Even-odd rule point-in-polygon test."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        ):
            inside = not inside
    return inside


def _polygon_area(polygon: List[Tuple[float, float]]) -> float:
    area = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


class ParticleFilter:
    """A simple particle filter for 2D LIDAR localization.

    Units: x and y are in meters; theta is in radians.  The map data is stored
    in meters (see map_format).
    """

    def __init__(
        self,
        map_data: MapData,
        num_particles: int = 300,
        num_beams: int = 36,
        max_range_m: float = 10.0,
        measurement_sigma_m: float = 0.15,
        forward_noise_m: float = 0.01,
        theta_noise_rad: float = 0.01,
        random_fraction: float = 0.05,
    ):
        self.map_data = map_data
        self.num_particles = num_particles
        self.num_beams = num_beams
        self.max_range = max_range_m
        self.measurement_sigma = measurement_sigma_m

        self.forward_noise_m = forward_noise_m
        self.theta_noise_rad = theta_noise_rad
        self.random_fraction = random_fraction

        # Radius of the local random-injection disc used after an anchor.
        self.local_inject_radius_m = 2.0

        # Beam angles relative to the particle's forward direction.
        self.beam_angles = np.linspace(-math.pi, math.pi, num_beams, endpoint=False)

        self.particles = np.zeros((num_particles, 3), dtype=np.float64)
        self.weights = np.ones(num_particles) / num_particles
        self.initialized = False

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------

    def initialize_uniform(self):
        """Spread particles uniformly across free space (room polygons)."""
        rooms = self.map_data.rooms
        if not rooms:
            # Fall back to the canvas bounding box; reject points inside walls.
            width, height = self.map_data.metadata.size_m
            for i in range(self.num_particles):
                while True:
                    x = np.random.uniform(0, width)
                    y = np.random.uniform(0, height)
                    if not self._inside_wall(x, y):
                        break
                self.particles[i] = [x, y, np.random.uniform(0, 2 * math.pi)]
        else:
            areas = np.array([_polygon_area(r.polygon) for r in rooms])
            probs = areas / areas.sum()
            room_indices = np.random.choice(
                len(rooms), size=self.num_particles, p=probs
            )

            for i, room_idx in enumerate(room_indices):
                room = rooms[room_idx]
                xs = [p[0] for p in room.polygon]
                ys = [p[1] for p in room.polygon]
                while True:
                    x = np.random.uniform(min(xs), max(xs))
                    y = np.random.uniform(min(ys), max(ys))
                    if _point_in_polygon(x, y, room.polygon):
                        break
                self.particles[i] = [x, y, np.random.uniform(0, 2 * math.pi)]

        self.weights = np.ones(self.num_particles) / self.num_particles
        self.initialized = True

    def initialize_near(
        self,
        x: float,
        y: float,
        theta: float,
        std_xy_m: float = 1.0,
        std_theta_rad: float = 0.3,
    ):
        """Initialise particles around a known pose with Gaussian noise.

        This matches how GPS-based initialisation works in real particle
        filters: spread particles around the initial estimate with
        uncertainty proportional to the sensor accuracy.
        """
        n = self.num_particles
        rooms = self.map_data.rooms
        for i in range(n):
            while True:
                sx = np.random.normal(x, std_xy_m)
                sy = np.random.normal(y, std_xy_m)
                pt = np.random.normal(theta, std_theta_rad)

                if rooms:
                    for room in rooms:
                        if _point_in_polygon(sx, sy, room.polygon):
                            self.particles[i] = [sx, sy, pt]
                            break
                    else:
                        # Not inside any room — retry.
                        continue
                else:
                    if not self._inside_wall(sx, sy):
                        self.particles[i] = [sx, sy, pt]
                    else:
                        # Inside a wall — retry.
                        continue
                # Accepted.
                break

        self.weights = np.ones(n) / n
        self.initialized = True

    def anchor_fraction(
        self,
        x: float,
        y: float,
        theta: float,
        std_xy_m: float = 0.25,
        std_theta_rad: float = 0.05,
        fraction: float = 0.3,
    ):
        """Inject a fraction of particles tightly around an anchor pose.

        Unlike initialize_near() which replaces ALL particles, this only
        replaces `fraction` of them, keeping the rest as-is.  With
        ``fraction=1.0`` it resets every particle to the anchor pose, so the
        filter jumps directly to an absolute measurement (e.g., an AprilTag).
        """
        if not self.initialized:
            return

        n = self.num_particles
        n_anchor = max(1, int(n * fraction))
        indices = np.random.choice(n, size=n_anchor, replace=False)

        rooms = self.map_data.rooms
        for idx in indices:
            while True:
                sx = np.random.normal(x, std_xy_m)
                sy = np.random.normal(y, std_xy_m)
                pt = np.random.normal(theta, std_theta_rad)

                if rooms:
                    for room in rooms:
                        if _point_in_polygon(sx, sy, room.polygon):
                            self.particles[idx] = [sx, sy, pt]
                            break
                    else:
                        continue
                else:
                    if not self._inside_wall(sx, sy):
                        self.particles[idx] = [sx, sy, pt]
                    else:
                        continue
                break

        # Re-normalize weights so injected particles get equal weight.
        self.weights = np.ones(n) / n

    def _inside_wall(self, x: float, y: float) -> bool:
        """Rough check whether a point is inside a wall buffer (not used with rooms)."""
        # Simple bounding-box rejection; adequate for fallback initialization.
        for wall in self.map_data.walls:
            min_x = min(wall.x1, wall.x2) - 0.25
            max_x = max(wall.x1, wall.x2) + 0.25
            min_y = min(wall.y1, wall.y2) - 0.25
            max_y = max(wall.y1, wall.y2) + 0.25
            if min_x <= x <= max_x and min_y <= y <= max_y:
                return True
        return False

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(self, delta_forward_m: float, delta_theta: float):
        """Advance particles by one odometry step with motion noise."""
        if not self.initialized:
            return

        n = self.num_particles
        # Motion noise scales with the magnitude of the odometry step.
        forward_std = max(0.01, 0.1 * abs(delta_forward_m)) + self.forward_noise_m
        theta_std = max(0.005, 0.1 * abs(delta_theta)) + self.theta_noise_rad

        forward_noisy = delta_forward_m + np.random.normal(0.0, forward_std, n)
        theta_noisy = delta_theta + np.random.normal(0.0, theta_std, n)

        self.particles[:, 2] += theta_noisy
        self.particles[:, 0] += forward_noisy * np.cos(self.particles[:, 2])
        self.particles[:, 1] += forward_noisy * np.sin(self.particles[:, 2])

    # -----------------------------------------------------------------------
    # Update
    # -----------------------------------------------------------------------

    def update(self, observed_hits: List[RayHit]):
        """Weight particles by how well each explains the observed LIDAR scan.

        Implements the standard SIR (bootstrap) weight update:
            new_weight ∝ p(measurement | particle) × old_weight
        """
        if not self.initialized:
            return

        observed_distances = self._hits_to_distances(observed_hits)
        expected = self._expected_distances()

        # Gaussian beam likelihood with a relative floor.
        diff = expected - observed_distances
        raw_lik = np.exp(
            -(diff * diff)
            / (2.0 * self.measurement_sigma * self.measurement_sigma)
        )
        likelihoods = raw_lik + 0.001 * np.max(raw_lik, axis=1, keepdims=True)

        # Multiply by previous weights (sequential importance sampling).
        # Work in log space to avoid floating-point underflow.
        log_likelihood = np.sum(np.log(likelihoods), axis=1)
        log_prior = np.log(np.maximum(self.weights, 1e-300))
        log_weights = log_prior + log_likelihood
        weights = np.exp(log_weights - np.max(log_weights))
        total = np.sum(weights)
        if total > 0:
            self.weights = weights / total
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles

    def _expected_distances(self) -> np.ndarray:
        """Vectorized ray casting for all particles and beams."""
        n = self.num_particles
        ox = self.particles[:, 0]
        oy = self.particles[:, 1]
        theta = self.particles[:, 2]

        expected = np.full((n, self.num_beams), self.max_range)

        for j, rel_angle in enumerate(self.beam_angles):
            directions = theta + rel_angle
            dx = np.cos(directions)
            dy = np.sin(directions)

            closest = np.full(n, self.max_range)

            for wall in self.map_data.walls:
                wx = wall.x2 - wall.x1
                wy = wall.y2 - wall.y1

                denom = dx * wy - dy * wx
                # Avoid division by zero: parallel rays are ignored for this wall.
                valid = np.abs(denom) > 1e-9

                t_numer = (wall.x1 - ox) * wy - (wall.y1 - oy) * wx
                u_numer = (wall.x1 - ox) * dy - (wall.y1 - oy) * dx

                t = np.full(n, np.inf)
                u = np.full(n, np.inf)
                t[valid] = t_numer[valid] / denom[valid]
                u[valid] = u_numer[valid] / denom[valid]

                hit = valid & (t >= 0) & (u >= 0) & (u <= 1) & (t < closest)
                closest[hit] = t[hit]

            expected[:, j] = np.minimum(closest, self.max_range)

        return expected

    def _hits_to_distances(self, observed_hits: List[RayHit]) -> np.ndarray:
        """Convert observed LIDAR hits to beam distances matched by angle.

        Each beam picks up the closest LIDAR hit within its angular sector.
        Beams where no LIDAR hit lands within the sector stay at max_range so
        they contribute zero information.
        """
        observed = np.full(self.num_beams, self.max_range)
        if not observed_hits:
            return observed

        hit_angles = np.array([h.angle for h in observed_hits])
        hit_distances = np.array([h.distance for h in observed_hits])

        # Maximum angular deviation we accept (half the beam spacing).
        max_angle_err = math.pi / self.num_beams

        for i, beam_angle in enumerate(self.beam_angles):
            diff = np.abs(hit_angles - beam_angle)
            diff = np.minimum(diff, 2 * math.pi - diff)  # circular wrap
            mask = diff <= max_angle_err
            if np.any(mask):
                observed[i] = np.min(hit_distances[mask])

        return observed

    # -----------------------------------------------------------------------
    # Resampling
    # -----------------------------------------------------------------------

    def resample(
        self,
        inject_mode: str = "global",
        anchor: Optional[Tuple[float, float]] = None,
    ):
        """Resample particles proportional to their weights and add roughening noise.

        Uses systematic resampling (stochastic universal sampling) which has
        lower variance than simple multinomial resampling.

        ``inject_mode`` controls where the random sample fraction is drawn
        from:

        - ``"global"`` — uniform across all rooms (recovers from divergence /
          kidnapping, but scatters particles across the whole map).
        - ``"local"`` — a disc of radius ``local_inject_radius_m`` around
          ``anchor`` (used after a confident absolute measurement such as an
          AprilTag, so we don't waste particles far from the robot).
        """
        if not self.initialized:
            return

        # --- Systematic resampling (stochastic universal sampling) ---
        n = self.num_particles
        cdf = np.cumsum(self.weights)
        cdf[-1] = 1.0  # guard against floating-point drift

        u0 = np.random.uniform(0.0, 1.0 / n)
        us = u0 + np.arange(n) / n

        indices = np.searchsorted(cdf, us)
        self.particles = self.particles[indices].copy()

        # --- Roughening / jittering ---
        # Use small noise so good particles stay near the true pose.
        self.particles[:, 0] += np.random.normal(0.0, 0.015, n)
        self.particles[:, 1] += np.random.normal(0.0, 0.015, n)
        self.particles[:, 2] += np.random.normal(0.0, 0.02, n)

        # --- Random sample injection ---
        if self.random_fraction > 0:
            n_random = int(n * self.random_fraction)
            if inject_mode == "local" and anchor is not None:
                self._inject_local(n_random, anchor[0], anchor[1])
            else:
                self._inject_global(n_random)

        self.weights = np.ones(n) / n

    def _inject_global(self, n_random: int):
        """Re-inject particles uniformly across free space (all rooms)."""
        n = self.num_particles
        rooms = self.map_data.rooms
        if not rooms:
            width, height = self.map_data.metadata.size_m
            for _ in range(n_random):
                while True:
                    x = np.random.uniform(0, width)
                    y = np.random.uniform(0, height)
                    if not self._inside_wall(x, y):
                        break
                idx = np.random.randint(n)
                self.particles[idx] = [x, y, np.random.uniform(0, 2 * math.pi)]
            return

        areas = np.array([_polygon_area(r.polygon) for r in rooms])
        probs = areas / areas.sum()
        room_indices = np.random.choice(len(rooms), size=n_random, p=probs)
        for room_idx in room_indices:
            room = rooms[room_idx]
            xs = [p[0] for p in room.polygon]
            ys = [p[1] for p in room.polygon]
            while True:
                x = np.random.uniform(min(xs), max(xs))
                y = np.random.uniform(min(ys), max(ys))
                if _point_in_polygon(x, y, room.polygon):
                    break
            idx = np.random.randint(n)
            self.particles[idx] = [x, y, np.random.uniform(0, 2 * math.pi)]

    def _inject_local(self, n_random: int, x: float, y: float):
        """Re-inject particles in a disc around (x, y) to preserve diversity.

        Uniformly samples the disc of radius ``local_inject_radius_m`` and
        rejects points that fall outside free space (room polygons).
        """
        n = self.num_particles
        rooms = self.map_data.rooms
        radius = self.local_inject_radius_m
        for _ in range(n_random):
            while True:
                # Uniform disc sampling: r ~ sqrt(U), angle ~ U(0, 2π).
                r = radius * math.sqrt(np.random.uniform(0.0, 1.0))
                a = np.random.uniform(0.0, 2 * math.pi)
                px = x + r * math.cos(a)
                py = y + r * math.sin(a)

                if rooms:
                    if any(_point_in_polygon(px, py, room.polygon) for room in rooms):
                        break
                elif not self._inside_wall(px, py):
                    break
            idx = np.random.randint(n)
            self.particles[idx] = [px, py, np.random.uniform(0, 2 * math.pi)]

    # -----------------------------------------------------------------------
    # Estimate
    # -----------------------------------------------------------------------

    def estimate(self) -> Tuple[float, float, float]:
        """Return the weighted mean pose estimate."""
        x_mean = float(np.average(self.particles[:, 0], weights=self.weights))
        y_mean = float(np.average(self.particles[:, 1], weights=self.weights))

        # Circular mean for theta.
        sin_sum = float(np.sum(np.sin(self.particles[:, 2]) * self.weights))
        cos_sum = float(np.sum(np.cos(self.particles[:, 2]) * self.weights))
        theta_mean = math.atan2(sin_sum, cos_sum)

        return x_mean, y_mean, theta_mean
