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

    Units follow the map file: x and y are in map pixels; theta is in radians.
    The caller converts to/from meters using the map scale where needed.
    """

    def __init__(
        self,
        map_data: MapData,
        num_particles: int = 300,
        num_beams: int = 36,
        max_range_m: float = 10.0,
        measurement_sigma_m: float = 0.15,
        forward_noise_px: float = 0.2,
        theta_noise_rad: float = 0.01,
        random_fraction: float = 0.05,
    ):
        self.map_data = map_data
        self.num_particles = num_particles
        self.num_beams = num_beams
        self.scale_m_per_px = map_data.metadata.scale_m_per_px
        self.max_range_px = max_range_m / self.scale_m_per_px
        self.measurement_sigma_px = measurement_sigma_m / self.scale_m_per_px

        self.forward_noise_px = forward_noise_px
        self.theta_noise_rad = theta_noise_rad
        self.random_fraction = random_fraction

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
            width, height = self.map_data.metadata.size_px
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
        std_xy_px: float = 20.0,
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
                px = np.random.normal(x, std_xy_px)
                py = np.random.normal(y, std_xy_px)
                pt = np.random.normal(theta, std_theta_rad)

                if rooms:
                    for room in rooms:
                        if _point_in_polygon(px, py, room.polygon):
                            self.particles[i] = [px, py, pt]
                            break
                    else:
                        # Not inside any room — retry.
                        continue
                else:
                    if not self._inside_wall(px, py):
                        self.particles[i] = [px, py, pt]
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
        std_xy_px: float = 5.0,
        std_theta_rad: float = 0.05,
        fraction: float = 0.3,
    ):
        """Inject a fraction of particles tightly around an anchor pose.

        Unlike initialize_near() which replaces ALL particles, this only
        replaces `fraction` of them, keeping the rest as-is.  This pulls
        the filter toward the anchor without discarding prior information.

        Call this when an absolute pose measurement (e.g., AprilTag) arrives.
        """
        if not self.initialized:
            return

        n = self.num_particles
        n_anchor = max(1, int(n * fraction))
        indices = np.random.choice(n, size=n_anchor, replace=False)

        rooms = self.map_data.rooms
        for idx in indices:
            while True:
                px = np.random.normal(x, std_xy_px)
                py = np.random.normal(y, std_xy_px)
                pt = np.random.normal(theta, std_theta_rad)

                if rooms:
                    for room in rooms:
                        if _point_in_polygon(px, py, room.polygon):
                            self.particles[idx] = [px, py, pt]
                            break
                    else:
                        continue
                else:
                    if not self._inside_wall(px, py):
                        self.particles[idx] = [px, py, pt]
                    else:
                        continue
                break

        # Re-normalize weights so injected particles get equal weight.
        self.weights = np.ones(n) / n

    def _inside_wall(self, x: float, y: float) -> bool:
        """Rough check whether a point is inside a wall buffer (not used with rooms)."""
        # Simple bounding-box rejection; adequate for fallback initialization.
        for wall in self.map_data.walls:
            min_x = min(wall.x1, wall.x2) - 5
            max_x = max(wall.x1, wall.x2) + 5
            min_y = min(wall.y1, wall.y2) - 5
            max_y = max(wall.y1, wall.y2) + 5
            if min_x <= x <= max_x and min_y <= y <= max_y:
                return True
        return False

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(self, delta_forward_px: float, delta_theta: float):
        """Advance particles by one odometry step with motion noise."""
        if not self.initialized:
            return

        n = self.num_particles
        # Motion noise scales with the magnitude of the odometry step.
        forward_std = max(0.2, 0.1 * abs(delta_forward_px)) + self.forward_noise_px
        theta_std = max(0.005, 0.1 * abs(delta_theta)) + self.theta_noise_rad

        forward_noisy = delta_forward_px + np.random.normal(0.0, forward_std, n)
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
            / (2.0 * self.measurement_sigma_px * self.measurement_sigma_px)
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

        expected = np.full((n, self.num_beams), self.max_range_px)

        for j, rel_angle in enumerate(self.beam_angles):
            directions = theta + rel_angle
            dx = np.cos(directions)
            dy = np.sin(directions)

            closest = np.full(n, self.max_range_px)

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

            expected[:, j] = np.minimum(closest, self.max_range_px)

        return expected

    def _hits_to_distances(self, observed_hits: List[RayHit]) -> np.ndarray:
        """Convert observed LIDAR hits to beam distances matched by angle.

        Each beam picks up the closest LIDAR hit within its angular sector.
        Beams where no LIDAR hit lands within the sector stay at max_range so
        they contribute zero information.
        """
        observed = np.full(self.num_beams, self.max_range_px)
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

    def resample(self):
        """Resample particles proportional to their weights and add roughening noise.

        Uses systematic resampling (stochastic universal sampling) which has
        lower variance than simple multinomial resampling.
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
        self.particles[:, 0] += np.random.normal(0.0, 0.3, n)
        self.particles[:, 1] += np.random.normal(0.0, 0.3, n)
        self.particles[:, 2] += np.random.normal(0.0, 0.02, n)

        # --- Random sample injection ---
        if self.random_fraction > 0:
            n_random = int(n * self.random_fraction)
            rooms = self.map_data.rooms
            if rooms:
                areas = np.array([_polygon_area(r.polygon) for r in rooms])
                probs = areas / areas.sum()
                room_indices = np.random.choice(len(rooms), size=n_random, p=probs)
                for i, room_idx in enumerate(room_indices):
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

        self.weights = np.ones(n) / n

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
