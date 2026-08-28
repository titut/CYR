"""Local scan-matching refinement on top of the particle filter.

The particle filter's weighted mean can linger at a biased local optimum even
when a structured LIDAR scan is available.  This module minimizes the *squared
range error* between the observed scan and the scan predicted from a candidate
pose:

    E(x, y, theta) = sum_j ( z_obs[j] - z_expected[j](x, y, theta) )^2

over the "hit" beams (rays that returned a real surface well inside the sensor's
max range).  Two solvers share this cost function:

- ``refine_discrete``    — coordinate descent ("move in a direction, keep going
  while it helps, halve the step when stuck").  No derivatives, robust.
- ``refine_gauss_newton`` — point-to-line ICP (Gauss-Newton), the standard
  mathematical scan matcher.  Fast, quadratic convergence.

Both snap a pose to the locally-optimal match; the caller decides which to trust
and re-anchors the particle cloud / odometry fallback accordingly.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from core.map_format import Wall


def _raycast_expected(
    x: float,
    y: float,
    theta: float,
    walls: Sequence[Wall],
    angles: np.ndarray,
    max_range: float,
) -> np.ndarray:
    """Expected beam distances for one pose, vectorized over ``angles``.

    Mirrors the segment-intersection math in ``simulation/raycast.py`` but
    evaluates all beams at once (numpy) so the cost function is cheap enough to
    call many times from the discrete solver.
    """
    directions = theta + angles
    dx = np.cos(directions)
    dy = np.sin(directions)
    closest = np.full(len(angles), max_range, dtype=np.float64)

    for wall in walls:
        wx = wall.x2 - wall.x1
        wy = wall.y2 - wall.y1
        denom = dx * wy - dy * wx
        valid = np.abs(denom) > 1e-9

        # The ray origin is a single pose (scalar), so the t-numerator is a
        # scalar while the u-numerator is per-beam (dx/dy are arrays).
        t_num = (wall.x1 - x) * wy - (wall.y1 - y) * wx
        u_num = (wall.x1 - x) * dy - (wall.y1 - y) * dx
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(valid, t_num / denom, np.inf)
            u = np.where(valid, u_num / denom, np.inf)

        hit = valid & (t >= 0.0) & (u >= 0.0) & (u <= 1.0) & (t < closest)
        closest = np.where(hit, t, closest)

    return np.minimum(closest, max_range)


def scan_match_error(
    pose: Tuple[float, float, float],
    walls: Sequence[Wall],
    angles: np.ndarray,
    observed: np.ndarray,
    max_range: float,
    margin: float = 0.1,
) -> float:
    """Squared range error over the hit beams (sum of squared residuals).

    ``angles``/``observed`` are the relative beam angles (rad) and the observed
    distances (m) of one scan.  Only beams with an observed distance well inside
    ``max_range`` contribute ("no return" rays are ignored).
    """
    x, y, theta = pose
    expected = _raycast_expected(x, y, theta, walls, angles, max_range)
    hit = observed < (max_range - margin)
    if not np.any(hit):
        return float("inf")
    diff = observed[hit] - expected[hit]
    return float(np.sum(diff * diff))


def _nearest_wall(
    px: float, py: float, walls: Sequence[Wall]
) -> Optional[Tuple[float, float, float, float]]:
    """Return the wall segment (x1, y1, x2, y2) closest to point (px, py)."""
    best: Optional[Tuple[float, float, float, float]] = None
    best_d2 = float("inf")
    for wall in walls:
        wx1, wy1, wx2, wy2 = wall.x1, wall.y1, wall.x2, wall.y2
        wdx = wx2 - wx1
        wdy = wy2 - wy1
        l2 = wdx * wdx + wdy * wdy
        if l2 < 1e-12:
            cx, cy = wx1, wy1
        else:
            t = ((px - wx1) * wdx + (py - wy1) * wdy) / l2
            t = max(0.0, min(1.0, t))
            cx = wx1 + t * wdx
            cy = wy1 + t * wdy
        d2 = (px - cx) ** 2 + (py - cy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = (wx1, wy1, wx2, wy2)
    return best


def refine_discrete(
    pose: Tuple[float, float, float],
    walls: Sequence[Wall],
    angles: np.ndarray,
    observed: np.ndarray,
    max_range: float,
    margin: float = 0.1,
    steps: Tuple[float, float, float] = (0.5, 0.5, 0.2),
    min_steps: Tuple[float, float, float] = (0.01, 0.01, 0.004),
    max_iters: int = 80,
) -> Tuple[float, float, float]:
    """Coordinate descent: greedily move along x/y/theta while the error drops,
    halving the step sizes when stuck, until they fall below ``min_steps``.

    This is the discrete "move in a direction, check better/worse, keep going"
    search — derivative-free and robust, but only linearly convergent.
    """
    x, y, th = pose
    err = scan_match_error((x, y, th), walls, angles, observed, max_range, margin)
    sx, sy, sth = steps

    for _ in range(max_iters):
        improved = False

        for dx in (sx, -sx):
            e = scan_match_error((x + dx, y, th), walls, angles, observed, max_range, margin)
            if e < err:
                x += dx
                err = e
                improved = True
                break
        for dy in (sy, -sy):
            e = scan_match_error((x, y + dy, th), walls, angles, observed, max_range, margin)
            if e < err:
                y += dy
                err = e
                improved = True
                break
        for dth in (sth, -sth):
            e = scan_match_error((x, y, th + dth), walls, angles, observed, max_range, margin)
            if e < err:
                th += dth
                err = e
                improved = True
                break

        if not improved:
            sx *= 0.5
            sy *= 0.5
            sth *= 0.5
            if sx < min_steps[0] and sy < min_steps[1] and sth < min_steps[2]:
                break

    th = math.atan2(math.sin(th), math.cos(th))
    return (x, y, th)


def refine_gauss_newton(
    pose: Tuple[float, float, float],
    walls: Sequence[Wall],
    angles: np.ndarray,
    observed: np.ndarray,
    max_range: float,
    margin: float = 0.1,
    max_corr: float = 0.5,
    max_iters: int = 20,
    trans_cap: float = 0.5,
    rot_cap: float = 0.3,
    min_delta: float = 1e-4,
) -> Tuple[float, float, float]:
    """Point-to-line ICP solved by Gauss-Newton (the mathematical scan matcher).

    Each hit beam's endpoint is matched to its nearest wall; the residual is the
    signed perpendicular distance to that wall's line.  Linearizing the 2D rigid
    transform (small-angle rotation) gives a linear least-squares problem
    ``A·Δ = b`` solved each iteration.  ``max_corr`` gates the correspondence so
    a mismatched/outlier endpoint (e.g. a dynamic obstacle) is skipped.
    """
    x, y, th = pose

    for _ in range(max_iters):
        rows: List[List[float]] = []
        rhs: List[float] = []

        for i in range(len(angles)):
            z = observed[i]
            if z >= max_range - margin:
                continue
            a = th + angles[i]
            px = x + z * math.cos(a)
            py = y + z * math.sin(a)

            wall = _nearest_wall(px, py, walls)
            if wall is None:
                continue
            wx1, wy1, wx2, wy2 = wall
            wdx = wx2 - wx1
            wdy = wy2 - wy1
            wlen = math.hypot(wdx, wdy)
            if wlen < 1e-9:
                continue

            # Signed perpendicular distance from the endpoint to the wall line.
            dist = (wdx * (py - wy1) - wdy * (px - wx1)) / wlen
            if abs(dist) > max_corr:
                continue  # outlier / mismatched correspondence

            nx = -wdy / wlen
            ny = wdx / wlen
            # n · d(endpoint)/d(pose) = [nx, ny, -z·nx·sin(a) + z·ny·cos(a)]
            rows.append([nx, ny, -z * nx * math.sin(a) + z * ny * math.cos(a)])
            rhs.append(-dist)

        if len(rows) < 3:
            break

        A = np.asarray(rows, dtype=np.float64)
        b = np.asarray(rhs, dtype=np.float64)
        delta, *_ = np.linalg.lstsq(A, b, rcond=None)

        dx = max(-trans_cap, min(trans_cap, float(delta[0])))
        dy = max(-trans_cap, min(trans_cap, float(delta[1])))
        dth = max(-rot_cap, min(rot_cap, float(delta[2])))

        x += dx
        y += dy
        th += dth
        if abs(dx) < min_delta and abs(dy) < min_delta and abs(dth) < min_delta:
            break

    th = math.atan2(math.sin(th), math.cos(th))
    return (x, y, th)
