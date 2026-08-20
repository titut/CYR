"""Clock abstraction for deterministic time (T-020, minimal slice).

The system should never depend on a bare wall clock for logic: durations and
latencies measured against ``time.monotonic()``, deadline-driven loops, and a
record/replay story all need an injectable clock.  This module provides the
abstraction and two implementations:

    WallClock   — the real monotonic clock (what production nodes use).
    SimClock    — a manually-advanced clock for deterministic tests / replay.

Adopting the clock across the nodes (deadline-driven loops, sim-time) is the
rest of T-020; this slice is the abstraction + seeding helpers that the T-026
replay tests build on.
"""

from __future__ import annotations

import random
import time
from typing import Protocol

import numpy as np


class Clock(Protocol):
    """A source of monotonic time in seconds."""

    def monotonic(self) -> float: ...


class WallClock:
    """Real monotonic time."""

    def monotonic(self) -> float:
        return time.monotonic()


class SimClock:
    """A deterministic clock that only advances when told to.

    ``t`` starts at 0 and advances in fixed steps, so a replay can be
    reproduced exactly regardless of how long the machine actually takes.
    """

    def __init__(self, start: float = 0.0, step: float = 0.0):
        self._t = start
        self._step = step

    def monotonic(self) -> float:
        return self._t

    def tick(self) -> float:
        """Advance one ``step`` and return the new time."""
        self._t += self._step
        return self._t

    def advance(self, dt: float) -> float:
        """Advance by ``dt`` seconds and return the new time."""
        self._t += dt
        return self._t


def sleep_until(deadline: float) -> None:
    """Sleep until the given real monotonic deadline (deadline-driven loops).

    Sleeps for ``deadline - time.monotonic()`` if positive, so loops can be
    paced without cumulative drift from ``time.sleep(fixed_dt)`` jitter.
    """
    delay = deadline - time.monotonic()
    if delay > 0:
        time.sleep(delay)


def seed_all(seed: int = 0) -> None:
    """Seed both RNGs used by the stack for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
