"""Shared pytest fixtures and deterministic seeding.

The navigation/localization stack is stochastic (particle filter, RRT*), so
every test session is seeded for reproducible results.
"""

from __future__ import annotations

import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_rng():
    """Seed the global RNGs before every test for deterministic runs."""
    random.seed(0)
    np.random.seed(0)
    yield
