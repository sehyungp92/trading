"""Shared fixtures for all test modules."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Common price data generators
# ---------------------------------------------------------------------------

def make_trending_up(n: int = 100, start: float = 100.0, step: float = 0.5,
                     noise: float = 0.3, seed: int = 42) -> tuple:
    """Generate synthetic uptrending OHLC data."""
    rng = np.random.RandomState(seed)
    closes = np.array([start + i * step + rng.randn() * noise for i in range(n)])
    highs = closes + rng.uniform(0.1, 1.0, n)
    lows = closes - rng.uniform(0.1, 1.0, n)
    opens = closes - rng.uniform(-0.3, 0.3, n)
    volumes = rng.uniform(1e6, 5e6, n)
    return opens, highs, lows, closes, volumes


def make_trending_down(n: int = 100, start: float = 200.0, step: float = 0.5,
                       noise: float = 0.3, seed: int = 42) -> tuple:
    """Generate synthetic downtrending OHLC data."""
    rng = np.random.RandomState(seed)
    closes = np.array([start - i * step + rng.randn() * noise for i in range(n)])
    highs = closes + rng.uniform(0.1, 1.0, n)
    lows = closes - rng.uniform(0.1, 1.0, n)
    opens = closes + rng.uniform(-0.3, 0.3, n)
    volumes = rng.uniform(1e6, 5e6, n)
    return opens, highs, lows, closes, volumes


def make_range_bound(n: int = 100, center: float = 150.0, width: float = 5.0,
                     seed: int = 42) -> tuple:
    """Generate synthetic range-bound OHLC data."""
    rng = np.random.RandomState(seed)
    closes = center + rng.uniform(-width / 2, width / 2, n)
    highs = closes + rng.uniform(0.1, 1.0, n)
    lows = closes - rng.uniform(0.1, 1.0, n)
    opens = closes + rng.uniform(-0.3, 0.3, n)
    volumes = rng.uniform(1e6, 5e6, n)
    return opens, highs, lows, closes, volumes


@pytest.fixture
def trending_up_data():
    return make_trending_up()


@pytest.fixture
def trending_down_data():
    return make_trending_down()


@pytest.fixture
def range_bound_data():
    return make_range_bound()
