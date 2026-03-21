"""Keltner Momentum Breakout — data models."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Direction(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


@dataclass
class DailyState:
    """Computed daily indicators for one bar."""

    close: float = 0.0
    close_prev: float = 0.0
    kelt_upper: float = 0.0
    kelt_middle: float = 0.0
    kelt_lower: float = 0.0
    kelt_middle_prev: float = 0.0
    rsi: float = 50.0
    rsi_prev: float = 50.0
    roc: float = 0.0
    volume: float = 0.0
    volume_sma: float = 0.0
    atr: float = 0.0
