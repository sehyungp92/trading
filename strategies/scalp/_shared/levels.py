from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class IVBLevels:
    high: float
    low: float
    mid: float
    range_pts: float
    poc: float = 0.0
    vah: float = 0.0
    val: float = 0.0

    @classmethod
    def from_bounds(
        cls,
        high: float,
        low: float,
        *,
        poc: float = 0.0,
        vah: float = 0.0,
        val: float = 0.0,
    ) -> "IVBLevels":
        return cls(
            high=float(high),
            low=float(low),
            mid=(float(high) + float(low)) / 2.0,
            range_pts=max(0.0, float(high) - float(low)),
            poc=float(poc),
            vah=float(vah),
            val=float(val),
        )


@dataclass(frozen=True, slots=True)
class DealingRange:
    high: float
    low: float
    eq: float

    @classmethod
    def from_bounds(cls, high: float, low: float) -> "DealingRange":
        return cls(float(high), float(low), (float(high) + float(low)) / 2.0)


@dataclass(frozen=True, slots=True)
class FractalPivot:
    index: int
    price: float
    confirmed_at_index: int


def detect_fractal_pivots(
    highs: Iterable[float],
    lows: Iterable[float],
    left_n: int,
    right_n: int,
) -> tuple[list[FractalPivot], list[FractalPivot]]:
    """Return look-ahead-safe pivots with explicit confirmation index."""
    high_values = [float(value) for value in highs]
    low_values = [float(value) for value in lows]
    if len(high_values) != len(low_values):
        raise ValueError("highs and lows must have the same length")
    if left_n < 1 or right_n < 1:
        raise ValueError("left_n and right_n must be positive")

    swing_highs: list[FractalPivot] = []
    swing_lows: list[FractalPivot] = []
    for idx in range(left_n, len(high_values) - right_n):
        left_highs = high_values[idx - left_n:idx]
        right_highs = high_values[idx + 1:idx + right_n + 1]
        left_lows = low_values[idx - left_n:idx]
        right_lows = low_values[idx + 1:idx + right_n + 1]
        if high_values[idx] > max(left_highs) and high_values[idx] > max(right_highs):
            swing_highs.append(
                FractalPivot(idx, high_values[idx], confirmed_at_index=idx + right_n)
            )
        if low_values[idx] < min(left_lows) and low_values[idx] < min(right_lows):
            swing_lows.append(
                FractalPivot(idx, low_values[idx], confirmed_at_index=idx + right_n)
            )
    return swing_highs, swing_lows


def latest_confirmed_dealing_range(
    highs: Iterable[float],
    lows: Iterable[float],
    *,
    left_n: int,
    right_n: int,
    as_of_index: int,
) -> DealingRange | None:
    swing_highs, swing_lows = detect_fractal_pivots(highs, lows, left_n, right_n)
    high = next((p for p in reversed(swing_highs) if p.confirmed_at_index <= as_of_index), None)
    low = next((p for p in reversed(swing_lows) if p.confirmed_at_index <= as_of_index), None)
    if high is None or low is None:
        return None
    return DealingRange.from_bounds(high.price, low.price)

