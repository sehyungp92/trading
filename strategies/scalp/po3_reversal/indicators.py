from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .models import PriceBar


def atr(bars: Sequence[PriceBar], period: int = 14) -> float:
    if len(bars) < 2:
        return 0.0
    true_ranges: list[float] = []
    prev_close = bars[0].close
    for bar in bars[1:]:
        true_ranges.append(
            max(
                bar.high - bar.low,
                abs(bar.high - prev_close),
                abs(bar.low - prev_close),
            )
        )
        prev_close = bar.close
    sample = true_ranges[-period:]
    return float(np.mean(sample)) if sample else 0.0


def vwap(bars: Sequence[PriceBar]) -> float:
    total_volume = sum(max(bar.volume, 0.0) for bar in bars)
    if total_volume <= 0:
        return bars[-1].close if bars else 0.0
    return sum(((bar.high + bar.low + bar.close) / 3.0) * bar.volume for bar in bars) / total_volume


def rolling_percentile(value: float, history: Sequence[float]) -> float:
    if not history:
        return 50.0
    arr = np.asarray(history, dtype=float)
    return float(np.mean(arr <= value) * 100.0)


def median_range(bars: Sequence[PriceBar], lookback: int) -> float:
    sample = bars[-lookback:]
    if not sample:
        return 0.0
    return float(np.median([bar.high - bar.low for bar in sample]))

