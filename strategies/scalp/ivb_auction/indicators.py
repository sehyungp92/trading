from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def atr_from_ohlc(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float:
    if len(highs) < 2 or len(lows) < 2 or len(closes) < 2:
        return 0.0
    ranges: list[float] = []
    prev_close = float(closes[0])
    for high, low, close in zip(highs[1:], lows[1:], closes[1:], strict=False):
        high_f = float(high)
        low_f = float(low)
        ranges.append(max(high_f - low_f, abs(high_f - prev_close), abs(low_f - prev_close)))
        prev_close = float(close)
    sample = ranges[-period:]
    return float(np.mean(sample)) if sample else 0.0


def rolling_median(values: Sequence[float], lookback: int) -> float:
    sample = list(values[-lookback:])
    return float(np.median(sample)) if sample else 0.0


def rolling_delta(deltas: Sequence[float], lookback: int) -> float:
    return float(sum(deltas[-lookback:])) if deltas else 0.0


def rolling_volume(volumes: Sequence[float], lookback: int) -> float:
    sample = list(volumes[-lookback:])
    return float(np.median(sample)) if sample else 0.0

