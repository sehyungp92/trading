"""Keltner Momentum Breakout — pure indicator computation.

All functions accept numpy arrays and return arrays or scalars.
"""
from __future__ import annotations

import numpy as np


def ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average with SMA seed."""
    out = np.empty_like(arr, dtype=float)
    k = 2.0 / (period + 1)
    seed_len = min(period, len(arr))
    out[0] = float(np.mean(arr[:seed_len]))
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average True Range (Wilder smoothing)."""
    n = len(highs)
    tr = np.empty(n, dtype=float)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    out = np.empty(n, dtype=float)
    out[0] = tr[0]
    alpha = 1.0 / period
    for i in range(1, n):
        out[i] = out[i - 1] * (1 - alpha) + tr[i] * alpha
    return out


def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index (Wilder smoothing)."""
    n = len(closes)
    out = np.full(n, 50.0, dtype=float)
    if n < 2:
        return out

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    seed = min(period, len(gains))
    avg_gain = float(np.mean(gains[:seed]))
    avg_loss = float(np.mean(losses[:seed]))

    for i in range(seed):
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            out[i + 1] = 100.0 - 100.0 / (1.0 + rs)
        elif avg_gain > 0:
            out[i + 1] = 100.0
        else:
            out[i + 1] = 50.0

    alpha = 1.0 / period
    for i in range(seed, len(gains)):
        avg_gain = avg_gain * (1 - alpha) + gains[i] * alpha
        avg_loss = avg_loss * (1 - alpha) + losses[i] * alpha
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            out[i + 1] = 100.0 - 100.0 / (1.0 + rs)
        elif avg_gain > 0:
            out[i + 1] = 100.0
        else:
            out[i + 1] = 50.0

    return out


def roc(closes: np.ndarray, period: int = 10) -> np.ndarray:
    """Rate of Change (percentage)."""
    n = len(closes)
    out = np.zeros(n, dtype=float)
    for i in range(period, n):
        prev = closes[i - period]
        if prev != 0:
            out[i] = (closes[i] - prev) / prev * 100.0
    return out


def keltner_channel(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    ema_period: int = 20,
    atr_period: int = 14,
    atr_mult: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keltner Channel: (upper, middle, lower).

    Middle = EMA of closes.
    Upper = middle + ATR * mult.
    Lower = middle - ATR * mult.
    """
    middle = ema(closes, ema_period)
    atr_arr = atr(highs, lows, closes, atr_period)
    upper = middle + atr_arr * atr_mult
    lower = middle - atr_arr * atr_mult
    return upper, middle, lower


def volume_sma(volumes: np.ndarray, period: int = 20) -> np.ndarray:
    """Simple moving average of volume."""
    n = len(volumes)
    out = np.zeros(n, dtype=float)
    cumsum = 0.0
    for i in range(n):
        cumsum += volumes[i]
        if i >= period:
            cumsum -= volumes[i - period]
            out[i] = cumsum / period
        else:
            out[i] = cumsum / (i + 1)
    return out
