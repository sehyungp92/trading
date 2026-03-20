"""Indicator computation: EMA, ATR, MACD, percentiles over bar arrays."""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np

from .config import (
    ATR_PERIOD, EMA_FAST_PERIOD, EMA_SLOW_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    VOL_PERCENTILE_WINDOW, VOL_FACTOR_MIN, VOL_FACTOR_MAX, LOW_VOL_THRESHOLD,
    TF, Bar,
)


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    out = np.full_like(values, np.nan)
    if len(values) < period:
        return out
    k = 2.0 / (period + 1)
    out[period - 1] = np.mean(values[:period])
    for i in range(period, len(values)):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = ATR_PERIOD) -> np.ndarray:
    """Average true range."""
    n = len(highs)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    out = np.full(n, np.nan)
    if n < period:
        return out
    out[period - 1] = np.mean(tr[:period])
    k = 1.0 / period
    for i in range(period, n):
        out[i] = tr[i] * k + out[i - 1] * (1 - k)
    return out


def macd(closes: np.ndarray, fast: int = MACD_FAST, slow: int = MACD_SLOW, sig: int = MACD_SIGNAL):
    """Returns (macd_line, signal_line, histogram)."""
    ema_f = ema(closes, fast)
    ema_s = ema(closes, slow)
    line = ema_f - ema_s
    signal = ema(line[~np.isnan(line)], sig)
    # Align signal back
    sig_full = np.full_like(line, np.nan)
    valid_start = np.argmax(~np.isnan(line))
    if len(signal) > 0:
        sig_full[valid_start:valid_start + len(signal)] = signal
    hist = line - sig_full
    return line, sig_full, hist


def percentile_rank(value: float, history: np.ndarray) -> float:
    """Percentile rank of value within history (0-100)."""
    valid = history[~np.isnan(history)]
    if len(valid) == 0:
        return 50.0
    return float(np.sum(valid < value) / len(valid) * 100)


class BarSeries:
    """Maintains a rolling window of bars for a single timeframe with computed indicators."""

    def __init__(self, tf: TF, maxlen: int = 300):
        self.tf = tf
        self.bars: deque[Bar] = deque(maxlen=maxlen)
        # Cached indicator arrays (recomputed on each bar add)
        self._closes: Optional[np.ndarray] = None
        self._highs: Optional[np.ndarray] = None
        self._lows: Optional[np.ndarray] = None
        self._ema_fast: Optional[np.ndarray] = None
        self._ema_slow: Optional[np.ndarray] = None
        self._atr: Optional[np.ndarray] = None
        self._macd_line: Optional[np.ndarray] = None
        self._macd_signal: Optional[np.ndarray] = None
        self._macd_hist: Optional[np.ndarray] = None
        self._dirty = True

    def add_bar(self, bar: Bar) -> None:
        self.bars.append(bar)
        self._dirty = True

    def _recompute(self) -> None:
        if not self._dirty or len(self.bars) < 2:
            return
        self._closes = np.array([b.close for b in self.bars])
        self._highs = np.array([b.high for b in self.bars])
        self._lows = np.array([b.low for b in self.bars])
        self._ema_fast = ema(self._closes, EMA_FAST_PERIOD)
        self._ema_slow = ema(self._closes, EMA_SLOW_PERIOD)
        self._atr = atr(self._highs, self._lows, self._closes, ATR_PERIOD)
        self._macd_line, self._macd_signal, self._macd_hist = macd(self._closes)
        self._dirty = False

    @property
    def last_close(self) -> float:
        return self.bars[-1].close if self.bars else 0.0

    @property
    def last_high(self) -> float:
        return self.bars[-1].high if self.bars else 0.0

    @property
    def last_low(self) -> float:
        return self.bars[-1].low if self.bars else 0.0

    def close_n_ago(self, n: int) -> float:
        idx = len(self.bars) - 1 - n
        if 0 <= idx < len(self.bars):
            return self.bars[idx].close
        return 0.0

    def ema_fast(self) -> float:
        self._recompute()
        return float(self._ema_fast[-1]) if self._ema_fast is not None and not np.isnan(self._ema_fast[-1]) else 0.0

    def ema_slow(self) -> float:
        self._recompute()
        return float(self._ema_slow[-1]) if self._ema_slow is not None and not np.isnan(self._ema_slow[-1]) else 0.0

    def current_atr(self) -> float:
        self._recompute()
        return float(self._atr[-1]) if self._atr is not None and not np.isnan(self._atr[-1]) else 0.0

    def atr_at(self, idx: int) -> float:
        self._recompute()
        if self._atr is not None and 0 <= idx < len(self._atr) and not np.isnan(self._atr[idx]):
            return float(self._atr[idx])
        return 0.0

    def macd_line_at(self, idx: int) -> float:
        self._recompute()
        if self._macd_line is not None and 0 <= idx < len(self._macd_line) and not np.isnan(self._macd_line[idx]):
            return float(self._macd_line[idx])
        return 0.0

    def macd_line_now(self) -> float:
        self._recompute()
        if self._macd_line is not None and not np.isnan(self._macd_line[-1]):
            return float(self._macd_line[-1])
        return 0.0

    def macd_hist_now(self) -> float:
        self._recompute()
        if self._macd_hist is not None and not np.isnan(self._macd_hist[-1]):
            return float(self._macd_hist[-1])
        return 0.0

    def macd_line_n_ago(self, n: int) -> float:
        self._recompute()
        idx = len(self._macd_line) - 1 - n if self._macd_line is not None else -1
        if self._macd_line is not None and 0 <= idx < len(self._macd_line) and not np.isnan(self._macd_line[idx]):
            return float(self._macd_line[idx])
        return 0.0

    def macd_hist_n_ago(self, n: int) -> float:
        self._recompute()
        idx = len(self._macd_hist) - 1 - n if self._macd_hist is not None else -1
        if self._macd_hist is not None and 0 <= idx < len(self._macd_hist) and not np.isnan(self._macd_hist[idx]):
            return float(self._macd_hist[idx])
        return 0.0

    def highest_high(self, lookback: int) -> float:
        self._recompute()
        if self._highs is None or len(self._highs) == 0:
            return 0.0
        return float(np.max(self._highs[-lookback:]))

    def lowest_low(self, lookback: int) -> float:
        self._recompute()
        if self._lows is None or len(self._lows) == 0:
            return float("inf")
        return float(np.min(self._lows[-lookback:]))

    def bar_range(self) -> float:
        if not self.bars:
            return 0.0
        b = self.bars[-1]
        return b.high - b.low

    def trend_strength_at(self, idx: int) -> float:
        """Return |ema_fast - ema_slow| / atr at array index `idx`."""
        self._recompute()
        if (self._ema_fast is None or self._ema_slow is None or self._atr is None
                or not (0 <= idx < len(self._atr))):
            return 0.0
        a = float(self._atr[idx])
        if a <= 0 or np.isnan(a):
            return 0.0
        ef = float(self._ema_fast[idx])
        es = float(self._ema_slow[idx])
        if np.isnan(ef) or np.isnan(es):
            return 0.0
        return abs(ef - es) / a

    def atr_rolling(self, period: int) -> float:
        """ATR computed over last `period` bars (for Class R vol signature)."""
        self._recompute()
        if self._atr is None or len(self._atr) < period:
            return 0.0
        subset = self._atr[-period:]
        valid = subset[~np.isnan(subset)]
        return float(np.mean(valid)) if len(valid) > 0 else 0.0

    def atr_rolling_prev(self, period: int) -> float:
        """ATR rolling mean over `period` bars ending one bar before the latest."""
        self._recompute()
        if self._atr is None or len(self._atr) < period + 1:
            return 0.0
        subset = self._atr[-(period + 1):-1]
        valid = subset[~np.isnan(subset)]
        return float(np.mean(valid)) if len(valid) > 0 else 0.0


class VolEngine:
    """Daily volatility engine: ATR_base (median), vol_pct, VolFactor."""

    def __init__(self):
        self.atr_base: float = 0.0
        self.vol_pct: float = 50.0
        self.vol_factor: float = 1.0
        self.extreme_vol: bool = False

    def update(self, daily_series: BarSeries) -> None:
        daily_series._recompute()
        if daily_series._atr is None:
            return
        atr_arr = daily_series._atr
        valid = atr_arr[~np.isnan(atr_arr)]
        if len(valid) < 10:
            return

        window = valid[-VOL_PERCENTILE_WINDOW:] if len(valid) >= VOL_PERCENTILE_WINDOW else valid
        self.atr_base = float(np.median(window))
        atr_today = float(valid[-1])

        self.vol_pct = percentile_rank(atr_today, window)
        self.extreme_vol = self.vol_pct > 95

        if atr_today > 0:
            raw = self.atr_base / atr_today
            self.vol_factor = max(VOL_FACTOR_MIN, min(VOL_FACTOR_MAX, raw))
        else:
            self.vol_factor = 1.0

        if self.vol_pct < LOW_VOL_THRESHOLD:
            self.vol_factor = min(self.vol_factor, 1.0)
