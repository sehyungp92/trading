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
    """Maintains a rolling window of bars with incrementally updated indicators.

    Instead of recomputing all indicators from scratch on every bar (O(N) per
    bar, ~6 full-array passes), this uses O(1) incremental EMA/ATR/MACD updates
    after a one-time warmup period.
    """

    # Minimum bars needed before incremental mode activates.
    # Must be >= max(EMA_SLOW_PERIOD, MACD_SLOW + MACD_SIGNAL - 1, ATR_PERIOD)
    _WARMUP = max(EMA_SLOW_PERIOD, MACD_SLOW + MACD_SIGNAL - 1, ATR_PERIOD)

    def __init__(self, tf: TF, maxlen: int = 300):
        self.tf = tf
        self.bars: deque[Bar] = deque(maxlen=maxlen)
        self._maxlen = maxlen

        # ── EMA multipliers (precomputed) ──
        self._k_ema_fast = 2.0 / (EMA_FAST_PERIOD + 1)
        self._k_ema_slow = 2.0 / (EMA_SLOW_PERIOD + 1)
        self._k_atr = 1.0 / ATR_PERIOD
        self._k_macd_fast = 2.0 / (MACD_FAST + 1)
        self._k_macd_slow = 2.0 / (MACD_SLOW + 1)
        self._k_macd_sig = 2.0 / (MACD_SIGNAL + 1)

        # ── Running state for incremental updates ──
        self._ema_fast_val: float = 0.0
        self._ema_slow_val: float = 0.0
        self._atr_val: float = 0.0
        self._macd_fast_ema: float = 0.0
        self._macd_slow_ema: float = 0.0
        self._macd_sig_ema: float = 0.0
        self._prev_close: float = 0.0
        self._warmed_up: bool = False
        self._bar_count: int = 0

        # ── Indicator history deques ──
        self._closes_d: deque[float] = deque(maxlen=maxlen)
        self._highs_d: deque[float] = deque(maxlen=maxlen)
        self._lows_d: deque[float] = deque(maxlen=maxlen)
        self._ema_fast_d: deque[float] = deque(maxlen=maxlen)
        self._ema_slow_d: deque[float] = deque(maxlen=maxlen)
        self._atr_d: deque[float] = deque(maxlen=maxlen)
        self._macd_line_d: deque[float] = deque(maxlen=maxlen)
        self._macd_signal_d: deque[float] = deque(maxlen=maxlen)
        self._macd_hist_d: deque[float] = deque(maxlen=maxlen)

    def add_bar(self, bar: Bar) -> None:
        self.bars.append(bar)
        self._bar_count += 1
        c, h, l = bar.close, bar.high, bar.low

        self._closes_d.append(c)
        self._highs_d.append(h)
        self._lows_d.append(l)

        if not self._warmed_up:
            if self._bar_count >= self._WARMUP:
                self._full_seed()
                self._warmed_up = True
            else:
                # Append NaN placeholders during warmup
                self._ema_fast_d.append(float("nan"))
                self._ema_slow_d.append(float("nan"))
                self._atr_d.append(float("nan"))
                self._macd_line_d.append(float("nan"))
                self._macd_signal_d.append(float("nan"))
                self._macd_hist_d.append(float("nan"))
                self._prev_close = c
            return

        # ── Incremental O(1) updates ──

        # True Range
        tr = max(h - l, abs(h - self._prev_close), abs(l - self._prev_close))

        # ATR (Wilder smoothing: k = 1/period)
        self._atr_val = tr * self._k_atr + self._atr_val * (1 - self._k_atr)

        # EMA fast/slow
        self._ema_fast_val = c * self._k_ema_fast + self._ema_fast_val * (1 - self._k_ema_fast)
        self._ema_slow_val = c * self._k_ema_slow + self._ema_slow_val * (1 - self._k_ema_slow)

        # MACD internal EMAs
        self._macd_fast_ema = c * self._k_macd_fast + self._macd_fast_ema * (1 - self._k_macd_fast)
        self._macd_slow_ema = c * self._k_macd_slow + self._macd_slow_ema * (1 - self._k_macd_slow)
        macd_line_val = self._macd_fast_ema - self._macd_slow_ema
        self._macd_sig_ema = macd_line_val * self._k_macd_sig + self._macd_sig_ema * (1 - self._k_macd_sig)
        macd_hist_val = macd_line_val - self._macd_sig_ema

        # Append to history deques
        self._ema_fast_d.append(self._ema_fast_val)
        self._ema_slow_d.append(self._ema_slow_val)
        self._atr_d.append(self._atr_val)
        self._macd_line_d.append(macd_line_val)
        self._macd_signal_d.append(self._macd_sig_ema)
        self._macd_hist_d.append(macd_hist_val)

        self._prev_close = c

    def _full_seed(self) -> None:
        """Compute full indicator arrays from deque to seed running state.

        Called once when bar_count reaches _WARMUP. After this, all updates
        are incremental.
        """
        closes = np.array(self._closes_d)
        highs = np.array(self._highs_d)
        lows = np.array(self._lows_d)
        n = len(closes)

        # EMA fast/slow
        ema_f = ema(closes, EMA_FAST_PERIOD)
        ema_s = ema(closes, EMA_SLOW_PERIOD)

        # ATR
        atr_arr = atr(highs, lows, closes, ATR_PERIOD)

        # MACD (with separate internal EMAs)
        macd_f_ema = ema(closes, MACD_FAST)
        macd_s_ema = ema(closes, MACD_SLOW)
        macd_line_arr = macd_f_ema - macd_s_ema
        # Signal EMA over valid MACD line values
        valid_mask = ~np.isnan(macd_line_arr)
        valid_line = macd_line_arr[valid_mask]
        sig_arr = ema(valid_line, MACD_SIGNAL) if len(valid_line) >= MACD_SIGNAL else np.array([])
        # Map signal back to full array
        macd_signal_full = np.full(n, np.nan)
        valid_start = np.argmax(valid_mask) if valid_mask.any() else 0
        if len(sig_arr) > 0:
            macd_signal_full[valid_start:valid_start + len(sig_arr)] = sig_arr
        macd_hist_arr = macd_line_arr - macd_signal_full

        # Populate history deques (replace NaN placeholders)
        self._ema_fast_d.clear()
        self._ema_slow_d.clear()
        self._atr_d.clear()
        self._macd_line_d.clear()
        self._macd_signal_d.clear()
        self._macd_hist_d.clear()
        for i in range(n):
            self._ema_fast_d.append(float(ema_f[i]))
            self._ema_slow_d.append(float(ema_s[i]))
            self._atr_d.append(float(atr_arr[i]))
            self._macd_line_d.append(float(macd_line_arr[i]))
            self._macd_signal_d.append(float(macd_signal_full[i]))
            self._macd_hist_d.append(float(macd_hist_arr[i]))

        # Seed running state from last valid values
        self._ema_fast_val = float(ema_f[n - 1])
        self._ema_slow_val = float(ema_s[n - 1])
        self._atr_val = float(atr_arr[n - 1]) if not np.isnan(atr_arr[n - 1]) else 0.0
        self._macd_fast_ema = float(macd_f_ema[n - 1])
        self._macd_slow_ema = float(macd_s_ema[n - 1])
        self._macd_sig_ema = float(macd_signal_full[n - 1]) if not np.isnan(macd_signal_full[n - 1]) else 0.0
        self._prev_close = float(closes[n - 1])

    @property
    def last_close(self) -> float:
        return self._closes_d[-1] if self._closes_d else 0.0

    @property
    def last_high(self) -> float:
        return self._highs_d[-1] if self._highs_d else 0.0

    @property
    def last_low(self) -> float:
        return self._lows_d[-1] if self._lows_d else 0.0

    def close_n_ago(self, n: int) -> float:
        idx = len(self._closes_d) - 1 - n
        if 0 <= idx < len(self._closes_d):
            return self._closes_d[idx]
        return 0.0

    def ema_fast(self) -> float:
        if not self._ema_fast_d:
            return 0.0
        v = self._ema_fast_d[-1]
        return v if not math.isnan(v) else 0.0

    def ema_slow(self) -> float:
        if not self._ema_slow_d:
            return 0.0
        v = self._ema_slow_d[-1]
        return v if not math.isnan(v) else 0.0

    def current_atr(self) -> float:
        if not self._atr_d:
            return 0.0
        v = self._atr_d[-1]
        return v if not math.isnan(v) else 0.0

    def atr_at(self, idx: int) -> float:
        if 0 <= idx < len(self._atr_d):
            v = self._atr_d[idx]
            if not math.isnan(v):
                return v
        return 0.0

    def macd_line_at(self, idx: int) -> float:
        if 0 <= idx < len(self._macd_line_d):
            v = self._macd_line_d[idx]
            if not math.isnan(v):
                return v
        return 0.0

    def macd_line_now(self) -> float:
        if not self._macd_line_d:
            return 0.0
        v = self._macd_line_d[-1]
        return v if not math.isnan(v) else 0.0

    def macd_hist_now(self) -> float:
        if not self._macd_hist_d:
            return 0.0
        v = self._macd_hist_d[-1]
        return v if not math.isnan(v) else 0.0

    def macd_line_n_ago(self, n: int) -> float:
        idx = len(self._macd_line_d) - 1 - n
        if 0 <= idx < len(self._macd_line_d):
            v = self._macd_line_d[idx]
            if not math.isnan(v):
                return v
        return 0.0

    def macd_hist_n_ago(self, n: int) -> float:
        idx = len(self._macd_hist_d) - 1 - n
        if 0 <= idx < len(self._macd_hist_d):
            v = self._macd_hist_d[idx]
            if not math.isnan(v):
                return v
        return 0.0

    def highest_high(self, lookback: int) -> float:
        if not self._highs_d:
            return 0.0
        # Iterate last N elements of deque (faster than converting to array)
        n = min(lookback, len(self._highs_d))
        start = len(self._highs_d) - n
        best = self._highs_d[start]
        for i in range(start + 1, len(self._highs_d)):
            if self._highs_d[i] > best:
                best = self._highs_d[i]
        return best

    def lowest_low(self, lookback: int) -> float:
        if not self._lows_d:
            return float("inf")
        n = min(lookback, len(self._lows_d))
        start = len(self._lows_d) - n
        best = self._lows_d[start]
        for i in range(start + 1, len(self._lows_d)):
            if self._lows_d[i] < best:
                best = self._lows_d[i]
        return best

    def bar_range(self) -> float:
        if not self.bars:
            return 0.0
        b = self.bars[-1]
        return b.high - b.low

    def trend_strength_at(self, idx: int) -> float:
        """Return |ema_fast - ema_slow| / atr at deque index `idx`."""
        if not (0 <= idx < len(self._atr_d)):
            return 0.0
        a = self._atr_d[idx]
        if a <= 0 or math.isnan(a):
            return 0.0
        ef = self._ema_fast_d[idx] if idx < len(self._ema_fast_d) else float("nan")
        es = self._ema_slow_d[idx] if idx < len(self._ema_slow_d) else float("nan")
        if math.isnan(ef) or math.isnan(es):
            return 0.0
        return abs(ef - es) / a

    def atr_rolling(self, period: int) -> float:
        """ATR computed over last `period` bars (for Class R vol signature)."""
        if len(self._atr_d) < period:
            return 0.0
        total = 0.0
        count = 0
        start = len(self._atr_d) - period
        for i in range(start, len(self._atr_d)):
            v = self._atr_d[i]
            if not math.isnan(v):
                total += v
                count += 1
        return total / count if count > 0 else 0.0

    def atr_rolling_prev(self, period: int) -> float:
        """ATR rolling mean over `period` bars ending one bar before the latest."""
        if len(self._atr_d) < period + 1:
            return 0.0
        total = 0.0
        count = 0
        start = len(self._atr_d) - period - 1
        end = len(self._atr_d) - 1
        for i in range(start, end):
            v = self._atr_d[i]
            if not math.isnan(v):
                total += v
                count += 1
        return total / count if count > 0 else 0.0


class VolEngine:
    """Daily volatility engine: ATR_base (median), vol_pct, VolFactor."""

    def __init__(self):
        self.atr_base: float = 0.0
        self.vol_pct: float = 50.0
        self.vol_factor: float = 1.0
        self.extreme_vol: bool = False

    def update(self, daily_series: BarSeries) -> None:
        if len(daily_series._atr_d) < 10:
            return

        # Collect valid (non-NaN) ATR values from deque
        valid = [v for v in daily_series._atr_d if not math.isnan(v)]
        if len(valid) < 10:
            return

        window = valid[-VOL_PERCENTILE_WINDOW:] if len(valid) >= VOL_PERCENTILE_WINDOW else valid
        window_arr = np.array(window)
        self.atr_base = float(np.median(window_arr))
        atr_today = valid[-1]

        self.vol_pct = percentile_rank(atr_today, window_arr)
        self.extreme_vol = self.vol_pct > 95

        if atr_today > 0:
            raw = self.atr_base / atr_today
            self.vol_factor = max(VOL_FACTOR_MIN, min(VOL_FACTOR_MAX, raw))
        else:
            self.vol_factor = 1.0

        if self.vol_pct < LOW_VOL_THRESHOLD:
            self.vol_factor = min(self.vol_factor, 1.0)
