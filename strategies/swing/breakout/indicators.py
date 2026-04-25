"""Multi-Asset Swing Breakout v3.3-ETF — pure indicator computation (no side effects).

All functions accept numpy arrays and return scalars or arrays.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Optional

import numpy as np

from .config import (
    ADX_PERIOD,
    ATR_DAILY_PERIOD,
    ATR_DAILY_LONG_PERIOD,
    ATR_HOURLY_PERIOD,
    EMA_1H_PERIOD,
    EMA_4H_PERIOD,
    EMA_DAILY_PERIOD,
    LOOKBACK_RVOL_D,
    LOOKBACK_SLOT_WEEKS,
    LOOKBACK_SQ,
)
from .models import Regime4H


# ---------------------------------------------------------------------------
# Primitive indicators
# ---------------------------------------------------------------------------

def ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average with SMA seed."""
    out = np.empty_like(arr, dtype=float)
    k = 2.0 / (period + 1)
    out[0] = float(np.mean(arr[:min(period, len(arr))]))
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average."""
    out = np.empty_like(arr, dtype=float)
    for i in range(len(arr)):
        start = max(0, i - period + 1)
        out[i] = float(np.mean(arr[start:i + 1]))
    return out


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
        period: int) -> np.ndarray:
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


def adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
        period: int = ADX_PERIOD) -> np.ndarray:
    """Average Directional Index."""
    n = len(highs)
    plus_dm = np.zeros(n, dtype=float)
    minus_dm = np.zeros(n, dtype=float)
    tr = np.zeros(n, dtype=float)
    tr[0] = highs[0] - lows[0]

    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    alpha = 1.0 / period
    atr_s = np.empty(n, dtype=float)
    plus_di_s = np.empty(n, dtype=float)
    minus_di_s = np.empty(n, dtype=float)
    atr_s[0] = tr[0]
    plus_di_s[0] = plus_dm[0]
    minus_di_s[0] = minus_dm[0]

    for i in range(1, n):
        atr_s[i] = atr_s[i - 1] * (1 - alpha) + tr[i] * alpha
        plus_di_s[i] = plus_di_s[i - 1] * (1 - alpha) + plus_dm[i] * alpha
        minus_di_s[i] = minus_di_s[i - 1] * (1 - alpha) + minus_dm[i] * alpha

    dx = np.zeros(n, dtype=float)
    for i in range(n):
        if atr_s[i] > 0:
            pdi = 100.0 * plus_di_s[i] / atr_s[i]
            mdi = 100.0 * minus_di_s[i] / atr_s[i]
            total = pdi + mdi
            dx[i] = 100.0 * abs(pdi - mdi) / total if total > 0 else 0.0

    adx_out = np.empty(n, dtype=float)
    adx_out[0] = dx[0]
    for i in range(1, n):
        adx_out[i] = adx_out[i - 1] * (1 - alpha) + dx[i] * alpha
    return adx_out


# ---------------------------------------------------------------------------
# Rolling high / low
# ---------------------------------------------------------------------------

def highest(arr: np.ndarray, period: int) -> np.ndarray:
    """Rolling highest over period."""
    out = np.empty_like(arr, dtype=float)
    for i in range(len(arr)):
        start = max(0, i - period + 1)
        out[i] = float(np.max(arr[start:i + 1]))
    return out


def lowest(arr: np.ndarray, period: int) -> np.ndarray:
    """Rolling lowest over period."""
    out = np.empty_like(arr, dtype=float)
    for i in range(len(arr)):
        start = max(0, i - period + 1)
        out[i] = float(np.min(arr[start:i + 1]))
    return out


# ---------------------------------------------------------------------------
# Past-only quantile (spec §5.4, §8.2)
# ---------------------------------------------------------------------------

def past_only_quantile(series: list[float], q: float, lookback: int) -> float:
    """Quantile using only past values (never peeks forward).

    series should contain only values strictly before 'now'.
    """
    if not series:
        return 0.0
    window = series[-lookback:] if len(series) >= lookback else list(series)
    return float(np.quantile(window, q))


# ---------------------------------------------------------------------------
# Containment (spec §5.2)
# ---------------------------------------------------------------------------

def containment_ratio(closes: np.ndarray, range_low: float, range_high: float,
                      L: int) -> float:
    """Fraction of last L closes inside [range_low, range_high]."""
    window = closes[-L:] if len(closes) >= L else closes
    inside = np.sum((window >= range_low) & (window <= range_high))
    return float(inside / len(window))


# ---------------------------------------------------------------------------
# AVWAP (spec §6.1) — campaign-anchored VWAP
# ---------------------------------------------------------------------------

def compute_avwap(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    bar_times: list[datetime],
    anchor_time: datetime,
) -> np.ndarray:
    """AVWAP anchored at anchor_time.

    tp = (H+L+C)/3; AVWAP = cumsum(tp*vol)/cumsum(vol) from anchor.
    Returns array same length as input; values before anchor are NaN.
    """
    n = len(closes)
    out = np.full(n, np.nan, dtype=float)
    cum_tpv = 0.0
    cum_vol = 0.0

    for i in range(n):
        if bar_times[i] < anchor_time:
            continue
        tp = (highs[i] + lows[i] + closes[i]) / 3.0
        cum_tpv += tp * volumes[i]
        cum_vol += volumes[i]
        out[i] = cum_tpv / cum_vol if cum_vol > 0 else closes[i]
    return out


def avwap_last(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    bar_times: list[datetime],
    anchor_time: datetime,
) -> float:
    """Return only the final AVWAP value (no array allocation).

    Vectorized: finds anchor index, then uses np.dot for cumulative VWAP.
    """
    n = len(closes)
    start = 0
    for i in range(n):
        if bar_times[i] >= anchor_time:
            start = i
            break
    else:
        return float('nan')
    tp = (highs[start:] + lows[start:] + closes[start:]) / 3.0
    vol = volumes[start:]
    cum_vol = float(vol.sum())
    if cum_vol <= 0:
        return float(closes[-1])
    return float(np.dot(tp, vol) / cum_vol)


# ---------------------------------------------------------------------------
# WVWAP (spec §6.2) — weekly VWAP, resets Monday 09:30 ET
# ---------------------------------------------------------------------------

def compute_wvwap(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    bar_times: list[datetime],
) -> np.ndarray:
    """Weekly VWAP resetting at Monday 09:30 ET."""
    n = len(closes)
    out = np.empty(n, dtype=float)
    cum_tpv = 0.0
    cum_vol = 0.0
    monday_reset = time(9, 30)

    for i in range(n):
        dt = bar_times[i]
        # Reset on Monday at/after 09:30
        if dt.weekday() == 0 and dt.time() <= monday_reset:
            cum_tpv = 0.0
            cum_vol = 0.0

        tp = (highs[i] + lows[i] + closes[i]) / 3.0
        cum_tpv += tp * volumes[i]
        cum_vol += volumes[i]
        out[i] = cum_tpv / cum_vol if cum_vol > 0 else closes[i]
    return out


# ---------------------------------------------------------------------------
# RVOL_D (spec §3)
# ---------------------------------------------------------------------------

def compute_rvol_d(volumes: np.ndarray, lookback: int = LOOKBACK_RVOL_D) -> float:
    """RVOL_D = volume_D / median(volume_D, past 20 sessions)."""
    if len(volumes) < 2:
        return 1.0
    past = volumes[-(lookback + 1):-1] if len(volumes) > lookback else volumes[:-1]
    if len(past) == 0:
        return 1.0
    med = float(np.median(past))
    return float(volumes[-1] / med) if med > 0 else 1.0


# ---------------------------------------------------------------------------
# RVOL_H slot-normalized (spec §12.1)
# ---------------------------------------------------------------------------

def get_slot_key(dt: datetime) -> tuple[int, int]:
    """Return (day_of_week, hour_slot) for RVOL_H normalization.

    Slots: 09:30-10:30 → 9, 10:30-11:30 → 10, ..., 15:00-16:00 → 15.
    """
    hour = dt.hour
    minute = dt.minute
    # Assign to nearest hour-start slot (bar close time → slot)
    slot_hour = hour if minute <= 30 else hour
    return (dt.weekday(), slot_hour)


def compute_rvol_h(
    current_volume: float,
    slot_key: tuple[int, int],
    slot_medians: dict[tuple[int, int], float],
) -> float:
    """Slot-normalized RVOL_H.

    RVOL_H = volume_H / median(volume_H for same slot over past N weeks).
    """
    med = slot_medians.get(slot_key, 0.0)
    if med <= 0:
        return 1.0
    return current_volume / med


def update_slot_medians(
    volumes_by_slot: dict[tuple[int, int], list[float]],
    lookback_weeks: int = LOOKBACK_SLOT_WEEKS,
) -> dict[tuple[int, int], float]:
    """Recompute slot medians from rolling volume history."""
    medians: dict[tuple[int, int], float] = {}
    for key, vols in volumes_by_slot.items():
        window = vols[-lookback_weeks:] if len(vols) >= lookback_weeks else vols
        medians[key] = float(np.median(window)) if window else 0.0
    return medians


# ---------------------------------------------------------------------------
# Volume score (spec §9.1)
# ---------------------------------------------------------------------------

def volume_score_component_daily(rvol_d: float) -> int:
    """RVOL_D → score component."""
    if rvol_d >= 1.5:
        return 1
    if rvol_d >= 1.1:
        return 0
    return -1


# ---------------------------------------------------------------------------
# 4H Regime classification (spec §7.1)
# ---------------------------------------------------------------------------

def compute_regime_4h(
    closes_4h: np.ndarray,
    highs_4h: np.ndarray,
    lows_4h: np.ndarray,
) -> tuple[Regime4H, float, float]:
    """Classify 4H regime.

    Returns (regime, slope_4h, adx_4h).
    """
    if len(closes_4h) < EMA_4H_PERIOD + 4:
        return Regime4H.RANGE_CHOP, 0.0, 0.0

    ema50_4h = ema(closes_4h, EMA_4H_PERIOD)
    atr14_4h = atr(highs_4h, lows_4h, closes_4h, ATR_HOURLY_PERIOD)
    adx_4h = adx(highs_4h, lows_4h, closes_4h, ADX_PERIOD)

    slope_4h = float(ema50_4h[-1] - ema50_4h[-4]) if len(ema50_4h) >= 4 else 0.0
    slope_th = 0.10 * float(atr14_4h[-1])
    price = float(closes_4h[-1])
    ema_val = float(ema50_4h[-1])
    adx_val = float(adx_4h[-1])

    if adx_val < 20:
        return Regime4H.RANGE_CHOP, slope_4h, adx_val
    if slope_4h > slope_th and price > ema_val:
        return Regime4H.BULL_TREND, slope_4h, adx_val
    if slope_4h < -slope_th and price < ema_val:
        return Regime4H.BEAR_TREND, slope_4h, adx_val
    return Regime4H.RANGE_CHOP, slope_4h, adx_val


# ---------------------------------------------------------------------------
# Daily slope (spec §7.2)
# ---------------------------------------------------------------------------

def compute_daily_slope(closes_d: np.ndarray) -> float:
    """daily_slope = EMA50_D[t] - EMA50_D[t-5]."""
    if len(closes_d) < EMA_DAILY_PERIOD + 6:
        return 0.0
    ema50_d = ema(closes_d, EMA_DAILY_PERIOD)
    return float(ema50_d[-1] - ema50_d[-6])


# ---------------------------------------------------------------------------
# 4H bar construction from 1H bars
# ---------------------------------------------------------------------------

def construct_4h_bars(
    hourly_highs: np.ndarray,
    hourly_lows: np.ndarray,
    hourly_closes: np.ndarray,
    hourly_volumes: np.ndarray,
    hourly_times: list[datetime],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[datetime]]:
    """Aggregate hourly bars into 4H bars (groups of 4).

    Returns (highs_4h, lows_4h, closes_4h, volumes_4h, times_4h).
    """
    n = len(hourly_closes)
    n_4h = n // 4
    if n_4h == 0:
        return (np.array([], dtype=float), np.array([], dtype=float),
                np.array([], dtype=float), np.array([], dtype=float), [])

    # Trim to multiple of 4 from the end
    start = n - n_4h * 4
    highs_4h = np.empty(n_4h, dtype=float)
    lows_4h = np.empty(n_4h, dtype=float)
    closes_4h = np.empty(n_4h, dtype=float)
    volumes_4h = np.empty(n_4h, dtype=float)
    times_4h: list[datetime] = []

    for i in range(n_4h):
        s = start + i * 4
        e = s + 4
        highs_4h[i] = float(np.max(hourly_highs[s:e]))
        lows_4h[i] = float(np.min(hourly_lows[s:e]))
        closes_4h[i] = float(hourly_closes[e - 1])
        volumes_4h[i] = float(np.sum(hourly_volumes[s:e]))
        times_4h.append(hourly_times[e - 1])

    return highs_4h, lows_4h, closes_4h, volumes_4h, times_4h


# ---------------------------------------------------------------------------
# Pullback ref selector (spec §6.3)
# ---------------------------------------------------------------------------

def pullback_ref(price: float, wvwap: float, avwap_h: float,
                 ema20_h: float, atr14_d: float) -> float:
    """Select pullback reference for adds."""
    if atr14_d > 0 and abs(price - wvwap) / atr14_d <= 2.0:
        return wvwap
    if atr14_d > 0 and abs(price - avwap_h) / atr14_d <= 2.0:
        return avwap_h
    return ema20_h


# ---------------------------------------------------------------------------
# Correlation (spec §18.1)
# ---------------------------------------------------------------------------

def rolling_correlation(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    lookback: int = 60,
) -> float:
    """Rolling correlation of 4H returns."""
    n = min(len(returns_a), len(returns_b), lookback)
    if n < 10:
        return 0.0
    a = returns_a[-n:]
    b = returns_b[-n:]
    corr = np.corrcoef(a, b)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0
