"""BRS-specific indicators wrapping ATRSS/breakout primitives.

Reuses: ema(), atr(), adx_suite() from strategy.indicators
New: vol_factor, bear_conviction, 4H structure, EMA slope
"""
from __future__ import annotations

import numpy as np

from strategy.indicators import adx_suite, atr, ema


# ---------------------------------------------------------------------------
# EMA slope
# ---------------------------------------------------------------------------

def compute_ema_slope(ema_arr: np.ndarray, lookback: int = 5) -> np.ndarray:
    """Slope of EMA over lookback bars: ema[i] - ema[i-lookback].

    Returns array of same length; first `lookback` values are 0.0.
    """
    out = np.zeros_like(ema_arr, dtype=np.float64)
    if len(ema_arr) <= lookback:
        return out
    out[lookback:] = ema_arr[lookback:] - ema_arr[:-lookback]
    return out


# ---------------------------------------------------------------------------
# VolFactor (spec §3.1)
# ---------------------------------------------------------------------------

def compute_vol_factor(
    atr14_d_today: float,
    atr14_d_history: np.ndarray,
    lookback: int = 60,
    vf_clamp_min: float = 0.35,
    vf_clamp_max: float = 1.5,
    extreme_vol_pct: float = 95.0,
) -> tuple[float, float]:
    """Compute VolFactor and vol_pct (percentile rank).

    Returns:
        (vol_factor, vol_pct) — both floats
    """
    if len(atr14_d_history) < 2 or atr14_d_today <= 0:
        return 1.0, 50.0

    window = atr14_d_history[-lookback:] if len(atr14_d_history) > lookback else atr14_d_history
    valid = window[~np.isnan(window)]
    if len(valid) < 2:
        return 1.0, 50.0

    atr_base = float(np.median(valid))
    if atr_base <= 0:
        return 1.0, 50.0

    # Percentile rank (past-only)
    vol_pct = float(np.sum(valid < atr14_d_today) / len(valid) * 100.0)

    # Raw VolFactor
    vf_raw = atr_base / atr14_d_today
    vf = max(vf_clamp_min, min(vf_clamp_max, vf_raw))

    # Crisis cap (spec §3.1)
    if vol_pct > extreme_vol_pct:
        vf = min(vf, 0.50)
    elif vol_pct < 20:
        vf = min(vf, 1.0)

    return vf, vol_pct


# ---------------------------------------------------------------------------
# Risk regime adjustment (spec §3.2)
# ---------------------------------------------------------------------------

def compute_risk_regime(
    atr14_d: float,
    atr14_d_history: np.ndarray,
    sma_period: int = 50,
) -> float:
    """Risk regime = ATR14_D / SMA(ATR14_D, 50).

    Returns clamp(1.0/risk_regime, 0.60, 1.05).
    """
    if len(atr14_d_history) < sma_period or atr14_d <= 0:
        return 1.0
    sma_val = float(np.mean(atr14_d_history[-sma_period:]))
    if sma_val <= 0:
        return 1.0
    risk_regime = atr14_d / sma_val
    return max(0.60, min(1.05, 1.0 / risk_regime))


# ---------------------------------------------------------------------------
# Bear conviction score (spec §2.5)
# ---------------------------------------------------------------------------

def compute_bear_conviction(
    adx_val: float,
    minus_di: float,
    plus_di: float,
    ema_sep_pct: float,
    price_below_both_emas: bool,
    ema_fast_slope_neg: bool,
) -> float:
    """Monotonic composite bear conviction score 0–100.

    Components:
      - ADX magnitude: 0–30 points (scaled linearly, ADX 10→0, ADX 40→30)
      - -DI dominance: 0–25 points ((-DI - +DI) scaled, 0→0, 20→25)
      - EMA separation: 0–20 points (ema_sep_pct scaled, 0→0, 2.0→20)
      - Price below both EMAs: +15 points
      - EMA_fast slope negative: +10 points
    """
    # ADX magnitude: linear 10-40 → 0-30
    adx_pts = max(0.0, min(30.0, (adx_val - 10.0) / 30.0 * 30.0))

    # -DI dominance
    di_diff = minus_di - plus_di
    di_pts = max(0.0, min(25.0, di_diff / 20.0 * 25.0))

    # EMA separation
    ema_pts = max(0.0, min(20.0, ema_sep_pct / 2.0 * 20.0))

    # Binary components
    below_pts = 15.0 if price_below_both_emas else 0.0
    slope_pts = 10.0 if ema_fast_slope_neg else 0.0

    return min(100.0, adx_pts + di_pts + ema_pts + below_pts + slope_pts)


# ---------------------------------------------------------------------------
# 4H structure (spec §2.4) — slope-based, different from breakout's ADX-based
# ---------------------------------------------------------------------------

def compute_4h_structure(
    closes_4h: np.ndarray,
    ema50_4h: np.ndarray,
    atr14_4h: np.ndarray,
    adx14_4h: np.ndarray,
    price: float,
    slope_lookback: int = 3,
) -> tuple[str, float]:
    """Compute 4H regime using slope + price position.

    Returns:
        (regime_str, slope_val) where regime_str is "BEAR_4H", "BULL_4H", or "CHOP_4H"
    """
    if len(ema50_4h) < slope_lookback + 1 or len(atr14_4h) < 1:
        return "CHOP_4H", 0.0

    slope = float(ema50_4h[-1] - ema50_4h[-1 - slope_lookback])
    slope_th = 0.10 * float(atr14_4h[-1])
    ema50_val = float(ema50_4h[-1])
    adx_val = float(adx14_4h[-1]) if len(adx14_4h) > 0 else 0.0

    # ADX too low → chop
    if adx_val < 18:
        return "CHOP_4H", slope

    if slope < -slope_th and price < ema50_val:
        return "BEAR_4H", slope
    elif slope > slope_th and price > ema50_val:
        return "BULL_4H", slope
    else:
        return "CHOP_4H", slope


# ---------------------------------------------------------------------------
# Donchian channels (for S3 impulse)
# ---------------------------------------------------------------------------

def swing_high_confirmed(highs: np.ndarray, lookback: int = 5) -> np.ndarray:
    """Detect confirmed swing highs with no look-ahead bias.

    At bar i, checks if bar i-lookback is a local maximum over
    highs[i-2*lookback : i+1]. Returns array where non-NaN values
    indicate a confirmed swing high price.
    """
    n = len(highs)
    out = np.full(n, np.nan, dtype=np.float64)
    min_idx = 2 * lookback
    for i in range(min_idx, n):
        candidate = i - lookback
        window = highs[max(0, candidate - lookback):i + 1]
        if highs[candidate] >= np.nanmax(window):
            out[i] = float(highs[candidate])
    return out


def volume_sma(volumes: np.ndarray, period: int = 20) -> np.ndarray:
    """Simple moving average of volume over period bars."""
    n = len(volumes)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out
    cumsum = np.cumsum(volumes.astype(np.float64))
    out[period - 1:] = (cumsum[period - 1:] - np.concatenate([[0], cumsum[:-period]])) / period
    return out


# ---------------------------------------------------------------------------
# Donchian channels (for S3 impulse)
# ---------------------------------------------------------------------------

def donchian_low(lows: np.ndarray, period: int) -> np.ndarray:
    """Rolling lowest low over period (vectorized)."""
    n = len(lows)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or period <= 0:
        return out
    # Use expanding window for first `period` bars, then sliding
    for i in range(min(period, n)):
        out[i] = float(np.nanmin(lows[:i + 1]))
    if n > period:
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(lows, period)
        out[period - 1:] = np.nanmin(windows, axis=1)
    return out


def donchian_high(highs: np.ndarray, period: int) -> np.ndarray:
    """Rolling highest high over period (vectorized)."""
    n = len(highs)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or period <= 0:
        return out
    for i in range(min(period, n)):
        out[i] = float(np.nanmax(highs[:i + 1]))
    if n > period:
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(highs, period)
        out[period - 1:] = np.nanmax(windows, axis=1)
    return out


# ---------------------------------------------------------------------------
# Containment ratio (for S2 box detection, spec §5.2)
# ---------------------------------------------------------------------------

def containment_ratio(
    closes: np.ndarray,
    range_low: float,
    range_high: float,
    lookback: int,
) -> float:
    """Fraction of closes within [range_low, range_high] over lookback bars."""
    if lookback <= 0 or range_high <= range_low:
        return 0.0
    window = closes[-lookback:]
    inside = np.sum((window >= range_low) & (window <= range_high))
    return float(inside / len(window))


# ---------------------------------------------------------------------------
# AVWAP (anchored VWAP) — simplified for backtest (close-only proxy)
# ---------------------------------------------------------------------------

def compute_avwap(
    closes: np.ndarray,
    volumes: np.ndarray,
    anchor_idx: int,
) -> float:
    """Compute anchored VWAP from anchor_idx to end of arrays."""
    if anchor_idx < 0 or anchor_idx >= len(closes):
        return float(closes[-1]) if len(closes) > 0 else 0.0
    c = closes[anchor_idx:]
    v = volumes[anchor_idx:]
    total_vol = np.sum(v)
    if total_vol <= 0:
        return float(np.mean(c))
    return float(np.sum(c * v) / total_vol)


# ---------------------------------------------------------------------------
# Adaptive box length (spec §5.1)
# ---------------------------------------------------------------------------

def compute_box_length(atr14_d: float, atr50_d: float) -> int:
    """Compute adaptive campaign box length from ATR ratio."""
    if atr50_d <= 0:
        return 10
    atr_ratio = atr14_d / atr50_d
    if atr_ratio < 0.70:
        return 6
    elif atr_ratio <= 1.25:
        return 10
    else:
        return 14
