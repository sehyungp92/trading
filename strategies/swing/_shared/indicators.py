"""Shared indicator primitives for swing ETF strategies.

All functions are pure numpy-in/numpy-out. Strategy-specific composition
belongs in each strategy's ``indicators.py`` module.
"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
from numpy import ndarray


def sma(arr: ndarray, period: int) -> ndarray:
    values = np.asarray(arr, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if period <= 0 or len(values) < period:
        return out
    csum = np.cumsum(np.insert(values, 0, 0.0))
    out[period - 1:] = (csum[period:] - csum[:-period]) / period
    return out


def ema(arr: ndarray, period: int) -> ndarray:
    values = np.asarray(arr, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if period <= 0 or len(values) < period:
        return out
    seed = float(np.nanmean(values[:period]))
    out[period - 1] = seed
    k = 2.0 / (period + 1.0)
    prev = seed
    for i in range(period, len(values)):
        value = values[i]
        if np.isnan(value):
            out[i] = prev
            continue
        prev = value * k + prev * (1.0 - k)
        out[i] = prev
    return out


def true_range(highs: ndarray, lows: ndarray, closes: ndarray) -> ndarray:
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    out = np.full(len(closes), np.nan, dtype=float)
    if len(closes) == 0:
        return out
    out[0] = highs[0] - lows[0]
    for i in range(1, len(closes)):
        out[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    return out


def atr(highs: ndarray, lows: ndarray, closes: ndarray, period: int) -> ndarray:
    tr = true_range(highs, lows, closes)
    out = np.full(len(tr), np.nan, dtype=float)
    if period <= 0 or len(tr) < period:
        return out
    seed = float(np.nanmean(tr[:period]))
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(tr)):
        prev = (prev * (period - 1) + tr[i]) / period
        out[i] = prev
    return out


def rolling_percentile(arr: ndarray, lookback: int) -> ndarray:
    values = np.asarray(arr, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if lookback <= 1:
        return out
    for i in range(lookback - 1, len(values)):
        window = values[i - lookback + 1:i + 1]
        window = window[~np.isnan(window)]
        current = values[i]
        if len(window) == 0 or np.isnan(current):
            continue
        out[i] = float(np.mean(window <= current) * 100.0)
    return out


def rsi(closes: ndarray, period: int = 14) -> ndarray:
    values = np.asarray(closes, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if period <= 0 or len(values) <= period:
        return out
    deltas = np.diff(values)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.nanmean(gains[:period]))
    avg_loss = float(np.nanmean(losses[:period]))
    out[period] = _rsi_value(avg_gain, avg_loss)
    for i in range(period + 1, len(values)):
        gain = gains[i - 1]
        loss = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = _rsi_value(avg_gain, avg_loss)
    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def adx(
    highs: ndarray,
    lows: ndarray,
    closes: ndarray,
    period: int = 14,
) -> tuple[ndarray, ndarray, ndarray]:
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    adx_arr = np.full(n, np.nan, dtype=float)
    plus_di = np.full(n, np.nan, dtype=float)
    minus_di = np.full(n, np.nan, dtype=float)
    if period <= 0 or n <= period * 2:
        return adx_arr, plus_di, minus_di

    tr = true_range(highs, lows, closes)
    plus_dm = np.zeros(n, dtype=float)
    minus_dm = np.zeros(n, dtype=float)
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm[i] = up if up > down and up > 0 else 0.0
        minus_dm[i] = down if down > up and down > 0 else 0.0

    tr_s = np.full(n, np.nan, dtype=float)
    plus_s = np.full(n, np.nan, dtype=float)
    minus_s = np.full(n, np.nan, dtype=float)
    tr_s[period] = np.nansum(tr[1:period + 1])
    plus_s[period] = np.nansum(plus_dm[1:period + 1])
    minus_s[period] = np.nansum(minus_dm[1:period + 1])
    for i in range(period + 1, n):
        tr_s[i] = tr_s[i - 1] - tr_s[i - 1] / period + tr[i]
        plus_s[i] = plus_s[i - 1] - plus_s[i - 1] / period + plus_dm[i]
        minus_s[i] = minus_s[i - 1] - minus_s[i - 1] / period + minus_dm[i]
        if tr_s[i] > 0:
            plus_di[i] = 100.0 * plus_s[i] / tr_s[i]
            minus_di[i] = 100.0 * minus_s[i] / tr_s[i]

    dx = np.full(n, np.nan, dtype=float)
    denom = plus_di + minus_di
    valid = denom > 0
    dx[valid] = 100.0 * np.abs(plus_di[valid] - minus_di[valid]) / denom[valid]
    seed_idx = period * 2
    adx_arr[seed_idx] = float(np.nanmean(dx[period + 1:seed_idx + 1]))
    for i in range(seed_idx + 1, n):
        adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx[i]) / period
    return adx_arr, plus_di, minus_di


def macd(
    closes: ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[ndarray, ndarray, ndarray]:
    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)
    line = fast_ema - slow_ema
    valid_start = slow - 1
    sig = np.full(len(line), np.nan, dtype=float)
    if 0 <= valid_start < len(line):
        sig_tail = ema(line[valid_start:], signal)
        sig[valid_start:] = sig_tail
    hist = line - sig
    return line, sig, hist


def bollinger_bands(
    closes: ndarray,
    period: int = 20,
    std_mult: float = 2.0,
) -> tuple[ndarray, ndarray, ndarray]:
    values = np.asarray(closes, dtype=float)
    middle = sma(values, period)
    upper = np.full(len(values), np.nan, dtype=float)
    lower = np.full(len(values), np.nan, dtype=float)
    if period <= 0 or len(values) < period:
        return upper, middle, lower
    for i in range(period - 1, len(values)):
        window = values[i - period + 1:i + 1]
        std = float(np.nanstd(window))
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
    return upper, middle, lower


def bb_width(upper: ndarray, lower: ndarray, middle: ndarray) -> ndarray:
    upper = np.asarray(upper, dtype=float)
    lower = np.asarray(lower, dtype=float)
    middle = np.asarray(middle, dtype=float)
    out = np.full(len(middle), np.nan, dtype=float)
    valid = middle != 0
    out[valid] = (upper[valid] - lower[valid]) / middle[valid]
    return out


def volume_sma(volumes: ndarray, period: int = 20) -> ndarray:
    return sma(np.asarray(volumes, dtype=float), period)


def z_score_from_ema(closes: ndarray, ema_arr: ndarray, atr_arr: ndarray) -> ndarray:
    closes = np.asarray(closes, dtype=float)
    ema_arr = np.asarray(ema_arr, dtype=float)
    atr_arr = np.asarray(atr_arr, dtype=float)
    out = np.full(len(closes), np.nan, dtype=float)
    valid = atr_arr > 0
    out[valid] = (closes[valid] - ema_arr[valid]) / atr_arr[valid]
    return out


def vwap_anchored(
    highs: ndarray,
    lows: ndarray,
    closes: ndarray,
    volumes: ndarray,
    timestamps: ndarray,
    anchor_hour: int = 9,
    anchor_minute: int = 30,
    tz: str = "America/New_York",
) -> ndarray:
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    out = np.full(len(closes), np.nan, dtype=float)
    zone = ZoneInfo(tz)
    current_day = None
    anchored = False
    pv_sum = 0.0
    vol_sum = 0.0
    for i, raw_ts in enumerate(timestamps):
        dt = _coerce_dt(raw_ts).astimezone(zone)
        day = dt.date()
        if day != current_day:
            current_day = day
            anchored = False
            pv_sum = 0.0
            vol_sum = 0.0
        if not anchored:
            anchored = (dt.hour, dt.minute) >= (anchor_hour, anchor_minute)
        if not anchored:
            continue
        typical = (highs[i] + lows[i] + closes[i]) / 3.0
        vol = 0.0 if np.isnan(volumes[i]) else max(float(volumes[i]), 0.0)
        pv_sum += typical * vol
        vol_sum += vol
        if vol_sum > 0:
            out[i] = pv_sum / vol_sum
    return out


def _coerce_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, np.datetime64):
        seconds = value.astype("datetime64[s]").astype(np.int64)
        return datetime.fromtimestamp(int(seconds), tz=timezone.utc)
    if isinstance(value, (int, np.integer)):
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    return np.datetime64(value).astype("datetime64[s]").astype(datetime).replace(tzinfo=timezone.utc)

