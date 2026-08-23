"""Causal daily/intraday price-basis alignment for equity replay data."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from backtests.stock.data.calendar import EXCHANGE_TIMEZONE


_COMMON_SPLIT_FACTORS = (0.1, 0.2, 0.25, 1.0 / 3.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0)
_PRICE_COLUMNS = ("open", "high", "low", "close", "wap")


def _snap_split_factor(ratio: float, *, relative_tolerance: float = 0.08) -> float:
    """Return a common split factor only when the ratio is unambiguous."""
    if not np.isfinite(ratio) or ratio <= 0:
        return 1.0
    nearest = min(_COMMON_SPLIT_FACTORS, key=lambda value: abs(ratio / value - 1.0))
    if nearest == 1.0:
        return 1.0
    relative_error = abs(ratio / nearest - 1.0)
    return float(nearest) if relative_error <= relative_tolerance else 1.0


def align_intraday_to_daily_price_basis(
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[date, float]]:
    """Align intraday bars to the daily series using only session-open data.

    Some retained IBKR intraday histories are raw across a later stock split
    while a subsequent daily backfill is split-adjusted.  The daily open and
    first RTH intraday open describe the same time-available price, so a clear
    common split ratio between them can be corrected without using the day's
    future close or strategy outcomes.
    """
    if intraday.empty or daily.empty or "open" not in intraday or "open" not in daily:
        return intraday, {}

    intraday_index = pd.DatetimeIndex(pd.to_datetime(intraday.index, utc=True))
    local_index = intraday_index.tz_convert(EXCHANGE_TIMEZONE)
    intraday_dates = local_index.date
    daily_dates = pd.DatetimeIndex(pd.to_datetime(daily.index, utc=True)).date
    daily_open = {
        day: float(value)
        for day, value in zip(daily_dates, daily["open"].to_numpy(), strict=True)
        if np.isfinite(float(value)) and float(value) > 0
    }
    mask = (local_index.hour == 9) & (local_index.minute == 30)
    first_rth: dict[date, float] = {}
    opens = intraday["open"].to_numpy(dtype=float)
    for idx in np.flatnonzero(mask):
        day = intraday_dates[idx]
        if day not in first_rth and np.isfinite(opens[idx]) and opens[idx] > 0:
            first_rth[day] = float(opens[idx])

    factors: dict[date, float] = {}
    for day, intraday_open in first_rth.items():
        reference_open = daily_open.get(day)
        if reference_open is None:
            continue
        factor = _snap_split_factor(intraday_open / reference_open)
        if factor != 1.0:
            factors[day] = factor
    if not factors:
        return intraday, {}

    result = intraday.copy()
    row_factors = np.fromiter((factors.get(day, 1.0) for day in intraday_dates), dtype=float, count=len(intraday_dates))
    adjusted = row_factors != 1.0
    for column in _PRICE_COLUMNS:
        if column in result.columns:
            values = result[column].to_numpy(dtype=float, copy=True)
            values[adjusted] = values[adjusted] / row_factors[adjusted]
            result[column] = values
    if "volume" in result.columns:
        volumes = result["volume"].to_numpy(dtype=float, copy=True)
        volumes[adjusted] = volumes[adjusted] * row_factors[adjusted]
        result["volume"] = volumes
    result.attrs["daily_price_basis_adjustments"] = {
        day.isoformat(): factor for day, factor in sorted(factors.items())
    }
    return result, factors
