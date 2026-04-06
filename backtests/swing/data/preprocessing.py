"""Data preprocessing: gap filling, timezone normalization, alignment."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to Regular Trading Hours only (09:30-16:00 ET).

    Removes pre-market and after-hours bars so that signal detection
    runs exclusively on tradeable bars.  Bars at 09:00 ET are included
    to provide indicator context (entry is still restricted by the engine
    until 09:45 ET).
    """
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    idx_et = df.index.tz_convert(et)
    minutes = idx_et.hour * 60 + idx_et.minute
    # Keep bars from 09:00 (540) through 15:59 (959) on weekdays
    mask = (minutes >= 540) & (minutes < 960) & (idx_et.weekday < 5)
    return df.loc[mask]


def normalize_timezone(df: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    """Ensure DatetimeIndex is in the specified timezone."""
    if df.index.tz is None:
        df = df.tz_localize(tz)
    else:
        df = df.tz_convert(tz)
    return df


def fill_gaps(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """Forward-fill missing timestamps, mark gaps.

    Inserts rows for missing timestamps with NaN OHLCV and a ``gap=True``
    column.  Existing rows get ``gap=False``.
    """
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq, tz=df.index.tz)
    df = df.reindex(full_idx)
    df["gap"] = df["close"].isna()
    # Forward-fill close only (for indicator seeding); leave OHLCV as NaN for gap bars
    df["close"] = df["close"].ffill()
    return df


def mark_invalid_blocks(df: pd.DataFrame, max_consecutive: int = 5) -> pd.DataFrame:
    """Mark contiguous gap blocks longer than max_consecutive.

    Adds ``invalid=True`` for bars inside blocks where the engine should skip
    entry evaluation.
    """
    if "gap" not in df.columns:
        df["invalid"] = False
        return df

    gap_groups = (~df["gap"]).cumsum()
    gap_lengths = df.groupby(gap_groups)["gap"].transform("sum")
    df["invalid"] = df["gap"] & (gap_lengths > max_consecutive)
    return df


def align_daily_to_hourly(
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> np.ndarray:
    """Map each hourly bar to the index of the most recent *completed* daily bar.

    Returns an integer array of length ``len(hourly_df)`` where each element
    is the positional index into ``daily_df``.  This prevents look-ahead bias:
    a daily bar is only available after its close.

    Daily bars are assumed to represent the close of that calendar day.
    An hourly bar at time ``t`` uses the daily bar whose date is strictly
    before ``t``'s date (i.e., yesterday's daily bar during the current day,
    switching to today's daily bar only on the first bar of the next day).
    """
    daily_dates = daily_df.index.normalize()
    hourly_dates = hourly_df.index.normalize()

    idx_map = np.empty(len(hourly_df), dtype=np.int64)
    daily_pos = 0

    for i in range(len(hourly_df)):
        h_date = hourly_dates[i]
        # Advance daily_pos to the last daily bar whose date < hourly bar's date
        while daily_pos < len(daily_dates) - 1 and daily_dates[daily_pos + 1] < h_date:
            daily_pos += 1
        # Use daily bar only if its date is strictly before the hourly bar's date
        if daily_dates[daily_pos] < h_date:
            idx_map[i] = daily_pos
        elif daily_pos > 0:
            idx_map[i] = daily_pos - 1
        else:
            idx_map[i] = 0

    return idx_map


@dataclass
class NumpyBars:
    """Contiguous numpy arrays extracted from a DataFrame."""

    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    volumes: np.ndarray
    times: np.ndarray  # datetime64

    def __len__(self) -> int:
        return len(self.closes)


def resample_1h_to_4h(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1H OHLCV bars to 4H using standard aggregation.

    Uses UTC boundaries (00:00, 04:00, 08:00, 12:00, 16:00, 20:00).
    The resulting DataFrame has the same timezone as the input.
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in hourly_df.columns:
        agg["volume"] = "sum"

    resampled = hourly_df.resample("4h", offset="0h").agg(agg)
    # Drop rows where all OHLC are NaN (incomplete periods at boundaries)
    resampled = resampled.dropna(subset=["open", "close"])
    return resampled


def align_4h_to_hourly(
    hourly_df: pd.DataFrame,
    four_hour_df: pd.DataFrame,
) -> np.ndarray:
    """Map each hourly bar to the index of the most recent *completed* 4H bar.

    Returns an integer array of length ``len(hourly_df)`` where each element
    is the positional index into ``four_hour_df``.

    A 4H bar is available only after its close. For example, the 4H bar
    closing at 04:00 is available starting from the 05:00 hourly bar.
    The hourly bars at 01:00-04:00 use the previous 4H bar (closing at 00:00).
    """
    four_hour_times = four_hour_df.index
    hourly_times = hourly_df.index

    idx_map = np.empty(len(hourly_df), dtype=np.int64)
    fh_pos = 0

    for i in range(len(hourly_times)):
        h_time = hourly_times[i]
        # Advance fh_pos to the last 4H bar whose time < hourly bar's time
        while fh_pos < len(four_hour_times) - 1 and four_hour_times[fh_pos + 1] < h_time:
            fh_pos += 1
        # Use 4H bar only if its time is strictly before the hourly bar's time
        if four_hour_times[fh_pos] < h_time:
            idx_map[i] = fh_pos
        elif fh_pos > 0:
            idx_map[i] = fh_pos - 1
        else:
            idx_map[i] = 0

    return idx_map


def build_numpy_arrays(df: pd.DataFrame) -> NumpyBars:
    """Extract OHLCV columns as contiguous float64 arrays."""
    return NumpyBars(
        opens=np.ascontiguousarray(df["open"].values, dtype=np.float64),
        highs=np.ascontiguousarray(df["high"].values, dtype=np.float64),
        lows=np.ascontiguousarray(df["low"].values, dtype=np.float64),
        closes=np.ascontiguousarray(df["close"].values, dtype=np.float64),
        volumes=np.ascontiguousarray(df["volume"].values, dtype=np.float64) if "volume" in df.columns else np.zeros(len(df)),
        times=df.index.values,
    )
