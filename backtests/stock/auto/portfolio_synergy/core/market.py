from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import pandas as pd


class CausalPriceBook:
    """Return only prices from bars completed strictly before an event."""

    def __init__(
        self,
        bars_by_symbol: Mapping[str, Any] | None = None,
        *,
        daily_close: pd.DataFrame | None = None,
    ) -> None:
        self._intraday: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for symbol, frame in (bars_by_symbol or {}).items():
            if frame is None or "close" not in frame:
                continue
            series = frame["close"].dropna().copy()
            if series.empty:
                continue
            index = pd.to_datetime(series.index, utc=True)
            order = np.argsort(index.asi8)
            self._intraday[str(symbol)] = (
                index.asi8[order],
                series.to_numpy(dtype=float)[order],
            )

        self._daily: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if daily_close is not None:
            for symbol in daily_close.columns:
                series = daily_close[symbol].dropna()
                if series.empty:
                    continue
                series_index = pd.to_datetime(series.index, utc=True)
                order = np.argsort(series_index.asi8)
                self._daily[str(symbol)] = (
                    series_index.asi8[order],
                    series.to_numpy(dtype=float)[order],
                )

    def __call__(self, symbol: str, at: datetime) -> float | None:
        timestamp = _aware_utc(at)
        value = _strictly_prior(self._intraday.get(symbol), timestamp)
        if value is not None:
            return value

        # Daily bars are session-close observations.  A normalized timestamp
        # for the event's date would be look-ahead during that session, so use
        # only a strictly earlier session.
        prior_session = datetime.combine(
            timestamp.date(), datetime.min.time(), tzinfo=timezone.utc
        )
        return _strictly_prior(self._daily.get(symbol), prior_session)


def _strictly_prior(
    series: tuple[np.ndarray, np.ndarray] | None,
    timestamp: datetime,
) -> float | None:
    if series is None:
        return None
    times, values = series
    target = pd.Timestamp(timestamp).value
    index = int(np.searchsorted(times, target, side="left")) - 1
    if index < 0:
        return None
    value = float(values[index])
    return value if np.isfinite(value) and value > 0.0 else None


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
