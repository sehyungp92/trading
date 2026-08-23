"""Versioned US-equity session rules used by stock data and replay.

The implementation is deliberately local to the standalone ``trading`` repository.
It models the regular XNYS/XNAS weekday calendar, the recurring full-day holidays,
and the published early closes needed by the retained and official research window.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd


EXCHANGE_TIMEZONE = "America/New_York"
CALENDAR_VERSION = "us_equities_xnys_xnas_rules_2024_2026_v2"
RTH_SESSION_POLICY = "us_equity_rth_0930_exchange_close_v1"
EXTENDED_SESSION_POLICY = "us_equity_extended_0400_2000_v1"
RAW_SESSION_POLICY = "raw_cache_unfiltered_v1"

_RTH_OPEN = time(9, 30)
_RTH_CLOSE = time(16, 0)
_EARLY_CLOSE = time(13, 0)
_EXTENDED_OPEN = time(4, 0)
_EXTENDED_CLOSE = time(20, 0)

# Explicitly versioned rather than silently relying on an optional calendar package.
# Extend the calendar version whenever this table is extended or corrected.
_EARLY_CLOSE_DATES = {
    date(2024, 7, 3),
    date(2024, 11, 29),
    date(2024, 12, 24),
    date(2025, 7, 3),
    date(2025, 11, 28),
    date(2025, 12, 24),
    date(2026, 11, 27),
    date(2026, 12, 24),
}

_AD_HOC_CLOSURE_DATES = {
    date(2025, 1, 9),  # National day of mourning for President Jimmy Carter.
}


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    current = date(year, month, 1)
    current += timedelta(days=(weekday - current.weekday()) % 7)
    return current + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        current = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        current = date(year, month + 1, 1) - timedelta(days=1)
    return current - timedelta(days=(current.weekday() - weekday) % 7)


def _easter_sunday(year: int) -> date:
    """Gregorian Easter using the Anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def exchange_holidays(year: int) -> set[date]:
    holidays = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Washington's Birthday
        _easter_sunday(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0),  # Memorial Day
        _observed(date(year, 6, 19)),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),
    }
    # New Year's Day can be observed in the preceding calendar year.
    next_new_year = _observed(date(year + 1, 1, 1))
    if next_new_year.year == year:
        holidays.add(next_new_year)
    return holidays


def is_trading_day(day: date) -> bool:
    return (
        day.weekday() < 5
        and day not in exchange_holidays(day.year)
        and day not in _AD_HOC_CLOSURE_DATES
    )


def session_close(day: date) -> time:
    if not is_trading_day(day):
        raise ValueError(f"{day.isoformat()} is not a US-equity trading day")
    return _EARLY_CLOSE if day in _EARLY_CLOSE_DATES else _RTH_CLOSE


def trading_days(start: date, end: date) -> list[date]:
    if end < start:
        return []
    return [
        value.date()
        for value in pd.date_range(start=start, end=end, freq="D")
        if is_trading_day(value.date())
    ]


def timeframe_delta(timeframe: str) -> timedelta:
    normalized = timeframe.lower()
    if normalized in {"1d", "daily"}:
        return timedelta(days=1)
    suffix = normalized[-1]
    amount = int(normalized[:-1])
    if suffix == "m":
        return timedelta(minutes=amount)
    if suffix == "h":
        return timedelta(hours=amount)
    raise ValueError(f"unsupported stock timeframe: {timeframe}")


def expected_bar_opens(
    start: datetime,
    end: datetime,
    timeframe: str,
    session_policy: str,
) -> pd.DatetimeIndex:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")

    if timeframe.lower() in {"1d", "daily"}:
        labels = [pd.Timestamp(day, tz="UTC") for day in trading_days(start_ts.date(), end_ts.date())]
        return pd.DatetimeIndex(labels)

    delta = timeframe_delta(timeframe)
    timezone = ZoneInfo(EXCHANGE_TIMEZONE)
    local_start = start_ts.tz_convert(timezone)
    local_end = end_ts.tz_convert(timezone)
    values: list[pd.Timestamp] = []
    for day in trading_days(local_start.date(), local_end.date()):
        if session_policy == RTH_SESSION_POLICY:
            open_time = _RTH_OPEN
            close_time = session_close(day)
        elif session_policy == EXTENDED_SESSION_POLICY:
            open_time = _EXTENDED_OPEN
            close_time = _EXTENDED_CLOSE
        else:
            raise ValueError(f"unknown session policy: {session_policy}")
        cursor = datetime.combine(day, open_time, tzinfo=timezone)
        close_dt = datetime.combine(day, close_time, tzinfo=timezone)
        while cursor < close_dt:
            stamp = pd.Timestamp(cursor).tz_convert("UTC")
            if start_ts <= stamp <= end_ts:
                values.append(stamp)
            cursor += delta
    return pd.DatetimeIndex(values)


def rth_mask(index: pd.DatetimeIndex) -> pd.Series:
    utc_index = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    local = utc_index.tz_convert(EXCHANGE_TIMEZONE)
    flags = []
    for stamp in local:
        day = stamp.date()
        flags.append(
            is_trading_day(day)
            and stamp.time().replace(tzinfo=None) >= _RTH_OPEN
            and stamp.time().replace(tzinfo=None) < session_close(day)
        )
    return pd.Series(flags, index=index, dtype=bool)


def bar_open_in_session(timestamp: datetime, session_policy: str) -> bool:
    """Return whether a bar-open timestamp belongs to the requested session.

    ``RAW_SESSION_POLICY`` deliberately preserves the historical replay
    behaviour and is intended only for controlled validity comparisons.
    """
    if session_policy == RAW_SESSION_POLICY:
        return True

    stamp = timestamp
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=ZoneInfo("UTC"))
    local = stamp.astimezone(ZoneInfo(EXCHANGE_TIMEZONE))
    day = local.date()
    if not is_trading_day(day):
        return False
    local_time = local.time().replace(tzinfo=None)
    if session_policy == RTH_SESSION_POLICY:
        return _RTH_OPEN <= local_time < session_close(day)
    if session_policy == EXTENDED_SESSION_POLICY:
        return _EXTENDED_OPEN <= local_time < _EXTENDED_CLOSE
    raise ValueError(f"unknown session policy: {session_policy}")


def session_dates(index: pd.DatetimeIndex, *, daily_labels: bool = False) -> list[date]:
    utc_index = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    if daily_labels:
        return [stamp.date() for stamp in utc_index]
    local = utc_index.tz_convert(EXCHANGE_TIMEZONE)
    return [stamp.date() for stamp in local]
