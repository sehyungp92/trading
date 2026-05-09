"""Session/timezone utilities shared by ETF swing strategies."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")


def is_in_session_window(
    timestamp: datetime,
    windows_et: tuple[tuple[int, int, int, int], ...],
) -> bool:
    if not windows_et:
        return True
    local = _to_et(timestamp)
    minute = local.hour * 60 + local.minute
    for sh, sm, eh, em in windows_et:
        start = sh * 60 + sm
        end = eh * 60 + em
        if start <= minute <= end:
            return True
    return False


def minutes_to_next_event(
    timestamp: datetime,
    event_times_et: list[tuple[int, int]],
) -> int:
    if not event_times_et:
        return 10**9
    local = _to_et(timestamp)
    candidates = []
    for hour, minute in event_times_et:
        event = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if event < local:
            event += timedelta(days=1)
        candidates.append(int((event - local).total_seconds() // 60))
    return min(candidates)


def compute_gap_pct(prev_close: float, current_open: float) -> float:
    if prev_close == 0:
        return 0.0
    return (current_open - prev_close) / prev_close


def get_trading_day_date(timestamp: datetime) -> date:
    local = _to_et(timestamp)
    session_open = local.replace(hour=9, minute=30, second=0, microsecond=0)
    if local < session_open:
        local = local - timedelta(days=1)
        while local.weekday() >= 5:
            local = local - timedelta(days=1)
    return local.date()


def _to_et(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=ZoneInfo("UTC"))
    return timestamp.astimezone(_ET)

