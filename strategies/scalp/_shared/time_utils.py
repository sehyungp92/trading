from __future__ import annotations

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = timezone.utc


def ensure_aware(value: datetime, default_tz: ZoneInfo | timezone = UTC) -> datetime:
    return value.replace(tzinfo=default_tz) if value.tzinfo is None else value


def to_et(value: datetime) -> datetime:
    return ensure_aware(value).astimezone(ET)


def et_time(value: datetime) -> time:
    return to_et(value).time()


def session_date(value: datetime) -> date:
    return to_et(value).date()


def combine_et(day: date, hhmm: str) -> datetime:
    hour, minute, *rest = hhmm.split(":")
    second = int(rest[0]) if rest else 0
    return datetime(day.year, day.month, day.day, int(hour), int(minute), second, tzinfo=ET)


def in_time_window(
    value: datetime,
    start: time,
    end: time,
    *,
    inclusive_end: bool = False,
) -> bool:
    current = et_time(value)
    if inclusive_end:
        return start <= current <= end
    return start <= current < end


def minutes_since_midnight_et(value: datetime) -> int:
    current = et_time(value)
    return current.hour * 60 + current.minute

