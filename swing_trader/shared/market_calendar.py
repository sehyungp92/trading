"""Market calendar — holiday and half-day awareness for US equity and CME futures.

Pure-stdlib implementation, no external dependencies.
Year-cached holiday computation for fast repeated lookups.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from functools import lru_cache


class AssetClass(Enum):
    EQUITY = "equity"          # NYSE / NASDAQ
    CME_FUTURES = "cme_futures"  # CME / COMEX / NYMEX


# ---------------------------------------------------------------------------
# Internal date helpers
# ---------------------------------------------------------------------------

def _observe(d: date) -> date:
    """Apply NYSE observed-holiday rule: Sat→Friday, Sun→Monday."""
    if d.weekday() == 5:   # Saturday
        return d - timedelta(days=1)
    if d.weekday() == 6:   # Sunday
        return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the *n*-th occurrence of *weekday* (0=Mon) in *month*/*year*."""
    first = date(year, month, 1)
    # Days until first target weekday
    delta = (weekday - first.weekday()) % 7
    first_occ = first + timedelta(days=delta)
    return first_occ + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of *weekday* (0=Mon) in *month*/*year*."""
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    delta = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=delta)


def _easter_sunday(year: int) -> date:
    """Meeus/Jones/Butcher algorithm for Easter Sunday."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _good_friday(year: int) -> date:
    return _easter_sunday(year) - timedelta(days=2)


# ---------------------------------------------------------------------------
# Holiday sets (per year, cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _equity_holidays(year: int) -> frozenset[date]:
    """NYSE/NASDAQ holidays for *year* (10/year)."""
    holidays: list[date] = []

    # New Year's Day
    holidays.append(_observe(date(year, 1, 1)))

    # MLK Day — 3rd Monday in January
    holidays.append(_nth_weekday(year, 1, 0, 3))

    # Presidents' Day — 3rd Monday in February
    holidays.append(_nth_weekday(year, 2, 0, 3))

    # Good Friday
    holidays.append(_good_friday(year))

    # Memorial Day — last Monday in May
    holidays.append(_last_weekday(year, 5, 0))

    # Juneteenth (since 2022)
    if year >= 2022:
        holidays.append(_observe(date(year, 6, 19)))

    # Independence Day
    holidays.append(_observe(date(year, 7, 4)))

    # Labor Day — 1st Monday in September
    holidays.append(_nth_weekday(year, 9, 0, 1))

    # Thanksgiving — 4th Thursday in November
    holidays.append(_nth_weekday(year, 11, 3, 4))

    # Christmas
    holidays.append(_observe(date(year, 12, 25)))

    return frozenset(holidays)


@lru_cache(maxsize=16)
def _cme_holidays(year: int) -> frozenset[date]:
    """CME/COMEX/NYMEX holidays for *year* (7/year).

    CME stays open on MLK Day, Presidents' Day, and Juneteenth.
    """
    holidays: list[date] = []

    holidays.append(_observe(date(year, 1, 1)))      # New Year's
    holidays.append(_good_friday(year))               # Good Friday
    holidays.append(_last_weekday(year, 5, 0))        # Memorial Day
    holidays.append(_observe(date(year, 7, 4)))       # Independence Day
    holidays.append(_nth_weekday(year, 9, 0, 1))      # Labor Day
    holidays.append(_nth_weekday(year, 11, 3, 4))     # Thanksgiving
    holidays.append(_observe(date(year, 12, 25)))     # Christmas

    return frozenset(holidays)


@lru_cache(maxsize=16)
def _half_days(year: int) -> frozenset[date]:
    """NYSE early-close days (1:00 PM ET).

    Three per year:
    - Day before Independence Day (if weekday)
    - Day after Thanksgiving (Black Friday)
    - Christmas Eve (if weekday)
    """
    days: list[date] = []

    # Day before Independence Day
    july4_observed = _observe(date(year, 7, 4))
    before_july4 = july4_observed - timedelta(days=1)
    if before_july4.weekday() < 5:
        days.append(before_july4)

    # Day after Thanksgiving (always a Friday)
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    black_friday = thanksgiving + timedelta(days=1)
    days.append(black_friday)

    # Christmas Eve
    xmas_observed = _observe(date(year, 12, 25))
    xmas_eve = xmas_observed - timedelta(days=1)
    if xmas_eve.weekday() < 5:
        days.append(xmas_eve)

    return frozenset(days)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MarketCalendar:
    """Holiday and half-day calendar for US equity and CME futures markets."""

    def is_market_holiday(self, d: date, asset_class: AssetClass = AssetClass.EQUITY) -> bool:
        """True if *d* is a full market closure (includes weekends)."""
        if d.weekday() >= 5:
            return True
        holidays = (
            _equity_holidays(d.year)
            if asset_class == AssetClass.EQUITY
            else _cme_holidays(d.year)
        )
        return d in holidays

    def is_half_day(self, d: date, asset_class: AssetClass = AssetClass.EQUITY) -> bool:
        """True if *d* is an early-close session (1:00 PM ET)."""
        if d.weekday() >= 5:
            return False
        return d in _half_days(d.year)

    def is_trading_day(self, d: date, asset_class: AssetClass = AssetClass.EQUITY) -> bool:
        """True if *d* is a regular or half-day session (not weekend, not holiday)."""
        if d.weekday() >= 5:
            return False
        holidays = (
            _equity_holidays(d.year)
            if asset_class == AssetClass.EQUITY
            else _cme_holidays(d.year)
        )
        return d not in holidays

    def next_trading_day(self, d: date, asset_class: AssetClass = AssetClass.EQUITY) -> date:
        """Return the next open session strictly after *d*."""
        candidate = d + timedelta(days=1)
        while not self.is_trading_day(candidate, asset_class):
            candidate += timedelta(days=1)
        return candidate

    def market_close_time_et(self, d: date, asset_class: AssetClass = AssetClass.EQUITY) -> time:
        """Return 13:00 for half days, 16:00 for normal sessions."""
        if self.is_half_day(d, asset_class):
            return time(13, 0)
        return time(16, 0)

    def is_entry_blocked(
        self, now_utc: datetime, asset_class: AssetClass = AssetClass.EQUITY,
    ) -> str | None:
        """Return denial reason if entries should be blocked, else None.

        Blocks on:
        - Market holidays (full closure)
        - Half-day sessions after 12:00 PM ET (no new entries in last hour)
        """
        from zoneinfo import ZoneInfo
        now_et = now_utc.astimezone(ZoneInfo("America/New_York"))
        today = now_et.date()

        if self.is_market_holiday(today, asset_class):
            return f"Market holiday: {today}"

        if self.is_half_day(today, asset_class):
            if now_et.hour >= 12:
                return f"Half-day session — no new entries after 12:00 ET ({today})"

        return None
