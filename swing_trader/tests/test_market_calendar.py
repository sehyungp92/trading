"""Unit tests for shared.market_calendar — holiday/half-day computation."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import pytest

from shared.market_calendar import (
    AssetClass,
    MarketCalendar,
    _easter_sunday,
    _equity_holidays,
    _cme_holidays,
    _good_friday,
    _half_days,
    _nth_weekday,
    _last_weekday,
    _observe,
)


@pytest.fixture
def cal():
    return MarketCalendar()


# ===========================================================================
# === EASTER / GOOD FRIDAY ===
# ===========================================================================


class TestEasterAndGoodFriday:
    """Verify Easter computation against known dates."""

    KNOWN_EASTERS = {
        2024: date(2024, 3, 31),
        2025: date(2025, 4, 20),
        2026: date(2026, 4, 5),
        2027: date(2027, 3, 28),
        2028: date(2028, 4, 16),
        2029: date(2029, 4, 1),
        2030: date(2030, 4, 21),
    }

    @pytest.mark.parametrize("year, expected", list(KNOWN_EASTERS.items()))
    def test_easter_known_dates(self, year, expected):
        assert _easter_sunday(year) == expected

    @pytest.mark.parametrize("year, easter", list(KNOWN_EASTERS.items()))
    def test_good_friday_is_easter_minus_2(self, year, easter):
        assert _good_friday(year) == easter - timedelta(days=2)


# ===========================================================================
# === OBSERVED RULE ===
# ===========================================================================


class TestObservedRule:
    """Saturday→Friday, Sunday→Monday, weekday unchanged."""

    def test_saturday_to_friday(self):
        # 2026-07-04 is Saturday
        assert _observe(date(2026, 7, 4)) == date(2026, 7, 3)

    def test_sunday_to_monday(self):
        # 2027-07-04 is Sunday
        assert _observe(date(2027, 7, 4)) == date(2027, 7, 5)

    def test_weekday_unchanged(self):
        # 2025-07-04 is Friday
        assert _observe(date(2025, 7, 4)) == date(2025, 7, 4)


# ===========================================================================
# === FLOATING HOLIDAYS ===
# ===========================================================================


class TestFloatingHolidays:
    """Verify nth_weekday and last_weekday helpers."""

    def test_mlk_day_2026(self):
        """MLK Day 2026 = 3rd Monday in Jan = Jan 19."""
        assert _nth_weekday(2026, 1, 0, 3) == date(2026, 1, 19)

    def test_thanksgiving_2026(self):
        """Thanksgiving 2026 = 4th Thursday in Nov = Nov 26."""
        assert _nth_weekday(2026, 11, 3, 4) == date(2026, 11, 26)

    def test_memorial_day_2026(self):
        """Memorial Day 2026 = last Monday in May = May 25."""
        assert _last_weekday(2026, 5, 0) == date(2026, 5, 25)

    def test_labor_day_2026(self):
        """Labor Day 2026 = 1st Monday in Sep = Sep 7."""
        assert _nth_weekday(2026, 9, 0, 1) == date(2026, 9, 7)


# ===========================================================================
# === HOLIDAY COUNT ===
# ===========================================================================


class TestHolidayCount:
    """Verify correct number of holidays per year."""

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027, 2028])
    def test_equity_holiday_count(self, year):
        """10 equity holidays per year (since 2022 when Juneteenth added)."""
        assert len(_equity_holidays(year)) == 10

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2027, 2028])
    def test_cme_holiday_count(self, year):
        """7 CME holidays per year."""
        assert len(_cme_holidays(year)) == 7


# ===========================================================================
# === HALF DAYS ===
# ===========================================================================


class TestHalfDays:
    """Verify half-day computation."""

    def test_black_friday_always_half(self, cal):
        """Day after Thanksgiving is always a half day."""
        for year in [2024, 2025, 2026, 2027]:
            thanksgiving = _nth_weekday(year, 11, 3, 4)
            black_friday = thanksgiving + timedelta(days=1)
            assert cal.is_half_day(black_friday)

    def test_weekend_not_half_day(self, cal):
        """Weekends should never be half days."""
        # 2026-01-03 is Saturday
        assert not cal.is_half_day(date(2026, 1, 3))

    def test_normal_day_not_half(self, cal):
        """Regular trading day is not a half day."""
        assert not cal.is_half_day(date(2026, 3, 10))  # regular Monday

    def test_half_day_count(self):
        """Most years have 3 half days (some edge cases may differ)."""
        halves = _half_days(2026)
        assert len(halves) >= 2  # at least Black Friday + Christmas Eve
        assert len(halves) <= 3


# ===========================================================================
# === IS_ENTRY_BLOCKED ===
# ===========================================================================


class TestIsEntryBlocked:
    """Tests for MarketCalendar.is_entry_blocked()."""

    def test_holiday_blocked(self, cal):
        """Entries blocked on holidays."""
        # 2026 Christmas observed = Fri Dec 25
        xmas = _observe(date(2026, 12, 25))
        dt = datetime(xmas.year, xmas.month, xmas.day, 14, 0, tzinfo=timezone.utc)
        result = cal.is_entry_blocked(dt)
        assert result is not None
        assert "holiday" in result.lower()

    def test_half_day_noon_blocked(self, cal):
        """Entries blocked on half day after noon ET."""
        # Black Friday 2026 = Nov 27
        thanksgiving = _nth_weekday(2026, 11, 3, 4)
        black_friday = thanksgiving + timedelta(days=1)
        # 12:30 ET = 17:30 UTC
        dt = datetime(black_friday.year, black_friday.month, black_friday.day,
                      17, 30, tzinfo=timezone.utc)
        result = cal.is_entry_blocked(dt)
        assert result is not None
        assert "half-day" in result.lower()

    def test_half_day_morning_allowed(self, cal):
        """Entries allowed on half day before noon ET."""
        thanksgiving = _nth_weekday(2026, 11, 3, 4)
        black_friday = thanksgiving + timedelta(days=1)
        # 10:00 ET = 15:00 UTC
        dt = datetime(black_friday.year, black_friday.month, black_friday.day,
                      15, 0, tzinfo=timezone.utc)
        result = cal.is_entry_blocked(dt)
        assert result is None

    def test_normal_day_allowed(self, cal):
        """Entries allowed on normal trading day."""
        # 2026-03-10 is a regular Monday
        dt = datetime(2026, 3, 10, 15, 0, tzinfo=timezone.utc)
        result = cal.is_entry_blocked(dt)
        assert result is None

    def test_weekend_blocked(self, cal):
        """Entries blocked on weekends."""
        # 2026-03-07 is Saturday
        dt = datetime(2026, 3, 7, 15, 0, tzinfo=timezone.utc)
        result = cal.is_entry_blocked(dt)
        assert result is not None


# ===========================================================================
# === IS_TRADING_DAY / NEXT_TRADING_DAY ===
# ===========================================================================


class TestTradingDay:
    """Tests for is_trading_day() and next_trading_day()."""

    def test_regular_weekday(self, cal):
        assert cal.is_trading_day(date(2026, 3, 10))

    def test_weekend_not_trading(self, cal):
        assert not cal.is_trading_day(date(2026, 3, 7))   # Saturday
        assert not cal.is_trading_day(date(2026, 3, 8))   # Sunday

    def test_holiday_not_trading(self, cal):
        xmas = _observe(date(2026, 12, 25))
        assert not cal.is_trading_day(xmas)

    def test_next_trading_day_skips_weekend(self, cal):
        """Friday -> next Monday."""
        friday = date(2026, 3, 6)
        assert cal.next_trading_day(friday) == date(2026, 3, 9)

    def test_next_trading_day_skips_long_weekend(self, cal):
        """Thursday before Good Friday -> Monday (skips Fri+Sat+Sun)."""
        gf = _good_friday(2026)
        thursday = gf - timedelta(days=1)
        next_td = cal.next_trading_day(thursday)
        assert next_td.weekday() == 0  # Monday
        assert next_td > gf


# ===========================================================================
# === CME VS EQUITY ===
# ===========================================================================


class TestCMEvsEquity:
    """Verify CME and equity calendars differ on specific holidays."""

    def test_mlk_day_equity_closed_cme_open(self, cal):
        """MLK Day: equity closed, CME open."""
        mlk = _nth_weekday(2026, 1, 0, 3)
        assert cal.is_market_holiday(mlk, AssetClass.EQUITY)
        assert not cal.is_market_holiday(mlk, AssetClass.CME_FUTURES)

    def test_presidents_day_equity_closed_cme_open(self, cal):
        """Presidents' Day: equity closed, CME open."""
        pres = _nth_weekday(2026, 2, 0, 3)
        assert cal.is_market_holiday(pres, AssetClass.EQUITY)
        assert not cal.is_market_holiday(pres, AssetClass.CME_FUTURES)

    def test_juneteenth_equity_closed_cme_open(self, cal):
        """Juneteenth: equity closed, CME open."""
        juneteenth = _observe(date(2026, 6, 19))
        assert cal.is_market_holiday(juneteenth, AssetClass.EQUITY)
        assert not cal.is_market_holiday(juneteenth, AssetClass.CME_FUTURES)

    def test_good_friday_both_closed(self, cal):
        """Good Friday: both equity and CME closed."""
        gf = _good_friday(2026)
        assert cal.is_market_holiday(gf, AssetClass.EQUITY)
        assert cal.is_market_holiday(gf, AssetClass.CME_FUTURES)

    def test_cme_entry_blocked_on_cme_holiday(self, cal):
        """CME entries blocked on CME holidays."""
        gf = _good_friday(2026)
        dt = datetime(gf.year, gf.month, gf.day, 15, 0, tzinfo=timezone.utc)
        result = cal.is_entry_blocked(dt, AssetClass.CME_FUTURES)
        assert result is not None

    def test_cme_entry_allowed_on_equity_only_holiday(self, cal):
        """CME entries allowed on equity-only holidays (e.g., MLK)."""
        mlk = _nth_weekday(2026, 1, 0, 3)
        dt = datetime(mlk.year, mlk.month, mlk.day, 15, 0, tzinfo=timezone.utc)
        result = cal.is_entry_blocked(dt, AssetClass.CME_FUTURES)
        assert result is None


# ===========================================================================
# === YEAR CACHING ===
# ===========================================================================


class TestYearCaching:
    """Verify that cached holiday sets are consistent across calls."""

    def test_same_set_returned(self):
        a = _equity_holidays(2026)
        b = _equity_holidays(2026)
        assert a is b

    def test_different_years_different_sets(self):
        a = _equity_holidays(2025)
        b = _equity_holidays(2026)
        assert a != b


# ===========================================================================
# === MARKET CLOSE TIME ===
# ===========================================================================


class TestMarketCloseTime:
    """Tests for market_close_time_et()."""

    def test_normal_day_closes_at_16(self, cal):
        assert cal.market_close_time_et(date(2026, 3, 10)) == time(16, 0)

    def test_half_day_closes_at_13(self, cal):
        thanksgiving = _nth_weekday(2026, 11, 3, 4)
        black_friday = thanksgiving + timedelta(days=1)
        assert cal.market_close_time_et(black_friday) == time(13, 0)
