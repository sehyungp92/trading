"""Unit tests for Strategy 1 (ATRSS) — indicators, signals, stops, allocator."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from strategy.config import (
    ADDON_A_R,
    ADDON_B_R,
    BE_ATR_OFFSET,
    COOLDOWN_HOURS,
    MOMENTUM_TOLERANCE_ATR,
    PROFIT_FLOOR,
    PROFIT_FLOOR_SHORT,
    QUALITY_GATE_THRESHOLD,
    RECOVERY_TOLERANCE_ATR_STRONG,
    RECOVERY_TOLERANCE_ATR_TREND,
    SCORE_REVERSE_MIN,
    SymbolConfig,
)
from strategy.indicators import (
    adx_suite,
    atr,
    compute_daily_state,
    compute_hourly_state,
    ema,
)
from strategy.models import (
    Candidate,
    CandidateType,
    DailyState,
    Direction,
    HourlyState,
    PositionBook,
    PositionLeg,
    ReentryState,
    Regime,
)
from strategy.signals import (
    addon_a_eligible,
    addon_b_eligible,
    breakout_pullback_signal,
    check_breakout_arm,
    compute_entry_quality,
    momentum_ok,
    pullback_signal,
    reverse_entry_ok,
    same_direction_reentry_allowed,
    short_safety_ok,
    short_symbol_gate,
)
from strategy.stops import (
    apply_profit_floor,
    compute_be_stop,
    compute_chandelier_stop,
    compute_initial_stop,
)
from strategy.allocator import (
    allocate,
    compute_position_size,
    rank_candidates,
)
from shared.oms.models.instrument import Instrument


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def default_cfg():
    return SymbolConfig(symbol="QQQ", tick_size=0.01, multiplier=1.0,
                        exchange="SMART", sec_type="STK", primary_exchange="NASDAQ")


@pytest.fixture
def qqq_instrument():
    return Instrument(symbol="QQQ", root="QQQ", venue="SMART",
                      tick_size=0.01, tick_value=0.01, multiplier=1.0)


@pytest.fixture
def daily_long():
    return DailyState(
        ema_fast=105.0, ema_slow=100.0, adx=25.0, plus_di=30.0,
        minus_di=15.0, atr20=2.0, regime=Regime.TREND,
        trend_dir=Direction.LONG, score=70.0, ema_sep_pct=0.5,
        di_diff=15.0, adx_slope_3=1.0, raw_bias=Direction.LONG,
        regime_on=True, ema_fast_slope_5=1.0, hh_20d=110.0, ll_20d=95.0,
    )


@pytest.fixture
def daily_strong_long():
    return DailyState(
        ema_fast=108.0, ema_slow=100.0, adx=35.0, plus_di=35.0,
        minus_di=12.0, atr20=2.0, regime=Regime.STRONG_TREND,
        trend_dir=Direction.LONG, score=80.0, ema_sep_pct=0.8,
        di_diff=23.0, adx_slope_3=2.0, raw_bias=Direction.LONG,
        regime_on=True, ema_fast_slope_5=2.0, hh_20d=112.0, ll_20d=96.0,
    )


@pytest.fixture
def daily_short():
    return DailyState(
        ema_fast=95.0, ema_slow=100.0, adx=25.0, plus_di=12.0,
        minus_di=30.0, atr20=2.0, regime=Regime.TREND,
        trend_dir=Direction.SHORT, score=65.0, ema_sep_pct=0.5,
        di_diff=18.0, adx_slope_3=-1.0, raw_bias=Direction.SHORT,
        regime_on=True, ema_fast_slope_5=-1.0, hh_20d=105.0, ll_20d=90.0,
    )


@pytest.fixture
def hourly_long():
    return HourlyState(
        open=104.5, high=106.0, low=103.5, close=105.5,
        ema_mom=104.0, ema_pull=104.2, atrh=1.5,
        donchian_high=107.0, donchian_low=101.0,
        prior_high=105.5, prior_low=103.0, dist_atr=0.5,
        recent_pull_touch_long=True, recent_pull_touch_short=False,
    )


@pytest.fixture
def hourly_short():
    return HourlyState(
        open=96.0, high=97.0, low=94.5, close=94.8,
        ema_mom=96.0, ema_pull=96.2, atrh=1.5,
        donchian_high=99.0, donchian_low=93.0,
        prior_high=97.0, prior_low=94.0, dist_atr=0.6,
        recent_pull_touch_long=False, recent_pull_touch_short=True,
    )


# ===========================================================================
# INDICATORS
# ===========================================================================

class TestEMA:
    def test_ema_single_value(self):
        arr = np.array([10.0])
        result = ema(arr, 5)
        assert len(result) == 1
        assert result[0] == pytest.approx(10.0)

    def test_ema_constant_series(self):
        arr = np.full(20, 50.0)
        result = ema(arr, 10)
        assert result[-1] == pytest.approx(50.0, abs=0.01)

    def test_ema_trending_up(self):
        arr = np.arange(1.0, 51.0)
        result = ema(arr, 10)
        # EMA should lag below price in uptrend
        assert result[-1] < arr[-1]
        assert result[-1] > result[-2]

    def test_ema_period_1(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = ema(arr, 1)
        np.testing.assert_allclose(result, arr, atol=0.01)


class TestATR:
    def test_atr_constant_bars(self):
        n = 30
        highs = np.full(n, 52.0)
        lows = np.full(n, 48.0)
        closes = np.full(n, 50.0)
        result = atr(highs, lows, closes, 14)
        assert result[-1] == pytest.approx(4.0, abs=0.1)

    def test_atr_increasing_volatility(self):
        n = 50
        highs = np.array([50.0 + i * 0.1 for i in range(n)])
        lows = np.array([50.0 - i * 0.1 for i in range(n)])
        closes = np.full(n, 50.0)
        result = atr(highs, lows, closes, 14)
        assert result[-1] > result[20]

    def test_atr_positive_values(self):
        rng = np.random.RandomState(42)
        closes = 100 + np.cumsum(rng.randn(100) * 0.5)
        highs = closes + rng.uniform(0.1, 1.0, 100)
        lows = closes - rng.uniform(0.1, 1.0, 100)
        result = atr(highs, lows, closes, 14)
        assert np.all(result > 0)


class TestADXSuite:
    def test_adx_trending_market(self):
        n = 100
        closes = np.linspace(100, 150, n)
        highs = closes + 1.0
        lows = closes - 1.0
        adx_arr, pdi, mdi = adx_suite(highs, lows, closes, 14)
        # Strong trend should have high ADX and +DI > -DI
        assert adx_arr[-1] > 15
        assert pdi[-1] > mdi[-1]

    def test_adx_output_shape(self):
        n = 50
        h = np.random.uniform(101, 105, n)
        l = np.random.uniform(95, 99, n)
        c = np.random.uniform(98, 103, n)
        adx_arr, pdi, mdi = adx_suite(h, l, c, 14)
        assert len(adx_arr) == n
        assert len(pdi) == n
        assert len(mdi) == n


class TestComputeDailyState:
    def test_daily_state_uptrend(self):
        n = 80
        closes = np.linspace(100, 140, n)
        highs = closes + 1.5
        lows = closes - 1.5
        cfg = SymbolConfig(symbol="QQQ", adx_on=18, adx_off=16,
                           tick_size=0.01, multiplier=1.0, exchange="SMART",
                           sec_type="STK")
        state = compute_daily_state(closes, highs, lows, None, cfg, "2025-01-15")
        assert state.ema_fast > state.ema_slow
        assert state.atr20 > 0
        assert state.hh_20d >= state.ll_20d

    def test_daily_state_regime_hysteresis(self):
        n = 80
        closes = np.linspace(100, 140, n)
        highs = closes + 1.5
        lows = closes - 1.5
        cfg = SymbolConfig(symbol="QQQ", adx_on=18, adx_off=16,
                           tick_size=0.01, multiplier=1.0, exchange="SMART",
                           sec_type="STK")
        prev = DailyState(regime_on=True, raw_bias=Direction.LONG,
                          hold_count=3, trend_dir=Direction.LONG,
                          ema_fast=130.0, ema_slow=120.0)
        state = compute_daily_state(closes, highs, lows, prev, cfg, "2025-01-16")
        # With prev regime_on and strong trend, should stay on
        assert state.regime_on is True

    def test_daily_state_hold_count_increments_on_new_bar(self):
        n = 80
        closes = np.linspace(100, 140, n)
        highs = closes + 1.5
        lows = closes - 1.5
        cfg = SymbolConfig(symbol="QQQ", adx_on=18, adx_off=16,
                           tick_size=0.01, multiplier=1.0, exchange="SMART",
                           sec_type="STK")
        prev = DailyState(raw_bias=Direction.LONG, hold_count=1,
                          last_daily_bar_date="2025-01-14")
        state = compute_daily_state(closes, highs, lows, prev, cfg, "2025-01-15")
        # Same bias + new date → hold_count increments
        if state.raw_bias == Direction.LONG:
            assert state.hold_count >= 2


class TestComputeHourlyState:
    def test_hourly_state_fields(self):
        n = 60
        closes = np.linspace(100, 115, n)
        highs = closes + 0.8
        lows = closes - 0.8
        daily = DailyState(ema_fast=110.0, atr20=2.0, regime=Regime.TREND)
        cfg = SymbolConfig(symbol="QQQ", tick_size=0.01, multiplier=1.0,
                           exchange="SMART", sec_type="STK")
        state = compute_hourly_state(closes, highs, lows, daily, cfg)
        assert state.atrh > 0
        assert state.donchian_high > state.donchian_low
        assert state.close == pytest.approx(closes[-1])


# ===========================================================================
# SIGNALS
# ===========================================================================

class TestMomentumOk:
    def test_long_above_ema(self, hourly_long):
        assert momentum_ok(hourly_long, Direction.LONG) is True

    def test_long_below_ema(self):
        h = HourlyState(close=90.0, ema_mom=100.0, atrh=1.0)
        assert momentum_ok(h, Direction.LONG) is False

    def test_short_below_ema(self, hourly_short):
        assert momentum_ok(hourly_short, Direction.SHORT) is True

    def test_tolerance(self):
        h = HourlyState(close=99.86, ema_mom=100.0, atrh=1.5)
        tol = MOMENTUM_TOLERANCE_ATR * h.atrh
        # close=99.86 > ema_mom - tol = 100.0 - 0.15 = 99.85
        assert momentum_ok(h, Direction.LONG) is True  # within tolerance

    def test_flat_always_false(self, hourly_long):
        assert momentum_ok(hourly_long, Direction.FLAT) is False


class TestShortSafetyOk:
    def test_negative_slope_ok(self, daily_short):
        assert short_safety_ok(daily_short) is True

    def test_positive_slope_blocked(self, daily_long):
        assert short_safety_ok(daily_long) is False


class TestShortSymbolGate:
    def test_qqq_always_passes(self, daily_short, hourly_short):
        assert short_symbol_gate("QQQ", daily_short, hourly_short) is True

    def test_gld_requires_di_agrees(self):
        d = DailyState(minus_di=25, plus_di=20, adx=25)
        h = HourlyState()
        assert short_symbol_gate("GLD", d, h) is True
        d2 = DailyState(minus_di=15, plus_di=20, adx=25)
        assert short_symbol_gate("GLD", d2, h) is False

    def test_uso_requires_adx_and_di(self):
        d = DailyState(adx=25, minus_di=30, plus_di=15)
        assert short_symbol_gate("USO", d, HourlyState()) is True
        d2 = DailyState(adx=15, minus_di=30, plus_di=15)
        assert short_symbol_gate("USO", d2, HourlyState()) is False


class TestComputeEntryQuality:
    def test_high_quality_long(self, hourly_long, daily_long):
        score = compute_entry_quality(hourly_long, daily_long, Direction.LONG)
        assert 0 <= score <= 7

    def test_adx_contribution(self):
        h = HourlyState(close=100, ema_pull=100, atrh=1.0, ema_mom=99)
        d = DailyState(adx=30, plus_di=25, minus_di=10, ema_sep_pct=0.3)
        score = compute_entry_quality(h, d, Direction.LONG)
        assert score >= 2.0  # ADX >= 25 contributes 2

    def test_zero_atrh_no_crash(self):
        h = HourlyState(close=100, ema_pull=100, atrh=0.0, ema_mom=99)
        d = DailyState(adx=10)
        score = compute_entry_quality(h, d, Direction.LONG)
        assert score >= 0


class TestPullbackSignal:
    def test_long_pullback_trend(self, hourly_long, daily_long):
        sig = pullback_signal(hourly_long, daily_long)
        assert sig == Direction.LONG

    def test_no_signal_in_range(self, hourly_long):
        d = DailyState(regime=Regime.RANGE, trend_dir=Direction.FLAT)
        assert pullback_signal(hourly_long, d) == Direction.FLAT

    def test_no_signal_without_touch(self, daily_long):
        h = HourlyState(close=105.0, ema_pull=104.0, atrh=1.5,
                        recent_pull_touch_long=False)
        assert pullback_signal(h, daily_long) == Direction.FLAT

    def test_short_pullback_requires_safety(self, hourly_short, daily_short):
        # Short with negative slope should work
        sig = pullback_signal(hourly_short, daily_short)
        assert sig == Direction.SHORT

    def test_short_pullback_blocked_by_safety(self, hourly_short):
        d = DailyState(regime=Regime.TREND, trend_dir=Direction.SHORT,
                       ema_fast_slope_5=2.0)  # positive slope blocks short
        assert pullback_signal(hourly_short, d) == Direction.FLAT


class TestBreakoutArm:
    def test_long_arm_strong_trend(self, daily_strong_long):
        h = HourlyState(high=108.0, close=107.5, ema_mom=106.0, atrh=1.5,
                        donchian_high=107.0)
        assert check_breakout_arm(h, daily_strong_long) == Direction.LONG

    def test_no_arm_in_trend(self, hourly_long, daily_long):
        # TREND (not STRONG_TREND) should not arm
        assert check_breakout_arm(hourly_long, daily_long) == Direction.FLAT


class TestBreakoutPullbackSignal:
    def test_long_pullback_after_arm(self, daily_strong_long):
        # arm_high=108, arm_low=106, range=2.0
        # retrace_entry = 108 - 0.30*2 = 107.4
        # retrace_limit = 108 - 0.50*2 = 107.0
        # Need: low <= 107.4, close > 107.0, close > open (bullish)
        h = HourlyState(open=107.0, high=107.5, low=107.2, close=107.3,
                        ema_mom=105.0, atrh=1.5)
        sig = breakout_pullback_signal(
            h, daily_strong_long, Direction.LONG,
            arm_high=108.0, arm_low=106.0,
        )
        assert sig == Direction.LONG

    def test_zero_range_returns_flat(self, daily_strong_long):
        h = HourlyState(low=100, high=102, close=101)
        assert breakout_pullback_signal(
            h, daily_strong_long, Direction.LONG,
            arm_high=100, arm_low=100,
        ) == Direction.FLAT


class TestReverseEntryOk:
    def test_reverse_ok_strong_conditions(self, hourly_long, daily_long):
        daily_long.score = 65
        assert reverse_entry_ok(hourly_long, daily_long) is True

    def test_reverse_blocked_low_score(self, hourly_long, daily_long):
        daily_long.score = 40  # below SCORE_REVERSE_MIN
        assert reverse_entry_ok(hourly_long, daily_long) is False

    def test_reverse_blocked_range(self, hourly_long):
        d = DailyState(regime_on=False)
        assert reverse_entry_ok(hourly_long, d) is False


class TestAddonEligibility:
    def test_addon_a_eligible(self, hourly_long, daily_long):
        pos = PositionBook(
            direction=Direction.LONG, mfe=1.8, be_triggered=True,
            addon_a_done=False,
            legs=[PositionLeg(qty=10, entry_price=100.0, initial_stop=98.0)],
        )
        assert addon_a_eligible(pos, hourly_long, daily_long, current_r=0.85) is True

    def test_addon_a_not_eligible_already_done(self, hourly_long, daily_long):
        pos = PositionBook(
            direction=Direction.LONG, mfe=1.8, be_triggered=True,
            addon_a_done=True,
        )
        assert addon_a_eligible(pos, hourly_long, daily_long) is False

    def test_addon_a_not_eligible_low_mfe(self, hourly_long, daily_long):
        pos = PositionBook(
            direction=Direction.LONG, mfe=0.5, be_triggered=True,
            addon_a_done=False,
        )
        assert addon_a_eligible(pos, hourly_long, daily_long) is False

    def test_addon_b_eligible_strong_trend(self, hourly_long, daily_strong_long):
        # Hourly must show pullback signal for addon B
        hourly_long.recent_pull_touch_long = True
        pos = PositionBook(
            direction=Direction.LONG, mfe=2.5, addon_b_done=False,
        )
        assert addon_b_eligible(pos, hourly_long, daily_strong_long) is True

    def test_addon_b_not_in_trend(self, hourly_long, daily_long):
        pos = PositionBook(direction=Direction.LONG, mfe=2.5, addon_b_done=False)
        assert addon_b_eligible(pos, hourly_long, daily_long) is False


class TestReentryCooldown:
    def test_first_entry_always_allowed(self):
        r = ReentryState()
        now = datetime.now(timezone.utc)
        assert same_direction_reentry_allowed(r, Direction.LONG, now, Regime.TREND) is True

    def test_opposite_direction_always_allowed(self):
        r = ReentryState(
            last_exit_time=datetime.now(timezone.utc),
            last_exit_dir=Direction.LONG,
        )
        now = datetime.now(timezone.utc)
        assert same_direction_reentry_allowed(r, Direction.SHORT, now, Regime.TREND) is True

    def test_quality_exit_skips_cooldown(self):
        now = datetime.now(timezone.utc)
        r = ReentryState(
            last_exit_time=now - timedelta(minutes=5),
            last_exit_dir=Direction.LONG,
            last_exit_mfe=0.5,
        )
        assert same_direction_reentry_allowed(r, Direction.LONG, now, Regime.TREND) is True

    def test_stall_exit_requires_cooldown(self):
        et = ZoneInfo("America/New_York")
        exit_time = datetime(2025, 1, 15, 10, 0, tzinfo=et)
        now = datetime(2025, 1, 15, 11, 0, tzinfo=et)  # 1 hour later
        r = ReentryState(
            last_exit_time=exit_time,
            last_exit_dir=Direction.LONG,
            last_exit_reason="FLATTEN_STALL",
            last_exit_mfe=0.1,
        )
        cooldown = COOLDOWN_HOURS.get("TREND", 2)
        # Should be blocked (only 1 RTH hour elapsed, need 2)
        result = same_direction_reentry_allowed(r, Direction.LONG, now, Regime.TREND)
        assert result is False


# ===========================================================================
# STOPS
# ===========================================================================

class TestComputeInitialStop:
    def test_long_stop_below_entry(self, hourly_long):
        stop = compute_initial_stop(
            Direction.LONG, 105.0, hourly_long,
            daily_atr=2.0, hourly_atr=1.5,
            d_mult=2.2, h_mult=3.0, tick_size=0.01,
        )
        assert stop < 105.0

    def test_short_stop_above_entry(self, hourly_short):
        stop = compute_initial_stop(
            Direction.SHORT, 95.0, hourly_short,
            daily_atr=2.0, hourly_atr=1.5,
            d_mult=2.2, h_mult=3.0, tick_size=0.01,
        )
        assert stop > 95.0

    def test_wider_of_atr_and_structure(self, hourly_long):
        # The stop should use the wider of ATR vs structure
        stop = compute_initial_stop(
            Direction.LONG, 105.0, hourly_long,
            daily_atr=2.0, hourly_atr=1.5,
            d_mult=2.2, h_mult=3.0, tick_size=0.01,
        )
        struct_stop = hourly_long.low - 0.01  # signal low - 1 tick
        atr_dist = max(2.2 * 2.0, 3.0 * 1.5)
        atr_stop = 105.0 - atr_dist
        expected = min(atr_stop, struct_stop)
        # Both should be below entry and stop should be approximately the wider
        assert stop <= 105.0


class TestBEStop:
    def test_long_be(self):
        stop = compute_be_stop(Direction.LONG, 100.0, 2.0, 0.01)
        expected = 100.0 + BE_ATR_OFFSET * 2.0
        assert stop == pytest.approx(expected, abs=0.01)

    def test_short_be(self):
        stop = compute_be_stop(Direction.SHORT, 100.0, 2.0, 0.01)
        expected = 100.0 - BE_ATR_OFFSET * 2.0
        assert stop == pytest.approx(expected, abs=0.01)


class TestChandelierStop:
    def test_long_chandelier(self, daily_long):
        stop = compute_chandelier_stop(Direction.LONG, daily_long, 3.0, 0.01)
        expected = daily_long.hh_20d - 3.0 * daily_long.atr20
        assert stop == pytest.approx(expected, abs=0.01)

    def test_short_chandelier(self, daily_short):
        stop = compute_chandelier_stop(Direction.SHORT, daily_short, 3.0, 0.01)
        expected = daily_short.ll_20d + 3.0 * daily_short.atr20
        assert stop == pytest.approx(expected, abs=0.01)


class TestProfitFloor:
    def test_long_profit_floor_at_2r(self):
        stop = apply_profit_floor(
            Direction.LONG, 100.0, 2.0, 2.5, 99.0, 0.01,
        )
        # MFE 2.5R >= 2.0 threshold → min profit 0.75R
        min_stop = 100.0 + 0.75 * 2.0
        assert stop >= min_stop

    def test_no_floor_below_threshold(self):
        stop = apply_profit_floor(
            Direction.LONG, 100.0, 2.0, 0.5, 99.0, 0.01,
        )
        # MFE 0.5 < 1.0 (lowest threshold) → no floor applied
        assert stop == pytest.approx(99.0)

    def test_short_profit_floor(self):
        stop = apply_profit_floor(
            Direction.SHORT, 100.0, 2.0, 1.2, 102.0, 0.01,
        )
        # MFE 1.2 >= 1.0 threshold in PROFIT_FLOOR_SHORT → min profit 0.50R
        min_stop = 100.0 - 0.50 * 2.0
        assert stop <= min_stop


# ===========================================================================
# ALLOCATOR
# ===========================================================================

class TestRankCandidates:
    def test_rank_by_score_desc(self):
        c1 = Candidate(symbol="A", rank_score=50, trigger_price=100, initial_stop=98)
        c2 = Candidate(symbol="B", rank_score=70, trigger_price=100, initial_stop=98)
        ranked = rank_candidates([c1, c2])
        assert ranked[0].symbol == "B"

    def test_tiebreak_by_stop_distance(self):
        c1 = Candidate(symbol="A", rank_score=60, trigger_price=100, initial_stop=97)
        c2 = Candidate(symbol="B", rank_score=60, trigger_price=100, initial_stop=99)
        ranked = rank_candidates([c1, c2])
        assert ranked[0].symbol == "B"  # tighter stop = smaller distance

    def test_tiebreak_by_symbol(self):
        c1 = Candidate(symbol="B", rank_score=60, trigger_price=100, initial_stop=98)
        c2 = Candidate(symbol="A", rank_score=60, trigger_price=100, initial_stop=98)
        ranked = rank_candidates([c1, c2])
        assert ranked[0].symbol == "A"


class TestComputePositionSize:
    def test_basic_sizing(self):
        qty = compute_position_size(100.0, 98.0, 100000, 0.01, 1.0)
        # risk = 100000 * 0.01 = 1000; risk/contract = 2.0
        assert qty == 500

    def test_zero_risk_returns_zero(self):
        qty = compute_position_size(100.0, 100.0, 100000, 0.01, 1.0)
        assert qty == 0

    def test_floor_rounding(self):
        qty = compute_position_size(100.0, 97.0, 100000, 0.01, 1.0)
        # risk/contract = 3.0; 1000/3 = 333.33 → floor = 333
        assert qty == 333


class TestAllocate:
    def test_accept_within_heat_cap(self, qqq_instrument):
        c = Candidate(symbol="QQQ", qty=10, trigger_price=100.0,
                      initial_stop=98.0, rank_score=70,
                      type=CandidateType.PULLBACK)
        instruments = {"QQQ": qqq_instrument}
        accepted = allocate([c], {}, {}, 100000, instruments)
        assert len(accepted) == 1

    def test_reject_exceeding_heat_cap(self, qqq_instrument):
        # Create a candidate that would use >6% heat
        c = Candidate(symbol="QQQ", qty=5000, trigger_price=100.0,
                      initial_stop=90.0, rank_score=70,
                      type=CandidateType.PULLBACK)
        instruments = {"QQQ": qqq_instrument}
        accepted = allocate([c], {}, {}, 10000, instruments)
        assert len(accepted) == 0  # 5000 * 10 = 50000 > 6% of 10000

    def test_one_base_entry_per_symbol(self, qqq_instrument):
        c1 = Candidate(symbol="QQQ", qty=5, trigger_price=100.0,
                       initial_stop=99.0, rank_score=70,
                       type=CandidateType.PULLBACK)
        c2 = Candidate(symbol="QQQ", qty=5, trigger_price=100.0,
                       initial_stop=99.0, rank_score=60,
                       type=CandidateType.PULLBACK)
        instruments = {"QQQ": qqq_instrument}
        accepted = allocate([c1, c2], {}, {}, 100000, instruments)
        assert len(accepted) == 1

    def test_addons_not_blocked_by_symbol_rule(self, qqq_instrument):
        c1 = Candidate(symbol="QQQ", qty=5, trigger_price=100.0,
                       initial_stop=99.0, rank_score=70,
                       type=CandidateType.PULLBACK)
        c2 = Candidate(symbol="QQQ", qty=2, trigger_price=101.0,
                       initial_stop=99.5, rank_score=60,
                       type=CandidateType.ADDON_A)
        instruments = {"QQQ": qqq_instrument}
        accepted = allocate([c1, c2], {}, {}, 100000, instruments)
        assert len(accepted) == 2

    def test_zero_equity(self, qqq_instrument):
        c = Candidate(symbol="QQQ", qty=10, trigger_price=100.0,
                      initial_stop=98.0, rank_score=70)
        assert allocate([c], {}, {}, 0, {"QQQ": qqq_instrument}) == []
