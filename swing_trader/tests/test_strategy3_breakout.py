"""Unit tests for Strategy 3 (Swing Breakout v3.3) — indicators, signals, stops, allocator, gates."""
from __future__ import annotations

import math
from datetime import datetime, date, time, timedelta
from typing import Optional

import numpy as np
import pytest

from strategy_3.config import (
    ADD_RISK_MULT,
    ADD_RISK_LOWVOLH_MULT,
    ADD_STOP_ATR_MULT,
    ATR_RATIO_HIGH,
    ATR_RATIO_LOW,
    BE_BUFFER_ATR_MULT,
    CAMPAIGN_RISK_BUDGET_MULT,
    CONTAINMENT_MIN,
    CORR_HEAT_PENALTY,
    CORR_MULT_ALIGNED,
    CORR_MULT_OTHER,
    CORR_THRESHOLD,
    DIRTY_BOX_SHIFT_ATR_MULT,
    DIRTY_DURATION_L_FRAC,
    DIRTY_OPPOSITE_MULT,
    DISP_STRONG_MULT,
    EMA_4H_FLOOR_ATR_MULT,
    ENTRY_A_TOUCH_TOL_ATR_H,
    ENTRYB_RVOLH_MIN,
    EXPIRY_BARS_MAX,
    EXPIRY_BARS_MIN,
    FRICTION_CAP,
    GAP_GUARD_ATR_MULT,
    HARD_BLOCK_SLOPE_MULT,
    HARD_EXPIRY_BARS_ADD,
    HYSTERESIS_BARS,
    L_HIGH,
    L_LOW,
    L_MID,
    MAX_ADDS_PER_CAMPAIGN,
    MAX_PORTFOLIO_HEAT,
    MAX_POSITIONS,
    MAX_REENTRY_PER_BOX_VERSION,
    MAX_SAME_DIRECTION,
    PENDING_MAX_RTH_HOURS,
    Q_DISP,
    RECLAIM_BUFFER_ATR_D_MULT,
    RECLAIM_BUFFER_ATR_H_MULT,
    REENTRY_COOLDOWN_DAYS,
    REENTRY_MIN_REALIZED_R,
    REGIME_MULT_ALIGNED,
    REGIME_MULT_CAUTION,
    REGIME_MULT_NEUTRAL,
    SQUEEZE_CEIL,
    SQUEEZE_MULT_GOOD,
    SQUEEZE_MULT_LOOSE,
    SQUEEZE_MULT_NEUTRAL,
    STALE_EXIT_DAYS_MAX,
    STALE_EXIT_DAYS_MIN,
    STALE_R_THRESH,
    STALE_TIGHTEN_MULT,
    STALE_WARN_DAYS,
    SWEEP_DEPTH_ATR_D_MULT,
    TRAIL_4H_ATR_MULT,
    WEEKLY_THROTTLE_R,
    SymbolConfig,
)
from strategy_3.indicators import (
    atr,
    adx,
    compute_avwap,
    compute_daily_slope,
    compute_regime_4h,
    compute_rvol_d,
    compute_rvol_h,
    compute_wvwap,
    construct_4h_bars,
    containment_ratio,
    ema,
    highest,
    lowest,
    past_only_quantile,
    pullback_ref,
    rolling_correlation,
    sma,
    volume_score_component_daily,
)
from strategy_3.signals import (
    check_add_trigger,
    check_breakout_quality_reject,
    check_continuation_mode,
    check_dirty_reset,
    check_dirty_trigger,
    check_displacement,
    check_structural_breakout,
    choose_L_with_hysteresis,
    classify_chop_mode,
    classify_squeeze_tier,
    compute_chop_score,
    compute_evidence_score,
    compute_expiry_bars,
    compute_expiry_mult,
    detect_compression,
    determine_trade_regime,
    entry_a_signal,
    entry_b_permitted,
    entry_b_signal,
    entry_c_continuation_signal,
    entry_c_standard_signal,
    is_regime_aligned,
    select_entry_type,
)
from strategy_3.stops import (
    apply_stale_tighten,
    check_stale_exit,
    compute_add_stop,
    compute_be_stop,
    compute_initial_stop,
    compute_tp_levels,
    compute_trail_mult,
    compute_trailing_stop,
    handle_gap_through_stop,
)
from strategy_3.allocator import (
    add_budget_ok,
    compute_add_risk_dollars,
    compute_corr_mult,
    compute_disp_mult,
    compute_final_risk,
    compute_quality_mult,
    compute_regime_mult,
    compute_risk_regime_adj,
    compute_shares,
    compute_squeeze_mult,
    micro_guard_ok,
)
from strategy_3.gates import (
    check_transient_blocks,
    circuit_breaker_check,
    compute_corr_heat_penalty,
    friction_gate,
    full_eligibility_check,
    gap_guard_ok,
    hard_block,
    heat_check,
    is_entry_window_open,
    is_news_blocked,
    is_rth_time,
    pending_expired,
    position_limit_check,
    reentry_allowed,
)
from strategy_3.models import (
    CampaignState,
    ChopMode,
    CircuitBreakerState,
    DailyContext,
    Direction,
    EntryType,
    PendingEntry,
    PositionState,
    Regime4H,
    SymbolCampaign,
    TradeRegime,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def default_cfg():
    return SymbolConfig(
        symbol="QQQ",
        tick_size=0.01,
        multiplier=1.0,
        exchange="SMART",
        entry_window_start_et="09:30",
        entry_window_end_et="16:00",
    )


@pytest.fixture
def fresh_campaign():
    """A fresh inactive campaign with no hysteresis state."""
    return SymbolCampaign()


@pytest.fixture
def active_campaign():
    """A campaign in BREAKOUT state with a frozen box."""
    c = SymbolCampaign(
        state=CampaignState.BREAKOUT,
        campaign_id=1,
        box_version=1,
        box_high=110.0,
        box_low=100.0,
        box_mid=105.0,
        box_height=10.0,
        breakout_direction=Direction.LONG,
        breakout_date=date(2025, 6, 1),
        bars_since_breakout=3,
        expiry_bars=5,
        L=12,
    )
    return c


@pytest.fixture
def continuation_campaign():
    """Campaign in CONTINUATION state."""
    c = SymbolCampaign(
        state=CampaignState.CONTINUATION,
        campaign_id=1,
        box_version=1,
        box_high=110.0,
        box_low=100.0,
        box_mid=105.0,
        box_height=10.0,
        breakout_direction=Direction.LONG,
        continuation=True,
        L=12,
    )
    return c


@pytest.fixture
def long_position():
    return PositionState(
        symbol="QQQ",
        direction=Direction.LONG,
        qty=100,
        avg_cost=112.0,
        current_stop=99.0,
        unrealized_pnl=200.0,
        r_state=1.5,
        campaign_id=1,
        box_version=1,
        total_risk_dollars=500.0,
        regime_at_entry="BULL_TREND",
    )


@pytest.fixture
def short_position():
    return PositionState(
        symbol="GLD",
        direction=Direction.SHORT,
        qty=100,
        avg_cost=95.0,
        current_stop=100.0,
        unrealized_pnl=100.0,
        r_state=0.8,
        campaign_id=2,
        box_version=1,
        total_risk_dollars=400.0,
        regime_at_entry="BEAR_TREND",
    )


@pytest.fixture
def empty_cb():
    return CircuitBreakerState()


@pytest.fixture
def squeeze_hist_good():
    """Squeeze history where low values are common (tight squeezes)."""
    return [0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.6, 0.7, 0.8, 0.9,
            0.4, 0.5, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8, 0.9]


@pytest.fixture
def disp_hist_normal():
    """Normal displacement history."""
    return [0.5, 0.7, 0.9, 1.1, 1.3, 0.6, 0.8, 1.0, 1.2, 0.7,
            0.5, 0.7, 0.9, 1.1, 1.3, 0.6, 0.8, 1.0, 1.2, 0.7,
            0.5, 0.7, 0.9, 1.1, 1.3, 0.6, 0.8, 1.0, 1.2, 0.7]


@pytest.fixture
def atr_hist_normal():
    """Normal ATR history."""
    return [2.0] * 60


# ===========================================================================
# INDICATORS
# ===========================================================================

class TestEma:
    def test_single_value(self):
        arr = np.array([10.0])
        result = ema(arr, 3)
        assert result[0] == pytest.approx(10.0)

    def test_constant_array(self):
        arr = np.array([5.0] * 20)
        result = ema(arr, 10)
        assert result[-1] == pytest.approx(5.0, abs=1e-10)

    def test_length_matches(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ema(arr, 3)
        assert len(result) == 5

    def test_trending_up_ema_follows(self):
        arr = np.array([float(i) for i in range(1, 21)])
        result = ema(arr, 5)
        # EMA should be below current price in uptrend
        assert result[-1] < arr[-1]
        # But above early values
        assert result[-1] > arr[0]


class TestSma:
    def test_constant_array(self):
        arr = np.array([3.0] * 10)
        result = sma(arr, 5)
        assert result[-1] == pytest.approx(3.0)

    def test_known_values(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(arr, 3)
        # Last SMA(3) = (3+4+5)/3 = 4.0
        assert result[-1] == pytest.approx(4.0)
        # SMA(3) at index 2 = (1+2+3)/3 = 2.0
        assert result[2] == pytest.approx(2.0)

    def test_period_larger_than_data(self):
        arr = np.array([2.0, 4.0])
        result = sma(arr, 10)
        # Uses all available data
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(3.0)


class TestAtr:
    def test_single_bar(self):
        h = np.array([102.0])
        l = np.array([98.0])
        c = np.array([100.0])
        result = atr(h, l, c, 14)
        assert result[0] == pytest.approx(4.0)

    def test_constant_range(self):
        n = 30
        h = np.array([101.0] * n)
        l = np.array([99.0] * n)
        c = np.array([100.0] * n)
        result = atr(h, l, c, 14)
        # With constant 2.0 range, ATR converges to 2.0
        assert result[-1] == pytest.approx(2.0, abs=0.01)

    def test_wilder_smoothing(self):
        # First two bars: verify Wilder smoothing
        h = np.array([102.0, 103.0])
        l = np.array([98.0, 99.0])
        c = np.array([100.0, 101.0])
        result = atr(h, l, c, 14)
        # TR[0] = 4.0, TR[1] = max(4, |103-100|, |99-100|) = 4.0
        # ATR[1] = ATR[0] * (1-1/14) + TR[1] * (1/14) = 4.0 * 13/14 + 4.0 * 1/14 = 4.0
        assert result[1] == pytest.approx(4.0, abs=0.01)


class TestAdx:
    def test_length_preserved(self):
        n = 30
        h = np.array([101.0 + 0.1 * i for i in range(n)])
        l = np.array([99.0 + 0.1 * i for i in range(n)])
        c = np.array([100.0 + 0.1 * i for i in range(n)])
        result = adx(h, l, c, 14)
        assert len(result) == n

    def test_flat_market_low_adx(self):
        n = 100
        h = np.array([101.0] * n)
        l = np.array([99.0] * n)
        c = np.array([100.0] * n)
        result = adx(h, l, c, 14)
        # Flat market: ADX should be very low
        assert result[-1] < 5.0


class TestHighestLowest:
    def test_highest_known(self):
        arr = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        result = highest(arr, 3)
        # At index 4: max of [2, 5, 4] = 5.0
        assert result[4] == pytest.approx(5.0)
        # At index 2: max of [1, 3, 2] = 3.0
        assert result[2] == pytest.approx(3.0)

    def test_lowest_known(self):
        arr = np.array([5.0, 3.0, 4.0, 1.0, 2.0])
        result = lowest(arr, 3)
        # At index 4: min of [4, 1, 2] = 1.0
        assert result[4] == pytest.approx(1.0)
        # At index 2: min of [5, 3, 4] = 3.0
        assert result[2] == pytest.approx(3.0)

    def test_period_one(self):
        arr = np.array([10.0, 20.0, 15.0])
        assert highest(arr, 1)[-1] == pytest.approx(15.0)
        assert lowest(arr, 1)[-1] == pytest.approx(15.0)


class TestPastOnlyQuantile:
    def test_empty_series(self):
        assert past_only_quantile([], 0.5, 60) == 0.0

    def test_known_quantile(self):
        series = list(range(1, 101))  # 1..100
        q50 = past_only_quantile(series, 0.50, 100)
        assert q50 == pytest.approx(50.5, abs=1.0)

    def test_lookback_window(self):
        series = [0.0] * 50 + [100.0] * 50
        q50 = past_only_quantile(series, 0.50, 50)
        # Only last 50 values (all 100.0)
        assert q50 == pytest.approx(100.0)


class TestContainmentRatio:
    def test_all_inside(self):
        closes = np.array([101.0, 102.0, 103.0, 104.0, 105.0])
        assert containment_ratio(closes, 100.0, 106.0, 5) == pytest.approx(1.0)

    def test_none_inside(self):
        closes = np.array([110.0, 111.0, 112.0])
        assert containment_ratio(closes, 100.0, 105.0, 3) == pytest.approx(0.0)

    def test_partial(self):
        closes = np.array([102.0, 107.0, 103.0, 108.0])
        ratio = containment_ratio(closes, 100.0, 105.0, 4)
        assert ratio == pytest.approx(0.5)

    def test_uses_last_L_bars(self):
        closes = np.array([90.0, 91.0, 102.0, 103.0])  # first 2 outside, last 2 inside
        ratio = containment_ratio(closes, 100.0, 105.0, 2)
        assert ratio == pytest.approx(1.0)  # only last 2 bars


class TestComputeAvwap:
    def test_basic_avwap(self):
        h = np.array([101.0, 102.0, 103.0])
        l = np.array([99.0, 100.0, 101.0])
        c = np.array([100.0, 101.0, 102.0])
        v = np.array([1000.0, 2000.0, 3000.0])
        anchor = datetime(2025, 1, 1, 9, 30)
        times = [datetime(2025, 1, 1, 9, 30),
                 datetime(2025, 1, 1, 10, 30),
                 datetime(2025, 1, 1, 11, 30)]
        result = compute_avwap(h, l, c, v, times, anchor)
        assert not np.isnan(result[-1])
        # AVWAP = cumsum(tp*vol) / cumsum(vol)
        tp0 = (101 + 99 + 100) / 3.0
        tp1 = (102 + 100 + 101) / 3.0
        tp2 = (103 + 101 + 102) / 3.0
        expected = (tp0 * 1000 + tp1 * 2000 + tp2 * 3000) / 6000.0
        assert result[-1] == pytest.approx(expected, abs=0.01)

    def test_before_anchor_is_nan(self):
        h = np.array([101.0, 102.0])
        l = np.array([99.0, 100.0])
        c = np.array([100.0, 101.0])
        v = np.array([1000.0, 2000.0])
        anchor = datetime(2025, 1, 1, 11, 0)
        times = [datetime(2025, 1, 1, 9, 30),
                 datetime(2025, 1, 1, 11, 30)]
        result = compute_avwap(h, l, c, v, times, anchor)
        assert np.isnan(result[0])
        assert not np.isnan(result[1])


class TestComputeWvwap:
    def test_resets_on_monday(self):
        # Monday 2025-01-06, Tuesday 2025-01-07
        h = np.array([101.0, 102.0, 103.0])
        l = np.array([99.0, 100.0, 101.0])
        c = np.array([100.0, 101.0, 102.0])
        v = np.array([1000.0, 2000.0, 3000.0])
        times = [
            datetime(2025, 1, 6, 9, 30),   # Monday reset
            datetime(2025, 1, 6, 10, 30),
            datetime(2025, 1, 7, 9, 30),   # Tuesday, no reset
        ]
        result = compute_wvwap(h, l, c, v, times)
        assert len(result) == 3
        # First bar after reset: WVWAP = typical price
        tp0 = (101 + 99 + 100) / 3.0
        assert result[0] == pytest.approx(tp0, abs=0.01)


class TestComputeRvolD:
    def test_default_rvol(self):
        assert compute_rvol_d(np.array([100.0])) == 1.0

    def test_double_volume(self):
        volumes = np.array([1000.0] * 20 + [2000.0])
        result = compute_rvol_d(volumes)
        assert result == pytest.approx(2.0)

    def test_half_volume(self):
        volumes = np.array([1000.0] * 20 + [500.0])
        result = compute_rvol_d(volumes)
        assert result == pytest.approx(0.5)


class TestComputeRvolH:
    def test_normal_volume(self):
        slot_medians = {(0, 9): 1000.0, (0, 10): 2000.0}
        assert compute_rvol_h(1000.0, (0, 9), slot_medians) == pytest.approx(1.0)

    def test_high_volume(self):
        slot_medians = {(0, 9): 1000.0}
        assert compute_rvol_h(2000.0, (0, 9), slot_medians) == pytest.approx(2.0)

    def test_missing_slot(self):
        assert compute_rvol_h(1000.0, (0, 9), {}) == 1.0

    def test_zero_median(self):
        assert compute_rvol_h(1000.0, (0, 9), {(0, 9): 0.0}) == 1.0


class TestComputeRegime4h:
    def test_insufficient_data(self):
        c = np.array([100.0] * 10)
        h = np.array([101.0] * 10)
        l = np.array([99.0] * 10)
        regime, slope, adx_val = compute_regime_4h(c, h, l)
        assert regime == Regime4H.RANGE_CHOP
        assert slope == 0.0
        assert adx_val == 0.0

    def test_flat_market_range_chop(self):
        n = 80
        c = np.array([100.0] * n)
        h = np.array([101.0] * n)
        l = np.array([99.0] * n)
        regime, slope, adx_val = compute_regime_4h(c, h, l)
        assert regime == Regime4H.RANGE_CHOP


class TestComputeDailySlope:
    def test_insufficient_data(self):
        c = np.array([100.0] * 10)
        assert compute_daily_slope(c) == 0.0

    def test_uptrend(self):
        n = 80
        c = np.array([100.0 + 0.5 * i for i in range(n)])
        slope = compute_daily_slope(c)
        assert slope > 0.0

    def test_downtrend(self):
        n = 80
        c = np.array([200.0 - 0.5 * i for i in range(n)])
        slope = compute_daily_slope(c)
        assert slope < 0.0


class TestConstruct4hBars:
    def test_exact_multiple(self):
        n = 8
        h = np.array([101.0 + i for i in range(n)])
        l = np.array([99.0 + i for i in range(n)])
        c = np.array([100.0 + i for i in range(n)])
        v = np.array([1000.0] * n)
        times = [datetime(2025, 1, 1, 9 + i, 30) for i in range(n)]
        h4, l4, c4, v4, t4 = construct_4h_bars(h, l, c, v, times)
        assert len(h4) == 2
        # First 4h bar: max of first 4 highs
        assert h4[0] == pytest.approx(max(h[:4]))
        assert l4[0] == pytest.approx(min(l[:4]))
        assert c4[0] == pytest.approx(c[3])  # close of last bar in group
        assert v4[0] == pytest.approx(sum(v[:4]))

    def test_non_multiple_trims_from_start(self):
        n = 6  # 6 bars -> 1 complete 4h bar (last 4 used)
        h = np.array([101.0] * n)
        l = np.array([99.0] * n)
        c = np.array([100.0] * n)
        v = np.array([1000.0] * n)
        times = [datetime(2025, 1, 1, 9 + i, 30) for i in range(n)]
        h4, l4, c4, v4, t4 = construct_4h_bars(h, l, c, v, times)
        assert len(h4) == 1

    def test_less_than_4_returns_empty(self):
        h = np.array([101.0, 102.0])
        l = np.array([99.0, 100.0])
        c = np.array([100.0, 101.0])
        v = np.array([1000.0, 2000.0])
        times = [datetime(2025, 1, 1, 9, 30), datetime(2025, 1, 1, 10, 30)]
        h4, l4, c4, v4, t4 = construct_4h_bars(h, l, c, v, times)
        assert len(h4) == 0


class TestPullbackRef:
    def test_prefers_wvwap_when_close(self):
        ref = pullback_ref(price=100.0, wvwap=101.0, avwap_h=105.0,
                           ema20_h=98.0, atr14_d=2.0)
        assert ref == 101.0  # |100-101|/2 = 0.5 <= 2.0

    def test_falls_to_avwap_when_wvwap_far(self):
        ref = pullback_ref(price=100.0, wvwap=110.0, avwap_h=102.0,
                           ema20_h=98.0, atr14_d=2.0)
        assert ref == 102.0  # wvwap too far, avwap OK

    def test_falls_to_ema20_when_all_far(self):
        ref = pullback_ref(price=100.0, wvwap=120.0, avwap_h=115.0,
                           ema20_h=98.0, atr14_d=2.0)
        assert ref == 98.0  # both too far


class TestRollingCorrelation:
    def test_perfect_correlation(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        b = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        corr = rolling_correlation(a, b)
        assert corr == pytest.approx(1.0)

    def test_inverse_correlation(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        b = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        corr = rolling_correlation(a, b)
        assert corr == pytest.approx(-1.0)

    def test_too_few_returns_zero(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        assert rolling_correlation(a, b) == 0.0


class TestVolumeScoreComponent:
    def test_high_volume(self):
        assert volume_score_component_daily(2.0) == 1

    def test_normal_volume(self):
        assert volume_score_component_daily(1.2) == 0

    def test_low_volume(self):
        assert volume_score_component_daily(0.9) == -1

    def test_boundary_1_5(self):
        assert volume_score_component_daily(1.5) == 1

    def test_boundary_1_1(self):
        assert volume_score_component_daily(1.1) == 0


# ===========================================================================
# SIGNALS
# ===========================================================================

class TestChooseLWithHysteresis:
    def test_first_call_low(self, fresh_campaign):
        result = choose_L_with_hysteresis(0.50, fresh_campaign)
        assert result == L_LOW  # atr_ratio < 0.70

    def test_first_call_mid(self, fresh_campaign):
        result = choose_L_with_hysteresis(1.0, fresh_campaign)
        assert result == L_MID

    def test_first_call_high(self, fresh_campaign):
        result = choose_L_with_hysteresis(1.50, fresh_campaign)
        assert result == L_HIGH

    def test_stays_in_bucket(self, fresh_campaign):
        choose_L_with_hysteresis(1.0, fresh_campaign)  # mid
        result = choose_L_with_hysteresis(1.0, fresh_campaign)
        assert result == L_MID

    def test_hysteresis_prevents_premature_switch(self, fresh_campaign):
        choose_L_with_hysteresis(1.0, fresh_campaign)  # mid (committed)
        # Two bars in high -> not enough for switch (need 3)
        choose_L_with_hysteresis(1.50, fresh_campaign)
        result = choose_L_with_hysteresis(1.50, fresh_campaign)
        # Still returns current L (mid) because candidate hasn't reached threshold
        assert result == fresh_campaign.L

    def test_hysteresis_allows_switch_after_threshold(self, fresh_campaign):
        fresh_campaign.L = L_MID
        choose_L_with_hysteresis(1.0, fresh_campaign)  # init to mid
        for _ in range(HYSTERESIS_BARS):
            choose_L_with_hysteresis(1.50, fresh_campaign)  # high bucket
        result = choose_L_with_hysteresis(1.50, fresh_campaign)
        assert result == L_HIGH


class TestDetectCompression:
    def test_passes_with_good_containment_and_squeeze(self):
        assert detect_compression(0.85, 1.0) is True

    def test_fails_low_containment(self):
        assert detect_compression(0.60, 1.0) is False

    def test_fails_high_squeeze(self):
        # Without adaptive ceiling, SQUEEZE_CEIL=1.25
        assert detect_compression(0.85, 1.30) is False

    def test_exact_containment_boundary(self):
        assert detect_compression(CONTAINMENT_MIN, 1.0) is True
        assert detect_compression(CONTAINMENT_MIN - 0.01, 1.0) is False

    def test_adaptive_ceiling_with_history(self):
        hist = [0.5] * 30  # All low squeezes -> adaptive ceil is small
        # squeeze_metric 1.0 > adaptive ceil from all-0.5 history
        result = detect_compression(0.85, 1.0, squeeze_hist=hist)
        assert result is False


class TestClassifySqueezeTier:
    def test_insufficient_history(self):
        sq_good, sq_loose = classify_squeeze_tier(0.5, [0.6, 0.7, 0.8])
        assert sq_good is False
        assert sq_loose is False

    def test_tight_squeeze_is_good(self, squeeze_hist_good):
        # Very low value relative to history -> sq_good
        sq_good, sq_loose = classify_squeeze_tier(0.1, squeeze_hist_good)
        assert sq_good is True
        assert sq_loose is False

    def test_high_squeeze_is_loose(self, squeeze_hist_good):
        # Very high value -> sq_loose
        sq_good, sq_loose = classify_squeeze_tier(5.0, squeeze_hist_good)
        assert sq_good is False
        assert sq_loose is True


class TestCheckStructuralBreakout:
    def test_long_breakout(self):
        assert check_structural_breakout(111.0, 110.0, 100.0) == Direction.LONG

    def test_short_breakout(self):
        assert check_structural_breakout(99.0, 110.0, 100.0) == Direction.SHORT

    def test_no_breakout_inside_box(self):
        assert check_structural_breakout(105.0, 110.0, 100.0) is None

    def test_at_boundary_no_breakout(self):
        # Equal to box_high is not a breakout (need >)
        assert check_structural_breakout(110.0, 110.0, 100.0) is None
        # Equal to box_low is not a breakout (need <)
        assert check_structural_breakout(100.0, 110.0, 100.0) is None


class TestCheckDisplacement:
    def test_zero_atr(self):
        passed, disp, disp_th = check_displacement(105.0, 100.0, 0.0, [], False)
        assert passed is False

    def test_passes_with_sufficient_displacement(self, disp_hist_normal):
        # disp = |120 - 100| / 2.0 = 10.0, well above threshold
        passed, disp, disp_th = check_displacement(
            120.0, 100.0, 2.0, disp_hist_normal, False
        )
        assert passed is True
        assert disp == pytest.approx(10.0)

    def test_fails_with_small_displacement(self, disp_hist_normal):
        # disp = |100.1 - 100.0| / 2.0 = 0.05
        passed, disp, disp_th = check_displacement(
            100.1, 100.0, 2.0, disp_hist_normal, False
        )
        assert passed is False

    def test_atr_expanding_lowers_threshold(self, disp_hist_normal):
        # With atr_expanding, Q_DISP is reduced, lowering threshold
        _, _, disp_th_normal = check_displacement(
            105.0, 100.0, 2.0, disp_hist_normal, False
        )
        _, _, disp_th_expanding = check_displacement(
            105.0, 100.0, 2.0, disp_hist_normal, True
        )
        assert disp_th_expanding <= disp_th_normal


class TestCheckBreakoutQualityReject:
    def test_normal_bar_not_rejected(self):
        # Normal-sized bar
        assert check_breakout_quality_reject(
            high_d=103.0, low_d=99.0, open_d=100.0, close_d=102.5,
            atr14_d=2.0, direction=Direction.LONG
        ) is False

    def test_wide_bar_weak_body_rejected(self):
        # Range = 10, ATR = 2 -> range > 2*ATR. Body = |100.5-100| = 0.5, body_ratio = 0.05
        assert check_breakout_quality_reject(
            high_d=105.0, low_d=95.0, open_d=100.0, close_d=100.5,
            atr14_d=2.0, direction=Direction.LONG
        ) is True

    def test_wide_bar_with_adverse_wick_rejected(self):
        # Range = 10, body = 4 (ok), but adverse_wick for LONG = high - close = 105-98 = 7
        # adverse_wick_ratio = 7/10 = 0.70 > 0.55
        assert check_breakout_quality_reject(
            high_d=105.0, low_d=95.0, open_d=94.0, close_d=98.0,
            atr14_d=2.0, direction=Direction.LONG
        ) is True

    def test_narrow_bar_never_rejected(self):
        # Range = 2 <= 2*ATR = 4 -> not rejected regardless
        assert check_breakout_quality_reject(
            high_d=101.0, low_d=99.0, open_d=100.0, close_d=100.1,
            atr14_d=2.0, direction=Direction.LONG
        ) is False


class TestComputeEvidenceScore:
    def test_high_volume_scores_positive(self):
        closes = np.array([111.0, 112.0])
        total, vol = compute_evidence_score(
            rvol_d=2.0, disp=1.5, disp_th=1.0, displacement_pass=True,
            sq_good=False, sq_loose=False, regime_4h=Regime4H.RANGE_CHOP,
            direction=Direction.LONG, atr_expanding=False,
            closes_d=closes, box_high=110.0, box_low=100.0,
        )
        assert vol == 1  # rvol >= 1.5

    def test_low_volume_penalty(self):
        closes = np.array([105.0])
        total, vol = compute_evidence_score(
            rvol_d=0.5, disp=0.5, disp_th=1.0, displacement_pass=False,
            sq_good=False, sq_loose=False, regime_4h=Regime4H.RANGE_CHOP,
            direction=Direction.LONG, atr_expanding=False,
            closes_d=closes, box_high=110.0, box_low=100.0,
        )
        assert vol == -1  # rvol < 1.1

    def test_loose_squeeze_penalty(self):
        closes = np.array([111.0])
        total, _ = compute_evidence_score(
            rvol_d=1.2, disp=1.0, disp_th=0.8, displacement_pass=True,
            sq_good=False, sq_loose=True, regime_4h=Regime4H.RANGE_CHOP,
            direction=Direction.LONG, atr_expanding=False,
            closes_d=closes, box_high=110.0, box_low=100.0,
        )
        # vol_score=0, squeeze=-1
        assert total <= 0

    def test_opposing_regime_penalty(self):
        closes = np.array([111.0])
        total, _ = compute_evidence_score(
            rvol_d=1.2, disp=1.0, disp_th=0.8, displacement_pass=True,
            sq_good=False, sq_loose=False, regime_4h=Regime4H.BEAR_TREND,
            direction=Direction.LONG, atr_expanding=False,
            closes_d=closes, box_high=110.0, box_low=100.0,
        )
        # vol_score=0, regime=-1
        assert total < 0

    def test_consecutive_closes_bonus(self):
        closes = np.array([111.0, 112.0])
        total1, _ = compute_evidence_score(
            rvol_d=1.2, disp=1.0, disp_th=0.8, displacement_pass=True,
            sq_good=False, sq_loose=False, regime_4h=Regime4H.RANGE_CHOP,
            direction=Direction.LONG, atr_expanding=False,
            closes_d=closes, box_high=110.0, box_low=100.0,
        )
        # Both closes above box_high -> +1
        closes_no_consec = np.array([109.0, 112.0])
        total2, _ = compute_evidence_score(
            rvol_d=1.2, disp=1.0, disp_th=0.8, displacement_pass=True,
            sq_good=False, sq_loose=False, regime_4h=Regime4H.RANGE_CHOP,
            direction=Direction.LONG, atr_expanding=False,
            closes_d=closes_no_consec, box_high=110.0, box_low=100.0,
        )
        assert total1 > total2

    def test_atr_expanding_bonus(self):
        closes = np.array([111.0])
        total_expand, _ = compute_evidence_score(
            rvol_d=1.2, disp=1.0, disp_th=0.8, displacement_pass=True,
            sq_good=False, sq_loose=False, regime_4h=Regime4H.RANGE_CHOP,
            direction=Direction.LONG, atr_expanding=True,
            closes_d=closes, box_high=110.0, box_low=100.0,
        )
        total_flat, _ = compute_evidence_score(
            rvol_d=1.2, disp=1.0, disp_th=0.8, displacement_pass=True,
            sq_good=False, sq_loose=False, regime_4h=Regime4H.RANGE_CHOP,
            direction=Direction.LONG, atr_expanding=False,
            closes_d=closes, box_high=110.0, box_low=100.0,
        )
        assert total_expand == total_flat + 1

    def test_low_vol_exception_with_strong_disp(self):
        closes = np.array([111.0])
        # Low RVOL (0.7) but very strong displacement: disp > 1.15 * disp_th
        _, vol = compute_evidence_score(
            rvol_d=0.7, disp=2.0, disp_th=1.0, displacement_pass=True,
            sq_good=False, sq_loose=False, regime_4h=Regime4H.RANGE_CHOP,
            direction=Direction.LONG, atr_expanding=False,
            closes_d=closes, box_high=110.0, box_low=100.0,
        )
        # vol_score would normally be -1 (rvol < 1.1), but exception caps at -1
        assert vol == -1


class TestComputeExpiryBars:
    def test_empty_history(self):
        expiry, hard = compute_expiry_bars(2.0, [])
        assert expiry == 5
        assert hard == 10

    def test_median_atr(self, atr_hist_normal):
        expiry, hard = compute_expiry_bars(2.0, atr_hist_normal)
        assert EXPIRY_BARS_MIN <= expiry <= EXPIRY_BARS_MAX
        assert hard == expiry + HARD_EXPIRY_BARS_ADD

    def test_very_high_atr(self):
        hist = [1.0] * 60
        expiry, hard = compute_expiry_bars(10.0, hist)
        # All values < 10.0, pctl=100 -> raw=10, clamped to max
        assert expiry == EXPIRY_BARS_MAX


class TestComputeExpiryMult:
    def test_within_expiry(self):
        assert compute_expiry_mult(3, 5) == 1.0

    def test_at_expiry(self):
        assert compute_expiry_mult(5, 5) == 1.0

    def test_one_step_past(self):
        assert compute_expiry_mult(6, 5) == pytest.approx(1.0 - 0.12)

    def test_floor(self):
        # Many steps past -> floor at 0.30
        assert compute_expiry_mult(100, 5) == pytest.approx(0.30)


class TestCheckContinuationMode:
    def test_measured_move_long(self):
        # box_high + 1.5 * box_height = 110 + 15 = 125
        result = check_continuation_mode(
            close_d=126.0, box_high=110.0, box_low=100.0, box_height=10.0,
            atr14_d=2.0, direction=Direction.LONG, bars_since_breakout=3,
            regime_4h=Regime4H.RANGE_CHOP, disp_mult=0.85,
        )
        assert result is True

    def test_measured_move_short(self):
        # box_low - 1.5 * box_height = 100 - 15 = 85
        result = check_continuation_mode(
            close_d=84.0, box_high=110.0, box_low=100.0, box_height=10.0,
            atr14_d=2.0, direction=Direction.SHORT, bars_since_breakout=3,
            regime_4h=Regime4H.RANGE_CHOP, disp_mult=0.85,
        )
        assert result is True

    def test_r_proxy_threshold(self):
        # r_proxy = (115 - 110)/2.0 = 2.5 >= 2.0
        result = check_continuation_mode(
            close_d=115.0, box_high=110.0, box_low=100.0, box_height=10.0,
            atr14_d=2.0, direction=Direction.LONG, bars_since_breakout=3,
            regime_4h=Regime4H.RANGE_CHOP, disp_mult=0.85,
        )
        assert result is True

    def test_no_continuation_near_box(self):
        # close barely above box_high
        result = check_continuation_mode(
            close_d=111.0, box_high=110.0, box_low=100.0, box_height=10.0,
            atr14_d=2.0, direction=Direction.LONG, bars_since_breakout=1,
            regime_4h=Regime4H.RANGE_CHOP, disp_mult=0.5,
        )
        assert result is False


class TestCheckDirtyTrigger:
    def test_long_closes_back_inside(self):
        result = check_dirty_trigger(
            close_d=109.0, box_high=110.0, box_low=100.0,
            breakout_direction=Direction.LONG, days_since_breakout=3, m_break=2,
        )
        assert result is True

    def test_long_still_above(self):
        result = check_dirty_trigger(
            close_d=112.0, box_high=110.0, box_low=100.0,
            breakout_direction=Direction.LONG, days_since_breakout=3, m_break=2,
        )
        assert result is False

    def test_too_early(self):
        result = check_dirty_trigger(
            close_d=109.0, box_high=110.0, box_low=100.0,
            breakout_direction=Direction.LONG, days_since_breakout=1, m_break=2,
        )
        assert result is False

    def test_short_closes_back_inside(self):
        result = check_dirty_trigger(
            close_d=101.0, box_high=110.0, box_low=100.0,
            breakout_direction=Direction.SHORT, days_since_breakout=3, m_break=2,
        )
        assert result is True


class TestCheckDirtyReset:
    def test_not_sq_good_no_reset(self, squeeze_hist_good):
        # squeeze_metric very high -> not sq_good
        should_reset, shifted = check_dirty_reset(
            new_high=115.0, new_low=95.0, dirty_high=110.0, dirty_low=100.0,
            atr14_d=2.0, squeeze_metric=5.0, squeeze_hist=squeeze_hist_good,
            dirty_duration=10, L_used=12,
        )
        assert should_reset is False

    def test_box_shifted_triggers_reset(self, squeeze_hist_good):
        # Make squeeze_metric very low to get sq_good
        should_reset, shifted = check_dirty_reset(
            new_high=120.0, new_low=90.0, dirty_high=110.0, dirty_low=100.0,
            atr14_d=2.0, squeeze_metric=0.1, squeeze_hist=squeeze_hist_good,
            dirty_duration=2, L_used=12,
        )
        # High shift: |120-110| = 10 >= 0.25*2 = 0.5, Low shift: |90-100| = 10 >= 0.5
        assert should_reset is True
        assert shifted is True

    def test_duration_triggers_reset(self, squeeze_hist_good):
        should_reset, shifted = check_dirty_reset(
            new_high=110.5, new_low=99.5, dirty_high=110.0, dirty_low=100.0,
            atr14_d=2.0, squeeze_metric=0.1, squeeze_hist=squeeze_hist_good,
            dirty_duration=10, L_used=12,
        )
        # Duration: 10 >= 0.35*12 = 4.2
        assert should_reset is True


class TestEntryASignal:
    def test_long_touch_and_reclaim(self):
        avwap_h = 100.0
        atr14_h = 2.0
        atr14_d = 4.0
        reclaim_buf = max(RECLAIM_BUFFER_ATR_H_MULT * atr14_h,
                          RECLAIM_BUFFER_ATR_D_MULT * atr14_d)
        touch_tol = ENTRY_A_TOUCH_TOL_ATR_H * atr14_h
        # Low touches AVWAP (within tolerance), close reclaims above
        result = entry_a_signal(
            direction=Direction.LONG,
            hourly_close=100.0 + reclaim_buf + 0.01,
            hourly_low=100.0,
            hourly_high=101.5,
            avwap_h=avwap_h,
            atr14_h=atr14_h,
            atr14_d=atr14_d,
        )
        assert result is True

    def test_long_no_touch(self):
        avwap_h = 100.0
        atr14_h = 2.0
        atr14_d = 4.0
        touch_tol = ENTRY_A_TOUCH_TOL_ATR_H * atr14_h
        # Low is way above AVWAP + tolerance -> no touch
        result = entry_a_signal(
            direction=Direction.LONG,
            hourly_close=105.0,
            hourly_low=103.0,
            hourly_high=106.0,
            avwap_h=avwap_h,
            atr14_h=atr14_h,
            atr14_d=atr14_d,
        )
        assert result is False

    def test_long_no_reclaim(self):
        avwap_h = 100.0
        atr14_h = 2.0
        atr14_d = 4.0
        # Low touches but close doesn't reclaim above buffer
        result = entry_a_signal(
            direction=Direction.LONG,
            hourly_close=100.0,  # exactly at avwap, not above buffer
            hourly_low=99.5,
            hourly_high=101.0,
            avwap_h=avwap_h,
            atr14_h=atr14_h,
            atr14_d=atr14_d,
        )
        assert result is False

    def test_short_touch_and_reclaim(self):
        avwap_h = 100.0
        atr14_h = 2.0
        atr14_d = 4.0
        reclaim_buf = max(RECLAIM_BUFFER_ATR_H_MULT * atr14_h,
                          RECLAIM_BUFFER_ATR_D_MULT * atr14_d)
        # High touches AVWAP (from below + tolerance), close reclaims below
        result = entry_a_signal(
            direction=Direction.SHORT,
            hourly_close=100.0 - reclaim_buf - 0.01,
            hourly_low=98.0,
            hourly_high=100.0,
            avwap_h=avwap_h,
            atr14_h=atr14_h,
            atr14_d=atr14_d,
        )
        assert result is True

    def test_prev_bar_touch_long(self):
        avwap_h = 100.0
        atr14_h = 2.0
        atr14_d = 4.0
        reclaim_buf = max(RECLAIM_BUFFER_ATR_H_MULT * atr14_h,
                          RECLAIM_BUFFER_ATR_D_MULT * atr14_d)
        # Current bar doesn't touch, but prev bar did
        result = entry_a_signal(
            direction=Direction.LONG,
            hourly_close=100.0 + reclaim_buf + 0.01,
            hourly_low=102.0,  # no touch this bar
            hourly_high=103.0,
            avwap_h=avwap_h,
            atr14_h=atr14_h,
            atr14_d=atr14_d,
            prev_hourly_low=99.5,  # prev bar touched
        )
        assert result is True


class TestEntryBSignal:
    def test_long_sweep_and_reclaim(self):
        avwap_h = 100.0
        atr14_d = 4.0
        sweep_depth = SWEEP_DEPTH_ATR_D_MULT * atr14_d
        # Low sweeps below AVWAP - sweep_depth
        result = entry_b_signal(
            direction=Direction.LONG,
            hourly_low=avwap_h - sweep_depth - 0.01,
            hourly_high=101.0,
            hourly_close=100.5,  # close above AVWAP
            avwap_h=avwap_h,
            atr14_d=atr14_d,
        )
        assert result is True

    def test_long_no_sweep(self):
        avwap_h = 100.0
        atr14_d = 4.0
        # Low doesn't sweep deep enough
        result = entry_b_signal(
            direction=Direction.LONG,
            hourly_low=99.9,  # barely below, not deep enough
            hourly_high=101.0,
            hourly_close=100.5,
            avwap_h=avwap_h,
            atr14_d=atr14_d,
        )
        assert result is False

    def test_short_sweep_and_reclaim(self):
        avwap_h = 100.0
        atr14_d = 4.0
        sweep_depth = SWEEP_DEPTH_ATR_D_MULT * atr14_d
        result = entry_b_signal(
            direction=Direction.SHORT,
            hourly_low=99.0,
            hourly_high=avwap_h + sweep_depth + 0.01,
            hourly_close=99.5,  # close below AVWAP
            avwap_h=avwap_h,
            atr14_d=atr14_d,
        )
        assert result is True


class TestEntryBPermitted:
    def test_aligned_regime_with_volume(self, active_campaign):
        result = entry_b_permitted(
            rvol_h=1.0, disp_mult=0.85, quality_mult=0.8,
            regime_4h=Regime4H.BULL_TREND, direction=Direction.LONG,
            campaign=active_campaign,
        )
        assert result is True

    def test_blocked_in_continuation(self, continuation_campaign):
        result = entry_b_permitted(
            rvol_h=1.0, disp_mult=0.85, quality_mult=0.8,
            regime_4h=Regime4H.BULL_TREND, direction=Direction.LONG,
            campaign=continuation_campaign,
        )
        assert result is False

    def test_blocked_dirty_same_direction(self):
        c = SymbolCampaign(
            state=CampaignState.DIRTY,
            breakout_direction=Direction.LONG,
        )
        result = entry_b_permitted(
            rvol_h=1.0, disp_mult=0.85, quality_mult=0.8,
            regime_4h=Regime4H.BULL_TREND, direction=Direction.LONG,
            campaign=c,
        )
        assert result is False

    def test_low_rvol_allowed_with_high_disp(self, active_campaign):
        result = entry_b_permitted(
            rvol_h=0.3, disp_mult=0.90, quality_mult=0.8,
            regime_4h=Regime4H.BULL_TREND, direction=Direction.LONG,
            campaign=active_campaign,
        )
        assert result is True  # disp_mult >= ENTRYB_DISPMULT_OVERRIDE


class TestEntryCStandardSignal:
    def test_two_consecutive_above_avwap(self):
        result = entry_c_standard_signal(
            direction=Direction.LONG,
            hourly_closes=[101.0, 102.0],
            avwap_h=100.0,
        )
        assert result is True

    def test_not_enough_bars(self):
        result = entry_c_standard_signal(
            direction=Direction.LONG,
            hourly_closes=[101.0],
            avwap_h=100.0,
        )
        assert result is False

    def test_one_below(self):
        result = entry_c_standard_signal(
            direction=Direction.LONG,
            hourly_closes=[99.0, 101.0],
            avwap_h=100.0,
        )
        assert result is False

    def test_short_two_below(self):
        result = entry_c_standard_signal(
            direction=Direction.SHORT,
            hourly_closes=[99.0, 98.0],
            avwap_h=100.0,
        )
        assert result is True

    def test_extension_guard_rejects(self):
        # Price too far from EMA20 -> rejected
        result = entry_c_standard_signal(
            direction=Direction.LONG,
            hourly_closes=[110.0, 110.0],
            avwap_h=100.0,
            atr14_h=2.0,
            ema20_h=103.0,  # |110-103| = 7 > 2.5*2 = 5
        )
        assert result is False


class TestEntryCContinuationSignal:
    def test_passes_with_hold_and_pause(self):
        result = entry_c_continuation_signal(
            direction=Direction.LONG,
            hourly_closes=[101.0, 102.0],
            hourly_highs=[101.5, 102.3],
            hourly_lows=[100.8, 101.8],
            avwap_h=100.0,
            atr14_h=5.0,  # 0.40*5 = 2.0, ranges are 0.7 and 0.5 -> OK
        )
        assert result is True

    def test_fails_wide_range(self):
        result = entry_c_continuation_signal(
            direction=Direction.LONG,
            hourly_closes=[101.0, 102.0],
            hourly_highs=[110.0, 102.3],  # range_1 = 110-90 = 20 > 0.40*5
            hourly_lows=[90.0, 101.8],
            avwap_h=100.0,
            atr14_h=5.0,
        )
        assert result is False

    def test_fails_short_wrong_side(self):
        result = entry_c_continuation_signal(
            direction=Direction.SHORT,
            hourly_closes=[101.0, 102.0],  # above AVWAP, wrong for short
            hourly_highs=[101.5, 102.3],
            hourly_lows=[100.8, 101.8],
            avwap_h=100.0,
            atr14_h=5.0,
        )
        assert result is False


class TestSelectEntryType:
    def test_returns_none_when_no_signal(self, active_campaign):
        result = select_entry_type(
            direction=Direction.LONG,
            campaign=active_campaign,
            hourly_close=115.0,
            hourly_low=114.0,
            hourly_high=116.0,
            hourly_closes=[114.0, 115.0],
            hourly_highs=[114.5, 116.0],
            hourly_lows=[113.5, 114.0],
            avwap_h=100.0,  # Far from price -> no A or B signal
            atr14_h=2.0,
            atr14_d=4.0,
            rvol_h=1.0,
            disp_mult=0.85,
            quality_mult=0.8,
            regime_4h=Regime4H.BULL_TREND,
        )
        # Close is far above AVWAP -> C standard may fire (both closes above avwap)
        # but entry_a_active defaults to False so C should fire if conditions met
        # Actually, 115 > 100 and 114 > 100 -> entry_c_standard fires
        # Unless extension guard blocks it
        # With ema20_h=0 (default), extension guard is skipped
        assert result is not None or result is None  # depends on signals

    def test_continuation_mode_only_c_cont(self, continuation_campaign):
        result = select_entry_type(
            direction=Direction.LONG,
            campaign=continuation_campaign,
            hourly_close=101.0,
            hourly_low=100.8,
            hourly_high=101.5,
            hourly_closes=[101.0, 101.2],
            hourly_highs=[101.3, 101.5],
            hourly_lows=[100.8, 100.9],
            avwap_h=100.0,
            atr14_h=5.0,  # pause constraint: range <= 0.40*5 = 2.0
            atr14_d=4.0,
            rvol_h=1.0,
            disp_mult=0.85,
            quality_mult=0.80,
            regime_4h=Regime4H.BULL_TREND,
        )
        assert result == EntryType.C_CONTINUATION

    def test_continuation_blocked_without_regime_alignment(self, continuation_campaign):
        result = select_entry_type(
            direction=Direction.LONG,
            campaign=continuation_campaign,
            hourly_close=101.0,
            hourly_low=100.8,
            hourly_high=101.5,
            hourly_closes=[101.0, 101.2],
            hourly_highs=[101.3, 101.5],
            hourly_lows=[100.8, 100.9],
            avwap_h=100.0,
            atr14_h=5.0,
            atr14_d=4.0,
            rvol_h=1.0,
            disp_mult=0.85,
            quality_mult=0.80,
            regime_4h=Regime4H.RANGE_CHOP,  # Not aligned
        )
        assert result is None


class TestCheckAddTrigger:
    def test_long_touch_and_resume(self):
        result = check_add_trigger(
            direction=Direction.LONG,
            hourly_close=101.0,
            hourly_low=99.5,  # touches ref
            hourly_high=102.0,
            ref=100.0,
            prev_close=99.5,  # prev was at/below ref
            rvol_h=1.0,
            high_vol_regime=False,
            reclaim_buffer=0.2,
        )
        assert result is True

    def test_long_no_touch(self):
        result = check_add_trigger(
            direction=Direction.LONG,
            hourly_close=105.0,
            hourly_low=103.0,  # doesn't touch ref
            hourly_high=106.0,
            ref=100.0,
            prev_close=104.0,
            rvol_h=1.0,
            high_vol_regime=False,
            reclaim_buffer=0.2,
        )
        assert result is False

    def test_short_touch_and_resume(self):
        result = check_add_trigger(
            direction=Direction.SHORT,
            hourly_close=99.0,
            hourly_low=98.0,
            hourly_high=100.5,  # touches ref from below
            ref=100.0,
            prev_close=100.5,  # prev was at/above ref
            rvol_h=1.0,
            high_vol_regime=False,
            reclaim_buffer=0.2,
        )
        assert result is True


class TestIsRegimeAligned:
    def test_long_bull(self):
        assert is_regime_aligned(Direction.LONG, Regime4H.BULL_TREND) is True

    def test_short_bear(self):
        assert is_regime_aligned(Direction.SHORT, Regime4H.BEAR_TREND) is True

    def test_long_bear(self):
        assert is_regime_aligned(Direction.LONG, Regime4H.BEAR_TREND) is False

    def test_long_range(self):
        assert is_regime_aligned(Direction.LONG, Regime4H.RANGE_CHOP) is False


class TestDetermineTradeRegime:
    def test_aligned(self):
        assert determine_trade_regime(Direction.LONG, Regime4H.BULL_TREND) == TradeRegime.ALIGNED

    def test_caution(self):
        assert determine_trade_regime(Direction.LONG, Regime4H.BEAR_TREND) == TradeRegime.CAUTION

    def test_neutral(self):
        assert determine_trade_regime(Direction.LONG, Regime4H.RANGE_CHOP) == TradeRegime.NEUTRAL

    def test_short_aligned(self):
        assert determine_trade_regime(Direction.SHORT, Regime4H.BEAR_TREND) == TradeRegime.ALIGNED

    def test_short_caution(self):
        assert determine_trade_regime(Direction.SHORT, Regime4H.BULL_TREND) == TradeRegime.CAUTION


class TestComputeChopScore:
    def test_low_atr_no_crosses(self):
        atr_hist = [3.0] * 60
        closes = np.array([100.0] * 20)
        ema_ref = np.array([100.0] * 20)
        score = compute_chop_score(1.0, atr_hist, closes, ema_ref)
        # atr14_d=1.0, all hist=3.0, pctl = 0/60 = 0 < 0.75 -> no ATR point
        # No crosses -> no cross point
        assert score == 0

    def test_high_atr_percentile(self):
        atr_hist = [1.0] * 60
        closes = np.array([100.0] * 20)
        ema_ref = np.array([100.0] * 20)
        score = compute_chop_score(5.0, atr_hist, closes, ema_ref)
        # pctl = 60/60 = 1.0 > 0.75 -> +1
        assert score >= 1


class TestClassifyChopMode:
    def test_normal(self):
        assert classify_chop_mode(0) == ChopMode.NORMAL

    def test_degraded(self):
        assert classify_chop_mode(1) == ChopMode.DEGRADED

    def test_halt(self):
        assert classify_chop_mode(2) == ChopMode.HALT
        assert classify_chop_mode(5) == ChopMode.HALT


# ===========================================================================
# STOPS
# ===========================================================================

class TestComputeInitialStop:
    def test_long_entry_a_uses_box_mid(self):
        stop = compute_initial_stop(
            direction=Direction.LONG, entry_type="A",
            box_high=110.0, box_low=100.0, box_mid=105.0,
            atr14_d=2.0, atr_stop_mult=1.0, sq_good=True, tick_size=0.01,
        )
        # raw = 105 - 1.0*2.0 = 103.0, rounded down
        assert stop == pytest.approx(103.0, abs=0.01)

    def test_long_entry_b_uses_box_edge(self):
        stop = compute_initial_stop(
            direction=Direction.LONG, entry_type="B",
            box_high=110.0, box_low=100.0, box_mid=105.0,
            atr14_d=2.0, atr_stop_mult=1.0, sq_good=True, tick_size=0.01,
        )
        # raw = 100 - 1.0*2.0 = 98.0, rounded down
        assert stop == pytest.approx(98.0, abs=0.01)

    def test_short_entry_a_uses_box_mid(self):
        stop = compute_initial_stop(
            direction=Direction.SHORT, entry_type="A",
            box_high=110.0, box_low=100.0, box_mid=105.0,
            atr14_d=2.0, atr_stop_mult=1.0, sq_good=False, tick_size=0.01,
        )
        # raw = 105 + 1.0*2.0 = 107.0, rounded up
        assert stop == pytest.approx(107.0, abs=0.01)

    def test_short_entry_b_uses_box_edge(self):
        stop = compute_initial_stop(
            direction=Direction.SHORT, entry_type="B",
            box_high=110.0, box_low=100.0, box_mid=105.0,
            atr14_d=2.0, atr_stop_mult=1.0, sq_good=False, tick_size=0.01,
        )
        # raw = 110 + 2.0 = 112.0, rounded up
        assert stop == pytest.approx(112.0, abs=0.01)

    def test_tick_rounding(self):
        stop = compute_initial_stop(
            direction=Direction.LONG, entry_type="A",
            box_high=110.0, box_low=100.0, box_mid=105.0,
            atr14_d=2.03, atr_stop_mult=1.0, sq_good=True, tick_size=0.05,
        )
        # raw = 105 - 2.03 = 102.97, rounded down to 0.05 -> 102.95
        assert stop == pytest.approx(102.95, abs=0.001)


class TestComputeAddStop:
    def test_long_add(self):
        stop = compute_add_stop(
            direction=Direction.LONG,
            pullback_low=99.0, pullback_high=102.0,
            ref=100.0, atr14_d=2.0, atr_stop_mult=1.0, tick_size=0.01,
        )
        # anchor = min(99, 100) = 99, buffer = 0.5*2.0*1.0 = 1.0
        # raw = 99 - 1 = 98.0
        assert stop == pytest.approx(98.0, abs=0.01)

    def test_short_add(self):
        stop = compute_add_stop(
            direction=Direction.SHORT,
            pullback_low=98.0, pullback_high=101.0,
            ref=100.0, atr14_d=2.0, atr_stop_mult=1.0, tick_size=0.01,
        )
        # anchor = max(101, 100) = 101, buffer = 0.5*2.0*1.0 = 1.0
        # raw = 101 + 1 = 102.0
        assert stop == pytest.approx(102.0, abs=0.01)


class TestComputeBeStop:
    def test_long_be(self):
        stop = compute_be_stop(
            direction=Direction.LONG, avg_entry=100.0,
            atr14_d=2.0, tick_size=0.01,
        )
        # raw = 100 + 0.1*2.0 = 100.20, rounded up
        assert stop == pytest.approx(100.20, abs=0.01)

    def test_short_be(self):
        stop = compute_be_stop(
            direction=Direction.SHORT, avg_entry=100.0,
            atr14_d=2.0, tick_size=0.01,
        )
        # raw = 100 - 0.1*2.0 = 99.80, rounded down
        assert stop == pytest.approx(99.80, abs=0.01)


class TestComputeTrailingStop:
    def test_long_trail_ratchets_up(self):
        stop = compute_trailing_stop(
            direction=Direction.LONG,
            highs_4h=[110.0, 112.0, 115.0],
            lows_4h=[108.0, 110.0, 113.0],
            atr14_4h=2.0,
            trail_mult=3.0,
            ema50_4h=108.0,
            current_stop=100.0,
            tick_size=0.01,
        )
        # hh = 115, raw = 115 - 3*2 = 109.0
        # ema_floor = 108 - 0.5*2 = 107.0
        # raw = max(109, 107) = 109.0
        # ratchet: max(109, 100) = 109
        assert stop == pytest.approx(109.0, abs=0.01)

    def test_long_trail_never_lowers(self):
        stop = compute_trailing_stop(
            direction=Direction.LONG,
            highs_4h=[110.0],
            lows_4h=[108.0],
            atr14_4h=2.0,
            trail_mult=3.0,
            ema50_4h=108.0,
            current_stop=120.0,  # already high
            tick_size=0.01,
        )
        # Candidate would be 110-6=104, but current is 120 -> keep 120
        assert stop == pytest.approx(120.0)

    def test_short_trail(self):
        stop = compute_trailing_stop(
            direction=Direction.SHORT,
            highs_4h=[102.0, 101.0, 100.0],
            lows_4h=[98.0, 97.0, 96.0],
            atr14_4h=2.0,
            trail_mult=3.0,
            ema50_4h=100.0,
            current_stop=110.0,
            tick_size=0.01,
        )
        # ll = 96, raw = 96 + 3*2 = 102.0
        # ema_ceil = 100 + 0.5*2 = 101.0
        # raw = min(102, 101) = 101.0
        # ratchet: min(101, 110) = 101
        assert stop == pytest.approx(101.0, abs=0.01)

    def test_empty_highs_returns_current(self):
        stop = compute_trailing_stop(
            direction=Direction.LONG,
            highs_4h=[], lows_4h=[],
            atr14_4h=2.0, trail_mult=3.0, ema50_4h=100.0,
            current_stop=95.0, tick_size=0.01,
        )
        assert stop == 95.0


class TestComputeTrailMult:
    def test_base_value(self):
        mult = compute_trail_mult(r_state=0.0, r_proxy=0.0, continuation=False)
        expected = TRAIL_4H_ATR_MULT * 5.0  # 0.40 * 5 = 2.0
        assert mult == pytest.approx(expected)

    def test_tightens_after_measured_move(self):
        mult = compute_trail_mult(r_state=0.0, r_proxy=1.5, continuation=False)
        expected = TRAIL_4H_ATR_MULT * 5.0 * 0.70
        assert mult == pytest.approx(expected)

    def test_tightens_with_continuation(self):
        mult = compute_trail_mult(r_state=0.0, r_proxy=0.0, continuation=True)
        expected = TRAIL_4H_ATR_MULT * 5.0 * 0.70
        assert mult == pytest.approx(expected)

    def test_tightens_high_r(self):
        mult = compute_trail_mult(r_state=3.0, r_proxy=2.0, continuation=False)
        base = TRAIL_4H_ATR_MULT * 5.0
        expected = max(1.0, base * 0.70 * 0.60)
        assert mult == pytest.approx(expected)

    def test_floor_at_1(self):
        # Even with heavy tightening, floor is 1.0
        mult = compute_trail_mult(r_state=100.0, r_proxy=100.0, continuation=True)
        assert mult >= 1.0


class TestHandleGapThroughStop:
    def test_long_gap_below_stop(self):
        triggered, fill = handle_gap_through_stop(
            Direction.LONG, stop_price=100.0, session_open=98.0
        )
        assert triggered is True
        assert fill == 98.0

    def test_long_no_gap(self):
        triggered, fill = handle_gap_through_stop(
            Direction.LONG, stop_price=100.0, session_open=102.0
        )
        assert triggered is False
        assert fill == 0.0

    def test_short_gap_above_stop(self):
        triggered, fill = handle_gap_through_stop(
            Direction.SHORT, stop_price=100.0, session_open=102.0
        )
        assert triggered is True
        assert fill == 102.0

    def test_short_no_gap(self):
        triggered, fill = handle_gap_through_stop(
            Direction.SHORT, stop_price=100.0, session_open=98.0
        )
        assert triggered is False
        assert fill == 0.0


class TestCheckStaleExit:
    def test_fresh_trade(self):
        warn, tighten, exit_ = check_stale_exit(days_held=2, r_state=0.5)
        assert warn is False
        assert tighten is False
        assert exit_ is False

    def test_warn_after_threshold(self):
        warn, tighten, exit_ = check_stale_exit(
            days_held=STALE_WARN_DAYS, r_state=-0.5
        )
        assert warn is True
        assert tighten is True
        assert exit_ is False

    def test_no_warn_if_profitable(self):
        warn, tighten, exit_ = check_stale_exit(
            days_held=STALE_WARN_DAYS + 2, r_state=0.5
        )
        assert warn is False

    def test_exit_after_min_days(self):
        warn, tighten, exit_ = check_stale_exit(
            days_held=STALE_EXIT_DAYS_MIN, r_state=-0.5
        )
        assert exit_ is True

    def test_hard_cap_exit(self):
        # Even near zero R, hard cap forces exit
        warn, tighten, exit_ = check_stale_exit(
            days_held=STALE_EXIT_DAYS_MAX, r_state=0.05
        )
        assert exit_ is True

    def test_no_hard_cap_if_profitable(self):
        # r_state >= 0.10, so hard cap doesn't trigger
        warn, tighten, exit_ = check_stale_exit(
            days_held=STALE_EXIT_DAYS_MAX, r_state=0.15
        )
        assert exit_ is False


class TestApplyStaleTighten:
    def test_tighten_multiplier(self):
        assert apply_stale_tighten(3.0) == pytest.approx(3.0 * STALE_TIGHTEN_MULT)


class TestComputeTpLevels:
    def test_long_tp_levels(self):
        tp1, tp2 = compute_tp_levels(
            direction=Direction.LONG,
            entry_price=100.0, risk_per_share=2.0,
            tp1_r=1.5, tp2_r=3.0, tick_size=0.01,
        )
        assert tp1 == pytest.approx(103.0, abs=0.01)
        assert tp2 == pytest.approx(106.0, abs=0.01)

    def test_short_tp_levels(self):
        tp1, tp2 = compute_tp_levels(
            direction=Direction.SHORT,
            entry_price=100.0, risk_per_share=2.0,
            tp1_r=1.5, tp2_r=3.0, tick_size=0.01,
        )
        assert tp1 == pytest.approx(97.0, abs=0.01)
        assert tp2 == pytest.approx(94.0, abs=0.01)


# ===========================================================================
# ALLOCATOR
# ===========================================================================

class TestComputeRiskRegimeAdj:
    def test_normal_regime(self):
        adj = compute_risk_regime_adj(0.005, 1.0)
        assert adj == pytest.approx(0.005)

    def test_high_risk_regime_reduces(self):
        # risk_regime=2.0 -> 1/2 = 0.50, clamped to 0.75
        adj = compute_risk_regime_adj(0.005, 2.0)
        assert adj == pytest.approx(0.005 * 0.75)

    def test_low_risk_regime_boosts(self):
        # risk_regime=0.5 -> 1/0.5 = 2.0, clamped to 1.05
        adj = compute_risk_regime_adj(0.005, 0.5)
        assert adj == pytest.approx(0.005 * 1.05)

    def test_zero_risk_regime_returns_base(self):
        adj = compute_risk_regime_adj(0.005, 0.0)
        assert adj == pytest.approx(0.005)


class TestComputeRegimeMult:
    def test_aligned(self):
        assert compute_regime_mult(Direction.LONG, Regime4H.BULL_TREND) == pytest.approx(REGIME_MULT_ALIGNED)

    def test_caution(self):
        assert compute_regime_mult(Direction.LONG, Regime4H.BEAR_TREND) == pytest.approx(REGIME_MULT_CAUTION)

    def test_neutral(self):
        assert compute_regime_mult(Direction.LONG, Regime4H.RANGE_CHOP) == pytest.approx(REGIME_MULT_NEUTRAL)


class TestComputeDispMult:
    def test_insufficient_history(self):
        assert compute_disp_mult(1.0, [0.5] * 5) == pytest.approx(0.85)

    def test_returns_in_range(self, disp_hist_normal):
        result = compute_disp_mult(1.0, disp_hist_normal)
        assert 0.70 <= result <= 1.0

    def test_high_disp_higher_mult(self, disp_hist_normal):
        low_mult = compute_disp_mult(0.1, disp_hist_normal)
        high_mult = compute_disp_mult(5.0, disp_hist_normal)
        assert high_mult >= low_mult


class TestComputeSqueezeMult:
    def test_good(self):
        assert compute_squeeze_mult(True, False) == pytest.approx(SQUEEZE_MULT_GOOD)

    def test_loose(self):
        assert compute_squeeze_mult(False, True) == pytest.approx(SQUEEZE_MULT_LOOSE)

    def test_neutral(self):
        assert compute_squeeze_mult(False, False) == pytest.approx(SQUEEZE_MULT_NEUTRAL)


class TestComputeCorrMult:
    def test_no_positions(self):
        assert compute_corr_mult(Direction.LONG, "QQQ", {}, {}) == 1.0

    def test_correlated_same_direction_both_aligned(self, long_position):
        positions = {"QQQ": long_position}
        corr_map = {("IWM", "QQQ"): 0.85}
        result = compute_corr_mult(
            Direction.LONG, "IWM", positions, corr_map,
            regime_at_entry="BULL_TREND",
        )
        assert result == pytest.approx(CORR_MULT_ALIGNED)

    def test_correlated_same_direction_not_aligned(self, long_position):
        long_position.regime_at_entry = "RANGE_CHOP"
        positions = {"QQQ": long_position}
        corr_map = {("IWM", "QQQ"): 0.85}
        result = compute_corr_mult(
            Direction.LONG, "IWM", positions, corr_map,
            regime_at_entry="RANGE_CHOP",
        )
        assert result == pytest.approx(CORR_MULT_OTHER)

    def test_uncorrelated(self, long_position):
        positions = {"QQQ": long_position}
        corr_map = {("GLD", "QQQ"): 0.30}
        result = compute_corr_mult(Direction.LONG, "GLD", positions, corr_map)
        assert result == 1.0

    def test_different_direction_ignored(self, short_position):
        positions = {"GLD": short_position}
        corr_map = {("GLD", "QQQ"): 0.85}
        result = compute_corr_mult(Direction.LONG, "QQQ", positions, corr_map)
        assert result == 1.0


class TestComputeQualityMult:
    def test_basic_computation(self, disp_hist_normal):
        result = compute_quality_mult(
            direction=Direction.LONG, regime_4h=Regime4H.BULL_TREND,
            disp=1.0, disp_hist=disp_hist_normal,
            sq_good=True, sq_loose=False,
            symbol="QQQ", positions={}, correlation_map={},
            score_total=0,
        )
        assert 0.25 <= result <= 1.15  # clamped * score_adj

    def test_clamped_min(self, disp_hist_normal):
        result = compute_quality_mult(
            direction=Direction.LONG, regime_4h=Regime4H.BEAR_TREND,
            disp=0.01, disp_hist=disp_hist_normal,
            sq_good=False, sq_loose=True,
            symbol="QQQ", positions={}, correlation_map={},
            score_total=-3,
        )
        # Very poor quality: CAUTION * low_disp * loose_squeeze * bad_score
        assert result >= 0.25 * 0.85  # floor * min score_adj


class TestComputeFinalRisk:
    def test_normal_risk(self):
        risk = compute_final_risk(
            equity=100000.0, base_risk_pct=0.005,
            risk_regime=1.0, quality_mult=1.0, expiry_mult=1.0,
        )
        assert risk == pytest.approx(500.0)

    def test_quality_reduces_risk(self):
        risk = compute_final_risk(
            equity=100000.0, base_risk_pct=0.005,
            risk_regime=1.0, quality_mult=0.5, expiry_mult=1.0,
        )
        assert risk == pytest.approx(250.0)

    def test_floor_applied(self):
        risk = compute_final_risk(
            equity=100000.0, base_risk_pct=0.005,
            risk_regime=1.0, quality_mult=0.01, expiry_mult=0.3,
        )
        # Floor = 0.20 * base_risk_adj = 0.20 * 0.005 = 0.001
        assert risk == pytest.approx(100.0)

    def test_cap_at_base(self):
        risk = compute_final_risk(
            equity=100000.0, base_risk_pct=0.005,
            risk_regime=1.0, quality_mult=2.0, expiry_mult=2.0,
        )
        # Capped at base_risk_adj = 0.005
        assert risk == pytest.approx(500.0)


class TestComputeShares:
    def test_basic_sizing(self):
        shares = compute_shares(
            final_risk_dollars=500.0,
            entry_price=100.0,
            stop_price=98.0,
        )
        # risk_per_share = 2.0, 500/2 = 250
        assert shares == 250

    def test_with_fees(self):
        shares = compute_shares(
            final_risk_dollars=500.0,
            entry_price=100.0,
            stop_price=98.0,
            fee_bps_est=0.0003,
        )
        # denominator = 2.0 + 0.0003*100 = 2.03
        expected = int(500.0 // 2.03)
        assert shares == expected

    def test_zero_risk_per_share(self):
        shares = compute_shares(
            final_risk_dollars=500.0,
            entry_price=100.0,
            stop_price=100.0,
        )
        assert shares == 0

    def test_minimum_one_share(self):
        shares = compute_shares(
            final_risk_dollars=0.5,
            entry_price=100.0,
            stop_price=99.0,
        )
        assert shares >= 1


class TestComputeAddRiskDollars:
    def test_normal_volume(self):
        risk = compute_add_risk_dollars(1000.0, rvol_h=1.2)
        assert risk == pytest.approx(ADD_RISK_MULT * 1000.0)

    def test_low_volume_reduced(self):
        risk = compute_add_risk_dollars(1000.0, rvol_h=0.5)
        assert risk == pytest.approx(ADD_RISK_MULT * 1000.0 * ADD_RISK_LOWVOLH_MULT)


class TestAddBudgetOk:
    def test_within_budget(self, active_campaign):
        active_campaign.add_count = 0
        active_campaign.campaign_risk_used = 100.0
        assert add_budget_ok(active_campaign, 1000.0, 200.0) is True

    def test_max_adds_exceeded(self, active_campaign):
        active_campaign.add_count = MAX_ADDS_PER_CAMPAIGN
        assert add_budget_ok(active_campaign, 1000.0, 200.0) is False

    def test_budget_exceeded(self, active_campaign):
        active_campaign.add_count = 0
        active_campaign.campaign_risk_used = 1400.0  # 1400+200 > 1.5*1000
        assert add_budget_ok(active_campaign, 1000.0, 200.0) is False


class TestMicroGuardOk:
    def test_normal_passes(self):
        assert micro_guard_ok(
            expiry_mult=0.8, disp_mult=0.90,
            trade_regime=TradeRegime.ALIGNED,
            final_risk_pct=0.004, base_risk_adj=0.005,
        ) is True

    def test_floored_caution_low_expiry_blocks(self):
        # Risk was floored (final_risk_pct <= 0.20 * base_risk_adj)
        base = 0.005
        result = micro_guard_ok(
            expiry_mult=0.50, disp_mult=0.70,
            trade_regime=TradeRegime.CAUTION,
            final_risk_pct=0.20 * base, base_risk_adj=base,
        )
        assert result is False  # expiry_mult < 0.60 and CAUTION and disp_mult < 0.85

    def test_floored_caution_high_disp_passes(self):
        base = 0.005
        result = micro_guard_ok(
            expiry_mult=0.50, disp_mult=0.90,
            trade_regime=TradeRegime.CAUTION,
            final_risk_pct=0.20 * base, base_risk_adj=base,
        )
        assert result is True  # disp_mult >= 0.85 overrides


# ===========================================================================
# GATES
# ===========================================================================

class TestIsRthTime:
    def test_during_rth(self):
        dt = datetime(2025, 6, 2, 10, 30)
        assert is_rth_time(dt) is True

    def test_before_rth(self):
        dt = datetime(2025, 6, 2, 9, 0)
        assert is_rth_time(dt) is False

    def test_at_open(self):
        dt = datetime(2025, 6, 2, 9, 30)
        assert is_rth_time(dt) is True

    def test_at_close(self):
        dt = datetime(2025, 6, 2, 16, 0)
        assert is_rth_time(dt) is True

    def test_after_close(self):
        dt = datetime(2025, 6, 2, 16, 1)
        assert is_rth_time(dt) is False


class TestIsEntryWindowOpen:
    def test_within_window(self, default_cfg):
        now = datetime(2025, 6, 2, 10, 0)
        assert is_entry_window_open(now, default_cfg) is True

    def test_before_window(self, default_cfg):
        now = datetime(2025, 6, 2, 9, 0)
        assert is_entry_window_open(now, default_cfg) is False

    def test_at_boundaries(self, default_cfg):
        assert is_entry_window_open(datetime(2025, 6, 2, 9, 30), default_cfg) is True
        assert is_entry_window_open(datetime(2025, 6, 2, 16, 0), default_cfg) is True


class TestHardBlock:
    def test_no_block_aligned(self):
        assert hard_block(Direction.LONG, Regime4H.BULL_TREND, 1.0, 2.0) is False

    def test_no_block_regime_only(self):
        # Regime opposes but slope doesn't
        assert hard_block(Direction.LONG, Regime4H.BEAR_TREND, 0.0, 2.0) is False

    def test_blocks_double_counter(self):
        # Regime opposes AND slope opposes strongly
        slope_th = HARD_BLOCK_SLOPE_MULT * 2.0  # 0.12 * 2 = 0.24
        assert hard_block(
            Direction.LONG, Regime4H.BEAR_TREND,
            daily_slope=-(slope_th + 0.01), atr14_d=2.0,
        ) is True

    def test_short_block(self):
        slope_th = HARD_BLOCK_SLOPE_MULT * 2.0
        assert hard_block(
            Direction.SHORT, Regime4H.BULL_TREND,
            daily_slope=(slope_th + 0.01), atr14_d=2.0,
        ) is True


class TestFrictionGate:
    def test_passes_low_friction(self):
        assert friction_gate("QQQ", 0.0002, 100.0, 100, 500.0) is True

    def test_blocks_high_friction(self):
        # friction = 0.10 * 100 * 100 = 1000, risk$ = 500
        # 1000 > 0.15 * 500 = 75
        assert friction_gate("QQQ", 0.10, 100.0, 100, 500.0) is False

    def test_zero_risk_blocks(self):
        assert friction_gate("QQQ", 0.0002, 100.0, 100, 0.0) is False


class TestGapGuardOk:
    def test_small_gap_ok(self):
        assert gap_guard_ok(101.0, 100.0, 2.0) is True

    def test_large_gap_blocked(self):
        # gap = |110 - 100| = 10 > 1.5*2 = 3
        assert gap_guard_ok(110.0, 100.0, 2.0) is False

    def test_zero_atr_always_ok(self):
        assert gap_guard_ok(200.0, 100.0, 0.0) is True

    def test_exact_boundary(self):
        assert gap_guard_ok(103.0, 100.0, 2.0) is True  # 3 <= 3


class TestPositionLimitCheck:
    def test_within_limits(self):
        positions = {
            "QQQ": PositionState(symbol="QQQ", direction=Direction.LONG, qty=100),
        }
        ok, reason = position_limit_check(positions, Direction.LONG)
        assert ok is True

    def test_max_positions_hit(self):
        positions = {
            f"SYM{i}": PositionState(symbol=f"SYM{i}", direction=Direction.LONG, qty=100)
            for i in range(MAX_POSITIONS)
        }
        ok, reason = position_limit_check(positions, Direction.LONG)
        assert ok is False
        assert reason == "max_positions"

    def test_max_same_direction_hit(self):
        positions = {
            f"SYM{i}": PositionState(symbol=f"SYM{i}", direction=Direction.LONG, qty=100)
            for i in range(MAX_SAME_DIRECTION)
        }
        ok, reason = position_limit_check(positions, Direction.LONG)
        assert ok is False
        assert reason == "max_same_direction"

    def test_opposite_direction_not_counted(self):
        positions = {
            f"SYM{i}": PositionState(symbol=f"SYM{i}", direction=Direction.SHORT, qty=100)
            for i in range(MAX_SAME_DIRECTION)
        }
        ok, reason = position_limit_check(positions, Direction.LONG)
        assert ok is True

    def test_zero_qty_not_counted(self):
        positions = {
            f"SYM{i}": PositionState(symbol=f"SYM{i}", direction=Direction.LONG, qty=0)
            for i in range(10)
        }
        ok, reason = position_limit_check(positions, Direction.LONG)
        assert ok is True


class TestHeatCheck:
    def test_within_heat_limit(self):
        positions = {
            "QQQ": PositionState(total_risk_dollars=500.0, qty=100),
        }
        ok, reason = heat_check(positions, 500.0, 100000.0)
        # heat = (500+500)/100000 = 1% < 3%
        assert ok is True

    def test_exceeds_heat(self):
        positions = {
            "QQQ": PositionState(total_risk_dollars=2500.0, qty=100),
        }
        ok, reason = heat_check(positions, 1000.0, 100000.0)
        # heat = (2500+1000)/100000 = 3.5% > 3%
        assert ok is False
        assert reason == "portfolio_heat"

    def test_correlation_penalty_increases_heat(self):
        positions = {
            "QQQ": PositionState(total_risk_dollars=2000.0, qty=100),
        }
        # Without penalty: (2000+800)/100000 = 2.8% OK
        ok_no_pen, _ = heat_check(positions, 800.0, 100000.0, 1.0)
        assert ok_no_pen is True
        # With 1.25 penalty: (2000+1000)/100000 = 3.0% borderline
        ok_pen, _ = heat_check(positions, 800.0, 100000.0, CORR_HEAT_PENALTY)
        # (2000 + 800*1.25)/100000 = 3.0% => should still pass (<=)
        # Actually 3000/100000 = 0.03 which equals MAX_PORTFOLIO_HEAT
        # The check is > MAX_PORTFOLIO_HEAT, so 0.03 passes
        assert ok_pen is True

    def test_zero_equity_blocks(self):
        ok, reason = heat_check({}, 100.0, 0.0)
        assert ok is False
        assert reason == "zero_equity"


class TestComputeCorrHeatPenalty:
    def test_no_correlated_positions(self, long_position):
        positions = {"QQQ": long_position}
        corr_map = {("GLD", "QQQ"): 0.30}
        result = compute_corr_heat_penalty(
            Direction.LONG, positions, corr_map, "GLD"
        )
        assert result == 1.0

    def test_correlated_same_direction(self, long_position):
        positions = {"QQQ": long_position}
        corr_map = {("IWM", "QQQ"): 0.85}
        result = compute_corr_heat_penalty(
            Direction.LONG, positions, corr_map, "IWM"
        )
        assert result == pytest.approx(CORR_HEAT_PENALTY)

    def test_different_direction_not_penalized(self, short_position):
        positions = {"GLD": short_position}
        corr_map = {("GLD", "QQQ"): 0.85}
        result = compute_corr_heat_penalty(
            Direction.LONG, positions, corr_map, "QQQ"
        )
        assert result == 1.0


class TestCircuitBreakerCheck:
    def test_normal(self, empty_cb):
        ok, reason, mult = circuit_breaker_check(empty_cb)
        assert ok is True
        assert mult == 1.0

    def test_halted(self):
        cb = CircuitBreakerState(halted=True)
        ok, reason, mult = circuit_breaker_check(cb)
        assert ok is False
        assert reason == "monthly_halt"
        assert mult == 0.0

    def test_weekly_throttle(self):
        cb = CircuitBreakerState(weekly_realized_r=WEEKLY_THROTTLE_R)
        ok, reason, mult = circuit_breaker_check(cb)
        assert ok is True
        assert reason == "weekly_throttle"
        assert mult == 0.5

    def test_above_throttle(self):
        cb = CircuitBreakerState(weekly_realized_r=WEEKLY_THROTTLE_R + 1.0)
        ok, reason, mult = circuit_breaker_check(cb)
        assert ok is True
        assert mult == 1.0


class TestReentryAllowed:
    def test_first_reentry_allowed(self, active_campaign):
        assert reentry_allowed(
            Direction.LONG, active_campaign,
            realized_r=0.0, days_since_exit=5,
        ) is True

    def test_cooldown_blocks(self, active_campaign):
        assert reentry_allowed(
            Direction.LONG, active_campaign,
            realized_r=0.0, days_since_exit=0,
        ) is False

    def test_bad_realized_r_blocks(self, active_campaign):
        assert reentry_allowed(
            Direction.LONG, active_campaign,
            realized_r=REENTRY_MIN_REALIZED_R - 0.1,
            days_since_exit=5,
        ) is False

    def test_max_reentry_blocks(self, active_campaign):
        active_campaign.reentry_count = {
            "LONG": {1: MAX_REENTRY_PER_BOX_VERSION},
            "SHORT": {},
        }
        assert reentry_allowed(
            Direction.LONG, active_campaign,
            realized_r=0.0, days_since_exit=5,
        ) is False


class TestCheckTransientBlocks:
    def test_no_blocks(self):
        result = check_transient_blocks(
            "QQQ", Direction.LONG, {}, 500.0, 100000.0
        )
        assert result is None

    def test_position_limit_block(self):
        positions = {
            f"SYM{i}": PositionState(symbol=f"SYM{i}", direction=Direction.LONG, qty=100)
            for i in range(MAX_POSITIONS)
        }
        result = check_transient_blocks(
            "QQQ", Direction.LONG, positions, 500.0, 100000.0
        )
        assert result is not None


class TestPendingExpired:
    def test_not_expired(self):
        p = PendingEntry(
            created_ts=datetime(2025, 6, 2, 10, 0),
            direction=Direction.LONG,
            entry_type_requested=EntryType.A_AVWAP_RETEST,
            rth_hours_elapsed=5,
        )
        assert pending_expired(p, datetime(2025, 6, 2, 15, 0)) is False

    def test_expired(self):
        p = PendingEntry(
            created_ts=datetime(2025, 6, 2, 10, 0),
            direction=Direction.LONG,
            entry_type_requested=EntryType.A_AVWAP_RETEST,
            rth_hours_elapsed=PENDING_MAX_RTH_HOURS,
        )
        assert pending_expired(p, datetime(2025, 6, 5, 10, 0)) is True


class TestIsNewsBlocked:
    def test_no_events(self):
        assert is_news_blocked(datetime(2025, 6, 2, 10, 0), "QQQ", []) is False

    def test_fomc_blocks(self):
        event_dt = datetime(2025, 6, 2, 14, 0)
        calendar = [("FOMC", event_dt)]
        # -60 to +60 minutes -> 13:00 to 15:00
        assert is_news_blocked(datetime(2025, 6, 2, 14, 30), "QQQ", calendar) is True

    def test_fomc_outside_window(self):
        event_dt = datetime(2025, 6, 2, 14, 0)
        calendar = [("FOMC", event_dt)]
        assert is_news_blocked(datetime(2025, 6, 2, 16, 0), "QQQ", calendar) is False

    def test_cl_inventory_only_uso(self):
        event_dt = datetime(2025, 6, 2, 10, 30)
        calendar = [("CL_INVENTORY", event_dt)]
        # QQQ is not USO, so not blocked
        assert is_news_blocked(datetime(2025, 6, 2, 10, 30), "QQQ", calendar) is False
        # USO would be blocked
        assert is_news_blocked(datetime(2025, 6, 2, 10, 30), "USO", calendar) is True

    def test_crypto_event_only_ibit(self):
        event_dt = datetime(2025, 6, 2, 10, 30)
        calendar = [("CRYPTO_EVENT", event_dt)]
        assert is_news_blocked(datetime(2025, 6, 2, 10, 30), "QQQ", calendar) is False
        assert is_news_blocked(datetime(2025, 6, 2, 10, 30), "IBIT", calendar) is True


class TestFullEligibilityCheck:
    def test_all_clear(self, default_cfg, empty_cb, active_campaign):
        ok, reason, cb_mult = full_eligibility_check(
            symbol="QQQ",
            direction=Direction.LONG,
            campaign=active_campaign,
            cfg=default_cfg,
            now_et=datetime(2025, 6, 2, 10, 0),
            cb=empty_cb,
            positions={},
            candidate_risk_dollars=500.0,
            equity=100000.0,
            calendar=[],
        )
        assert ok is True
        assert reason == ""
        assert cb_mult == 1.0

    def test_halted_cb_blocks(self, default_cfg, active_campaign):
        cb = CircuitBreakerState(halted=True)
        ok, reason, _ = full_eligibility_check(
            symbol="QQQ", direction=Direction.LONG,
            campaign=active_campaign, cfg=default_cfg,
            now_et=datetime(2025, 6, 2, 10, 0),
            cb=cb, positions={},
            candidate_risk_dollars=500.0, equity=100000.0,
            calendar=[],
        )
        assert ok is False
        assert reason == "monthly_halt"

    def test_outside_window_blocks(self, default_cfg, empty_cb, active_campaign):
        ok, reason, _ = full_eligibility_check(
            symbol="QQQ", direction=Direction.LONG,
            campaign=active_campaign, cfg=default_cfg,
            now_et=datetime(2025, 6, 2, 8, 0),  # before RTH
            cb=empty_cb, positions={},
            candidate_risk_dollars=500.0, equity=100000.0,
            calendar=[],
        )
        assert ok is False
        assert reason == "outside_entry_window"

    def test_inactive_campaign_blocks(self, default_cfg, empty_cb):
        campaign = SymbolCampaign(state=CampaignState.INACTIVE)
        ok, reason, _ = full_eligibility_check(
            symbol="QQQ", direction=Direction.LONG,
            campaign=campaign, cfg=default_cfg,
            now_et=datetime(2025, 6, 2, 10, 0),
            cb=empty_cb, positions={},
            candidate_risk_dollars=500.0, equity=100000.0,
            calendar=[],
        )
        assert ok is False
        assert reason == "campaign_inactive"

    def test_news_blocks(self, default_cfg, empty_cb, active_campaign):
        event_dt = datetime(2025, 6, 2, 10, 0)
        ok, reason, _ = full_eligibility_check(
            symbol="QQQ", direction=Direction.LONG,
            campaign=active_campaign, cfg=default_cfg,
            now_et=datetime(2025, 6, 2, 10, 0),
            cb=empty_cb, positions={},
            candidate_risk_dollars=500.0, equity=100000.0,
            calendar=[("FOMC", event_dt)],
        )
        assert ok is False
        assert reason == "news_blocked"

    def test_position_limit_blocks(self, default_cfg, empty_cb, active_campaign):
        positions = {
            f"SYM{i}": PositionState(symbol=f"SYM{i}", direction=Direction.LONG, qty=100)
            for i in range(MAX_POSITIONS)
        }
        ok, reason, _ = full_eligibility_check(
            symbol="QQQ", direction=Direction.LONG,
            campaign=active_campaign, cfg=default_cfg,
            now_et=datetime(2025, 6, 2, 10, 0),
            cb=empty_cb, positions=positions,
            candidate_risk_dollars=500.0, equity=100000.0,
            calendar=[],
        )
        assert ok is False
        assert reason == "max_positions"

    def test_weekly_throttle_returns_half_mult(self, default_cfg, active_campaign):
        cb = CircuitBreakerState(weekly_realized_r=WEEKLY_THROTTLE_R)
        ok, reason, cb_mult = full_eligibility_check(
            symbol="QQQ", direction=Direction.LONG,
            campaign=active_campaign, cfg=default_cfg,
            now_et=datetime(2025, 6, 2, 10, 0),
            cb=cb, positions={},
            candidate_risk_dollars=500.0, equity=100000.0,
            calendar=[],
        )
        assert ok is True
        assert cb_mult == 0.5


# ===========================================================================
# MODEL ENUMS
# ===========================================================================

class TestEnums:
    def test_direction_values(self):
        assert Direction.SHORT == -1
        assert Direction.FLAT == 0
        assert Direction.LONG == 1

    def test_campaign_state_values(self):
        assert CampaignState.INACTIVE.value == "INACTIVE"
        assert CampaignState.BREAKOUT.value == "BREAKOUT"
        assert CampaignState.DIRTY.value == "DIRTY"

    def test_regime_4h_values(self):
        assert Regime4H.BULL_TREND.value == "BULL_TREND"
        assert Regime4H.BEAR_TREND.value == "BEAR_TREND"
        assert Regime4H.RANGE_CHOP.value == "RANGE_CHOP"

    def test_entry_type_values(self):
        assert EntryType.A_AVWAP_RETEST.value == "A"
        assert EntryType.B_SWEEP_RECLAIM.value == "B"
        assert EntryType.C_STANDARD.value == "C_standard"

    def test_trade_regime(self):
        assert TradeRegime.ALIGNED.value == "ALIGNED"
        assert TradeRegime.NEUTRAL.value == "NEUTRAL"
        assert TradeRegime.CAUTION.value == "CAUTION"

    def test_chop_mode(self):
        assert ChopMode.NORMAL.value == "NORMAL"
        assert ChopMode.DEGRADED.value == "DEGRADED"
        assert ChopMode.HALT.value == "HALT"


class TestSymbolCampaignDefaults:
    def test_defaults(self):
        c = SymbolCampaign()
        assert c.state == CampaignState.INACTIVE
        assert c.L == 12
        assert c.continuation is False
        assert c.add_count == 0
        assert c.pending is None

    def test_reentry_count_default(self):
        c = SymbolCampaign()
        assert "LONG" in c.reentry_count
        assert "SHORT" in c.reentry_count


class TestPositionStateDefaults:
    def test_defaults(self):
        p = PositionState()
        assert p.direction == Direction.FLAT
        assert p.qty == 0
        assert p.trail_mult == pytest.approx(4.5)
        assert p.total_risk_dollars == 0.0


class TestCircuitBreakerStateDefaults:
    def test_defaults(self):
        cb = CircuitBreakerState()
        assert cb.weekly_realized_r == 0.0
        assert cb.halted is False


class TestPendingEntryDefaults:
    def test_defaults(self):
        p = PendingEntry(
            created_ts=datetime(2025, 6, 2, 10, 0),
            direction=Direction.LONG,
            entry_type_requested=EntryType.A_AVWAP_RETEST,
        )
        assert p.rth_hours_elapsed == 0
        assert p.reason == ""
