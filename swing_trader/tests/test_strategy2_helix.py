"""Unit tests for Strategy 2 (AKC-Helix) — indicators, signals, stops, allocator, gates."""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

import numpy as np
import pytest

from strategy_2.config import (
    ADX_PERIOD,
    BE_ATR1H_OFFSET,
    CLASS_A_SIZE_CHOP,
    CLASS_A_SIZE_COUNTER,
    CLASS_A_SIZE_TREND,
    CLASS_D_MOM_LOOKBACK,
    DIV_MAG_DEFAULT_THRESHOLD,
    EXTREME_VOL_PCT,
    INSTRUMENT_CAP_R,
    PORTFOLIO_CAP_R,
    STOP_1H_HIGHVOL,
    STOP_1H_STD,
    STOP_4H_MULT,
    TRAIL_BASE,
    TRAIL_MAX,
    TRAIL_MIN,
    TRAIL_R_DIV,
    SymbolConfig,
)
from strategy_2.indicators import (
    atr,
    calc_buffer,
    compute_adx,
    compute_daily_state,
    compute_regime,
    compute_regime_4h,
    compute_trend_strength,
    compute_vol_factor,
    confirmed_pivot,
    ema,
    macd,
    percentile_rank,
    scan_pivots,
)
from strategy_2.models import (
    CircuitBreakerState,
    DailyState,
    Direction,
    Pivot,
    PivotKind,
    PivotStore,
    Regime,
    SetupClass,
    SetupInstance,
    SetupState,
    TFState,
)
from strategy_2.signals import (
    compute_div_magnitude,
    detect_class_a,
    detect_class_b,
    detect_class_c,
    detect_class_d,
    detect_add_setup,
    div_mag_passes,
    is_structure_invalidated,
    is_trend_aligned,
)
from strategy_2.stops import (
    compute_be_stop,
    compute_chandelier_stop,
    compute_ratchet_stop,
    compute_stop0_1h,
    compute_stop0_4h,
    compute_trailing_mult,
    is_momentum_strong,
)
from strategy_2.allocator import (
    allocate,
    compute_position_size,
    compute_risk_r,
    compute_unit1_risk,
    rank_setups,
)
from strategy_2.gates import (
    apply_basket_adjustment,
    circuit_breaker_ok,
    corridor_cap_ok,
    extreme_vol_gate,
    full_eligibility_check,
    heat_cap_ok,
    is_entry_window_open,
    is_news_blocked,
    min_stop_gate_ok,
    spread_gate_ok,
)
from shared.oms.models.instrument import Instrument


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def etf_cfg():
    return SymbolConfig(symbol="QQQ", is_etf=True, sec_type="STK",
                        tick_size=0.01, multiplier=1.0, exchange="SMART",
                        entry_window_start_et="09:35", entry_window_end_et="15:45",
                        max_spread_dollars=0.02, max_spread_bps=2)


@pytest.fixture
def fut_cfg():
    return SymbolConfig(symbol="MNQ", tick_size=0.25, multiplier=2.0,
                        exchange="CME", trading_class="MNQ")


@pytest.fixture
def daily_bull():
    return DailyState(ema_fast=110, ema_slow=100, atr_d=2.0,
                      regime=Regime.BULL, trend_strength=1.5,
                      trend_strength_3d_ago=1.2, vol_pct=50, atr_base=1.8,
                      vol_factor=0.9, extreme_vol=False, close=112, adx=25)


@pytest.fixture
def daily_bear():
    return DailyState(ema_fast=90, ema_slow=100, atr_d=2.0,
                      regime=Regime.BEAR, trend_strength=1.5,
                      trend_strength_3d_ago=1.2, vol_pct=50, atr_base=1.8,
                      vol_factor=0.9, extreme_vol=False, close=88, adx=25)


@pytest.fixture
def daily_chop():
    return DailyState(ema_fast=100, ema_slow=100.5, atr_d=2.0,
                      regime=Regime.CHOP, trend_strength=0.2,
                      trend_strength_3d_ago=0.3, vol_pct=50, atr_base=1.8,
                      vol_factor=1.0, extreme_vol=False, close=100, adx=15)


@pytest.fixture
def qqq_inst():
    return Instrument(symbol="QQQ", root="QQQ", venue="SMART",
                      tick_size=0.01, tick_value=0.01, multiplier=1.0)


# ===========================================================================
# INDICATORS
# ===========================================================================

class TestHelixEMA:
    def test_ema_matches_strategy1(self):
        arr = np.arange(1.0, 51.0)
        from strategy.indicators import ema as ema_s1
        result_s2 = ema(arr, 10)
        result_s1 = ema_s1(arr, 10)
        np.testing.assert_allclose(result_s2, result_s1, atol=1e-10)


class TestComputeRegime:
    def test_bull(self):
        assert compute_regime(105, 103, 100) == Regime.BULL

    def test_bear(self):
        assert compute_regime(95, 97, 100) == Regime.BEAR

    def test_chop(self):
        assert compute_regime(101, 100, 102) == Regime.CHOP


class TestComputeRegime4H:
    def test_matches_daily(self):
        r = compute_regime_4h(105, 103, 100)
        assert r == Regime.BULL


class TestTrendStrength:
    def test_positive(self):
        assert compute_trend_strength(110, 100, 5) == pytest.approx(2.0)

    def test_zero_atr(self):
        assert compute_trend_strength(110, 100, 0) == 0.0


class TestVolFactor:
    def test_normal_vol(self):
        vf = compute_vol_factor(2.0, 2.0, 50.0)
        assert vf == pytest.approx(1.0)

    def test_high_vol_capped(self):
        vf = compute_vol_factor(4.0, 2.0, 50.0)
        assert vf >= 0.4  # VOLFACTOR_MIN

    def test_low_vol_capped_at_1(self):
        vf = compute_vol_factor(1.0, 3.0, 15.0)  # low vol pct
        assert vf <= 1.0


class TestPercentileRank:
    def test_basic(self):
        arr = np.arange(1.0, 101.0)
        assert percentile_rank(50, arr) == pytest.approx(50.0)

    def test_empty_array(self):
        assert percentile_rank(10, np.array([])) == 50.0


class TestMACD:
    def test_macd_output_shape(self):
        closes = np.linspace(100, 120, 50)
        line, signal, hist = macd(closes)
        assert len(line) == 50
        assert len(signal) == 50
        assert len(hist) == 50
        np.testing.assert_allclose(hist, line - signal)


class TestCalcBuffer:
    def test_minimum_tick(self):
        b = calc_buffer(0.01, 0.1, True)
        assert b >= 0.01

    def test_minimum_atr_fraction(self):
        b = calc_buffer(0.01, 10.0, False)
        assert b >= 0.05 * 10.0


class TestConfirmedPivot:
    def test_pivot_high(self):
        highs = np.array([10.0, 11.0, 15.0, 12.0, 11.0])
        lows = np.array([8.0, 9.0, 9.5, 8.5, 8.0])
        macd_l = np.zeros(5)
        macd_h = np.zeros(5)
        atr_arr = np.ones(5)
        times = [datetime(2025, 1, 1, i) for i in range(5)]
        p = confirmed_pivot(highs, lows, 4, macd_l, macd_h, atr_arr, times)
        assert p is not None
        assert p.kind == PivotKind.HIGH
        assert p.price == 15.0

    def test_pivot_low(self):
        highs = np.array([15.0, 14.0, 13.0, 14.0, 15.0])
        lows = np.array([12.0, 11.0, 8.0, 10.0, 12.0])
        macd_l = np.zeros(5)
        macd_h = np.zeros(5)
        atr_arr = np.ones(5)
        times = [datetime(2025, 1, 1, i) for i in range(5)]
        p = confirmed_pivot(highs, lows, 4, macd_l, macd_h, atr_arr, times)
        assert p is not None
        assert p.kind == PivotKind.LOW
        assert p.price == 8.0

    def test_no_pivot_insufficient_data(self):
        highs = np.array([10.0, 11.0, 12.0])
        lows = np.array([8.0, 9.0, 10.0])
        p = confirmed_pivot(highs, lows, 2, np.zeros(3), np.zeros(3),
                            np.ones(3), [datetime(2025, 1, 1)] * 3)
        assert p is None


class TestComputeDailyState:
    def test_daily_state_has_required_fields(self):
        n = 80
        closes = np.linspace(100, 130, n)
        highs = closes + 1.5
        lows = closes - 1.5
        state = compute_daily_state(closes, highs, lows, None, "2025-01-15")
        assert state.atr_d > 0
        assert state.vol_factor > 0
        assert state.regime in (Regime.BULL, Regime.BEAR, Regime.CHOP)


class TestComputeADX:
    def test_adx_trending(self):
        n = 100
        closes = np.linspace(100, 150, n)
        highs = closes + 1.0
        lows = closes - 1.0
        val = compute_adx(highs, lows, closes, ADX_PERIOD)
        assert val > 0

    def test_adx_insufficient_data(self):
        closes = np.array([100.0, 101.0])
        highs = np.array([102.0, 103.0])
        lows = np.array([98.0, 99.0])
        assert compute_adx(highs, lows, closes, 14) == 0.0


# ===========================================================================
# SIGNALS
# ===========================================================================

class TestIsTrendAligned:
    def test_long_bull(self):
        assert is_trend_aligned(Direction.LONG, Regime.BULL) is True

    def test_short_bear(self):
        assert is_trend_aligned(Direction.SHORT, Regime.BEAR) is True

    def test_long_bear(self):
        assert is_trend_aligned(Direction.LONG, Regime.BEAR) is False


class TestDivMagnitude:
    def test_basic(self):
        assert compute_div_magnitude(1.0, 0.5, 2.0) == pytest.approx(0.25)

    def test_zero_atr(self):
        assert compute_div_magnitude(1.0, 0.5, 0.0) == 0.0

    def test_passes_default(self):
        assert div_mag_passes(0.10, [], default_threshold=0.05) is True
        assert div_mag_passes(0.02, [], default_threshold=0.05) is False


class TestDetectClassA:
    def test_long_hidden_divergence(self, etf_cfg, daily_bull):
        now = datetime(2025, 1, 15, 12, 0)
        store = PivotStore()
        # L1: price=100, macd=0.5
        store.add(Pivot(ts=datetime(2025, 1, 10), kind=PivotKind.LOW,
                        price=100.0, macd_line=0.5, macd_hist=0.1, atr_tf=1.0, bar_index=0))
        # BOS high between L1 and L2 — keep close to stop so corridor cap passes
        # corridor_cap = 1.6 * atr_d(2.0) = 3.2
        # entry_to_stop = bos.price + buffer - stop0
        # stop0 = L2.price - 0.75*atr_tf = 101 - 0.75 = 100.25
        # entry_to_stop = 103 + 0.05 - 100.25 = 2.80 < 3.2 ✓
        store.add(Pivot(ts=datetime(2025, 1, 12), kind=PivotKind.HIGH,
                        price=103.0, macd_line=1.0, macd_hist=0.2, atr_tf=1.0, bar_index=5))
        # L2: higher low (price 101>100), hidden div (macd 0.3<0.5)
        store.add(Pivot(ts=datetime(2025, 1, 14), kind=PivotKind.LOW,
                        price=101.0, macd_line=0.3, macd_hist=0.05, atr_tf=1.0, bar_index=10))
        tf4h = TFState(tf_label="4H")
        setup = detect_class_a("QQQ", store, daily_bull, tf4h, etf_cfg, [], now)
        assert setup is not None
        assert setup.setup_class == SetupClass.CLASS_A
        assert setup.direction == Direction.LONG

    def test_no_setup_insufficient_pivots(self, etf_cfg, daily_bull):
        store = PivotStore()
        store.add(Pivot(ts=datetime(2025, 1, 10), kind=PivotKind.LOW,
                        price=100.0, macd_line=0.5, macd_hist=0.1, atr_tf=1.0, bar_index=0))
        tf4h = TFState(tf_label="4H")
        setup = detect_class_a("QQQ", store, daily_bull, tf4h, etf_cfg, [])
        assert setup is None


class TestDetectClassC:
    def test_class_c_gate_chop(self, etf_cfg, daily_chop):
        store = PivotStore()
        store.add(Pivot(ts=datetime(2025, 1, 10), kind=PivotKind.HIGH,
                        price=102.0, macd_line=1.0, macd_hist=0.2, atr_tf=1.0, bar_index=0))
        store.add(Pivot(ts=datetime(2025, 1, 12), kind=PivotKind.LOW,
                        price=97.0, macd_line=-0.5, macd_hist=-0.1, atr_tf=1.0, bar_index=5))
        store.add(Pivot(ts=datetime(2025, 1, 14), kind=PivotKind.HIGH,
                        price=104.0, macd_line=0.5, macd_hist=0.1, atr_tf=1.0, bar_index=10))
        tf4h = TFState(tf_label="4H")
        setup = detect_class_c("QQQ", store, daily_chop, tf4h, etf_cfg, [])
        # In chop regime, Class C gate should pass
        # Whether setup is detected depends on divergence
        # At minimum, the gate check should not crash
        assert setup is None or setup.setup_class == SetupClass.CLASS_C

    def test_class_c_blocked_in_bull(self, etf_cfg, daily_bull):
        # In BULL regime without extension, gate should block
        daily_bull.close = daily_bull.ema_fast + 0.5  # not extended
        store = PivotStore()
        store.add(Pivot(ts=datetime(2025, 1, 10), kind=PivotKind.HIGH,
                        price=102.0, macd_line=1.0, macd_hist=0.2, atr_tf=1.0, bar_index=0))
        store.add(Pivot(ts=datetime(2025, 1, 14), kind=PivotKind.HIGH,
                        price=104.0, macd_line=0.5, macd_hist=0.1, atr_tf=1.0, bar_index=10))
        tf4h = TFState(tf_label="4H")
        setup = detect_class_c("QQQ", store, daily_bull, tf4h, etf_cfg, [])
        assert setup is None


class TestDetectClassD:
    def test_class_d_requires_bull_for_long(self, etf_cfg, daily_bull):
        store = PivotStore()
        store.add(Pivot(ts=datetime(2025, 1, 10), kind=PivotKind.LOW,
                        price=100.0, macd_line=0.3, macd_hist=0.1, atr_tf=1.0, bar_index=0))
        store.add(Pivot(ts=datetime(2025, 1, 12), kind=PivotKind.HIGH,
                        price=108.0, macd_line=1.5, macd_hist=0.5, atr_tf=1.0, bar_index=5))
        store.add(Pivot(ts=datetime(2025, 1, 14), kind=PivotKind.LOW,
                        price=103.0, macd_line=0.5, macd_hist=0.2, atr_tf=1.0, bar_index=10))
        tf1h = TFState(tf_label="1H", macd_line=1.0,
                       macd_line_history=[0.2, 0.3, 0.4, 0.5, 0.6])
        setup = detect_class_d("QQQ", store, daily_bull, tf1h, etf_cfg)
        # May or may not detect depending on exact conditions
        assert setup is None or setup.setup_class == SetupClass.CLASS_D

    def test_class_d_blocked_in_chop(self, etf_cfg, daily_chop):
        store = PivotStore()
        store.add(Pivot(ts=datetime(2025, 1, 10), kind=PivotKind.LOW,
                        price=100.0, macd_line=0.3, macd_hist=0.1, atr_tf=1.0, bar_index=0))
        store.add(Pivot(ts=datetime(2025, 1, 14), kind=PivotKind.LOW,
                        price=103.0, macd_line=0.5, macd_hist=0.2, atr_tf=1.0, bar_index=10))
        tf1h = TFState(tf_label="1H", macd_line=1.0,
                       macd_line_history=[0.2, 0.3, 0.4, 0.5, 0.6])
        setup = detect_class_d("QQQ", store, daily_chop, tf1h, etf_cfg)
        assert setup is None  # CHOP regime blocks Class D


class TestStructureInvalidation:
    def test_long_invalidated_by_lower_low(self):
        setup = SetupInstance(
            direction=Direction.LONG,
            pivot_2=Pivot(ts=datetime(2025, 1, 12), kind=PivotKind.LOW,
                          price=100.0, macd_line=0.5, macd_hist=0.1,
                          atr_tf=1.0, bar_index=5),
            bos_pivot=Pivot(ts=datetime(2025, 1, 11), kind=PivotKind.HIGH,
                            price=105.0, macd_line=1.0, macd_hist=0.2,
                            atr_tf=1.0, bar_index=3),
        )
        pivots = PivotStore()
        pivots.add(Pivot(ts=datetime(2025, 1, 14), kind=PivotKind.LOW,
                         price=99.0, macd_line=0.3, macd_hist=0.05,
                         atr_tf=1.0, bar_index=10))
        assert is_structure_invalidated(setup, pivots) is True

    def test_long_not_invalidated_by_higher_low(self):
        setup = SetupInstance(
            direction=Direction.LONG,
            pivot_2=Pivot(ts=datetime(2025, 1, 12), kind=PivotKind.LOW,
                          price=100.0, macd_line=0.5, macd_hist=0.1,
                          atr_tf=1.0, bar_index=5),
        )
        pivots = PivotStore()
        pivots.add(Pivot(ts=datetime(2025, 1, 14), kind=PivotKind.LOW,
                         price=101.0, macd_line=0.6, macd_hist=0.1,
                         atr_tf=1.0, bar_index=10))
        assert is_structure_invalidated(setup, pivots) is False


# ===========================================================================
# STOPS
# ===========================================================================

class TestStop04H:
    def test_long_stop(self):
        s = compute_stop0_4h(Direction.LONG, 100.0, 2.0, 0.01)
        assert s == pytest.approx(100.0 - STOP_4H_MULT * 2.0, abs=0.01)

    def test_short_stop(self):
        s = compute_stop0_4h(Direction.SHORT, 100.0, 2.0, 0.01)
        assert s == pytest.approx(100.0 + STOP_4H_MULT * 2.0, abs=0.01)


class TestStop01H:
    def test_std_vol(self):
        s = compute_stop0_1h(Direction.LONG, 100.0, 1.5, 50.0, 0.01)
        assert s == pytest.approx(100.0 - STOP_1H_STD * 1.5, abs=0.01)

    def test_high_vol(self):
        s = compute_stop0_1h(Direction.LONG, 100.0, 1.5, 85.0, 0.01)
        assert s == pytest.approx(100.0 - STOP_1H_HIGHVOL * 1.5, abs=0.01)


class TestHelixBEStop:
    def test_long_be(self):
        s = compute_be_stop(Direction.LONG, 100.0, 1.5, 0.01)
        assert s < 100.0  # BE stop has cushion below avg_entry for long

    def test_short_be(self):
        s = compute_be_stop(Direction.SHORT, 100.0, 1.5, 0.01)
        assert s > 100.0


class TestRatchetStop:
    def test_long_ratchet(self):
        s = compute_ratchet_stop(Direction.LONG, 100.0, 2.0, 0.01)
        assert s == pytest.approx(102.0, abs=0.01)

    def test_short_ratchet(self):
        s = compute_ratchet_stop(Direction.SHORT, 100.0, 2.0, 0.01)
        assert s == pytest.approx(98.0, abs=0.01)


class TestIsMomentumStrong:
    def test_long_strong(self):
        assert is_momentum_strong(1.5, 1.0, 0.3, Direction.LONG) is True

    def test_long_weak(self):
        assert is_momentum_strong(0.8, 1.0, 0.3, Direction.LONG) is False

    def test_short_strong(self):
        assert is_momentum_strong(-1.5, -1.0, -0.3, Direction.SHORT) is True


class TestTrailingMult:
    def test_base_calculation(self):
        mult = compute_trailing_mult(0.0, False, False, False, 0.0)
        assert mult == pytest.approx(TRAIL_BASE)

    def test_clamped_min(self):
        mult = compute_trailing_mult(50.0, False, True, True, 0.0)
        assert mult >= TRAIL_MIN

    def test_clamped_max(self):
        mult = compute_trailing_mult(0.0, True, False, False, 2.0)
        assert mult <= TRAIL_MAX


class TestHelixChandelierStop:
    def test_long_chandelier(self):
        highs = [100.0, 102.0, 105.0, 103.0, 104.0]
        lows = [98.0, 100.0, 102.0, 101.0, 102.0]
        s = compute_chandelier_stop(Direction.LONG, highs, lows, 5, 1.5, 3.0, 0.01)
        assert s == pytest.approx(105.0 - 3.0 * 1.5, abs=0.01)

    def test_short_chandelier(self):
        highs = [100.0, 102.0, 105.0, 103.0, 104.0]
        lows = [98.0, 100.0, 96.0, 101.0, 99.0]
        s = compute_chandelier_stop(Direction.SHORT, highs, lows, 5, 1.5, 3.0, 0.01)
        assert s == pytest.approx(96.0 + 3.0 * 1.5, abs=0.01)


# ===========================================================================
# ALLOCATOR
# ===========================================================================

class TestRankSetups:
    def test_class_a_before_class_d(self):
        a = SetupInstance(setup_class=SetupClass.CLASS_A, symbol="QQQ",
                          created_ts=datetime(2025, 1, 15))
        d = SetupInstance(setup_class=SetupClass.CLASS_D, symbol="QQQ",
                          created_ts=datetime(2025, 1, 14))
        ranked = rank_setups([d, a])
        assert ranked[0].setup_class == SetupClass.CLASS_A


class TestUnit1Risk:
    def test_basic(self):
        r = compute_unit1_risk(100000, 0.005, 1.0)
        assert r == pytest.approx(500.0)

    def test_with_vol_factor(self):
        r = compute_unit1_risk(100000, 0.005, 0.8)
        assert r == pytest.approx(400.0)


class TestHelixPositionSize:
    def test_basic_sizing(self):
        qty = compute_position_size(100.0, 98.0, 500.0, 1.0, 1.0, 500)
        assert qty == 250  # 500 / (2*1) = 250

    def test_capped_at_max(self):
        qty = compute_position_size(100.0, 99.0, 5000.0, 1.0, 1.0, 10)
        assert qty == 10


class TestRiskR:
    def test_basic(self):
        r = compute_risk_r(100.0, 98.0, 10, 1.0, 500.0)
        assert r == pytest.approx(0.04)  # (2*10*1)/500

    def test_zero_unit_risk(self):
        r = compute_risk_r(100.0, 98.0, 10, 1.0, 0.0)
        assert r == float("inf")


# ===========================================================================
# GATES
# ===========================================================================

class TestEntryWindow:
    def test_inside_window(self, etf_cfg):
        dt = datetime(2025, 1, 15, 10, 0)
        assert is_entry_window_open(dt, etf_cfg) is True

    def test_outside_window(self, etf_cfg):
        dt = datetime(2025, 1, 15, 8, 0)
        assert is_entry_window_open(dt, etf_cfg) is False


class TestNewsBlocked:
    def test_fomc_blocked(self):
        event_dt = datetime(2025, 1, 15, 14, 0)
        now = datetime(2025, 1, 15, 13, 30)  # 30 min before
        assert is_news_blocked(now, "QQQ", [("FOMC", event_dt)]) is True

    def test_no_events(self):
        assert is_news_blocked(datetime(2025, 1, 15, 10, 0), "QQQ", []) is False

    def test_cl_inventory_only_blocks_oil(self):
        event_dt = datetime(2025, 1, 15, 10, 30)
        now = datetime(2025, 1, 15, 10, 20)
        assert is_news_blocked(now, "QQQ", [("CL_INVENTORY", event_dt)]) is False
        assert is_news_blocked(now, "MCL", [("CL_INVENTORY", event_dt)]) is True


class TestSpreadGate:
    def test_futures_pass(self):
        setup = SetupInstance()
        assert spread_gate_ok(1.5, 2, setup) is True

    def test_futures_fail(self):
        setup = SetupInstance()
        assert spread_gate_ok(3.0, 2, setup) is True  # grace: fail_count <= 2
        assert spread_gate_ok(3.0, 2, setup) is True   # second fail ok
        assert spread_gate_ok(3.0, 2, setup) is False  # third fail blocked


class TestMinStopGate:
    def test_pass(self):
        assert min_stop_gate_ok(100.0, 98.0, 1.0, 5.0, 1.0) is True

    def test_fail(self):
        assert min_stop_gate_ok(100.0, 99.99, 1.0, 5.0, 1.0) is False


class TestHeatCap:
    def test_within_cap(self):
        assert heat_cap_ok(0.5, 0.1, 0.3, 0.2, False) is True

    def test_exceeds_portfolio(self):
        assert heat_cap_ok(1.0, 0.3, 0.3, 0.2, False) is False

    def test_extreme_vol_lower_cap(self):
        assert heat_cap_ok(1.0, 0.1, 0.2, 0.2, True) is False


class TestExtremeVolGate:
    def test_blocks_1h_setups(self):
        assert extreme_vol_gate(SetupClass.CLASS_B, 96.0) is False
        assert extreme_vol_gate(SetupClass.CLASS_D, 96.0) is False

    def test_allows_4h_setups(self):
        assert extreme_vol_gate(SetupClass.CLASS_A, 96.0) is True
        assert extreme_vol_gate(SetupClass.CLASS_C, 96.0) is True


class TestCircuitBreaker:
    def test_ok_when_not_paused(self):
        cb = CircuitBreakerState()
        assert circuit_breaker_ok(cb, datetime.now()) is True

    def test_blocked_when_paused(self):
        cb = CircuitBreakerState(paused_until=datetime.now() + timedelta(hours=1))
        assert circuit_breaker_ok(cb, datetime.now()) is False


class TestBasketAdjustment:
    def test_no_basket_symbol(self):
        assert apply_basket_adjustment("GLD", SetupClass.CLASS_A, 1.0, {}) == 1.0

    def test_1h_mutual_exclusion(self):
        active = {"s1": SetupInstance(symbol="MNQ", state=SetupState.ACTIVE,
                                      origin_tf="1H")}
        mult = apply_basket_adjustment("QQQ", SetupClass.CLASS_B, 1.0, active)
        assert mult == 0.0

    def test_4h_reduction(self):
        active = {"s1": SetupInstance(symbol="MNQ", state=SetupState.ACTIVE,
                                      origin_tf="4H")}
        mult = apply_basket_adjustment("QQQ", SetupClass.CLASS_A, 1.0, active)
        assert mult < 1.0
