"""Unit tests for Strategy 4 (Keltner Momentum Breakout) —
indicators, models, signals, config.
"""
from __future__ import annotations

import numpy as np
import pytest

from strategy_4.config import (
    SYMBOL_CONFIGS,
    S5_PB_CONFIGS,
    S5_DUAL_CONFIGS,
    SymbolConfig,
    build_instruments,
)
from strategy_4.indicators import (
    atr,
    ema,
    keltner_channel,
    roc,
    rsi,
    volume_sma,
)
from strategy_4.models import DailyState, Direction
from strategy_4.signals import (
    entry_signal,
    should_exit_signal,
)


# ===========================================================================
# Fixtures — configs
# ===========================================================================

@pytest.fixture
def default_cfg():
    """Breakout mode, shorts enabled (default)."""
    return SymbolConfig(symbol="QQQ")


@pytest.fixture
def pullback_cfg():
    return SymbolConfig(symbol="QQQ", entry_mode="pullback")


@pytest.fixture
def momentum_cfg():
    return SymbolConfig(symbol="QQQ", entry_mode="momentum")


@pytest.fixture
def dual_cfg():
    return SymbolConfig(symbol="QQQ", entry_mode="dual")


@pytest.fixture
def midline_exit_cfg():
    return SymbolConfig(symbol="QQQ", exit_mode="midline")


@pytest.fixture
def reversal_exit_cfg():
    return SymbolConfig(symbol="QQQ", exit_mode="reversal")


@pytest.fixture
def no_shorts_cfg():
    return SymbolConfig(symbol="QQQ", shorts_enabled=False)


# ===========================================================================
# Fixtures — DailyState scenarios
# ===========================================================================

@pytest.fixture
def state_breakout_long():
    """Close above upper band, RSI bullish, ROC positive, volume above SMA."""
    return DailyState(
        close=110.0, close_prev=105.0,
        kelt_upper=108.0, kelt_middle=100.0, kelt_lower=92.0,
        kelt_middle_prev=99.5,
        rsi=65.0, rsi_prev=55.0, roc=5.0,
        volume=2e6, volume_sma=1.5e6, atr=3.0,
    )


@pytest.fixture
def state_breakout_short():
    """Close below lower band, RSI bearish, ROC negative, volume above SMA."""
    return DailyState(
        close=88.0, close_prev=95.0,
        kelt_upper=108.0, kelt_middle=100.0, kelt_lower=92.0,
        kelt_middle_prev=100.5,
        rsi=35.0, rsi_prev=45.0, roc=-5.0,
        volume=2e6, volume_sma=1.5e6, atr=3.0,
    )


@pytest.fixture
def state_pullback_long():
    """Price crosses above midline (pullback recovery), RSI/ROC bullish."""
    return DailyState(
        close=101.0, close_prev=99.0,
        kelt_upper=108.0, kelt_middle=100.0, kelt_lower=92.0,
        kelt_middle_prev=100.0,
        rsi=55.0, rsi_prev=52.0, roc=2.0,
        volume=2e6, volume_sma=1.5e6, atr=3.0,
    )


@pytest.fixture
def state_pullback_short():
    """Price crosses below midline, RSI/ROC bearish."""
    return DailyState(
        close=99.0, close_prev=101.0,
        kelt_upper=108.0, kelt_middle=100.0, kelt_lower=92.0,
        kelt_middle_prev=100.0,
        rsi=45.0, rsi_prev=48.0, roc=-2.0,
        volume=2e6, volume_sma=1.5e6, atr=3.0,
    )


@pytest.fixture
def state_momentum_long():
    """RSI crosses above 50, price above midline, ROC positive."""
    return DailyState(
        close=102.0, close_prev=101.0,
        kelt_upper=108.0, kelt_middle=100.0, kelt_lower=92.0,
        kelt_middle_prev=99.5,
        rsi=52.0, rsi_prev=48.0, roc=3.0,
        volume=2e6, volume_sma=1.5e6, atr=3.0,
    )


@pytest.fixture
def state_momentum_short():
    """RSI crosses below 50, price below midline, ROC negative."""
    return DailyState(
        close=98.0, close_prev=99.0,
        kelt_upper=108.0, kelt_middle=100.0, kelt_lower=92.0,
        kelt_middle_prev=100.5,
        rsi=48.0, rsi_prev=52.0, roc=-3.0,
        volume=2e6, volume_sma=1.5e6, atr=3.0,
    )


@pytest.fixture
def state_flat():
    """No conditions met — close between bands, RSI at 50, ROC zero."""
    return DailyState(
        close=100.0, close_prev=100.0,
        kelt_upper=108.0, kelt_middle=100.0, kelt_lower=92.0,
        kelt_middle_prev=100.0,
        rsi=50.0, rsi_prev=50.0, roc=0.0,
        volume=2e6, volume_sma=1.5e6, atr=3.0,
    )


@pytest.fixture
def state_midline_exit_long():
    """Long position: close crosses below midline."""
    return DailyState(
        close=99.0, close_prev=101.0,
        kelt_upper=108.0, kelt_middle=100.0, kelt_lower=92.0,
        kelt_middle_prev=100.0,
        rsi=48.0, rsi_prev=52.0, roc=-1.0,
        volume=2e6, volume_sma=1.5e6, atr=3.0,
    )


@pytest.fixture
def state_midline_exit_short():
    """Short position: close crosses above midline."""
    return DailyState(
        close=101.0, close_prev=99.0,
        kelt_upper=108.0, kelt_middle=100.0, kelt_lower=92.0,
        kelt_middle_prev=100.0,
        rsi=52.0, rsi_prev=48.0, roc=1.0,
        volume=2e6, volume_sma=1.5e6, atr=3.0,
    )


@pytest.fixture
def state_reversal_bearish():
    """Full bearish conditions: close < lower, RSI bearish, ROC negative."""
    return DailyState(
        close=88.0, close_prev=95.0,
        kelt_upper=108.0, kelt_middle=100.0, kelt_lower=92.0,
        kelt_middle_prev=100.0,
        rsi=35.0, rsi_prev=45.0, roc=-5.0,
        volume=2e6, volume_sma=1.5e6, atr=3.0,
    )


@pytest.fixture
def state_reversal_bullish():
    """Full bullish conditions: close > upper, RSI bullish, ROC positive."""
    return DailyState(
        close=112.0, close_prev=105.0,
        kelt_upper=108.0, kelt_middle=100.0, kelt_lower=92.0,
        kelt_middle_prev=100.0,
        rsi=65.0, rsi_prev=55.0, roc=5.0,
        volume=2e6, volume_sma=1.5e6, atr=3.0,
    )


# ===========================================================================
# === INDICATORS ===
# ===========================================================================


class TestEMA:
    """Tests for indicators.ema()."""

    def test_sma_seed(self):
        """First value should be SMA of first `period` elements."""
        arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = ema(arr, period=3)
        assert result[0] == pytest.approx(np.mean(arr[:3]), abs=1e-10)

    def test_constant_series_converges(self):
        """EMA of a constant series should converge to the constant."""
        arr = np.full(50, 42.0)
        result = ema(arr, period=10)
        assert result[-1] == pytest.approx(42.0, abs=1e-8)

    def test_uptrend_lags_price(self):
        """EMA should lag behind a linearly increasing series."""
        arr = np.arange(1.0, 101.0)
        result = ema(arr, period=20)
        assert result[-1] < arr[-1]

    def test_period_one_equals_input(self):
        """EMA with period=1 should track input exactly after seed."""
        arr = np.array([5.0, 10.0, 15.0, 20.0])
        result = ema(arr, period=1)
        # k = 2/(1+1) = 1.0, so out[i] = arr[i]*1.0 + out[i-1]*0.0
        np.testing.assert_allclose(result, arr, atol=1e-10)

    def test_output_length(self):
        """Output length should match input length."""
        arr = np.random.rand(73)
        result = ema(arr, period=14)
        assert len(result) == len(arr)


class TestATR:
    """Tests for indicators.atr()."""

    def test_single_bar_true_range(self):
        """First bar TR = high - low."""
        h = np.array([105.0])
        l = np.array([95.0])
        c = np.array([100.0])
        result = atr(h, l, c, period=14)
        assert result[0] == pytest.approx(10.0, abs=1e-10)

    def test_constant_range_converges(self):
        """ATR of constant H-L bars should converge to that range."""
        n = 200
        h = np.full(n, 110.0)
        l = np.full(n, 100.0)
        c = np.full(n, 105.0)
        result = atr(h, l, c, period=14)
        assert result[-1] == pytest.approx(10.0, abs=0.1)

    def test_wilder_smoothing(self):
        """Wilder smoothing: out[i] = out[i-1]*(1-1/p) + tr[i]*(1/p)."""
        h = np.array([110.0, 112.0, 108.0])
        l = np.array([100.0, 101.0, 99.0])
        c = np.array([105.0, 106.0, 103.0])
        result = atr(h, l, c, period=14)
        alpha = 1.0 / 14
        tr_0 = 10.0  # 110-100
        tr_1 = max(112 - 101, abs(112 - 105), abs(101 - 105))  # 11
        expected_1 = tr_0 * (1 - alpha) + tr_1 * alpha
        assert result[1] == pytest.approx(expected_1, abs=1e-10)

    def test_increasing_volatility_increases_atr(self):
        """ATR should increase when true range grows."""
        n = 100
        c = np.full(n, 100.0)
        h_small = c + 1.0
        l_small = c - 1.0
        result_small = atr(h_small, l_small, c, period=14)

        h_big = c + 5.0
        l_big = c - 5.0
        result_big = atr(h_big, l_big, c, period=14)
        assert result_big[-1] > result_small[-1]

    def test_always_positive(self):
        """ATR should always be positive."""
        rng = np.random.RandomState(0)
        c = 100 + np.cumsum(rng.randn(100))
        h = c + rng.uniform(0.1, 2.0, 100)
        l = c - rng.uniform(0.1, 2.0, 100)
        result = atr(h, l, c, period=14)
        assert np.all(result > 0)


class TestRSI:
    """Tests for indicators.rsi()."""

    def test_single_value_is_50(self):
        """RSI of single element should be 50."""
        result = rsi(np.array([100.0]), period=14)
        assert result[0] == pytest.approx(50.0, abs=1e-10)

    def test_monotonic_up_approaches_100(self):
        """RSI of monotonically increasing series should approach 100."""
        arr = np.arange(1.0, 102.0)
        result = rsi(arr, period=14)
        assert result[-1] > 90.0

    def test_monotonic_down_approaches_0(self):
        """RSI of monotonically decreasing series should approach 0."""
        arr = np.arange(200.0, 99.0, -1.0)
        result = rsi(arr, period=14)
        assert result[-1] < 10.0

    def test_range_0_100(self):
        """RSI should always be within [0, 100]."""
        rng = np.random.RandomState(42)
        arr = 100 + np.cumsum(rng.randn(200))
        result = rsi(arr, period=14)
        assert np.all(result >= 0)
        assert np.all(result <= 100)

    def test_constant_is_50(self):
        """RSI of constant series should be 50."""
        arr = np.full(50, 100.0)
        result = rsi(arr, period=14)
        assert result[-1] == pytest.approx(50.0, abs=1e-10)

    def test_zero_loss_is_100(self):
        """If all changes are gains (no losses), RSI should be 100."""
        arr = np.array([100.0, 101.0, 103.0, 106.0, 110.0, 115.0,
                        121.0, 128.0, 136.0, 145.0, 155.0, 166.0,
                        178.0, 191.0, 205.0, 220.0])
        result = rsi(arr, period=14)
        assert result[-1] == pytest.approx(100.0, abs=1e-10)


class TestROC:
    """Tests for indicators.roc()."""

    def test_zeros_before_period(self):
        """ROC should be 0 for indices < period."""
        arr = np.arange(1.0, 21.0)
        result = roc(arr, period=10)
        np.testing.assert_array_equal(result[:10], 0.0)

    def test_known_percentage(self):
        """ROC should compute correct percentage change."""
        arr = np.array([100.0, 101.0, 102.0, 103.0, 104.0,
                        105.0, 106.0, 107.0, 108.0, 109.0,
                        120.0])  # 120 vs 100 = +20%
        result = roc(arr, period=10)
        assert result[10] == pytest.approx(20.0, abs=1e-10)

    def test_negative_decline(self):
        """ROC should be negative for declining prices."""
        arr = np.array([200.0, 199.0, 198.0, 197.0, 196.0,
                        195.0, 194.0, 193.0, 192.0, 191.0,
                        180.0])  # 180 vs 200 = -10%
        result = roc(arr, period=10)
        assert result[10] == pytest.approx(-10.0, abs=1e-10)

    def test_zero_prev_guard(self):
        """ROC should return 0 when previous close is 0 (avoid division)."""
        arr = np.array([0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0,
                        100.0])
        result = roc(arr, period=10)
        assert result[10] == 0.0

    def test_output_length(self):
        """Output should have same length as input."""
        arr = np.random.rand(50)
        result = roc(arr, period=10)
        assert len(result) == 50


class TestKeltnerChannel:
    """Tests for indicators.keltner_channel()."""

    def test_returns_three_arrays(self):
        """Should return (upper, middle, lower) tuple."""
        n = 50
        c = np.full(n, 100.0)
        h = c + 1.0
        l = c - 1.0
        result = keltner_channel(c, h, l)
        assert len(result) == 3
        assert all(len(a) == n for a in result)

    def test_upper_gt_middle_gt_lower(self):
        """Upper > middle > lower at every point."""
        rng = np.random.RandomState(42)
        c = 100 + np.cumsum(rng.randn(100) * 0.5)
        h = c + rng.uniform(0.5, 2.0, 100)
        l = c - rng.uniform(0.5, 2.0, 100)
        upper, middle, lower = keltner_channel(c, h, l)
        assert np.all(upper > middle)
        assert np.all(middle > lower)

    def test_middle_equals_ema(self):
        """Middle band should equal EMA of closes."""
        c = np.arange(1.0, 51.0)
        h = c + 1.0
        l = c - 1.0
        upper, middle, lower = keltner_channel(c, h, l, ema_period=20)
        expected_ema = ema(c, 20)
        np.testing.assert_allclose(middle, expected_ema, atol=1e-10)

    def test_symmetric_bands(self):
        """Upper - middle should equal middle - lower (both = ATR * mult)."""
        c = np.full(100, 100.0)
        h = c + 2.0
        l = c - 2.0
        upper, middle, lower = keltner_channel(c, h, l, atr_mult=2.0)
        band_width_up = upper - middle
        band_width_dn = middle - lower
        np.testing.assert_allclose(band_width_up, band_width_dn, atol=1e-10)

    def test_wider_mult_wider_bands(self):
        """Larger ATR multiplier should produce wider bands."""
        c = np.full(50, 100.0)
        h = c + 2.0
        l = c - 2.0
        u1, m1, l1 = keltner_channel(c, h, l, atr_mult=1.0)
        u2, m2, l2 = keltner_channel(c, h, l, atr_mult=3.0)
        assert (u2[-1] - m2[-1]) > (u1[-1] - m1[-1])


class TestVolumeSMA:
    """Tests for indicators.volume_sma()."""

    def test_constant_converges(self):
        """Volume SMA of constant series should equal that constant."""
        arr = np.full(50, 1e6)
        result = volume_sma(arr, period=20)
        assert result[-1] == pytest.approx(1e6, abs=1.0)

    def test_expanding_window_before_period(self):
        """Before period, uses expanding window average."""
        arr = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        result = volume_sma(arr, period=20)
        # index 0: 100/1 = 100
        assert result[0] == pytest.approx(100.0, abs=1e-10)
        # index 1: (100+200)/2 = 150
        assert result[1] == pytest.approx(150.0, abs=1e-10)
        # index 2: (100+200+300)/3 = 200
        assert result[2] == pytest.approx(200.0, abs=1e-10)

    def test_known_rolling_values(self):
        """After period, should compute correct rolling average."""
        arr = np.arange(1.0, 26.0)  # 1..25
        result = volume_sma(arr, period=5)
        # index 5: should be mean of arr[1..5] = (2+3+4+5+6)/5 = 4.0
        expected = np.mean(arr[1:6])
        assert result[5] == pytest.approx(expected, abs=1e-10)


# ===========================================================================
# === MODELS ===
# ===========================================================================


class TestDirection:
    """Tests for models.Direction."""

    def test_values(self):
        assert int(Direction.SHORT) == -1
        assert int(Direction.FLAT) == 0
        assert int(Direction.LONG) == 1

    def test_arithmetic(self):
        assert Direction.LONG + Direction.SHORT == 0
        assert Direction.LONG * 2 == 2

    def test_negation(self):
        assert -Direction.LONG == Direction.SHORT
        assert -Direction.SHORT == Direction.LONG


class TestDailyState:
    """Tests for models.DailyState."""

    def test_defaults(self):
        s = DailyState()
        assert s.rsi == 50.0
        assert s.close == 0.0
        assert s.atr == 0.0
        assert s.roc == 0.0

    def test_custom_values(self):
        s = DailyState(close=150.0, rsi=70.0, atr=2.5)
        assert s.close == 150.0
        assert s.rsi == 70.0
        assert s.atr == 2.5

    def test_all_twelve_fields(self):
        s = DailyState(
            close=100.0, close_prev=99.0,
            kelt_upper=110.0, kelt_middle=100.0, kelt_lower=90.0,
            kelt_middle_prev=99.5,
            rsi=55.0, rsi_prev=50.0, roc=3.0,
            volume=2e6, volume_sma=1.5e6, atr=2.0,
        )
        assert s.close == 100.0
        assert s.close_prev == 99.0
        assert s.kelt_upper == 110.0
        assert s.kelt_middle == 100.0
        assert s.kelt_lower == 90.0
        assert s.kelt_middle_prev == 99.5
        assert s.rsi == 55.0
        assert s.rsi_prev == 50.0
        assert s.roc == 3.0
        assert s.volume == 2e6
        assert s.volume_sma == 1.5e6
        assert s.atr == 2.0


# ===========================================================================
# === SIGNALS — ENTRY ===
# ===========================================================================


class TestBreakoutEntry:
    """Tests for breakout entry mode."""

    def test_long_all_conditions(self, default_cfg, state_breakout_long):
        result = entry_signal(state_breakout_long, default_cfg)
        assert result == Direction.LONG

    def test_short_all_conditions(self, default_cfg, state_breakout_short):
        result = entry_signal(state_breakout_short, default_cfg)
        assert result == Direction.SHORT

    def test_rsi_missing_blocks_long(self, default_cfg):
        """RSI below threshold blocks breakout long."""
        state = DailyState(
            close=110.0, kelt_upper=108.0, kelt_middle=100.0, kelt_lower=92.0,
            rsi=45.0, roc=5.0, volume=2e6, volume_sma=1.5e6,
        )
        result = entry_signal(state, default_cfg)
        assert result == Direction.FLAT

    def test_roc_missing_blocks_long(self, default_cfg):
        """ROC <= 0 blocks breakout long."""
        state = DailyState(
            close=110.0, kelt_upper=108.0, kelt_middle=100.0, kelt_lower=92.0,
            rsi=60.0, roc=0.0, volume=2e6, volume_sma=1.5e6,
        )
        result = entry_signal(state, default_cfg)
        assert result == Direction.FLAT

    def test_close_below_upper_blocks_long(self, default_cfg):
        """Close not above upper band blocks breakout long."""
        state = DailyState(
            close=107.0, kelt_upper=108.0, kelt_middle=100.0, kelt_lower=92.0,
            rsi=60.0, roc=5.0, volume=2e6, volume_sma=1.5e6,
        )
        result = entry_signal(state, default_cfg)
        assert result == Direction.FLAT

    def test_shorts_disabled_blocks_short(self, no_shorts_cfg, state_breakout_short):
        """shorts_enabled=False should block short entries."""
        result = entry_signal(state_breakout_short, no_shorts_cfg)
        assert result == Direction.FLAT


class TestPullbackEntry:
    """Tests for pullback entry mode."""

    def test_long_crossover(self, pullback_cfg, state_pullback_long):
        result = entry_signal(state_pullback_long, pullback_cfg)
        assert result == Direction.LONG

    def test_short_crossover(self, pullback_cfg, state_pullback_short):
        result = entry_signal(state_pullback_short, pullback_cfg)
        assert result == Direction.SHORT

    def test_no_crossover_flat(self, pullback_cfg):
        """Both bars on same side of midline -> FLAT."""
        state = DailyState(
            close=102.0, close_prev=101.0,
            kelt_middle=100.0, kelt_middle_prev=100.0,
            rsi=55.0, roc=2.0, volume=2e6, volume_sma=1.5e6,
        )
        result = entry_signal(state, pullback_cfg)
        assert result == Direction.FLAT

    def test_rsi_missing_blocks(self, pullback_cfg):
        """RSI below threshold blocks pullback long."""
        state = DailyState(
            close=101.0, close_prev=99.0,
            kelt_middle=100.0, kelt_middle_prev=100.0,
            rsi=45.0, roc=2.0, volume=2e6, volume_sma=1.5e6,
        )
        result = entry_signal(state, pullback_cfg)
        assert result == Direction.FLAT

    def test_roc_missing_blocks(self, pullback_cfg):
        """ROC <= 0 blocks pullback long."""
        state = DailyState(
            close=101.0, close_prev=99.0,
            kelt_middle=100.0, kelt_middle_prev=100.0,
            rsi=55.0, roc=0.0, volume=2e6, volume_sma=1.5e6,
        )
        result = entry_signal(state, pullback_cfg)
        assert result == Direction.FLAT

    def test_boundary_close_equals_midline(self, pullback_cfg):
        """close == midline and prev < midline_prev -> long (>=)."""
        state = DailyState(
            close=100.0, close_prev=99.0,
            kelt_middle=100.0, kelt_middle_prev=100.0,
            rsi=55.0, roc=2.0, volume=2e6, volume_sma=1.5e6,
        )
        result = entry_signal(state, pullback_cfg)
        assert result == Direction.LONG


class TestMomentumEntry:
    """Tests for momentum entry mode."""

    def test_rsi_crossover_long(self, momentum_cfg, state_momentum_long):
        result = entry_signal(state_momentum_long, momentum_cfg)
        assert result == Direction.LONG

    def test_rsi_crossover_short(self, momentum_cfg, state_momentum_short):
        result = entry_signal(state_momentum_short, momentum_cfg)
        assert result == Direction.SHORT

    def test_no_crossover_flat(self, momentum_cfg):
        """RSI both above threshold (no crossover) -> FLAT."""
        state = DailyState(
            close=102.0, close_prev=101.0,
            kelt_middle=100.0, kelt_middle_prev=100.0,
            rsi=55.0, rsi_prev=52.0, roc=3.0,
            volume=2e6, volume_sma=1.5e6,
        )
        result = entry_signal(state, momentum_cfg)
        assert result == Direction.FLAT

    def test_below_midline_blocks_long(self, momentum_cfg):
        """Price below midline blocks momentum long."""
        state = DailyState(
            close=98.0, close_prev=97.0,
            kelt_middle=100.0, kelt_middle_prev=100.0,
            rsi=52.0, rsi_prev=48.0, roc=3.0,
            volume=2e6, volume_sma=1.5e6,
        )
        result = entry_signal(state, momentum_cfg)
        assert result == Direction.FLAT

    def test_roc_blocks_long(self, momentum_cfg):
        """ROC <= 0 blocks momentum long."""
        state = DailyState(
            close=102.0, close_prev=101.0,
            kelt_middle=100.0, kelt_middle_prev=100.0,
            rsi=52.0, rsi_prev=48.0, roc=0.0,
            volume=2e6, volume_sma=1.5e6,
        )
        result = entry_signal(state, momentum_cfg)
        assert result == Direction.FLAT


class TestDualEntry:
    """Tests for dual (breakout | pullback) entry mode."""

    def test_fires_breakout(self, dual_cfg, state_breakout_long):
        """Dual mode fires breakout when breakout conditions met."""
        result = entry_signal(state_breakout_long, dual_cfg)
        assert result == Direction.LONG

    def test_falls_through_to_pullback(self, dual_cfg, state_pullback_long):
        """Dual mode fires pullback when breakout fails but pullback fires."""
        result = entry_signal(state_pullback_long, dual_cfg)
        assert result == Direction.LONG

    def test_flat_when_neither(self, dual_cfg, state_flat):
        """Dual returns FLAT when neither breakout nor pullback fires."""
        result = entry_signal(state_flat, dual_cfg)
        assert result == Direction.FLAT


class TestVolumeFilter:
    """Tests for volume filter in entry_signal()."""

    def test_blocks_low_volume(self, default_cfg):
        """Volume below SMA should block entry."""
        state = DailyState(
            close=110.0, kelt_upper=108.0, kelt_middle=100.0, kelt_lower=92.0,
            rsi=65.0, roc=5.0,
            volume=1e6, volume_sma=2e6, atr=3.0,
        )
        result = entry_signal(state, default_cfg)
        assert result == Direction.FLAT

    def test_passes_high_volume(self, default_cfg, state_breakout_long):
        """Volume above SMA should allow entry."""
        result = entry_signal(state_breakout_long, default_cfg)
        assert result == Direction.LONG

    def test_disabled_bypasses(self):
        """volume_filter=False should bypass volume check."""
        cfg = SymbolConfig(symbol="QQQ", volume_filter=False)
        state = DailyState(
            close=110.0, kelt_upper=108.0, kelt_middle=100.0, kelt_lower=92.0,
            rsi=65.0, roc=5.0,
            volume=1e6, volume_sma=2e6, atr=3.0,
        )
        result = entry_signal(state, cfg)
        assert result == Direction.LONG

    def test_zero_sma_bypasses(self, default_cfg):
        """volume_sma=0 should bypass volume check (avoid div-by-zero)."""
        state = DailyState(
            close=110.0, kelt_upper=108.0, kelt_middle=100.0, kelt_lower=92.0,
            rsi=65.0, roc=5.0,
            volume=1e6, volume_sma=0.0, atr=3.0,
        )
        result = entry_signal(state, default_cfg)
        assert result == Direction.LONG

    def test_equality_passes(self, default_cfg):
        """Volume exactly equal to SMA should not be blocked (not < SMA)."""
        state = DailyState(
            close=110.0, kelt_upper=108.0, kelt_middle=100.0, kelt_lower=92.0,
            rsi=65.0, roc=5.0,
            volume=2e6, volume_sma=2e6, atr=3.0,
        )
        result = entry_signal(state, default_cfg)
        assert result == Direction.LONG


# ===========================================================================
# === SIGNALS — EXIT ===
# ===========================================================================


class TestShouldExitTrailOnly:
    """Tests for trail_only exit mode."""

    def test_long_always_false(self, default_cfg, state_flat):
        should, reason = should_exit_signal(state_flat, Direction.LONG, default_cfg)
        assert should is False
        assert reason == ""

    def test_short_always_false(self, default_cfg, state_flat):
        should, reason = should_exit_signal(state_flat, Direction.SHORT, default_cfg)
        assert should is False
        assert reason == ""


class TestShouldExitMidline:
    """Tests for midline exit mode (default when exit_mode not trail_only/reversal)."""

    def test_long_crosses_below(self, midline_exit_cfg, state_midline_exit_long):
        should, reason = should_exit_signal(
            state_midline_exit_long, Direction.LONG, midline_exit_cfg,
        )
        assert should is True
        assert reason == "KELT_MID_EXIT"

    def test_short_crosses_above(self, midline_exit_cfg, state_midline_exit_short):
        should, reason = should_exit_signal(
            state_midline_exit_short, Direction.SHORT, midline_exit_cfg,
        )
        assert should is True
        assert reason == "KELT_MID_EXIT"

    def test_no_crossover_no_exit(self, midline_exit_cfg):
        """No midline crossover -> no exit."""
        state = DailyState(
            close=102.0, close_prev=101.0,
            kelt_middle=100.0, kelt_middle_prev=100.0,
            rsi=55.0, rsi_prev=52.0,
        )
        should, reason = should_exit_signal(state, Direction.LONG, midline_exit_cfg)
        assert should is False


class TestShouldExitReversal:
    """Tests for reversal exit mode."""

    def test_long_exits_on_full_bearish(self, reversal_exit_cfg, state_reversal_bearish):
        should, reason = should_exit_signal(
            state_reversal_bearish, Direction.LONG, reversal_exit_cfg,
        )
        assert should is True
        assert reason == "REVERSAL_EXIT"

    def test_short_exits_on_full_bullish(self, reversal_exit_cfg, state_reversal_bullish):
        should, reason = should_exit_signal(
            state_reversal_bullish, Direction.SHORT, reversal_exit_cfg,
        )
        assert should is True
        assert reason == "REVERSAL_EXIT"

    def test_partial_conditions_no_exit(self, reversal_exit_cfg):
        """Only some bearish conditions met -> no reversal exit from long."""
        state = DailyState(
            close=88.0,  # below lower band
            kelt_upper=108.0, kelt_middle=100.0, kelt_lower=92.0,
            rsi=55.0,  # NOT bearish
            roc=-5.0,
        )
        should, reason = should_exit_signal(
            state, Direction.LONG, reversal_exit_cfg,
        )
        assert should is False


# ===========================================================================
# === CONFIG ===
# ===========================================================================


class TestConfig:
    """Tests for strategy_4.config module."""

    def test_symbol_configs_keys(self):
        assert set(SYMBOL_CONFIGS.keys()) == {"QQQ", "GLD", "IBIT"}

    def test_default_values(self):
        cfg = SymbolConfig(symbol="TEST")
        assert cfg.entry_mode == "breakout"
        assert cfg.exit_mode == "trail_only"
        assert cfg.shorts_enabled is True
        assert cfg.kelt_ema_period == 20
        assert cfg.kelt_atr_period == 14
        assert cfg.kelt_atr_mult == 2.0

    def test_s5_pb_configs(self):
        assert "IBIT" in S5_PB_CONFIGS
        cfg = S5_PB_CONFIGS["IBIT"]
        assert cfg.entry_mode == "pullback"
        assert cfg.kelt_ema_period == 10

    def test_s5_dual_configs(self):
        assert "GLD" in S5_DUAL_CONFIGS
        assert "IBIT" in S5_DUAL_CONFIGS
        cfg = S5_DUAL_CONFIGS["GLD"]
        assert cfg.entry_mode == "dual"
        assert cfg.shorts_enabled is False

    def test_frozen_dataclass(self):
        cfg = SymbolConfig(symbol="QQQ")
        with pytest.raises(AttributeError):
            cfg.symbol = "SPY"  # type: ignore[misc]

    def test_build_instruments(self):
        instruments = build_instruments({"QQQ": SymbolConfig(symbol="QQQ")})
        assert "QQQ" in instruments
        inst = instruments["QQQ"]
        assert inst.symbol == "QQQ"
        assert inst.point_value == pytest.approx(1.0, abs=1e-10)
