"""BRS single-symbol bar-by-bar backtest engine.

Follows RegimeEngine structural pattern (position tracking via instance vars,
run() → _run_loop() → per-bar _manage_position()), but with:
  - 3 timeframes: daily (regime/context), 4H (structure), 1H (execution)
  - 4 entry signal checkers (S1/S2/S3/L1)
  - Richer exit logic: catastrophic cap, BE, chandelier trail, stale, time decay
  - Bear conviction scoring and vol-adjusted sizing
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

import numpy as np

from backtest.config import SlippageConfig
from backtest.config_brs import BRSConfig, BRSSymbolConfig
from backtest.data.preprocessing import NumpyBars
from backtest.engine.backtest_engine import SymbolResult, TradeRecord

from strategies.swing.brs.exits import (
    check_exits,
    check_scale_out,
    compute_initial_stop,
    update_trailing_stop,
)
from strategies.swing.brs.indicators import (
    adx_suite,
    atr,
    compute_4h_structure,
    compute_bear_conviction,
    compute_ema_slope,
    compute_risk_regime,
    compute_vol_factor,
    donchian_low,
    ema,
    swing_high_confirmed,
    volume_sma,
)
from strategies.swing.brs.models import (
    BDArmState,
    BRSRegime,
    BiasState,
    DailyContext,
    Direction,
    EntrySignal,
    EntryType,
    ExitReason,
    HourlyContext,
    LHArmState,
    Regime4H,
    S2ArmState,
    S3ArmState,
    VolState,
)
from strategies.swing.brs.regime import (
    classify_regime,
    compute_raw_bias,
    compute_regime_on,
    update_bias,
)
from strategies.swing.brs.signals import (
    check_bd_arm,
    check_bd_continuation,
    check_l1,
    check_lh_rejection,
    check_s1,
    check_s2,
    check_s2_arm,
    check_s3,
    check_s3_arm,
)
from strategies.swing.brs.sizing import compute_position_size

logger = logging.getLogger(__name__)


class BRSEngine:
    """Single-symbol BRS backtest engine."""

    def __init__(
        self,
        symbol: str,
        sym_cfg: BRSSymbolConfig,
        cfg: BRSConfig,
    ):
        self.symbol = symbol
        self.sym_cfg = sym_cfg
        self.cfg = cfg

        # State
        self.equity = cfg.initial_equity
        self.daily_ctx = DailyContext()
        self.prev_regime = BRSRegime.RANGE_CHOP
        self._prev_regime_on = False
        self._bias = BiasState()
        self._short_bias_no_trade: int = 0  # R8: persistence override counter

        # Position tracking
        self._in_position = False
        self._pos_direction = Direction.FLAT
        self._pos_entry_price: float = 0.0
        self._pos_initial_stop: float = 0.0
        self._pos_current_stop: float = 0.0
        self._pos_qty: int = 0
        self._pos_entry_time: datetime | None = None
        self._pos_risk_per_unit: float = 0.0
        self._pos_bars_held: int = 0
        self._pos_mfe_price: float = 0.0
        self._pos_mfe_r: float = 0.0
        self._pos_mae_price: float = 0.0
        self._pos_be_triggered: bool = False
        self._pos_regime_entry = ""
        self._pos_entry_type = ""
        self._pos_adx_entry: float = 0.0
        self._pos_score_entry: float = 0.0
        self._pos_di_agrees: bool = False
        self._pos_quality: float = 0.0
        self._pos_chand_bonus: float = 0.0  # widened after scale-out (Change #6)
        self._hourly_highs: list[float] = []
        self._hourly_lows: list[float] = []

        # Scale-out state (Change #6)
        self._pos_tranche_b_open: bool = False
        self._pos_original_qty: int = 0

        # Pyramiding state (R9)
        self._pyramid_count: int = 0
        self._last_close: float = 0.0

        # Arming state
        self._s2_arm = S2ArmState()
        self._s3_arm = S3ArmState()
        self._lh_arm = LHArmState()
        self._bd_arm = BDArmState()
        self._swing_highs: list[float] = []  # confirmed swing highs for LH detection

        # Cooldown
        self._cooldown_until_bar: int = 0

        # Concurrent position tracking (for portfolio-level heat — simplified single-symbol)
        self._concurrent_positions: int = 0
        self._current_heat_r: float = 0.0

        # Results
        self.trades: list[TradeRecord] = []
        self.equity_curve: list[float] = []
        self.timestamps: list = []
        self.total_commission: float = 0.0

        # Slippage config
        self._slip = cfg.slippage

        # QQQ regime timeline (reserved for future cross-symbol L1 logic)
        self._qqq_regimes: list[BRSRegime] | None = None
        self._regime_history: list[BRSRegime] = []  # track this symbol's regime history

        # Crisis state log — lightweight daily snapshot during crisis windows only
        self.crisis_state_log: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        daily: NumpyBars,
        hourly: NumpyBars,
        four_hour: NumpyBars,
        daily_idx_map: np.ndarray,
        four_hour_idx_map: np.ndarray,
    ) -> SymbolResult:
        """Run the backtest over all hourly bars."""
        self._run_loop(daily, hourly, four_hour, daily_idx_map, four_hour_idx_map)

        result = SymbolResult(
            symbol=self.symbol,
            trades=self.trades,
            equity_curve=np.array(self.equity_curve),
            timestamps=np.array(self.timestamps),
            total_commission=self.total_commission,
        )
        result.crisis_state_log = self.crisis_state_log  # type: ignore[attr-defined]
        return result

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_loop(
        self,
        daily: NumpyBars,
        hourly: NumpyBars,
        four_hour: NumpyBars,
        daily_idx_map: np.ndarray,
        four_hour_idx_map: np.ndarray,
    ) -> None:
        warmup_d = self.cfg.warmup_daily
        warmup_h = self.cfg.warmup_hourly
        warmup_4h = self.cfg.warmup_4h

        self._daily_times = daily.times  # store for crisis state logging

        last_daily_idx = -1
        last_4h_idx = -1

        # Pre-compute daily indicators (full arrays)
        d_closes = daily.closes
        d_highs = daily.highs
        d_lows = daily.lows
        n_daily = len(d_closes)

        # Daily EMAs (full array)
        ema_fast_d = ema(d_closes, self.cfg.ema_fast_period)
        ema_slow_d = ema(d_closes, self.cfg.ema_slow_period)
        ema_fast_slope_d = compute_ema_slope(ema_fast_d, 5)
        ema_slow_slope_d = compute_ema_slope(ema_slow_d, 5)
        atr14_d_arr = atr(d_highs, d_lows, d_closes, 14)
        atr50_d_arr = atr(d_highs, d_lows, d_closes, 50)
        adx_d, plus_di_d, minus_di_d = adx_suite(d_highs, d_lows, d_closes, 14)

        # 4H indicators
        ema50_4h = ema(four_hour.closes, 50)
        atr14_4h = atr(four_hour.highs, four_hour.lows, four_hour.closes, 14)
        if len(four_hour.closes) > 14:
            adx14_4h_arr, _, _ = adx_suite(four_hour.highs, four_hour.lows, four_hour.closes, 14)
        else:
            adx14_4h_arr = np.zeros(len(four_hour.closes))

        # Hourly indicators (pre-compute full arrays)
        h_closes = hourly.closes
        h_highs = hourly.highs
        h_lows = hourly.lows
        h_opens = hourly.opens

        ema20_h = ema(h_closes, 20)
        ema34_h = ema(h_closes, 34)
        ema50_h = ema(h_closes, 50)
        atr14_h_arr = atr(h_highs, h_lows, h_closes, 14)
        donch_low_h = donchian_low(h_lows, self.cfg.s3_donchian_period)
        donch_low_bd = donchian_low(h_lows, self.cfg.bd_donchian_period)

        # Pre-compute new signal indicators
        swing_highs_h = swing_high_confirmed(h_highs, self.cfg.lh_swing_lookback)
        h_volumes = hourly.volumes if hasattr(hourly, 'volumes') else None
        vol_sma20_h = volume_sma(h_volumes, 20) if h_volumes is not None else None

        for i in range(len(hourly)):
            bar_time = self._to_datetime(hourly.times[i])
            O = h_opens[i]
            H = h_highs[i]
            L = h_lows[i]
            C = h_closes[i]

            # Skip NaN
            if np.isnan(O) or np.isnan(H) or np.isnan(L):
                self.equity_curve.append(self.equity)
                self.timestamps.append(hourly.times[i])
                continue

            # 1. Update daily state when d_idx changes
            d_idx = int(daily_idx_map[i])
            if d_idx != last_daily_idx and d_idx >= warmup_d and d_idx < n_daily:
                self._update_daily(
                    d_idx, d_closes, d_highs, d_lows, daily.volumes,
                    ema_fast_d, ema_slow_d, ema_fast_slope_d, ema_slow_slope_d,
                    atr14_d_arr, atr50_d_arr, adx_d, plus_di_d, minus_di_d,
                )
                last_daily_idx = d_idx

            # 2. Update 4H state
            fh_idx = int(four_hour_idx_map[i])
            if fh_idx != last_4h_idx and fh_idx >= warmup_4h and fh_idx < len(four_hour.closes):
                regime_4h_str, _ = compute_4h_structure(
                    four_hour.closes[:fh_idx + 1],
                    ema50_4h[:fh_idx + 1],
                    atr14_4h[:fh_idx + 1],
                    adx14_4h_arr[:fh_idx + 1],
                    float(four_hour.closes[fh_idx]),
                )
                self.daily_ctx.regime_4h = Regime4H(regime_4h_str)
                last_4h_idx = fh_idx

            if not self.daily_ctx.regime_on and self.daily_ctx.regime == BRSRegime.RANGE_CHOP:
                if d_idx < warmup_d:
                    self.equity_curve.append(self.equity)
                    self.timestamps.append(hourly.times[i])
                    continue

            # 3. Compute hourly context
            if i < warmup_h:
                self.equity_curve.append(self.equity)
                self.timestamps.append(hourly.times[i])
                continue

            h_ctx = HourlyContext(
                close=C,
                high=H,
                low=L,
                open=O,
                prior_high=float(h_highs[i - 1]) if i > 0 else 0.0,
                prior_low=float(h_lows[i - 1]) if i > 0 else 0.0,
                ema_pull=self._get_ema_pull(i, ema34_h, ema50_h),
                ema_mom=float(ema20_h[i]),
                atr14_h=float(atr14_h_arr[i]) if not np.isnan(atr14_h_arr[i]) else 0.0,
                avwap_h=0.0,  # computed in S2 arm state
                volume=float(h_volumes[i]) if h_volumes is not None else 0.0,
                volume_sma20=float(vol_sma20_h[i]) if vol_sma20_h is not None and not np.isnan(vol_sma20_h[i]) else 0.0,
                prior_close_3=float(h_closes[i - 3]) if i >= 3 else 0.0,
            )

            # Track last close for pyramid R-multiple calc (R9)
            self._last_close = C

            # 4. Position management or entry
            if self._in_position:
                self._manage_position(h_ctx, bar_time, i)
                # Pyramid check: if winning and pyramid allowed (R9)
                if (self._in_position
                        and self.cfg.pyramid_enabled
                        and self._pyramid_count < self.cfg.max_pyramid_adds):
                    cur_r = self._current_r()
                    if cur_r >= self.cfg.pyramid_min_r:
                        self._check_pyramid(h_ctx, bar_time, i)
            else:
                self._check_entries(h_ctx, bar_time, i, donch_low_h,
                                     d_closes, d_lows, d_highs, daily.volumes,
                                     atr14_d_arr, atr50_d_arr,
                                     ema_fast_slope_d, ema_slow_slope_d, d_idx,
                                     swing_highs_h, h_volumes, vol_sma20_h,
                                     donch_low_bd)

            self.equity_curve.append(self.equity)
            self.timestamps.append(hourly.times[i])

    # ------------------------------------------------------------------
    # Daily state update
    # ------------------------------------------------------------------

    def _update_daily(
        self,
        d_idx: int,
        d_closes: np.ndarray,
        d_highs: np.ndarray,
        d_lows: np.ndarray,
        d_volumes: np.ndarray,
        ema_fast_d: np.ndarray,
        ema_slow_d: np.ndarray,
        ema_fast_slope_d: np.ndarray,
        ema_slow_slope_d: np.ndarray,
        atr14_d_arr: np.ndarray,
        atr50_d_arr: np.ndarray,
        adx_d: np.ndarray,
        plus_di_d: np.ndarray,
        minus_di_d: np.ndarray,
    ) -> None:
        close = float(d_closes[d_idx])
        ef = float(ema_fast_d[d_idx])
        es = float(ema_slow_d[d_idx])
        ef_slope = float(ema_fast_slope_d[d_idx])
        es_slope = float(ema_slow_slope_d[d_idx])
        a14 = float(atr14_d_arr[d_idx]) if not np.isnan(atr14_d_arr[d_idx]) else 0.0
        a50 = float(atr50_d_arr[d_idx]) if not np.isnan(atr50_d_arr[d_idx]) else 0.0
        adx_val = float(adx_d[d_idx]) if not np.isnan(adx_d[d_idx]) else 0.0
        p_di = float(plus_di_d[d_idx]) if not np.isnan(plus_di_d[d_idx]) else 0.0
        m_di = float(minus_di_d[d_idx]) if not np.isnan(minus_di_d[d_idx]) else 0.0

        ema_sep_pct = abs(ef - es) / close * 100.0 if close > 0 else 0.0

        # Peak drawdown trigger (Part A — Round 6)
        # Uses d_highs (not closes) for rolling peak — captures intraday highs
        peak_drop_triggered = False
        if self.cfg.peak_drop_enabled and d_idx >= self.cfg.peak_drop_lookback:
            lookback_start = max(0, d_idx - self.cfg.peak_drop_lookback + 1)
            rolling_peak = float(np.max(d_highs[lookback_start:d_idx + 1]))
            if rolling_peak > 0:
                drop_pct = (close - rolling_peak) / rolling_peak
                if drop_pct <= self.cfg.peak_drop_pct:
                    peak_drop_triggered = True

        # Regime ON/OFF
        regime_on = compute_regime_on(
            adx_val, self._prev_regime_on,
            self.sym_cfg.adx_on, self.sym_cfg.adx_off,
        )
        # Peak drop forces regime_on
        if peak_drop_triggered:
            regime_on = True

        # GLD cross-symbol override: force GLD regime_on when QQQ is in bear regime (R8)
        if (self.cfg.gld_qqq_override_enabled
                and self.symbol == "GLD"
                and self._qqq_regimes
                and len(self._qqq_regimes) > d_idx):
            qqq_regime = self._qqq_regimes[d_idx]
            if (qqq_regime == BRSRegime.BEAR_STRONG
                    or (self.cfg.gld_qqq_override_bear_trend
                        and qqq_regime == BRSRegime.BEAR_TREND)):
                regime_on = True

        self._prev_regime_on = regime_on

        # Classify regime
        self.prev_regime = self.daily_ctx.regime
        regime = classify_regime(
            adx_val, p_di, m_di, close, ef, es,
            regime_on, self.cfg.adx_strong,
            bear_min_conditions=self.cfg.regime_bear_min_conditions,
        )

        # Bear conviction
        price_below = close < ef and close < es
        slope_neg = ef_slope < 0
        conviction = compute_bear_conviction(
            adx_val, m_di, p_di, ema_sep_pct, price_below, slope_neg,
        )

        # VolFactor (configurable via param_overrides)
        vf_min = self.cfg.resolve_param("vf_clamp_min") or 0.35
        vf_max = self.cfg.resolve_param("vf_clamp_max") or 1.5
        extreme_pct = self.cfg.resolve_param("extreme_vol_pct") or 95.0
        vf, vol_pct = compute_vol_factor(
            a14, atr14_d_arr[:d_idx + 1],
            vf_clamp_min=vf_min, vf_clamp_max=vf_max,
            extreme_vol_pct=extreme_pct,
        )
        risk_adj = compute_risk_regime(a14, atr14_d_arr[:d_idx + 1])

        # Bias update
        raw_dir = compute_raw_bias(close, ef, es)
        di_diff = m_di - p_di
        daily_return = (close - float(d_closes[d_idx - 1])) / float(d_closes[d_idx - 1]) if d_idx > 0 else 0.0
        atr_ratio = a14 / a50 if a50 > 0 else 1.0

        # Cumulative N-day return for Path G (always compute for crisis logging)
        cum_return_lookback = self.cfg.cum_return_lookback
        cum_return = 0.0
        if d_idx >= cum_return_lookback:
            lookback_close = float(d_closes[d_idx - cum_return_lookback])
            if lookback_close > 0:
                cum_return = (close - lookback_close) / lookback_close

        self._bias = update_bias(
            self._bias, regime, regime_on, raw_dir,
            conviction, adx_val, di_diff, ema_sep_pct,
            daily_return=daily_return,
            atr_ratio=atr_ratio,
            fast_crash_enabled=self.cfg.fast_crash_enabled,
            fast_crash_return_thresh=self.cfg.fast_crash_return_thresh,
            fast_crash_atr_ratio=self.cfg.fast_crash_atr_ratio,
            crash_override_enabled=self.cfg.crash_override_enabled,
            crash_override_return=self.cfg.crash_override_return,
            crash_override_atr_ratio=self.cfg.crash_override_atr_ratio,
            # Path F: return-only (Round 6 Part B)
            return_only_enabled=self.cfg.return_only_enabled,
            return_only_thresh=self.cfg.return_only_thresh,
            close=close,
            ema_fast=ef,
            # 4H bias accelerator (Round 6 Part C)
            bias_4h_accel_enabled=self.cfg.bias_4h_accel_enabled,
            bias_4h_accel_reduction=self.cfg.bias_4h_accel_reduction,
            regime_4h_bear=self.daily_ctx.regime_4h == Regime4H.BEAR,
            # Path G: cumulative return (Round 7)
            cum_return_enabled=self.cfg.cum_return_enabled,
            cum_return=cum_return,
            cum_return_thresh=self.cfg.cum_return_thresh,
            # R8: churn bridge
            churn_bridge_bars=self.cfg.churn_bridge_bars,
        )

        # Peak drop override: force instant SHORT confirmation (Part A — Round 6)
        if peak_drop_triggered and self._bias.confirmed_direction != Direction.SHORT:
            self._bias.confirmed_direction = Direction.SHORT
            self._bias.peak_drop_override = True

        # Persistence override counter: consecutive SHORT-bias bars without position (R8)
        if self._bias.confirmed_direction == Direction.SHORT and not self._in_position:
            self._short_bias_no_trade += 1
        else:
            self._short_bias_no_trade = 0

        # Track regime history for cross-symbol extraction
        self._regime_history.append(regime)

        # Build daily context
        self.daily_ctx = DailyContext(
            regime=regime,
            regime_on=regime_on,
            regime_4h=self.daily_ctx.regime_4h,  # preserve 4H state
            bias=self._bias,
            vol=VolState(
                vol_factor=vf,
                vol_pct=vol_pct,
                risk_regime=1.0 / risk_adj if risk_adj > 0 else 1.0,
                base_risk_adj=risk_adj,
                extreme_vol=vol_pct > extreme_pct,
            ),
            bear_conviction=conviction,
            adx=adx_val,
            plus_di=p_di,
            minus_di=m_di,
            ema_fast=ef,
            ema_slow=es,
            ema_fast_slope=ef_slope,
            ema_slow_slope=es_slope,
            ema_sep_pct=ema_sep_pct,
            atr14_d=a14,
            atr50_d=a50,
            atr_ratio=atr_ratio,
            close=close,
            persistence_active=(self.cfg.persistence_override_bars > 0
                                and self._short_bias_no_trade >= self.cfg.persistence_override_bars),
        )

        # Crisis state logging — only during crisis windows (memory-efficient)
        from backtests.swing.analysis.brs_diagnostics import CRISIS_WINDOWS
        bar_dt = self._to_datetime(self._daily_times[d_idx])
        bar_dt_naive = bar_dt.replace(tzinfo=None) if bar_dt.tzinfo else bar_dt
        for cname, cstart, cend in CRISIS_WINDOWS:
            if cstart <= bar_dt_naive <= cend:
                self.crisis_state_log.append({
                    "date": bar_dt_naive.strftime("%Y-%m-%d"),
                    "crisis": cname,
                    "regime_on": regime_on,
                    "regime": regime.value,
                    "adx": round(adx_val, 1),
                    "plus_di": round(p_di, 1),
                    "minus_di": round(m_di, 1),
                    "bias_raw": raw_dir.name,
                    "bias_confirmed": self._bias.confirmed_direction.name,
                    "hold_count": self._bias.hold_count,
                    "crash_override": self._bias.crash_override,
                    "peak_drop_override": self._bias.peak_drop_override,
                    "cum_return": round(cum_return * 100, 2),
                    "cum_return_override": self._bias.cum_return_override,
                    "daily_return": round(daily_return * 100, 2),
                    "atr_ratio": round(atr_ratio, 2),
                    "ema_fast": round(ef, 2),
                    "ema_slow": round(es, 2),
                    "close": round(close, 2),
                    "conviction": round(conviction, 1),
                    "in_position": self._in_position,
                })
                break

        # Check S2 arming on daily bar
        arm = check_s2_arm(
            d_closes, d_lows, d_highs, d_volumes,
            a14, a50, self.cfg, d_idx,
        )
        if arm is not None:
            self._s2_arm = arm

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _manage_position(self, h: HourlyContext, bar_time: datetime, bar_idx: int) -> None:
        if not self._in_position:
            return

        if self._is_rth(bar_time):
            self._pos_bars_held += 1

        self._hourly_highs.append(h.high)
        self._hourly_lows.append(h.low)

        # Update MFE/MAE
        rpu = self._pos_risk_per_unit
        if self._pos_direction == Direction.SHORT:
            if h.low < self._pos_mfe_price or self._pos_mfe_price == 0:
                self._pos_mfe_price = h.low
            if h.high > self._pos_mae_price or self._pos_mae_price == 0:
                self._pos_mae_price = h.high
            if rpu > 0:
                self._pos_mfe_r = max(self._pos_mfe_r,
                    (self._pos_entry_price - self._pos_mfe_price) / rpu)
                cur_r = (self._pos_entry_price - h.close) / rpu
            else:
                cur_r = 0.0
        else:
            if h.high > self._pos_mfe_price or self._pos_mfe_price == 0:
                self._pos_mfe_price = h.high
            if h.low < self._pos_mae_price or self._pos_mae_price == 0:
                self._pos_mae_price = h.low
            if rpu > 0:
                self._pos_mfe_r = max(self._pos_mfe_r,
                    (self._pos_mfe_price - self._pos_entry_price) / rpu)
                cur_r = (h.close - self._pos_entry_price) / rpu
            else:
                cur_r = 0.0

        # Scale-out check: exit tranche B at target (Change #6)
        if (self.cfg.scale_out_enabled
                and check_scale_out(cur_r, self.cfg.scale_out_target_r, self._pos_tranche_b_open)):
            self._partial_close(h.close, bar_time, cur_r)

        # Check exits
        is_long = self._pos_direction == Direction.LONG
        exit_result = check_exits(
            direction=self._pos_direction,
            entry_price=self._pos_entry_price,
            current_stop=self._pos_current_stop,
            risk_per_unit=rpu,
            bar_high=h.high,
            bar_low=h.low,
            bar_close=h.close,
            atr14_h=h.atr14_h,
            atr14_d=self.daily_ctx.atr14_d,
            bars_held=self._pos_bars_held,
            mfe_r=self._pos_mfe_r,
            cur_r=cur_r,
            be_triggered=self._pos_be_triggered,
            regime=self.daily_ctx.regime,
            sym_cfg=self.sym_cfg,
            catastrophic_cap_r=self.cfg.catastrophic_cap_r,
            be_trigger_r=self.cfg.be_trigger_r,
            trail_trigger_r=self.cfg.trail_trigger_r,
            stale_bars=self.cfg.stale_bars_short if not is_long else self.cfg.stale_bars_long,
            stale_early_bars=self.cfg.stale_early_bars,
            time_decay_hours=self.cfg.time_decay_hours,
            is_long=is_long,
            hourly_highs=self._hourly_highs,
            hourly_lows=self._hourly_lows,
            min_hold_bars=self.cfg.min_hold_bars,
        )

        if exit_result is not None:
            reason, price = exit_result
            self._close_position(price, bar_time, reason.value)
            # Regime-adaptive cooldown (Change #5)
            base_cd = self.sym_cfg.cooldown_bars
            regime = self.daily_ctx.regime if self.daily_ctx else None
            if regime == BRSRegime.BEAR_STRONG:
                cd = self.cfg.cooldown_bear_strong
            elif regime == BRSRegime.BEAR_TREND:
                cd = self.cfg.cooldown_bear_trend
            else:
                cd = base_cd
            self._cooldown_until_bar = bar_idx + cd
            return

        # Update trailing stop
        new_stop, new_be = update_trailing_stop(
            direction=self._pos_direction,
            current_stop=self._pos_current_stop,
            entry_price=self._pos_entry_price,
            risk_per_unit=rpu,
            mfe_r=self._pos_mfe_r,
            atr14_d=self.daily_ctx.atr14_d,
            atr14_h=h.atr14_h,
            be_triggered=self._pos_be_triggered,
            regime=self.daily_ctx.regime,
            prev_regime=self.prev_regime,
            sym_cfg=self.sym_cfg,
            be_trigger_r=self.cfg.be_trigger_r,
            trail_trigger_r=self.cfg.trail_trigger_r,
            profit_floor_scale=self.cfg.profit_floor_scale,
            hourly_highs=self._hourly_highs,
            hourly_lows=self._hourly_lows,
            chand_bonus=self._pos_chand_bonus,
            trail_regime_scaling=self.cfg.trail_regime_scaling,
        )
        self._pos_current_stop = new_stop
        self._pos_be_triggered = new_be

    # ------------------------------------------------------------------
    # Entry checks
    # ------------------------------------------------------------------

    def _check_entries(
        self,
        h: HourlyContext,
        bar_time: datetime,
        bar_idx: int,
        donch_low_h: np.ndarray,
        d_closes: np.ndarray,
        d_lows: np.ndarray,
        d_highs: np.ndarray,
        d_volumes: np.ndarray,
        atr14_d_arr: np.ndarray,
        atr50_d_arr: np.ndarray,
        ema_fast_slope_d: np.ndarray,
        ema_slow_slope_d: np.ndarray,
        d_idx: int,
        swing_highs_h: np.ndarray | None = None,
        h_volumes: np.ndarray | None = None,
        vol_sma20_h: np.ndarray | None = None,
        donch_low_bd: np.ndarray | None = None,
    ) -> None:
        if self._in_position:
            return
        if not self._is_rth(bar_time):
            return
        if bar_idx < self._cooldown_until_bar:
            return

        d = self.daily_ctx

        # TOD filter: no entries 14:00-15:30 ET (lunch reversal trap zone)
        if self.cfg.tod_filter_enabled:
            dt_et = bar_time.astimezone(self._get_et_zone())
            tod_minutes = dt_et.hour * 60 + dt_et.minute
            if self.cfg.tod_filter_start <= tod_minutes < self.cfg.tod_filter_end:
                return

        # Expire arms
        if self._s2_arm.armed and bar_idx > self._s2_arm.armed_until_bar:
            self._s2_arm = S2ArmState()
        if self._s3_arm.armed and bar_idx > self._s3_arm.armed_until_bar:
            self._s3_arm = S3ArmState()
        if self._lh_arm.armed and bar_idx > self._lh_arm.armed_until_bar:
            self._lh_arm = LHArmState()
        if self._bd_arm.armed and bar_idx > self._bd_arm.armed_until_bar:
            self._bd_arm = BDArmState()

        # Swing high tracking for LH arming
        if swing_highs_h is not None and bar_idx < len(swing_highs_h):
            sh_val = swing_highs_h[bar_idx]
            if not np.isnan(sh_val):
                # New confirmed swing high — check for lower-high
                if self._swing_highs and sh_val < self._swing_highs[-1]:
                    # Lower high confirmed → arm LH
                    self._lh_arm = LHArmState(
                        armed=True,
                        swing_high_price=sh_val,
                        prior_swing_high_price=self._swing_highs[-1],
                        armed_bar=bar_idx,
                        armed_until_bar=bar_idx + self.cfg.lh_arm_bars,
                    )
                self._swing_highs.append(sh_val)

        # BD arming check (hourly Donchian break with volume)
        # Use prior bar's Donchian low — current bar is included in donch_low_h[bar_idx],
        # so low < donch_low_h[bar_idx] is impossible. Compare against bar_idx-1 instead.
        bd_donch = donch_low_bd if donch_low_bd is not None else donch_low_h
        if not self._bd_arm.armed and bar_idx >= 1 and bar_idx < len(bd_donch):
            vol_val = float(h_volumes[bar_idx]) if h_volumes is not None else 0.0
            vsma_val = float(vol_sma20_h[bar_idx]) if vol_sma20_h is not None and not np.isnan(vol_sma20_h[bar_idx]) else 0.0
            prior_donch = float(bd_donch[bar_idx - 1])
            # R8: allow BD arming in RANGE_CHOP when chop override conditions met
            chop_arm_ok = (self.cfg.chop_short_entry_enabled
                           and self._bias.peak_drop_override
                           and self._bias.confirmed_direction == Direction.SHORT)
            arm_bd = check_bd_arm(
                h.low, h.high, h.close, vol_val,
                prior_donch, vsma_val,
                d.regime, self.cfg, bar_idx,
                chop_arm_allowed=chop_arm_ok,
                persistence_override=d.persistence_active,
            )
            if arm_bd is not None:
                self._bd_arm = arm_bd

        # Check S3 arming (Donchian breakout) — use prior bar's Donchian (same fix as BD)
        if not self._s3_arm.armed and bar_idx >= 1 and bar_idx < len(donch_low_h):
            arm3 = check_s3_arm(
                d.regime, d.bias.confirmed_direction == Direction.SHORT,
                h.low, h.close, float(donch_low_h[bar_idx - 1]), h.ema_mom, bar_idx,
            )
            if arm3 is not None:
                self._s3_arm = arm3

        ef_slope = float(ema_fast_slope_d[d_idx]) if d_idx < len(ema_fast_slope_d) else 0.0
        es_slope = float(ema_slow_slope_d[d_idx]) if d_idx < len(ema_slow_slope_d) else 0.0

        # Try signals in priority order: LH > BD > S2 > S1 > S3 > L1
        signals: list[EntrySignal | None] = [
            check_lh_rejection(d, h, self.sym_cfg, self.cfg, self._lh_arm),
            check_bd_continuation(d, h, self.sym_cfg, self.cfg, self._bd_arm),
            check_s2(d, h, self.sym_cfg, self.cfg, self._s2_arm),
            check_s1(d, h, self.sym_cfg, self.cfg),
            check_s3(d, h, self.sym_cfg, self.cfg, self._s3_arm),
            check_l1(d, h, self.sym_cfg, self.cfg, self.symbol, ef_slope, es_slope),
        ]

        for sig in signals:
            if sig is not None:
                self._enter_position(sig, h, bar_time)
                return

    # ------------------------------------------------------------------
    # Enter position
    # ------------------------------------------------------------------

    def _enter_position(self, signal: EntrySignal, h: HourlyContext, bar_time: datetime) -> None:
        qty = compute_position_size(
            signal, self.equity, self.sym_cfg, self.daily_ctx, self.cfg,
            self._current_heat_r, self._concurrent_positions,
        )
        if qty <= 0:
            return

        # Apply slippage
        entry_price = signal.signal_price
        if signal.direction == Direction.SHORT:
            entry_price -= 0.01  # 1 tick adverse
        else:
            entry_price += 0.01

        # Recompute stop from actual entry
        stop_price = compute_initial_stop(
            signal.direction, entry_price,
            signal.signal_high, signal.signal_low,
            self.daily_ctx.atr14_d, h.atr14_h, self.sym_cfg,
        )
        # Change #7: Wider stops in BEAR_STRONG to survive intraday noise
        if self.daily_ctx.regime == BRSRegime.BEAR_STRONG and h.atr14_h > 0:
            min_floor = self.sym_cfg.stop_floor_atr * self.cfg.stop_floor_bear_strong_mult * h.atr14_h
            current_dist = abs(stop_price - entry_price)
            if current_dist < min_floor:
                if signal.direction == Direction.SHORT:
                    stop_price = entry_price + min_floor
                else:
                    stop_price = entry_price - min_floor
        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit <= 0:
            return

        # Commission
        commission = qty * self._slip.commission_per_share_etf
        self.equity -= commission
        self.total_commission += commission

        # Set position state
        self._in_position = True
        self._pos_direction = signal.direction
        self._pos_entry_price = entry_price
        self._pos_initial_stop = stop_price
        self._pos_current_stop = stop_price
        self._pos_qty = qty
        self._pos_entry_time = bar_time
        self._pos_risk_per_unit = risk_per_unit
        self._pos_bars_held = 0
        self._pos_mfe_price = 0.0
        self._pos_mfe_r = 0.0
        self._pos_mae_price = 0.0
        self._pos_be_triggered = False
        self._pos_regime_entry = signal.regime_at_entry.value
        self._pos_entry_type = signal.entry_type.value
        self._pos_adx_entry = self.daily_ctx.adx
        self._pos_score_entry = signal.bear_conviction
        self._pos_di_agrees = self.daily_ctx.minus_di > self.daily_ctx.plus_di
        self._pos_quality = signal.quality_score
        self._pos_chand_bonus = 0.0
        self._pos_tranche_b_open = self.cfg.scale_out_enabled
        self._pos_original_qty = qty
        self._hourly_highs = []
        self._hourly_lows = []
        self._concurrent_positions += 1

        # Disarm if entered from armed state
        if signal.entry_type == EntryType.S2_BREAKDOWN:
            self._s2_arm = S2ArmState()
        elif signal.entry_type == EntryType.S3_IMPULSE:
            self._s3_arm = S3ArmState()
        elif signal.entry_type == EntryType.LH_REJECTION:
            self._lh_arm = LHArmState()
        elif signal.entry_type == EntryType.BD_CONTINUATION:
            self._bd_arm = BDArmState()

    # ------------------------------------------------------------------
    # Close position
    # ------------------------------------------------------------------

    def _close_position(self, exit_price: float, bar_time: datetime, reason: str) -> None:
        if not self._in_position:
            return

        d = self._pos_direction
        entry = self._pos_entry_price
        rpu = self._pos_risk_per_unit

        # PnL
        if d == Direction.SHORT:
            pnl_pts = entry - exit_price
            mae_pts = self._pos_mae_price - entry if self._pos_mae_price > 0 else 0.0
        else:
            pnl_pts = exit_price - entry
            mae_pts = entry - self._pos_mae_price if self._pos_mae_price > 0 else 0.0

        pnl_dollars = pnl_pts * self._pos_qty
        r_mult = pnl_pts / rpu if rpu > 0 else 0.0
        mae_r = mae_pts / rpu if rpu > 0 else 0.0

        # Exit commission
        commission = self._pos_qty * self._slip.commission_per_share_etf
        self.total_commission += commission

        trade = TradeRecord(
            symbol=self.symbol,
            direction=int(d.value),
            entry_type=self._pos_entry_type,
            entry_time=self._pos_entry_time,
            exit_time=bar_time,
            entry_price=entry,
            exit_price=exit_price,
            qty=self._pos_qty,
            initial_stop=self._pos_initial_stop,
            exit_reason=reason,
            pnl_points=pnl_pts,
            pnl_dollars=pnl_dollars,
            r_multiple=r_mult,
            mfe_r=self._pos_mfe_r,
            mae_r=mae_r,
            bars_held=self._pos_bars_held,
            commission=commission,
            adx_entry=self._pos_adx_entry,
            score_entry=self._pos_score_entry,
            di_agrees=self._pos_di_agrees,
            quality_score=self._pos_quality,
            regime_entry=self._pos_regime_entry,
        )
        self.trades.append(trade)
        self.equity += pnl_dollars - commission

        # Reset
        self._in_position = False
        self._pos_direction = Direction.FLAT
        self._pos_entry_price = 0.0
        self._pos_initial_stop = 0.0
        self._pos_current_stop = 0.0
        self._pos_qty = 0
        self._pos_entry_time = None
        self._pos_risk_per_unit = 0.0
        self._pos_bars_held = 0
        self._pos_mfe_price = 0.0
        self._pos_mfe_r = 0.0
        self._pos_mae_price = 0.0
        self._pos_be_triggered = False
        self._pos_regime_entry = ""
        self._pos_entry_type = ""
        self._pos_chand_bonus = 0.0
        self._pos_tranche_b_open = False
        self._pos_original_qty = 0
        self._pyramid_count = 0
        self._hourly_highs = []
        self._hourly_lows = []
        self._concurrent_positions = max(0, self._concurrent_positions - 1)

    # ------------------------------------------------------------------
    # Partial close (scale-out) — Change #6
    # ------------------------------------------------------------------

    def _partial_close(self, exit_price: float, bar_time: datetime, cur_r: float) -> None:
        """Exit tranche B (partial position) at target R-multiple."""
        if not self._in_position or not self._pos_tranche_b_open:
            return

        # Compute tranche B qty
        tranche_b_qty = int(math.floor(self._pos_original_qty * self.cfg.scale_out_pct))
        if tranche_b_qty <= 0:
            self._pos_tranche_b_open = False
            return

        d = self._pos_direction
        entry = self._pos_entry_price
        rpu = self._pos_risk_per_unit

        # PnL for tranche B
        if d == Direction.SHORT:
            pnl_pts = entry - exit_price
            mae_pts = self._pos_mae_price - entry if self._pos_mae_price > 0 else 0.0
        else:
            pnl_pts = exit_price - entry
            mae_pts = entry - self._pos_mae_price if self._pos_mae_price > 0 else 0.0

        pnl_dollars = pnl_pts * tranche_b_qty
        r_mult = pnl_pts / rpu if rpu > 0 else 0.0
        mae_r = mae_pts / rpu if rpu > 0 else 0.0

        # Commission
        commission = tranche_b_qty * self._slip.commission_per_share_etf
        self.total_commission += commission

        # Record tranche B exit as separate trade
        trade = TradeRecord(
            symbol=self.symbol,
            direction=int(d.value),
            entry_type=self._pos_entry_type,
            entry_time=self._pos_entry_time,
            exit_time=bar_time,
            entry_price=entry,
            exit_price=exit_price,
            qty=tranche_b_qty,
            initial_stop=self._pos_initial_stop,
            exit_reason="TARGET",
            pnl_points=pnl_pts,
            pnl_dollars=pnl_dollars,
            r_multiple=r_mult,
            mfe_r=self._pos_mfe_r,
            mae_r=mae_r,
            bars_held=self._pos_bars_held,
            commission=commission,
            adx_entry=self._pos_adx_entry,
            score_entry=self._pos_score_entry,
            di_agrees=self._pos_di_agrees,
            quality_score=self._pos_quality,
            regime_entry=self._pos_regime_entry,
        )
        self.trades.append(trade)
        self.equity += pnl_dollars - commission

        # Reduce remaining position
        self._pos_qty -= tranche_b_qty
        self._pos_tranche_b_open = False

        # Widen trailing stop for remaining tranche A
        self._pos_chand_bonus = self.cfg.scale_out_trail_bonus

        # If nothing left (edge case: very small positions), clean up fully
        if self._pos_qty <= 0:
            self._in_position = False
            self._pos_direction = Direction.FLAT
            self._concurrent_positions = max(0, self._concurrent_positions - 1)

    # ------------------------------------------------------------------
    # Pyramiding (R9)
    # ------------------------------------------------------------------

    def _current_r(self) -> float:
        """Current R-multiple of open position."""
        if not self._in_position or self._pos_risk_per_unit <= 0:
            return 0.0
        if self._pos_direction == Direction.SHORT:
            return (self._pos_entry_price - self._last_close) / self._pos_risk_per_unit
        return (self._last_close - self._pos_entry_price) / self._pos_risk_per_unit

    def _check_pyramid(self, h: HourlyContext, bar_time: datetime, bar_idx: int) -> None:
        """Check for add-on entry signals while in a winning position."""
        d = self.daily_ctx
        signals = [
            check_lh_rejection(d, h, self.sym_cfg, self.cfg, self._lh_arm),
            check_bd_continuation(d, h, self.sym_cfg, self.cfg, self._bd_arm),
        ]
        for sig in signals:
            if sig is not None and sig.direction == self._pos_direction:
                self._pyramid_add(sig, h, bar_time)
                return

    def _pyramid_add(self, signal: EntrySignal, h: HourlyContext, bar_time: datetime) -> None:
        """Add to existing position on new signal confirmation."""
        add_qty = int(math.floor(self._pos_original_qty * self.cfg.pyramid_scale))
        if add_qty <= 0:
            return

        # Heat cap check (don't exceed portfolio risk limits)
        if self._pos_risk_per_unit > 0:
            base_risk = self.equity * self.sym_cfg.base_risk_pct
            if base_risk > 0:
                add_risk_r = add_qty * self._pos_risk_per_unit / base_risk
                if self._current_heat_r + add_risk_r > self.cfg.heat_cap_r:
                    return

        # Blend entry price
        old_cost = self._pos_entry_price * self._pos_qty
        add_price = signal.signal_price
        if signal.direction == Direction.SHORT:
            add_price -= 0.01
        else:
            add_price += 0.01

        new_qty = self._pos_qty + add_qty
        new_entry = (old_cost + add_price * add_qty) / new_qty

        # Update position state
        self._pos_entry_price = new_entry
        self._pos_qty = new_qty
        self._pos_risk_per_unit = abs(new_entry - self._pos_current_stop)
        self._pyramid_count += 1

        # Commission
        commission = add_qty * self._slip.commission_per_share_etf
        self.equity -= commission
        self.total_commission += commission

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_ema_pull(self, i: int, ema34: np.ndarray, ema50: np.ndarray) -> float:
        """Get regime-adaptive EMA_pull value."""
        if self.daily_ctx.regime == BRSRegime.BEAR_STRONG:
            return float(ema34[i]) if not np.isnan(ema34[i]) else 0.0
        return float(ema50[i]) if not np.isnan(ema50[i]) else 0.0

    _ET_ZONE = None

    @classmethod
    def _get_et_zone(cls):
        if cls._ET_ZONE is None:
            from zoneinfo import ZoneInfo
            cls._ET_ZONE = ZoneInfo("America/New_York")
        return cls._ET_ZONE

    @staticmethod
    def _is_rth(dt: datetime) -> bool:
        et_zone = BRSEngine._get_et_zone()
        dt_et = dt.astimezone(et_zone)
        if dt_et.weekday() >= 5:
            return False
        t = dt_et.hour * 60 + dt_et.minute
        return 570 <= t < 960  # 09:30-16:00

    @staticmethod
    def _to_datetime(ts) -> datetime:
        if isinstance(ts, datetime):
            return ts
        if hasattr(ts, 'astype'):
            unix_epoch = np.datetime64(0, 'ns')
            one_second = np.timedelta64(1, 's')
            seconds = (ts - unix_epoch) / one_second
            return datetime.fromtimestamp(float(seconds), tz=timezone.utc)
        return datetime.now(timezone.utc)
