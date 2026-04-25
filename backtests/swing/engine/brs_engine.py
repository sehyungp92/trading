"""BRS single-symbol backtest engine with broker-routed execution."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace as _dc_replace
from datetime import datetime, timezone

import numpy as np

from backtest.config_brs import BRSConfig, BRSSymbolConfig
from backtest.data.preprocessing import NumpyBars
from backtest.engine.backtest_engine import SymbolResult, TradeRecord
from backtest.engine.sim_broker import (
    FillResult,
    FillStatus,
    OrderSide,
    OrderType,
    SimBroker,
    SimOrder,
)

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

_TICK_SIZE = 0.01


@dataclass
class _PendingEntry:
    order_id: str
    signal: EntrySignal
    qty: int
    signal_time: datetime
    signal_bar_index: int
    signal_atr14_h: float
    signal_atr14_d: float
    regime_at_signal: BRSRegime
    adx_entry: float
    di_agrees: bool
    quality_score: float


@dataclass
class _PendingAdd:
    order_id: str
    signal: EntrySignal
    qty: int
    signal_time: datetime
    signal_bar_index: int


class BRSEngine:
    """Single-symbol BRS backtest engine."""

    def __init__(
        self,
        symbol: str,
        sym_cfg: BRSSymbolConfig,
        cfg: BRSConfig,
        *,
        starting_equity: float | None = None,
    ):
        self.symbol = symbol
        self.sym_cfg = sym_cfg
        self.cfg = cfg

        self.equity = cfg.initial_equity if starting_equity is None else starting_equity
        self._starting_equity = self.equity
        self._sizing_equity = max(cfg.initial_equity, 1.0)
        self._last_mark_price: float = 0.0

        broker_slip = _dc_replace(
            cfg.slippage,
            commission_per_contract=cfg.slippage.commission_per_share_etf,
        )
        self.broker = SimBroker(slippage_config=broker_slip)

        self.daily_ctx = DailyContext()
        self.prev_regime = BRSRegime.RANGE_CHOP
        self._prev_regime_on = False
        self._bias = BiasState()
        self._short_bias_no_trade: int = 0

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
        self._pos_chand_bonus: float = 0.0
        self._pos_tranche_b_open: bool = False
        self._pos_original_qty: int = 0
        self._pos_open_commission_basis: float = 0.0
        self._pos_signal_time: datetime | None = None
        self._pos_fill_time: datetime | None = None
        self._pos_signal_bar_index: int = -1
        self._pos_fill_bar_index: int = -1
        self._campaign_id: str = ""
        self._protective_stop_live: bool = False
        self._hourly_highs: list[float] = []
        self._hourly_lows: list[float] = []

        self._pyramid_count: int = 0
        self._last_close: float = 0.0

        self._s2_arm = S2ArmState()
        self._s3_arm = S3ArmState()
        self._lh_arm = LHArmState()
        self._bd_arm = BDArmState()
        self._swing_highs: list[float] = []

        self._pending_entry: _PendingEntry | None = None
        self._pending_add: _PendingAdd | None = None
        self._pending_exit_order_id: str | None = None
        self._pending_exit_reason: str = ""
        self._pending_partial_order_id: str | None = None
        self._pending_partial_qty: int = 0
        self._campaign_seq: int = 0

        self._cooldown_until_bar: int = 0
        self._concurrent_positions: int = 0
        self._current_heat_r: float = 0.0

        self.trades: list[TradeRecord] = []
        self.equity_curve: list[float] = []
        self.timestamps: list = []
        self.total_commission: float = 0.0

        self._qqq_regimes: list[BRSRegime] | None = None
        self._regime_history: list[BRSRegime] = []
        self.crisis_state_log: list[dict] = []

        self._prepared = False
        self._last_daily_idx = -1
        self._last_4h_idx = -1

    def run(
        self,
        daily: NumpyBars,
        hourly: NumpyBars,
        four_hour: NumpyBars,
        daily_idx_map: np.ndarray,
        four_hour_idx_map: np.ndarray,
    ) -> SymbolResult:
        """Run the backtest over all hourly bars."""
        self._prepare_run_context(
            daily=daily,
            hourly=hourly,
            four_hour=four_hour,
            daily_idx_map=daily_idx_map,
            four_hour_idx_map=four_hour_idx_map,
        )

        for bar_idx in range(len(self._hourly.closes)):
            self._step_bar(bar_idx)

        self._finalize_end_of_data()
        if self.equity_curve:
            self.equity_curve[-1] = self._mtm_equity()
        return self._build_result()

    def _build_result(self, *, equity_offset: float = 0.0) -> SymbolResult:
        result = SymbolResult(
            symbol=self.symbol,
            trades=self.trades,
            equity_curve=np.array(self.equity_curve, dtype=np.float64) + equity_offset,
            timestamps=np.array(self.timestamps),
            total_commission=self.total_commission,
        )
        result.crisis_state_log = self.crisis_state_log  # type: ignore[attr-defined]
        return result

    def _prepare_run_context(
        self,
        daily: NumpyBars,
        hourly: NumpyBars,
        four_hour: NumpyBars,
        daily_idx_map: np.ndarray,
        four_hour_idx_map: np.ndarray,
    ) -> None:
        self._daily = daily
        self._hourly = hourly
        self._four_hour = four_hour
        self._daily_idx_map = daily_idx_map
        self._four_hour_idx_map = four_hour_idx_map

        self._warmup_d = self.cfg.warmup_daily
        self._warmup_h = self.cfg.warmup_hourly
        self._warmup_4h = self.cfg.warmup_4h
        self._n_daily = len(daily.closes)
        self._daily_times = daily.times
        self._hourly_datetimes = [self._to_datetime(ts) for ts in hourly.times]

        self._d_closes = daily.closes
        self._d_highs = daily.highs
        self._d_lows = daily.lows
        self._d_volumes = daily.volumes

        self._ema_fast_d = ema(self._d_closes, self.cfg.ema_fast_period)
        self._ema_slow_d = ema(self._d_closes, self.cfg.ema_slow_period)
        self._ema_fast_slope_d = compute_ema_slope(self._ema_fast_d, 5)
        self._ema_slow_slope_d = compute_ema_slope(self._ema_slow_d, 5)
        self._atr14_d_arr = atr(self._d_highs, self._d_lows, self._d_closes, 14)
        self._atr50_d_arr = atr(self._d_highs, self._d_lows, self._d_closes, 50)
        self._adx_d, self._plus_di_d, self._minus_di_d = adx_suite(
            self._d_highs, self._d_lows, self._d_closes, 14,
        )

        self._ema50_4h = ema(four_hour.closes, 50)
        self._atr14_4h = atr(four_hour.highs, four_hour.lows, four_hour.closes, 14)
        if len(four_hour.closes) > 14:
            self._adx14_4h_arr, _, _ = adx_suite(
                four_hour.highs,
                four_hour.lows,
                four_hour.closes,
                14,
            )
        else:
            self._adx14_4h_arr = np.zeros(len(four_hour.closes))

        self._h_opens = hourly.opens
        self._h_highs = hourly.highs
        self._h_lows = hourly.lows
        self._h_closes = hourly.closes
        self._h_volumes = hourly.volumes if hasattr(hourly, "volumes") else None

        self._ema20_h = ema(self._h_closes, 20)
        self._ema34_h = ema(self._h_closes, 34)
        self._ema50_h = ema(self._h_closes, 50)
        self._atr14_h_arr = atr(self._h_highs, self._h_lows, self._h_closes, 14)
        self._donch_low_h = donchian_low(self._h_lows, self.cfg.s3_donchian_period)
        self._donch_low_bd = donchian_low(self._h_lows, self.cfg.bd_donchian_period)
        self._swing_highs_h = swing_high_confirmed(self._h_highs, self.cfg.lh_swing_lookback)
        self._vol_sma20_h = volume_sma(self._h_volumes, 20) if self._h_volumes is not None else None

        self._prepared = True

    def _step_bar(
        self,
        bar_idx: int,
        *,
        sizing_equity: float | None = None,
        shared_reserved_risk_dollars: float | None = None,
        shared_concurrent_positions: int | None = None,
    ) -> None:
        if not self._prepared:
            raise RuntimeError("BRS engine must be prepared before stepping bars.")

        bar_time = self._hourly_datetimes[bar_idx]
        O = float(self._h_opens[bar_idx])
        H = float(self._h_highs[bar_idx])
        L = float(self._h_lows[bar_idx])
        C = float(self._h_closes[bar_idx])

        if not np.isnan(C):
            self._last_close = C
            self._last_mark_price = C

        if np.isnan(O) or np.isnan(H) or np.isnan(L):
            current_price = C if not np.isnan(C) else self._last_mark_price
            self.equity_curve.append(self._mtm_equity(current_price))
            self.timestamps.append(self._hourly.times[bar_idx])
            return

        self._activate_protective_stop_if_due(bar_time)
        fills = self.broker.process_bar(
            self.symbol,
            bar_time,
            O,
            H,
            L,
            C,
            _TICK_SIZE,
        )
        for fill in self._sort_fills(fills):
            self._handle_fill(fill, bar_time, bar_idx)

        d_idx = int(self._daily_idx_map[bar_idx])
        if d_idx != self._last_daily_idx and d_idx >= self._warmup_d and d_idx < self._n_daily:
            self._update_daily(
                d_idx,
                self._d_closes,
                self._d_highs,
                self._d_lows,
                self._d_volumes,
                self._ema_fast_d,
                self._ema_slow_d,
                self._ema_fast_slope_d,
                self._ema_slow_slope_d,
                self._atr14_d_arr,
                self._atr50_d_arr,
                self._adx_d,
                self._plus_di_d,
                self._minus_di_d,
            )
            self._last_daily_idx = d_idx

        fh_idx = int(self._four_hour_idx_map[bar_idx])
        if fh_idx != self._last_4h_idx and fh_idx >= self._warmup_4h and fh_idx < len(self._four_hour.closes):
            regime_4h_str, _ = compute_4h_structure(
                self._four_hour.closes[:fh_idx + 1],
                self._ema50_4h[:fh_idx + 1],
                self._atr14_4h[:fh_idx + 1],
                self._adx14_4h_arr[:fh_idx + 1],
                float(self._four_hour.closes[fh_idx]),
            )
            self.daily_ctx.regime_4h = Regime4H(regime_4h_str)
            self._last_4h_idx = fh_idx

        if not self.daily_ctx.regime_on and self.daily_ctx.regime == BRSRegime.RANGE_CHOP:
            if d_idx < self._warmup_d:
                self.equity_curve.append(self._mtm_equity(C))
                self.timestamps.append(self._hourly.times[bar_idx])
                return

        if bar_idx < self._warmup_h:
            self.equity_curve.append(self._mtm_equity(C))
            self.timestamps.append(self._hourly.times[bar_idx])
            return

        self._sizing_equity = max(
            sizing_equity if sizing_equity is not None else self._mtm_equity(C),
            1.0,
        )
        reserved_risk = (
            shared_reserved_risk_dollars
            if shared_reserved_risk_dollars is not None
            else self.reserved_risk_dollars()
        )
        risk_unit_dollars = self._sizing_equity * self.sym_cfg.base_risk_pct
        self._current_heat_r = reserved_risk / risk_unit_dollars if risk_unit_dollars > 0 else 0.0
        self._concurrent_positions = (
            shared_concurrent_positions
            if shared_concurrent_positions is not None
            else self.reserved_position_slots()
        )

        h_ctx = HourlyContext(
            close=C,
            high=H,
            low=L,
            open=O,
            prior_high=float(self._h_highs[bar_idx - 1]) if bar_idx > 0 else 0.0,
            prior_low=float(self._h_lows[bar_idx - 1]) if bar_idx > 0 else 0.0,
            ema_pull=self._get_ema_pull(bar_idx, self._ema34_h, self._ema50_h),
            ema_mom=float(self._ema20_h[bar_idx]),
            atr14_h=float(self._atr14_h_arr[bar_idx]) if not np.isnan(self._atr14_h_arr[bar_idx]) else 0.0,
            avwap_h=0.0,
            volume=float(self._h_volumes[bar_idx]) if self._h_volumes is not None else 0.0,
            volume_sma20=float(self._vol_sma20_h[bar_idx])
            if self._vol_sma20_h is not None and not np.isnan(self._vol_sma20_h[bar_idx])
            else 0.0,
            prior_close_3=float(self._h_closes[bar_idx - 3]) if bar_idx >= 3 else 0.0,
        )

        if self._in_position:
            self._manage_position(h_ctx, bar_time, bar_idx)
            if (
                self._in_position
                and self.cfg.pyramid_enabled
                and self._pending_add is None
                and self._pending_exit_order_id is None
                and self._pyramid_count < self.cfg.max_pyramid_adds
                and self._current_r() >= self.cfg.pyramid_min_r
            ):
                self._check_pyramid(h_ctx, bar_time, bar_idx)
        elif self._pending_entry is None:
            self._check_entries(
                h_ctx,
                bar_time,
                bar_idx,
                self._donch_low_h,
                self._d_closes,
                self._d_lows,
                self._d_highs,
                self._d_volumes,
                self._atr14_d_arr,
                self._atr50_d_arr,
                self._ema_fast_slope_d,
                self._ema_slow_slope_d,
                d_idx,
                self._swing_highs_h,
                self._h_volumes,
                self._vol_sma20_h,
                self._donch_low_bd,
            )

        self.equity_curve.append(self._mtm_equity(C))
        self.timestamps.append(self._hourly.times[bar_idx])

    def unrealized_pnl(self, mark_price: float | None = None) -> float:
        if not self._in_position:
            return 0.0
        price = self._last_mark_price if mark_price is None else mark_price
        if price == 0.0 or np.isnan(price):
            return 0.0
        direction = 1 if self._pos_direction == Direction.LONG else -1
        return (price - self._pos_entry_price) * direction * self._pos_qty

    def reserved_risk_dollars(self) -> float:
        reserved = 0.0
        if self._in_position and self._pos_qty > 0 and self._pos_current_stop > 0:
            reserved += abs(self._pos_entry_price - self._pos_current_stop) * self._pos_qty
        if self._pending_entry is not None:
            reserved += self._pending_entry.signal.risk_per_unit * self._pending_entry.qty
        if self._pending_add is not None and self._pos_risk_per_unit > 0:
            reserved += self._pos_risk_per_unit * self._pending_add.qty
        return reserved

    def reserved_position_slots(self) -> int:
        return 1 if self._in_position or self._pending_entry is not None else 0

    @staticmethod
    def _sort_fills(fills: list[FillResult]) -> list[FillResult]:
        priorities = {
            "protective_stop": 0,
            "exit": 1,
            "partial": 2,
            "add_on": 3,
            "entry": 4,
        }
        return sorted(fills, key=lambda fill: priorities.get(fill.order.tag, 99))

    def _handle_fill(self, fill: FillResult, bar_time: datetime, bar_idx: int) -> None:
        if fill.status in (FillStatus.EXPIRED, FillStatus.REJECTED, FillStatus.CANCELLED):
            if self._pending_entry is not None and fill.order.order_id == self._pending_entry.order_id:
                self._pending_entry = None
            if self._pending_add is not None and fill.order.order_id == self._pending_add.order_id:
                self._pending_add = None
            if self._pending_exit_order_id == fill.order.order_id:
                self._pending_exit_order_id = None
                self._pending_exit_reason = ""
            if self._pending_partial_order_id == fill.order.order_id:
                self._pending_partial_order_id = None
                self._pending_partial_qty = 0
            return

        if fill.status != FillStatus.FILLED:
            return

        if fill.order.tag == "protective_stop":
            self._on_stop_fill(fill, bar_time, bar_idx)
        elif fill.order.tag == "exit":
            self._on_exit_fill(fill, bar_time, bar_idx)
        elif fill.order.tag == "partial":
            self._on_partial_fill(fill, bar_time, bar_idx)
        elif fill.order.tag == "add_on":
            self._on_add_fill(fill, bar_time, bar_idx)
        elif fill.order.tag == "entry":
            self._on_entry_fill(fill, bar_time, bar_idx)

    def _on_entry_fill(self, fill: FillResult, bar_time: datetime, bar_idx: int) -> None:
        pending = self._pending_entry
        if pending is None or fill.order.order_id != pending.order_id:
            return

        self._pending_entry = None
        self.total_commission += fill.commission
        self.equity -= fill.commission

        stop_price = compute_initial_stop(
            pending.signal.direction,
            fill.fill_price,
            pending.signal.signal_high,
            pending.signal.signal_low,
            pending.signal_atr14_d,
            pending.signal_atr14_h,
            self.sym_cfg,
        )
        if pending.regime_at_signal == BRSRegime.BEAR_STRONG and pending.signal_atr14_h > 0:
            min_floor = (
                self.sym_cfg.stop_floor_atr
                * self.cfg.stop_floor_bear_strong_mult
                * pending.signal_atr14_h
            )
            current_dist = abs(stop_price - fill.fill_price)
            if current_dist < min_floor:
                if pending.signal.direction == Direction.SHORT:
                    stop_price = fill.fill_price + min_floor
                else:
                    stop_price = fill.fill_price - min_floor

        risk_per_unit = abs(fill.fill_price - stop_price)
        if risk_per_unit <= 0:
            return

        self._campaign_seq += 1
        self._campaign_id = f"{self.symbol}-{self._campaign_seq}"
        self._in_position = True
        self._pos_direction = pending.signal.direction
        self._pos_entry_price = fill.fill_price
        self._pos_initial_stop = stop_price
        self._pos_current_stop = stop_price
        self._pos_qty = fill.order.qty
        self._pos_entry_time = bar_time
        self._pos_risk_per_unit = risk_per_unit
        self._pos_bars_held = 0
        self._pos_mfe_price = fill.fill_price
        self._pos_mfe_r = 0.0
        self._pos_mae_price = fill.fill_price
        self._pos_be_triggered = False
        self._pos_regime_entry = pending.signal.regime_at_entry.value
        self._pos_entry_type = pending.signal.entry_type.value
        self._pos_adx_entry = pending.adx_entry
        self._pos_score_entry = pending.signal.bear_conviction
        self._pos_di_agrees = pending.di_agrees
        self._pos_quality = pending.quality_score
        self._pos_chand_bonus = 0.0
        self._pos_tranche_b_open = self.cfg.scale_out_enabled
        self._pos_original_qty = fill.order.qty
        self._pos_open_commission_basis = fill.commission
        self._pos_signal_time = pending.signal_time
        self._pos_fill_time = bar_time
        self._pos_signal_bar_index = pending.signal_bar_index
        self._pos_fill_bar_index = bar_idx
        self._protective_stop_live = self.cfg.min_hold_bars <= 0
        self._hourly_highs = []
        self._hourly_lows = []
        self._pyramid_count = 0

        if self._protective_stop_live:
            self._sync_protective_stop(bar_time)

    def _on_add_fill(self, fill: FillResult, bar_time: datetime, _bar_idx: int) -> None:
        pending = self._pending_add
        if pending is None or fill.order.order_id != pending.order_id or not self._in_position:
            return

        self._pending_add = None
        self.total_commission += fill.commission
        self.equity -= fill.commission

        old_qty = self._pos_qty
        new_qty = old_qty + fill.order.qty
        if new_qty <= 0:
            return

        old_cost = self._pos_entry_price * old_qty
        add_cost = fill.fill_price * fill.order.qty
        self._pos_entry_price = (old_cost + add_cost) / new_qty
        self._pos_qty = new_qty
        self._pos_risk_per_unit = abs(self._pos_entry_price - self._pos_current_stop)
        self._pos_open_commission_basis += fill.commission
        self._pyramid_count += 1
        if self._protective_stop_live:
            self._sync_protective_stop(bar_time)

    def _on_partial_fill(self, fill: FillResult, bar_time: datetime, _bar_idx: int) -> None:
        if (
            not self._in_position
            or self._pending_partial_order_id is None
            or fill.order.order_id != self._pending_partial_order_id
        ):
            return

        qty = min(self._pending_partial_qty or fill.order.qty, self._pos_qty)
        self._pending_partial_order_id = None
        self._pending_partial_qty = 0
        if qty <= 0:
            return

        self.total_commission += fill.commission
        entry_alloc = self._allocate_open_commission(qty)
        pnl_points, pnl_dollars, r_multiple, mae_r = self._trade_outcome(fill.fill_price, qty)

        self._record_trade(
            qty=qty,
            exit_price=fill.fill_price,
            exit_time=bar_time,
            reason="TARGET",
            pnl_points=pnl_points,
            pnl_dollars=pnl_dollars,
            r_multiple=r_multiple,
            mae_r=mae_r,
            report_commission=entry_alloc + fill.commission,
        )
        self.equity += pnl_dollars - fill.commission

        self._pos_qty -= qty
        self._pos_tranche_b_open = False
        self._pos_chand_bonus = self.cfg.scale_out_trail_bonus

        if self._pos_qty <= 0:
            self._finalize_flat_position()
            return

        if self._protective_stop_live:
            self._sync_protective_stop(bar_time)

    def _on_exit_fill(self, fill: FillResult, bar_time: datetime, bar_idx: int) -> None:
        if (
            not self._in_position
            or self._pending_exit_order_id is None
            or fill.order.order_id != self._pending_exit_order_id
        ):
            return

        reason = self._pending_exit_reason or ExitReason.TIME_DECAY.value
        self._pending_exit_order_id = None
        self._pending_exit_reason = ""
        self._close_full_position(fill.fill_price, bar_time, reason, fill.commission, bar_idx)

    def _on_stop_fill(self, fill: FillResult, bar_time: datetime, bar_idx: int) -> None:
        if not self._in_position:
            return
        self._close_full_position(fill.fill_price, bar_time, ExitReason.STOP.value, fill.commission, bar_idx)

    def _manage_position(self, h: HourlyContext, bar_time: datetime, bar_idx: int) -> None:
        if not self._in_position:
            return

        if self._is_rth(bar_time):
            self._pos_bars_held += 1

        self._hourly_highs.append(h.high)
        self._hourly_lows.append(h.low)

        rpu = self._pos_risk_per_unit
        if self._pos_direction == Direction.SHORT:
            if h.low < self._pos_mfe_price or self._pos_mfe_price == 0.0:
                self._pos_mfe_price = h.low
            if h.high > self._pos_mae_price or self._pos_mae_price == 0.0:
                self._pos_mae_price = h.high
            if rpu > 0:
                self._pos_mfe_r = max(self._pos_mfe_r, (self._pos_entry_price - self._pos_mfe_price) / rpu)
                cur_r = (self._pos_entry_price - h.close) / rpu
            else:
                cur_r = 0.0
        else:
            if h.high > self._pos_mfe_price or self._pos_mfe_price == 0.0:
                self._pos_mfe_price = h.high
            if h.low < self._pos_mae_price or self._pos_mae_price == 0.0:
                self._pos_mae_price = h.low
            if rpu > 0:
                self._pos_mfe_r = max(self._pos_mfe_r, (self._pos_mfe_price - self._pos_entry_price) / rpu)
                cur_r = (h.close - self._pos_entry_price) / rpu
            else:
                cur_r = 0.0

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
        stop_changed = abs(new_stop - self._pos_current_stop) > 1e-12
        self._pos_current_stop = new_stop
        self._pos_be_triggered = new_be
        if stop_changed:
            self._sync_protective_stop(bar_time)

        if (
            self.cfg.scale_out_enabled
            and self._pending_partial_order_id is None
            and self._pending_exit_order_id is None
            and check_scale_out(cur_r, self.cfg.scale_out_target_r, self._pos_tranche_b_open)
        ):
            tranche_b_qty = int(math.floor(self._pos_original_qty * self.cfg.scale_out_pct))
            tranche_b_qty = min(tranche_b_qty, self._pos_qty)
            if tranche_b_qty > 0:
                self._submit_partial_exit(tranche_b_qty, bar_time)

        is_long = self._pos_direction == Direction.LONG
        exit_reason = check_exits(
            risk_per_unit=rpu,
            bars_held=self._pos_bars_held,
            cur_r=cur_r,
            be_triggered=self._pos_be_triggered,
            regime=self.daily_ctx.regime,
            catastrophic_cap_r=self.cfg.catastrophic_cap_r,
            stale_bars=self.cfg.stale_bars_short if not is_long else self.cfg.stale_bars_long,
            stale_early_bars=self.cfg.stale_early_bars,
            time_decay_hours=self.cfg.time_decay_hours,
            is_long=is_long,
            min_hold_bars=self.cfg.min_hold_bars,
        )
        if exit_reason is not None and self._pending_exit_order_id is None:
            if self._pending_partial_order_id is not None:
                self.broker.cancel_orders(self.symbol, tag="partial")
                self._pending_partial_order_id = None
                self._pending_partial_qty = 0
            self._submit_market_exit(exit_reason.value, bar_time)

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
        if self._in_position or self._pending_entry is not None:
            return
        if not self._is_rth(bar_time):
            return
        if bar_idx < self._cooldown_until_bar:
            return

        d = self.daily_ctx

        if self.cfg.tod_filter_enabled:
            dt_et = bar_time.astimezone(self._get_et_zone())
            tod_minutes = dt_et.hour * 60 + dt_et.minute
            if self.cfg.tod_filter_start <= tod_minutes < self.cfg.tod_filter_end:
                return

        if self._s2_arm.armed and bar_idx > self._s2_arm.armed_until_bar:
            self._s2_arm = S2ArmState()
        if self._s3_arm.armed and bar_idx > self._s3_arm.armed_until_bar:
            self._s3_arm = S3ArmState()
        if self._lh_arm.armed and bar_idx > self._lh_arm.armed_until_bar:
            self._lh_arm = LHArmState()
        if self._bd_arm.armed and bar_idx > self._bd_arm.armed_until_bar:
            self._bd_arm = BDArmState()

        if swing_highs_h is not None and bar_idx < len(swing_highs_h):
            sh_val = swing_highs_h[bar_idx]
            if not np.isnan(sh_val):
                if self._swing_highs and sh_val < self._swing_highs[-1]:
                    self._lh_arm = LHArmState(
                        armed=True,
                        swing_high_price=sh_val,
                        prior_swing_high_price=self._swing_highs[-1],
                        armed_bar=bar_idx,
                        armed_until_bar=bar_idx + self.cfg.lh_arm_bars,
                    )
                self._swing_highs.append(sh_val)

        bd_donch = donch_low_bd if donch_low_bd is not None else donch_low_h
        if not self._bd_arm.armed and bar_idx >= 1 and bar_idx < len(bd_donch):
            vol_val = float(h_volumes[bar_idx]) if h_volumes is not None else 0.0
            vsma_val = (
                float(vol_sma20_h[bar_idx])
                if vol_sma20_h is not None and not np.isnan(vol_sma20_h[bar_idx])
                else 0.0
            )
            prior_donch = float(bd_donch[bar_idx - 1])
            chop_arm_ok = (
                self.cfg.chop_short_entry_enabled
                and self._bias.peak_drop_override
                and self._bias.confirmed_direction == Direction.SHORT
            )
            arm_bd = check_bd_arm(
                h.low,
                h.high,
                h.close,
                vol_val,
                prior_donch,
                vsma_val,
                d.regime,
                self.cfg,
                bar_idx,
                chop_arm_allowed=chop_arm_ok,
                persistence_override=d.persistence_active,
            )
            if arm_bd is not None:
                self._bd_arm = arm_bd

        if not self._s3_arm.armed and bar_idx >= 1 and bar_idx < len(donch_low_h):
            arm3 = check_s3_arm(
                d.regime,
                d.bias.confirmed_direction == Direction.SHORT,
                h.low,
                h.close,
                float(donch_low_h[bar_idx - 1]),
                h.ema_mom,
                bar_idx,
            )
            if arm3 is not None:
                self._s3_arm = arm3

        ef_slope = float(ema_fast_slope_d[d_idx]) if d_idx < len(ema_fast_slope_d) else 0.0
        es_slope = float(ema_slow_slope_d[d_idx]) if d_idx < len(ema_slow_slope_d) else 0.0

        signals: list[EntrySignal | None] = [
            check_lh_rejection(d, h, self.sym_cfg, self.cfg, self._lh_arm),
            check_bd_continuation(d, h, self.sym_cfg, self.cfg, self._bd_arm),
            check_s2(d, h, self.sym_cfg, self.cfg, self._s2_arm),
            check_s1(d, h, self.sym_cfg, self.cfg),
            check_s3(d, h, self.sym_cfg, self.cfg, self._s3_arm),
            check_l1(d, h, self.sym_cfg, self.cfg, self.symbol, ef_slope, es_slope),
        ]

        for signal in signals:
            if signal is None:
                continue
            self._submit_entry_order(signal, h, bar_time, bar_idx)
            if self._pending_entry is not None:
                self._disarm_signal_state(signal.entry_type)
            return

    def _submit_entry_order(
        self,
        signal: EntrySignal,
        h: HourlyContext,
        bar_time: datetime,
        bar_idx: int,
    ) -> None:
        qty = compute_position_size(
            signal,
            self._sizing_equity,
            self.sym_cfg,
            self.daily_ctx,
            self.cfg,
            self._current_heat_r,
            self._concurrent_positions,
        )
        if qty <= 0:
            return

        order_id = self.broker.next_order_id()
        self.broker.submit_order(
            SimOrder(
                order_id=order_id,
                symbol=self.symbol,
                side=OrderSide.SELL if signal.direction == Direction.SHORT else OrderSide.BUY,
                order_type=OrderType.MARKET,
                qty=qty,
                tick_size=_TICK_SIZE,
                submit_time=bar_time,
                ttl_hours=0,
                tag="entry",
            )
        )
        self._pending_entry = _PendingEntry(
            order_id=order_id,
            signal=signal,
            qty=qty,
            signal_time=bar_time,
            signal_bar_index=bar_idx,
            signal_atr14_h=h.atr14_h,
            signal_atr14_d=self.daily_ctx.atr14_d,
            regime_at_signal=self.daily_ctx.regime,
            adx_entry=self.daily_ctx.adx,
            di_agrees=self.daily_ctx.minus_di > self.daily_ctx.plus_di,
            quality_score=signal.quality_score,
        )

    def _submit_market_exit(self, reason: str, bar_time: datetime) -> None:
        if not self._in_position or self._pending_exit_order_id is not None:
            return
        exit_side = OrderSide.SELL if self._pos_direction == Direction.LONG else OrderSide.BUY
        order_id = self.broker.next_order_id()
        self.broker.submit_order(
            SimOrder(
                order_id=order_id,
                symbol=self.symbol,
                side=exit_side,
                order_type=OrderType.MARKET,
                qty=self._pos_qty,
                tick_size=_TICK_SIZE,
                submit_time=bar_time,
                ttl_hours=0,
                tag="exit",
            )
        )
        self._pending_exit_order_id = order_id
        self._pending_exit_reason = reason

    def _submit_partial_exit(self, qty: int, bar_time: datetime) -> None:
        if not self._in_position or qty <= 0 or qty > self._pos_qty or self._pending_partial_order_id is not None:
            return
        exit_side = OrderSide.SELL if self._pos_direction == Direction.LONG else OrderSide.BUY
        order_id = self.broker.next_order_id()
        self.broker.submit_order(
            SimOrder(
                order_id=order_id,
                symbol=self.symbol,
                side=exit_side,
                order_type=OrderType.MARKET,
                qty=qty,
                tick_size=_TICK_SIZE,
                submit_time=bar_time,
                ttl_hours=0,
                tag="partial",
            )
        )
        self._pending_partial_order_id = order_id
        self._pending_partial_qty = qty

    def _submit_add_order(self, signal: EntrySignal, bar_time: datetime, bar_idx: int) -> None:
        if not self._in_position or self._pending_add is not None:
            return
        add_qty = int(math.floor(self._pos_original_qty * self.cfg.pyramid_scale))
        if add_qty <= 0:
            return
        if self._pos_risk_per_unit > 0:
            base_risk = self._sizing_equity * self.sym_cfg.base_risk_pct
            if base_risk > 0:
                add_risk_r = add_qty * self._pos_risk_per_unit / base_risk
                heat_cap = self.cfg.crisis_heat_cap_r if self.daily_ctx.vol.extreme_vol else self.cfg.heat_cap_r
                if self._current_heat_r + add_risk_r > heat_cap:
                    return

        order_id = self.broker.next_order_id()
        self.broker.submit_order(
            SimOrder(
                order_id=order_id,
                symbol=self.symbol,
                side=OrderSide.SELL if signal.direction == Direction.SHORT else OrderSide.BUY,
                order_type=OrderType.MARKET,
                qty=add_qty,
                tick_size=_TICK_SIZE,
                submit_time=bar_time,
                ttl_hours=0,
                tag="add_on",
            )
        )
        self._pending_add = _PendingAdd(
            order_id=order_id,
            signal=signal,
            qty=add_qty,
            signal_time=bar_time,
            signal_bar_index=bar_idx,
        )

    def _sync_protective_stop(self, bar_time: datetime) -> None:
        self.broker.cancel_orders(self.symbol, tag="protective_stop")
        if (
            not self._in_position
            or not self._protective_stop_live
            or self._pos_qty <= 0
            or self._pos_current_stop <= 0
        ):
            return

        stop_side = OrderSide.SELL if self._pos_direction == Direction.LONG else OrderSide.BUY
        self.broker.submit_order(
            SimOrder(
                order_id=self.broker.next_order_id(),
                symbol=self.symbol,
                side=stop_side,
                order_type=OrderType.STOP,
                qty=self._pos_qty,
                stop_price=self._pos_current_stop,
                tick_size=_TICK_SIZE,
                submit_time=bar_time,
                ttl_hours=0,
                tag="protective_stop",
            )
        )

    def _activate_protective_stop_if_due(self, bar_time: datetime) -> None:
        if not self._in_position or self._protective_stop_live or self._pos_qty <= 0:
            return
        if self.cfg.min_hold_bars > 0:
            if not self._is_rth(bar_time):
                return
            if self._pos_bars_held + 1 < self.cfg.min_hold_bars:
                return
        self._protective_stop_live = True
        self._sync_protective_stop(bar_time)

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

        peak_drop_triggered = False
        if self.cfg.peak_drop_enabled and d_idx >= self.cfg.peak_drop_lookback:
            lookback_start = max(0, d_idx - self.cfg.peak_drop_lookback + 1)
            rolling_peak = float(np.max(d_highs[lookback_start:d_idx + 1]))
            if rolling_peak > 0:
                drop_pct = (close - rolling_peak) / rolling_peak
                if drop_pct <= self.cfg.peak_drop_pct:
                    peak_drop_triggered = True

        regime_on = compute_regime_on(
            adx_val,
            self._prev_regime_on,
            self.sym_cfg.adx_on,
            self.sym_cfg.adx_off,
        )
        if peak_drop_triggered:
            regime_on = True

        if (
            self.cfg.gld_qqq_override_enabled
            and self.symbol == "GLD"
            and self._qqq_regimes
            and len(self._qqq_regimes) > d_idx
        ):
            qqq_regime = self._qqq_regimes[d_idx]
            if (
                qqq_regime == BRSRegime.BEAR_STRONG
                or (
                    self.cfg.gld_qqq_override_bear_trend
                    and qqq_regime == BRSRegime.BEAR_TREND
                )
            ):
                regime_on = True

        self._prev_regime_on = regime_on

        self.prev_regime = self.daily_ctx.regime
        regime = classify_regime(
            adx_val,
            p_di,
            m_di,
            close,
            ef,
            es,
            regime_on,
            self.cfg.adx_strong,
            bear_min_conditions=self.cfg.regime_bear_min_conditions,
        )

        price_below = close < ef and close < es
        slope_neg = ef_slope < 0
        conviction = compute_bear_conviction(
            adx_val,
            m_di,
            p_di,
            ema_sep_pct,
            price_below,
            slope_neg,
        )

        vf_min = self.cfg.resolve_param("vf_clamp_min") or 0.35
        vf_max = self.cfg.resolve_param("vf_clamp_max") or 1.5
        extreme_pct = self.cfg.resolve_param("extreme_vol_pct") or 95.0
        vf, vol_pct = compute_vol_factor(
            a14,
            atr14_d_arr[:d_idx + 1],
            vf_clamp_min=vf_min,
            vf_clamp_max=vf_max,
            extreme_vol_pct=extreme_pct,
        )
        risk_adj = compute_risk_regime(a14, atr14_d_arr[:d_idx + 1])

        raw_dir = compute_raw_bias(close, ef, es)
        di_diff = m_di - p_di
        daily_return = (close - float(d_closes[d_idx - 1])) / float(d_closes[d_idx - 1]) if d_idx > 0 else 0.0
        atr_ratio = a14 / a50 if a50 > 0 else 1.0

        cum_return_lookback = self.cfg.cum_return_lookback
        cum_return = 0.0
        if d_idx >= cum_return_lookback:
            lookback_close = float(d_closes[d_idx - cum_return_lookback])
            if lookback_close > 0:
                cum_return = (close - lookback_close) / lookback_close

        self._bias = update_bias(
            self._bias,
            regime,
            regime_on,
            raw_dir,
            conviction,
            adx_val,
            di_diff,
            ema_sep_pct,
            daily_return=daily_return,
            atr_ratio=atr_ratio,
            fast_crash_enabled=self.cfg.fast_crash_enabled,
            fast_crash_return_thresh=self.cfg.fast_crash_return_thresh,
            fast_crash_atr_ratio=self.cfg.fast_crash_atr_ratio,
            crash_override_enabled=self.cfg.crash_override_enabled,
            crash_override_return=self.cfg.crash_override_return,
            crash_override_atr_ratio=self.cfg.crash_override_atr_ratio,
            return_only_enabled=self.cfg.return_only_enabled,
            return_only_thresh=self.cfg.return_only_thresh,
            close=close,
            ema_fast=ef,
            bias_4h_accel_enabled=self.cfg.bias_4h_accel_enabled,
            bias_4h_accel_reduction=self.cfg.bias_4h_accel_reduction,
            regime_4h_bear=self.daily_ctx.regime_4h == Regime4H.BEAR,
            cum_return_enabled=self.cfg.cum_return_enabled,
            cum_return=cum_return,
            cum_return_thresh=self.cfg.cum_return_thresh,
            churn_bridge_bars=self.cfg.churn_bridge_bars,
        )

        if peak_drop_triggered and self._bias.confirmed_direction != Direction.SHORT:
            self._bias.confirmed_direction = Direction.SHORT
            self._bias.peak_drop_override = True

        if self._bias.confirmed_direction == Direction.SHORT and not self._in_position:
            self._short_bias_no_trade += 1
        else:
            self._short_bias_no_trade = 0

        self._regime_history.append(regime)

        self.daily_ctx = DailyContext(
            regime=regime,
            regime_on=regime_on,
            regime_4h=self.daily_ctx.regime_4h,
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
            persistence_active=(
                self.cfg.persistence_override_bars > 0
                and self._short_bias_no_trade >= self.cfg.persistence_override_bars
            ),
        )

        from backtests.swing.analysis.brs_diagnostics import CRISIS_WINDOWS

        bar_dt = self._to_datetime(self._daily_times[d_idx])
        bar_dt_naive = bar_dt.replace(tzinfo=None) if bar_dt.tzinfo else bar_dt
        for cname, cstart, cend in CRISIS_WINDOWS:
            if cstart <= bar_dt_naive <= cend:
                self.crisis_state_log.append(
                    {
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
                    }
                )
                break

        arm = check_s2_arm(
            d_closes,
            d_lows,
            d_highs,
            d_volumes,
            a14,
            a50,
            self.cfg,
            d_idx,
        )
        if arm is not None:
            self._s2_arm = arm

    def _current_r(self) -> float:
        if not self._in_position or self._pos_risk_per_unit <= 0:
            return 0.0
        if self._pos_direction == Direction.SHORT:
            return (self._pos_entry_price - self._last_close) / self._pos_risk_per_unit
        return (self._last_close - self._pos_entry_price) / self._pos_risk_per_unit

    def _check_pyramid(self, h: HourlyContext, bar_time: datetime, bar_idx: int) -> None:
        d = self.daily_ctx
        signals = [
            check_lh_rejection(d, h, self.sym_cfg, self.cfg, self._lh_arm),
            check_bd_continuation(d, h, self.sym_cfg, self.cfg, self._bd_arm),
        ]
        for signal in signals:
            if signal is not None and signal.direction == self._pos_direction:
                self._submit_add_order(signal, bar_time, bar_idx)
                return

    def _close_full_position(
        self,
        exit_price: float,
        bar_time: datetime,
        reason: str,
        exit_commission: float,
        bar_idx: int,
    ) -> None:
        if not self._in_position or self._pos_qty <= 0:
            return

        self.total_commission += exit_commission
        qty = self._pos_qty
        entry_alloc = self._allocate_open_commission(qty)
        pnl_points, pnl_dollars, r_multiple, mae_r = self._trade_outcome(exit_price, qty)
        self._record_trade(
            qty=qty,
            exit_price=exit_price,
            exit_time=bar_time,
            reason=reason,
            pnl_points=pnl_points,
            pnl_dollars=pnl_dollars,
            r_multiple=r_multiple,
            mae_r=mae_r,
            report_commission=entry_alloc + exit_commission,
        )
        self.equity += pnl_dollars - exit_commission
        self._set_cooldown(bar_idx)
        self._finalize_flat_position()

    def _trade_outcome(self, exit_price: float, qty: int) -> tuple[float, float, float, float]:
        if self._pos_direction == Direction.SHORT:
            pnl_points = self._pos_entry_price - exit_price
            mae_points = self._pos_mae_price - self._pos_entry_price if self._pos_mae_price > 0 else 0.0
        else:
            pnl_points = exit_price - self._pos_entry_price
            mae_points = self._pos_entry_price - self._pos_mae_price if self._pos_mae_price > 0 else 0.0

        pnl_dollars = pnl_points * qty
        risk_dollars = self._pos_risk_per_unit * qty
        r_multiple = pnl_dollars / risk_dollars if risk_dollars > 0 else 0.0
        mae_r = mae_points / self._pos_risk_per_unit if self._pos_risk_per_unit > 0 else 0.0
        return pnl_points, pnl_dollars, r_multiple, mae_r

    def _record_trade(
        self,
        *,
        qty: int,
        exit_price: float,
        exit_time: datetime,
        reason: str,
        pnl_points: float,
        pnl_dollars: float,
        r_multiple: float,
        mae_r: float,
        report_commission: float,
    ) -> None:
        self.trades.append(
            TradeRecord(
                symbol=self.symbol,
                direction=int(self._pos_direction.value),
                entry_type=self._pos_entry_type,
                entry_time=self._pos_entry_time,
                exit_time=exit_time,
                entry_price=self._pos_entry_price,
                exit_price=exit_price,
                qty=qty,
                initial_stop=self._pos_initial_stop,
                exit_reason=reason,
                pnl_points=pnl_points,
                pnl_dollars=pnl_dollars,
                r_multiple=r_multiple,
                mfe_r=self._pos_mfe_r,
                mae_r=mae_r,
                bars_held=self._pos_bars_held,
                commission=report_commission,
                adx_entry=self._pos_adx_entry,
                score_entry=self._pos_score_entry,
                di_agrees=self._pos_di_agrees,
                quality_score=self._pos_quality,
                regime_entry=self._pos_regime_entry,
                signal_time=self._pos_signal_time,
                fill_time=self._pos_fill_time,
                signal_bar_index=self._pos_signal_bar_index,
                fill_bar_index=self._pos_fill_bar_index,
                campaign_id=self._campaign_id,
            )
        )

    @staticmethod
    def _commission_share(total_commission: float, qty: int, total_qty: int) -> float:
        if total_commission <= 0 or qty <= 0 or total_qty <= 0:
            return 0.0
        return total_commission * qty / total_qty

    def _allocate_open_commission(self, qty: int) -> float:
        if qty <= 0 or self._pos_qty <= 0 or self._pos_open_commission_basis <= 0:
            return 0.0
        qty = min(qty, self._pos_qty)
        alloc = self._commission_share(self._pos_open_commission_basis, qty, self._pos_qty)
        self._pos_open_commission_basis = max(self._pos_open_commission_basis - alloc, 0.0)
        return alloc

    def _finalize_end_of_data(self) -> None:
        self.broker.cancel_orders(self.symbol, tag="entry")
        self.broker.cancel_orders(self.symbol, tag="add_on")
        self.broker.cancel_orders(self.symbol, tag="partial")
        self.broker.cancel_orders(self.symbol, tag="exit")
        self.broker.cancel_orders(self.symbol, tag="protective_stop")

        self._pending_entry = None
        self._pending_add = None
        self._pending_partial_order_id = None
        self._pending_partial_qty = 0

        exit_reason = self._pending_exit_reason or "END_OF_DATA"
        self._pending_exit_order_id = None
        self._pending_exit_reason = ""

        if not self._in_position or self._pos_qty <= 0:
            return

        last_index = len(self._h_closes) - 1
        if last_index < 0:
            return

        last_time = self._hourly_datetimes[last_index]
        last_price = self._last_mark_price
        if last_price <= 0.0 or np.isnan(last_price):
            last_price = float(self._h_closes[last_index])
        if last_price <= 0.0 or np.isnan(last_price):
            last_price = self._pos_entry_price

        exit_side = OrderSide.SELL if self._pos_direction == Direction.LONG else OrderSide.BUY
        fill = self.broker._fill_market(
            SimOrder(
                order_id=self.broker.next_order_id(),
                symbol=self.symbol,
                side=exit_side,
                order_type=OrderType.MARKET,
                qty=self._pos_qty,
                tick_size=_TICK_SIZE,
                submit_time=last_time,
                ttl_hours=0,
                tag="exit",
            ),
            last_time,
            last_price,
            _TICK_SIZE,
        )
        self._close_full_position(fill.fill_price, last_time, exit_reason, fill.commission, last_index)

    def _finalize_flat_position(self) -> None:
        self.broker.cancel_orders(self.symbol, tag="protective_stop")
        self.broker.cancel_orders(self.symbol, tag="partial")
        self.broker.cancel_orders(self.symbol, tag="exit")
        self.broker.cancel_orders(self.symbol, tag="add_on")

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
        self._pos_adx_entry = 0.0
        self._pos_score_entry = 0.0
        self._pos_di_agrees = False
        self._pos_quality = 0.0
        self._pos_chand_bonus = 0.0
        self._pos_tranche_b_open = False
        self._pos_original_qty = 0
        self._pos_open_commission_basis = 0.0
        self._pos_signal_time = None
        self._pos_fill_time = None
        self._pos_signal_bar_index = -1
        self._pos_fill_bar_index = -1
        self._campaign_id = ""
        self._protective_stop_live = False
        self._pyramid_count = 0
        self._hourly_highs = []
        self._hourly_lows = []
        self._pending_add = None
        self._pending_exit_order_id = None
        self._pending_exit_reason = ""
        self._pending_partial_order_id = None
        self._pending_partial_qty = 0

    def _set_cooldown(self, bar_idx: int) -> None:
        base_cd = self.sym_cfg.cooldown_bars
        regime = self.daily_ctx.regime if self.daily_ctx else None
        if regime == BRSRegime.BEAR_STRONG:
            cooldown = self.cfg.cooldown_bear_strong
        elif regime == BRSRegime.BEAR_TREND:
            cooldown = self.cfg.cooldown_bear_trend
        else:
            cooldown = base_cd
        self._cooldown_until_bar = bar_idx + cooldown

    def _disarm_signal_state(self, entry_type: EntryType) -> None:
        if entry_type == EntryType.S2_BREAKDOWN:
            self._s2_arm = S2ArmState()
        elif entry_type == EntryType.S3_IMPULSE:
            self._s3_arm = S3ArmState()
        elif entry_type == EntryType.LH_REJECTION:
            self._lh_arm = LHArmState()
        elif entry_type == EntryType.BD_CONTINUATION:
            self._bd_arm = BDArmState()

    def _mtm_equity(self, current_price: float | None = None) -> float:
        price = self._last_mark_price if current_price is None else current_price
        return self.equity + self.unrealized_pnl(price)

    def _get_ema_pull(self, i: int, ema34: np.ndarray, ema50: np.ndarray) -> float:
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
        dt_et = dt.astimezone(BRSEngine._get_et_zone())
        if dt_et.weekday() >= 5:
            return False
        minute = dt_et.hour * 60 + dt_et.minute
        return 570 <= minute < 960

    @staticmethod
    def _to_datetime(ts) -> datetime:
        if isinstance(ts, datetime):
            return ts
        if hasattr(ts, "astype"):
            unix_epoch = np.datetime64(0, "ns")
            one_second = np.timedelta64(1, "s")
            seconds = (ts - unix_epoch) / one_second
            return datetime.fromtimestamp(float(seconds), tz=timezone.utc)
        return datetime.now(timezone.utc)
