"""Daily bar-by-bar Keltner Momentum Breakout backtesting engine.

Mirrors S6Engine architecture for consistency. Uses SimBroker for fill simulation.
Strategy logic called via pure functions in strategy_4/*.py.
"""
from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone

from libs.broker_ibkr.risk_support.tick_rules import round_to_tick

from strategy_4 import indicators, signals
from strategy_4.config import SymbolConfig, SYMBOL_CONFIGS
from strategy_4.models import DailyState, Direction

from backtest.config import SlippageConfig
from backtest.config_s5 import S5BacktestConfig
from backtest.data.preprocessing import NumpyBars
from backtest.engine.sim_broker import (
    FillResult,
    FillStatus,
    OrderSide,
    OrderType,
    SimBroker,
    SimOrder,
)

logger = logging.getLogger(__name__)


@dataclass
class S5TradeRecord:
    """One completed Strategy-5 trade."""

    symbol: str = ""
    direction: int = 0
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    qty: int = 0
    initial_stop: float = 0.0
    exit_reason: str = ""
    pnl_points: float = 0.0
    pnl_dollars: float = 0.0
    r_multiple: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    bars_held: int = 0
    commission: float = 0.0


@dataclass
class _ActivePosition:
    """Internal position state during backtest."""

    direction: Direction = Direction.FLAT
    fill_price: float = 0.0
    qty: int = 0
    initial_stop: float = 0.0
    current_stop: float = 0.0
    r_price: float = 0.0
    entry_time: datetime | None = None
    bars_held: int = 0
    mfe_price: float = 0.0
    mae_price: float = 0.0
    commission: float = 0.0


class S5Engine:
    """Single-symbol daily bar-by-bar Keltner Momentum engine."""

    def __init__(
        self,
        symbol: str,
        cfg: SymbolConfig,
        bt_config: S5BacktestConfig,
        point_value: float = 1.0,
    ) -> None:
        self.symbol = symbol
        self.cfg = cfg
        self.bt_config = bt_config
        self.point_value = point_value

        # Apply variant overrides
        from dataclasses import replace as _replace
        self.cfg = _replace(
            cfg,
            kelt_ema_period=bt_config.kelt_ema_period,
            kelt_atr_period=bt_config.kelt_atr_period,
            kelt_atr_mult=bt_config.kelt_atr_mult,
            rsi_period=bt_config.rsi_period,
            rsi_entry_long=bt_config.rsi_entry_long,
            rsi_entry_short=bt_config.rsi_entry_short,
            roc_period=bt_config.roc_period,
            vol_sma_period=bt_config.vol_sma_period,
            volume_filter=bt_config.volume_filter,
            entry_mode=bt_config.entry_mode,
            exit_mode=bt_config.exit_mode,
            atr_period=bt_config.atr_period,
            atr_stop_mult=bt_config.atr_stop_mult,
            trail_atr_mult=bt_config.trail_atr_mult,
            base_risk_pct=bt_config.risk_pct,
            shorts_enabled=bt_config.shorts_enabled,
        )

        # Use ETF commission rate
        slippage = bt_config.slippage
        slippage = _replace(slippage,
                            commission_per_contract=slippage.commission_per_share_etf)
        self.broker = SimBroker(slippage_config=slippage)

        self.equity = bt_config.initial_equity
        self.sizing_equity = bt_config.initial_equity

        self.active_position: _ActivePosition | None = None
        self.pending_exit: bool = False
        self._pending_exit_reason: str = ""

        self.trades: list[S5TradeRecord] = []
        self.equity_curve: list[float] = []
        self.timestamps: list = []
        self.total_commission: float = 0.0

    def force_flatten(self) -> None:
        """Reverse an entry rejected by portfolio heat cap."""
        self.broker.cancel_all(self.symbol)
        # Reverse entry commission leaked by the phantom fill
        if self.active_position is not None:
            entry_comm = getattr(self.active_position, "commission", 0.0)
            if entry_comm > 0:
                self.equity += entry_comm
                self.total_commission -= entry_comm
        self.active_position = None

    def run(self, daily: NumpyBars) -> None:
        """Run full backtest over daily bars."""
        warmup = self.bt_config.warmup_daily

        for t in range(len(daily.times)):
            self._step_bar(daily, t, warmup)

        if self.active_position is not None:
            self._flatten_position(
                self.active_position,
                daily.closes[-1],
                self._to_datetime(daily.times[-1]),
                "END_OF_DATA",
            )

    def _step_bar(self, daily: NumpyBars, t: int, warmup: int) -> None:
        bar_time = self._to_datetime(daily.times[t])
        O = daily.opens[t]
        H = daily.highs[t]
        L = daily.lows[t]
        C = daily.closes[t]

        if np.isnan(O) or np.isnan(C):
            self.equity_curve.append(self.equity)
            self.timestamps.append(daily.times[t])
            return

        fills = self.broker.process_bar(
            self.symbol, bar_time, O, H, L, C, self.cfg.tick_size,
        )
        for fill in fills:
            self._handle_fill(fill, bar_time, C)

        if self.active_position is not None:
            self._update_trailing_stop(daily, t, H, L)

        if t < warmup:
            self.equity_curve.append(self.equity)
            self.timestamps.append(daily.times[t])
            return

        state = self._compute_state(daily, t)

        if self.active_position is not None:
            should_exit, reason = signals.should_exit_signal(
                state, self.active_position.direction, self.cfg,
            )
            if should_exit:
                self._submit_exit_order(bar_time, reason)

        if self.active_position is None and not self.pending_exit:
            direction = signals.entry_signal(state, self.cfg)
            if direction != Direction.FLAT:
                self._submit_entry(state, direction, daily, t, bar_time)

        self.equity_curve.append(self.equity)
        self.timestamps.append(daily.times[t])

    def _compute_state(self, daily: NumpyBars, t: int) -> DailyState:
        end = t + 1
        closes = daily.closes[:end]
        highs = daily.highs[:end]
        lows = daily.lows[:end]
        volumes = daily.volumes[:end]
        cfg = self.cfg

        # Keltner Channel
        kelt_upper, kelt_middle, kelt_lower = indicators.keltner_channel(
            closes, highs, lows,
            cfg.kelt_ema_period, cfg.kelt_atr_period, cfg.kelt_atr_mult,
        )

        # RSI
        rsi_arr = indicators.rsi(closes, cfg.rsi_period)

        # ROC
        roc_arr = indicators.roc(closes, cfg.roc_period)

        # Volume SMA
        vol_sma_arr = indicators.volume_sma(volumes, cfg.vol_sma_period)

        # ATR for stops
        atr_arr = indicators.atr(highs, lows, closes, cfg.atr_period)

        # Previous values
        close_prev = float(closes[-2]) if len(closes) >= 2 else float(closes[-1])
        kelt_mid_prev = float(kelt_middle[-2]) if len(kelt_middle) >= 2 else float(kelt_middle[-1])
        rsi_prev = float(rsi_arr[-2]) if len(rsi_arr) >= 2 else float(rsi_arr[-1])

        return DailyState(
            close=float(closes[-1]),
            close_prev=close_prev,
            kelt_upper=float(kelt_upper[-1]),
            kelt_middle=float(kelt_middle[-1]),
            kelt_lower=float(kelt_lower[-1]),
            kelt_middle_prev=kelt_mid_prev,
            rsi=float(rsi_arr[-1]),
            rsi_prev=rsi_prev,
            roc=float(roc_arr[-1]),
            volume=float(volumes[-1]),
            volume_sma=float(vol_sma_arr[-1]),
            atr=float(atr_arr[-1]),
        )

    def _submit_entry(self, state, direction, daily, t, bar_time):
        cfg = self.cfg
        atr_val = state.atr
        if atr_val <= 0:
            return

        stop_dist = cfg.atr_stop_mult * atr_val
        if direction == Direction.LONG:
            stop_price = state.close - stop_dist
        else:
            stop_price = state.close + stop_dist

        risk_per_share = stop_dist * self.point_value
        if risk_per_share <= 0:
            return
        raw_qty = (self.sizing_equity * cfg.base_risk_pct) / risk_per_share
        qty = max(1, int(raw_qty))

        max_notional = self.sizing_equity * 0.20
        if state.close > 0:
            max_qty = int(max_notional / state.close)
            qty = min(qty, max(1, max_qty))

        side = OrderSide.BUY if direction == Direction.LONG else OrderSide.SELL
        order = SimOrder(
            order_id=self.broker.next_order_id(),
            symbol=self.symbol, side=side, order_type=OrderType.MARKET,
            qty=qty, tick_size=cfg.tick_size, submit_time=bar_time,
            ttl_hours=48, tag="entry",
        )
        self.broker.submit_order(order)
        self._pending_entry_context = {
            "direction": direction, "stop_dist": stop_dist,
        }

    def _submit_exit_order(self, bar_time, reason):
        pos = self.active_position
        if pos is None:
            return
        self.broker.cancel_all(self.symbol)
        side = OrderSide.SELL if pos.direction == Direction.LONG else OrderSide.BUY
        order = SimOrder(
            order_id=self.broker.next_order_id(),
            symbol=self.symbol, side=side, order_type=OrderType.MARKET,
            qty=pos.qty, tick_size=self.cfg.tick_size, submit_time=bar_time,
            ttl_hours=48, tag="signal_exit",
        )
        self.broker.submit_order(order)
        self.pending_exit = True
        self._pending_exit_reason = reason

    def _handle_fill(self, fill, bar_time, close):
        if fill.status == FillStatus.EXPIRED or fill.status == FillStatus.REJECTED:
            if fill.order.tag == "entry":
                self._pending_entry_context = None
            return
        if fill.status != FillStatus.FILLED:
            return

        tag = fill.order.tag
        if tag == "entry":
            self._on_entry_fill(fill, bar_time)
        elif tag == "protective_stop":
            self._on_stop_fill(fill, bar_time)
        elif tag == "signal_exit":
            self._on_signal_exit_fill(fill, bar_time)

    def _on_entry_fill(self, fill, bar_time):
        ctx = getattr(self, '_pending_entry_context', None)
        if ctx is None:
            return
        direction = ctx["direction"]
        stop_dist = ctx["stop_dist"]
        if direction == Direction.LONG:
            stop_price = fill.fill_price - stop_dist
        else:
            stop_price = fill.fill_price + stop_dist
        stop_price = round_to_tick(stop_price, self.cfg.tick_size, "nearest")
        r_price = abs(fill.fill_price - stop_price)

        self.active_position = _ActivePosition(
            direction=direction, fill_price=fill.fill_price,
            qty=fill.order.qty, initial_stop=stop_price, current_stop=stop_price,
            r_price=r_price, entry_time=bar_time,
            mfe_price=fill.fill_price, mae_price=fill.fill_price,
            commission=fill.commission,
        )
        self.equity -= fill.commission
        self.total_commission += fill.commission
        self._submit_protective_stop(stop_price, fill.order.qty, bar_time)
        self._pending_entry_context = None
        self.pending_exit = False

    def _on_stop_fill(self, fill, bar_time):
        self._flatten_position(
            self.active_position, fill.fill_price, bar_time, "STOP", fill.commission,
        )

    def _on_signal_exit_fill(self, fill, bar_time):
        reason = self._pending_exit_reason or "SIGNAL_EXIT"
        self._flatten_position(
            self.active_position, fill.fill_price, bar_time, reason, fill.commission,
        )
        self.pending_exit = False
        self._pending_exit_reason = ""

    def _flatten_position(self, pos, exit_price, exit_time, reason, exit_commission=0.0):
        if pos is None:
            return
        direction = pos.direction
        pnl_points = (exit_price - pos.fill_price) * direction
        pnl_dollars = pnl_points * pos.qty * self.point_value
        pnl_dollars -= exit_commission
        self.equity += pnl_dollars
        self.total_commission += exit_commission

        r_mult = pnl_points / pos.r_price if pos.r_price > 0 else 0.0
        if pos.r_price > 0:
            if direction == Direction.LONG:
                mfe_r = (pos.mfe_price - pos.fill_price) / pos.r_price
                mae_r = (pos.fill_price - pos.mae_price) / pos.r_price
            else:
                mfe_r = (pos.fill_price - pos.mfe_price) / pos.r_price
                mae_r = (pos.mae_price - pos.fill_price) / pos.r_price
        else:
            mfe_r = mae_r = 0.0

        trade = S5TradeRecord(
            symbol=self.symbol, direction=int(direction),
            entry_time=pos.entry_time, exit_time=exit_time,
            entry_price=pos.fill_price, exit_price=exit_price,
            qty=pos.qty, initial_stop=pos.initial_stop, exit_reason=reason,
            pnl_points=pnl_points, pnl_dollars=pnl_dollars,
            r_multiple=r_mult, mfe_r=mfe_r, mae_r=mae_r,
            bars_held=pos.bars_held, commission=pos.commission + exit_commission,
        )
        self.trades.append(trade)
        self.broker.cancel_all(self.symbol)
        self.active_position = None

    def _submit_protective_stop(self, stop_price, qty, bar_time):
        self.broker.cancel_orders(self.symbol, tag="protective_stop")
        pos = self.active_position
        if pos is None:
            return
        side = OrderSide.SELL if pos.direction == Direction.LONG else OrderSide.BUY
        order = SimOrder(
            order_id=self.broker.next_order_id(),
            symbol=self.symbol, side=side, order_type=OrderType.STOP,
            qty=qty, stop_price=stop_price, tick_size=self.cfg.tick_size,
            submit_time=bar_time, ttl_hours=0, tag="protective_stop",
        )
        self.broker.submit_order(order)

    def _update_trailing_stop(self, daily, t, H, L):
        pos = self.active_position
        if pos is None:
            return
        pos.bars_held += 1

        if pos.direction == Direction.LONG:
            if H > pos.mfe_price:
                pos.mfe_price = H
            if L < pos.mae_price:
                pos.mae_price = L
        else:
            if L < pos.mfe_price:
                pos.mfe_price = L
            if H > pos.mae_price:
                pos.mae_price = H

        end = t + 1
        if end < self.cfg.atr_period + 1:
            return
        atr_arr = indicators.atr(
            daily.highs[:end], daily.lows[:end], daily.closes[:end],
            self.cfg.atr_period,
        )
        cur_atr = float(atr_arr[-1])

        if pos.r_price > 0:
            if pos.direction == Direction.LONG:
                current_r = (pos.mfe_price - pos.fill_price) / pos.r_price
            else:
                current_r = (pos.fill_price - pos.mfe_price) / pos.r_price

            if current_r >= 1.0:
                trail_dist = self.cfg.trail_atr_mult * cur_atr
                if pos.direction == Direction.LONG:
                    new_stop = pos.mfe_price - trail_dist
                    new_stop = round_to_tick(new_stop, self.cfg.tick_size, "down")
                    if new_stop > pos.current_stop:
                        pos.current_stop = new_stop
                        bar_time = self._to_datetime(daily.times[t])
                        self._submit_protective_stop(new_stop, pos.qty, bar_time)
                else:
                    new_stop = pos.mfe_price + trail_dist
                    new_stop = round_to_tick(new_stop, self.cfg.tick_size, "up")
                    if new_stop < pos.current_stop:
                        pos.current_stop = new_stop
                        bar_time = self._to_datetime(daily.times[t])
                        self._submit_protective_stop(new_stop, pos.qty, bar_time)

    @staticmethod
    def _to_datetime(ts) -> datetime:
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                return ts.replace(tzinfo=timezone.utc)
            return ts
        if hasattr(ts, 'astype'):
            unix_epoch = np.datetime64(0, 'ns')
            one_second = np.timedelta64(1, 's')
            seconds = (ts - unix_epoch) / one_second
            return datetime.fromtimestamp(float(seconds), tz=timezone.utc)
        import pandas as pd
        dt = pd.Timestamp(ts).to_pydatetime()
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
