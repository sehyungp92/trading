"""Keltner Momentum Breakout — async live trading engine.

Single class instantiated twice: once as S5_PB (IBIT pullback) and once as
S5_DUAL (GLD+IBIT dual mode).  Daily-only cycle at 16:15 ET — no intraday
bars, no campaigns, no add-ons.  The simplest live engine in the system.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np

from libs.broker_ibkr.risk_support.tick_rules import round_to_tick
from libs.oms.models.events import OMSEventType
from libs.oms.models.intent import Intent, IntentType
from libs.oms.models.order import (
    EntryPolicy,
    OMSOrder,
    OrderRole,
    OrderSide,
    OrderType,
    RiskContext,
)
from libs.oms.risk.calculator import RiskCalculator
from libs.services.trade_recorder import TradeRecorder

from . import indicators, signals
from .config import SymbolConfig
from .models import DailyState, Direction

logger = logging.getLogger(__name__)

from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Internal position tracker
# ---------------------------------------------------------------------------

@dataclass
class _LivePosition:
    """Tracks one open position for a single symbol."""

    symbol: str
    direction: Direction
    fill_price: float
    qty: int
    initial_stop: float
    current_stop: float
    r_price: float  # |entry - stop|
    entry_time: datetime
    stop_order_id: str = ""
    mfe_price: float = 0.0
    mae_price: float = 0.0
    bars_held: int = 0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class KeltnerEngine:
    """Async live engine for Keltner Momentum Breakout (daily bars only)."""

    def __init__(
        self,
        strategy_id: str,
        ib_session: Any,
        oms_service: Any,
        instruments: dict[str, Any],
        config: dict[str, SymbolConfig],
        trade_recorder: TradeRecorder | None = None,
        equity: float = 100_000.0,
        market_calendar: Any | None = None,
        kit: Any | None = None,
        equity_offset: float = 0.0,
        equity_alloc_pct: float = 1.0,
    ) -> None:
        self._strategy_id = strategy_id
        self._ib = ib_session
        self._oms = oms_service
        self._instruments = instruments
        self._config = config
        self._recorder = trade_recorder
        self._equity = equity
        self._equity_offset = equity_offset
        self._equity_alloc_pct = equity_alloc_pct
        self._market_cal = market_calendar
        self._kit = kit

        # Wire drawdown tracker with initial equity
        if self._kit and self._kit.ctx and self._kit.ctx.drawdown_tracker:
            self._kit.ctx.drawdown_tracker.update_equity(self._equity)

        # Per-symbol state
        self.positions: dict[str, _LivePosition] = {}
        self._pending_entry: dict[str, dict] = {}  # sym -> {direction, stop_dist}
        self._order_to_symbol: dict[str, str] = {}  # oms_order_id -> symbol
        self._order_role: dict[str, str] = {}  # oms_order_id -> "entry"|"stop"|"signal_exit"
        self._risk_halted = False
        self._risk_halt_reason = ""

        # Resolved IB contracts
        self.contracts: dict[str, Any] = {}

        # Async tasks
        self._event_task: asyncio.Task | None = None
        self._daily_task: asyncio.Task | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe OMS events, resolve contracts, start scheduler."""
        logger.info("%s engine starting …", self._strategy_id)
        self._running = True

        # OMS event subscription
        event_queue = self._oms.stream_events(self._strategy_id)
        self._event_task = asyncio.create_task(self._process_events(event_queue))

        # Resolve ETF contracts
        for sym in self._config:
            try:
                from ib_async import Stock
                contract = Stock(sym, "SMART", "USD")
                qualified = await self._ib.ib.qualifyContractsAsync(contract)
                if qualified:
                    self.contracts[sym] = qualified[0]
            except Exception as e:
                logger.warning("%s: Could not resolve contract for %s: %s",
                               self._strategy_id, sym, e)

        # Start daily scheduler
        self._daily_task = asyncio.create_task(self._daily_scheduler())

        logger.info("%s engine started (symbols: %s)",
                    self._strategy_id, list(self._config.keys()))

    async def stop(self) -> None:
        """Cancel async tasks."""
        logger.info("%s engine stopping …", self._strategy_id)
        self._running = False

        for task in [self._event_task, self._daily_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("%s engine stopped", self._strategy_id)

    # ------------------------------------------------------------------
    # Daily scheduler
    # ------------------------------------------------------------------

    async def _daily_scheduler(self) -> None:
        """Sleep until 16:15 ET each weekday, then run _daily_cycle."""
        while self._running:
            now = datetime.now(timezone.utc)
            try:
                now_et = now.astimezone(ET)
            except Exception:
                now_et = now
            target = now_et.replace(hour=16, minute=15, second=0, microsecond=0)
            if target <= now_et:
                target += timedelta(days=1)
            while target.weekday() >= 5 or (
                self._market_cal
                and not self._market_cal.is_trading_day(target.date())
            ):
                target += timedelta(days=1)
            wait = (target - now_et).total_seconds()
            await asyncio.sleep(max(1, wait))
            if not self._running:
                break
            try:
                await self._daily_cycle()
            except Exception:
                logger.exception("%s: Error in daily cycle", self._strategy_id)

    async def _daily_cycle(self) -> None:
        """Run daily close for all symbols."""
        logger.info("%s: === Daily cycle ===", self._strategy_id)
        await self._refresh_equity()

        for sym in self._config:
            try:
                await self._on_daily_close(sym)
            except Exception:
                logger.exception("%s: Error in daily close for %s",
                                 self._strategy_id, sym)

        # Hook 1: Market snapshot + regime classification (post-decision)
        if self._kit:
            for sym in self._config:
                self._kit.capture_snapshot(sym)
                self._kit.classify_regime(sym)

    # ------------------------------------------------------------------
    # Per-symbol daily logic
    # ------------------------------------------------------------------

    async def _on_daily_close(self, symbol: str) -> None:
        """Fetch bars, compute state, evaluate signals for one symbol."""
        contract = self.contracts.get(symbol)
        if not contract:
            return
        cfg = self._config[symbol]

        # Fetch 200 daily bars
        bars = await self._ib.ib.reqHistoricalDataAsync(
            contract, endDateTime="", durationStr="200 D",
            barSizeSetting="1 day", whatToShow="TRADES",
            useRTH=True, formatDate=1,
        )
        if not bars or len(bars) < 50:
            logger.warning("%s: Insufficient bars for %s (%d)",
                           self._strategy_id, symbol, len(bars) if bars else 0)
            return

        # Build numpy arrays
        closes = np.array([b.close for b in bars], dtype=float)
        highs = np.array([b.high for b in bars], dtype=float)
        lows = np.array([b.low for b in bars], dtype=float)
        volumes = np.array([b.volume for b in bars], dtype=float)

        if self._kit and len(closes) > 0:
            self._kit.record_close(symbol, float(closes[-1]))

        # Compute DailyState (same as backtest S5Engine._compute_state)
        state = self._compute_state(closes, highs, lows, volumes, cfg)

        today_high = float(highs[-1])
        today_low = float(lows[-1])

        # --- Position management ---
        pos = self.positions.get(symbol)
        if pos is not None:
            pos.bars_held += 1
            # Update MFE / MAE
            if pos.direction == Direction.LONG:
                if today_high > pos.mfe_price:
                    pos.mfe_price = today_high
                if today_low < pos.mae_price:
                    pos.mae_price = today_low
            else:
                if today_low < pos.mfe_price:
                    pos.mfe_price = today_low
                if today_high > pos.mae_price:
                    pos.mae_price = today_high

            # Trailing stop
            self._update_trailing_stop(pos, highs, lows, closes, cfg)

            # Signal exit
            should_exit, reason = signals.should_exit_signal(
                state, pos.direction, cfg,
            )
            if should_exit:
                await self._submit_signal_exit(symbol, pos, reason)
            return

        # --- Entry evaluation (flat) ---
        if symbol in self._pending_entry:
            return  # already have a pending entry order

        direction = signals.entry_signal(state, cfg)

        # Emit indicator snapshot at Keltner entry evaluation
        if self._kit:
            self._kit.on_indicator_snapshot(
                pair=symbol,
                indicators={
                    "close": state.close,
                    "kelt_upper": state.kelt_upper,
                    "kelt_middle": state.kelt_middle,
                    "kelt_lower": state.kelt_lower,
                    "rsi": state.rsi,
                    "roc": state.roc,
                    "volume": state.volume,
                    "volume_sma": state.volume_sma,
                    "atr": state.atr,
                },
                signal_name=f"keltner_{cfg.entry_mode}",
                signal_strength=0.0,
                decision="enter" if direction != Direction.FLAT else "skip",
                strategy_id=self._strategy_id,
            )
            if direction != Direction.FLAT:
                _snap = self._kit.capture_snapshot(symbol)
                if _snap:
                    self._kit.on_orderbook_context(
                        pair=symbol,
                        best_bid=_snap.get("bid", 0),
                        best_ask=_snap.get("ask", 0),
                        trade_context="signal_eval",
                    )

        if direction == Direction.FLAT:
            # Log missed if price was outside Keltner bands but volume filter blocked
            if self._kit:
                vol_blocked = (cfg.volume_filter and state.volume_sma > 0
                               and state.volume < state.volume_sma)
                price_outside = (state.close > state.kelt_upper
                                 or state.close < state.kelt_lower)
                if vol_blocked and price_outside:
                    missed_side = "LONG" if state.close > state.kelt_upper else "SHORT"
                    self._kit.on_filter_decision(
                        pair=symbol, filter_name="volume_filter",
                        passed=False, threshold=state.volume_sma,
                        actual_value=state.volume,
                        signal_name=f"keltner_{cfg.entry_mode}",
                        strategy_id=self._strategy_id,
                    )
                    self._kit.log_missed(
                        pair=symbol,
                        side=missed_side,
                        signal="keltner_breakout",
                        signal_id=f"{symbol}_vol_filter_{datetime.now(timezone.utc).date()}",
                        signal_strength=0.0,
                        blocked_by="volume_filter",
                        block_reason=f"volume {state.volume:.0f} < SMA {state.volume_sma:.0f}",
                    )
            return

        await self._submit_entry(symbol, state, direction, cfg)

    def _compute_state(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        cfg: SymbolConfig,
    ) -> DailyState:
        """Compute DailyState from full bar arrays (identical to backtest)."""
        kelt_upper, kelt_middle, kelt_lower = indicators.keltner_channel(
            closes, highs, lows,
            cfg.kelt_ema_period, cfg.kelt_atr_period, cfg.kelt_atr_mult,
        )
        rsi_arr = indicators.rsi(closes, cfg.rsi_period)
        roc_arr = indicators.roc(closes, cfg.roc_period)
        vol_sma_arr = indicators.volume_sma(volumes, cfg.vol_sma_period)
        atr_arr = indicators.atr(highs, lows, closes, cfg.atr_period)

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

    # ------------------------------------------------------------------
    # Entry submission
    # ------------------------------------------------------------------

    async def _submit_entry(
        self,
        symbol: str,
        state: DailyState,
        direction: Direction,
        cfg: SymbolConfig,
    ) -> None:
        """Size, submit MARKET entry via OMS."""
        if self._risk_halted:
            logger.warning(
                "%s: entry suppressed for %s while OMS risk halt is active: %s",
                self._strategy_id,
                symbol,
                self._risk_halt_reason or "unspecified",
            )
            return
        atr_val = state.atr
        if atr_val <= 0:
            return

        stop_dist = cfg.atr_stop_mult * atr_val

        inst = self._instruments.get(symbol)
        if inst is None:
            return
        point_value = inst.point_value if hasattr(inst, "point_value") else 1.0

        # Position sizing: risk-based
        risk_per_share = stop_dist * point_value
        if risk_per_share <= 0:
            return
        raw_qty = (self._equity * cfg.base_risk_pct) / risk_per_share
        qty = max(1, int(raw_qty))

        # Cap at 20% notional
        if state.close > 0:
            max_qty = int((self._equity * 0.20) / state.close)
            qty = min(qty, max(1, max_qty))

        # Stop price
        if direction == Direction.LONG:
            stop_price = state.close - stop_dist
        else:
            stop_price = state.close + stop_dist
        stop_price = round_to_tick(stop_price, cfg.tick_size, "nearest")

        # Risk context
        risk_dollars = RiskCalculator.compute_order_risk_dollars(
            state.close, stop_price, qty, point_value,
        )

        side = OrderSide.BUY if direction == Direction.LONG else OrderSide.SELL
        entry_order = OMSOrder(
            strategy_id=self._strategy_id,
            instrument=inst,
            side=side,
            qty=qty,
            order_type=OrderType.MARKET,
            tif="GTC",
            role=OrderRole.ENTRY,
            entry_policy=EntryPolicy(ttl_seconds=24 * 3600),
            risk_context=RiskContext(
                stop_for_risk=stop_price,
                planned_entry_price=state.close,
                risk_dollars=risk_dollars,
            ),
        )

        receipt = await self._oms.submit_intent(
            Intent(
                intent_type=IntentType.NEW_ORDER,
                strategy_id=self._strategy_id,
                order=entry_order,
            )
        )
        if receipt.oms_order_id:
            self._order_to_symbol[receipt.oms_order_id] = symbol
            self._order_role[receipt.oms_order_id] = "entry"
            self._pending_entry[symbol] = {
                "direction": direction,
                "stop_dist": stop_dist,
                # Carry-forward for enriched telemetry
                "rsi": state.rsi,
                "roc": state.roc,
                "kelt_upper": state.kelt_upper,
                "kelt_middle": state.kelt_middle,
                "kelt_lower": state.kelt_lower,
                "volume": state.volume,
                "volume_sma": state.volume_sma,
                "atr": state.atr,
                "close": state.close,
            }
            logger.info(
                "%s: %s entry %s submitted — qty=%d, est_stop=%.2f",
                self._strategy_id, symbol, direction.name, qty, stop_price,
            )

            if self._kit:
                self._kit.on_order_event(
                    order_id=receipt.oms_order_id,
                    pair=symbol,
                    side="LONG" if direction == Direction.LONG else "SHORT",
                    order_type="MARKET",
                    status="SUBMITTED",
                    requested_qty=float(qty),
                    requested_price=state.close,
                    strategy_id=self._strategy_id,
                )

    # ------------------------------------------------------------------
    # Protective stop
    # ------------------------------------------------------------------

    async def _submit_protective_stop(
        self, symbol: str, pos: _LivePosition,
    ) -> None:
        """Submit standing STOP order for a position."""
        inst = self._instruments.get(symbol)
        if inst is None:
            return

        cfg = self._config[symbol]
        exit_side = OrderSide.SELL if pos.direction == Direction.LONG else OrderSide.BUY
        stop_order = OMSOrder(
            strategy_id=self._strategy_id,
            instrument=inst,
            side=exit_side,
            qty=pos.qty,
            order_type=OrderType.STOP,
            stop_price=round_to_tick(pos.current_stop, cfg.tick_size, "nearest"),
            tif="GTC",
            role=OrderRole.STOP,
        )

        receipt = await self._oms.submit_intent(
            Intent(
                intent_type=IntentType.NEW_ORDER,
                strategy_id=self._strategy_id,
                order=stop_order,
            )
        )
        if receipt.oms_order_id:
            pos.stop_order_id = receipt.oms_order_id
            self._order_to_symbol[receipt.oms_order_id] = symbol
            self._order_role[receipt.oms_order_id] = "stop"
            logger.info(
                "%s: %s protective stop placed at %.2f (order %s)",
                self._strategy_id, symbol, pos.current_stop,
                receipt.oms_order_id,
            )

    # ------------------------------------------------------------------
    # Trailing stop
    # ------------------------------------------------------------------

    def _update_trailing_stop(
        self,
        pos: _LivePosition,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        cfg: SymbolConfig,
    ) -> None:
        """Ratchet trailing stop when R >= 1.0 (same logic as backtest)."""
        if pos.r_price <= 0:
            return

        # Current R from MFE
        if pos.direction == Direction.LONG:
            current_r = (pos.mfe_price - pos.fill_price) / pos.r_price
        else:
            current_r = (pos.fill_price - pos.mfe_price) / pos.r_price

        if current_r < 1.0:
            return

        # Compute current ATR
        if len(closes) < cfg.atr_period + 1:
            return
        atr_arr = indicators.atr(highs, lows, closes, cfg.atr_period)
        cur_atr = float(atr_arr[-1])

        trail_dist = cfg.trail_atr_mult * cur_atr
        if pos.direction == Direction.LONG:
            new_stop = pos.mfe_price - trail_dist
            new_stop = round_to_tick(new_stop, cfg.tick_size, "down")
            if new_stop > pos.current_stop:
                old_stop = pos.current_stop
                pos.current_stop = new_stop
                self._schedule_stop_replace(pos.symbol, pos, old_stop)
                if self._kit:
                    self._kit.log_stop_adjustment(
                        trade_id=pos.trade_id or f"KELT-{pos.symbol}",
                        symbol=pos.symbol, old_stop=old_stop, new_stop=new_stop,
                        adjustment_type="trailing", trigger="atr_mfe_trail",
                    )
        else:
            new_stop = pos.mfe_price + trail_dist
            new_stop = round_to_tick(new_stop, cfg.tick_size, "up")
            if new_stop < pos.current_stop:
                old_stop = pos.current_stop
                pos.current_stop = new_stop
                self._schedule_stop_replace(pos.symbol, pos, old_stop)
                if self._kit:
                    self._kit.log_stop_adjustment(
                        trade_id=pos.trade_id or f"KELT-{pos.symbol}",
                        symbol=pos.symbol, old_stop=old_stop, new_stop=new_stop,
                        adjustment_type="trailing", trigger="atr_mfe_trail",
                    )

    def _schedule_stop_replace(
        self, symbol: str, pos: _LivePosition, old_stop: float,
    ) -> None:
        """Fire-and-forget REPLACE_ORDER to ratchet the standing stop."""
        if not pos.stop_order_id:
            return

        cfg = self._config[symbol]
        new_price = round_to_tick(pos.current_stop, cfg.tick_size, "nearest")

        async def _do_replace() -> None:
            try:
                await self._oms.submit_intent(
                    Intent(
                        intent_type=IntentType.REPLACE_ORDER,
                        strategy_id=self._strategy_id,
                        target_oms_order_id=pos.stop_order_id,
                        new_stop_price=new_price,
                    )
                )
                logger.info(
                    "%s: %s trailing stop ratcheted %.2f → %.2f",
                    self._strategy_id, symbol, old_stop, new_price,
                )
            except Exception:
                logger.exception("%s: Failed to replace stop for %s",
                                 self._strategy_id, symbol)

        asyncio.create_task(_do_replace())

    # ------------------------------------------------------------------
    # Signal exit
    # ------------------------------------------------------------------

    async def _submit_signal_exit(
        self, symbol: str, pos: _LivePosition, reason: str,
    ) -> None:
        """Cancel standing stop, submit MARKET exit."""
        # Cancel existing stop
        if pos.stop_order_id:
            try:
                await self._oms.submit_intent(
                    Intent(
                        intent_type=IntentType.CANCEL_ORDER,
                        strategy_id=self._strategy_id,
                        target_oms_order_id=pos.stop_order_id,
                    )
                )
            except Exception:
                logger.warning("%s: Failed to cancel stop for %s",
                               self._strategy_id, symbol)

        inst = self._instruments.get(symbol)
        if inst is None:
            return

        exit_side = OrderSide.SELL if pos.direction == Direction.LONG else OrderSide.BUY
        exit_order = OMSOrder(
            strategy_id=self._strategy_id,
            instrument=inst,
            side=exit_side,
            qty=pos.qty,
            order_type=OrderType.MARKET,
            tif="GTC",
            role=OrderRole.EXIT,
        )

        receipt = await self._oms.submit_intent(
            Intent(
                intent_type=IntentType.NEW_ORDER,
                strategy_id=self._strategy_id,
                order=exit_order,
            )
        )
        if receipt.oms_order_id:
            self._order_to_symbol[receipt.oms_order_id] = symbol
            self._order_role[receipt.oms_order_id] = "signal_exit"
            logger.info(
                "%s: %s signal exit submitted — reason=%s",
                self._strategy_id, symbol, reason,
            )

    # ------------------------------------------------------------------
    # OMS event processing
    # ------------------------------------------------------------------

    async def _process_events(self, event_queue: asyncio.Queue) -> None:
        """Poll OMS event queue, route fills and terminal events."""
        while self._running:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                oms_order_id = getattr(event, "oms_order_id", None)
                if not oms_order_id:
                    continue

                symbol = self._order_to_symbol.get(oms_order_id)
                if not symbol:
                    continue

                role = self._order_role.get(oms_order_id, "")
                evt = event.event_type

                if evt == OMSEventType.RISK_HALT:
                    await self._on_risk_halt((event.payload or {}).get("reason", ""))
                elif evt == OMSEventType.FILL:
                    fill_price = float((event.payload or {}).get("price", 0))
                    fill_qty = int((event.payload or {}).get("qty", 0))
                    await self._on_fill(symbol, role, oms_order_id,
                                        fill_price, fill_qty)

                elif evt == OMSEventType.ORDER_FILLED:
                    fill_price = float((event.payload or {}).get("avg_fill_price", 0))
                    fill_qty = int((event.payload or {}).get("filled_qty", 0))
                    await self._on_fill(symbol, role, oms_order_id,
                                        fill_price, fill_qty)

                elif evt in (OMSEventType.ORDER_CANCELLED,
                             OMSEventType.ORDER_REJECTED,
                             OMSEventType.ORDER_EXPIRED):
                    self._on_terminal(symbol, role, oms_order_id, evt)

            except Exception:
                logger.exception("%s: Error processing OMS event",
                                 self._strategy_id)

    async def _on_risk_halt(self, reason: str) -> None:
        """Pause new entries and cancel outstanding entry intents."""
        if self._risk_halted:
            return

        self._risk_halted = True
        self._risk_halt_reason = reason or "OMS risk halt"
        logger.error("%s: risk halt engaged: %s", self._strategy_id, self._risk_halt_reason)

        for oms_order_id, role in list(self._order_role.items()):
            if role != "entry":
                continue
            try:
                await self._oms.submit_intent(
                    Intent(
                        intent_type=IntentType.CANCEL_ORDER,
                        strategy_id=self._strategy_id,
                        target_oms_order_id=oms_order_id,
                    )
                )
            except Exception:
                logger.warning(
                    "%s: failed to cancel entry order %s during risk halt",
                    self._strategy_id,
                    oms_order_id,
                )

    async def _on_fill(
        self,
        symbol: str,
        role: str,
        oms_order_id: str,
        fill_price: float,
        fill_qty: int,
    ) -> None:
        """Handle a fill event based on order role."""
        cfg = self._config.get(symbol)
        if cfg is None:
            return

        if role == "entry":
            ctx = self._pending_entry.pop(symbol, None)
            if ctx is None:
                return

            direction: Direction = ctx["direction"]
            stop_dist: float = ctx["stop_dist"]

            if direction == Direction.LONG:
                stop_price = fill_price - stop_dist
            else:
                stop_price = fill_price + stop_dist
            stop_price = round_to_tick(stop_price, cfg.tick_size, "nearest")
            r_price = abs(fill_price - stop_price)

            pos = _LivePosition(
                symbol=symbol,
                direction=direction,
                fill_price=fill_price,
                qty=fill_qty,
                initial_stop=stop_price,
                current_stop=stop_price,
                r_price=r_price,
                entry_time=datetime.now(timezone.utc),
                mfe_price=fill_price,
                mae_price=fill_price,
            )
            self.positions[symbol] = pos
            logger.info(
                "%s: %s entry filled %d @ %.2f, stop=%.2f",
                self._strategy_id, symbol, fill_qty, fill_price, stop_price,
            )

            # Hook 4: Instrumentation trade entry
            if self._kit:
                side_str = "LONG" if direction == Direction.LONG else "SHORT"
                pending = ctx  # ctx was popped from _pending_entry
                vol_ratio = pending.get("volume", 0) / pending.get("volume_sma", 1) if pending.get("volume_sma", 0) > 0 else 0
                _cfg_dict = dataclasses.asdict(cfg)
                _param_set_id = hashlib.md5(
                    json.dumps(_cfg_dict, sort_keys=True, default=str).encode()
                ).hexdigest()[:8]
                self._kit.log_entry(
                    trade_id=f"{symbol}_{pos.entry_time.isoformat()}",
                    pair=symbol,
                    side=side_str,
                    entry_price=fill_price,
                    position_size=float(fill_qty),
                    position_size_quote=fill_price * fill_qty,
                    entry_signal="keltner_breakout",
                    entry_signal_id=f"{symbol}_kelt_{pos.entry_time.isoformat()}",
                    entry_signal_strength=0.5,
                    active_filters=["volume_filter", "rsi_threshold"],
                    passed_filters=["volume_filter", "rsi_threshold"],
                    filter_decisions=[
                        {"filter_name": "volume_filter", "threshold": pending.get("volume_sma", 0),
                         "actual_value": pending.get("volume", 0), "passed": True,
                         "margin_pct": round((vol_ratio - 1.0) * 100, 1) if vol_ratio > 0 else 0},
                        {"filter_name": "rsi_threshold",
                         "threshold": cfg.rsi_entry_long if direction == Direction.LONG else cfg.rsi_entry_short,
                         "actual_value": pending.get("rsi", 50), "passed": True,
                         "margin_pct": 0},
                    ],
                    strategy_params={
                        "param_set_id": _param_set_id,
                        "config": _cfg_dict,
                        "stop_dist": stop_dist,
                        "r_price": r_price,
                        "kelt_ema_period": cfg.kelt_ema_period,
                        "kelt_atr_mult": cfg.kelt_atr_mult,
                        "rsi_period": cfg.rsi_period,
                        "roc_period": cfg.roc_period,
                        "atr_stop_mult": cfg.atr_stop_mult,
                        "entry_mode": cfg.entry_mode,
                        "base_risk_pct": cfg.base_risk_pct,
                        "rsi_at_entry": pending.get("rsi"),
                        "roc_at_entry": pending.get("roc"),
                        "atr_at_entry": pending.get("atr"),
                    },
                    expected_entry_price=fill_price,
                    signal_factors=[
                        {"factor_name": "rsi", "factor_value": pending.get("rsi", 50),
                         "threshold": cfg.rsi_entry_long if direction == Direction.LONG else cfg.rsi_entry_short,
                         "contribution": "momentum_filter"},
                        {"factor_name": "roc", "factor_value": pending.get("roc", 0),
                         "threshold": 0.0, "contribution": "rate_of_change"},
                        {"factor_name": "volume_ratio", "factor_value": round(vol_ratio, 2),
                         "threshold": 1.0, "contribution": "volume_confirmation"},
                        {"factor_name": "entry_mode", "factor_value": cfg.entry_mode,
                         "threshold": "breakout", "contribution": "signal_type"},
                    ],
                    sizing_inputs={
                        "target_risk_pct": cfg.base_risk_pct,
                        "account_equity": self._equity,
                        "volatility_basis": stop_dist,
                        "sizing_model": "keltner_atr",
                    },
                    portfolio_state_at_entry={
                        "num_positions": len(self.positions),
                        "symbols_held": list(self.positions.keys()),
                    },
                    concurrent_positions_strategy=len(self.positions),
                )

                self._kit.on_order_event(
                    order_id=oms_order_id,
                    pair=symbol,
                    side="LONG" if direction == Direction.LONG else "SHORT",
                    order_type="MARKET",
                    status="FILLED",
                    requested_qty=float(fill_qty),
                    filled_qty=float(fill_qty),
                    requested_price=fill_price,
                    fill_price=fill_price,
                    related_trade_id=f"{symbol}_{pos.entry_time.isoformat()}",
                    strategy_id=self._strategy_id,
                )

            # Submit protective stop
            await self._submit_protective_stop(symbol, pos)

        elif role == "stop":
            pos = self.positions.pop(symbol, None)
            if pos:
                logger.info(
                    "%s: %s stop filled @ %.2f — position closed",
                    self._strategy_id, symbol, fill_price,
                )
                await self._record_trade(pos, fill_price, "STOP")
                # Hook 5: Instrumentation trade exit + process scoring
                if self._kit:
                    tid = f"{symbol}_{pos.entry_time.isoformat()}"
                    # Compute MFE/MAE metrics
                    if pos.direction == Direction.LONG:
                        _mfe_pct = (pos.mfe_price - pos.fill_price) / pos.fill_price if pos.fill_price > 0 else None
                        _mae_pct = (pos.fill_price - pos.mae_price) / pos.fill_price if pos.fill_price > 0 else None
                        _pnl_pct = (fill_price - pos.fill_price) / pos.fill_price if pos.fill_price > 0 else None
                        _mfe_r = (pos.mfe_price - pos.fill_price) / pos.r_price if pos.r_price > 0 else None
                        _mae_r = (pos.fill_price - pos.mae_price) / pos.r_price if pos.r_price > 0 else None
                    else:
                        _mfe_pct = (pos.fill_price - pos.mfe_price) / pos.fill_price if pos.fill_price > 0 else None
                        _mae_pct = (pos.mae_price - pos.fill_price) / pos.fill_price if pos.fill_price > 0 else None
                        _pnl_pct = (pos.fill_price - fill_price) / pos.fill_price if pos.fill_price > 0 else None
                        _mfe_r = (pos.fill_price - pos.mfe_price) / pos.r_price if pos.r_price > 0 else None
                        _mae_r = (pos.mae_price - pos.fill_price) / pos.r_price if pos.r_price > 0 else None
                    self._kit.log_exit(
                        trade_id=tid, exit_price=fill_price,
                        exit_reason="STOP_LOSS",
                        mfe_price=pos.mfe_price, mae_price=pos.mae_price,
                        mfe_r=_mfe_r, mae_r=_mae_r,
                        mfe_pct=_mfe_pct, mae_pct=_mae_pct,
                        pnl_pct=_pnl_pct,
                    )

                    self._kit.on_order_event(
                        order_id=oms_order_id,
                        pair=symbol,
                        side="SELL" if pos.direction == Direction.LONG else "BUY",
                        order_type="STOP",
                        status="FILLED",
                        requested_qty=float(pos.qty),
                        filled_qty=float(fill_qty),
                        requested_price=pos.current_stop,
                        fill_price=fill_price,
                        related_trade_id=tid,
                        strategy_id=self._strategy_id,
                    )

        elif role == "signal_exit":
            pos = self.positions.pop(symbol, None)
            if pos:
                logger.info(
                    "%s: %s signal exit filled @ %.2f — position closed",
                    self._strategy_id, symbol, fill_price,
                )
                await self._record_trade(pos, fill_price, "SIGNAL_EXIT")
                # Hook 5: Instrumentation trade exit + process scoring
                if self._kit:
                    tid = f"{symbol}_{pos.entry_time.isoformat()}"
                    # Compute MFE/MAE metrics
                    if pos.direction == Direction.LONG:
                        _mfe_pct = (pos.mfe_price - pos.fill_price) / pos.fill_price if pos.fill_price > 0 else None
                        _mae_pct = (pos.fill_price - pos.mae_price) / pos.fill_price if pos.fill_price > 0 else None
                        _pnl_pct = (fill_price - pos.fill_price) / pos.fill_price if pos.fill_price > 0 else None
                        _mfe_r = (pos.mfe_price - pos.fill_price) / pos.r_price if pos.r_price > 0 else None
                        _mae_r = (pos.fill_price - pos.mae_price) / pos.r_price if pos.r_price > 0 else None
                    else:
                        _mfe_pct = (pos.fill_price - pos.mfe_price) / pos.fill_price if pos.fill_price > 0 else None
                        _mae_pct = (pos.mae_price - pos.fill_price) / pos.fill_price if pos.fill_price > 0 else None
                        _pnl_pct = (pos.fill_price - fill_price) / pos.fill_price if pos.fill_price > 0 else None
                        _mfe_r = (pos.fill_price - pos.mfe_price) / pos.r_price if pos.r_price > 0 else None
                        _mae_r = (pos.mae_price - pos.fill_price) / pos.r_price if pos.r_price > 0 else None
                    self._kit.log_exit(
                        trade_id=tid, exit_price=fill_price,
                        exit_reason="SIGNAL",
                        mfe_price=pos.mfe_price, mae_price=pos.mae_price,
                        mfe_r=_mfe_r, mae_r=_mae_r,
                        mfe_pct=_mfe_pct, mae_pct=_mae_pct,
                        pnl_pct=_pnl_pct,
                    )

                    self._kit.on_order_event(
                        order_id=oms_order_id,
                        pair=symbol,
                        side="SELL" if pos.direction == Direction.LONG else "BUY",
                        order_type="MARKET",
                        status="FILLED",
                        requested_qty=float(pos.qty),
                        filled_qty=float(fill_qty),
                        requested_price=fill_price,
                        fill_price=fill_price,
                        related_trade_id=tid,
                        strategy_id=self._strategy_id,
                    )

        # Cleanup order tracking
        self._order_to_symbol.pop(oms_order_id, None)
        self._order_role.pop(oms_order_id, None)

    def _on_terminal(self, symbol: str, role: str, oms_order_id: str, evt: Any = None) -> None:
        """Handle cancel/reject/expire — clean up pending state."""
        if role == "entry":
            ctx = self._pending_entry.pop(symbol, None)
            logger.info(
                "%s: %s entry order terminal (%s)",
                self._strategy_id, symbol, evt,
            )

            if self._kit and ctx:
                status_map = {
                    OMSEventType.ORDER_REJECTED: "REJECTED",
                    OMSEventType.ORDER_CANCELLED: "CANCELLED",
                    OMSEventType.ORDER_EXPIRED: "EXPIRED",
                }
                self._kit.on_order_event(
                    order_id=oms_order_id,
                    pair=symbol,
                    side="LONG" if ctx.get("direction") == Direction.LONG else "SHORT",
                    order_type="MARKET",
                    status=status_map.get(evt, "CANCELLED"),
                    requested_qty=0,
                    requested_price=ctx.get("close", 0),
                    strategy_id=self._strategy_id,
                )
        elif role == "stop":
            # Stop cancelled externally — position still open, will be
            # re-evaluated next daily cycle
            logger.warning(
                "%s: %s stop order terminal — position may be unprotected",
                self._strategy_id, symbol,
            )

        self._order_to_symbol.pop(oms_order_id, None)
        self._order_role.pop(oms_order_id, None)

    # ------------------------------------------------------------------
    # Trade recording
    # ------------------------------------------------------------------

    async def _record_trade(
        self, pos: _LivePosition, exit_price: float, reason: str,
    ) -> None:
        """Persist trade record via TradeRecorder."""
        if not self._recorder:
            return

        point_value = 1.0
        inst = self._instruments.get(pos.symbol)
        if inst and hasattr(inst, "point_value"):
            point_value = inst.point_value

        pnl_points = (exit_price - pos.fill_price) * int(pos.direction)
        pnl_dollars = pnl_points * pos.qty * point_value
        r_mult = pnl_points / pos.r_price if pos.r_price > 0 else 0.0

        record = {
            "strategy_id": self._strategy_id,
            "symbol": pos.symbol,
            "direction": int(pos.direction),
            "entry_time": pos.entry_time.isoformat(),
            "exit_time": datetime.now(timezone.utc).isoformat(),
            "entry_price": pos.fill_price,
            "exit_price": exit_price,
            "qty": pos.qty,
            "initial_stop": pos.initial_stop,
            "exit_reason": reason,
            "pnl_points": pnl_points,
            "pnl_dollars": pnl_dollars,
            "r_multiple": r_mult,
            "bars_held": pos.bars_held,
        }
        try:
            await self._recorder.record(record)
        except Exception:
            logger.exception("%s: Failed to record trade for %s",
                             self._strategy_id, pos.symbol)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _refresh_equity(self) -> None:
        """Fetch current account equity from IB."""
        try:
            accounts = self._ib.ib.managedAccounts()
            if accounts:
                for item in self._ib.ib.accountValues():
                    if item.tag == "NetLiquidation" and item.currency == "USD" and item.account == accounts[0]:
                        raw = float(item.value)
                        self._equity = raw * self._equity_alloc_pct + self._equity_offset
                        if self._kit and self._kit.ctx and self._kit.ctx.drawdown_tracker:
                            self._kit.ctx.drawdown_tracker.update_equity(self._equity)
                        return
        except Exception:
            logger.warning("%s: Could not refresh equity, using $%.2f",
                           self._strategy_id, self._equity)
