"""BRS R9 live engine -- async event-driven, multi-symbol (QQQ + GLD).

Follows the Helix4Engine async pattern: subscribe to IB historical bar
streams with keepUpToDate=True, process each new bar in an eval lock,
execute entries/exits via the OMS intent system.
"""
from __future__ import annotations

import asyncio
import logging
import math
import uuid
from copy import deepcopy
from datetime import datetime, time, timedelta, timezone
from typing import Any, Optional

import numpy as np
from zoneinfo import ZoneInfo

from .bar_series import Bar, BRSBarSeries
from .config import (
    BRS_SYMBOL_DEFAULTS,
    STRATEGY_ID,
    TF,
    BRSConfig,
    BRSSymbolConfig,
)
from .execution import BRSExecutionEngine
from .indicators import (
    compute_4h_structure,
    compute_bear_conviction,
    compute_risk_regime,
    compute_vol_factor,
)
from .models import (
    BDArmState,
    BRSRegime,
    BiasState,
    DailyContext,
    Direction,
    EntrySignal,
    EntryType,
    HourlyContext,
    LHArmState,
    Regime4H,
    S2ArmState,
    S3ArmState,
    VolState,
)
from .core import logic as brs_core_logic
from .core.logic import apply_core_state as apply_core_runtime_state
from .core.logic import build_core_state as build_core_runtime_state
from .core.serializers import restore_state as restore_core_state
from .core.serializers import snapshot_state as snapshot_core_state
from .core.state import (
    BRSAddOnRequest,
    BRSEntryFillContext,
    BRSEntryRequest,
    BRSFill,
    BRSFlattenRequest,
    BRSOrderUpdate,
    BRSScaleOutRequest,
    BRSStopUpdateRequest,
)
from strategies.core.actions import (
    FlattenPosition,
    ReplaceProtectiveStop,
    SubmitAddOnEntry,
    SubmitEntry,
    SubmitExit,
    SubmitPartialExit,
)
from .positions import ActionResult, BRSPositionState, PendingOrder, PositionAction
from .regime import classify_regime, compute_raw_bias, compute_regime_on, update_bias
from .signals import (
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
from .sizing import compute_position_size
from libs.oms.models.events import OMSEventType

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# RTH window
RTH_START = time(9, 30)
RTH_END = time(16, 0)


class BRSLiveEngine:
    """Async event-driven BRS engine managing QQQ + GLD."""

    def __init__(
        self,
        ib_session: Any,
        oms_service: Any,
        instruments: list[Any],
        trade_recorder: Any = None,
        equity: float = 10_000.0,
        instrumentation: Any = None,
        cfg: BRSConfig | None = None,
        equity_alloc_pct: float = 1.0,
    ) -> None:
        self._cfg = cfg or BRSConfig()
        self._sym_cfgs: dict[str, BRSSymbolConfig] = {
            sym: self._cfg.get_symbol_config(sym) for sym in self._cfg.symbols
        }
        self._symbols = list(self._cfg.symbols)

        self.ib = ib_session
        self._oms = oms_service
        self._instruments_list = instruments
        self._instruments_map: dict[str, Any] = {i.symbol: i for i in instruments}
        self._trade_recorder = trade_recorder
        self._equity = equity
        self._equity_alloc_pct = equity_alloc_pct
        self._instrumentation = instrumentation

        # Per-symbol bar series (3 TF each)
        self._daily_series: dict[str, BRSBarSeries] = {}
        self._h4_series: dict[str, BRSBarSeries] = {}
        self._h1_series: dict[str, BRSBarSeries] = {}

        # Per-symbol state
        self._daily_ctx: dict[str, DailyContext] = {}
        self._bias: dict[str, BiasState] = {}
        self._prev_regime_on: dict[str, bool] = {}
        self._prev_regime: dict[str, BRSRegime] = {}
        self._position: dict[str, Optional[BRSPositionState]] = {}
        self._cooldown_until: dict[str, datetime] = {}
        self._lh_arm: dict[str, LHArmState] = {}
        self._bd_arm: dict[str, BDArmState] = {}
        self._s2_arm: dict[str, S2ArmState] = {}
        self._s3_arm: dict[str, S3ArmState] = {}
        self._swing_highs: dict[str, list[float]] = {}
        self._short_bias_no_trade: dict[str, int] = {}
        self._hourly_bar_count: dict[str, int] = {}
        self._last_daily_close: dict[str, float] = {}

        for sym in self._symbols:
            self._daily_series[sym] = BRSBarSeries(TF.D1, maxlen=200)
            self._h4_series[sym] = BRSBarSeries(TF.H4, maxlen=300)
            self._h1_series[sym] = BRSBarSeries(TF.H1, maxlen=500)
            self._daily_ctx[sym] = DailyContext()
            self._bias[sym] = BiasState()
            self._prev_regime_on[sym] = False
            self._prev_regime[sym] = BRSRegime.RANGE_CHOP
            self._position[sym] = None
            self._cooldown_until[sym] = datetime.min.replace(tzinfo=ET)
            self._lh_arm[sym] = LHArmState()
            self._bd_arm[sym] = BDArmState()
            self._s2_arm[sym] = S2ArmState()
            self._s3_arm[sym] = S3ArmState()
            self._swing_highs[sym] = []
            self._short_bias_no_trade[sym] = 0
            self._hourly_bar_count[sym] = 0
            self._last_daily_close[sym] = 0.0

        # Shared state
        self._contracts: dict[str, Any] = {}
        self._bar_streams: list[Any] = []
        self._exec: Optional[BRSExecutionEngine] = None
        self._eval_lock = asyncio.Lock()
        self._running = False

        # Diagnostic pulse state
        self._last_decision_code: str = "IDLE"
        self._last_decision_details: dict[str, Any] = {}
        self._last_bar_ts: datetime | None = None
        self._symbol_last_bar_ts: dict[str, datetime] = {}

        self._bar_processed_count: dict[tuple[str, str], int] = {}  # (symbol, tf_name) -> count
        self._pending_orders: dict[str, PendingOrder] = {}  # oms_order_id -> PendingOrder
        self._filled_order_ids: set[str] = set()  # exec_ids already processed (dedup)
        self._closing: dict[str, str] = {}  # symbol -> exit_reason
        self._event_task: Optional[asyncio.Task] = None

    def _record_decision(self, code: str, details: dict | None = None) -> None:
        self._last_decision_code = code
        self._last_decision_details = details or {}

    # ══════════════════════════════════════════════════════════════════
    # Lifecycle
    # ══════════════════════════════════════════════════════════════════

    async def start(self) -> None:
        """Qualify contracts, subscribe to bar streams, start OMS listener."""
        self._running = True
        self._exec = BRSExecutionEngine(self._oms, self._instruments_map)

        # 1. Qualify ETF contracts
        cf = getattr(self.ib, "_contract_factory", None)
        for sym in self._symbols:
            try:
                if cf is not None:
                    contract, _ = await cf.resolve(
                        sym,
                        instrument=self._instruments_map.get(sym),
                    )
                    self._contracts[sym] = contract
                    logger.info("Qualified contract: %s", sym)
                    continue
                from ib_async import Stock
                contract = Stock(sym, "SMART", "USD")
                qualified = await self.ib.ib.qualifyContractsAsync(contract)
                if qualified:
                    self._contracts[sym] = qualified[0]
                    logger.info("Qualified contract: %s", sym)
                else:
                    logger.error("Failed to qualify: %s", sym)
            except Exception:
                logger.exception("Failed to qualify: %s", sym)

        # 2. Subscribe to historical bars (6 streams: 3 TF x 2 symbols)
        bar_specs = [
            ("1 day",    "200 D", TF.D1),
            ("4 hours",  "60 D",  TF.H4),
            ("1 hour",   "30 D",  TF.H1),
        ]
        for sym in self._symbols:
            contract = self._contracts.get(sym)
            if contract is None:
                continue
            for bar_size, duration, tf in bar_specs:
                try:
                    bars = await self.ib.req_historical_data(
                        contract,
                        endDateTime="",
                        durationStr=duration,
                        barSizeSetting=bar_size,
                        whatToShow="TRADES",
                        useRTH=False,
                        formatDate=1,
                        keepUpToDate=True,
                        request_kind="subscription",
                    )
                    # Backfill from historical bars
                    series = self._get_series(sym, tf)
                    for b in bars[:-1]:
                        series.add_bar(self._convert_ib_bar(b))
                    self._bar_processed_count[(sym, tf.name)] = len(bars) - 1
                    logger.info(
                        "Backfilled %s %s: %d bars", sym, tf.name, len(bars) - 1,
                    )

                    # Process initial daily context from backfill
                    if tf == TF.D1:
                        self._update_daily_context(sym)
                    elif tf == TF.H4:
                        self._update_4h_context(sym)

                    # Register streaming callback
                    bars.updateEvent += (
                        lambda b, new, _s=sym, _t=tf: asyncio.ensure_future(
                            self._on_bar(b, new, _s, _t)
                        )
                    )
                    self._bar_streams.append(bars)
                except Exception:
                    logger.exception("Bar subscription failed: %s %s", sym, bar_size)

        # 3. Subscribe to OMS fill events
        event_queue = self._oms.stream_events(STRATEGY_ID)
        self._event_task = asyncio.create_task(self._process_events(event_queue))

        logger.info("BRSLiveEngine started -- %d streams", len(self._bar_streams))

    async def stop(self) -> None:
        """Cancel bar streams and stop engine."""
        self._running = False
        if self._event_task and not self._event_task.done():
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass
        for bars in self._bar_streams:
            try:
                self.ib.ib.cancelHistoricalData(bars)
            except Exception:
                pass
        self._bar_streams.clear()
        logger.info("BRSLiveEngine stopped")

    def health_status(self) -> dict[str, Any]:
        positions = {s: p.entry_type if p else None for s, p in self._position.items()}
        regimes = {s: c.regime.value for s, c in self._daily_ctx.items()}
        return {
            "strategy_id": STRATEGY_ID,
            "running": self._running,
            "streams": len(self._bar_streams),
            "positions": positions,
            "regimes": regimes,
            "last_decision_code": self._last_decision_code,
            "last_decision_details": self._last_decision_details,
            "last_bar_ts": self._last_bar_ts.isoformat() if self._last_bar_ts else None,
        }

    def liveness_payload(self) -> dict:
        return {
            "bars_processed": sum(self._hourly_bar_count.values()),
            "symbol_freshness": {
                sym: ts.isoformat() for sym, ts in self._symbol_last_bar_ts.items()
            },
        }

    def snapshot_state(self) -> dict[str, Any]:
        return snapshot_core_state(build_core_runtime_state(self))

    async def hydrate(self, snapshot: dict[str, Any]) -> None:
        if not snapshot:
            return
        apply_core_runtime_state(self, restore_core_state(snapshot))

    # ══════════════════════════════════════════════════════════════════
    # Bar routing
    # ══════════════════════════════════════════════════════════════════

    def _get_series(self, symbol: str, tf: TF) -> BRSBarSeries:
        if tf == TF.D1:
            return self._daily_series[symbol]
        elif tf == TF.H4:
            return self._h4_series[symbol]
        return self._h1_series[symbol]

    @staticmethod
    def _convert_ib_bar(ib_bar: Any) -> Bar:
        ts = ib_bar.date if isinstance(ib_bar.date, datetime) else datetime.combine(
            ib_bar.date, time(16, 0), tzinfo=ET,
        )
        if ts.tzinfo is None:
            ts = ET.localize(ts)
        return Bar(
            ts=ts,
            open=float(ib_bar.open),
            high=float(ib_bar.high),
            low=float(ib_bar.low),
            close=float(ib_bar.close),
            volume=float(ib_bar.volume),
        )

    async def _on_bar(self, bars: Any, has_new: bool, symbol: str, tf: TF) -> None:
        if not has_new or not self._running:
            return
        async with self._eval_lock:
            key = (symbol, tf.name)
            last_idx = self._bar_processed_count.get(key, 0)
            new_end = len(bars) - 1  # exclude developing bar
            if new_end <= last_idx:
                return
            series = self._get_series(symbol, tf)
            for i in range(last_idx, new_end):
                series.add_bar(self._convert_ib_bar(bars[i]))
            self._bar_processed_count[key] = new_end

            # Track bar timestamp for diagnostic pulse
            last_ib_bar = bars[new_end - 1] if new_end > 0 else None
            if last_ib_bar is not None:
                bar_ts = last_ib_bar.date if isinstance(last_ib_bar.date, datetime) else None
                if bar_ts is not None:
                    self._last_bar_ts = bar_ts if bar_ts.tzinfo else bar_ts.replace(tzinfo=ET)
                    self._symbol_last_bar_ts[symbol] = self._last_bar_ts

            if tf == TF.D1:
                self._update_daily_context(symbol)
                # Refresh equity from broker on new daily bar (first symbol only)
                if symbol == self._symbols[0]:
                    await self._refresh_equity()
            elif tf == TF.H4:
                self._update_4h_context(symbol)
            elif tf == TF.H1:
                await self._on_hourly_bar(symbol)

    # ══════════════════════════════════════════════════════════════════
    # OMS event processing (fill-driven model)
    # ══════════════════════════════════════════════════════════════════

    async def _process_events(self, event_queue: asyncio.Queue) -> None:
        """Poll OMS event queue, route fills and terminal events."""
        while self._running:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                self._expire_pending_orders()
                continue
            except asyncio.CancelledError:
                break

            try:
                oms_order_id = getattr(event, "oms_order_id", None)
                if not oms_order_id:
                    continue

                evt = event.event_type
                payload = getattr(event, "payload", {}) or {}

                if evt == OMSEventType.FILL:
                    async with self._eval_lock:
                        await self._dispatch_fill(
                            oms_order_id,
                            float(payload.get("price", 0)),
                            int(payload.get("qty", 0)),
                            float(payload.get("commission", 0.0) or 0.0),
                            payload,
                        )

                elif evt == OMSEventType.ORDER_FILLED:
                    async with self._eval_lock:
                        await self._dispatch_fill(
                            oms_order_id,
                            float(payload.get("avg_fill_price", 0)),
                            int(payload.get("filled_qty", 0)),
                            float(payload.get("commission", 0.0) or 0.0),
                            payload,
                        )

                elif evt in (OMSEventType.ORDER_CANCELLED,
                             OMSEventType.ORDER_REJECTED,
                             OMSEventType.ORDER_EXPIRED):
                    self._on_terminal(oms_order_id, evt)

                elif evt == OMSEventType.RISK_HALT:
                    logger.warning("Risk halt: %s", payload.get("reason", ""))

            except Exception:
                logger.exception("Error processing OMS event")

    def _expire_pending_orders(self) -> None:
        """Remove pending orders older than 60s and stale _closing entries."""
        now = datetime.now(timezone.utc)
        expired = [
            oid for oid, po in self._pending_orders.items()
            if (now - po.submitted_at).total_seconds() > 60
        ]
        for oid in expired:
            po = self._pending_orders.pop(oid)
            logger.warning("[%s] Pending %s order expired: %s", po.symbol, po.role, oid)

        # Clear _closing entries with no matching position (flatten already completed
        # or failed silently). Without this, _on_hourly_bar is blocked forever.
        stale_closing = [
            sym for sym in self._closing
            if self._position.get(sym) is None
        ]
        for sym in stale_closing:
            logger.warning("[%s] Clearing stale _closing state", sym)
            self._closing.pop(sym)

    def _on_terminal(self, oms_order_id: str, evt: Any) -> None:
        """Handle cancel/reject/expire through core state machine."""
        # Capture pre-state for instrumentation before core modifies it
        pending = self._pending_orders.get(oms_order_id)
        symbol = ""
        order_role: str = "unknown"
        if pending:
            symbol = pending.symbol
            order_role = pending.role
        else:
            for sym, pos in self._position.items():
                if pos is not None and pos.stop_oms_id == oms_order_id:
                    symbol = sym
                    order_role = "stop"
                    break

        status_str = evt.value if hasattr(evt, "value") else str(evt)

        # Route through core -- pops pending, clears stop_oms_id
        update = BRSOrderUpdate(
            oms_order_id=oms_order_id,
            status=status_str,
            symbol=symbol,
            order_role=order_role,
        )
        state = build_core_runtime_state(self)
        new_state, _, events = brs_core_logic.on_order_update(state, update)
        apply_core_runtime_state(self, new_state)

        # Logging and instrumentation (unchanged behaviour)
        if pending:
            logger.info("[%s] %s order %s (%s)", pending.symbol, pending.role,
                        status_str, oms_order_id)
            if pending.role == "entry" and self._instrumentation:
                self._instrumentation.on_order_event(
                    order_id=oms_order_id,
                    pair=pending.symbol,
                    side="SELL" if pending.signal and pending.signal.direction == Direction.SHORT else "BUY",
                    order_type="LIMIT",
                    status=status_str,
                    requested_qty=float(pending.qty),
                    requested_price=pending.signal.signal_price if pending.signal else 0,
                    strategy_id=STRATEGY_ID,
                )
        elif order_role == "stop":
            logger.warning(
                "[%s] Stop order terminal (%s) -- position may be unprotected",
                symbol, status_str,
            )

    async def _dispatch_fill(
        self, oms_order_id: str, fill_price: float, fill_qty: int,
        fill_commission: float, payload: dict,
    ) -> None:
        """Route fill event through shared core decision machine."""
        # Dedup by exec_id: same fill arrives as both FILL and ORDER_FILLED events
        # with the same exec_id.  Using exec_id (not oms_order_id) preserves
        # distinct partial fills for the same order (different exec_ids).
        fill_exec_id = payload.get("exec_id", oms_order_id)
        if fill_exec_id in self._filled_order_ids:
            return
        self._filled_order_ids.add(fill_exec_id)
        # Cap set size to prevent unbounded growth
        if len(self._filled_order_ids) > 500:
            self._filled_order_ids = set(list(self._filled_order_ids)[-250:])

        # Snapshot positions before core call (needed for exit instrumentation)
        pre_positions = {sym: pos for sym, pos in self._position.items() if pos is not None}

        # Build entry context if this is an entry fill
        pending = self._pending_orders.get(oms_order_id)
        entry_context = None
        if pending is not None and pending.role == "entry" and pending.signal:
            symbol = pending.symbol
            d = self._daily_ctx[symbol]
            sym_cfg = self._sym_cfgs[symbol]
            cfg = self._cfg
            cd_bars = sym_cfg.cooldown_bars
            if d.regime == BRSRegime.BEAR_STRONG:
                cd_bars = cfg.cooldown_bear_strong
            elif d.regime == BRSRegime.BEAR_TREND:
                cd_bars = cfg.cooldown_bear_trend
            entry_context = BRSEntryFillContext(
                cooldown_until=datetime.now(ET) + timedelta(hours=cd_bars),
                reset_arm_state=pending.signal.entry_type,
                reset_short_bias=True,
                tranche_b_qty=0,
            )

        # Build fill and route through core
        fill = BRSFill(
            oms_order_id=oms_order_id,
            fill_price=fill_price,
            fill_qty=fill_qty,
            fill_time=datetime.now(timezone.utc),
            commission=fill_commission,
            symbol=payload.get("symbol", ""),
            entry_context=entry_context,
        )
        core_state = build_core_runtime_state(self)
        new_state, actions, events = brs_core_logic.on_fill(core_state, fill)
        apply_core_runtime_state(self, new_state)

        # Post-processing: setup scale-out schedule for new entry positions
        if pending is not None and pending.role == "entry" and pending.signal:
            pos = self._position.get(pending.symbol)
            if pos is not None:
                pos.setup_scale_out(self._cfg)

        # Dispatch core actions (protective stop after entry fill)
        for action in actions:
            if isinstance(action, SubmitExit):
                symbol = action.symbol
                pos = self._position.get(symbol)
                if pos is not None:
                    stop_id = await self._exec.place_stop(symbol, pos, action.stop_price)
                    if stop_id:
                        pos.stop_oms_id = stop_id

        # Process events for equity updates, trade recording, and instrumentation
        for event in events:
            if event.code == "ENTRY_FILLED":
                await self._on_entry_fill_instrumentation(
                    event, pending, fill_price, fill_qty, fill_commission,
                )
            elif event.code == "ADD_ON_FILLED":
                self._on_pyramid_fill_instrumentation(
                    event, pending, fill_price, fill_qty, fill_commission,
                )
            elif event.code == "PARTIAL_EXIT_FILLED":
                # Equity update for intermediate scale-out
                pre_pos = pre_positions.get(event.symbol)
                if pre_pos is not None:
                    is_short = pre_pos.direction == Direction.SHORT
                    exit_qty = event.details.get("qty", fill_qty)
                    chunk_pnl = ((pre_pos.entry_price - fill_price) if is_short else
                                 (fill_price - pre_pos.entry_price)) * exit_qty
                    self._equity += chunk_pnl - fill_commission
                self._on_scale_out_fill_instrumentation(
                    event, pending, fill_price, fill_qty, fill_commission,
                )
            elif event.code == "EXIT_FILLED":
                pre_pos = pre_positions.get(event.symbol)
                if pre_pos is not None:
                    exit_qty = event.details.get("qty", fill_qty)
                    is_short = pre_pos.direction == Direction.SHORT
                    gross_pnl = ((pre_pos.entry_price - fill_price) if is_short else
                                 (fill_price - pre_pos.entry_price)) * exit_qty
                    total_commission = pre_pos.commission + fill_commission
                    net_pnl = gross_pnl - total_commission
                    self._equity += net_pnl
                    exit_type = event.details.get("exit_type", "unknown")
                    r_mult = pre_pos.current_r(fill_price)
                    await self._on_exit_fill_instrumentation(
                        event.symbol, pre_pos, fill_price, fill_qty,
                        fill_commission, exit_type, net_pnl, r_mult,
                    )

        if not events:
            logger.debug("Unmatched fill event: %s", oms_order_id)

    async def _on_entry_fill_instrumentation(
        self, event: Any, pending: PendingOrder | None, fill_price: float,
        fill_qty: int, fill_commission: float,
    ) -> None:
        """Log entry fill via instrumentation (state already applied by core)."""
        if pending is None or pending.signal is None:
            return
        signal = pending.signal
        symbol = pending.symbol
        cfg = self._cfg
        sym_cfg = self._sym_cfgs[symbol]
        risk_per_unit = abs(signal.stop_price - fill_price)

        logger.info(
            "[%s] ENTRY FILL %s %s %d @ %.2f (signal=%.2f) stop=%.2f comm=%.4f",
            symbol, signal.entry_type.value, signal.direction.name,
            fill_qty, fill_price, signal.signal_price, signal.stop_price,
            fill_commission,
        )

        if self._instrumentation:
            _side_str = "SHORT" if signal.direction == Direction.SHORT else "LONG"
            concurrent = sum(1 for p in self._position.values() if p is not None)
            current_heat_r = self._compute_heat()
            risk_dollars = fill_qty * risk_per_unit
            d = self._daily_ctx[symbol]
            _bias = self._bias[symbol]
            _heat_cap = cfg.crisis_heat_cap_r if d.vol.extreme_vol else cfg.heat_cap_r
            cd_bars = sym_cfg.cooldown_bars
            if d.regime == BRSRegime.BEAR_STRONG:
                cd_bars = cfg.cooldown_bear_strong
            elif d.regime == BRSRegime.BEAR_TREND:
                cd_bars = cfg.cooldown_bear_trend

            self._instrumentation.log_entry(
                trade_id=pending.pos_id,
                pair=symbol,
                side=_side_str,
                entry_price=fill_price,
                position_size=float(fill_qty),
                position_size_quote=fill_price * fill_qty,
                entry_signal=signal.entry_type.value,
                entry_signal_id=pending.pos_id,
                entry_signal_strength=signal.quality_score,
                active_filters=["regime_gate", "bias_confirmation", "heat_cap", "max_concurrent"],
                passed_filters=["regime_gate", "bias_confirmation", "heat_cap", "max_concurrent"],
                strategy_params={
                    "regime": d.regime.value,
                    "regime_4h": d.regime_4h.value,
                    "bear_conviction": d.bear_conviction,
                    "vol_factor": d.vol.vol_factor,
                    "stop_price": signal.stop_price,
                    "risk_per_unit": risk_per_unit,
                    "risk_dollars": risk_dollars,
                    "cooldown_bars": cd_bars,
                    "pyramid_enabled": cfg.pyramid_enabled,
                    "actual_fill_price": fill_price,
                    "signal_price": signal.signal_price,
                    "slippage": fill_price - signal.signal_price,
                },
                signal_factors=[
                    {"factor": "regime", "value": d.regime.value},
                    {"factor": "regime_4h", "value": d.regime_4h.value},
                    {"factor": "bear_conviction", "value": d.bear_conviction, "threshold": 50},
                    {"factor": "bias_path", "value": (
                        "crash_override" if _bias.crash_override else
                        "peak_drop" if _bias.peak_drop_override else
                        "cum_return" if _bias.cum_return_override else
                        "normal"
                    )},
                    {"factor": "vol_factor", "value": signal.vol_factor},
                    {"factor": "quality_score", "value": signal.quality_score},
                ],
                filter_decisions=[
                    {"filter_name": "regime_gate", "threshold": d.regime.value,
                     "actual_value": d.regime.value, "passed": True},
                    {"filter_name": "heat_cap", "threshold": _heat_cap,
                     "actual_value": current_heat_r, "passed": True},
                    {"filter_name": "max_concurrent", "threshold": float(cfg.max_concurrent),
                     "actual_value": float(concurrent), "passed": True},
                ],
                sizing_inputs={
                    "base_risk_pct": sym_cfg.base_risk_pct,
                    "account_equity": self._equity,
                    "quality_mult": signal.quality_score,
                    "vol_factor": d.vol.vol_factor,
                },
                portfolio_state_at_entry={
                    "concurrent_positions": concurrent,
                    "current_heat_r": round(current_heat_r, 4),
                },
                concurrent_positions_strategy=concurrent,
            )

            self._instrumentation.on_order_event(
                order_id=pending.oms_order_id,
                pair=symbol,
                side="SELL" if signal.direction == Direction.SHORT else "BUY",
                order_type="LIMIT",
                status="FILLED",
                requested_qty=float(pending.qty),
                filled_qty=float(fill_qty),
                requested_price=signal.signal_price,
                fill_price=fill_price,
                related_trade_id=pending.pos_id,
                strategy_id=STRATEGY_ID,
            )

    def _on_pyramid_fill_instrumentation(
        self, event: Any, pending: PendingOrder | None, fill_price: float,
        fill_qty: int, fill_commission: float,
    ) -> None:
        """Log pyramid fill via instrumentation (state already applied by core)."""
        symbol = event.symbol
        pos = self._position.get(symbol)
        logger.info(
            "[%s] PYRAMID FILL @ %.2f qty=%d (blended=%.2f, total=%d)",
            symbol, fill_price, fill_qty,
            pos.entry_price if pos else 0.0, pos.qty if pos else 0,
        )
        if self._instrumentation and pending is not None and pos is not None:
            self._instrumentation.on_order_event(
                order_id=pending.oms_order_id,
                pair=symbol,
                side="SELL" if pos.direction == Direction.SHORT else "BUY",
                order_type="LIMIT",
                status="FILLED",
                requested_qty=float(pending.qty),
                filled_qty=float(fill_qty),
                requested_price=pending.signal.signal_price if pending.signal else fill_price,
                fill_price=fill_price,
                related_trade_id=pos.pos_id,
                strategy_id=STRATEGY_ID,
            )

    def _on_scale_out_fill_instrumentation(
        self, event: Any, pending: PendingOrder | None, fill_price: float,
        fill_qty: int, fill_commission: float,
    ) -> None:
        """Log scale-out fill via instrumentation (state + equity already applied)."""
        symbol = event.symbol
        exit_qty = event.details.get("qty", fill_qty)
        pos = self._position.get(symbol)
        logger.info(
            "[%s] SCALE-OUT FILL %d @ %.2f (remaining=%d)",
            symbol, exit_qty, fill_price, pos.qty if pos else 0,
        )
        if self._instrumentation and pending is not None:
            pos_id = pending.pos_id if pending.pos_id else (pos.pos_id if pos else "")
            self._instrumentation.on_order_event(
                order_id=pending.oms_order_id,
                pair=symbol,
                side="BUY" if (pos and pos.direction == Direction.SHORT) else "SELL",
                order_type="LIMIT",
                status="FILLED",
                requested_qty=float(pending.qty),
                filled_qty=float(exit_qty),
                fill_price=fill_price,
                related_trade_id=pos_id,
                strategy_id=STRATEGY_ID,
            )

    async def _on_exit_fill_instrumentation(
        self, symbol: str, pos: BRSPositionState, fill_price: float,
        fill_qty: int, fill_commission: float, exit_type: str,
        net_pnl: float, r_mult: float,
    ) -> None:
        """Record trade exit and log via instrumentation (state + equity already applied by core)."""
        is_short = pos.direction == Direction.SHORT

        logger.info(
            "[%s] EXIT FILL %s @ %.2f qty=%d pnl=$%.2f R=%.2f",
            symbol, exit_type, fill_price, fill_qty, net_pnl, r_mult,
        )

        # Trade recorder
        if self._trade_recorder:
            try:
                await self._trade_recorder.record({
                    "strategy_id": STRATEGY_ID,
                    "symbol": symbol,
                    "direction": "SHORT" if is_short else "LONG",
                    "entry_price": pos.entry_price,
                    "exit_price": fill_price,
                    "shares": pos.original_qty,
                    "realized_pnl": net_pnl,
                    "r_multiple": r_mult,
                    "entry_type": pos.entry_type,
                    "exit_reason": exit_type,
                    "entry_ts": pos.entry_ts,
                    "exit_ts": datetime.now(timezone.utc),
                    "trade_id": pos.pos_id,
                    "hold_bars": pos.bars_held,
                    "regime": pos.regime_at_entry.value,
                    "commission": pos.commission + fill_commission,
                    "pyramid_count": pos.pyramid_count,
                    "mfe_r": pos.mfe_r,
                    "mae_r": pos.mae_r,
                })
            except Exception:
                logger.warning("Trade recorder failed for %s", pos.pos_id, exc_info=True)

        # Instrumentation
        if self._instrumentation:
            if is_short:
                _mfe_pct = (pos.entry_price - pos.mfe_price) / pos.entry_price if pos.entry_price > 0 else None
                _mae_pct = (pos.mae_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else None
                _pnl_pct = (pos.entry_price - fill_price) / pos.entry_price if pos.entry_price > 0 else None
            else:
                _mfe_pct = (pos.mfe_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else None
                _mae_pct = (pos.entry_price - pos.mae_price) / pos.entry_price if pos.entry_price > 0 else None
                _pnl_pct = (fill_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else None

            self._instrumentation.log_exit(
                trade_id=pos.pos_id,
                exit_price=fill_price,
                exit_reason=exit_type,
                mfe_price=pos.mfe_price,
                mae_price=pos.mae_price,
                mfe_r=pos.mfe_r,
                mae_r=pos.mae_r,
                mfe_pct=_mfe_pct,
                mae_pct=_mae_pct,
                pnl_pct=_pnl_pct,
            )

            self._instrumentation.on_order_event(
                order_id=pos.entry_oms_id or pos.pos_id,
                pair=symbol,
                side="BUY" if is_short else "SELL",
                order_type="MARKET",
                status="FILLED",
                requested_qty=float(pos.qty),
                filled_qty=float(fill_qty),
                fill_price=fill_price,
                related_trade_id=pos.pos_id,
                strategy_id=STRATEGY_ID,
            )

    # ══════════════════════════════════════════════════════════════════
    # Equity refresh
    # ══════════════════════════════════════════════════════════════════

    async def _refresh_equity(self) -> None:
        """Sync equity from IB account (matches other swing engine pattern)."""
        try:
            accounts = self.ib.ib.managedAccounts()
            if accounts:
                for item in self.ib.ib.accountValues():
                    if item.tag == "NetLiquidation" and item.currency == "USD" and item.account == accounts[0]:
                        self._equity = float(item.value) * self._equity_alloc_pct
                        logger.info("BRS equity refreshed: $%.2f", self._equity)
                        return
        except Exception:
            logger.debug("BRS equity refresh failed, using $%.2f", self._equity)

    # ══════════════════════════════════════════════════════════════════
    # Daily context update
    # ══════════════════════════════════════════════════════════════════

    def _update_daily_context(self, symbol: str) -> None:
        ds = self._daily_series[symbol]
        ds.ensure_computed()
        cfg = self._cfg
        sym_cfg = self._sym_cfgs[symbol]

        if len(ds) < 2:
            return

        close = ds.closes[-1]
        ema_f = ds.latest_ema_fast
        ema_s = ds.latest_ema_slow
        adx_val = ds.latest_adx
        plus_di = ds.latest_plus_di
        minus_di = ds.latest_minus_di
        atr14 = ds.latest_atr14
        atr50 = ds.latest_atr50

        # Peak drop check -- must precede regime_on (forces it True)
        peak_drop_triggered = False
        if cfg.peak_drop_enabled and len(ds.highs) > cfg.peak_drop_lookback:
            peak = float(np.max(ds.highs[-cfg.peak_drop_lookback:]))
            if peak > 0:
                drop_pct = (close - peak) / peak
                if drop_pct <= cfg.peak_drop_pct:
                    peak_drop_triggered = True

        # Regime on/off
        regime_on = compute_regime_on(adx_val, self._prev_regime_on[symbol], sym_cfg.adx_on, sym_cfg.adx_off)

        if peak_drop_triggered:
            regime_on = True

        # GLD cross-symbol override: force regime_on when QQQ is in bear regime (R8)
        if (cfg.gld_qqq_override_enabled
                and symbol == "GLD"
                and "QQQ" in self._daily_ctx):
            qqq_regime = self._daily_ctx["QQQ"].regime
            if (qqq_regime == BRSRegime.BEAR_STRONG
                    or (cfg.gld_qqq_override_bear_trend
                        and qqq_regime == BRSRegime.BEAR_TREND)):
                regime_on = True

        self._prev_regime_on[symbol] = regime_on

        # Classify regime
        regime = classify_regime(
            adx_val, plus_di, minus_di, close, ema_f, ema_s,
            regime_on, cfg.adx_strong, cfg.regime_bear_min_conditions,
        )

        # EMA separation
        ema_sep_pct = abs(ema_f - ema_s) / max(close, 1e-9) * 100.0 if close > 0 else 0.0

        # Daily return
        prev_close = self._last_daily_close.get(symbol, close)
        daily_return = (close - prev_close) / max(prev_close, 1e-9) if prev_close > 0 else 0.0
        self._last_daily_close[symbol] = close

        # ATR ratio
        atr_ratio = atr14 / max(atr50, 1e-9) if atr50 > 0 else 1.0

        # Raw bias
        raw_dir = compute_raw_bias(close, ema_f, ema_s)

        # Bear conviction
        bear_conv = compute_bear_conviction(
            adx_val, minus_di, plus_di, ema_sep_pct,
            close < ema_f and close < ema_s,
            ds.latest_ema_fast_slope < 0,
        )

        # 4H regime for bias acceleration
        h4 = self._h4_series[symbol]
        regime_4h_bear = False
        regime_4h = Regime4H.CHOP
        if len(h4) > 3:
            h4.ensure_computed()
            r4h_str, _ = compute_4h_structure(
                h4.closes, h4.ema50_arr, h4.atr14_arr, h4.adx_arr, close,
            )
            regime_4h = Regime4H(r4h_str)
            regime_4h_bear = regime_4h == Regime4H.BEAR

        # Cumulative return for Path G bias confirmation
        cum_return = 0.0
        if cfg.cum_return_enabled and len(ds.closes) > cfg.cum_return_lookback:
            prev_c = float(ds.closes[-cfg.cum_return_lookback - 1])
            cum_return = (close - prev_c) / max(prev_c, 1e-9) if prev_c > 0 else 0.0

        # Bias update
        self._bias[symbol] = update_bias(
            prev_bias=self._bias[symbol],
            regime=regime,
            regime_on=regime_on,
            raw_direction=raw_dir,
            bear_score=bear_conv,
            adx=adx_val,
            di_diff=minus_di - plus_di,
            ema_sep_pct=ema_sep_pct,
            daily_return=daily_return,
            atr_ratio=atr_ratio,
            fast_crash_enabled=cfg.fast_crash_enabled,
            fast_crash_return_thresh=cfg.fast_crash_return_thresh,
            fast_crash_atr_ratio=cfg.fast_crash_atr_ratio,
            crash_override_enabled=cfg.crash_override_enabled,
            crash_override_return=cfg.crash_override_return,
            crash_override_atr_ratio=cfg.crash_override_atr_ratio,
            return_only_enabled=cfg.return_only_enabled,
            return_only_thresh=cfg.return_only_thresh,
            close=close,
            ema_fast=ema_f,
            bias_4h_accel_enabled=cfg.bias_4h_accel_enabled,
            bias_4h_accel_reduction=cfg.bias_4h_accel_reduction,
            regime_4h_bear=regime_4h_bear,
            cum_return_enabled=cfg.cum_return_enabled,
            cum_return=cum_return,
            cum_return_thresh=cfg.cum_return_thresh,
            churn_bridge_bars=cfg.churn_bridge_bars,
        )

        # Peak drop: force instant SHORT confirmation (matches backtest)
        if peak_drop_triggered and self._bias[symbol].confirmed_direction != Direction.SHORT:
            self._bias[symbol].confirmed_direction = Direction.SHORT
            self._bias[symbol].peak_drop_override = True

        # Vol state
        vf, vol_pct = compute_vol_factor(
            atr14, ds.atr14_history, extreme_vol_pct=cfg.extreme_vol_pct,
        )  # vf_clamp_min/max use function defaults (0.35, 1.5) matching backtest
        rr = compute_risk_regime(atr14, ds.atr14_history)
        vol = VolState(
            vol_factor=vf,
            vol_pct=vol_pct,
            risk_regime=(1.0 / rr if rr > 0 else 1.0),
            base_risk_adj=rr,
            extreme_vol=vol_pct > cfg.extreme_vol_pct,
        )

        # Persistence override counter
        if self._bias[symbol].confirmed_direction == Direction.SHORT and self._position[symbol] is None:
            self._short_bias_no_trade[symbol] += 1
        else:
            self._short_bias_no_trade[symbol] = 0

        persistence_active = (
            cfg.persistence_override_bars > 0
            and self._short_bias_no_trade[symbol] >= cfg.persistence_override_bars
        )

        self._prev_regime[symbol] = self._daily_ctx[symbol].regime

        self._daily_ctx[symbol] = DailyContext(
            regime=regime,
            regime_on=regime_on,
            regime_4h=regime_4h,
            bias=self._bias[symbol],
            vol=vol,
            bear_conviction=bear_conv,
            adx=adx_val,
            plus_di=plus_di,
            minus_di=minus_di,
            ema_fast=ema_f,
            ema_slow=ema_s,
            ema_fast_slope=ds.latest_ema_fast_slope,
            ema_slow_slope=ds.latest_ema_slow_slope,
            ema_sep_pct=ema_sep_pct,
            atr14_d=atr14,
            atr50_d=atr50,
            atr_ratio=atr_ratio,
            close=close,
            persistence_active=persistence_active,
        )

        # S2 arming (daily-level campaign box detection)
        if not cfg.disable_s2 and len(ds) > 15:
            d_idx = len(ds) - 1
            new_s2 = check_s2_arm(
                daily_closes=ds.closes,
                daily_lows=ds.lows,
                daily_highs=ds.highs,
                daily_volumes=ds.volumes,
                atr14_d=atr14,
                atr50_d=atr50,
                cfg=cfg,
                current_bar_idx=d_idx,
            )
            if new_s2 is not None:
                self._s2_arm[symbol] = new_s2
                if self._instrumentation:
                    self._instrumentation.on_filter_decision(
                        pair=symbol,
                        filter_name="s2_arm",
                        passed=True,
                        threshold=new_s2.box_low,
                        actual_value=new_s2.box_high,
                        signal_name="s2_breakdown_armed",
                        strategy_id="BRS_R9",
                    )

        logger.info(
            "[%s] Daily: regime=%s regime_on=%s bias=%s adx=%.1f conv=%.0f vf=%.2f",
            symbol, regime.value, regime_on,
            self._bias[symbol].confirmed_direction.name,
            adx_val, bear_conv, vf,
        )

        if self._instrumentation:
            _prev_r = self._prev_regime.get(symbol, BRSRegime.RANGE_CHOP)
            _transition = regime != _prev_r
            self._instrumentation.on_indicator_snapshot(
                pair=symbol,
                indicators={
                    "regime": regime.value,
                    "prev_regime": _prev_r.value,
                    "regime_transition": float(_transition),
                    "regime_on": float(regime_on),
                    "regime_4h": regime_4h.value,
                    "bias_confirmed": self._bias[symbol].confirmed_direction.name,
                    "bias_hold_count": float(self._bias[symbol].hold_count),
                    "bias_crash_override": float(self._bias[symbol].crash_override),
                    "bias_peak_drop_override": float(self._bias[symbol].peak_drop_override),
                    "bear_conviction": bear_conv,
                    "adx": adx_val,
                    "plus_di": plus_di,
                    "minus_di": minus_di,
                    "ema_fast": ema_f,
                    "ema_slow": ema_s,
                    "ema_sep_pct": ema_sep_pct,
                    "ema_fast_slope": ds.latest_ema_fast_slope,
                    "atr14_d": atr14,
                    "atr50_d": atr50,
                    "atr_ratio": atr_ratio,
                    "vol_factor": vf,
                    "vol_pct": vol_pct,
                    "daily_return": daily_return,
                    "persistence_active": float(persistence_active),
                },
                signal_name="daily_regime_update",
                signal_strength=bear_conv / 100.0,
                decision="regime_transition" if _transition else "regime_hold",
                strategy_id="BRS_R9",
            )
            self._instrumentation.record_close(symbol, close)

    # ══════════════════════════════════════════════════════════════════
    # 4H context update
    # ══════════════════════════════════════════════════════════════════

    def _update_4h_context(self, symbol: str) -> None:
        h4 = self._h4_series[symbol]
        h4.ensure_computed()
        if len(h4) < 4:
            return
        ds = self._daily_series[symbol]
        close = ds.closes[-1] if len(ds) > 0 else 0.0

        r4h_str, _ = compute_4h_structure(
            h4.closes, h4.ema50_arr, h4.atr14_arr, h4.adx_arr, close,
        )
        self._daily_ctx[symbol].regime_4h = Regime4H(r4h_str)

    # ══════════════════════════════════════════════════════════════════
    # Hourly bar processing (main signal + position logic)
    # ══════════════════════════════════════════════════════════════════

    async def _on_hourly_bar(self, symbol: str) -> None:
        hs = self._h1_series[symbol]
        if len(hs) < 30:
            self._record_decision("AWAITING_DATA", {"symbol": symbol, "bars": len(hs), "need": 30})
            return

        hs.ensure_computed()
        bar = hs.last_bar
        if bar is None:
            return

        # Skip if position is closing (exit fill pending)
        if symbol in self._closing:
            return

        # RTH filter
        bar_et = bar.ts.astimezone(ET)
        if not (RTH_START <= bar_et.time() < RTH_END):
            self._record_decision("OUTSIDE_RTH", {"symbol": symbol})
            return

        self._hourly_bar_count[symbol] += 1
        bar_idx = self._hourly_bar_count[symbol]

        d_ctx = self._daily_ctx[symbol]
        h_ctx = self._build_hourly_context(symbol)
        cfg = self._cfg
        sym_cfg = self._sym_cfgs[symbol]

        # ── Arming updates ────────────────────────────────────────────
        self._update_arming(symbol, h_ctx, d_ctx, bar_idx)

        # ── Position management ───────────────────────────────────────
        pos = self._position[symbol]
        if pos is not None:
            self._record_decision("MANAGING_POSITION", {"symbol": symbol, "entry_type": pos.entry_type})
            _old_stop = pos.stop_price
            actions = pos.manage(h_ctx, d_ctx, cfg, sym_cfg, self._prev_regime[symbol])
            await self._execute_actions(symbol, actions, _old_stop)

            # Pyramid check (only if still in position after management)
            if self._position[symbol] is not None and cfg.pyramid_enabled:
                await self._check_pyramid(symbol, h_ctx, d_ctx)
        else:
            await self._check_entries(symbol, h_ctx, d_ctx, bar_idx)

    def _build_hourly_context(self, symbol: str) -> HourlyContext:
        hs = self._h1_series[symbol]
        cfg = self._cfg
        d_ctx = self._daily_ctx[symbol]
        bar = hs.last_bar
        prior = hs.prior_bar

        # EMA_pull is regime-adaptive (same as backtest)
        regime = d_ctx.regime
        if regime == BRSRegime.BEAR_STRONG:
            ema_pull = hs.latest_ema34  # tighter pullback in strong trend
        else:
            ema_pull = hs.latest_ema50_h  # wider pullback otherwise

        return HourlyContext(
            close=bar.close if bar else 0.0,
            high=bar.high if bar else 0.0,
            low=bar.low if bar else 0.0,
            open=bar.open if bar else 0.0,
            prior_high=prior.high if prior else 0.0,
            prior_low=prior.low if prior else 0.0,
            ema_pull=ema_pull,
            ema_mom=hs.latest_ema20,
            atr14_h=hs.latest_atr14,
            avwap_h=0.0,  # AVWAP not maintained in live (backtest-only)
            volume=bar.volume if bar else 0.0,
            volume_sma20=hs.latest_volume_sma20,
            prior_close_3=float(hs.closes[-4]) if len(hs) >= 4 else 0.0,
        )

    # ── arming ────────────────────────────────────────────────────────

    def _update_arming(
        self, symbol: str, h: HourlyContext, d: DailyContext, bar_idx: int,
    ) -> None:
        cfg = self._cfg

        # LH arm expiry
        lh = self._lh_arm[symbol]
        if lh.armed and bar_idx > lh.armed_until_bar:
            if self._instrumentation:
                self._instrumentation.on_filter_decision(
                    pair=symbol, filter_name="lh_arm_expiry", passed=False,
                    threshold=float(lh.armed_until_bar), actual_value=float(bar_idx),
                    signal_name="lh_rejection_expired", strategy_id="BRS_R9",
                )
            self._lh_arm[symbol] = LHArmState()

        # BD arm expiry
        bd = self._bd_arm[symbol]
        if bd.armed and bar_idx > bd.armed_until_bar:
            if self._instrumentation:
                self._instrumentation.on_filter_decision(
                    pair=symbol, filter_name="bd_arm_expiry", passed=False,
                    threshold=float(bd.armed_until_bar), actual_value=float(bar_idx),
                    signal_name="bd_continuation_expired", strategy_id="BRS_R9",
                )
            self._bd_arm[symbol] = BDArmState()

        # S2 arm expiry
        s2 = self._s2_arm[symbol]
        if s2.armed and bar_idx > s2.armed_until_bar:
            if self._instrumentation:
                self._instrumentation.on_filter_decision(
                    pair=symbol, filter_name="s2_arm_expiry", passed=False,
                    threshold=float(s2.armed_until_bar), actual_value=float(bar_idx),
                    signal_name="s2_breakdown_expired", strategy_id="BRS_R9",
                )
            self._s2_arm[symbol] = S2ArmState()

        # S3 arm expiry
        s3 = self._s3_arm[symbol]
        if s3.armed and bar_idx > s3.armed_until_bar:
            if self._instrumentation:
                self._instrumentation.on_filter_decision(
                    pair=symbol, filter_name="s3_arm_expiry", passed=False,
                    threshold=float(s3.armed_until_bar), actual_value=float(bar_idx),
                    signal_name="s3_impulse_expired", strategy_id="BRS_R9",
                )
            self._s3_arm[symbol] = S3ArmState()

        hs = self._h1_series[symbol]

        # Swing high tracking for LH (cap at 50 to prevent unbounded growth)
        sh = hs.latest_swing_high
        if not np.isnan(sh) and sh > 0:
            swh = self._swing_highs[symbol]
            swh.append(sh)
            if len(swh) > 50:
                self._swing_highs[symbol] = swh[-50:]
            highs = self._swing_highs[symbol]
            if len(highs) >= 2 and highs[-1] < highs[-2]:
                self._lh_arm[symbol] = LHArmState(
                    armed=True,
                    swing_high_price=highs[-1],
                    prior_swing_high_price=highs[-2],
                    armed_bar=bar_idx,
                    armed_until_bar=bar_idx + cfg.lh_arm_bars,
                )
                if self._instrumentation:
                    self._instrumentation.on_filter_decision(
                        pair=symbol, filter_name="lh_arm", passed=True,
                        threshold=highs[-2], actual_value=highs[-1],
                        signal_name="lh_rejection_armed", strategy_id="BRS_R9",
                    )

        # BD arming
        donch_low_bd = hs.prior_donchian_low_bd
        if donch_low_bd > 0 and not self._bd_arm[symbol].armed:
            _chop_override = (
                cfg.chop_short_entry_enabled
                and d.regime == BRSRegime.RANGE_CHOP
                and d.bias.confirmed_direction == Direction.SHORT
                and d.bias.peak_drop_override
            )
            new_bd = check_bd_arm(
                hourly_low=h.low,
                hourly_high=h.high,
                hourly_close=h.close,
                hourly_volume=h.volume,
                donchian_low_val=donch_low_bd,
                volume_sma20=h.volume_sma20,
                regime=d.regime,
                cfg=cfg,
                bar_idx=bar_idx,
                chop_arm_allowed=_chop_override,
                persistence_override=d.persistence_active,
            )
            if new_bd is not None:
                self._bd_arm[symbol] = new_bd
                if self._instrumentation:
                    self._instrumentation.on_filter_decision(
                        pair=symbol, filter_name="bd_arm", passed=True,
                        threshold=donch_low_bd, actual_value=h.low,
                        signal_name="bd_continuation_armed", strategy_id="BRS_R9",
                    )

        # S3 arming (hourly Donchian breakout -- prior bar's Donchian)
        if not cfg.disable_s3 and not self._s3_arm[symbol].armed:
            donch_low_s3 = hs.prior_donchian_low_26
            if donch_low_s3 > 0:
                new_s3 = check_s3_arm(
                    regime=d.regime,
                    confirmed_bear=d.bias.confirmed_direction == Direction.SHORT,
                    hourly_low=h.low,
                    hourly_close=h.close,
                    donchian_low_val=donch_low_s3,
                    ema_mom=h.ema_mom,
                    current_bar_idx=bar_idx,
                )
                if new_s3 is not None:
                    self._s3_arm[symbol] = new_s3
                    if self._instrumentation:
                        self._instrumentation.on_filter_decision(
                            pair=symbol, filter_name="s3_arm", passed=True,
                            threshold=donch_low_s3, actual_value=h.low,
                            signal_name="s3_impulse_armed", strategy_id="BRS_R9",
                        )

    # ── entries ───────────────────────────────────────────────────────

    async def _check_entries(
        self, symbol: str, h: HourlyContext, d: DailyContext, bar_idx: int,
    ) -> None:
        cfg = self._cfg
        sym_cfg = self._sym_cfgs[symbol]
        now = datetime.now(ET)

        # Skip if entry already pending for this symbol
        if any(p.symbol == symbol and p.role == "entry"
               for p in self._pending_orders.values()):
            return

        # Cooldown check
        if now < self._cooldown_until[symbol]:
            if self._instrumentation:
                try:
                    self._instrumentation.log_missed(
                        pair=symbol, side="UNKNOWN", signal="cooldown_blocked",
                        signal_id=f"BRS_{symbol}_cooldown_{bar_idx}",
                        signal_strength=0.0, blocked_by="cooldown",
                        block_reason=f"cooldown until {self._cooldown_until[symbol]}",
                        strategy_params={"regime": d.regime.value},
                        market_regime=d.regime.value,
                        filter_decisions=[
                            {"filter_name": "cooldown", "threshold": self._cooldown_until[symbol].isoformat(),
                             "actual_value": now.isoformat(), "passed": False},
                        ],
                    )
                except Exception:
                    pass
            self._record_decision("SIGNAL_FILTERED", {"symbol": symbol, "filter": "cooldown"})
            return

        # Check concurrent positions (include pending entries to prevent race)
        pending_entries = sum(1 for p in self._pending_orders.values() if p.role == "entry")
        concurrent = sum(1 for p in self._position.values() if p is not None) + pending_entries

        # Signal priority: LH > BD > S2 > S1 > S3 > L1 (matches backtest)
        signal: Optional[EntrySignal] = None

        sig = check_lh_rejection(d, h, sym_cfg, cfg, self._lh_arm[symbol])
        if sig is not None:
            signal = sig

        if signal is None:
            sig = check_bd_continuation(d, h, sym_cfg, cfg, self._bd_arm[symbol])
            if sig is not None:
                signal = sig

        if signal is None:
            sig = check_s2(d, h, sym_cfg, cfg, self._s2_arm[symbol])
            if sig is not None:
                signal = sig

        if signal is None:
            sig = check_s1(d, h, sym_cfg, cfg)
            if sig is not None:
                signal = sig

        if signal is None:
            sig = check_s3(d, h, sym_cfg, cfg, self._s3_arm[symbol])
            if sig is not None:
                signal = sig

        if signal is None:
            sig = check_l1(
                d, h, sym_cfg, cfg, symbol,
                self._daily_series[symbol].latest_ema_fast_slope,
                self._daily_series[symbol].latest_ema_slow_slope,
            )
            if sig is not None:
                signal = sig

        if signal is None:
            self._record_decision("NO_SIGNAL", {"symbol": symbol, "bar_idx": bar_idx})
            return

        # Sizing
        current_heat_r = self._compute_heat()
        qty = compute_position_size(
            signal=signal,
            equity=self._equity,
            sym_cfg=sym_cfg,
            daily_ctx=d,
            cfg=cfg,
            current_heat_r=current_heat_r,
            concurrent_positions=concurrent,
        )
        if qty <= 0:
            if self._instrumentation:
                _hcap = cfg.crisis_heat_cap_r if d.vol.extreme_vol else cfg.heat_cap_r
                self._instrumentation.log_missed(
                    pair=symbol,
                    side="SHORT" if signal.direction == Direction.SHORT else "LONG",
                    signal=signal.entry_type.value,
                    signal_id=f"BRS_{symbol}_{signal.entry_type.value}_{bar_idx}",
                    signal_strength=signal.quality_score,
                    blocked_by="position_sizing",
                    block_reason=f"qty=0 heat={current_heat_r:.2f} concurrent={concurrent} max={cfg.max_concurrent}",
                    strategy_params={"regime": d.regime.value, "vol_factor": d.vol.vol_factor},
                    market_regime=d.regime.value,
                    filter_decisions=[
                        {"filter_name": "cooldown", "threshold": self._cooldown_until[symbol].isoformat(),
                         "actual_value": now.isoformat(), "passed": True},
                        {"filter_name": "regime_gate", "threshold": d.regime.value,
                         "actual_value": d.regime.value, "passed": True},
                        {"filter_name": "heat_cap", "threshold": _hcap,
                         "actual_value": round(current_heat_r, 4), "passed": current_heat_r < _hcap},
                        {"filter_name": "max_concurrent", "threshold": float(cfg.max_concurrent),
                         "actual_value": float(concurrent), "passed": concurrent < cfg.max_concurrent},
                        {"filter_name": "position_sizing", "threshold": 1,
                         "actual_value": 0, "passed": False},
                    ],
                )
            return

        # Wider stop for BEAR_STRONG (both directions, per backtest)
        if d.regime == BRSRegime.BEAR_STRONG and h.atr14_h > 0:
            floor = sym_cfg.stop_floor_atr * cfg.stop_floor_bear_strong_mult * h.atr14_h
            if signal.risk_per_unit < floor:
                if signal.direction == Direction.SHORT:
                    new_stop = signal.signal_price + floor
                else:
                    new_stop = signal.signal_price - floor
                signal = EntrySignal(
                    entry_type=signal.entry_type,
                    direction=signal.direction,
                    signal_price=signal.signal_price,
                    signal_high=signal.signal_high,
                    signal_low=signal.signal_low,
                    stop_price=new_stop,
                    risk_per_unit=floor,
                    bear_conviction=signal.bear_conviction,
                    quality_score=signal.quality_score,
                    regime_at_entry=signal.regime_at_entry,
                    vol_factor=signal.vol_factor,
                )

        # Route entry through core
        pos_id = f"BRS_{symbol}_{uuid.uuid4().hex[:8]}"
        entry_request = BRSEntryRequest(
            client_order_id=str(uuid.uuid4()),
            symbol=symbol,
            signal=signal,
            qty=qty,
            pos_id=pos_id,
        )
        state = build_core_runtime_state(self)
        new_state, core_actions, events = brs_core_logic.on_bar(
            state, bar_ts=self._last_bar_ts, entry_request=entry_request,
        )
        apply_core_runtime_state(self, new_state)

        # Dispatch SubmitEntry action via execution engine
        submit_action = next((a for a in core_actions if isinstance(a, SubmitEntry)), None)
        if submit_action is None:
            return

        risk_dollars = qty * signal.risk_per_unit
        oms_id = await self._exec.submit_entry(
            symbol=symbol,
            pos_id=pos_id,
            direction=signal.direction,
            qty=qty,
            limit_price=signal.signal_price,
            stop_for_risk=signal.stop_price,
            risk_dollars=risk_dollars,
        )

        if oms_id is None:
            self._record_decision("ENTRY_DENIED", {"symbol": symbol, "signal": signal.entry_type.value, "denial_reason": "oms_rejection"})
            if self._instrumentation:
                _hcap = cfg.crisis_heat_cap_r if d.vol.extreme_vol else cfg.heat_cap_r
                self._instrumentation.log_missed(
                    pair=symbol,
                    side="SHORT" if signal.direction == Direction.SHORT else "LONG",
                    signal=signal.entry_type.value,
                    signal_id=pos_id,
                    signal_strength=signal.quality_score,
                    blocked_by="oms_rejection",
                    block_reason="OMS denied entry intent",
                    market_regime=d.regime.value,
                    filter_decisions=[
                        {"filter_name": "cooldown", "threshold": self._cooldown_until[symbol].isoformat(),
                         "actual_value": now.isoformat(), "passed": True},
                        {"filter_name": "regime_gate", "threshold": d.regime.value,
                         "actual_value": d.regime.value, "passed": True},
                        {"filter_name": "heat_cap", "threshold": _hcap,
                         "actual_value": round(current_heat_r, 4), "passed": True},
                        {"filter_name": "max_concurrent", "threshold": float(cfg.max_concurrent),
                         "actual_value": float(concurrent), "passed": True},
                        {"filter_name": "position_sizing", "threshold": 1,
                         "actual_value": float(qty), "passed": True},
                        {"filter_name": "oms_gate", "threshold": 1.0,
                         "actual_value": 0.0, "passed": False},
                    ],
                )
            return

        self._record_decision("ENTRY_SUBMITTED", {"symbol": symbol, "signal": signal.entry_type.value, "oms_id": oms_id})
        # Store pending entry -- position created on actual fill event
        self._pending_orders[oms_id] = PendingOrder(
            oms_order_id=oms_id,
            symbol=symbol,
            role="entry",
            qty=qty,
            signal=signal,
            pos_id=pos_id,
        )
        logger.info(
            "[%s] ENTRY SUBMITTED %s %s %d @ %.2f stop=%.2f",
            symbol, signal.entry_type.value, signal.direction.name,
            qty, signal.signal_price, signal.stop_price,
        )

    # ── pyramid ───────────────────────────────────────────────────────

    async def _check_pyramid(
        self, symbol: str, h: HourlyContext, d: DailyContext,
    ) -> None:
        cfg = self._cfg
        pos = self._position[symbol]
        if pos is None or pos.pyramid_count >= cfg.max_pyramid_adds:
            return

        cur_r = pos.current_r(h.close)
        if cur_r < cfg.pyramid_min_r:
            return

        # Must have a confirming signal to pyramid (LH/BD only, per backtest)
        sym_cfg = self._sym_cfgs[symbol]
        sig = check_lh_rejection(d, h, sym_cfg, cfg, self._lh_arm[symbol])
        if sig is None:
            sig = check_bd_continuation(d, h, sym_cfg, cfg, self._bd_arm[symbol])
        if sig is None:
            return

        # Only pyramid in same direction
        if sig.direction != pos.direction:
            return

        add_qty = max(1, int(math.floor(pos.original_qty * cfg.pyramid_scale)))
        pyr_pos_id = f"{pos.pos_id}_pyr{pos.pyramid_count + 1}"

        # Route through core
        add_on_request = BRSAddOnRequest(
            client_order_id=str(uuid.uuid4()),
            symbol=symbol,
            signal=sig,
            qty=add_qty,
            pos_id=pyr_pos_id,
        )
        state = build_core_runtime_state(self)
        new_state, core_actions, events = brs_core_logic.on_bar(
            state, bar_ts=self._last_bar_ts, add_on_request=add_on_request,
        )
        apply_core_runtime_state(self, new_state)

        # Dispatch SubmitAddOnEntry action
        submit_action = next((a for a in core_actions if isinstance(a, SubmitAddOnEntry)), None)
        if submit_action is None:
            return

        risk_dollars = add_qty * sig.risk_per_unit
        oms_id = await self._exec.submit_entry(
            symbol=symbol,
            pos_id=pyr_pos_id,
            direction=sig.direction,
            qty=add_qty,
            limit_price=sig.signal_price,
            stop_for_risk=sig.stop_price,
            risk_dollars=risk_dollars,
        )
        if oms_id:
            self._pending_orders[oms_id] = PendingOrder(
                oms_order_id=oms_id,
                symbol=symbol,
                role="pyramid",
                qty=add_qty,
                signal=sig,
                pos_id=pos.pos_id,
            )
            logger.info("[%s] PYRAMID SUBMITTED #%d: %d @ %.2f",
                        symbol, pos.pyramid_count + 1, add_qty, sig.signal_price)

    # ── action execution ──────────────────────────────────────────────

    async def _execute_actions(
        self, symbol: str, actions: list[ActionResult], old_stop: float = 0.0,
    ) -> None:
        """Route position-management ActionResults through core, dispatch OMS actions."""
        pos = self._position[symbol]
        if pos is None:
            return

        # Build core requests from ActionResults
        flatten_req = None
        stop_req = None
        scale_req = None
        has_exit = False

        for act in actions:
            if act.action == PositionAction.EXIT:
                flatten_req = BRSFlattenRequest(
                    symbol=symbol,
                    reason=act.exit_reason.value if act.exit_reason else "unknown",
                )
                has_exit = True
                break  # EXIT overrides other actions
            elif act.action == PositionAction.STOP_UPDATE:
                stop_req = BRSStopUpdateRequest(
                    symbol=symbol,
                    stop_price=act.new_stop,
                    qty=pos.qty,
                    reason="trail",
                )
            elif act.action == PositionAction.SCALE_OUT:
                scale_req = BRSScaleOutRequest(
                    client_order_id=f"{pos.pos_id}-scale-{uuid.uuid4().hex[:6]}",
                    symbol=symbol,
                    qty=act.scale_out_qty,
                    limit_price=act.exit_price,
                    pos_id=pos.pos_id,
                )

        if flatten_req is None and stop_req is None and scale_req is None:
            return

        # Single core call with all requests
        state = build_core_runtime_state(self)
        new_state, core_actions, events = brs_core_logic.on_bar(
            state,
            bar_ts=self._last_bar_ts,
            flatten_request=flatten_req,
            stop_update=stop_req,
            scale_out_request=scale_req,
        )
        apply_core_runtime_state(self, new_state)

        # Dispatch core actions via execution engine
        for ca in core_actions:
            if isinstance(ca, FlattenPosition):
                await self._exec.flatten(symbol, reason=ca.reason)
                logger.info(
                    "[%s] EXIT SUBMITTED %s bars=%d mfe_r=%.2f",
                    symbol, ca.reason, pos.bars_held, pos.mfe_r,
                )
            elif isinstance(ca, ReplaceProtectiveStop):
                await self._exec.modify_stop(pos, ca.stop_price)
            elif isinstance(ca, SubmitPartialExit):
                oms_id = await self._exec.submit_scale_out(
                    symbol, pos, ca.qty, ca.limit_price,
                )
                if oms_id:
                    self._pending_orders[oms_id] = PendingOrder(
                        oms_order_id=oms_id,
                        symbol=symbol,
                        role="scale_out",
                        qty=ca.qty,
                        pos_id=pos.pos_id,
                    )

        # Instrumentation for stop updates
        if stop_req is not None and self._instrumentation and old_stop != stop_req.stop_price:
            try:
                self._instrumentation.log_stop_adjustment(
                    trade_id=pos.pos_id,
                    symbol=symbol,
                    old_stop=old_stop,
                    new_stop=stop_req.stop_price,
                    adjustment_type="trailing",
                    trigger=f"stop_ratchet_{pos.entry_type}",
                    metadata={
                        "bars_held": pos.bars_held,
                        "mfe_r": round(pos.mfe_r, 3),
                        "entry_type": pos.entry_type,
                    },
                )
            except Exception:
                pass

    # ── helpers ───────────────────────────────────────────────────────

    def _compute_heat(self) -> float:
        """Compute current portfolio heat in R-units.

        Each position contributes risk_per_unit * qty / risk_budget,
        where risk_budget = equity * sym_cfg.base_risk_pct.
        """
        heat = 0.0
        for sym, pos in self._position.items():
            if pos is not None and pos.risk_per_unit > 0:
                sym_cfg = self._sym_cfgs.get(sym)
                if sym_cfg is None:
                    heat += 1.0
                    continue
                risk_budget = self._equity * sym_cfg.base_risk_pct
                if risk_budget > 0:
                    heat += (pos.risk_per_unit * pos.qty) / risk_budget
                else:
                    heat += 1.0
        return heat
