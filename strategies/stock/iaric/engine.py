"""Pullback hybrid engine for IARIC V2.

Replaces the T1 FSM engine with a 7-trigger daily selection + 5-min intraday
confirmation engine. Preserves the exact interface contract expected by
StockFamilyCoordinator.

Entry routes (checked in order each 5m bar):
  1. OPENING_RECLAIM (bars 1-5): flush + reclaim detection
  2. OPEN_SCORED_ENTRY (bars 1+): score-ranked fallback, max 4 slots
  3. DELAYED_CONFIRM (bars 6+): confirmation acceptance
  4. VWAP_BOUNCE (bars 12+): VWAP touch + reclaim
  5. AFTERNOON_RETEST (bars 48+): session low retest
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, time, timezone
from decimal import Decimal
from typing import Any

from libs.oms.models.events import OMSEventType
from libs.oms.models.intent import Intent, IntentType
from libs.oms.models.order import OrderRole
from libs.oms.instrumentation.runtime_refs import fill_runtime_refs
from strategies.core.actions import CancelAction, FlattenPosition, ReplaceProtectiveStop, SubmitEntry, SubmitMarketExit, SubmitProtectiveStop

from .artifact_store import IntradayStateSnapshot, load_intraday_state, persist_intraday_state
from .bar_policy import (
    Completed5mContractError,
    Completed5mGapError,
    apply_completed_5m_bar,
    validate_completed_5m_bar,
    validate_next_completed_5m_bar,
)
from .config import ET, PROXY_SYMBOLS, STRATEGY_ID, StrategySettings, build_proxy_instruments
from .core import logic as iaric_core_logic
from .core.lanes import (
    IssuerEntryCandidate,
    anchor_exit_enabled,
    issuer_batch_arbitration,
    issuer_exposure_decision,
    lane_daily_cap,
    lane_id_for_route,
)
from .core.residual import causal_relative_dislocation_atr
from .core.logic import apply_core_state as apply_core_runtime_state
from .core.logic import build_core_state as build_core_runtime_state
from .core.state import (
    IARICFill,
    IARICFlattenRequest,
    IARICOrderUpdate,
    IARICPartialExitRequest,
    IARICStopUpdateRequest,
)
from .diagnostics import JsonlDiagnostics
from .entry_request import build_ready_entry_request
from .execution import build_entry_order, build_market_exit, build_stock_instrument, build_stop_order
from .exits import (
    _route_param,
    carry_quality_gate,
    check_v2_partial,
    compute_overnight_stop,
    compute_stale_tighten,
    partial_exit_quantity,
    run_exit_chain,
    should_carry_overnight,
    update_mfe_stages,
)
from .models import (
    Bar,
    MarketSnapshot,
    PBSymbolState,
    PendingOrderState,
    PortfolioState,
    PositionState,
    QuoteSnapshot,
    SymbolIntradayState,
    VWAPLedger,
    WatchlistArtifact,
)
from .risk import timing_gate_allows_entry

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _BatchEntryCandidate:
    symbol: str
    bar: Bar
    timestamp: datetime
    route: str
    score: float
    entry_rank_pct: float
    entry_rsi: float
    artifact_index: int


class IARICEngine:
    """Live IARIC pullback hybrid engine.

    Constructor signature and public interface preserved for coordinator compatibility.
    """

    def __init__(
        self,
        oms_service,
        artifact: WatchlistArtifact,
        account_id: str,
        nav: float,
        settings: StrategySettings | None = None,
        trade_recorder=None,
        diagnostics: JsonlDiagnostics | None = None,
        instrumentation=None,
        disable_background_tasks: bool = False,
    ) -> None:
        self._oms = oms_service
        self._artifact = artifact
        self._items = artifact.by_symbol
        self._artifact_index = {item.symbol: index for index, item in enumerate(artifact.items)}
        self._account_id = account_id
        self._settings = settings or StrategySettings()
        if str(self._settings.pb_open_scored_fill_timing).lower() != "next_5m_open":
            raise ValueError(
                "Live IARIC requires pb_open_scored_fill_timing='next_5m_open'; "
                "same-open research has no production-parity order path"
            )
        self._trade_recorder = trade_recorder
        self._diagnostics = diagnostics or JsonlDiagnostics(self._settings.diagnostics_dir, enabled=False)
        self._instrumentation = instrumentation
        self._disable_background_tasks = bool(disable_background_tasks)

        self._symbols: dict[str, PBSymbolState] = {}
        self._markets: dict[str, MarketSnapshot] = {}
        self._session_vwap: dict[str, VWAPLedger] = {}
        self._portfolio = PortfolioState(account_equity=nav, base_risk_fraction=self._settings.base_risk_fraction)
        self._symbol_to_sector = {item.symbol: item.sector for item in artifact.items}
        self._active_symbols: set[str] = set()
        self._order_index: dict[str, tuple[str, str]] = {}
        self._flow_reversal_flags = {held.symbol: held.flow_reversal_flag for held in artifact.held_positions}
        self._market_wide_institutional_selling = artifact.market_wide_institutional_selling
        self._expected_stop_cancels: set[str] = set()
        self._last_quote_volume: dict[str, float] = {}
        self._last_save_ts: datetime | None = None
        self._open_scored_count: int = 0
        self._aperture_family_counts: dict[str, int] = {}
        self._daily_entry_symbols: list[str] = []
        self._rescue_entry_count: int = 0
        self._lane_entry_counts: dict[str, int] = {}
        self._kit_cache = None

        self._event_queue = None
        self._event_task: asyncio.Task | None = None
        self._pulse_task: asyncio.Task | None = None
        self._running = False

        # Diagnostic pulse state
        self._last_decision_code: str = "IDLE"
        self._last_decision_details: dict = {}
        self._last_bar_ts: datetime | None = None
        self._bars_processed: int = 0
        self._symbol_last_bar_ts: dict[str, datetime] = {}
        self._pending_5m_batches: dict[datetime, dict[str, Bar]] = {}
        self._pending_5m_batch_tasks: dict[datetime, asyncio.Task] = {}
        self._collecting_entry_candidates: list[_BatchEntryCandidate] | None = None
        self._bar_arrival_latency_s: dict[str, float] = {}
        self._duplicate_5m_bars: int = 0
        self._gap_5m_bars: int = 0
        self._rejected_5m_bars: int = 0
        self._expected_5m_bars: int = 0
        self._missing_5m_bars: int = 0
        self._last_missing_5m_symbols: list[str] = []
        self._recovery_cutoff_by_symbol: dict[str, datetime] = {}

        self._initialize_from_artifact()

    def _record_decision(self, code: str, details: dict | None = None) -> None:
        """Record the latest decision for diagnostic pulse reporting."""
        self._last_decision_code = code
        self._last_decision_details = details or {}

    def liveness_payload(self) -> dict:
        return {
            "bars_processed": self._bars_processed,
            "symbol_freshness": {
                sym: ts.isoformat() for sym, ts in self._symbol_last_bar_ts.items()
            },
            "completed_5m": {
                "arrival_latency_s": dict(self._bar_arrival_latency_s),
                "duplicates": self._duplicate_5m_bars,
                "gaps": self._gap_5m_bars,
                "rejected": self._rejected_5m_bars,
                "expected": self._expected_5m_bars,
                "missing": self._missing_5m_bars,
                "missing_rate": (
                    self._missing_5m_bars / self._expected_5m_bars
                    if self._expected_5m_bars
                    else 0.0
                ),
                "last_missing_symbols": list(self._last_missing_5m_symbols),
                "pending_batches": len(self._pending_5m_batches),
            },
        }

    @property
    def _instr_kit(self):
        """Lazy InstrumentationKit for direct facade calls."""
        if self._kit_cache is None and self._instrumentation is not None:
            try:
                from strategies.stock.instrumentation.src.facade import InstrumentationKit
                self._kit_cache = InstrumentationKit(self._instrumentation, strategy_type="strategy_iaric")
            except Exception:
                pass
        return self._kit_cache

    # ── Initialization ──────────────────────────────────────────────

    def _initialize_from_artifact(self) -> None:
        ranked_symbols = [item.symbol for item in self._artifact.items]
        self._active_symbols = set(ranked_symbols[: self._settings.active_monitoring_target])

        for item in self._artifact.items:
            symbol = item.symbol
            self._symbols[symbol] = PBSymbolState(
                symbol=symbol,
                daily_signal_score=item.daily_signal_score,
                trigger_types=list(item.trigger_types),
                trigger_tier=item.trigger_tier,
                trend_tier=item.trend_tier,
                rescue_flow_candidate=item.rescue_flow_candidate,
                sizing_mult=item.sizing_mult,
                daily_atr=item.daily_atr_estimate,
                cdd_value=item.cdd_value,
                ema10_daily=item.ema10_daily,
                rsi14_daily=item.rsi14_daily,
                entry_rank=int(getattr(item, "entry_rank", 0)),
                entry_rank_pct=float(getattr(item, "entry_rank_pct", 100.0)),
                entry_rsi=float(getattr(item, "entry_rsi", 50.0)),
            )
            self._markets[symbol] = MarketSnapshot(symbol=symbol)
            self._session_vwap[symbol] = VWAPLedger()

        # Restore held positions
        for held in self._artifact.held_positions:
            sym = self._symbols.get(held.symbol)
            if sym is None:
                sym = PBSymbolState(symbol=held.symbol, daily_atr=0.01)
                self._symbols[held.symbol] = sym
                self._markets.setdefault(held.symbol, MarketSnapshot(symbol=held.symbol))
                self._session_vwap.setdefault(held.symbol, VWAPLedger())
            position = PositionState(
                entry_price=held.entry_price,
                qty_entry=held.size,
                qty_open=held.size,
                final_stop=held.stop,
                current_stop=held.stop,
                entry_time=held.entry_time,
                initial_risk_per_share=max(held.initial_r, 0.01),
                max_favorable_price=held.entry_price,
                max_adverse_price=held.entry_price,
                setup_tag=held.setup_tag or "PB_CARRY",
            )
            sym.position = position
            sym.in_position = True
            sym.stage = "IN_POSITION"
            sym.risk_per_share = max(held.initial_r, 0.01)
            sym.stop_level = held.stop
            self._portfolio.open_positions[held.symbol] = position
            self._active_symbols.add(held.symbol)

    # ── Lifecycle (coordinator interface) ────────────────────────────

    @staticmethod
    def _log_task_exception(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Unhandled exception in background task: %s", exc, exc_info=exc)

    async def _reconcile_after_reconnect(self) -> None:
        logger.warning("IB reconnected -- triggering OMS reconciliation")
        try:
            await self._oms.request_reconciliation()
            logger.info("Post-reconnect OMS reconciliation complete")
        except Exception as exc:
            logger.error("Post-reconnect reconciliation failed: %s", exc, exc_info=exc)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._event_queue = self._oms.stream_events(STRATEGY_ID)
        self._event_task = asyncio.create_task(self._event_loop())
        if not self._disable_background_tasks:
            self._pulse_task = asyncio.create_task(self._pulse_loop())

    async def stop(self) -> None:
        self._running = False
        for task in self._pending_5m_batch_tasks.values():
            task.cancel()
        self._pending_5m_batch_tasks.clear()
        self._pending_5m_batches.clear()
        await self._save_state("stop")
        for task in (self._pulse_task, self._event_task):
            if task is None:
                continue
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    def hydrate_state(self, snapshot: IntradayStateSnapshot) -> None:
        """Restore intraday state from persisted snapshot.

        Accepts legacy SymbolIntradayState objects for backward compatibility.
        """
        restored_core = True
        for stored in snapshot.symbols:
            if not isinstance(stored, (PBSymbolState, SymbolIntradayState)):
                restored_core = False
                break
        if restored_core:
            apply_core_runtime_state(self, snapshot)
            self._arm_completed_5m_recovery()
            return

        self._active_symbols = set(snapshot.meta.get("active_symbols", self._active_symbols))
        for stored in snapshot.symbols:
            symbol_name = stored.symbol
            current = self._symbols.get(symbol_name)

            if isinstance(stored, PBSymbolState):
                # Native PB state
                if current is None:
                    self._symbols[symbol_name] = stored
                    self._markets.setdefault(symbol_name, MarketSnapshot(symbol=symbol_name))
                    self._session_vwap.setdefault(symbol_name, VWAPLedger())
                else:
                    # Merge key fields
                    current.stage = stored.stage
                    current.route_family = stored.route_family
                    current.in_position = stored.in_position
                    current.position = stored.position
                    current.entry_order = stored.entry_order
                    current.exit_order = stored.exit_order
                    current.pending_hard_exit = stored.pending_hard_exit
                    current.mfe_stage = stored.mfe_stage
                    current.breakeven_activated = stored.breakeven_activated
                    current.trail_active = stored.trail_active
                    current.hold_bars = stored.hold_bars
                    current.v2_partial_taken = stored.v2_partial_taken
                    current.stop_level = stored.stop_level
                    current.risk_per_share = stored.risk_per_share
                    current.bars_seen_today = stored.bars_seen_today
                    current.active_order_id = stored.active_order_id
                    current.last_transition_reason = stored.last_transition_reason
                    current.consecutive_bars_below_vwap = stored.consecutive_bars_below_vwap
                if stored.position is not None:
                    self._portfolio.open_positions[symbol_name] = stored.position
                self._restore_order_state(symbol_name, stored)

            elif isinstance(stored, SymbolIntradayState):
                # Legacy T1 state -- convert positions only
                if stored.position is not None and current is not None:
                    current.position = stored.position
                    current.in_position = True
                    current.stage = "IN_POSITION"
                    current.stop_level = stored.position.current_stop
                    current.risk_per_share = stored.position.initial_risk_per_share
                    self._portfolio.open_positions[symbol_name] = stored.position
                if stored.entry_order is not None and current is not None:
                    current.entry_order = stored.entry_order
                    self._order_index[stored.entry_order.oms_order_id] = (symbol_name, "ENTRY")
                if stored.exit_order is not None and current is not None:
                    current.exit_order = stored.exit_order
                    self._order_index[stored.exit_order.oms_order_id] = (symbol_name, stored.exit_order.role)
        self._arm_completed_5m_recovery()

    def _arm_completed_5m_recovery(self) -> None:
        self._recovery_cutoff_by_symbol = {
            symbol: state.last_5m_bar_time
            for symbol, state in self._symbols.items()
            if state.last_5m_bar_time is not None
        }

    def snapshot_state(self) -> IntradayStateSnapshot:
        return build_core_runtime_state(self)

    def subscription_instruments(self) -> list:
        instruments = build_proxy_instruments()
        seen = {instrument.symbol for instrument in instruments}
        for symbol in sorted(self._active_symbols):
            state = self._symbols.get(symbol)
            if state is None:
                continue
            # Subscribe to all active symbols (pullback monitors everything)
            item = self._items.get(symbol)
            if item and symbol not in seen:
                instruments.append(build_stock_instrument(item))
                seen.add(symbol)
        # Also subscribe to symbols with open positions
        for symbol in self._portfolio.open_positions:
            item = self._items.get(symbol)
            if item and symbol not in seen:
                instruments.append(build_stock_instrument(item))
                seen.add(symbol)
        return instruments

    def polling_instruments(self) -> list[tuple[Any, int]]:
        requests: list[tuple[Any, int]] = []
        for symbol, item in self._items.items():
            state = self._symbols.get(symbol)
            if state is None:
                continue
            if not item.tradable_flag and not state.in_position:
                continue
            # Quotes remain streaming for HOT symbols; authoritative action
            # bars for every eligible symbol come from completed 5m history.
            interval = min(int(self._settings.warm_poll_interval_s), 30)
            requests.append((build_stock_instrument(item), interval))
        return requests

    def health_status(self) -> dict:
        return {
            "engine": "IARICEngine_PB_V2",
            "running": self._running,
            "symbols_tracked": len(self._symbols),
            "active_symbols": len(self._active_symbols),
            "open_positions": len(self._portfolio.open_positions),
            "pending_orders": len(self._order_index),
            "open_scored_count": self._open_scored_count,
            "aperture_family_counts": dict(self._aperture_family_counts),
            "rescue_entry_count": self._rescue_entry_count,
            "lane_entry_counts": dict(self._lane_entry_counts),
            "last_decision_code": self._last_decision_code,
            "last_decision_details": self._last_decision_details,
            "last_bar_ts": self._last_bar_ts.isoformat() if self._last_bar_ts else None,
            "completed_5m": self.liveness_payload()["completed_5m"],
        }

    # ── Market data callbacks ───────────────────────────────────────

    def on_quote(self, symbol: str, quote: QuoteSnapshot) -> None:
        normalized = symbol.upper()
        if normalized in PROXY_SYMBOLS:
            return
        market = self._markets.get(normalized)
        if market is None:
            return
        market.last_quote = quote
        market.bid = quote.bid
        market.ask = quote.ask
        market.spread_pct = quote.spread_pct
        market.last_price = quote.last if quote.last > 0 else market.last_price
        # Tick pressure accumulation
        previous_volume = self._last_quote_volume.get(normalized, quote.cumulative_volume)
        volume_delta = max(0.0, quote.cumulative_volume - previous_volume)
        self._last_quote_volume[normalized] = quote.cumulative_volume
        midpoint = ((quote.bid + quote.ask) / 2.0) if quote.bid > 0 and quote.ask > 0 else quote.last
        signed = quote.last * volume_delta
        if quote.last < midpoint:
            signed *= -1.0
        if volume_delta > 0:
            market.tick_pressure_window.append((quote.ts, signed))

    def on_completed_5m_bar(
        self,
        symbol: str,
        bar: Bar,
        *,
        received_at: datetime | None = None,
    ) -> bool:
        """Queue one authoritative completed 5m bar for timestamp-batched evaluation.

        This is the only action-generating market-bar ingress for IARIC. Quotes
        remain a separate, non-clock input used for spread and execution context.
        """

        normalized = symbol.upper()
        market = self._markets.get(normalized)
        item = self._items.get(normalized)
        if market is None or item is None:
            return False
        receipt = received_at or datetime.now(timezone.utc)
        try:
            validate_completed_5m_bar(bar, received_at=receipt, expected_symbol=normalized)
        except Completed5mContractError:
            self._rejected_5m_bars += 1
            raise
        if bar.start_time.astimezone(ET).date() != self._artifact.trade_date:
            # A one-day bootstrap request can legitimately include the prior
            # RTH session. It is completed 5m data, but not input for today's
            # artifact clock.
            return False

        pending = self._pending_5m_batches.setdefault(bar.end_time, {})
        pending_bar = pending.get(normalized)
        if pending_bar is not None:
            if pending_bar == bar:
                self._duplicate_5m_bars += 1
                return False
            self._rejected_5m_bars += 1
            raise Completed5mContractError(
                f"IARIC received conflicting 5m bars for {normalized} ending {bar.end_time.isoformat()}"
            )
        if market.last_5m_bar is not None and bar.start_time == market.last_5m_bar.start_time:
            self._duplicate_5m_bars += 1
            return False
        prior_pending = [
            queued[normalized]
            for queued in self._pending_5m_batches.values()
            if normalized in queued and queued[normalized].start_time < bar.start_time
        ]
        later_pending = [
            queued[normalized]
            for queued in self._pending_5m_batches.values()
            if normalized in queued and queued[normalized].start_time > bar.start_time
        ]
        if later_pending:
            self._rejected_5m_bars += 1
            raise Completed5mContractError(
                f"IARIC received an out-of-order 5m bar for {normalized} ending {bar.end_time.isoformat()}"
            )
        previous = max(
            ([market.last_5m_bar] if market.last_5m_bar is not None else []) + prior_pending,
            key=lambda prior: prior.start_time,
            default=None,
        )
        try:
            validate_next_completed_5m_bar(previous, bar)
        except Completed5mGapError:
            self._gap_5m_bars += 1
            raise
        except Completed5mContractError:
            self._rejected_5m_bars += 1
            raise

        pending[normalized] = bar
        self._symbol_last_bar_ts[normalized] = receipt
        self._bar_arrival_latency_s[normalized] = max(0.0, (receipt - bar.end_time).total_seconds())
        self._schedule_5m_batch_flush(bar.end_time)

        expected = self._expected_5m_batch_symbols()
        if expected and expected.issubset(pending):
            self.flush_completed_5m_batch(bar.end_time)
        return True

    def _expected_5m_batch_symbols(self) -> set[str]:
        return {
            symbol
            for symbol, item in self._items.items()
            if item.tradable_flag or symbol in self._portfolio.open_positions
        }

    def _schedule_5m_batch_flush(self, end_time: datetime) -> None:
        if end_time in self._pending_5m_batch_tasks:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        grace_s = max(0.0, float(getattr(self._settings, "completed_5m_batch_grace_s", 30.0)))
        task = loop.create_task(self._flush_5m_batch_after_grace(end_time, grace_s))
        task.add_done_callback(self._log_task_exception)
        self._pending_5m_batch_tasks[end_time] = task

    async def _flush_5m_batch_after_grace(self, end_time: datetime, grace_s: float) -> None:
        if grace_s > 0:
            await asyncio.sleep(grace_s)
        self._pending_5m_batch_tasks.pop(end_time, None)
        self.flush_completed_5m_batch(end_time)

    def flush_completed_5m_batch(self, end_time: datetime) -> int:
        """Apply all same-close bars before deterministically allocating entries."""

        batch = self._pending_5m_batches.pop(end_time, None)
        task = self._pending_5m_batch_tasks.pop(end_time, None)
        current_task = None
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            pass
        if task is not None and task is not current_task:
            task.cancel()
        if not batch:
            return 0

        ordered = sorted(batch.items(), key=lambda row: row[0])
        expected = self._expected_5m_batch_symbols()
        missing = sorted(expected.difference(batch))
        self._expected_5m_bars += len(expected)
        self._missing_5m_bars += len(missing)
        self._last_missing_5m_symbols = missing
        evaluable: list[tuple[str, Bar]] = []
        for symbol, bar in ordered:
            market = self._markets[symbol]
            state = self._symbols.get(symbol)
            recovery_cutoff = self._recovery_cutoff_by_symbol.get(symbol)
            recovering = recovery_cutoff is not None and bar.end_time <= recovery_cutoff
            apply_completed_5m_bar(market, bar, state=None if recovering else state)
            if not recovering:
                self._recovery_cutoff_by_symbol.pop(symbol, None)
                evaluable.append((symbol, bar))
            self._bars_processed += 1

        self._causal_relative_dislocations = causal_relative_dislocation_atr(
            {
                symbol: list(market.bars_5m)
                for symbol, market in self._markets.items()
                if market.bars_5m
            },
            self._symbol_to_sector,
            {
                symbol: float(getattr(item, "daily_atr_estimate", 0.0))
                for symbol, item in self._items.items()
            },
        )

        self._collecting_entry_candidates = []
        try:
            for symbol, bar in evaluable:
                self._last_bar_ts = bar.end_time
                self._process_intraday_bar(symbol, bar, bar.end_time)
            candidates = list(self._collecting_entry_candidates)
        finally:
            self._collecting_entry_candidates = None
        self._dispatch_batched_entry_candidates(candidates)
        return len(ordered)

    def flush_all_completed_5m_batches(self) -> int:
        applied = 0
        for end_time in sorted(self._pending_5m_batches):
            applied += self.flush_completed_5m_batch(end_time)
        return applied

    def get_position_snapshot(self) -> list[dict[str, Any]]:
        snapshots = []
        for symbol, state in self._symbols.items():
            market = self._markets.get(symbol)
            if state.position is None or market is None or market.last_price is None:
                continue
            unrealized_r = (market.last_price - state.position.entry_price) / max(state.risk_per_share, 1e-9)
            snapshots.append({
                "strategy_type": "strategy_iaric",
                "symbol": symbol,
                "direction": "LONG",
                "entry_price": state.position.entry_price,
                "qty": state.position.qty_open,
                "unrealized_pnl_r": round(unrealized_r, 3),
                "route_family": state.route_family,
                "mfe_stage": state.mfe_stage,
            })
        return snapshots

    def open_order_count(self) -> int:
        return len(self._order_index)

    # ── Intraday processing ─────────────────────────────────────────

    def _process_intraday_bar(self, symbol: str, bar_5m: Bar, now: datetime) -> None:
        """Core pullback processing on each 5m bar."""
        state = self._symbols.get(symbol)
        item = self._items.get(symbol)
        market = self._markets.get(symbol)
        if state is None or item is None or market is None:
            return

        # Periodic indicator snapshot (every 6th bar = 30 min)
        if (state.bars_seen_today % 6 == 0
                and state.stage not in ("WATCHING", "INVALIDATED")):
            kit = self._instr_kit
            if kit:
                try:
                    kit.on_indicator_snapshot(
                        pair=symbol,
                        indicators={
                            "bars_seen_today": float(state.bars_seen_today),
                            "daily_signal_score": state.daily_signal_score,
                            "intraday_score": state.intraday_score,
                            "mfe_stage": float(state.mfe_stage),
                            "stop_level": state.stop_level,
                            "daily_atr": state.daily_atr,
                            "hold_bars": float(state.hold_bars),
                        },
                        signal_name=f"iaric_pb_{state.route_family.lower()}" if state.route_family else "iaric_pb",
                        signal_strength=state.intraday_score / 100.0,
                        decision="IN_POSITION" if state.in_position else state.stage,
                        strategy_type="strategy_iaric",
                        exchange_timestamp=now,
                        context={
                            "route_family": state.route_family,
                            "trigger_tier": state.trigger_tier,
                            "trend_tier": state.trend_tier,
                            "stage": state.stage,
                        },
                    )
                except Exception:
                    pass

        if state.in_position:
            self._record_decision("MANAGING_POSITION", {"symbol": symbol})
            self._manage_position_intraday(symbol, bar_5m, now)
        else:
            self._check_entry_routes(symbol, bar_5m, now)

    def _check_entry_routes(self, symbol: str, bar_5m: Bar, now: datetime) -> None:
        """Check all 5 entry routes in priority order."""
        state = self._symbols[symbol]
        item = self._items[symbol]
        market = self._markets[symbol]
        cfg = self._settings

        # Skip if not tradable or already has pending order
        if not item.tradable_flag and not item.daily_signal_score:
            self._record_decision("NO_SIGNAL", {"symbol": symbol, "reason": "no_tradable_flag"})
            self._log_missed(symbol=symbol, blocked_by="not_tradable",
                             block_reason="no_signal_no_tradable_flag",
                             exchange_timestamp=now, route="ENTRY_CHECK")
            return
        if state.entry_order is not None or state.active_order_id is not None:
            return
        if not timing_gate_allows_entry(now, cfg):
            if state.intraday_score > 0:
                self._log_missed(symbol=symbol, blocked_by="timing_gate",
                                 block_reason="outside_entry_window", exchange_timestamp=now,
                                 route="ENTRY_CHECK")
            return
        if self._portfolio.regime_allows_no_new_entries:
            if state.intraday_score > 0:
                self._log_missed(symbol=symbol, blocked_by="regime_gate",
                                 block_reason="regime_no_new_entries", exchange_timestamp=now,
                                 route="ENTRY_CHECK")
            return

        # Check max positions
        max_pos = cfg.pb_max_positions
        if self._artifact.regime.tier == "B":
            max_pos = min(max_pos, cfg.max_positions_tier_b)
        if len(self._portfolio.open_positions) + len(self._portfolio.pending_entry_risk) >= max_pos:
            self._log_missed(symbol=symbol, blocked_by="max_positions",
                             block_reason="at_max_positions", exchange_timestamp=now,
                             route="ENTRY_CHECK")
            return

        # Sector cap
        if self._portfolio.sector_position_count(self._symbol_to_sector, item.sector) >= cfg.max_positions_per_sector:
            self._log_missed(symbol=symbol, blocked_by="sector_limit",
                             block_reason="sector_cap_reached", exchange_timestamp=now,
                             route="ENTRY_CHECK")
            return

        # Spread guard
        if market.spread_pct > cfg.max_median_spread_pct * 2.0:
            self._log_missed(symbol=symbol, blocked_by="spread_gate",
                             block_reason="spread_too_wide", exchange_timestamp=now,
                             route="ENTRY_CHECK")
            return

        bar_idx = max(state.bars_seen_today - 1, 0)
        if iaric_core_logic.maybe_reset_invalidated_state(state, bar_idx):
            pass
        elif state.stage == "INVALIDATED":
            return

        if state.stage in {"WATCHING", "APERTURE_CONFIRM_ARMED"}:
            if self._try_aperture_entry(symbol, bar_5m, now):
                return

        if state.stage == "RETEST_ARMED":
            self._try_open_scored_retest(symbol, bar_5m, now)
            return

        if state.stage in {"FLUSH_LOCKED", "RECLAIMING"}:
            self._try_opening_reclaim(symbol, bar_5m, now)
            return

        if state.stage == "READY":
            self._try_ready_entry(symbol, bar_5m, now)
            return

        if state.stage != "WATCHING":
            return

        if cfg.pb_opening_reclaim_enabled and self._try_opening_reclaim(symbol, bar_5m, now):
            return

        if cfg.pb_delayed_confirm_enabled and self._try_delayed_confirm(symbol, bar_5m, now):
            return

        first_open = market.bars_5m[0].open if market.bars_5m else bar_5m.open
        if not iaric_core_logic.opening_gap_eligible(cfg, item.previous_close, first_open):
            iaric_core_logic.invalidate_route_state(state, "opening_gap_reject", int(state.bars_seen_today) + 78)
            self._log_missed(symbol=symbol, blocked_by="gap_gate",
                             block_reason="opening_gap_outside_range", exchange_timestamp=now,
                             route="ENTRY_CHECK")
            return

        if cfg.pb_v2_vwap_bounce_enabled and self._try_vwap_bounce(symbol, bar_5m, now):
            return

        if cfg.pb_v2_afternoon_retest_enabled:
            if self._try_afternoon_retest(symbol, bar_5m, now):
                return

        open_after_bar = int(getattr(cfg, "pb_v2_open_scored_after_bar", 0)) if cfg.pb_v2_enabled else 0
        if iaric_core_logic.route_enabled(cfg, "OPEN_SCORED_ENTRY") and bar_idx >= open_after_bar:
            if self._try_open_scored_entry(symbol, bar_5m, now):
                return

    def _session_atr(self, symbol: str) -> float:
        """Estimate intraday ATR through the shared causal parity helper."""
        return iaric_core_logic.estimate_session_atr(
            self._items[symbol],
            list(self._markets[symbol].bars_5m),
            self._symbols[symbol].daily_atr,
        )

    def _initial_stop(self, setup_low: float, daily_atr: float, session_atr: float) -> float:
        """Compute initial stop: session ATR based with daily ATR cap (research parity)."""
        return iaric_core_logic.compute_initial_stop(self._settings, setup_low, daily_atr, session_atr)

    def _volume_ratio(self, bar: Bar, symbol: str) -> float:
        """Compute bar volume / expected 5m volume."""
        return iaric_core_logic.compute_volume_ratio(bar, self._items.get(symbol))

    def _try_opening_reclaim(self, symbol: str, bar_5m: Bar, now: datetime) -> bool:
        state = self._symbols[symbol]
        market = self._markets[symbol]
        step = iaric_core_logic.advance_opening_reclaim_route(
            self._settings,
            state,
            self._items[symbol],
            bar_5m,
            market,
            max(state.bars_seen_today - 1, 0),
            self._session_atr(symbol),
            bars=list(market.bars_5m),
        )
        if step is not None and step.stage == "READY":
            state.entry_atr = self._session_atr(symbol)
        return step is not None

    def _try_open_scored_entry(self, symbol: str, bar_5m: Bar, now: datetime) -> bool:
        """Open-scored entry: score-ranked broad entry for qualified candidates."""
        state = self._symbols[symbol]
        market = self._markets[symbol]
        cfg = self._settings

        if not iaric_core_logic.open_scored_eligible(
            cfg,
            {
                "daily_signal_score": state.daily_signal_score,
                "daily_signal_rank_pct": state.entry_rank_pct,
                "rescue_flow_candidate": state.rescue_flow_candidate,
                "trigger_types": list(state.trigger_types),
            },
        ):
            return False

        bar_idx = max(state.bars_seen_today - 1, 0)
        transition = iaric_core_logic.open_scored_transition(cfg)
        if transition == "confirmed_retest":
            after_bar = (
                int(getattr(cfg, "pb_v2_open_scored_after_bar", 0))
                if cfg.pb_v2_enabled
                else 0
            )
            if bar_idx != after_bar:
                return False
            step = iaric_core_logic.arm_open_scored_retest_route(
                cfg,
                state,
                self._items[symbol],
                bar_5m,
                market,
                bar_idx,
                self._session_atr(symbol),
                bars=list(market.bars_5m),
            )
            if step is None:
                self._log_missed(
                    symbol=symbol,
                    blocked_by="entry_transition",
                    block_reason="open_scored_retest_not_armed",
                    exchange_timestamp=now,
                    route="OPEN_SCORED_RETEST",
                )
                return False
            state.entry_atr = self._session_atr(symbol)
            self._record_decision(
                "RETEST_ARMED",
                {
                    "symbol": symbol,
                    "route": "OPEN_SCORED_RETEST",
                    "target": state.target_entry_price,
                    "expires_bar": state.improvement_expires,
                    "score": state.intraday_score,
                },
            )
            return True

        if transition == "resting_retrace":
            after_bar = (
                int(getattr(cfg, "pb_v2_open_scored_after_bar", 0))
                if cfg.pb_v2_enabled
                else 0
            )
            if bar_idx != after_bar:
                return False
            step = iaric_core_logic.arm_open_scored_retrace_limit_route(
                cfg,
                state,
                self._items[symbol],
                bar_5m,
                market,
                bar_idx,
                self._session_atr(symbol),
                bars=list(market.bars_5m),
            )
            if step is None or step.acceptance is None:
                self._log_missed(
                    symbol=symbol,
                    blocked_by="entry_transition",
                    block_reason="open_scored_retrace_limit_not_armed",
                    exchange_timestamp=now,
                    route="OPEN_SCORED_RETRACE_LIMIT",
                )
                return False
            iaric_core_logic.apply_entry_acceptance(state, step.acceptance)
            state.entry_atr = self._session_atr(symbol)
            self._record_decision(
                "RETRACE_LIMIT_ARMED",
                {
                    "symbol": symbol,
                    "route": "OPEN_SCORED_RETRACE_LIMIT",
                    "target": state.target_entry_price,
                    "expires_bar": state.improvement_expires,
                    "score": state.intraday_score,
                },
            )
            self._fire_entry(symbol, bar_5m, now, "OPEN_SCORED_RETRACE_LIMIT")
            return True

        step = iaric_core_logic.activate_open_scored_direct_route(
            cfg,
            state,
            self._items[symbol],
            bar_5m,
            market,
            bar_idx,
            self._session_atr(symbol),
            bars=list(market.bars_5m),
        )
        if step is None or step.acceptance is None:
            return False
        iaric_core_logic.apply_entry_acceptance(state, step.acceptance)
        state.entry_atr = float(step.acceptance.session_atr)
        self._fire_entry(symbol, bar_5m, now, step.acceptance.route_family)
        return True

    def _try_open_scored_retest(self, symbol: str, bar_5m: Bar, now: datetime) -> bool:
        """Advance the shared completed-bar retest and submit for next-bar fill."""

        state = self._symbols[symbol]
        market = self._markets[symbol]
        step = iaric_core_logic.advance_open_scored_retest_route(
            self._settings,
            state,
            self._items[symbol],
            bar_5m,
            market,
            max(state.bars_seen_today - 1, 0),
            self._session_atr(symbol),
            bars=list(market.bars_5m),
        )
        if step is None:
            return False
        if step.stage == "INVALIDATED":
            self._log_missed(
                symbol=symbol,
                blocked_by="entry_transition",
                block_reason=step.reason,
                exchange_timestamp=now,
                route="OPEN_SCORED_RETEST",
            )
            return True
        if step.acceptance is not None:
            iaric_core_logic.apply_entry_acceptance(state, step.acceptance)
            state.entry_atr = self._session_atr(symbol)
            self._fire_entry(symbol, bar_5m, now, step.acceptance.route_family)
            return True
        return False

    def _try_delayed_confirm(self, symbol: str, bar_5m: Bar, now: datetime) -> bool:
        state = self._symbols[symbol]
        market = self._markets[symbol]
        step = iaric_core_logic.activate_delayed_confirm_route(
            self._settings,
            state,
            self._items[symbol],
            bar_5m,
            market,
            max(state.bars_seen_today - 1, 0),
            self._session_atr(symbol),
            bars=list(market.bars_5m),
        )
        if step is None:
            return False
        state.entry_atr = self._session_atr(symbol)
        return True

    def _try_vwap_bounce(self, symbol: str, bar_5m: Bar, now: datetime) -> bool:
        state = self._symbols[symbol]
        market = self._markets[symbol]
        step = iaric_core_logic.activate_vwap_bounce_route(
            self._settings,
            state,
            self._items[symbol],
            bar_5m,
            market,
            max(state.bars_seen_today - 1, 0),
            self._session_atr(symbol),
            bars=list(market.bars_5m),
        )
        if step is None:
            return False
        state.entry_atr = self._session_atr(symbol)
        return True

    def _try_afternoon_retest(self, symbol: str, bar_5m: Bar, now: datetime) -> bool:
        state = self._symbols[symbol]
        market = self._markets[symbol]
        step = iaric_core_logic.activate_afternoon_retest_route(
            self._settings,
            state,
            self._items[symbol],
            bar_5m,
            market,
            max(state.bars_seen_today - 1, 0),
            self._session_atr(symbol),
            bars=list(market.bars_5m),
        )
        if step is None:
            return False
        state.entry_atr = self._session_atr(symbol)
        return True

    def _try_aperture_entry(self, symbol: str, bar_5m: Bar, now: datetime) -> bool:
        state = self._symbols[symbol]
        market = self._markets[symbol]
        step = iaric_core_logic.advance_aperture_route(
            self._settings,
            state,
            self._items[symbol],
            bar_5m,
            market,
            max(state.bars_seen_today - 1, 0),
            self._session_atr(symbol),
            bars=list(market.bars_5m),
            relative_dislocation_atr=getattr(
                self, "_causal_relative_dislocations", {}
            ).get(symbol),
        )
        if state.opportunity_audit_bar_idx == max(state.bars_seen_today - 1, 0):
            for audit in state.opportunity_audit_events:
                self._diagnostics.log_decision(
                    "APERTURE_LANE_FUNNEL",
                    {"symbol": symbol, **audit, "exchange_timestamp": now},
                )
        if step is None:
            return False
        if step.acceptance is not None:
            iaric_core_logic.apply_entry_acceptance(state, step.acceptance)
            state.entry_atr = float(step.acceptance.session_atr)
            self._fire_entry(symbol, bar_5m, now, step.acceptance.route_family)
        return True

    def _try_ready_entry(self, symbol: str, bar_5m: Bar, now: datetime) -> bool:
        state = self._symbols[symbol]
        market = self._markets[symbol]
        step = iaric_core_logic.evaluate_ready_entry(
            self._settings,
            state,
            self._items[symbol],
            bar_5m,
            market,
            max(state.bars_seen_today - 1, 0),
            self._session_atr(symbol),
            bars=list(market.bars_5m),
        )
        if step is None:
            return False
        if step.acceptance is not None:
            iaric_core_logic.apply_entry_acceptance(state, step.acceptance)
            state.entry_atr = self._session_atr(symbol)
            self._fire_entry(symbol, bar_5m, now, step.acceptance.route_family)
        return True

    def _fire_entry(self, symbol: str, bar_5m: Bar, now: datetime, route: str) -> None:
        """Common entry submission for all routes."""
        state = self._symbols[symbol]
        if self._collecting_entry_candidates is not None:
            item = self._items[symbol]
            self._collecting_entry_candidates.append(
                _BatchEntryCandidate(
                    symbol=symbol,
                    bar=bar_5m,
                    timestamp=now,
                    route=route,
                    score=float(state.intraday_score),
                    entry_rank_pct=float(state.entry_rank_pct),
                    entry_rsi=float(state.entry_rsi),
                    artifact_index=self._artifact_index.get(symbol, len(self._artifact_index)),
                )
            )
            return
        self._dispatch_entry_candidate(
            _BatchEntryCandidate(
                symbol=symbol,
                bar=bar_5m,
                timestamp=now,
                route=route,
                score=float(state.intraday_score),
                entry_rank_pct=float(state.entry_rank_pct),
                entry_rsi=float(state.entry_rsi),
                artifact_index=self._artifact_index.get(symbol, len(self._artifact_index)),
            )
        )

    def _dispatch_entry_candidate(self, candidate: _BatchEntryCandidate) -> None:
        state = self._symbols[candidate.symbol]
        aperture_family = iaric_core_logic.aperture_family_from_route(candidate.route)
        if aperture_family:
            cap = iaric_core_logic.aperture_family_daily_cap(
                self._settings,
                aperture_family,
            )
            if cap is not None and self._aperture_family_counts.get(aperture_family, 0) >= cap:
                iaric_core_logic.reset_route_state(state)
                self._log_missed(
                    symbol=candidate.symbol,
                    blocked_by="route_capacity",
                    block_reason="aperture_family_daily_cap",
                    exchange_timestamp=candidate.timestamp,
                    route=candidate.route,
                )
                return
        lane_capacity_reason = self._entry_lane_capacity_reason(candidate)
        if lane_capacity_reason:
            iaric_core_logic.reset_route_state(state)
            self._log_missed(
                symbol=candidate.symbol,
                blocked_by=(
                    "issuer_exposure"
                    if lane_capacity_reason.startswith("issuer_")
                    else "route_capacity"
                ),
                block_reason=lane_capacity_reason,
                exchange_timestamp=candidate.timestamp,
                route=candidate.route,
            )
            return
        state.active_order_id = "SUBMITTING_ENTRY"
        if iaric_core_logic.is_open_scored_route(candidate.route):
            self._open_scored_count += 1
        if aperture_family:
            self._aperture_family_counts[aperture_family] = (
                self._aperture_family_counts.get(aperture_family, 0) + 1
            )
        self._daily_entry_symbols.append(candidate.symbol)
        if bool(getattr(state, "rescue_flow_candidate", False)):
            self._rescue_entry_count += 1
        lane_id = lane_id_for_route(
            candidate.route,
            rescue_candidate=bool(getattr(state, "rescue_flow_candidate", False)),
        )
        self._lane_entry_counts[lane_id] = self._lane_entry_counts.get(lane_id, 0) + 1
        task = asyncio.create_task(
            self._submit_entry(candidate.symbol, candidate.timestamp, candidate.route)
        )
        task.add_done_callback(self._log_task_exception)

    def _entry_lane_capacity_reason(self, candidate: _BatchEntryCandidate) -> str:
        """Return a shared lane/issuer reservation rejection, if any."""

        state = self._symbols[candidate.symbol]
        active_symbols = list(self._portfolio.open_positions)
        active_symbols.extend(self._portfolio.pending_entry_risk)
        active_symbols.extend(
            symbol
            for symbol, symbol_state in self._symbols.items()
            if symbol_state.active_order_id or symbol_state.entry_order is not None
        )
        issuer_decision = issuer_exposure_decision(
            self._settings,
            candidate.symbol,
            active_symbols=active_symbols,
            daily_entry_symbols=self._daily_entry_symbols,
        )
        if not issuer_decision.allowed:
            return issuer_decision.reason
        lane_id = lane_id_for_route(
            candidate.route,
            rescue_candidate=bool(getattr(state, "rescue_flow_candidate", False)),
        )
        cap = lane_daily_cap(self._settings, lane_id)
        if cap is not None and self._lane_entry_counts.get(lane_id, 0) >= cap:
            return "lane_daily_cap"
        if bool(getattr(state, "rescue_flow_candidate", False)) and (
            self._rescue_entry_count
            >= max(int(getattr(self._settings, "pb_rescue_max_per_day", 1)), 0)
        ):
            return "rescue_daily_cap"
        return ""

    def _dispatch_batched_entry_candidates(self, candidates: list[_BatchEntryCandidate]) -> None:
        """Apply replay-compatible score priority before portfolio capacity."""

        if not candidates:
            return
        cfg = self._settings
        max_positions = cfg.pb_max_positions
        if self._artifact.regime.tier == "B":
            max_positions = min(max_positions, cfg.max_positions_tier_b)
        used_slots = len(self._portfolio.open_positions) + len(self._portfolio.pending_entry_risk)
        available_slots = max(max_positions - used_slots, 0)
        open_scored_cap = iaric_core_logic.open_scored_slot_cap(
            cfg,
            available_slots,
            has_intraday_candidates=any(
                not iaric_core_logic.is_open_scored_route(candidate.route)
                for candidate in candidates
            ) or any(
                not state.in_position
                and state.stage
                in {"WATCHING", "FLUSH_LOCKED", "RECLAIMING", "RETEST_ARMED", "READY"}
                for state in self._symbols.values()
            ),
        )
        sector_counts: dict[str, int] = {}
        for symbol in self._portfolio.open_positions:
            sector = self._symbol_to_sector.get(symbol, "")
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        ranked = sorted(
            candidates,
            key=lambda candidate: (
                iaric_core_logic.route_priority_value(
                    cfg,
                    candidate.route,
                    candidate.score,
                ),
                candidate.entry_rank_pct,
                candidate.entry_rsi,
                candidate.artifact_index,
            ),
        )
        issuer_arbitration = issuer_batch_arbitration(
            cfg,
            (
                IssuerEntryCandidate(
                    symbol=candidate.symbol,
                    route_family=candidate.route,
                    score=candidate.score,
                    stable_rank=int(round(candidate.entry_rank_pct * 10_000)),
                )
                for candidate in ranked
            ),
        )
        for candidate in ranked:
            state = self._symbols[candidate.symbol]
            item = self._items[candidate.symbol]
            reason = ""
            if candidate.symbol not in issuer_arbitration.selected_symbols:
                winner = issuer_arbitration.rejected_by_winner.get(candidate.symbol, "")
                iaric_core_logic.reset_route_state(state)
                self._log_missed(
                    symbol=candidate.symbol,
                    blocked_by="issuer_exposure",
                    block_reason="issuer_duplicate_event",
                    exchange_timestamp=candidate.timestamp,
                    route=candidate.route,
                )
                self._record_decision(
                    "ISSUER_EVENT_DEDUPED",
                    {"symbol": candidate.symbol, "winner": winner, "route": candidate.route},
                )
                continue
            aperture_family = iaric_core_logic.aperture_family_from_route(
                candidate.route
            )
            aperture_cap = (
                iaric_core_logic.aperture_family_daily_cap(cfg, aperture_family)
                if aperture_family
                else None
            )
            lane_capacity_reason = self._entry_lane_capacity_reason(candidate)
            if lane_capacity_reason:
                iaric_core_logic.reset_route_state(state)
                self._log_missed(
                    symbol=candidate.symbol,
                    blocked_by=(
                        "issuer_exposure"
                        if lane_capacity_reason.startswith("issuer_")
                        else "route_capacity"
                    ),
                    block_reason=lane_capacity_reason,
                    exchange_timestamp=candidate.timestamp,
                    route=candidate.route,
                )
                continue
            if aperture_cap is not None and self._aperture_family_counts.get(
                aperture_family,
                0,
            ) >= aperture_cap:
                iaric_core_logic.reset_route_state(state)
                self._log_missed(
                    symbol=candidate.symbol,
                    blocked_by="route_capacity",
                    block_reason="aperture_family_daily_cap",
                    exchange_timestamp=candidate.timestamp,
                    route=candidate.route,
                )
                continue
            if iaric_core_logic.is_open_scored_route(candidate.route) and (
                self._open_scored_count >= open_scored_cap
            ):
                iaric_core_logic.reset_route_state(state)
                self._log_missed(
                    symbol=candidate.symbol,
                    blocked_by="route_capacity",
                    block_reason="open_scored_slot_cap",
                    exchange_timestamp=candidate.timestamp,
                    route=candidate.route,
                )
                continue
            if used_slots >= max_positions:
                reason = "slot_cap_reject"
            elif sector_counts.get(item.sector, 0) >= cfg.max_positions_per_sector:
                reason = "sector_cap_reject"
            if reason:
                iaric_core_logic.invalidate_route_state(
                    state,
                    reason,
                    int(state.bars_seen_today) + 78,
                )
                self._log_missed(
                    symbol=candidate.symbol,
                    blocked_by="max_positions" if reason.startswith("slot") else "sector_limit",
                    block_reason=reason,
                    exchange_timestamp=candidate.timestamp,
                    route=candidate.route,
                )
                continue
            used_slots += 1
            sector_counts[item.sector] = sector_counts.get(item.sector, 0) + 1
            self._dispatch_entry_candidate(candidate)

    def _compute_micropressure(self, symbol: str, bar_5m: Bar) -> str:
        """Route-aligned micropressure proxy from completed 5m bars."""
        market = self._markets.get(symbol)
        item = self._items.get(symbol)
        state = self._symbols.get(symbol)
        if market is None or item is None or state is None:
            return "NEUTRAL"
        bars = list(market.bars_5m)
        if not bars:
            bars = [bar_5m]
        return iaric_core_logic.micropressure_label(
            bars,
            len(bars) - 1,
            state.reclaim_level if state.reclaim_level > 0 else bar_5m.close,
            item,
        )

    # ── Position management ─────────────────────────────────────────

    def _manage_position_intraday(self, symbol: str, bar_5m: Bar, now: datetime) -> None:
        """Manage open position: exits, MFE stages, partials."""
        state = self._symbols[symbol]
        market = self._markets[symbol]
        position = state.position
        if position is None or market.last_price is None:
            return

        # The protective stop carried into the bar owns all intrabar lows.
        # A replacement justified by this bar's completed high is active only
        # for subsequent market data.  Resolve the conservative stop-first
        # case before crediting the bar's MFE, matching replay semantics.
        active_stop_level = float(state.stop_level)
        if bar_5m.low <= active_stop_level:
            state.stopped_out_today = True
            self._diagnostics.log_decision(
                "EXIT",
                {"symbol": symbol, "reason": "STOP_HIT", "active_stop": active_stop_level},
            )
            self._request_full_exit(symbol, "STOP_HIT")
            return

        # Update MFE tracking
        position.max_favorable_price = max(position.max_favorable_price, bar_5m.high)
        position.max_adverse_price = min(position.max_adverse_price, bar_5m.low)
        state.hold_bars += 1

        entry_price = position.entry_price
        risk_per_share = max(state.risk_per_share, position.initial_risk_per_share, 0.01)
        unrealized_r = (bar_5m.close - entry_price) / risk_per_share
        max_mfe_r = (position.max_favorable_price - entry_price) / risk_per_share
        # Entry-time session ATR is the shared live/replay trail basis.  Daily
        # ATR is a different scale and made live trails materially looser.
        entry_atr = max(state.entry_atr, 0.01)

        # Update MFE stages (3->2->1 order, uses entry_atr for trail)
        prev_mfe_stage = state.mfe_stage
        new_stop = update_mfe_stages(
            state=state,
            bar_high=bar_5m.high,
            entry_price=entry_price,
            risk_per_share=risk_per_share,
            entry_atr=entry_atr,
            config=self._settings,
        )
        if state.mfe_stage != prev_mfe_stage:
            self._diagnostics.log_decision("MFE_STAGE", {
                "symbol": symbol, "from": prev_mfe_stage, "to": state.mfe_stage,
                "mfe_r": round(max_mfe_r, 3), "new_stop": round(new_stop, 5),
            })

        # Stale position tighten (research parity: tightens stop, does NOT exit)
        stale_stop = compute_stale_tighten(
            hold_bars=state.hold_bars,
            max_mfe_r=max_mfe_r,
            entry_price=entry_price,
            risk_per_share=risk_per_share,
            current_stop=new_stop,
            stale_bars=self._settings.pb_v2_stale_bars,
            stale_mfe_thresh=self._settings.pb_v2_stale_mfe_thresh,
            stale_tighten_pct=getattr(self._settings, 'pb_v2_stale_tighten_pct', 0.50),
        )
        if stale_stop is not None:
            new_stop = max(new_stop, stale_stop)

        if new_stop > state.stop_level:
            old_stop = state.stop_level
            stop_request = IARICStopUpdateRequest(
                symbol=symbol,
                stop_price=new_stop,
                qty=position.qty_open,
                reason="mfe_stage_trail",
            )
            core_state = build_core_runtime_state(self)
            new_state, actions, _events = iaric_core_logic.on_bar(
                core_state,
                bar_ts=self._last_bar_ts,
                stop_update=stop_request,
            )
            apply_core_runtime_state(self, new_state)
            state = self._symbols[symbol]
            position = state.position
            if any(isinstance(action, ReplaceProtectiveStop) for action in actions):
                asyncio.create_task(self._replace_stop(symbol)).add_done_callback(self._log_task_exception)
            kit = self._kit_cache
            if kit:
                kit.log_stop_adjustment(
                    trade_id=position.trade_id or f"IARIC-{symbol}",
                    symbol=symbol, old_stop=old_stop, new_stop=new_stop,
                    adjustment_type="trailing", trigger="mfe_stage_trail",
                )

        if (
            anchor_exit_enabled(self._settings, state.route_family)
            and position.reversion_anchor > position.entry_price
            and bar_5m.close >= position.reversion_anchor
        ):
            self._diagnostics.log_decision(
                "EXIT",
                {
                    "symbol": symbol,
                    "reason": "REVERSION_ANCHOR",
                    "anchor": position.reversion_anchor,
                    "event_id": position.opportunity_event_id,
                },
            )
            self._request_full_exit(symbol, "REVERSION_ANCHOR")
            return

        # V2 partial profit (triggers on MFE, not unrealized -- research parity)
        partial_trigger_r = float(self._settings.pb_v2_partial_profit_trigger_r)
        if (
            check_v2_partial(max_mfe_r, state.v2_partial_taken, partial_trigger_r)
            and position.pending_partial_stop <= 0
            and state.exit_order is None
        ):
            partial_qty = partial_exit_quantity(
                current_qty=position.qty_open,
                original_qty=position.qty_entry,
                fraction=self._settings.pb_v2_partial_profit_fraction,
                minimum_remaining_size_pct=self._settings.minimum_remaining_size_pct,
            )
            if partial_qty <= 0:
                return
            self._diagnostics.log_decision("V2_PARTIAL", {
                "symbol": symbol, "mfe_r": round(max_mfe_r, 3),
                "partial_qty": partial_qty,
            })
            partial_stop = entry_price + self._settings.pb_v2_partial_profit_remainder_stop_r * risk_per_share
            item = self._items[symbol]
            partial_request = IARICPartialExitRequest(
                client_order_id=f"{symbol}-partial-{int(now.timestamp())}",
                symbol=symbol,
                qty=partial_qty,
                reason="TP",
                remainder_stop_price=partial_stop,
                execution_buffer=item.tick_size,
            )
            core_state = build_core_runtime_state(self)
            new_state, actions, _events = iaric_core_logic.on_bar(
                core_state,
                bar_ts=self._last_bar_ts,
                partial_exit_request=partial_request,
            )
            apply_core_runtime_state(self, new_state)
            if any(isinstance(action, SubmitMarketExit) for action in actions):
                asyncio.create_task(
                    self._submit_market_exit(symbol, partial_qty, OrderRole.TP)
                ).add_done_callback(self._log_task_exception)
            return

        hold_days = (now.astimezone(ET).date() - position.entry_time.astimezone(ET).date()).days

        ema10 = state.ema10_daily if state.ema10_daily > 0 else None
        rsi14 = state.rsi14_daily if state.rsi14_daily > 0 else None
        flow_hist = None
        item = self._items.get(symbol)
        if item and hasattr(item, 'flow_proxy_gate_pass'):
            if not item.flow_proxy_gate_pass:
                flow_hist = [-1.0, -1.0]

        # Route-specific exit params via _route_param
        quick_exit_loss_r = abs(_route_param(state.route_family, "quick_exit_loss_r", self._settings))
        stale_exit_bars = int(_route_param(state.route_family, "stale_exit_bars", self._settings))
        stale_exit_min_r = _route_param(state.route_family, "stale_exit_min_r", self._settings)

        should_exit, reason = run_exit_chain(
            state=state,
            bar=bar_5m,
            now=now,
            unrealized_r=unrealized_r,
            max_mfe_r=max_mfe_r,
            ema10_value=ema10,
            rsi_value=rsi14,
            session_vwap=market.session_vwap,
            hold_days=hold_days,
            flow_history=flow_hist,
            recent_5m_bars=list(market.bars_5m),
            quick_exit_loss_r=quick_exit_loss_r,
            config=self._settings,
            stale_exit_bars=stale_exit_bars,
            stale_exit_min_r=stale_exit_min_r,
            active_stop_level=active_stop_level,
        )

        if should_exit:
            if reason == "STOP_HIT":
                state.stopped_out_today = True
            self._diagnostics.log_decision("EXIT", {"symbol": symbol, "reason": reason, "unrealized_r": round(unrealized_r, 3)})
            self._request_full_exit(symbol, reason)
            return

        # EOD carry check (near close)
        et_time = now.astimezone(ET).time()
        if self._settings.pb_carry_enabled and et_time >= self._settings.close_block_start:
            close_in_range = 0.0
            if state.session_high > state.session_low > 0:
                daily_range = state.session_high - state.session_low
                close_in_range = (bar_5m.close - state.session_low) / max(daily_range, 1e-9)

            should_carry, decision_path = should_carry_overnight(
                state=state,
                unrealized_r=unrealized_r,
                close_in_range_pct=close_in_range,
                regime_tier=self._artifact.regime.tier,
                flow_history=flow_hist,
                hold_days=hold_days,
                config=self._settings,
            )
            state.carry_decision_path = decision_path

            if not should_carry:
                self._diagnostics.log_decision("FLATTEN_EOD", {"symbol": symbol, "reason": decision_path})
                self._request_full_exit(symbol, f"eod_flatten:{decision_path}")
            elif not carry_quality_gate(state.route_family, close_in_range, max_mfe_r, self._settings):
                state.carry_decision_path = "v2_quality_reject"
                self._diagnostics.log_decision("FLATTEN_EOD", {"symbol": symbol, "reason": "v2_quality_reject"})
                self._request_full_exit(symbol, "eod_flatten:v2_quality_reject")
            else:
                self._diagnostics.log_decision("CARRY_OVERNIGHT", {
                    "symbol": symbol, "path": decision_path,
                    "unrealized_r": round(unrealized_r, 3), "hold_days": hold_days,
                })
                overnight_stop = compute_overnight_stop(
                    entry_price, state.stop_level, risk_per_share, unrealized_r, self._settings,
                )
                if overnight_stop > state.stop_level:
                    old_sl = state.stop_level
                    state.stop_level = overnight_stop
                    position.current_stop = overnight_stop
                    if position.stop_order_id:
                        asyncio.create_task(self._replace_stop(symbol)).add_done_callback(self._log_task_exception)
                    kit = self._kit_cache
                    if kit:
                        kit.log_stop_adjustment(
                            trade_id=position.trade_id or f"IARIC-{symbol}",
                            symbol=symbol, old_stop=old_sl, new_stop=overnight_stop,
                            adjustment_type="time_decay", trigger="overnight_tighten",
                        )

    # ── Order execution ─────────────────────────────────────────────

    async def _submit_entry(self, symbol: str, now: datetime, route: str) -> None:
        state = self._symbols[symbol]
        item = self._items[symbol]
        market = self._markets[symbol]

        if state.in_position or state.entry_order is not None:
            if state.active_order_id == "SUBMITTING_ENTRY":
                state.active_order_id = None
            return

        if market.last_price is None or state.stop_level <= 0:
            if state.active_order_id == "SUBMITTING_ENTRY":
                state.active_order_id = None
            return

        request_build = build_ready_entry_request(
            symbol=symbol,
            state=state,
            item=item,
            market=market,
            portfolio=self._portfolio,
            symbol_to_sector=self._symbol_to_sector,
            settings=self._settings,
            now=now,
            route=route,
        )
        if request_build.entry_request is None:
            self._diagnostics.log_decision("ENTRY_BLOCKED", {"symbol": symbol, "reason": request_build.reason, "route": route})
            self._log_missed(symbol=symbol, blocked_by="portfolio_constraints",
                             block_reason=request_build.reason, exchange_timestamp=now, route=route)
            if state.active_order_id == "SUBMITTING_ENTRY":
                state.active_order_id = None
            return

        entry_price = request_build.entry_price
        state.risk_per_share = max(entry_price - state.stop_level, 0.01)
        entry_request = request_build.entry_request
        core_state = build_core_runtime_state(self)
        new_state, actions, _events = iaric_core_logic.on_bar(
            core_state,
            bar_ts=self._last_bar_ts,
            entry_request=entry_request,
        )
        apply_core_runtime_state(self, new_state)
        state = self._symbols[symbol]
        submit_action = next((action for action in actions if isinstance(action, SubmitEntry)), None)
        if submit_action is None:
            if state.active_order_id == "SUBMITTING_ENTRY":
                state.active_order_id = None
            return

        order = build_entry_order(
            item,
            self._account_id,
            submit_action.qty,
            submit_action.limit_price or entry_price,
            float(submit_action.risk_context.get("stop_for_risk", state.stop_level)),
            signal_id=f"{symbol}:{route}:{int(now.timestamp())}",
            bar_id=f"{symbol}:{self._last_bar_ts.isoformat()}" if self._last_bar_ts else "",
            exchange_timestamp=self._last_bar_ts or now,
            ttl_seconds=(
                int(getattr(self._settings, "pb_open_scored_retrace_limit_ttl_seconds", 3600))
                if iaric_core_logic.is_retrace_limit_route(route)
                else 30
            ),
        )
        receipt = await self._oms.submit_intent(Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=order))
        if receipt.oms_order_id:
            state.entry_order = PendingOrderState(
                oms_order_id=receipt.oms_order_id,
                submitted_at=now,
                role="ENTRY",
                requested_qty=submit_action.qty,
                limit_price=submit_action.limit_price or entry_price,
            )
            state.active_order_id = receipt.oms_order_id
            self._portfolio.pending_entry_risk[symbol] = submit_action.qty * state.risk_per_share
            self._order_index[receipt.oms_order_id] = (symbol, "ENTRY")
            self._record_decision("ENTRY_SUBMITTED", {"symbol": symbol, "qty": submit_action.qty, "price": submit_action.limit_price or entry_price, "route": route})
            self._diagnostics.log_order(symbol, "submit_entry", {
                "qty": submit_action.qty, "limit_price": submit_action.limit_price or entry_price, "route": route,
                "sizing_mult": round(request_build.sizing_mult, 3), "gap_up_size_mult": round(request_build.gap_up_mult, 3),
                "daily_score": state.daily_signal_score,
            })
            kit = self._instr_kit
            if kit:
                try:
                    kit.on_order_event(
                        order_id=receipt.oms_order_id,
                        pair=symbol, side="BUY", order_type="LIMIT_ENTRY",
                        status="SUBMITTED", requested_qty=submit_action.qty,
                        requested_price=submit_action.limit_price or entry_price,
                        strategy_type="strategy_iaric",
                        session=self._current_session_type(now),
                        exchange_timestamp=now,
                    )
                except Exception:
                    pass
        else:
            self._record_decision("ENTRY_DENIED", {"symbol": symbol, "denial_reason": receipt.denial_reason or "unknown"})
            if state.active_order_id == "SUBMITTING_ENTRY":
                state.active_order_id = None

    async def _submit_stop(self, symbol: str) -> None:
        state = self._symbols[symbol]
        item = self._items.get(symbol)
        if state.position is None or state.position.qty_open <= 0 or state.position.stop_order_id or item is None:
            return
        try:
            order = build_stop_order(item, self._account_id, state.position.qty_open, state.position.current_stop)
            receipt = await self._oms.submit_intent(Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=order))
            if receipt.oms_order_id:
                state.position.stop_order_id = receipt.oms_order_id
                self._order_index[receipt.oms_order_id] = (symbol, "STOP")
                self._diagnostics.log_order(symbol, "submit_stop", {"qty": state.position.qty_open, "stop_price": state.position.current_stop})
        except Exception as exc:
            logger.error("submit_stop failed for %s: %s", symbol, exc, exc_info=exc)

    async def _replace_stop(self, symbol: str) -> None:
        state = self._symbols[symbol]
        if state.position is None or not state.position.stop_order_id:
            return
        try:
            await self._oms.submit_intent(
                Intent(
                    intent_type=IntentType.REPLACE_ORDER,
                    strategy_id=STRATEGY_ID,
                    target_oms_order_id=state.position.stop_order_id,
                    new_qty=state.position.qty_open,
                    new_stop_price=state.position.current_stop,
                )
            )
            self._diagnostics.log_order(symbol, "replace_stop", {"qty": state.position.qty_open, "stop_price": state.position.current_stop})
        except Exception as exc:
            logger.error("replace_stop failed for %s: %s", symbol, exc, exc_info=exc)

    async def _cancel_stop(self, symbol: str) -> None:
        state = self._symbols[symbol]
        if state.position and state.position.stop_order_id:
            self._expected_stop_cancels.add(state.position.stop_order_id)
            await self._cancel_order(state.position.stop_order_id)

    async def _submit_market_exit(self, symbol: str, qty: int, role: OrderRole) -> None:
        state = self._symbols[symbol]
        item = self._items.get(symbol)
        position = state.position
        if position is None or qty <= 0 or state.exit_order is not None or item is None:
            return
        requested_qty = min(qty, position.qty_open)
        if requested_qty <= 0:
            return
        market = self._markets.get(symbol)
        expected_exit_price = float(market.bid or market.last_price or 0.0) if market else 0.0
        try:
            order = build_market_exit(item, self._account_id, requested_qty, role)
            receipt = await self._oms.submit_intent(Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=order))
            if receipt.oms_order_id:
                state.exit_order = PendingOrderState(
                    oms_order_id=receipt.oms_order_id,
                    submitted_at=datetime.now(timezone.utc),
                    role=role.value,
                    requested_qty=requested_qty,
                    limit_price=expected_exit_price if expected_exit_price > 0 else None,
                )
                self._order_index[receipt.oms_order_id] = (symbol, role.value)
                self._diagnostics.log_order(symbol, "submit_exit", {"qty": requested_qty, "role": role.value})
            elif role == OrderRole.TP and state.position is not None:
                state.position.pending_partial_stop = 0.0
                state.position.pending_partial_stop_buffer = 0.0
        except Exception as exc:
            if role == OrderRole.TP and state.position is not None:
                state.position.pending_partial_stop = 0.0
                state.position.pending_partial_stop_buffer = 0.0
            logger.error("submit_market_exit failed for %s: %s", symbol, exc, exc_info=exc)

    async def _cancel_order(self, oms_order_id: str) -> None:
        await self._oms.submit_intent(Intent(intent_type=IntentType.CANCEL_ORDER, strategy_id=STRATEGY_ID, target_oms_order_id=oms_order_id))

    def _request_full_exit(self, symbol: str, reason: str) -> None:
        state = self._symbols[symbol]
        position = state.position
        if position is None or position.qty_open <= 0:
            return
        flatten_request = IARICFlattenRequest(symbol=symbol, reason=reason, qty=position.qty_open)
        core_state = build_core_runtime_state(self)
        new_state, actions, _events = iaric_core_logic.on_bar(
            core_state,
            bar_ts=self._last_bar_ts,
            flatten_request=flatten_request,
        )
        apply_core_runtime_state(self, new_state)
        for action in actions:
            if isinstance(action, CancelAction):
                task = asyncio.create_task(self._cancel_order(action.target_order_id))
                task.add_done_callback(self._log_task_exception)
            elif isinstance(action, FlattenPosition):
                task = asyncio.create_task(
                    self._cancel_then_exit(action.symbol, action.qty or position.qty_open),
                )
                task.add_done_callback(self._log_task_exception)

    async def _cancel_then_exit(self, symbol: str, qty: int) -> None:
        await self._cancel_stop(symbol)
        await self._submit_market_exit(symbol, qty, OrderRole.EXIT)

    # ── Event handling ──────────────────────────────────────────────

    async def advance(self, now: datetime) -> None:
        await self._refresh_portfolio()
        for symbol, state in self._symbols.items():
            if state.in_position:
                # Staleness watchdog
                if state.last_5m_bar_time is not None:
                    gap = (now - state.last_5m_bar_time).total_seconds()
                    stale_after = float(getattr(self._settings, "completed_5m_stale_after_s", 390.0))
                    if gap > stale_after and now.astimezone(ET).time() >= time(9, 30):
                        logger.warning("IARIC STALE DATA: %s -- no bar for %.0fs", symbol, gap)
        if self._last_save_ts is None or (now - self._last_save_ts).total_seconds() >= 60:
            await self._save_state("interval")

    async def _pulse_loop(self) -> None:
        while self._running:
            await self.advance(datetime.now(timezone.utc))
            await asyncio.sleep(1.0)

    async def _event_loop(self) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            await self._handle_event(event)

    async def _refresh_portfolio(self) -> None:
        strategy_halted = False
        portfolio_halted = False
        try:
            risk_state = await self._oms.get_strategy_risk(STRATEGY_ID)
            strategy_halted = bool(getattr(risk_state, "halted", False))
        except Exception:
            strategy_halted = False
        try:
            portfolio_risk = await self._oms.get_portfolio_risk()
            portfolio_halted = bool(getattr(portfolio_risk, "halted", False))
        except Exception:
            portfolio_halted = False
        self._portfolio.regime_allows_no_new_entries = (
            strategy_halted
            or portfolio_halted
            or self._artifact.regime.tier == "C"
        )

    async def _handle_event(self, event) -> None:
        if event.event_type == OMSEventType.FILL:
            await self._handle_fill(event)
        elif event.event_type == OMSEventType.RISK_HALT:
            await self._handle_risk_halt((event.payload or {}).get("reason", ""))
        elif event.event_type in (OMSEventType.ORDER_CANCELLED, OMSEventType.ORDER_EXPIRED, OMSEventType.ORDER_REJECTED):
            await self._handle_terminal(event)

    async def _handle_risk_halt(self, reason: str) -> None:
        self._portfolio.regime_allows_no_new_entries = True
        self._diagnostics.log_order("PORTFOLIO", "risk_halt", {"reason": reason or "OMS risk halt"})
        for state in self._symbols.values():
            if state.entry_order and not state.entry_order.cancel_requested:
                state.entry_order.cancel_requested = True
                await self._cancel_order(state.entry_order.oms_order_id)

    async def _handle_fill(self, event) -> None:
        payload = event.payload or {}
        symbol, role = self._resolve_order(event.oms_order_id, payload)
        if not symbol:
            if event.oms_order_id:
                self._order_index.pop(event.oms_order_id, None)
            return
        state = self._symbols.get(symbol)
        item = self._items.get(symbol)
        if state is None or item is None:
            return
        fill_qty = int(float(payload.get("qty", 0.0) or 0.0))
        fill_price = float(payload.get("price", 0.0) or 0.0)
        if fill_qty <= 0:
            return

        # Capture pre-fill state for exit instrumentation
        pre_position = deepcopy(state.position) if state.position else None
        pre_sym_state = deepcopy(state)

        # Build core fill
        commission = float(payload.get("commission", 0.0) or 0.0)
        fill = IARICFill(
            oms_order_id=event.oms_order_id or "",
            fill_price=fill_price,
            fill_qty=fill_qty,
            fill_time=event.timestamp,
            commission=commission,
            symbol=symbol,
            order_role=role if role in ("ENTRY", "TP", "EXIT", "STOP") else "UNKNOWN",
        )

        # Route through core
        core_state = build_core_runtime_state(self)
        new_state, actions, events = iaric_core_logic.on_fill(core_state, fill)
        apply_core_runtime_state(self, new_state)

        # Dispatch OMS actions
        for action in actions:
            if isinstance(action, SubmitProtectiveStop):
                await self._submit_stop(action.symbol)
            elif isinstance(action, ReplaceProtectiveStop):
                await self._replace_stop(action.symbol)
            elif isinstance(action, FlattenPosition):
                sym = self._symbols.get(action.symbol)
                if sym and sym.position and sym.position.stop_order_id:
                    await self._cancel_stop(action.symbol)
                await self._submit_market_exit(action.symbol, action.qty, OrderRole.EXIT)

        # Record decision events
        for ev in events:
            self._record_decision(ev.code, ev.details)

        # Entry instrumentation (new position created)
        if any(ev.code == "ENTRY_FILLED" for ev in events):
            await self._record_entry_instrumentation(
                symbol=symbol, event=event,
                fill_price=fill_price, fill_qty=fill_qty,
                payload=payload,
            )

        # Exit instrumentation (full close: position existed, now gone)
        cur_state = self._symbols.get(symbol)
        if pre_position and (cur_state is None or cur_state.position is None):
            await self._record_exit_instrumentation(
                symbol=symbol, event=event,
                fill_price=fill_price, fill_qty=fill_qty,
                commission=commission, role=role,
                pre_position=pre_position,
                pre_sym_state=pre_sym_state,
                payload=payload,
            )

    async def _record_entry_instrumentation(
        self, *, symbol: str, event, fill_price: float, fill_qty: int, payload: dict,
    ) -> None:
        state = self._symbols.get(symbol)
        if not state or not state.position:
            return
        position = state.position
        if self._trade_recorder:
            position.trade_id = await self._trade_recorder.record_entry(
                strategy_id=STRATEGY_ID,
                instrument=symbol,
                direction="LONG",
                quantity=fill_qty,
                entry_price=Decimal(str(fill_price)),
                entry_ts=event.timestamp,
                setup_tag=position.setup_tag,
                entry_type="marketable_limit",
                meta={
                    "entry_signal": f"PB_{state.route_family}",
                    "entry_signal_id": event.oms_order_id or symbol,
                    "entry_signal_strength": state.intraday_score / 100.0,
                    "strategy_params": {
                        "route_family": state.route_family,
                        "daily_signal_score": state.daily_signal_score,
                        "trigger_types": state.trigger_types,
                        "trigger_tier": state.trigger_tier,
                        "trend_tier": state.trend_tier,
                        "sizing_mult": state.sizing_mult,
                        "mfe_stage": state.mfe_stage,
                        "stop0": state.stop_level,
                        "cdd_value": state.cdd_value,
                        "entry_atr": state.entry_atr,
                        "regime_tier": self._artifact.regime.tier if self._artifact.regime else "",
                        "regime_score": getattr(self._artifact.regime, 'score', 0.0) if self._artifact.regime else 0.0,
                    },
                    "sizing_inputs": {
                        "entry_price": fill_price,
                        "stop_level": state.stop_level,
                        "qty": fill_qty,
                        "risk_per_share": state.risk_per_share,
                        "sizing_mult": state.sizing_mult,
                        "base_risk_fraction": self._portfolio.base_risk_fraction,
                        "account_equity": self._portfolio.account_equity,
                    },
                    "signal_factors": self._entry_signal_factors(symbol),
                    "filter_decisions": self._entry_filter_decisions(symbol),
                    "portfolio_state": self._portfolio_state_snapshot(),
                    "session_type": self._current_session_type(event.timestamp),
                    "exchange_timestamp": event.timestamp,
                    "concurrent_positions": len(self._portfolio.open_positions),
                },
                account_id=self._account_id,
            )
        kit = self._instr_kit
        if kit:
            try:
                kit.log_entry(
                    trade_id=position.trade_id or f"IARIC-{symbol}",
                    pair=symbol,
                    side="LONG",
                    entry_price=fill_price,
                    position_size=float(fill_qty),
                    position_size_quote=float(fill_price * fill_qty),
                    entry_signal=f"PB_{state.route_family}",
                    entry_signal_id=event.oms_order_id or symbol,
                    entry_signal_strength=state.intraday_score / 100.0,
                    signal_factors=self._entry_signal_factors(symbol),
                    filter_decisions=self._entry_filter_decisions(symbol),
                    conviction_factors=dict(state.score_components) if getattr(state, 'score_components', None) else None,
                    sizing_inputs={
                        "entry_price": fill_price,
                        "stop_level": state.stop_level,
                        "qty": fill_qty,
                        "risk_per_share": state.risk_per_share,
                        "sizing_mult": state.sizing_mult,
                        "base_risk_fraction": self._portfolio.base_risk_fraction,
                        "account_equity": self._portfolio.account_equity,
                    },
                    exchange_timestamp=event.timestamp,
                    strategy_params={
                        "route_family": state.route_family,
                        "daily_signal_score": state.daily_signal_score,
                        "trigger_tier": state.trigger_tier,
                        "trend_tier": state.trend_tier,
                    },
                    portfolio_state={
                        "account_equity": self._portfolio.account_equity,
                        "open_positions": len(self._portfolio.open_positions),
                        "pending_entry_risk": sum(self._portfolio.pending_entry_risk.values()),
                        "base_risk_fraction": self._portfolio.base_risk_fraction,
                        "regime_allows_no_new_entries": self._portfolio.regime_allows_no_new_entries,
                        "symbols_held": sorted(self._portfolio.open_positions.keys()),
                    },
                    concurrent_positions=len(self._portfolio.open_positions),
                    session_type=self._current_session_type(event.timestamp),
                    **fill_runtime_refs(event.oms_order_id or "", payload, fill_qty=fill_qty),
                )
            except Exception:
                pass

    async def _record_exit_instrumentation(
        self, *, symbol: str, event, fill_price: float, fill_qty: int,
        commission: float, role: str,
        pre_position: PositionState, pre_sym_state: PBSymbolState,
        payload: dict | None = None,
    ) -> None:
        # Compute final values from pre-fill state + this fill's contribution
        exit_qty = min(fill_qty, pre_position.qty_open)
        total_exit_comm = pre_position.exit_commission + commission
        total_fees = pre_position.entry_commission + total_exit_comm
        realized_pnl = pre_position.realized_pnl_usd + (fill_price - pre_position.entry_price) * exit_qty
        net_pnl = realized_pnl - total_fees
        realized_r = net_pnl / max(pre_position.total_initial_risk_usd, 1e-9)
        max_fav = max(pre_position.max_favorable_price, fill_price)
        max_adv = min(pre_position.max_adverse_price, fill_price)
        exit_reason = pre_sym_state.last_transition_reason or role or "EXIT"

        if self._trade_recorder and pre_position.trade_id:
            await self._trade_recorder.record_exit(
                trade_id=pre_position.trade_id,
                exit_price=Decimal(str(fill_price)),
                exit_ts=event.timestamp,
                exit_reason=exit_reason,
                realized_r=Decimal(str(round(realized_r, 4))),
                realized_usd=Decimal(str(round(net_pnl, 2))),
                mfe_r=Decimal(str(round(
                    (max_fav - pre_position.entry_price) / max(pre_position.initial_risk_per_share, 1e-9), 4,
                ))),
                mae_r=Decimal(str(round(
                    (max_adv - pre_position.entry_price) / max(pre_position.initial_risk_per_share, 1e-9), 4,
                ))),
                max_adverse_price=Decimal(str(max_adv)),
                max_favorable_price=Decimal(str(max_fav)),
                meta={
                    "exchange_timestamp": event.timestamp,
                    "route_family": pre_sym_state.route_family,
                    "mfe_stage": pre_sym_state.mfe_stage,
                    "hold_bars": pre_sym_state.hold_bars,
                    "exit_reason_detail": pre_sym_state.last_transition_reason,
                    "fees_paid": total_fees,
                    "hold_days": (event.timestamp.astimezone(ET).date() - pre_position.entry_time.astimezone(ET).date()).days if pre_position.entry_time else 0,
                    "carry_decision_path": pre_sym_state.carry_decision_path,
                    "v2_partial_taken": pre_sym_state.v2_partial_taken,
                    "trail_active": pre_sym_state.trail_active,
                    "breakeven_activated": pre_sym_state.breakeven_activated,
                    "daily_signal_score": pre_sym_state.daily_signal_score,
                    "trigger_tier": pre_sym_state.trigger_tier,
                    "trend_tier": pre_sym_state.trend_tier,
                },
            )
        kit = self._instr_kit
        if kit and pre_position.trade_id:
            try:
                kit.log_exit(
                    trade_id=pre_position.trade_id,
                    exit_price=fill_price,
                    exit_reason=exit_reason,
                    exchange_timestamp=event.timestamp,
                    mfe_r=round(
                        (max_fav - pre_position.entry_price)
                        / max(pre_position.initial_risk_per_share, 1e-9), 4),
                    mae_r=round(
                        (max_adv - pre_position.entry_price)
                        / max(pre_position.initial_risk_per_share, 1e-9), 4),
                    mfe_price=max_fav,
                    mae_price=max_adv,
                    **fill_runtime_refs(event.oms_order_id or "", payload, fill_qty=fill_qty, is_exit=True),
                )
            except Exception:
                pass

    _TERMINAL_STATUS_MAP = {
        OMSEventType.ORDER_CANCELLED: "cancelled",
        OMSEventType.ORDER_EXPIRED: "expired",
        OMSEventType.ORDER_REJECTED: "rejected",
    }

    async def _handle_terminal(self, event) -> None:
        payload = event.payload or {}
        symbol, role = self._resolve_order(event.oms_order_id, payload)
        if not symbol:
            if event.oms_order_id:
                self._order_index.pop(event.oms_order_id, None)
            return
        state = self._symbols.get(symbol)
        if state is None:
            return

        # Build core order update
        status = self._TERMINAL_STATUS_MAP.get(event.event_type, "cancelled")
        update = IARICOrderUpdate(
            oms_order_id=event.oms_order_id or "",
            status=status,
            timestamp=event.timestamp,
            symbol=symbol,
            order_role=role if role in ("ENTRY", "TP", "EXIT", "STOP") else "UNKNOWN",
        )

        # Route through core
        core_state = build_core_runtime_state(self)
        new_state, actions, events = iaric_core_logic.on_order_update(core_state, update)
        apply_core_runtime_state(self, new_state)

        # Dispatch OMS actions
        for action in actions:
            if isinstance(action, SubmitProtectiveStop):
                await self._submit_stop(action.symbol)
            elif isinstance(action, FlattenPosition):
                sym = self._symbols.get(action.symbol)
                if sym and sym.position and sym.position.stop_order_id:
                    await self._cancel_stop(action.symbol)
                await self._submit_market_exit(action.symbol, action.qty, OrderRole.EXIT)

        # Record decision events
        for ev in events:
            self._record_decision(ev.code, ev.details)

    # ── Helpers ─────────────────────────────────────────────────────

    def _resolve_order(self, oms_order_id: str | None, payload: dict[str, Any]) -> tuple[str, str]:
        if oms_order_id and oms_order_id in self._order_index:
            return self._order_index[oms_order_id]
        return str(payload.get("symbol", "")).upper(), str(payload.get("role", ""))

    def _entry_signal_factors(self, symbol: str) -> list[dict]:
        """Build signal_factors list from score components for TA analysis."""
        state = self._symbols.get(symbol)
        if state is None:
            return []
        c = state.score_components or {}
        return [
            {"factor_name": "daily_signal_score", "factor_value": state.daily_signal_score,
             "threshold": self._settings.pb_daily_signal_min_score,
             "contribution": c.get("daily_signal", 0.0) / 100.0},
            {"factor_name": "intraday_score", "factor_value": state.intraday_score,
             "threshold": self._settings.pb_entry_score_min,
             "contribution": state.intraday_score / 100.0},
            {"factor_name": "reclaim", "factor_value": c.get("reclaim", 0.0),
             "threshold": 0.0, "contribution": c.get("reclaim", 0.0) / 8.0},
            {"factor_name": "volume", "factor_value": c.get("volume", 0.0),
             "threshold": 0.0, "contribution": c.get("volume", 0.0) / 12.0},
            {"factor_name": "vwap_hold", "factor_value": c.get("vwap_hold", 0.0),
             "threshold": 0.0, "contribution": c.get("vwap_hold", 0.0) / 5.0},
            {"factor_name": "cpr", "factor_value": c.get("cpr", 0.0),
             "threshold": 0.0, "contribution": c.get("cpr", 0.0) / 6.0},
            {"factor_name": "speed", "factor_value": c.get("speed", 0.0),
             "threshold": 0.0, "contribution": c.get("speed", 0.0) / 8.0},
            {"factor_name": "context", "factor_value": c.get("context", 0.0),
             "threshold": 0.0, "contribution": c.get("context", 0.0) / 100.0},
            {"factor_name": "extension", "factor_value": c.get("extension", 0.0),
             "threshold": 0.0, "contribution": c.get("extension", 0.0) / 100.0},
        ]

    def _entry_filter_decisions(self, symbol: str) -> list[dict]:
        """Build filter_decisions list for TA filter analysis."""
        state = self._symbols.get(symbol)
        item = self._items.get(symbol)
        market = self._markets.get(symbol)
        if state is None or item is None:
            return []
        cfg = self._settings
        current_pos = len(self._portfolio.open_positions) + len(self._portfolio.pending_entry_risk)
        sector_count = self._portfolio.sector_position_count(self._symbol_to_sector, item.sector)
        spread_pct = market.spread_pct if market else 0.0
        decisions = [
            {"filter_name": "max_positions", "threshold": cfg.pb_max_positions,
             "actual_value": current_pos, "passed": current_pos < cfg.pb_max_positions},
            {"filter_name": "sector_limit", "threshold": cfg.max_positions_per_sector,
             "actual_value": sector_count, "passed": sector_count < cfg.max_positions_per_sector},
            {"filter_name": "spread_gate", "threshold": round(cfg.max_median_spread_pct * 2.0, 4),
             "actual_value": round(spread_pct, 4), "passed": spread_pct <= cfg.max_median_spread_pct * 2.0},
            {"filter_name": "regime_gate", "threshold": True,
             "actual_value": not self._portfolio.regime_allows_no_new_entries,
             "passed": not self._portfolio.regime_allows_no_new_entries},
        ]
        if state.intraday_score > 0:
            decisions.append(
                {"filter_name": "entry_score", "threshold": cfg.pb_entry_score_min,
                 "actual_value": state.intraday_score, "passed": state.intraday_score >= cfg.pb_entry_score_min})
        if state.route_family:
            decisions.append(
                {"filter_name": "stopped_out_today", "threshold": False,
                 "actual_value": state.stopped_out_today, "passed": not state.stopped_out_today})
        return decisions

    def _portfolio_state_snapshot(self) -> dict:
        """Snapshot portfolio state for TA enrichment."""
        return {
            "open_positions": len(self._portfolio.open_positions),
            "pending_entries": len(self._portfolio.pending_entry_risk),
            "account_equity": self._portfolio.account_equity,
            "base_risk_fraction": self._portfolio.base_risk_fraction,
            "sectors_in_use": sorted(set(
                self._symbol_to_sector.get(s, "") for s in self._portfolio.open_positions
            )),
        }

    def _log_missed(self, *, symbol: str, blocked_by: str, block_reason: str,
                    exchange_timestamp: datetime, route: str = "") -> None:
        """Fire-and-forget missed opportunity via Kit."""
        kit = self._instr_kit
        if kit is None:
            return
        state = self._symbols.get(symbol)
        try:
            kit.log_missed(
                pair=symbol, side="LONG",
                signal=f"iaric_pb_{route.lower()}" if route else "iaric_pb_entry",
                signal_id=f"{symbol}:{blocked_by}:{int(exchange_timestamp.timestamp())}",
                signal_strength=state.intraday_score / 100.0 if state else 0.0,
                blocked_by=blocked_by, block_reason=block_reason,
                strategy_params={
                    "route_family": state.route_family if state else "",
                    "daily_signal_score": state.daily_signal_score if state else 0.0,
                    "trigger_tier": state.trigger_tier if state else "",
                    "trend_tier": state.trend_tier if state else "",
                    "bars_seen_today": state.bars_seen_today if state else 0,
                },
                filter_decisions=self._entry_filter_decisions(symbol),
                concurrent_positions=len(self._portfolio.open_positions),
                exchange_timestamp=exchange_timestamp,
            )
        except Exception:
            pass

    def _current_session_type(self, now: datetime) -> str:
        et_now = now.astimezone(ET).time()
        if et_now < self._settings.market_open:
            return "PREMARKET"
        if et_now >= self._settings.close_block_start:
            return "LATE_DAY"
        return "RTH"

    async def _save_state(self, reason: str) -> None:
        persist_intraday_state(self.snapshot_state(), settings=self._settings)
        self._last_save_ts = datetime.now(timezone.utc)
        self._diagnostics.log_decision("STATE_SAVE", {"reason": reason})

    def _restore_order_state(self, symbol: str, state: PBSymbolState) -> None:
        if state.entry_order is not None:
            self._order_index[state.entry_order.oms_order_id] = (symbol, "ENTRY")
        if state.exit_order is not None:
            self._order_index[state.exit_order.oms_order_id] = (symbol, state.exit_order.role)
        if state.position is not None and state.position.stop_order_id:
            self._order_index[state.position.stop_order_id] = (symbol, "STOP")

    @classmethod
    def try_load_state(cls, trade_date, settings: StrategySettings | None = None) -> IntradayStateSnapshot | None:
        try:
            return load_intraday_state(trade_date, settings=settings or StrategySettings())
        except FileNotFoundError:
            return None
