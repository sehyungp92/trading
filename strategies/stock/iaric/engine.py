"""Async state-machine engine for IARIC."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from statistics import fmean
from typing import Any

from libs.oms.models.events import OMSEventType
from libs.oms.models.intent import Intent, IntentType
from libs.oms.models.order import OrderRole

from .artifact_store import IntradayStateSnapshot, load_intraday_state, persist_intraday_state
from .config import ET, PROXY_SYMBOLS, STRATEGY_ID, StrategySettings, build_proxy_instruments
from .data import CanonicalBarBuilder
from .diagnostics import JsonlDiagnostics
from .execution import build_entry_order, build_market_exit, build_position_from_fill, build_stock_instrument, build_stop_order
from .exits import carry_eligible, classify_trade, flow_reversal_exit_due, should_exit_for_avwap_breakdown, should_exit_for_time_stop, should_take_partial
from .models import AVWAPLedger, Bar, MarketSnapshot, PendingOrderState, PortfolioState, PositionState, QuoteSnapshot, SymbolIntradayState, VWAPLedger, WatchlistArtifact
from .risk import adjust_qty_for_portfolio_constraints, compute_final_risk_unit, compute_order_quantity, timing_gate_allows_entry
from .signals import alpha_step, compute_flowproxy_signal, compute_location_grade, compute_micropressure_from_ticks, compute_micropressure_proxy, resolve_confidence, update_symbol_tier

logger = logging.getLogger(__name__)


class IARICEngine:
    """Live IARIC engine kept thin around pure helpers."""

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
    ) -> None:
        self._oms = oms_service
        self._artifact = artifact
        self._items = artifact.by_symbol
        self._account_id = account_id
        self._settings = settings or StrategySettings()
        self._trade_recorder = trade_recorder
        self._diagnostics = diagnostics or JsonlDiagnostics(self._settings.diagnostics_dir, enabled=False)
        self._instrumentation = instrumentation

        self._symbols: dict[str, SymbolIntradayState] = {}
        self._markets: dict[str, MarketSnapshot] = {}
        self._session_vwap: dict[str, VWAPLedger] = {}
        self._avwap: dict[str, AVWAPLedger] = {}
        self._bar_builder = CanonicalBarBuilder()
        self._portfolio = PortfolioState(account_equity=nav, base_risk_fraction=self._settings.base_risk_fraction)
        self._symbol_to_sector = {item.symbol: item.sector for item in artifact.items}
        self._active_symbols: set[str] = set()
        self._order_index: dict[str, tuple[str, str]] = {}
        self._flow_reversal_flags = {held.symbol: held.flow_reversal_flag for held in artifact.held_positions}
        self._market_wide_institutional_selling = artifact.market_wide_institutional_selling
        self._expected_stop_cancels: set[str] = set()
        self._last_quote_volume: dict[str, float] = {}
        self._last_save_ts: datetime | None = None

        self._event_queue = None
        self._event_task: asyncio.Task | None = None
        self._pulse_task: asyncio.Task | None = None
        self._running = False

        self._initialize_from_artifact()

    def _initialize_from_artifact(self) -> None:
        ranked_symbols = [item.symbol for item in self._artifact.items]
        self._active_symbols = set(ranked_symbols[: self._settings.active_monitoring_target])
        initial_hot = set(ranked_symbols[: min(5, len(ranked_symbols), self._settings.hot_max)])
        for item in self._artifact.items:
            symbol = item.symbol
            self._symbols[symbol] = SymbolIntradayState(
                symbol=symbol,
                tier="HOT" if symbol in initial_hot else "WARM" if symbol in self._active_symbols else "COLD",
                sponsorship_signal=item.sponsorship_state,
                average_30m_volume=item.average_30m_volume,
                expected_volume_pct=item.expected_5m_volume,
            )
            self._markets[symbol] = MarketSnapshot(symbol=symbol)
            self._session_vwap[symbol] = VWAPLedger()
            self._avwap[symbol] = AVWAPLedger.bootstrap(item.avwap_ref)

        for held in self._artifact.held_positions:
            sym = self._symbols.setdefault(held.symbol, SymbolIntradayState(symbol=held.symbol, tier="HOT"))
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
                setup_tag=held.setup_tag or "UNCLASSIFIED",
                time_stop_deadline=held.time_stop_deadline,
            )
            sym.position = position
            sym.in_position = True
            sym.position_qty = held.size
            sym.avg_price = held.entry_price
            sym.fsm_state = "IN_POSITION"
            sym.setup_tag = held.setup_tag
            self._portfolio.open_positions[held.symbol] = position
            self._active_symbols.add(held.symbol)

    @staticmethod
    def _log_task_exception(task: asyncio.Task) -> None:
        """Done-callback for fire-and-forget tasks so exceptions are never silently lost."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Unhandled exception in background task: %s", exc, exc_info=exc)

    async def _reconcile_after_reconnect(self) -> None:
        """Re-sync OMS state after an IB Gateway reconnection."""
        logger.warning("IB reconnected — triggering OMS reconciliation")
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
        self._pulse_task = asyncio.create_task(self._pulse_loop())

    async def stop(self) -> None:
        self._running = False
        await self._save_state("stop")
        for task in (self._pulse_task, self._event_task):
            if task is None:
                continue
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    def hydrate_state(self, snapshot: IntradayStateSnapshot) -> None:
        self._active_symbols = set(snapshot.meta.get("active_symbols", self._active_symbols))
        for stored in snapshot.symbols:
            current = self._symbols.get(stored.symbol)
            if current is None:
                item = self._items.get(stored.symbol)
                avwap_ref = stored.avwap_live if stored.avwap_live is not None else (item.avwap_ref if item is not None else 0.0)
                if stored.entry_order is None and stored.active_order_id == "SUBMITTING_ENTRY":
                    stored.active_order_id = None
                self._symbols[stored.symbol] = stored
                self._markets.setdefault(stored.symbol, MarketSnapshot(symbol=stored.symbol))
                self._session_vwap.setdefault(stored.symbol, VWAPLedger())
                self._avwap.setdefault(stored.symbol, AVWAPLedger.bootstrap(avwap_ref))
                self._restore_order_state(stored.symbol, stored)
                continue
            current.tier = stored.tier
            current.fsm_state = stored.fsm_state
            current.in_position = stored.in_position
            current.position_qty = stored.position_qty
            current.avg_price = stored.avg_price
            current.setup_type = stored.setup_type
            current.setup_low = stored.setup_low
            current.reclaim_level = stored.reclaim_level
            current.stop_level = stored.stop_level
            current.setup_time = stored.setup_time
            current.invalidated_at = stored.invalidated_at
            current.acceptance_count = stored.acceptance_count
            current.required_acceptance_count = stored.required_acceptance_count
            current.location_grade = stored.location_grade
            current.session_vwap = stored.session_vwap
            current.avwap_live = stored.avwap_live
            current.sponsorship_signal = stored.sponsorship_signal
            current.micropressure_signal = stored.micropressure_signal
            current.micropressure_mode = stored.micropressure_mode
            current.flowproxy_signal = stored.flowproxy_signal
            current.confidence = stored.confidence
            current.last_1m_bar_time = stored.last_1m_bar_time
            current.last_5m_bar_time = stored.last_5m_bar_time
            current.active_order_id = None if stored.entry_order is None and stored.active_order_id == "SUBMITTING_ENTRY" else stored.active_order_id
            current.time_stop_deadline = stored.time_stop_deadline
            current.setup_tag = stored.setup_tag
            current.expected_volume_pct = stored.expected_volume_pct
            current.average_30m_volume = stored.average_30m_volume
            current.last_transition_reason = stored.last_transition_reason
            current.entry_order = stored.entry_order
            current.position = stored.position
            current.exit_order = stored.exit_order
            current.pending_hard_exit = stored.pending_hard_exit
            if stored.position is not None:
                self._portfolio.open_positions[stored.symbol] = stored.position
            self._restore_order_state(stored.symbol, current)

    def snapshot_state(self) -> IntradayStateSnapshot:
        return IntradayStateSnapshot(
            trade_date=self._artifact.trade_date,
            saved_at=datetime.now(timezone.utc),
            symbols=list(self._symbols.values()),
            last_decision_code="snapshot",
            meta={"active_symbols": sorted(self._active_symbols)},
        )

    def subscription_instruments(self) -> list:
        instruments = build_proxy_instruments()
        seen = {instrument.symbol for instrument in instruments}
        for symbol in sorted(self._active_symbols):
            state = self._symbols.get(symbol)
            if state is None or (state.tier != "HOT" and not state.in_position):
                continue
            item = self._items.get(symbol)
            if item and symbol not in seen:
                instruments.append(build_stock_instrument(item))
                seen.add(symbol)
        return instruments

    def polling_instruments(self) -> list[tuple[Any, int]]:
        requests: list[tuple[Any, int]] = []
        for symbol, item in self._items.items():
            state = self._symbols.get(symbol)
            if state is None or state.in_position or state.tier == "HOT":
                continue
            if not item.tradable_flag and symbol not in self._active_symbols:
                continue
            interval = self._settings.warm_poll_interval_s if state.tier == "WARM" or symbol in self._active_symbols else self._settings.cold_poll_interval_s
            requests.append((build_stock_instrument(item), interval))
        return requests

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
        previous_volume = self._last_quote_volume.get(normalized, quote.cumulative_volume)
        volume_delta = max(0.0, quote.cumulative_volume - previous_volume)
        self._last_quote_volume[normalized] = quote.cumulative_volume
        midpoint = ((quote.bid + quote.ask) / 2.0) if quote.bid > 0 and quote.ask > 0 else quote.last
        signed = quote.last * volume_delta
        if quote.last < midpoint:
            signed *= -1.0
        if volume_delta > 0:
            market.tick_pressure_window.append((quote.ts, signed))

    def on_bar(self, symbol: str, bar: Bar) -> None:
        normalized = symbol.upper()
        market = self._markets.get(normalized)
        item = self._items.get(normalized)
        if market is None or item is None:
            return
        if bar.start_time.astimezone(ET).date() != self._artifact.trade_date:
            return
        if market.last_1m_bar is not None and market.last_1m_bar.start_time >= bar.start_time:
            return

        self._bar_builder.ingest_bar(bar)
        market.minute_bars.append(bar)
        market.last_1m_bar = bar
        market.last_price = bar.close
        market.session_high = max(market.session_high or bar.high, bar.high)
        market.session_low = min(market.session_low or bar.low, bar.low)
        self._session_vwap[normalized].update(bar)
        self._avwap[normalized].update(bar)
        market.session_vwap = self._session_vwap[normalized].value
        market.avwap_live = self._avwap[normalized].value
        self._symbols[normalized].session_vwap = market.session_vwap
        self._symbols[normalized].avwap_live = market.avwap_live
        self._symbols[normalized].last_1m_bar_time = bar.end_time
        self._diagnostics.log_ledger(normalized, market.session_vwap, market.avwap_live)

        self._process_alpha(normalized, bar.end_time, bar_1m=bar)
        for bar_5m in self._bar_builder.aggregate_new_bars(normalized, 5):
            market.last_5m_bar = bar_5m
            market.bars_5m.append(bar_5m)
            self._process_signals(normalized, bar_5m)
            self._process_alpha(normalized, bar_5m.end_time, bar_1m=bar, bar_5m=bar_5m)
        for bar_30m in self._bar_builder.aggregate_new_bars(normalized, 30):
            market.last_30m_bar = bar_30m
            market.bars_30m.append(bar_30m)
            self._symbols[normalized].average_30m_volume = fmean(sample.volume for sample in list(market.bars_30m)[-10:])
            self._manage_position(normalized, bar_30m.end_time)

        self._manage_position(normalized, bar.end_time)

    def get_position_snapshot(self) -> list[dict[str, Any]]:
        snapshots = []
        for symbol, state in self._symbols.items():
            market = self._markets.get(symbol)
            if state.position is None or market is None or market.last_price is None:
                continue
            unrealized_r = (market.last_price - state.position.entry_price) / max(state.position.initial_risk_per_share, 1e-9)
            snapshots.append(
                {
                    "strategy_type": "strategy_iaric",
                    "symbol": symbol,
                    "direction": "LONG",
                    "entry_price": state.position.entry_price,
                    "qty": state.position.qty_open,
                    "unrealized_pnl_r": round(unrealized_r, 3),
                }
            )
        return snapshots

    def open_order_count(self) -> int:
        return len(self._order_index)

    def _confidence_score(self, confidence: str | None) -> float:
        mapping = {
            "RED": 0.0,
            "YELLOW": 0.5,
            "GREEN": 1.0,
        }
        return mapping.get((confidence or "").upper(), 0.0)

    def _current_session_type(self, now: datetime) -> str:
        et_now = now.astimezone(ET).time()
        if et_now < self._settings.market_open:
            return "PREMARKET"
        if et_now >= self._settings.close_block_start:
            return "LATE_DAY"
        return "RTH"

    def _portfolio_state_snapshot(self) -> dict[str, Any]:
        return {
            "open_positions": len(self._portfolio.open_positions),
            "pending_entries": len(self._portfolio.pending_entry_risk),
            "active_symbols": len(self._active_symbols),
            "regime_allows_no_new_entries": self._portfolio.regime_allows_no_new_entries,
            "sectors_in_use": sorted(
                {
                    self._symbol_to_sector.get(symbol, "")
                    for symbol in self._portfolio.open_positions
                    if self._symbol_to_sector.get(symbol)
                }
            ),
        }

    def _entry_filter_decisions(
        self,
        *,
        state: SymbolIntradayState,
        market: MarketSnapshot,
        timing_ok: bool,
    ) -> list[dict[str, Any]]:
        return [
            {
                "filter_name": "confidence_gate",
                "threshold": 0.5,
                "actual_value": self._confidence_score(state.confidence),
                "passed": state.confidence != "RED",
            },
            {
                "filter_name": "timing_gate",
                "threshold": 1.0,
                "actual_value": 1.0 if timing_ok else 0.0,
                "passed": timing_ok,
            },
            {
                "filter_name": "spread_gate",
                "threshold": self._settings.max_median_spread_pct * 2.0,
                "actual_value": market.spread_pct,
                "passed": market.spread_pct <= (self._settings.max_median_spread_pct * 2.0),
            },
        ]

    def _entry_signal_factors(self, state: SymbolIntradayState) -> list[dict[str, Any]]:
        grade_scores = {"A": 1.0, "B": 0.6, "C": 0.25}
        label_scores = {
            "STRONG": 1.0,
            "ACCUMULATE": 1.0,
            "RECLAIM": 1.0,
            "NEUTRAL": 0.0,
            "HOLD": 0.0,
            "WEAK": -0.5,
            "DISTRIBUTE": -1.0,
            "BREAKDOWN": -1.0,
        }
        factors: list[dict[str, Any]] = []

        location_value = grade_scores.get((state.location_grade or "").upper(), 0.0)
        factors.append(
            {
                "factor_name": "location_grade",
                "factor_value": location_value,
                "threshold": 0.6,
                "contribution": location_value,
            }
        )

        for factor_name, raw_value in (
            ("sponsorship_signal", state.sponsorship_signal),
            ("micropressure_signal", state.micropressure_signal),
            ("flowproxy_signal", state.flowproxy_signal),
        ):
            normalized = str(raw_value or "").upper()
            factors.append(
                {
                    "factor_name": factor_name,
                    "factor_value": label_scores.get(normalized, 0.0),
                    "threshold": 0.0,
                    "contribution": label_scores.get(normalized, 0.0),
                }
            )

        return factors

    def _log_indicator_snapshot(self, symbol: str, bar: Bar) -> None:
        if self._instrumentation is None:
            return
        state = self._symbols[symbol]
        market = self._markets[symbol]
        try:
            self._instrumentation.on_indicator_snapshot(
                pair=symbol,
                indicators={
                    "spread_pct": float(market.spread_pct or 0.0),
                    "atr_5m_pct": float(self._atr_5m_pct(symbol)),
                    "session_vwap": float(market.session_vwap or 0.0),
                    "avwap_live": float(market.avwap_live or 0.0),
                    "average_30m_volume": float(state.average_30m_volume or 0.0),
                    "expected_volume_pct": float(state.expected_volume_pct or 0.0),
                },
                signal_name=state.setup_type or "iaric_signal",
                signal_strength=self._confidence_score(state.confidence),
                decision=state.fsm_state,
                strategy_type="strategy_iaric",
                exchange_timestamp=bar.end_time,
                bar_id=bar.end_time.isoformat(),
                context={
                    "location_grade": state.location_grade,
                    "micropressure_signal": state.micropressure_signal,
                    "micropressure_mode": state.micropressure_mode,
                    "flowproxy_signal": state.flowproxy_signal,
                },
            )
        except Exception:
            pass

    def _log_orderbook_context(
        self,
        *,
        symbol: str,
        trade_context: str,
        related_trade_id: str = "",
        exchange_timestamp: datetime | None = None,
    ) -> None:
        if self._instrumentation is None:
            return
        market = self._markets.get(symbol)
        if market is None:
            return
        quote = market.last_quote
        best_bid = float(market.bid or 0.0)
        best_ask = float(market.ask or 0.0)
        if best_bid <= 0 and market.last_price is not None:
            best_bid = float(market.last_price)
        if best_ask <= 0 and market.last_price is not None:
            best_ask = float(market.last_price)
        if best_bid <= 0 or best_ask <= 0:
            return
        bid_depth = float(getattr(quote, "bid_size", 0.0) or 0.0)
        ask_depth = float(getattr(quote, "ask_size", 0.0) or 0.0)
        bid_levels = [{"price": best_bid, "size": bid_depth}] if bid_depth > 0 else None
        ask_levels = [{"price": best_ask, "size": ask_depth}] if ask_depth > 0 else None
        try:
            self._instrumentation.on_orderbook_context(
                pair=symbol,
                best_bid=best_bid,
                best_ask=best_ask,
                trade_context=trade_context,
                related_trade_id=related_trade_id or None,
                bid_depth_10bps=bid_depth,
                ask_depth_10bps=ask_depth,
                bid_levels=bid_levels,
                ask_levels=ask_levels,
                exchange_timestamp=exchange_timestamp,
            )
        except Exception:
            pass

    def _log_missed(
        self,
        *,
        symbol: str,
        blocked_by: str,
        block_reason: str,
        signal: str,
        signal_strength: float | None = None,
        exchange_timestamp: datetime | None = None,
        filter_decisions: list[dict[str, Any]] | None = None,
        strategy_params: dict[str, Any] | None = None,
    ) -> None:
        if self._instrumentation is None:
            return
        state = self._symbols[symbol]
        try:
            self._instrumentation.log_missed(
                pair=symbol,
                side="LONG",
                signal=signal,
                signal_id=f"{symbol}:{blocked_by}:{int((exchange_timestamp or datetime.now(timezone.utc)).timestamp())}",
                signal_strength=signal_strength if signal_strength is not None else self._confidence_score(state.confidence),
                blocked_by=blocked_by,
                block_reason=block_reason,
                strategy_params={
                    "setup_type": state.setup_type,
                    "setup_tag": state.setup_tag,
                    "location_grade": state.location_grade,
                    "reclaim_level": state.reclaim_level,
                    "stop_level": state.stop_level,
                    **(strategy_params or {}),
                },
                filter_decisions=filter_decisions,
                session_type=self._current_session_type(exchange_timestamp or datetime.now(timezone.utc)),
                concurrent_positions=len(self._portfolio.open_positions),
                exchange_timestamp=exchange_timestamp,
            )
        except Exception:
            pass

    def _log_order_event(
        self,
        *,
        order_id: str,
        symbol: str,
        status: str,
        requested_qty: int,
        order_type: str,
        related_trade_id: str = "",
        requested_price: float | None = None,
        fill_price: float | None = None,
        reject_reason: str = "",
        exchange_timestamp: datetime | None = None,
    ) -> None:
        if self._instrumentation is None:
            return
        try:
            self._instrumentation.on_order_event(
                order_id=order_id,
                pair=symbol,
                side="SELL" if status in {"CANCEL_REQUESTED", "REPLACE_REQUESTED"} or order_type in {"STOP", "MARKET_EXIT"} else "BUY",
                order_type=order_type,
                status=status,
                requested_qty=requested_qty,
                requested_price=requested_price,
                fill_price=fill_price,
                reject_reason=reject_reason,
                related_trade_id=related_trade_id,
                strategy_type="strategy_iaric",
                session=self._current_session_type(exchange_timestamp or datetime.now(timezone.utc)),
                exchange_timestamp=exchange_timestamp,
            )
        except Exception:
            pass

    async def advance(self, now: datetime) -> None:
        await self._refresh_portfolio()
        self._rebalance_tiers()
        for symbol in self._symbols:
            self._manage_position(symbol, now)
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

    def _rebalance_tiers(self) -> None:
        desired_hot: list[str] = []
        desired_warm: list[str] = []
        for symbol, item in self._items.items():
            state = self._symbols[symbol]
            market = self._markets[symbol]
            target = "COLD" if symbol not in self._active_symbols and not state.in_position else update_symbol_tier(item, state, market, self._settings)
            if target == "HOT":
                desired_hot.append(symbol)
            elif target == "WARM":
                desired_warm.append(symbol)

        protected = [s for s in desired_hot if self._symbols[s].in_position or self._symbols[s].fsm_state in {"SETUP_DETECTED", "ACCEPTING", "IN_POSITION"}]
        remaining = [s for s in desired_hot if s not in protected]
        allowed_hot = set(protected + remaining[: max(0, self._settings.hot_max - len(protected))])
        for symbol, state in self._symbols.items():
            old_tier = state.tier
            state.tier = "HOT" if symbol in allowed_hot else "WARM" if symbol in desired_warm else "COLD"
            if old_tier != state.tier:
                self._diagnostics.log_tier_change(symbol, old_tier, state.tier, "rebalance")

        ranked = [item.symbol for item in self._artifact.items]
        protected_symbols = [symbol for symbol, state in self._symbols.items() if state.in_position or state.fsm_state in {"SETUP_DETECTED", "ACCEPTING"}]
        refreshed = []
        for symbol in ranked:
            if symbol in refreshed or symbol in protected_symbols:
                continue
            refreshed.append(symbol)
            if len(refreshed) >= self._settings.active_monitoring_target - len(protected_symbols):
                break
        self._active_symbols = set(protected_symbols + refreshed)

    def _atr_5m_pct(self, symbol: str) -> float:
        market = self._markets[symbol]
        bars = list(market.bars_5m)[-14:]
        if len(bars) < 2:
            return self._items[symbol].intraday_atr_seed
        true_ranges: list[float] = []
        prev_close = bars[0].close
        for bar in bars[1:]:
            true_ranges.append(max(bar.high - bar.low, abs(bar.high - prev_close), abs(bar.low - prev_close)))
            prev_close = bar.close
        atr = fmean(true_ranges) if true_ranges else 0.0
        return atr / max(bars[-1].close, 1e-9)

    def _process_signals(self, symbol: str, bar_5m: Bar) -> None:
        state = self._symbols[symbol]
        item = self._items[symbol]
        market = self._markets[symbol]
        state.sponsorship_signal = item.sponsorship_state
        if state.tier == "HOT" and market.tick_pressure_window:
            state.micropressure_mode = "TICK"
            state.micropressure_signal = compute_micropressure_from_ticks(market.tick_pressure_window)
        else:
            state.micropressure_mode = "PROXY"
            state.micropressure_signal = compute_micropressure_proxy(
                bar_5m=bar_5m,
                expected_volume=max(item.expected_5m_volume, item.average_30m_volume / 6.0, 1.0),
                median20_volume=max(item.average_30m_volume, item.expected_5m_volume * 6.0, 1.0),
                reclaim_level=state.reclaim_level or bar_5m.close,
            )
        state.flowproxy_signal = compute_flowproxy_signal(None)
        state.location_grade = compute_location_grade(item, market)
        state.confidence = resolve_confidence(state)
        self._log_indicator_snapshot(symbol, bar_5m)

    def _process_alpha(self, symbol: str, now: datetime, bar_1m: Bar | None = None, bar_5m: Bar | None = None) -> None:
        state = self._symbols[symbol]
        item = self._items[symbol]
        market = self._markets[symbol]
        if not item.tradable_flag and not state.in_position:
            return
        if state.tier == "HOT" and market.tick_pressure_window:
            state.micropressure_mode = "TICK"
        elif state.fsm_state != "IN_POSITION":
            state.micropressure_mode = "PROXY"
        action, adders = alpha_step(
            item=item,
            sym=state,
            market=market,
            bar_1m=bar_1m,
            bar_5m=bar_5m,
            now=now,
            atr_5m_pct=self._atr_5m_pct(symbol),
            settings=self._settings,
            market_wide_institutional_selling=self._market_wide_institutional_selling,
        )
        if action == "SETUP_DETECTED":
            self._diagnostics.log_setup(symbol, state.setup_type or "", state.location_grade or "B", state.reclaim_level or 0.0, state.stop_level or 0.0)
        elif action == "MOVE_TO_ACCEPTING":
            self._diagnostics.log_acceptance(symbol, state.acceptance_count, state.required_acceptance_count, adders, state.confidence)
        elif action == "READY_TO_ENTER":
            if state.entry_order is None and state.position is None and state.active_order_id is None:
                state.active_order_id = "SUBMITTING_ENTRY"
                asyncio.create_task(self._submit_entry(symbol, now)).add_done_callback(self._log_task_exception)

    async def _submit_entry(self, symbol: str, now: datetime) -> None:
        state = self._symbols[symbol]
        item = self._items[symbol]
        market = self._markets[symbol]
        timing_ok = timing_gate_allows_entry(now, self._settings)
        filter_decisions = self._entry_filter_decisions(state=state, market=market, timing_ok=timing_ok)
        if state.fsm_state != "ACCEPTING" or state.entry_order is not None or state.position is not None:
            if state.active_order_id == "SUBMITTING_ENTRY":
                state.active_order_id = None
            return
        if state.active_order_id not in (None, "SUBMITTING_ENTRY"):
            return
        if state.confidence == "RED" or not timing_ok:
            self._log_missed(
                symbol=symbol,
                blocked_by="entry_gate",
                block_reason="confidence_red" if state.confidence == "RED" else "timing_gate",
                signal=state.setup_type or "iaric_entry",
                exchange_timestamp=now,
                filter_decisions=filter_decisions,
            )
            if state.active_order_id == "SUBMITTING_ENTRY":
                state.active_order_id = None
            return
        if market.last_price is None or state.stop_level is None:
            if state.active_order_id == "SUBMITTING_ENTRY":
                state.active_order_id = None
            return
        entry_price = market.ask if market.ask > 0 else market.last_price + item.tick_size
        if market.spread_pct > (self._settings.max_median_spread_pct * 2.0):
            self._diagnostics.log_decision("ENTRY_BLOCKED", {"symbol": symbol, "reason": "live_spread_wide"})
            self._log_missed(
                symbol=symbol,
                blocked_by="spread_guard",
                block_reason="live_spread_wide",
                signal=state.setup_type or "iaric_entry",
                exchange_timestamp=now,
                filter_decisions=filter_decisions,
                strategy_params={"entry_price": entry_price},
            )
            if state.active_order_id == "SUBMITTING_ENTRY":
                state.active_order_id = None
            return
        final_risk_unit = compute_final_risk_unit(item, state, now, self._settings)
        qty = compute_order_quantity(
            account_equity=self._portfolio.account_equity,
            base_risk_fraction=self._portfolio.base_risk_fraction,
            final_risk_unit=final_risk_unit,
            entry_price=entry_price,
            stop_level=state.stop_level,
        )
        qty, reason = adjust_qty_for_portfolio_constraints(
            portfolio=self._portfolio,
            item=item,
            intended_qty=qty,
            entry_price=entry_price,
            stop_level=state.stop_level,
            symbol_to_sector=self._symbol_to_sector,
            settings=self._settings,
        )
        if qty <= 0:
            self._diagnostics.log_decision("ENTRY_BLOCKED", {"symbol": symbol, "reason": reason})
            self._log_missed(
                symbol=symbol,
                blocked_by="portfolio_constraints",
                block_reason=reason,
                signal=state.setup_type or "iaric_entry",
                exchange_timestamp=now,
                filter_decisions=filter_decisions,
                strategy_params={
                    "entry_price": entry_price,
                    "final_risk_unit": final_risk_unit,
                },
            )
            if state.active_order_id == "SUBMITTING_ENTRY":
                state.active_order_id = None
            return
        order = build_entry_order(item, self._account_id, qty, entry_price, state.stop_level)
        receipt = await self._oms.submit_intent(Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=order))
        if receipt.oms_order_id:
            state.entry_order = PendingOrderState(
                oms_order_id=receipt.oms_order_id,
                submitted_at=now,
                role="ENTRY",
                requested_qty=qty,
                limit_price=entry_price,
            )
            state.active_order_id = receipt.oms_order_id
            state.fsm_state = "IN_POSITION_PENDING"
            self._portfolio.pending_entry_risk[symbol] = qty * max(entry_price - state.stop_level, 0.01)
            self._order_index[receipt.oms_order_id] = (symbol, "ENTRY")
            self._diagnostics.log_order(symbol, "submit_entry", {"qty": qty, "limit_price": entry_price})
            self._log_order_event(
                order_id=receipt.oms_order_id,
                symbol=symbol,
                status="SUBMITTED",
                requested_qty=qty,
                order_type="LIMIT_ENTRY",
                requested_price=entry_price,
                exchange_timestamp=now,
            )
            self._log_orderbook_context(
                symbol=symbol,
                trade_context="entry",
                exchange_timestamp=now,
            )
            return
        self._log_missed(
            symbol=symbol,
            blocked_by="oms_submit",
            block_reason="receipt_missing_order_id",
            signal=state.setup_type or "iaric_entry",
            exchange_timestamp=now,
            filter_decisions=filter_decisions,
            strategy_params={
                "entry_price": entry_price,
                "qty": qty,
            },
        )
        if state.active_order_id == "SUBMITTING_ENTRY":
            state.active_order_id = None

    async def _submit_stop(self, symbol: str) -> None:
        state = self._symbols[symbol]
        item = self._items[symbol]
        if state.position is None or state.position.qty_open <= 0 or state.position.stop_order_id:
            return
        try:
            order = build_stop_order(item, self._account_id, state.position.qty_open, state.position.current_stop)
            receipt = await self._oms.submit_intent(Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=order))
            if receipt.oms_order_id:
                state.position.stop_order_id = receipt.oms_order_id
                self._order_index[receipt.oms_order_id] = (symbol, "STOP")
                self._diagnostics.log_order(symbol, "submit_stop", {"qty": state.position.qty_open, "stop_price": state.position.current_stop})
                self._log_order_event(
                    order_id=receipt.oms_order_id,
                    symbol=symbol,
                    status="SUBMITTED",
                    requested_qty=state.position.qty_open,
                    order_type="STOP",
                    requested_price=state.position.current_stop,
                )
        except Exception as exc:
            logger.error("submit_stop failed for %s: %s", symbol, exc, exc_info=exc)
            if self._instrumentation:
                try:
                    self._instrumentation.log_error(
                        error_type="submit_stop_failed",
                        message=str(exc),
                        severity="high",
                        category="engine",
                        context={"symbol": symbol},
                        exc=exc,
                    )
                except Exception:
                    pass

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
            self._log_order_event(
                order_id=state.position.stop_order_id,
                symbol=symbol,
                status="REPLACE_REQUESTED",
                requested_qty=state.position.qty_open,
                order_type="STOP",
                requested_price=state.position.current_stop,
            )
        except Exception as exc:
            logger.error("replace_stop failed for %s: %s", symbol, exc, exc_info=exc)
            if self._instrumentation:
                try:
                    self._instrumentation.log_error(
                        error_type="replace_stop_failed",
                        message=str(exc),
                        severity="high",
                        category="engine",
                        context={"symbol": symbol},
                        exc=exc,
                    )
                except Exception:
                    pass

    async def _cancel_stop(self, symbol: str) -> None:
        state = self._symbols[symbol]
        if state.position and state.position.stop_order_id:
            self._expected_stop_cancels.add(state.position.stop_order_id)
            await self._cancel_order(state.position.stop_order_id)

    async def _submit_market_exit(self, symbol: str, qty: int, role: OrderRole) -> None:
        state = self._symbols[symbol]
        item = self._items[symbol]
        market = self._markets.get(symbol)
        position = state.position
        if position is None or qty <= 0 or state.exit_order is not None:
            return
        requested_qty = min(qty, position.qty_open)
        if requested_qty <= 0:
            return
        expected_exit_price = 0.0
        if market is not None:
            expected_exit_price = float(market.bid or market.last_price or 0.0)
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
                self._log_order_event(
                    order_id=receipt.oms_order_id,
                    symbol=symbol,
                    status="SUBMITTED",
                    requested_qty=requested_qty,
                    order_type="MARKET_EXIT",
                    requested_price=expected_exit_price if expected_exit_price > 0 else None,
                    related_trade_id=position.trade_id if position else "",
                )
                self._log_orderbook_context(
                    symbol=symbol,
                    trade_context="exit",
                    related_trade_id=position.trade_id if position else "",
                )
        except Exception as exc:
            logger.error("submit_market_exit failed for %s: %s", symbol, exc, exc_info=exc)
            if self._instrumentation:
                try:
                    self._instrumentation.log_error(
                        error_type="submit_market_exit_failed",
                        message=str(exc),
                        severity="high",
                        category="engine",
                        context={"symbol": symbol, "qty": requested_qty, "role": role.value},
                        exc=exc,
                    )
                except Exception:
                    pass

    async def _cancel_order(self, oms_order_id: str) -> None:
        symbol, role = self._resolve_order(oms_order_id, {})
        requested_qty = 0
        requested_price = None
        if symbol:
            state = self._symbols.get(symbol)
            if role == "ENTRY" and state and state.entry_order:
                requested_qty = state.entry_order.requested_qty
                requested_price = state.entry_order.limit_price
            elif role in {OrderRole.TP.value, OrderRole.EXIT.value} and state and state.exit_order:
                requested_qty = state.exit_order.requested_qty
                requested_price = state.exit_order.limit_price
            elif role == "STOP" and state and state.position:
                requested_qty = state.position.qty_open
                requested_price = state.position.current_stop
        self._log_order_event(
            order_id=oms_order_id,
            symbol=symbol or "UNKNOWN",
            status="CANCEL_REQUESTED",
            requested_qty=requested_qty,
            order_type=role or "UNKNOWN",
            requested_price=requested_price,
        )
        await self._oms.submit_intent(Intent(intent_type=IntentType.CANCEL_ORDER, strategy_id=STRATEGY_ID, target_oms_order_id=oms_order_id))

    def _request_partial_exit(self, symbol: str, qty: int) -> None:
        state = self._symbols[symbol]
        if state.position is None or qty <= 0 or state.exit_order is not None or state.pending_hard_exit:
            return
        asyncio.create_task(self._submit_market_exit(symbol, qty, OrderRole.TP)).add_done_callback(self._log_task_exception)

    def _request_full_exit(self, symbol: str, reason: str) -> None:
        state = self._symbols[symbol]
        position = state.position
        if position is None or position.qty_open <= 0:
            return
        state.last_transition_reason = reason
        if state.exit_order is not None:
            if state.exit_order.role == OrderRole.EXIT.value:
                return
            state.pending_hard_exit = True
            if not state.exit_order.cancel_requested:
                state.exit_order.cancel_requested = True
                task = asyncio.create_task(self._cancel_order(state.exit_order.oms_order_id))
                task.add_done_callback(self._log_task_exception)
            return
        asyncio.create_task(
            self._cancel_then_exit(symbol, position.qty_open),
        ).add_done_callback(self._log_task_exception)

    async def _cancel_then_exit(self, symbol: str, qty: int) -> None:
        """Await stop cancel before submitting market exit to avoid double-execution."""
        await self._cancel_stop(symbol)
        await self._submit_market_exit(symbol, qty, OrderRole.EXIT)

    def _manage_position(self, symbol: str, now: datetime) -> None:
        state = self._symbols[symbol]
        market = self._markets[symbol]
        item = self._items.get(symbol)
        position = state.position
        if position is None or item is None or market.last_price is None:
            return
        position.max_favorable_price = max(position.max_favorable_price, market.last_price)
        position.max_adverse_price = min(position.max_adverse_price, market.last_price)
        if position.setup_tag == "UNCLASSIFIED":
            position.setup_tag = classify_trade(market, position)
            state.setup_tag = position.setup_tag

        if flow_reversal_exit_due(
            flow_reversal_flag=self._flow_reversal_flags.get(symbol, False),
            now=now,
            opened_today=position.entry_time.astimezone(ET).date() == self._artifact.trade_date,
        ):
            if state.exit_order is None or state.exit_order.role != OrderRole.EXIT.value:
                self._diagnostics.log_exit(symbol, "FLOW_REVERSAL", {"last_price": market.last_price})
            self._request_full_exit(symbol, "flow_reversal")
            return

        if should_exit_for_time_stop(position, now, market.last_price):
            if state.exit_order is None or state.exit_order.role != OrderRole.EXIT.value:
                self._diagnostics.log_exit(symbol, "TIME_STOP", {"last_price": market.last_price})
            self._request_full_exit(symbol, "time_stop")
            return

        partial_triggered, partial_fraction = should_take_partial(
            position, market.last_price, self._settings, regime_multiplier=item.regime_risk_multiplier,
        )
        if partial_triggered and state.exit_order is None and not state.pending_hard_exit:
            qty = max(1, int(position.qty_open * partial_fraction))
            self._diagnostics.log_exit(symbol, "PARTIAL", {"qty": qty, "last_price": market.last_price})
            self._request_partial_exit(symbol, qty)

        if market.last_30m_bar and market.avwap_live is not None:
            if should_exit_for_avwap_breakdown(market.last_30m_bar, market.avwap_live, max(state.average_30m_volume, 1.0), self._settings):
                if state.exit_order is None or state.exit_order.role != OrderRole.EXIT.value:
                    self._diagnostics.log_exit(symbol, "AVWAP_BREAKDOWN", {"last_price": market.last_price})
                self._request_full_exit(symbol, "avwap_breakdown")
                return

        if state.micropressure_signal == "DISTRIBUTE":
            tightened = max(position.current_stop, position.entry_price - item.tick_size)
            if tightened > position.current_stop and not state.pending_hard_exit and state.exit_order is None:
                position.current_stop = tightened
                asyncio.create_task(self._replace_stop(symbol)).add_done_callback(self._log_task_exception)

        if now.astimezone(ET).time() >= self._settings.forced_flatten:
            eligible, reason = carry_eligible(item, market, position, self._flow_reversal_flags.get(symbol, False))
            if state.exit_order is None or state.exit_order.role != OrderRole.EXIT.value:
                self._diagnostics.log_carry(symbol, eligible, reason)
            if not eligible:
                self._request_full_exit(symbol, "forced_flatten")

    async def _handle_event(self, event) -> None:
        if event.event_type == OMSEventType.FILL:
            await self._handle_fill(event)
        elif event.event_type == OMSEventType.ORDER_FILLED:
            pass  # order-state notification; real execution data comes via FILL
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
        if event.oms_order_id:
            self._order_index.pop(event.oms_order_id, None)
        if not symbol:
            return
        state = self._symbols.get(symbol)
        item = self._items.get(symbol)
        market = self._markets.get(symbol)
        if state is None or item is None:
            return
        fill_qty = int(float(payload.get("qty", 0.0) or 0.0))
        fill_price = float(payload.get("price", 0.0) or 0.0)
        if fill_qty <= 0:
            return

        if role == "ENTRY":
            entry_order = state.entry_order
            state.entry_order = None
            state.active_order_id = None
            self._portfolio.pending_entry_risk.pop(symbol, None)
            position = build_position_from_fill(
                fill_price=fill_price,
                fill_qty=fill_qty,
                stop_price=state.stop_level or max(fill_price - item.tick_size, 0.01),
                fill_time=event.timestamp,
                setup_tag=state.setup_tag or "UNCLASSIFIED",
            )
            position.time_stop_deadline = event.timestamp + timedelta(minutes=self._settings.time_stop_minutes)
            state.position = position
            state.in_position = True
            state.position_qty = fill_qty
            state.avg_price = fill_price
            state.fsm_state = "IN_POSITION"
            state.time_stop_deadline = position.time_stop_deadline
            self._portfolio.open_positions[symbol] = position
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
                        "entry_signal": state.setup_type or position.setup_tag,
                        "entry_signal_id": event.oms_order_id or symbol,
                        "entry_signal_strength": self._confidence_score(state.confidence),
                        "strategy_params": {
                            "setup_type": state.setup_type,
                            "location_grade": state.location_grade,
                            "sponsorship_signal": state.sponsorship_signal,
                            "micropressure_signal": state.micropressure_signal,
                            "flowproxy_signal": state.flowproxy_signal,
                            "stop0": state.stop_level,
                        },
                        "signal_factors": self._entry_signal_factors(state),
                        "filter_decisions": self._entry_filter_decisions(
                            state=state,
                            market=market,
                            timing_ok=True,
                        ),
                        "sizing_inputs": {
                            "entry_price": fill_price,
                            "stop_level": state.stop_level,
                            "qty": fill_qty,
                        },
                        "portfolio_state": self._portfolio_state_snapshot(),
                        "session_type": self._current_session_type(event.timestamp),
                        "exchange_timestamp": event.timestamp,
                        "expected_entry_price": entry_order.limit_price if entry_order else fill_price,
                        "concurrent_positions": len(self._portfolio.open_positions),
                        "drawdown_pct": getattr(self._portfolio, 'total_pnl_pct', None),
                        "bar_id": f"{symbol}:{event.timestamp.strftime('%Y%m%dT%H%M%S')}",
                        "entry_latency_ms": (
                            int((event.timestamp - entry_order.submitted_at).total_seconds() * 1000)
                            if entry_order and entry_order.submitted_at
                            else None
                        ),
                        "execution_timestamps": {
                            "order_submitted_at": (
                                entry_order.submitted_at.isoformat()
                                if entry_order and entry_order.submitted_at
                                else None
                            ),
                            "fill_received_at": event.timestamp.isoformat(),
                        },
                    },
                    account_id=self._account_id,
                )
            position.entry_commission = float(payload.get("commission", 0.0) or 0.0)
            self._log_orderbook_context(
                symbol=symbol,
                trade_context="entry",
                related_trade_id=position.trade_id,
                exchange_timestamp=event.timestamp,
            )
            await self._submit_stop(symbol)
            return

        position = state.position
        if position is None:
            return
        exit_order = state.exit_order
        if state.exit_order and event.oms_order_id == state.exit_order.oms_order_id:
            state.exit_order = None
        position.max_favorable_price = max(position.max_favorable_price, fill_price)
        position.max_adverse_price = min(position.max_adverse_price, fill_price)
        exit_qty = min(fill_qty, position.qty_open)
        position.realized_pnl_usd += (fill_price - position.entry_price) * exit_qty
        position.qty_open = max(0, position.qty_open - exit_qty)
        state.position_qty = position.qty_open
        if role == "TP":
            position.partial_taken = True
            position.current_stop = max(position.current_stop, position.entry_price - item.tick_size)
            if position.qty_open > 0 and position.stop_order_id:
                await self._replace_stop(symbol)
            if state.pending_hard_exit and position.qty_open > 0:
                state.pending_hard_exit = False
                await self._cancel_stop(symbol)
                await self._submit_market_exit(symbol, position.qty_open, OrderRole.EXIT)
        elif role == "EXIT" and position.qty_open > 0 and not position.stop_order_id:
            await self._submit_stop(symbol)
        elif role == "STOP":
            position.stop_order_id = ""

        if position.qty_open <= 0:
            if self._trade_recorder and position.trade_id:
                realized_r = position.realized_pnl_usd / max(position.total_initial_risk_usd, 1e-9)
                await self._trade_recorder.record_exit(
                    trade_id=position.trade_id,
                    exit_price=Decimal(str(fill_price)),
                    exit_ts=event.timestamp,
                    exit_reason=role or "EXIT",
                    realized_r=Decimal(str(round(realized_r, 4))),
                    realized_usd=Decimal(str(round(position.realized_pnl_usd, 2))),
                    mfe_r=Decimal(str(round(
                        (position.max_favorable_price - position.entry_price) / max(position.initial_risk_per_share, 1e-9),
                        4,
                    ))),
                    mae_r=Decimal(str(round(
                        (position.max_adverse_price - position.entry_price) / max(position.initial_risk_per_share, 1e-9),
                        4,
                    ))),
                    max_adverse_price=Decimal(str(position.max_adverse_price)),
                    max_favorable_price=Decimal(str(position.max_favorable_price)),
                    meta={
                        "exchange_timestamp": event.timestamp,
                        "expected_exit_price": (
                            exit_order.limit_price
                            if exit_order and exit_order.limit_price is not None
                            else fill_price
                        ),
                        "fees_paid": float(payload.get("commission", 0.0) or 0.0) + getattr(position, 'entry_commission', 0.0),
                        "session_transitions": [state.last_transition_reason] if state.last_transition_reason else [],
                        "exit_latency_ms": (
                            int((event.timestamp - exit_order.submitted_at).total_seconds() * 1000)
                            if exit_order and exit_order.submitted_at
                            else None
                        ),
                    },
                )
            self._log_orderbook_context(
                symbol=symbol,
                trade_context="exit",
                related_trade_id=position.trade_id,
                exchange_timestamp=event.timestamp,
            )
            self._portfolio.open_positions.pop(symbol, None)
            state.position = None
            state.in_position = False
            state.fsm_state = "INVALIDATED"
            state.invalidated_at = event.timestamp
            state.exit_order = None
            state.pending_hard_exit = False

    async def _handle_terminal(self, event) -> None:
        payload = event.payload or {}
        symbol, role = self._resolve_order(event.oms_order_id, payload)
        if event.oms_order_id:
            self._order_index.pop(event.oms_order_id, None)
        if not symbol:
            return
        state = self._symbols.get(symbol)
        if state is None:
            return

        if role == "ENTRY":
            state.entry_order = None
            state.active_order_id = None
            self._portfolio.pending_entry_risk.pop(symbol, None)
            state.fsm_state = "INVALIDATED"
            state.invalidated_at = event.timestamp
            state.last_transition_reason = "entry_terminal"
            self._log_missed(
                symbol=symbol,
                blocked_by="entry_terminal",
                block_reason=getattr(event.event_type, "value", str(event.event_type)),
                signal=state.setup_type or "iaric_entry",
                exchange_timestamp=event.timestamp,
            )
            return

        if role in {OrderRole.TP.value, OrderRole.EXIT.value}:
            pending_hard_exit = state.pending_hard_exit
            if state.exit_order and event.oms_order_id == state.exit_order.oms_order_id:
                state.exit_order = None
            state.pending_hard_exit = False
            if pending_hard_exit and state.position and state.position.qty_open > 0:
                await self._cancel_stop(symbol)
                await self._submit_market_exit(symbol, state.position.qty_open, OrderRole.EXIT)
            elif role == OrderRole.EXIT.value and state.position and state.position.qty_open > 0 and not state.position.stop_order_id:
                await self._submit_stop(symbol)
            return

        if role == "STOP" and state.position and state.position.qty_open > 0:
            if event.oms_order_id in self._expected_stop_cancels:
                self._expected_stop_cancels.discard(event.oms_order_id)
                state.position.stop_order_id = ""
                return
            state.position.stop_order_id = ""
            await self._submit_market_exit(symbol, state.position.qty_open, OrderRole.EXIT)

    def _resolve_order(self, oms_order_id: str | None, payload: dict[str, Any]) -> tuple[str, str]:
        if oms_order_id and oms_order_id in self._order_index:
            return self._order_index[oms_order_id]
        return str(payload.get("symbol", "")).upper(), str(payload.get("role", ""))

    async def _save_state(self, reason: str) -> None:
        persist_intraday_state(self.snapshot_state(), settings=self._settings)
        self._last_save_ts = datetime.now(timezone.utc)
        self._diagnostics.log_decision("STATE_SAVE", {"reason": reason})

    def _restore_order_state(self, symbol: str, state: SymbolIntradayState) -> None:
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
