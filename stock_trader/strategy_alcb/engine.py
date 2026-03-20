"""Async campaign engine for ALCB."""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import suppress
from copy import deepcopy
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from shared.oms.models.events import OMSEventType
from shared.oms.models.intent import Intent, IntentType
from shared.oms.models.order import OrderRole

from .artifact_store import load_latest_intraday_state, persist_intraday_state
from .config import ET, PROXY_SYMBOLS, STRATEGY_ID, STRATEGY_TYPE, StrategySettings, build_proxy_instruments
from .data import CanonicalBarBuilder, StrategyDataStore, aggregate_bars
from .diagnostics import JsonlDiagnostics
from .execution import (
    build_entry_order,
    build_market_exit,
    build_position_from_fill,
    build_stock_instrument,
    build_stop_order,
    build_tp_order,
)
from .exits import (
    add_trigger_price,
    breakeven_plus_buffer,
    business_days_between,
    can_add,
    gap_through_stop,
    maybe_enable_continuation,
    ratchet_runner_stop,
    stale_exit_needed,
    update_dirty_state,
)
from .models import (
    CampaignState,
    CandidateArtifact,
    Direction,
    EntryType,
    IntradayStateSnapshot,
    MarketSnapshot,
    PendingOrderState,
    PortfolioState,
    PositionPlan,
    PositionState,
    QuoteSnapshot,
    Regime,
    SymbolRuntimeState,
)
from .risk import (
    base_risk_fraction,
    choose_stop,
    event_block,
    friction_gate_pass,
    max_positions_pass,
    portfolio_heat_after,
    position_size,
    quality_mult,
    regime_mult,
    sector_limit_pass,
)
from .signals import (
    atr_from_bars,
    classify_4h_regime,
    compute_campaign_avwap,
    compute_weekly_vwap,
    determine_intraday_mode,
    directional_regime_pass,
    entry_a_trigger,
    entry_b_trigger,
    entry_c_trigger,
    in_entry_window,
    intraday_evidence_score,
    market_regime_from_proxies,
)

logger = logging.getLogger(__name__)


class ALCBEngine:
    """Live ALCB engine kept thin around nightly artifacts and pure helpers."""

    def __init__(
        self,
        oms_service,
        artifact: CandidateArtifact,
        account_id: str,
        nav: float,
        settings: StrategySettings | None = None,
        trade_recorder=None,
        diagnostics: JsonlDiagnostics | None = None,
        instrumentation=None,
    ) -> None:
        self._oms = oms_service
        self._artifact = artifact
        self._settings = settings or StrategySettings()
        self._account_id = account_id
        self._trade_recorder = trade_recorder
        self._diagnostics = diagnostics or JsonlDiagnostics(self._settings.diagnostics_dir, enabled=False)
        self._instrumentation = instrumentation

        self._items = artifact.by_symbol
        self._symbols: dict[str, SymbolRuntimeState] = {}
        self._markets: dict[str, MarketSnapshot] = {}
        self._portfolio = PortfolioState(
            account_equity=nav,
            base_risk_fraction=self._settings.base_risk_fraction,
        )
        self._symbol_to_sector = {item.symbol: item.sector for item in artifact.items}
        self._hot_symbols = {item.symbol for item in artifact.tradable}
        self._order_index: dict[str, tuple[str, str]] = {}
        self._pending_plans: dict[str, PositionPlan] = {}
        self._order_submission_times: dict[str, datetime] = {}
        self._exit_price_hints: dict[str, float] = {}
        self._expected_stop_cancels: set[str] = set()
        self._setup_emitted: set[tuple[str, int]] = set()
        self._bar_builder = CanonicalBarBuilder()
        self._last_save_ts: datetime | None = None

        self._event_queue = None
        self._event_task: asyncio.Task | None = None
        self._pulse_task: asyncio.Task | None = None
        self._running = False

        self._initialize_from_artifact()

    def _initialize_from_artifact(self) -> None:
        for item in self._artifact.items:
            campaign = deepcopy(item.campaign)
            campaign.symbol = item.symbol
            runtime = SymbolRuntimeState(symbol=item.symbol, campaign=campaign)
            self._symbols[item.symbol] = runtime
            bars_4h = aggregate_bars(item.bars_30m, 8)
            self._markets[item.symbol] = MarketSnapshot(
                symbol=item.symbol,
                last_price=item.price,
                daily_bars=item.daily_bars[:],
                bars_30m=item.bars_30m[:],
                last_30m_bar=(item.bars_30m[-1] if item.bars_30m else None),
                bars_4h=bars_4h,
                last_4h_bar=(bars_4h[-1] if bars_4h else None),
            )
        for instrument in build_proxy_instruments():
            self._markets.setdefault(instrument.symbol, MarketSnapshot(symbol=instrument.symbol))

    @staticmethod
    def _log_task_exception(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Unhandled exception in background task: %s", exc, exc_info=exc)

    def _store(self) -> StrategyDataStore:
        return StrategyDataStore(self._items, self._markets)

    async def _reconcile_after_reconnect(self) -> None:
        if self._oms is None:
            return
        logger.warning("IB reconnected; requesting OMS reconciliation")
        try:
            await self._oms.request_reconciliation()
            logger.info("OMS reconciliation complete")
        except Exception as exc:
            logger.error("OMS reconciliation failed: %s", exc, exc_info=exc)

    async def start(self) -> None:
        if self._running or self._oms is None:
            return
        self._running = True
        self._event_queue = self._oms.stream_events(STRATEGY_ID)
        self._event_task = asyncio.create_task(self._event_loop())
        self._pulse_task = asyncio.create_task(self._pulse_loop())

    async def stop(self) -> None:
        self._running = False
        await self._cancel_open_entries("shutdown")
        await self._save_state("stop")
        for task in (self._pulse_task, self._event_task):
            if task is None:
                continue
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    def hydrate_state(self, snapshot: IntradayStateSnapshot) -> None:
        for market in snapshot.markets:
            self._markets[market.symbol] = market

        self._portfolio.open_positions.clear()
        self._portfolio.pending_entry_risk.clear()
        self._order_index.clear()
        self._pending_plans.clear()
        self._order_submission_times.clear()
        for stored in snapshot.symbols:
            current = self._symbols.get(stored.symbol)
            if current is None:
                self._symbols[stored.symbol] = stored
            else:
                current.campaign = (
                    current.campaign
                    if self._campaign_rebuilt(current.campaign, stored.campaign)
                    else stored.campaign
                )
                current.intraday_score = stored.intraday_score
                current.intraday_detail = dict(stored.intraday_detail)
                current.mode = stored.mode
                current.last_transition_reason = stored.last_transition_reason
                current.last_30m_bar_time = stored.last_30m_bar_time
                current.entry_order = stored.entry_order
                current.stop_order = stored.stop_order
                current.exit_order = stored.exit_order
                current.tp1_order = stored.tp1_order
                current.tp2_order = stored.tp2_order
                current.position = stored.position
                current.pending_hard_exit = stored.pending_hard_exit
                current.pending_add = stored.pending_add
                current.last_signal_factors = dict(stored.last_signal_factors)
                stored = current
            if stored.position is not None:
                self._portfolio.open_positions[stored.symbol] = stored.position
            self._restore_order_state(stored.symbol, stored)

    @staticmethod
    def _campaign_rebuilt(fresh, stored) -> bool:
        if not stored.reentry_block_same_direction:
            return False
        if fresh.box is None:
            return False
        if stored.box is None:
            return True
        box_changed = (
            fresh.box.start_date != stored.box.start_date
            or fresh.box.end_date != stored.box.end_date
            or fresh.box.L_used != stored.box.L_used
            or abs(fresh.box.high - stored.box.high) > 1e-9
            or abs(fresh.box.low - stored.box.low) > 1e-9
        )
        breakout_changed = (
            fresh.breakout is not None
            and (
                stored.breakout is None
                or fresh.breakout.direction != stored.breakout.direction
                or fresh.breakout.breakout_date != stored.breakout.breakout_date
            )
        )
        return box_changed or breakout_changed

    def snapshot_state(self) -> IntradayStateSnapshot:
        return IntradayStateSnapshot(
            trade_date=self._artifact.trade_date,
            saved_at=datetime.now(timezone.utc),
            symbols=list(self._symbols.values()),
            markets=list(self._markets.values()),
            last_decision_code="snapshot",
            meta={"hot_symbols": sorted(self._hot_symbols)},
        )

    def subscription_instruments(self) -> list:
        instruments = build_proxy_instruments()
        seen = {instrument.symbol for instrument in instruments}
        for symbol in sorted(self._hot_symbols | set(self._portfolio.open_positions)):
            item = self._items.get(symbol)
            if item is None or symbol in seen:
                continue
            instruments.append(build_stock_instrument(item))
            seen.add(symbol)
        return instruments

    def polling_instruments(self) -> list[tuple[Any, int]]:
        requests: list[tuple[Any, int]] = []
        for item in self._artifact.overflow:
            if item.symbol in self._hot_symbols:
                continue
            requests.append((build_stock_instrument(item), self._settings.cold_poll_interval_s))
        return requests

    def on_quote(self, symbol: str, quote: QuoteSnapshot) -> None:
        normalized = symbol.upper()
        market = self._markets.get(normalized)
        if market is None:
            return
        market.last_quote = quote
        market.bid = quote.bid
        market.ask = quote.ask
        market.spread_pct = quote.spread_pct
        if quote.vwap is not None and quote.vwap > 0:
            market.session_vwap = quote.vwap
        elif quote.cumulative_volume > 0 and quote.cumulative_value > 0:
            market.session_vwap = quote.cumulative_value / quote.cumulative_volume
        market.last_price = quote.last if quote.last > 0 else market.last_price

    def on_bar(self, symbol: str, bar) -> None:
        normalized = symbol.upper()
        market = self._markets.get(normalized)
        state = self._symbols.get(normalized)
        if market is None:
            return
        if bar.start_time.astimezone(ET).date() != self._artifact.trade_date:
            return
        if market.minute_bars and market.minute_bars[-1].start_time >= bar.start_time:
            return

        market.minute_bars.append(bar)
        market.last_1m_bar = bar
        market.last_price = bar.close
        self._bar_builder.ingest_bar(bar)

        for bar_30m in self._bar_builder.aggregate_new_bars(normalized, 30):
            if market.bars_30m and market.bars_30m[-1].end_time >= bar_30m.end_time:
                continue
            market.bars_30m.append(bar_30m)
            market.last_30m_bar = bar_30m
            market.bars_4h = aggregate_bars(market.bars_30m, 8)
            market.last_4h_bar = market.bars_4h[-1] if market.bars_4h else None
            if state is None:
                continue
            state.last_30m_bar_time = bar_30m.end_time
            self._refresh_live_metrics(normalized)
            asyncio.create_task(self._advance_symbol(normalized, bar_30m.end_time)).add_done_callback(
                self._log_task_exception
            )

    def _refresh_live_metrics(self, symbol: str) -> None:
        state = self._symbols.get(symbol)
        market = self._markets.get(symbol)
        if state is None or market is None:
            return
        store = self._store()
        market.weekly_vwap = compute_weekly_vwap(store.bars_30m(symbol))
        market.avwap_live = compute_campaign_avwap(store.bars_30m(symbol), state.campaign.avwap_anchor_ts)

    def _live_market_regime(self, store: StrategyDataStore) -> tuple[Regime, dict[str, str]]:
        market_regime, proxy_detail = market_regime_from_proxies(store, proxy_symbols=PROXY_SYMBOLS[:2])
        if proxy_detail:
            return market_regime, proxy_detail
        fallback = (
            Regime(self._artifact.regime.market_regime)
            if self._artifact.regime.market_regime in Regime._value2member_map_
            else Regime.TRANSITIONAL
        )
        return fallback, {}

    @staticmethod
    def _plan_from_order_state(symbol: str, order: PendingOrderState | None) -> PositionPlan | None:
        if order is None:
            return None
        if (
            order.direction is None
            or order.entry_type is None
            or order.entry_price is None
            or order.planned_stop_price is None
            or order.planned_tp1_price is None
            or order.planned_tp2_price is None
            or order.risk_per_share is None
            or order.risk_dollars is None
        ):
            return None
        return PositionPlan(
            symbol=symbol,
            direction=order.direction,
            entry_type=order.entry_type,
            entry_price=order.entry_price,
            stop_price=order.planned_stop_price,
            tp1_price=order.planned_tp1_price,
            tp2_price=order.planned_tp2_price,
            quantity=order.requested_qty,
            risk_per_share=order.risk_per_share,
            risk_dollars=order.risk_dollars,
            quality_mult=1.0,
            regime_mult=1.0,
            corr_mult=1.0,
        )

    def get_position_snapshot(self) -> list[dict[str, Any]]:
        snapshots: list[dict[str, Any]] = []
        for symbol, state in self._symbols.items():
            market = self._markets.get(symbol)
            if state.position is None or market is None or market.last_price is None:
                continue
            snapshots.append(
                {
                    "strategy_type": STRATEGY_TYPE,
                    "symbol": symbol,
                    "direction": state.position.direction.value,
                    "entry_price": state.position.entry_price,
                    "qty": state.position.qty_open,
                    "unrealized_pnl_r": round(state.position.unrealized_r(market.last_price), 3),
                }
            )
        return snapshots

    def open_order_count(self) -> int:
        return len(self._order_index)

    def _session_type(self, now: datetime) -> str:
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
            "halt_new_entries": self._portfolio.halt_new_entries,
            "flatten_all": self._portfolio.flatten_all,
            "total_pnl_pct": self._portfolio.total_pnl_pct,
            "sectors_in_use": sorted(
                {
                    self._symbol_to_sector.get(symbol, "")
                    for symbol in self._portfolio.open_positions
                    if self._symbol_to_sector.get(symbol)
                }
            ),
        }

    def _intraday_threshold(self, mode: str) -> int:
        if mode == "NORMAL":
            return self._settings.normal_intraday_score_min
        return self._settings.degraded_intraday_score_min

    @staticmethod
    def _decision_bar_id(symbol: str, ts: datetime | None) -> str | None:
        if ts is None:
            return None
        return f"{symbol}:{ts.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

    @staticmethod
    def _latency_ms(submitted_at: datetime | None, completed_at: datetime | None) -> int | None:
        if submitted_at is None or completed_at is None:
            return None
        return max(0, int(round((completed_at - submitted_at).total_seconds() * 1000)))

    @staticmethod
    def _pct_delta(value: float | None, reference: float | None) -> float | None:
        if value is None or reference is None or abs(reference) <= 1e-9:
            return None
        return ((value - reference) / reference) * 100.0

    def _track_order_submission(self, order_id: str | None, submitted_at: datetime | None) -> None:
        if order_id and submitted_at is not None:
            self._order_submission_times[order_id] = submitted_at

    def _clear_order_tracking(
        self,
        order_id: str | None,
        *,
        drop_plan: bool = False,
        drop_exit_hint: bool = True,
    ) -> None:
        if not order_id:
            return
        self._order_index.pop(order_id, None)
        self._order_submission_times.pop(order_id, None)
        if drop_plan:
            self._pending_plans.pop(order_id, None)
        if drop_exit_hint:
            self._exit_price_hints.pop(order_id, None)

    @staticmethod
    def _requested_qty_from_fill(
        order_state: PendingOrderState | None,
        payload: dict[str, Any],
        fill_qty: int,
    ) -> int:
        requested_qty = int(float(payload.get("requested_qty", 0.0) or 0.0))
        if order_state is not None:
            requested_qty = max(requested_qty, int(order_state.requested_qty or 0))
        return max(requested_qty, fill_qty)

    def _apply_fill_to_order(
        self,
        order_state: PendingOrderState | None,
        *,
        fill_qty: int,
        requested_qty: int,
    ) -> tuple[int, int, bool, int]:
        previous_filled = 0
        if order_state is None:
            remaining_qty = max(0, requested_qty - fill_qty)
            return requested_qty, remaining_qty, remaining_qty == 0, previous_filled
        previous_filled = int(order_state.filled_qty or 0)
        order_state.requested_qty = max(int(order_state.requested_qty or 0), requested_qty)
        order_state.filled_qty = min(order_state.requested_qty, previous_filled + fill_qty)
        remaining_qty = max(0, order_state.requested_qty - order_state.filled_qty)
        return order_state.requested_qty, remaining_qty, remaining_qty == 0, previous_filled

    def _entry_fill_risk(self, entry_plan: PositionPlan, fill_qty: int) -> float:
        return max(0.0, float(entry_plan.risk_per_share)) * max(0, fill_qty)

    def _remaining_entry_risk(self, entry_plan: PositionPlan, remaining_qty: int) -> float:
        return max(0.0, float(entry_plan.risk_per_share)) * max(0, remaining_qty)

    def _new_exit_oca_group(self, symbol: str) -> str:
        return f"ALCB-{symbol}-EXIT-{uuid.uuid4().hex[:12]}"

    def _current_stop_order_id(self, state: SymbolRuntimeState) -> str:
        if state.stop_order is not None:
            return state.stop_order.oms_order_id
        if state.position is None:
            return ""
        return state.position.stop_order_id

    def _drop_stop_order_state(self, symbol: str) -> None:
        state = self._symbols[symbol]
        stop_order_id = self._current_stop_order_id(state)
        if stop_order_id:
            self._clear_order_tracking(stop_order_id)
            self._expected_stop_cancels.add(stop_order_id)
        state.stop_order = None
        if state.position is not None:
            state.position.stop_order_id = ""
            state.position.stop_submitted_at = None

    def _drop_tp_order_state(self, symbol: str, which: str) -> None:
        state = self._symbols[symbol]
        order_state = getattr(state, which)
        if order_state is not None:
            self._clear_order_tracking(order_state.oms_order_id)
        setattr(state, which, None)
        if state.position is None:
            return
        if which == "tp1_order":
            state.position.tp1_order_id = ""
        elif which == "tp2_order":
            state.position.tp2_order_id = ""

    async def _sync_exit_orders_after_entry_fill(self, symbol: str) -> None:
        state = self._symbols[symbol]
        position = state.position
        if position is None or position.qty_open <= 0:
            return
        if not position.exit_oca_group:
            position.exit_oca_group = self._new_exit_oca_group(symbol)
        if self._current_stop_order_id(state):
            await self._replace_stop(symbol)
        else:
            await self._submit_stop(symbol)
        await self._submit_tp_orders(symbol)

    def _missing_order_id_error(
        self,
        *,
        action: str,
        symbol: str,
        context: dict[str, Any] | None = None,
        exchange_timestamp: datetime | None = None,
    ) -> None:
        error = RuntimeError(f"{action} OMS receipt missing order id")
        logger.error("%s for %s returned no OMS order id", action, symbol)
        self._log_engine_error(
            error_type=f"{action}_failed",
            symbol=symbol,
            exc=error,
            context=context,
            exchange_timestamp=exchange_timestamp,
        )

    def _log_engine_error(
        self,
        *,
        error_type: str,
        symbol: str,
        exc: Exception,
        context: dict[str, Any] | None = None,
        exchange_timestamp: datetime | None = None,
    ) -> None:
        if self._instrumentation is None:
            return
        try:
            self._instrumentation.log_error(
                error_type=error_type,
                message=str(exc),
                severity="high",
                category="engine",
                context={"symbol": symbol, **(context or {})},
                exc=exc,
                exchange_timestamp=exchange_timestamp,
            )
        except Exception:
            pass

    def _emit_indicator_snapshot(
        self,
        *,
        symbol: str,
        now: datetime,
        decision: str,
        stock_regime: Regime,
        market_regime: Regime,
        proxy_regimes: dict[str, str],
        entry_type: EntryType | None = None,
    ) -> None:
        if self._instrumentation is None:
            return
        state = self._symbols.get(symbol)
        item = self._items.get(symbol)
        market = self._markets.get(symbol)
        if state is None or item is None or market is None or state.campaign.breakout is None:
            return

        indicators: dict[str, float] = {
            "spread_pct": float(market.spread_pct or item.median_spread_pct or 0.0),
            "selection_score": float(item.selection_score),
            "intraday_score": float(state.intraday_score),
            "mode_threshold": float(self._intraday_threshold(state.mode)),
            "disp_ratio": float(
                (
                    state.campaign.breakout.disp_value / state.campaign.breakout.disp_threshold
                    if state.campaign.breakout.disp_threshold > 0
                    else 0.0
                )
            ),
            "breakout_rvol_d": float(state.campaign.breakout.rvol_d or 0.0),
        }
        for key, value in sorted(state.intraday_detail.items()):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                indicators[key] = float(value)

        last_vs_session_vwap_pct = self._pct_delta(market.last_price, market.session_vwap)
        if last_vs_session_vwap_pct is not None:
            indicators["last_vs_session_vwap_pct"] = float(last_vs_session_vwap_pct)
        last_vs_avwap_pct = self._pct_delta(market.last_price, market.avwap_live)
        if last_vs_avwap_pct is not None:
            indicators["last_vs_avwap_pct"] = float(last_vs_avwap_pct)

        try:
            self._instrumentation.on_indicator_snapshot(
                pair=symbol,
                indicators=indicators,
                signal_name="alcb_intraday_decision",
                signal_strength=float(state.intraday_score or 0.0),
                decision=decision,
                strategy_type=STRATEGY_TYPE,
                exchange_timestamp=now,
                bar_id=self._decision_bar_id(symbol, state.last_30m_bar_time or now),
                context={
                    "campaign_state": state.campaign.state.value,
                    "mode": state.mode,
                    "direction": state.campaign.breakout.direction.value,
                    "candidate_entry_type": entry_type.value if entry_type is not None else "",
                    "stock_regime": stock_regime.value,
                    "market_regime": market_regime.value,
                    "proxy_regimes": dict(proxy_regimes),
                },
            )
        except Exception:
            pass

    def _entry_signal_factors(self, state: SymbolRuntimeState) -> list[dict[str, Any]]:
        factors = state.last_signal_factors
        threshold_map: dict[str, float] = {
            "intraday_score": float(
                self._settings.normal_intraday_score_min
                if state.mode == "NORMAL"
                else self._settings.degraded_intraday_score_min
            ),
            "mode_threshold": 0.0,
            "rvol_30m": self._settings.intraday_rvol_min,
            "adx14": 20.0,
            "disp_ratio": 1.0,
            "acceptance_clv": 0.55,
            "spread_pct": self._settings.max_median_spread_pct,
        }
        result: list[dict[str, Any]] = []
        for key, value in sorted(factors.items()):
            if not isinstance(value, (int, float)):
                continue
            thresh = threshold_map.get(key, 0.0)
            contribution = (
                (float(value) - thresh) / thresh if thresh > 0 else float(value)
            )
            result.append({
                "factor_name": key,
                "factor_value": float(value),
                "threshold": thresh,
                "contribution": round(contribution, 4),
            })
        # Add factors available from campaign/market context but not in last_signal_factors
        item = self._items.get(state.symbol)
        market = self._markets.get(state.symbol)
        if item is not None:
            if "spread_at_signal" not in factors:
                result.append({
                    "factor_name": "spread_at_signal",
                    "factor_value": item.median_spread_pct,
                    "threshold": self._settings.max_median_spread_pct,
                    "contribution": round(
                        (item.median_spread_pct - self._settings.max_median_spread_pct)
                        / self._settings.max_median_spread_pct
                        if self._settings.max_median_spread_pct > 0 else 0.0, 4
                    ),
                })
        if state.campaign.box is not None and "compression_tier" not in factors:
            result.append({
                "factor_name": "compression_tier",
                "factor_value": {"GOOD": 2, "NEUTRAL": 1, "LOOSE": 0}.get(
                    state.campaign.box.tier.value, 0
                ),
                "threshold": 1.0,
                "contribution": 1.0 if state.campaign.box.tier.value == "GOOD" else 0.0,
            })
        if state.campaign.box is not None and "box_length" not in factors:
            result.append({
                "factor_name": "box_length",
                "factor_value": float(state.campaign.box.L_used),
                "threshold": float(self._settings.box_length_mid),
                "contribution": round(
                    (state.campaign.box.L_used - self._settings.box_length_mid)
                    / self._settings.box_length_mid, 4
                ),
            })
        if market is not None and market.avwap_live and market.last_price and "avwap_distance_pct" not in factors:
            avwap_dist = abs(market.last_price - market.avwap_live) / market.last_price if market.last_price > 0 else 0.0
            result.append({
                "factor_name": "avwap_distance_pct",
                "factor_value": round(avwap_dist, 6),
                "threshold": 0.0,
                "contribution": round(avwap_dist, 6),
            })
        return result

    def _entry_filter_decisions(
        self,
        item,
        *,
        timing_ok: bool,
        score_ok: bool,
        intraday_score: int,
        threshold: int,
        heat_ok: bool | None = None,
        heat_value: float | None = None,
        friction_ok: bool | None = None,
    ) -> list[dict[str, Any]]:
        decisions = [
            {
                "filter_name": "timing_gate",
                "threshold": 1.0,
                "actual_value": 1.0 if timing_ok else 0.0,
                "passed": timing_ok,
            },
            {
                "filter_name": "intraday_score",
                "threshold": float(threshold),
                "actual_value": float(intraday_score),
                "passed": score_ok,
            },
            {
                "filter_name": "spread_gate",
                "threshold": self._settings.max_median_spread_pct,
                "actual_value": item.median_spread_pct,
                "passed": item.median_spread_pct <= self._settings.max_median_spread_pct,
            },
        ]
        if heat_ok is not None:
            decisions.append(
                {
                    "filter_name": "portfolio_heat",
                    "threshold": self._settings.max_portfolio_heat_fraction,
                    "actual_value": float(heat_value or 0.0),
                    "passed": heat_ok,
                }
            )
        if friction_ok is not None:
            decisions.append(
                {
                    "filter_name": "friction_gate",
                    "threshold": self._settings.max_friction_to_risk,
                    "actual_value": 1.0 if friction_ok else 0.0,
                    "passed": friction_ok,
                }
            )
        return decisions

    def _log_missed(
        self,
        *,
        symbol: str,
        direction: Direction,
        blocked_by: str,
        block_reason: str,
        exchange_timestamp: datetime,
        state: SymbolRuntimeState | None = None,
        strategy_params: dict[str, Any] | None = None,
        filter_decisions: list[dict[str, Any]] | None = None,
    ) -> None:
        if self._instrumentation is None:
            return
        runtime = state or self._symbols.get(symbol)
        item = self._items.get(symbol)
        try:
            self._instrumentation.log_missed(
                pair=symbol,
                side=direction.value,
                signal="alcb_breakout",
                signal_id=f"{symbol}:{blocked_by}:{int(exchange_timestamp.timestamp())}",
                signal_strength=float(runtime.intraday_score if runtime else 0.0),
                blocked_by=blocked_by,
                block_reason=block_reason,
                strategy_params={
                    "campaign_state": runtime.campaign.state.value if runtime else "",
                    "entry_type": (
                        runtime.campaign.last_entry_type.value
                        if runtime and runtime.campaign.last_entry_type is not None
                        else ""
                    ),
                    "selection_score": item.selection_score if item else 0,
                    **(strategy_params or {}),
                },
                filter_decisions=filter_decisions or [],
                session_type=self._session_type(exchange_timestamp),
                concurrent_positions=len(self._portfolio.open_positions),
                drawdown_pct=self._portfolio.total_pnl_pct,
                exchange_timestamp=exchange_timestamp,
            )
        except Exception:
            pass

    def _log_order_event(
        self,
        *,
        order_id: str,
        symbol: str,
        side: str,
        status: str,
        requested_qty: int,
        order_type: str,
        requested_price: float | None = None,
        related_trade_id: str = "",
        exchange_timestamp: datetime | None = None,
    ) -> None:
        if self._instrumentation is None:
            return
        try:
            self._instrumentation.on_order_event(
                order_id=order_id,
                pair=symbol,
                side=side,
                order_type=order_type,
                status=status,
                requested_qty=requested_qty,
                requested_price=requested_price,
                related_trade_id=related_trade_id,
                strategy_type=STRATEGY_TYPE,
                session=self._session_type(exchange_timestamp or datetime.now(timezone.utc)),
                exchange_timestamp=exchange_timestamp,
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
        quote = None if market is None else market.last_quote
        if market is None or quote is None:
            return
        best_bid = float(quote.bid or market.last_price or 0.0)
        best_ask = float(quote.ask or market.last_price or 0.0)
        if best_bid <= 0 or best_ask <= 0:
            return
        bid_depth = float(getattr(quote, "bid_size", 0.0) or 0.0)
        ask_depth = float(getattr(quote, "ask_size", 0.0) or 0.0)
        try:
            self._instrumentation.on_orderbook_context(
                pair=symbol,
                best_bid=best_bid,
                best_ask=best_ask,
                trade_context=trade_context,
                related_trade_id=related_trade_id or None,
                bid_depth_10bps=bid_depth,
                ask_depth_10bps=ask_depth,
                bid_levels=([{"price": best_bid, "size": bid_depth}] if bid_depth > 0 else None),
                ask_levels=([{"price": best_ask, "size": ask_depth}] if ask_depth > 0 else None),
                exchange_timestamp=exchange_timestamp,
            )
        except Exception:
            pass

    async def advance(self, now: datetime) -> None:
        await self._refresh_portfolio()
        if now.astimezone(ET).time() >= self._settings.forced_flatten:
            await self._flatten_all("forced_flatten")

    async def _pulse_loop(self) -> None:
        while self._running:
            await self.advance(datetime.now(timezone.utc))
            should_save = (
                self._last_save_ts is None
                or (datetime.now(timezone.utc) - self._last_save_ts).total_seconds() >= 60.0
            )
            if should_save:
                await self._save_state("pulse")
            await asyncio.sleep(5.0)

    async def _event_loop(self) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            try:
                await self._handle_event(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                payload = getattr(event, "payload", {}) or {}
                symbol = str(payload.get("symbol", "")).upper()
                logger.error(
                    "Unhandled OMS event for %s/%s: %s",
                    getattr(event, "event_type", "UNKNOWN"),
                    getattr(event, "oms_order_id", ""),
                    exc,
                    exc_info=exc,
                )
                self._log_engine_error(
                    error_type="oms_event_processing_failed",
                    symbol=symbol or "UNKNOWN",
                    exc=exc,
                    context={
                        "event_type": getattr(getattr(event, "event_type", None), "value", str(getattr(event, "event_type", ""))),
                        "oms_order_id": getattr(event, "oms_order_id", None),
                    },
                    exchange_timestamp=getattr(event, "timestamp", None),
                )

    async def _refresh_portfolio(self) -> None:
        if self._oms is None:
            return
        strategy_halted = False
        portfolio_halted = False
        try:
            risk_state = await self._oms.get_strategy_risk(STRATEGY_ID)
            strategy_halted = bool(getattr(risk_state, "halted", False))
        except Exception:
            pass
        try:
            portfolio_risk = await self._oms.get_portfolio_risk()
            portfolio_halted = bool(getattr(portfolio_risk, "halted", False))
        except Exception:
            pass
        self._portfolio.halt_new_entries = strategy_halted or portfolio_halted or self._artifact.regime.tier == "C"
        self._portfolio.flatten_all = strategy_halted or portfolio_halted

    async def _advance_symbol(self, symbol: str, now: datetime) -> None:
        state = self._symbols.get(symbol)
        item = self._items.get(symbol)
        if state is None or item is None:
            return
        if state.position is not None:
            await self._manage_position(symbol, now)
            return
        if state.entry_order is not None:
            return

        campaign = state.campaign
        if campaign.state not in {
            CampaignState.COMPRESSION,
            CampaignState.BREAKOUT,
            CampaignState.CONTINUATION,
            CampaignState.DIRTY,
        }:
            return
        if campaign.breakout is None:
            return
        if campaign.reentry_block_same_direction:
            self._log_missed(
                symbol=symbol,
                direction=campaign.breakout.direction,
                blocked_by="dirty_block",
                block_reason="reentry_block_same_direction",
                exchange_timestamp=now,
                state=state,
            )
            return
        if self._portfolio.halt_new_entries or self._portfolio.flatten_all:
            return
        if event_block(item):
            self._log_missed(
                symbol=symbol,
                direction=campaign.breakout.direction,
                blocked_by="event_block",
                block_reason="event_risk",
                exchange_timestamp=now,
                state=state,
            )
            return

        timing_ok = in_entry_window(now, self._settings)
        store = self._store()
        stock_regime = classify_4h_regime(store.bars_4h(symbol))
        market_regime, proxy_regimes = self._live_market_regime(store)
        regime_pass, regime_detail = directional_regime_pass(
            campaign.breakout.direction,
            stock_regime,
            market_regime,
            item.daily_trend_sign,
        )
        regime_detail = {**regime_detail, **{f"proxy_{name.lower()}": value for name, value in proxy_regimes.items()}}
        if not regime_pass:
            self._log_missed(
                symbol=symbol,
                direction=campaign.breakout.direction,
                blocked_by="regime_block",
                block_reason="directional_regime_opposes",
                exchange_timestamp=now,
                state=state,
                strategy_params=regime_detail,
            )
            return

        setup_key = (symbol, campaign.box_version)
        if campaign.state == CampaignState.COMPRESSION and campaign.box is not None and setup_key not in self._setup_emitted:
            self._setup_emitted.add(setup_key)
            self._diagnostics.log_setup_detected(symbol, {
                "campaign_state": campaign.state.value,
                "box_tier": campaign.box.tier.value,
                "box_length": campaign.box.L_used,
                "breakout_direction": campaign.breakout.direction.value,
                "stock_regime": stock_regime.value,
                "market_regime": market_regime.value,
            })

        intraday_score, detail = intraday_evidence_score(symbol, campaign, store, self._settings)
        state.intraday_score = intraday_score
        state.intraday_detail = detail
        state.mode = determine_intraday_mode(campaign.breakout.direction, stock_regime, market_regime)
        state.last_market_regime = market_regime.value
        state.last_stock_regime = stock_regime.value
        threshold = self._intraday_threshold(state.mode)
        score_ok = intraday_score >= threshold
        state.last_signal_factors = {
            "intraday_score": intraday_score,
            "mode_threshold": threshold,
            "stock_regime": stock_regime.value,
            "market_regime": market_regime.value,
            "disp_ratio": (
                campaign.breakout.disp_value / campaign.breakout.disp_threshold
                if campaign.breakout.disp_threshold > 0
                else 0.0
            ),
            **detail,
        }
        filter_decisions = self._entry_filter_decisions(
            item,
            timing_ok=timing_ok,
            score_ok=score_ok,
            intraday_score=intraday_score,
            threshold=threshold,
        )
        if not timing_ok or not score_ok:
            self._log_missed(
                symbol=symbol,
                direction=campaign.breakout.direction,
                blocked_by="entry_gate",
                block_reason="timing_gate" if not timing_ok else "intraday_score",
                exchange_timestamp=now,
                state=state,
                filter_decisions=filter_decisions,
            )
            self._emit_indicator_snapshot(
                symbol=symbol,
                now=now,
                decision="entry_gate_block",
                stock_regime=stock_regime,
                market_regime=market_regime,
                proxy_regimes=proxy_regimes,
            )
            return

        entry_type, entry_price = self._choose_entry(symbol)
        if entry_type is None or entry_price is None:
            self._emit_indicator_snapshot(
                symbol=symbol,
                now=now,
                decision="no_entry_setup",
                stock_regime=stock_regime,
                market_regime=market_regime,
                proxy_regimes=proxy_regimes,
            )
            return

        stop_price = choose_stop(
            entry_type,
            campaign.breakout.direction,
            item,
            campaign,
            self._settings,
            stock_regime=stock_regime,
            market_regime=market_regime,
        )
        reg_mult = regime_mult(campaign.breakout.direction, stock_regime, market_regime)
        if reg_mult <= 0:
            self._log_missed(
                symbol=symbol,
                direction=campaign.breakout.direction,
                blocked_by="sizing_block",
                block_reason="regime_sized_to_zero",
                exchange_timestamp=now,
                state=state,
                strategy_params={
                    "stock_regime": stock_regime.value,
                    "market_regime": market_regime.value,
                },
            )
            self._emit_indicator_snapshot(
                symbol=symbol,
                now=now,
                decision="sizing_regime_block",
                stock_regime=stock_regime,
                market_regime=market_regime,
                proxy_regimes=proxy_regimes,
                entry_type=entry_type,
            )
            return
        q_mult = quality_mult(campaign, intraday_score, self._settings)
        if q_mult <= 0:
            self._log_missed(
                symbol=symbol,
                direction=campaign.breakout.direction,
                blocked_by="sizing_block",
                block_reason="quality_sized_to_zero",
                exchange_timestamp=now,
                state=state,
                strategy_params={
                    "intraday_score": intraday_score,
                    "stock_regime": stock_regime.value,
                    "market_regime": market_regime.value,
                },
            )
            self._emit_indicator_snapshot(
                symbol=symbol,
                now=now,
                decision="sizing_quality_block",
                stock_regime=stock_regime,
                market_regime=market_regime,
                proxy_regimes=proxy_regimes,
                entry_type=entry_type,
            )
            return
        plan = position_size(
            item,
            entry_price,
            stop_price,
            campaign.breakout.direction,
            campaign,
            intraday_score,
            self._portfolio,
            self._items,
            self._settings,
            entry_type=entry_type,
            stock_regime=stock_regime,
            market_regime=market_regime,
        )
        if plan is None:
            self._log_missed(
                symbol=symbol,
                direction=campaign.breakout.direction,
                blocked_by="sizing_block",
                block_reason="position_size_zero",
                exchange_timestamp=now,
                state=state,
                strategy_params={
                    "stock_regime": stock_regime.value,
                    "market_regime": market_regime.value,
                },
            )
            self._emit_indicator_snapshot(
                symbol=symbol,
                now=now,
                decision="no_entry_setup",
                stock_regime=stock_regime,
                market_regime=market_regime,
                proxy_regimes=proxy_regimes,
                entry_type=entry_type,
            )
            return

        projected_heat = portfolio_heat_after(plan, self._portfolio)
        heat_ok = projected_heat <= self._settings.max_portfolio_heat_fraction
        friction_ok = friction_gate_pass(item, plan, self._settings)
        filter_decisions = self._entry_filter_decisions(
            item,
            timing_ok=timing_ok,
            score_ok=score_ok,
            intraday_score=intraday_score,
            threshold=threshold,
            heat_ok=heat_ok,
            heat_value=projected_heat,
            friction_ok=friction_ok,
        )
        if not max_positions_pass(self._portfolio, self._settings):
            self._log_missed(
                symbol=symbol,
                direction=campaign.breakout.direction,
                blocked_by="portfolio_block",
                block_reason="max_positions",
                exchange_timestamp=now,
                state=state,
                filter_decisions=filter_decisions,
            )
            self._emit_indicator_snapshot(
                symbol=symbol,
                now=now,
                decision="max_positions_block",
                stock_regime=stock_regime,
                market_regime=market_regime,
                proxy_regimes=proxy_regimes,
                entry_type=entry_type,
            )
            return
        if not sector_limit_pass(item, self._portfolio, self._symbol_to_sector, self._settings):
            self._log_missed(
                symbol=symbol,
                direction=campaign.breakout.direction,
                blocked_by="portfolio_block",
                block_reason="sector_limit",
                exchange_timestamp=now,
                state=state,
                filter_decisions=filter_decisions,
            )
            self._emit_indicator_snapshot(
                symbol=symbol,
                now=now,
                decision="sector_limit_block",
                stock_regime=stock_regime,
                market_regime=market_regime,
                proxy_regimes=proxy_regimes,
                entry_type=entry_type,
            )
            return
        if not heat_ok:
            self._log_missed(
                symbol=symbol,
                direction=campaign.breakout.direction,
                blocked_by="risk_block",
                block_reason="heat_cap",
                exchange_timestamp=now,
                state=state,
                filter_decisions=filter_decisions,
            )
            self._emit_indicator_snapshot(
                symbol=symbol,
                now=now,
                decision="heat_cap_block",
                stock_regime=stock_regime,
                market_regime=market_regime,
                proxy_regimes=proxy_regimes,
                entry_type=entry_type,
            )
            return
        if not friction_ok:
            self._log_missed(
                symbol=symbol,
                direction=campaign.breakout.direction,
                blocked_by="friction_block",
                block_reason="friction_to_risk",
                exchange_timestamp=now,
                state=state,
                filter_decisions=filter_decisions,
            )
            self._emit_indicator_snapshot(
                symbol=symbol,
                now=now,
                decision="friction_block",
                stock_regime=stock_regime,
                market_regime=market_regime,
                proxy_regimes=proxy_regimes,
                entry_type=entry_type,
            )
            return

        if await self._submit_entry(symbol, plan, now):
            self._emit_indicator_snapshot(
                symbol=symbol,
                now=now,
                decision="entry_submitted",
                stock_regime=stock_regime,
                market_regime=market_regime,
                proxy_regimes=proxy_regimes,
                entry_type=entry_type,
            )

    def _choose_entry(self, symbol: str) -> tuple[EntryType | None, float | None]:
        state = self._symbols[symbol]
        store = self._store()
        if state.campaign.continuation_enabled or state.campaign.state == CampaignState.CONTINUATION:
            price = entry_c_trigger(symbol, state.campaign, store)
            self._diagnostics.log_signal_evaluation(
                symbol, entry_type="entry_c", triggered=price is not None,
                conditions={"continuation_enabled": True, "campaign_state": state.campaign.state.value},
                limit_price=price,
            )
            if price is not None:
                return EntryType.C_CONTINUATION, price
            return None, None
        price_a = entry_a_trigger(symbol, state.campaign, store)
        self._diagnostics.log_signal_evaluation(
            symbol, entry_type="entry_a", triggered=price_a is not None,
            conditions={"avwap_retest": price_a is not None},
            limit_price=price_a,
        )
        if price_a is not None:
            return EntryType.A_AVWAP_RETEST, price_a
        price_b = entry_b_trigger(symbol, state.campaign, store, self._settings)
        self._diagnostics.log_signal_evaluation(
            symbol, entry_type="entry_b", triggered=price_b is not None,
            conditions={"sweep_reclaim": price_b is not None},
            limit_price=price_b,
        )
        if price_b is not None:
            return EntryType.B_SWEEP_RECLAIM, price_b
        price_c = entry_c_trigger(symbol, state.campaign, store)
        self._diagnostics.log_signal_evaluation(
            symbol, entry_type="entry_c", triggered=price_c is not None,
            conditions={"continuation_fallback": True},
            limit_price=price_c,
        )
        if price_c is not None:
            return EntryType.C_CONTINUATION, price_c
        return None, None

    async def _submit_entry(self, symbol: str, plan: PositionPlan, now: datetime) -> bool:
        if self._oms is None:
            return False
        state = self._symbols[symbol]
        item = self._items[symbol]
        if state.entry_order is not None:
            return False
        try:
            order = build_entry_order(item, self._account_id, plan)
            receipt = await self._oms.submit_intent(
                Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=order)
            )
            if receipt.oms_order_id:
                state.entry_order = PendingOrderState(
                    oms_order_id=receipt.oms_order_id,
                    submitted_at=now,
                    role=OrderRole.ENTRY.value,
                    requested_qty=plan.quantity,
                    limit_price=order.limit_price if order.limit_price is not None else None,
                    direction=plan.direction,
                    entry_type=plan.entry_type,
                    entry_price=plan.entry_price,
                    planned_stop_price=plan.stop_price,
                    planned_tp1_price=plan.tp1_price,
                    planned_tp2_price=plan.tp2_price,
                    risk_per_share=plan.risk_per_share,
                    risk_dollars=plan.risk_dollars,
                )
                self._pending_plans[receipt.oms_order_id] = plan
                self._order_index[receipt.oms_order_id] = (symbol, "ENTRY")
                self._track_order_submission(receipt.oms_order_id, now)
                self._portfolio.pending_entry_risk[symbol] = plan.risk_dollars
                side = "BUY" if plan.direction == Direction.LONG else "SELL"
                self._log_order_event(
                    order_id=receipt.oms_order_id,
                    symbol=symbol,
                    side=side,
                    status="SUBMITTED",
                    requested_qty=plan.quantity,
                    order_type=order.order_type.value,
                    requested_price=order.limit_price if order.limit_price is not None else None,
                    exchange_timestamp=now,
                )
                self._log_orderbook_context(symbol=symbol, trade_context="entry", exchange_timestamp=now)
                self._diagnostics.log_order(
                    symbol,
                    "submit_entry",
                    {
                        "entry_type": plan.entry_type.value,
                        "direction": plan.direction.value,
                        "qty": plan.quantity,
                        "entry_price": plan.entry_price,
                        "stop_price": plan.stop_price,
                    },
                )
                return True
            self._missing_order_id_error(
                action="submit_entry",
                symbol=symbol,
                context={"qty": plan.quantity, "entry_type": plan.entry_type.value},
                exchange_timestamp=now,
            )
        except Exception as exc:
            logger.error("submit_entry failed for %s: %s", symbol, exc, exc_info=exc)
            self._log_engine_error(
                error_type="submit_entry_failed",
                symbol=symbol,
                exc=exc,
                context={"qty": plan.quantity, "entry_type": plan.entry_type.value},
                exchange_timestamp=now,
            )
        return False

    def _tp_quantities(self, qty: int) -> tuple[int, int]:
        if qty <= 1:
            return 0, 0
        if qty <= 3:
            return max(1, qty // 2), 0
        tp1 = max(1, int(round(qty * self._settings.tp1_fraction)))
        tp2 = max(1, int(round(qty * self._settings.tp2_fraction)))
        if tp1 + tp2 >= qty:
            tp2 = max(0, qty - tp1 - 1)
        return tp1, tp2

    async def _submit_stop(self, symbol: str) -> None:
        if self._oms is None:
            return
        state = self._symbols[symbol]
        item = self._items[symbol]
        position = state.position
        if position is None or position.qty_open <= 0 or self._current_stop_order_id(state):
            return
        try:
            if not position.exit_oca_group:
                position.exit_oca_group = self._new_exit_oca_group(symbol)
            order = build_stop_order(
                item,
                self._account_id,
                position.qty_open,
                position.current_stop,
                position.direction,
                oca_group=position.exit_oca_group,
                oca_type=1,
            )
            receipt = await self._oms.submit_intent(
                Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=order)
            )
            if receipt.oms_order_id:
                submitted_at = datetime.now(timezone.utc)
                state.stop_order = PendingOrderState(
                    oms_order_id=receipt.oms_order_id,
                    submitted_at=submitted_at,
                    role=OrderRole.STOP.value,
                    requested_qty=position.qty_open,
                    stop_price=position.current_stop,
                )
                position.stop_order_id = receipt.oms_order_id
                position.stop_submitted_at = submitted_at
                self._order_index[receipt.oms_order_id] = (symbol, "STOP")
                self._track_order_submission(receipt.oms_order_id, submitted_at)
                self._log_order_event(
                    order_id=receipt.oms_order_id,
                    symbol=symbol,
                    side="SELL" if position.direction == Direction.LONG else "BUY",
                    status="SUBMITTED",
                    requested_qty=position.qty_open,
                    order_type="STOP",
                    requested_price=position.current_stop,
                )
            else:
                self._missing_order_id_error(
                    action="submit_stop",
                    symbol=symbol,
                    context={"qty": position.qty_open, "stop_price": position.current_stop},
                )
        except Exception as exc:
            logger.error("submit_stop failed for %s: %s", symbol, exc, exc_info=exc)
            self._log_engine_error(error_type="submit_stop_failed", symbol=symbol, exc=exc)

    async def _submit_tp_orders(self, symbol: str) -> None:
        if self._oms is None:
            return
        state = self._symbols[symbol]
        item = self._items[symbol]
        position = state.position
        if position is None or position.qty_open <= 1:
            return
        if not position.exit_oca_group:
            position.exit_oca_group = self._new_exit_oca_group(symbol)
        tp1_qty, tp2_qty = self._tp_quantities(position.qty_entry)
        desired_tp1_qty = 0 if position.partial_taken else min(tp1_qty, position.qty_open)
        desired_tp2_qty = 0 if position.tp2_taken else min(tp2_qty, position.qty_open)

        if desired_tp1_qty > 0:
            if state.tp1_order is None:
                try:
                    order = build_tp_order(
                        item,
                        self._account_id,
                        desired_tp1_qty,
                        position.tp1_price,
                        position.direction,
                        "tp1",
                        oca_group=position.exit_oca_group,
                        oca_type=1,
                    )
                    receipt = await self._oms.submit_intent(
                        Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=order)
                    )
                    if receipt.oms_order_id:
                        submitted_at = datetime.now(timezone.utc)
                        state.tp1_order = PendingOrderState(
                            oms_order_id=receipt.oms_order_id,
                            submitted_at=submitted_at,
                            role=OrderRole.TP.value,
                            requested_qty=desired_tp1_qty,
                            limit_price=position.tp1_price,
                        )
                        position.tp1_order_id = receipt.oms_order_id
                        self._order_index[receipt.oms_order_id] = (symbol, "TP1")
                        self._track_order_submission(receipt.oms_order_id, submitted_at)
                    else:
                        self._missing_order_id_error(
                            action="submit_tp1",
                            symbol=symbol,
                            context={"qty": desired_tp1_qty, "price": position.tp1_price},
                        )
                except Exception as exc:
                    logger.error("submit_tp1 failed for %s: %s", symbol, exc, exc_info=exc)
                    self._log_engine_error(
                        error_type="submit_tp1_failed",
                        symbol=symbol,
                        exc=exc,
                        context={"qty": desired_tp1_qty, "price": position.tp1_price},
                    )
            elif state.tp1_order.filled_qty <= 0 and state.tp1_order.requested_qty != desired_tp1_qty:
                await self._replace_tp_order(
                    symbol,
                    label="TP1",
                    order_state=state.tp1_order,
                    new_qty=desired_tp1_qty,
                    new_price=position.tp1_price,
                )

        if desired_tp2_qty > 0:
            if state.tp2_order is None:
                try:
                    order = build_tp_order(
                        item,
                        self._account_id,
                        desired_tp2_qty,
                        position.tp2_price,
                        position.direction,
                        "tp2",
                        oca_group=position.exit_oca_group,
                        oca_type=1,
                    )
                    receipt = await self._oms.submit_intent(
                        Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=order)
                    )
                    if receipt.oms_order_id:
                        submitted_at = datetime.now(timezone.utc)
                        state.tp2_order = PendingOrderState(
                            oms_order_id=receipt.oms_order_id,
                            submitted_at=submitted_at,
                            role=OrderRole.TP.value,
                            requested_qty=desired_tp2_qty,
                            limit_price=position.tp2_price,
                        )
                        position.tp2_order_id = receipt.oms_order_id
                        self._order_index[receipt.oms_order_id] = (symbol, "TP2")
                        self._track_order_submission(receipt.oms_order_id, submitted_at)
                    else:
                        self._missing_order_id_error(
                            action="submit_tp2",
                            symbol=symbol,
                            context={"qty": desired_tp2_qty, "price": position.tp2_price},
                        )
                except Exception as exc:
                    logger.error("submit_tp2 failed for %s: %s", symbol, exc, exc_info=exc)
                    self._log_engine_error(
                        error_type="submit_tp2_failed",
                        symbol=symbol,
                        exc=exc,
                        context={"qty": desired_tp2_qty, "price": position.tp2_price},
                    )
            elif state.tp2_order.filled_qty <= 0 and state.tp2_order.requested_qty != desired_tp2_qty:
                await self._replace_tp_order(
                    symbol,
                    label="TP2",
                    order_state=state.tp2_order,
                    new_qty=desired_tp2_qty,
                    new_price=position.tp2_price,
                )

    async def _replace_tp_order(
        self,
        symbol: str,
        *,
        label: str,
        order_state: PendingOrderState,
        new_qty: int,
        new_price: float,
    ) -> None:
        if self._oms is None or new_qty <= 0:
            return
        try:
            await self._oms.submit_intent(
                Intent(
                    intent_type=IntentType.REPLACE_ORDER,
                    strategy_id=STRATEGY_ID,
                    target_oms_order_id=order_state.oms_order_id,
                    new_qty=new_qty,
                    new_limit_price=new_price,
                )
            )
            submitted_at = datetime.now(timezone.utc)
            order_state.requested_qty = new_qty
            order_state.limit_price = new_price
            order_state.submitted_at = submitted_at
            self._track_order_submission(order_state.oms_order_id, submitted_at)
            state = self._symbols[symbol]
            position = state.position
            self._log_order_event(
                order_id=order_state.oms_order_id,
                symbol=symbol,
                side="SELL" if position and position.direction == Direction.LONG else "BUY",
                status="REPLACE_REQUESTED",
                requested_qty=new_qty,
                order_type=label,
                requested_price=new_price,
                related_trade_id=position.trade_id if position else "",
            )
        except Exception as exc:
            logger.error("replace_%s failed for %s: %s", label.lower(), symbol, exc, exc_info=exc)
            self._log_engine_error(
                error_type=f"replace_{label.lower()}_failed",
                symbol=symbol,
                exc=exc,
                context={"qty": new_qty, "price": new_price},
            )

    async def _replace_stop(self, symbol: str) -> None:
        if self._oms is None:
            return
        state = self._symbols[symbol]
        position = state.position
        stop_order_id = self._current_stop_order_id(state)
        if position is None or not stop_order_id:
            return
        try:
            await self._oms.submit_intent(
                Intent(
                    intent_type=IntentType.REPLACE_ORDER,
                    strategy_id=STRATEGY_ID,
                    target_oms_order_id=stop_order_id,
                    new_qty=position.qty_open,
                    new_stop_price=position.current_stop,
                )
            )
            submitted_at = datetime.now(timezone.utc)
            if state.stop_order is not None:
                state.stop_order.requested_qty = position.qty_open
                state.stop_order.stop_price = position.current_stop
                state.stop_order.submitted_at = submitted_at
            position.stop_submitted_at = submitted_at
            self._track_order_submission(stop_order_id, submitted_at)
            self._log_order_event(
                order_id=stop_order_id,
                symbol=symbol,
                side="SELL" if position.direction == Direction.LONG else "BUY",
                status="REPLACE_REQUESTED",
                requested_qty=position.qty_open,
                order_type="STOP",
                requested_price=position.current_stop,
            )
        except Exception as exc:
            logger.error("replace_stop failed for %s: %s", symbol, exc, exc_info=exc)
            self._log_engine_error(error_type="replace_stop_failed", symbol=symbol, exc=exc)

    async def _cancel_order(self, oms_order_id: str) -> None:
        if self._oms is None:
            return
        symbol, role = self._resolve_order(oms_order_id, {})
        await self._oms.submit_intent(
            Intent(
                intent_type=IntentType.CANCEL_ORDER,
                strategy_id=STRATEGY_ID,
                target_oms_order_id=oms_order_id,
            )
        )
        if not symbol:
            return
        state = self._symbols.get(symbol)
        position = None if state is None else state.position
        side = "BUY"
        requested_qty = 0
        requested_price = None
        if position is not None:
            exit_side = "SELL" if position.direction == Direction.LONG else "BUY"
            entry_side = "BUY" if position.direction == Direction.LONG else "SELL"
            side = exit_side if role in {"STOP", "TP1", "TP2", "EXIT"} else entry_side
        if role == "ENTRY" and state and state.entry_order:
            requested_qty = state.entry_order.requested_qty
            requested_price = state.entry_order.limit_price
        elif role == "TP1" and state and state.tp1_order:
            requested_qty = state.tp1_order.requested_qty
            requested_price = state.tp1_order.limit_price
        elif role == "TP2" and state and state.tp2_order:
            requested_qty = state.tp2_order.requested_qty
            requested_price = state.tp2_order.limit_price
        elif role == "STOP":
            if state and state.stop_order is not None:
                requested_qty = state.stop_order.requested_qty
                requested_price = state.stop_order.stop_price
            elif position:
                requested_qty = position.qty_open
                requested_price = position.current_stop
        elif role == "EXIT" and state and state.exit_order:
            requested_qty = state.exit_order.requested_qty
            requested_price = state.exit_order.limit_price
        self._log_order_event(
            order_id=oms_order_id,
            symbol=symbol,
            side=side,
            status="CANCEL_REQUESTED",
            requested_qty=requested_qty,
            order_type=role or "UNKNOWN",
            requested_price=requested_price,
        )

    async def _cancel_stop(self, symbol: str) -> None:
        state = self._symbols[symbol]
        stop_order_id = self._current_stop_order_id(state)
        if not stop_order_id:
            return
        self._expected_stop_cancels.add(stop_order_id)
        if state.stop_order is not None:
            state.stop_order.cancel_requested = True
        await self._cancel_order(stop_order_id)

    async def _cancel_tp_orders(self, symbol: str) -> None:
        state = self._symbols[symbol]
        for order_state in (state.tp1_order, state.tp2_order):
            if order_state is None or order_state.cancel_requested:
                continue
            order_state.cancel_requested = True
            await self._cancel_order(order_state.oms_order_id)

    async def _submit_market_exit(self, symbol: str, qty: int, role: OrderRole) -> None:
        if self._oms is None:
            return
        state = self._symbols[symbol]
        item = self._items[symbol]
        market = self._markets.get(symbol)
        position = state.position
        if position is None or qty <= 0 or state.exit_order is not None:
            return
        requested_qty = min(qty, position.qty_open)
        if requested_qty <= 0:
            return
        expected_price = 0.0
        if market is not None:
            if position.direction == Direction.LONG:
                expected_price = float(market.bid or market.last_price or 0.0)
            else:
                expected_price = float(market.ask or market.last_price or 0.0)
        try:
            if not position.exit_oca_group:
                position.exit_oca_group = self._new_exit_oca_group(symbol)
            order = build_market_exit(
                item,
                self._account_id,
                requested_qty,
                position.direction,
                role=role,
                oca_group=position.exit_oca_group,
                oca_type=1,
            )
            receipt = await self._oms.submit_intent(
                Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=order)
            )
            if receipt.oms_order_id:
                submitted_at = datetime.now(timezone.utc)
                state.exit_order = PendingOrderState(
                    oms_order_id=receipt.oms_order_id,
                    submitted_at=submitted_at,
                    role=role.value,
                    requested_qty=requested_qty,
                    limit_price=expected_price if expected_price > 0 else None,
                )
                self._order_index[receipt.oms_order_id] = (symbol, role.value)
                self._track_order_submission(receipt.oms_order_id, submitted_at)
                if expected_price > 0:
                    self._exit_price_hints[receipt.oms_order_id] = expected_price
                side = "SELL" if position.direction == Direction.LONG else "BUY"
                self._log_order_event(
                    order_id=receipt.oms_order_id,
                    symbol=symbol,
                    side=side,
                    status="SUBMITTED",
                    requested_qty=requested_qty,
                    order_type="MARKET_EXIT",
                    requested_price=expected_price if expected_price > 0 else None,
                    related_trade_id=position.trade_id,
                )
                self._log_orderbook_context(
                    symbol=symbol,
                    trade_context="exit",
                    related_trade_id=position.trade_id,
                )
            else:
                self._missing_order_id_error(
                    action="submit_market_exit",
                    symbol=symbol,
                    context={"qty": requested_qty, "role": role.value},
                )
        except Exception as exc:
            logger.error("submit_market_exit failed for %s: %s", symbol, exc, exc_info=exc)
            self._log_engine_error(
                error_type="submit_market_exit_failed",
                symbol=symbol,
                exc=exc,
                context={"qty": requested_qty, "role": role.value},
            )

    async def _request_full_exit(self, symbol: str, reason: str) -> None:
        state = self._symbols[symbol]
        position = state.position
        if position is None or position.qty_open <= 0:
            return
        if state.exit_order is not None and state.exit_order.role == OrderRole.EXIT.value:
            return
        state.pending_hard_exit = True
        state.last_transition_reason = reason
        if state.entry_order is not None and not state.entry_order.cancel_requested:
            state.entry_order.cancel_requested = True
            await self._cancel_order(state.entry_order.oms_order_id)
        await self._submit_market_exit(symbol, position.qty_open, OrderRole.EXIT)
        await self._cancel_tp_orders(symbol)
        await self._cancel_stop(symbol)

    async def _flatten_all(self, reason: str) -> None:
        for symbol, state in self._symbols.items():
            if state.position is not None:
                await self._request_full_exit(symbol, reason)
            elif state.entry_order is not None and not state.entry_order.cancel_requested:
                state.entry_order.cancel_requested = True
                await self._cancel_order(state.entry_order.oms_order_id)

    async def _cancel_open_entries(self, reason: str) -> None:
        for symbol, state in self._symbols.items():
            if state.entry_order is None or state.entry_order.cancel_requested:
                continue
            state.entry_order.cancel_requested = True
            state.last_transition_reason = reason
            await self._cancel_order(state.entry_order.oms_order_id)

    async def _request_add(self, symbol: str, now: datetime) -> None:
        state = self._symbols[symbol]
        item = self._items[symbol]
        position = state.position
        if position is None or state.entry_order is not None or state.pending_add:
            return
        store = self._store()
        if not can_add(item, state.campaign, position, store, self._settings):
            return
        stock_regime = classify_4h_regime(store.bars_4h(symbol))
        market_regime, _ = self._live_market_regime(store)
        if determine_intraday_mode(state.campaign.breakout.direction, stock_regime, market_regime) == "DEGRADED":
            return
        entry_price = add_trigger_price(item, state.campaign, store)
        if entry_price is None or state.campaign.breakout is None:
            return
        stop_price = choose_stop(
            EntryType.C_CONTINUATION,
            state.campaign.breakout.direction,
            item,
            state.campaign,
            self._settings,
            stock_regime=stock_regime,
            market_regime=market_regime,
        )
        plan = position_size(
            item,
            entry_price,
            stop_price,
            state.campaign.breakout.direction,
            state.campaign,
            max(state.intraday_score, self._settings.normal_intraday_score_min),
            self._portfolio,
            self._items,
            self._settings,
            entry_type=EntryType.C_CONTINUATION,
            stock_regime=stock_regime,
            market_regime=market_regime,
        )
        if plan is None:
            return
        max_add_risk = max(0.0, position.total_initial_risk_usd * 0.5)
        remaining_campaign_risk = max(
            0.0,
            (position.total_initial_risk_usd * self._settings.max_campaign_risk_mult)
            - state.campaign.campaign_risk_used,
        )
        allowed_risk = min(max_add_risk, remaining_campaign_risk)
        if allowed_risk <= 0 or plan.risk_per_share <= 0:
            return
        plan.quantity = min(plan.quantity, int(allowed_risk // plan.risk_per_share))
        if plan.quantity < 1:
            return
        plan.risk_dollars = plan.quantity * plan.risk_per_share
        state.pending_add = True
        if not await self._submit_entry(symbol, plan, now):
            state.pending_add = False

    async def _manage_position(self, symbol: str, now: datetime) -> None:
        state = self._symbols[symbol]
        item = self._items.get(symbol)
        market = self._markets.get(symbol)
        position = state.position
        if item is None or market is None or position is None:
            return
        store = self._store()
        bars_30m = store.bars_30m(symbol)
        if not bars_30m:
            return
        latest = bars_30m[-1]
        self._update_position_extremes(position, latest.close)
        self._refresh_live_metrics(symbol)
        stock_regime = classify_4h_regime(store.bars_4h(symbol))
        market_regime, _ = self._live_market_regime(store)
        state.last_stock_regime = stock_regime.value
        state.last_market_regime = market_regime.value
        state.campaign.position_open = True
        state.campaign.state = (
            CampaignState.CONTINUATION
            if state.campaign.continuation_enabled
            else CampaignState.POSITION_OPEN
        )

        if position.entry_time.date() < latest.end_time.date() and gap_through_stop(position, latest.open):
            self._diagnostics.log_exit_decision(symbol, "gap_through_stop", {
                "open_price": latest.open,
                "stop_level": position.current_stop,
                "gap_size": abs(latest.open - position.current_stop),
                "current_r": round(position.unrealized_r(latest.close), 4),
                "mfe_r": round(self._mfe_r(position), 4),
                "position_age_bars": len(bars_30m),
            })
            await self._request_full_exit(symbol, "gap_through_stop")
            return

        daily_atr = atr_from_bars(item.daily_bars, 14)
        if position.partial_taken:
            be_stop = breakeven_plus_buffer(position, daily_atr, self._settings)
            new_stop = (
                max(position.current_stop, be_stop)
                if position.direction == Direction.LONG
                else min(position.current_stop, be_stop)
            )
            if new_stop != position.current_stop:
                position.current_stop = new_stop
                await self._replace_stop(symbol)

        if position.partial_taken and market.bars_4h:
            old_stop = position.current_stop
            runner_stop, trail_detail = ratchet_runner_stop(position, market.bars_4h, self._settings, avwap=market.avwap_live or 0.0)
            if runner_stop != position.current_stop:
                self._diagnostics.log_exit_decision(symbol, "ratchet_stop", {
                    "old_stop": old_stop, "new_stop": runner_stop,
                    "runner_qty": position.qty_open,
                    "current_r": round(position.unrealized_r(latest.close), 4),
                    **trail_detail,
                })
                position.current_stop = runner_stop
                await self._replace_stop(symbol)

        prev_continuation = state.campaign.continuation_enabled
        maybe_enable_continuation(state.campaign, latest.close, daily_atr, self._settings)
        if state.campaign.continuation_enabled and not prev_continuation:
            self._diagnostics.log_continuation_check(symbol, {
                "triggered": True,
                "latest_close": latest.close,
                "daily_atr": round(daily_atr, 4),
                "campaign_state": state.campaign.state.value,
            })
        update_dirty_state(state.campaign, latest.close, self._settings, latest.end_time.date(), daily_bars=item.daily_bars)

        if stale_exit_needed(position, now, latest.close, self._settings):
            self._diagnostics.log_exit_decision(symbol, "stale_exit", {
                "days_held": business_days_between(position.entry_time.date(), now.date()) + 1,
                "current_r": round(position.unrealized_r(latest.close), 4),
                "mfe_r": round(self._mfe_r(position), 4),
                "partial_taken": position.partial_taken,
                "stock_regime": state.last_stock_regime or "",
                "market_regime": state.last_market_regime or "",
            })
            await self._request_full_exit(symbol, "stale_exit")
            return

        if (
            state.campaign.profit_funded
            and state.campaign.add_count < self._settings.max_adds
            and not state.pending_hard_exit
        ):
            await self._request_add(symbol, now)

    def _update_position_extremes(self, position: PositionState, price: float) -> None:
        if position.direction == Direction.LONG:
            position.max_favorable_price = max(position.max_favorable_price, price)
            position.max_adverse_price = min(position.max_adverse_price, price)
            return
        position.max_favorable_price = min(position.max_favorable_price, price)
        position.max_adverse_price = max(position.max_adverse_price, price)

    def _realized_delta(self, position: PositionState, price: float, qty: int) -> float:
        if position.direction == Direction.LONG:
            return (price - position.entry_price) * qty
        return (position.entry_price - price) * qty

    def _mfe_r(self, position: PositionState) -> float:
        risk = max(position.initial_risk_per_share, 1e-9)
        if position.direction == Direction.LONG:
            return (position.max_favorable_price - position.entry_price) / risk
        return (position.entry_price - position.max_favorable_price) / risk

    def _mae_r(self, position: PositionState) -> float:
        risk = max(position.initial_risk_per_share, 1e-9)
        if position.direction == Direction.LONG:
            return (position.max_adverse_price - position.entry_price) / risk
        return (position.entry_price - position.max_adverse_price) / risk

    async def _handle_event(self, event) -> None:
        if event.event_type == OMSEventType.FILL:
            await self._handle_fill(event)
        elif event.event_type == OMSEventType.ORDER_FILLED:
            pass  # order-state notification; real execution data comes via FILL
        elif event.event_type == OMSEventType.RISK_HALT:
            await self._handle_risk_halt((event.payload or {}).get("reason", ""))
        elif event.event_type in (
            OMSEventType.ORDER_CANCELLED,
            OMSEventType.ORDER_EXPIRED,
            OMSEventType.ORDER_REJECTED,
        ):
            await self._handle_terminal(event)

    async def _handle_risk_halt(self, reason: str) -> None:
        self._portfolio.halt_new_entries = True
        self._diagnostics.log_order("PORTFOLIO", "risk_halt", {"reason": reason or "OMS risk halt"})
        for state in self._symbols.values():
            if state.entry_order and not state.entry_order.cancel_requested:
                state.entry_order.cancel_requested = True
                await self._cancel_order(state.entry_order.oms_order_id)

    async def _handle_fill(self, event) -> None:
        payload = event.payload or {}
        symbol, role = self._resolve_order(event.oms_order_id, payload)
        plan = self._pending_plans.get(event.oms_order_id, None) if event.oms_order_id else None
        submitted_at = self._order_submission_times.get(event.oms_order_id, None) if event.oms_order_id else None
        if not symbol:
            return

        state = self._symbols.get(symbol)
        item = self._items.get(symbol)
        if state is None or item is None:
            return
        fill_qty = int(float(payload.get("qty", 0.0) or 0.0))
        fill_price = float(payload.get("price", 0.0) or 0.0)
        if fill_qty <= 0:
            return

        if role == "ENTRY":
            entry_order = state.entry_order
            entry_submitted_at = submitted_at or (entry_order.submitted_at if entry_order is not None else None)
            entry_plan = plan or self._plan_from_order_state(symbol, entry_order)
            if entry_plan is None:
                fallback_direction = state.campaign.breakout.direction if state.campaign.breakout else Direction.LONG
                entry_plan = PositionPlan(
                    symbol=symbol,
                    direction=fallback_direction,
                    entry_type=EntryType.A_AVWAP_RETEST,
                    entry_price=fill_price,
                    stop_price=choose_stop(
                        EntryType.A_AVWAP_RETEST,
                        fallback_direction,
                        item,
                        state.campaign,
                        self._settings,
                    ),
                    tp1_price=fill_price,
                    tp2_price=fill_price,
                    quantity=fill_qty,
                    risk_per_share=max(item.tick_size, 0.01),
                    risk_dollars=max(item.tick_size, 0.01) * fill_qty,
                    quality_mult=1.0,
                    regime_mult=1.0,
                    corr_mult=1.0,
                )
            requested_qty = self._requested_qty_from_fill(entry_order, payload, fill_qty)
            _, remaining_entry_qty, entry_terminal, previous_entry_filled = self._apply_fill_to_order(
                entry_order,
                fill_qty=fill_qty,
                requested_qty=requested_qty,
            )
            fill_risk_dollars = self._entry_fill_risk(entry_plan, fill_qty)
            if remaining_entry_qty > 0:
                self._portfolio.pending_entry_risk[symbol] = self._remaining_entry_risk(
                    entry_plan,
                    remaining_entry_qty,
                )
            else:
                self._portfolio.pending_entry_risk.pop(symbol, None)

            if state.position is None:
                position = build_position_from_fill(
                    direction=entry_plan.direction,
                    fill_price=fill_price,
                    fill_qty=fill_qty,
                    stop_price=entry_plan.stop_price,
                    tp1_price=entry_plan.tp1_price,
                    tp2_price=entry_plan.tp2_price,
                    fill_time=event.timestamp,
                    setup_tag=entry_plan.entry_type.value,
                )
                state.position = position
            else:
                position = state.position
                total_cost = (position.entry_price * position.qty_open) + (fill_price * fill_qty)
                position.qty_entry += fill_qty
                position.qty_open += fill_qty
                position.entry_price = total_cost / max(position.qty_open, 1)
                self._update_position_extremes(position, fill_price)

            state.campaign.position_open = True
            state.campaign.state = CampaignState.POSITION_OPEN
            state.campaign.last_entry_type = entry_plan.entry_type
            state.campaign.campaign_risk_used += fill_risk_dollars
            self._portfolio.open_positions[symbol] = position
            position.entry_commission += float(payload.get("commission", 0.0) or 0.0)

            if not state.pending_add and not position.trade_id and self._trade_recorder:
                decision_bar_time = state.last_30m_bar_time or position.entry_time
                item_base_risk = base_risk_fraction(item, self._settings)
                raw_risk_frac = item_base_risk * entry_plan.regime_mult * entry_plan.quality_mult * entry_plan.corr_mult
                clamped_risk_frac = min(max(raw_risk_frac, self._settings.final_risk_min_mult * item_base_risk), self._settings.final_risk_max_mult * item_base_risk)
                execution_timestamps = {
                    "order_submitted_at": (
                        entry_submitted_at.isoformat() if entry_submitted_at is not None else None
                    ),
                    "fill_received_at": position.entry_time.isoformat(),
                }
                try:
                    position.trade_id = await self._trade_recorder.record_entry(
                        strategy_id=STRATEGY_ID,
                        instrument=symbol,
                        direction=position.direction.value,
                        quantity=position.qty_entry,
                        entry_price=Decimal(str(position.entry_price)),
                        entry_ts=position.entry_time,
                        setup_tag=position.setup_tag,
                        entry_type=entry_plan.entry_type.value,
                        meta={
                            "entry_signal": entry_plan.entry_type.value,
                            "entry_signal_id": event.oms_order_id or symbol,
                            "entry_signal_strength": float(state.intraday_score),
                            "market_regime": state.last_market_regime,
                            "stock_regime": state.last_stock_regime,
                            "strategy_params": {
                                "selection_score": item.selection_score,
                                "compression_tier": (
                                    state.campaign.box.tier.value if state.campaign.box is not None else ""
                                ),
                                "stop0": entry_plan.stop_price,
                                "tp1": entry_plan.tp1_price,
                                "tp2": entry_plan.tp2_price,
                            },
                            "signal_factors": self._entry_signal_factors(state),
                            "filter_decisions": self._entry_filter_decisions(
                                item,
                                timing_ok=True,
                                score_ok=True,
                                intraday_score=state.intraday_score,
                                threshold=(
                                    self._settings.normal_intraday_score_min
                                    if state.mode == "NORMAL"
                                    else self._settings.degraded_intraday_score_min
                                ),
                            ),
                            "sizing_inputs": {
                                "qty": position.qty_entry,
                                "entry_price": position.entry_price,
                                "stop_price": entry_plan.stop_price,
                                "risk_dollars": self._entry_fill_risk(entry_plan, position.qty_entry),
                                "regime_mult": entry_plan.regime_mult,
                                "quality_mult": entry_plan.quality_mult,
                                "correlation_mult": entry_plan.corr_mult,
                                "base_risk_fraction": item_base_risk,
                                "final_risk_fraction": round(clamped_risk_frac, 6),
                                "nav": self._portfolio.account_equity,
                            },
                            "portfolio_state": self._portfolio_state_snapshot(),
                            "session_type": self._session_type(position.entry_time),
                            "concurrent_positions": len(self._portfolio.open_positions),
                            "drawdown_pct": self._portfolio.total_pnl_pct,
                            "exchange_timestamp": position.entry_time,
                            "bar_id": self._decision_bar_id(symbol, decision_bar_time),
                            "entry_latency_ms": self._latency_ms(entry_submitted_at, position.entry_time),
                            "execution_timestamps": execution_timestamps,
                            "expected_entry_price": (
                                entry_order.limit_price
                                if entry_order and entry_order.limit_price is not None
                                else position.entry_price
                            ),
                        },
                        account_id=self._account_id,
                    )
                except Exception as exc:
                    logger.error("record_entry failed for %s: %s", symbol, exc, exc_info=exc)
                    self._log_engine_error(
                        error_type="record_entry_failed",
                        symbol=symbol,
                        exc=exc,
                        context={"qty": position.qty_entry, "entry_type": entry_plan.entry_type.value},
                        exchange_timestamp=position.entry_time,
                    )

            self._log_orderbook_context(
                symbol=symbol,
                trade_context="entry",
                related_trade_id=position.trade_id,
                exchange_timestamp=event.timestamp,
            )
            if state.pending_add:
                if previous_entry_filled <= 0:
                    state.campaign.add_count += 1
                if entry_terminal:
                    state.pending_add = False
                    state.entry_order = None
                    self._clear_order_tracking(event.oms_order_id, drop_plan=True, drop_exit_hint=False)
                if self._current_stop_order_id(state):
                    await self._replace_stop(symbol)
                else:
                    await self._submit_stop(symbol)
                return

            await self._sync_exit_orders_after_entry_fill(symbol)
            if entry_terminal:
                state.entry_order = None
                self._clear_order_tracking(event.oms_order_id, drop_plan=True, drop_exit_hint=False)
            return

        position = state.position
        if position is None:
            return

        if state.entry_order is not None and not state.entry_order.cancel_requested:
            state.entry_order.cancel_requested = True
            await self._cancel_order(state.entry_order.oms_order_id)

        exit_order_state: PendingOrderState | None = None
        if role == "TP1":
            exit_order_state = state.tp1_order
        elif role == "TP2":
            exit_order_state = state.tp2_order
        elif role == OrderRole.EXIT.value:
            exit_order_state = state.exit_order
        elif role == "STOP":
            exit_order_state = state.stop_order

        exit_submitted_at = submitted_at
        if exit_submitted_at is None:
            if exit_order_state is not None:
                exit_submitted_at = exit_order_state.submitted_at
            elif role == "STOP":
                exit_submitted_at = position.stop_submitted_at

        expected_exit_price = self._exit_price_hints.get(event.oms_order_id, None)
        if expected_exit_price is None:
            if exit_order_state and exit_order_state.limit_price is not None:
                expected_exit_price = exit_order_state.limit_price
            elif role == "STOP":
                if exit_order_state and exit_order_state.stop_price is not None:
                    expected_exit_price = exit_order_state.stop_price
                elif position.current_stop > 0:
                    expected_exit_price = position.current_stop

        requested_qty = self._requested_qty_from_fill(exit_order_state, payload, fill_qty)
        _, remaining_exit_order_qty, exit_terminal, _ = self._apply_fill_to_order(
            exit_order_state,
            fill_qty=fill_qty,
            requested_qty=requested_qty,
        )
        exit_qty = min(fill_qty, position.qty_open)
        if exit_qty <= 0:
            return

        self._update_position_extremes(position, fill_price)
        position.realized_pnl_usd += self._realized_delta(position, fill_price, exit_qty)
        position.qty_open = max(0, position.qty_open - exit_qty)
        position.exit_commission += float(payload.get("commission", 0.0) or 0.0)

        if role == "TP1":
            self._diagnostics.log_exit_decision(symbol, "TP1", {
                "fill_price": fill_price,
                "tp1_price": position.tp1_price,
                "current_r": round(position.unrealized_r(fill_price), 4),
                "mfe_r": round(self._mfe_r(position), 4),
                "qty_exited": exit_qty, "qty_remaining": position.qty_open,
            })
            position.partial_taken = True
            position.profit_funded = True
            state.campaign.profit_funded = True
            be_stop = breakeven_plus_buffer(position, atr_from_bars(item.daily_bars, 14), self._settings)
            position.current_stop = (
                max(position.current_stop, be_stop)
                if position.direction == Direction.LONG
                else min(position.current_stop, be_stop)
            )
            if exit_terminal:
                state.tp1_order = None
                position.tp1_order_id = ""
                self._clear_order_tracking(event.oms_order_id)
                if position.qty_open > 0:
                    self._drop_stop_order_state(symbol)
                    self._drop_tp_order_state(symbol, "tp2_order")
                    position.exit_oca_group = self._new_exit_oca_group(symbol)
                    await self._submit_stop(symbol)
                    await self._submit_tp_orders(symbol)
            elif position.qty_open > 0:
                if self._current_stop_order_id(state):
                    await self._replace_stop(symbol)
                else:
                    await self._submit_stop(symbol)
        elif role == "TP2":
            self._diagnostics.log_exit_decision(symbol, "TP2", {
                "fill_price": fill_price,
                "tp2_price": position.tp2_price,
                "current_r": round(position.unrealized_r(fill_price), 4),
                "mfe_r": round(self._mfe_r(position), 4),
                "qty_exited": exit_qty, "qty_remaining": position.qty_open,
            })
            position.tp2_taken = True
            if position.qty_open > 0 and self._markets[symbol].bars_4h:
                _mkt = self._markets[symbol]
                position.current_stop = ratchet_runner_stop(position, _mkt.bars_4h, self._settings, avwap=_mkt.avwap_live or 0.0)
            if exit_terminal:
                state.tp2_order = None
                position.tp2_order_id = ""
                self._clear_order_tracking(event.oms_order_id)
                if position.qty_open > 0:
                    self._drop_stop_order_state(symbol)
                    position.exit_oca_group = self._new_exit_oca_group(symbol)
                    await self._submit_stop(symbol)
            elif position.qty_open > 0:
                if self._current_stop_order_id(state):
                    await self._replace_stop(symbol)
                else:
                    await self._submit_stop(symbol)
        elif role == OrderRole.EXIT.value:
            if exit_terminal:
                state.exit_order = None
                self._clear_order_tracking(event.oms_order_id)
        elif role == "STOP":
            self._diagnostics.log_exit_decision(symbol, "STOP", {
                "fill_price": fill_price,
                "stop_level": position.current_stop,
                "current_r": round(position.unrealized_r(fill_price), 4),
                "mfe_r": round(self._mfe_r(position), 4),
                "partial_taken": position.partial_taken,
                "qty_exited": exit_qty, "qty_remaining": position.qty_open,
            })
            if exit_terminal:
                state.stop_order = None
                position.stop_order_id = ""
                position.stop_submitted_at = None
                self._clear_order_tracking(event.oms_order_id)

        if position.qty_open <= 0:
            if self._trade_recorder and position.trade_id:
                realized_r = position.realized_pnl_usd / max(position.total_initial_risk_usd, 1e-9)
                mfe_r = round(self._mfe_r(position), 4)
                mae_r = round(self._mae_r(position), 4)
                try:
                    await self._trade_recorder.record_exit(
                        trade_id=position.trade_id,
                        exit_price=Decimal(str(fill_price)),
                        exit_ts=event.timestamp,
                        exit_reason=role or "EXIT",
                        realized_r=Decimal(str(round(realized_r, 4))),
                        realized_usd=Decimal(str(round(position.realized_pnl_usd, 2))),
                        mfe_r=Decimal(str(mfe_r)),
                        mae_r=Decimal(str(mae_r)),
                        max_adverse_price=Decimal(str(position.max_adverse_price)),
                        max_favorable_price=Decimal(str(position.max_favorable_price)),
                        meta={
                            "exchange_timestamp": event.timestamp,
                            "expected_exit_price": expected_exit_price or fill_price,
                            "exit_latency_ms": self._latency_ms(exit_submitted_at, event.timestamp),
                            "fees_paid": position.entry_commission + position.exit_commission,
                            "session_transitions": [state.last_transition_reason] if state.last_transition_reason else [],
                            "market_regime": state.last_market_regime,
                            "stock_regime": state.last_stock_regime,
                            "exit_context": {
                                "exit_role": role or "EXIT",
                                "realized_r": round(realized_r, 4),
                                "mfe_r": mfe_r,
                                "mfe_capture_pct": round(
                                    realized_r / max(mfe_r, 1e-9) * 100, 1
                                ) if mfe_r > 0 else 0.0,
                                "partial_taken": position.partial_taken,
                                "tp2_taken": position.tp2_taken,
                                "days_held": business_days_between(
                                    position.entry_time.date(), event.timestamp.date()
                                ) + 1,
                                "stop_level_at_exit": position.current_stop,
                                "tp1_price": position.tp1_price,
                                "tp2_price": position.tp2_price,
                            },
                        },
                    )
                except Exception as exc:
                    logger.error("record_exit failed for %s: %s", symbol, exc, exc_info=exc)
                    self._log_engine_error(
                        error_type="record_exit_failed",
                        symbol=symbol,
                        exc=exc,
                        context={"qty": fill_qty, "role": role or "EXIT"},
                        exchange_timestamp=event.timestamp,
                    )
            self._log_orderbook_context(
                symbol=symbol,
                trade_context="exit",
                related_trade_id=position.trade_id,
                exchange_timestamp=event.timestamp,
            )
            if state.entry_order is not None and not state.entry_order.cancel_requested:
                state.entry_order.cancel_requested = True
                await self._cancel_order(state.entry_order.oms_order_id)
            self._portfolio.open_positions.pop(symbol, None)
            state.position = None
            state.stop_order = None
            state.exit_order = None
            state.tp1_order = None
            state.tp2_order = None
            state.pending_hard_exit = False
            state.pending_add = False
            state.campaign.position_open = False
            state.campaign.profit_funded = False
            market = self._markets.get(symbol)
            latest_close = (
                market.last_30m_bar.close
                if market and market.last_30m_bar is not None
                else item.daily_bars[-1].close
            )
            update_dirty_state(state.campaign, latest_close, self._settings, event.timestamp.date(), daily_bars=item.daily_bars)
            if state.campaign.state == CampaignState.POSITION_OPEN:
                state.campaign.state = CampaignState.BREAKOUT if state.campaign.breakout else CampaignState.INACTIVE

    async def _handle_terminal(self, event) -> None:
        payload = event.payload or {}
        symbol, role = self._resolve_order(event.oms_order_id, payload)
        if not symbol:
            return

        state = self._symbols.get(symbol)
        if state is None:
            return

        self._clear_order_tracking(
            event.oms_order_id,
            drop_plan=(role == "ENTRY"),
        )

        if role == "ENTRY":
            if state.entry_order is not None and event.oms_order_id and state.entry_order.oms_order_id != event.oms_order_id:
                return
            state.entry_order = None
            self._portfolio.pending_entry_risk.pop(symbol, None)
            if state.position is not None:
                state.pending_add = False
                return
            self._log_missed(
                symbol=symbol,
                direction=state.campaign.breakout.direction if state.campaign.breakout else Direction.LONG,
                blocked_by="entry_terminal",
                block_reason=getattr(event.event_type, "value", str(event.event_type)),
                exchange_timestamp=event.timestamp,
                state=state,
            )
            return

        if role == "TP1":
            if state.tp1_order is not None and event.oms_order_id and state.tp1_order.oms_order_id != event.oms_order_id:
                return
            state.tp1_order = None
            if state.position is not None:
                state.position.tp1_order_id = ""
            return
        if role == "TP2":
            if state.tp2_order is not None and event.oms_order_id and state.tp2_order.oms_order_id != event.oms_order_id:
                return
            state.tp2_order = None
            if state.position is not None:
                state.position.tp2_order_id = ""
            return
        if role == OrderRole.EXIT.value:
            if state.exit_order is not None and event.oms_order_id and state.exit_order.oms_order_id != event.oms_order_id:
                return
            state.exit_order = None
            return
        if role == "STOP" and state.position is not None:
            current_stop_order_id = self._current_stop_order_id(state)
            if current_stop_order_id and event.oms_order_id and current_stop_order_id != event.oms_order_id:
                self._expected_stop_cancels.discard(event.oms_order_id)
                return
            if event.oms_order_id in self._expected_stop_cancels:
                self._expected_stop_cancels.discard(event.oms_order_id)
                state.stop_order = None
                state.position.stop_order_id = ""
                state.position.stop_submitted_at = None
                return
            state.stop_order = None
            state.position.stop_order_id = ""
            state.position.stop_submitted_at = None
            await self._submit_market_exit(symbol, state.position.qty_open, OrderRole.EXIT)

    def _resolve_order(self, oms_order_id: str | None, payload: dict[str, Any]) -> tuple[str, str]:
        if oms_order_id and oms_order_id in self._order_index:
            return self._order_index[oms_order_id]
        return str(payload.get("symbol", "")).upper(), str(payload.get("role", ""))

    async def _save_state(self, reason: str) -> None:
        persist_intraday_state(self.snapshot_state(), settings=self._settings)
        self._last_save_ts = datetime.now(timezone.utc)
        self._diagnostics.log_decision("STATE_SAVE", {"reason": reason})

    def _restore_order_state(self, symbol: str, state: SymbolRuntimeState) -> None:
        if state.entry_order is not None:
            self._order_index[state.entry_order.oms_order_id] = (symbol, "ENTRY")
            self._track_order_submission(state.entry_order.oms_order_id, state.entry_order.submitted_at)
            if state.entry_order.risk_dollars is not None:
                self._portfolio.pending_entry_risk[symbol] = state.entry_order.risk_dollars
            restored_plan = self._plan_from_order_state(symbol, state.entry_order)
            if restored_plan is not None:
                self._pending_plans[state.entry_order.oms_order_id] = restored_plan
        if state.stop_order is not None:
            self._order_index[state.stop_order.oms_order_id] = (symbol, "STOP")
            self._track_order_submission(state.stop_order.oms_order_id, state.stop_order.submitted_at)
            if state.position is not None:
                state.position.stop_order_id = state.stop_order.oms_order_id
                state.position.stop_submitted_at = state.stop_order.submitted_at
        if state.exit_order is not None:
            self._order_index[state.exit_order.oms_order_id] = (symbol, OrderRole.EXIT.value)
            self._track_order_submission(state.exit_order.oms_order_id, state.exit_order.submitted_at)
        if state.tp1_order is not None:
            self._order_index[state.tp1_order.oms_order_id] = (symbol, "TP1")
            self._track_order_submission(state.tp1_order.oms_order_id, state.tp1_order.submitted_at)
        if state.tp2_order is not None:
            self._order_index[state.tp2_order.oms_order_id] = (symbol, "TP2")
            self._track_order_submission(state.tp2_order.oms_order_id, state.tp2_order.submitted_at)
        if state.stop_order is None and state.position is not None and state.position.stop_order_id:
            self._order_index[state.position.stop_order_id] = (symbol, "STOP")
            self._track_order_submission(state.position.stop_order_id, state.position.stop_submitted_at)

    @classmethod
    def try_load_state(
        cls,
        trade_date: date,
        settings: StrategySettings | None = None,
    ) -> IntradayStateSnapshot | None:
        return load_latest_intraday_state(trade_date, settings=settings or StrategySettings())
