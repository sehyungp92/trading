"""Async state-machine engine for the U.S. ORB strategy."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from libs.oms.models.events import OMSEventType
from libs.oms.models.intent import Intent, IntentType
from libs.oms.models.order import OrderRole

from .config import ET, PROXY_SYMBOLS, STRATEGY_ID, StrategySettings, build_proxy_instruments
from .data import UniversePoolManager
from .diagnostics import JsonlDiagnostics
from .execution import build_entry_order, build_market_exit, build_stock_instrument, build_stop_order
from .indicators import compute_opening_range, compute_rvol, five_min_return, last_n_value
from .models import DangerState, PendingOrderState, PortfolioSnapshot, PositionState, QuoteSnapshot, RegimeSnapshot, State, SymbolContext
from .regime import compute_regime, ewma, from_open_return
from .signals import (
    acceptance_passed,
    apply_gap_policy,
    breakout_triggered,
    candidate_pre_score,
    compute_order_plan,
    exit_signal,
    live_gate_pass,
    quality_score,
    update_acceptance,
    volatility_danger_snapshot,
)

logger = logging.getLogger(__name__)


class USORBEngine:
    """Live ORB engine that stays thin around pure signal helpers."""

    def __init__(
        self,
        oms_service,
        cache: dict[str, Any],
        account_id: str,
        nav: float,
        settings: StrategySettings | None = None,
        trade_recorder=None,
        diagnostics: JsonlDiagnostics | None = None,
        instrumentation=None,
    ) -> None:
        self._oms = oms_service
        self._cache = cache
        self._account_id = account_id
        self._settings = settings or StrategySettings()
        self._trade_recorder = trade_recorder
        self._diagnostics = diagnostics or JsonlDiagnostics(self._settings.diagnostics_dir, enabled=False)
        self._instrumentation = instrumentation
        self._pool = UniversePoolManager(cache, self._settings.scanner_cap)

        self._symbols: dict[str, SymbolContext] = {}
        self._proxies: dict[str, QuoteSnapshot] = {}
        self._proxy_bars = {symbol: [] for symbol in PROXY_SYMBOLS}
        self._proxy_returns = {symbol: 0.0 for symbol in PROXY_SYMBOLS}
        self._flow_ewma = {symbol: 0.0 for symbol in ("SPY", "QQQ")}
        self._portfolio = PortfolioSnapshot(nav=nav)
        self._regime = RegimeSnapshot()
        self._live_pool: set[str] = set(PROXY_SYMBOLS)
        self._order_index: dict[str, tuple[str, str]] = {}
        self._exit_price_hints: dict[str, float] = {}
        self._expected_stop_cancels: set[str] = set()
        self._scan_finalized_date = None
        self._flatten_reason = ""
        self._risk_halted = False
        self._risk_halt_reason = ""

        self._event_queue = None
        self._event_task: asyncio.Task | None = None
        self._pulse_task: asyncio.Task | None = None
        self._running = False

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
        for task in (self._pulse_task, self._event_task):
            if task is None:
                continue
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    def update_scanner_symbols(self, symbols: list[str], now: datetime) -> None:
        et = now.astimezone(ET)
        if et.time() < self._settings.scanner_start or et.time() >= self._settings.forced_flatten:
            return
        shortlist = self._pool.shortlist(symbols)
        self._live_pool = set(shortlist) | set(PROXY_SYMBOLS)
        for symbol in shortlist:
            self._ensure_context(symbol)

    def subscription_instruments(self) -> list:
        instruments = build_proxy_instruments()
        seen = {instrument.symbol for instrument in instruments}
        for symbol in sorted(self._live_pool):
            if symbol in PROXY_SYMBOLS:
                continue
            if symbol not in seen:
                instruments.append(build_stock_instrument(self._ensure_context(symbol).cached))
                seen.add(symbol)
        for ctx in self._symbols.values():
            if ctx.position and ctx.symbol not in seen:
                instruments.append(build_stock_instrument(ctx.cached))
                seen.add(ctx.symbol)
        return instruments

    def on_quote(self, symbol: str, quote: QuoteSnapshot, imbalance_90s: float) -> None:
        normalized = symbol.upper()
        if normalized in PROXY_SYMBOLS:
            self._proxies[normalized] = quote
            if normalized in self._flow_ewma:
                self._flow_ewma[normalized] = ewma(
                    self._flow_ewma[normalized],
                    imbalance_90s,
                    self._settings.flow_alpha,
                )
            return

        ctx = self._ensure_context(normalized)
        ctx.quote = quote
        ctx.imbalance_90s = imbalance_90s
        ctx.spread_pct = quote.spread_pct
        if quote.vwap is not None:
            ctx.vwap = quote.vwap

        if quote.past_limit:
            ctx.past_limit_events.append(quote.ts)
        if quote.ask_past_high:
            ctx.ask_past_high_events.append(quote.ts)
        if quote.bid_past_low:
            ctx.bid_past_low_events.append(quote.ts)
        if quote.is_halted and not ctx.was_halted:
            ctx.halt_events.append(quote.ts)
        if quote.resumed_from_halt or (ctx.was_halted and not quote.is_halted):
            ctx.resume_events.append(quote.ts)
        ctx.was_halted = quote.is_halted

        self._refresh_symbol_danger(ctx, quote.ts)

        if quote.past_limit:
            self._apply_cooldown(ctx, quote.ts, self._settings.halt_cooldown_s, "past_limit")
        elif quote.ask_past_high:
            self._apply_cooldown(ctx, quote.ts, self._settings.ask_past_high_cooldown_s, "ask_past_high")
        elif quote.is_halted or quote.resumed_from_halt:
            self._apply_cooldown(ctx, quote.ts, self._settings.halt_cooldown_s, "halt_or_resume")
        elif ctx.vdm.state in (DangerState.DANGER, DangerState.BLOCKED):
            self._apply_cooldown(
                ctx,
                quote.ts,
                ctx.vdm.cooldown_seconds or self._settings.danger_cooldown_s,
                ctx.vdm.state.value.lower(),
            )

    def on_bar(self, symbol: str, bar) -> None:
        normalized = symbol.upper()
        if normalized in PROXY_SYMBOLS:
            self._proxy_bars[normalized].append(bar)
            self._proxy_bars[normalized] = self._proxy_bars[normalized][-240:]
            self._proxy_returns[normalized] = from_open_return(self._proxy_bars[normalized], bar.close)
            return

        ctx = self._ensure_context(normalized)
        ctx.bars.append(bar)
        total_volume = sum(item.volume for item in ctx.bars)
        total_value = sum(item.dollar_value for item in ctx.bars)
        ctx.vwap = (total_value / total_volume) if total_volume > 0 else ctx.vwap
        ctx.last5m_value = last_n_value(ctx.bars, 5)
        ctx.gap_pct = ((ctx.bars[0].open - ctx.cached.prior_close) / ctx.cached.prior_close) if ctx.cached.prior_close > 0 else 0.0
        ctx.rvol_1m = compute_rvol(bar.volume, ctx.cached.minute_volume_baseline.get(bar.minute_key, 0.0))
        ctx.relative_strength_5m = self._relative_strength(ctx)
        self._refresh_symbol_danger(ctx, bar.ts)

    def get_position_snapshot(self) -> list[dict[str, Any]]:
        snapshots = []
        for ctx in self._symbols.values():
            if not ctx.position or ctx.last_price is None:
                continue
            unrealized_r = (ctx.last_price - ctx.position.entry_price) / ctx.position.initial_risk_per_share if ctx.position.initial_risk_per_share > 0 else 0.0
            snapshots.append(
                {
                    "strategy_type": "strategy_orb",
                    "symbol": ctx.symbol,
                    "direction": "LONG",
                    "entry_price": ctx.position.entry_price,
                    "qty": ctx.position.qty_open,
                    "unrealized_pnl_r": round(unrealized_r, 3),
                }
            )
        return snapshots

    def open_order_count(self) -> int:
        return len(self._order_index)

    def _session_type(self, now: datetime) -> str:
        et_now = now.astimezone(ET).time()
        if et_now < self._settings.entry_start:
            return "OPENING_RANGE"
        if et_now >= self._settings.forced_flatten:
            return "LATE_DAY"
        return "RTH"

    def _portfolio_state_snapshot(self) -> dict[str, Any]:
        return {
            "open_positions": self._portfolio.open_positions,
            "halt_new_entries": self._portfolio.halt_new_entries,
            "flatten_all": self._portfolio.flatten_all,
            "total_pnl_pct": self._portfolio.total_pnl_pct,
            "sectors_in_use": sorted(self._portfolio.sectors_in_use),
            "regime_ok": self._regime.regime_ok,
            "risk_off": self._regime.risk_off,
        }

    def _entry_filter_decisions(self, ctx: SymbolContext) -> list[dict[str, Any]]:
        caution_passed = (
            ctx.vdm.state != DangerState.CAUTION
            or float(ctx.quality_score or 0.0) >= self._settings.caution_quality_min
        )
        return [
            {
                "filter_name": "quality_score_gate",
                "threshold": float(self._settings.minimum_quality_score),
                "actual_value": float(ctx.quality_score or 0.0),
                "passed": float(ctx.quality_score or 0.0) >= self._settings.minimum_quality_score,
            },
            {
                "filter_name": "spread_gate",
                "threshold": float(self._settings.spread_limit_pct),
                "actual_value": float(ctx.spread_pct or 0.0),
                "passed": float(ctx.spread_pct or 0.0) <= self._settings.spread_limit_pct,
            },
            {
                "filter_name": "caution_quality_gate",
                "threshold": float(self._settings.caution_quality_min),
                "actual_value": float(ctx.quality_score or 0.0),
                "passed": caution_passed,
            },
        ]

    def _entry_signal_factors(self, ctx: SymbolContext) -> list[dict[str, Any]]:
        vwap = ctx.vwap or 0.0
        price = ctx.last_price or 0.0
        return [
            {
                "factor_name": "pre_score",
                "factor_value": float(ctx.pre_score or 0.0),
                "threshold": 70.0,
                "contribution": float(ctx.pre_score or 0.0) / 100.0,
            },
            {
                "factor_name": "quality_score",
                "factor_value": float(ctx.quality_score or 0.0),
                "threshold": float(self._settings.minimum_quality_score),
                "contribution": float(ctx.quality_score or 0.0) / 100.0,
            },
            {
                "factor_name": "surge",
                "factor_value": float(ctx.surge or 0.0),
                "threshold": 2.0,
                "contribution": float(ctx.surge or 0.0),
            },
            {
                "factor_name": "rvol_1m",
                "factor_value": float(ctx.rvol_1m or 0.0),
                "threshold": 1.5,
                "contribution": float(ctx.rvol_1m or 0.0),
            },
            {
                "factor_name": "relative_strength_5m",
                "factor_value": float(ctx.relative_strength_5m or 0.0),
                "threshold": 0.0,
                "contribution": float(ctx.relative_strength_5m or 0.0),
            },
            {
                "factor_name": "vwap_distance",
                "factor_value": (price - vwap) / vwap if vwap > 0 else 0.0,
                "threshold": 0.0,
                "contribution": 0.0,
            },
            {
                "factor_name": "imbalance_90s",
                "factor_value": float(ctx.imbalance_90s or 0.0),
                "threshold": 0.0,
                "contribution": float(ctx.imbalance_90s or 0.0),
            },
            {
                "factor_name": "gap_pct",
                "factor_value": float(ctx.gap_pct or 0.0),
                "threshold": 0.0,
                "contribution": float(ctx.gap_pct or 0.0),
            },
            {
                "factor_name": "or_pct",
                "factor_value": float(ctx.or_pct or 0.0),
                "threshold": float(self._settings.min_or_pct),
                "contribution": float(ctx.or_pct or 0.0),
            },
        ]

    def _emit_indicator_snapshot(self, ctx: SymbolContext, now: datetime, decision: str) -> None:
        if self._instrumentation is None:
            return
        try:
            self._instrumentation.on_indicator_snapshot(
                pair=ctx.symbol,
                indicators={
                    "spread_pct": float(ctx.spread_pct or 0.0),
                    "rvol_1m": float(ctx.rvol_1m or 0.0),
                    "relative_strength_5m": float(ctx.relative_strength_5m or 0.0),
                    "pre_score": float(ctx.pre_score or 0.0),
                    "quality_score": float(ctx.quality_score or 0.0),
                    "surge": float(ctx.surge or 0.0),
                    "imbalance_90s": float(ctx.imbalance_90s or 0.0),
                },
                signal_name="us_orb_decision",
                signal_strength=float(ctx.quality_score or ctx.pre_score or 0.0),
                decision=decision,
                strategy_type="strategy_orb",
                exchange_timestamp=now,
                bar_id=now.isoformat(),
                context={
                    "state": ctx.state.value,
                    "vdm_state": ctx.vdm.state.value,
                    "flow_regime": self._regime.flow_regime,
                },
            )
        except Exception:
            pass

    def _log_orderbook_context(
        self,
        *,
        ctx: SymbolContext,
        trade_context: str,
        related_trade_id: str = "",
        exchange_timestamp: datetime | None = None,
    ) -> None:
        if self._instrumentation is None or ctx.quote is None:
            return
        best_bid = float(ctx.quote.bid or 0.0)
        best_ask = float(ctx.quote.ask or 0.0)
        if best_bid <= 0 and ctx.last_price is not None:
            best_bid = float(ctx.last_price)
        if best_ask <= 0 and ctx.last_price is not None:
            best_ask = float(ctx.last_price)
        if best_bid <= 0 or best_ask <= 0:
            return
        bid_depth = float(getattr(ctx.quote, "bid_size", 0.0) or 0.0)
        ask_depth = float(getattr(ctx.quote, "ask_size", 0.0) or 0.0)
        bid_levels = [{"price": best_bid, "size": bid_depth}] if bid_depth > 0 else None
        ask_levels = [{"price": best_ask, "size": ask_depth}] if ask_depth > 0 else None
        try:
            self._instrumentation.on_orderbook_context(
                pair=ctx.symbol,
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

    def _update_position_excursions(self, ctx: SymbolContext) -> None:
        if ctx.position is None or ctx.last_price is None:
            return
        ctx.position.max_favorable_price = max(ctx.position.max_favorable_price, ctx.last_price)
        ctx.position.max_adverse_price = min(ctx.position.max_adverse_price, ctx.last_price)

    def _log_missed(
        self,
        *,
        ctx: SymbolContext,
        blocked_by: str,
        block_reason: str,
        exchange_timestamp: datetime,
        strategy_params: dict[str, Any] | None = None,
        filter_decisions: list[dict[str, Any]] | None = None,
    ) -> None:
        if self._instrumentation is None:
            return
        if filter_decisions is None:
            filter_decisions = self._entry_filter_decisions(ctx)
        try:
            self._instrumentation.log_missed(
                pair=ctx.symbol,
                side="LONG",
                signal="us_orb_breakout",
                signal_id=f"{ctx.symbol}:{blocked_by}:{int(exchange_timestamp.timestamp())}",
                signal_strength=float(ctx.quality_score or ctx.pre_score or 0.0),
                blocked_by=blocked_by,
                block_reason=block_reason,
                strategy_params={
                    "state": ctx.state.value,
                    "vdm_state": ctx.vdm.state.value,
                    "sector": ctx.sector,
                    "planned_entry": ctx.planned_entry,
                    "planned_limit": ctx.planned_limit,
                    "final_stop": ctx.final_stop,
                    "rearms_used": ctx.rearms_used,
                    "size_penalty": ctx.size_penalty,
                    **(strategy_params or {}),
                },
                filter_decisions=filter_decisions,
                session_type=self._session_type(exchange_timestamp),
                concurrent_positions=self._portfolio.open_positions,
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
                side="SELL" if order_type in {"STOP", "MARKET_EXIT"} else "BUY",
                order_type=order_type,
                status=status,
                requested_qty=requested_qty,
                requested_price=requested_price,
                related_trade_id=related_trade_id,
                strategy_type="strategy_orb",
                session=self._session_type(exchange_timestamp or datetime.now(timezone.utc)),
                exchange_timestamp=exchange_timestamp,
            )
        except Exception:
            pass

    async def advance(self, now: datetime) -> None:
        self._recompute_regime()
        if self._scan_finalized_date != now.astimezone(ET).date() and now.astimezone(ET).time() >= self._settings.or_end:
            self._finalize_opening_scan(now)
            self._recompute_regime()

        await self._refresh_portfolio()
        flatten_reason = self._flatten_reason_for(now)
        if flatten_reason:
            await self._flatten_all(flatten_reason)

        for ctx in self._symbols.values():
            await self._advance_symbol(ctx, now)

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

    def _ensure_context(self, symbol: str) -> SymbolContext:
        if symbol not in self._symbols:
            self._symbols[symbol] = SymbolContext(cached=self._cache[symbol])
        return self._symbols[symbol]

    def _refresh_symbol_danger(self, ctx: SymbolContext, now: datetime) -> None:
        previous = ctx.vdm
        ctx.vdm = volatility_danger_snapshot(ctx, now, self._settings)
        if ctx.vdm != previous:
            self._diagnostics.log_vdm(ctx.symbol, ctx.vdm.state.value, ctx.vdm.score, ctx.vdm.reasons)

    def _apply_cooldown(self, ctx: SymbolContext, now: datetime, seconds: int, reason: str) -> None:
        ctx.cooldown_until = now + timedelta(seconds=max(0, seconds))
        ctx.block_reason = reason
        if ctx.position is None and ctx.state not in (State.DONE, State.IN_POSITION, State.ORDER_SENT):
            ctx.state = State.COOLDOWN
            self._diagnostics.log_state(ctx.symbol, ctx.state.value, reason)

    def _relative_strength(self, ctx: SymbolContext) -> float:
        stock_return = five_min_return(ctx.bars)
        spy_return = five_min_return(self._proxy_bars["SPY"]) if self._proxy_bars["SPY"] else 0.0
        qqq_return = five_min_return(self._proxy_bars["QQQ"]) if self._proxy_bars["QQQ"] else 0.0
        benchmark = qqq_return if (ctx.cached.tech_tag or "tech" in ctx.cached.sector.lower()) else max(spy_return, qqq_return)
        return stock_return - benchmark

    def _recompute_regime(self) -> None:
        self._regime = compute_regime(
            symbols=self._symbols,
            proxies=self._proxies,
            proxy_returns=self._proxy_returns,
            flow_ewma=self._flow_ewma,
            settings=self._settings,
            breadth_negative_persistent=False,
            weak_directional_progress=False,
            repeated_vwap_crossings=False,
            poor_leader_follow_through=False,
        )
        self._diagnostics.log_regime(
            {
                "regime_ok": self._regime.regime_ok,
                "risk_off": self._regime.risk_off,
                "leader_count": self._regime.leader_count,
                "flow_regime": self._regime.flow_regime,
            }
        )

    async def _refresh_portfolio(self) -> None:
        strategy_halted = False
        portfolio_halted = False
        try:
            risk_state = await self._oms.get_strategy_risk(STRATEGY_ID)
            self._portfolio.realized_pnl_usd = float(getattr(risk_state, "daily_realized_pnl", 0.0))
            strategy_halted = bool(getattr(risk_state, "halted", False))
        except Exception:
            self._portfolio.realized_pnl_usd = 0.0
        try:
            portfolio_risk = await self._oms.get_portfolio_risk()
            portfolio_halted = bool(getattr(portfolio_risk, "halted", False))
        except Exception:
            portfolio_halted = False

        unrealized = 0.0
        sectors = set()
        open_positions = 0
        for ctx in self._symbols.values():
            if not ctx.position or ctx.last_price is None:
                continue
            unrealized += (ctx.last_price - ctx.position.entry_price) * ctx.position.qty_open
            open_positions += 1
            if ctx.sector:
                sectors.add(ctx.sector)
        self._portfolio.unrealized_pnl_usd = unrealized
        self._portfolio.open_positions = open_positions
        self._portfolio.sectors_in_use = sectors
        self._portfolio.halt_new_entries = (
            strategy_halted
            or portfolio_halted
            or self._portfolio.total_pnl_pct <= self._settings.halt_new_entries_pnl_pct
        )
        self._portfolio.flatten_all = self._portfolio.total_pnl_pct <= self._settings.flatten_all_pnl_pct or self._regime.risk_off

    def _flatten_reason_for(self, now: datetime) -> str:
        et = now.astimezone(ET)
        if et.time() >= self._settings.forced_flatten:
            return "forced_flatten"
        if self._portfolio.total_pnl_pct <= self._settings.flatten_all_pnl_pct:
            return "day_pnl_flatten"
        if self._regime.risk_off:
            return "risk_off"
        return ""

    def _finalize_opening_scan(self, now: datetime) -> None:
        candidates: list[SymbolContext] = []
        for symbol in sorted(self._live_pool):
            if symbol in PROXY_SYMBOLS:
                continue
            ctx = self._ensure_context(symbol)
            or_bars = [
                bar
                for bar in ctx.bars
                if self._settings.or_start <= bar.ts.astimezone(ET).time() <= self._settings.or_end
            ]
            if not or_bars:
                continue

            ctx.or_high, ctx.or_low, ctx.or_mid, ctx.or_pct, ctx.value15 = compute_opening_range(or_bars)
            ctx.surge = (ctx.value15 / ctx.cached.baseline_value15) if ctx.cached.baseline_value15 > 0 else 0.0
            self._refresh_symbol_danger(ctx, now)
            ctx.pre_score = candidate_pre_score(ctx, self._regime)
            allowed, size_penalty, reason = apply_gap_policy(ctx.gap_pct, ctx.pre_score, ctx.spread_pct)
            ctx.size_penalty = size_penalty
            ctx.planned_entry = (ctx.or_high or 0.0) + 0.01

            eligible = (
                ctx.cached.adv20 >= self._settings.secondary_adv_threshold
                and ctx.cached.trend_ok
                and allowed
                and ctx.last_price is not None
                and ctx.last_price >= self._settings.min_price
                and ctx.vwap is not None
                and ctx.last_price >= ctx.vwap
                and ctx.spread_pct <= self._settings.spread_limit_pct
                and ctx.surge >= self._settings.surge_threshold
                and ctx.or_pct is not None
                and self._settings.min_or_pct <= ctx.or_pct <= self._settings.max_or_pct
                and ctx.vdm.state != DangerState.BLOCKED
                and not (ctx.quote and ctx.quote.is_halted)
            )
            if eligible:
                ctx.state = State.CANDIDATE
                ctx.done_reason = ""
                candidates.append(ctx)
                self._diagnostics.log_candidate(ctx.symbol, ctx.pre_score, ctx.surge, ctx.rvol_1m)
                self._emit_indicator_snapshot(ctx, now, "candidate")
            else:
                ctx.state = State.DONE
                ctx.done_reason = reason if not allowed else ctx.vdm.reasons[0] if ctx.vdm.state == DangerState.BLOCKED and ctx.vdm.reasons else "precheck_fail"

        candidates.sort(
            key=lambda ctx: (
                -ctx.pre_score,
                -ctx.surge,
                -ctx.rvol_1m,
                -ctx.imbalance_90s,
                -ctx.relative_strength_5m,
                ctx.spread_pct,
            )
        )
        for ctx in candidates[self._settings.scanner_cap :]:
            ctx.state = State.DONE
            ctx.done_reason = "rank_cut"
        self._scan_finalized_date = now.astimezone(ET).date()

    async def _advance_symbol(self, ctx: SymbolContext, now: datetime) -> None:
        if ctx.state == State.DONE and ctx.position is None:
            return

        self._update_position_excursions(ctx)
        self._refresh_symbol_danger(ctx, now)
        et = now.astimezone(ET)
        if ctx.state == State.COOLDOWN:
            if ctx.cooldown_until and now >= ctx.cooldown_until and et.time() <= self._settings.entry_end:
                ctx.state = State.WAIT_BREAK
                self._diagnostics.log_state(ctx.symbol, ctx.state.value, "cooldown_complete")
            return

        if self._portfolio.flatten_all or self._flatten_reason:
            if ctx.position is None:
                ctx.state = State.DONE
                ctx.done_reason = self._flatten_reason or "flatten_all"
                return

        if ctx.position is None and ctx.vdm.state in (DangerState.DANGER, DangerState.BLOCKED):
            self._apply_cooldown(
                ctx,
                now,
                ctx.vdm.cooldown_seconds or self._settings.danger_cooldown_s,
                ctx.vdm.state.value.lower(),
            )
            return

        if ctx.state == State.CANDIDATE:
            if self._portfolio.halt_new_entries:
                return
            if self._portfolio.open_positions >= self._settings.max_positions:
                return
            if ctx.sector and ctx.sector in self._portfolio.sectors_in_use:
                ctx.state = State.DONE
                ctx.done_reason = "sector_cap"
                self._log_missed(
                    ctx=ctx,
                    blocked_by="sector_cap",
                    block_reason="sector_cap",
                    exchange_timestamp=now,
                )
                return
            if live_gate_pass(ctx, self._regime, now, self._settings):
                ctx.state = State.ARMED
                self._diagnostics.log_state(ctx.symbol, ctx.state.value)
            return

        if ctx.state == State.ARMED:
            if et.time() > self._settings.entry_end:
                ctx.state = State.DONE
                ctx.done_reason = "entry_window_closed"
                return
            if et.time() >= self._settings.entry_start:
                ctx.state = State.WAIT_BREAK
                self._diagnostics.log_state(ctx.symbol, ctx.state.value)
            return

        if ctx.state == State.WAIT_BREAK:
            if et.time() > self._settings.entry_end:
                ctx.state = State.DONE
                ctx.done_reason = "entry_window_closed"
                return
            if not live_gate_pass(ctx, self._regime, now, self._settings):
                return
            if breakout_triggered(ctx):
                ctx.planned_entry = (ctx.or_high or 0.0) + 0.01
                if ctx.vdm.state in (DangerState.DANGER, DangerState.BLOCKED):
                    self._apply_cooldown(
                        ctx,
                        now,
                        ctx.vdm.cooldown_seconds or self._settings.danger_cooldown_s,
                        ctx.vdm.reasons[0] if ctx.vdm.reasons else ctx.vdm.state.value.lower(),
                    )
                    return
                ctx.acceptance.reset()
                ctx.acceptance.break_time = now
                ctx.acceptance.deadline = now + timedelta(seconds=self._settings.acceptance_timeout_s)
                ctx.state = State.WAIT_ACCEPTANCE
                self._diagnostics.log_state(ctx.symbol, ctx.state.value)
            return

        if ctx.state == State.WAIT_ACCEPTANCE:
            update_acceptance(ctx)
            if ctx.vdm.state in (DangerState.DANGER, DangerState.BLOCKED):
                self._apply_cooldown(
                    ctx,
                    now,
                    ctx.vdm.cooldown_seconds or self._settings.danger_cooldown_s,
                    "acceptance_escalation",
                )
            elif acceptance_passed(ctx, now):
                ctx.state = State.READY
                self._diagnostics.log_state(ctx.symbol, ctx.state.value)
            elif ctx.acceptance.deadline and now > ctx.acceptance.deadline:
                ctx.state = State.DONE
                ctx.done_reason = "acceptance_timeout"
                self._log_missed(
                    ctx=ctx,
                    blocked_by="acceptance_timeout",
                    block_reason="acceptance_timeout",
                    exchange_timestamp=now,
                )
            return

        if ctx.state == State.READY:
            if self._portfolio.halt_new_entries or self._risk_halted:
                return
            sector_penalty = bool(ctx.sector and ctx.sector in self._portfolio.sectors_in_use)
            ctx.quality_score = quality_score(ctx, self._regime, sector_penalty)
            ctx.planned_entry, ctx.planned_limit, ctx.final_stop, ctx.qty = compute_order_plan(
                ctx, self._portfolio, self._regime, now, self._settings
            )
            self._emit_indicator_snapshot(ctx, now, "ready")
            if ctx.vdm.state in (DangerState.DANGER, DangerState.BLOCKED):
                self._apply_cooldown(
                    ctx,
                    now,
                    ctx.vdm.cooldown_seconds or self._settings.danger_cooldown_s,
                    "ready_escalation",
                )
                return
            if ctx.quality_score < self._settings.minimum_quality_score or ctx.qty <= 0:
                ctx.state = State.DONE
                ctx.done_reason = "quality_or_qty"
                self._log_missed(
                    ctx=ctx,
                    blocked_by="quality_or_qty",
                    block_reason="quality_or_qty",
                    exchange_timestamp=now,
                    strategy_params={"qty": ctx.qty, "quality_score": ctx.quality_score},
                )
                return
            if ctx.vdm.state == DangerState.CAUTION and ctx.quality_score < self._settings.caution_quality_min:
                ctx.state = State.DONE
                ctx.done_reason = "caution_quality"
                self._log_missed(
                    ctx=ctx,
                    blocked_by="caution_quality",
                    block_reason="caution_quality",
                    exchange_timestamp=now,
                    strategy_params={"quality_score": ctx.quality_score},
                )
                return
            await self._submit_entry(ctx, now)
            return

        if ctx.state == State.ORDER_SENT:
            if ctx.entry_order and ctx.vdm.state in (DangerState.DANGER, DangerState.BLOCKED) and not ctx.entry_order.cancel_requested:
                ctx.entry_order.cancel_requested = True
                await self._cancel_order(ctx.entry_order.oms_order_id)
                return
            if ctx.entry_order and not ctx.entry_order.cancel_requested:
                age = (now - ctx.entry_order.submitted_at).total_seconds()
                if age >= self._settings.entry_ttl_s:
                    ctx.entry_order.cancel_requested = True
                    await self._cancel_order(ctx.entry_order.oms_order_id)
                    self._diagnostics.log_missed_fill(
                        ctx.symbol,
                        {
                            "stop": ctx.entry_order.stop_price,
                            "limit": ctx.entry_order.limit_price,
                            "last": ctx.last_price,
                            "spread_pct": ctx.spread_pct,
                            "reason": "entry_ttl_expired",
                        },
                    )
                    self._log_missed(
                        ctx=ctx,
                        blocked_by="entry_ttl",
                        block_reason="entry_ttl_expired",
                        exchange_timestamp=now,
                        strategy_params={
                            "age_seconds": age,
                            "spread_pct": ctx.spread_pct,
                            "last_price": ctx.last_price,
                        },
                    )
            return

        if ctx.state == State.IN_POSITION:
            action, level = exit_signal(ctx, self._regime, now, self._settings)
            if action == "partial" and ctx.position and ctx.exit_order is None and not ctx.pending_hard_exit:
                qty = max(1, int(ctx.position.qty_open * self._settings.partial_exit_fraction))
                await self._submit_market_exit(ctx, qty, OrderRole.TP)
            elif action == "exit" and ctx.position:
                await self._request_full_exit(ctx)
            elif action == "trail" and ctx.position and level is not None and not ctx.pending_hard_exit and ctx.exit_order is None:
                ctx.position.current_stop = max(ctx.position.current_stop, level)
                await self._replace_stop(ctx)

    async def _submit_entry(self, ctx: SymbolContext, now: datetime) -> None:
        order = build_entry_order(ctx, self._account_id, self._settings)
        receipt = await self._oms.submit_intent(Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=order))
        if receipt.oms_order_id:
            ctx.entry_order = PendingOrderState(
                oms_order_id=receipt.oms_order_id,
                submitted_at=now,
                role="ENTRY",
                requested_qty=ctx.qty,
                limit_price=ctx.planned_limit,
                stop_price=ctx.planned_entry,
            )
            self._order_index[receipt.oms_order_id] = (ctx.symbol, "ENTRY")
            ctx.state = State.ORDER_SENT
            self._diagnostics.log_state(ctx.symbol, ctx.state.value)
            self._log_order_event(
                order_id=receipt.oms_order_id,
                symbol=ctx.symbol,
                status="SUBMITTED",
                requested_qty=ctx.qty,
                order_type="STOP_LIMIT_ENTRY",
                requested_price=ctx.planned_limit,
                exchange_timestamp=now,
            )
            self._log_orderbook_context(
                ctx=ctx,
                trade_context="entry",
                exchange_timestamp=now,
            )
        else:
            ctx.state = State.DONE
            ctx.done_reason = "entry_denied"
            self._log_missed(
                ctx=ctx,
                blocked_by="oms_submit",
                block_reason=receipt.denial_reason or "entry_denied",
                exchange_timestamp=now,
            )

    async def _submit_stop(self, ctx: SymbolContext) -> None:
        if ctx.position is None or ctx.position.qty_open <= 0:
            return
        try:
            order = build_stop_order(ctx, self._account_id, ctx.position.qty_open, ctx.position.current_stop)
            receipt = await self._oms.submit_intent(Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=order))
            if receipt.oms_order_id:
                ctx.position.stop_order_id = receipt.oms_order_id
                self._order_index[receipt.oms_order_id] = (ctx.symbol, "STOP")
                self._log_order_event(
                    order_id=receipt.oms_order_id,
                    symbol=ctx.symbol,
                    status="SUBMITTED",
                    requested_qty=ctx.position.qty_open,
                    order_type="STOP",
                    requested_price=ctx.position.current_stop,
                )
        except Exception as exc:
            logger.error("submit_stop failed for %s: %s", ctx.symbol, exc, exc_info=exc)
            if self._instrumentation:
                try:
                    self._instrumentation.log_error(
                        error_type="submit_stop_failed",
                        message=str(exc),
                        severity="high",
                        category="engine",
                        context={"symbol": ctx.symbol},
                        exc=exc,
                    )
                except Exception:
                    pass

    async def _replace_stop(self, ctx: SymbolContext) -> None:
        if ctx.position is None or not ctx.position.stop_order_id:
            return
        try:
            await self._oms.submit_intent(
                Intent(
                    intent_type=IntentType.REPLACE_ORDER,
                    strategy_id=STRATEGY_ID,
                    target_oms_order_id=ctx.position.stop_order_id,
                    new_qty=ctx.position.qty_open,
                    new_stop_price=ctx.position.current_stop,
                )
            )
            self._log_order_event(
                order_id=ctx.position.stop_order_id,
                symbol=ctx.symbol,
                status="REPLACE_REQUESTED",
                requested_qty=ctx.position.qty_open,
                order_type="STOP",
                requested_price=ctx.position.current_stop,
            )
        except Exception as exc:
            logger.error("replace_stop failed for %s: %s", ctx.symbol, exc, exc_info=exc)
            if self._instrumentation:
                try:
                    self._instrumentation.log_error(
                        error_type="replace_stop_failed",
                        message=str(exc),
                        severity="high",
                        category="engine",
                        context={"symbol": ctx.symbol},
                        exc=exc,
                    )
                except Exception:
                    pass

    async def _cancel_stop(self, ctx: SymbolContext) -> None:
        if ctx.position and ctx.position.stop_order_id:
            self._expected_stop_cancels.add(ctx.position.stop_order_id)
            await self._cancel_order(ctx.position.stop_order_id)

    async def _request_full_exit(self, ctx: SymbolContext) -> None:
        if ctx.position is None or ctx.position.qty_open <= 0:
            return
        if ctx.exit_order is not None:
            if ctx.exit_order.role == OrderRole.EXIT.value:
                return
            ctx.pending_hard_exit = True
            if not ctx.exit_order.cancel_requested:
                ctx.exit_order.cancel_requested = True
                await self._cancel_order(ctx.exit_order.oms_order_id)
            return
        await self._cancel_stop(ctx)
        await self._submit_market_exit(ctx, ctx.position.qty_open, OrderRole.EXIT)

    async def _submit_market_exit(self, ctx: SymbolContext, qty: int, role: OrderRole) -> None:
        if ctx.position is None or qty <= 0 or ctx.exit_order is not None:
            return
        requested_qty = min(qty, ctx.position.qty_open)
        if requested_qty <= 0:
            return
        expected_exit_price = 0.0
        if ctx.quote is not None:
            expected_exit_price = float(ctx.quote.bid or ctx.last_price or 0.0)
        elif ctx.last_price is not None:
            expected_exit_price = float(ctx.last_price)
        try:
            order = build_market_exit(ctx, self._account_id, requested_qty, role)
            receipt = await self._oms.submit_intent(Intent(intent_type=IntentType.NEW_ORDER, strategy_id=STRATEGY_ID, order=order))
            if receipt.oms_order_id:
                ctx.exit_order = PendingOrderState(
                    oms_order_id=receipt.oms_order_id,
                    submitted_at=datetime.now(timezone.utc),
                    role=role.value,
                    requested_qty=requested_qty,
                    limit_price=expected_exit_price if expected_exit_price > 0 else None,
                )
                self._order_index[receipt.oms_order_id] = (ctx.symbol, role.value)
                if expected_exit_price > 0:
                    self._exit_price_hints[receipt.oms_order_id] = expected_exit_price
                self._log_order_event(
                    order_id=receipt.oms_order_id,
                    symbol=ctx.symbol,
                    status="SUBMITTED",
                    requested_qty=requested_qty,
                    order_type="MARKET_EXIT",
                    requested_price=expected_exit_price if expected_exit_price > 0 else None,
                    related_trade_id=ctx.position.trade_id if ctx.position else "",
                )
                self._log_orderbook_context(
                    ctx=ctx,
                    trade_context="exit",
                    related_trade_id=ctx.position.trade_id if ctx.position else "",
                )
        except Exception as exc:
            logger.error("submit_market_exit failed for %s: %s", ctx.symbol, exc, exc_info=exc)
            if self._instrumentation:
                try:
                    self._instrumentation.log_error(
                        error_type="submit_market_exit_failed",
                        message=str(exc),
                        severity="high",
                        category="engine",
                        context={"symbol": ctx.symbol, "qty": requested_qty, "role": role.value},
                        exc=exc,
                    )
                except Exception:
                    pass

    async def _cancel_order(self, oms_order_id: str) -> None:
        symbol, role = self._resolve_order(oms_order_id, {})
        requested_qty = 0
        requested_price = None
        if symbol:
            ctx = self._ensure_context(symbol)
            if role == "ENTRY" and ctx.entry_order:
                requested_qty = ctx.entry_order.requested_qty
                requested_price = ctx.entry_order.limit_price
            elif role == "STOP" and ctx.position:
                requested_qty = ctx.position.qty_open
                requested_price = ctx.position.current_stop
            elif role in {OrderRole.TP.value, OrderRole.EXIT.value} and ctx.exit_order:
                requested_qty = ctx.exit_order.requested_qty
                requested_price = ctx.exit_order.limit_price
            elif ctx.position:
                requested_qty = ctx.position.qty_open
                requested_price = self._exit_price_hints.get(oms_order_id)
        self._log_order_event(
            order_id=oms_order_id,
            symbol=symbol or "UNKNOWN",
            status="CANCEL_REQUESTED",
            requested_qty=requested_qty,
            order_type=role or "UNKNOWN",
            requested_price=requested_price,
        )
        await self._oms.submit_intent(
            Intent(intent_type=IntentType.CANCEL_ORDER, strategy_id=STRATEGY_ID, target_oms_order_id=oms_order_id)
        )

    async def _flatten_all(self, reason: str) -> None:
        if self._flatten_reason == reason:
            return
        self._flatten_reason = reason
        for ctx in self._symbols.values():
            if ctx.position and ctx.position.stop_order_id:
                self._expected_stop_cancels.add(ctx.position.stop_order_id)
                await self._cancel_order(ctx.position.stop_order_id)
        await self._oms.submit_intent(Intent(intent_type=IntentType.FLATTEN, strategy_id=STRATEGY_ID))
        for ctx in self._symbols.values():
            if ctx.state != State.DONE:
                ctx.done_reason = reason

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
        if self._risk_halted:
            return
        self._risk_halted = True
        self._risk_halt_reason = reason or "OMS risk halt"
        self._portfolio.halt_new_entries = True
        for ctx in self._symbols.values():
            if ctx.entry_order and not ctx.entry_order.cancel_requested:
                ctx.entry_order.cancel_requested = True
                await self._cancel_order(ctx.entry_order.oms_order_id)

    async def _handle_fill(self, event) -> None:
        payload = event.payload or {}
        symbol, role = self._resolve_order(event.oms_order_id, payload)
        if event.oms_order_id:
            self._order_index.pop(event.oms_order_id, None)
        if not symbol:
            return
        ctx = self._ensure_context(symbol)
        fill_qty = int(float(payload.get("qty", 0.0) or 0.0))
        fill_price = float(payload.get("price", 0.0) or 0.0)
        role = role or payload.get("role", "")
        expected_exit_price = self._exit_price_hints.pop(event.oms_order_id, None)

        if role == "ENTRY":
            entry_order = ctx.entry_order
            risk_per_share = max(0.01, fill_price - (ctx.final_stop or fill_price - 0.01))
            if ctx.position is None:
                ctx.position = PositionState(
                    entry_price=fill_price,
                    qty_entry=fill_qty,
                    qty_open=fill_qty,
                    final_stop=ctx.final_stop or (fill_price - risk_per_share),
                    current_stop=ctx.final_stop or (fill_price - risk_per_share),
                    entry_time=event.timestamp,
                    initial_risk_per_share=risk_per_share,
                    max_favorable_price=fill_price,
                    max_adverse_price=fill_price,
                )
                if self._trade_recorder:
                    ctx.position.trade_id = await self._trade_recorder.record_entry(
                        strategy_id=STRATEGY_ID,
                        instrument=ctx.symbol,
                        direction="LONG",
                        quantity=fill_qty,
                        entry_price=Decimal(str(fill_price)),
                        entry_ts=event.timestamp,
                        setup_tag="orb",
                        entry_type="stop_limit",
                        meta={
                            "entry_signal": "us_orb_breakout",
                            "entry_signal_id": event.oms_order_id or ctx.symbol,
                            "entry_signal_strength": float(ctx.quality_score or ctx.pre_score or 0.0),
                            "strategy_params": {
                                "pre_score": ctx.pre_score,
                                "quality_score": ctx.quality_score,
                                "surge": ctx.surge,
                                "rvol_1m": ctx.rvol_1m,
                                "relative_strength_5m": ctx.relative_strength_5m,
                                "stop0": ctx.final_stop,
                            },
                            "signal_factors": self._entry_signal_factors(ctx),
                            "filter_decisions": self._entry_filter_decisions(ctx),
                            "sizing_inputs": {
                                "qty": fill_qty,
                                "planned_entry": ctx.planned_entry,
                                "planned_limit": ctx.planned_limit,
                                "final_stop": ctx.final_stop,
                            },
                            "portfolio_state": self._portfolio_state_snapshot(),
                            "session_type": self._session_type(event.timestamp),
                            "exchange_timestamp": event.timestamp,
                            "expected_entry_price": (
                                entry_order.limit_price
                                if entry_order and entry_order.limit_price is not None
                                else (ctx.planned_limit or ctx.planned_entry)
                            ),
                            "concurrent_positions": self._portfolio.open_positions,
                            "drawdown_pct": self._portfolio.total_pnl_pct,
                            "bar_id": f"{ctx.symbol}:{event.timestamp.strftime('%Y%m%dT%H%M%S')}",
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
                    self._log_orderbook_context(
                        ctx=ctx,
                        trade_context="entry",
                        related_trade_id=ctx.position.trade_id,
                        exchange_timestamp=event.timestamp,
                    )
                ctx.position.entry_commission = float(payload.get("commission", 0.0) or 0.0)
            else:
                total_cost = (ctx.position.entry_price * ctx.position.qty_open) + (fill_price * fill_qty)
                ctx.position.qty_entry += fill_qty
                ctx.position.qty_open += fill_qty
                ctx.position.entry_price = total_cost / ctx.position.qty_open
                ctx.position.max_favorable_price = max(ctx.position.max_favorable_price, fill_price)
                ctx.position.max_adverse_price = min(ctx.position.max_adverse_price, fill_price)

            ctx.entry_order = None
            ctx.exit_order = None
            ctx.pending_hard_exit = False
            ctx.state = State.IN_POSITION
            if ctx.position.stop_order_id:
                await self._replace_stop(ctx)
            else:
                await self._submit_stop(ctx)
            return

        if ctx.position is None:
            return

        exit_order = ctx.exit_order
        if ctx.exit_order and event.oms_order_id == ctx.exit_order.oms_order_id:
            ctx.exit_order = None

        ctx.position.max_favorable_price = max(ctx.position.max_favorable_price, fill_price)
        ctx.position.max_adverse_price = min(ctx.position.max_adverse_price, fill_price)
        exit_qty = fill_qty if fill_qty > 0 else ctx.position.qty_open
        ctx.position.realized_pnl_usd += (fill_price - ctx.position.entry_price) * exit_qty
        ctx.position.qty_open = max(0, ctx.position.qty_open - exit_qty)

        if role == "TP":
            ctx.position.partial_taken = True
            ctx.position.current_stop = max(ctx.position.current_stop, ctx.position.entry_price - ctx.cached.tick_size)
            if ctx.pending_hard_exit and ctx.position.qty_open > 0:
                ctx.pending_hard_exit = False
                await self._cancel_stop(ctx)
                await self._submit_market_exit(ctx, ctx.position.qty_open, OrderRole.EXIT)
            else:
                await self._replace_stop(ctx)
        elif role == "EXIT" and ctx.position.qty_open > 0 and not ctx.position.stop_order_id:
            await self._submit_stop(ctx)
        elif role == "STOP":
            ctx.position.stop_order_id = ""

        if ctx.position.qty_open <= 0:
            if self._trade_recorder and ctx.position.trade_id:
                realized_r = (
                    ctx.position.realized_pnl_usd / ctx.position.total_initial_risk_usd
                    if ctx.position.total_initial_risk_usd > 0
                    else 0.0
                )
                await self._trade_recorder.record_exit(
                    trade_id=ctx.position.trade_id,
                    exit_price=Decimal(str(fill_price)),
                    exit_ts=event.timestamp,
                    exit_reason=role or "EXIT",
                    realized_r=Decimal(str(round(realized_r, 4))),
                    realized_usd=Decimal(str(round(ctx.position.realized_pnl_usd, 2))),
                    mfe_r=Decimal(str(round(
                        (ctx.position.max_favorable_price - ctx.position.entry_price) / max(ctx.position.initial_risk_per_share, 1e-9),
                        4,
                    ))),
                    mae_r=Decimal(str(round(
                        (ctx.position.max_adverse_price - ctx.position.entry_price) / max(ctx.position.initial_risk_per_share, 1e-9),
                        4,
                    ))),
                    max_adverse_price=Decimal(str(ctx.position.max_adverse_price)),
                    max_favorable_price=Decimal(str(ctx.position.max_favorable_price)),
                    meta={
                        "exchange_timestamp": event.timestamp,
                        "expected_exit_price": expected_exit_price or fill_price,
                        "fees_paid": float(payload.get("commission", 0.0) or 0.0) + getattr(ctx.position, 'entry_commission', 0.0),
                        "session_transitions": [],
                        "exit_latency_ms": (
                            int((event.timestamp - exit_order.submitted_at).total_seconds() * 1000)
                            if exit_order and exit_order.submitted_at
                            else None
                        ),
                    },
                )
            self._log_orderbook_context(
                ctx=ctx,
                trade_context="exit",
                related_trade_id=ctx.position.trade_id,
                exchange_timestamp=event.timestamp,
            )
            ctx.position = None
            ctx.exit_order = None
            ctx.pending_hard_exit = False
            ctx.state = State.DONE
            ctx.done_reason = f"filled_{role.lower()}" if role else "filled_exit"

    async def _handle_terminal(self, event) -> None:
        payload = event.payload or {}
        symbol, role = self._resolve_order(event.oms_order_id, payload)
        if event.oms_order_id:
            self._order_index.pop(event.oms_order_id, None)
        if not symbol:
            return
        if event.oms_order_id:
            self._exit_price_hints.pop(event.oms_order_id, None)
        ctx = self._ensure_context(symbol)

        if role == "ENTRY" and ctx.state == State.ORDER_SENT:
            ctx.entry_order = None
            if self._flatten_reason:
                ctx.state = State.DONE
                ctx.done_reason = self._flatten_reason
            elif ctx.cooldown_until and event.timestamp < ctx.cooldown_until:
                ctx.state = State.COOLDOWN
                self._diagnostics.log_state(ctx.symbol, ctx.state.value, ctx.block_reason)
            elif ctx.vdm.state in (DangerState.DANGER, DangerState.BLOCKED):
                self._apply_cooldown(
                    ctx,
                    event.timestamp,
                    ctx.vdm.cooldown_seconds or self._settings.danger_cooldown_s,
                    ctx.vdm.reasons[0] if ctx.vdm.reasons else "entry_risk_escalation",
                )
            elif ctx.rearms_used < self._settings.max_rearms:
                ctx.rearms_used += 1
                ctx.state = State.WAIT_BREAK
            else:
                ctx.state = State.DONE
                ctx.done_reason = "entry_terminal"
            self._log_missed(
                ctx=ctx,
                blocked_by="entry_terminal",
                block_reason=getattr(event.event_type, "value", str(event.event_type)),
                exchange_timestamp=event.timestamp,
            )
            return

        if role in {OrderRole.TP.value, OrderRole.EXIT.value}:
            pending_hard_exit = ctx.pending_hard_exit
            if ctx.exit_order and event.oms_order_id == ctx.exit_order.oms_order_id:
                ctx.exit_order = None
            ctx.pending_hard_exit = False
            if pending_hard_exit and ctx.position and ctx.position.qty_open > 0:
                await self._cancel_stop(ctx)
                await self._submit_market_exit(ctx, ctx.position.qty_open, OrderRole.EXIT)
            elif role == OrderRole.EXIT.value and ctx.position and ctx.position.qty_open > 0 and not ctx.position.stop_order_id:
                await self._submit_stop(ctx)
            return

        if role == "STOP" and ctx.position and ctx.position.qty_open > 0:
            if event.oms_order_id in self._expected_stop_cancels:
                self._expected_stop_cancels.discard(event.oms_order_id)
                ctx.position.stop_order_id = ""
                return
            ctx.position.stop_order_id = ""
            await self._submit_market_exit(ctx, ctx.position.qty_open, OrderRole.EXIT)

    def _resolve_order(self, oms_order_id: str | None, payload: dict[str, Any]) -> tuple[str, str]:
        if oms_order_id and oms_order_id in self._order_index:
            return self._order_index[oms_order_id]
        return str(payload.get("symbol", "")).upper(), str(payload.get("role", ""))
