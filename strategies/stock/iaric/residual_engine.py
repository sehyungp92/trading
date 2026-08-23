"""Thin live OMS adapter for the shared IARIC daily residual core."""
from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime, time, timezone
from decimal import Decimal
from typing import Any

from libs.oms.models.events import OMSEventType
from libs.oms.models.intent import Intent, IntentType
from libs.oms.models.order import OrderRole
from strategies.core.actions import (
    CancelAction,
    ReplaceProtectiveStop,
    SubmitEntry,
    SubmitMarketExit,
    SubmitPartialExit,
    SubmitProtectiveStop,
)

from .artifact_store import persist_intraday_state
from .config import ET, STRATEGY_ID, StrategySettings
from .core.daily_residual import (
    DAILY_RESIDUAL_SLEEVE,
    DailyResidualExecutionState,
    DailyResidualFill,
    DailyResidualSymbolState,
    apply_daily_residual_fill,
    build_daily_residual_execution_state,
    plan_daily_residual_forced_exit,
    plan_daily_residual_session_orders,
)
from .diagnostics import JsonlDiagnostics
from .execution import (
    build_market_entry,
    build_market_exit,
    build_stock_instrument,
    build_stop_order,
)
from .models import Bar, IntradayStateSnapshot, QuoteSnapshot, WatchlistArtifact

logger = logging.getLogger(__name__)


class IARICDailyResidualEngine:
    """Production adapter; all economic transitions live in ``core``."""

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
        if artifact.strategy_mode != DAILY_RESIDUAL_SLEEVE:
            raise ValueError("residual engine requires a daily residual artifact")
        if artifact.selection_contract_version not in {
            "daily_residual_shared_selector_v2",
            "daily_residual_shared_selector_v3",
            "daily_residual_shared_selector_v4",
        }:
            raise ValueError("unsupported or missing residual selection contract")
        if artifact.strategy_parameters.get("entry_clock") != "next_session_open":
            raise ValueError("live residual engine only supports next-session-open entry")
        if artifact.strategy_parameters.get("universe_contract") != (
            "frozen_98_intraday_symbols_only"
        ):
            raise ValueError("live residual artifact does not use the frozen-98 contract")

        self._oms = oms_service
        self._artifact = artifact
        self._account_id = account_id
        self._settings = settings or StrategySettings()
        self._trade_recorder = trade_recorder
        self._diagnostics = diagnostics or JsonlDiagnostics(
            self._settings.diagnostics_dir, enabled=False
        )
        self._instrumentation = instrumentation
        self._disable_background_tasks = bool(disable_background_tasks)
        self._state = build_daily_residual_execution_state(
            artifact,
            nav=float(nav),
            catastrophic_stop_atr=float(
                self._settings.daily_residual_catastrophic_stop_atr
            ),
            catastrophic_stop_residual_r=float(
                self._settings.daily_residual_catastrophic_stop_residual_r
            ),
        )
        self._decision_events: list[Any] = []
        self._client_to_oms: dict[str, str] = {}
        self._oms_to_order: dict[str, tuple[str, str, str]] = {}
        self._last_quotes: dict[str, QuoteSnapshot] = {}
        self._last_5m_bar: dict[str, datetime] = {}
        self._event_queue = None
        self._event_task: asyncio.Task | None = None
        self._pulse_task: asyncio.Task | None = None
        self._running = False
        self._last_save_ts: datetime | None = None
        self._last_risk_refresh: datetime | None = None
        self._risk_halted = False
        self._last_error = ""

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
        await self._save_state("stop")
        for task in (self._pulse_task, self._event_task):
            if task is None:
                continue
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    def subscription_instruments(self) -> list:
        return [
            build_stock_instrument(self._state.symbols[symbol])
            for symbol in sorted(self._state.symbols)
        ]

    def polling_instruments(self) -> list[tuple[Any, int]]:
        # Daily residual decisions are formed by the nightly completed-session
        # artifact; no five-minute polling is required for the core sleeve.
        return []

    def on_quote(self, symbol: str, quote: QuoteSnapshot) -> None:
        if symbol in self._state.symbols:
            self._last_quotes[symbol] = quote

    def on_completed_5m_bar(
        self,
        symbol: str,
        bar: Bar,
        *,
        received_at: datetime | None = None,
    ) -> bool:
        # Market data remains useful for liveness/MTM but cannot mutate the
        # nightly residual decision or manufacture an intraday entry.
        if symbol in self._state.symbols:
            self._last_5m_bar[symbol] = bar.end_time
        return True

    def flush_completed_5m_batch(self, end_time: datetime) -> int:
        return 0

    async def advance(self, now: datetime) -> None:
        await self._refresh_risk(now)
        local = now.astimezone(ET)
        if local.date() == self._state.trade_date and local.time() >= self._settings.premarket_start:
            if not self._state.session_orders_planned:
                allow_entries = local.time() < self._settings.market_open and not self._risk_halted
                self._state, actions, events = plan_daily_residual_session_orders(
                    self._state,
                    ts=now,
                    allow_entries=allow_entries,
                )
                self._record_events(events)
                await self._execute_actions(actions, now)
        if self._last_save_ts is None or (now - self._last_save_ts).total_seconds() >= 60:
            await self._save_state("interval")

    async def _refresh_risk(self, now: datetime) -> None:
        if self._last_risk_refresh and (now - self._last_risk_refresh).total_seconds() < 5:
            return
        self._last_risk_refresh = now
        try:
            strategy_risk, portfolio_risk = await asyncio.gather(
                self._oms.get_strategy_risk(STRATEGY_ID),
                self._oms.get_portfolio_risk(),
            )
            self._risk_halted = bool(
                getattr(strategy_risk, "halted", False)
                or getattr(portfolio_risk, "halted", False)
            )
        except Exception as exc:
            # Preserve the existing live infrastructure's fail-closed posture:
            # inability to verify risk blocks new entries, never management.
            self._risk_halted = True
            self._last_error = f"risk_refresh_failed:{type(exc).__name__}"

    async def _execute_actions(self, actions, now: datetime) -> None:
        for action in actions:
            try:
                if isinstance(action, CancelAction):
                    target = self._client_to_oms.get(
                        action.target_order_id, action.target_order_id
                    )
                    if target:
                        await self._oms.submit_intent(
                            Intent(
                                intent_type=IntentType.CANCEL_ORDER,
                                strategy_id=STRATEGY_ID,
                                target_oms_order_id=target,
                            )
                        )
                    continue
                if isinstance(action, ReplaceProtectiveStop):
                    target = self._client_to_oms.get(
                        action.target_order_id, action.target_order_id
                    )
                    if not target:
                        raise RuntimeError("protective stop replace target is unavailable")
                    await self._oms.submit_intent(
                        Intent(
                            intent_type=IntentType.REPLACE_ORDER,
                            strategy_id=STRATEGY_ID,
                            target_oms_order_id=target,
                            new_qty=action.qty,
                            new_stop_price=action.stop_price,
                        )
                    )
                    continue

                symbol_state = self._state.symbols[action.symbol]
                if isinstance(action, SubmitEntry):
                    order = build_market_entry(
                        symbol_state,
                        self._account_id,
                        action.qty,
                        float(action.risk_context["planned_entry_price"]),
                        float(action.risk_context["stop_for_risk"]),
                        client_order_id=action.client_order_id,
                        exchange_timestamp=now,
                    )
                    role = "ENTRY"
                elif isinstance(action, SubmitProtectiveStop):
                    order = build_stop_order(
                        symbol_state,
                        self._account_id,
                        action.qty,
                        action.stop_price,
                        oca_group=action.oca_group,
                    )
                    order.client_order_id = action.client_order_id
                    role = "STOP"
                elif isinstance(action, (SubmitPartialExit, SubmitMarketExit)):
                    order = build_market_exit(
                        symbol_state,
                        self._account_id,
                        action.qty,
                        OrderRole.EXIT,
                        oca_group=action.oca_group,
                    )
                    order.client_order_id = action.client_order_id
                    role = "PARTIAL_EXIT" if isinstance(action, SubmitPartialExit) else "EXIT"
                else:
                    raise TypeError(f"unsupported residual neutral action {type(action).__name__}")
                receipt = await self._oms.submit_intent(
                    Intent(
                        intent_type=IntentType.NEW_ORDER,
                        strategy_id=STRATEGY_ID,
                        order=order,
                    )
                )
                if not receipt.oms_order_id:
                    raise RuntimeError(receipt.denial_reason or "OMS denied residual order")
                self._client_to_oms[action.client_order_id] = receipt.oms_order_id
                self._oms_to_order[receipt.oms_order_id] = (
                    action.symbol,
                    role,
                    action.client_order_id,
                )
                self._diagnostics.log_order(
                    action.symbol,
                    "submit_residual_order",
                    {
                        "role": role,
                        "qty": action.qty,
                        "client_order_id": action.client_order_id,
                        "oms_order_id": receipt.oms_order_id,
                    },
                )
            except Exception as exc:
                self._last_error = f"action_failed:{type(exc).__name__}:{exc}"
                self._diagnostics.log_degraded(
                    "daily_residual_execution",
                    self._last_error,
                    {"action": type(action).__name__},
                )
                logger.error("IARIC residual action failed: %s", exc, exc_info=exc)
                if isinstance(action, (SubmitProtectiveStop, ReplaceProtectiveStop)):
                    await self._emergency_flatten(
                        action.symbol,
                        now,
                        reason="protective_stop_submission_failed",
                    )

    async def _emergency_flatten(
        self,
        symbol: str,
        ts: datetime,
        *,
        reason: str,
    ) -> None:
        symbol_state = self._state.symbols.get(symbol)
        position = symbol_state.position if symbol_state else None
        if symbol_state is None or position is None or position.qty_open <= 0:
            return
        if symbol_state.pending_remaining_qty > 0:
            return
        self._state, action, event = plan_daily_residual_forced_exit(
            self._state,
            symbol=symbol,
            ts=ts,
            reason=reason,
        )
        self._record_events((event,))
        await self._execute_actions((action,), ts)

    async def _event_loop(self) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            try:
                if event.event_type == OMSEventType.FILL:
                    await self._handle_fill(event)
                elif event.event_type == OMSEventType.RISK_HALT:
                    self._risk_halted = True
                elif event.event_type in {
                    OMSEventType.ORDER_CANCELLED,
                    OMSEventType.ORDER_EXPIRED,
                    OMSEventType.ORDER_REJECTED,
                }:
                    await self._handle_terminal(event)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                # An unresolved broker event is an execution-authority failure.
                # Preserve the event loop for subsequent stop/fill events while
                # failing closed to new risk and surfacing the reconciliation
                # requirement in health/diagnostics.
                self._risk_halted = True
                self._last_error = (
                    "oms_event_reconciliation_required:"
                    f"{type(exc).__name__}:{exc}"
                )
                self._diagnostics.log_degraded(
                    "daily_residual_oms_event",
                    self._last_error,
                    {
                        "event_type": str(getattr(event, "event_type", "")),
                        "oms_order_id": str(
                            getattr(event, "oms_order_id", "") or ""
                        ),
                    },
                )
                logger.error(
                    "IARIC residual OMS event requires reconciliation: %s",
                    exc,
                    exc_info=exc,
                )
                await self._save_state("oms_event_error")

    async def _handle_fill(self, event) -> None:
        mapping = self._oms_to_order.get(event.oms_order_id or "")
        if mapping is None:
            self._last_error = f"quarantined_unmatched_fill:{event.oms_order_id}"
            self._diagnostics.log_degraded(
                "daily_residual_fill", self._last_error, event.payload or {}
            )
            return
        symbol, role, client_order_id = mapping
        payload = event.payload or {}
        qty = int(float(payload.get("qty", 0.0) or 0.0))
        price = float(payload.get("price", 0.0) or 0.0)
        commission = float(payload.get("commission", 0.0) or 0.0)
        pre_position = self._state.symbols[symbol].position
        pre_open_qty = pre_position.qty_open if pre_position else 0
        competing_exit_oms = ""
        if role == "STOP":
            pending_client = self._state.symbols[symbol].pending_client_order_id
            if pending_client and pending_client != client_order_id:
                competing_exit_oms = self._client_to_oms.get(pending_client, "")
        self._state, actions, events = apply_daily_residual_fill(
            self._state,
            DailyResidualFill(
                client_order_id=client_order_id,
                symbol=symbol,
                role=role,
                qty=qty,
                price=price,
                ts=event.timestamp,
                commission=commission,
            ),
        )
        self._record_events(events)
        if competing_exit_oms:
            await self._oms.submit_intent(
                Intent(
                    intent_type=IntentType.CANCEL_ORDER,
                    strategy_id=STRATEGY_ID,
                    target_oms_order_id=competing_exit_oms,
                )
            )
        await self._execute_actions(actions, event.timestamp)
        await self._record_trade_fill(
            symbol=symbol,
            role=role,
            qty=qty,
            price=price,
            ts=event.timestamp,
            pre_open_qty=pre_open_qty,
        )
        current = self._state.symbols[symbol]
        if role == "STOP" or current.pending_remaining_qty == 0:
            self._oms_to_order.pop(event.oms_order_id or "", None)
            self._client_to_oms.pop(client_order_id, None)
        await self._save_state("fill")

    async def _record_trade_fill(
        self,
        *,
        symbol: str,
        role: str,
        qty: int,
        price: float,
        ts: datetime,
        pre_open_qty: int,
    ) -> None:
        if self._trade_recorder is None:
            return
        position = self._state.symbols[symbol].position
        try:
            if role == "ENTRY" and position is not None and not position.trade_id:
                position.trade_id = await self._trade_recorder.record_entry(
                    strategy_id=STRATEGY_ID,
                    instrument=symbol,
                    direction="LONG",
                    quantity=qty,
                    entry_price=Decimal(str(price)),
                    entry_ts=ts,
                    setup_tag=DAILY_RESIDUAL_SLEEVE,
                    entry_type="next_session_open_market",
                    meta={
                        "strategy_params": self._artifact.strategy_parameters,
                        "selection_contract_version": (
                            self._artifact.selection_contract_version
                        ),
                    },
                    account_id=self._account_id,
                )
            elif (
                role in {"EXIT", "STOP"}
                and position is not None
                and position.trade_id
                and position.qty_open == 0
            ):
                initial_risk = max(
                    position.qty_entry * position.initial_risk_per_share,
                    1e-9,
                )
                await self._trade_recorder.record_exit(
                    trade_id=position.trade_id,
                    exit_price=Decimal(str(price)),
                    exit_ts=ts,
                    exit_reason=(
                        "catastrophic_stop"
                        if role == "STOP"
                        else self._state.symbols[symbol].pending_management_reason
                    ),
                    realized_r=Decimal(
                        str(round(position.realized_pnl_usd / initial_risk, 6))
                    ),
                    realized_usd=Decimal(str(round(position.realized_pnl_usd, 2))),
                )
        except Exception as exc:
            self._diagnostics.log_degraded(
                "daily_residual_trade_recorder", type(exc).__name__, {"symbol": symbol}
            )

    async def _handle_terminal(self, event) -> None:
        mapping = self._oms_to_order.pop(event.oms_order_id or "", None)
        if mapping is None:
            return
        symbol, role, client_order_id = mapping
        self._client_to_oms.pop(client_order_id, None)
        terminal = str(event.event_type.value).upper().removeprefix("ORDER_")
        state = self._state.symbols[symbol]
        self._last_error = f"order_{terminal.lower()}:{symbol}:{role}"
        if role == "ENTRY":
            state.entry_skipped_reason = f"OMS_{terminal}"
            state.pending_client_order_id = ""
            state.pending_role = ""
            state.pending_remaining_qty = 0
        elif role in {"PARTIAL_EXIT", "EXIT"}:
            state.pending_client_order_id = ""
            state.pending_role = ""
            state.pending_remaining_qty = 0
            await self._emergency_flatten(
                symbol,
                event.timestamp,
                reason=f"residual_management_order_{terminal.lower()}",
            )
        elif role == "STOP":
            state.protective_stop_client_order_id = ""
            state.protective_stop_qty = 0
            management_exit_is_working = (
                state.pending_role in {"PARTIAL_EXIT", "EXIT"}
                and state.pending_remaining_qty > 0
            )
            if not management_exit_is_working:
                await self._emergency_flatten(
                    symbol,
                    event.timestamp,
                    reason=f"protective_stop_{terminal.lower()}",
                )
        await self._save_state("terminal_order")

    def _record_events(self, events) -> None:
        for event in events:
            self._decision_events.append(event)
            self._diagnostics.log_decision(event.code, asdict(event))

    async def _pulse_loop(self) -> None:
        while self._running:
            await self.advance(datetime.now(timezone.utc))
            await asyncio.sleep(1.0)

    def snapshot_state(self) -> IntradayStateSnapshot:
        return IntradayStateSnapshot(
            trade_date=self._state.trade_date,
            saved_at=datetime.now(timezone.utc),
            symbols=list(self._state.symbols.values()),
            last_decision_code=self._state.last_decision_code,
            meta={
                "strategy_mode": DAILY_RESIDUAL_SLEEVE,
                "schema_version": self._state.schema_version,
                "nav": self._state.nav,
                "session_orders_planned": self._state.session_orders_planned,
                "entry_orders_planned": self._state.entry_orders_planned,
                "exit_orders_planned": self._state.exit_orders_planned,
                "client_to_oms": dict(self._client_to_oms),
                "oms_to_order": {
                    key: list(value) for key, value in self._oms_to_order.items()
                },
            },
        )

    def hydrate_state(self, snapshot: IntradayStateSnapshot) -> None:
        if snapshot.meta.get("strategy_mode") != DAILY_RESIDUAL_SLEEVE:
            raise ValueError("cannot hydrate legacy state into residual engine")
        if snapshot.trade_date != self._artifact.trade_date:
            raise ValueError("residual state date does not match artifact date")
        if snapshot.meta.get("schema_version") != "iaric_daily_residual_execution_v2":
            raise ValueError("residual state does not use the frozen-model v2 schema")
        self._state = DailyResidualExecutionState(
            trade_date=snapshot.trade_date,
            nav=float(snapshot.meta.get("nav", self._state.nav)),
            symbols={state.symbol: state for state in snapshot.symbols},
            session_orders_planned=bool(
                snapshot.meta.get("session_orders_planned", False)
            ),
            entry_orders_planned=bool(snapshot.meta.get("entry_orders_planned", False)),
            exit_orders_planned=bool(snapshot.meta.get("exit_orders_planned", False)),
            last_decision_code=snapshot.last_decision_code,
            schema_version=str(
                snapshot.meta.get(
                    "schema_version", "iaric_daily_residual_execution_v2"
                )
            ),
        )
        self._client_to_oms = {
            str(key): str(value)
            for key, value in snapshot.meta.get("client_to_oms", {}).items()
        }
        self._oms_to_order = {
            str(key): tuple(value)
            for key, value in snapshot.meta.get("oms_to_order", {}).items()
        }

    async def _save_state(self, reason: str) -> None:
        try:
            persist_intraday_state(self.snapshot_state(), settings=self._settings)
            self._last_save_ts = datetime.now(timezone.utc)
        except Exception as exc:
            self._last_error = f"state_persist_failed:{type(exc).__name__}"
            logger.error("IARIC residual state save failed (%s): %s", reason, exc)

    def health_status(self) -> dict[str, Any]:
        return {
            "engine": "IARICDailyResidualEngine",
            "running": self._running,
            "strategy_mode": DAILY_RESIDUAL_SLEEVE,
            "selection_contract_version": self._artifact.selection_contract_version,
            "symbols_tracked": len(self._state.symbols),
            "open_positions": sum(
                1
                for value in self._state.symbols.values()
                if value.position is not None and value.position.qty_open > 0
            ),
            "pending_orders": len(self._oms_to_order),
            "session_orders_planned": self._state.session_orders_planned,
            "risk_halted": self._risk_halted,
            "decision_events": len(self._decision_events),
            "last_decision_code": self._state.last_decision_code,
            "last_error": self._last_error,
        }

    @property
    def decision_events(self) -> tuple[Any, ...]:
        return tuple(self._decision_events)
