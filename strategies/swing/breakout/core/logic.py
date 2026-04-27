from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from strategies.core.actions import (
    FlattenPosition,
    ReplaceProtectiveStop,
    SubmitAddOnEntry,
    SubmitEntry,
    SubmitPartialExit,
    SubmitProtectiveStop,
)
from strategies.core.events import DecisionEvent
from strategies.swing.breakout.models import Direction, PositionState, SetupInstance, SetupState

from .state import (
    BreakoutBarInput,
    BreakoutCoreState,
    BreakoutEntryRequest,
    BreakoutFill,
    BreakoutFlattenRequest,
    BreakoutOrderUpdate,
    BreakoutPartialExitRequest,
    BreakoutStopUpdateRequest,
)

_TERMINAL_STATUSES = {
    "cancelled",
    "expired",
    "rejected",
    "order_cancelled",
    "order_expired",
    "order_rejected",
}


def build_core_state(engine) -> BreakoutCoreState:
    return BreakoutCoreState(
        campaigns=deepcopy(engine.campaigns),
        histories=deepcopy(engine.histories),
        hourly_states=deepcopy(engine.hourly_states),
        positions=deepcopy(engine.positions),
        active_setups=deepcopy(engine.active_setups),
        circuit_breaker=deepcopy(engine.circuit_breaker),
        correlation_map=deepcopy(engine.correlation_map),
        order_to_setup=deepcopy(engine._order_to_setup),
        order_kind=deepcopy(engine._order_kind),
        order_requested_qty=deepcopy(engine._order_requested_qty),
        oca_counter=engine._oca_counter,
        risk_halted=engine._risk_halted,
        risk_halt_reason=engine._risk_halt_reason,
        last_decision_code=engine._last_decision_code,
        last_decision_details=deepcopy(engine._last_decision_details),
        last_bar_ts=engine._last_bar_ts,
    )


def apply_core_state(engine, state: BreakoutCoreState) -> None:
    engine.campaigns = deepcopy(state.campaigns)
    engine.histories = deepcopy(state.histories)
    engine.hourly_states = deepcopy(state.hourly_states)
    engine.positions = deepcopy(state.positions)
    engine.active_setups = deepcopy(state.active_setups)
    engine.circuit_breaker = deepcopy(state.circuit_breaker)
    engine.correlation_map = deepcopy(state.correlation_map)
    engine._order_to_setup = deepcopy(state.order_to_setup)
    engine._order_kind = deepcopy(state.order_kind)
    engine._order_requested_qty = deepcopy(state.order_requested_qty)
    engine._oca_counter = state.oca_counter
    engine._risk_halted = state.risk_halted
    engine._risk_halt_reason = state.risk_halt_reason
    engine._last_decision_code = state.last_decision_code
    engine._last_decision_details = deepcopy(state.last_decision_details)
    engine._last_bar_ts = state.last_bar_ts


def on_bar(
    state: BreakoutCoreState,
    payload: BreakoutBarInput | None = None,
    *,
    bar_ts: datetime | None = None,
    entry_request: BreakoutEntryRequest | None = None,
    stop_update: BreakoutStopUpdateRequest | None = None,
    partial_exit_request: BreakoutPartialExitRequest | None = None,
    flatten_request: BreakoutFlattenRequest | None = None,
) -> tuple[
    BreakoutCoreState,
    list[SubmitEntry | SubmitAddOnEntry | ReplaceProtectiveStop | SubmitPartialExit | FlattenPosition],
    list[DecisionEvent],
]:
    next_state = deepcopy(state)
    actions: list[SubmitEntry | SubmitAddOnEntry | ReplaceProtectiveStop | SubmitPartialExit | FlattenPosition] = []
    events: list[DecisionEvent] = []

    if payload is not None and all(
        request is None
        for request in (entry_request, stop_update, partial_exit_request, flatten_request)
    ):
        events = _legacy_bar_events(payload)
        if payload.bar_ts is not None:
            next_state.last_bar_ts = payload.bar_ts
        _update_last_decision(next_state, events)
        return next_state, [], events

    if bar_ts is not None:
        next_state.last_bar_ts = bar_ts
    event_ts = bar_ts or datetime.now(timezone.utc)

    if entry_request is not None:
        setup = deepcopy(entry_request.setup)
        setup.state = SetupState.ARMED
        next_state.active_setups[setup.setup_id] = setup
        action_cls = SubmitAddOnEntry if setup.is_add else SubmitEntry
        actions.append(
            action_cls(
                client_order_id=entry_request.client_order_id,
                symbol=setup.symbol,
                side="BUY" if setup.direction == Direction.LONG else "SELL",
                qty=setup.shares_planned,
                order_type=entry_request.order_type,
                tif=entry_request.tif,
                limit_price=setup.entry_price if entry_request.order_type == "LIMIT" else None,
                role="add_on_entry" if setup.is_add else "entry",
                oca_group=setup.oca_group,
                risk_context={
                    "stop_for_risk": setup.stop0,
                    "planned_entry_price": setup.entry_price,
                },
                metadata={"setup_id": setup.setup_id, "entry_type": setup.entry_type.value},
            )
        )
        events.append(
            _event(
                code="ADD_REQUESTED" if setup.is_add else "ENTRY_REQUESTED",
                ts=event_ts,
                symbol=setup.symbol,
                details={
                    "setup_id": setup.setup_id,
                    "entry_type": setup.entry_type.value,
                    "qty": setup.shares_planned,
                    "entry_price": setup.entry_price,
                    "stop0": setup.stop0,
                },
            )
        )

    if stop_update is not None:
        setup = next_state.active_setups.get(stop_update.setup_id)
        if setup is not None and setup.stop_order_id:
            setup.current_stop = stop_update.stop_price
            position = next_state.positions.get(stop_update.symbol)
            if position is not None:
                position.current_stop = stop_update.stop_price
            actions.append(
                ReplaceProtectiveStop(
                    symbol=stop_update.symbol,
                    target_order_id=setup.stop_order_id,
                    side="SELL" if setup.direction == Direction.LONG else "BUY",
                    stop_price=stop_update.stop_price,
                    qty=min(stop_update.qty, max(setup.qty_open, 0)),
                    reason=stop_update.reason,
                )
            )
            events.append(
                _event(
                    code="STOP_REPLACEMENT_REQUESTED",
                    ts=event_ts,
                    symbol=stop_update.symbol,
                    details={"setup_id": setup.setup_id, "stop_price": stop_update.stop_price},
                )
            )

    if partial_exit_request is not None:
        setup = next_state.active_setups.get(partial_exit_request.setup_id)
        if setup is not None and setup.qty_open > 0:
            actions.append(
                SubmitPartialExit(
                    client_order_id=partial_exit_request.client_order_id,
                    symbol=partial_exit_request.symbol,
                    side="SELL" if setup.direction == Direction.LONG else "BUY",
                    qty=min(partial_exit_request.qty, setup.qty_open),
                    order_type=partial_exit_request.order_type,
                    tif=partial_exit_request.tif,
                    metadata={"setup_id": setup.setup_id, "reason": partial_exit_request.reason},
                )
            )
            events.append(
                _event(
                    code="PARTIAL_EXIT_REQUESTED",
                    ts=event_ts,
                    symbol=partial_exit_request.symbol,
                    details={"setup_id": setup.setup_id, "qty": min(partial_exit_request.qty, setup.qty_open)},
                )
            )

    if flatten_request is not None:
        setup = next_state.active_setups.get(flatten_request.setup_id)
        if setup is not None and setup.qty_open > 0:
            actions.append(
                FlattenPosition(
                    symbol=flatten_request.symbol,
                    reason=flatten_request.reason,
                    side="SELL" if setup.direction == Direction.LONG else "BUY",
                    qty=setup.qty_open,
                )
            )
            events.append(
                _event(
                    code="FLATTEN_REQUESTED",
                    ts=event_ts,
                    symbol=flatten_request.symbol,
                    details={"setup_id": setup.setup_id, "reason": flatten_request.reason},
                )
            )

    _update_last_decision(next_state, events)
    return next_state, actions, events


def on_order_update(
    state: BreakoutCoreState,
    update: BreakoutOrderUpdate,
) -> tuple[BreakoutCoreState, list[ReplaceProtectiveStop], list[DecisionEvent]]:
    next_state = deepcopy(state)
    event_ts = update.timestamp or datetime.now(timezone.utc)
    events: list[DecisionEvent] = []

    if update.status.lower() in _TERMINAL_STATUSES and update.oms_order_id:
        setup_id = next_state.order_to_setup.pop(update.oms_order_id, "")
        order_kind = next_state.order_kind.pop(update.oms_order_id, "")
        next_state.order_requested_qty.pop(update.oms_order_id, None)
        setup = next_state.active_setups.get(setup_id) if setup_id else None
        if setup is not None:
            if order_kind == "stop" and setup.stop_order_id == update.oms_order_id:
                setup.stop_order_id = ""
            elif order_kind == "tp1" and setup.tp1_order_id == update.oms_order_id:
                setup.tp1_order_id = ""
            elif order_kind == "tp2" and setup.tp2_order_id == update.oms_order_id:
                setup.tp2_order_id = ""
            events.append(
                _event(
                    code="ORDER_TERMINAL",
                    ts=event_ts,
                    symbol=setup.symbol,
                    details={"setup_id": setup_id, "order_kind": order_kind, "status": update.status.lower()},
                )
            )

    if not events and update.decision_code:
        events.append(
            _event(
                code=update.decision_code,
                ts=event_ts,
                symbol=update.symbol,
                details=update.decision_details,
            )
        )

    _update_last_decision(next_state, events, preserve_last_bar_ts=True)
    return next_state, [], events


def on_fill(
    state: BreakoutCoreState,
    fill: BreakoutFill,
) -> tuple[
    BreakoutCoreState,
    list[SubmitProtectiveStop | ReplaceProtectiveStop],
    list[DecisionEvent],
]:
    next_state = deepcopy(state)
    event_ts = fill.fill_time or datetime.now(timezone.utc)
    actions: list[SubmitProtectiveStop | ReplaceProtectiveStop] = []
    events: list[DecisionEvent] = []

    setup_id = next_state.order_to_setup.get(fill.oms_order_id, "")
    order_kind = next_state.order_kind.get(fill.oms_order_id, fill.order_role)
    setup = next_state.active_setups.get(setup_id) if setup_id else None
    if setup is None:
        if fill.decision_code:
            events.append(
                _event(
                    code=fill.decision_code,
                    ts=event_ts,
                    symbol=fill.symbol,
                    details=fill.decision_details,
                )
            )
        _update_last_decision(next_state, events, preserve_last_bar_ts=True)
        return next_state, actions, events

    fill_price = fill.fill_price or setup.entry_price
    fill_qty = fill.fill_qty or next_state.order_requested_qty.get(fill.oms_order_id, setup.shares_planned)

    if order_kind in {"primary_entry", "entry", "add", "add_entry", "rescue", "catchup"}:
        if (setup.is_add or order_kind in {"add", "add_entry"}) and setup.symbol in next_state.positions:
            position = next_state.positions[setup.symbol]
            total_before = position.qty
            position.qty += fill_qty
            if total_before > 0:
                position.avg_cost = ((position.avg_cost * total_before) + (fill_price * fill_qty)) / max(position.qty, 1)
            else:
                position.avg_cost = fill_price
            position.add_count += 1
            setup.fill_qty += fill_qty
            setup.qty_open += fill_qty
            setup.avg_entry = position.avg_cost
            setup.state = SetupState.ACTIVE
            if setup.stop_order_id:
                actions.append(
                    ReplaceProtectiveStop(
                        symbol=setup.symbol,
                        target_order_id=setup.stop_order_id,
                        side="SELL" if setup.direction == Direction.LONG else "BUY",
                        stop_price=setup.current_stop,
                        qty=setup.qty_open,
                        reason="add_resize",
                    )
                )
            events.append(
                _event(
                    code="ADD_FILLED",
                    ts=event_ts,
                    symbol=setup.symbol,
                    details={"setup_id": setup.setup_id, "qty": fill_qty, "price": fill_price},
                )
            )
        else:
            setup.state = SetupState.ACTIVE
            setup.fill_price = fill_price
            setup.fill_qty = fill_qty
            setup.qty_open = fill_qty
            setup.avg_entry = fill_price
            setup.fill_ts = event_ts
            next_state.positions[setup.symbol] = PositionState(
                symbol=setup.symbol,
                direction=setup.direction,
                qty=fill_qty,
                avg_cost=fill_price,
                current_stop=setup.current_stop or setup.stop0,
                campaign_id=setup.campaign_id,
                box_version=setup.box_version,
                total_risk_dollars=setup.final_risk_dollars,
                regime_at_entry=setup.regime_at_entry,
            )
            actions.append(
                SubmitProtectiveStop(
                    client_order_id=f"{setup.symbol}-stop-{setup.setup_id}",
                    symbol=setup.symbol,
                    side="SELL" if setup.direction == Direction.LONG else "BUY",
                    qty=fill_qty,
                    stop_price=setup.current_stop or setup.stop0,
                    oca_group=setup.oca_group,
                )
            )
            events.append(
                _event(
                    code="ENTRY_FILLED",
                    ts=event_ts,
                    symbol=setup.symbol,
                    details={"setup_id": setup.setup_id, "qty": fill_qty, "price": fill_price},
                )
            )
    elif order_kind in {"tp1", "tp2", "partial_close"}:
        exit_qty = min(fill_qty, setup.qty_open)
        setup.qty_open = max(0, setup.qty_open - exit_qty)
        position = next_state.positions.get(setup.symbol)
        if position is not None:
            position.qty = max(0, position.qty - exit_qty)
        if setup.qty_open > 0 and setup.stop_order_id:
            actions.append(
                ReplaceProtectiveStop(
                    symbol=setup.symbol,
                    target_order_id=setup.stop_order_id,
                    side="SELL" if setup.direction == Direction.LONG else "BUY",
                    stop_price=setup.current_stop,
                    qty=setup.qty_open,
                    reason="partial_resize",
                )
            )
        events.append(
            _event(
                code="PARTIAL_EXIT_FILLED" if setup.qty_open > 0 else "EXIT_FILLED",
                ts=event_ts,
                symbol=setup.symbol,
                details={"setup_id": setup.setup_id, "order_kind": order_kind, "qty": exit_qty, "price": fill_price},
            )
        )
        if setup.qty_open <= 0:
            setup.state = SetupState.CLOSED
            next_state.positions.pop(setup.symbol, None)
    elif order_kind == "stop":
        setup.qty_open = 0
        setup.state = SetupState.CLOSED
        next_state.positions.pop(setup.symbol, None)
        events.append(
            _event(
                code="STOP_FILLED",
                ts=event_ts,
                symbol=setup.symbol,
                details={"setup_id": setup.setup_id, "price": fill_price},
            )
        )

    _update_last_decision(next_state, events, preserve_last_bar_ts=True)
    return next_state, actions, events


def _legacy_bar_events(payload: BreakoutBarInput) -> list[DecisionEvent]:
    if not payload.decision_code:
        return []
    return [
        _event(
            code=payload.decision_code,
            ts=payload.bar_ts or datetime.now(timezone.utc),
            symbol=payload.symbol,
            details=payload.decision_details,
        )
    ]


def _event(*, code: str, ts: datetime, symbol: str, details: dict[str, Any]) -> DecisionEvent:
    return DecisionEvent(code=code, ts=ts, symbol=symbol, timeframe="1h", details=dict(details))


def _update_last_decision(
    state: BreakoutCoreState,
    events: list[DecisionEvent],
    *,
    preserve_last_bar_ts: bool = False,
) -> None:
    if not events:
        return
    latest = events[-1]
    state.last_decision_code = latest.code
    state.last_decision_details = dict(latest.details)
    if latest.ts is not None and not preserve_last_bar_ts:
        state.last_bar_ts = latest.ts
