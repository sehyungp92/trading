from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from strategies.core.actions import (
    FlattenPosition,
    ReplaceProtectiveStop,
    SubmitEntry,
    SubmitProfitTarget,
    SubmitProtectiveStop,
)
from strategies.core.events import DecisionEvent
from strategies.scalp._shared.nq_contract import spec_for
from strategies.scalp.po3_reversal.config import (
    MAX_FULL_LOSSES_PER_DAY,
    MAX_TRADES_PER_DAY,
    Po3Phase,
    SetupTier,
    TradeDirection,
)
from strategies.scalp.po3_reversal.models import Po3Position, Po3Setup

from .state import (
    Po3BarInput,
    Po3EntryRequest,
    Po3Fill,
    Po3FlattenRequest,
    Po3OrderUpdate,
    Po3ReversalCoreState,
    Po3StopUpdateRequest,
)

_TERMINAL_STATUSES = {"cancelled", "expired", "rejected", "filled"}


def build_core_state(engine) -> Po3ReversalCoreState:
    return Po3ReversalCoreState(
        phase=deepcopy(getattr(engine, "phase", Po3Phase.IDLE)),
        context=deepcopy(getattr(engine, "context", None)) or Po3ReversalCoreState().context,
        active_setup=deepcopy(getattr(engine, "active_setup", None)),
        position=deepcopy(getattr(engine, "position", None)),
        order_to_setup=deepcopy(getattr(engine, "_order_to_setup", {})),
        order_kind=deepcopy(getattr(engine, "_order_kind", {})),
        daily_pnl=float(getattr(engine, "daily_pnl", 0.0)),
        weekly_pnl=float(getattr(engine, "weekly_pnl", 0.0)),
        trades_today=int(getattr(engine, "trades_today", 0)),
        full_losses_today=int(getattr(engine, "full_losses_today", 0)),
        last_signal_score=float(getattr(engine, "last_signal_score", 0.0)),
        last_tier=str(getattr(engine, "last_tier", "")),
        last_decision_code=str(getattr(engine, "_last_decision_code", "IDLE")),
        last_decision_details=deepcopy(getattr(engine, "_last_decision_details", {})),
        last_bar_ts=getattr(engine, "_last_bar_ts", None),
    )


def apply_core_state(engine, state: Po3ReversalCoreState) -> None:
    engine.phase = deepcopy(state.phase)
    engine.context = deepcopy(state.context)
    engine.active_setup = deepcopy(state.active_setup)
    engine.position = deepcopy(state.position)
    engine._order_to_setup = deepcopy(state.order_to_setup)
    engine._order_kind = deepcopy(state.order_kind)
    engine.daily_pnl = state.daily_pnl
    engine.weekly_pnl = state.weekly_pnl
    engine.trades_today = state.trades_today
    engine.full_losses_today = state.full_losses_today
    engine.last_signal_score = state.last_signal_score
    engine.last_tier = state.last_tier
    engine._last_decision_code = state.last_decision_code
    engine._last_decision_details = deepcopy(state.last_decision_details)
    engine._last_bar_ts = state.last_bar_ts


def on_bar(
    state: Po3ReversalCoreState,
    payload: Po3BarInput | None = None,
    *,
    entry_request: Po3EntryRequest | None = None,
    stop_update: Po3StopUpdateRequest | None = None,
    flatten_request: Po3FlattenRequest | None = None,
) -> tuple[Po3ReversalCoreState, list, list[DecisionEvent]]:
    next_state = deepcopy(state)
    actions: list = []
    events: list[DecisionEvent] = []
    event_ts = payload.bar_ts if payload is not None else datetime.now(timezone.utc)
    if payload is not None:
        next_state.last_bar_ts = payload.bar_ts
        next_state.context = deepcopy(payload.context)
        next_state.last_signal_score = payload.signal_score
        next_state.last_tier = payload.tier.value
        _advance_from_payload(next_state, payload, events)
        if _payload_requests_entry(payload, next_state):
            entry_request = Po3EntryRequest(
                client_order_id=f"{payload.symbol}-po3-entry-{payload.bar_ts.strftime('%Y%m%d%H%M%S')}",
                setup=_setup_from_payload(payload),
            )

    if entry_request is not None:
        setup = deepcopy(entry_request.setup)
        setup.state = "ARMED"
        setup.entry_order_id = entry_request.client_order_id
        next_state.active_setup = setup
        next_state.order_to_setup[entry_request.client_order_id] = setup.setup_id
        next_state.order_kind[entry_request.client_order_id] = "entry"
        next_state.phase = Po3Phase.ORDER_WORKING
        actions.append(
            SubmitEntry(
                client_order_id=entry_request.client_order_id,
                symbol=setup.symbol,
                side="BUY" if setup.direction is TradeDirection.LONG else "SELL",
                qty=setup.qty,
                order_type=entry_request.order_type,
                tif=entry_request.tif,
                stop_price=setup.entry_price if entry_request.order_type == "STOP" else None,
                limit_price=setup.entry_price if entry_request.order_type == "LIMIT" else None,
                role="entry",
                risk_context={"stop_for_risk": setup.stop_price, "planned_entry_price": setup.entry_price},
                metadata={
                    "setup_id": setup.setup_id,
                    "tier": setup.tier.value,
                    "entry_type": setup.entry_type.value,
                    "target_price": setup.target_price,
                    "rr": setup.rr,
                },
            )
        )
        events.append(_event("ENTRY_REQUESTED", event_ts, setup.symbol, {"setup_id": setup.setup_id, "score": setup.score}))

    if stop_update is not None and next_state.active_setup is not None:
        setup = next_state.active_setup
        if setup.setup_id == stop_update.setup_id and setup.stop_order_id:
            setup.stop_price = stop_update.stop_price
            if next_state.position is not None:
                next_state.position.stop_price = stop_update.stop_price
            actions.append(
                ReplaceProtectiveStop(
                    symbol=stop_update.symbol,
                    target_order_id=setup.stop_order_id,
                    side="SELL" if setup.direction is TradeDirection.LONG else "BUY",
                    stop_price=stop_update.stop_price,
                    qty=min(stop_update.qty, setup.qty_open or setup.qty),
                    reason=stop_update.reason,
                )
            )
            events.append(_event("STOP_REPLACEMENT_REQUESTED", event_ts, stop_update.symbol, {"setup_id": setup.setup_id}))

    if flatten_request is not None and next_state.position is not None:
        position = next_state.position
        if flatten_request.setup_id == position.setup_id:
            actions.append(
                FlattenPosition(
                    symbol=flatten_request.symbol,
                    reason=flatten_request.reason,
                    side="SELL" if position.direction is TradeDirection.LONG else "BUY",
                    qty=position.qty,
                    metadata={"setup_id": position.setup_id},
                )
            )
            events.append(_event("FLATTEN_REQUESTED", event_ts, flatten_request.symbol, {"setup_id": position.setup_id}))

    _update_last_decision(next_state, events)
    return next_state, actions, events


def on_fill(
    state: Po3ReversalCoreState,
    fill: Po3Fill,
) -> tuple[Po3ReversalCoreState, list, list[DecisionEvent]]:
    next_state = deepcopy(state)
    actions: list = []
    events: list[DecisionEvent] = []
    event_ts = fill.fill_time or datetime.now(timezone.utc)
    setup_id = next_state.order_to_setup.pop(fill.oms_order_id, "")
    order_kind = next_state.order_kind.pop(fill.oms_order_id, fill.order_role)
    setup = next_state.active_setup if next_state.active_setup and next_state.active_setup.setup_id == setup_id else None
    if setup is None:
        if fill.decision_code:
            events.append(_event(fill.decision_code, event_ts, fill.symbol, fill.decision_details))
        _update_last_decision(next_state, events, preserve_last_bar_ts=True)
        return next_state, actions, events

    fill_qty = fill.fill_qty or setup.qty
    fill_price = fill.fill_price or setup.entry_price
    if order_kind == "entry":
        setup.qty_open = fill_qty
        setup.avg_entry = fill_price
        setup.state = "ACTIVE"
        setup.metadata["_entry_commission"] = fill.commission
        stop_id = f"{setup.symbol}-po3-stop-{setup.setup_id}"
        target_id = f"{setup.symbol}-po3-target-{setup.setup_id}"
        setup.stop_order_id = stop_id
        setup.target_order_id = target_id
        next_state.order_to_setup[stop_id] = setup.setup_id
        next_state.order_kind[stop_id] = "stop"
        next_state.order_to_setup[target_id] = setup.setup_id
        next_state.order_kind[target_id] = "target"
        next_state.position = Po3Position(
            setup_id=setup.setup_id,
            symbol=setup.symbol,
            direction=setup.direction,
            qty=fill_qty,
            avg_entry=fill_price,
            stop_price=setup.stop_price,
            target_price=setup.target_price,
            opened_at=event_ts,
            initial_risk_points=abs(fill_price - setup.stop_price),
        )
        next_state.phase = Po3Phase.IN_POSITION
        exit_side = "SELL" if setup.direction is TradeDirection.LONG else "BUY"
        actions.extend(
            [
                SubmitProtectiveStop(
                    client_order_id=stop_id,
                    symbol=setup.symbol,
                    side=exit_side,
                    qty=fill_qty,
                    stop_price=setup.stop_price,
                    oca_group=f"PO3-{setup.setup_id}",
                    metadata={"setup_id": setup.setup_id, "stop_for_risk": setup.stop_price},
                ),
                SubmitProfitTarget(
                    client_order_id=target_id,
                    symbol=setup.symbol,
                    side=exit_side,
                    qty=fill_qty,
                    limit_price=setup.target_price,
                    oca_group=f"PO3-{setup.setup_id}",
                    metadata={"setup_id": setup.setup_id, "stop_for_risk": setup.stop_price},
                ),
            ]
        )
        events.append(_event("ENTRY_FILLED", event_ts, setup.symbol, {"setup_id": setup.setup_id, "price": fill_price, "qty": fill_qty}))
    elif order_kind in {"stop", "target", "flatten"}:
        exit_code = "STOP_FILLED" if order_kind == "stop" else "TARGET_FILLED" if order_kind == "target" else "EXIT_FILLED"
        _record_realized_exit(next_state, setup, fill_price, fill_qty or setup.qty_open, fill.commission, order_kind)
        setup.qty_open = 0
        setup.state = "CLOSED"
        next_state.position = None
        next_state.phase = Po3Phase.DONE_FOR_DAY if order_kind in {"stop", "target"} else Po3Phase.IDLE
        next_state.trades_today += 1
        events.append(_event(exit_code, event_ts, setup.symbol, {"setup_id": setup.setup_id, "price": fill_price}))

    _update_last_decision(next_state, events, preserve_last_bar_ts=True)
    return next_state, actions, events


def on_order_update(
    state: Po3ReversalCoreState,
    update: Po3OrderUpdate,
) -> tuple[Po3ReversalCoreState, list, list[DecisionEvent]]:
    next_state = deepcopy(state)
    events: list[DecisionEvent] = []
    event_ts = update.timestamp or datetime.now(timezone.utc)
    status = update.status.lower()
    if status in _TERMINAL_STATUSES and update.oms_order_id:
        setup_id = next_state.order_to_setup.pop(update.oms_order_id, "")
        order_kind = next_state.order_kind.pop(update.oms_order_id, update.order_role)
        if setup_id:
            if (
                order_kind == "entry"
                and next_state.active_setup is not None
                and next_state.active_setup.setup_id == setup_id
                and status in {"cancelled", "expired", "rejected"}
            ):
                next_state.active_setup.state = status.upper()
                if next_state.position is None:
                    next_state.phase = Po3Phase.IDLE
            events.append(_event("ORDER_TERMINAL", event_ts, update.symbol, {"setup_id": setup_id, "order_kind": order_kind, "status": status}))
    elif update.decision_code:
        events.append(_event(update.decision_code, event_ts, update.symbol, update.decision_details))
    _update_last_decision(next_state, events, preserve_last_bar_ts=True)
    return next_state, [], events


def _advance_from_payload(state: Po3ReversalCoreState, payload: Po3BarInput, events: list[DecisionEvent]) -> None:
    if payload.decision_code:
        events.append(_event(payload.decision_code, payload.bar_ts, payload.symbol, payload.decision_details))
    if payload.context.daily_bias is not TradeDirection.FLAT or payload.context.h4_bias is not TradeDirection.FLAT:
        state.phase = max_phase(state.phase, Po3Phase.BIAS_SET)
    if payload.direction is not TradeDirection.FLAT and payload.context.h4_bias == payload.direction:
        state.phase = max_phase(state.phase, Po3Phase.H4_PRIMED)
    if payload.sweep is not None and payload.sweep.swept:
        state.phase = max_phase(state.phase, Po3Phase.LIQUIDITY_ARMED)
    if payload.smt is not None and payload.smt.present:
        state.phase = max_phase(state.phase, Po3Phase.SMT_CONFIRMED)
    if payload.ifvg is not None:
        state.phase = max_phase(state.phase, Po3Phase.IFVG_CONFIRMED)


def _payload_requests_entry(payload: Po3BarInput, state: Po3ReversalCoreState) -> bool:
    if state.phase in {Po3Phase.ORDER_WORKING, Po3Phase.IN_POSITION, Po3Phase.BREAK_EVEN_SET, Po3Phase.DONE_FOR_DAY}:
        return False
    if state.position is not None:
        return False
    if any(kind == "entry" for kind in state.order_kind.values()):
        return False
    if state.trades_today >= MAX_TRADES_PER_DAY or state.full_losses_today >= MAX_FULL_LOSSES_PER_DAY:
        return False
    if payload.tier is SetupTier.NONE:
        return False
    if payload.direction is TradeDirection.FLAT or payload.qty <= 0:
        return False
    if not payload.risk_approved or payload.depth_confirmed is False:
        return False
    if payload.signal_threshold <= 0 or payload.signal_score < payload.signal_threshold:
        return False
    if payload.entry_price <= 0 or payload.stop_price <= 0 or payload.target_price <= 0:
        return False
    return True


def _record_realized_exit(
    state: Po3ReversalCoreState,
    setup: Po3Setup,
    fill_price: float,
    exit_qty: int,
    exit_commission: float,
    order_kind: str,
) -> None:
    if exit_qty <= 0 or setup.avg_entry <= 0:
        return
    direction = 1 if setup.direction is TradeDirection.LONG else -1
    gross = (fill_price - setup.avg_entry) * direction * exit_qty * spec_for(setup.symbol).point_value
    net = gross - float(setup.metadata.get("_entry_commission", 0.0)) - exit_commission
    state.daily_pnl += net
    state.weekly_pnl += net
    if order_kind == "stop" or net < 0:
        state.full_losses_today += 1


def _setup_from_payload(payload: Po3BarInput) -> Po3Setup:
    direction_name = "long" if payload.direction is TradeDirection.LONG else "short"
    return Po3Setup(
        setup_id=f"PO3-{payload.bar_ts.strftime('%Y%m%d%H%M%S')}-{direction_name}",
        symbol=payload.symbol,
        direction=payload.direction,
        tier=payload.tier,
        entry_type=__import__("strategies.scalp.po3_reversal.config", fromlist=["EntryType"]).EntryType.STOP_CONFIRMATION,
        signal_time=payload.bar_ts,
        score=payload.signal_score,
        entry_price=payload.entry_price,
        stop_price=payload.stop_price,
        target_price=payload.target_price,
        qty=payload.qty,
        rr=payload.rr,
        metadata=dict(payload.decision_details),
    )


_PHASE_RANK = {phase: rank for rank, phase in enumerate(Po3Phase)}


def max_phase(current: Po3Phase, candidate: Po3Phase) -> Po3Phase:
    if current in {Po3Phase.ORDER_WORKING, Po3Phase.IN_POSITION, Po3Phase.DONE_FOR_DAY}:
        return current
    return candidate if _PHASE_RANK[candidate] > _PHASE_RANK[current] else current


def _event(code: str, ts: datetime, symbol: str, details: dict) -> DecisionEvent:
    return DecisionEvent(code=code, ts=ts, symbol=symbol, timeframe="1m", details=dict(details))


def _update_last_decision(
    state: Po3ReversalCoreState,
    events: list[DecisionEvent],
    *,
    preserve_last_bar_ts: bool = False,
) -> None:
    if not events:
        return
    latest = events[-1]
    state.last_decision_code = latest.code
    state.last_decision_details = dict(latest.details)
    if not preserve_last_bar_ts:
        state.last_bar_ts = latest.ts
