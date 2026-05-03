from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from strategies.core.actions import (
    FlattenPosition,
    ReplaceProtectiveStop,
    SubmitEntry,
    SubmitPartialExit,
    SubmitProfitTarget,
    SubmitProtectiveStop,
)
from strategies.core.events import DecisionEvent
from strategies.scalp._shared.nq_contract import spec_for
from strategies.scalp._shared.session import ScalpSessionBlock, entries_allowed, must_flatten
from strategies.scalp.ivb_auction.config import (
    A1_MIN_SCORE,
    A2_MIN_SCORE,
    MAX_LOSSES_PER_DAY,
    MAX_TRADES_PER_DAY,
    IvbAuctionPhase,
    IvbModule,
    TradeDirection,
)
from strategies.scalp.ivb_auction.models import IvbSetup, ScalpPosition, ScalpTick

from .state import (
    IvbAuctionCoreState,
    IvbBarInput,
    IvbEntryRequest,
    IvbFill,
    IvbFlattenRequest,
    IvbOrderUpdate,
    IvbPartialExitRequest,
    IvbStopUpdateRequest,
)

_TERMINAL_STATUSES = {"cancelled", "expired", "rejected", "filled"}


def build_core_state(engine) -> IvbAuctionCoreState:
    return IvbAuctionCoreState(
        phase=deepcopy(getattr(engine, "phase", IvbAuctionPhase.S0_PRE_OPEN)),
        ivb_levels=deepcopy(getattr(engine, "ivb_levels", None)),
        break_direction=deepcopy(getattr(engine, "break_direction", TradeDirection.FLAT)),
        break_price=float(getattr(engine, "break_price", 0.0)),
        break_bar_idx=int(getattr(engine, "break_bar_idx", -1)),
        positions=deepcopy(getattr(engine, "positions", {})),
        active_setups=deepcopy(getattr(engine, "active_setups", {})),
        order_to_setup=deepcopy(getattr(engine, "_order_to_setup", {})),
        order_kind=deepcopy(getattr(engine, "_order_kind", {})),
        cumulative_delta=float(getattr(engine, "cumulative_delta", 0.0)),
        absorption_events=deepcopy(getattr(engine, "absorption_events", [])),
        daily_pnl=float(getattr(engine, "daily_pnl", 0.0)),
        daily_trades=int(getattr(engine, "daily_trades", 0)),
        daily_losses=int(getattr(engine, "daily_losses", 0)),
        daily_risk_used=float(getattr(engine, "daily_risk_used", 0.0)),
        last_signal_score=float(getattr(engine, "last_signal_score", 0.0)),
        last_signal_module=str(getattr(engine, "last_signal_module", "")),
        last_decision_code=str(getattr(engine, "_last_decision_code", "IDLE")),
        last_decision_details=deepcopy(getattr(engine, "_last_decision_details", {})),
        last_bar_ts=getattr(engine, "_last_bar_ts", None),
        bar_index=int(getattr(engine, "bar_index", 0)),
    )


def apply_core_state(engine, state: IvbAuctionCoreState) -> None:
    engine.phase = deepcopy(state.phase)
    engine.ivb_levels = deepcopy(state.ivb_levels)
    engine.break_direction = deepcopy(state.break_direction)
    engine.break_price = state.break_price
    engine.break_bar_idx = state.break_bar_idx
    engine.positions = deepcopy(state.positions)
    engine.active_setups = deepcopy(state.active_setups)
    engine._order_to_setup = deepcopy(state.order_to_setup)
    engine._order_kind = deepcopy(state.order_kind)
    engine.cumulative_delta = state.cumulative_delta
    engine.absorption_events = deepcopy(state.absorption_events)
    engine.daily_pnl = state.daily_pnl
    engine.daily_trades = state.daily_trades
    engine.daily_losses = state.daily_losses
    engine.daily_risk_used = state.daily_risk_used
    engine.last_signal_score = state.last_signal_score
    engine.last_signal_module = state.last_signal_module
    engine._last_decision_code = state.last_decision_code
    engine._last_decision_details = deepcopy(state.last_decision_details)
    engine._last_bar_ts = state.last_bar_ts
    engine.bar_index = state.bar_index


def on_tick(
    state: IvbAuctionCoreState,
    tick: ScalpTick,
) -> tuple[IvbAuctionCoreState, list, list[DecisionEvent]]:
    next_state = deepcopy(state)
    if tick.side:
        next_state.cumulative_delta += tick.side * tick.size
    if tick.side and tick.size > 0:
        next_state.absorption_events = next_state.absorption_events[-100:]
    return next_state, [], []


def on_bar(
    state: IvbAuctionCoreState,
    payload: IvbBarInput | None = None,
    *,
    entry_request: IvbEntryRequest | None = None,
    stop_update: IvbStopUpdateRequest | None = None,
    partial_exit_request: IvbPartialExitRequest | None = None,
    flatten_request: IvbFlattenRequest | None = None,
) -> tuple[IvbAuctionCoreState, list, list[DecisionEvent]]:
    next_state = deepcopy(state)
    actions: list = []
    events: list[DecisionEvent] = []
    event_ts = payload.bar_ts if payload is not None else datetime.now(timezone.utc)

    if payload is not None:
        next_state.last_bar_ts = payload.bar_ts
        next_state.bar_index += 1
        if payload.footprint_state is not None:
            next_state.cumulative_delta = payload.footprint_state.cumulative_delta
        _advance_phase(next_state, payload, events)
        if _payload_requests_entry(payload, next_state):
            entry_request = IvbEntryRequest(
                client_order_id=f"{payload.symbol}-ivb-entry-{payload.bar_ts.strftime('%Y%m%d%H%M%S')}",
                setup=_setup_from_payload(payload),
                order_type="LIMIT" if payload.module is IvbModule.A1_CONTINUATION else "STOP",
            )

    if entry_request is not None:
        setup = deepcopy(entry_request.setup)
        setup.state = "ARMED"
        setup.entry_order_id = entry_request.client_order_id
        next_state.active_setups[setup.setup_id] = setup
        next_state.order_to_setup[entry_request.client_order_id] = setup.setup_id
        next_state.order_kind[entry_request.client_order_id] = "entry"
        next_state.phase = IvbAuctionPhase.S5_ENTRY_HUNT
        actions.append(
            SubmitEntry(
                client_order_id=entry_request.client_order_id,
                symbol=setup.symbol,
                side="BUY" if setup.direction is TradeDirection.LONG else "SELL",
                qty=setup.qty,
                order_type=entry_request.order_type,
                tif=entry_request.tif,
                limit_price=setup.entry_price if entry_request.order_type == "LIMIT" else None,
                stop_price=setup.entry_price if entry_request.order_type == "STOP" else None,
                role="entry",
                risk_context={"stop_for_risk": setup.stop_price, "planned_entry_price": setup.entry_price},
                metadata={"setup_id": setup.setup_id, "module": setup.module.value, "trigger": setup.trigger.value},
            )
        )
        events.append(_event("ENTRY_REQUESTED", event_ts, setup.symbol, {"setup_id": setup.setup_id, "score": setup.score}))

    if stop_update is not None:
        setup = next_state.active_setups.get(stop_update.setup_id)
        if setup is not None and setup.stop_order_id:
            setup.stop_price = stop_update.stop_price
            position = next_state.positions.get(setup.symbol)
            if position is not None:
                position.current_stop = stop_update.stop_price
            actions.append(
                ReplaceProtectiveStop(
                    symbol=stop_update.symbol,
                    target_order_id=setup.stop_order_id,
                    side="SELL" if setup.direction is TradeDirection.LONG else "BUY",
                    stop_price=stop_update.stop_price,
                    qty=min(stop_update.qty, max(setup.qty_open, 0)),
                    reason=stop_update.reason,
                )
            )
            events.append(_event("STOP_REPLACEMENT_REQUESTED", event_ts, setup.symbol, {"setup_id": setup.setup_id}))

    if partial_exit_request is not None:
        setup = next_state.active_setups.get(partial_exit_request.setup_id)
        if setup is not None and setup.qty_open > 0:
            qty = min(partial_exit_request.qty, setup.qty_open)
            next_state.order_to_setup[partial_exit_request.client_order_id] = setup.setup_id
            next_state.order_kind[partial_exit_request.client_order_id] = "partial"
            actions.append(
                SubmitPartialExit(
                    client_order_id=partial_exit_request.client_order_id,
                    symbol=partial_exit_request.symbol,
                    side="SELL" if setup.direction is TradeDirection.LONG else "BUY",
                    qty=qty,
                    order_type=partial_exit_request.order_type,
                    limit_price=partial_exit_request.limit_price,
                    metadata={"setup_id": setup.setup_id, "reason": partial_exit_request.reason},
                )
            )
            events.append(_event("PARTIAL_EXIT_REQUESTED", event_ts, setup.symbol, {"setup_id": setup.setup_id, "qty": qty}))

    if flatten_request is not None:
        setup = next_state.active_setups.get(flatten_request.setup_id)
        if setup is not None and setup.qty_open > 0:
            actions.append(
                FlattenPosition(
                    symbol=flatten_request.symbol,
                    reason=flatten_request.reason,
                    side="SELL" if setup.direction is TradeDirection.LONG else "BUY",
                    qty=setup.qty_open,
                    metadata={"setup_id": setup.setup_id},
                )
            )
            events.append(_event("FLATTEN_REQUESTED", event_ts, setup.symbol, {"setup_id": setup.setup_id}))

    _update_last_decision(next_state, events)
    return next_state, actions, events


def on_fill(
    state: IvbAuctionCoreState,
    fill: IvbFill,
) -> tuple[IvbAuctionCoreState, list, list[DecisionEvent]]:
    next_state = deepcopy(state)
    actions: list = []
    events: list[DecisionEvent] = []
    event_ts = fill.fill_time or datetime.now(timezone.utc)
    setup_id = next_state.order_to_setup.pop(fill.oms_order_id, "")
    order_kind = next_state.order_kind.pop(fill.oms_order_id, fill.order_role)
    setup = next_state.active_setups.get(setup_id) if setup_id else None
    if setup is None:
        if fill.decision_code:
            events.append(_event(fill.decision_code, event_ts, fill.symbol, fill.decision_details))
        _update_last_decision(next_state, events, preserve_last_bar_ts=True)
        return next_state, actions, events

    fill_price = fill.fill_price or setup.entry_price
    fill_qty = fill.fill_qty or setup.qty
    if order_kind == "entry":
        setup.qty_open = fill_qty
        setup.avg_entry = fill_price
        setup.state = "ACTIVE"
        setup.metadata["_entry_commission"] = fill.commission
        setup.metadata["_remaining_entry_commission"] = fill.commission
        next_state.daily_trades += 1
        stop_id = f"{setup.symbol}-ivb-stop-{setup.setup_id}"
        tp1_id = f"{setup.symbol}-ivb-tp1-{setup.setup_id}"
        setup.stop_order_id = stop_id
        setup.tp1_order_id = tp1_id
        next_state.order_to_setup[stop_id] = setup.setup_id
        next_state.order_kind[stop_id] = "stop"
        next_state.order_to_setup[tp1_id] = setup.setup_id
        next_state.order_kind[tp1_id] = "tp1"
        next_state.positions[setup.symbol] = ScalpPosition(
            setup_id=setup.setup_id,
            symbol=setup.symbol,
            module=setup.module,
            direction=setup.direction,
            qty=fill_qty,
            avg_entry=fill_price,
            current_stop=setup.stop_price,
            tp1_price=setup.tp1_price,
            tp2_price=setup.tp2_price,
            opened_at=event_ts,
            initial_risk_points=abs(fill_price - setup.stop_price),
        )
        next_state.phase = IvbAuctionPhase.S6_IN_TRADE
        exit_side = "SELL" if setup.direction is TradeDirection.LONG else "BUY"
        actions.extend(
            [
                SubmitProtectiveStop(
                    client_order_id=stop_id,
                    symbol=setup.symbol,
                    side=exit_side,
                    qty=fill_qty,
                    stop_price=setup.stop_price,
                    oca_group=f"IVB-{setup.setup_id}",
                    metadata={"setup_id": setup.setup_id, "stop_for_risk": setup.stop_price},
                ),
                SubmitProfitTarget(
                    client_order_id=tp1_id,
                    symbol=setup.symbol,
                    side=exit_side,
                    qty=fill_qty,
                    limit_price=setup.tp1_price,
                    oca_group=f"IVB-{setup.setup_id}",
                    metadata={"setup_id": setup.setup_id, "stop_for_risk": setup.stop_price},
                ),
            ]
        )
        events.append(_event("ENTRY_FILLED", event_ts, setup.symbol, {"setup_id": setup.setup_id, "price": fill_price, "qty": fill_qty}))
    elif order_kind in {"tp1", "tp2", "partial"}:
        exit_qty = min(fill_qty, setup.qty_open)
        _record_realized_exit(next_state, setup, fill_price, exit_qty, fill.commission, order_kind)
        setup.qty_open = max(0, setup.qty_open - exit_qty)
        position = next_state.positions.get(setup.symbol)
        if position is not None:
            position.qty = max(0, position.qty - exit_qty)
            position.tp1_filled = True
        if setup.qty_open <= 0:
            setup.state = "CLOSED"
            next_state.positions.pop(setup.symbol, None)
            next_state.phase = IvbAuctionPhase.S7_FLAT_DONE
        events.append(_event("PARTIAL_EXIT_FILLED" if setup.qty_open > 0 else "EXIT_FILLED", event_ts, setup.symbol, {"setup_id": setup.setup_id, "qty": exit_qty}))
    elif order_kind in {"stop", "flatten"}:
        exit_qty = setup.qty_open or fill_qty
        _record_realized_exit(next_state, setup, fill_price, exit_qty, fill.commission, order_kind)
        setup.qty_open = 0
        setup.state = "CLOSED"
        next_state.positions.pop(setup.symbol, None)
        next_state.phase = IvbAuctionPhase.S8_RECLAIM_WATCH if setup.module is IvbModule.A1_CONTINUATION else IvbAuctionPhase.S7_FLAT_DONE
        events.append(_event("STOP_FILLED" if order_kind == "stop" else "EXIT_FILLED", event_ts, setup.symbol, {"setup_id": setup.setup_id, "price": fill_price}))

    _update_last_decision(next_state, events, preserve_last_bar_ts=True)
    return next_state, actions, events


def on_order_update(
    state: IvbAuctionCoreState,
    update: IvbOrderUpdate,
) -> tuple[IvbAuctionCoreState, list, list[DecisionEvent]]:
    next_state = deepcopy(state)
    events: list[DecisionEvent] = []
    event_ts = update.timestamp or datetime.now(timezone.utc)
    if update.status.lower() in _TERMINAL_STATUSES and update.oms_order_id:
        setup_id = next_state.order_to_setup.pop(update.oms_order_id, "")
        order_kind = next_state.order_kind.pop(update.oms_order_id, update.order_role)
        if setup_id:
            events.append(_event("ORDER_TERMINAL", event_ts, update.symbol, {"setup_id": setup_id, "order_kind": order_kind, "status": update.status.lower()}))
    elif update.decision_code:
        events.append(_event(update.decision_code, event_ts, update.symbol, update.decision_details))
    _update_last_decision(next_state, events, preserve_last_bar_ts=True)
    return next_state, [], events


def _advance_phase(state: IvbAuctionCoreState, payload: IvbBarInput, events: list[DecisionEvent]) -> None:
    if payload.decision_code:
        events.append(_event(payload.decision_code, payload.bar_ts, payload.symbol, payload.decision_details))
    if payload.session_block is ScalpSessionBlock.IVB_FORMING:
        state.phase = IvbAuctionPhase.S1_IVB_FORMING
    if payload.ivb_levels is not None:
        state.ivb_levels = payload.ivb_levels
        if state.phase in {IvbAuctionPhase.S0_PRE_OPEN, IvbAuctionPhase.S1_IVB_FORMING}:
            state.phase = IvbAuctionPhase.S2_IVB_LOCKED
    if payload.session_block in {ScalpSessionBlock.RTH_PRIME, ScalpSessionBlock.RTH_ACTIVE} and state.phase is IvbAuctionPhase.S2_IVB_LOCKED:
        state.phase = IvbAuctionPhase.S3_AWAITING_BREAK
    if payload.breakout_accepted and payload.breakout_direction is not TradeDirection.FLAT:
        state.break_direction = payload.breakout_direction
        state.break_price = payload.bar_ohlcv[3]
        state.break_bar_idx = state.bar_index
        state.phase = IvbAuctionPhase.S4_BREAK_CONFIRMED
    if payload.module is IvbModule.A1_CONTINUATION and payload.signal_score >= A1_MIN_SCORE:
        state.phase = IvbAuctionPhase.S5_ENTRY_HUNT
    elif payload.module is IvbModule.A2_RECLAIM and payload.signal_score >= A2_MIN_SCORE:
        state.phase = IvbAuctionPhase.S8_RECLAIM_WATCH
    state.last_signal_score = payload.signal_score
    state.last_signal_module = payload.module.value if payload.module else ""


def _payload_requests_entry(payload: IvbBarInput, state: IvbAuctionCoreState) -> bool:
    if payload.module is None or payload.trigger is None:
        return False
    if not entries_allowed(payload.session_block, "ivb_auction") or must_flatten(payload.bar_ts):
        return False
    if state.positions.get(payload.symbol) is not None:
        return False
    if any(kind == "entry" for kind in state.order_kind.values()):
        return False
    if state.daily_trades >= MAX_TRADES_PER_DAY or state.daily_losses >= MAX_LOSSES_PER_DAY:
        return False
    if payload.depth_confirmed is False:
        return False
    min_score = A2_MIN_SCORE if payload.module is IvbModule.A2_RECLAIM else A1_MIN_SCORE
    if payload.signal_score < min_score or payload.size_multiplier <= 0:
        return False
    return payload.qty > 0 and payload.entry_price > 0 and payload.stop_price > 0 and payload.tp1_price > 0


def _record_realized_exit(
    state: IvbAuctionCoreState,
    setup: IvbSetup,
    fill_price: float,
    exit_qty: int,
    exit_commission: float,
    order_kind: str,
) -> None:
    if exit_qty <= 0 or setup.avg_entry <= 0:
        return
    direction = 1 if setup.direction is TradeDirection.LONG else -1
    gross = (fill_price - setup.avg_entry) * direction * exit_qty * spec_for(setup.symbol).point_value
    remaining_entry_commission = float(setup.metadata.get("_remaining_entry_commission", 0.0))
    open_qty = max(setup.qty_open, exit_qty)
    entry_commission_delta = remaining_entry_commission if exit_qty >= open_qty else remaining_entry_commission * (exit_qty / open_qty)
    setup.metadata["_remaining_entry_commission"] = max(0.0, remaining_entry_commission - entry_commission_delta)
    net = gross - entry_commission_delta - exit_commission
    state.daily_pnl += net
    if order_kind == "stop" or net < 0:
        state.daily_losses += 1


def _setup_from_payload(payload: IvbBarInput) -> IvbSetup:
    assert payload.module is not None
    assert payload.trigger is not None
    direction_name = "long" if payload.breakout_direction is TradeDirection.LONG else "short"
    setup_id = f"IVB-{payload.module.value}-{payload.bar_ts.strftime('%Y%m%d%H%M%S')}-{direction_name}"
    return IvbSetup(
        setup_id=setup_id,
        symbol=payload.symbol,
        module=payload.module,
        direction=payload.breakout_direction,
        trigger=payload.trigger,
        signal_time=payload.bar_ts,
        score=payload.signal_score,
        entry_price=payload.entry_price,
        stop_price=payload.stop_price,
        tp1_price=payload.tp1_price,
        tp2_price=payload.tp2_price,
        qty=payload.qty,
        size_multiplier=payload.size_multiplier,
        rr_to_tp1=payload.rr_to_tp1,
        metadata=dict(payload.decision_details),
    )


def _event(code: str, ts: datetime, symbol: str, details: dict) -> DecisionEvent:
    return DecisionEvent(code=code, ts=ts, symbol=symbol, timeframe="1m", details=dict(details))


def _update_last_decision(
    state: IvbAuctionCoreState,
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
