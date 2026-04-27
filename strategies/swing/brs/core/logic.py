from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from strategies.core.actions import (
    FlattenPosition,
    ReplaceProtectiveStop,
    SubmitAddOnEntry,
    SubmitEntry,
    SubmitExit,
    SubmitPartialExit,
)
from strategies.core.events import DecisionEvent
from strategies.swing.brs.models import BDArmState, Direction, EntryType, LHArmState, S2ArmState, S3ArmState
from strategies.swing.brs.positions import ActionResult, BRSPositionState, PendingOrder, PositionAction

from .state import (
    BRSAddOnRequest,
    BRSCoreState,
    BRSEntryRequest,
    BRSFill,
    BRSFlattenRequest,
    BRSOrderUpdate,
    BRSScaleOutRequest,
    BRSStopUpdateRequest,
)

_TERMINAL_STATUSES = {
    "cancelled",
    "expired",
    "rejected",
    "order_cancelled",
    "order_expired",
    "order_rejected",
}
_ACK_STATUSES = {"accepted", "acknowledged", "submitted"}


def build_core_state(engine) -> BRSCoreState:
    return BRSCoreState(
        daily_ctx=deepcopy(engine._daily_ctx),
        bias=deepcopy(engine._bias),
        prev_regime_on=deepcopy(engine._prev_regime_on),
        prev_regime=deepcopy(engine._prev_regime),
        position=deepcopy(engine._position),
        cooldown_until=deepcopy(engine._cooldown_until),
        lh_arm=deepcopy(engine._lh_arm),
        bd_arm=deepcopy(engine._bd_arm),
        s2_arm=deepcopy(engine._s2_arm),
        s3_arm=deepcopy(engine._s3_arm),
        swing_highs=deepcopy(engine._swing_highs),
        short_bias_no_trade=deepcopy(engine._short_bias_no_trade),
        hourly_bar_count=deepcopy(engine._hourly_bar_count),
        last_daily_close=deepcopy(engine._last_daily_close),
        pending_orders=deepcopy(engine._pending_orders),
        filled_order_ids=set(engine._filled_order_ids),
        closing=deepcopy(engine._closing),
        last_decision_code=engine._last_decision_code,
        last_decision_details=deepcopy(engine._last_decision_details),
        last_bar_ts=engine._last_bar_ts,
    )


def apply_core_state(engine, state: BRSCoreState) -> None:
    engine._daily_ctx = deepcopy(state.daily_ctx)
    engine._bias = deepcopy(state.bias)
    engine._prev_regime_on = deepcopy(state.prev_regime_on)
    engine._prev_regime = deepcopy(state.prev_regime)
    engine._position = deepcopy(state.position)
    engine._cooldown_until = deepcopy(state.cooldown_until)
    engine._lh_arm = deepcopy(state.lh_arm)
    engine._bd_arm = deepcopy(state.bd_arm)
    engine._s2_arm = deepcopy(state.s2_arm)
    engine._s3_arm = deepcopy(state.s3_arm)
    engine._swing_highs = deepcopy(state.swing_highs)
    engine._short_bias_no_trade = deepcopy(state.short_bias_no_trade)
    engine._hourly_bar_count = deepcopy(state.hourly_bar_count)
    engine._last_daily_close = deepcopy(state.last_daily_close)
    engine._pending_orders = deepcopy(state.pending_orders)
    engine._filled_order_ids = set(state.filled_order_ids)
    engine._closing = deepcopy(state.closing)
    engine._last_decision_code = state.last_decision_code
    engine._last_decision_details = deepcopy(state.last_decision_details)
    engine._last_bar_ts = state.last_bar_ts


def translate_position_actions(
    symbol: str,
    position: BRSPositionState,
    actions: list[ActionResult],
):
    side = "BUY" if position.direction == Direction.SHORT else "SELL"
    translated = []
    for index, action in enumerate(actions, start=1):
        if action.action == PositionAction.EXIT:
            translated.append(
                FlattenPosition(
                    symbol=symbol,
                    reason=action.exit_reason.value if action.exit_reason else "unknown",
                    side=side,
                    qty=position.qty,
                    metadata={"pos_id": position.pos_id, "exit_price": action.exit_price},
                )
            )
            continue
        if action.action == PositionAction.STOP_UPDATE and position.stop_oms_id:
            translated.append(
                ReplaceProtectiveStop(
                    symbol=symbol,
                    target_order_id=position.stop_oms_id,
                    side=side,
                    stop_price=action.new_stop,
                    qty=position.qty,
                    reason="trail",
                    metadata={"pos_id": position.pos_id},
                )
            )
            continue
        if action.action == PositionAction.SCALE_OUT:
            translated.append(
                SubmitPartialExit(
                    client_order_id=f"{position.pos_id}-scale-{index}",
                    symbol=symbol,
                    side=side,
                    qty=action.scale_out_qty,
                    order_type="LIMIT",
                    limit_price=action.exit_price,
                    metadata={"pos_id": position.pos_id},
                )
            )
    return translated


def on_bar(
    state: BRSCoreState,
    *,
    bar_ts: datetime | None = None,
    entry_request: BRSEntryRequest | None = None,
    add_on_request: BRSAddOnRequest | None = None,
    stop_update: BRSStopUpdateRequest | None = None,
    scale_out_request: BRSScaleOutRequest | None = None,
    flatten_request: BRSFlattenRequest | None = None,
) -> tuple[
    BRSCoreState,
    list[SubmitEntry | SubmitAddOnEntry | ReplaceProtectiveStop | SubmitPartialExit | FlattenPosition],
    list[DecisionEvent],
]:
    next_state = deepcopy(state)
    actions: list[SubmitEntry | SubmitAddOnEntry | ReplaceProtectiveStop | SubmitPartialExit | FlattenPosition] = []
    events: list[DecisionEvent] = []
    event_ts = bar_ts or datetime.now(timezone.utc)

    if bar_ts is not None:
        next_state.last_bar_ts = bar_ts

    if entry_request is not None:
        actions.append(
            SubmitEntry(
                client_order_id=entry_request.client_order_id,
                symbol=entry_request.symbol,
                side="BUY" if entry_request.signal.direction == Direction.LONG else "SELL",
                qty=entry_request.qty,
                order_type=entry_request.order_type,
                tif=entry_request.tif,
                limit_price=entry_request.signal.signal_price,
                metadata={
                    "pos_id": entry_request.pos_id,
                    "entry_type": entry_request.signal.entry_type.value,
                    "role": "entry",
                },
                risk_context={
                    "stop_for_risk": entry_request.signal.stop_price,
                    "planned_entry_price": entry_request.signal.signal_price,
                },
            )
        )
        events.append(
            DecisionEvent(
                code="ENTRY_REQUESTED",
                ts=event_ts,
                symbol=entry_request.symbol,
                timeframe="1h",
                details={
                    "pos_id": entry_request.pos_id,
                    "entry_type": entry_request.signal.entry_type.value,
                    "qty": entry_request.qty,
                },
            )
        )

    if add_on_request is not None:
        actions.append(
            SubmitAddOnEntry(
                client_order_id=add_on_request.client_order_id,
                symbol=add_on_request.symbol,
                side="BUY" if add_on_request.signal.direction == Direction.LONG else "SELL",
                qty=add_on_request.qty,
                order_type=add_on_request.order_type,
                tif=add_on_request.tif,
                limit_price=add_on_request.signal.signal_price,
                metadata={
                    "pos_id": add_on_request.pos_id,
                    "entry_type": add_on_request.signal.entry_type.value,
                },
                risk_context={
                    "stop_for_risk": add_on_request.signal.stop_price,
                    "planned_entry_price": add_on_request.signal.signal_price,
                },
            )
        )
        events.append(
            DecisionEvent(
                code="ADD_ON_REQUESTED",
                ts=event_ts,
                symbol=add_on_request.symbol,
                timeframe="1h",
                details={
                    "pos_id": add_on_request.pos_id,
                    "entry_type": add_on_request.signal.entry_type.value,
                    "qty": add_on_request.qty,
                },
            )
        )

    if stop_update is not None:
        position = next_state.position.get(stop_update.symbol)
        if position is not None and position.stop_oms_id:
            actions.append(
                ReplaceProtectiveStop(
                    symbol=stop_update.symbol,
                    target_order_id=position.stop_oms_id,
                    side="BUY" if position.direction == Direction.SHORT else "SELL",
                    stop_price=stop_update.stop_price,
                    qty=stop_update.qty,
                    reason=stop_update.reason,
                    metadata={"pos_id": position.pos_id},
                )
            )
            events.append(
                DecisionEvent(
                    code="STOP_REPLACEMENT_REQUESTED",
                    ts=event_ts,
                    symbol=stop_update.symbol,
                    timeframe="1h",
                    details={"stop_price": stop_update.stop_price, "reason": stop_update.reason},
                )
            )

    if scale_out_request is not None:
        position = next_state.position.get(scale_out_request.symbol)
        if position is not None:
            actions.append(
                SubmitPartialExit(
                    client_order_id=scale_out_request.client_order_id,
                    symbol=scale_out_request.symbol,
                    side="BUY" if position.direction == Direction.SHORT else "SELL",
                    qty=scale_out_request.qty,
                    order_type=scale_out_request.order_type,
                    tif=scale_out_request.tif,
                    limit_price=scale_out_request.limit_price,
                    metadata={"pos_id": scale_out_request.pos_id},
                )
            )
            events.append(
                DecisionEvent(
                    code="PARTIAL_EXIT_REQUESTED",
                    ts=event_ts,
                    symbol=scale_out_request.symbol,
                    timeframe="1h",
                    details={"qty": scale_out_request.qty, "limit_price": scale_out_request.limit_price},
                )
            )

    if flatten_request is not None:
        position = next_state.position.get(flatten_request.symbol)
        if position is not None:
            next_state.closing[flatten_request.symbol] = flatten_request.reason
            actions.append(
                FlattenPosition(
                    symbol=flatten_request.symbol,
                    reason=flatten_request.reason,
                    side="BUY" if position.direction == Direction.SHORT else "SELL",
                    qty=position.qty,
                    metadata={"pos_id": position.pos_id},
                )
            )
            events.append(
                DecisionEvent(
                    code="FLATTEN_REQUESTED",
                    ts=event_ts,
                    symbol=flatten_request.symbol,
                    timeframe="1h",
                    details={"reason": flatten_request.reason},
                )
            )

    _update_last_decision(next_state, events)
    return next_state, actions, events


def on_order_update(
    state: BRSCoreState,
    update: BRSOrderUpdate,
) -> tuple[BRSCoreState, list[SubmitExit], list[DecisionEvent]]:
    next_state = deepcopy(state)
    actions: list[SubmitExit] = []
    events: list[DecisionEvent] = []
    event_ts = update.timestamp or datetime.now(timezone.utc)
    status = update.status.lower()

    if update.accepted_order is not None and status in _ACK_STATUSES:
        accepted = deepcopy(update.accepted_order)
        accepted.oms_order_id = update.oms_order_id
        next_state.pending_orders[update.oms_order_id] = accepted
        code = {
            "entry": "ENTRY_SUBMITTED",
            "pyramid": "ADD_ON_SUBMITTED",
            "scale_out": "PARTIAL_EXIT_SUBMITTED",
        }.get(accepted.role, "ORDER_SUBMITTED")
        events.append(
            DecisionEvent(
                code=code,
                ts=event_ts,
                symbol=accepted.symbol,
                timeframe="1h",
                details={"oms_order_id": update.oms_order_id, "role": accepted.role, "qty": accepted.qty},
            )
        )
    elif update.order_role == "stop" and status in _ACK_STATUSES:
        symbol = update.symbol
        position = next_state.position.get(symbol)
        if position is not None:
            position.stop_oms_id = update.oms_order_id
            events.append(
                DecisionEvent(
                    code="PROTECTIVE_STOP_SUBMITTED",
                    ts=event_ts,
                    symbol=symbol,
                    timeframe="1h",
                    details={"stop_oms_order_id": update.oms_order_id},
                )
            )
    elif status in _TERMINAL_STATUSES:
        pending = next_state.pending_orders.pop(update.oms_order_id, None)
        if pending is not None:
            events.append(
                DecisionEvent(
                    code="ORDER_TERMINATED",
                    ts=event_ts,
                    symbol=pending.symbol,
                    timeframe="1h",
                    details={"oms_order_id": update.oms_order_id, "role": pending.role, "status": update.status},
                )
            )
        symbol = update.symbol
        position = next_state.position.get(symbol)
        if position is not None and position.stop_oms_id == update.oms_order_id:
            position.stop_oms_id = None
            events.append(
                DecisionEvent(
                    code="PROTECTIVE_STOP_CLEARED",
                    ts=event_ts,
                    symbol=symbol,
                    timeframe="1h",
                    details={"status": update.status},
                )
            )

    _update_last_decision(next_state, events)
    return next_state, actions, events


def on_fill(
    state: BRSCoreState,
    fill: BRSFill,
) -> tuple[BRSCoreState, list[SubmitExit], list[DecisionEvent]]:
    next_state = deepcopy(state)
    actions: list[SubmitExit] = []
    events: list[DecisionEvent] = []
    event_ts = fill.fill_time or datetime.now(timezone.utc)

    pending = next_state.pending_orders.get(fill.oms_order_id)
    if pending is not None:
        if pending.role == "entry" and pending.signal is not None:
            next_state.pending_orders.pop(fill.oms_order_id, None)
            qty = fill.fill_qty or pending.qty
            risk_per_unit = abs(pending.signal.stop_price - fill.fill_price)
            pos = BRSPositionState(
                symbol=pending.symbol,
                pos_id=pending.pos_id,
                direction=pending.signal.direction,
                entry_price=fill.fill_price,
                qty=qty,
                original_qty=qty,
                risk_per_unit=max(risk_per_unit, 1e-9),
                stop_price=pending.signal.stop_price,
                entry_type=pending.signal.entry_type.value,
                regime_at_entry=pending.signal.regime_at_entry,
                entry_ts=event_ts,
                mfe_price=fill.fill_price,
                mae_price=fill.fill_price,
                quality_score=pending.signal.quality_score,
                vol_factor=pending.signal.vol_factor,
                commission=fill.commission,
            )
            pos.entry_oms_id = fill.oms_order_id
            if fill.entry_context is not None:
                if fill.entry_context.tranche_b_qty > 0:
                    pos.tranche_b_open = True
                    pos.tranche_b_qty = fill.entry_context.tranche_b_qty
                if fill.entry_context.cooldown_until is not None:
                    next_state.cooldown_until[pending.symbol] = fill.entry_context.cooldown_until
                if fill.entry_context.reset_short_bias:
                    next_state.short_bias_no_trade[pending.symbol] = 0
                _consume_arm_state(next_state, pending.symbol, fill.entry_context.reset_arm_state)
            next_state.position[pending.symbol] = pos
            actions.append(
                SubmitExit(
                    client_order_id=f"{fill.oms_order_id}:protective_stop",
                    symbol=pending.symbol,
                    side="BUY" if pos.direction == Direction.SHORT else "SELL",
                    qty=qty,
                    order_type="STOP",
                    tif="GTC",
                    stop_price=pos.stop_price,
                    metadata={"role": "protective_stop", "pos_id": pos.pos_id},
                )
            )
            events.append(
                DecisionEvent(
                    code="ENTRY_FILLED",
                    ts=event_ts,
                    symbol=pending.symbol,
                    timeframe="1h",
                    details={"fill_price": fill.fill_price, "qty": qty, "entry_type": pos.entry_type},
                )
            )
        elif pending.role == "pyramid":
            next_state.pending_orders.pop(fill.oms_order_id, None)
            position = next_state.position.get(pending.symbol)
            if position is not None:
                qty = fill.fill_qty or pending.qty
                position.apply_pyramid(qty, fill.fill_price)
                position.commission += fill.commission
                events.append(
                    DecisionEvent(
                        code="ADD_ON_FILLED",
                        ts=event_ts,
                        symbol=pending.symbol,
                        timeframe="1h",
                        details={"fill_price": fill.fill_price, "qty": qty, "pos_id": position.pos_id},
                    )
                )
        elif pending.role == "scale_out":
            next_state.pending_orders.pop(fill.oms_order_id, None)
            position = next_state.position.get(pending.symbol)
            if position is not None:
                exit_qty = min(fill.fill_qty or pending.qty, position.qty)
                position.commission += fill.commission
                if exit_qty >= position.qty:
                    next_state.position[pending.symbol] = None
                    next_state.closing.pop(pending.symbol, None)
                    events.append(
                        DecisionEvent(
                            code="EXIT_FILLED",
                            ts=event_ts,
                            symbol=pending.symbol,
                            timeframe="1h",
                            details={"fill_price": fill.fill_price, "qty": exit_qty, "exit_type": "scale_out_complete"},
                        )
                    )
                else:
                    position.qty -= exit_qty
                    events.append(
                        DecisionEvent(
                            code="PARTIAL_EXIT_FILLED",
                            ts=event_ts,
                            symbol=pending.symbol,
                            timeframe="1h",
                            details={"fill_price": fill.fill_price, "qty": exit_qty},
                        )
                    )
        _update_last_decision(next_state, events)
        return next_state, actions, events

    for symbol, position in next_state.position.items():
        if position is None:
            continue
        matched_stop = fill.oms_order_id == position.stop_oms_id
        matched_flatten = bool(fill.symbol) and fill.symbol == symbol and symbol in next_state.closing
        if not matched_stop and not matched_flatten:
            continue
        next_state.position[symbol] = None
        next_state.closing.pop(symbol, None)
        events.append(
            DecisionEvent(
                code="EXIT_FILLED",
                ts=event_ts,
                symbol=symbol,
                timeframe="1h",
                details={
                    "fill_price": fill.fill_price,
                    "qty": fill.fill_qty or position.qty,
                    "exit_type": fill.exit_type or ("stop" if matched_stop else "flatten"),
                },
            )
        )
        break

    _update_last_decision(next_state, events)
    return next_state, actions, events


def _consume_arm_state(state: BRSCoreState, symbol: str, entry_type: EntryType | None) -> None:
    if entry_type == EntryType.S2_BREAKDOWN:
        state.s2_arm[symbol] = S2ArmState()
    elif entry_type == EntryType.S3_IMPULSE:
        state.s3_arm[symbol] = S3ArmState()
    elif entry_type == EntryType.LH_REJECTION:
        state.lh_arm[symbol] = LHArmState()
    elif entry_type == EntryType.BD_CONTINUATION:
        state.bd_arm[symbol] = BDArmState()


def _update_last_decision(state: BRSCoreState, events: list[DecisionEvent]) -> None:
    if not events:
        return
    latest = events[-1]
    state.last_decision_code = latest.code
    state.last_decision_details = dict(latest.details)
    state.last_bar_ts = latest.ts
