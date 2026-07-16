from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime, timezone

from strategies.core.actions import (
    CancelAction,
    FlattenPosition,
    ReplaceProtectiveStop,
    SubmitEntry,
    SubmitExit,
)
from strategies.core.events import DecisionEvent
from strategies.core.idle_market import idle_market_details
from strategies.momentum.downturn.models import ActivePosition, WorkingEntry

from .state import (
    DownturnAuthorization,
    DownturnCoreState,
    DownturnEntryRequest,
    DownturnFill,
    DownturnOrderUpdate,
    DownturnStopUpdateRequest,
)
from .entry_decision import (
    DownturnEntryProposal,
    entry_action,
    with_quantity,
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


def on_bar(
    state: DownturnCoreState,
    *,
    bar_count_5m: int | None = None,
    bar_ts: datetime | None = None,
    entry_request: DownturnEntryRequest | None = None,
    entry_proposal: DownturnEntryProposal | None = None,
    stop_update: DownturnStopUpdateRequest | None = None,
    flatten_reason: str | None = None,
    expire_entries: bool = False,
    idle_market_bars: Sequence[object] | None = None,
    idle_market_symbol: str = "",
    idle_market_timeframe: str = "5m",
) -> tuple[
    DownturnCoreState,
    list[SubmitEntry | ReplaceProtectiveStop | CancelAction | FlattenPosition],
    list[DecisionEvent],
]:
    next_state = deepcopy(state)
    actions: list[SubmitEntry | ReplaceProtectiveStop | CancelAction | FlattenPosition] = []
    events: list[DecisionEvent] = []
    event_ts = bar_ts or datetime.now(timezone.utc)

    if bar_count_5m is not None:
        next_state.bar_count_5m = bar_count_5m
    if bar_ts is not None:
        next_state.last_bar_ts = bar_ts

    if entry_request is not None:
        next_state.symbol = entry_request.symbol or next_state.symbol
        actions.append(
            SubmitEntry(
                client_order_id=entry_request.client_order_id,
                symbol=entry_request.symbol,
                side=entry_request.side,
                qty=entry_request.qty,
                order_type=entry_request.order_type,
                tif=entry_request.tif,
                price=entry_request.price,
                limit_price=entry_request.limit_price,
                stop_price=entry_request.stop_price,
                metadata={
                    "engine_tag": entry_request.engine_tag.value,
                    "signal_class": entry_request.signal_class,
                    "role": "entry",
                },
            )
        )
        events.append(
            DecisionEvent(
                code="ENTRY_REQUESTED",
                ts=event_ts,
                symbol=entry_request.symbol,
                timeframe="5m",
                details={
                    "engine_tag": entry_request.engine_tag.value,
                    "signal_class": entry_request.signal_class,
                    "qty": entry_request.qty,
                    "entry_price": entry_request.entry_price,
                    "stop0": entry_request.stop0,
                },
            )
        )

    if entry_proposal is not None:
        next_state.symbol = entry_proposal.symbol or next_state.symbol
        next_state.pending_entries[entry_proposal.client_order_id] = entry_proposal
        actions.append(entry_action(entry_proposal))
        events.append(
            DecisionEvent(
                code="ENTRY_PROPOSED",
                ts=event_ts,
                symbol=entry_proposal.symbol,
                timeframe="5m",
                details={
                    "engine_tag": entry_proposal.engine_tag.value,
                    "signal_class": entry_proposal.signal_class,
                    "qty": entry_proposal.qty,
                    "entry_price": entry_proposal.entry_price,
                    "stop0": entry_proposal.stop0,
                    "risk_dollars": entry_proposal.risk_dollars,
                },
            )
        )

    if stop_update is not None and next_state.position and next_state.position.stop_oms_order_id:
        actions.append(
            ReplaceProtectiveStop(
                symbol=next_state.symbol,
                target_order_id=next_state.position.stop_oms_order_id,
                side="BUY",
                stop_price=stop_update.stop_price,
                qty=stop_update.qty,
                reason=stop_update.reason,
                metadata={"entry_oms_order_id": next_state.position.entry_oms_order_id},
            )
        )
        events.append(
            DecisionEvent(
                code="STOP_REPLACEMENT_REQUESTED",
                ts=event_ts,
                symbol=next_state.symbol,
                timeframe="5m",
                details={"stop_price": stop_update.stop_price, "reason": stop_update.reason},
            )
        )

    if flatten_reason and next_state.position is not None:
        actions.append(
            FlattenPosition(
                symbol=next_state.symbol,
                reason=flatten_reason,
                metadata={"entry_oms_order_id": next_state.position.entry_oms_order_id},
            )
        )
        events.append(
            DecisionEvent(
                code="FLATTEN_REQUESTED",
                ts=event_ts,
                symbol=next_state.symbol,
                timeframe="5m",
                details={"reason": flatten_reason},
            )
        )

    if expire_entries:
        still_active: list[WorkingEntry] = []
        for entry in next_state.working_entries:
            bars_elapsed = next_state.bar_count_5m - entry.submitted_bar_idx
            if bars_elapsed >= entry.ttl_bars:
                actions.append(
                    CancelAction(
                        symbol=next_state.symbol,
                        target_order_id=entry.oms_order_id,
                        reason="ttl_expiry",
                        metadata={"engine_tag": entry.engine_tag.value, "signal_class": entry.signal_class},
                    )
                )
                events.append(
                    DecisionEvent(
                        code="ENTRY_EXPIRED",
                        ts=event_ts,
                        symbol=next_state.symbol,
                        timeframe="5m",
                        details={"ttl_bars": entry.ttl_bars, "signal_class": entry.signal_class},
                    )
                )
            else:
                still_active.append(entry)
        next_state.working_entries = still_active

    if idle_market_bars is not None and not actions and not events:
        observed_symbol = idle_market_symbol or next_state.symbol
        events.append(
            DecisionEvent(
                code="IDLE_MARKET_OBSERVED",
                ts=event_ts,
                symbol=observed_symbol,
                timeframe=idle_market_timeframe,
                details=idle_market_details(
                    idle_market_bars,
                    symbol=observed_symbol,
                    timeframe=idle_market_timeframe,
                ),
            )
        )

    _update_last_decision(next_state, events)
    return next_state, actions, events


def on_authorization(
    state: DownturnCoreState,
    authorization: DownturnAuthorization,
) -> tuple[DownturnCoreState, list[SubmitEntry], list[DecisionEvent]]:
    """Apply family portfolio feedback before an entry is submitted."""

    next_state = deepcopy(state)
    proposal = next_state.pending_entries.pop(authorization.client_order_id, None)
    if proposal is None:
        return next_state, [], []
    approved_qty = min(
        max(int(authorization.approved_qty), 0),
        max(int(authorization.requested_qty), 0),
    )
    approved = bool(authorization.approved and approved_qty > 0)
    if not approved:
        events = [
            DecisionEvent(
                code="ENTRY_DENIED",
                ts=authorization.timestamp,
                symbol=authorization.symbol or proposal.symbol,
                timeframe="5m",
                details={
                    "client_order_id": authorization.client_order_id,
                    "requested_qty": authorization.requested_qty,
                    "reason": authorization.denial_reason,
                },
            )
        ]
        _update_last_decision(next_state, events)
        return next_state, [], events

    approved_proposal = with_quantity(proposal, approved_qty)
    events = [
        DecisionEvent(
            code="ENTRY_AUTHORIZED",
            ts=authorization.timestamp,
            symbol=authorization.symbol or proposal.symbol,
            timeframe="5m",
            details={
                "client_order_id": authorization.client_order_id,
                "requested_qty": authorization.requested_qty,
                "approved_qty": approved_qty,
                "portfolio_decision_ref": authorization.portfolio_decision_ref,
            },
        )
    ]
    _update_last_decision(next_state, events)
    return next_state, [entry_action(approved_proposal)], events


def on_order_update(
    state: DownturnCoreState,
    update: DownturnOrderUpdate,
) -> tuple[DownturnCoreState, list[SubmitExit], list[DecisionEvent]]:
    next_state = deepcopy(state)
    actions: list[SubmitExit] = []
    events: list[DecisionEvent] = []
    event_ts = update.timestamp or datetime.now(timezone.utc)
    status = update.status.lower()

    if update.accepted_entry is not None and status in _ACK_STATUSES:
        accepted = update.accepted_entry
        next_state.symbol = accepted.symbol or next_state.symbol
        next_state.working_entries.append(
            WorkingEntry(
                oms_order_id=update.oms_order_id,
                engine_tag=accepted.engine_tag,
                signal_class=accepted.signal_class,
                entry_price=accepted.entry_price,
                stop0=accepted.stop0,
                qty=accepted.qty,
                submitted_bar_idx=accepted.submitted_bar_idx,
                ttl_bars=accepted.ttl_bars,
                composite_regime=accepted.composite_regime,
                vol_state=accepted.vol_state,
                in_correction=accepted.in_correction,
                predator=accepted.predator,
                tp_schedule=list(accepted.tp_schedule),
                signal_strength=accepted.signal_strength,
            )
        )
        events.append(
            DecisionEvent(
                code="ENTRY_SUBMITTED",
                ts=event_ts,
                symbol=accepted.symbol,
                timeframe="5m",
                details={
                    "oms_order_id": update.oms_order_id,
                    "signal_class": accepted.signal_class,
                    "qty": accepted.qty,
                },
            )
        )
    elif update.order_role == "stop" and next_state.position is not None and status in _ACK_STATUSES:
        next_state.position.stop_oms_order_id = update.oms_order_id
        events.append(
            DecisionEvent(
                        code="PROTECTIVE_STOP_SUBMITTED",
                        ts=event_ts,
                        symbol=next_state.symbol,
                timeframe="5m",
                details={"stop_oms_order_id": update.oms_order_id},
            )
        )
    elif status in _TERMINAL_STATUSES:
        next_state.working_entries = [
            entry for entry in next_state.working_entries if entry.oms_order_id != update.oms_order_id
        ]
        if next_state.position is not None and next_state.position.stop_oms_order_id == update.oms_order_id:
            next_state.position.stop_oms_order_id = ""
            events.append(
                DecisionEvent(
                        code="PROTECTIVE_STOP_CLEARED",
                        ts=event_ts,
                        symbol=next_state.symbol,
                    timeframe="5m",
                    details={"status": update.status},
                )
            )

    _update_last_decision(next_state, events)
    return next_state, actions, events


def on_fill(
    state: DownturnCoreState,
    fill: DownturnFill,
) -> tuple[DownturnCoreState, list[SubmitExit], list[DecisionEvent]]:
    next_state = deepcopy(state)
    actions: list[SubmitExit] = []
    events: list[DecisionEvent] = []
    event_ts = fill.fill_time or datetime.now(timezone.utc)

    matched_entry = None
    for entry in next_state.working_entries:
        if entry.oms_order_id == fill.oms_order_id:
            matched_entry = entry
            break

    if matched_entry is not None:
        next_state.working_entries = [
            entry for entry in next_state.working_entries if entry.oms_order_id != fill.oms_order_id
        ]
        qty = fill.fill_qty or matched_entry.qty
        next_state.position = ActivePosition(
            engine_tag=matched_entry.engine_tag,
            signal_class=matched_entry.signal_class,
            trade_id=matched_entry.oms_order_id,
            entry_price=fill.fill_price,
            stop0=matched_entry.stop0,
            qty=qty,
            remaining_qty=qty,
            entry_oms_order_id=matched_entry.oms_order_id,
            entry_time=event_ts,
            mfe_price=fill.fill_price,
            mae_price=fill.fill_price,
            chandelier_stop=matched_entry.stop0,
            tp_schedule=list(matched_entry.tp_schedule),
            composite_regime=matched_entry.composite_regime,
            vol_state=matched_entry.vol_state,
            in_correction=matched_entry.in_correction,
            predator=matched_entry.predator,
            commission=fill.commission,
        )
        next_state.bars_since_last_entry = 0
        actions.append(
            SubmitExit(
                client_order_id=f"{fill.oms_order_id}:protective_stop",
                symbol=next_state.symbol,
                side="BUY",
                qty=qty,
                order_type="STOP",
                tif="GTC",
                stop_price=matched_entry.stop0,
                metadata={"role": "protective_stop", "entry_oms_order_id": matched_entry.oms_order_id},
            )
        )
        events.append(
            DecisionEvent(
                code="ENTRY_FILLED",
                ts=event_ts,
                symbol=next_state.symbol,
                timeframe="5m",
                details={"fill_price": fill.fill_price, "qty": qty},
            )
        )
    elif next_state.position is not None and (
        fill.oms_order_id == next_state.position.stop_oms_order_id or fill.exit_type
    ):
        trade_id = next_state.position.trade_id
        next_state.position = None
        events.append(
            DecisionEvent(
                code="EXIT_FILLED",
                ts=event_ts,
                symbol=next_state.symbol or trade_id,
                timeframe="5m",
                details={"fill_price": fill.fill_price, "qty": fill.fill_qty, "exit_type": fill.exit_type},
            )
        )

    _update_last_decision(next_state, events)
    return next_state, actions, events


def _update_last_decision(state: DownturnCoreState, events: list[DecisionEvent]) -> None:
    if not events:
        return
    latest = events[-1]
    state.last_decision_code = latest.code
    state.last_decision_details = dict(latest.details)
    state.last_bar_ts = latest.ts
