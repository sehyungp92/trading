from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from strategies.core.actions import (
    FlattenPosition,
    NeutralAction,
    ReplaceProtectiveStop,
    SubmitEntry,
    SubmitExit,
)
from strategies.core.events import DecisionEvent
from strategies.momentum.helix_v40.config import (
    PositionState,
    SetupClass,
    SetupState,
)

from .state import (
    HelixV40CoreState,
    HelixV40EntryArmed,
    HelixV40ExpireSignatures,
    HelixV40Fill,
    HelixV40FlattenRequest,
    HelixV40OrderUpdate,
    HelixV40StopUpdateRequest,
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


# ── State sync (unchanged) ──────────────────────────────────────────


def build_core_state(engine) -> HelixV40CoreState:
    return HelixV40CoreState(
        positions=deepcopy(getattr(engine.positions, "positions", [])),
        pending_setups=deepcopy(getattr(engine.exec, "pending_setups", [])),
        open_risk_r=float(getattr(engine.risk, "open_risk_r", 0.0)),
        pending_risk_r=float(getattr(engine.risk, "pending_risk_r", 0.0)),
        dir_risk_r=deepcopy(getattr(engine.risk, "dir_risk_r", {})),
        vol_pct=float(getattr(engine.vol, "vol_pct", 0.0)),
        ts_history=list(getattr(engine, "_ts_history", [])),
        placed_signatures=set(getattr(engine, "_placed_signatures", set())),
        sig_expiry=deepcopy(getattr(engine, "_sig_expiry", {})),
        last_m_bar=deepcopy(getattr(engine, "_last_m_bar", {})),
        spread_recheck=deepcopy(getattr(engine, "_spread_recheck", [])),
        last_decision_code=engine._last_decision_code,
        last_decision_details=deepcopy(engine._last_decision_details),
        last_bar_ts=engine._last_bar_ts,
    )


def apply_core_state(engine, state: HelixV40CoreState) -> None:
    engine.positions.positions = deepcopy(state.positions)
    engine.exec.pending_setups = deepcopy(state.pending_setups)
    engine.risk.open_risk_r = float(state.open_risk_r)
    engine.risk.pending_risk_r = float(state.pending_risk_r)
    engine.risk.dir_risk_r = deepcopy(state.dir_risk_r)
    engine.vol.vol_pct = float(state.vol_pct)
    engine._ts_history = deque(deepcopy(state.ts_history), maxlen=10)
    engine._placed_signatures = set(state.placed_signatures)
    engine._sig_expiry = deepcopy(state.sig_expiry)
    engine._last_m_bar = deepcopy(state.last_m_bar)
    engine._spread_recheck = deepcopy(state.spread_recheck)
    engine._last_decision_code = state.last_decision_code
    engine._last_decision_details = deepcopy(state.last_decision_details)
    engine._last_bar_ts = state.last_bar_ts


# ── on_bar ───────────────────────────────────────────────────────────


def on_bar(
    state: HelixV40CoreState,
    *,
    bar_ts: datetime | None = None,
    armed_entries: list[HelixV40EntryArmed] | None = None,
    stop_updates: list[HelixV40StopUpdateRequest] | None = None,
    flatten_requests: list[HelixV40FlattenRequest] | None = None,
    expire_sigs: HelixV40ExpireSignatures | None = None,
    decision_code: str = "",
    decision_details: dict[str, Any] | None = None,
) -> tuple[HelixV40CoreState, list[NeutralAction], list[DecisionEvent]]:
    """Process a bar-level decision cycle.

    The engine evaluates signals, gates, and sizing, then passes results here
    as typed request objects.  The core records state changes and emits actions.
    """
    next_state = deepcopy(state)
    actions: list[NeutralAction] = []
    events: list[DecisionEvent] = []
    event_ts = bar_ts or datetime.now(timezone.utc)

    if bar_ts is not None:
        next_state.last_bar_ts = bar_ts

    # ── Entries armed ────────────────────────────────────────────
    for entry_armed in (armed_entries or []):
        setup = entry_armed.setup

        # Signature dedup
        if entry_armed.signature:
            next_state.placed_signatures.add(entry_armed.signature)
            next_state.sig_expiry[entry_armed.signature] = entry_armed.sig_expiry_ts

        # Class M tracking for Class T suppression
        if setup.cls == SetupClass.M:
            next_state.last_m_bar[setup.direction] = entry_armed.bar_idx_1h

        # Risk bookkeeping
        next_state.pending_risk_r += entry_armed.risk_r
        next_state.dir_risk_r.setdefault(setup.direction, 0.0)
        next_state.dir_risk_r[setup.direction] += entry_armed.risk_r

        # Add to pending setups
        setup_copy = deepcopy(setup)
        setup_copy.state = SetupState.PENDING
        setup_copy.contracts = entry_armed.contracts
        setup_copy.armed_risk_r = entry_armed.risk_r
        next_state.pending_setups.append(setup_copy)

        # Emit SubmitEntry
        side = "BUY" if setup.direction == 1 else "SELL"
        actions.append(
            SubmitEntry(
                client_order_id=f"{setup.setup_id}:entry",
                symbol="",
                side=side,
                qty=entry_armed.contracts,
                order_type="STOP_LIMIT",
                tif="DAY",
                stop_price=setup.entry_stop,
                limit_price=setup.entry_stop,
                metadata={
                    "setup_id": setup.setup_id,
                    "cls": setup.cls.value,
                    "direction": setup.direction,
                    "stop0": setup.stop0,
                    "armed_risk_r": entry_armed.risk_r,
                    "role": "entry",
                },
            )
        )
        events.append(
            DecisionEvent(
                code="ENTRY_ARMED",
                ts=event_ts,
                symbol=setup.setup_id,
                timeframe="1H",
                details={
                    "setup_id": setup.setup_id,
                    "cls": setup.cls.value,
                    "direction": setup.direction,
                    "contracts": entry_armed.contracts,
                    "entry_stop": setup.entry_stop,
                    "stop0": setup.stop0,
                    "armed_risk_r": entry_armed.risk_r,
                },
            )
        )

    # ── Stop updates ────────────────────────────────────────────
    for update in stop_updates or []:
        matched_pos = _find_position(next_state.positions, update.pos_id)
        if matched_pos is not None and matched_pos.stop_oms_id:
            matched_pos.stop_price = update.stop_price
            exit_side = "SELL" if matched_pos.direction == 1 else "BUY"
            actions.append(
                ReplaceProtectiveStop(
                    symbol=update.symbol,
                    target_order_id=matched_pos.stop_oms_id,
                    side=exit_side,
                    stop_price=update.stop_price,
                    qty=update.qty,
                    reason=update.reason,
                    metadata={"pos_id": matched_pos.pos_id},
                )
            )
            events.append(
                DecisionEvent(
                    code="STOP_REPLACEMENT_REQUESTED",
                    ts=event_ts,
                    symbol=update.symbol,
                    timeframe="1H",
                    details={
                        "pos_id": update.pos_id,
                        "stop_price": update.stop_price,
                        "reason": update.reason,
                    },
                )
            )

    # ── Flatten requests ────────────────────────────────────────
    for flatten in flatten_requests or []:
        matched_pos = _find_position(next_state.positions, flatten.pos_id)
        if matched_pos is not None:
            actions.append(
                FlattenPosition(
                    symbol=flatten.symbol,
                    reason=flatten.reason,
                    metadata={"pos_id": flatten.pos_id},
                )
            )
            events.append(
                DecisionEvent(
                    code="FLATTEN_REQUESTED",
                    ts=event_ts,
                    symbol=flatten.symbol,
                    timeframe="1H",
                    details={"reason": flatten.reason, "pos_id": flatten.pos_id},
                )
            )

    # ── Expire signatures ───────────────────────────────────────
    if expire_sigs is not None:
        cutoff = expire_sigs.now
        expired = [
            sig for sig, ts in next_state.sig_expiry.items() if ts <= cutoff
        ]
        for sig in expired:
            next_state.placed_signatures.discard(sig)
            next_state.sig_expiry.pop(sig, None)

    # ── Record decision ───────────────────────────────────────
    if decision_code:
        next_state.last_decision_code = decision_code
        next_state.last_decision_details = dict(decision_details or {})
        events.append(DecisionEvent(
            code=decision_code,
            ts=event_ts,
            symbol="",
            timeframe="1H",
            details=dict(decision_details or {}),
        ))

    _update_last_decision(next_state, events)
    return next_state, actions, events


# ── on_order_update ──────────────────────────────────────────────────


def on_order_update(
    state: HelixV40CoreState,
    update: HelixV40OrderUpdate,
) -> tuple[HelixV40CoreState, list[NeutralAction], list[DecisionEvent]]:
    """Process an OMS order status update."""
    next_state = deepcopy(state)
    actions: list[NeutralAction] = []
    events: list[DecisionEvent] = []
    event_ts = update.timestamp or datetime.now(timezone.utc)
    status = update.status.lower()

    if update.order_role == "stop" and status in _ACK_STATUSES:
        # Stop order accepted -- record stop_oms_id on position
        if update.pos_id:
            pos = _find_position(next_state.positions, update.pos_id)
            if pos is not None:
                pos.stop_oms_id = update.oms_order_id
        events.append(
            DecisionEvent(
                code="PROTECTIVE_STOP_SUBMITTED",
                ts=event_ts,
                symbol="",
                timeframe="1H",
                details={
                    "stop_oms_order_id": update.oms_order_id,
                    "pos_id": update.pos_id,
                },
            )
        )

    elif status in _TERMINAL_STATUSES:
        # Check pending setups first
        removed_setup = None
        for setup in next_state.pending_setups:
            if (
                setup.entry_oms_id == update.oms_order_id
                or setup.catchup_oms_id == update.oms_order_id
            ):
                removed_setup = setup
                break

        if removed_setup is not None:
            next_state.pending_setups = [
                s
                for s in next_state.pending_setups
                if not (
                    s.entry_oms_id == update.oms_order_id
                    or s.catchup_oms_id == update.oms_order_id
                )
            ]
            removed_setup.state = SetupState.CANCELED
            next_state.pending_risk_r = max(
                0.0, next_state.pending_risk_r - removed_setup.armed_risk_r
            )
            if removed_setup.direction in next_state.dir_risk_r:
                next_state.dir_risk_r[removed_setup.direction] = max(
                    0.0,
                    next_state.dir_risk_r[removed_setup.direction]
                    - removed_setup.armed_risk_r,
                )
            events.append(
                DecisionEvent(
                    code="ENTRY_CANCELLED",
                    ts=event_ts,
                    symbol="",
                    timeframe="1H",
                    details={
                        "oms_order_id": update.oms_order_id,
                        "status": update.status,
                    },
                )
            )
        else:
            # Check stop orders on positions
            for pos in next_state.positions:
                if pos.stop_oms_id == update.oms_order_id:
                    pos.stop_oms_id = None
                    events.append(
                        DecisionEvent(
                            code="PROTECTIVE_STOP_CLEARED",
                            ts=event_ts,
                            symbol="",
                            timeframe="1H",
                            details={
                                "status": update.status,
                                "pos_id": pos.pos_id,
                            },
                        )
                    )
                    break

    _update_last_decision(next_state, events)
    return next_state, actions, events


# ── on_fill ──────────────────────────────────────────────────────────


def on_fill(
    state: HelixV40CoreState,
    fill: HelixV40Fill,
) -> tuple[HelixV40CoreState, list[NeutralAction], list[DecisionEvent]]:
    """Process an OMS fill event."""
    next_state = deepcopy(state)
    actions: list[NeutralAction] = []
    events: list[DecisionEvent] = []
    event_ts = fill.fill_time or datetime.now(timezone.utc)
    pv = fill.point_value

    # ── Entry fill (context provided by engine, or match by oms_order_id) ──
    entry_setup = None
    is_catastrophic = fill.is_catastrophic
    is_teleport = fill.is_teleport

    if fill.entry_context is not None:
        entry_setup = fill.entry_context.setup
        is_catastrophic = is_catastrophic or fill.entry_context.is_catastrophic
        is_teleport = is_teleport or fill.entry_context.is_teleport
    else:
        # Fallback: match against pending setups by oms_order_id
        for s in next_state.pending_setups:
            if s.entry_oms_id == fill.oms_order_id or s.catchup_oms_id == fill.oms_order_id:
                entry_setup = s
                break

    if entry_setup is not None:
        setup = entry_setup

        if is_catastrophic:
            # Catastrophic fill -- flatten immediately
            next_state.pending_setups = [
                s for s in next_state.pending_setups
                if s.setup_id != setup.setup_id
            ]
            next_state.pending_risk_r = max(
                0.0, next_state.pending_risk_r - setup.armed_risk_r
            )
            if setup.direction in next_state.dir_risk_r:
                next_state.dir_risk_r[setup.direction] = max(
                    0.0,
                    next_state.dir_risk_r[setup.direction] - setup.armed_risk_r,
                )

            actions.append(
                FlattenPosition(
                    symbol="",
                    reason="CATASTROPHIC_FILL",
                    metadata={"setup_id": setup.setup_id},
                )
            )
            events.append(
                DecisionEvent(
                    code="CATASTROPHIC_FILL",
                    ts=event_ts,
                    symbol=setup.setup_id,
                    timeframe="1H",
                    details={
                        "fill_price": fill.fill_price,
                        "expected_price": setup.entry_stop,
                    },
                )
            )
            _update_last_decision(next_state, events)
            return next_state, actions, events

        # Normal entry fill -- create position and emit protective stop
        qty = fill.fill_qty or setup.contracts
        pos = PositionState(
            pos_id=setup.setup_id,
            direction=setup.direction,
            avg_entry=fill.fill_price,
            contracts=qty,
            unit1_risk_usd=setup.unit1_risk_usd,
            origin_class=setup.cls,
            origin_setup_id=setup.setup_id,
            entry_ts=event_ts,
            stop_price=setup.stop0,
            entry_contracts=qty,
            alignment_score_at_entry=setup.alignment_score,
            teleport_penalty=is_teleport,
            highest_since_entry=fill.fill_price,
            lowest_since_entry=fill.fill_price,
        )
        next_state.positions.append(pos)

        # Risk: promote pending -> open
        next_state.pending_risk_r = max(
            0.0, next_state.pending_risk_r - setup.armed_risk_r
        )
        next_state.open_risk_r += setup.armed_risk_r

        # Remove from pending
        next_state.pending_setups = [
            s for s in next_state.pending_setups
            if s.setup_id != setup.setup_id
        ]

        # Emit protective stop
        exit_side = "SELL" if setup.direction == 1 else "BUY"
        actions.append(
            SubmitExit(
                client_order_id=f"{setup.setup_id}:protective_stop",
                symbol="",
                side=exit_side,
                qty=qty,
                order_type="STOP",
                tif="GTC",
                stop_price=setup.stop0,
                metadata={
                    "role": "protective_stop",
                    "setup_id": setup.setup_id,
                },
            )
        )
        events.append(
            DecisionEvent(
                code="ENTRY_FILLED",
                ts=event_ts,
                symbol=setup.setup_id,
                timeframe="1H",
                details={
                    "fill_price": fill.fill_price,
                    "qty": qty,
                    "setup_id": setup.setup_id,
                    "cls": setup.cls.value,
                    "direction": setup.direction,
                },
            )
        )
        _update_last_decision(next_state, events)
        return next_state, actions, events

    # ── Stop fill ───────────────────────────────────────────────
    for pos in next_state.positions:
        if pos.stop_oms_id == fill.oms_order_id:
            actual_pnl = (
                (fill.fill_price - pos.avg_entry)
                * pos.direction
                * pv
                * fill.fill_qty
            )
            pos.realized_partial_usd += actual_pnl
            pos.contracts = max(0, pos.contracts - fill.fill_qty)

            if pos.contracts <= 0:
                exit_reason = (
                    "TRAILING_STOP" if pos.trailing_active else "INITIAL_STOP"
                )
                next_state.open_risk_r = max(
                    0.0, next_state.open_risk_r - pos.current_risk_r
                )
                if pos.direction in next_state.dir_risk_r:
                    next_state.dir_risk_r[pos.direction] = max(
                        0.0,
                        next_state.dir_risk_r[pos.direction] - pos.current_risk_r,
                    )
                next_state.positions = [
                    p for p in next_state.positions if p.pos_id != pos.pos_id
                ]
                events.append(
                    DecisionEvent(
                        code="EXIT_FILLED",
                        ts=event_ts,
                        symbol=pos.origin_setup_id,
                        timeframe="1H",
                        details={
                            "fill_price": fill.fill_price,
                            "qty": fill.fill_qty,
                            "exit_type": exit_reason,
                            "r_mult": _compute_r_mult(pos),
                        },
                    )
                )

            _update_last_decision(next_state, events)
            return next_state, actions, events

    # ── Exit fill (partial / market exit) ───────────────────────
    for pos in next_state.positions:
        if fill.oms_order_id in pos.exit_oms_ids:
            actual_pnl = (
                (fill.fill_price - pos.avg_entry)
                * pos.direction
                * pv
                * fill.fill_qty
            )
            if fill.oms_order_id in pos.pending_exit_estimates:
                estimated = pos.pending_exit_estimates.pop(fill.oms_order_id)
                correction = actual_pnl - estimated
                pos.realized_partial_usd += correction
            else:
                pos.realized_partial_usd += actual_pnl

            pos.exit_oms_ids.remove(fill.oms_order_id)

            if pos.contracts <= 0:
                next_state.open_risk_r = max(
                    0.0, next_state.open_risk_r - pos.current_risk_r
                )
                if pos.direction in next_state.dir_risk_r:
                    next_state.dir_risk_r[pos.direction] = max(
                        0.0,
                        next_state.dir_risk_r[pos.direction] - pos.current_risk_r,
                    )
                next_state.positions = [
                    p for p in next_state.positions if p.pos_id != pos.pos_id
                ]
                events.append(
                    DecisionEvent(
                        code="EXIT_FILLED",
                        ts=event_ts,
                        symbol=pos.origin_setup_id,
                        timeframe="1H",
                        details={
                            "fill_price": fill.fill_price,
                            "qty": fill.fill_qty,
                            "exit_type": "EXIT_FILL",
                            "r_mult": _compute_r_mult(pos),
                        },
                    )
                )

            _update_last_decision(next_state, events)
            return next_state, actions, events

    # No match
    _update_last_decision(next_state, events)
    return next_state, actions, events


# ── Helpers ──────────────────────────────────────────────────────────


def _find_position(
    positions: list[PositionState], pos_id: str
) -> PositionState | None:
    for pos in positions:
        if pos.pos_id == pos_id:
            return pos
    return None


def _compute_r_mult(pos: PositionState) -> float:
    if pos.unit1_risk_usd > 0 and pos.entry_contracts > 0:
        return pos.realized_partial_usd / (
            pos.unit1_risk_usd * pos.entry_contracts
        )
    return 0.0


def _update_last_decision(
    state: HelixV40CoreState, events: list[DecisionEvent]
) -> None:
    if not events:
        return
    latest = events[-1]
    state.last_decision_code = latest.code
    state.last_decision_details = dict(latest.details)
    state.last_bar_ts = latest.ts
