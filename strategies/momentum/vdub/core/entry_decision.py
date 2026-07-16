"""Pure Vdub pre-submit signal, gate, sizing, and order proposal logic."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from strategies.core.actions import SubmitEntry
from strategies.momentum.vdub import config as C
from strategies.momentum.vdub import risk, signals
from strategies.momentum.vdub.models import (
    DayCounters,
    Direction,
    EntryType,
    PivotPoint,
    PositionState,
    RegimeState,
    SessionWindow,
    SubWindow,
    WorkingEntry,
)


@dataclass(slots=True, frozen=True)
class VdubSignalSelection:
    entry_type: EntryType
    signal: dict[str, Any]
    vwap_used: float = 0.0


@dataclass(slots=True, frozen=True)
class VdubEntryProposal:
    client_order_id: str
    symbol: str
    direction: Direction
    qty: int
    stop_entry: float
    limit_entry: float
    initial_stop: float
    entry_type: EntryType
    is_flip: bool
    is_pyramid: bool
    class_mult: float
    vwap_used: float
    session: SessionWindow
    sub_window: SubWindow
    submitted_bar_idx: int
    signal_id: str
    bar_id: str
    exchange_timestamp: datetime
    unit_risk: float
    r_points: float
    risk_dollars: float


def select_entry_signal(
    *,
    closes_15m: np.ndarray,
    lows_15m: np.ndarray,
    highs_15m: np.ndarray,
    svwap: np.ndarray,
    vwap_a: np.ndarray,
    pivots_1h: list[PivotPoint],
    n_1h_bars: int,
    atr15: float,
    direction: Direction,
    sub_window: SubWindow,
    trend_1h: int,
    type_a_enabled: bool = True,
    type_b_enabled: bool = True,
    type_c_enabled: bool = False,
) -> VdubSignalSelection | None:
    """Select the first eligible Vdub signal without mutating strategy state."""

    signal = None
    entry_type = EntryType.TYPE_A
    vwap_used = 0.0
    if type_a_enabled:
        signal = signals.type_a_check(
            closes_15m,
            lows_15m,
            highs_15m,
            svwap,
            vwap_a,
            atr15,
            direction,
            sub_window,
        )
        if signal:
            vwap_used = float(signal.get("vwap_used", 0.0) or 0.0)

    if (
        signal is None
        and C.USE_TYPE_B
        and type_b_enabled
        and sub_window.value in C.TYPE_B_ALLOWED_WINDOWS
    ):
        type_b_ok = True
        if C.TYPE_B_REQUIRE_1H_ALIGN:
            type_b_ok = (
                direction == Direction.LONG and trend_1h == 1
            ) or (
                direction == Direction.SHORT and trend_1h == -1
            )
        if type_b_ok:
            signal = signals.type_b_check(
                closes_15m,
                lows_15m,
                highs_15m,
                pivots_1h,
                n_1h_bars,
                atr15,
                direction,
            )
            if signal:
                entry_type = EntryType.TYPE_B

    if signal is None and C.USE_TYPE_C and type_c_enabled:
        signal = signals.type_c_continuation_check(
            closes_15m,
            lows_15m,
            highs_15m,
            svwap,
            vwap_a,
            atr15,
            direction,
            sub_window,
        )
        if signal:
            entry_type = EntryType.TYPE_C
            vwap_used = float(signal.get("vwap_used", 0.0) or 0.0)

    if signal is None:
        return None
    return VdubSignalSelection(
        entry_type=entry_type,
        signal=dict(signal),
        vwap_used=vwap_used,
    )


def build_entry_proposal(
    *,
    selection: VdubSignalSelection,
    symbol: str,
    direction: Direction,
    session: SessionWindow,
    sub_window: SubWindow,
    now: datetime,
    bar_idx: int,
    bar_high: float,
    bar_low: float,
    close_price: float,
    atr15: float,
    atr1h: float,
    pivots_1h: list[PivotPoint],
    regime: RegimeState,
    counters: DayCounters,
    positions: Sequence[PositionState],
    equity: float,
    is_flip: bool,
    class_mult: float,
    session_mult: float,
    hourly_mult: float,
    point_value: float,
    tick_size: float,
    entry_size_mult: float = 1.0,
    fixed_qty: int | None = None,
    post_size_multipliers: Sequence[float] = (),
    signal_id: str = "",
    bar_id: str = "",
) -> tuple[VdubEntryProposal | None, str]:
    """Build the deterministic order proposal after common state-dependent gates."""

    active = [position for position in positions if position.qty_open > 0]
    if any(position.direction != direction for position in active):
        return None, "opposite_position"

    is_pyramid = False
    existing = next((position for position in active if position.direction == direction), None)
    if existing is not None:
        if not risk.pyramid_eligible(existing, direction, close_price, counters):
            return None, "pyramid_not_eligible"
        is_pyramid = True

    atr15_ticks = atr15 / tick_size
    stop_entry, limit_entry = risk.compute_entry_prices(
        bar_high,
        bar_low,
        atr15_ticks,
        direction,
    )
    initial_stop = risk.compute_initial_stop(
        stop_entry,
        direction,
        pivots_1h,
        atr1h,
        atr15,
    )
    r_points = abs(stop_entry - initial_stop)
    if r_points == 0:
        return None, "zero_risk"

    unit_risk = risk.compute_unit_risk(equity, regime.vol_state)
    effective_risk = risk.compute_effective_risk(
        unit_risk,
        class_mult,
        session_mult * hourly_mult,
    )
    if is_pyramid:
        effective_risk = risk.compute_addon_risk(effective_risk)
    if fixed_qty is None:
        qty = risk.compute_qty(effective_risk * entry_size_mult, r_points)
    else:
        qty = max(1, int(fixed_qty * hourly_mult * entry_size_mult))
    for multiplier in post_size_multipliers:
        if multiplier < 1.0:
            qty = max(1, int(qty * multiplier))
    if qty < 1:
        return None, "zero_quantity"

    resolved_signal_id = signal_id or (
        f"{selection.entry_type.value}_{direction.name}_{bar_idx}"
    )
    resolved_bar_id = bar_id or f"{symbol}:15m:{now.isoformat()}"
    client_order_id = f"{C.STRATEGY_ID}:{resolved_signal_id}"
    return VdubEntryProposal(
        client_order_id=client_order_id,
        symbol=symbol,
        direction=direction,
        qty=qty,
        stop_entry=stop_entry,
        limit_entry=limit_entry,
        initial_stop=initial_stop,
        entry_type=selection.entry_type,
        is_flip=is_flip,
        is_pyramid=is_pyramid,
        class_mult=class_mult,
        vwap_used=selection.vwap_used,
        session=session,
        sub_window=sub_window,
        submitted_bar_idx=bar_idx,
        signal_id=resolved_signal_id,
        bar_id=resolved_bar_id,
        exchange_timestamp=now,
        unit_risk=unit_risk,
        r_points=r_points,
        risk_dollars=r_points * point_value * qty,
    ), ""


def evaluate_proposal_gates(
    proposal: VdubEntryProposal,
    *,
    counters: DayCounters,
    open_risk: float,
    viability_enabled: bool = True,
    risk_gates_enabled: bool = True,
) -> tuple[bool, str]:
    if viability_enabled:
        approved, reason = risk.pass_viability(
            proposal.qty,
            proposal.r_points,
            proposal.sub_window,
        )
        if not approved:
            return False, f"viability_{reason}"
    if risk_gates_enabled:
        approved, reason = risk.pass_risk_gates(
            counters,
            proposal.direction,
            open_risk,
            proposal.risk_dollars,
            proposal.unit_risk,
        )
        if not approved:
            return False, f"risk_gate_{reason}"
    return True, ""


def with_entry_prices(
    proposal: VdubEntryProposal,
    *,
    stop_entry: float,
    limit_entry: float,
) -> VdubEntryProposal:
    """Apply a source-owned micro trigger without changing the original stop basis."""

    return replace(proposal, stop_entry=stop_entry, limit_entry=limit_entry)


def explicit_entry_proposal(
    *,
    symbol: str,
    direction: Direction,
    qty: int,
    stop_entry: float,
    limit_entry: float,
    initial_stop: float,
    entry_type: EntryType,
    is_flip: bool,
    is_pyramid: bool,
    class_mult: float,
    vwap_used: float,
    session: SessionWindow,
    sub_window: SubWindow,
    bar_idx: int,
    now: datetime,
    signal_id: str,
    bar_id: str,
    point_value: float,
) -> VdubEntryProposal:
    r_points = abs(stop_entry - initial_stop)
    return VdubEntryProposal(
        client_order_id=f"{C.STRATEGY_ID}:{signal_id}",
        symbol=symbol,
        direction=direction,
        qty=qty,
        stop_entry=stop_entry,
        limit_entry=limit_entry,
        initial_stop=initial_stop,
        entry_type=entry_type,
        is_flip=is_flip,
        is_pyramid=is_pyramid,
        class_mult=class_mult,
        vwap_used=vwap_used,
        session=session,
        sub_window=sub_window,
        submitted_bar_idx=bar_idx,
        signal_id=signal_id,
        bar_id=bar_id,
        exchange_timestamp=now,
        unit_risk=0.0,
        r_points=r_points,
        risk_dollars=r_points * point_value * qty,
    )


def entry_action(proposal: VdubEntryProposal) -> SubmitEntry:
    return SubmitEntry(
        client_order_id=proposal.client_order_id,
        symbol=proposal.symbol,
        side="BUY" if proposal.direction == Direction.LONG else "SELL",
        qty=proposal.qty,
        order_type="STOP_LIMIT",
        tif="GTC",
        limit_price=proposal.limit_entry,
        stop_price=proposal.stop_entry,
        role="entry",
        session=proposal.session.value,
        risk_context={
            "stop_for_risk": proposal.initial_stop,
            "planned_entry_price": proposal.stop_entry,
            "risk_dollars": proposal.risk_dollars,
            "signal_id": proposal.signal_id,
            "bar_id": proposal.bar_id,
            "exchange_timestamp": proposal.exchange_timestamp,
        },
        metadata={
            "entry_type": proposal.entry_type.value,
            "is_flip": proposal.is_flip,
            "is_pyramid": proposal.is_pyramid,
            "class_mult": proposal.class_mult,
            "vwap_used": proposal.vwap_used,
            "initial_stop": proposal.initial_stop,
            "submitted_bar_idx": proposal.submitted_bar_idx,
        },
    )


def working_entry_from_proposal(
    proposal: VdubEntryProposal,
    *,
    oms_order_id: str,
    qty: int,
    filter_decisions: list[dict] | None = None,
) -> WorkingEntry:
    return WorkingEntry(
        oms_order_id=oms_order_id,
        entry_type=proposal.entry_type,
        direction=proposal.direction,
        stop_entry=proposal.stop_entry,
        limit_entry=proposal.limit_entry,
        qty=qty,
        submitted_bar_idx=proposal.submitted_bar_idx,
        ttl_bars=C.TTL_BARS,
        initial_stop=proposal.initial_stop,
        vwap_used=proposal.vwap_used,
        class_mult=proposal.class_mult,
        session=proposal.session,
        is_flip=proposal.is_flip,
        is_addon=proposal.is_pyramid,
        filter_decisions=filter_decisions,
        signal_id=proposal.signal_id,
        bar_id=proposal.bar_id,
        exchange_timestamp=proposal.exchange_timestamp,
    )
