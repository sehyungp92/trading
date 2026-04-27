from __future__ import annotations

from datetime import datetime, timezone

from strategies.core.actions import FlattenPosition, ReplaceProtectiveStop, SubmitEntry, SubmitProtectiveStop
from strategies.swing.atrss.core import logic as atrss_logic
from strategies.swing.atrss.core.state import (
    ATRSSCoreState,
    ATRSSEntryRequest,
    ATRSSFill,
    ATRSSFlattenRequest,
    ATRSSPartialExitRequest,
)
from strategies.swing.atrss.models import Candidate, CandidateType, Direction, HourlyState, PositionBook, PositionLeg

UTC = timezone.utc


def test_atrss_on_bar_entry_request_emits_submit_entry() -> None:
    candidate = Candidate(
        symbol="QQQ",
        type=CandidateType.PULLBACK,
        direction=Direction.LONG,
        trigger_price=510.25,
        initial_stop=503.5,
        qty=3,
        signal_bar=HourlyState(time=datetime(2026, 4, 26, 13, 0, tzinfo=UTC)),
    )
    state, actions, events = atrss_logic.on_bar(
        ATRSSCoreState(),
        bar_ts=datetime(2026, 4, 26, 13, 0, tzinfo=UTC),
        entry_request=ATRSSEntryRequest(
            client_order_id="ENTRY-1",
            symbol="QQQ",
            candidate=candidate,
            limit_price=510.75,
        ),
    )

    assert len(actions) == 1
    assert isinstance(actions[0], SubmitEntry)
    assert actions[0].qty == 3
    assert actions[0].stop_price == 510.25
    assert events[0].code == "ENTRY_REQUESTED"
    assert state.last_decision_code == "ENTRY_REQUESTED"


def test_atrss_on_fill_entry_creates_position_and_protective_stop() -> None:
    state = ATRSSCoreState(
        pending_orders={
            "ENTRY-1": {
                "symbol": "QQQ",
                "type": CandidateType.PULLBACK,
                "direction": Direction.LONG,
                "trigger_price": 510.25,
                "initial_stop": 503.5,
                "qty": 3,
            }
        }
    )

    next_state, actions, events = atrss_logic.on_fill(
        state,
        ATRSSFill(
            oms_order_id="ENTRY-1",
            fill_price=510.5,
            fill_qty=3,
            fill_time=datetime(2026, 4, 26, 13, 5, tzinfo=UTC),
        ),
    )

    position = next_state.positions["QQQ"]
    assert position.direction is Direction.LONG
    assert position.base_leg is not None
    assert position.base_leg.qty == 3
    assert position.stop_pending is True
    assert len(actions) == 1
    assert isinstance(actions[0], SubmitProtectiveStop)
    assert actions[0].stop_price == 503.5
    assert events[0].code == "ENTRY_FILLED"


def test_atrss_on_fill_partial_close_resizes_stop() -> None:
    state = ATRSSCoreState(
        positions={
            "QQQ": PositionBook(
                symbol="QQQ",
                direction=Direction.LONG,
                legs=[
                    PositionLeg(
                        qty=3,
                        entry_price=510.5,
                        initial_stop=503.5,
                        fill_time=datetime(2026, 4, 26, 13, 5, tzinfo=UTC),
                    )
                ],
                current_stop=507.0,
                stop_oms_order_id="STOP-1",
            )
        },
        pending_orders={
            "PARTIAL-1": {
                "symbol": "QQQ",
                "type": "PARTIAL_CLOSE",
                "direction": Direction.LONG,
                "partial_qty": 1,
                "reason": "TP1",
            }
        },
    )

    next_state, actions, events = atrss_logic.on_fill(
        state,
        ATRSSFill(
            oms_order_id="PARTIAL-1",
            fill_price=514.0,
            fill_qty=1,
            fill_time=datetime(2026, 4, 26, 14, 0, tzinfo=UTC),
        ),
    )

    assert next_state.positions["QQQ"].base_leg is not None
    assert next_state.positions["QQQ"].base_leg.qty == 2
    assert len(actions) == 1
    assert isinstance(actions[0], ReplaceProtectiveStop)
    assert actions[0].qty == 2
    assert events[0].code == "PARTIAL_EXIT_FILLED"


def test_atrss_on_fill_stop_grants_voucher_and_removes_position() -> None:
    state = ATRSSCoreState(
        positions={
            "QQQ": PositionBook(
                symbol="QQQ",
                direction=Direction.LONG,
                legs=[
                    PositionLeg(
                        qty=2,
                        entry_price=510.5,
                        initial_stop=503.5,
                        fill_time=datetime(2026, 4, 26, 13, 5, tzinfo=UTC),
                    )
                ],
                current_stop=507.0,
                stop_oms_order_id="STOP-1",
                mfe=1.25,
            )
        }
    )

    next_state, actions, events = atrss_logic.on_fill(
        state,
        ATRSSFill(
            oms_order_id="STOP-1",
            fill_price=507.0,
            fill_qty=2,
            fill_time=datetime(2026, 4, 26, 14, 30, tzinfo=UTC),
            exit_type="STOP",
        ),
    )

    assert "QQQ" not in next_state.positions
    assert next_state.reentry_states["QQQ"].voucher_long is True
    assert actions == []
    assert events[0].code == "STOP_FILLED"


def test_atrss_on_bar_flatten_emits_flatten_action() -> None:
    state = ATRSSCoreState(
        positions={
            "QQQ": PositionBook(
                symbol="QQQ",
                direction=Direction.SHORT,
                legs=[PositionLeg(qty=2, entry_price=510.5, initial_stop=517.5)],
            )
        }
    )

    _state, actions, events = atrss_logic.on_bar(
        state,
        bar_ts=datetime(2026, 4, 26, 15, 0, tzinfo=UTC),
        flatten_request=ATRSSFlattenRequest(symbol="QQQ", reason="FLATTEN_TIME_DECAY"),
    )

    assert len(actions) == 1
    assert isinstance(actions[0], FlattenPosition)
    assert actions[0].side == "BUY"
    assert events[0].code == "FLATTEN_REQUESTED"
