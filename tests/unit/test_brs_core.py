from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from backtests.shared.parity.decision_capture import normalize_decision_stream
from backtests.shared.parity.replay_driver import ReplayStep, run_replay
from strategies.core.actions import FlattenPosition, ReplaceProtectiveStop, SubmitPartialExit
from strategies.swing.brs.core.logic import on_bar, on_fill, on_order_update, translate_position_actions
from strategies.swing.brs.core.serializers import restore_state, snapshot_state
from strategies.swing.brs.core.state import (
    BRSAddOnRequest,
    BRSCoreState,
    BRSEntryFillContext,
    BRSEntryRequest,
    BRSFill,
    BRSFlattenRequest,
    BRSOrderUpdate,
    BRSScaleOutRequest,
    BRSStopUpdateRequest,
)
from strategies.swing.brs.engine import BRSLiveEngine
from strategies.swing.brs.models import (
    BRSRegime,
    DailyContext,
    Direction,
    EntrySignal,
    EntryType,
    ExitReason,
    S2ArmState,
)
from strategies.swing.brs.positions import ActionResult, BRSPositionState, PendingOrder, PositionAction

UTC = timezone.utc


def _entry_signal() -> EntrySignal:
    return EntrySignal(
        entry_type=EntryType.S2_BREAKDOWN,
        direction=Direction.SHORT,
        signal_price=440.0,
        signal_high=442.0,
        signal_low=439.5,
        stop_price=446.0,
        risk_per_unit=6.0,
        regime_at_entry=BRSRegime.BEAR_TREND,
        quality_score=0.81,
        vol_factor=1.1,
    )


def _position() -> BRSPositionState:
    return BRSPositionState.from_signal(
        symbol="QQQ",
        pos_id="BRS-1",
        signal=_entry_signal(),
        qty=3,
        entry_ts=datetime(2026, 4, 25, 14, 0, tzinfo=UTC),
    )


def _entry_request() -> BRSEntryRequest:
    return BRSEntryRequest(
        client_order_id="entry-1",
        symbol="QQQ",
        signal=_entry_signal(),
        qty=3,
        pos_id="BRS-1",
    )


def _pending_order(*, order_id: str = "OMS-1", role: str = "entry", qty: int = 3) -> PendingOrder:
    return PendingOrder(
        oms_order_id=order_id,
        symbol="QQQ",
        role=role,
        qty=qty,
        signal=_entry_signal(),
        pos_id="BRS-1",
        submitted_at=datetime(2026, 4, 25, 14, 1, tzinfo=UTC),
    )


def test_brs_core_snapshot_roundtrip_preserves_positions_and_pending_orders() -> None:
    position = _position()
    position.stop_oms_id = "STOP-1"
    state = BRSCoreState(
        daily_ctx={"QQQ": DailyContext(regime=BRSRegime.BEAR_TREND)},
        position={"QQQ": position, "GLD": None},
        pending_orders={
            "OMS-1": PendingOrder(
                oms_order_id="OMS-1",
                symbol="QQQ",
                role="entry",
                qty=3,
                signal=_entry_signal(),
                pos_id="BRS-1",
                submitted_at=datetime(2026, 4, 25, 14, 1, tzinfo=UTC),
            )
        },
        filled_order_ids={"FILL-1"},
        closing={"QQQ": "TIME_DECAY"},
        last_decision_code="ENTRY_FILLED",
        last_decision_details={"symbol": "QQQ"},
        last_bar_ts=datetime(2026, 4, 25, 15, 0, tzinfo=UTC),
    )

    restored = restore_state(snapshot_state(state))

    assert restored.position["QQQ"] is not None
    assert restored.position["QQQ"].qty == 3
    assert restored.position["QQQ"].direction == Direction.SHORT
    assert restored.pending_orders["OMS-1"].signal is not None
    assert restored.pending_orders["OMS-1"].signal.entry_type == EntryType.S2_BREAKDOWN
    assert restored.filled_order_ids == {"FILL-1"}
    assert restored.last_decision_code == "ENTRY_FILLED"


def test_brs_core_entry_lifecycle_partial_and_flatten_transitions() -> None:
    state = BRSCoreState(
        s2_arm={"QQQ": S2ArmState()},
        position={"QQQ": None},
        short_bias_no_trade={"QQQ": 3},
    )

    state, actions, events = on_bar(
        state,
        bar_ts=datetime(2026, 4, 25, 14, 0, tzinfo=UTC),
        entry_request=_entry_request(),
        add_on_request=BRSAddOnRequest(
            client_order_id="add-1",
            symbol="QQQ",
            signal=_entry_signal(),
            qty=1,
            pos_id="BRS-1",
        ),
    )

    assert len(actions) == 2
    assert events[0].code == "ENTRY_REQUESTED"
    assert events[1].code == "ADD_ON_REQUESTED"

    state, _, events = on_order_update(
        state,
        BRSOrderUpdate(
            oms_order_id="OMS-1",
            status="accepted",
            timestamp=datetime(2026, 4, 25, 14, 1, tzinfo=UTC),
            accepted_order=_pending_order(order_id="OMS-1"),
        ),
    )

    assert state.pending_orders["OMS-1"].role == "entry"
    assert events[-1].code == "ENTRY_SUBMITTED"

    state, actions, events = on_fill(
        state,
        BRSFill(
            oms_order_id="OMS-1",
            fill_price=439.75,
            fill_qty=3,
            commission=1.2,
            fill_time=datetime(2026, 4, 25, 14, 2, tzinfo=UTC),
            entry_context=BRSEntryFillContext(
                cooldown_until=datetime(2026, 4, 25, 20, 0, tzinfo=UTC),
                reset_arm_state=EntryType.S2_BREAKDOWN,
                tranche_b_qty=1,
            ),
        ),
    )

    assert state.position["QQQ"] is not None
    assert state.position["QQQ"].entry_oms_id == "OMS-1"
    assert state.position["QQQ"].tranche_b_open is True
    assert state.short_bias_no_trade["QQQ"] == 0
    assert state.s2_arm["QQQ"] == S2ArmState()
    assert actions[0].order_type == "STOP"
    assert actions[0].stop_price == pytest.approx(_entry_signal().stop_price)
    assert events[-1].code == "ENTRY_FILLED"

    state, _, _ = on_order_update(
        state,
        BRSOrderUpdate(
            oms_order_id="STOP-1",
            status="accepted",
            timestamp=datetime(2026, 4, 25, 14, 3, tzinfo=UTC),
            order_role="stop",
            symbol="QQQ",
        ),
    )
    assert state.position["QQQ"].stop_oms_id == "STOP-1"

    state, actions, events = on_bar(
        state,
        bar_ts=datetime(2026, 4, 25, 14, 4, tzinfo=UTC),
        stop_update=BRSStopUpdateRequest(symbol="QQQ", stop_price=443.5, qty=3, reason="trail"),
        scale_out_request=BRSScaleOutRequest(
            client_order_id="scale-1",
            symbol="QQQ",
            qty=1,
            limit_price=432.0,
            pos_id="BRS-1",
        ),
    )
    assert isinstance(actions[0], ReplaceProtectiveStop)
    assert isinstance(actions[1], SubmitPartialExit)
    assert [event.code for event in events] == [
        "STOP_REPLACEMENT_REQUESTED",
        "PARTIAL_EXIT_REQUESTED",
    ]

    state, _, _ = on_order_update(
        state,
        BRSOrderUpdate(
            oms_order_id="PARTIAL-1",
            status="accepted",
            timestamp=datetime(2026, 4, 25, 14, 5, tzinfo=UTC),
            accepted_order=PendingOrder(
                oms_order_id="PARTIAL-1",
                symbol="QQQ",
                role="scale_out",
                qty=1,
                pos_id="BRS-1",
            ),
        ),
    )
    state, actions, events = on_fill(
        state,
        BRSFill(
            oms_order_id="PARTIAL-1",
            fill_price=432.0,
            fill_qty=1,
            commission=0.3,
            fill_time=datetime(2026, 4, 25, 14, 6, tzinfo=UTC),
        ),
    )
    assert state.position["QQQ"] is not None
    assert state.position["QQQ"].qty == 2
    assert actions == []
    assert events[-1].code == "PARTIAL_EXIT_FILLED"

    state, actions, events = on_bar(
        state,
        bar_ts=datetime(2026, 4, 25, 14, 7, tzinfo=UTC),
        flatten_request=BRSFlattenRequest(symbol="QQQ", reason="risk_off"),
    )
    assert isinstance(actions[0], FlattenPosition)
    assert state.closing["QQQ"] == "risk_off"
    assert events[-1].code == "FLATTEN_REQUESTED"

    state, actions, events = on_fill(
        state,
        BRSFill(
            oms_order_id="UNRELATED-FLAT",
            fill_price=431.0,
            fill_qty=2,
            fill_time=datetime(2026, 4, 25, 14, 8, tzinfo=UTC),
            symbol="QQQ",
            exit_type="risk_off",
        ),
    )
    assert state.position["QQQ"] is None
    assert state.closing == {}
    assert actions == []
    assert events[-1].code == "EXIT_FILLED"


def test_brs_core_order_terminal_and_unmatched_fill_are_safe_noops() -> None:
    state = BRSCoreState(
        position={"QQQ": _position()},
        pending_orders={"OMS-1": _pending_order(order_id="OMS-1")},
    )
    state.position["QQQ"].stop_oms_id = "STOP-1"

    state, _, events = on_order_update(
        state,
        BRSOrderUpdate(
            oms_order_id="OMS-1",
            status="cancelled",
            timestamp=datetime(2026, 4, 25, 14, 3, tzinfo=UTC),
            symbol="QQQ",
        ),
    )
    assert "OMS-1" not in state.pending_orders
    assert events[-1].code == "ORDER_TERMINATED"

    next_state, actions, events = on_fill(
        state,
        BRSFill(
            oms_order_id="UNRELATED",
            fill_price=430.0,
            fill_qty=1,
            fill_time=datetime(2026, 4, 25, 14, 4, tzinfo=UTC),
        ),
    )
    assert next_state.position["QQQ"] is not None
    assert next_state.position["QQQ"].qty == 3
    assert actions == []
    assert events == []


def test_brs_replay_driver_produces_normalized_decision_stream() -> None:
    steps = [
        ReplayStep(
            bar_input={
                "bar_ts": datetime(2026, 4, 25, 14, 0, tzinfo=UTC),
                "entry_request": _entry_request(),
            }
        ),
        ReplayStep(
            order_updates=[
                BRSOrderUpdate(
                    oms_order_id="OMS-1",
                    status="accepted",
                    timestamp=datetime(2026, 4, 25, 14, 1, tzinfo=UTC),
                    accepted_order=_pending_order(order_id="OMS-1"),
                )
            ]
        ),
        ReplayStep(
            fills=[
                BRSFill(
                    oms_order_id="OMS-1",
                    fill_price=439.75,
                    fill_qty=3,
                    fill_time=datetime(2026, 4, 25, 14, 2, tzinfo=UTC),
                    entry_context=BRSEntryFillContext(reset_arm_state=EntryType.S2_BREAKDOWN),
                )
            ]
        ),
    ]

    result = run_replay(
        BRSCoreState(position={"QQQ": None}),
        steps=steps,
        on_bar=lambda state, payload: on_bar(state, **payload),
        on_order_update=on_order_update,
        on_fill=on_fill,
    )

    codes = [event["code"] for event in normalize_decision_stream(result.events)]
    assert codes == ["ENTRY_REQUESTED", "ENTRY_SUBMITTED", "ENTRY_FILLED"]


def test_brs_position_actions_translate_to_neutral_action_seam() -> None:
    position = _position()
    position.stop_oms_id = "STOP-1"

    translated = translate_position_actions(
        "QQQ",
        position,
        [
            ActionResult(action=PositionAction.STOP_UPDATE, new_stop=443.5),
            ActionResult(action=PositionAction.SCALE_OUT, scale_out_qty=1, exit_price=432.0),
            ActionResult(action=PositionAction.EXIT, exit_reason=ExitReason.TIME_DECAY, exit_price=431.0),
        ],
    )

    assert isinstance(translated[0], ReplaceProtectiveStop)
    assert translated[0].side == "BUY"
    assert translated[0].stop_price == 443.5
    assert isinstance(translated[1], SubmitPartialExit)
    assert translated[1].limit_price == 432.0
    assert isinstance(translated[2], FlattenPosition)
    assert translated[2].reason == "TIME_DECAY"
    assert translated[2].qty == 3


@pytest.mark.asyncio
async def test_brs_engine_snapshot_and_hydrate_preserve_runtime_state() -> None:
    instruments = [SimpleNamespace(symbol="QQQ"), SimpleNamespace(symbol="GLD")]
    engine = BRSLiveEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments=instruments,
    )
    position = _position()
    position.stop_oms_id = "STOP-1"
    engine._position["QQQ"] = position
    engine._pending_orders["OMS-1"] = PendingOrder(
        oms_order_id="OMS-1",
        symbol="QQQ",
        role="entry",
        qty=3,
        signal=_entry_signal(),
        pos_id="BRS-1",
        submitted_at=datetime(2026, 4, 25, 14, 1, tzinfo=UTC),
    )
    engine._last_decision_code = "ENTRY_FILLED"
    engine._last_decision_details = {"symbol": "QQQ"}

    restored = BRSLiveEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments=instruments,
    )
    await restored.hydrate(engine.snapshot_state())

    assert restored._position["QQQ"] is not None
    assert restored._position["QQQ"].pos_id == "BRS-1"
    assert restored._pending_orders["OMS-1"].role == "entry"
    assert restored.health_status()["last_decision_code"] == "ENTRY_FILLED"
