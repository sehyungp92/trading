from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from backtests.shared.parity.decision_capture import normalize_decision_stream
from backtests.shared.parity.replay_driver import ReplayStep, run_replay
from libs.oms.models.events import OMSEventType
from strategies.core.actions import CancelAction, FlattenPosition, ReplaceProtectiveStop, SubmitEntry, SubmitExit
from strategies.momentum.nqdtc.engine import NQDTCEngine
from strategies.momentum.nqdtc.core.logic import on_bar, on_fill, on_order_update
from strategies.momentum.nqdtc.core.serializers import restore_state, snapshot_state
from strategies.momentum.nqdtc.core.state import (
    NQDTCCoreState,
    NQDTCEntryFillContext,
    NQDTCEntryRequest,
    NQDTCFill,
    NQDTCOrderUpdate,
    NQDTCSimpleRequest,
)
from strategies.momentum.nqdtc.models import Direction, EntrySubtype, ExitTier, PositionState, Session, TPLevel, WorkingOrder

UTC = timezone.utc


def _entry_request() -> NQDTCEntryRequest:
    return NQDTCEntryRequest(
        client_order_id="entry-1",
        symbol="NQ",
        subtype=EntrySubtype.A_RETEST,
        direction=Direction.LONG,
        qty=2,
        stop_for_risk=19950.0,
        order_type="STOP_LIMIT",
        price=19975.0,
        limit_price=19975.0,
        stop_price=19976.0,
        oca_group="OCA-1",
        quality_mult=1.2,
        submitted_bar_idx=15,
        ttl_bars=4,
    )


def test_nqdtc_core_entry_and_exit_roundtrip() -> None:
    state = NQDTCCoreState(symbol="NQ")

    state, actions, events = on_bar(
        state,
        bar_count_5m=15,
        bar_ts=datetime(2026, 4, 25, 11, 0, tzinfo=UTC),
        entry_request=_entry_request(),
    )
    assert isinstance(actions[0], SubmitEntry)
    assert events[-1].code == "ENTRY_REQUESTED"

    state, _, events = on_order_update(
        state,
        NQDTCOrderUpdate(
            oms_order_id="OMS-1",
            status="accepted",
            timestamp=datetime(2026, 4, 25, 11, 1, tzinfo=UTC),
            order_role="entry",
            accepted_entry=_entry_request(),
        ),
    )
    assert state.working_orders[0].oms_order_id == "OMS-1"
    assert events[-1].code == "ENTRY_SUBMITTED"

    fill_context = NQDTCEntryFillContext(
        exit_tier=ExitTier.ALIGNED,
        tp_levels=[TPLevel(r_target=1.0, pct=0.5, qty=1)],
        mm_level=20020.0,
        mm_reached=False,
        box_high_at_entry=19980.0,
        box_low_at_entry=19940.0,
        box_mid_at_entry=19960.0,
        entry_session=Session.RTH,
        tp1_only_cap=False,
    )
    state, actions, events = on_fill(
        state,
        NQDTCFill(
            oms_order_id="OMS-1",
            fill_price=19977.0,
            fill_qty=2,
            fill_time=datetime(2026, 4, 25, 11, 2, tzinfo=UTC),
            entry_context=fill_context,
        ),
    )
    assert state.position.open is True
    assert isinstance(actions[0], SubmitExit)
    assert actions[0].side == "SELL"
    assert events[-1].code == "ENTRY_FILLED"

    state, _, _ = on_order_update(
        state,
        NQDTCOrderUpdate(
            oms_order_id="STOP-1",
            status="accepted",
            timestamp=datetime(2026, 4, 25, 11, 3, tzinfo=UTC),
            order_role="stop",
        ),
    )
    assert state.position.stop_oms_order_id == "STOP-1"

    snapshot = snapshot_state(state)
    restored = restore_state(snapshot)
    assert restored.position.open is True
    assert restored.position.entry_price == 19977.0
    assert restored.working_orders == []

    state, _, events = on_fill(
        state,
        NQDTCFill(
            oms_order_id="STOP-1",
            fill_price=19950.0,
            fill_qty=2,
            fill_time=datetime(2026, 4, 25, 11, 4, tzinfo=UTC),
            exit_type="stop",
        ),
    )
    assert state.position.open is False
    assert events[-1].code == "EXIT_FILLED"


def test_nqdtc_core_stop_cancel_and_flatten_transitions() -> None:
    state = NQDTCCoreState(
        symbol="NQ",
        position=PositionState(
            open=True,
            symbol="NQ",
            direction=Direction.LONG,
            entry_subtype=EntrySubtype.A_RETEST,
            entry_price=19977.0,
            stop_price=19950.0,
            initial_stop_price=19950.0,
            qty=2,
            qty_open=2,
            stop_oms_order_id="STOP-1",
        ),
        working_orders=[
            WorkingOrder(
                oms_order_id="OMS-2",
                subtype=EntrySubtype.C_STANDARD,
                direction=Direction.LONG,
                price=19990.0,
                qty=1,
                submitted_bar_idx=14,
                ttl_bars=6,
            )
        ],
    )

    state, actions, events = on_bar(
        state,
        bar_ts=datetime(2026, 4, 25, 11, 5, tzinfo=UTC),
        stop_update=NQDTCSimpleRequest(reason="trail", price=19960.0, qty=2),
        cancel_order_ids=["OMS-2"],
        flatten_request=NQDTCSimpleRequest(reason="risk_off"),
    )

    assert isinstance(actions[0], ReplaceProtectiveStop)
    assert isinstance(actions[1], CancelAction)
    assert isinstance(actions[2], FlattenPosition)
    assert [event.code for event in events] == [
        "STOP_REPLACEMENT_REQUESTED",
        "ORDER_CANCEL_REQUESTED",
        "FLATTEN_REQUESTED",
    ]


def test_nqdtc_core_ignores_unmatched_fill_without_exit_context() -> None:
    state = NQDTCCoreState(
        symbol="NQ",
        position=PositionState(
            open=True,
            symbol="NQ",
            direction=Direction.LONG,
            entry_subtype=EntrySubtype.A_RETEST,
            entry_price=19977.0,
            stop_price=19950.0,
            initial_stop_price=19950.0,
            qty=2,
            qty_open=2,
            stop_oms_order_id="STOP-1",
        ),
    )

    next_state, actions, events = on_fill(
        state,
        NQDTCFill(
            oms_order_id="UNRELATED",
            fill_price=19990.0,
            fill_qty=1,
            fill_time=datetime(2026, 4, 25, 11, 6, tzinfo=UTC),
        ),
    )

    assert next_state.position.open is True
    assert next_state.position.stop_oms_order_id == "STOP-1"
    assert actions == []
    assert events == []


def test_nqdtc_replay_driver_produces_normalized_decision_stream() -> None:
    fill_context = NQDTCEntryFillContext(
        exit_tier=ExitTier.ALIGNED,
        tp_levels=[TPLevel(r_target=1.0, pct=0.5, qty=1)],
        mm_level=20020.0,
        mm_reached=False,
        box_high_at_entry=19980.0,
        box_low_at_entry=19940.0,
        box_mid_at_entry=19960.0,
        entry_session=Session.RTH,
        tp1_only_cap=False,
    )
    steps = [
        ReplayStep(
            bar_input={
                "bar_count_5m": 15,
                "bar_ts": datetime(2026, 4, 25, 11, 0, tzinfo=UTC),
                "entry_request": _entry_request(),
            }
        ),
        ReplayStep(
            order_updates=[
                NQDTCOrderUpdate(
                    oms_order_id="OMS-1",
                    status="accepted",
                    timestamp=datetime(2026, 4, 25, 11, 1, tzinfo=UTC),
                    order_role="entry",
                    accepted_entry=_entry_request(),
                )
            ]
        ),
        ReplayStep(
            fills=[
                NQDTCFill(
                    oms_order_id="OMS-1",
                    fill_price=19977.0,
                    fill_qty=2,
                    fill_time=datetime(2026, 4, 25, 11, 2, tzinfo=UTC),
                    entry_context=fill_context,
                )
            ]
        ),
    ]

    result = run_replay(
        NQDTCCoreState(symbol="NQ"),
        steps=steps,
        on_bar=lambda state, payload: on_bar(state, **payload),
        on_order_update=on_order_update,
        on_fill=on_fill,
    )

    codes = [event["code"] for event in normalize_decision_stream(result.events)]
    assert codes == ["ENTRY_REQUESTED", "ENTRY_SUBMITTED", "ENTRY_FILLED"]


class _DummyIB:
    pass


class _DummyOMS:
    pass


@pytest.mark.asyncio
async def test_nqdtc_engine_snapshot_and_hydrate_preserve_wrapper_contract(tmp_path) -> None:
    engine = NQDTCEngine(
        ib_session=_DummyIB(),
        oms_service=_DummyOMS(),
        instruments={},
        state_dir=tmp_path,
        instrumentation=None,
    )
    engine._position = PositionState(
        open=True,
        symbol="NQ",
        direction=Direction.LONG,
        entry_subtype=EntrySubtype.A_RETEST,
        entry_price=19977.0,
        stop_price=19950.0,
        initial_stop_price=19950.0,
        qty=2,
        qty_open=2,
        stop_oms_order_id="STOP-1",
        tp_levels=[TPLevel(r_target=1.0, pct=0.5, qty=1)],
    )
    engine._working_orders = [
        WorkingOrder(
            oms_order_id="OMS-2",
            subtype=EntrySubtype.C_STANDARD,
            direction=Direction.LONG,
            price=19990.0,
            qty=1,
            submitted_bar_idx=14,
            ttl_bars=6,
            oca_group="OCA-1",
        )
    ]
    engine._bar_count_5m = 15
    engine._last_decision_code = "PROTECTIVE_STOP_SUBMITTED"
    engine._last_decision_details = {"stop_oms_order_id": "STOP-1"}
    engine._last_bar_ts = datetime(2026, 4, 25, 11, 3, tzinfo=UTC)

    snapshot = engine.snapshot_state()

    restored = NQDTCEngine(
        ib_session=_DummyIB(),
        oms_service=_DummyOMS(),
        instruments={},
        state_dir=tmp_path,
        instrumentation=None,
    )
    await restored.hydrate(snapshot)

    assert restored.health_status()["last_decision_code"] == "PROTECTIVE_STOP_SUBMITTED"
    assert restored._position.open is True
    assert restored._position.stop_oms_order_id == "STOP-1"
    assert restored._working_orders[0].oms_order_id == "OMS-2"
    assert restored._bar_count_5m == 15
    assert restored._symbol == "NQ"


@pytest.mark.asyncio
async def test_nqdtc_engine_entry_fill_routes_through_shared_core(tmp_path) -> None:
    engine = NQDTCEngine(
        ib_session=_DummyIB(),
        oms_service=_DummyOMS(),
        instruments={},
        state_dir=tmp_path,
        instrumentation=None,
    )
    engine._working_orders = [
        WorkingOrder(
            oms_order_id="OMS-1",
            subtype=EntrySubtype.A_RETEST,
            direction=Direction.LONG,
            price=19975.0,
            qty=2,
            submitted_bar_idx=15,
            ttl_bars=4,
            oca_group="OCA-1",
            is_limit=True,
            quality_mult=1.2,
            stop_for_risk=19950.0,
            expected_fill_price=19975.0,
        ),
        WorkingOrder(
            oms_order_id="OMS-2",
            subtype=EntrySubtype.A_LATCH,
            direction=Direction.LONG,
            price=19976.0,
            qty=2,
            submitted_bar_idx=15,
            ttl_bars=4,
            oca_group="OCA-1",
            quality_mult=1.2,
            stop_for_risk=19950.0,
            expected_fill_price=19976.0,
        ),
    ]
    engine._bar_count_5m = 15
    engine._bars_daily = {"ema50": [], "atr14": []}

    await engine._on_fill(
        SimpleNamespace(
            event_type=OMSEventType.FILL,
            oms_order_id="OMS-1",
            payload={"price": 19977.0, "qty": 2},
            timestamp=datetime(2026, 4, 25, 11, 2, tzinfo=UTC),
        )
    )

    assert engine._position.open is True
    assert engine._position.entry_price == 19977.0
    assert engine._position.entry_subtype == EntrySubtype.A_RETEST
    assert [order.oms_order_id for order in engine._working_orders] == []
    assert engine.health_status()["last_decision_code"] == "ENTRY_FILLED"


@pytest.mark.asyncio
async def test_nqdtc_engine_rejected_filled_entry_clears_working_orders(tmp_path) -> None:
    engine = NQDTCEngine(
        ib_session=_DummyIB(),
        oms_service=_DummyOMS(),
        instruments={},
        state_dir=tmp_path,
        instrumentation=None,
    )
    engine._working_orders = [
        WorkingOrder(
            oms_order_id="OMS-1",
            subtype=EntrySubtype.A_RETEST,
            direction=Direction.LONG,
            price=19975.0,
            qty=2,
            submitted_bar_idx=15,
            ttl_bars=4,
            oca_group="OCA-1",
            is_limit=True,
            quality_mult=1.2,
            stop_for_risk=19976.5,
            expected_fill_price=19975.0,
        ),
        WorkingOrder(
            oms_order_id="OMS-2",
            subtype=EntrySubtype.A_LATCH,
            direction=Direction.LONG,
            price=19976.0,
            qty=2,
            submitted_bar_idx=15,
            ttl_bars=4,
            oca_group="OCA-1",
            quality_mult=1.2,
            stop_for_risk=19976.5,
            expected_fill_price=19976.0,
        ),
    ]
    engine._bar_count_5m = 15
    engine._bars_daily = {"ema50": [], "atr14": []}

    await engine._on_fill(
        SimpleNamespace(
            event_type=OMSEventType.FILL,
            oms_order_id="OMS-1",
            payload={"price": 19977.0, "qty": 2},
            timestamp=datetime(2026, 4, 25, 11, 2, tzinfo=UTC),
        )
    )

    assert engine._position.open is False
    assert engine._working_orders == []
    assert engine.health_status()["last_decision_code"] == "ENTRY_FILL_REJECTED"
    assert engine.health_status()["last_decision_details"]["reason"] == "MIN_STOP_DISTANCE"
