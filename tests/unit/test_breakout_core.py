from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from backtests.shared.parity.replay_driver import ReplayStep, run_replay
from libs.oms.models.events import OMSEventType
import pytest

from strategies.swing.breakout.core.logic import build_core_state as build_breakout_runtime_state
from strategies.core.actions import FlattenPosition, ReplaceProtectiveStop, SubmitEntry, SubmitProtectiveStop
from strategies.swing.breakout.core import logic as breakout_logic
from strategies.swing.breakout.core.serializers import restore_state as restore_breakout_state
from strategies.swing.breakout.core.serializers import snapshot_state as snapshot_breakout_state
from strategies.swing.breakout.core.state import (
    BreakoutCoreState,
    BreakoutEntryRequest,
    BreakoutFill,
    BreakoutFlattenRequest,
)
from strategies.swing.breakout.engine import BreakoutEngine
from strategies.swing.breakout.models import Direction, EntryType, ExitTier, PositionState, SetupInstance, SetupState

UTC = timezone.utc


def _setup(*, setup_id: str = "SETUP-1") -> SetupInstance:
    return SetupInstance(
        setup_id=setup_id,
        symbol="SPY",
        direction=Direction.LONG,
        entry_type=EntryType.A_AVWAP_RETEST,
        state=SetupState.NEW,
        created_ts=datetime(2026, 4, 26, 13, 0, tzinfo=UTC),
        campaign_id=7,
        box_version=2,
        entry_price=510.5,
        stop0=504.0,
        current_stop=504.0,
        final_risk_dollars=130.0,
        shares_planned=5,
        oca_group="BKO-1",
        exit_tier=ExitTier.ALIGNED,
        regime_at_entry="BULL_TREND",
    )


def test_breakout_on_bar_entry_request_emits_submit_entry() -> None:
    state, actions, events = breakout_logic.on_bar(
        BreakoutCoreState(),
        bar_ts=datetime(2026, 4, 26, 13, 0, tzinfo=UTC),
        entry_request=BreakoutEntryRequest(
            client_order_id="ENTRY-1",
            setup=_setup(),
            order_type="LIMIT",
        ),
    )

    assert len(actions) == 1
    assert isinstance(actions[0], SubmitEntry)
    assert actions[0].qty == 5
    assert actions[0].limit_price == 510.5
    assert events[0].code == "ENTRY_REQUESTED"
    assert state.active_setups["SETUP-1"].state is SetupState.ARMED


def test_breakout_on_fill_entry_creates_position_and_protective_stop() -> None:
    setup = _setup()
    state = BreakoutCoreState(
        active_setups={setup.setup_id: setup},
        order_to_setup={"ENTRY-1": setup.setup_id},
        order_kind={"ENTRY-1": "primary_entry"},
        order_requested_qty={"ENTRY-1": 5},
    )

    next_state, actions, events = breakout_logic.on_fill(
        state,
        BreakoutFill(
            oms_order_id="ENTRY-1",
            fill_price=510.75,
            fill_qty=5,
            fill_time=datetime(2026, 4, 26, 13, 5, tzinfo=UTC),
        ),
    )

    position = next_state.positions["SPY"]
    assert position.direction is Direction.LONG
    assert position.qty == 5
    assert len(actions) == 1
    assert isinstance(actions[0], SubmitProtectiveStop)
    assert actions[0].stop_price == 504.0
    assert events[0].code == "ENTRY_FILLED"


def test_breakout_partial_fill_resizes_stop() -> None:
    setup = _setup()
    setup.state = SetupState.ACTIVE
    setup.fill_price = 510.75
    setup.fill_qty = 5
    setup.qty_open = 5
    setup.avg_entry = 510.75
    setup.stop_order_id = "STOP-1"

    state = BreakoutCoreState(
        active_setups={setup.setup_id: setup},
        positions={
            setup.symbol: PositionState(
                symbol=setup.symbol,
                direction=setup.direction,
                qty=5,
                avg_cost=510.75,
                current_stop=504.0,
            )
        },
        order_to_setup={"PART-1": setup.setup_id},
        order_kind={"PART-1": "partial_close"},
    )

    next_state, actions, events = breakout_logic.on_fill(
        state,
        BreakoutFill(
            oms_order_id="PART-1",
            fill_price=514.25,
            fill_qty=2,
            fill_time=datetime(2026, 4, 26, 14, 0, tzinfo=UTC),
        ),
    )

    assert next_state.active_setups[setup.setup_id].qty_open == 3
    assert next_state.positions["SPY"].qty == 3
    assert len(actions) == 1
    assert isinstance(actions[0], ReplaceProtectiveStop)
    assert actions[0].qty == 3
    assert events[0].code == "PARTIAL_EXIT_FILLED"


def test_breakout_add_entry_fill_uses_live_order_kind_alias() -> None:
    setup = _setup()
    setup.state = SetupState.ACTIVE
    setup.fill_price = 510.75
    setup.fill_qty = 5
    setup.qty_open = 5
    setup.avg_entry = 510.75
    setup.is_add = True
    setup.stop_order_id = "STOP-1"

    state = BreakoutCoreState(
        active_setups={setup.setup_id: setup},
        positions={
            setup.symbol: PositionState(
                symbol=setup.symbol,
                direction=setup.direction,
                qty=5,
                avg_cost=510.75,
                current_stop=504.0,
            )
        },
        order_to_setup={"ADD-1": setup.setup_id},
        order_kind={"ADD-1": "add_entry"},
        order_requested_qty={"ADD-1": 2},
    )

    next_state, actions, events = breakout_logic.on_fill(
        state,
        BreakoutFill(
            oms_order_id="ADD-1",
            fill_price=511.5,
            fill_qty=2,
            fill_time=datetime(2026, 4, 26, 14, 0, tzinfo=UTC),
        ),
    )

    assert next_state.active_setups[setup.setup_id].qty_open == 7
    assert next_state.positions["SPY"].qty == 7
    assert len(actions) == 1
    assert isinstance(actions[0], ReplaceProtectiveStop)
    assert actions[0].qty == 7
    assert events[0].code == "ADD_FILLED"


def test_breakout_flatten_request_emits_flatten_position() -> None:
    setup = _setup()
    setup.state = SetupState.ACTIVE
    setup.qty_open = 5

    state, actions, events = breakout_logic.on_bar(
        BreakoutCoreState(active_setups={setup.setup_id: setup}),
        bar_ts=datetime(2026, 4, 26, 15, 0, tzinfo=UTC),
        flatten_request=BreakoutFlattenRequest(
            setup_id=setup.setup_id,
            symbol=setup.symbol,
            reason="time_exit",
        ),
    )

    assert len(actions) == 1
    assert isinstance(actions[0], FlattenPosition)
    assert actions[0].qty == 5
    assert events[0].code == "FLATTEN_REQUESTED"
    assert state.last_decision_code == "FLATTEN_REQUESTED"


@pytest.mark.asyncio
async def test_breakout_live_wrapper_entry_fill_matches_replay_core_state(monkeypatch) -> None:
    setup = _setup()
    engine = BreakoutEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments={},
        config={},
    )
    engine.active_setups[setup.setup_id] = setup
    engine._order_to_setup["ENTRY-1"] = setup.setup_id
    engine._order_kind["ENTRY-1"] = "primary_entry"
    engine._order_requested_qty["ENTRY-1"] = 5

    submitted_brackets: list[tuple[str, float, int]] = []

    async def _fake_submit_brackets(active_setup: SetupInstance) -> None:
        submitted_brackets.append((active_setup.symbol, active_setup.current_stop, active_setup.qty_open))

    monkeypatch.setattr(engine, "_submit_bracket_orders", _fake_submit_brackets)
    monkeypatch.setattr(engine, "_record_entry_instrumentation", lambda *_args, **_kwargs: None)

    initial_state = restore_breakout_state(snapshot_breakout_state(build_breakout_runtime_state(engine)))

    event = SimpleNamespace(
        event_type=OMSEventType.FILL,
        payload={"price": 510.75, "qty": 5, "commission": 0.0},
        timestamp=datetime(2026, 4, 26, 13, 5, tzinfo=UTC),
    )
    await engine._on_fill_event(event, setup, "primary_entry", "ENTRY-1", 5)

    wrapper_snapshot = snapshot_breakout_state(build_breakout_runtime_state(engine))
    replay = run_replay(
        initial_state,
        steps=[
            ReplayStep(
                fills=[
                    BreakoutFill(
                        oms_order_id="ENTRY-1",
                        fill_price=510.75,
                        fill_qty=5,
                        symbol=setup.symbol,
                        fill_time=event.timestamp,
                        order_role="primary_entry",
                    )
                ]
            )
        ],
        on_bar=lambda state, payload: breakout_logic.on_bar(state, **payload),
        on_order_update=breakout_logic.on_order_update,
        on_fill=breakout_logic.on_fill,
    )

    assert submitted_brackets == [("SPY", 504.0, 5)]
    assert replay.events[-1].code == engine.health_status()["last_decision_code"] == "ENTRY_FILLED"
    assert snapshot_breakout_state(replay.state) == wrapper_snapshot
