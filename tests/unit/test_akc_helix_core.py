from __future__ import annotations

from datetime import datetime, timezone

from strategies.core.actions import ReplaceProtectiveStop, SubmitEntry, SubmitProtectiveStop
from strategies.swing.akc_helix.core import logic as akc_helix_logic
from strategies.swing.akc_helix.core.state import (
    AKCHelixCoreState,
    AKCHelixEntryRequest,
    AKCHelixFill,
)
from strategies.swing.akc_helix.models import Direction, SetupClass, SetupInstance, SetupState

UTC = timezone.utc


def _setup(*, setup_id: str = "HELIX-1") -> SetupInstance:
    return SetupInstance(
        setup_id=setup_id,
        symbol="QQQ",
        setup_class=SetupClass.CLASS_A,
        direction=Direction.LONG,
        origin_tf="4H",
        state=SetupState.NEW,
        created_ts=datetime(2026, 4, 26, 13, 0, tzinfo=UTC),
        bos_level=505.5,
        stop0=499.0,
        current_stop=499.0,
        qty_planned=3,
        oca_group="HELIX-OCA",
    )


def test_akc_helix_on_bar_entry_request_emits_submit_entry() -> None:
    state, actions, events = akc_helix_logic.on_bar(
        AKCHelixCoreState(),
        bar_ts=datetime(2026, 4, 26, 13, 0, tzinfo=UTC),
        entry_request=AKCHelixEntryRequest(
            client_order_id="ENTRY-1",
            setup=_setup(),
            order_type="STOP_LIMIT",
            limit_price=505.75,
        ),
    )

    assert len(actions) == 1
    assert isinstance(actions[0], SubmitEntry)
    assert actions[0].qty == 3
    assert actions[0].stop_price == 505.5
    assert actions[0].limit_price == 505.75
    assert events[0].code == "ENTRY_REQUESTED"
    assert state.pending_setups["HELIX-1"].state is SetupState.ARMED


def test_akc_helix_on_fill_entry_creates_stop_action() -> None:
    setup = _setup()
    state = AKCHelixCoreState(
        pending_setups={setup.setup_id: setup},
        order_to_setup={"ENTRY-1": setup.setup_id},
    )

    next_state, actions, events = akc_helix_logic.on_fill(
        state,
        AKCHelixFill(
            oms_order_id="ENTRY-1",
            fill_price=505.75,
            fill_qty=3,
            fill_time=datetime(2026, 4, 26, 13, 5, tzinfo=UTC),
            order_role="entry",
        ),
    )

    filled_setup = next_state.active_setups[setup.setup_id]
    assert filled_setup.state is SetupState.ACTIVE
    assert filled_setup.qty_open == 3
    assert len(actions) == 1
    assert isinstance(actions[0], SubmitProtectiveStop)
    assert actions[0].stop_price == 499.0
    assert events[0].code == "ENTRY_FILLED"


def test_akc_helix_partial_fill_resizes_stop() -> None:
    setup = _setup()
    setup.state = SetupState.ACTIVE
    setup.fill_price = 505.75
    setup.fill_qty = 3
    setup.qty_open = 3
    setup.stop_order_id = "STOP-1"

    state = AKCHelixCoreState(
        active_setups={setup.setup_id: setup},
        order_to_setup={"PARTIAL-1": setup.setup_id},
    )

    next_state, actions, events = akc_helix_logic.on_fill(
        state,
        AKCHelixFill(
            oms_order_id="PARTIAL-1",
            fill_price=508.5,
            fill_qty=1,
            fill_time=datetime(2026, 4, 26, 14, 0, tzinfo=UTC),
            order_role="partial",
        ),
    )

    assert next_state.active_setups[setup.setup_id].qty_open == 2
    assert len(actions) == 1
    assert isinstance(actions[0], ReplaceProtectiveStop)
    assert actions[0].qty == 2
    assert events[0].code == "PARTIAL_EXIT_FILLED"
