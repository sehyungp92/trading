from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from strategies.swing.akc_helix.engine import HelixEngine
from strategies.swing.akc_helix.models import Direction as HelixDirection
from strategies.swing.akc_helix.models import SetupClass, SetupInstance as HelixSetup, SetupState as HelixSetupState
from strategies.swing.breakout.engine import BreakoutEngine
from strategies.swing.breakout.models import (
    Direction as BreakoutDirection,
    EntryType,
    ExitTier,
    PositionState,
    SetupInstance as BreakoutSetup,
    SetupState as BreakoutSetupState,
)

UTC = timezone.utc


def _breakout_setup() -> BreakoutSetup:
    return BreakoutSetup(
        setup_id="BK-1",
        symbol="SPY",
        direction=BreakoutDirection.LONG,
        entry_type=EntryType.A_AVWAP_RETEST,
        state=BreakoutSetupState.NEW,
        created_ts=datetime(2026, 4, 27, 13, 0, tzinfo=UTC),
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


def _helix_setup() -> HelixSetup:
    return HelixSetup(
        setup_id="HELIX-1",
        symbol="QQQ",
        setup_class=SetupClass.CLASS_A,
        direction=HelixDirection.LONG,
        origin_tf="4H",
        state=HelixSetupState.NEW,
        created_ts=datetime(2026, 4, 27, 13, 0, tzinfo=UTC),
        bos_level=505.5,
        stop0=499.0,
        current_stop=499.0,
        qty_planned=3,
        oca_group="HELIX-OCA",
    )


def test_breakout_live_wrapper_routes_entry_and_exit_requests_through_core() -> None:
    engine = BreakoutEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments={},
        config={},
    )
    setup = _breakout_setup()

    routed_setup = engine._route_core_entry_request(
        bar_ts=datetime(2026, 4, 27, 13, 0, tzinfo=UTC),
        setup=setup,
        client_order_id="OMS-BK-ENTRY",
        order_type="LIMIT",
    )

    assert routed_setup.state is BreakoutSetupState.ARMED
    assert engine.active_setups[setup.setup_id].state is BreakoutSetupState.ARMED
    assert engine.health_status()["last_decision_code"] == "ENTRY_REQUESTED"

    routed_setup.state = BreakoutSetupState.ACTIVE
    routed_setup.qty_open = 5
    routed_setup.fill_price = 510.75
    routed_setup.fill_qty = 5
    routed_setup.avg_entry = 510.75
    routed_setup.stop_order_id = "STOP-1"
    engine.positions[setup.symbol] = PositionState(
        symbol=setup.symbol,
        direction=setup.direction,
        qty=5,
        avg_cost=510.75,
        current_stop=504.0,
    )

    engine._route_core_stop_update(
        bar_ts=datetime(2026, 4, 27, 14, 0, tzinfo=UTC),
        setup_id=setup.setup_id,
        symbol=setup.symbol,
        stop_price=505.25,
        qty=5,
        reason="trail",
    )
    assert engine.active_setups[setup.setup_id].current_stop == 505.25
    assert engine.health_status()["last_decision_code"] == "STOP_REPLACEMENT_REQUESTED"

    engine._route_core_partial_exit_request(
        bar_ts=datetime(2026, 4, 27, 14, 5, tzinfo=UTC),
        setup_id=setup.setup_id,
        symbol=setup.symbol,
        client_order_id="OMS-BK-PARTIAL",
        qty=2,
        reason="tp1",
    )
    assert engine.health_status()["last_decision_code"] == "PARTIAL_EXIT_REQUESTED"

    engine._route_core_flatten_request(
        bar_ts=datetime(2026, 4, 27, 14, 10, tzinfo=UTC),
        setup_id=setup.setup_id,
        symbol=setup.symbol,
        reason="time_exit",
    )
    assert engine.health_status()["last_decision_code"] == "FLATTEN_REQUESTED"


def test_akc_helix_live_wrapper_routes_entry_and_exit_requests_through_core() -> None:
    engine = HelixEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments={},
        config={},
    )
    setup = _helix_setup()

    routed_setup = engine._route_core_entry_request(
        bar_ts=datetime(2026, 4, 27, 13, 0, tzinfo=UTC),
        setup=setup,
        client_order_id="OMS-HELIX-ENTRY",
        order_type="STOP_LIMIT",
        limit_price=505.75,
    )

    assert routed_setup.state is HelixSetupState.ARMED
    assert engine.pending_setups[setup.setup_id].state is HelixSetupState.ARMED
    assert engine.health_status()["last_decision_code"] == "ENTRY_REQUESTED"

    routed_setup.state = HelixSetupState.ACTIVE
    routed_setup.fill_price = 505.75
    routed_setup.fill_qty = 3
    routed_setup.qty_open = 3
    routed_setup.stop_order_id = "STOP-1"
    engine.pending_setups.pop(setup.setup_id, None)
    engine.active_setups[setup.setup_id] = routed_setup

    engine._route_core_stop_update(
        bar_ts=datetime(2026, 4, 27, 14, 0, tzinfo=UTC),
        setup_id=setup.setup_id,
        symbol=setup.symbol,
        stop_price=500.5,
        qty=3,
        reason="trailing",
    )
    assert engine.active_setups[setup.setup_id].current_stop == 500.5
    assert engine.health_status()["last_decision_code"] == "STOP_REPLACEMENT_REQUESTED"

    engine._route_core_partial_exit_request(
        bar_ts=datetime(2026, 4, 27, 14, 5, tzinfo=UTC),
        setup_id=setup.setup_id,
        symbol=setup.symbol,
        client_order_id="OMS-HELIX-PARTIAL",
        qty=1,
        reason="partial",
    )
    assert engine.health_status()["last_decision_code"] == "PARTIAL_EXIT_REQUESTED"

    engine._route_core_flatten_request(
        bar_ts=datetime(2026, 4, 27, 14, 10, tzinfo=UTC),
        setup_id=setup.setup_id,
        symbol=setup.symbol,
        reason="bias_flip",
    )
    assert engine.active_setups[setup.setup_id].state is HelixSetupState.CLOSING
    assert engine.health_status()["last_decision_code"] == "FLATTEN_REQUESTED"
