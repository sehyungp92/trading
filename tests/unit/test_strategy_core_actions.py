from __future__ import annotations

from datetime import datetime, timezone

from backtests.shared.parity.decision_capture import normalize_decision_stream
from backtests.shared.parity.execution_adapters import neutral_action_to_sim_order
from libs.oms.models.instrument import Instrument
from strategies.core.actions import (
    FlattenPosition,
    ReplaceProtectiveStop,
    SubmitAddOnEntry,
    SubmitEntry,
    SubmitMarketExit,
    SubmitPartialExit,
    SubmitProfitTarget,
    SubmitProtectiveStop,
)
from strategies.core.events import DecisionEvent
from strategies.core.oms_adapter import neutral_action_to_oms_order

UTC = timezone.utc


def test_neutral_action_to_sim_order_maps_entry_and_stop_actions() -> None:
    entry = SubmitEntry(
        client_order_id="entry-1",
        symbol="MNQ",
        side="SELL",
        qty=2,
        order_type="STOP_LIMIT",
        price=18990.0,
        limit_price=18990.0,
        stop_price=18992.0,
        metadata={"role": "entry"},
    )
    stop = ReplaceProtectiveStop(
        symbol="MNQ",
        target_order_id="stop-1",
        side="BUY",
        stop_price=19010.0,
        qty=2,
        reason="trail",
    )

    entry_order = neutral_action_to_sim_order(entry, tick_size=0.25)
    stop_order = neutral_action_to_sim_order(stop, tick_size=0.25)

    assert entry_order.order_id == "entry-1"
    assert entry_order.stop_price == 18992.0
    assert entry_order.limit_price == 18990.0
    assert stop_order.order_id == "stop-1"
    assert stop_order.stop_price == 19010.0


def test_neutral_action_to_sim_order_maps_profit_target_partial_and_flatten_actions() -> None:
    target = SubmitProfitTarget(
        client_order_id="tp-1",
        symbol="NQ",
        side="SELL",
        qty=1,
        limit_price=20025.0,
        oca_group="OCA-1",
    )
    partial = SubmitPartialExit(
        client_order_id="partial-1",
        symbol="NQ",
        side="SELL",
        qty=1,
        order_type="LIMIT",
        limit_price=20010.0,
        role="partial_exit",
        oca_group="OCA-1",
    )
    flatten = FlattenPosition(
        symbol="NQ",
        reason="risk_off",
        side="SELL",
        qty=2,
        parent_order_id="flatten-1",
    )

    target_order = neutral_action_to_sim_order(target, tick_size=0.25)
    partial_order = neutral_action_to_sim_order(partial, tick_size=0.25)
    flatten_order = neutral_action_to_sim_order(flatten, tick_size=0.25)

    assert target_order.order_type.value == "LIMIT"
    assert target_order.limit_price == 20025.0
    assert target_order.oca_group == "OCA-1"
    assert partial_order.order_type.value == "LIMIT"
    assert partial_order.limit_price == 20010.0
    assert flatten_order.order_type.value == "MARKET"
    assert flatten_order.order_id == "flatten-1"


def test_neutral_action_to_sim_order_maps_add_on_entry_and_oca_metadata() -> None:
    add_on = SubmitAddOnEntry(
        client_order_id="add-1",
        symbol="NQ",
        side="BUY",
        qty=1,
        order_type="LIMIT",
        limit_price=20005.0,
        tif="DAY",
        parent_order_id="entry-1",
        oca_group="ADD-LEG-1",
        role="add_on_entry",
        metadata={"ttl_seconds": 30},
    )
    stop = SubmitProtectiveStop(
        client_order_id="stop-1",
        symbol="NQ",
        side="SELL",
        qty=1,
        stop_price=19975.0,
        parent_order_id="entry-1",
        oca_group="ADD-LEG-1",
        role="protective_stop",
    )

    add_on_order = neutral_action_to_sim_order(add_on, tick_size=0.25)
    stop_order = neutral_action_to_sim_order(stop, tick_size=0.25)

    assert add_on_order.order_type.value == "LIMIT"
    assert add_on_order.limit_price == 20005.0
    assert add_on_order.oca_group == "ADD-LEG-1"
    assert add_on_order.tag == "add_on_entry"
    assert stop_order.order_type.value == "STOP"
    assert stop_order.stop_price == 19975.0
    assert stop_order.oca_group == "ADD-LEG-1"
    assert stop_order.tag == "protective_stop"


def test_neutral_action_to_oms_order_preserves_roles_and_risk_context() -> None:
    instrument = Instrument(
        symbol="MNQ",
        root="MNQ",
        venue="CME",
        tick_size=0.25,
        tick_value=0.5,
        multiplier=2.0,
    )
    entry = SubmitEntry(
        client_order_id="entry-1",
        symbol="MNQ",
        side="BUY",
        qty=2,
        order_type="STOP_LIMIT",
        limit_price=18990.0,
        stop_price=18992.0,
        metadata={"ttl_bars": 6},
        risk_context={
            "stop_for_risk": 18970.0,
            "planned_entry_price": 18990.0,
            "risk_dollars": 80.0,
            "risk_budget_tag": "trend",
        },
    )
    stop = SubmitProtectiveStop(
        client_order_id="stop-1",
        symbol="MNQ",
        side="SELL",
        qty=2,
        stop_price=18970.0,
        parent_order_id="entry-1",
    )
    market_exit = SubmitMarketExit(
        client_order_id="exit-1",
        symbol="MNQ",
        side="SELL",
        qty=2,
    )

    entry_order = neutral_action_to_oms_order(
        entry,
        strategy_id="TEST",
        instrument=instrument,
        account_id="DU123",
    )
    stop_order = neutral_action_to_oms_order(
        stop,
        strategy_id="TEST",
        instrument=instrument,
    )
    exit_order = neutral_action_to_oms_order(
        market_exit,
        strategy_id="TEST",
        instrument=instrument,
    )

    assert entry_order.role.value == "ENTRY"
    assert entry_order.entry_policy is not None
    assert entry_order.entry_policy.ttl_bars == 6
    assert entry_order.risk_context is not None
    assert entry_order.risk_context.stop_for_risk == 18970.0
    assert stop_order.role.value == "STOP"
    assert stop_order.stop_price == 18970.0
    assert exit_order.role.value == "EXIT"
    assert exit_order.order_type.value == "MARKET"


def test_neutral_action_to_oms_order_maps_add_on_entry_with_entry_policy_and_oca_group() -> None:
    instrument = Instrument(
        symbol="NQ",
        root="NQ",
        venue="CME",
        tick_size=0.25,
        tick_value=5.0,
        multiplier=20.0,
    )
    add_on = SubmitAddOnEntry(
        client_order_id="add-1",
        symbol="NQ",
        side="BUY",
        qty=1,
        order_type="LIMIT",
        tif="IOC",
        limit_price=20005.0,
        parent_order_id="entry-1",
        oca_group="ADD-LEG-1",
        metadata={"ttl_seconds": 45},
        risk_context={
            "stop_for_risk": 19975.0,
            "planned_entry_price": 20005.0,
            "risk_dollars": 150.0,
            "portfolio_size_mult": 0.5,
        },
    )

    order = neutral_action_to_oms_order(
        add_on,
        strategy_id="TEST",
        instrument=instrument,
        account_id="DU123",
    )

    assert order.role.value == "ENTRY"
    assert order.oca_group == "ADD-LEG-1"
    assert order.entry_policy is not None
    assert order.entry_policy.ttl_seconds == 45
    assert order.risk_context is not None
    assert order.risk_context.portfolio_size_mult == 0.5
    assert order.limit_price == 20005.0


def test_normalize_decision_stream_preserves_core_fields() -> None:
    events = [
        DecisionEvent(
            code="ENTRY_FILLED",
            ts=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
            symbol="MNQ",
            timeframe="5m",
            details={"qty": 1, "nested": {"b": 2, "a": 1}},
        )
    ]

    normalized = normalize_decision_stream(events)

    assert normalized == [
        {
            "code": "ENTRY_FILLED",
            "ts": "2026-04-25T10:00:00+00:00",
            "symbol": "MNQ",
            "timeframe": "5m",
            "details": {"nested": {"a": 1, "b": 2}, "qty": 1},
        }
    ]
