from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from shared.oms.events.bus import EventBus
from shared.oms.execution.router import ExecutionRouter, OrderPriority
from shared.oms.models.events import OMSEvent, OMSEventType
from shared.oms.models.instrument import Instrument
from shared.oms.models.order import OMSOrder, OrderRole, OrderSide, OrderStatus, OrderType
from shared.oms.persistence.in_memory import InMemoryRepository
from strategy_3.config import SymbolConfig as BreakoutSymbolConfig
from strategy_3.engine import BreakoutEngine
from strategy_3.models import CampaignState, Direction as BreakoutDirection, EntryType, ExitTier, PositionState, SetupInstance, SetupState
from strategy_4.config import SymbolConfig as KeltnerSymbolConfig
from strategy_4.engine import KeltnerEngine
from strategy_4.models import Direction as KeltnerDirection


def _make_instrument(symbol: str = "QQQ") -> Instrument:
    return Instrument(
        symbol=symbol,
        root=symbol,
        venue="SMART",
        tick_size=0.01,
        tick_value=0.01,
        multiplier=1.0,
        currency="USD",
    )


def _make_order(strategy_id: str, status: OrderStatus) -> OMSOrder:
    now = datetime.now(timezone.utc)
    return OMSOrder(
        strategy_id=strategy_id,
        instrument=_make_instrument(),
        side=OrderSide.BUY,
        qty=5,
        order_type=OrderType.LIMIT,
        limit_price=100.0,
        role=OrderRole.ENTRY,
        status=status,
        reject_reason="blocked by test",
        created_at=now,
        last_update_at=now,
        filled_qty=2,
        remaining_qty=3,
        avg_fill_price=101.25,
    )


async def _run_breakout_event(engine: BreakoutEngine, event: OMSEvent) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    engine._running = True
    task = asyncio.create_task(engine._process_events(queue))
    await queue.put(event)
    await asyncio.sleep(0.05)
    engine._running = False
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def _make_breakout_filled_setup(
    *,
    fill_qty: int = 10,
    qty_open: int | None = None,
    fill_price: float = 100.0,
) -> SetupInstance:
    open_qty = fill_qty if qty_open is None else qty_open
    return SetupInstance(
        symbol="QQQ",
        direction=BreakoutDirection.LONG,
        entry_type=EntryType.A_AVWAP_RETEST,
        state=SetupState.FILLED,
        campaign_id=1,
        box_version=1,
        entry_price=100.0,
        stop0=95.0,
        final_risk_dollars=500.0,
        quality_mult=1.0,
        expiry_mult=1.0,
        shares_planned=fill_qty,
        current_stop=95.0,
        oca_group="BRK_QQQ_1",
        exit_tier=ExitTier.NEUTRAL,
        fill_price=fill_price,
        fill_qty=fill_qty,
        fill_ts=datetime.now(timezone.utc),
        avg_entry=fill_price,
        qty_open=open_qty,
    )


@pytest.mark.asyncio
async def test_event_bus_broadcasts_global_risk_halt_and_order_payloads() -> None:
    bus = EventBus()
    strategy_a = bus.subscribe("strategy_a")
    strategy_b = bus.subscribe("strategy_b")
    global_q = bus.subscribe_all()

    bus.emit_risk_halt("", "callback exception")

    event_a = strategy_a.get_nowait()
    event_b = strategy_b.get_nowait()
    event_global = global_q.get_nowait()
    assert event_a.event_type == OMSEventType.RISK_HALT
    assert event_b.event_type == OMSEventType.RISK_HALT
    assert event_global.payload["reason"] == "callback exception"

    order = _make_order("strategy_a", OrderStatus.REJECTED)
    bus.emit_order_event(order)

    reject_event = strategy_a.get_nowait()
    assert reject_event.event_type == OMSEventType.ORDER_REJECTED
    assert reject_event.payload["reject_reason"] == "blocked by test"
    assert reject_event.payload["role"] == OrderRole.ENTRY.value
    assert reject_event.payload["order_type"] == OrderType.LIMIT.value
    assert reject_event.payload["filled_qty"] == 2


@pytest.mark.asyncio
async def test_execution_router_expires_stale_queued_orders_while_congested() -> None:
    adapter = SimpleNamespace(is_congested=True, submit_order=AsyncMock())
    repo = InMemoryRepository()
    bus = EventBus()
    queue = bus.subscribe("router_test")
    router = ExecutionRouter(adapter, repo, bus=bus)

    order = _make_order("router_test", OrderStatus.RISK_APPROVED)
    await repo.save_order(order)
    router._queue.append(  # noqa: SLF001 - targeted regression coverage
        (
            OrderPriority.NEW_ENTRY,
            order,
            {"queued_at": datetime(2020, 1, 1, tzinfo=timezone.utc)},
        )
    )

    await router._expire_stale_queued_orders()  # noqa: SLF001 - targeted regression coverage

    updated = await repo.get_order(order.oms_order_id)
    assert updated is not None
    assert updated.status == OrderStatus.EXPIRED
    assert len(router._queue) == 0  # noqa: SLF001 - targeted regression coverage
    assert adapter.submit_order.await_count == 0
    assert repo._events[-1]["event_type"] == "QUEUE_EXPIRED"  # noqa: SLF001 - targeted regression coverage
    expired_event = queue.get_nowait()
    assert expired_event.event_type == OMSEventType.ORDER_EXPIRED


@pytest.mark.asyncio
async def test_keltner_engine_uses_fill_event_price_and_qty_payload() -> None:
    oms = SimpleNamespace(
        submit_intent=AsyncMock(return_value=SimpleNamespace(oms_order_id="stop-1"))
    )
    engine = KeltnerEngine(
        strategy_id="S5_TEST",
        ib_session=None,
        oms_service=oms,
        instruments={"QQQ": _make_instrument()},
        config={"QQQ": KeltnerSymbolConfig(symbol="QQQ")},
    )
    engine._running = True
    engine._pending_entry["QQQ"] = {
        "direction": KeltnerDirection.LONG,
        "stop_dist": 2.5,
    }
    engine._order_to_symbol["entry-1"] = "QQQ"
    engine._order_role["entry-1"] = "entry"

    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(engine._process_events(queue))
    await queue.put(
        OMSEvent(
            event_type=OMSEventType.FILL,
            timestamp=datetime.now(timezone.utc),
            strategy_id="S5_TEST",
            oms_order_id="entry-1",
            payload={"price": 101.5, "qty": 3},
        )
    )
    await asyncio.sleep(0.05)
    engine._running = False
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    position = engine.positions["QQQ"]
    assert position.fill_price == 101.5
    assert position.qty == 3
    assert position.initial_stop == pytest.approx(99.0)
    assert oms.submit_intent.await_count == 1


@pytest.mark.asyncio
async def test_breakout_engine_uses_fill_event_payload_and_real_event_names() -> None:
    oms = SimpleNamespace(
        submit_intent=AsyncMock(
            side_effect=[
                SimpleNamespace(oms_order_id="stop-1"),
                SimpleNamespace(oms_order_id="tp1-1"),
                SimpleNamespace(oms_order_id="tp2-1"),
            ]
        )
    )
    engine = BreakoutEngine(
        ib_session=None,
        oms_service=oms,
        instruments={"QQQ": _make_instrument()},
        config={"QQQ": BreakoutSymbolConfig(symbol="QQQ")},
    )
    engine._running = True
    engine.campaigns["QQQ"].state = CampaignState.BREAKOUT

    setup = SetupInstance(
        symbol="QQQ",
        direction=BreakoutDirection.LONG,
        entry_type=EntryType.A_AVWAP_RETEST,
        state=SetupState.ARMED,
        campaign_id=1,
        box_version=1,
        entry_price=100.0,
        stop0=95.0,
        final_risk_dollars=500.0,
        quality_mult=1.0,
        expiry_mult=1.0,
        shares_planned=10,
        current_stop=95.0,
        oca_group="BRK_QQQ_1",
        exit_tier=ExitTier.NEUTRAL,
    )
    engine.active_setups[setup.setup_id] = setup
    engine._order_to_setup["entry-1"] = setup.setup_id
    engine._order_kind["entry-1"] = "primary_entry"

    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(engine._process_events(queue))
    await queue.put(
        OMSEvent(
            event_type=OMSEventType.FILL,
            timestamp=datetime.now(timezone.utc),
            strategy_id="BREAKOUT",
            oms_order_id="entry-1",
            payload={"price": 101.0, "qty": 10},
        )
    )
    await asyncio.sleep(0.05)
    engine._running = False
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    position = engine.positions["QQQ"]
    assert setup.state == SetupState.FILLED
    assert setup.fill_price == 101.0
    assert setup.fill_qty == 10
    assert position.qty == 10
    assert engine._order_kind["stop-1"] == "stop"
    assert engine._order_kind["tp1-1"] == "tp1"
    assert engine._order_kind["tp2-1"] == "tp2"


@pytest.mark.asyncio
async def test_breakout_engine_applies_add_fill_to_live_position_state() -> None:
    engine = BreakoutEngine(
        ib_session=None,
        oms_service=SimpleNamespace(submit_intent=AsyncMock()),
        instruments={"QQQ": _make_instrument()},
        config={"QQQ": BreakoutSymbolConfig(symbol="QQQ")},
    )
    engine.campaigns["QQQ"].state = CampaignState.POSITION_OPEN

    setup = _make_breakout_filled_setup(fill_qty=10, qty_open=10, fill_price=100.0)
    engine.active_setups[setup.setup_id] = setup
    engine.positions["QQQ"] = PositionState(
        symbol="QQQ",
        direction=BreakoutDirection.LONG,
        qty=10,
        avg_cost=100.0,
        current_stop=95.0,
        campaign_id=1,
        box_version=1,
    )
    engine._track_order("add-1", setup.setup_id, "add_entry", 5)  # noqa: SLF001

    await _run_breakout_event(
        engine,
        OMSEvent(
            event_type=OMSEventType.FILL,
            timestamp=datetime.now(timezone.utc),
            strategy_id="BREAKOUT",
            oms_order_id="add-1",
            payload={"price": 110.0, "qty": 5},
        ),
    )

    position = engine.positions["QQQ"]
    assert setup.state == SetupState.ACTIVE
    assert setup.fill_qty == 15
    assert setup.qty_open == 15
    assert setup.avg_entry == pytest.approx((100.0 * 10 + 110.0 * 5) / 15)
    assert setup.add_count == 1
    assert position.qty == 15
    assert position.avg_cost == pytest.approx(setup.avg_entry)
    assert position.add_count == 1
    assert "add-1" not in engine._order_kind


@pytest.mark.asyncio
async def test_breakout_engine_applies_tp_fill_to_live_position_state() -> None:
    receipt = SimpleNamespace(oms_order_id="new-stop-1")
    oms = SimpleNamespace(submit_intent=AsyncMock(return_value=receipt))
    engine = BreakoutEngine(
        ib_session=None,
        oms_service=oms,
        instruments={"QQQ": _make_instrument()},
        config={"QQQ": BreakoutSymbolConfig(symbol="QQQ")},
    )
    engine.campaigns["QQQ"].state = CampaignState.POSITION_OPEN

    setup = _make_breakout_filled_setup(fill_qty=9, qty_open=9, fill_price=100.0)
    setup.stop_order_id = "stop-1"
    setup.tp1_order_id = "tp1-1"
    setup.tp2_order_id = "tp2-1"
    engine.active_setups[setup.setup_id] = setup
    engine.positions["QQQ"] = PositionState(
        symbol="QQQ",
        direction=BreakoutDirection.LONG,
        qty=9,
        avg_cost=100.0,
        current_stop=95.0,
        campaign_id=1,
        box_version=1,
    )
    engine._track_order("stop-1", setup.setup_id, "stop", 9)  # noqa: SLF001
    engine._track_order("tp1-1", setup.setup_id, "tp1", 3)  # noqa: SLF001
    engine._track_order("tp2-1", setup.setup_id, "tp2", 3)  # noqa: SLF001

    await _run_breakout_event(
        engine,
        OMSEvent(
            event_type=OMSEventType.FILL,
            timestamp=datetime.now(timezone.utc),
            strategy_id="BREAKOUT",
            oms_order_id="tp1-1",
            payload={"price": 103.0, "qty": 3},
        ),
    )

    position = engine.positions["QQQ"]
    assert setup.tp1_done is True
    assert position.tp1_done is True
    assert setup.state == SetupState.FILLED
    assert setup.qty_open == 6
    assert position.qty == 6
    assert setup.realized_pnl == pytest.approx(9.0)
    assert setup.tp1_order_id == ""
    # Stop order was amended: old cancelled, new submitted with qty=6
    assert setup.stop_order_id == "new-stop-1"
    assert engine._order_kind["new-stop-1"] == "stop"
    assert engine._order_requested_qty["new-stop-1"] == 6


@pytest.mark.asyncio
async def test_breakout_engine_applies_stop_fill_to_live_position_state() -> None:
    engine = BreakoutEngine(
        ib_session=None,
        oms_service=SimpleNamespace(submit_intent=AsyncMock()),
        instruments={"QQQ": _make_instrument()},
        config={"QQQ": BreakoutSymbolConfig(symbol="QQQ")},
    )
    engine.campaigns["QQQ"].state = CampaignState.POSITION_OPEN

    setup = _make_breakout_filled_setup(fill_qty=10, qty_open=10, fill_price=100.0)
    setup.stop_order_id = "stop-1"
    setup.tp1_order_id = "tp1-1"
    setup.tp2_order_id = "tp2-1"
    engine.active_setups[setup.setup_id] = setup
    engine.positions["QQQ"] = PositionState(
        symbol="QQQ",
        direction=BreakoutDirection.LONG,
        qty=10,
        avg_cost=100.0,
        current_stop=95.0,
        campaign_id=1,
        box_version=1,
    )
    engine._track_order("stop-1", setup.setup_id, "stop", 10)  # noqa: SLF001
    engine._track_order("tp1-1", setup.setup_id, "tp1", 3)  # noqa: SLF001
    engine._track_order("tp2-1", setup.setup_id, "tp2", 3)  # noqa: SLF001

    await _run_breakout_event(
        engine,
        OMSEvent(
            event_type=OMSEventType.FILL,
            timestamp=datetime.now(timezone.utc),
            strategy_id="BREAKOUT",
            oms_order_id="stop-1",
            payload={"price": 94.0, "qty": 10},
        ),
    )

    position = engine.positions["QQQ"]
    campaign = engine.campaigns["QQQ"]
    assert setup.state == SetupState.CLOSED
    assert setup.qty_open == 0
    assert position.qty == 0
    assert setup.realized_pnl == pytest.approx(-60.0)
    assert setup.r_state == pytest.approx(-0.12)
    assert setup.stop_order_id == ""
    assert setup.tp1_order_id == ""
    assert setup.tp2_order_id == ""
    assert campaign.last_exit_direction == BreakoutDirection.LONG
    assert "stop-1" not in engine._order_kind
    assert "tp1-1" not in engine._order_kind
    assert "tp2-1" not in engine._order_kind


# ---------------------------------------------------------------------------
# Exit lifecycle hardening tests (Steps 2-6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_fill_cancels_outstanding_tp_orders() -> None:
    """Stop fill should submit CANCEL_ORDER intents for TP1 and TP2."""
    cancel_ids = []

    async def mock_submit(intent):
        if intent.intent_type.value == "CANCEL_ORDER":
            cancel_ids.append(intent.target_oms_order_id)
        return SimpleNamespace(oms_order_id=f"receipt-{len(cancel_ids)}")

    oms = SimpleNamespace(submit_intent=mock_submit)
    engine = BreakoutEngine(
        ib_session=None,
        oms_service=oms,
        instruments={"QQQ": _make_instrument()},
        config={"QQQ": BreakoutSymbolConfig(symbol="QQQ")},
    )
    engine.campaigns["QQQ"].state = CampaignState.POSITION_OPEN

    setup = _make_breakout_filled_setup(fill_qty=10, qty_open=10)
    setup.stop_order_id = "stop-1"
    setup.tp1_order_id = "tp1-1"
    setup.tp2_order_id = "tp2-1"
    engine.active_setups[setup.setup_id] = setup
    engine.positions["QQQ"] = PositionState(
        symbol="QQQ", direction=BreakoutDirection.LONG,
        qty=10, avg_cost=100.0, current_stop=95.0,
        campaign_id=1, box_version=1,
    )
    engine._track_order("stop-1", setup.setup_id, "stop", 10)
    engine._track_order("tp1-1", setup.setup_id, "tp1", 3)
    engine._track_order("tp2-1", setup.setup_id, "tp2", 3)

    await _run_breakout_event(
        engine,
        OMSEvent(
            event_type=OMSEventType.FILL,
            timestamp=datetime.now(timezone.utc),
            strategy_id="BREAKOUT",
            oms_order_id="stop-1",
            payload={"price": 94.0, "qty": 10},
        ),
    )

    assert "tp1-1" in cancel_ids
    assert "tp2-1" in cancel_ids
    assert setup.state == SetupState.CLOSED


@pytest.mark.asyncio
async def test_tp_fill_closing_position_cancels_stop_order() -> None:
    """TP fill that brings qty_open to 0 should cancel the stop order."""
    cancel_ids = []

    async def mock_submit(intent):
        if intent.intent_type.value == "CANCEL_ORDER":
            cancel_ids.append(intent.target_oms_order_id)
        return SimpleNamespace(oms_order_id=f"receipt-{len(cancel_ids)}")

    oms = SimpleNamespace(submit_intent=mock_submit)
    engine = BreakoutEngine(
        ib_session=None,
        oms_service=oms,
        instruments={"QQQ": _make_instrument()},
        config={"QQQ": BreakoutSymbolConfig(symbol="QQQ")},
    )
    engine.campaigns["QQQ"].state = CampaignState.POSITION_OPEN

    # Setup with only 3 shares left (simulating post-TP1)
    setup = _make_breakout_filled_setup(fill_qty=10, qty_open=3)
    setup.stop_order_id = "stop-1"
    setup.tp1_done = True
    setup.tp1_order_id = ""
    setup.tp2_order_id = "tp2-1"
    engine.active_setups[setup.setup_id] = setup
    engine.positions["QQQ"] = PositionState(
        symbol="QQQ", direction=BreakoutDirection.LONG,
        qty=3, avg_cost=100.0, current_stop=95.0,
        campaign_id=1, box_version=1,
    )
    engine._track_order("stop-1", setup.setup_id, "stop", 3)
    engine._track_order("tp2-1", setup.setup_id, "tp2", 3)

    await _run_breakout_event(
        engine,
        OMSEvent(
            event_type=OMSEventType.FILL,
            timestamp=datetime.now(timezone.utc),
            strategy_id="BREAKOUT",
            oms_order_id="tp2-1",
            payload={"price": 110.0, "qty": 3},
        ),
    )

    assert "stop-1" in cancel_ids
    assert setup.state == SetupState.CLOSED
    assert setup.qty_open == 0


@pytest.mark.asyncio
async def test_close_position_noop_when_qty_open_zero() -> None:
    """_close_position should be a no-op when qty_open is already 0."""
    oms = SimpleNamespace(submit_intent=AsyncMock())
    engine = BreakoutEngine(
        ib_session=None,
        oms_service=oms,
        instruments={"QQQ": _make_instrument()},
        config={"QQQ": BreakoutSymbolConfig(symbol="QQQ")},
    )

    setup = _make_breakout_filled_setup(fill_qty=10, qty_open=0)
    setup.state = SetupState.CLOSED
    engine.active_setups[setup.setup_id] = setup

    await engine._close_position(setup, 105.0, "test", datetime.now(timezone.utc))

    # submit_intent should never have been called
    oms.submit_intent.assert_not_awaited()


@pytest.mark.asyncio
async def test_close_position_noop_when_exit_pending() -> None:
    """_close_position should skip if a stop order is already in flight."""
    oms = SimpleNamespace(submit_intent=AsyncMock())
    engine = BreakoutEngine(
        ib_session=None,
        oms_service=oms,
        instruments={"QQQ": _make_instrument()},
        config={"QQQ": BreakoutSymbolConfig(symbol="QQQ")},
    )

    setup = _make_breakout_filled_setup(fill_qty=10, qty_open=10)
    setup.stop_order_id = "stop-1"
    engine.active_setups[setup.setup_id] = setup
    engine._track_order("stop-1", setup.setup_id, "stop", 10)

    await engine._close_position(setup, 94.0, "stop_hit", datetime.now(timezone.utc))

    # submit_intent should never have been called — broker owns the exit
    oms.submit_intent.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_close_orders_tracked_and_fills_reconciled() -> None:
    """_submit_partial_close should track the order; fill should update qty."""
    receipt = SimpleNamespace(oms_order_id="partial-1")
    oms = SimpleNamespace(submit_intent=AsyncMock(return_value=receipt))
    engine = BreakoutEngine(
        ib_session=None,
        oms_service=oms,
        instruments={"QQQ": _make_instrument()},
        config={"QQQ": BreakoutSymbolConfig(symbol="QQQ")},
    )
    engine.campaigns["QQQ"].state = CampaignState.POSITION_OPEN

    setup = _make_breakout_filled_setup(fill_qty=9, qty_open=9)
    engine.active_setups[setup.setup_id] = setup
    engine.positions["QQQ"] = PositionState(
        symbol="QQQ", direction=BreakoutDirection.LONG,
        qty=9, avg_cost=100.0, current_stop=95.0,
        campaign_id=1, box_version=1,
    )

    await engine._submit_partial_close(setup, 3, 105.0, "tp1", datetime.now(timezone.utc))

    # Order should be tracked
    assert "partial-1" in engine._order_kind
    assert engine._order_kind["partial-1"] == "partial_close"
    # qty_open should NOT be decremented yet (fill handler does it)
    assert setup.qty_open == 9

    # Now simulate the fill
    await _run_breakout_event(
        engine,
        OMSEvent(
            event_type=OMSEventType.FILL,
            timestamp=datetime.now(timezone.utc),
            strategy_id="BREAKOUT",
            oms_order_id="partial-1",
            payload={"price": 105.0, "qty": 3},
        ),
    )

    assert setup.qty_open == 6
    assert "partial-1" not in engine._order_kind


@pytest.mark.asyncio
async def test_hourly_exit_skips_setups_with_broker_brackets() -> None:
    """_manage_exits should not fire stop-hit when stop_order_id is set."""
    oms = SimpleNamespace(submit_intent=AsyncMock())
    engine = BreakoutEngine(
        ib_session=None,
        oms_service=oms,
        instruments={"QQQ": _make_instrument()},
        config={"QQQ": BreakoutSymbolConfig(symbol="QQQ")},
    )
    engine._running = True
    engine.campaigns["QQQ"].state = CampaignState.POSITION_OPEN

    setup = _make_breakout_filled_setup(fill_qty=10, qty_open=10, fill_price=100.0)
    setup.stop_order_id = "stop-1"  # Broker owns the stop
    setup.fill_ts = datetime.now(timezone.utc) - timedelta(hours=2)
    engine.active_setups[setup.setup_id] = setup
    engine.positions["QQQ"] = PositionState(
        symbol="QQQ", direction=BreakoutDirection.LONG,
        qty=10, avg_cost=100.0, current_stop=95.0,
        campaign_id=1, box_version=1,
    )
    engine._track_order("stop-1", setup.setup_id, "stop", 10)

    # Price is below the stop — but broker owns it, so engine should NOT close
    from strategy_3.models import HourlyState
    engine.hourly_states["QQQ"] = HourlyState()
    engine.hourly_states["QQQ"].close = 93.0  # below stop of 95

    now_et = datetime(2026, 3, 13, 11, 0, tzinfo=timezone.utc)
    await engine._manage_exits(now_et)

    # Engine should NOT have submitted any close orders
    oms.submit_intent.assert_not_awaited()
    # Setup should still be open
    assert setup.state == SetupState.FILLED
    assert setup.qty_open == 10


@pytest.mark.asyncio
async def test_finalized_setup_removed_from_active_setups() -> None:
    """_finalize_closed_setup should pop the setup from active_setups."""
    oms = SimpleNamespace(submit_intent=AsyncMock())
    engine = BreakoutEngine(
        ib_session=None,
        oms_service=oms,
        instruments={"QQQ": _make_instrument()},
        config={"QQQ": BreakoutSymbolConfig(symbol="QQQ")},
    )
    engine.campaigns["QQQ"].state = CampaignState.POSITION_OPEN

    setup = _make_breakout_filled_setup(fill_qty=10, qty_open=10)
    engine.active_setups[setup.setup_id] = setup
    engine.positions["QQQ"] = PositionState(
        symbol="QQQ", direction=BreakoutDirection.LONG,
        qty=10, avg_cost=100.0, current_stop=95.0,
        campaign_id=1, box_version=1,
    )

    await engine._finalize_closed_setup(
        setup, 94.0, "stop_fill", datetime.now(timezone.utc), order_id="stop-1",
    )

    assert setup.setup_id not in engine.active_setups
    assert setup.state == SetupState.CLOSED


@pytest.mark.asyncio
async def test_tp_fill_amends_stop_qty() -> None:
    """After partial TP fill, stop order should be replaced with reduced qty."""
    receipt = SimpleNamespace(oms_order_id="new-stop-1")
    oms = SimpleNamespace(submit_intent=AsyncMock(return_value=receipt))
    engine = BreakoutEngine(
        ib_session=None,
        oms_service=oms,
        instruments={"QQQ": _make_instrument()},
        config={"QQQ": BreakoutSymbolConfig(symbol="QQQ")},
    )
    engine._running = True
    engine.campaigns["QQQ"].state = CampaignState.POSITION_OPEN

    setup = _make_breakout_filled_setup(fill_qty=10, qty_open=10, fill_price=100.0)
    setup.stop_order_id = "old-stop-1"
    setup.tp1_order_id = "tp1-1"
    engine.active_setups[setup.setup_id] = setup
    engine.positions["QQQ"] = PositionState(
        symbol="QQQ", direction=BreakoutDirection.LONG,
        qty=10, avg_cost=100.0, current_stop=95.0,
        campaign_id=1, box_version=1,
    )
    engine._track_order("old-stop-1", setup.setup_id, "stop", 10)
    engine._track_order("tp1-1", setup.setup_id, "tp1", 3)

    # Simulate TP1 fill of 3 shares
    event = OMSEvent(
        event_type=OMSEventType.ORDER_FILLED,
        timestamp=datetime.now(timezone.utc),
        strategy_id="BREAKOUT",
        oms_order_id="tp1-1",
        payload={"fill_price": 105.0, "fill_qty": 3},
    )
    await _run_breakout_event(engine, event)

    # qty_open should be reduced
    assert setup.qty_open == 7
    assert setup.tp1_done is True

    # Stop should have been amended: cancel old + submit new
    calls = oms.submit_intent.call_args_list
    # First call: cancel old stop, second call: new stop with qty=7
    cancel_calls = [c for c in calls if c.args[0].intent_type.value == "CANCEL_ORDER"]
    new_calls = [c for c in calls if c.args[0].intent_type.value == "NEW_ORDER"]
    assert len(cancel_calls) >= 1
    assert cancel_calls[0].args[0].target_oms_order_id == "old-stop-1"
    assert len(new_calls) >= 1
    assert new_calls[0].args[0].order.qty == 7
    assert setup.stop_order_id == "new-stop-1"
