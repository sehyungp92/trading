from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from libs.oms.engine.fill_processor import FillProcessor
from libs.oms.engine.timeout_monitor import OrderTimeoutMonitor
from libs.oms.execution.router import ExecutionRouter, OrderPriority
from libs.oms.intent.handler import IntentHandler
from libs.oms.models.instrument import Instrument
from libs.oms.models.intent import Intent, IntentResult, IntentType
from libs.oms.models.order import OMSOrder, OrderRole, OrderSide, OrderStatus, OrderType
from libs.oms.models.risk_state import PortfolioRiskState
from libs.oms.persistence.in_memory import InMemoryRepository
from libs.oms.persistence.repository import OMSPersistenceInvariantError, OMSRepository
from libs.oms.services.factory import _wire_adapter_callbacks, _wire_adapter_callbacks_multi


def _instrument(symbol: str = "QQQ") -> Instrument:
    return Instrument(
        symbol=symbol,
        root=symbol,
        venue="SMART",
        tick_size=0.01,
        tick_value=0.01,
        multiplier=1.0,
        point_value=1.0,
        currency="USD",
        primary_exchange="NASDAQ",
        sec_type="STK",
    )


def _entry_order(symbol: str = "QQQ") -> OMSOrder:
    return OMSOrder(
        oms_order_id=f"oms-{symbol.lower()}",
        client_order_id=f"client-{symbol.lower()}",
        strategy_id="TEST",
        account_id="DU123",
        instrument=_instrument(symbol),
        side=OrderSide.BUY,
        qty=10,
        order_type=OrderType.LIMIT,
        limit_price=100.0,
        role=OrderRole.ENTRY,
        status=OrderStatus.CREATED,
    )


class _Acquire:
    def __init__(self, conn) -> None:
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _BrokenConnection:
    async def execute(self, *args, **kwargs):
        raise RuntimeError("fk violation")


class _BrokenPool:
    def __init__(self, conn) -> None:
        self._conn = conn

    def acquire(self):
        return _Acquire(self._conn)


class _DelayedWorkingSaveRepository(InMemoryRepository):
    def __init__(self) -> None:
        super().__init__()
        self.working_save_started = asyncio.Event()
        self.release_working_save = asyncio.Event()
        self.saved_statuses: list[OrderStatus] = []

    async def save_order(self, order: OMSOrder) -> None:
        self.saved_statuses.append(order.status)
        if order.status == OrderStatus.WORKING and not self.release_working_save.is_set():
            self.working_save_started.set()
            await self.release_working_save.wait()
        await super().save_order(order)


async def _wait_for_order_status(
    repo: InMemoryRepository,
    oms_order_id: str,
    expected: OrderStatus,
) -> OMSOrder:
    for _ in range(200):
        order = await repo.get_order(oms_order_id)
        if order is not None and order.status == expected:
            return order
        await asyncio.sleep(0.01)
    raise AssertionError(f"Order {oms_order_id} never reached {expected.value}")


@pytest.mark.asyncio
async def test_intent_handler_denial_uses_atomic_helper() -> None:
    risk = MagicMock()
    risk.check_entry = AsyncMock(return_value="risk denied")
    router = MagicMock()
    router.route = AsyncMock()
    repo = MagicMock()
    repo.get_order_id_by_client_order_id = AsyncMock(return_value=None)
    repo.get_positions = AsyncMock(return_value=[])
    repo.save_order_and_event = AsyncMock()
    bus = MagicMock()
    bus.emit_risk_denial = MagicMock()
    bus.emit_order_event = MagicMock()

    handler = IntentHandler(risk, router, repo, bus)
    order = _entry_order()

    receipt = await handler.submit(
        Intent(intent_type=IntentType.NEW_ORDER, strategy_id="TEST", order=order)
    )

    assert receipt.result == IntentResult.DENIED
    repo.save_order_and_event.assert_awaited_once()
    saved_order, event_type, payload = repo.save_order_and_event.await_args.args
    assert saved_order.status == OrderStatus.REJECTED
    assert event_type == "RISK_DENIED"
    assert payload == {"reason": "risk denied"}


@pytest.mark.asyncio
async def test_intent_handler_approval_uses_atomic_helper() -> None:
    risk = MagicMock()
    risk.check_entry = AsyncMock(return_value=None)
    router = MagicMock()
    router.route = AsyncMock()
    repo = MagicMock()
    repo.get_order_id_by_client_order_id = AsyncMock(return_value=None)
    repo.get_positions = AsyncMock(return_value=[])
    repo.save_order_and_event = AsyncMock()
    bus = MagicMock()
    bus.emit_risk_denial = MagicMock()
    bus.emit_order_event = MagicMock()

    handler = IntentHandler(risk, router, repo, bus)
    order = _entry_order("GLD")

    receipt = await handler.submit(
        Intent(intent_type=IntentType.NEW_ORDER, strategy_id="TEST", order=order)
    )

    assert receipt.result == IntentResult.ACCEPTED
    repo.save_order_and_event.assert_awaited_once()
    saved_order, event_type, payload = repo.save_order_and_event.await_args.args
    assert saved_order.status == OrderStatus.RISK_APPROVED
    assert event_type == "RISK_APPROVED"
    assert payload == {}


@pytest.mark.asyncio
async def test_execution_router_queue_expiry_uses_atomic_helper() -> None:
    adapter = MagicMock()
    adapter.is_congested = True
    repo = MagicMock()
    repo.save_order_and_event = AsyncMock()
    bus = MagicMock()
    bus.emit_order_event = MagicMock()
    router = ExecutionRouter(adapter, repo, bus)
    now = datetime.now(timezone.utc)
    order = _entry_order()
    order.status = OrderStatus.RISK_APPROVED
    router._queue = [
        (
            OrderPriority.NEW_ENTRY,
            order,
            {"queued_at": now - timedelta(seconds=301)},
        )
    ]

    await router._expire_stale_queued_orders()

    assert order.status == OrderStatus.EXPIRED
    repo.save_order_and_event.assert_awaited_once()
    assert router._queue == []


@pytest.mark.asyncio
async def test_execution_router_submit_failure_persists_rejection_event_and_emits_bus() -> None:
    adapter = MagicMock()
    adapter.is_congested = False
    adapter.submit_order = AsyncMock(side_effect=RuntimeError("submit exploded"))
    repo = MagicMock()
    repo.save_order = AsyncMock()
    repo.save_order_and_event = AsyncMock()
    bus = MagicMock()
    bus.emit_order_event = MagicMock()
    router = ExecutionRouter(adapter, repo, bus)
    order = _entry_order("MSFT")
    order.status = OrderStatus.RISK_APPROVED
    order.remaining_qty = order.qty

    await router._submit_to_adapter(order)

    assert order.status == OrderStatus.REJECTED
    assert order.reject_reason == "submit exploded"
    assert order.last_update_at is not None
    repo.save_order.assert_awaited_once()
    repo.save_order_and_event.assert_awaited_once()
    saved_order, event_type, payload = repo.save_order_and_event.await_args.args
    assert saved_order is order
    assert event_type == "BROKER_SUBMIT_FAILED"
    assert payload["error_type"] == "RuntimeError"
    assert payload["error"] == "submit exploded"
    assert payload["instrument_backed"] is True
    bus.emit_order_event.assert_called_once_with(order)


@pytest.mark.asyncio
async def test_execution_router_passes_instrument_to_adapter_submit() -> None:
    adapter = MagicMock()
    adapter.is_congested = False
    adapter.submit_order = AsyncMock(
        return_value=SimpleNamespace(broker_order_id=101, perm_id=202)
    )
    repo = MagicMock()
    repo.save_order = AsyncMock()
    router = ExecutionRouter(adapter, repo)
    order = _entry_order("IBIT")
    order.status = OrderStatus.RISK_APPROVED
    order.remaining_qty = order.qty

    await router._submit_to_adapter(order)

    assert adapter.submit_order.await_args.kwargs["instrument"] is order.instrument
    assert order.broker_order_id == 101
    assert order.perm_id == 202


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "timestamp_attr", "payload_reason"),
    [
        (OrderStatus.ROUTED, "submitted_at", "routed_timeout"),
        (OrderStatus.CANCEL_REQUESTED, "last_update_at", "cancel_timeout"),
    ],
)
async def test_timeout_monitor_uses_atomic_helper(
    status: OrderStatus,
    timestamp_attr: str,
    payload_reason: str,
) -> None:
    repo = MagicMock()
    repo.save_order_and_event = AsyncMock()
    bus = MagicMock()
    bus.emit_order_event = MagicMock()
    monitor = OrderTimeoutMonitor(repo, bus, routed_timeout_s=30.0, cancel_timeout_s=15.0)
    order = _entry_order("NFLX")
    order.status = status
    stale_ts = datetime.now(timezone.utc) - timedelta(seconds=60)
    setattr(order, timestamp_attr, stale_ts)
    order.created_at = stale_ts
    repo.get_all_working_orders = AsyncMock(return_value=[order])

    await monitor._scan_stuck_orders()

    assert order.status == OrderStatus.CANCELLED
    repo.save_order_and_event.assert_awaited_once()
    _, event_type, payload = repo.save_order_and_event.await_args.args
    assert event_type == "TIMEOUT_CANCELLED"
    assert payload["reason"] == payload_reason


@pytest.mark.asyncio
async def test_fill_processor_uses_atomic_fill_helper_only() -> None:
    repo = MagicMock()
    repo.fill_exists = AsyncMock(return_value=False)
    repo.save_fill = AsyncMock()
    repo.save_order_fill_and_event = AsyncMock(return_value=True)
    order = _entry_order("IBIT")
    order.status = OrderStatus.WORKING
    order.remaining_qty = 10
    repo.get_order = AsyncMock(return_value=order)
    processor = FillProcessor(repo)

    await processor.process_fill(
        oms_order_id=order.oms_order_id,
        broker_fill_id="exec-1",
        price=101.25,
        qty=10,
        timestamp=datetime.now(timezone.utc),
        fees=1.5,
    )

    repo.save_fill.assert_not_awaited()
    repo.save_order_fill_and_event.assert_awaited_once()
    saved_order, fill, event_type, payload = repo.save_order_fill_and_event.await_args.args
    assert saved_order.status == OrderStatus.FILLED
    assert fill.broker_fill_id == "exec-1"
    assert event_type == "FILL"
    assert payload["qty"] == 10


@pytest.mark.asyncio
async def test_fill_processor_ignores_duplicate_fill_after_race() -> None:
    repo = MagicMock()
    repo.fill_exists = AsyncMock(return_value=False)
    repo.save_order_fill_and_event = AsyncMock(return_value=False)
    order = _entry_order("IBIT")
    order.status = OrderStatus.WORKING
    order.remaining_qty = 10
    repo.get_order = AsyncMock(return_value=order)
    processor = FillProcessor(repo)

    await processor.process_fill(
        oms_order_id=order.oms_order_id,
        broker_fill_id="exec-race",
        price=101.25,
        qty=10,
        timestamp=datetime.now(timezone.utc),
        fees=1.5,
    )

    repo.save_order_fill_and_event.assert_awaited_once()
    assert order.filled_qty == 0.0
    assert order.remaining_qty == 10


@pytest.mark.asyncio
async def test_fill_processor_serializes_distinct_partial_fills_per_order() -> None:
    repo = InMemoryRepository()
    order = _entry_order("IBIT")
    order.status = OrderStatus.WORKING
    order.remaining_qty = 10
    await repo.save_order(order)
    processor = FillProcessor(repo)
    now = datetime.now(timezone.utc)

    await asyncio.gather(
        processor.process_fill(
            oms_order_id=order.oms_order_id,
            broker_fill_id="exec-1",
            price=100.0,
            qty=5,
            timestamp=now,
        ),
        processor.process_fill(
            oms_order_id=order.oms_order_id,
            broker_fill_id="exec-2",
            price=101.0,
            qty=5,
            timestamp=now + timedelta(seconds=1),
        ),
    )

    updated = await repo.get_order(order.oms_order_id)
    assert updated is not None
    assert updated.filled_qty == 10
    assert updated.remaining_qty == 0
    assert updated.status == OrderStatus.FILLED


@pytest.mark.asyncio
async def test_fill_processor_accepts_status_first_terminal_state_without_warning(caplog) -> None:
    repo = InMemoryRepository()
    order = _entry_order("NVDA")
    order.status = OrderStatus.FILLED
    order.remaining_qty = 10
    await repo.save_order(order)
    processor = FillProcessor(repo)
    caplog.set_level(logging.WARNING)

    await processor.process_fill(
        oms_order_id=order.oms_order_id,
        broker_fill_id="exec-status-first",
        price=101.25,
        qty=10,
        timestamp=datetime.now(timezone.utc),
        fees=1.5,
    )

    updated = await repo.get_order(order.oms_order_id)
    assert updated is not None
    assert updated.status == OrderStatus.FILLED
    assert updated.filled_qty == 10
    assert "Invalid transition" not in caplog.text
    assert "Fill transition rejected" not in caplog.text


@pytest.mark.asyncio
async def test_single_oms_callbacks_serialize_status_ack_then_fill(caplog) -> None:
    repo = _DelayedWorkingSaveRepository()
    order = _entry_order("QQQ")
    order.status = OrderStatus.ROUTED
    order.remaining_qty = order.qty
    await repo.save_order(order)
    repo.saved_statuses.clear()
    bus = MagicMock()
    bus.emit_order_event = MagicMock()
    bus.emit_fill_event = MagicMock()
    adapter = SimpleNamespace()
    caplog.set_level(logging.WARNING)

    _wire_adapter_callbacks(
        adapter=adapter,
        bus=bus,
        repo=repo,
        fill_proc=FillProcessor(repo),
        router=MagicMock(),
        strategy_risk_states={},
        portfolio_risk_state=PortfolioRiskState(trade_date=date.today()),
        unit_risk_dollars=100.0,
        open_positions={},
    )

    adapter.on_status(order.oms_order_id, "Submitted", order.qty)
    await asyncio.wait_for(repo.working_save_started.wait(), timeout=1.0)
    adapter.on_ack(order.oms_order_id, SimpleNamespace(broker_order_id=77))
    adapter.on_fill(
        order.oms_order_id,
        "exec-single",
        101.0,
        order.qty,
        datetime.now(timezone.utc),
        0.0,
    )
    repo.release_working_save.set()

    updated = await _wait_for_order_status(repo, order.oms_order_id, OrderStatus.FILLED)

    assert updated.filled_qty == order.qty
    assert updated.remaining_qty == 0
    assert updated.acked_at is not None
    assert updated.broker_order_ref is not None
    assert repo.saved_statuses[0] == OrderStatus.WORKING
    assert repo.saved_statuses[-1] == OrderStatus.FILLED
    assert "Invalid transition" not in caplog.text
    assert "Fill transition rejected" not in caplog.text


@pytest.mark.asyncio
async def test_multi_oms_callbacks_serialize_status_ack_then_fill(caplog) -> None:
    repo = _DelayedWorkingSaveRepository()
    order = _entry_order("GLD")
    order.status = OrderStatus.ROUTED
    order.remaining_qty = order.qty
    await repo.save_order(order)
    repo.saved_statuses.clear()
    bus = MagicMock()
    bus.emit_order_event = MagicMock()
    bus.emit_fill_event = MagicMock()
    adapter = SimpleNamespace()
    coordinator = MagicMock()
    caplog.set_level(logging.WARNING)

    _wire_adapter_callbacks_multi(
        adapter=adapter,
        bus=bus,
        repo=repo,
        fill_proc=FillProcessor(repo),
        router=MagicMock(),
        strategy_risk_states={},
        portfolio_risk_state=PortfolioRiskState(trade_date=date.today()),
        unit_risk_map={order.strategy_id: 100.0},
        open_positions={},
        coordinator=coordinator,
        portfolio_urd=100.0,
    )

    adapter.on_status(order.oms_order_id, "Submitted", order.qty)
    await asyncio.wait_for(repo.working_save_started.wait(), timeout=1.0)
    adapter.on_ack(order.oms_order_id, SimpleNamespace(broker_order_id=88))
    adapter.on_fill(
        order.oms_order_id,
        "exec-multi",
        99.5,
        order.qty,
        datetime.now(timezone.utc),
        0.0,
    )
    repo.release_working_save.set()

    updated = await _wait_for_order_status(repo, order.oms_order_id, OrderStatus.FILLED)

    assert updated.filled_qty == order.qty
    assert updated.remaining_qty == 0
    assert updated.acked_at is not None
    assert updated.broker_order_ref is not None
    assert repo.saved_statuses[0] == OrderStatus.WORKING
    assert repo.saved_statuses[-1] == OrderStatus.FILLED
    assert "Invalid transition" not in caplog.text
    assert "Fill transition rejected" not in caplog.text


@pytest.mark.asyncio
async def test_save_event_raises_invariant_on_orphan_fk() -> None:
    repo = OMSRepository(_BrokenPool(_BrokenConnection()))
    repo._is_fk_violation = lambda exc: True  # type: ignore[method-assign]

    with pytest.raises(OMSPersistenceInvariantError, match="save_event failed"):
        await repo.save_event("oms-missing", "RISK_APPROVED", {})


@pytest.mark.asyncio
async def test_save_fill_raises_invariant_on_orphan_fk() -> None:
    repo = OMSRepository(_BrokenPool(_BrokenConnection()))
    repo._is_fk_violation = lambda exc: True  # type: ignore[method-assign]

    with pytest.raises(OMSPersistenceInvariantError, match="save_fill failed"):
        await repo.save_fill(
            fill=MagicMock(
                oms_order_id="oms-missing",
                fill_id="fill-1",
                broker_fill_id="exec-1",
                price=1.0,
                qty=1.0,
                timestamp=datetime.now(timezone.utc),
                fees=0.0,
            )
        )
