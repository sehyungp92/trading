from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from libs.oms.engine.fill_processor import FillProcessor
from libs.oms.models.instrument import Instrument
from libs.oms.models.order import OMSOrder, OrderRole, OrderSide, OrderStatus, OrderType
from libs.oms.persistence.in_memory import InMemoryRepository
from libs.oms.reconciliation.orchestrator import ReconciliationOrchestrator

sys.path.append(str(Path(__file__).resolve().parent))
from fake_ibkr import FakeIBKRExecutionAdapter  # noqa: E402


@pytest.mark.asyncio
@pytest.mark.parity_nightly
async def test_oms_restart_imports_offline_fills_before_next_bar() -> None:
    repo = InMemoryRepository()
    order = _order(status=OrderStatus.ROUTED)
    await repo.save_order(order)
    adapter = FakeIBKRExecutionAdapter()
    adapter.executions = [_exec("EXEC-1", broker_order_id=10001, qty=2, price=101.25)]
    orchestrator = ReconciliationOrchestrator(
        adapter,
        repo,
        _Bus(),
        fill_processor=FillProcessor(repo),
    )

    await orchestrator.startup_reconciliation()

    imported = await repo.get_order("OMS-1")
    assert await repo.fill_exists("EXEC-1") is True
    assert imported.status is OrderStatus.FILLED
    assert imported.filled_qty == 2
    assert imported.remaining_qty == 0
    assert imported.avg_fill_price == 101.25
    assert adapter.cache.is_fill_seen("EXEC-1") is True


@pytest.mark.asyncio
@pytest.mark.parity_nightly
async def test_oms_restart_does_not_double_import_fills() -> None:
    repo = InMemoryRepository()
    await repo.save_order(_order(status=OrderStatus.ROUTED))
    adapter = FakeIBKRExecutionAdapter()
    adapter.executions = [_exec("EXEC-1", broker_order_id=10001, qty=2, price=101.25)]
    orchestrator = ReconciliationOrchestrator(
        adapter,
        repo,
        _Bus(),
        fill_processor=FillProcessor(repo),
    )

    await orchestrator.startup_reconciliation()
    await orchestrator.startup_reconciliation()

    imported = await repo.get_order("OMS-1")
    assert len(repo._fills) == 1
    assert imported.filled_qty == 2
    assert imported.remaining_qty == 0


@pytest.mark.asyncio
@pytest.mark.parity_nightly
async def test_fill_processor_walks_routed_to_acked_to_filled() -> None:
    repo = InMemoryRepository()
    await repo.save_order(_order(status=OrderStatus.ROUTED))

    inserted = await FillProcessor(repo).process_fill(
        oms_order_id="OMS-1",
        broker_fill_id="EXEC-1",
        price=101.25,
        qty=2,
        timestamp=datetime(2026, 5, 20, 14, 30, tzinfo=timezone.utc),
        fees=1.24,
    )

    order = await repo.get_order("OMS-1")
    assert inserted is True
    assert order.status is OrderStatus.FILLED
    assert order.filled_qty == 2
    assert order.remaining_qty == 0


def _order(*, status: OrderStatus) -> OMSOrder:
    return OMSOrder(
        oms_order_id="OMS-1",
        client_order_id="CLIENT-1",
        strategy_id="PARITY_RESTART",
        account_id="DU123",
        instrument=_instrument(),
        side=OrderSide.BUY,
        qty=2,
        order_type=OrderType.LIMIT,
        role=OrderRole.ENTRY,
        status=status,
        broker_order_id=10001,
        perm_id=20001,
        remaining_qty=2,
    )


def _instrument() -> Instrument:
    return Instrument(
        symbol="MNQ",
        root="MNQ",
        venue="CME",
        tick_size=0.25,
        tick_value=0.5,
        multiplier=2.0,
    )


def _exec(exec_id: str, *, broker_order_id: int, qty: float, price: float) -> SimpleNamespace:
    return SimpleNamespace(
        exec_id=exec_id,
        broker_order_id=broker_order_id,
        perm_id=20001,
        symbol="MNQ",
        side="BOT",
        qty=qty,
        price=price,
        fill_time=datetime(2026, 5, 20, 14, 31, tzinfo=timezone.utc),
        commission=1.24,
    )


class _Bus:
    def emit_risk_halt(self, *_args, **_kwargs) -> None:
        return None

    def emit_order_event(self, *_args, **_kwargs) -> None:
        return None
