import asyncio
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from shared.ibkr_core.reconciler.discrepancy_policy import DiscrepancyAction
from shared.oms.engine.fill_processor import FillProcessor
from shared.oms.events.bus import EventBus
from shared.oms.models.events import OMSEventType
from shared.oms.models.instrument import Instrument
from shared.oms.models.order import OMSOrder, OrderRole, OrderSide, OrderStatus, OrderType, RiskContext
from shared.oms.models.risk_state import PortfolioRiskState
from shared.oms.persistence.in_memory import InMemoryRepository
from shared.oms.reconciliation.orchestrator import ReconciliationOrchestrator
from shared.oms.services.factory import _wire_adapter_callbacks


def _make_instrument(symbol: str) -> Instrument:
    return Instrument(
        symbol=symbol,
        root=symbol,
        venue="SMART",
        tick_size=0.01,
        tick_value=0.01,
        multiplier=1.0,
        point_value=1.0,
        currency="USD",
        sec_type="STK",
        primary_exchange="NASDAQ",
    )


def _make_order(
    *,
    oms_order_id: str,
    symbol: str,
    side: OrderSide,
    qty: int,
    role: OrderRole,
    status: OrderStatus,
    risk_dollars: float = 0.0,
) -> OMSOrder:
    risk_context = None
    if role == OrderRole.ENTRY:
        price = 10.0 if symbol == "AAPL" else 20.0
        risk_context = RiskContext(
            stop_for_risk=price - 1.0,
            planned_entry_price=price,
            risk_dollars=risk_dollars,
        )
    now = datetime.now(timezone.utc)
    return OMSOrder(
        oms_order_id=oms_order_id,
        strategy_id="US_ORB_v1",
        account_id="DU1",
        instrument=_make_instrument(symbol),
        side=side,
        qty=qty,
        order_type=OrderType.MARKET if role != OrderRole.ENTRY else OrderType.LIMIT,
        limit_price=None if role != OrderRole.ENTRY else (10.0 if symbol == "AAPL" else 20.0),
        role=role,
        status=status,
        risk_context=risk_context,
        created_at=now,
        last_update_at=now,
    )


@pytest.mark.asyncio
async def test_event_bus_broadcasts_global_risk_halt() -> None:
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


@pytest.mark.asyncio
async def test_fill_processor_allows_fill_before_ack() -> None:
    repo = InMemoryRepository()
    fill_proc = FillProcessor(repo)
    order = _make_order(
        oms_order_id="fill-first",
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=10,
        role=OrderRole.ENTRY,
        status=OrderStatus.ROUTED,
        risk_dollars=100.0,
    )
    await repo.save_order(order)

    await fill_proc.process_fill(
        oms_order_id=order.oms_order_id,
        broker_fill_id="broker-fill-1",
        price=10.0,
        qty=10,
        timestamp=datetime.now(timezone.utc),
        fees=0.0,
    )

    saved = await repo.get_order(order.oms_order_id)
    assert saved is not None
    assert saved.status == OrderStatus.FILLED
    assert saved.remaining_qty == 0


@pytest.mark.asyncio
async def test_adapter_callbacks_keep_positions_and_risk_per_symbol() -> None:
    repo = InMemoryRepository()
    bus = EventBus()
    fill_proc = FillProcessor(repo)
    adapter = SimpleNamespace()
    router = SimpleNamespace(route=AsyncMock())
    strategy_risk_states = {}
    portfolio_risk_state = PortfolioRiskState(trade_date=date.today())
    open_positions = {}

    _wire_adapter_callbacks(
        adapter,
        bus,
        repo,
        fill_proc,
        router,
        strategy_risk_states,
        portfolio_risk_state,
        unit_risk_dollars=100.0,
        open_positions=open_positions,
    )

    aapl_entry = _make_order(
        oms_order_id="aapl-entry",
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=10,
        role=OrderRole.ENTRY,
        status=OrderStatus.ROUTED,
        risk_dollars=100.0,
    )
    msft_entry = _make_order(
        oms_order_id="msft-entry",
        symbol="MSFT",
        side=OrderSide.BUY,
        qty=5,
        role=OrderRole.ENTRY,
        status=OrderStatus.ROUTED,
        risk_dollars=100.0,
    )
    aapl_exit = _make_order(
        oms_order_id="aapl-exit",
        symbol="AAPL",
        side=OrderSide.SELL,
        qty=10,
        role=OrderRole.EXIT,
        status=OrderStatus.ROUTED,
    )

    for order in (aapl_entry, msft_entry, aapl_exit):
        await repo.save_order(order)

    adapter.on_fill("aapl-entry", "exec-aapl-entry", 10.0, 10, datetime.now(timezone.utc), 0.0)
    adapter.on_fill("msft-entry", "exec-msft-entry", 20.0, 5, datetime.now(timezone.utc), 0.0)
    adapter.on_fill("aapl-exit", "exec-aapl-exit", 11.0, 10, datetime.now(timezone.utc), 0.0)
    await asyncio.sleep(0.05)

    positions = {pos.instrument_symbol: pos for pos in await repo.get_positions("US_ORB_v1")}
    assert positions["AAPL"].net_qty == 0
    assert positions["MSFT"].net_qty == 5
    assert positions["MSFT"].avg_price == pytest.approx(20.0)
    assert positions["MSFT"].open_risk_R == pytest.approx(1.0)
    assert strategy_risk_states["US_ORB_v1"].open_risk_R == pytest.approx(1.0)
    assert strategy_risk_states["US_ORB_v1"].daily_realized_pnl == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_reconciliation_halt_calls_callback_and_broadcasts() -> None:
    bus = EventBus()
    strategy_q = bus.subscribe("US_ORB_v1")
    halted = {}

    async def _halt_trading(reason: str) -> None:
        halted["reason"] = reason

    adapter = SimpleNamespace(cache=SimpleNamespace(lookup_oms_id=lambda _: None))
    orchestrator = ReconciliationOrchestrator(
        adapter=adapter,
        repo=InMemoryRepository(),
        bus=bus,
        halt_trading=_halt_trading,
    )

    discrepancy = SimpleNamespace(
        type="POSITION_MISMATCH",
        action=DiscrepancyAction.HALT_AND_ALERT,
        details={"symbol": "AAPL", "broker_qty": 10, "oms_qty": 0},
    )

    await orchestrator._apply_discrepancies([discrepancy])

    assert halted["reason"].startswith("Reconciliation: POSITION_MISMATCH")
    event = strategy_q.get_nowait()
    assert event.event_type == OMSEventType.RISK_HALT
    assert "POSITION_MISMATCH" in event.payload["reason"]
