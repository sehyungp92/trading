from __future__ import annotations

import pytest

from libs.broker_ibkr.models.types import BrokerOrderStatus, OrderStatusEvent
from libs.broker_ibkr.reconciler.discrepancy_policy import (
    DiscrepancyAction,
    DiscrepancyPolicy,
)
from libs.broker_ibkr.reconciler.sync import Discrepancy, ReconcilerSync
from libs.broker_ibkr.state.cache import IBCache
from libs.oms.models.order import OMSOrder, OrderStatus
from libs.oms.reconciliation.orchestrator import ReconciliationOrchestrator


def _broker_order(order_ref: str, broker_order_id: int = 101) -> OrderStatusEvent:
    return OrderStatusEvent(
        broker_order_id=broker_order_id,
        perm_id=9001,
        status=BrokerOrderStatus.SUBMITTED,
        filled_qty=0.0,
        remaining_qty=1.0,
        avg_fill_price=0.0,
        order_ref=order_ref,
    )


class _Repo:
    def __init__(self, order: OMSOrder | None = None) -> None:
        self.order = order
        self.saved: list[OMSOrder] = []

    async def get_order(self, oms_order_id: str) -> OMSOrder | None:
        if self.order and self.order.oms_order_id == oms_order_id:
            return self.order
        return None

    async def save_order(self, order: OMSOrder) -> None:
        self.saved.append(order)


class _Bus:
    def __init__(self) -> None:
        self.order_events: list[OMSOrder] = []
        self.risk_halts: list[str] = []

    def emit_order_event(self, order: OMSOrder) -> None:
        self.order_events.append(order)

    def emit_risk_halt(self, strategy_id: str, reason: str) -> None:
        self.risk_halts.append(reason)


class _Adapter:
    def __init__(self) -> None:
        self.cache = IBCache()
        self.cancelled: list[tuple[int, int]] = []

    async def cancel_order(self, broker_order_id: int, perm_id: int = 0) -> None:
        self.cancelled.append((broker_order_id, perm_id))


def test_blank_ref_broker_orphan_stays_cancellable() -> None:
    sync = ReconcilerSync(DiscrepancyPolicy())

    discrepancies = sync.reconcile_orders(
        [_broker_order("")],
        oms_working_ids=set(),
        known_order_refs={},
    )

    assert len(discrepancies) == 1
    assert discrepancies[0].action == DiscrepancyAction.CANCEL


def test_reconcile_orders_normalizes_broker_id_inputs() -> None:
    sync = ReconcilerSync(DiscrepancyPolicy())

    discrepancies = sync.reconcile_orders(
        [_broker_order("", broker_order_id="101")],  # type: ignore[arg-type]
        oms_working_ids={"101"},  # type: ignore[arg-type]
        known_order_refs={},
    )

    assert discrepancies == []


@pytest.mark.asyncio
async def test_known_order_ref_repairs_mapping_without_halt() -> None:
    order = OMSOrder(
        oms_order_id="oms-1",
        client_order_id="client-1",
        strategy_id="S1",
        status=OrderStatus.RISK_APPROVED,
        qty=1,
    )
    repo = _Repo(order)
    bus = _Bus()
    adapter = _Adapter()
    halted: list[str] = []
    async def _halt(reason: str) -> None:
        halted.append(reason)
    orchestrator = ReconciliationOrchestrator(
        adapter,
        repo,
        bus,
        halt_trading=_halt,
    )
    discrepancy = Discrepancy(
        "repair_order_mapping",
        DiscrepancyAction.REPAIR_MAPPING,
        {"order": _broker_order("client-1"), "order_ref": "client-1", "oms_order_id": "oms-1"},
    )

    await orchestrator._apply_discrepancies([discrepancy])

    assert halted == []
    assert bus.risk_halts == []
    assert order.broker_order_id == 101
    assert order.perm_id == 9001
    assert order.status == OrderStatus.WORKING
    assert adapter.cache.lookup_oms_id(101) == "oms-1"
    assert repo.saved == [order]
    assert bus.order_events == [order]


@pytest.mark.asyncio
async def test_unknown_nonblank_order_ref_halts() -> None:
    sync = ReconcilerSync(DiscrepancyPolicy())
    discrepancies = sync.reconcile_orders(
        [_broker_order("mystery-ref")],
        oms_working_ids=set(),
        known_order_refs={},
    )
    bus = _Bus()
    halted: list[str] = []
    async def _halt(reason: str) -> None:
        halted.append(reason)
    orchestrator = ReconciliationOrchestrator(
        _Adapter(),
        _Repo(),
        bus,
        halt_trading=_halt,
    )

    await orchestrator._apply_discrepancies(discrepancies)

    assert discrepancies[0].action == DiscrepancyAction.IMPORT
    assert halted
    assert bus.risk_halts


@pytest.mark.asyncio
async def test_position_quantity_mismatch_halts() -> None:
    bus = _Bus()
    halted: list[str] = []
    async def _halt(reason: str) -> None:
        halted.append(reason)
    orchestrator = ReconciliationOrchestrator(
        _Adapter(),
        _Repo(),
        bus,
        halt_trading=_halt,
    )
    discrepancy = Discrepancy(
        "position_mismatch",
        DiscrepancyAction.ADJUST_POSITION,
        {"symbol": "MNQ", "broker_qty": 2.0, "oms_qty": 1.0},
    )

    await orchestrator._apply_discrepancies([discrepancy])

    assert halted
    assert "position mismatch" in halted[0].lower()
    assert bus.risk_halts == halted
