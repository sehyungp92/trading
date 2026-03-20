import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.oms.execution.router import ExecutionRouter
from shared.oms.models.instrument import Instrument
from shared.oms.models.order import OMSOrder, OrderRole, OrderSide, OrderStatus, OrderType
from shared.oms.persistence.in_memory import InMemoryRepository


class _FakeAdapter:
    def __init__(self):
        self.is_congested = False
        self.submissions = []

    async def submit_order(self, **kwargs):
        self.submissions.append(kwargs)

        class _Ref:
            broker_order_id = 77
            perm_id = 88

        return _Ref()


def test_router_passes_full_instrument_to_adapter():
    repo = InMemoryRepository()
    adapter = _FakeAdapter()
    router = ExecutionRouter(adapter, repo)
    instrument = Instrument(
        symbol="AAPL",
        root="AAPL",
        venue="SMART",
        sec_type="STK",
        primary_exchange="NASDAQ",
        tick_size=0.01,
        tick_value=0.01,
        multiplier=1.0,
    )
    order = OMSOrder(
        strategy_id="US_ORB_v1",
        instrument=instrument,
        side=OrderSide.BUY,
        qty=25,
        order_type=OrderType.STOP_LIMIT,
        limit_price=101.2,
        stop_price=101.0,
        role=OrderRole.ENTRY,
        status=OrderStatus.RISK_APPROVED,
    )

    asyncio.run(router._submit_to_adapter(order))

    assert adapter.submissions
    submitted = adapter.submissions[0]
    assert submitted["instrument"].symbol == "AAPL"
    assert submitted["instrument"].sec_type == "STK"
    assert submitted["instrument"].primary_exchange == "NASDAQ"
    assert order.broker_order_id == 77
